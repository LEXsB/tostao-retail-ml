"""API de inferencia FastAPI para los tres casos (AIOps §11).

Endpoints versionados por caso con validación Pydantic y health/readiness. Al
arrancar, construye las tablas maestras desde ``data/01_raw`` y entrena/deriva los
artefactos de servicio en memoria (modelo de gasto del Caso C y combos del Caso B).
El Caso A se sirve con el solver newsvendor (no requiere modelo persistido).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

import tostao_ml
from tostao_ml.cases import caso_b, caso_c, masters
from tostao_ml.framework.models import GBRRegressionModel
from tostao_ml.framework.optimization import NewsvendorPolicy

from .schemas import (
    Combo,
    HealthResponse,
    OrderRequest,
    OrderResponse,
    SpendRequest,
    SpendResponse,
)

logger = logging.getLogger("tostao_ml.serving")
RAW = Path("data/01_raw")
STATE: dict[str, Any] = {"ready": False}


def _load_raw(case_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / case_dir / f"{name}.csv")


def _warmup() -> None:
    """Construye masters y entrena/deriva artefactos de servicio en memoria."""
    # Caso B — combos.
    tickets = _load_raw("02_product_bundles", "tickets")
    detalle = _load_raw("02_product_bundles", "detalle_tickets")
    catalogo_b = _load_raw("02_product_bundles", "catalogo_productos")
    master_b, _ = masters.build_master_b(tickets, detalle, catalogo_b)
    baskets = masters.build_baskets_b(master_b)
    STATE["combos"] = caso_b.run_case_b(master_b, baskets).combos

    # Caso C — modelo de gasto del cliente recurrente.
    trans = _load_raw("03_aov_drivers", "transacciones_resumen")
    loyalty = _load_raw("03_aov_drivers", "clientes_loyalty")
    exo = _load_raw("03_aov_drivers", "variables_exogenas")
    promo = _load_raw("03_aov_drivers", "promociones_activas")
    master_c, _ = masters.build_master_c(trans, loyalty, exo, promo)
    STATE["spend_model"], STATE["segmentos"] = _train_spend_model(master_c)
    STATE["ready"] = True
    logger.info("Warmup completado: combos y modelo de gasto listos.")


def _train_spend_model(master_c: pd.DataFrame) -> tuple[GBRRegressionModel, list[str]]:
    """Entrena el modelo de gasto (features RFM + loyalty) para servir."""
    rfm = caso_c.RFMTransformer("id_cliente", "fecha", "total_venta", score=False).fit_transform(
        master_c
    )
    profile = master_c.groupby("id_cliente", observed=True).agg(
        edad=("edad", "first"),
        segmento=("segmento", "first"),
        antiguedad=("antiguedad_cliente_dias", "max"),
        ticket_medio=("total_venta", "mean"),
    )
    data = rfm.join(profile).dropna(subset=["ticket_medio"])
    data = data[data["frequency"] >= 2]
    feat = pd.get_dummies(
        data[["recency", "frequency", "edad", "antiguedad", "segmento"]],
        columns=["segmento"],
        drop_first=True,
    ).astype(float)
    model = GBRRegressionModel(max_iter=200).fit(feat, data["ticket_medio"].astype(float))
    return model, list(feat.columns)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida: warmup al arrancar (tolerante a fallos de datos)."""
    try:
        _warmup()
    except Exception as exc:  # pragma: no cover - datos ausentes en el entorno
        logger.warning("Warmup fallido (%s); el servicio arranca en modo degradado.", exc)
    yield
    STATE.clear()


app = FastAPI(
    title="Tostao Retail ML API",
    version=tostao_ml.__version__,
    description="Inferencia de abastecimiento (A), combos (B) y gasto/AOV (C).",
    lifespan=lifespan,
)

# Observabilidad: expone métricas Prometheus en /metrics si la librería está
# disponible (extra `observability`). Prometheus las scrapea (ver deployment/).
try:  # pragma: no cover - depende del extra instalado
    from prometheus_client import make_asgi_app

    app.mount("/metrics", make_asgi_app())
except ImportError:  # pragma: no cover
    logger.info("prometheus_client no instalado; /metrics deshabilitado.")


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    """Liveness + readiness del servicio."""
    return HealthResponse(
        status="ok", version=tostao_ml.__version__, models_ready=bool(STATE.get("ready"))
    )


@app.post("/v1/caso_a/order", response_model=OrderResponse, tags=["caso_a"])
def recommend_order(req: OrderRequest) -> OrderResponse:
    """Recomienda la cantidad a pedir (política newsvendor / critical fractile)."""
    cu = req.precio_venta - req.costo_unitario
    policy = NewsvendorPolicy(
        cu=cu, co=req.costo_almacenamiento_semanal, safety_factor=req.safety_factor
    )
    decision = policy.order(req.quantiles, req.stock_actual)
    return OrderResponse(**decision)


@app.get("/v1/caso_b/combos", response_model=list[Combo], tags=["caso_b"])
def list_combos(cluster: int | None = None) -> list[Combo]:
    """Devuelve los combos propuestos, opcionalmente filtrados por cluster."""
    if not STATE.get("ready"):
        raise HTTPException(
            status_code=503, detail="Servicio calentando; reintente en unos segundos."
        )
    combos: pd.DataFrame = STATE["combos"]
    if cluster is not None:
        combos = combos[combos["cluster"] == cluster]
    return [Combo(**row) for row in combos.to_dict(orient="records")]


@app.post("/v1/caso_c/spend", response_model=SpendResponse, tags=["caso_c"])
def predict_spend(req: SpendRequest) -> SpendResponse:
    """Predice el gasto esperado del cliente recurrente en su próxima visita."""
    if not STATE.get("ready"):
        raise HTTPException(
            status_code=503, detail="Servicio calentando; reintente en unos segundos."
        )
    model: GBRRegressionModel = STATE["spend_model"]
    columns: list[str] = STATE["segmentos"]
    row = {
        "recency": req.recency,
        "frequency": req.frequency,
        "edad": req.edad,
        "antiguedad": req.antiguedad,
    }
    for col in columns:
        if col.startswith("segmento_"):
            row[col] = 1.0 if col == f"segmento_{req.segmento}" else 0.0
    features = pd.DataFrame([row]).reindex(columns=columns, fill_value=0.0)
    pred = float(model.predict(features)[0])
    return SpendResponse(gasto_esperado=round(pred, 4))
