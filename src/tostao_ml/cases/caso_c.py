"""Caso C — AOV: drivers del ticket (GLM inferencial) + gasto esperado (predictivo).

Reutiliza el framework: winsorización de outliers, GLM inferencial con IC, RFM por
cliente, gradient boosting, métricas de regresión e interpretabilidad SHAP.
Dos modelos complementarios: uno explica los drivers del ticket, otro predice el
gasto esperado del cliente recurrente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tostao_ml.framework.evaluation import kfold_splitter, metrics
from tostao_ml.framework.features import RFMTransformer, Winsorizer
from tostao_ml.framework.models import GBRRegressionModel, GLMModel
from tostao_ml.framework.narrate import Insight, Narrative, Severity, rules
from tostao_ml.framework.tuning import TuningResult, tune_model

_HPO_SPACE = {
    "learning_rate": {"type": "float", "low": 0.02, "high": 0.3, "log": True},
    "max_depth": {"type": "int", "low": 2, "high": 8},
    "max_iter": {"type": "int", "low": 80, "high": 350},
}

TARGET = "total_venta"
_DRIVERS_NUM = [
    "total_articulos",
    "edad",
    "competitor_price_index",
    "indice_trafico",
    "n_promos_activas",
    "hora",
    "dia_semana",
    "antiguedad_cliente_dias",
]
_DRIVERS_CAT = ["segmento", "clima"]


@dataclass(slots=True)
class CaseCResult:
    """Resultado del Caso C: drivers inferenciales y modelo predictivo de gasto."""

    coefficients: pd.DataFrame
    inferential_metrics: dict[str, float]
    predictive_metrics: dict[str, float]
    predictive_test: pd.DataFrame
    narrative: Narrative = field(default_factory=Narrative)
    predictive_model: object = None
    predictive_features: pd.DataFrame | None = None
    tuning: TuningResult | None = None


def _design_matrix(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    """Matriz de diseño numérica con one-hot de categóricas (nombres interpretables)."""
    present_num = [c for c in num if c in df.columns]
    present_cat = [c for c in cat if c in df.columns]
    X = pd.get_dummies(df[present_num + present_cat], columns=present_cat, drop_first=True)
    return X.astype(float)


def inferential_drivers(master_c: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], Narrative]:
    """GLM inferencial de los drivers del ticket (coeficientes + IC), con outliers domados."""
    df = Winsorizer([TARGET], 0.01, 0.99).fit_transform(master_c)
    X = _design_matrix(df, _DRIVERS_NUM, _DRIVERS_CAT)
    y = df[TARGET].astype(float)
    model = GLMModel(family="gaussian").fit(X, y)
    coefs = model.coefficients_frame().drop(index="const", errors="ignore")
    coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)

    narrative = Narrative()
    significant = coefs[coefs["pvalue"] < 0.05]
    for name, row in significant.head(5).iterrows():
        direction = "aumenta" if row["coef"] > 0 else "reduce"
        narrative.add(
            Insight(
                text=(
                    f"«{name}» {direction} el ticket: β = {row['coef']:+.3f} "
                    f"(IC95% [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], p = {row['pvalue']:.3g})."
                ),
                severity=Severity.GOOD,
                metrics={"coef": round(float(row["coef"]), 4)},
                tags=("caso_c", "drivers"),
            )
        )
    inf_metrics = {
        "aic": float(model.metadata.extra.get("aic", float("nan"))),
        "n_drivers_significativos": len(significant),
    }
    return coefs, inf_metrics, narrative


def predictive_spend(
    master_c: pd.DataFrame, *, seed: int = 42, tune: bool = False, n_trials: int = 20
) -> tuple[
    dict[str, float], pd.DataFrame, GBRRegressionModel, pd.DataFrame, TuningResult | None, Narrative
]:
    """Modelo predictivo del gasto esperado del cliente recurrente (features RFM + loyalty).

    Si ``tune`` es ``True``, ajusta los hiperparámetros del boosting con Optuna
    (K-Fold) optimizando el WAPE antes del entrenamiento final.
    """
    rfm = RFMTransformer("id_cliente", "fecha", TARGET, score=False).fit_transform(master_c)
    profile = master_c.groupby("id_cliente", observed=True).agg(
        edad=("edad", "first"),
        segmento=("segmento", "first"),
        antiguedad=("antiguedad_cliente_dias", "max"),
        ticket_medio=("total_venta", "mean"),
    )
    data = rfm.join(profile).dropna(subset=["ticket_medio"])
    # Clientes recurrentes (≥2 visitas): el gasto futuro es predecible por su perfil.
    data = data[data["frequency"] >= 2]

    y = data["ticket_medio"].astype(float)
    feat = _design_matrix(
        data.assign()[["recency", "frequency", "edad", "antiguedad", "segmento"]],
        ["recency", "frequency", "edad", "antiguedad"],
        ["segmento"],
    )
    rng = np.random.default_rng(seed)
    mask = rng.random(len(data)) < 0.8

    tuning: TuningResult | None = None
    hp: dict[str, object] = {"max_iter": 200}
    if tune:
        try:
            tuning = tune_model(
                "gbr",
                _HPO_SPACE,
                feat[mask].reset_index(drop=True),
                y[mask].reset_index(drop=True),
                scorer=lambda m, xv, yv: metrics.wape(yv, m.predict(xv)),
                splitter=kfold_splitter(n_splits=4, seed=seed),
                direction="minimize",
                n_trials=n_trials,
                sampler="tpe",
                pruner="none",
                seed=seed,
            )
            hp = {k: tuning.best_params[k] for k in ("learning_rate", "max_depth", "max_iter")}
        except (ValueError, RuntimeError):  # pragma: no cover - datos insuficientes
            tuning = None

    model = GBRRegressionModel(random_state=seed, **hp).fit(feat[mask], y[mask])  # type: ignore[arg-type]
    pred = model.predict(feat[~mask])
    test = data[~mask].assign(pred=pred)

    report = metrics.regression_report(y[~mask], pred)
    baseline = np.full(len(pred), y[mask].mean())
    report["wape_baseline"] = metrics.wape(y[~mask], baseline)

    narrative = Narrative()
    narrative.add(
        rules.metric_vs_baseline_insight(
            "WAPE", report["wape"], report["wape_baseline"], higher_is_better=False
        )
    )
    narrative.add(
        Insight(
            text=(
                f"Modelo de gasto esperado sobre {int(mask.sum())} clientes recurrentes: "
                f"R² = {report['r2']:.2f}, MAE = {report['mae']:.2f}."
            ),
            severity=Severity.GOOD if report["r2"] > 0 else Severity.WARNING,
            metrics={"r2": round(report["r2"], 4), "mae": round(report["mae"], 4)},
            tags=("caso_c", "predictivo"),
            title="Predicción de gasto",
        )
    )
    if tuning is not None:
        narrative.extend(tuning.narrative)
    return report, test, model, feat[~mask], tuning, narrative


def run_case_c(
    master_c: pd.DataFrame, *, seed: int = 42, tune: bool = False, n_trials: int = 20
) -> CaseCResult:
    """Ejecuta el Caso C completo: drivers inferenciales + gasto esperado."""
    coefs, inf_metrics, inf_narr = inferential_drivers(master_c)
    pred_metrics, pred_test, pred_model, pred_feat, tuning, pred_narr = predictive_spend(
        master_c, seed=seed, tune=tune, n_trials=n_trials
    )
    narrative = Narrative()
    narrative.extend(inf_narr)
    narrative.extend(pred_narr)
    return CaseCResult(
        coefs,
        inf_metrics,
        pred_metrics,
        pred_test,
        narrative,
        predictive_model=pred_model,
        predictive_features=pred_feat,
        tuning=tuning,
    )
