"""Caso A — Abastecimiento: forecast probabilístico + optimización de pedido.

Reutiliza el framework: transformadores de features (rezagos group-aware,
codificación cíclica y de frecuencia), el modelo cuantílico, las métricas de
evaluación y el solver newsvendor. Aquí solo vive la orquestación del caso.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tostao_ml.framework.evaluation import metrics
from tostao_ml.framework.features import CyclicalEncoder, FrequencyEncoder, GroupLagFeatures
from tostao_ml.framework.models import QuantileGBRModel
from tostao_ml.framework.narrate import Insight, Narrative, Severity, rules
from tostao_ml.framework.optimization import NewsvendorPolicy, expected_cost

GROUP = ["id_tienda", "id_producto"]
TARGET = "unidades_vendidas"
_STATIC_NUM = [
    "precio_venta",
    "costo_unitario",
    "costo_almacenamiento_semanal",
    "tamaño_m2",
    "stock_actual",
    "dias_con_venta",
]
_LAG_COLS = [f"{TARGET}_lag_1", f"{TARGET}_lag_2", f"{TARGET}_rollmean_3"]
_CYCLIC = ["semana_sin", "semana_cos"]
_FREQ = ["categoria_freq", "ciudad_freq", "trend_type_freq"]
FEATURES = _LAG_COLS + _CYCLIC + _FREQ + _STATIC_NUM


@dataclass(slots=True)
class CaseAResult:
    """Resultado del Caso A: métricas, decisión de pedido y narrativa."""

    metrics: dict[str, float]
    test: pd.DataFrame
    orders: pd.DataFrame
    cost_model: float
    cost_naive: float
    narrative: Narrative = field(default_factory=Narrative)
    model: object = None
    feature_names: list[str] = field(default_factory=list)


def build_features_a(weekly: pd.DataFrame) -> pd.DataFrame:
    """Genera features de forecast reutilizando los transformadores del framework."""
    df = weekly.sort_values([*GROUP, "semana"]).copy()
    df = GroupLagFeatures(GROUP, "semana", TARGET, lags=(1, 2), rolling_windows=(3,)).fit_transform(
        df
    )
    df = CyclicalEncoder(["semana"], {"semana": 52}, drop_original=False).fit_transform(df)
    df = FrequencyEncoder(["categoria", "ciudad", "trend_type"], drop_original=False).fit_transform(
        df
    )
    return df.dropna(subset=[f"{TARGET}_lag_1"]).reset_index(drop=True)


def temporal_split(
    features: pd.DataFrame, test_weeks: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corta las últimas ``test_weeks`` semanas como holdout (sin leakage)."""
    cutoff = features["semana"].max() - test_weeks
    return features[features["semana"] <= cutoff].copy(), features[
        features["semana"] > cutoff
    ].copy()


def run_case_a(
    weekly: pd.DataFrame,
    *,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    max_iter: int = 200,
    seed: int = 42,
) -> CaseAResult:
    """Entrena, evalúa y optimiza el pedido para el Caso A."""
    features = build_features_a(weekly)
    train, test = temporal_split(features)

    model = QuantileGBRModel(quantiles=quantiles, max_iter=max_iter, random_state=seed)
    model.fit(train[FEATURES], train[TARGET])

    q_pred = model.predict_quantiles(test[FEATURES])
    interval = model.predict_interval(test[FEATURES], coverage=quantiles[-1] - quantiles[0])
    test = test.assign(pred=interval["median"], lower=interval["lower"], upper=interval["upper"])

    y = test[TARGET].to_numpy(dtype=float)
    naive = test[f"{TARGET}_lag_1"].to_numpy(dtype=float)  # baseline: demanda de la semana previa
    result_metrics = {
        **metrics.regression_report(y, test["pred"]),
        "wape_naive": metrics.wape(y, naive),
        "pinball_p90": metrics.pinball_loss(y, q_pred[f"q{quantiles[-1]}"], quantiles[-1]),
        "picp": metrics.picp(y, test["lower"], test["upper"]),
        "mpiw": metrics.mpiw(test["lower"], test["upper"]),
    }

    orders, cost_model, cost_naive = _optimize_orders(test, q_pred, quantiles)
    narrative = _narrate(result_metrics, cost_model, cost_naive, quantiles)
    return CaseAResult(
        result_metrics,
        test,
        orders,
        cost_model,
        cost_naive,
        narrative,
        model=model,
        feature_names=list(FEATURES),
    )


def _optimize_orders(
    test: pd.DataFrame, q_pred: pd.DataFrame, quantiles: tuple[float, ...]
) -> tuple[pd.DataFrame, float, float]:
    """Aplica la política newsvendor y compara su costo esperado con el ingenuo."""
    cu = (test["precio_venta"] - test["costo_unitario"]).to_numpy(dtype=float)
    co = test["costo_almacenamiento_semanal"].to_numpy(dtype=float)
    stock = test["stock_actual"].to_numpy(dtype=float)
    demand = test[TARGET].to_numpy(dtype=float)
    naive_order = test[f"{TARGET}_lag_1"].to_numpy(dtype=float)

    records, cost_model, cost_naive = [], 0.0, 0.0
    for i in range(len(test)):
        quantile_map = {q: float(q_pred.iloc[i][f"q{q}"]) for q in quantiles}
        policy = NewsvendorPolicy(cu=float(cu[i]), co=float(co[i]))
        decision = policy.order(quantile_map, float(stock[i]))
        records.append(decision)
        cost_model += expected_cost(
            decision["order_qty"], stock[i], np.array([demand[i]]), cu[i], co[i]
        )
        naive_qty = max(0.0, naive_order[i] - stock[i])
        cost_naive += expected_cost(naive_qty, stock[i], np.array([demand[i]]), cu[i], co[i])
    return pd.DataFrame(records, index=test.index), float(cost_model), float(cost_naive)


def _narrate(
    m: dict[str, float], cost_model: float, cost_naive: float, quantiles: tuple[float, ...]
) -> Narrative:
    """Redacta las conclusiones del caso con cifras reales."""
    narrative = Narrative()
    narrative.add(
        rules.metric_vs_baseline_insight("WAPE", m["wape"], m["wape_naive"], higher_is_better=False)
    )
    nominal = quantiles[-1] - quantiles[0]
    narrative.add(rules.interval_coverage_insight(m["picp"], nominal, m["mpiw"]))
    saved = cost_naive - cost_model
    saved_pct = (saved / cost_naive * 100.0) if cost_naive else 0.0
    narrative.add(
        Insight(
            text=(
                f"Costo esperado de faltante+sobrante: política óptima = {cost_model:,.0f} vs "
                f"ingenua = {cost_naive:,.0f} ⇒ ahorro de {saved:,.0f} ({saved_pct:.1f}%)."
            ),
            severity=Severity.GOOD if saved > 0 else Severity.WARNING,
            metrics={
                "cost_model": round(cost_model, 2),
                "cost_naive": round(cost_naive, 2),
                "saved_pct": round(saved_pct, 2),
            },
            tags=("business", "caso_a"),
            title="Impacto de negocio",
        )
    )
    return narrative
