"""Comparación de varios modelos sobre una misma partición.

Entrena/evalúa un conjunto de modelos con las mismas métricas y devuelve una
tabla ordenada por desempeño, para validar más de un modelo por caso y decidir
con evidencia cuál (o qué combinación) usar.
"""

from __future__ import annotations

import pandas as pd

from tostao_ml.framework.models.base import BaseModel
from tostao_ml.framework.narrate import Insight, Severity

from . import metrics


def compare_models(
    models: dict[str, BaseModel],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    sort_by: str = "wape",
) -> pd.DataFrame:
    """Entrena (si hace falta) y evalúa varios modelos de regresión en el holdout.

    Args:
        models: Mapa nombre→modelo (``BaseModel``).
        X_train, y_train: Datos de entrenamiento.
        X_test, y_test: Datos de evaluación (holdout).
        sort_by: Métrica por la que ordenar (ascendente para errores).

    Returns:
        DataFrame indexado por modelo con MAE/RMSE/WAPE/sMAPE/R², ordenado.
    """
    rows = []
    for name, model in models.items():
        if not model.is_fitted_:
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"modelo": name, **metrics.regression_report(y_test, pred)})
    table = pd.DataFrame(rows).set_index("modelo")
    ascending = sort_by not in {"r2"}
    return table.sort_values(sort_by, ascending=ascending)


def narrate_comparison(
    comparison: pd.DataFrame, incumbent: str, *, metric: str = "wape"
) -> Insight:
    """Redacta qué modelo gana y si supera al modelo base (incumbente)."""
    best = str(comparison.index[0])
    best_v = float(comparison.iloc[0][metric])
    inc_v = float(comparison.loc[incumbent, metric]) if incumbent in comparison.index else best_v
    delta = (inc_v - best_v) / inc_v * 100 if inc_v else 0.0
    if best == incumbent or delta <= 0.5:
        text = (
            f"Se validaron {len(comparison)} modelos; el mejor es «{best}» ({metric.upper()} "
            f"{best_v:.1%}) y no hay una mejora relevante al combinarlos."
        )
        sev = Severity.INFO
    else:
        text = (
            f"Se validaron {len(comparison)} modelos; «{best}» supera al modelo base «{incumbent}», "
            f"reduciendo el {metric.upper()} en {delta:.0f}% (de {inc_v:.1%} a {best_v:.1%})."
        )
        sev = Severity.GOOD
    return Insight(
        text=text,
        severity=sev,
        metrics={f"best_{metric}": round(best_v, 4)},
        tags=("comparacion",),
        title="Comparación de modelos",
    )
