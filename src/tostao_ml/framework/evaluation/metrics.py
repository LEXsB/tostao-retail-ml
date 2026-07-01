"""Métricas de evaluación para regresión, forecasting e intervalos y clasificación.

Funciones puras sobre arrays; los pipelines las combinan según el tipo de tarea.
Incluye métricas específicas de negocio-retail (WAPE) y de forecasting
probabilístico (pinball, PICP, MPIW, Theil's U).
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def _to_arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


# --------------------------------------------------------------------------- #
# Regresión / forecasting puntual
# --------------------------------------------------------------------------- #
def mae(y_true: object, y_pred: object) -> float:
    """Error absoluto medio."""
    a, b = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true: object, y_pred: object) -> float:
    """Raíz del error cuadrático medio."""
    a, b = _to_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def wape(y_true: object, y_pred: object) -> float:
    """Weighted Absolute Percentage Error = Σ|y-ŷ| / Σ|y|.

    Robusta a ceros (a diferencia de MAPE), estándar en demanda de retail.
    """
    a, b = _to_arrays(y_true, y_pred)
    denom = np.sum(np.abs(a))
    return float(np.sum(np.abs(a - b)) / denom) if denom else float("nan")


def smape(y_true: object, y_pred: object) -> float:
    """Symmetric MAPE ∈ [0, 2]; evita división por cero simétricamente."""
    a, b = _to_arrays(y_true, y_pred)
    denom = np.abs(a) + np.abs(b)
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(a[mask] - b[mask]) / denom[mask]))


def r2(y_true: object, y_pred: object) -> float:
    """Coeficiente de determinación R²."""
    a, b = _to_arrays(y_true, y_pred)
    return float(skm.r2_score(a, b))


def theil_u2(y_true: object, y_pred: object) -> float:
    """Estadístico U2 de Theil vs. pronóstico ingenuo (persistencia).

    <1 mejor que la persistencia; =1 igual; >1 peor.
    """
    a, b = _to_arrays(y_true, y_pred)
    if a.size < 2:
        return float("nan")
    naive = a[:-1]
    num = np.sqrt(np.mean((b[1:] - a[1:]) ** 2))
    den = np.sqrt(np.mean((naive - a[1:]) ** 2))
    return float(num / den) if den else float("nan")


# --------------------------------------------------------------------------- #
# Forecasting probabilístico / intervalos
# --------------------------------------------------------------------------- #
def pinball_loss(y_true: object, y_pred: object, quantile: float) -> float:
    """Pérdida pinball (quantile loss) para un cuantil dado."""
    a, b = _to_arrays(y_true, y_pred)
    diff = a - b
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def picp(y_true: object, lower: object, upper: object) -> float:
    """Prediction Interval Coverage Probability: fracción cubierta por el intervalo."""
    a, lo, hi = (
        np.asarray(y_true, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    return float(np.mean((a >= lo) & (a <= hi)))


def mpiw(lower: object, upper: object) -> float:
    """Mean Prediction Interval Width: ancho medio del intervalo."""
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    return float(np.mean(hi - lo))


def regression_report(y_true: object, y_pred: object) -> dict[str, float]:
    """Panel de métricas de regresión/forecasting puntual."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Clasificación
# --------------------------------------------------------------------------- #
def ks_statistic(y_true: object, y_score: object) -> float:
    """Estadístico KS: máxima separación entre CDFs de score por clase."""
    a = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    pos, neg = np.sort(s[a == 1]), np.sort(s[a == 0])
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    grid = np.sort(np.unique(s))
    cdf_pos = np.searchsorted(pos, grid, side="right") / pos.size
    cdf_neg = np.searchsorted(neg, grid, side="right") / neg.size
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def classification_report(
    y_true: object, y_pred: object, y_score: object | None = None
) -> dict[str, float]:
    """Panel de métricas de clasificación binaria."""
    a = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    out = {
        "accuracy": float(skm.accuracy_score(a, p)),
        "precision": float(skm.precision_score(a, p, zero_division=0)),
        "recall": float(skm.recall_score(a, p, zero_division=0)),
        "f1": float(skm.f1_score(a, p, zero_division=0)),
    }
    if y_score is not None:
        s = np.asarray(y_score, dtype=float)
        if len(np.unique(a)) > 1:
            out["roc_auc"] = float(skm.roc_auc_score(a, s))
            out["pr_auc"] = float(skm.average_precision_score(a, s))
            out["ks"] = ks_statistic(a, s)
            out["brier"] = float(skm.brier_score_loss(a, s))
    return out
