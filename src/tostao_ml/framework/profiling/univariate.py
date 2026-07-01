"""Análisis univariado (§4.2): estadística, outliers y normalidad.

Funciones puras que reciben una Series y devuelven diccionarios de métricas,
listas para tabular o narrar. La elección de qué calcular la hace el motor según
el :class:`VariableKind`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def numeric_summary(series: pd.Series) -> dict[str, float]:
    """Resumen estadístico de una variable numérica.

    Incluye tendencia central, dispersión (incl. CV), forma (skew/kurtosis),
    percentiles y proporciones de nulos/ceros.
    """
    values = series.dropna().to_numpy(dtype=float)
    n = int(series.shape[0])
    if values.size == 0:
        return {"count": 0, "pct_null": 100.0}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    summary = {
        "count": float(values.size),
        "mean": mean,
        "median": float(np.median(values)),
        "std": std,
        "cv": float(std / mean) if mean else np.nan,
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "skewness": float(stats.skew(values)) if values.size > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(values)) if values.size > 3 else 0.0,
        "pct_null": float((n - values.size) / n * 100.0) if n else 0.0,
        "pct_zero": float(np.mean(values == 0) * 100.0),
    }
    return summary


def outlier_fraction_iqr(series: pd.Series, k: float = 1.5) -> float:
    """Porcentaje de valores fuera de ``[Q1 - k·IQR, Q3 + k·IQR]``."""
    values = series.dropna().to_numpy(dtype=float)
    if values.size < 4:
        return 0.0
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return float(np.mean((values < lower) | (values > upper)) * 100.0)


def outlier_fraction_mad(series: pd.Series, threshold: float = 3.5) -> float:
    """Porcentaje de outliers por z-score robusto (MAD, Iglewicz-Hoaglin)."""
    values = series.dropna().to_numpy(dtype=float)
    if values.size < 4:
        return 0.0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return 0.0
    robust_z = 0.6745 * (values - median) / mad
    return float(np.mean(np.abs(robust_z) > threshold) * 100.0)


def normality_test(series: pd.Series) -> dict[str, float | str]:
    """Test de normalidad adaptado al tamaño muestral.

    Usa Shapiro-Wilk para muestras pequeñas y D'Agostino-Pearson para grandes
    (Shapiro es poco fiable con n muy grande).
    """
    values = series.dropna().to_numpy(dtype=float)
    if values.size < 8 or np.ptp(values) == 0:
        return {"test": "n/a", "statistic": np.nan, "pvalue": np.nan}
    if values.size <= 5000:
        stat, pvalue = stats.shapiro(values)
        test = "Shapiro-Wilk"
    else:
        stat, pvalue = stats.normaltest(values)
        test = "D'Agostino-Pearson"
    return {"test": test, "statistic": float(stat), "pvalue": float(pvalue)}


def shannon_entropy(series: pd.Series, base: float = 2.0) -> float:
    """Entropía de Shannon de una variable categórica (en bits por defecto)."""
    counts = series.value_counts(dropna=True).to_numpy(dtype=float)
    if counts.size <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)) / np.log(base))


def categorical_summary(series: pd.Series) -> dict[str, float | str]:
    """Resumen de una variable categórica: cardinalidad, moda, entropía, nulos."""
    n = int(series.shape[0])
    non_null = series.dropna()
    if non_null.empty:
        return {"count": 0, "cardinality": 0, "pct_null": 100.0}
    counts = non_null.value_counts()
    return {
        "count": int(non_null.size),
        "cardinality": int(non_null.nunique()),
        "mode": str(counts.index[0]),
        "mode_freq_pct": float(counts.iloc[0] / non_null.size * 100.0),
        "entropy_bits": shannon_entropy(non_null),
        "pct_null": float((n - non_null.size) / n * 100.0) if n else 0.0,
    }
