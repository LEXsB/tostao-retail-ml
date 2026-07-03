"""Evaluación: métricas, validadores CV/temporales, gráficos de desempeño y comparación."""

from __future__ import annotations

from . import metrics, performance
from .comparison import compare_models, narrate_comparison
from .validation import (
    kfold_splitter,
    temporal_holdout,
    time_series_splitter,
)

__all__ = [
    "compare_models",
    "kfold_splitter",
    "metrics",
    "narrate_comparison",
    "performance",
    "temporal_holdout",
    "time_series_splitter",
]
