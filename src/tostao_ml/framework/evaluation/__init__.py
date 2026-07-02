"""Evaluación: métricas, validadores CV/temporales, gráficos de desempeño y KPIs."""

from __future__ import annotations

from . import metrics, performance
from .comparison import compare_models, narrate_comparison
from .validation import (
    kfold_splitter,
    stratified_kfold_splitter,
    temporal_holdout,
    time_series_splitter,
)

__all__ = [
    "compare_models",
    "kfold_splitter",
    "metrics",
    "narrate_comparison",
    "performance",
    "stratified_kfold_splitter",
    "temporal_holdout",
    "time_series_splitter",
]
