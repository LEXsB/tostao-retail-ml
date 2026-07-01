"""Motor de HPO reutilizable (Optuna): estudios, espacios y visualización."""

from __future__ import annotations

from .engine import TuningResult, make_pruner, make_sampler, tune_model
from .figures import build_hpo_figures
from .spaces import suggest_params

__all__ = [
    "TuningResult",
    "build_hpo_figures",
    "make_pruner",
    "make_sampler",
    "suggest_params",
    "tune_model",
]
