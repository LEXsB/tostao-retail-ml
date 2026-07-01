"""Interpretabilidad reutilizable: SHAP, permutación y PDP + sus figuras."""

from __future__ import annotations

from .explain import ShapResult, compute_shap, partial_dependence, permutation_importance
from .figures import pdp_figure, permutation_bar, shap_beeswarm, shap_summary_bar

__all__ = [
    "ShapResult",
    "compute_shap",
    "partial_dependence",
    "pdp_figure",
    "permutation_bar",
    "permutation_importance",
    "shap_beeswarm",
    "shap_summary_bar",
]
