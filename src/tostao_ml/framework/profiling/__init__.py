"""Motor de EDA reutilizable (§4): tipado, univariado, bivariado, multivariado.

Ejecuta el mismo análisis estadístico para A/B/C parametrizado por el target y
emite un perfil con conclusiones autogeneradas. Los submódulos (``univariate``,
``bivariate``, ``associations``) contienen las funciones estadísticas puras.
"""

from __future__ import annotations

from . import associations, bivariate, univariate
from .engine import DatasetProfile, profile_dataset
from .figures import (
    build_correlation_figures,
    build_eda_figures,
    build_target_analysis,
    build_univariate_figures,
    by_target_figure,
    categorical_summary_table,
    numeric_summary_table,
    variable_dictionary,
)
from .typing import TypingConfig, group_by_kind, infer_variable_types

__all__ = [
    "DatasetProfile",
    "TypingConfig",
    "associations",
    "bivariate",
    "build_correlation_figures",
    "build_eda_figures",
    "build_target_analysis",
    "build_univariate_figures",
    "by_target_figure",
    "categorical_summary_table",
    "group_by_kind",
    "infer_variable_types",
    "numeric_summary_table",
    "profile_dataset",
    "univariate",
    "variable_dictionary",
]
