"""Motor de EDA reutilizable (§4): tipado, univariado, bivariado, multivariado.

Ejecuta el mismo análisis estadístico para A/B/C parametrizado por el target y
emite un perfil con conclusiones autogeneradas. Los submódulos (``univariate``,
``bivariate``, ``associations``) contienen las funciones estadísticas puras.
"""

from __future__ import annotations

from . import associations, bivariate, univariate
from .engine import DatasetProfile, profile_dataset
from .figures import build_eda_figures
from .typing import TypingConfig, group_by_kind, infer_variable_types

__all__ = [
    "DatasetProfile",
    "TypingConfig",
    "associations",
    "bivariate",
    "build_eda_figures",
    "group_by_kind",
    "infer_variable_types",
    "profile_dataset",
    "univariate",
]
