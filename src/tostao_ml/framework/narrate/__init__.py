"""Motor de narración: mini-conclusiones analíticas autogeneradas.

Recibe las salidas numéricas de un nodo y produce texto con umbrales/reglas
(p. ej. «VIF>10 ⇒ multicolinealidad severa en X»). Alimenta el reporte HTML.
"""

from __future__ import annotations

from . import rules
from .insight import Insight, Narrative, Severity
from .thresholds import NarrationThresholds

__all__ = [
    "Insight",
    "NarrationThresholds",
    "Narrative",
    "Severity",
    "rules",
]
