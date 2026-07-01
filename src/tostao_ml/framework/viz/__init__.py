"""Fábrica de gráficos Plotly reutilizables (EDA + desempeño + negocio).

Punto único para producir figuras con estética consistente. Las funciones de
desempeño se añaden en ``performance`` (ver ``evaluation``) y las de negocio en
``business``; aquí se exponen el tema, la serialización y las figuras de EDA.
"""

from __future__ import annotations

from . import eda
from .serialize import figure_to_div, plotly_js_bundle, save_figure
from .theme import COLORS, QUALITATIVE, apply_theme, register_theme

__all__ = [
    "COLORS",
    "QUALITATIVE",
    "apply_theme",
    "eda",
    "figure_to_div",
    "plotly_js_bundle",
    "register_theme",
    "save_figure",
]
