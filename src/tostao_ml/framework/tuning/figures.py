"""Figuras de HPO (Plotly) a partir de un estudio Optuna.

Envuelve ``optuna.visualization`` aplicando el tema de marca, de modo que los
gráficos de tuning se integran con el resto del reporte.
"""

from __future__ import annotations

from collections.abc import Callable

import optuna
import plotly.graph_objects as go
from optuna import visualization as viz

from tostao_ml.framework.viz.theme import apply_theme


def build_hpo_figures(study: optuna.Study) -> dict[str, go.Figure]:
    """Construye las figuras clave de un estudio de HPO.

    Incluye historia de optimización, importancia de hiperparámetros,
    coordenadas paralelas y slice plot. Cada figura es tolerante a fallos
    (estudios con pocos trials) y se omite si no puede generarse.

    Args:
        study: Estudio Optuna finalizado.

    Returns:
        Mapa ``clave → figura`` con las visualizaciones disponibles.
    """
    figures: dict[str, go.Figure] = {}
    builders: dict[str, Callable[[optuna.Study], go.Figure]] = {
        "hpo_history": viz.plot_optimization_history,
        "hpo_importances": viz.plot_param_importances,
        "hpo_parallel": viz.plot_parallel_coordinate,
        "hpo_slice": viz.plot_slice,
    }
    for key, builder in builders.items():
        try:
            figures[key] = apply_theme(builder(study))
        except (
            ValueError,
            RuntimeError,
            ZeroDivisionError,
        ):  # pragma: no cover - estudio degenerado
            continue
    return figures
