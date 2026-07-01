"""Ensamblado de figuras EDA a partir de un :class:`DatasetProfile`.

Mantiene ``viz/`` como fábrica pura de gráficos y ``profiling/`` como
orquestador: aquí se decide *qué* figura corresponde a cada variable según su
tipo, reutilizando las funciones de ``viz.eda``.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from tostao_ml.framework.types import VariableKind
from tostao_ml.framework.viz import eda

from .engine import DatasetProfile


def build_eda_figures(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    *,
    max_numeric: int = 8,
    max_categorical: int = 6,
) -> dict[str, go.Figure]:
    """Construye un diccionario de figuras EDA clave para reporte/notebook.

    Args:
        frame: Datos perfilados.
        profile: Perfil calculado por :func:`profile_dataset`.
        max_numeric: Máx. de variables numéricas a graficar individualmente.
        max_categorical: Máx. de categóricas a graficar (Pareto).

    Returns:
        Mapa ``clave → figura`` con distribuciones, Pareto, correlación,
        faltantes y (si hay target) distribución agrupada.
    """
    figures: dict[str, go.Figure] = {}
    numeric = [c for c, k in profile.types.items() if k.is_numeric][:max_numeric]
    categorical = [c for c, k in profile.types.items() if k is VariableKind.CATEGORICAL_NOMINAL][
        :max_categorical
    ]

    for col in numeric:
        figures[f"dist__{col}"] = eda.histogram_kde(frame[col], name=col)

    for col in categorical:
        figures[f"pareto__{col}"] = eda.pareto(frame[col].value_counts(), name=col)

    pearson = profile.correlations.get("pearson")
    if pearson is not None and not pearson.empty:
        figures["correlation_heatmap"] = eda.correlation_heatmap(pearson)

    if frame.isna().any().any():
        figures["missingness"] = eda.missingness_bar(frame)

    target = profile.target
    if target is not None and target in frame.columns and profile.types[target].is_categorical:
        for col in numeric:
            if col != target:
                figures[f"target__{col}"] = eda.grouped_distribution(frame, col, target)
                break

    return figures
