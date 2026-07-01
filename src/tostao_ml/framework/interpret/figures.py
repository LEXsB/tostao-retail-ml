"""Figuras Plotly de interpretabilidad (SHAP, permutación, PDP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tostao_ml.framework.viz.theme import COLORS, apply_theme

from .explain import ShapResult


def shap_summary_bar(result: ShapResult, top: int = 15) -> go.Figure:
    """Importancia global SHAP: media de |valor SHAP| por feature."""
    mean_abs = result.mean_abs.head(top)[::-1]
    fig = go.Figure(
        go.Bar(
            x=mean_abs.to_numpy(),
            y=list(mean_abs.index),
            orientation="h",
            marker_color=COLORS["primary"],
        )
    )
    fig.update_xaxes(title="media(|SHAP|)")
    return apply_theme(fig, title="Importancia SHAP (global)")


def shap_beeswarm(result: ShapResult, top: int = 12) -> go.Figure:
    """Beeswarm SHAP en Plotly: dispersión de valores SHAP coloreada por el valor del feature."""
    order = result.mean_abs.head(top).index[::-1]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for i, feat in enumerate(order):
        idx = list(result.sample.columns).index(feat)
        shap_vals = result.values[:, idx]  # noqa: PD011 - atributo propio, no pandas
        feat_vals = result.sample[feat].to_numpy(dtype=float)
        jitter = i + rng.uniform(-0.18, 0.18, size=shap_vals.size)
        fig.add_scatter(
            x=shap_vals,
            y=jitter,
            mode="markers",
            marker={
                "color": feat_vals,
                "colorscale": "RdBu",
                "reversescale": True,
                "size": 6,
                "opacity": 0.7,
                "colorbar": {"title": "valor"} if i == len(order) - 1 else None,
            },
            name=str(feat),
            showlegend=False,
            hovertext=[str(feat)] * shap_vals.size,
        )
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(order))), ticktext=[str(f) for f in order]
    )
    fig.update_xaxes(title="Valor SHAP (impacto en la predicción)")
    fig.add_vline(x=0, line={"color": COLORS["neutral"], "dash": "dash"})
    return apply_theme(fig, title="SHAP beeswarm")


def permutation_bar(importances: pd.Series, top: int = 15) -> go.Figure:
    """Barras de importancia por permutación."""
    imp = importances.head(top)[::-1]
    fig = go.Figure(
        go.Bar(
            x=imp.to_numpy(), y=list(imp.index), orientation="h", marker_color=COLORS["secondary"]
        )
    )
    fig.update_xaxes(title="Degradación de la métrica")
    return apply_theme(fig, title="Importancia por permutación")


def pdp_figure(grid: np.ndarray, avg: np.ndarray, feature: str) -> go.Figure:
    """Curva de dependencia parcial 1D."""
    fig = go.Figure(go.Scatter(x=grid, y=avg, mode="lines", line={"color": COLORS["accent"]}))
    fig.update_xaxes(title=feature)
    fig.update_yaxes(title="Predicción media")
    return apply_theme(fig, title=f"Dependencia parcial: {feature}")
