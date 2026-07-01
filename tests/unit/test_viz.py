"""Pruebas de la fábrica de gráficos Plotly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tostao_ml.framework.viz import eda, figure_to_div
from tostao_ml.framework.viz.theme import TEMPLATE_NAME


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "valor": rng.normal(10, 2, 200),
            "grupo": rng.choice(["A", "B", "C"], 200),
            "target": rng.choice(["si", "no"], 200),
        }
    )


@pytest.mark.unit
def test_eda_figures_return_themed_figures(frame: pd.DataFrame) -> None:
    figs = [
        eda.histogram_kde(frame["valor"]),
        eda.box_violin(frame["valor"]),
        eda.ecdf(frame["valor"]),
        eda.qq_plot(frame["valor"]),
        eda.pareto(frame["grupo"].value_counts()),
        eda.grouped_distribution(frame, "valor", "grupo"),
        eda.stacked_proportions(frame, "grupo", "target"),
        eda.missingness_bar(frame),
    ]
    for fig in figs:
        assert isinstance(fig, go.Figure)
        assert fig.layout.template is not None


@pytest.mark.unit
def test_correlation_heatmap(frame: pd.DataFrame) -> None:
    corr = frame[["valor"]].assign(otro=frame["valor"] * 2).corr()
    fig = eda.correlation_heatmap(corr)
    assert isinstance(fig, go.Figure)


@pytest.mark.unit
def test_figure_to_div_is_selfstanding_html(frame: pd.DataFrame) -> None:
    div = figure_to_div(eda.histogram_kde(frame["valor"]), div_id="fig1")
    assert isinstance(div, str)
    assert "fig1" in div
    assert TEMPLATE_NAME  # tema registrado
