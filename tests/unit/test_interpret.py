"""Pruebas de interpretabilidad (SHAP, permutación, PDP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tostao_ml.framework.evaluation import metrics
from tostao_ml.framework.interpret import (
    compute_shap,
    partial_dependence,
    permutation_bar,
    permutation_importance,
    shap_summary_bar,
)
from tostao_ml.framework.models import build_model


@pytest.fixture
def fitted_model() -> tuple[object, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 200
    x_relevante = rng.normal(size=n)
    x_ruido = rng.normal(size=n)
    y = pd.Series(4.0 * x_relevante + rng.normal(0, 0.1, n), name="y")
    X = pd.DataFrame({"x_relevante": x_relevante, "x_ruido": x_ruido})
    model = build_model("ridge").fit(X, y)
    return model, X, y


@pytest.mark.unit
def test_permutation_importance_ranks_relevant_first(fitted_model) -> None:
    model, X, y = fitted_model
    imp = permutation_importance(model, X, y, metrics.rmse, n_repeats=3)
    assert imp.index[0] == "x_relevante"
    assert imp["x_relevante"] > imp["x_ruido"]


@pytest.mark.unit
def test_partial_dependence_monotonic_for_linear(fitted_model) -> None:
    model, X, _ = fitted_model
    _grid, avg = partial_dependence(model, X, "x_relevante", grid_resolution=20)
    assert np.all(np.diff(avg) > 0)  # relación positiva ⇒ PDP creciente


@pytest.mark.unit
def test_compute_shap_and_figure(fitted_model) -> None:
    model, X, _ = fitted_model
    result = compute_shap(model, X, background_size=40, sample_size=60)
    assert result.values.shape[1] == X.shape[1]  # noqa: PD011 - atributo propio, no pandas
    assert result.mean_abs.index[0] == "x_relevante"
    assert isinstance(shap_summary_bar(result), go.Figure)


@pytest.mark.unit
def test_permutation_bar_figure(fitted_model) -> None:
    model, X, y = fitted_model
    imp = permutation_importance(model, X, y, metrics.rmse, n_repeats=2)
    assert isinstance(permutation_bar(imp), go.Figure)
