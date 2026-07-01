"""Pruebas del motor de HPO (Optuna)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from tostao_ml.framework.models.base import BaseModel
from tostao_ml.framework.tuning import build_hpo_figures, make_pruner, make_sampler, tune_model


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    y = 2.0 * x1 - x2 + rng.normal(0, 0.1, 200)
    return pd.DataFrame({"x1": x1, "x2": x2}), pd.Series(y, name="y")


def _mse_scorer(model: BaseModel, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    return float(mean_squared_error(y_val, model.predict(X_val)))


@pytest.mark.unit
def test_make_sampler_and_pruner_validation() -> None:
    assert make_sampler("tpe", 42) is not None
    assert make_pruner("median") is not None
    with pytest.raises(ValueError, match="Sampler"):
        make_sampler("inexistente", 1)
    with pytest.raises(ValueError, match="Pruner"):
        make_pruner("inexistente")


@pytest.mark.unit
def test_tune_model_optimizes_ridge(data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = data
    space = {"alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True}}
    result = tune_model(
        "ridge",
        space,
        X,
        y,
        scorer=_mse_scorer,
        splitter=lambda frame: KFold(n_splits=3, shuffle=True, random_state=0).split(frame),
        direction="minimize",
        n_trials=10,
        sampler="tpe",
        pruner="none",
        seed=42,
    )
    assert len(result.study.trials) == 10
    assert "alpha" in result.best_params
    assert np.isfinite(result.best_value)
    assert len(result.narrative) >= 1


@pytest.mark.unit
def test_hpo_figures_build(data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = data
    space = {"alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True}}
    result = tune_model(
        "ridge",
        space,
        X,
        y,
        scorer=_mse_scorer,
        splitter=lambda frame: KFold(n_splits=3).split(frame),
        n_trials=8,
        pruner="none",
    )
    figures = build_hpo_figures(result.study)
    assert isinstance(figures, dict)
    assert "hpo_history" in figures
