"""Pruebas de modelos, registro y persistencia."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.framework.models import (
    GLMModel,
    KMeansModel,
    QuantileGBRModel,
    available_models,
    build_model,
)
from tostao_ml.framework.models.registry import register_model


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 3.0 * x1 - 2.0 * x2 + rng.normal(0, 0.1, n) + 10
    return pd.DataFrame({"x1": x1, "x2": x2}), pd.Series(y, name="y")


@pytest.mark.unit
def test_registry_build_and_list() -> None:
    assert {"ridge", "gbr", "quantile_gbr", "kmeans", "glm"}.issubset(set(available_models()))
    model = build_model("ridge", alpha=0.5)
    assert model.name == "ridge"
    with pytest.raises(KeyError):
        build_model("no_existe")


@pytest.mark.unit
def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="ya registrado"):

        @register_model("ridge")
        class _Dup:  # pragma: no cover - solo dispara el error
            pass


@pytest.mark.unit
def test_ridge_fit_predict_and_persistence(regression_data, tmp_path) -> None:
    X, y = regression_data
    model = build_model("ridge", alpha=0.1).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.corrcoef(preds, y)[0, 1] > 0.99
    path = model.save(tmp_path / "ridge.joblib")
    reloaded = type(model).load(path)
    np.testing.assert_allclose(reloaded.predict(X), preds)


@pytest.mark.unit
def test_quantile_model_intervals(regression_data) -> None:
    X, y = regression_data
    model = QuantileGBRModel(quantiles=(0.1, 0.5, 0.9), max_iter=80).fit(X, y)
    interval = model.predict_interval(X, coverage=0.8)
    assert np.all(interval["lower"] <= interval["upper"] + 1e-9)
    assert np.all(interval["median"] >= interval["lower"] - 1e-9)
    quantiles = model.predict_quantiles(X)
    assert list(quantiles.columns) == ["q0.1", "q0.5", "q0.9"]


@pytest.mark.unit
def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        build_model("gbr").predict(pd.DataFrame({"x": [1.0]}))


@pytest.mark.unit
def test_kmeans_labels_and_quality() -> None:
    rng = np.random.default_rng(1)
    blob_a = rng.normal(0, 0.3, (60, 2))
    blob_b = rng.normal(5, 0.3, (60, 2))
    X = pd.DataFrame(np.vstack([blob_a, blob_b]), columns=["f1", "f2"])
    model = KMeansModel(n_clusters=2).fit(X)
    labels = model.predict(X)
    assert set(labels) == {0, 1}
    assert model.metadata.extra["silhouette"] > 0.7


@pytest.mark.unit
def test_glm_recovers_coefficient_signs(regression_data) -> None:
    X, y = regression_data
    model = GLMModel(family="gaussian").fit(X, y)
    coefs = model.coefficients_frame()
    assert coefs.loc["x1", "coef"] > 0  # relación positiva conocida
    assert coefs.loc["x2", "coef"] < 0  # relación negativa conocida
    assert {"coef", "std_err", "pvalue", "ci_lower", "ci_upper"}.issubset(coefs.columns)
