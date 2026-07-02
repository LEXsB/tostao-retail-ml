"""Pruebas del ensemble y del comparador de modelos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.framework.evaluation import compare_models, narrate_comparison
from tostao_ml.framework.models import AveragingEnsemble, GBRRegressionModel, RidgeRegressionModel


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2] ** 2 + rng.normal(0, 0.1, 400)
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    return df.iloc[:300], pd.Series(y[:300]), df.iloc[300:], pd.Series(y[300:])


@pytest.mark.unit
def test_ensemble_averages_predictions(data) -> None:
    X_tr, y_tr, X_te, _ = data
    ridge = RidgeRegressionModel().fit(X_tr, y_tr)
    gbr = GBRRegressionModel(max_iter=80).fit(X_tr, y_tr)
    ens = AveragingEnsemble([("ridge", ridge), ("gbr", gbr)]).fit(X_tr, y_tr)
    expected = (ridge.predict(X_te) + gbr.predict(X_te)) / 2
    np.testing.assert_allclose(ens.predict(X_te), expected, rtol=1e-6)


@pytest.mark.unit
def test_compare_models_returns_sorted_table(data) -> None:
    X_tr, y_tr, X_te, y_te = data
    models = {
        "ridge": RidgeRegressionModel(),
        "gbr": GBRRegressionModel(max_iter=120),
        "ensemble": AveragingEnsemble(
            [("r", RidgeRegressionModel()), ("g", GBRRegressionModel(max_iter=120))]
        ),
    }
    table = compare_models(models, X_tr, y_tr, X_te, y_te, sort_by="wape")
    assert list(table.index) == sorted(table.index, key=lambda m: table.loc[m, "wape"])
    assert {"mae", "rmse", "wape", "r2"}.issubset(table.columns)
    # En datos no lineales, GBR o ensemble deberían superar a Ridge.
    assert table.iloc[0].name in {"gbr", "ensemble"}


@pytest.mark.unit
def test_narrate_comparison(data) -> None:
    X_tr, y_tr, X_te, y_te = data
    models = {"ridge": RidgeRegressionModel(), "gbr": GBRRegressionModel(max_iter=120)}
    table = compare_models(models, X_tr, y_tr, X_te, y_te)
    insight = narrate_comparison(table, "ridge")
    assert "modelos" in insight.text.lower()
