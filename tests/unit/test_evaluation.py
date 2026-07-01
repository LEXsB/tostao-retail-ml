"""Pruebas de métricas, validación y gráficos de desempeño."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tostao_ml.framework.evaluation import metrics, performance, time_series_splitter
from tostao_ml.framework.evaluation.validation import temporal_holdout


@pytest.mark.unit
def test_regression_metrics() -> None:
    y = np.array([10.0, 20.0, 30.0, 40.0])
    yhat = np.array([12.0, 18.0, 33.0, 39.0])
    assert metrics.mae(y, yhat) == pytest.approx(2.0)
    assert metrics.wape(y, yhat) == pytest.approx(8 / 100)
    report = metrics.regression_report(y, yhat)
    assert {"mae", "rmse", "wape", "smape", "r2"} == set(report)


@pytest.mark.unit
def test_pinball_and_interval_metrics() -> None:
    y = np.array([5.0, 5.0, 5.0])
    # Cuantil 0.9 subestimando penaliza más el lado de subestimación.
    assert metrics.pinball_loss(y, np.array([4.0, 4.0, 4.0]), 0.9) == pytest.approx(0.9)
    lower = np.array([3.0, 3.0, 3.0])
    upper = np.array([7.0, 7.0, 7.0])
    assert metrics.picp(y, lower, upper) == 1.0
    assert metrics.mpiw(lower, upper) == pytest.approx(4.0)


@pytest.mark.unit
def test_classification_report_and_ks() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.6, 0.9])
    report = metrics.classification_report(y, (score > 0.5).astype(int), score)
    assert report["roc_auc"] == 1.0
    assert 0.0 <= report["ks"] <= 1.0


@pytest.mark.unit
def test_time_series_splitter_is_expanding() -> None:
    X = pd.DataFrame({"x": range(20)})
    splits = list(time_series_splitter(n_splits=3)(X))
    assert len(splits) == 3
    # cada train usa solo índices anteriores al val (sin leakage)
    for train_idx, val_idx in splits:
        assert train_idx.max() < val_idx.min()


@pytest.mark.unit
def test_temporal_holdout_selects_tail() -> None:
    df = pd.DataFrame({"fecha": pd.date_range("2024-01-01", periods=10)})
    mask = temporal_holdout(df, "fecha", test_fraction=0.3)
    assert mask.sum() == 3
    assert mask[-3:].all()  # el bloque final


@pytest.mark.unit
def test_performance_figures() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yhat = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    score = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    label = np.array([0, 0, 1, 1, 1])
    figs = [
        performance.pred_vs_actual(y, yhat),
        performance.residuals_vs_pred(y, yhat),
        performance.roc(label, score),
        performance.precision_recall(label, score),
        performance.confusion(label, (score > 0.5).astype(int)),
        performance.cumulative_gain(label, score),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)
