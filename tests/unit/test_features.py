"""Pruebas de los transformadores de features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.framework.features import (
    CorrelationSelector,
    CyclicalEncoder,
    DateTimeFeatures,
    FrequencyEncoder,
    GroupLagFeatures,
    RFMTransformer,
    VIFSelector,
    Winsorizer,
    build_column_transformer,
)
from tostao_ml.framework.types import VariableKind


@pytest.mark.unit
def test_datetime_features() -> None:
    df = pd.DataFrame({"fecha": pd.to_datetime(["2024-01-06", "2024-07-15"]), "x": [1, 2]})
    out = DateTimeFeatures("fecha").fit_transform(df)
    assert "fecha_year" in out.columns and "fecha" not in out.columns
    assert out["fecha_is_weekend"].tolist() == [1, 0]  # 2024-01-06 es sábado
    assert out["fecha_quarter"].tolist() == [1, 3]


@pytest.mark.unit
def test_cyclical_encoder_is_continuous_across_boundary() -> None:
    df = pd.DataFrame({"mes": [12, 1]})
    out = CyclicalEncoder(["mes"], {"mes": 12}, drop_original=False).fit_transform(df)
    # dic (12) y ene (1) deben quedar cercanos en el espacio seno/coseno
    dist = np.hypot(
        out["mes_sin"].iloc[0] - out["mes_sin"].iloc[1],
        out["mes_cos"].iloc[0] - out["mes_cos"].iloc[1],
    )
    assert dist < 0.6


@pytest.mark.unit
def test_group_lag_features_no_leakage() -> None:
    df = pd.DataFrame(
        {
            "sku": ["A", "A", "A", "B", "B"],
            "fecha": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-01", "2024-01-08"]
            ),
            "y": [10, 20, 30, 100, 200],
        }
    )
    out = GroupLagFeatures(["sku"], "fecha", "y", lags=(1,), rolling_windows=(2,)).fit_transform(df)
    a = out[out["sku"] == "A"].sort_values("fecha")
    assert np.isnan(a["y_lag_1"].iloc[0])  # primer valor sin rezago
    assert a["y_lag_1"].iloc[1] == 10  # rezago dentro del grupo
    assert np.isnan(a["y_rollmean_2"].iloc[0])  # media rodante no usa el actual


@pytest.mark.unit
def test_rfm_transformer() -> None:
    df = pd.DataFrame(
        {
            "id_cliente": ["c1", "c1", "c2"],
            "fecha": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-05"]),
            "monto": [100.0, 50.0, 200.0],
        }
    )
    rfm = RFMTransformer("id_cliente", "fecha", "monto", score=True).fit_transform(df)
    assert rfm.loc["c1", "frequency"] == 2
    assert rfm.loc["c1", "monetary"] == 150.0
    assert {"r_score", "f_score", "m_score", "rfm_score"}.issubset(rfm.columns)


@pytest.mark.unit
def test_frequency_encoder_handles_unseen() -> None:
    train = pd.DataFrame({"cat": ["a", "a", "b"]})
    enc = FrequencyEncoder(["cat"]).fit(train)
    out = enc.transform(pd.DataFrame({"cat": ["a", "z"]}))
    assert out["cat_freq"].iloc[0] == pytest.approx(2 / 3)
    assert out["cat_freq"].iloc[1] == 0.0  # categoría no vista


@pytest.mark.unit
def test_winsorizer_clips_with_train_bounds() -> None:
    train = pd.DataFrame({"x": list(range(100))})
    w = Winsorizer(["x"], 0.05, 0.95).fit(train)
    out = w.transform(pd.DataFrame({"x": [-50, 500]}))
    assert out["x"].iloc[0] >= train["x"].quantile(0.05)
    assert out["x"].iloc[1] <= train["x"].quantile(0.95)


@pytest.mark.unit
def test_vif_and_correlation_selectors_drop_redundant() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    df = pd.DataFrame({"a": base, "b": base * 2 + 1e-9, "c": rng.normal(size=200)})
    assert (
        "b" in VIFSelector(threshold=10.0).fit(df).dropped_
        or "a" in VIFSelector(threshold=10.0).fit(df).dropped_
    )
    assert CorrelationSelector(threshold=0.95).fit(df).dropped_ == ["b"]


@pytest.mark.unit
def test_build_column_transformer_shapes() -> None:
    df = pd.DataFrame(
        {"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"], "target": [0, 1, 0, 1]}
    )
    types = {
        "num": VariableKind.NUMERIC_CONTINUOUS,
        "cat": VariableKind.CATEGORICAL_NOMINAL,
        "target": VariableKind.BINARY,
    }
    ct = build_column_transformer(types, exclude={"target"})
    arr = ct.fit_transform(df)
    assert arr.shape[0] == 4
    names = list(ct.get_feature_names_out())
    assert "num" in names
