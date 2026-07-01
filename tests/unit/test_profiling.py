"""Pruebas del motor de EDA (tipado, estadística y orquestación)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.framework.profiling import (
    associations,
    bivariate,
    infer_variable_types,
    profile_dataset,
    univariate,
)
from tostao_ml.framework.types import VariableKind


@pytest.fixture
def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    grupo = rng.choice(["A", "B"], n)
    valor = np.where(grupo == "A", rng.normal(10, 2, n), rng.normal(14, 2, n))
    return pd.DataFrame(
        {
            "id_cliente": [f"C{i:05d}" for i in range(n)],
            "fecha": pd.date_range("2024-01-01", periods=n, freq="h"),
            "grupo": grupo,
            "valor": valor,
            "redundante": valor * 2.0 + rng.normal(0, 1e-6, n),
            "conteo": rng.integers(0, 5, n),
            "target_bin": (valor > 12).astype(int),
        }
    )


@pytest.mark.unit
def test_type_inference(synthetic: pd.DataFrame) -> None:
    types = infer_variable_types(synthetic)
    assert types["id_cliente"] is VariableKind.IDENTIFIER
    assert types["fecha"] is VariableKind.TEMPORAL
    assert types["grupo"] is VariableKind.BINARY
    assert types["valor"] is VariableKind.NUMERIC_CONTINUOUS
    assert types["conteo"] is VariableKind.NUMERIC_DISCRETE
    assert types["target_bin"] is VariableKind.BINARY


@pytest.mark.unit
def test_type_override_wins(synthetic: pd.DataFrame) -> None:
    types = infer_variable_types(synthetic, overrides={"conteo": "categorical_ordinal"})
    assert types["conteo"] is VariableKind.CATEGORICAL_ORDINAL


@pytest.mark.unit
def test_numeric_summary_and_outliers() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    summary = univariate.numeric_summary(s)
    assert summary["min"] == 1.0 and summary["max"] == 100.0
    assert univariate.outlier_fraction_iqr(s) > 0


@pytest.mark.unit
def test_vif_detects_perfect_collinearity() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"a": x, "b": x * 3.0, "c": rng.normal(size=200)})
    vif = associations.vif_scores(df)
    assert vif["a"] > 10 and vif["b"] > 10


@pytest.mark.unit
def test_cramers_v_and_correlation_ratio() -> None:
    x = pd.Series(["a", "a", "b", "b"] * 25)
    y = pd.Series(["x", "x", "y", "y"] * 25)
    v, pvalue = associations.cramers_v(x, y)
    assert v > 0.9 and pvalue < 0.05
    cats = pd.Series(["a"] * 50 + ["b"] * 50)
    vals = pd.Series(list(np.zeros(50)) + list(np.ones(50)))
    assert associations.correlation_ratio(cats, vals) > 0.9


@pytest.mark.unit
def test_bivariate_group_difference() -> None:
    a = pd.Series(np.random.default_rng(1).normal(0, 1, 100))
    groups = pd.Series(["g1"] * 50 + ["g2"] * 50)
    values = pd.concat([a[:50], a[50:] + 5], ignore_index=True)
    res = bivariate.test_numeric_vs_categorical(values, groups)
    assert res["effect_name"] == "Cohen's d"
    assert res["pvalue"] < 0.05


@pytest.mark.unit
def test_profile_dataset_end_to_end(synthetic: pd.DataFrame) -> None:
    profile = profile_dataset(synthetic, name="synthetic", target="target_bin")
    assert profile.n_rows == 400
    assert len(profile.narrative) > 0
    # La colinealidad valor↔redundante debe aparecer como par redundante.
    pairs = profile.redundant_pairs(threshold=0.95)
    assert any({"valor", "redundante"} == {a, b} for a, b, _ in pairs)
    # El bivariado debe incluir a «valor» frente al target.
    assert "valor" in profile.bivariate["feature"].tolist()
