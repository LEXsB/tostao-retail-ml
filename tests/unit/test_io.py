"""Pruebas de utilidades de reproducibilidad y hashing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.framework.io import hash_dataframe, set_global_seed, short_hash


@pytest.mark.unit
def test_hash_is_deterministic_and_column_order_invariant() -> None:
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = df1[["b", "a"]]  # mismo contenido, columnas reordenadas
    assert hash_dataframe(df1) == hash_dataframe(df2)
    assert len(hash_dataframe(df1)) == 64


@pytest.mark.unit
def test_hash_changes_with_content() -> None:
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 4]})
    assert hash_dataframe(df1) != hash_dataframe(df2)
    assert len(short_hash(df1)) == 12


@pytest.mark.unit
def test_set_global_seed_makes_numpy_reproducible() -> None:
    set_global_seed(123)
    first = np.random.rand(5)  # noqa: NPY002 - se valida justamente el RandomState global
    set_global_seed(123)
    second = np.random.rand(5)  # noqa: NPY002
    np.testing.assert_array_equal(first, second)
