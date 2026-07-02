"""Pruebas de utilidades de reproducibilidad."""

from __future__ import annotations

import numpy as np
import pytest

from tostao_ml.framework.io import set_global_seed


@pytest.mark.unit
def test_set_global_seed_makes_numpy_reproducible() -> None:
    set_global_seed(123)
    first = np.random.rand(5)  # noqa: NPY002 - se valida justamente el RandomState global
    set_global_seed(123)
    second = np.random.rand(5)  # noqa: NPY002
    np.testing.assert_array_equal(first, second)
