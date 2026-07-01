"""Pruebas de validación de contratos e integridad referencial."""

from __future__ import annotations

import pandas as pd
import pytest

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa

from tostao_ml.framework.validation import check_referential_integrity, validate_dataframe


@pytest.mark.data_contract
def test_schema_validation_passes_on_valid_data() -> None:
    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(str, unique=True),
            "precio": pa.Column(float, pa.Check.ge(0)),
        }
    )
    df = pd.DataFrame({"id": ["a", "b"], "precio": [1.0, 2.5]})
    result = validate_dataframe(df, schema, name="productos")
    assert result.passed
    assert result.narrative.worst_severity.value == "good"


@pytest.mark.data_contract
def test_schema_validation_flags_negative_price() -> None:
    schema = pa.DataFrameSchema({"precio": pa.Column(float, pa.Check.ge(0))})
    df = pd.DataFrame({"precio": [1.0, -5.0, -1.0]})
    result = validate_dataframe(df, schema, name="precios")
    assert not result.passed
    assert result.n_failures >= 1


@pytest.mark.data_contract
def test_referential_integrity_detects_orphans() -> None:
    parent = pd.DataFrame({"id_producto": ["P1", "P2", "P3"]})
    child = pd.DataFrame({"id_producto": ["P1", "P2", "P9", "P9"]})
    result = check_referential_integrity(
        child, parent, "id_producto", "id_producto", name="detalle→catalogo"
    )
    assert not result.passed
    assert result.n_failures == 2  # dos filas huérfanas (P9, P9)
    assert "P9" in result.failure_cases["id_producto"].tolist()


@pytest.mark.data_contract
def test_referential_integrity_ok() -> None:
    parent = pd.DataFrame({"id": ["P1", "P2"]})
    child = pd.DataFrame({"id": ["P1", "P1", "P2"]})
    result = check_referential_integrity(child, parent, "id", "id", name="ok")
    assert result.passed
