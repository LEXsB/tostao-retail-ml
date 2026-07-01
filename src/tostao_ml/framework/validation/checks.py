"""Validación de contratos de datos con Pandera + chequeos de integridad.

Fallar rápido ante datos inválidos: tipos, rangos, nulos, unicidad e integridad
referencial entre tiendas/productos/tickets/clientes.
"""

from __future__ import annotations

import pandas as pd

try:  # Pandera >= 0.20 separa la API de pandas en ``pandera.pandas``.
    import pandera.pandas as pa
    from pandera.errors import SchemaErrors
except ImportError:  # pragma: no cover - compatibilidad con versiones antiguas
    import pandera as pa  # type: ignore[no-redef]
    from pandera.errors import SchemaErrors

from .result import ValidationResult


def validate_dataframe(
    frame: pd.DataFrame, schema: pa.DataFrameSchema, name: str
) -> ValidationResult:
    """Valida un DataFrame contra un esquema Pandera de forma perezosa.

    Args:
        frame: Datos a validar.
        schema: Contrato Pandera.
        name: Nombre legible del dataset/contrato.

    Returns:
        :class:`ValidationResult` con casos de falla y narrativa.
    """
    n_checks = _count_checks(schema)
    try:
        schema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases if isinstance(exc.failure_cases, pd.DataFrame) else pd.DataFrame()
        result = ValidationResult(
            name=name,
            passed=False,
            n_checks=n_checks,
            n_failures=len(cases),
            failure_cases=cases,
        )
        result.build_narrative()
        return result
    result = ValidationResult(name=name, passed=True, n_checks=n_checks)
    result.build_narrative()
    return result


def check_referential_integrity(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    child_key: str,
    parent_key: str,
    name: str,
) -> ValidationResult:
    """Verifica que toda clave foránea del hijo exista en el padre.

    Args:
        child: Tabla con la clave foránea (p. ej. detalle de tickets).
        parent: Tabla de referencia (p. ej. catálogo de productos).
        child_key: Columna FK en ``child``.
        parent_key: Columna PK en ``parent``.
        name: Nombre legible de la comprobación.

    Returns:
        :class:`ValidationResult`; los casos de falla son las claves huérfanas.
    """
    valid_keys = set(parent[parent_key].dropna().unique())
    child_keys = child[child_key].dropna()
    orphan_mask = ~child_keys.isin(valid_keys)
    n_orphans = int(orphan_mask.sum())
    orphans = (
        child_keys[orphan_mask].value_counts().rename_axis(child_key).reset_index(name="conteo")
        if n_orphans
        else pd.DataFrame(columns=[child_key, "conteo"])
    )
    result = ValidationResult(
        name=name,
        passed=n_orphans == 0,
        n_checks=1,
        n_failures=n_orphans,
        failure_cases=orphans,
    )
    result.build_narrative()
    return result


def _count_checks(schema: pa.DataFrameSchema) -> int:
    """Cuenta comprobaciones declaradas en un esquema (columnas + checks)."""
    total = len(schema.columns)
    for col in schema.columns.values():
        total += len(getattr(col, "checks", []) or [])
    return total
