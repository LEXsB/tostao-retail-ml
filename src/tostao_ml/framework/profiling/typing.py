"""Tipado automático de variables (§4.1 del diseño de EDA).

Clasifica cada columna en un :class:`VariableKind` combinando dtype, cardinalidad
y heurísticas de nombre. El tipo decide qué estadística, qué test y qué gráfico
se aplican, y admite override explícito por configuración (YAML).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api import types as pdt

from tostao_ml.framework.types import VariableKind


@dataclass(frozen=True, slots=True)
class TypingConfig:
    """Umbrales de las heurísticas de tipado.

    Attributes:
        max_discrete_unique: Máx. de valores únicos para tratar un entero como
            conteo/discreto (por encima se considera continuo).
        identifier_unique_ratio: Ratio nunique/n a partir del cual una columna de
            alta cardinalidad se considera identificador.
        text_avg_len: Longitud media de cadena a partir de la cual es texto libre.
        max_categorical_unique: Máx. de niveles para seguir siendo categórica
            nominal (por encima, si no es id/texto, se marca alta cardinalidad).
    """

    max_discrete_unique: int = 20
    identifier_unique_ratio: float = 0.9
    text_avg_len: float = 40.0
    max_categorical_unique: int = 50


def infer_variable_types(
    frame: pd.DataFrame,
    overrides: dict[str, str | VariableKind] | None = None,
    config: TypingConfig | None = None,
) -> dict[str, VariableKind]:
    """Infiere el :class:`VariableKind` de cada columna del DataFrame.

    Args:
        frame: Datos a tipar.
        overrides: Mapa columna→tipo forzado (gana sobre la heurística).
        config: Umbrales de las heurísticas.

    Returns:
        Diccionario columna → :class:`VariableKind`.
    """
    config = config or TypingConfig()
    normalized: dict[str, VariableKind] = {k: VariableKind(v) for k, v in (overrides or {}).items()}
    result: dict[str, VariableKind] = {}
    n = len(frame)
    for col in frame.columns:
        if col in normalized:
            result[col] = normalized[col]
            continue
        result[col] = _infer_one(frame[col], n, col, config)
    return result


def _infer_one(series: pd.Series, n: int, name: str, config: TypingConfig) -> VariableKind:
    """Clasifica una única columna."""
    non_null = series.dropna()
    nunique = int(non_null.nunique())

    if pdt.is_datetime64_any_dtype(series) or pdt.is_timedelta64_dtype(series):
        return VariableKind.TEMPORAL

    if nunique <= 2 and n > 2:
        return VariableKind.BINARY

    if pdt.is_bool_dtype(series):
        return VariableKind.BINARY

    if pdt.is_numeric_dtype(series):
        unique_ratio = nunique / n if n else 0.0
        if _looks_like_id(name) and unique_ratio >= config.identifier_unique_ratio:
            return VariableKind.IDENTIFIER
        if pdt.is_integer_dtype(series) and nunique <= config.max_discrete_unique:
            return VariableKind.NUMERIC_DISCRETE
        if (
            pdt.is_float_dtype(series)
            and _is_integer_valued(non_null)
            and nunique <= config.max_discrete_unique
        ):
            return VariableKind.NUMERIC_DISCRETE
        return VariableKind.NUMERIC_CONTINUOUS

    # object / categórica / string
    unique_ratio = nunique / n if n else 0.0
    if unique_ratio >= config.identifier_unique_ratio or (_looks_like_id(name) and nunique == n):
        return VariableKind.IDENTIFIER
    avg_len = non_null.astype(str).str.len().mean() if not non_null.empty else 0.0
    if avg_len and avg_len >= config.text_avg_len and nunique > config.max_categorical_unique:
        return VariableKind.TEXT
    return VariableKind.CATEGORICAL_NOMINAL


def _looks_like_id(name: str) -> bool:
    """Heurística de nombre: ``id``, ``*_id``, ``id_*``, ``uuid``, ``codigo``."""
    low = name.lower()
    return (
        low == "id"
        or low.startswith("id_")
        or low.endswith("_id")
        or "uuid" in low
        or low.startswith("codigo")
    )


def _is_integer_valued(series: pd.Series) -> bool:
    """True si todos los valores flotantes son enteros (p. ej. 3.0)."""
    try:
        return bool((series.dropna() % 1 == 0).all())
    except TypeError:  # pragma: no cover - series no numérica
        return False


def group_by_kind(types: dict[str, VariableKind]) -> dict[VariableKind, list[str]]:
    """Agrupa columnas por su tipo inferido (útil para iterar por familia)."""
    grouped: dict[VariableKind, list[str]] = {}
    for col, kind in types.items():
        grouped.setdefault(kind, []).append(col)
    return grouped
