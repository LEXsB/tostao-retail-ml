"""Tipos y enumeraciones compartidos por todo el framework.

Centralizar estas definiciones evita literales mágicos dispersos y da una
interfaz común a profiling, features, models y evaluation.
"""

from __future__ import annotations

from enum import StrEnum


class VariableKind(StrEnum):
    """Clasificación semántica de una variable (dirige el análisis y el gráfico).

    El motor de perfilado (``profiling/``) infiere este tipo por dtype,
    cardinalidad y heurísticas, permitiendo override por configuración. El tipo
    determina qué estadística, qué test y qué visual se aplican a cada variable.
    """

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    BINARY = "binary"
    TEMPORAL = "temporal"
    IDENTIFIER = "identifier"
    TEXT = "text"

    @property
    def is_numeric(self) -> bool:
        """True si la variable admite operaciones numéricas."""
        return self in (VariableKind.NUMERIC_CONTINUOUS, VariableKind.NUMERIC_DISCRETE)

    @property
    def is_categorical(self) -> bool:
        """True si la variable es categórica (nominal, ordinal o binaria)."""
        return self in (
            VariableKind.CATEGORICAL_NOMINAL,
            VariableKind.CATEGORICAL_ORDINAL,
            VariableKind.BINARY,
        )

    @property
    def is_modelable(self) -> bool:
        """True si la variable es candidata directa a feature (no id ni texto)."""
        return self not in (VariableKind.IDENTIFIER, VariableKind.TEXT)


class TaskType(StrEnum):
    """Tipo de tarea de modelado; selecciona métricas, validadores y gráficos."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"
    CLUSTERING = "clustering"
    ASSOCIATION = "association"

    @property
    def is_supervised(self) -> bool:
        """True para tareas con variable objetivo observada."""
        return self in (
            TaskType.REGRESSION,
            TaskType.CLASSIFICATION,
            TaskType.FORECASTING,
        )

    @property
    def is_temporal(self) -> bool:
        """True para tareas que exigen validación temporal (sin leakage)."""
        return self is TaskType.FORECASTING
