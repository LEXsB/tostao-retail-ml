"""Traducción de espacios de búsqueda declarativos (YAML) a sugerencias Optuna.

El espacio vive en configuración, no en código. Cada entrada describe el tipo y
el rango; los condicionales se resuelven en tiempo de trial.
"""

from __future__ import annotations

from typing import Any

import optuna


def suggest_params(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Genera un conjunto de hiperparámetros para un ``trial`` desde el espacio.

    Formato de cada entrada de ``space``::

        alpha:  {type: float, low: 0.01, high: 10.0, log: true}
        depth:  {type: int, low: 2, high: 12}
        loss:   {type: categorical, choices: [l1, l2]}

    Args:
        trial: Trial de Optuna en curso.
        space: Diccionario nombre→especificación.

    Returns:
        Diccionario de hiperparámetros muestreados.
    """
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:  # pragma: no cover - configuración inválida
            raise ValueError(f"Tipo de espacio no soportado para '{name}': {kind}")
    return params
