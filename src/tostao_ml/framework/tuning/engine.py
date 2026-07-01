"""Motor de HPO reutilizable sobre Optuna (§6).

Optimiza hiperparámetros de cualquier modelo del registro con validación cruzada
(temporal o estándar), samplers configurables (TPE/CMA-ES/NSGA-II) y pruners.
Todo el estudio es reproducible (semilla) y su mejor configuración se narra.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd

from tostao_ml.framework.models import build_model
from tostao_ml.framework.models.base import BaseModel
from tostao_ml.framework.narrate import Insight, Narrative, Severity

from .spaces import suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

Scorer = Callable[[BaseModel, pd.DataFrame, "pd.Series[Any]"], float]
Splitter = Callable[[pd.DataFrame], Iterator[tuple[np.ndarray, np.ndarray]]]

_SAMPLERS = {
    "tpe": optuna.samplers.TPESampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "random": optuna.samplers.RandomSampler,
    "nsga2": optuna.samplers.NSGAIISampler,
}
_PRUNERS = {
    "median": optuna.pruners.MedianPruner,
    "hyperband": optuna.pruners.HyperbandPruner,
    "none": optuna.pruners.NopPruner,
}


@dataclass(slots=True)
class TuningResult:
    """Resultado de un estudio de HPO.

    Attributes:
        best_params: Mejor configuración encontrada.
        best_value: Valor objetivo de la mejor configuración.
        direction: ``minimize`` o ``maximize``.
        study: El estudio Optuna completo (para gráficos/trazabilidad).
        param_importances: Importancia relativa de cada hiperparámetro.
        narrative: Conclusiones del estudio.
    """

    best_params: dict[str, Any]
    best_value: float
    direction: str
    study: optuna.Study
    param_importances: dict[str, float] = field(default_factory=dict)
    narrative: Narrative = field(default_factory=Narrative)


def make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    """Instancia un sampler por nombre con semilla fija."""
    if name not in _SAMPLERS:
        raise ValueError(f"Sampler '{name}' desconocido. Opciones: {sorted(_SAMPLERS)}")
    return _SAMPLERS[name](seed=seed)  # type: ignore[no-any-return]


def make_pruner(name: str) -> optuna.pruners.BasePruner:
    """Instancia un pruner por nombre."""
    if name not in _PRUNERS:
        raise ValueError(f"Pruner '{name}' desconocido. Opciones: {sorted(_PRUNERS)}")
    return _PRUNERS[name]()  # type: ignore[no-any-return]


def tune_model(
    model_name: str,
    search_space: dict[str, dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    scorer: Scorer,
    splitter: Splitter,
    *,
    direction: str = "minimize",
    n_trials: int = 50,
    sampler: str = "tpe",
    pruner: str = "median",
    seed: int = 42,
    fixed_params: dict[str, Any] | None = None,
    study_name: str | None = None,
) -> TuningResult:
    """Optimiza los hiperparámetros de un modelo con validación cruzada.

    Args:
        model_name: Nombre del modelo registrado a optimizar.
        search_space: Espacio de búsqueda declarativo (ver :func:`suggest_params`).
        X, y: Datos de entrenamiento.
        scorer: ``(modelo, X_val, y_val) -> float`` alineado al negocio.
        splitter: Objeto/función que produce índices de train/val (CV temporal
            o estándar).
        direction: Sentido de la optimización.
        n_trials: Número de trials.
        sampler: Estrategia de muestreo (``tpe``/``cmaes``/``nsga2``/``random``).
        pruner: Estrategia de poda.
        seed: Semilla del estudio (reproducibilidad).
        fixed_params: Parámetros fijos combinados con los muestreados.
        study_name: Nombre del estudio.

    Returns:
        :class:`TuningResult` con la mejor configuración y su narrativa.
    """
    fixed_params = fixed_params or {}

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed_params, **suggest_params(trial, search_space)}
        scores: list[float] = []
        for train_idx, val_idx in splitter(X):
            model = build_model(model_name, **params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(scorer(model, X.iloc[val_idx], y.iloc[val_idx]))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction=direction,
        sampler=make_sampler(sampler, seed),
        pruner=make_pruner(pruner),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    importances = _safe_param_importances(study)
    result = TuningResult(
        best_params={**fixed_params, **study.best_params},
        best_value=float(study.best_value),
        direction=direction,
        study=study,
        param_importances=importances,
    )
    _narrate_study(result)
    return result


def _safe_param_importances(study: optuna.Study) -> dict[str, float]:
    """Importancia de hiperparámetros, tolerante a estudios degenerados."""
    try:
        return {k: float(v) for k, v in optuna.importance.get_param_importances(study).items()}
    except (ValueError, RuntimeError):  # pragma: no cover - pocos trials o sin variación
        return {}


def _narrate_study(result: TuningResult) -> None:
    """Redacta la conclusión del estudio (mejor config y HP dominante)."""
    result.narrative.add(
        Insight(
            text=(
                f"HPO ({len(result.study.trials)} trials): mejor objetivo = "
                f"{result.best_value:.4g} con {result.best_params}."
            ),
            severity=Severity.GOOD,
            metrics={
                "best_value": round(result.best_value, 6),
                "n_trials": len(result.study.trials),
            },
            tags=("hpo",),
            title="Optimización",
        )
    )
    if result.param_importances:
        top = max(result.param_importances.items(), key=lambda kv: kv[1])
        result.narrative.add(
            Insight(
                text=f"El hiperparámetro más influyente fue «{top[0]}» (importancia {top[1]:.0%}).",
                severity=Severity.INFO,
                metrics={"top_param": top[0], "importance": round(top[1], 4)},
                tags=("hpo", "importance"),
            )
        )
