"""Interpretabilidad model-agnóstica: SHAP, permutación y dependencia parcial.

Funciona sobre cualquier :class:`BaseModel` a través de su ``predict``, de modo
que un mismo código explica GLM, GBR o el forecast cuantílico.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from tostao_ml.framework.models.base import BaseModel

Scorer = Callable[[np.ndarray, np.ndarray], float]


@dataclass(slots=True)
class ShapResult:
    """Valores SHAP y la muestra sobre la que se calcularon."""

    values: np.ndarray
    sample: pd.DataFrame
    base_value: float

    @property
    def mean_abs(self) -> pd.Series:
        """Importancia global: media de |SHAP| por feature (ordenada)."""
        importance = np.abs(self.values).mean(axis=0)
        return pd.Series(importance, index=self.sample.columns, name="mean_abs_shap").sort_values(
            ascending=False
        )


def compute_shap(
    model: BaseModel,
    X: pd.DataFrame,
    *,
    background_size: int = 100,
    sample_size: int = 150,
    seed: int = 42,
) -> ShapResult:
    """Calcula valores SHAP con un explainer agnóstico (muestreando por eficiencia).

    Args:
        model: Modelo entrenado.
        X: Datos sobre los que explicar.
        background_size: Tamaño del fondo de referencia.
        sample_size: Nº de observaciones a explicar.
        seed: Semilla del muestreo.

    Returns:
        :class:`ShapResult` con la matriz de valores y la muestra usada.
    """
    sample = X.sample(min(sample_size, len(X)), random_state=seed) if len(X) > sample_size else X
    background = X.sample(min(background_size, len(X)), random_state=seed)
    explainer = shap.Explainer(model.predict, background)
    explanation = explainer(sample)
    values = np.asarray(explanation.values, dtype=float)
    base = explanation.base_values
    base_value = float(np.mean(base)) if np.ndim(base) else float(base)
    return ShapResult(values=values, sample=sample, base_value=base_value)


def permutation_importance(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    scorer: Scorer,
    *,
    n_repeats: int = 5,
    seed: int = 42,
    higher_is_better: bool = False,
) -> pd.Series:
    """Importancia por permutación: degradación de la métrica al mezclar cada feature.

    Args:
        model: Modelo entrenado.
        X: Features.
        y: Objetivo real.
        scorer: ``(y_true, y_pred) -> float`` (p. ej. RMSE).
        n_repeats: Repeticiones de la permutación por feature.
        seed: Semilla.
        higher_is_better: Sentido de la métrica (afecta el signo de la importancia).

    Returns:
        Serie de importancia por feature (mayor ⇒ más importante), ordenada.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=float)
    baseline = scorer(y_arr, model.predict(X))
    importances: dict[str, float] = {}
    for col in X.columns:
        deltas = []
        for _ in range(n_repeats):
            permuted = X.copy()
            permuted[col] = rng.permutation(permuted[col].to_numpy())
            score = scorer(y_arr, model.predict(permuted))
            deltas.append((baseline - score) if higher_is_better else (score - baseline))
        importances[col] = float(np.mean(deltas))
    return pd.Series(importances, name="permutation_importance").sort_values(ascending=False)


def partial_dependence(
    model: BaseModel, X: pd.DataFrame, feature: str, *, grid_resolution: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Dependencia parcial 1D: efecto marginal medio de ``feature`` sobre la predicción.

    Returns:
        ``(grid, avg_pred)`` — malla del feature y predicción media marginal.
    """
    values = X[feature].to_numpy(dtype=float)
    grid = np.linspace(np.percentile(values, 2), np.percentile(values, 98), grid_resolution)
    avg = np.empty(grid_resolution)
    for i, g in enumerate(grid):
        perturbed = X.copy()
        perturbed[feature] = g
        avg[i] = float(np.mean(model.predict(perturbed)))
    return grid, avg
