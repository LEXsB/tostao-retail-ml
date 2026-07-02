"""Ensembles reutilizables bajo la interfaz :class:`BaseModel`.

Un ensemble por promedio (opcionalmente ponderado) combina varias predicciones
para reducir varianza y, con modelos que se equivocan de forma distinta, mejorar
el error. Se usa cuando la comparación muestra que la combinación supera a cada
modelo individual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tostao_ml.framework.types import TaskType

from .base import BaseModel


class AveragingEnsemble(BaseModel):
    """Promedio (ponderado) de las predicciones de varios modelos base.

    Args:
        estimators: Lista de ``(nombre, modelo)`` a combinar.
        weights: Pesos por modelo (se normalizan); ``None`` = promedio simple.
    """

    task = TaskType.REGRESSION

    def __init__(
        self, estimators: list[tuple[str, BaseModel]], weights: list[float] | None = None
    ) -> None:
        super().__init__(name="ensemble_promedio")
        if not estimators:
            raise ValueError("El ensemble requiere al menos un modelo base.")
        self.estimators = estimators
        w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(estimators))
        self.weights = w / w.sum()
        self.metadata.params = {
            "models": [n for n, _ in estimators],
            "weights": self.weights.tolist(),
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> AveragingEnsemble:
        for _, model in self.estimators:
            if not model.is_fitted_:
                model.fit(X, y)
        self.metadata.feature_names = list(X.columns)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        preds = np.column_stack([model.predict(X) for _, model in self.estimators])
        return np.asarray(preds @ self.weights, dtype=float)
