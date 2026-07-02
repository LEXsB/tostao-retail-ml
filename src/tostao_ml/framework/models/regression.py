"""Modelos de regresión puntual y forecasting probabilístico (cuantílico).

Operan sobre features ya preprocesadas (matriz numérica), dejando el
preprocesamiento a ``features/``. El modelo cuantílico habilita intervalos de
predicción, necesarios para la decisión de pedido del Caso A.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from tostao_ml.framework.types import TaskType

from .base import BaseModel
from .registry import register_model


@register_model("ridge")
class RidgeRegressionModel(BaseModel):
    """Regresión lineal regularizada (baseline interpretable y estable)."""

    task = TaskType.REGRESSION

    def __init__(self, alpha: float = 1.0, random_state: int = 42, **kwargs: object) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        super().__init__(name="ridge")
        self.alpha = alpha
        self.random_state = random_state
        self._estimator = Ridge(alpha=alpha, random_state=random_state, **kwargs)
        self.metadata.params = {"alpha": alpha, "random_state": random_state, **kwargs}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> RidgeRegressionModel:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        self._estimator.fit(X, y)
        self.metadata.feature_names = list(X.columns)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las predicciones para las observaciones de entrada."""
        self._check_fitted()
        return np.asarray(self._estimator.predict(X))


@register_model("gbr")
class GBRRegressionModel(BaseModel):
    """Gradient boosting por histogramas (no lineal, robusto, rápido)."""

    task = TaskType.REGRESSION

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_depth: int | None = None,
        max_iter: int = 300,
        l2_regularization: float = 0.0,
        random_state: int = 42,
        **kwargs: object,
    ) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        super().__init__(name="gbr")
        self._estimator = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            l2_regularization=l2_regularization,
            random_state=random_state,
            **kwargs,
        )
        self.metadata.params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "max_iter": max_iter,
            "l2_regularization": l2_regularization,
            "random_state": random_state,
            **kwargs,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> GBRRegressionModel:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        self._estimator.fit(X, y)
        self.metadata.feature_names = list(X.columns)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las predicciones para las observaciones de entrada."""
        self._check_fitted()
        return np.asarray(self._estimator.predict(X))


@register_model("quantile_gbr")
class QuantileGBRModel(BaseModel):
    """Forecast probabilístico vía gradient boosting con pérdida pinball.

    Entrena un estimador por cuantil; ``predict`` devuelve la mediana y
    ``predict_interval`` un intervalo central con la cobertura solicitada.

    Args:
        quantiles: Cuantiles a estimar (deben incluir la mediana 0.5).
        learning_rate/max_depth/max_iter: Hiperparámetros del boosting.
        random_state: Semilla.
    """

    task = TaskType.FORECASTING

    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        learning_rate: float = 0.1,
        max_depth: int | None = None,
        max_iter: int = 300,
        random_state: int = 42,
        **kwargs: object,
    ) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        super().__init__(name="quantile_gbr")
        self.quantiles = tuple(sorted(quantiles))
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.random_state = random_state
        self._kwargs = kwargs
        self._estimators: dict[float, HistGradientBoostingRegressor] = {}
        self.metadata.params = {
            "quantiles": list(self.quantiles),
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "max_iter": max_iter,
            "random_state": random_state,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> QuantileGBRModel:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        for q in self.quantiles:
            est = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                max_iter=self.max_iter,
                random_state=self.random_state,
                **self._kwargs,
            )
            est.fit(X, y)
            self._estimators[q] = est
        self.metadata.feature_names = list(X.columns)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las predicciones para las observaciones de entrada."""
        self._check_fitted()
        median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
        preds = np.asarray(self._estimators[median_q].predict(X))
        # La regresión cuantílica puede cruzar cuantiles; se garantiza no-negatividad
        # de la demanda recortando en 0 (unidades vendidas ≥ 0).
        return np.asarray(np.clip(preds, 0.0, None), dtype=float)

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicción de todos los cuantiles entrenados (columnas ``q{q}``)."""
        self._check_fitted()
        out = {f"q{q}": np.clip(self._estimators[q].predict(X), 0.0, None) for q in self.quantiles}
        return pd.DataFrame(out, index=X.index)

    def predict_interval(self, X: pd.DataFrame, coverage: float = 0.8) -> dict[str, np.ndarray]:
        """Devuelve el intervalo de prediccion para las observaciones."""
        self._check_fitted()
        lower_q = min(self.quantiles)
        upper_q = max(self.quantiles)
        median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
        lower = np.clip(self._estimators[lower_q].predict(X), 0.0, None)
        upper = np.clip(self._estimators[upper_q].predict(X), 0.0, None)
        median = np.clip(self._estimators[median_q].predict(X), 0.0, None)
        # Los cuantiles estimados pueden cruzarse puntualmente; se impone
        # monotonicidad (lower ≤ median ≤ upper) para un intervalo coherente.
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        median = np.clip(median, lower, upper)
        return {
            "lower": np.asarray(lower, dtype=float),
            "median": np.asarray(median, dtype=float),
            "upper": np.asarray(upper, dtype=float),
            "coverage": np.full(len(X), upper_q - lower_q),
        }
