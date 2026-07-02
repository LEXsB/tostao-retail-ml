"""Modelo de clustering (K-Means) bajo la interfaz común.

Usado por el Caso B para segmentar tiendas antes de derivar combos por cluster.
``predict`` devuelve la etiqueta de cluster; ``fit`` expone también métricas de
calidad (silhouette, inercia) para elegir k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tostao_ml.framework.types import TaskType

from .base import BaseModel
from .registry import register_model


@register_model("kmeans")
class KMeansModel(BaseModel):
    """K-Means con métricas de calidad de agrupamiento.

    Args:
        n_clusters: Número de clusters k.
        random_state: Semilla.
        n_init: Reinicios del algoritmo (robustez a inicialización).
    """

    task = TaskType.CLUSTERING

    def __init__(
        self, n_clusters: int = 4, random_state: int = 42, n_init: int = 10, **kwargs: object
    ) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        super().__init__(name="kmeans")
        self.n_clusters = n_clusters
        self._estimator = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=n_init, **kwargs
        )
        self.metadata.params = {
            "n_clusters": n_clusters,
            "random_state": random_state,
            "n_init": n_init,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> KMeansModel:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        labels = self._estimator.fit_predict(X)
        self.metadata.feature_names = list(X.columns)
        self.metadata.extra["inertia"] = float(self._estimator.inertia_)
        if 1 < self.n_clusters < len(X):
            self.metadata.extra["silhouette"] = float(silhouette_score(X, labels))
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las predicciones para las observaciones de entrada."""
        self._check_fitted()
        return np.asarray(self._estimator.predict(X))

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Centroides de los clusters (en el espacio de features de entrada)."""
        self._check_fitted()
        return np.asarray(self._estimator.cluster_centers_)
