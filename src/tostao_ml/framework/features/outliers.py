"""Tratamiento de outliers: winsorización por percentiles (aprendida en fit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import PandasTransformer


class Winsorizer(PandasTransformer):
    """Recorta valores extremos a percentiles aprendidos en el entrenamiento.

    Reduce el efecto de outliers sin eliminarlos (clave para modelos sensibles a
    colas, p. ej. GLM del Caso C). Los límites se aprenden en ``fit`` y se aplican
    idénticos en ``transform`` (sin leakage).

    Args:
        columns: Columnas numéricas a recortar.
        lower_quantile: Percentil inferior (0-1).
        upper_quantile: Percentil superior (0-1).
    """

    def __init__(
        self, columns: list[str], lower_quantile: float = 0.01, upper_quantile: float = 0.99
    ) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y: object = None) -> Winsorizer:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        X = self._check_dataframe(X)
        self.bounds_: dict[str, tuple[float, float]] = {}
        for col in self.columns:
            lo = float(X[col].quantile(self.lower_quantile))
            hi = float(X[col].quantile(self.upper_quantile))
            self.bounds_[col] = (lo, hi)
        self.feature_names_out_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica la transformacion y devuelve un DataFrame."""
        X = self._check_dataframe(X).copy()
        for col in self.columns:
            lo, hi = self.bounds_[col]
            X[col] = np.clip(X[col].to_numpy(dtype=float), lo, hi)
        return X
