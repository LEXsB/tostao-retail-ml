"""Codificadores de variables categóricas robustos a categorías no vistas."""

from __future__ import annotations

import pandas as pd

from .base import PandasTransformer


class FrequencyEncoder(PandasTransformer):
    """Reemplaza cada categoría por su frecuencia relativa en el entrenamiento.

    Útil para alta cardinalidad sin explotar la dimensionalidad. Las categorías
    no vistas en ``transform`` reciben frecuencia 0.

    Args:
        columns: Columnas categóricas a codificar.
        suffix: Sufijo de las columnas generadas.
        drop_original: Si elimina las columnas originales.
    """

    def __init__(
        self, columns: list[str], suffix: str = "_freq", drop_original: bool = True
    ) -> None:
        self.columns = columns
        self.suffix = suffix
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: object = None) -> FrequencyEncoder:
        X = self._check_dataframe(X)
        self.frequencies_: dict[str, dict] = {}
        for col in self.columns:
            self.frequencies_[col] = (X[col].value_counts(normalize=True)).to_dict()
        kept = [c for c in X.columns if not (self.drop_original and c in self.columns)]
        self.feature_names_out_ = kept + [f"{c}{self.suffix}" for c in self.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._check_dataframe(X).copy()
        for col in self.columns:
            X[f"{col}{self.suffix}"] = X[col].map(self.frequencies_[col]).fillna(0.0)
        if self.drop_original:
            X = X.drop(columns=list(self.columns))
        return X
