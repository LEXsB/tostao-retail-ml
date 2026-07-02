"""Selección de variables: por multicolinealidad (VIF) y por correlación.

Reutiliza el motor de asociaciones del EDA para decidir qué eliminar, de modo
que EDA y feature engineering comparten criterio (sin duplicar lógica).
"""

from __future__ import annotations

import pandas as pd

from tostao_ml.framework.profiling.associations import vif_scores

from .base import PandasTransformer


class VIFSelector(PandasTransformer):
    """Elimina iterativamente la variable con VIF más alto hasta bajar del umbral.

    Args:
        threshold: VIF máximo tolerado (típicamente 5 o 10).
        columns: Subconjunto a evaluar; si ``None`` usa todas las numéricas.
    """

    def __init__(self, threshold: float = 10.0, columns: list[str] | None = None) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        self.threshold = threshold
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> VIFSelector:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        X = self._check_dataframe(X)
        candidates = list(self.columns) if self.columns else list(X.select_dtypes("number").columns)
        self.dropped_: list[str] = []
        while len(candidates) > 1:
            vifs = vif_scores(X[candidates])
            worst = vifs.idxmax()
            if not pd.notna(vifs.max()) or vifs.max() <= self.threshold:
                break
            self.dropped_.append(str(worst))
            candidates.remove(str(worst))
        self.kept_ = candidates
        self.feature_names_out_ = [c for c in X.columns if c not in self.dropped_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica la transformacion y devuelve un DataFrame."""
        X = self._check_dataframe(X)
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])


class CorrelationSelector(PandasTransformer):
    """Elimina una de cada par de variables con |correlación| ≥ umbral.

    Conserva la primera de cada par (orden de columnas) para determinismo.

    Args:
        threshold: |correlación| a partir de la cual se descarta una del par.
        method: Método de correlación (``pearson``/``spearman``).
    """

    def __init__(self, threshold: float = 0.95, method: str = "pearson") -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        self.threshold = threshold
        self.method = method

    def fit(self, X: pd.DataFrame, y: object = None) -> CorrelationSelector:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        X = self._check_dataframe(X)
        numeric = X.select_dtypes("number")
        corr = numeric.corr(method=self.method).abs()
        self.dropped_: list[str] = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[j] in self.dropped_ or cols[i] in self.dropped_:
                    continue
                if pd.notna(corr.iloc[i, j]) and corr.iloc[i, j] >= self.threshold:
                    self.dropped_.append(cols[j])
        self.feature_names_out_ = [c for c in X.columns if c not in self.dropped_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica la transformacion y devuelve un DataFrame."""
        X = self._check_dataframe(X)
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])
