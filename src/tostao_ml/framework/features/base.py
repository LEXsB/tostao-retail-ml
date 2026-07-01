"""Utilidades base para transformadores estilo scikit-learn.

Todos los transformadores del framework aceptan y devuelven ``DataFrame`` para
preservar nombres de columnas (clave para interpretabilidad y para el reporte),
son componibles en ``Pipeline``/``ColumnTransformer`` y no mutan su entrada.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PandasTransformer(TransformerMixin, BaseEstimator):
    """Base común: valida que la entrada sea ``DataFrame`` y fija ``set_output``.

    Las subclases implementan ``fit`` (aprendizaje de estado) y ``transform``
    (devolviendo un ``DataFrame``). Exponen ``get_feature_names_out`` para
    integrarse con la introspección de sklearn.
    """

    feature_names_out_: list[str]

    def _check_dataframe(self, X: object) -> pd.DataFrame:
        """Garantiza que la entrada sea un ``DataFrame`` (o lo convierte)."""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def get_feature_names_out(self, input_features: object = None) -> list[str]:
        """Nombres de columnas producidas (tras ``fit``)."""
        return list(getattr(self, "feature_names_out_", []))
