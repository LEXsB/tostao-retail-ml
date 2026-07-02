"""Features RFM (Recency, Frequency, Monetary) por cliente.

Reutilizable por el Caso C (predicción de gasto del cliente recurrente) y como
insumo de segmentación. Agrega transacciones a nivel cliente y opcionalmente
puntúa cada dimensión en quintiles.
"""

from __future__ import annotations

import pandas as pd

from .base import PandasTransformer


class RFMTransformer(PandasTransformer):
    """Calcula R/F/M por cliente a partir de transacciones.

    Args:
        customer_col: Columna identificadora del cliente.
        date_col: Columna de fecha de la transacción.
        monetary_col: Columna de importe de la transacción.
        reference_date: Fecha de corte para la recencia; si ``None`` usa
            ``max(date) + 1 día`` (recencia mínima = 1).
        score: Si añade quintiles R/F/M (1-5) y un ``rfm_score`` combinado.
    """

    def __init__(
        self,
        customer_col: str,
        date_col: str,
        monetary_col: str,
        reference_date: pd.Timestamp | None = None,
        score: bool = True,
    ) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        self.customer_col = customer_col
        self.date_col = date_col
        self.monetary_col = monetary_col
        self.reference_date = reference_date
        self.score = score

    def fit(self, X: pd.DataFrame, y: object = None) -> RFMTransformer:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        cols = ["recency", "frequency", "monetary", "monetary_mean"]
        if self.score:
            cols += ["r_score", "f_score", "m_score", "rfm_score"]
        self.feature_names_out_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica la transformacion y devuelve un DataFrame."""
        X = self._check_dataframe(X).copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        ref = self.reference_date or (X[self.date_col].max() + pd.Timedelta(days=1))
        agg = X.groupby(self.customer_col).agg(
            recency=(self.date_col, lambda s: (ref - s.max()).days),
            frequency=(self.date_col, "count"),
            monetary=(self.monetary_col, "sum"),
            monetary_mean=(self.monetary_col, "mean"),
        )
        if self.score:
            agg["r_score"] = _quintile(agg["recency"], ascending=False)
            agg["f_score"] = _quintile(agg["frequency"], ascending=True)
            agg["m_score"] = _quintile(agg["monetary"], ascending=True)
            agg["rfm_score"] = agg[["r_score", "f_score", "m_score"]].sum(axis=1)
        return agg


def _quintile(series: pd.Series, ascending: bool) -> pd.Series:
    """Puntúa una serie en quintiles 1-5 (5 = mejor), robusto a empates.

    Para recencia (``ascending=False``) menor es mejor, así que se invierte.
    """
    ranks = series.rank(method="first", ascending=ascending)
    try:
        scores = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5])
    except ValueError:  # pragma: no cover - pocas observaciones o sin variación
        return pd.Series(3, index=series.index, dtype=int)
    return scores.astype(int)
