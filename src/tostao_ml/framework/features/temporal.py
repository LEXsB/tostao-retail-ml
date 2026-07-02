"""Transformadores temporales: calendario, codificación cíclica y rezagos.

Reutilizables por el Caso A (forecast de demanda) y el Caso C (exógenas por
fecha). Los rezagos/rodantes son *group-aware* para no filtrar información entre
SKU-tienda o entre clientes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import PandasTransformer


class DateTimeFeatures(PandasTransformer):
    """Extrae atributos de calendario de una columna de fecha.

    Args:
        column: Nombre de la columna datetime de entrada.
        drop_original: Si elimina la columna original tras extraer.
    """

    def __init__(self, column: str, drop_original: bool = True) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        self.column = column
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: object = None) -> DateTimeFeatures:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        X = self._check_dataframe(X)
        base = [c for c in X.columns if c != self.column] if self.drop_original else list(X.columns)
        self.feature_names_out_ = [
            *base,
            f"{self.column}_year",
            f"{self.column}_month",
            f"{self.column}_day",
            f"{self.column}_dayofweek",
            f"{self.column}_weekofyear",
            f"{self.column}_quarter",
            f"{self.column}_is_weekend",
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica la transformacion y devuelve un DataFrame."""
        X = self._check_dataframe(X).copy()
        dt = pd.to_datetime(X[self.column])
        X[f"{self.column}_year"] = dt.dt.year
        X[f"{self.column}_month"] = dt.dt.month
        X[f"{self.column}_day"] = dt.dt.day
        X[f"{self.column}_dayofweek"] = dt.dt.dayofweek
        X[f"{self.column}_weekofyear"] = dt.dt.isocalendar().week.astype(int)
        X[f"{self.column}_quarter"] = dt.dt.quarter
        X[f"{self.column}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        if self.drop_original:
            X = X.drop(columns=[self.column])
        return X


class CyclicalEncoder(PandasTransformer):
    """Codifica variables cíclicas (mes, día de semana) en seno/coseno.

    Evita la discontinuidad artificial de tratar diciembre(12) y enero(1) como
    lejanos. Añade ``{col}_sin`` y ``{col}_cos`` por columna.

    Args:
        columns: Columnas cíclicas a codificar.
        periods: Periodo de cada columna (p. ej. 12 para mes, 7 para día).
        drop_original: Si elimina las columnas originales.
    """

    def __init__(
        self, columns: list[str], periods: dict[str, int], drop_original: bool = True
    ) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        self.columns = columns
        self.periods = periods
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: object = None) -> CyclicalEncoder:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        X = self._check_dataframe(X)
        kept = [c for c in X.columns if not (self.drop_original and c in self.columns)]
        generated = [f"{c}_{fn}" for c in self.columns for fn in ("sin", "cos")]
        self.feature_names_out_ = kept + generated
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica la transformacion y devuelve un DataFrame."""
        X = self._check_dataframe(X).copy()
        for col in self.columns:
            period = self.periods[col]
            radians = 2.0 * np.pi * X[col].to_numpy(dtype=float) / period
            X[f"{col}_sin"] = np.sin(radians)
            X[f"{col}_cos"] = np.cos(radians)
        if self.drop_original:
            X = X.drop(columns=list(self.columns))
        return X


class GroupLagFeatures(PandasTransformer):
    """Rezagos y estadísticos rodantes por grupo para series temporales.

    Ordena por ``time_col`` dentro de cada grupo y calcula ``lag_k`` y medias
    rodantes, desplazadas para no incluir el valor actual (sin leakage).

    Args:
        group_cols: Columnas que definen la serie (p. ej. SKU + tienda).
        time_col: Columna temporal para ordenar.
        target: Columna sobre la que se calculan los rezagos.
        lags: Lista de rezagos a generar.
        rolling_windows: Ventanas para media/desviación rodante.
    """

    def __init__(
        self,
        group_cols: list[str],
        time_col: str,
        target: str,
        lags: tuple[int, ...] = (1, 2, 4),
        rolling_windows: tuple[int, ...] = (4,),
    ) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        self.group_cols = group_cols
        self.time_col = time_col
        self.target = target
        self.lags = lags
        self.rolling_windows = rolling_windows

    def fit(self, X: pd.DataFrame, y: object = None) -> GroupLagFeatures:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        X = self._check_dataframe(X)
        generated = [f"{self.target}_lag_{k}" for k in self.lags]
        for w in self.rolling_windows:
            generated += [f"{self.target}_rollmean_{w}", f"{self.target}_rollstd_{w}"]
        self.feature_names_out_ = list(X.columns) + generated
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica la transformacion y devuelve un DataFrame."""
        X = self._check_dataframe(X).copy()
        X = X.sort_values([*self.group_cols, self.time_col])
        grouped = X.groupby(self.group_cols, observed=True)[self.target]
        for k in self.lags:
            X[f"{self.target}_lag_{k}"] = grouped.shift(k)
        for w in self.rolling_windows:
            # shift(1) antes de la ventana ⇒ la media/desv. no incluye el valor actual.
            X[f"{self.target}_rollmean_{w}"] = grouped.transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
            X[f"{self.target}_rollstd_{w}"] = grouped.transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).std()
            )
        return X
