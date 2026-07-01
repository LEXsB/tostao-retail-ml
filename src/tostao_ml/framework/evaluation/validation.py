"""Validadores de partición: temporal (walk-forward) y estándar.

Devuelven *splitters* con la firma que consume el motor de HPO
(``Callable[[DataFrame], Iterator[(train_idx, val_idx)]]``), de modo que la misma
estrategia de validación se usa en tuning y en evaluación (sin leakage en A/C).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

Splitter = Callable[[pd.DataFrame], Iterator[tuple[np.ndarray, np.ndarray]]]


def time_series_splitter(n_splits: int = 5, test_size: int | None = None, gap: int = 0) -> Splitter:
    """Walk-forward: cada validación usa solo el pasado (evita leakage temporal).

    Args:
        n_splits: Número de particiones expansivas.
        test_size: Tamaño del bloque de validación (filas); ``None`` lo infiere.
        gap: Filas descartadas entre train y val (para no filtrar rezagos).
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    def _split(X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from splitter.split(X)

    return _split


def kfold_splitter(n_splits: int = 5, shuffle: bool = True, seed: int = 42) -> Splitter:
    """K-Fold estándar (para tareas sin dependencia temporal)."""
    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)

    def _split(X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from splitter.split(X)

    return _split


def stratified_kfold_splitter(
    y: pd.Series | np.ndarray, n_splits: int = 5, shuffle: bool = True, seed: int = 42
) -> Splitter:
    """K-Fold estratificado por ``y`` (clasificación desbalanceada)."""
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None
    )
    y_arr = np.asarray(y)

    def _split(X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from splitter.split(X, y_arr)

    return _split


def temporal_holdout(frame: pd.DataFrame, time_col: str, test_fraction: float = 0.2) -> np.ndarray:
    """Máscara booleana del bloque final como holdout temporal.

    Args:
        frame: Datos con la columna temporal.
        time_col: Columna de fecha/tiempo para ordenar.
        test_fraction: Fracción final reservada como test.

    Returns:
        Array booleano alineado a ``frame`` (``True`` = pertenece al test).
    """
    order = frame[time_col].rank(method="first")
    cutoff = order.quantile(1.0 - test_fraction)
    return np.asarray((order > cutoff).to_numpy(), dtype=bool)
