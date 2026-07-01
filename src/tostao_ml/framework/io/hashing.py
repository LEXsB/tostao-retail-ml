"""Hashing de datos para trazabilidad y linaje reproducible."""

from __future__ import annotations

import hashlib

import pandas as pd


def hash_dataframe(frame: pd.DataFrame) -> str:
    """Devuelve un hash SHA-256 estable del contenido de un DataFrame.

    Se ordena por columnas para independizar el hash del orden de columnas.
    Útil para versionar entradas/salidas y detectar cambios de datos.

    Args:
        frame: DataFrame a hashear.

    Returns:
        Digest hexadecimal de 64 caracteres.
    """
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    row_hashes = pd.util.hash_pandas_object(ordered, index=True).to_numpy()
    digest = hashlib.sha256(row_hashes.tobytes())
    digest.update(",".join(ordered.columns).encode("utf-8"))
    return digest.hexdigest()


def short_hash(frame: pd.DataFrame, length: int = 12) -> str:
    """Versión corta de :func:`hash_dataframe` para etiquetas legibles."""
    return hash_dataframe(frame)[:length]
