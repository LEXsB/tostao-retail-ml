"""IO reutilizable: reproducibilidad y linaje de datos.

La carga/persistencia por capas la gestiona el ``DataCatalog`` de Kedro; aquí
viven utilidades transversales (semillas deterministas, hashing de datos).
"""

from __future__ import annotations

from .hashing import hash_dataframe, short_hash
from .reproducibility import set_global_seed

__all__ = ["hash_dataframe", "set_global_seed", "short_hash"]
