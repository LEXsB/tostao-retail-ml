"""IO reutilizable: reproducibilidad.

La carga/persistencia por capas la gestiona el ``DataCatalog`` de Kedro; aquí
vive la utilidad transversal de semilla determinista.
"""

from __future__ import annotations

from .reproducibility import set_global_seed

__all__ = ["set_global_seed"]
