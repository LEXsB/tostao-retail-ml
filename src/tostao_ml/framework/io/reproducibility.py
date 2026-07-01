"""Utilidades de reproducibilidad: semillas globales y determinismo.

Reproducibilidad total es un principio no negociable del producto: cada run debe
partir de un estado aleatorio conocido.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> int:
    """Fija la semilla global de ``random``, ``numpy`` y el hashing de Python.

    Args:
        seed: Semilla a aplicar.

    Returns:
        La semilla aplicada (para registrarla en MLflow/artefactos).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Se fija el RandomState global de numpy a propósito: muchas librerías
    # (sklearn, statsmodels, shap) muestrean de él si no reciben un Generator.
    np.random.seed(seed)  # noqa: NPY002
    return seed
