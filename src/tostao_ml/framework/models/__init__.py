"""Modelos bajo interfaz común + registro/factory por configuración.

Importar este paquete registra todas las implementaciones (regresión, forecast
probabilístico, clustering, GLM) en el ``registry``, disponibles vía
``build_model(name, **params)``.
"""

from __future__ import annotations

from .base import BaseModel, ModelMetadata
from .clustering import KMeansModel
from .ensemble import AveragingEnsemble
from .linear import GLMModel
from .registry import available_models, build_model, register_model
from .regression import GBRRegressionModel, QuantileGBRModel, RidgeRegressionModel

__all__ = [
    "AveragingEnsemble",
    "BaseModel",
    "GBRRegressionModel",
    "GLMModel",
    "KMeansModel",
    "ModelMetadata",
    "QuantileGBRModel",
    "RidgeRegressionModel",
    "available_models",
    "build_model",
    "register_model",
]
