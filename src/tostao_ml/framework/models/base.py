"""Interfaz común de modelos: ``BaseModel`` (ABC) y metadatos.

Regresión, forecasting probabilístico y clustering se exponen bajo una misma
interfaz (fit/predict/save/load), de modo que pipelines, evaluación e
interpretabilidad los tratan de forma uniforme. Las capacidades opcionales
(intervalos de predicción) se declaran vía métodos que por defecto informan que
no están soportadas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tostao_ml.framework.types import TaskType


@dataclass(slots=True)
class ModelMetadata:
    """Metadatos de trazabilidad de un modelo entrenado.

    Attributes:
        name: Nombre lógico del modelo (clave del registro).
        task: Tipo de tarea que resuelve.
        params: Hiperparámetros efectivos.
        feature_names: Columnas de entrada esperadas (tras preprocesamiento).
        extra: Campos adicionales (semilla, hash de datos, métricas de fit).
    """

    name: str
    task: TaskType
    params: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class BaseModel(ABC):
    """Clase base abstracta para todos los modelos del framework.

    Las subclases implementan ``fit`` y ``predict``. ``predict_interval`` es
    opcional (forecasting probabilístico). La persistencia usa ``joblib`` sobre
    el estado interno, delegando en ``_state``/``_restore`` de la subclase.
    """

    task: TaskType

    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata = ModelMetadata(name=name, task=self.task)
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> BaseModel:
        """Entrena el modelo y devuelve ``self``."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicción puntual."""

    def predict_interval(self, X: pd.DataFrame, coverage: float = 0.8) -> dict[str, np.ndarray]:
        """Intervalo de predicción; por defecto no soportado.

        Las subclases probabilísticas devuelven ``{"lower", "median", "upper"}``.
        """
        raise NotImplementedError(f"{self.name} no soporta intervalos de predicción.")

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"El modelo '{self.name}' no ha sido entrenado (llama a fit).")

    def save(self, path: str | Path) -> Path:
        """Serializa el modelo entrenado a disco con ``joblib``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Carga un modelo serializado con :meth:`save`."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, BaseModel):  # pragma: no cover - archivo corrupto
            raise TypeError(f"El archivo {path} no contiene un BaseModel.")
        return obj
