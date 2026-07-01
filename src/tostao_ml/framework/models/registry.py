"""Registro y factory de modelos por configuración.

Permite instanciar modelos por nombre desde YAML (configuración sobre código),
manteniendo el catálogo de implementaciones desacoplado de los pipelines.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Decorador que registra una implementación de modelo bajo un nombre.

    Args:
        name: Clave única del modelo en el registro.

    Returns:
        Decorador que devuelve la clase sin modificarla.
    """

    def _decorator(cls: type[BaseModel]) -> type[BaseModel]:
        if name in _REGISTRY:
            raise ValueError(f"Modelo '{name}' ya registrado ({_REGISTRY[name].__name__}).")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def build_model(name: str, **params: object) -> BaseModel:
    """Crea un modelo registrado a partir de su nombre y parámetros.

    Args:
        name: Clave del modelo en el registro.
        **params: Hiperparámetros pasados al constructor de la implementación.

    Returns:
        Instancia de :class:`BaseModel`.

    Raises:
        KeyError: Si el nombre no está registrado.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Modelo '{name}' no registrado. Disponibles: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)  # type: ignore


def available_models() -> list[str]:
    """Lista los nombres de modelos registrados."""
    return sorted(_REGISTRY)
