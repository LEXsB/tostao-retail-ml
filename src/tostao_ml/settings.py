"""Ajustes de proyecto Kedro para ``tostao_ml``.

Registro central de configuración del framework: cargador de configuración
(OmegaConf), hooks del ciclo de vida y patrones de parámetros por caso. Se mantiene
deliberadamente declarativo — la lógica vive en ``tostao_ml.framework``.
"""

from __future__ import annotations

from kedro.config import OmegaConfigLoader

# --------------------------------------------------------------------------- #
# Cargador de configuración
# --------------------------------------------------------------------------- #
CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS: dict = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "catalog": ["catalog*", "catalog*/**", "**/catalog*"],
        "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
        "mlflow": ["mlflow*", "mlflow*/**", "**/mlflow*"],
    },
}

# --------------------------------------------------------------------------- #
# Hooks del ciclo de vida (se activan a medida que el framework los provee)
# --------------------------------------------------------------------------- #
# from tostao_ml.framework.hooks import NarrationHook, SeedHook
# HOOKS = (SeedHook(), NarrationHook())
HOOKS: tuple = ()

# Directorio de sesiones de Kedro; None usa el valor por defecto (.kedro).
# SESSION_STORE_CLASS = ...

# Clase de catálogo de datos (por defecto DataCatalog de Kedro).
# DATA_CATALOG_CLASS = ...
