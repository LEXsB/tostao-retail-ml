"""Registro de pipelines de ``tostao_ml``.

Kedro descubre aquí los pipelines disponibles. A medida que se construyen el
framework y los tres casos, cada uno registra su pipeline y el ``__default__``
compone el flujo unificado (ingesta → validación → EDA → features → HPO →
entrenamiento → evaluación → reporte).
"""

from __future__ import annotations

from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Registra los pipelines del proyecto.

    Returns:
        Mapa ``nombre -> Pipeline``. La clave ``__default__`` define el pipeline
        que ejecuta ``kedro run`` sin argumentos.
    """
    pipelines: dict[str, Pipeline] = {}
    # Los pipelines se registran incrementalmente a medida que se construyen:
    #   from tostao_ml.pipelines import data_ingestion, data_validation, eda, ...
    #   pipelines["data_ingestion"] = data_ingestion.create_pipeline()
    #   ...
    pipelines["__default__"] = sum(pipelines.values(), start=Pipeline([]))
    return pipelines
