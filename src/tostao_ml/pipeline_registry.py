"""Registro de pipelines de ``tostao_ml``.

Kedro descubre aquí los pipelines disponibles. Cada caso registra su pipeline
(ingesta cruzada → modelado → salidas) y el ``__default__`` compone el flujo
unificado de los tres casos.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from tostao_ml.pipelines import caso_a, caso_b, caso_c


def register_pipelines() -> dict[str, Pipeline]:
    """Registra los pipelines del proyecto.

    Returns:
        Mapa ``nombre -> Pipeline``. ``__default__`` ejecuta los tres casos.
    """
    pipelines: dict[str, Pipeline] = {
        "caso_a": caso_a.create_pipeline(),
        "caso_b": caso_b.create_pipeline(),
        "caso_c": caso_c.create_pipeline(),
    }
    pipelines["__default__"] = pipelines["caso_a"] + pipelines["caso_b"] + pipelines["caso_c"]
    return pipelines
