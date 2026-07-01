"""Pruebas de humo del paquete y del proyecto Kedro."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_package_import_and_version() -> None:
    """El paquete importa y expone una versión semántica."""
    import tostao_ml

    assert tostao_ml.__version__
    assert tostao_ml.__version__.count(".") >= 2


@pytest.mark.unit
def test_pipeline_registry_has_all_cases() -> None:
    """El registro expone los tres casos y un ``__default__`` que los une."""
    from tostao_ml.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    assert {"caso_a", "caso_b", "caso_c", "__default__"}.issubset(pipelines)
    # El default debe orquestar los nodos de los tres casos.
    default_nodes = {n.name for n in pipelines["__default__"].nodes}
    assert {"build_master_a", "build_master_b", "build_master_c"}.issubset(default_nodes)
