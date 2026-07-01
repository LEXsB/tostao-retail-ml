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
def test_pipeline_registry_has_default() -> None:
    """El registro de pipelines siempre expone un ``__default__``."""
    from tostao_ml.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    assert "__default__" in pipelines
