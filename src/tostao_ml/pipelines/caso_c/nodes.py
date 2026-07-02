"""Nodos Kedro del Caso C (drivers GLM + gasto predictivo)."""

from __future__ import annotations

import pandas as pd

from tostao_ml.cases import caso_c, masters


def node_build_master_c(
    transacciones: pd.DataFrame,
    loyalty: pd.DataFrame,
    exogenas: pd.DataFrame,
    promociones: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Cruza las 4 fuentes en la master de AOV."""
    master, report = masters.build_master_c(transacciones, loyalty, exogenas, promociones)
    return master, report


def node_aov_models(
    master_c: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, dict, pd.DataFrame, list, object]:
    """Ajusta el GLM inferencial y el modelo predictivo de gasto.

    Devuelve además el modelo predictivo entrenado para persistirlo como artefacto.
    """
    result = caso_c.run_case_c(master_c, seed=int(params.get("random_seed", 42)))
    coefficients = result.coefficients.reset_index(names="driver")
    metrics = {
        **{f"inferencial_{k}": v for k, v in result.inferential_metrics.items()},
        **{f"predictivo_{k}": v for k, v in result.predictive_metrics.items()},
    }
    return (
        coefficients,
        metrics,
        result.predictive_test,
        result.narrative.to_dicts(),
        result.predictive_model,
    )
