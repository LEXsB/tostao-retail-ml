"""Nodos Kedro del Caso A (orquestan ``cases.masters`` y ``cases.caso_a``)."""

from __future__ import annotations

import pandas as pd

from tostao_ml.cases import caso_a, masters


def node_build_master_a(
    ventas: pd.DataFrame,
    catalogo: pd.DataFrame,
    tiendas: pd.DataFrame,
    inventario: pd.DataFrame,
    trends: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cruza las 5 fuentes en la master diaria y su agregación semanal."""
    master, report = masters.build_master_a(ventas, catalogo, tiendas, inventario, trends)
    weekly = masters.aggregate_weekly_a(master)
    return master, weekly, report


def node_forecast_and_optimize(
    weekly: pd.DataFrame, params: dict
) -> tuple[dict, pd.DataFrame, pd.DataFrame, list, object]:
    """Forecast probabilístico + optimización de pedido (newsvendor).

    Devuelve además el modelo entrenado para persistirlo como artefacto y poder
    calificar periodos futuros sin reentrenar.
    """
    result = caso_a.run_case_a(
        weekly,
        quantiles=tuple(params.get("quantiles", (0.1, 0.5, 0.9))),
        max_iter=int(params.get("max_iter", 200)),
        seed=int(params.get("random_seed", 42)),
    )
    return result.metrics, result.orders, result.test, result.narrative.to_dicts(), result.model
