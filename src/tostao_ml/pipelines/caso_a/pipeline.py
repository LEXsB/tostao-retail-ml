"""Pipeline Kedro del Caso A: ingesta cruzada → forecast → optimización."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import node_build_master_a, node_forecast_and_optimize


def create_pipeline(**kwargs) -> Pipeline:
    """Compone el pipeline del Caso A."""
    return pipeline(
        [
            node(
                node_build_master_a,
                inputs=[
                    "a_ventas_historicas",
                    "a_catalogo_productos",
                    "a_maestro_tiendas",
                    "a_inventario_actual",
                    "a_ground_truth_trends",
                ],
                outputs=["master_caso_a_diario", "master_caso_a_semanal", "a_join_report"],
                name="build_master_a",
            ),
            node(
                node_forecast_and_optimize,
                inputs=["master_caso_a_semanal", "params:caso_a"],
                outputs=[
                    "a_forecast_metrics",
                    "a_orders",
                    "a_forecast_test",
                    "a_narrative",
                    "a_modelo",
                ],
                name="forecast_and_optimize_a",
            ),
        ],
        tags="caso_a",
    )
