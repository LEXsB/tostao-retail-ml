"""Pipeline Kedro del Caso C: ingesta cruzada → drivers GLM → gasto predictivo."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import node_aov_models, node_build_master_c


def create_pipeline(**kwargs) -> Pipeline:
    """Compone el pipeline del Caso C."""
    return pipeline(
        [
            node(
                node_build_master_c,
                inputs=[
                    "c_transacciones_resumen",
                    "c_clientes_loyalty",
                    "c_variables_exogenas",
                    "c_promociones_activas",
                ],
                outputs=["master_caso_c", "c_join_report"],
                name="build_master_c",
            ),
            node(
                node_aov_models,
                inputs=["master_caso_c", "params:caso_c"],
                outputs=["c_coefficients", "c_metrics", "c_predictive_test", "c_narrative"],
                name="aov_models_c",
            ),
        ],
        tags="caso_c",
    )
