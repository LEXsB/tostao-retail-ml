"""Pipeline Kedro del Caso B: ingesta cruzada → clustering → reglas → combos."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import node_build_master_b, node_combos


def create_pipeline(**kwargs) -> Pipeline:
    """Compone el pipeline del Caso B."""
    return pipeline(
        [
            node(
                node_build_master_b,
                inputs=["b_tickets", "b_detalle_tickets", "b_catalogo_productos"],
                outputs=["master_caso_b", "b_baskets", "b_join_report"],
                name="build_master_b",
            ),
            node(
                node_combos,
                inputs=["master_caso_b", "b_baskets", "params:caso_b"],
                outputs=["b_store_clusters", "b_rules", "b_combos", "b_narrative"],
                name="cluster_and_combos_b",
            ),
        ],
        tags="caso_b",
    )
