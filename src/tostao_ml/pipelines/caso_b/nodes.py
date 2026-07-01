"""Nodos Kedro del Caso B (clustering de tiendas + reglas de asociación)."""

from __future__ import annotations

import pandas as pd

from tostao_ml.cases import caso_b, masters


def node_build_master_b(
    tickets: pd.DataFrame, detalle: pd.DataFrame, catalogo: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cruza las 3 fuentes y arma la matriz de cestas."""
    master, report = masters.build_master_b(tickets, detalle, catalogo)
    baskets = masters.build_baskets_b(master)
    return master, baskets, report


def node_combos(
    master_b: pd.DataFrame, baskets: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """Segmenta tiendas y propone combos por cluster."""
    result = caso_b.run_case_b(
        master_b,
        baskets,
        n_clusters=int(params.get("n_clusters", 3)),
        top_n=int(params.get("top_n_combos", 5)),
    )
    clusters = result.store_clusters.rename("cluster").reset_index()
    rules = _serialize_rules(result.rules)
    return clusters, rules, result.combos, result.narrative.to_dicts()


def _serialize_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Convierte los ``frozenset`` de antecedentes/consecuentes en texto (parquet-safe)."""
    if rules.empty:
        return pd.DataFrame()
    out = rules.copy()
    for col in ("antecedents", "consequents"):
        out[col] = out[col].map(lambda s: ", ".join(sorted(s)))
    keep = ["antecedents", "consequents", "support", "confidence", "lift"]
    return out[[c for c in keep if c in out.columns]].round(4)
