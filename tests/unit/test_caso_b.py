"""Pruebas del pipeline del Caso B (clustering + reglas de asociación)."""

from __future__ import annotations

import pandas as pd
import pytest

from tostao_ml.cases import masters
from tostao_ml.cases.caso_b import (
    association_rules_for,
    build_store_profiles,
    cluster_stores,
    run_case_b,
)

CATALOGO = pd.DataFrame(
    {
        "id_producto": ["P1", "P2", "P3", "P4"],
        "nombre": ["Buñuelo", "Avena", "Café", "Pan"],
        "categoria": ["Panadería", "Panadería", "Bebidas", "Bebidas"],
        "subcategoria": ["a", "b", "c", "d"],
    }
)


def _synthetic_case_b() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tiendas S1/S2 compran {P1,P2}; S3/S4 compran {P3,P4} (clusters claros)."""
    tickets, detalle = [], []
    tid = 0
    for store, (a, b) in [
        ("S1", ("P1", "P2")),
        ("S2", ("P1", "P2")),
        ("S3", ("P3", "P4")),
        ("S4", ("P3", "P4")),
    ]:
        for _ in range(30):
            tid += 1
            t = f"T{tid:04d}"
            tickets.append(
                {"id_ticket": t, "fecha": "2024-01-01", "id_tienda": store, "id_cliente": "C1"}
            )
            for prod in (a, b):
                detalle.append(
                    {"id_ticket": t, "id_producto": prod, "cantidad": 1, "precio_unitario": 1000}
                )
    master, _ = masters.build_master_b(pd.DataFrame(tickets), pd.DataFrame(detalle), CATALOGO)
    baskets = masters.build_baskets_b(master)
    return master, baskets


@pytest.mark.unit
def test_store_profiles_indexed_by_store() -> None:
    master, _ = _synthetic_case_b()
    profiles = build_store_profiles(master)
    assert set(profiles.index) == {"S1", "S2", "S3", "S4"}
    assert any(c.startswith("share_") for c in profiles.columns)


@pytest.mark.unit
def test_clustering_separates_store_profiles() -> None:
    master, _ = _synthetic_case_b()
    labels, silhouette = cluster_stores(build_store_profiles(master), n_clusters=2)
    # S1/S2 deben caer juntas y separadas de S3/S4
    assert labels["S1"] == labels["S2"]
    assert labels["S3"] == labels["S4"]
    assert labels["S1"] != labels["S3"]
    assert silhouette > 0.3


@pytest.mark.unit
def test_association_rules_find_pair() -> None:
    _, baskets = _synthetic_case_b()
    rules = association_rules_for(baskets, min_support=0.1)
    assert not rules.empty
    assert (rules["lift"] > 1).any()


@pytest.mark.unit
def test_run_case_b_end_to_end() -> None:
    master, baskets = _synthetic_case_b()
    result = run_case_b(master, baskets, n_clusters=2, top_n=3)
    assert not result.combos.empty
    assert {"cluster", "producto_a", "producto_b", "lift", "precio_combo"}.issubset(
        result.combos.columns
    )
    assert (result.combos["precio_combo"] < result.combos["precio_lista"]).all()
    assert len(result.narrative) >= 2
