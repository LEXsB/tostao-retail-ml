"""Pruebas de los constructores de tablas maestras por caso."""

from __future__ import annotations

import pandas as pd
import pytest

from tostao_ml.cases import masters


@pytest.mark.unit
def test_master_a_joins_and_derived() -> None:
    ventas = pd.DataFrame(
        {
            "fecha": ["2024-01-01", "2024-01-02"],
            "id_tienda": ["S1", "S1"],
            "id_producto": ["P1", "P1"],
            "unidades_vendidas": [3, 5],
        }
    )
    catalogo = pd.DataFrame(
        {
            "id_producto": ["P1"],
            "nombre": ["Tinto"],
            "categoria": ["Beb"],
            "costo_unitario": [800],
            "precio_venta": [2500],
            "costo_almacenamiento_semanal": [10],
        }
    )
    tiendas = pd.DataFrame({"id_tienda": ["S1"], "ciudad": ["Bogotá"], "tamaño_m2": [28]})
    inventario = pd.DataFrame({"id_tienda": ["S1"], "id_producto": ["P1"], "stock_actual": [4]})
    trends = pd.DataFrame({"id_tienda": ["S1"], "id_producto": ["P1"], "trend_type": ["up"]})

    master, report = masters.build_master_a(ventas, catalogo, tiendas, inventario, trends)
    assert all(cov == 1.0 for cov in report.values())
    assert master.loc[0, "margen_unitario"] == 1700
    assert master.loc[1, "ingreso"] == 5 * 2500
    weekly = masters.aggregate_weekly_a(master)
    assert weekly["unidades_vendidas"].sum() == 8


@pytest.mark.unit
def test_master_b_joins_and_baskets() -> None:
    tickets = pd.DataFrame(
        {
            "id_ticket": ["T1", "T2"],
            "fecha": ["2024-01-01", "2024-01-02"],
            "id_tienda": ["TOSTAO_1", "TOSTAO_1"],
            "id_cliente": ["C1", "C2"],
        }
    )
    detalle = pd.DataFrame(
        {
            "id_ticket": ["T1", "T1", "T2"],
            "id_producto": ["P1", "P2", "P1"],
            "cantidad": [1, 2, 1],
            "precio_unitario": [1000, 500, 1000],
        }
    )
    catalogo = pd.DataFrame(
        {
            "id_producto": ["P1", "P2"],
            "nombre": ["A", "B"],
            "categoria": ["x", "y"],
            "subcategoria": ["u", "v"],
        }
    )
    master, report = masters.build_master_b(tickets, detalle, catalogo)
    assert report["tickets"] == 1.0 and report["catalogo_productos"] == 1.0
    assert master.loc[1, "importe_linea"] == 2 * 500
    baskets = masters.build_baskets_b(master)
    assert baskets.loc["T1", "P1"] and baskets.loc["T1", "P2"]
    assert not baskets.loc["T2", "P2"]


@pytest.mark.unit
def test_master_c_joins_and_promo_intensity() -> None:
    trans = pd.DataFrame(
        {
            "id_ticket": ["T1", "T2"],
            "id_tienda": ["S1", "S2"],
            "timestamp": ["2024-02-10 10:00:00", "2024-02-10 11:00:00"],
            "total_venta": [10.0, 20.0],
            "total_articulos": [2, 4],
            "id_cliente": ["C1", "C2"],
        }
    )
    loyalty = pd.DataFrame(
        {
            "id_cliente": ["C1", "C2"],
            "fecha_registro": ["2023-01-01", "2023-06-01"],
            "edad": [30, 40],
            "segmento": ["Regular", "VIP"],
        }
    )
    exo = pd.DataFrame(
        {
            "fecha": ["2024-02-10", "2024-02-10"],
            "id_tienda": ["S1", "S2"],
            "clima": ["Sunny", "Rainy"],
            "competitor_price_index": [1.0, 1.1],
            "indice_trafico": [80, 90],
        }
    )
    promo = pd.DataFrame(
        {
            "fecha_inicio": ["2024-02-07"],
            "fecha_fin": ["2024-02-18"],
            "id_producto": ["P1"],
            "tipo_descuento": ["Fixed"],
            "id_tienda": ["S1"],
        }
    )
    master, report = masters.build_master_c(trans, loyalty, exo, promo)
    assert report["clientes_loyalty"] == 1.0 and report["variables_exogenas"] == 1.0
    # S1 tiene promo activa ese día; S2 no.
    assert master.set_index("id_tienda").loc["S1", "hay_promo"] == 1
    assert master.set_index("id_tienda").loc["S2", "hay_promo"] == 0
    assert master.loc[0, "antiguedad_cliente_dias"] > 0
