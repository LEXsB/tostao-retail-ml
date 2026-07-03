"""Pruebas de humo de los reportes ejecutivos por caso.

Verifican que el reporte ejecutivo trae la tarea propuesta, el enfoque, los
resultados y el impacto —pero **no** el análisis exploratorio a fondo (ese vive en
el reporte completo y en los notebooks)—.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.cases import caso_a, caso_b, caso_c, executive, masters


def _weekly_a() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for g in range(6):
        for w in range(1, 15):
            rows.append(
                {
                    "id_tienda": f"S{g % 2}",
                    "id_producto": f"P{g}",
                    "anio": 2024,
                    "semana": w,
                    "unidades_vendidas": max(0, int(10 + 3 * np.sin(w / 2) + rng.normal(0, 2))),
                    "ingreso": 0.0,
                    "dias_con_venta": 5,
                    "nombre": f"p{g}",
                    "categoria": "Bebidas" if g % 2 else "Panadería",
                    "costo_unitario": 800.0,
                    "precio_venta": 2500.0,
                    "costo_almacenamiento_semanal": 10.0,
                    "ciudad": "Bogotá",
                    "tamaño_m2": 30.0,
                    "stock_actual": 4.0,
                    "trend_type": "up",
                }
            )
    return pd.DataFrame(rows)


def _master_b() -> tuple[pd.DataFrame, pd.DataFrame]:
    tickets, detalle, tid = [], [], 0
    catalogo = pd.DataFrame(
        {
            "id_producto": ["P1", "P2", "P3", "P4"],
            "nombre": ["A", "B", "C", "D"],
            "categoria": ["Panadería", "Panadería", "Bebidas", "Bebidas"],
            "subcategoria": ["a", "b", "c", "d"],
        }
    )
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
            for p in (a, b):
                detalle.append(
                    {"id_ticket": t, "id_producto": p, "cantidad": 1, "precio_unitario": 1000}
                )
    master, _ = masters.build_master_b(pd.DataFrame(tickets), pd.DataFrame(detalle), catalogo)
    return master, masters.build_baskets_b(master)


def _master_c() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for c in range(40):
        for _ in range(6):
            arts = rng.integers(1, 6)
            rows.append(
                {
                    "id_cliente": f"C{c:03d}",
                    "fecha": pd.Timestamp("2024-01-01")
                    + pd.Timedelta(days=int(rng.integers(1, 90))),
                    "total_venta": float(8 * arts + rng.normal(0, 1)),
                    "total_articulos": int(arts),
                    "edad": int(rng.integers(20, 70)),
                    "segmento": rng.choice(["Regular", "VIP"]),
                    "clima": rng.choice(["Sunny", "Rainy"]),
                    "competitor_price_index": float(rng.normal(1, 0.1)),
                    "indice_trafico": int(rng.integers(50, 100)),
                    "n_promos_activas": int(rng.integers(0, 3)),
                    "hora": int(rng.integers(7, 20)),
                    "dia_semana": int(rng.integers(0, 7)),
                    "antiguedad_cliente_dias": int(rng.integers(30, 400)),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.integration
def test_executive_a_renders() -> None:
    weekly = _weekly_a()
    result = caso_a.run_case_a(weekly, tune=False)
    html = executive.executive_a(weekly, result).render()
    for marker in (
        "Tarea propuesta",
        "Contexto y enfoque",
        "Resultados clave",
        "Impacto de negocio",
    ):
        assert marker in html
    # El ejecutivo NO incluye el EDA a fondo.
    assert "Apertura por la variable objetivo" not in html
    assert "VIF" not in html
    assert "Plotly.newPlot" in html


@pytest.mark.integration
def test_executive_b_renders() -> None:
    master_b, baskets = _master_b()
    result = caso_b.run_case_b(master_b, baskets, tune=False)
    html = executive.executive_b(master_b, baskets, result).render()
    for marker in ("Tarea propuesta", "Resultados clave", "Combos recomendados"):
        assert marker in html
    assert "Apertura por la variable objetivo" not in html


@pytest.mark.integration
def test_executive_c_renders() -> None:
    master_c = _master_c()
    result = caso_c.run_case_c(master_c, tune=False)
    html = executive.executive_c(master_c, result).render()
    for marker in ("Tarea propuesta", "Resultados clave", "Predicción de gasto"):
        assert marker in html
    assert "Apertura por la variable objetivo" not in html
