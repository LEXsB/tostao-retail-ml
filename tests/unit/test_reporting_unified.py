"""Prueba de humo del reporte HTML unificado de los tres casos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.cases import masters
from tostao_ml.cases.reporting import build_unified_report


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
def test_unified_report_renders_all_cases() -> None:
    weekly_a = _weekly_a()
    master_b, baskets_b = _master_b()
    report = build_unified_report(weekly_a, weekly_a, master_b, baskets_b, _master_c())
    html = report.render()
    # Intro + historia de datos + EDA (target/multi por caso) + modelado interpretado.
    assert len(report.sections) >= 10
    for marker in (
        "Caso A",
        "Caso B",
        "Caso C",
        "Resumen ejecutivo",
        "Contexto y lectura de los datos",
        "Qué son los datos",
        "Apertura por la variable objetivo",
        "Estrategia de modelamiento",
        "Optuna",
    ):
        assert marker in html
    assert "Plotly.newPlot" in html
