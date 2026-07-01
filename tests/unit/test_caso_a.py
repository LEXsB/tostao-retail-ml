"""Pruebas del pipeline del Caso A (forecast + optimización)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.cases.caso_a import FEATURES, build_features_a, run_case_a, temporal_split


def _synthetic_weekly(n_groups: int = 4, n_weeks: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_groups):
        base = rng.integers(5, 20)
        for w in range(1, n_weeks + 1):
            rows.append(
                {
                    "id_tienda": f"S{g % 2}",
                    "id_producto": f"P{g}",
                    "anio": 2024,
                    "semana": w,
                    "unidades_vendidas": max(0, int(base + 3 * np.sin(w / 2) + rng.normal(0, 2))),
                    "ingreso": 0.0,
                    "dias_con_venta": rng.integers(1, 7),
                    "nombre": f"prod{g}",
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


@pytest.mark.unit
def test_build_features_creates_lags_without_leakage() -> None:
    feats = build_features_a(_synthetic_weekly())
    assert set(FEATURES).issubset(feats.columns)
    # el primer registro de cada serie (semana 1) se descarta por no tener lag_1
    assert feats["semana"].min() >= 2


@pytest.mark.unit
def test_temporal_split_is_ordered() -> None:
    feats = build_features_a(_synthetic_weekly())
    train, test = temporal_split(feats, test_weeks=3)
    assert train["semana"].max() < test["semana"].min()


@pytest.mark.unit
def test_run_case_a_end_to_end() -> None:
    result = run_case_a(_synthetic_weekly(n_groups=6, n_weeks=16), max_iter=60)
    assert {"mae", "rmse", "wape", "picp", "mpiw"}.issubset(result.metrics)
    assert len(result.orders) == len(result.test)
    assert {"order_qty", "order_up_to", "critical_fractile"}.issubset(result.orders.columns)
    assert np.isfinite(result.cost_model) and np.isfinite(result.cost_naive)
    assert len(result.narrative) >= 3
