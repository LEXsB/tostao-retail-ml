"""Pruebas del optimizador newsvendor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.framework.optimization import (
    NewsvendorPolicy,
    critical_fractile,
    expected_cost,
    optimize_orders,
    order_up_to_level,
)


@pytest.mark.unit
def test_critical_fractile() -> None:
    # margen alto vs almacenamiento bajo → nivel de servicio alto
    assert critical_fractile(cu=9.0, co=1.0) == pytest.approx(0.9)
    assert critical_fractile(cu=0.0, co=0.0) == 0.5


@pytest.mark.unit
def test_order_up_to_level_interpolates() -> None:
    quantiles = {0.1: 2.0, 0.5: 5.0, 0.9: 10.0}
    assert order_up_to_level(quantiles, 0.5) == pytest.approx(5.0)
    assert 5.0 < order_up_to_level(quantiles, 0.7) < 10.0


@pytest.mark.unit
def test_higher_margin_orders_more() -> None:
    quantiles = {0.1: 2.0, 0.5: 5.0, 0.9: 10.0}
    aggressive = NewsvendorPolicy(cu=9.0, co=1.0).order(quantiles, current_stock=0)
    conservative = NewsvendorPolicy(cu=1.0, co=9.0).order(quantiles, current_stock=0)
    assert aggressive["order_qty"] > conservative["order_qty"]


@pytest.mark.unit
def test_order_discounts_current_stock() -> None:
    quantiles = {0.1: 2.0, 0.5: 5.0, 0.9: 10.0}
    policy = NewsvendorPolicy(cu=1.0, co=1.0)  # cf=0.5 → S*=5
    assert policy.order(quantiles, current_stock=0)["order_qty"] == 5.0
    assert policy.order(quantiles, current_stock=5)["order_qty"] == 0.0


@pytest.mark.unit
def test_expected_cost_minimized_near_optimum() -> None:
    scenarios = np.array([2.0, 5.0, 5.0, 8.0, 10.0])
    cu, co = 3.0, 1.0
    costs = {q: expected_cost(q, 0.0, scenarios, cu, co) for q in range(0, 12)}
    best = min(costs, key=lambda q: costs[q])
    # con cu>co conviene pedir por encima de la mediana (5)
    assert best >= 5


@pytest.mark.unit
def test_optimize_orders_batch() -> None:
    qf = pd.DataFrame({"q0.1": [2.0, 1.0], "q0.5": [5.0, 3.0], "q0.9": [10.0, 6.0]})
    out = optimize_orders(
        qf, cu=np.array([9.0, 1.0]), co=np.array([1.0, 9.0]), current_stock=np.array([0.0, 0.0])
    )
    assert list(out.columns) == ["critical_fractile", "order_up_to", "order_qty"]
    assert out.loc[0, "order_qty"] > out.loc[1, "order_qty"]
