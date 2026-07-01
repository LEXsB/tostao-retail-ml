"""Optimización de decisiones: solver newsvendor / critical fractile (Caso A)."""

from __future__ import annotations

from .newsvendor import (
    NewsvendorPolicy,
    critical_fractile,
    expected_cost,
    optimize_orders,
    order_up_to_level,
)

__all__ = [
    "NewsvendorPolicy",
    "critical_fractile",
    "expected_cost",
    "optimize_orders",
    "order_up_to_level",
]
