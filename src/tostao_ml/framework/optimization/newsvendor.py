"""Optimizador de pedido newsvendor / critical fractile (Caso A).

Traduce un forecast probabilístico de demanda en una decisión de pedido que
minimiza el costo esperado, balanceando quiebre de stock (margen perdido) contra
sobre-stock (almacenamiento + capital), descontando el inventario actual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def critical_fractile(cu: float, co: float) -> float:
    """Fractil crítico Cu / (Cu + Co).

    Args:
        cu: Costo de subabastecer (margen perdido por unidad no vendida).
        co: Costo de sobreabastecer (almacenamiento + capital por unidad).

    Returns:
        Nivel de servicio óptimo ∈ [0, 1].
    """
    total = cu + co
    return float(cu / total) if total > 0 else 0.5


def order_up_to_level(quantiles: dict[float, float], target_fractile: float) -> float:
    """Nivel de inventario objetivo S* = F⁻¹(fractil) interpolando el forecast cuantílico.

    Args:
        quantiles: Mapa cuantil→demanda pronosticada (p. ej. {0.1: 3, 0.5: 5, 0.9: 9}).
        target_fractile: Fractil crítico objetivo.

    Returns:
        Nivel de inventario objetivo (demanda al fractil crítico).
    """
    levels = np.array(sorted(quantiles))
    values = np.array([quantiles[q] for q in levels], dtype=float)
    return float(np.interp(target_fractile, levels, values))


@dataclass(slots=True)
class NewsvendorPolicy:
    """Política de pedido para un ítem según su estructura de costos.

    Attributes:
        cu: Costo unitario de quiebre (margen perdido = precio − costo).
        co: Costo unitario de sobre-stock (almacenamiento semanal + capital).
        safety_factor: Ajuste de agresividad (>1 más agresivo, <1 conservador).
    """

    cu: float
    co: float
    safety_factor: float = 1.0

    @property
    def critical_fractile(self) -> float:
        """Fractil crítico efectivo, acotado a [0.01, 0.99]."""
        cf = critical_fractile(self.cu, self.co) * self.safety_factor
        return float(np.clip(cf, 0.01, 0.99))

    def order(self, quantiles: dict[float, float], current_stock: float) -> dict[str, float]:
        """Calcula la cantidad a pedir descontando el stock actual.

        Returns:
            Diccionario con ``critical_fractile``, ``order_up_to`` y ``order_qty``
            (entero no negativo).
        """
        cf = self.critical_fractile
        s_star = order_up_to_level(quantiles, cf)
        order_qty = max(0.0, np.ceil(s_star - current_stock))
        return {"critical_fractile": cf, "order_up_to": s_star, "order_qty": float(order_qty)}


def expected_cost(
    order_qty: float, current_stock: float, demand_scenarios: np.ndarray, cu: float, co: float
) -> float:
    """Costo esperado de una decisión de pedido sobre escenarios de demanda.

    Args:
        order_qty: Unidades pedidas.
        current_stock: Inventario disponible antes del pedido.
        demand_scenarios: Muestras/cuantiles de demanda (aprox. de la distribución).
        cu: Costo unitario de quiebre.
        co: Costo unitario de sobre-stock.

    Returns:
        Costo esperado (subabastecimiento + sobreabastecimiento).
    """
    available = current_stock + order_qty
    demand = np.asarray(demand_scenarios, dtype=float)
    understock = np.maximum(demand - available, 0.0) * cu
    overstock = np.maximum(available - demand, 0.0) * co
    return float(np.mean(understock + overstock))


def optimize_orders(
    quantile_frame: pd.DataFrame,
    cu: np.ndarray,
    co: np.ndarray,
    current_stock: np.ndarray,
    *,
    safety_factor: float = 1.0,
) -> pd.DataFrame:
    """Optimiza el pedido para un lote de ítems (vectorizado por filas).

    Args:
        quantile_frame: Columnas ``q{level}`` con la demanda pronosticada por cuantil.
        cu: Vector de costos de quiebre por ítem.
        co: Vector de costos de sobre-stock por ítem.
        current_stock: Vector de inventario actual por ítem.
        safety_factor: Ajuste global de agresividad de la política.

    Returns:
        DataFrame con ``critical_fractile``, ``order_up_to`` y ``order_qty`` por ítem.
    """
    levels = sorted(float(c[1:]) for c in quantile_frame.columns if c.startswith("q"))
    records = []
    for i, (_, row) in enumerate(quantile_frame.iterrows()):
        quantiles = {lvl: float(row[f"q{lvl}"]) for lvl in levels}
        policy = NewsvendorPolicy(cu=float(cu[i]), co=float(co[i]), safety_factor=safety_factor)
        records.append(policy.order(quantiles, float(current_stock[i])))
    return pd.DataFrame(records, index=quantile_frame.index)
