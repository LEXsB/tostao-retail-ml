"""Construcción de las tablas maestras por caso (cruce de TODAS las fuentes).

Cada caso tiene su propia master table, resultado de unir sus fuentes crudas en
un único DataFrame analítico enriquecido. Sobre esa tabla —no fuente por fuente—
se hace el EDA, el modelado y la estadística.

Los constructores son funciones puras (DataFrames → DataFrame) y cada uno reporta
la **cobertura de cada cruce** (``join_report``) para validar la integridad
referencial: qué proporción de filas encontró match en cada fuente unida.
"""

from __future__ import annotations

import pandas as pd


def _coverage(left: pd.DataFrame, merged: pd.DataFrame, added_cols: list[str]) -> float:
    """Proporción de filas del hecho que encontraron match (columna añadida no nula)."""
    if merged.empty or not added_cols:
        return 0.0
    return float(merged[added_cols[0]].notna().mean())


# =========================================================================== #
# Caso A — Optimización de Abastecimiento
# =========================================================================== #
def build_master_a(
    ventas: pd.DataFrame,
    catalogo: pd.DataFrame,
    tiendas: pd.DataFrame,
    inventario: pd.DataFrame,
    trends: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Master de abastecimiento: ventas diarias SKU-tienda enriquecidas.

    Cruza las 5 fuentes de ``01_supply_optimization``:
    ventas ⨝ catálogo (producto) ⨝ maestro_tiendas (tienda) ⨝ inventario
    (tienda+producto) ⨝ ground_truth_trends (tienda+producto).

    Returns:
        ``(master, join_report)`` — la tabla maestra diaria y la cobertura de
        cada cruce.
    """
    report: dict[str, float] = {}
    m = ventas.copy()
    m["fecha"] = pd.to_datetime(m["fecha"])

    m = m.merge(catalogo, on="id_producto", how="left")
    report["catalogo_productos"] = _coverage(ventas, m, ["nombre"])

    m = m.merge(tiendas, on="id_tienda", how="left")
    report["maestro_tiendas"] = _coverage(ventas, m, ["ciudad"])

    m = m.merge(inventario, on=["id_tienda", "id_producto"], how="left")
    report["inventario_actual"] = _coverage(ventas, m, ["stock_actual"])

    m = m.merge(trends, on=["id_tienda", "id_producto"], how="left")
    report["ground_truth_trends"] = _coverage(ventas, m, ["trend_type"])

    # Derivadas de negocio.
    m["ingreso"] = m["unidades_vendidas"] * m["precio_venta"]
    m["margen_unitario"] = m["precio_venta"] - m["costo_unitario"]
    m["margen_total"] = m["unidades_vendidas"] * m["margen_unitario"]
    m["anio"] = m["fecha"].dt.year
    m["semana"] = m["fecha"].dt.isocalendar().week.astype(int)
    m["dia_semana"] = m["fecha"].dt.dayofweek
    return m, report


def aggregate_weekly_a(master_a: pd.DataFrame) -> pd.DataFrame:
    """Agrega la master diaria a demanda semanal por SKU-tienda (grano de forecast)."""
    keys = ["id_tienda", "id_producto", "anio", "semana"]
    static = [
        "nombre",
        "categoria",
        "costo_unitario",
        "precio_venta",
        "costo_almacenamiento_semanal",
        "ciudad",
        "tamaño_m2",
        "stock_actual",
        "trend_type",
    ]
    agg = (
        master_a.groupby(keys, observed=True)
        .agg(
            unidades_vendidas=("unidades_vendidas", "sum"),
            ingreso=("ingreso", "sum"),
            dias_con_venta=("unidades_vendidas", lambda s: int((s > 0).sum())),
            **{c: (c, "first") for c in static},
        )
        .reset_index()
    )
    return agg


# =========================================================================== #
# Caso B — Creación de Combos
# =========================================================================== #
def build_master_b(
    tickets: pd.DataFrame, detalle: pd.DataFrame, catalogo: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Master de combos: líneas de ticket enriquecidas con cabecera y catálogo.

    Cruza las 3 fuentes de ``02_product_bundles``:
    detalle_tickets ⨝ tickets (id_ticket) ⨝ catálogo (id_producto).

    Returns:
        ``(master, join_report)`` — una fila por línea de ticket enriquecida.
    """
    report: dict[str, float] = {}
    m = detalle.copy()

    m = m.merge(tickets, on="id_ticket", how="left")
    report["tickets"] = _coverage(detalle, m, ["fecha"])

    m = m.merge(catalogo, on="id_producto", how="left")
    report["catalogo_productos"] = _coverage(detalle, m, ["nombre"])

    m["importe_linea"] = m["cantidad"] * m["precio_unitario"]
    if "fecha" in m.columns:
        m["fecha"] = pd.to_datetime(m["fecha"])
    return m, report


def build_baskets_b(master_b: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz cesta×producto (one-hot por ticket) para reglas de asociación."""
    baskets = (
        master_b.assign(presente=1)
        .pivot_table(
            index="id_ticket", columns="id_producto", values="presente", aggfunc="max", fill_value=0
        )
        .astype(bool)
    )
    return baskets


# =========================================================================== #
# Caso C — Modelado de AOV
# =========================================================================== #
def build_master_c(
    transacciones: pd.DataFrame,
    loyalty: pd.DataFrame,
    exogenas: pd.DataFrame,
    promociones: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Master de AOV: tickets enriquecidos con cliente, exógenas y promociones.

    Cruza las 4 fuentes de ``03_aov_drivers``:
    transacciones_resumen ⨝ clientes_loyalty (cliente) ⨝ variables_exogenas
    (fecha+tienda) ⨝ intensidad de promociones_activas (fecha+tienda).

    Returns:
        ``(master, join_report)`` — una fila por ticket enriquecida.
    """
    report: dict[str, float] = {}
    m = transacciones.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"])
    m["fecha"] = m["timestamp"].dt.normalize()

    m = m.merge(loyalty, on="id_cliente", how="left")
    report["clientes_loyalty"] = _coverage(transacciones, m, ["segmento"])

    exo = exogenas.copy()
    exo["fecha"] = pd.to_datetime(exo["fecha"]).dt.normalize()
    m = m.merge(exo, on=["fecha", "id_tienda"], how="left")
    report["variables_exogenas"] = _coverage(transacciones, m, ["clima"])

    promo_intensity = _promo_intensity(promociones)
    m = m.merge(promo_intensity, on=["fecha", "id_tienda"], how="left")
    m["n_promos_activas"] = m["n_promos_activas"].fillna(0).astype(int)
    m["hay_promo"] = (m["n_promos_activas"] > 0).astype(int)
    report["promociones_activas"] = float((m["n_promos_activas"] > 0).mean())

    # Derivadas de negocio.
    m["hora"] = m["timestamp"].dt.hour
    m["dia_semana"] = m["timestamp"].dt.dayofweek
    m["ticket_promedio_articulo"] = m["total_venta"] / m["total_articulos"].replace(0, pd.NA)
    if "fecha_registro" in m.columns:
        m["fecha_registro"] = pd.to_datetime(m["fecha_registro"])
        m["antiguedad_cliente_dias"] = (m["fecha"] - m["fecha_registro"]).dt.days
    return m, report


def _promo_intensity(promociones: pd.DataFrame) -> pd.DataFrame:
    """Expande promociones activas a un conteo por (fecha, tienda)."""
    promo = promociones.copy()
    promo["fecha_inicio"] = pd.to_datetime(promo["fecha_inicio"])
    promo["fecha_fin"] = pd.to_datetime(promo["fecha_fin"])
    rows = []
    for _, r in promo.iterrows():
        for day in pd.date_range(r["fecha_inicio"], r["fecha_fin"], freq="D"):
            rows.append({"fecha": day.normalize(), "id_tienda": r["id_tienda"]})
    if not rows:
        return pd.DataFrame(columns=["fecha", "id_tienda", "n_promos_activas"])
    expanded = pd.DataFrame(rows)
    return (
        expanded.groupby(["fecha", "id_tienda"], observed=True)
        .size()
        .reset_index(name="n_promos_activas")
    )
