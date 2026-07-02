"""Construye las tablas maestras por caso y su reporte HTML **completo**.

Ejecutar con: ``uv run python scripts/build_masters_eda.py``

Carga las fuentes crudas por el catálogo de Kedro, cruza TODAS las fuentes de
cada caso en su master table, la persiste en Parquet y genera un reporte HTML
completo por caso (EDA detallado + apertura por variable objetivo + modelado +
interpretabilidad + impacto de negocio + glosario) sobre la tabla cruzada.
"""

from __future__ import annotations

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from tostao_ml.cases import masters
from tostao_ml.cases.reporting import (
    build_unified_report,
    case_a_model_sections,
    case_b_model_sections,
    case_c_model_sections,
    full_case_report,
)

PRIMARY = Path("data/03_primary")
REPORTING = Path("data/08_reporting")


def main() -> None:
    bootstrap_project(Path.cwd())
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTING.mkdir(parents=True, exist_ok=True)

    with KedroSession.create(project_path=Path.cwd()) as session:
        catalog = session.load_context().catalog
        load = catalog.load
        master_a, rep_a = masters.build_master_a(
            load("a_ventas_historicas"),
            load("a_catalogo_productos"),
            load("a_maestro_tiendas"),
            load("a_inventario_actual"),
            load("a_ground_truth_trends"),
        )
        weekly_a = masters.aggregate_weekly_a(master_a)
        master_b, rep_b = masters.build_master_b(
            load("b_tickets"), load("b_detalle_tickets"), load("b_catalogo_productos")
        )
        baskets_b = masters.build_baskets_b(master_b)
        master_c, rep_c = masters.build_master_c(
            load("c_transacciones_resumen"),
            load("c_clientes_loyalty"),
            load("c_variables_exogenas"),
            load("c_promociones_activas"),
        )

    master_a.to_parquet(PRIMARY / "master_caso_a_diario.parquet")
    weekly_a.to_parquet(PRIMARY / "master_caso_a_semanal.parquet")
    master_b.to_parquet(PRIMARY / "master_caso_b.parquet")
    master_c.to_parquet(PRIMARY / "master_caso_c.parquet")

    print("Generando reporte Caso A…")
    full_case_report(
        "a",
        "Caso A — Abastecimiento · Reporte completo",
        weekly_a,
        "unidades_vendidas",
        "Demanda semanal por SKU-tienda enriquecida con catálogo, tiendas, inventario y tendencia.",
        rep_a,
        case_a_model_sections(weekly_a),
    ).save(REPORTING / "reporte_caso_a.html")

    print("Generando reporte Caso B…")
    full_case_report(
        "b",
        "Caso B — Combos · Reporte completo",
        master_b,
        "importe_linea",
        "Líneas de ticket enriquecidas con cabecera y catálogo; base para clustering y combos.",
        rep_b,
        case_b_model_sections(master_b, baskets_b),
    ).save(REPORTING / "reporte_caso_b.html")

    print("Generando reporte Caso C…")
    full_case_report(
        "c",
        "Caso C — AOV · Reporte completo",
        master_c,
        "total_venta",
        "Tickets enriquecidos con loyalty, exógenas e intensidad de promociones.",
        rep_c,
        case_c_model_sections(master_c),
    ).save(REPORTING / "reporte_caso_c.html")

    print("Generando reporte unificado…")
    build_unified_report(weekly_a, master_a, master_b, baskets_b, master_c).save(
        REPORTING / "reporte_unificado.html"
    )

    print("\n=== ARTEFACTOS ===")
    for p in sorted(REPORTING.glob("reporte_*.html")):
        print(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
