"""Construye las tablas maestras por caso y sus reportes HTML.

Ejecutar con: ``uv run python scripts/build_reports.py``

Carga las fuentes crudas por el catálogo de Kedro, cruza TODAS las fuentes de
cada caso en su master table, la persiste en Parquet y genera dos reportes por caso
sobre la tabla cruzada:

- **Reporte completo** (``data/08_reporting/reporte_caso_*.html``, no versionado):
  EDA a fondo + apertura por objetivo + modelado + interpretabilidad + negocio.
- **Reporte ejecutivo** (``reports/ejecutivo/reporte_caso_*.html``, versionado): la
  lectura para negocio (tarea, enfoque, métricas clave interpretadas e impacto),
  sin el análisis exploratorio a fondo.

Cada caso se entrena una sola vez y su resultado alimenta ambos reportes, de modo
que las cifras coinciden.
"""

from __future__ import annotations

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from tostao_ml.cases import caso_a, caso_b, caso_c, executive, masters
from tostao_ml.cases.reporting import (
    build_unified_report,
    case_a_model_sections,
    case_b_model_sections,
    case_c_model_sections,
    full_case_report,
)

PRIMARY = Path("data/03_primary")
REPORTING = Path("data/08_reporting")
EJECUTIVO = Path("reports/ejecutivo")


def main() -> None:
    """Genera las tablas maestras y los reportes completo y ejecutivo por caso."""
    bootstrap_project(Path.cwd())
    for folder in (PRIMARY, REPORTING, EJECUTIVO):
        folder.mkdir(parents=True, exist_ok=True)

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

    # Cada caso se entrena una sola vez; el resultado alimenta ambos reportes.
    print("Entrenando Caso A…")
    result_a = caso_a.run_case_a(weekly_a, tune=True, n_trials=20)
    print("Entrenando Caso B…")
    result_b = caso_b.run_case_b(master_b, baskets_b, tune=True)
    print("Entrenando Caso C…")
    result_c = caso_c.run_case_c(master_c, tune=True)

    print("Reporte completo Caso A…")
    full_case_report(
        "a",
        "Caso A — Abastecimiento · Reporte completo",
        weekly_a,
        "unidades_vendidas",
        "Demanda semanal por SKU-tienda enriquecida con catálogo, tiendas, inventario y tendencia.",
        rep_a,
        case_a_model_sections(weekly_a, result_a),
    ).save(REPORTING / "reporte_caso_a.html")
    executive.executive_a(weekly_a, result_a).save(EJECUTIVO / "reporte_caso_a.html", static=True)

    print("Reporte completo Caso B…")
    full_case_report(
        "b",
        "Caso B — Combos · Reporte completo",
        master_b,
        "importe_linea",
        "Líneas de ticket enriquecidas con cabecera y catálogo; base para clustering y combos.",
        rep_b,
        case_b_model_sections(master_b, baskets_b, result_b),
    ).save(REPORTING / "reporte_caso_b.html")
    executive.executive_b(master_b, baskets_b, result_b).save(
        EJECUTIVO / "reporte_caso_b.html", static=True
    )

    print("Reporte completo Caso C…")
    full_case_report(
        "c",
        "Caso C — AOV · Reporte completo",
        master_c,
        "total_venta",
        "Tickets enriquecidos con loyalty, exógenas e intensidad de promociones.",
        rep_c,
        case_c_model_sections(master_c, result_c),
    ).save(REPORTING / "reporte_caso_c.html")
    executive.executive_c(master_c, result_c).save(EJECUTIVO / "reporte_caso_c.html", static=True)

    print("Reporte unificado…")
    build_unified_report(weekly_a, master_a, master_b, baskets_b, master_c).save(
        REPORTING / "reporte_unificado.html"
    )

    print("\n=== ARTEFACTOS ===")
    for folder in (EJECUTIVO, REPORTING):
        for p in sorted(folder.glob("reporte_*.html")):
            print(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
