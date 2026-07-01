"""Genera el reporte HTML unificado de los tres casos.

Ejecutar con: ``uv run python scripts/build_unified_report.py``

Construye las tablas maestras desde el catálogo de Kedro, ejecuta los tres casos
y ensambla un único reporte HTML ejecutivo (EDA + desempeño + interpretabilidad +
impacto de negocio) con las mini-conclusiones autogeneradas de cada sección.
"""

from __future__ import annotations

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from tostao_ml.cases import masters
from tostao_ml.cases.reporting import build_unified_report

OUT = Path("data/08_reporting/reporte_unificado.html")


def main() -> None:
    bootstrap_project(Path.cwd())
    with KedroSession.create(project_path=Path.cwd()) as session:
        catalog = session.load_context().catalog
        master_a, _ = masters.build_master_a(
            catalog.load("a_ventas_historicas"),
            catalog.load("a_catalogo_productos"),
            catalog.load("a_maestro_tiendas"),
            catalog.load("a_inventario_actual"),
            catalog.load("a_ground_truth_trends"),
        )
        weekly_a = masters.aggregate_weekly_a(master_a)
        master_b, _ = masters.build_master_b(
            catalog.load("b_tickets"),
            catalog.load("b_detalle_tickets"),
            catalog.load("b_catalogo_productos"),
        )
        baskets_b = masters.build_baskets_b(master_b)
        master_c, _ = masters.build_master_c(
            catalog.load("c_transacciones_resumen"),
            catalog.load("c_clientes_loyalty"),
            catalog.load("c_variables_exogenas"),
            catalog.load("c_promociones_activas"),
        )

    report = build_unified_report(weekly_a, master_a, master_b, baskets_b, master_c)
    path = report.save(OUT)
    print(
        f"Reporte unificado escrito en: {path} ({path.stat().st_size / 1024:.0f} KB, {len(report.sections)} secciones)"
    )


if __name__ == "__main__":
    main()
