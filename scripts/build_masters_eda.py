"""Construye las tablas maestras por caso y su reporte HTML de EDA.

Ejecutar con: ``uv run python scripts/build_masters_eda.py``

Carga las fuentes crudas por el catálogo de Kedro, cruza TODAS las fuentes de
cada caso en su master table, la persiste en Parquet y genera un reporte HTML de
EDA (sobre la tabla cruzada) con las mini-conclusiones autogeneradas y el
diagnóstico de cobertura de cada cruce.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from tostao_ml.cases import masters
from tostao_ml.framework.profiling import build_eda_figures, profile_dataset
from tostao_ml.framework.reporting import HTMLReport, ReportSection

PRIMARY = Path("data/03_primary")
REPORTING = Path("data/08_reporting")


def _join_report_table(report: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fuente_cruzada": list(report),
            "cobertura_%": [round(v * 100, 2) for v in report.values()],
        }
    )


def build_eda_report(
    master: pd.DataFrame,
    *,
    name: str,
    title: str,
    target: str | None,
    join_report: dict[str, float],
    description: str,
) -> HTMLReport:
    """Perfila la master table y arma su reporte HTML de EDA."""
    profile = profile_dataset(master, name=name, target=target)
    figures = build_eda_figures(master, profile)

    report = HTMLReport(
        title=title, subtitle=f"EDA sobre la tabla maestra cruzada — {len(master):,} filas"
    )

    resumen = ReportSection(
        id="resumen",
        title="Resumen y cruce de fuentes",
        description=description,
        narrative=profile.narrative,
    )
    resumen.add_table(
        "Cobertura de cada cruce (integridad referencial)", _join_report_table(join_report)
    )
    resumen.add_table(
        "Tipos de variable inferidos",
        pd.DataFrame(
            {"variable": list(profile.types), "tipo": [t.value for t in profile.types.values()]}
        ),
    )
    report.add_section(resumen)

    univ = ReportSection(id="univariado", title="Distribuciones univariadas")
    for key, fig in figures.items():
        if key.startswith(("dist__", "pareto__")):
            univ.add_figure(key, fig)
    univ.add_table("Estadística univariada", profile.univariate.round(3))
    report.add_section(univ)

    multi = ReportSection(id="multivariado", title="Correlaciones y redundancia")
    if "correlation_heatmap" in figures:
        multi.add_figure("correlation_heatmap", figures["correlation_heatmap"])
    if not profile.vif.empty:
        multi.add_table("VIF (multicolinealidad)", profile.vif.round(3).to_frame())
    report.add_section(multi)

    if target is not None and not profile.bivariate.empty:
        biv = ReportSection(id="target", title=f"Relación con el target «{target}»")
        for key, fig in figures.items():
            if key.startswith("target__"):
                biv.add_figure(key, fig)
        biv.add_table("Tests bivariados feature–target", profile.bivariate.round(4))
        if not profile.mutual_information.empty:
            biv.add_table(
                "Información mutua feature→target", profile.mutual_information.round(4).to_frame()
            )
        report.add_section(biv)

    return report


def main() -> None:
    bootstrap_project(Path.cwd())
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTING.mkdir(parents=True, exist_ok=True)

    with KedroSession.create(project_path=Path.cwd()) as session:
        catalog = session.load_context().catalog
        raw = {
            name: catalog.load(name)
            for name in [
                "a_ventas_historicas",
                "a_catalogo_productos",
                "a_maestro_tiendas",
                "a_inventario_actual",
                "a_ground_truth_trends",
                "b_tickets",
                "b_detalle_tickets",
                "b_catalogo_productos",
                "c_transacciones_resumen",
                "c_clientes_loyalty",
                "c_variables_exogenas",
                "c_promociones_activas",
            ]
        }

    # --- Caso A -------------------------------------------------------------
    master_a, rep_a = masters.build_master_a(
        raw["a_ventas_historicas"],
        raw["a_catalogo_productos"],
        raw["a_maestro_tiendas"],
        raw["a_inventario_actual"],
        raw["a_ground_truth_trends"],
    )
    weekly_a = masters.aggregate_weekly_a(master_a)
    master_a.to_parquet(PRIMARY / "master_caso_a_diario.parquet")
    weekly_a.to_parquet(PRIMARY / "master_caso_a_semanal.parquet")
    build_eda_report(
        weekly_a,
        name="caso_a_semanal",
        title="Caso A — Abastecimiento · Tabla maestra + EDA",
        target="unidades_vendidas",
        join_report=rep_a,
        description=(
            "Demanda semanal por SKU-tienda enriquecida con catálogo (costos/precio), "
            "maestro de tiendas, inventario actual y tendencia de referencia."
        ),
    ).save(REPORTING / "eda_caso_a.html")

    # --- Caso B -------------------------------------------------------------
    master_b, rep_b = masters.build_master_b(
        raw["b_tickets"], raw["b_detalle_tickets"], raw["b_catalogo_productos"]
    )
    master_b.to_parquet(PRIMARY / "master_caso_b.parquet")
    build_eda_report(
        master_b,
        name="caso_b",
        title="Caso B — Combos · Tabla maestra + EDA",
        target="importe_linea",
        join_report=rep_b,
        description=(
            "Líneas de ticket enriquecidas con cabecera (fecha/tienda/cliente) y catálogo "
            "(categoría/subcategoría). Base para clustering y reglas de asociación."
        ),
    ).save(REPORTING / "eda_caso_b.html")

    # --- Caso C -------------------------------------------------------------
    master_c, rep_c = masters.build_master_c(
        raw["c_transacciones_resumen"],
        raw["c_clientes_loyalty"],
        raw["c_variables_exogenas"],
        raw["c_promociones_activas"],
    )
    master_c.to_parquet(PRIMARY / "master_caso_c.parquet")
    build_eda_report(
        master_c,
        name="caso_c",
        title="Caso C — AOV · Tabla maestra + EDA",
        target="total_venta",
        join_report=rep_c,
        description=(
            "Tickets enriquecidos con loyalty (edad/segmento/antigüedad), exógenas "
            "(clima/tráfico/competencia) e intensidad de promociones activas."
        ),
    ).save(REPORTING / "eda_caso_c.html")

    # --- Resumen a consola --------------------------------------------------
    print("\n===================== MASTER TABLES =====================")
    for name, df, rep in [
        ("A (semanal)", weekly_a, rep_a),
        ("B", master_b, rep_b),
        ("C", master_c, rep_c),
    ]:
        print(f"\n### CASO {name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print("  columnas:", list(df.columns))
        print("  cobertura de cruces:")
        for src, cov in rep.items():
            print(f"    - {src:24s}: {cov * 100:6.2f}% de filas con match")
    print("\n===================== ARTEFACTOS =====================")
    print("  data/03_primary/master_caso_a_diario.parquet")
    print("  data/03_primary/master_caso_a_semanal.parquet")
    print("  data/03_primary/master_caso_b.parquet")
    print("  data/03_primary/master_caso_c.parquet")
    print("  data/08_reporting/eda_caso_a.html")
    print("  data/08_reporting/eda_caso_b.html")
    print("  data/08_reporting/eda_caso_c.html")


if __name__ == "__main__":
    main()
