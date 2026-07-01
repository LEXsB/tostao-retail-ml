"""Ensamblado del reporte HTML unificado de los tres casos.

Reutiliza el motor de EDA, los runners de cada caso y las fábricas de gráficos
(EDA, desempeño, negocio) para producir un único entregable ejecutivo con la
narrativa autogenerada de cada sección.
"""

from __future__ import annotations

import pandas as pd

from tostao_ml.framework.evaluation import performance
from tostao_ml.framework.narrate import Insight, Narrative, Severity
from tostao_ml.framework.profiling import build_eda_figures, profile_dataset
from tostao_ml.framework.reporting import HTMLReport, ReportSection
from tostao_ml.framework.viz import eda as eda_viz

from . import caso_a, caso_b, caso_c


def _eda_section(
    master: pd.DataFrame, *, section_id: str, title: str, target: str | None, description: str
) -> ReportSection:
    """Sección de EDA sobre una tabla maestra."""
    profile = profile_dataset(master, name=section_id, target=target)
    figures = build_eda_figures(master, profile)
    section = ReportSection(
        id=section_id, title=title, description=description, narrative=profile.narrative
    )
    if "correlation_heatmap" in figures:
        section.add_figure("correlation_heatmap", figures["correlation_heatmap"])
    for key, fig in list(figures.items())[:3]:
        if key.startswith(("dist__", "target__")):
            section.add_figure(key, fig)
    if not profile.mutual_information.empty:
        section.add_table(
            "Información mutua feature→target", profile.mutual_information.round(4).to_frame("MI")
        )
    return section


def case_a_section(weekly: pd.DataFrame) -> ReportSection:
    """Sección del Caso A: desempeño del forecast + impacto de negocio."""
    result = caso_a.run_case_a(weekly)
    test = result.test
    section = ReportSection(
        id="caso_a",
        title="🚚 Caso A — Abastecimiento (forecast + newsvendor)",
        description="Forecast probabilístico de demanda semanal y política de pedido óptima.",
        narrative=result.narrative,
    )
    section.add_figure(
        "pred_vs_real", performance.pred_vs_actual(test["unidades_vendidas"], test["pred"])
    )
    section.add_figure(
        "residuales", performance.residuals_vs_pred(test["unidades_vendidas"], test["pred"])
    )
    kpis = {"Costo política óptima": result.cost_model, "Costo política ingenua": result.cost_naive}
    section.add_figure("costo", performance.model_comparison_bar(kpis, "costo esperado"))
    section.add_table(
        "Métricas de forecast",
        pd.DataFrame([result.metrics]).T.rename(columns={0: "valor"}).round(4),
    )
    return section


def case_b_section(master_b: pd.DataFrame, baskets: pd.DataFrame) -> ReportSection:
    """Sección del Caso B: clusters de tiendas y combos propuestos."""
    result = caso_b.run_case_b(master_b, baskets)
    section = ReportSection(
        id="caso_b",
        title="🥐 Caso B — Combos (clustering + reglas de asociación)",
        description="Segmentación de tiendas y combos de co-compra por cluster.",
        narrative=result.narrative,
    )
    cluster_sizes = result.store_clusters.value_counts().sort_index()
    section.add_figure("clusters", eda_viz.pareto(cluster_sizes, name="cluster"))
    if not result.combos.empty:
        top = result.combos.sort_values("lift", ascending=False).head(10)
        combo_lift = top.assign(combo=top["producto_a"] + " + " + top["producto_b"]).set_index(
            "combo"
        )["lift"]
        section.add_figure("combos_lift", eda_viz.pareto(combo_lift, name="combo (lift)"))
        section.add_table("Combos propuestos (Top por cluster)", result.combos)
    return section


def case_c_section(master_c: pd.DataFrame) -> ReportSection:
    """Sección del Caso C: drivers del ticket y modelo de gasto."""
    result = caso_c.run_case_c(master_c)
    section = ReportSection(
        id="caso_c",
        title="🧾 Caso C — AOV (drivers GLM + predicción de gasto)",
        description="Drivers inferenciales del ticket y predicción del gasto del cliente recurrente.",
        narrative=result.narrative,
    )
    coefs = result.coefficients.head(10)
    section.add_figure("coeficientes", eda_viz.pareto(coefs["coef"].abs(), name="|β| driver"))
    test = result.predictive_test
    section.add_figure(
        "gasto_pred_vs_real", performance.pred_vs_actual(test["ticket_medio"], test["pred"])
    )
    section.add_table("Coeficientes del GLM (drivers del ticket)", result.coefficients.round(4))
    section.add_table(
        "Métricas",
        pd.DataFrame([result.predictive_metrics]).T.rename(columns={0: "valor"}).round(4),
    )
    return section


def build_unified_report(
    weekly_a: pd.DataFrame,
    master_a: pd.DataFrame,
    master_b: pd.DataFrame,
    baskets_b: pd.DataFrame,
    master_c: pd.DataFrame,
) -> HTMLReport:
    """Ensambla el reporte HTML unificado de los tres casos."""
    report = HTMLReport(
        title="Tostao · Plataforma DS/ML retail — Reporte unificado",
        subtitle="Abastecimiento, Combos y AOV sobre un mismo framework reutilizable",
    )

    overview = Narrative()
    overview.add(
        Insight(
            text=(
                "Tres casos de negocio resueltos sobre un núcleo común: forecast probabilístico + "
                "optimización de pedido (A), clustering + reglas de asociación (B) y modelado "
                "inferencial/predictivo del ticket (C)."
            ),
            severity=Severity.INFO,
            tags=("overview",),
            title="Resumen ejecutivo",
        )
    )
    report.add_section(
        ReportSection(
            id="intro",
            title="Resumen ejecutivo",
            description="Cada caso construye su tabla maestra cruzando todas sus fuentes y reutiliza el mismo framework.",
            narrative=overview,
        )
    )

    report.add_section(
        _eda_section(
            weekly_a,
            section_id="eda_a",
            title="EDA · Caso A (demanda semanal)",
            target="unidades_vendidas",
            description="Tabla maestra de abastecimiento (ventas ⨝ catálogo ⨝ tiendas ⨝ inventario ⨝ tendencias).",
        )
    )
    report.add_section(case_a_section(weekly_a))

    report.add_section(
        _eda_section(
            master_b,
            section_id="eda_b",
            title="EDA · Caso B (líneas de ticket)",
            target="importe_linea",
            description="Tabla maestra de combos (detalle ⨝ tickets ⨝ catálogo).",
        )
    )
    report.add_section(case_b_section(master_b, baskets_b))

    report.add_section(
        _eda_section(
            master_c,
            section_id="eda_c",
            title="EDA · Caso C (tickets)",
            target="total_venta",
            description="Tabla maestra de AOV (transacciones ⨝ loyalty ⨝ exógenas ⨝ promos).",
        )
    )
    report.add_section(case_c_section(master_c))

    return report
