"""Reportes HTML ejecutivos por caso (presentación al negocio).

A diferencia del reporte completo (EDA a fondo + modelado interpretado en detalle),
el reporte ejecutivo es una lectura de una sola pasada pensada para quien decide:
la tarea propuesta, el enfoque en pocas líneas, las métricas clave **interpretadas**
y el impacto de negocio. No incluye el análisis exploratorio a fondo; ese detalle
vive en los notebooks (``notebooks/caso_*``) y en el reporte completo.

Reutiliza el mismo framework (``performance``) y la narración de ``storytelling``,
de modo que las cifras del reporte ejecutivo coinciden exactamente con las del
análisis completo.
"""

from __future__ import annotations

import pandas as pd

from tostao_ml.framework.evaluation import performance
from tostao_ml.framework.narrate import Insight, Narrative, Severity
from tostao_ml.framework.reporting import HTMLReport, ReportSection

from . import caso_a, caso_b, caso_c
from . import storytelling as st
from .reporting import _coef_figure

_FOOTER = "Tostao · Resumen ejecutivo — de los datos a la decisión."


def _approach(case_key: str, rows: list[tuple[str, str]], story: Narrative) -> ReportSection:
    """Sección de apertura: qué son los datos (story) + enfoque en pocas líneas."""
    return ReportSection(
        id=f"{case_key}_enfoque",
        title="Contexto y enfoque",
        description="Qué representan los datos y cómo se aborda el problema.",
        narrative=story,
    ).add_table("Enfoque de la solución", pd.DataFrame(rows, columns=["aspecto", "detalle"]))


def _report(
    case_key: str, title: str, frame: pd.DataFrame, sections: list[ReportSection]
) -> HTMLReport:
    """Ensambla un reporte ejecutivo con la tarea propuesta en el encabezado."""
    report = HTMLReport(
        title=title,
        subtitle=f"Resumen ejecutivo sobre {len(frame):,} registros — enfoque, resultados e impacto",
        context=st.context_html(case_key),
        footer=_FOOTER,
    )
    for section in sections:
        report.add_section(section)
    return report


def executive_a(weekly: pd.DataFrame, result: caso_a.CaseAResult | None = None) -> HTMLReport:
    """Reporte ejecutivo del Caso A (abastecimiento): pronóstico + política de pedido."""
    result = result or caso_a.run_case_a(weekly, tune=True, n_trials=20)
    m = result.metrics
    test = result.test

    enfoque = _approach(
        "a",
        [
            ("Objetivo", "Pronosticar la demanda semanal por SKU-tienda y decidir cuánto pedir."),
            ("Modelo", "Boosting cuantílico (intervalos) + optimizador de pedido newsvendor."),
            (
                "Validación",
                "Holdout temporal + walk-forward; HPO con Optuna; se comparan varios modelos.",
            ),
            (
                "Decisión",
                "El fractil crítico Cu/(Cu+Co) fija la cantidad que minimiza el costo esperado.",
            ),
        ],
        st.data_story_a(weekly),
    )

    resultados = ReportSection(
        id="a_resultados",
        title="Resultados clave",
        description="Calidad del pronóstico y fiabilidad de sus intervalos.",
        narrative=st.interpret_model_a(result),
    )
    resultados.add_table(
        "Métricas de forecast (holdout)",
        pd.DataFrame(
            {
                "métrica": ["WAPE", "R²", "MAE", "RMSE", "PICP (nom. 80%)"],
                "valor": [
                    f"{m['wape']:.1%}",
                    f"{m['r2']:.2f}",
                    f"{m['mae']:.2f}",
                    f"{m['rmse']:.2f}",
                    f"{m['picp']:.0%}",
                ],
            }
        ),
    )
    resultados.add_figure(
        "pred_vs_real", performance.pred_vs_actual(test["unidades_vendidas"], test["pred"])
    )

    impacto = ReportSection(
        id="a_impacto",
        title="Impacto de negocio",
        description="La política óptima frente a pedir lo del período anterior.",
        narrative=Narrative().add(
            Insight(
                text=(
                    f"La política newsvendor reduce el costo esperado de faltante+sobrante de "
                    f"{result.cost_naive:,.0f} a {result.cost_model:,.0f}: la incertidumbre del "
                    f"modelo se traduce en una decisión de pedido con ahorro material."
                ),
                severity=Severity.GOOD,
                title="Ahorro esperado",
            )
        ),
    )
    impacto.add_figure(
        "costo",
        performance.model_comparison_bar(
            {"Política óptima": result.cost_model, "Política ingenua": result.cost_naive},
            "costo esperado (menor es mejor)",
        ),
    )
    return _report("a", "Caso A — Abastecimiento", weekly, [enfoque, resultados, impacto])


def executive_b(
    master_b: pd.DataFrame, baskets: pd.DataFrame, result: caso_b.CaseBResult | None = None
) -> HTMLReport:
    """Reporte ejecutivo del Caso B (combos): segmentación + reglas de co-compra."""
    result = result or caso_b.run_case_b(master_b, baskets, tune=True)

    enfoque = _approach(
        "b",
        [
            ("Objetivo", "Proponer los mejores combos por segmento de tienda y su lift esperado."),
            ("Modelos", "Clustering de tiendas (K-Means) + reglas de asociación FP-Growth."),
            (
                "Validación",
                "k por máxima silhouette; reglas filtradas por lift>1 y soporte mínimo.",
            ),
            (
                "Decisión",
                "Por cluster se activan los combos de mayor lift con precio con descuento.",
            ),
        ],
        st.data_story_b(master_b),
    )

    n_rules = len(result.rules)
    best_lift = float(result.rules["lift"].max()) if not result.rules.empty else 0.0
    n_combos = len(result.combos)
    resultados = ReportSection(
        id="b_resultados",
        title="Resultados clave",
        description="Validez de la segmentación y robustez de las reglas de co-compra.",
        narrative=st.interpret_model_b(result),
    )
    resultados.add_table(
        "Resumen",
        pd.DataFrame(
            {
                "métrica": [
                    "Clusters de tienda",
                    "Silhouette",
                    "Reglas (lift>1)",
                    "Lift máximo",
                    "Combos propuestos",
                ],
                "valor": [
                    str(int(result.store_clusters.nunique())),
                    f"{result.silhouette:.2f}",
                    str(n_rules),
                    f"{best_lift:.1f}x",
                    str(n_combos),
                ],
            }
        ),
    )
    if not result.combos.empty:
        top = result.combos.sort_values("lift", ascending=False).head(10)
        combo_lift = top.assign(combo=top["producto_a"] + " + " + top["producto_b"]).set_index(
            "combo"
        )["lift"]
        resultados.add_figure(
            "combos_lift",
            performance.model_comparison_bar(combo_lift.to_dict(), "lift (mayor es mejor)"),
        )

    impacto = ReportSection(
        id="b_impacto",
        title="Combos recomendados",
        description="Los combos de mayor potencial por cluster, listos para activar.",
    )
    if not result.combos.empty:
        cols = [
            c
            for c in ["cluster", "producto_a", "producto_b", "lift", "precio_combo"]
            if c in result.combos
        ]
        impacto.add_table(
            "Top combos por cluster",
            result.combos.sort_values("lift", ascending=False).head(12)[cols].round(2),
        )
    return _report("b", "Caso B — Combos", master_b, [enfoque, resultados, impacto])


def executive_c(master_c: pd.DataFrame, result: caso_c.CaseCResult | None = None) -> HTMLReport:
    """Reporte ejecutivo del Caso C (AOV): drivers del ticket + gasto esperado."""
    result = result or caso_c.run_case_c(master_c, tune=True)
    pm = result.predictive_metrics
    n_sig = int((result.coefficients["pvalue"] < 0.05).sum())

    enfoque = _approach(
        "c",
        [
            (
                "Objetivo",
                "Entender qué mueve el ticket y predecir el gasto del cliente recurrente.",
            ),
            (
                "Modelos",
                "GLM inferencial (coeficientes + IC) + boosting predictivo sobre RFM+loyalty.",
            ),
            (
                "Validación",
                "GLM con p-valores e IC95%; predictivo en holdout 80/20 con HPO (Optuna).",
            ),
            (
                "Decisión",
                "Los drivers significativos orientan acciones; el gasto prioriza clientes por valor.",
            ),
        ],
        st.data_story_c(master_c),
    )

    resultados = ReportSection(
        id="c_resultados",
        title="Resultados clave",
        description="Qué drivers son significativos y qué tan bien se predice el gasto.",
        narrative=st.interpret_model_c(result),
    )
    resultados.add_table(
        "Resumen",
        pd.DataFrame(
            {
                "métrica": ["Drivers significativos (p<0.05)", "R² predictivo", "WAPE", "MAE"],
                "valor": [
                    str(n_sig),
                    f"{pm['r2']:.2f}",
                    f"{pm['wape']:.1%}",
                    f"{pm['mae']:.2f}",
                ],
            }
        ),
    )
    resultados.add_figure("coeficientes", _coef_figure(result.coefficients))

    impacto = ReportSection(
        id="c_impacto",
        title="Predicción de gasto",
        description="Ajuste del modelo de gasto esperado sobre el holdout.",
    )
    test = result.predictive_test
    impacto.add_figure(
        "pred_vs_real", performance.pred_vs_actual(test["ticket_medio"], test["pred"])
    )
    return _report("c", "Caso C — AOV", master_c, [enfoque, resultados, impacto])
