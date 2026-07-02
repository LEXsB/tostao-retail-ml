"""Ensamblado de reportes HTML **completos** por caso y unificado.

El reporte cuenta una historia: encabezado con la **tarea propuesta**, descripción
e interpretación de los datos (EDA), y **lectura de los resultados** del modelo
(si son significativos, si son robustos y qué decisión implican). Cada figura y
tabla acompaña a un análisis; no hay salidas sueltas ni definiciones de métricas.
Reutiliza el framework (``profiling``, ``performance``, ``interpret``) y la
narración de ``storytelling``.
"""

from __future__ import annotations

from collections import Counter
from typing import cast

import pandas as pd
import plotly.graph_objects as go

from tostao_ml.framework.evaluation import metrics, performance
from tostao_ml.framework.interpret import (
    compute_shap,
    permutation_bar,
    permutation_importance,
    shap_summary_bar,
)
from tostao_ml.framework.models.base import BaseModel
from tostao_ml.framework.narrate import Insight, Narrative, Severity
from tostao_ml.framework.profiling import (
    build_correlation_figures,
    build_target_analysis,
    build_univariate_figures,
    categorical_summary_table,
    numeric_summary_table,
    profile_dataset,
    variable_dictionary,
)
from tostao_ml.framework.profiling.engine import DatasetProfile
from tostao_ml.framework.reporting import HTMLReport, ReportSection
from tostao_ml.framework.tuning import TuningResult
from tostao_ml.framework.viz import COLORS
from tostao_ml.framework.viz import eda as eda_viz

from . import caso_a, caso_b, caso_c
from . import storytelling as st


def _pick(figures: dict[str, go.Figure], *prefixes: str) -> dict[str, go.Figure]:
    return {k: v for k, v in figures.items() if k.startswith(prefixes)}


def _dims_insight(frame: pd.DataFrame, profile: DatasetProfile) -> Insight:
    comp = Counter(t.value for t in profile.types.values())
    detalle = ", ".join(f"{v} {k}" for k, v in comp.items())
    return Insight(
        text=f"La tabla maestra tiene {frame.shape[0]:,} filas × {frame.shape[1]} columnas ({detalle}).",
        severity=Severity.INFO,
        metrics={"n_rows": frame.shape[0], "n_cols": frame.shape[1]},
        tags=("eda", "overview"),
        title="Dimensiones",
    )


# --------------------------------------------------------------------------- #
# Secciones de EDA (comunes a los tres casos)
# --------------------------------------------------------------------------- #
def eda_sections(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    *,
    prefix: str,
    join_report: dict[str, float] | None,
    description: str,
    full: bool = True,
) -> list[ReportSection]:
    """Construye las secciones de EDA (diccionario, univariado, target, multivariado)."""
    sections: list[ReportSection] = []
    target = profile.target

    # 1) Resumen + diccionario de variables + cobertura de cruces.
    overview_narr = Narrative().add(_dims_insight(frame, profile))
    overview_narr.extend(list(profile.narrative)[-1:])  # insight de composición
    overview = ReportSection(
        id=f"{prefix}_resumen",
        title="Resumen y variables",
        description=description,
        narrative=overview_narr,
    )
    if join_report:
        overview.add_table(
            "Cobertura de cada cruce (integridad referencial)",
            pd.DataFrame(
                {
                    "fuente_cruzada": list(join_report),
                    "cobertura_%": [round(v * 100, 2) for v in join_report.values()],
                }
            ),
        )
    overview.add_table("Diccionario de variables (tipo y rol)", variable_dictionary(profile))
    sections.append(overview)

    # 2) Univariado numérico (estadística detallada + figuras).
    num_tbl = numeric_summary_table(profile)
    if not num_tbl.empty:
        uni_num = ReportSection(
            id=f"{prefix}_uni_num",
            title="Univariado · variables numéricas",
            description="Estadística detallada (tendencia, dispersión, forma, outliers, normalidad) e histogramas/box por variable.",
        )
        uni_num.add_table("Estadística numérica", num_tbl)
        if full:
            uni_num.figures = _pick(build_univariate_figures(frame, profile), "dist__", "box__")
        sections.append(uni_num)

    # 3) Univariado categórico.
    cat_tbl = categorical_summary_table(profile)
    if not cat_tbl.empty:
        uni_cat = ReportSection(
            id=f"{prefix}_uni_cat",
            title="Univariado · variables categóricas",
            description="Cardinalidad, moda, entropía y % de nulos; Pareto de frecuencias por variable.",
        )
        uni_cat.add_table("Estadística categórica", cat_tbl)
        if full:
            uni_cat.figures = _pick(build_univariate_figures(frame, profile), "pareto__")
        sections.append(uni_cat)

    # 4) Apertura por la variable objetivo (§4.3) — clave.
    if target is not None:
        tgt = ReportSection(
            id=f"{prefix}_target",
            title=f"Apertura por la variable objetivo «{target}»",
            description=(
                "Distribución de cada feature condicionada al nivel del objetivo "
                "(el target continuo se discretiza en Bajo/Medio/Alto) más los tests de "
                "diferencia y la relación directa (dispersión + tendencia)."
            ),
            narrative=_target_narrative(profile),
        )
        if not profile.bivariate.empty:
            tgt.add_table(
                "Tests bivariados feature–objetivo (con tamaño de efecto)",
                profile.bivariate.round(4),
            )
        if not profile.mutual_information.empty:
            tgt.add_table(
                "Información mutua feature→objetivo",
                profile.mutual_information.round(4).to_frame("MI"),
            )
        if full:
            tgt.figures = build_target_analysis(frame, profile, max_features=14)
        sections.append(tgt)

    # 5) Multivariado: correlaciones, VIF, redundancia.
    multi = ReportSection(
        id=f"{prefix}_multi",
        title="Multivariado · correlaciones y redundancia",
        description="Correlaciones lineales (Pearson) y monótonas (Spearman), multicolinealidad (VIF) y pares redundantes.",
        narrative=_multi_narrative(profile),
    )
    if not profile.vif.empty:
        multi.add_table(
            "VIF por variable (multicolinealidad)", profile.vif.round(3).to_frame("VIF")
        )
    pairs = profile.redundant_pairs(0.8)
    if pairs:
        multi.add_table(
            "Pares redundantes (|Pearson| ≥ 0.8)",
            pd.DataFrame(pairs, columns=["variable_a", "variable_b", "pearson"]).round(3),
        )
    if full:
        multi.figures = build_correlation_figures(profile)
    sections.append(multi)
    return sections


def _target_narrative(profile: DatasetProfile) -> Narrative:
    narr = Narrative()
    for ins in profile.narrative:
        if "bivariate" in ins.tags or "mutual_information" in ins.tags or "association" in ins.tags:
            narr.add(ins)
    return narr


def _multi_narrative(profile: DatasetProfile) -> Narrative:
    narr = Narrative()
    for ins in profile.narrative:
        if "vif" in ins.tags or "correlation" in ins.tags:
            narr.add(ins)
    return narr


# --------------------------------------------------------------------------- #
# Secciones de modelado por caso
# --------------------------------------------------------------------------- #
def _coef_figure(coefs: pd.DataFrame, top: int = 12) -> go.Figure:
    """Barras horizontales de coeficientes GLM con barras de error (IC 95%)."""
    top_coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index).head(top)[
        ::-1
    ]
    err = (top_coefs["ci_upper"] - top_coefs["ci_lower"]) / 2
    fig = go.Figure(
        go.Bar(
            x=top_coefs["coef"],
            y=list(top_coefs.index),
            orientation="h",
            marker_color=[
                COLORS["positive"] if c > 0 else COLORS["critical"] for c in top_coefs["coef"]
            ],
            error_x={"type": "data", "array": err.to_numpy(), "color": COLORS["neutral"]},
        )
    )
    fig.add_vline(x=0, line={"color": COLORS["neutral"], "dash": "dash"})
    fig.update_xaxes(title="Coeficiente β (IC 95%)")
    return eda_viz.apply_theme(fig, title="Drivers del ticket (GLM)")


def _hpo_detail(tuning: TuningResult | None) -> str:
    """Frase con la mejor configuración de HPO y el hiperparámetro más influyente."""
    if tuning is None:
        return "No se ejecutó HPO en este run; se usó una configuración por defecto razonable."
    imp = tuning.param_importances or {}
    n = len(tuning.study.trials)
    top = max(imp.items(), key=lambda kv: kv[1])[0] if imp else "—"
    cfg = ", ".join(
        f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in tuning.best_params.items()
    )
    return f"Optuna (TPE, {n} trials); mejor config: {cfg}; hiperparámetro más influyente: «{top}»."


def _strategy_section(
    case_id: str, title: str, rows: list[tuple[str, str]], insights: list[Insight]
) -> ReportSection:
    """Sección corta y estándar de estrategia de modelamiento y validación."""
    narr = Narrative()
    for ins in insights:
        narr.add(ins)
    return ReportSection(
        id=case_id,
        title=title,
        description="Cómo se modela, cómo se valida, qué métricas se usan y qué decisión se toma (y por qué).",
        narrative=narr,
    ).add_table(
        "Estrategia de modelamiento y validación",
        pd.DataFrame(rows, columns=["aspecto", "detalle"]),
    )


def strategy_a(result: caso_a.CaseAResult) -> ReportSection:
    """Seccion con la estrategia de modelamiento y validacion del Caso A."""
    m = result.metrics
    rows = [
        (
            "Tipo de tarea",
            "Regresión / forecasting probabilístico de demanda semanal por SKU-tienda.",
        ),
        (
            "Modelo",
            "Gradient boosting cuantílico (HistGradientBoosting, pérdida pinball): un estimador por cuantil (0.1/0.5/0.9).",
        ),
        (
            "¿Por qué este modelo?",
            "La decisión de pedido depende de la incertidumbre, no solo del valor esperado; los cuantiles dan intervalos y permiten el newsvendor.",
        ),
        (
            "Partición train/test",
            "Holdout temporal: las últimas 3 semanas son test y el resto es train (sin leakage de futuro).",
        ),
        (
            "Validación",
            "Walk-forward (TimeSeriesSplit, 3 folds) dentro de train — respeta el orden temporal.",
        ),
        ("Optimización de hiperparámetros", _hpo_detail(result.tuning)),
        (
            "Métricas",
            "MAE/RMSE (unidades), WAPE (% error robusto), R² (varianza explicada); pinball, PICP y MPIW para los intervalos.",
        ),
        (
            "Decisión y por qué",
            "Con los cuantiles se calcula el fractil crítico Cu/(Cu+Co) → nivel objetivo S* → cantidad a pedir que minimiza el costo esperado de faltante+sobrante.",
        ),
    ]
    insights = [
        Insight(
            text="No hay tarea de clasificación en este caso; de haberla, se reportarían accuracy/F1/ROC-AUC/PR-AUC (disponibles en el framework).",
            severity=Severity.INFO,
            tags=("nota",),
        ),
    ]
    _ = m  # las métricas se leen e interpretan en la sección de resultados
    return _strategy_section(
        "a_estrategia", "Caso A · Estrategia de modelamiento y validación", rows, insights
    )


def strategy_b(result: caso_b.CaseBResult) -> ReportSection:
    """Seccion con la estrategia de modelamiento y validacion del Caso B."""
    hpo = (
        (
            f"Selección de k por máxima silhouette (barrido {list(result.k_selection['k'])}); "
            f"k* = {result.store_clusters.nunique()}."
        )
        if result.k_selection is not None
        else f"k fijado en {result.store_clusters.nunique()} (silhouette {result.silhouette:.2f})."
    )
    rows = [
        (
            "Tipo de tarea",
            "Aprendizaje NO supervisado: clustering de tiendas + reglas de asociación.",
        ),
        (
            "Modelos",
            "K-Means (segmentación por perfil de compra) + FP-Growth (reglas de co-compra).",
        ),
        (
            "¿Por qué?",
            "No hay etiqueta objetivo; se busca estructura (segmentos) y patrones de co-compra para diseñar combos.",
        ),
        ("Partición train/test", "No aplica (no supervisado): se usa todo el histórico de cestas."),
        ("Validación / selección", hpo + " Reglas filtradas por lift>1 y soporte mínimo."),
        (
            "Métricas",
            "Clustering: silhouette (−1..1; cohesión vs. separación). Reglas: support, confidence, lift, conviction.",
        ),
        (
            "Decisión y por qué",
            "Por cada cluster se eligen los combos con mayor lift (co-compra más frecuente de lo esperado) y se fija un precio con descuento.",
        ),
    ]
    return _strategy_section(
        "b_estrategia", "Caso B · Estrategia de modelamiento y validación", rows, []
    )


def strategy_c(result: caso_c.CaseCResult) -> ReportSection:
    """Seccion con la estrategia de modelamiento y validacion del Caso C."""
    pm = result.predictive_metrics
    rows = [
        (
            "Tipo de tarea",
            "Dos tareas: (1) inferencia de drivers del ticket; (2) regresión predictiva del gasto del cliente recurrente.",
        ),
        (
            "Modelos",
            "GLM gaussiano (inferencial, coeficientes + IC) + gradient boosting (predictivo, features RFM + loyalty).",
        ),
        (
            "¿Por qué?",
            "El GLM da efectos interpretables con significancia; el boosting captura no linealidades para predecir el gasto.",
        ),
        (
            "Partición train/test",
            "Predictivo: holdout aleatorio 80/20 sobre clientes recurrentes (≥2 visitas). GLM: se ajusta sobre todo el conjunto (inferencia, no predicción).",
        ),
        (
            "Validación",
            "Predictivo: métricas en holdout vs. baseline de la media. GLM: p-valores e IC95%; outliers winsorizados.",
        ),
        (
            "Optimización de hiperparámetros",
            _hpo_detail(result.tuning) + " (modelo predictivo, K-Fold).",
        ),
        (
            "Métricas",
            "GLM: β (efecto), p-valor (significancia), IC95%. Predictivo: MAE/RMSE/WAPE/R².",
        ),
        (
            "Decisión y por qué",
            "Los drivers significativos (p<0.05) orientan acciones de negocio; el modelo de gasto prioriza clientes por valor esperado.",
        ),
    ]
    _ = pm  # las métricas se leen e interpretan en la sección de resultados
    return _strategy_section(
        "c_estrategia", "Caso C · Estrategia de modelamiento y validación", rows, []
    )


def case_a_model_sections(weekly: pd.DataFrame) -> list[ReportSection]:
    """Construye las secciones de modelado e interpretabilidad del Caso A."""
    result = caso_a.run_case_a(weekly, tune=True, n_trials=20)
    test = result.test
    perf = ReportSection(
        id="a_modelo",
        title="Caso A · Lectura de resultados y desempeño",
        description="Se analiza si el pronóstico es bueno, si sus intervalos son fiables y qué impacto de negocio genera la política de pedido. Las figuras (predicho vs. real, residuales, error por producto) sustentan esta lectura.",
        narrative=st.interpret_model_a(result),
    )
    perf.add_table(
        "Métricas de forecast",
        pd.DataFrame([result.metrics]).T.rename(columns={0: "valor"}).round(4),
    )
    perf.add_figure(
        "pred_vs_real", performance.pred_vs_actual(test["unidades_vendidas"], test["pred"])
    )
    perf.add_figure(
        "residuales", performance.residuals_vs_pred(test["unidades_vendidas"], test["pred"])
    )
    perf.add_figure(
        "hist_residuales", performance.residual_hist(test["unidades_vendidas"], test["pred"])
    )
    perf.add_figure(
        "error_por_producto",
        performance.error_by_segment(
            test.assign(err=(test["unidades_vendidas"] - test["pred"]).abs()), "id_producto", "err"
        ),
    )
    if result.comparison is not None:
        perf.add_table(
            "Comparación de modelos en el holdout (ordenada por WAPE)", result.comparison.round(4)
        )
        perf.add_figure(
            "comparacion",
            performance.model_comparison_bar(
                result.comparison["wape"].to_dict(), "WAPE (menor es mejor)"
            ),
        )

    interp = ReportSection(
        id="a_interpret",
        title="Caso A · Interpretabilidad y negocio",
        description="Importancia de features (permutación) e impacto económico del optimizador de pedido.",
    )
    if result.model is not None:
        imp = permutation_importance(
            cast(BaseModel, result.model),
            test[result.feature_names],
            test["unidades_vendidas"],
            metrics.rmse,
            n_repeats=3,
        )
        interp.add_figure("importancia", permutation_bar(imp))
    interp.add_figure(
        "costo",
        performance.model_comparison_bar(
            {"Política óptima": result.cost_model, "Política ingenua": result.cost_naive},
            "costo esperado",
        ),
    )
    interp.add_table("Muestra de pedidos recomendados", result.orders.head(15).round(2))
    return [strategy_a(result), perf, interp]


def case_b_model_sections(master_b: pd.DataFrame, baskets: pd.DataFrame) -> list[ReportSection]:
    """Construye las secciones de segmentacion, reglas y combos del Caso B."""
    result = caso_b.run_case_b(master_b, baskets, tune=True)
    seg = ReportSection(
        id="b_modelo",
        title="Caso B · Lectura de la segmentación y las reglas",
        description="Se analiza si los clusters son válidos y si las reglas de co-compra son robustas (no ruido). Las tablas de perfil, reglas y el barrido de k sustentan la lectura.",
        narrative=st.interpret_model_b(result),
    )
    profiles = result.store_profiles.join(result.store_clusters)
    seg.add_table("Perfil de tiendas + cluster", profiles.round(3).reset_index())
    seg.add_figure(
        "clusters",
        eda_viz.pareto(result.store_clusters.value_counts().sort_index(), name="cluster"),
    )
    if not result.rules.empty:
        rules = result.rules.assign(
            antecedente=result.rules["antecedents"].map(lambda s: ", ".join(sorted(s))),
            consecuente=result.rules["consequents"].map(lambda s: ", ".join(sorted(s))),
        )
        seg.add_table(
            "Reglas de asociación (Top por lift)",
            rules[["antecedente", "consecuente", "support", "confidence", "lift"]]
            .head(20)
            .round(4),
        )

    combos = ReportSection(
        id="b_combos",
        title="Caso B · Combos propuestos",
        description="Top combos por cluster con precio propuesto (descuento) y lift esperado.",
    )
    if not result.combos.empty:
        top = result.combos.sort_values("lift", ascending=False).head(12)
        combo_lift = top.assign(combo=top["producto_a"] + " + " + top["producto_b"]).set_index(
            "combo"
        )["lift"]
        combos.add_figure("combos_lift", eda_viz.pareto(combo_lift, name="combo (lift)"))
        combos.add_table("Combos por cluster", result.combos)
    if result.k_selection is not None:
        seg.add_table("Selección de k por silhouette (K-Means)", result.k_selection)
    if result.clustering_comparison is not None:
        seg.add_table(
            "Comparación de algoritmos de clustering (silhouette)", result.clustering_comparison
        )
    if result.graph_centrality is not None:
        combos.add_table(
            "Grafo de co-compra — productos «hub» por centralidad (enfoque complementario)",
            result.graph_centrality,
        )
    return [strategy_b(result), seg, combos]


def case_c_model_sections(master_c: pd.DataFrame) -> list[ReportSection]:
    """Construye las secciones de drivers y prediccion del Caso C."""
    result = caso_c.run_case_c(master_c, tune=True)
    drivers = ReportSection(
        id="c_modelo",
        title="Caso C · Lectura de resultados (drivers y predicción)",
        description="Se analiza qué drivers mueven el ticket y si son significativos, y qué tan bien predice el gasto el modelo. Los coeficientes con IC y las métricas sustentan la lectura.",
        narrative=st.interpret_model_c(result),
    )
    drivers.add_figure("coeficientes", _coef_figure(result.coefficients))
    drivers.add_table("Coeficientes (β, error, p-valor, IC 95%)", result.coefficients.round(4))

    pred = ReportSection(
        id="c_pred",
        title="Caso C · Predicción de gasto e interpretabilidad",
        description="Modelo de gasto esperado del cliente recurrente (RFM + loyalty) e importancia de features (SHAP).",
    )
    test = result.predictive_test
    pred.add_table(
        "Métricas del modelo predictivo",
        pd.DataFrame([result.predictive_metrics]).T.rename(columns={0: "valor"}).round(4),
    )
    if result.comparison is not None:
        pred.add_table(
            "Comparación de modelos predictivos en el holdout (ordenada por WAPE)",
            result.comparison.round(4),
        )
        pred.add_figure(
            "comparacion",
            performance.model_comparison_bar(
                result.comparison["wape"].to_dict(), "WAPE (menor es mejor)"
            ),
        )
    pred.add_figure("pred_vs_real", performance.pred_vs_actual(test["ticket_medio"], test["pred"]))
    pred.add_figure("residuales", performance.residuals_vs_pred(test["ticket_medio"], test["pred"]))
    if result.predictive_model is not None and result.predictive_features is not None:
        shap_res = compute_shap(
            cast(BaseModel, result.predictive_model), result.predictive_features, sample_size=120
        )
        pred.add_figure("shap", shap_summary_bar(shap_res))
    return [strategy_c(result), drivers, pred]


# --------------------------------------------------------------------------- #
# Reportes completos
# --------------------------------------------------------------------------- #
_DATA_STORY = {"a": st.data_story_a, "b": st.data_story_b, "c": st.data_story_c}


def _story_section(case_key: str, frame: pd.DataFrame, profile: DatasetProfile) -> ReportSection:
    """Sección de apertura: historia de los datos + lectura del EDA (interpretación)."""
    narr = Narrative()
    narr.extend(_DATA_STORY[case_key](frame))
    narr.extend(st.interpret_eda(profile))
    return ReportSection(
        id=f"{case_key}_datos",
        title="Contexto y lectura de los datos",
        description="Qué representan los datos y qué revela el análisis exploratorio antes de modelar.",
        narrative=narr,
    )


def full_case_report(
    case_key: str,
    title: str,
    frame: pd.DataFrame,
    target: str,
    description: str,
    join_report: dict[str, float] | None,
    model_sections: list[ReportSection],
) -> HTMLReport:
    """Reporte HTML completo de un caso: contexto → datos → EDA → modelado interpretado."""
    profile = profile_dataset(frame, name=case_key, target=target)
    report = HTMLReport(
        title=title,
        subtitle=f"De los datos crudos a la decisión — tabla maestra de {len(frame):,} filas × {frame.shape[1]} columnas",
        context=st.context_html(case_key),
    )
    report.add_section(_story_section(case_key, frame, profile))
    for section in eda_sections(
        frame, profile, prefix=case_key, join_report=join_report, description=description, full=True
    ):
        report.add_section(section)
    for section in model_sections:
        report.add_section(section)
    return report


def build_unified_report(
    weekly_a: pd.DataFrame,
    master_a: pd.DataFrame,
    master_b: pd.DataFrame,
    baskets_b: pd.DataFrame,
    master_c: pd.DataFrame,
) -> HTMLReport:
    """Reporte HTML unificado ejecutivo (EDA condensado + modelado de los tres casos)."""
    report = HTMLReport(
        title="Tostao · Plataforma DS/ML retail — Reporte unificado",
        subtitle="Abastecimiento, Combos y AOV sobre un mismo framework reutilizable",
    )
    intro = Narrative().add(
        Insight(
            text=(
                "Tres casos resueltos sobre un núcleo común: forecast probabilístico + optimización de "
                "pedido (A), clustering + reglas de asociación (B) y modelado inferencial/predictivo del "
                "ticket (C). Cada caso cruza todas sus fuentes en una tabla maestra."
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
            description="Una plataforma, tres casos, un mismo framework reutilizable.",
            narrative=intro,
        )
    )

    for prefix, frame, target, desc in [
        (
            "a",
            weekly_a,
            "unidades_vendidas",
            "Caso A — demanda semanal (ventas ⨝ catálogo ⨝ tiendas ⨝ inventario ⨝ tendencias).",
        ),
        (
            "b",
            master_b,
            "importe_linea",
            "Caso B — líneas de ticket (detalle ⨝ tickets ⨝ catálogo).",
        ),
        (
            "c",
            master_c,
            "total_venta",
            "Caso C — tickets (transacciones ⨝ loyalty ⨝ exógenas ⨝ promos).",
        ),
    ]:
        profile = profile_dataset(frame, name=prefix, target=target)
        report.add_section(_story_section(prefix, frame, profile))
        # EDA condensado (diccionario + target + multivariado; sin todas las figuras univariadas).
        for section in eda_sections(
            frame, profile, prefix=f"u{prefix}", join_report=None, description=desc, full=False
        ):
            if section.id.endswith(("_target", "_multi")):
                report.add_section(section)

    for section in case_a_model_sections(weekly_a):
        report.add_section(section)
    for section in case_b_model_sections(master_b, baskets_b):
        report.add_section(section)
    for section in case_c_model_sections(master_c):
        report.add_section(section)
    return report
