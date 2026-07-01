"""Reglas de narración: convierten cifras de un paso en :class:`Insight`.

Cada función es pura (cifras → texto + severidad) y usa
:class:`NarrationThresholds` para decidir el juicio. Se reutilizan en EDA, HPO y
evaluación, de modo que el criterio narrativo es consistente en todo el proyecto.
"""

from __future__ import annotations

import math

from .insight import Insight, Severity
from .thresholds import NarrationThresholds

_DEFAULTS = NarrationThresholds()


def correlation_insight(
    var_a: str,
    var_b: str,
    corr: float,
    method: str = "Pearson",
    thresholds: NarrationThresholds = _DEFAULTS,
) -> Insight:
    """Describe la correlación entre dos variables (dirección y magnitud)."""
    direction = "positiva" if corr >= 0 else "negativa"
    label = thresholds.corr_label(corr)
    severity = Severity.WARNING if abs(corr) >= thresholds.corr_strong else Severity.INFO
    text = f"{method} entre «{var_a}» y «{var_b}» = {corr:+.2f}: correlación {label} {direction}."
    return Insight(
        text=text,
        severity=severity,
        metrics={f"{method}": round(corr, 4)},
        tags=("eda", "correlation"),
    )


def vif_insight(
    feature: str, vif: float, thresholds: NarrationThresholds = _DEFAULTS
) -> Insight | None:
    """Marca multicolinealidad según VIF; devuelve ``None`` si es bajo."""
    if math.isinf(vif) or vif >= thresholds.vif_severe:
        return Insight(
            text=(
                f"VIF({feature}) = {'∞' if math.isinf(vif) else f'{vif:.1f}'} "
                f"> {thresholds.vif_severe:.0f}: multicolinealidad severa; "
                f"considerar drop/PCA/regularización."
            ),
            severity=Severity.CRITICAL,
            metrics={"vif": float("inf") if math.isinf(vif) else round(vif, 2)},
            tags=("eda", "vif"),
        )
    if vif >= thresholds.vif_moderate:
        return Insight(
            text=(
                f"VIF({feature}) = {vif:.1f} entre {thresholds.vif_moderate:.0f} y "
                f"{thresholds.vif_severe:.0f}: multicolinealidad moderada."
            ),
            severity=Severity.WARNING,
            metrics={"vif": round(vif, 2)},
            tags=("eda", "vif"),
        )
    return None


def normality_insight(
    variable: str,
    pvalue: float,
    test_name: str = "Shapiro-Wilk",
    thresholds: NarrationThresholds = _DEFAULTS,
) -> Insight:
    """Interpreta un test de normalidad sobre una variable numérica."""
    normal = pvalue >= thresholds.pvalue_alpha
    text = (
        f"{test_name} sobre «{variable}»: p = {pvalue:.3g} "
        f"({'no se rechaza' if normal else 'se rechaza'} la normalidad al "
        f"{int((1 - thresholds.pvalue_alpha) * 100)}%)."
    )
    return Insight(
        text=text,
        severity=Severity.INFO if normal else Severity.WARNING,
        metrics={"pvalue": round(pvalue, 5)},
        tags=("eda", "normality"),
    )


def group_difference_insight(
    feature: str,
    target: str,
    pvalue: float,
    effect_size: float,
    test_name: str,
    effect_name: str = "Cohen's d",
    thresholds: NarrationThresholds = _DEFAULTS,
) -> Insight:
    """Resume una prueba de diferencia entre grupos con su tamaño de efecto."""
    significant = pvalue < thresholds.pvalue_alpha
    effect_label = thresholds.effect_label(effect_size)
    if significant and abs(effect_size) >= thresholds.effect_medium:
        severity = Severity.GOOD
    elif significant:
        severity = Severity.INFO
    else:
        severity = Severity.WARNING
    verdict = "discrimina" if significant else "no discrimina de forma significativa"
    text = (
        f"«{feature}» {verdict} «{target}» ({test_name}: p = {pvalue:.3g}; "
        f"{effect_name} = {effect_size:+.2f}, efecto {effect_label})."
    )
    return Insight(
        text=text,
        severity=severity,
        metrics={"pvalue": round(pvalue, 5), effect_name: round(effect_size, 4)},
        tags=("eda", "bivariate"),
    )


def mutual_information_insight(
    feature: str,
    target: str,
    mi: float,
    thresholds: NarrationThresholds = _DEFAULTS,
) -> Insight | None:
    """Reporta información mutua feature–target si supera el umbral de relevancia."""
    if mi < thresholds.mi_relevant:
        return None
    return Insight(
        text=(
            f"Información mutua «{feature}»→«{target}» = {mi:.3f}: aporta "
            f"señal no lineal relevante."
        ),
        severity=Severity.GOOD,
        metrics={"mutual_information": round(mi, 4)},
        tags=("eda", "mutual_information"),
    )


def association_insight(
    var_a: str,
    var_b: str,
    cramers_v: float,
    pvalue: float,
    thresholds: NarrationThresholds = _DEFAULTS,
) -> Insight:
    """Describe la asociación categórica–categórica (chi² + Cramér's V)."""
    significant = pvalue < thresholds.pvalue_alpha
    label = thresholds.corr_label(cramers_v)
    severity = Severity.INFO
    if significant and cramers_v >= thresholds.corr_strong:
        severity = Severity.WARNING
    text = (
        f"«{var_a}» vs «{var_b}»: Cramér's V = {cramers_v:.2f} (asociación {label}), "
        f"chi² p = {pvalue:.3g}."
    )
    return Insight(
        text=text,
        severity=severity,
        metrics={"cramers_v": round(cramers_v, 4), "pvalue": round(pvalue, 5)},
        tags=("eda", "association"),
    )


def outlier_insight(variable: str, pct_outliers: float, method: str = "IQR") -> Insight:
    """Reporta la proporción de outliers detectada en una variable."""
    severity = Severity.WARNING if pct_outliers >= 5.0 else Severity.INFO
    return Insight(
        text=f"«{variable}»: {pct_outliers:.1f}% de outliers ({method}).",
        severity=severity,
        metrics={"pct_outliers": round(pct_outliers, 2)},
        tags=("eda", "outliers"),
    )


def interval_coverage_insight(picp: float, nominal: float, mpiw: float) -> Insight:
    """Evalúa la calibración de intervalos de predicción (PICP vs nominal)."""
    gap = picp - nominal
    if abs(gap) <= 0.05:
        severity, verdict = Severity.GOOD, "bien calibrados"
    elif gap < 0:
        severity, verdict = Severity.WARNING, "sub-cubren (intervalos estrechos)"
    else:
        severity, verdict = Severity.INFO, "sobre-cubren (intervalos anchos)"
    text = (
        f"Cobertura de intervalos PICP = {picp:.1%} vs nominal {nominal:.0%} "
        f"({verdict}); ancho medio MPIW = {mpiw:.2f}."
    )
    return Insight(
        text=text,
        severity=severity,
        metrics={"picp": round(picp, 4), "nominal": nominal, "mpiw": round(mpiw, 4)},
        tags=("evaluation", "intervals"),
    )


def metric_vs_baseline_insight(
    metric_name: str,
    model_value: float,
    baseline_value: float,
    higher_is_better: bool,
) -> Insight:
    """Compara una métrica del modelo contra un baseline y cuantifica la mejora."""
    if baseline_value == 0:
        rel = math.inf if model_value != baseline_value else 0.0
    else:
        rel = (model_value - baseline_value) / abs(baseline_value) * 100.0
    improved = (
        (model_value > baseline_value) if higher_is_better else (model_value < baseline_value)
    )
    severity = Severity.GOOD if improved else Severity.WARNING
    arrow = "mejora" if improved else "empeora"
    text = (
        f"{metric_name}: modelo = {model_value:.4g} vs baseline = {baseline_value:.4g} "
        f"({arrow} {abs(rel):.1f}%)."
    )
    metrics: dict[str, float | int | str] = {
        f"{metric_name}_model": round(model_value, 6),
        f"{metric_name}_baseline": round(baseline_value, 6),
    }
    if math.isfinite(rel):
        metrics["rel_change_pct"] = round(rel, 2)
    return Insight(text=text, severity=severity, metrics=metrics, tags=("evaluation", "baseline"))
