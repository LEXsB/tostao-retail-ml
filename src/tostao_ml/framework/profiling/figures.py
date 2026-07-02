"""Ensamblado de figuras y tablas EDA a partir de un :class:`DatasetProfile`.

Mantiene ``viz/`` como fábrica pura de gráficos y ``profiling/`` como
orquestador: decide *qué* figura corresponde a cada variable según su tipo,
incluida la **apertura por la variable objetivo** para cada feature (§4.3).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from tostao_ml.framework.types import VariableKind
from tostao_ml.framework.viz import eda

from .engine import DatasetProfile

# Columnas de la tabla univariada relevantes para cada familia.
_NUMERIC_COLS = [
    "variable",
    "count",
    "pct_null",
    "pct_zero",
    "mean",
    "median",
    "std",
    "cv",
    "min",
    "p25",
    "p75",
    "p95",
    "max",
    "skewness",
    "kurtosis",
    "pct_outliers_iqr",
    "normality_test",
    "normality_pvalue",
]
_CATEGORICAL_COLS = [
    "variable",
    "count",
    "cardinality",
    "mode",
    "mode_freq_pct",
    "entropy_bits",
    "pct_null",
]


def _analyzable(profile: DatasetProfile) -> list[str]:
    """Columnas que entran al análisis (excluye identificadores y texto libre)."""
    return [
        c for c, k in profile.types.items() if k not in (VariableKind.IDENTIFIER, VariableKind.TEXT)
    ]


def variable_dictionary(profile: DatasetProfile) -> pd.DataFrame:
    """Diccionario de variables: nombre, tipo inferido y rol (target/feature/id)."""
    rows = []
    for col, kind in profile.types.items():
        if col == profile.target:
            role = "objetivo"
        elif kind in (VariableKind.IDENTIFIER, VariableKind.TEXT):
            role = "identificador (excluido)"
        else:
            role = "feature"
        rows.append({"variable": col, "tipo": kind.value, "rol": role})
    return pd.DataFrame(rows)


def numeric_summary_table(profile: DatasetProfile) -> pd.DataFrame:
    """Estadística univariada solo de numéricas analizables (sin IDs)."""
    if profile.univariate.empty:
        return pd.DataFrame()
    numeric_kinds = {VariableKind.NUMERIC_CONTINUOUS.value, VariableKind.NUMERIC_DISCRETE.value}
    tbl = profile.univariate[profile.univariate["kind"].isin(numeric_kinds)]
    cols = [c for c in _NUMERIC_COLS if c in tbl.columns]
    return tbl[cols].reset_index(drop=True).round(3)


def categorical_summary_table(profile: DatasetProfile) -> pd.DataFrame:
    """Estadística univariada de categóricas/binarias (sin IDs)."""
    if profile.univariate.empty:
        return pd.DataFrame()
    cat_kinds = {
        VariableKind.CATEGORICAL_NOMINAL.value,
        VariableKind.CATEGORICAL_ORDINAL.value,
        VariableKind.BINARY.value,
    }
    tbl = profile.univariate[profile.univariate["kind"].isin(cat_kinds)]
    cols = [c for c in _CATEGORICAL_COLS if c in tbl.columns]
    return tbl[cols].reset_index(drop=True).round(3)


def build_univariate_figures(
    frame: pd.DataFrame, profile: DatasetProfile, *, max_numeric: int = 12, max_categorical: int = 8
) -> dict[str, go.Figure]:
    """Figuras univariadas: histograma+KDE y box por numérica; Pareto por categórica."""
    figures: dict[str, go.Figure] = {}
    numeric = [c for c, k in profile.types.items() if k.is_numeric][:max_numeric]
    categorical = [
        c
        for c, k in profile.types.items()
        if k
        in (VariableKind.CATEGORICAL_NOMINAL, VariableKind.CATEGORICAL_ORDINAL, VariableKind.BINARY)
    ][:max_categorical]

    for col in numeric:
        figures[f"dist__{col}"] = eda.histogram_kde(frame[col], name=col)
        figures[f"box__{col}"] = eda.box_violin(frame[col], name=col)
    for col in categorical:
        figures[f"pareto__{col}"] = eda.pareto(frame[col].value_counts(), name=col)
    if frame.isna().any().any():
        figures["missingness"] = eda.missingness_bar(frame)
    return figures


def build_correlation_figures(profile: DatasetProfile) -> dict[str, go.Figure]:
    """Heatmaps de correlación (Pearson y Spearman)."""
    figures: dict[str, go.Figure] = {}
    pearson = profile.correlations.get("pearson")
    spearman = profile.correlations.get("spearman")
    if pearson is not None and not pearson.empty:
        figures["corr_pearson"] = eda.correlation_heatmap(
            pearson, title="Correlación de Pearson (lineal)"
        )
    if spearman is not None and not spearman.empty:
        figures["corr_spearman"] = eda.correlation_heatmap(
            spearman, title="Correlación de Spearman (monótona)"
        )
    return figures


def by_target_figure(
    frame: pd.DataFrame, feature: str, profile: DatasetProfile, *, n_bins: int = 3
) -> go.Figure | None:
    """Apertura de un feature por la variable objetivo (§4.3).

    Si el target es categórico, agrupa por sus niveles; si es continuo, lo
    discretiza en cuantiles (Bajo/Medio/Alto). Para features numéricos usa boxplot
    por grupo; para categóricos, composición 100% apilada.
    """
    target = profile.target
    if target is None or feature == target:
        return None
    target_kind = profile.types[target]
    if target_kind.is_categorical:
        working, group = frame, target
    else:
        working = frame.assign(__nivel_objetivo=eda.target_bins(frame[target], n_bins))
        group = "__nivel_objetivo"

    feat_kind = profile.types[feature]
    if feat_kind.is_numeric:
        return eda.box_by_group(working, feature, group)
    if feat_kind.is_categorical:
        return eda.stacked_proportions(working, feature, group)
    return None


def build_target_analysis(
    frame: pd.DataFrame, profile: DatasetProfile, *, n_bins: int = 3, max_features: int = 14
) -> dict[str, go.Figure]:
    """Una figura de apertura por la variable objetivo para CADA feature analizable.

    Además, para features numéricos con target continuo añade la relación directa
    (scatter + tendencia).
    """
    figures: dict[str, go.Figure] = {}
    target = profile.target
    if target is None:
        return figures
    target_continuo = profile.types[target].is_numeric
    features = [c for c in _analyzable(profile) if c != target][:max_features]
    for feat in features:
        fig = by_target_figure(frame, feat, profile, n_bins=n_bins)
        if fig is not None:
            figures[f"target__{feat}"] = fig
        if target_continuo and profile.types[feat].is_numeric:
            figures[f"scatter__{feat}"] = eda.scatter_trend(frame, feat, target)
    return figures


def build_eda_figures(
    frame: pd.DataFrame, profile: DatasetProfile, *, max_numeric: int = 8, max_categorical: int = 6
) -> dict[str, go.Figure]:
    """Conjunto compacto de figuras EDA (compatibilidad); ver builders específicos."""
    figures = build_univariate_figures(
        frame, profile, max_numeric=max_numeric, max_categorical=max_categorical
    )
    figures.update(build_correlation_figures(profile))
    figures.update(build_target_analysis(frame, profile, max_features=6))
    return figures
