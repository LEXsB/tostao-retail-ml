"""Figuras Plotly para análisis exploratorio (EDA).

Todas devuelven ``go.Figure`` con la plantilla de marca aplicada, de modo que se
reutilizan idénticas en notebooks y en el reporte HTML. No cargan datos ni
tienen efectos colaterales: reciben Series/DataFrames y devuelven la figura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .theme import COLORS, QUALITATIVE, apply_theme


def histogram_kde(series: pd.Series, name: str | None = None, nbins: int = 40) -> go.Figure:
    """Histograma de densidad con curva KDE superpuesta."""
    name = name or str(series.name or "variable")
    values = series.dropna().to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_histogram(
        x=values,
        histnorm="probability density",
        nbinsx=nbins,
        marker_color=COLORS["secondary"],
        opacity=0.75,
        name="Histograma",
    )
    if values.size > 1 and np.ptp(values) > 0:
        kde = stats.gaussian_kde(values)
        grid = np.linspace(values.min(), values.max(), 200)
        fig.add_scatter(
            x=grid,
            y=kde(grid),
            mode="lines",
            line={"color": COLORS["accent"], "width": 2},
            name="KDE",
        )
    return apply_theme(fig, title=f"Distribución de {name}")


def box_violin(series: pd.Series, name: str | None = None) -> go.Figure:
    """Violín con boxplot interno para ver forma, mediana y outliers."""
    name = name or str(series.name or "variable")
    fig = go.Figure()
    fig.add_violin(
        y=series.dropna().to_numpy(dtype=float),
        box_visible=True,
        meanline_visible=True,
        fillcolor=COLORS["secondary"],
        line_color=COLORS["primary"],
        opacity=0.7,
        name=name,
    )
    return apply_theme(fig, title=f"Dispersión de {name}")


def ecdf(series: pd.Series, name: str | None = None) -> go.Figure:
    """Función de distribución acumulada empírica (ECDF)."""
    name = name or str(series.name or "variable")
    values = np.sort(series.dropna().to_numpy(dtype=float))
    y = np.arange(1, values.size + 1) / max(values.size, 1)
    fig = go.Figure(go.Scatter(x=values, y=y, mode="lines", line={"color": COLORS["primary"]}))
    fig.update_yaxes(title="F(x)")
    fig.update_xaxes(title=name)
    return apply_theme(fig, title=f"ECDF de {name}")


def qq_plot(series: pd.Series, name: str | None = None) -> go.Figure:
    """QQ-plot contra la normal teórica (diagnóstico de normalidad)."""
    name = name or str(series.name or "variable")
    values = series.dropna().to_numpy(dtype=float)
    (osm, osr), (slope, intercept, _) = stats.probplot(values, dist="norm")
    fig = go.Figure()
    fig.add_scatter(
        x=osm, y=osr, mode="markers", marker={"color": COLORS["secondary"]}, name="Cuantiles"
    )
    fig.add_scatter(
        x=osm,
        y=slope * osm + intercept,
        mode="lines",
        line={"color": COLORS["accent"], "dash": "dash"},
        name="Referencia normal",
    )
    fig.update_xaxes(title="Cuantiles teóricos")
    fig.update_yaxes(title="Cuantiles muestrales")
    return apply_theme(fig, title=f"QQ-plot de {name}")


def pareto(counts: pd.Series, top: int = 15, name: str | None = None) -> go.Figure:
    """Barras de frecuencia con curva de Pareto (% acumulado)."""
    name = name or str(counts.name or "categoría")
    counts = counts.sort_values(ascending=False).head(top)
    cum_pct = counts.cumsum() / counts.sum() * 100.0
    fig = go.Figure()
    fig.add_bar(
        x=counts.index.astype(str),
        y=counts.to_numpy(),
        marker_color=COLORS["primary"],
        name="Frecuencia",
    )
    fig.add_scatter(
        x=counts.index.astype(str),
        y=cum_pct.to_numpy(),
        mode="lines+markers",
        line={"color": COLORS["accent"]},
        name="% acumulado",
        yaxis="y2",
    )
    fig.update_layout(
        yaxis2={
            "overlaying": "y",
            "side": "right",
            "range": [0, 105],
            "title": "% acumulado",
            "showgrid": False,
        }
    )
    return apply_theme(fig, title=f"Pareto de {name}")


def correlation_heatmap(corr: pd.DataFrame, title: str = "Matriz de correlación") -> go.Figure:
    """Heatmap de una matriz de correlación (diverging centrado en 0)."""
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "ρ"},
        )
    )
    fig.update_yaxes(autorange="reversed")
    return apply_theme(fig, title=title)


def grouped_distribution(
    frame: pd.DataFrame, feature: str, group: str, max_groups: int = 6
) -> go.Figure:
    """Distribución de ``feature`` condicionada a los niveles de ``group``.

    Muestra si la distribución de la variable difiere entre grupos del target
    (clave del análisis bivariado). Usa violines superpuestos por grupo.
    """
    levels = frame[group].value_counts().head(max_groups).index.tolist()
    fig = go.Figure()
    for i, level in enumerate(levels):
        subset = frame.loc[frame[group] == level, feature].dropna().to_numpy(dtype=float)
        fig.add_violin(
            y=subset,
            name=str(level),
            box_visible=True,
            meanline_visible=True,
            line_color=COLORS["primary"],
            opacity=0.65,
            marker_color=None if i else COLORS["accent"],
        )
    fig.update_yaxes(title=feature)
    fig.update_xaxes(title=group)
    return apply_theme(fig, title=f"«{feature}» por «{group}»")


def stacked_proportions(
    frame: pd.DataFrame, feature: str, target: str, normalize: bool = True
) -> go.Figure:
    """Barras 100% apiladas de una categórica frente al target categórico."""
    ct = pd.crosstab(frame[feature], frame[target], normalize="index" if normalize else False)
    fig = go.Figure()
    for i, col in enumerate(ct.columns):
        fig.add_bar(
            x=ct.index.astype(str),
            y=ct[col].to_numpy(),
            name=str(col),
            marker_color=None if i else COLORS["primary"],
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Proporción" if normalize else "Frecuencia")
    fig.update_xaxes(title=feature)
    return apply_theme(fig, title=f"Composición de «{target}» por «{feature}»")


def target_bins(target: pd.Series, n_bins: int = 3) -> pd.Series:
    """Discretiza un target continuo en grupos por cuantiles (Bajo/Medio/Alto…).

    Permite «abrir» cualquier feature por nivel del objetivo aunque el target sea
    numérico continuo.
    """
    labels = {
        2: ["Bajo", "Alto"],
        3: ["Bajo", "Medio", "Alto"],
        4: ["Q1", "Q2", "Q3", "Q4"],
        5: ["Q1", "Q2", "Q3", "Q4", "Q5"],
    }.get(n_bins)
    try:
        return pd.qcut(target, q=n_bins, labels=labels, duplicates="drop")
    except (ValueError, IndexError):  # pragma: no cover - pocos valores distintos
        return pd.cut(target, bins=min(n_bins, target.nunique()), labels=None)


def box_by_group(frame: pd.DataFrame, feature: str, group: str, max_groups: int = 8) -> go.Figure:
    """Boxplot de una numérica por nivel de ``group`` (comparación de medianas/IQR)."""
    levels = frame[group].value_counts().head(max_groups).index.tolist()
    fig = go.Figure()
    for i, level in enumerate(levels):
        subset = frame.loc[frame[group] == level, feature].dropna().to_numpy(dtype=float)
        fig.add_box(
            y=subset, name=str(level), boxmean=True, marker_color=QUALITATIVE[i % len(QUALITATIVE)]
        )
    fig.update_yaxes(title=feature)
    fig.update_xaxes(title=group)
    return apply_theme(fig, title=f"«{feature}» por «{group}»")


def overlaid_histograms(
    frame: pd.DataFrame, feature: str, group: str, max_groups: int = 5
) -> go.Figure:
    """Histogramas de densidad superpuestos de una numérica por nivel de ``group``."""
    levels = frame[group].value_counts().head(max_groups).index.tolist()
    fig = go.Figure()
    for i, level in enumerate(levels):
        subset = frame.loc[frame[group] == level, feature].dropna().to_numpy(dtype=float)
        fig.add_histogram(
            x=subset,
            histnorm="probability density",
            opacity=0.55,
            name=str(level),
            marker_color=QUALITATIVE[i % len(QUALITATIVE)],
            nbinsx=30,
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=feature)
    fig.update_yaxes(title="densidad")
    return apply_theme(fig, title=f"Distribución de «{feature}» por «{group}»")


def scatter_trend(
    frame: pd.DataFrame, feature: str, target: str, max_points: int = 3000
) -> go.Figure:
    """Dispersión feature vs. target (continuo) con recta de tendencia (OLS)."""
    df = frame[[feature, target]].dropna()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    x = df[feature].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_scatter(
        x=x,
        y=y,
        mode="markers",
        marker={"color": COLORS["secondary"], "opacity": 0.4, "size": 5},
        name="Observaciones",
    )
    if x.size >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        fig.add_scatter(
            x=xs,
            y=slope * xs + intercept,
            mode="lines",
            line={"color": COLORS["accent"], "width": 2},
            name="Tendencia (OLS)",
        )
    fig.update_xaxes(title=feature)
    fig.update_yaxes(title=target)
    return apply_theme(fig, title=f"«{feature}» vs. «{target}»")


def missingness_bar(frame: pd.DataFrame, top: int = 30) -> go.Figure:
    """Barras del porcentaje de valores nulos por columna."""
    pct = (frame.isna().mean() * 100.0).sort_values(ascending=False).head(top)
    pct = pct[pct > 0]
    fig = go.Figure(
        go.Bar(x=pct.to_numpy(), y=list(pct.index), orientation="h", marker_color=COLORS["warning"])
    )
    fig.update_xaxes(title="% nulos")
    fig.update_yaxes(autorange="reversed")
    return apply_theme(fig, title="Valores faltantes por columna")
