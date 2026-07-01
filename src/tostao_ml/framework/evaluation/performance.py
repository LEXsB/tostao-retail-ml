"""Fábrica de gráficos de desempeño (Plotly, §7).

Cubre regresión/forecasting (predicho vs real, residuales, bandas de intervalos)
y clasificación (ROC, PR, calibración, matriz de confusión, ganancia/lift). Todas
devuelven ``go.Figure`` estilizada, reutilizable en notebooks y reporte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from tostao_ml.framework.viz.theme import COLORS, apply_theme


# --------------------------------------------------------------------------- #
# Regresión / forecasting
# --------------------------------------------------------------------------- #
def pred_vs_actual(y_true: object, y_pred: object) -> go.Figure:
    """Dispersión predicho vs. real con la recta y=x de referencia."""
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    fig = go.Figure()
    fig.add_scatter(
        x=a,
        y=b,
        mode="markers",
        marker={"color": COLORS["secondary"], "opacity": 0.6},
        name="Predicciones",
    )
    fig.add_scatter(
        x=[lo, hi],
        y=[lo, hi],
        mode="lines",
        line={"color": COLORS["accent"], "dash": "dash"},
        name="y = x",
    )
    fig.update_xaxes(title="Real")
    fig.update_yaxes(title="Predicho")
    return apply_theme(fig, title="Predicho vs. real")


def residuals_vs_pred(y_true: object, y_pred: object) -> go.Figure:
    """Residuales frente a predicho (diagnóstico de heterocedasticidad/sesgo)."""
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    resid = a - b
    fig = go.Figure()
    fig.add_scatter(
        x=b,
        y=resid,
        mode="markers",
        marker={"color": COLORS["secondary"], "opacity": 0.6},
        name="Residual",
    )
    fig.add_hline(y=0, line={"color": COLORS["accent"], "dash": "dash"})
    fig.update_xaxes(title="Predicho")
    fig.update_yaxes(title="Residual (real − predicho)")
    return apply_theme(fig, title="Residuales vs. predicho")


def residual_hist(y_true: object, y_pred: object) -> go.Figure:
    """Histograma de residuales (chequeo de sesgo y forma del error)."""
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    fig = go.Figure(go.Histogram(x=a - b, marker_color=COLORS["primary"], nbinsx=40))
    fig.update_xaxes(title="Residual")
    fig.update_yaxes(title="Frecuencia")
    return apply_theme(fig, title="Distribución de residuales")


def forecast_with_intervals(
    index: object, y_true: object, median: object, lower: object, upper: object
) -> go.Figure:
    """Serie real vs. pronóstico con banda de intervalos de predicción."""
    x = list(np.asarray(index))
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    fig = go.Figure()
    fig.add_scatter(
        x=x + x[::-1],
        y=list(hi) + list(lo[::-1]),
        fill="toself",
        fillcolor="rgba(198,139,89,0.25)",
        line={"width": 0},
        name="Intervalo",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=x,
        y=np.asarray(median, dtype=float),
        mode="lines",
        line={"color": COLORS["accent"]},
        name="Pronóstico",
    )
    fig.add_scatter(
        x=x,
        y=np.asarray(y_true, dtype=float),
        mode="lines+markers",
        line={"color": COLORS["primary"]},
        name="Real",
    )
    fig.update_yaxes(title="Valor")
    return apply_theme(fig, title="Pronóstico con banda de incertidumbre")


# --------------------------------------------------------------------------- #
# Clasificación
# --------------------------------------------------------------------------- #
def roc(y_true: object, y_score: object) -> go.Figure:
    """Curva ROC con la diagonal de azar."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float))
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", line={"color": COLORS["accent"]}, name="ROC")
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line={"color": COLORS["neutral"], "dash": "dash"},
        name="Azar",
    )
    fig.update_xaxes(title="FPR")
    fig.update_yaxes(title="TPR")
    return apply_theme(fig, title="Curva ROC")


def precision_recall(y_true: object, y_score: object) -> go.Figure:
    """Curva Precision-Recall (clave con clases desbalanceadas)."""
    prec, rec, _ = precision_recall_curve(
        np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float)
    )
    fig = go.Figure(
        go.Scatter(x=rec, y=prec, mode="lines", line={"color": COLORS["primary"]}, name="PR")
    )
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return apply_theme(fig, title="Curva Precision-Recall")


def calibration(y_true: object, y_score: object, n_bins: int = 10) -> go.Figure:
    """Curva de calibración (reliability) frente a la diagonal perfecta."""
    frac_pos, mean_pred = calibration_curve(
        np.asarray(y_true).astype(int),
        np.asarray(y_score, dtype=float),
        n_bins=n_bins,
        strategy="quantile",
    )
    fig = go.Figure()
    fig.add_scatter(
        x=mean_pred,
        y=frac_pos,
        mode="lines+markers",
        line={"color": COLORS["accent"]},
        name="Modelo",
    )
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line={"color": COLORS["neutral"], "dash": "dash"},
        name="Perfecta",
    )
    fig.update_xaxes(title="Probabilidad predicha")
    fig.update_yaxes(title="Fracción positiva observada")
    return apply_theme(fig, title="Calibración (reliability)")


def confusion(y_true: object, y_pred: object, normalize: bool = True) -> go.Figure:
    """Matriz de confusión (normalizada por fila por defecto)."""
    cm = confusion_matrix(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        normalize="true" if normalize else None,
    )
    labels = [str(i) for i in range(cm.shape[0])]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Oranges",
            text=np.round(cm, 3),
            texttemplate="%{text}",
            colorbar={"title": "prop."},
        )
    )
    fig.update_xaxes(title="Predicho")
    fig.update_yaxes(title="Real", autorange="reversed")
    return apply_theme(fig, title="Matriz de confusión")


def cumulative_gain(y_true: object, y_score: object) -> go.Figure:
    """Curva de ganancia acumulada (cuántos positivos captura el top-k del score)."""
    a = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    order = np.argsort(-s)
    gains = np.cumsum(a[order]) / max(a.sum(), 1)
    pct = np.arange(1, a.size + 1) / a.size
    fig = go.Figure()
    fig.add_scatter(x=pct, y=gains, mode="lines", line={"color": COLORS["primary"]}, name="Modelo")
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line={"color": COLORS["neutral"], "dash": "dash"},
        name="Azar",
    )
    fig.update_xaxes(title="Proporción de población (ordenada por score)")
    fig.update_yaxes(title="Proporción de positivos captados")
    return apply_theme(fig, title="Ganancia acumulada")


# --------------------------------------------------------------------------- #
# Diagnóstico de entrenamiento
# --------------------------------------------------------------------------- #
def cv_boxplot(fold_scores: dict[str, list[float]]) -> go.Figure:
    """Boxplot de la métrica por fold y por modelo (estabilidad entre folds)."""
    fig = go.Figure()
    for i, (model, scores) in enumerate(fold_scores.items()):
        fig.add_box(
            y=scores, name=model, marker_color=None if i else COLORS["primary"], boxmean=True
        )
    fig.update_yaxes(title="Métrica")
    return apply_theme(fig, title="Estabilidad entre folds")


def model_comparison_bar(scores: dict[str, float], metric_name: str = "métrica") -> go.Figure:
    """Barras comparando una métrica entre modelos (baseline vs. candidatos)."""
    names = list(scores.keys())
    values = [scores[n] for n in names]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=COLORS["primary"],
            text=np.round(values, 4),
            textposition="outside",
        )
    )
    fig.update_yaxes(title=metric_name)
    return apply_theme(fig, title=f"Comparación de modelos ({metric_name})")


def error_by_segment(frame: pd.DataFrame, segment: str, error_col: str) -> go.Figure:
    """Error medio por segmento (SKU/tienda/tiempo) para detectar sesgos locales."""
    agg = (
        frame.groupby(segment, observed=True)[error_col]
        .mean()
        .sort_values(ascending=False)
        .head(20)
    )
    fig = go.Figure(
        go.Bar(x=agg.index.astype(str), y=agg.to_numpy(), marker_color=COLORS["secondary"])
    )
    fig.update_xaxes(title=segment)
    fig.update_yaxes(title=f"{error_col} medio")
    return apply_theme(fig, title=f"Error por «{segment}»")
