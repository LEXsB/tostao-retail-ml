"""Tema visual consistente para todas las figuras Plotly del proyecto.

Un único punto de verdad para paleta, tipografía y layout, de modo que EDA,
desempeño y negocio compartan estética. Las figuras se usan igual en notebooks
y en el reporte HTML autocontenido.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Paleta de marca (tonos café + acentos cálidos) --------------------------- #
COLORS = {
    "primary": "#6F4E37",  # café
    "secondary": "#C68B59",  # café con leche
    "accent": "#E4572E",  # naranja/rojo
    "positive": "#2E9E5B",  # verde (mejora / bien)
    "warning": "#E5A50A",  # ámbar
    "critical": "#C0392B",  # rojo
    "neutral": "#7F8C8D",  # gris
    "ink": "#2B2118",  # texto
    "paper": "#FFFFFF",
    "grid": "#ECE5DE",
}

# Secuencia cualitativa (categorías) y continua (mapas de calor).
QUALITATIVE = [
    "#6F4E37",
    "#E4572E",
    "#2E9E5B",
    "#3A6EA5",
    "#C68B59",
    "#8E44AD",
    "#E5A50A",
    "#16A085",
    "#D35400",
    "#7F8C8D",
]
SEQUENTIAL = "Oranges"
DIVERGING = "RdBu"

TEMPLATE_NAME = "tostao"


def _build_template() -> go.layout.Template:
    """Construye la plantilla Plotly de marca."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font={
            "family": "Inter, Segoe UI, Helvetica, Arial, sans-serif",
            "size": 13,
            "color": COLORS["ink"],
        },
        title={"font": {"size": 18, "color": COLORS["ink"]}, "x": 0.02, "xanchor": "left"},
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        colorway=QUALITATIVE,
        xaxis={
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
            "linecolor": COLORS["neutral"],
        },
        yaxis={
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
            "linecolor": COLORS["neutral"],
        },
        legend={"bgcolor": "rgba(0,0,0,0)", "borderwidth": 0},
        margin={"l": 60, "r": 30, "t": 60, "b": 55},
        colorscale={"sequential": SEQUENTIAL, "diverging": DIVERGING},
    )
    return template


def register_theme() -> None:
    """Registra la plantilla ``tostao`` en Plotly (idempotente)."""
    pio.templates[TEMPLATE_NAME] = _build_template()


def apply_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Aplica la plantilla de marca a una figura y, opcionalmente, su título.

    Args:
        fig: Figura Plotly a estilizar.
        title: Título a establecer (si se provee).

    Returns:
        La misma figura, estilizada (mutación in-place por eficiencia).
    """
    if TEMPLATE_NAME not in pio.templates:
        register_theme()
    fig.update_layout(template=TEMPLATE_NAME)
    if title is not None:
        fig.update_layout(title=title)
    return fig


# Registro al importar para que las figuras salgan con estilo por defecto.
register_theme()
