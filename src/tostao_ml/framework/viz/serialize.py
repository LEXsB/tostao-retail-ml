"""Serialización de figuras Plotly para el reporte HTML autocontenido.

El reporte embebe ``plotly.js`` una sola vez y cada figura como un ``<div>``
(sin dependencias online), de modo que el HTML final es portable.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio


def figure_to_div(fig: go.Figure, div_id: str | None = None) -> str:
    """Convierte una figura en un ``<div>`` HTML sin incluir la librería JS.

    Args:
        fig: Figura a serializar.
        div_id: Identificador HTML del div (para anclas/estilos).

    Returns:
        Fragmento HTML del gráfico (asume ``plotly.js`` ya cargado en la página).
    """
    return str(
        pio.to_html(
            fig,
            include_plotlyjs=False,
            full_html=False,
            div_id=div_id,
            config={"displaylogo": False, "responsive": True},
        )
    )


def plotly_js_bundle() -> str:
    """Devuelve el bundle de ``plotly.js`` embebible una sola vez en el reporte."""
    from plotly.offline import get_plotlyjs

    return f'<script type="text/javascript">{get_plotlyjs()}</script>'


def save_figure(fig: go.Figure, path: str | Path, *, self_contained: bool = True) -> Path:
    """Guarda una figura como HTML interactivo en disco.

    Args:
        fig: Figura a persistir.
        path: Ruta destino (``.html``).
        self_contained: Si ``True``, embebe ``plotly.js`` (portable, sin internet).

    Returns:
        La ruta escrita.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        fig,
        file=str(path),
        include_plotlyjs=True if self_contained else "cdn",
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    return path
