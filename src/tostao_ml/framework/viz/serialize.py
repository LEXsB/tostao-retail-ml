"""Serialización de figuras Plotly para el reporte HTML autocontenido.

El reporte embebe ``plotly.js`` una sola vez y cada figura como un ``<div>``
(sin dependencias online), de modo que el HTML final es portable.
"""

from __future__ import annotations

import re
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# El bundle de plotly.js trae algún carácter pictográfico en su tabla de símbolos
# (p. ej. la "x" -> U+274C). Se reemplaza por ASCII para que el HTML no contenga
# ningún icono/emoji, sin afectar el render de las figuras usadas.
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U00002b00-\U00002bff\U00002139\U0000fe0f]"
)


def _sin_iconos(js: str) -> str:
    """Reemplaza el símbolo pictográfico de plotly por ASCII (deja el HTML sin iconos)."""
    return _EMOJI.sub("x", js)


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

    return f'<script type="text/javascript">{_sin_iconos(get_plotlyjs())}</script>'


def figure_to_png_img(fig: go.Figure, *, width: int = 760, height: int = 430) -> str:
    """Convierte una figura en una imagen PNG estática embebida (data URI base64).

    Pensado para reportes livianos (ejecutivos): la página no necesita ``plotly.js``,
    por lo que el HTML pesa pocos KB y es imprimible. Usa la API directa de kaleido,
    compatible con la versión de Plotly instalada.

    Args:
        fig: Figura a rasterizar.
        width: Ancho del PNG en píxeles.
        height: Alto del PNG en píxeles.

    Returns:
        Etiqueta ``<img>`` con el PNG embebido como ``data:`` URI.
    """
    import base64

    import kaleido

    sized = go.Figure(fig)
    sized.update_layout(width=width, height=height)
    png = kaleido.calc_fig_sync(sized)
    b64 = base64.b64encode(png).decode("ascii")
    return f'<img alt="figura" style="width:100%;height:auto" src="data:image/png;base64,{b64}">'


def stop_image_backend() -> None:
    """Detiene el servidor de kaleido (libera el proceso de Chrome tras exportar)."""
    try:
        import kaleido

        kaleido.stop_sync_server()
    except Exception:  # pragma: no cover - limpieza best-effort
        pass


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
