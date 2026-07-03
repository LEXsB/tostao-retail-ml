"""Ensamblador del reporte HTML autocontenido (§8).

Recolecta secciones (narrativa + figuras Plotly + tablas) y las renderiza con
Jinja2 en un único HTML portable, con ``plotly.js`` embebido una sola vez. Los
mismos ``Figure`` se usan en notebooks y aquí.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from tostao_ml.framework.narrate import Narrative
from tostao_ml.framework.viz.serialize import figure_to_div, figure_to_png_img, plotly_js_bundle

_TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(slots=True)
class ReportSection:
    """Una sección del reporte: narrativa, figuras y tablas.

    Attributes:
        id: Ancla HTML (para el índice).
        title: Título visible de la sección.
        description: Texto introductorio opcional.
        narrative: Conclusiones autogeneradas de la sección.
        figures: Figuras Plotly (clave→figura).
        tables: Tablas (nombre→DataFrame).
    """

    id: str
    title: str
    description: str = ""
    narrative: Narrative | None = None
    figures: dict[str, go.Figure] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def add_figure(self, name: str, figure: go.Figure) -> ReportSection:
        """Añade una figura a la sección."""
        self.figures[name] = figure
        return self

    def add_table(self, name: str, table: pd.DataFrame) -> ReportSection:
        """Añade una tabla a la sección."""
        self.tables[name] = table
        return self


class HTMLReport:
    """Constructor del reporte HTML unificado."""

    def __init__(self, title: str, subtitle: str = "", footer: str = "", context: str = "") -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        self.title = title
        self.subtitle = subtitle
        self.context = context  # bloque HTML con la tarea/contexto de negocio (encabezado)
        self.footer = footer or "Reporte generado automáticamente por el pipeline de reporting."
        self.sections: list[ReportSection] = []
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def add_section(self, section: ReportSection) -> HTMLReport:
        """Añade una sección al reporte (en orden de aparición)."""
        self.sections.append(section)
        return self

    def render(self, *, static: bool = False) -> str:
        """Renderiza el reporte a una cadena HTML autocontenida.

        Args:
            static: Si ``True``, las figuras se embeben como PNG (sin ``plotly.js``),
                produciendo un HTML liviano e imprimible (usado por el reporte
                ejecutivo). Si ``False``, figuras interactivas con el bundle embebido.
        """
        template = self._env.get_template("report.html.j2")
        return template.render(
            title=self.title,
            subtitle=self.subtitle,
            context=self.context,
            footer=self.footer,
            plotly_js="" if static else plotly_js_bundle(),
            sections=[self._render_section(s, static=static) for s in self.sections],
        )

    def save(self, path: str | Path, *, static: bool = False) -> Path:
        """Renderiza y guarda el reporte en disco (UTF-8)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(static=static), encoding="utf-8")
        return path

    @staticmethod
    def _render_section(section: ReportSection, *, static: bool = False) -> dict[str, object]:
        """Prepara el contexto Jinja2 de una sección (figuras→div/img, tablas→html)."""
        insights = section.narrative.to_dicts() if section.narrative else []
        worst = (
            section.narrative.worst_severity.value
            if section.narrative and len(section.narrative)
            else ""
        )
        figures = {}
        for i, (name, fig) in enumerate(section.figures.items()):
            figures[name] = (
                figure_to_png_img(fig)
                if static
                else figure_to_div(fig, div_id=f"{section.id}__{i}")
            )
        return {
            "id": section.id,
            "title": section.title,
            "description": section.description,
            "insights": insights,
            "worst_severity": worst,
            "figures": figures,
            "tables": {
                name: df.to_html(classes="table", border=0, float_format=lambda v: f"{v:,.3f}")
                for name, df in section.tables.items()
            },
        }
