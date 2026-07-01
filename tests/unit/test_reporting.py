"""Pruebas del ensamblador de reporte HTML."""

from __future__ import annotations

import pandas as pd
import pytest

from tostao_ml.framework.narrate import Insight, Narrative, Severity
from tostao_ml.framework.reporting import HTMLReport, ReportSection
from tostao_ml.framework.viz import eda


@pytest.fixture
def section() -> ReportSection:
    narrative = Narrative().add(Insight("Conclusión de prueba con cifra 0.42", Severity.GOOD))
    sec = ReportSection(
        id="eda", title="Análisis exploratorio", description="Sección de EDA.", narrative=narrative
    )
    sec.add_figure("dist", eda.histogram_kde(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")))
    sec.add_table("resumen", pd.DataFrame({"metric": [1.0, 2.0]}))
    return sec


@pytest.mark.unit
def test_report_render_is_self_contained(section: ReportSection) -> None:
    html = HTMLReport("Reporte Tostao", subtitle="Prueba").add_section(section).render()
    assert "<!DOCTYPE html>" in html
    assert "Reporte Tostao" in html
    assert "Análisis exploratorio" in html
    assert "Conclusión de prueba con cifra 0.42" in html
    assert "Plotly.newPlot" in html  # figura embebida
    # Autocontenido: el bundle completo de plotly.js va embebido (varios MB),
    # no una etiqueta <script src=...cdn...> externa.
    assert len(html) > 1_000_000
    assert "<script src=" not in html.split("</head>")[0]


@pytest.mark.unit
def test_report_save(section: ReportSection, tmp_path) -> None:
    path = HTMLReport("R").add_section(section).save(tmp_path / "reporte.html")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "eda__0" in content  # id de la figura de la sección
