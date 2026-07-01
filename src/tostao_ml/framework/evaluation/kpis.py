"""Traducción de métricas técnicas a KPIs de negocio.

Contenedor genérico reutilizable por los tres casos; la lógica específica de cada
KPI (costo evitado, uplift de AOV) se calcula en el pipeline del caso y se
registra aquí con su narrativa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tostao_ml.framework.narrate import Insight, Narrative, Severity, rules


@dataclass(slots=True)
class BusinessKPIs:
    """Colección de KPIs de negocio con narrativa asociada.

    Attributes:
        values: Mapa nombre→valor del KPI.
        units: Unidad por KPI (para formateo en el reporte).
        narrative: Conclusiones de negocio autogeneradas.
    """

    values: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    narrative: Narrative = field(default_factory=Narrative)

    def add(
        self, name: str, value: float, unit: str = "", *, insight: Insight | None = None
    ) -> BusinessKPIs:
        """Registra un KPI y, opcionalmente, su insight."""
        self.values[name] = value
        self.units[name] = unit
        if insight is not None:
            self.narrative.add(insight)
        return self

    def add_vs_baseline(
        self,
        metric_name: str,
        model_value: float,
        baseline_value: float,
        *,
        higher_is_better: bool,
        unit: str = "",
    ) -> BusinessKPIs:
        """Registra una métrica frente a un baseline y narra la mejora relativa."""
        self.values[f"{metric_name}"] = model_value
        self.values[f"{metric_name}_baseline"] = baseline_value
        self.units[metric_name] = unit
        self.narrative.add(
            rules.metric_vs_baseline_insight(
                metric_name, model_value, baseline_value, higher_is_better
            )
        )
        return self


def improvement_pct(model_value: float, baseline_value: float) -> float:
    """Cambio relativo porcentual del modelo frente al baseline."""
    if baseline_value == 0:
        return float("nan")
    return (model_value - baseline_value) / abs(baseline_value) * 100.0


def summarize_impact(kpis: BusinessKPIs, headline: str) -> Insight:
    """Genera un insight de titular con el impacto agregado de negocio."""
    return Insight(
        text=headline,
        severity=Severity.GOOD,
        metrics=dict(kpis.values),
        tags=("business",),
        title="Impacto",
    )
