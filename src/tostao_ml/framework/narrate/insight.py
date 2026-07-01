"""Contenedores de conclusiones analíticas autogeneradas.

Cada nodo del pipeline emite ``Insight`` (mini-conclusión con cifras reales de
sus propias salidas). Una ``Narrative`` agrupa los insights de una sección y se
serializa a Markdown/HTML para alimentar el reporte final.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    """Nivel semántico de una conclusión, usado para estilo y priorización."""

    GOOD = "good"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def emoji(self) -> str:
        """Emoji representativo para render en texto/HTML."""
        return {
            Severity.GOOD: "✅",
            Severity.INFO: "ℹ️",
            Severity.WARNING: "⚠️",
            Severity.CRITICAL: "🔴",
        }[self]

    @property
    def rank(self) -> int:
        """Orden de severidad (mayor = más urgente)."""
        return {
            Severity.GOOD: 0,
            Severity.INFO: 1,
            Severity.WARNING: 2,
            Severity.CRITICAL: 3,
        }[self]


@dataclass(frozen=True, slots=True)
class Insight:
    """Una conclusión analítica derivada de cifras reales de un paso.

    Attributes:
        text: Texto narrativo con las cifras del run (no genérico).
        severity: Nivel semántico de la conclusión.
        metrics: Cifras que respaldan la conclusión (para trazabilidad/reporte).
        tags: Etiquetas de agrupación (p. ej. ``{"eda", "vif"}``).
        title: Título opcional de la conclusión.
    """

    text: str
    severity: Severity = Severity.INFO
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    title: str | None = None

    def to_markdown(self) -> str:
        """Renderiza la conclusión como una línea Markdown."""
        prefix = f"**{self.title}** — " if self.title else ""
        return f"{self.severity.emoji} {prefix}{self.text}"


class Narrative:
    """Colección ordenada de :class:`Insight` de una sección del análisis."""

    def __init__(self, insights: Iterable[Insight] | None = None) -> None:
        self._insights: list[Insight] = list(insights) if insights else []

    def add(self, insight: Insight | None) -> Narrative:
        """Añade un insight (ignora ``None`` para permitir reglas condicionales)."""
        if insight is not None:
            self._insights.append(insight)
        return self

    def extend(self, insights: Iterable[Insight]) -> Narrative:
        """Añade varios insights de una vez."""
        self._insights.extend(i for i in insights if i is not None)
        return self

    def sorted_by_severity(self) -> list[Insight]:
        """Devuelve los insights de más a menos urgente (estable)."""
        return sorted(self._insights, key=lambda i: -i.severity.rank)

    @property
    def worst_severity(self) -> Severity:
        """Severidad más alta presente (``GOOD`` si está vacía)."""
        if not self._insights:
            return Severity.GOOD
        return max((i.severity for i in self._insights), key=lambda s: s.rank)

    def to_markdown(self) -> str:
        """Renderiza toda la narrativa como lista Markdown."""
        return "\n".join(f"- {i.to_markdown()}" for i in self._insights)

    def to_dicts(self) -> list[dict[str, object]]:
        """Serializa a estructuras planas (para plantillas Jinja2/JSON)."""
        return [
            {
                "text": i.text,
                "severity": i.severity.value,
                "emoji": i.severity.emoji,
                "title": i.title,
                "metrics": dict(i.metrics),
                "tags": list(i.tags),
            }
            for i in self._insights
        ]

    def __len__(self) -> int:
        return len(self._insights)

    def __iter__(self) -> Iterator[Insight]:
        return iter(self._insights)

    def __repr__(self) -> str:
        return f"Narrative(n={len(self._insights)}, worst={self.worst_severity.value})"
