"""Resultado tipado de una validación de datos."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tostao_ml.framework.narrate import Insight, Narrative, Severity


@dataclass(slots=True)
class ValidationResult:
    """Salida estructurada de validar un dataset contra un contrato.

    Attributes:
        name: Nombre del dataset/contrato validado.
        passed: ``True`` si no hubo violaciones.
        n_checks: Número de comprobaciones aplicadas.
        n_failures: Número de casos que fallaron.
        failure_cases: Detalle de las violaciones (columna, check, valor).
        narrative: Mini-conclusión autogenerada de la validación.
    """

    name: str
    passed: bool
    n_checks: int = 0
    n_failures: int = 0
    failure_cases: pd.DataFrame = field(default_factory=pd.DataFrame)
    narrative: Narrative = field(default_factory=Narrative)

    def raise_for_status(self) -> None:
        """Lanza ``ValueError`` si la validación falló (fallar rápido)."""
        if not self.passed:
            raise ValueError(
                f"Validación '{self.name}' falló con {self.n_failures} violación(es). "
                f"Primeros casos:\n{self.failure_cases.head(10)}"
            )

    def build_narrative(self) -> Narrative:
        """Redacta la conclusión de la validación a partir de sus cifras."""
        if self.passed:
            self.narrative.add(
                Insight(
                    text=f"Contrato «{self.name}»: {self.n_checks} comprobaciones sin violaciones.",
                    severity=Severity.GOOD,
                    metrics={"n_checks": self.n_checks},
                    tags=("validation",),
                )
            )
        else:
            self.narrative.add(
                Insight(
                    text=(
                        f"Contrato «{self.name}»: {self.n_failures} violación(es) en "
                        f"{self.n_checks} comprobaciones — el pipeline debe fallar rápido."
                    ),
                    severity=Severity.CRITICAL,
                    metrics={"n_failures": self.n_failures, "n_checks": self.n_checks},
                    tags=("validation",),
                )
            )
        return self.narrative
