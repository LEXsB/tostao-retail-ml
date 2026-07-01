"""Detección de drift de datos (PSI + KS) para el ciclo AIOps.

Implementación ligera y testeable del monitoreo de drift entre un conjunto de
referencia (entrenamiento) y uno actual (producción). En despliegue se complementa
con Evidently para dashboards; aquí vive la lógica reutilizable y sin dependencias
pesadas que alimenta la narrativa y las alertas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from tostao_ml.framework.narrate import Insight, Narrative, Severity
from tostao_ml.framework.profiling.typing import infer_variable_types


@dataclass(slots=True)
class DriftReport:
    """Resultado del análisis de drift por feature."""

    table: pd.DataFrame
    n_drifted: int
    narrative: Narrative = field(default_factory=Narrative)

    @property
    def drift_share(self) -> float:
        """Proporción de features con drift detectado."""
        return float(self.n_drifted / len(self.table)) if len(self.table) else 0.0


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI entre dos muestras numéricas (buckets por cuantiles de la referencia).

    Reglas habituales: PSI < 0.1 estable, 0.1–0.25 cambio moderado, > 0.25 drift.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size < bins or cur.size == 0:
        return 0.0
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if quantiles.size < 3:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_perc = np.histogram(ref, bins=quantiles)[0] / ref.size
    cur_perc = np.histogram(cur, bins=quantiles)[0] / cur.size
    eps = 1e-6
    ref_perc = np.clip(ref_perc, eps, None)
    cur_perc = np.clip(cur_perc, eps, None)
    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    psi_threshold: float = 0.25,
    pvalue_alpha: float = 0.05,
) -> DriftReport:
    """Detecta drift por feature entre referencia y actual.

    Numéricas: PSI + KS de dos muestras. Categóricas: chi-cuadrado de frecuencias.

    Args:
        reference: Datos de referencia (p. ej. entrenamiento).
        current: Datos actuales (p. ej. ventana de producción).
        psi_threshold: PSI a partir del cual se marca drift.
        pvalue_alpha: Nivel de significancia de los tests.

    Returns:
        :class:`DriftReport` con la tabla por feature y la narrativa.
    """
    common = [c for c in reference.columns if c in current.columns]
    types = infer_variable_types(reference[common])
    rows: list[dict[str, object]] = []
    for col in common:
        kind = types[col]
        if kind.is_numeric:
            psi = population_stability_index(
                reference[col].dropna().to_numpy(dtype=float),
                current[col].dropna().to_numpy(dtype=float),
            )
            ref_v = reference[col].dropna().to_numpy(dtype=float)
            cur_v = current[col].dropna().to_numpy(dtype=float)
            pvalue = (
                float(stats.ks_2samp(ref_v, cur_v).pvalue) if ref_v.size and cur_v.size else np.nan
            )
            drifted = psi >= psi_threshold or (pd.notna(pvalue) and pvalue < pvalue_alpha)
            rows.append(
                {
                    "feature": col,
                    "tipo": kind.value,
                    "psi": round(psi, 4),
                    "ks_pvalue": round(pvalue, 5) if pd.notna(pvalue) else np.nan,
                    "drift": bool(drifted),
                }
            )
        elif kind.is_categorical:
            pvalue = _categorical_pvalue(reference[col], current[col])
            drifted = pd.notna(pvalue) and pvalue < pvalue_alpha
            rows.append(
                {
                    "feature": col,
                    "tipo": kind.value,
                    "psi": np.nan,
                    "ks_pvalue": round(pvalue, 5) if pd.notna(pvalue) else np.nan,
                    "drift": bool(drifted),
                }
            )

    table = pd.DataFrame(rows)
    n_drifted = int(table["drift"].sum()) if not table.empty else 0
    report = DriftReport(table=table, n_drifted=n_drifted)
    _narrate_drift(report, psi_threshold)
    return report


def _categorical_pvalue(reference: pd.Series, current: pd.Series) -> float:
    """Chi-cuadrado entre las distribuciones de frecuencia de una categórica."""
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    categories = ref_counts.index.union(cur_counts.index)
    ref_freq = ref_counts.reindex(categories, fill_value=0).to_numpy(dtype=float) + 1e-6
    cur_freq = cur_counts.reindex(categories, fill_value=0).to_numpy(dtype=float) + 1e-6
    if categories.size < 2:
        return float("nan")
    # Escala esperada a la magnitud observada.
    expected = ref_freq / ref_freq.sum() * cur_freq.sum()
    return float(stats.chisquare(cur_freq, expected).pvalue)


def _narrate_drift(report: DriftReport, psi_threshold: float) -> None:
    """Redacta la conclusión del monitoreo de drift."""
    if report.table.empty:
        return
    share = report.drift_share
    if report.n_drifted == 0:
        report.narrative.add(
            Insight(
                text="Sin drift detectado: las distribuciones actuales son consistentes con la referencia.",
                severity=Severity.GOOD,
                metrics={"drift_share": 0.0},
                tags=("aiops", "drift"),
                title="Monitoreo",
            )
        )
        return
    severity = Severity.CRITICAL if share >= 0.3 else Severity.WARNING
    drifted = report.table[report.table["drift"]]["feature"].tolist()
    report.narrative.add(
        Insight(
            text=(
                f"Drift en {report.n_drifted} de {len(report.table)} features ({share:.0%}): "
                f"{', '.join(drifted[:5])}. Sugiere revisar/reentrenar."
            ),
            severity=severity,
            metrics={"n_drifted": report.n_drifted, "drift_share": round(share, 3)},
            tags=("aiops", "drift"),
            title="Monitoreo",
        )
    )
