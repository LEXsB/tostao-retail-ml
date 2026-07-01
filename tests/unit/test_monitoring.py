"""Pruebas del monitoreo de drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.framework.monitoring import detect_drift, population_stability_index


@pytest.mark.unit
def test_psi_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    assert population_stability_index(x, x) < 0.01


@pytest.mark.unit
def test_psi_high_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(3, 1, 1000)  # media desplazada
    assert population_stability_index(ref, cur) > 0.25


@pytest.mark.unit
def test_detect_drift_flags_shifted_feature() -> None:
    rng = np.random.default_rng(1)
    reference = pd.DataFrame(
        {
            "estable": rng.normal(0, 1, 800),
            "desplazada": rng.normal(0, 1, 800),
            "cat": rng.choice(["a", "b", "c"], 800),
        }
    )
    current = pd.DataFrame(
        {
            "estable": rng.normal(0, 1, 800),
            "desplazada": rng.normal(4, 1, 800),  # drift fuerte
            "cat": rng.choice(["a", "b", "c"], 800),
        }
    )
    report = detect_drift(reference, current)
    drifted = set(report.table[report.table["drift"]]["feature"])
    assert "desplazada" in drifted
    assert "estable" not in drifted
    assert report.n_drifted >= 1
    assert len(report.narrative) >= 1


@pytest.mark.unit
def test_detect_drift_no_drift_when_identical() -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    report = detect_drift(df, df.copy())
    assert report.n_drifted == 0
    assert report.drift_share == 0.0
