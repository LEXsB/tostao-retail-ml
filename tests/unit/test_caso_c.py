"""Pruebas del pipeline del Caso C (drivers GLM + gasto predictivo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.cases.caso_c import inferential_drivers, predictive_spend, run_case_c


def _synthetic_master_c(n_customers: int = 60, visits: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_customers):
        edad = rng.integers(20, 70)
        seg = rng.choice(["Regular", "Premium", "VIP"])
        base_day = 0
        for _ in range(visits):
            base_day += rng.integers(3, 15)
            arts = rng.integers(1, 6)
            clima = rng.choice(["Sunny", "Rainy"])
            total = 8.0 * arts + (-1.5 if clima == "Rainy" else 0.0) + rng.normal(0, 1)
            rows.append(
                {
                    "id_cliente": f"C{c:03d}",
                    "fecha": pd.Timestamp("2024-01-01") + pd.Timedelta(days=base_day),
                    "total_venta": max(0.5, total),
                    "total_articulos": arts,
                    "edad": edad,
                    "segmento": seg,
                    "clima": clima,
                    "competitor_price_index": rng.normal(1, 0.1),
                    "indice_trafico": rng.integers(50, 100),
                    "n_promos_activas": rng.integers(0, 3),
                    "hora": rng.integers(7, 20),
                    "dia_semana": base_day % 7,
                    "antiguedad_cliente_dias": 200 + base_day,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_inferential_recovers_articulos_driver() -> None:
    coefs, inf_metrics, narrative = inferential_drivers(_synthetic_master_c())
    assert coefs.loc["total_articulos", "coef"] > 5  # driver dominante conocido
    assert coefs.loc["total_articulos", "pvalue"] < 0.01
    assert inf_metrics["n_drivers_significativos"] >= 1
    assert len(narrative) >= 1


@pytest.mark.unit
def test_predictive_beats_baseline() -> None:
    report, test, _model, _feat, _tuning, comparison, _narrative = predictive_spend(
        _synthetic_master_c()
    )
    assert {"mae", "rmse", "wape", "r2", "wape_baseline"}.issubset(report)
    assert len(comparison) >= 2 and "wape" in comparison.columns
    assert report["wape"] < report["wape_baseline"]  # mejora sobre la media
    assert "pred" in test.columns


@pytest.mark.unit
def test_run_case_c_end_to_end() -> None:
    result = run_case_c(_synthetic_master_c())
    assert not result.coefficients.empty
    assert result.predictive_metrics["r2"] > 0
    assert len(result.narrative) >= 2
