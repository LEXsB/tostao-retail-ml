"""Pruebas de la narración e interpretación (storytelling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tostao_ml.cases import storytelling as st
from tostao_ml.framework.profiling import profile_dataset


@pytest.mark.unit
def test_context_html_has_task_and_objective() -> None:
    for case in ("a", "b", "c"):
        html = st.context_html(case)
        assert "Tarea propuesta" in html
        assert "Objetivo" in html and "Contexto de negocio" in html


@pytest.mark.unit
def test_data_story_a_reports_real_facts() -> None:
    weekly = pd.DataFrame(
        {
            "id_tienda": ["S1", "S1", "S2", "S2"],
            "id_producto": ["P1", "P1", "P1", "P1"],
            "semana": [1, 2, 1, 2],
            "unidades_vendidas": [10, 0, 5, 8],
            "categoria": ["Bebidas"] * 4,
        }
    )
    narr = st.data_story_a(weekly)
    text = narr.to_markdown()
    assert "series SKU-tienda" in text
    assert "%" in text  # incluye intermitencia


@pytest.mark.unit
def test_interpret_eda_verdicts() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=300),
            "x2": rng.normal(size=300),
            "target": rng.normal(size=300),
        }
    )
    df["x1"] = df["target"] * 2 + rng.normal(0, 0.1, 300)  # x1 informativo
    profile = profile_dataset(df, name="t", target="target")
    narr = st.interpret_eda(profile)
    text = narr.to_markdown()
    assert "VIF" in text  # veredicto de multicolinealidad
    assert len(narr) >= 1
