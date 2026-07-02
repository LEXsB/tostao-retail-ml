"""Calificación periódica: simula un periodo futuro y evalúa el modelo.

Reproduce lo que ejecuta el workflow programado ``calificacion-mensual``: construye
la tabla maestra desde las fuentes crudas, simula la demanda del periodo siguiente y
califica el pronóstico. Sirve de prueba de regresión: si el pipeline o el modelo se
degradan, esta prueba falla.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from tostao_ml.cases import masters
from tostao_ml.cases.simulacion import calificar_demanda, simular_demanda


def _weekly_master() -> pd.DataFrame:
    bootstrap_project(Path.cwd())
    with KedroSession.create(project_path=Path.cwd()) as session:
        catalog = session.load_context().catalog
        master, _ = masters.build_master_a(
            catalog.load("a_ventas_historicas"),
            catalog.load("a_catalogo_productos"),
            catalog.load("a_maestro_tiendas"),
            catalog.load("a_inventario_actual"),
            catalog.load("a_ground_truth_trends"),
        )
    return masters.aggregate_weekly_a(master)


@pytest.mark.calificacion
def test_simulacion_genera_periodo_futuro() -> None:
    weekly = _weekly_master()
    futuro = simular_demanda(weekly, semanas=4)
    assert not futuro.empty
    # El periodo simulado es posterior al histórico.
    assert futuro["semana"].min() > weekly["semana"].max()
    # Conserva el esquema y la no-negatividad de la demanda.
    assert set(weekly.columns).issubset(futuro.columns)
    assert (futuro["unidades_vendidas"] >= 0).all()


@pytest.mark.calificacion
def test_calificacion_del_periodo_futuro() -> None:
    weekly = _weekly_master()
    futuro = simular_demanda(weekly, semanas=4)
    reporte = calificar_demanda(weekly, futuro)
    assert reporte["n"] > 0
    # El modelo debe puntuar el periodo simulado con un error razonable (salud del pipeline).
    assert 0 <= reporte["wape"] < 0.5
