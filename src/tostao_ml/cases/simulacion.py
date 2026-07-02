"""Simulación de periodos futuros y calificación del modelo de abastecimiento.

Sin datos reales de meses venideros, se simula la demanda del periodo siguiente a
partir de la distribución histórica de cada serie SKU-tienda y se **califica** el
pronóstico ya entrenado sobre ese periodo. Sirve como prueba de regresión
periódica (ver el workflow ``calificacion-mensual``) y como mecanismo para puntuar
meses futuros reutilizando el modelo guardado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tostao_ml.framework.evaluation import metrics
from tostao_ml.framework.models.base import BaseModel

from .caso_a import FEATURES, TARGET, build_features_a, run_case_a

_GRUPO = ["id_tienda", "id_producto"]
_ESTATICOS = [
    "nombre",
    "categoria",
    "costo_unitario",
    "precio_venta",
    "costo_almacenamiento_semanal",
    "ciudad",
    "tamaño_m2",
    "stock_actual",
    "trend_type",
]


def simular_demanda(weekly: pd.DataFrame, semanas: int = 4, seed: int = 7) -> pd.DataFrame:
    """Simula la demanda de las próximas ``semanas`` por SKU-tienda.

    Para cada serie se muestrea la demanda de una distribución de Poisson ajustada a
    su media histórica (conteo no negativo), conservando los atributos estáticos del
    producto y la tienda. Devuelve un DataFrame con el mismo esquema semanal.

    Args:
        weekly: Tabla maestra semanal histórica (grano SKU-tienda-semana).
        semanas: Número de semanas futuras a simular.
        seed: Semilla del generador aleatorio.

    Returns:
        Filas simuladas del periodo siguiente, listas para calificar.
    """
    rng = np.random.default_rng(seed)
    estaticos = [c for c in _ESTATICOS if c in weekly.columns]
    semana_max = int(weekly["semana"].max())
    filas = []
    for (tienda, producto), grupo in weekly.groupby(_GRUPO, observed=True):
        media = max(float(grupo[TARGET].mean()), 0.1)
        ultimo = grupo.sort_values("semana").iloc[-1]
        for i in range(1, semanas + 1):
            fila = {c: ultimo[c] for c in estaticos}
            fila.update(
                id_tienda=tienda,
                id_producto=producto,
                anio=int(ultimo["anio"]),
                semana=semana_max + i,
                unidades_vendidas=int(rng.poisson(media)),
                ingreso=0.0,
                dias_con_venta=int(rng.integers(1, 8)),
            )
            filas.append(fila)
    return pd.DataFrame(filas)


def calificar_demanda(
    weekly_hist: pd.DataFrame, weekly_fut: pd.DataFrame, modelo: BaseModel | None = None
) -> dict[str, float]:
    """Califica el pronóstico de demanda sobre un periodo (real o simulado).

    Concatena el histórico con el periodo a calificar, reconstruye las features
    (rezagos, estacionalidad) y predice las semanas futuras con el modelo entrenado;
    luego compara contra la demanda observada del periodo.

    Args:
        weekly_hist: Historia semanal usada para entrenar y para los rezagos.
        weekly_fut: Periodo a calificar (con la demanda observada/simulada).
        modelo: Modelo ya entrenado; si es ``None`` se entrena con ``weekly_hist``.

    Returns:
        Métricas de error (MAE, RMSE, WAPE, sMAPE, R²) sobre el periodo.
    """
    if modelo is None:
        modelo = run_case_a(weekly_hist).model  # type: ignore[assignment]

    semana_corte = int(weekly_hist["semana"].max())
    completo = pd.concat([weekly_hist, weekly_fut], ignore_index=True)
    features = build_features_a(completo)
    futuro = features[features["semana"] > semana_corte]
    if futuro.empty:
        return {"n": 0.0}
    pred = modelo.predict(futuro[FEATURES].fillna(0.0))  # type: ignore[union-attr]
    reporte = metrics.regression_report(futuro[TARGET].to_numpy(dtype=float), pred)
    reporte["n"] = float(len(futuro))
    return reporte
