# Calificación de periodos futuros

Sin datos reales de meses venideros, la plataforma incluye un mecanismo para
**simular el periodo siguiente y calificar el modelo** sobre él. Sirve para dos
cosas: (1) puntuar meses futuros reutilizando el modelo entrenado y (2) actuar como
prueba de regresión periódica (si el pipeline o el modelo se degradan, la prueba
falla).

## Cómo funciona

1. **Artefacto entrenado.** Al ejecutar `kedro run --pipeline caso_a` (o `caso_c`),
   el modelo entrenado se persiste en `data/06_models/` (`modelo_caso_a.pkl`,
   `modelo_caso_c.pkl`). Así se puede calificar un mes nuevo sin reentrenar.
2. **Simulación del periodo.** `cases/simulacion.py::simular_demanda` genera la
   demanda de las próximas semanas por SKU-tienda, muestreando de la distribución
   histórica de cada serie (Poisson ajustado a su media) y conservando el esquema.
3. **Calificación.** `calificar_demanda` reconstruye las features (rezagos,
   estacionalidad) sobre histórico + periodo nuevo, predice con el modelo y compara
   contra la demanda observada del periodo (MAE, RMSE, WAPE, R²).

```python
import pandas as pd
from tostao_ml.cases.simulacion import simular_demanda, calificar_demanda

weekly = pd.read_parquet("data/03_primary/master_caso_a_semanal.parquet")
futuro = simular_demanda(weekly, semanas=4)
print(calificar_demanda(weekly, futuro))   # {'wape': ~0.11, 'r2': ~0.89, ...}
```

## Automatización mensual (GitHub Actions)

El workflow `.github/workflows/calificacion-mensual.yml` corre el **primer día de
cada mes** (y bajo demanda) y ejecuta la calificación como prueba:

```bash
uv run pytest -m calificacion -v
```

Si la simulación deja de generarse bien o el modelo puntúa por encima del umbral de
error, el job falla y avisa. Con datos reales del nuevo mes basta con reemplazar el
periodo simulado por el observado y llamar a `calificar_demanda` con el modelo
guardado.
