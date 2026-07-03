# Casos de uso

## Caso A — Abastecimiento

Forecast probabilístico de demanda semanal por SKU-tienda y decisión de pedido
por *critical fractile* (newsvendor). Balancea el margen perdido por quiebre
contra el costo de sobre-stock, descontando el inventario actual.

- **Resultado:** WAPE ≈ 13 %, R² ≈ 0.88; ≈ 70 % menos costo esperado de
  faltante + sobrante vs. política ingenua.

## Caso B — Combos

Segmentación de tiendas por perfil de compra (K-Means) y reglas de asociación
(FP-Growth). Propone los mejores combos por cluster con precio y lift esperado.

- **Resultado:** 3 clusters (silhouette ≈ 0.32); combos con lift 5–9.

## Caso C — AOV

Modelo inferencial (GLM + IC) de los drivers del ticket y modelo predictivo del
gasto esperado del cliente recurrente (features RFM + loyalty).

- **Resultado:** driver dominante `total_articulos` (β ≈ 8); modelo de gasto con
  R² ≈ 0.77 y −54 % de WAPE vs. baseline.

## Ejecución

```bash
uv run kedro run --pipeline caso_a   # o caso_b / caso_c
```

## Notebooks y reportes

Cada caso tiene, en paralelo al pipeline, un notebook de **EDA**
(`notebooks/caso_*/01_eda.ipynb`) y uno de **modelamiento**
(`notebooks/caso_*/02_modelamiento.ipynb`) que despliega el modelo por dentro
(features, HPO, ajuste, evaluación) tal como producción. El
detalle por caso está en [Caso A](caso_a.md), [Caso B](caso_b.md) y
[Caso C](caso_c.md). El reporte **ejecutivo** por caso se genera en
`reports/ejecutivo/` con `uv run python scripts/build_reports.py`.
