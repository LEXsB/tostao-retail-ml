# Model Card — Plataforma DS/ML retail (Tostao)

Documento las decisiones y límites de los modelos de los tres casos. Todas las
cifras provienen de ejecuciones reales sobre los datos provistos.

## Caso A — Forecast de demanda + optimización de pedido

- **Tarea:** forecast probabilístico (cuantiles 0.1/0.5/0.9) de demanda semanal
  por SKU-tienda; decisión de pedido por *critical fractile* (newsvendor).
- **Modelo:** gradient boosting por histogramas con pérdida pinball
  (`HistGradientBoostingRegressor`, un estimador por cuantil).
- **Features:** rezagos y media rodante *group-aware* (sin leakage), estacionalidad
  cíclica (semana), codificación de frecuencia de categoría/ciudad/tendencia y
  atributos de catálogo/tienda/inventario.
- **Validación:** partición temporal (holdout de las últimas semanas).
- **Desempeño (holdout):** WAPE ≈ 13 %, R² ≈ 0.88; supera al baseline ingenuo
  (demanda de la semana previa).
- **Impacto de negocio:** la política óptima reduce ≈ 70 % el costo esperado de
  faltante + sobrante frente a la política ingenua.
- **Limitaciones:** series cortas (~13 semanas/serie); los intervalos tienden a
  **sub-cubrir** (PICP < nominal) — conviene calibrarlos (conformal / cuantiles
  más anchos) antes de decisiones críticas.

## Caso B — Combos (clustering + reglas de asociación)

- **Tarea:** segmentar tiendas y proponer combos de co-compra por segmento.
- **Modelos:** K-Means sobre el perfil de compra de la tienda; reglas de
  asociación FP-Growth (support/confidence/lift/conviction).
- **Salida:** Top-N combos por cluster con precio propuesto (descuento
  configurable) y lift esperado.
- **Desempeño:** 3 clusters (silhouette ≈ 0.32); combos con lift 5–9.
- **Limitaciones:** el lift alto puede reflejar baja frecuencia; validar con test
  A/B antes de fijar precios; los IDs de tienda difieren de formato entre fuentes.

## Caso C — AOV (drivers + gasto esperado)

- **Tareas:** (1) inferencia de drivers del ticket con GLM (coeficientes + IC),
  outliers winsorizados; (2) predicción del gasto esperado del cliente recurrente
  (features RFM + loyalty).
- **Modelos:** GLM gaussiano (inferencial) + gradient boosting (predictivo).
- **Desempeño:** driver dominante `total_articulos` (β ≈ 8, p ≈ 0), `clima`
  significativo; modelo de gasto con R² ≈ 0.77 y −54 % de WAPE vs. baseline.
- **Limitaciones:** el GLM asume relaciones aproximadamente lineales; correlación
  entre drivers puede inflar varianza (ver VIF en el EDA).

## Consideraciones transversales

- **Reproducibilidad:** semillas fijas, entorno bloqueado (`uv.lock`), datos de
  entrada versionados.
- **Trazabilidad:** parámetros, métricas y artefactos se registran con MLflow
  (`kedro-mlflow`); linaje con Kedro-Viz.
- **Monitoreo:** detección de drift (PSI/KS) para disparar reentrenamiento.
- **Uso responsable:** los modelos apoyan decisiones operativas; no sustituyen el
  criterio de negocio ni deben usarse fuera del dominio de entrenamiento.
