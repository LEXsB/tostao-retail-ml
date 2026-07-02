# Changelog

Todas las modificaciones notables de este proyecto se documentan aquí.
El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y el versionado semántico ([SemVer](https://semver.org/lang/es/)).

## [Sin publicar]

### Cambiado
- **Detalle técnico del modelamiento en cada reporte**: sección corta "Estrategia
  de modelamiento y validación" por caso que explica el tipo de tarea, el modelo y
  por qué, la **partición train/test** (holdout temporal en A, 80/20 en C, no
  aplica en B), la **validación** (walk-forward, silhouette, K-Fold), la
  **optimización de hiperparámetros** con Optuna (mejor config + HP influyente),
  las **métricas y qué significan** (incl. nota sobre clasificación) y la
  **decisión que se toma y por qué**. Se elimina la versión simple de los EDA.
- **HPO real activado** en la generación de reportes: Caso A (boosting cuantílico,
  Optuna sobre walk-forward, WAPE↓ a ~12.7%), Caso C (boosting predictivo, Optuna
  K-Fold, R²↑ a ~0.81) y Caso B (selección de k por barrido de silhouette → k=3).
- **Reportes rehechos a fondo** (calidad EDA + presentación): tipado que ahora
  clasifica IDs de alta cardinalidad como identificadores (excluidos del describe);
  estadística univariada **separada** en numéricas (media/mediana/std/CV/skew/
  kurtosis/percentiles/outliers/normalidad) y categóricas (cardinalidad/moda/
  entropía); **apertura de cada feature por la variable objetivo** (§4.3) —el
  target continuo se discretiza en Bajo/Medio/Alto— con box/violín, composición
  100% apilada, dispersión+tendencia y tests con tamaño de efecto; correlaciones
  Pearson+Spearman, VIF y pares redundantes; **reporte por caso completo**
  (EDA+modelado+interpretabilidad SHAP/coeficientes+negocio) y **glosario** que
  explica cada métrica. Plantilla HTML con paleta profesional y **tablas con scroll
  (sin desbordes)**. Reportes reales: ~50 figuras Plotly por caso.

### Añadido
- Scaffolding inicial del proyecto Kedro `tostao-retail-ml`: layout `src/`,
  configuración de entorno reproducible con `uv`, herramientas de calidad
  (ruff, black, mypy, pytest) y estructura de carpetas de datos por capas.
- CI de calidad (GitHub Actions) y configuración de `kedro-mlflow`.
- **Fundaciones del framework** (`framework/`): tipos compartidos
  (`VariableKind`, `TaskType`), motor de narración (`narrate/`: `Insight`,
  `Narrative`, umbrales y reglas analíticas), fábrica de gráficos Plotly con
  tema de marca (`viz/`: figuras EDA + serialización HTML autocontenida),
  utilidades de reproducibilidad/hashing (`io/`) y validación de contratos con
  Pandera + integridad referencial (`validation/`). Con pruebas unitarias.
- **Motor de EDA** (`profiling/`, §4): tipado automático de variables, estadística
  univariada (outliers IQR/MAD, normalidad), multivariado (correlaciones
  Pearson/Spearman/Kendall, VIF, número de condición, información mutua, Cramér's
  V, Theil's U, razón de correlación η, correlaciones canónicas) y bivariado con
  el target (t/ANOVA, Mann-Whitney/Kruskal, chi², KS, tamaños de efecto). El
  orquestador `profile_dataset` emite un `DatasetProfile` con narrativa
  autogenerada; validado sobre datos reales de ventas.
- **Transformadores de features** (`features/`): base `DataFrame`-first estilo
  scikit-learn, calendario/cíclicos/rezagos temporales *group-aware* sin leakage,
  RFM por cliente, `FrequencyEncoder`, `Winsorizer`, selección por VIF/correlación
  y ensamblado automático de `ColumnTransformer` desde el tipado del EDA.
- **Modelos** (`models/`): `BaseModel` (ABC) con fit/predict/predict_interval/
  save/load, registry/factory por configuración e implementaciones (Ridge, GBR,
  forecast probabilístico cuantílico, K-Means, GLM inferencial con IC).
- **Motor de HPO** (`tuning/`, §6): Optuna con espacios de búsqueda declarativos,
  samplers (TPE/CMA-ES/NSGA-II), pruners, validación cruzada inyectable, estudios
  reproducibles, importancia de hiperparámetros y figuras de optimización.
- **Evaluación** (`evaluation/`, §7): métricas de regresión/forecast (MAE, RMSE,
  WAPE, sMAPE, pinball, PICP, MPIW, Theil's U) y clasificación (ROC-AUC, PR-AUC,
  KS, Brier); validadores temporales (walk-forward) y estándar; fábrica de gráficos
  de desempeño (predicho vs real, residuales, ROC/PR, calibración, matriz de
  confusión, ganancia, bandas de intervalos) y contenedor de KPIs de negocio.
- **Interpretabilidad** (`interpret/`): SHAP model-agnóstico, importancia por
  permutación y dependencia parcial, con figuras Plotly (summary, beeswarm, PDP).
- **Optimización** (`optimization/`): solver newsvendor / critical fractile para
  la decisión de pedido del Caso A (política agresiva/conservadora, costo esperado).
- **Reporte HTML** (`reporting/`): ensamblador Jinja2 autocontenido (plotly.js
  embebido una sola vez), secciones con narrativa + figuras + tablas.
- **Tablas maestras por caso** (`cases/masters.py`): cada caso cruza TODAS sus
  fuentes en una master table única (A: ventas ⨝ catálogo ⨝ tiendas ⨝ inventario
  ⨝ tendencias; B: detalle ⨝ tickets ⨝ catálogo; C: transacciones ⨝ loyalty ⨝
  exógenas ⨝ intensidad de promociones), con diagnóstico de cobertura de cada
  cruce. Script `scripts/build_masters_eda.py` genera las masters (Parquet) y un
  reporte HTML de EDA por caso sobre la tabla cruzada.
- **Caso A** (`cases/caso_a.py`): forecast probabilístico (modelo cuantílico) con
  features de rezago/estacionalidad, evaluación (WAPE/pinball/PICP/MPIW) vs
  baseline ingenuo y optimización de pedido newsvendor. Sobre datos reales: WAPE
  13% (R² 0.88) y ~70% de reducción del costo esperado de faltante+sobrante.
- **Caso B** (`cases/caso_b.py`): clustering de tiendas por perfil de compra
  (K-Means) + reglas de asociación FP-Growth (support/confidence/lift/conviction)
  → Top-N combos por cluster con precio propuesto y lift esperado. Sobre datos
  reales: 3 clusters (silhouette 0.32) y combos con lift 5–9 (p. ej. Buñuelo+Avena).
- **Caso C** (`cases/caso_c.py`): modelo inferencial GLM de los drivers del ticket
  (coeficientes + IC, outliers winsorizados) y modelo predictivo del gasto esperado
  del cliente recurrente (features RFM + loyalty). Sobre datos reales: driver
  dominante `total_articulos` (β≈8, p≈0) y `clima_Rainy` significativo (−0.39);
  el modelo de gasto logra R² 0.77 y mejora ~54% el WAPE sobre el baseline.
- **Pipelines Kedro** (`pipelines/caso_{a,b,c}/`): orquestan ingesta cruzada →
  modelado → salidas para cada caso; `pipeline_registry` los registra y compone
  el `__default__`. Catálogo tipado de salidas (masters, métricas, órdenes,
  combos, coeficientes) y parámetros de modelo por caso. `kedro run --pipeline
  caso_a|caso_b|caso_c` (o `kedro run` para los tres) ejecuta end-to-end.
- **Reporte HTML unificado** (`cases/reporting.py` + `scripts/build_unified_report.py`):
  entregable ejecutivo autocontenido con 7 secciones (resumen + EDA y modelo de
  cada caso), integrando desempeño, interpretabilidad e impacto de negocio con la
  narrativa autogenerada.
- **API de inferencia** (`serving/api/`, §11): FastAPI con endpoints versionados
  por caso (recomendación de pedido newsvendor, combos, gasto esperado), esquemas
  Pydantic, health/readiness y `/metrics` Prometheus opcional. Warmup en el
  arranque (masters + modelos en memoria). Con pruebas de integración (TestClient).
- **DevOps/AIOps**: `Dockerfile` (pipelines) y `serving/Dockerfile` (API)
  multi-stage con `uv` y usuario no root; `deployment/docker-compose.yml` (API +
  MLflow + Prometheus + Grafana) y provisión de monitoreo.
- **AIOps — drift** (`framework/monitoring/drift.py`): detección de drift por
  feature (PSI + KS para numéricas, chi² para categóricas) con narrativa y
  umbrales, para cerrar el ciclo monitoreo→detección→reentrenamiento.
- **MLOps / Docs**: model card y data card (`reports/`), scaffolding DVC
  (`dvc.yaml`, `.dvc/`), documentación MkDocs (`mkdocs.yml`, `docs/`) con
  `mkdocstrings`, y **README** en primera persona con puesta en marcha
  reproducible, decisiones de diseño, resultados y limitaciones.
