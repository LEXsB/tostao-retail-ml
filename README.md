# tostao-retail-ml

Construí una **plataforma de ciencia de datos y machine learning de grado
producción** para una cadena de retail. No es un notebook de prueba: es un
producto de software modular, reutilizable, testeado y desplegable, siguiendo
prácticas de MLOps, DevOps y AIOps.

Diseñé primero un **framework reutilizable** y luego resolví tres casos de negocio
reutilizándolo, de modo que cada caso añade el mínimo código posible. Todo el
pipeline se orquesta con Kedro y cada nodo emite, además de sus artefactos, una
**mini-conclusión analítica** derivada de sus propias salidas.

| Caso | Problema | Enfoque | Resultado (datos reales) |
|------|----------|---------|--------------------------|
| **A — Abastecimiento** | ¿Cuánto pedir por SKU-tienda cada semana? | Forecast probabilístico + optimizador newsvendor (se validaron Ridge/GBR/cuantílico/ensemble) | WAPE ~12–13 %, R² ~0.90; **−70 %** costo esperado de faltante+sobrante |
| **B — Combos** | ¿Qué empaquetar y a qué precio? | Clustering (K-Means vs. Aglomerativo) + reglas FP-Growth + grafo de co-compra | 3 clusters (silhouette 0.32); combos con **lift 5–9** |
| **C — AOV** | ¿Qué mueve el ticket y cuánto gastará el cliente? | GLM inferencial + predictivo (Ridge/GBR/ensemble) sobre RFM | driver `total_articulos` (β~8); gasto **R² ~0.81** |

Explicación detallada de cómo funciona cada caso: [Caso A](docs/caso_a.md) ·
[Caso B](docs/caso_b.md) · [Caso C](docs/caso_c.md).

## Por qué lo construí así (decisiones de diseño)

- **Framework antes que casos.** Escribí un núcleo agnóstico (EDA, features,
  modelos, HPO, evaluación, interpretabilidad, optimización, narración, reporte)
  porque los tres casos comparten el 90 % de la mecánica. Cada caso nuevo redujo
  el código respecto al anterior: solo aporta su tabla maestra y su nodo de modelo.
- **Una tabla maestra por caso.** En lugar de analizar fuente por fuente, crucé
  todas las fuentes de cada caso en una única tabla maestra y sobre ella hice EDA,
  modelado y estadística. Cada cruce reporta su cobertura (integridad referencial),
  de modo que el pipeline falla temprano si se rompe una FK.
- **Validación de varios modelos por caso.** No me quedé con un solo modelo: comparo
  candidatos en el holdout y, donde aporta, los combino en un ensemble. El hallazgo
  honesto en estos datos es que, tras el feature engineering, el modelo lineal (Ridge)
  supera al boosting en las dos tareas de regresión.
- **Forecast probabilístico en el Caso A.** La decisión de pedido depende de la
  incertidumbre, no solo del valor esperado; por eso modelé cuantiles y resolví el
  pedido con el *critical fractile* (newsvendor).
- **Configuración sobre código.** Semillas, umbrales, costos y espacios de HPO
  viven en YAML versionados; nada hardcodeado.
- **Reproducibilidad total.** Fijé el entorno con `uv` + lockfile (Python 3.13) y
  semillas globales deterministas.

## Arquitectura

```text
conf/                     # configuración Kedro (catálogo, parámetros por caso, logging, mlflow)
data/                     # capas 01_raw ... 08_reporting (01_raw versionado; derivadas via DVC)
src/tostao_ml/
  framework/              # NÚCLEO reutilizable y agnóstico al caso:
                          #   io, validation, profiling, features, models, tuning,
                          #   evaluation, interpret, optimization, viz, narrate,
                          #   reporting, monitoring
  cases/                  # masters.py (cruces) + caso_a/b/c + reporting + storytelling
  pipelines/              # pipelines Kedro que orquestan el framework por caso
serving/api/              # API FastAPI de inferencia (endpoints por caso)
deployment/               # docker-compose + monitoreo (Prometheus/Grafana)
notebooks/                # guía de análisis en paralelo (EDA + modelos por caso)
tests/                    # unit + integration + data_contracts
reports/                  # model card, data card, y reports/ejecutivo (HTML por caso)
docs/                     # MkDocs (mkdocstrings) + un documento por caso
```

## Requisitos mínimos y entorno

| Requisito | Detalle |
|-----------|---------|
| **SO** | Linux, macOS o Windows 10/11 (probado en Windows 10 y en Ubuntu por CI). |
| **Python** | 3.13 (`>=3.13,<3.14`). No hace falta instalarlo: `uv` lo descarga y fija. |
| **Gestor de entorno** | [`uv`](https://docs.astral.sh/uv/) ≥ 0.11 (resuelve, instala y bloquea el entorno). |
| **Git** | Cualquier versión reciente. |
| **RAM** | 4 GB bastan; el dato es pequeño (miles de filas por caso). |
| **Disco** | ~2 GB para el entorno virtual (`.venv`) y las dependencias. |
| **Opcional** | Docker + Docker Compose para el stack de serving/observabilidad. |

**El stack (dependencias principales y su rol).** Las versiones exactas quedan
fijadas en `uv.lock`; estas son las cotas declaradas en `pyproject.toml`:

| Paquete | Versión | Para qué |
|---------|---------|----------|
| `kedro` | ≥ 0.19 | Orquestación: catálogo de datos, nodos y pipelines. |
| `kedro-datasets` / `kedro-viz` / `kedro-mlflow` | ≥ 4.0 / ≥ 10.0 / ≥ 0.13 | Datasets del catálogo, grafo del pipeline y tracking MLflow. |
| `pandas` / `polars` / `pyarrow` / `numpy` | ≥ 2.2 / ≥ 1.0 / ≥ 16.0 / ≥ 1.26 | Manipulación de datos y Parquet. |
| `scikit-learn` / `statsmodels` / `scipy` | ≥ 1.5 / ≥ 0.14 / ≥ 1.13 | Modelos, GLM inferencial y estadística. |
| `optuna` | ≥ 4.0 | Optimización de hiperparámetros (TPE). |
| `shap` | ≥ 0.46 | Interpretabilidad de modelos. |
| `pandera` | ≥ 0.20 | Validación de esquemas (contratos de datos). |
| `plotly` / `jinja2` | ≥ 5.22 / ≥ 3.1 | Figuras y ensamblado de reportes HTML. |
| `mlflow` | ≥ 2.14 | Registro de experimentos y modelos. |
| `mlxtend` / `networkx` *(extra `caso_b`)* | ≥ 0.23 / ≥ 3.3 | Reglas de asociación y grafo de co-compra. |
| `fastapi` / `uvicorn` *(extra `serving`)* | ≥ 0.111 / ≥ 0.30 | API de inferencia. |

Herramientas de desarrollo (grupo `dev`): `pytest`+`pytest-cov`, `ruff`, `black`,
`mypy`, `pre-commit`, `mkdocs`+`mkdocs-material`, `jupyter`.

## Puesta en marcha (reproducible)

Prerrequisitos: [`uv`](https://docs.astral.sh/uv/), `git`, y opcionalmente Docker.

```bash
# 1) Clonar
git clone https://github.com/LEXsB/tostao-retail-ml.git
cd tostao-retail-ml

# 2) Entorno reproducible + stack (Python 3.13 se instala solo vía uv)
uv sync --extra caso_b --extra serving --group dev

# 3) Verificación
uv run kedro info
uv run pytest

# 4) Ejecutar los pipelines (los tres casos)
uv run kedro run                      # o: --pipeline caso_a | caso_b | caso_c

# 5) Tablas maestras + reportes HTML (completo y ejecutivo por caso)
uv run python scripts/build_reports.py   # ejecutivo -> reports/ejecutivo/  ·  completo -> data/08_reporting/

# 6) API de inferencia
uv run uvicorn serving.api.main:app --reload     # http://localhost:8000/docs

# 7) MLflow UI (tras un kedro run)
uv run mlflow ui                                  # http://localhost:5000

# 8) Calificar un periodo futuro (simula el mes siguiente y evalúa el modelo)
uv run pytest -m calificacion -v

# 9) Grafo del pipeline
uv run kedro viz
```

### Qué es Kedro (y cómo se usa aquí)

**Kedro** es un framework de orquestación para proyectos de datos: impone una
estructura estándar (creada con la plantilla **cookiecutter** oficial de Kedro —
un generador de andamiaje que produce siempre las mismas carpetas: `conf/`, `data/`
por capas, `src/<paquete>/`, `pipelines/`) y aporta dos piezas clave:

- El **`DataCatalog`** (`conf/base/catalog*.yml`): declara cada dataset por nombre,
  su tipo y su ruta. El código nunca abre archivos con `pd.read_csv`; pide al
  catálogo `catalog.load("a_ventas_historicas")`. Cambiar un origen es editar YAML.
- Los **pipelines**: un pipeline es un grafo de **nodos** (funciones puras) donde
  las salidas de uno son las entradas de otro. Kedro resuelve el orden por
  dependencias y ejecuta.

Sí: **Kedro es el ejecutor**. `uv run kedro run` es el punto de entrada. Kedro lee
`[tool.kedro]` de `pyproject.toml` (paquete `tostao_ml`, `src/` como raíz), carga
`src/tostao_ml/pipeline_registry.py` —donde se registran los pipelines de los tres
casos— y ejecuta, para cada caso:

1. **Ingesta.** El `DataCatalog` carga tipadas las fuentes crudas de `data/01_raw`.
2. **Tabla maestra.** El nodo `build_master_*` cruza todas las fuentes del caso en
   una única tabla (`cases/masters.py`) y valida la cobertura de cada cruce.
3. **Modelado.** El nodo del caso llama a `cases/caso_*.py`, que reutiliza el
   framework (features, modelos, HPO, evaluación) y compara varios modelos.
4. **Salidas.** Métricas, órdenes/combos/coeficientes y el **modelo entrenado**
   (`data/06_models/*.pkl`) se persisten por el catálogo en las capas `>= 03`.

### Cómo se entrelaza todo (arquitectura funcional)

```
conf/ (catálogo + parámetros)                     data/01_raw (fuentes)
        │                                                │
        ▼                                                ▼
pipeline_registry ──> pipelines/caso_X ──> nodos ──> cases/masters (cruces)
                                             │
                                             ▼
                        cases/caso_X ──reutiliza──> framework/ (features, models,
                                             │        tuning, evaluation, interpret,
                                             │        optimization, viz, narrate)
                                             ▼
                        artefactos: métricas, modelo.pkl, órdenes/combos/coefs
                                             │
                 ┌───────────────────────────┼───────────────────────────┐
                 ▼                            ▼                            ▼
     cases/reporting + storytelling   cases/executive          serving/api (FastAPI)
     → reporte COMPLETO (08_reporting) → reporte EJECUTIVO       carga modelo.pkl y
       gitignored                        (reports/ejecutivo)     sirve predicciones
```

Las **capas** dependen unas de otras, no al revés: `pipelines/` solo orquesta;
`cases/` traduce cada problema de negocio a llamadas al `framework/`; el
`framework/` es agnóstico al caso y no conoce a Tostao. Por eso un caso nuevo solo
añade su master y su nodo de modelo.

Los **notebooks** (`notebooks/`) corren **en paralelo** a este flujo: son la guía de
exploración que un desarrollador hace antes (o al lado) de la puesta en producción.
Importan `tostao_ml` y cargan por el catálogo; por caso hay uno de **EDA**
(`01_eda`) y uno de **modelamiento** (`02_modelamiento`) que **despliega el modelo
por dentro** —features, partición, HPO, ajuste, evaluación, comparación e impacto—
con las mismas piezas del framework que usa producción, para que se vea el
modelamiento y se reproduzca. El detalle por caso está en [Caso A](docs/caso_a.md) ·
[Caso B](docs/caso_b.md) · [Caso C](docs/caso_c.md).

**Reportes.** `scripts/build_reports.py` genera dos por caso: el **completo** (EDA a
fondo + modelado interpretado, en `data/08_reporting`, no versionado) y el
**ejecutivo** para negocio (`reports/ejecutivo/reporte_caso_*.html`, versionado).
Nada está hardcodeado: parámetros, costos y espacios de HPO están en `conf/`.

### Docker

```bash
docker compose -f deployment/docker-compose.yml up --build
# API :8000  MLflow :5000  Prometheus :9090  Grafana :3000
```

## MLOps, DevOps y AIOps

- **Tracking/registry:** MLflow vía `kedro-mlflow`; **versionado** con DVC
  (`dvc.yaml`) y el catálogo de Kedro. Model/Data cards en `reports/`.
- **CI/CD:** GitHub Actions (ruff, black, mypy, pytest con cobertura, humo de
  Kedro) en cada push/PR; `pre-commit` con los mismos hooks.
- **Contenerización:** imágenes multi-stage con `uv` y usuario no root.
- **Serving:** FastAPI con validación Pydantic, health/readiness y `/metrics`.
- **Observabilidad:** Prometheus + Grafana; detección de drift (PSI/KS) para cerrar
  el ciclo monitoreo -> detección -> reentrenamiento.

## Resultados y limitaciones

Los resultados provienen de ejecuciones reales sobre los datos provistos.
Limitaciones que documento con honestidad: las series del Caso A son cortas
(~13 semanas) y los intervalos de predicción tienden a **sub-cubrir** (conviene
calibrarlos); en el Caso B el lift alto puede reflejar baja frecuencia (validar con
A/B antes de fijar precios); el GLM del Caso C asume relaciones aproximadamente
lineales. Ver `reports/model_card.md` para el detalle.

## Licencia

[MIT](LICENSE).
