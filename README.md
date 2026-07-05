# tostao-retail-ml

[![CI](https://github.com/LEXsB/tostao-retail-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/LEXsB/tostao-retail-ml/actions/workflows/ci.yml)

Este repositorio implementa una **plataforma de ciencia de datos y machine learning
de grado producción** para una cadena de retail. No se trata de un notebook de
prueba, sino de un producto de software modular, reutilizable, probado y desplegable,
desarrollado bajo prácticas de MLOps, DevOps y AIOps.

La estrategia parte de un **framework reutilizable** sobre el que se resuelven los
tres casos de negocio, de manera que cada caso añade el mínimo código posible. Todo
el flujo se orquesta con Kedro y cada nodo emite, junto a sus artefactos, una
**mini-conclusión analítica** derivada de sus propias salidas.

| Caso | Problema | Enfoque | Resultado (datos reales) |
|------|----------|---------|--------------------------|
| **A — Abastecimiento** | ¿Cuánto pedir por SKU-tienda cada semana? | Forecast probabilístico + optimizador newsvendor (se validaron Ridge/GBR/cuantílico/ensemble) | WAPE ~12–13 %, R² ~0.90; **−70 %** costo esperado de faltante+sobrante |
| **B — Combos** | ¿Qué empaquetar y a qué precio? | Clustering (K-Means vs. Aglomerativo) + reglas FP-Growth + grafo de co-compra | 3 clusters (silhouette 0.32); combos con **lift 5–9** |
| **C — AOV** | ¿Qué mueve el ticket y cuánto gastará el cliente? | GLM inferencial + predictivo (Ridge/GBR/ensemble) sobre RFM | driver `total_articulos` (β~8); gasto **R² ~0.81** |

Explicación detallada de cómo funciona cada caso: [Caso A](docs/caso_a.md) ·
[Caso B](docs/caso_b.md) · [Caso C](docs/caso_c.md).

## Glosario

Términos y herramientas que aparecen en el repositorio (los específicos de cada caso
están en su propio README):

- **uv** — gestor de entorno y paquetes de Python: fija la versión de Python, crea el
  `.venv`, instala dependencias con lockfile y ejecuta comandos (`uv run …`).
- **Kedro** — framework que orquesta el flujo de datos (catálogo de datasets +
  pipelines de nodos); es el ejecutor (`kedro run`).
- **cookiecutter** — generador de andamiaje que crea la estructura estándar del
  proyecto (`conf/`, `data/`, `src/`, …).
- **MLflow** — registro (*tracking*) de experimentos, métricas y modelos.
- **DVC** — versionado de datos y artefactos derivados (complementa a Git para
  archivos grandes/generados).
- **Pandera** — validación de esquemas de datos (tipos, rangos, unicidad): contratos
  que hacen fallar el pipeline ante datos inválidos.
- **Optuna** — optimización de hiperparámetros (búsqueda TPE).
- **SHAP** — interpretabilidad: contribución de cada variable a la predicción.
- **FastAPI + Uvicorn** — API de inferencia (*serving*) y su servidor.
- **Prometheus + Grafana** — métricas y tableros de observabilidad.
- **ruff / mypy / pytest** — lint y formato / tipos estáticos / pruebas.
- **HPO** — *Hyperparameter Optimization*, optimización de hiperparámetros.
- **holdout** — partición de datos reservada para evaluar, que no se usa al entrenar.
- **ensemble** — combinación de varios modelos para mejorar la predicción.
- **drift (PSI/KS)** — desviación de los datos en producción respecto al
  entrenamiento; se detecta con el índice PSI y la prueba KS.

## Por qué lo construimos así (decisiones de diseño)

- **Framework antes que casos.** El núcleo (EDA, features, modelos, HPO, evaluación,
  interpretabilidad, optimización, narración, reporte) es agnóstico al caso, porque
  los tres comparten el 90 % de la mecánica. Gracias a ello, cada caso nuevo reduce el
  código respecto al anterior: solo aporta su tabla maestra y su nodo de modelo.
- **Una tabla maestra por caso.** En lugar de analizar fuente por fuente, cruzamos
  todas las fuentes de cada caso en una única tabla maestra y sobre ella hacemos el
  EDA, el modelado y la estadística. Cada cruce reporta su cobertura (integridad
  referencial), de modo que el pipeline falla temprano si se rompe una relación.
- **Validación de varios modelos por caso.** No nos quedamos con un solo modelo:
  comparamos candidatos en el holdout y, donde aporta, los combinamos en un ensemble.
  El hallazgo honesto en estos datos es que, tras el feature engineering, el modelo
  lineal (Ridge) supera al boosting en las dos tareas de regresión.
- **Forecast probabilístico en el Caso A.** La decisión de pedido depende de la
  incertidumbre y no solo del valor esperado; por eso modelamos cuantiles y resolvemos
  el pedido con el *critical fractile* (newsvendor).
- **Configuración sobre código.** Semillas, umbrales, costos y espacios de HPO viven
  en YAML versionados; nada queda escrito a mano en el código.
- **Reproducibilidad total.** El entorno se fija con `uv` + lockfile (Python 3.13) y
  semillas globales deterministas.

## Cruces de las tablas maestras (resumen)

Cada caso construye **una tabla maestra** partiendo de su **tabla de hechos** y
uniéndole sus dimensiones con **LEFT JOIN** (para no perder ningún hecho). Las llaves
salen del grano de cada dimensión; todas las dimensiones son **únicas en su llave**,
por lo que los cruces son **N:1** y **no duplican** filas.

| Caso | Hecho (grano) | Dimensiones y llaves | Resultado verificado |
|------|---------------|----------------------|----------------------|
| **A** | `ventas_historicas` (fecha × tienda × producto) | catálogo (`id_producto`), tiendas (`id_tienda`), inventario y tendencia (`id_tienda + id_producto`) | 14.560 filas, **sin duplicidad**, cobertura **100 %** |
| **B** | `detalle_tickets` (línea de ticket) | cabecera (`id_ticket`), catálogo (`id_producto`) | 18.376 filas, **sin duplicidad**, cobertura **100 %** |
| **C** | `transacciones_resumen` (ticket) | loyalty (`id_cliente`), exógenas y promos (`fecha + id_tienda`) | 10.000 filas, **sin duplicidad**, integridad **100 %** (promo = penetración 98 %) |

Cada cruce emite su **cobertura** (`join_report`) como control de integridad, y la
maestra conserva exactamente las filas del hecho (sin *fan-out*). Un caso especial: en
el Caso C las promociones son **rangos de fecha**, así que se pre-agregan a una
**intensidad** por `(fecha, tienda)` antes de unir, para no duplicar tickets. El
detalle de cada cruce, con su **diagrama entidad-relación (ERD)**, está en
[Caso A](docs/caso_a.md), [Caso B](docs/caso_b.md) y [Caso C](docs/caso_c.md).

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
notebooks/                # guía de análisis en paralelo (EDA + modelamiento por caso)
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

**¿Qué es `uv`?** Es un gestor de proyectos y paquetes de Python, muy rápido (escrito
en Rust por el equipo de `ruff`). Reúne en una sola herramienta lo que antes exigía
varias: instala y **fija la versión de Python** (no hace falta tenerla preinstalada),
crea el **entorno virtual** (`.venv`), resuelve e instala las **dependencias** con un
**lockfile** (`uv.lock`) para reproducibilidad exacta, y **ejecuta** comandos dentro
del entorno (`uv run …`). En la práctica es el equivalente de `npm`/`yarn` para
Python. Por eso `uv sync` (prepara el entorno) y `uv run kedro run` (ejecuta el
pipeline) funcionan igual en cualquier máquina, sin instalar nada a mano. Se instala
en segundos (`pip install uv` o el instalador oficial); ver [docs](https://docs.astral.sh/uv/).

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

Herramientas de desarrollo (grupo `dev`): `pytest`+`pytest-cov`, `ruff` (lint y
formato), `mypy`, `pre-commit`, `mkdocs`+`mkdocs-material`, `jupyter`.

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

### Atajo: ejecutar todo desde un notebook

Para quien prefiere no usar la terminal (o trabaja en Google Colab), hay dos
notebooks orquestadores que se abren y se ejecutan con *Run All*:

- **`notebooks/run_pipeline.ipynb`** — prepara el entorno con `uv` y corre el
  pipeline completo (`kedro run` de los tres casos) más los reportes.
- **`notebooks/run_notebooks.ipynb`** — ejecuta uno a uno los notebooks de EDA y
  modelamiento de cada caso y guarda las copias ejecutadas (con gráficos) en
  `notebooks/_ejecutados/`.

Ambos instalan `uv` si falta y no requieren Python 3.13 preinstalado ni subir los
datos aparte (el crudo ya está versionado en `data/01_raw/`).

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

### Cómo se crea un proyecto Kedro (y qué genera)

Un proyecto Kedro se crea con el comando interactivo **`kedro new`**, que a partir de
la plantilla **cookiecutter** oficial pregunta el nombre y qué *tools* incluir
(Linting, Testing, Custom Logging, Documentation, Data Structure, PySpark, Kedro-Viz)
y si añadir un pipeline de ejemplo. Este repositorio se generó con las tools
**Linting, Testing, Custom Logging y Data Structure**, sin pipeline de ejemplo (se ve
en `[tool.kedro]` de `pyproject.toml`).

Con esa elección, Kedro genera de fábrica esta estructura (aquí ya poblada por el
proyecto):

```text
conf/
  base/               # configuración versionada
    catalog.yml       # datasets (aquí + catalog_cases.yml)
    parameters.yml    # parámetros por caso
    logging.yml       # (tool «Custom Logging»)
  local/              # overrides y credenciales LOCALES (gitignored)
data/                 # 8 capas 01_raw … 08_reporting (tool «Data Structure»)
src/tostao_ml/
  __init__.py
  __main__.py           # envoltorio del CLI de Kedro (python -m tostao_ml)
  settings.py           # config loader (OmegaConf), hooks y patrones de config
  pipeline_registry.py  # registro de pipelines (ver abajo)
  pipelines/            # un subpaquete por pipeline
tests/                  # (tool «Testing»)
notebooks/
pyproject.toml          # incluye la sección [tool.kedro]
```

El resto (`framework/`, `cases/`, los pipelines de cada caso, `serving/`,
`deployment/`, reportes) lo añadimos nosotros: Kedro solo aporta el andamiaje y los
puntos de extensión. Dos piezas de configuración clave:

- **`settings.py`** declara el *config loader* (`OmegaConfigLoader`), los entornos
  (`base`, versionado, y `local`, para secretos y gitignored) y los patrones que
  permiten dividir el catálogo y los parámetros en varios archivos (`catalog*`,
  `parameters*`, `mlflow*`) — por eso conviven `catalog.yml` y `catalog_cases.yml`.
- **`conf/base` vs `conf/local`:** `base` se versiona; `local` (credenciales, rutas de
  máquina) nunca sube al repositorio.

### El `pipeline_registry`

`src/tostao_ml/pipeline_registry.py` es el punto donde Kedro **descubre** los
pipelines. Expone una única función, `register_pipelines()`, que devuelve un
diccionario `nombre → Pipeline`:

```python
def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {
        "caso_a": caso_a.create_pipeline(),
        "caso_b": caso_b.create_pipeline(),
        "caso_c": caso_c.create_pipeline(),
    }
    pipelines["__default__"] = pipelines["caso_a"] + pipelines["caso_b"] + pipelines["caso_c"]
    return pipelines
```

Cómo se usa:

- Al arrancar, Kedro llama a `register_pipelines()` y guarda ese mapa.
- **`kedro run`** (sin argumentos) ejecuta la clave especial **`__default__`** —aquí,
  los tres casos encadenados—.
- **`kedro run --pipeline caso_a`** ejecuta solo ese pipeline por su nombre.
- **`kedro registry list`** lista los nombres registrados (se usa como prueba de humo
  en la CI).

Cada `create_pipeline()` (en `pipelines/caso_*/pipeline.py`) compone sus **nodos**
—funciones puras con entradas y salidas nombradas del catálogo— y los pipelines se
pueden **sumar** con `+` para combinarlos, que es justo como se arma el `__default__`.
Así, añadir un caso nuevo se reduce a escribir su `create_pipeline()` y registrarlo
con una línea.

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
- **CI/CD:** GitHub Actions (ruff check + ruff format, mypy, pytest con cobertura)
  en cada push/PR; `pre-commit` con los mismos hooks.
- **Contenerización:** imágenes multi-stage con `uv` y usuario no root.
- **Serving:** FastAPI con validación Pydantic, health/readiness y `/metrics`.
- **Observabilidad:** Prometheus + Grafana; detección de drift (PSI/KS) para cerrar
  el ciclo monitoreo -> detección -> reentrenamiento.

## Calidad: lint, tipos y pruebas

Las comprobaciones se ejecutan con `uv` y son **las mismas en local y en CI**:

```bash
uv run ruff check src tests           # lint (estilo, imports, bugs comunes)
uv run ruff format --check src tests   # formato (ruff; línea de 100)
uv run mypy src                        # tipos estáticos
uv run pytest                          # pruebas + cobertura (resumen en consola)
```

- **Lint y formato — `ruff`.** Reglas y estilo en `pyproject.toml` (`[tool.ruff]`).
  `ruff format` es el único formateador del proyecto.
- **Tipos — `mypy`** sobre `src/`.
- **Pruebas — `pytest`** (con `pytest-cov`) en `tests/`, organizadas por marcadores:
  `unit`, `integration`, `data_contract` y `calificacion`. Ejemplos:
  `uv run pytest -m unit` o `uv run pytest -m calificacion`.
- **`pre-commit`.** Los mismos hooks corren en cada commit; se activan una vez con
  `uv run pre-commit install`.

**Quién lo administra.** La CI de **GitHub Actions** (`.github/workflows/ci.yml`)
ejecuta lint + formato + tipos + pruebas + humo de Kedro en cada `push` y `pull
request` a `main` y `develop`; **ningún cambio se fusiona sin la CI en verde**. Un
workflow programado (`.github/workflows/calificacion-mensual.yml`) corre cada mes la
calificación de un periodo futuro. La aprobación de los PR la realiza el
administrador del repositorio (ver [Flujo de trabajo](#flujo-de-trabajo-ramas-pr-y-cicd)).

## Flujo de trabajo (ramas, PR y CI/CD)

El proyecto trabaja con dos ramas de larga vida —**`main`** (producción, siempre
desplegable) y **`develop`** (integración)— y ramas de trabajo por cada cambio:

1. **Una rama por feature o solicitud del backlog.** Cada funcionalidad, corrección
   o solicitud se desarrolla en su propia rama (`feature/…`, `fix/…`), nombrada según
   el ítem que resuelve.
2. **Pull Request a `develop`.** Al terminar, se abre un PR de la rama hacia
   `develop`. La CI (ruff check + ruff format, mypy, pytest con cobertura) debe pasar
   y **un administrador** revisa y **acepta** el PR antes de fusionar.
3. **Pull Request de `develop` a `main`.** En cada hito se abre un PR de `develop`
   hacia `main`, también revisado y **aceptado por un administrador**.

Cada fusión conserva el historial (merge commits `--no-ff`, sin *squash*), de modo
que se mantiene la trazabilidad de cada cambio. El repositorio incluye `CODEOWNERS`
y una plantilla de PR (`.github/PULL_REQUEST_TEMPLATE.md`) para estandarizar la
revisión.

> **Nota.** En esta entrega, al ser un proyecto individual, el mismo desarrollador
> crea las ramas, abre los PR y los acepta. El flujo correcto —y para el que el
> repositorio está preparado— es que la aprobación a `develop` y a `main` la realice
> un administrador distinto (con protección de ramas que exija la revisión).

## Resultados y limitaciones

Los resultados provienen de ejecuciones reales sobre los datos provistos.
Limitaciones que documento con honestidad: las series del Caso A son cortas
(~13 semanas) y los intervalos de predicción tienden a **sub-cubrir** (conviene
calibrarlos); en el Caso B el lift alto puede reflejar baja frecuencia (validar con
A/B antes de fijar precios); el GLM del Caso C asume relaciones aproximadamente
lineales. Ver `reports/model_card.md` para el detalle.

## Licencia

[MIT](LICENSE).
