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
| 🚚 **A — Abastecimiento** | ¿Cuánto pedir por SKU-tienda cada semana? | Forecast probabilístico + optimizador newsvendor | WAPE ≈ 13 % (R² 0.88); **−70 %** costo esperado de faltante+sobrante |
| 🥐 **B — Combos** | ¿Qué empaquetar y a qué precio? | Clustering de tiendas + reglas FP-Growth | 3 clusters (silhouette 0.32); combos con **lift 5–9** |
| 🧾 **C — AOV** | ¿Qué mueve el ticket y cuánto gastará el cliente? | GLM inferencial + modelo predictivo (RFM) | driver `total_articulos` (β≈8); gasto **R² 0.77**, −54 % WAPE |

## Por qué lo construí así (decisiones de diseño)

- **Framework antes que casos.** Escribí un núcleo agnóstico (EDA, features,
  modelos, HPO, evaluación, interpretabilidad, optimización, narración, reporte)
  porque los tres casos comparten el 90 % de la mecánica. Cada caso nuevo redujo
  el código respecto al anterior: solo aporta su tabla maestra y su nodo de modelo.
- **Una tabla maestra por caso.** En lugar de analizar fuente por fuente, **crucé
  todas las fuentes de cada caso en una única tabla maestra** y sobre ella hice
  EDA, modelado y estadística. Cada cruce reporta su cobertura (integridad
  referencial), de modo que el pipeline falla temprano si se rompe una FK.
- **Forecast probabilístico en el Caso A.** La decisión de pedido depende de la
  incertidumbre, no solo del valor esperado; por eso modelé cuantiles y resolví el
  pedido con el *critical fractile* (newsvendor), que balancea margen perdido vs.
  costo de almacenamiento.
- **Configuración sobre código.** Semillas, umbrales, costos y espacios de HPO
  viven en YAML versionados; nada hardcodeado.
- **Reproducibilidad total.** Fijé el entorno con `uv` + lockfile (Python 3.13),
  semillas globales y hashing de datos para linaje.

## Arquitectura

```text
conf/                     # configuración Kedro (catálogo, parámetros por caso, logging, mlflow)
data/                     # capas 01_raw … 08_reporting (01_raw versionado; derivadas via DVC)
src/tostao_ml/
├── framework/            # NÚCLEO reutilizable y agnóstico al caso
│   ├── io, validation, profiling, features, models, tuning, evaluation,
│   ├── interpret, optimization, viz, narrate, reporting, monitoring
├── cases/                # masters.py (cruces) + caso_a/b/c + reporting unificado
└── pipelines/            # pipelines Kedro que orquestan el framework por caso
serving/api/              # API FastAPI de inferencia (endpoints por caso)
deployment/               # docker-compose + monitoreo (Prometheus/Grafana)
notebooks/                # vitrina del análisis (importan src/, cargan por catálogo)
tests/                    # unit + integration + data_contracts
reports/                  # model card, data card
docs/                     # MkDocs (mkdocstrings)
```

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

# 5) Tablas maestras + EDA por caso, y reporte HTML unificado
uv run python scripts/build_masters_eda.py
uv run python scripts/build_unified_report.py   # -> data/08_reporting/reporte_unificado.html

# 6) API de inferencia
uv run uvicorn serving.api.main:app --reload     # http://localhost:8000/docs

# 7) MLflow UI (tras un kedro run)
uv run mlflow ui                                  # http://localhost:5000

# 8) Grafo del pipeline
uv run kedro viz
```

### Docker

```bash
docker compose -f deployment/docker-compose.yml up --build
# API :8000 · MLflow :5000 · Prometheus :9090 · Grafana :3000
```

## MLOps · DevOps · AIOps

- **Tracking/registry:** MLflow vía `kedro-mlflow`; **versionado** con DVC
  (`dvc.yaml`) y el catálogo de Kedro. **Model/Data cards** en `reports/`.
- **CI/CD:** GitHub Actions (ruff, black, mypy, pytest con cobertura, humo de
  Kedro) en cada push/PR; `pre-commit` con los mismos hooks.
- **Contenerización:** imágenes multi-stage con `uv` y usuario no root.
- **Serving:** FastAPI con validación Pydantic, health/readiness y `/metrics`.
- **Observabilidad:** Prometheus + Grafana; **detección de drift** (PSI/KS) para
  cerrar el ciclo monitoreo → detección → reentrenamiento.

## Resultados y limitaciones

Los resultados (arriba) provienen de ejecuciones reales sobre los datos provistos.
Limitaciones que documento con honestidad: las series del Caso A son cortas
(~13 semanas) y los intervalos de predicción tienden a **sub-cubrir** (conviene
calibrarlos); en el Caso B el lift alto puede reflejar baja frecuencia (validar con
A/B antes de fijar precios); el GLM del Caso C asume relaciones aproximadamente
lineales. Ver `reports/model_card.md` para el detalle.

## Licencia

[MIT](LICENSE).
