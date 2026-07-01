# Arquitectura

Kedro es la columna vertebral. El código se separa en dos capas:

- **`framework/`** — núcleo reutilizable y **agnóstico al caso**: `io`,
  `validation`, `profiling` (motor de EDA), `features`, `models` (BaseModel +
  registry), `tuning` (Optuna), `evaluation`, `interpret`, `optimization`
  (newsvendor), `viz`, `narrate`, `reporting`, `monitoring` (drift).
- **`pipelines/` y `cases/`** — orquestan el framework por caso. `cases/masters.py`
  cruza las fuentes en una tabla maestra por caso; `pipelines/caso_*` las
  ejecutan como nodos Kedro.

## Flujo de datos (capas Kedro)

```
01_raw → 02_intermediate → 03_primary → 04_feature →
05_model_input → 06_models → 07_model_output → 08_reporting
```

Los datasets crudos se declaran tipados en el catálogo; la persistencia ocurre en
capas ≥ 03. Cada caso construye su **tabla maestra** (03_primary) y produce
métricas, órdenes/combos/coeficientes (07_model_output) y reportes (08_reporting).

## MLOps / DevOps / AIOps

- **Tracking:** MLflow vía `kedro-mlflow` (params, métricas, artefactos).
- **Versionado:** Git para el crudo; DVC (`dvc.yaml`) para artefactos derivados.
- **CI/CD:** GitHub Actions (ruff, black, mypy, pytest, humo de Kedro).
- **Serving:** FastAPI (`serving/api`) con endpoints por caso.
- **Contenedores:** `Dockerfile` (pipelines) y `serving/Dockerfile` (API);
  `docker-compose` con MLflow + Prometheus + Grafana.
- **Observabilidad:** `/metrics` Prometheus + detección de drift (PSI/KS).
