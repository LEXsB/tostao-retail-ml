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
01_raw → 03_primary → 06_models → 07_model_output → 08_reporting
```

Sobre la convención de capas de Kedro, el proyecto materializa las que usa: los
datasets crudos se declaran tipados en el catálogo (`01_raw`, versionado) y la
persistencia ocurre en las capas derivadas (gitignored, se regeneran con
`kedro run`). Cada caso construye su **tabla maestra** (`03_primary`), entrena y
persiste su **modelo** (`06_models`), y produce métricas, órdenes/combos/coeficientes
(`07_model_output`) y reportes (`08_reporting`).

## MLOps / DevOps / AIOps

- **Tracking:** MLflow vía `kedro-mlflow` (params, métricas, artefactos).
- **Versionado:** Git para el crudo; DVC (`dvc.yaml`) para artefactos derivados.
- **CI/CD:** GitHub Actions (ruff check + ruff format, mypy, pytest con cobertura).
- **Serving:** FastAPI (`serving/api`) con endpoints por caso.
- **Contenedores:** `Dockerfile` (pipelines) y `serving/Dockerfile` (API);
  `docker-compose` con MLflow + Prometheus + Grafana.
- **Observabilidad:** `/metrics` Prometheus + detección de drift (PSI/KS).
