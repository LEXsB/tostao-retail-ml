# tostao-retail-ml

Plataforma de ciencia de datos y machine learning de grado producción para una cadena
de retail. Construí este proyecto como un **producto de software modular y reutilizable**
—no como un conjunto de notebooks sueltos— siguiendo prácticas de MLOps, DevOps y AIOps.

Resuelvo tres casos de negocio sobre un mismo núcleo de código compartido:

| Caso | Problema | Enfoque |
|------|----------|---------|
| 🚚 **A — Abastecimiento** | ¿Cuánto pedir por SKU-tienda cada semana? | Forecast probabilístico de demanda + optimizador de pedido (critical fractile / newsvendor). |
| 🥐 **B — Combos** | ¿Qué productos empaquetar y a qué precio? | Clustering de tiendas + reglas de asociación (FP-Growth) y grafos de co-compra. |
| 🧾 **C — AOV** | ¿Qué mueve el ticket promedio y cuánto gastará el cliente? | Modelo inferencial (GLM + SHAP) de drivers + modelo predictivo de gasto. |

> **Filosofía de diseño.** Escribí primero un *framework* reutilizable (carga, validación,
> EDA, features, modelos, HPO, evaluación, visualización, narración y reporte) y luego
> implementé los tres casos **reutilizándolo**, de modo que cada caso nuevo añade el mínimo
> código posible. Todo el pipeline se orquesta con Kedro y cada nodo emite, además de sus
> artefactos, una **mini-conclusión analítica** derivada de sus propias salidas.

## Estado

Proyecto en construcción incremental. Consulta el [CHANGELOG](CHANGELOG.md) para el detalle
de avances y el árbol de carpetas en la sección de arquitectura (más abajo, en expansión).

## Requisitos

- Python 3.13 (gestionado con [`uv`](https://docs.astral.sh/uv/)).
- `git` y, opcionalmente, `gh` (GitHub CLI) y Docker.

## Puesta en marcha rápida

```bash
# 1) Clonar
git clone https://github.com/LEXsB/tostao-retail-ml.git
cd tostao-retail-ml

# 2) Crear entorno reproducible e instalar el stack
uv sync --extra all --group dev

# 3) Verificación de humo
uv run kedro info
```

## Estructura (resumen)

```text
conf/                # configuración Kedro (catálogo, parámetros por caso, logging)
data/                # capas de datos (01_raw … 08_reporting)
src/tostao_ml/
├── framework/       # núcleo reutilizable y agnóstico al caso
└── pipelines/       # pipelines Kedro que orquestan el framework por caso
notebooks/           # vitrina del análisis (importan src/, cargan por catálogo)
tests/               # unit + integration + data_contracts
```

## Licencia

[MIT](LICENSE).
