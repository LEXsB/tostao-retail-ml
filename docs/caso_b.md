# Caso B — Creación de Combos

## Problema de negocio

Se busca incrementar el ticket promedio mediante venta cruzada, identificando
patrones de compra no evidentes. El objetivo es proponer los **Top-5 combos** por
cluster de tiendas, con su precio y el **lift** esperado, filtrando el ruido de
artículos de alta frecuencia y baja correlación específica.

## Datos y tabla maestra

Se cruzan las tres fuentes de `02_product_bundles`:

```
detalle_tickets  x  tickets            (id_ticket)
                 x  catalogo_productos (id_producto)
```

El resultado es una fila por línea de ticket enriquecida (cabecera + catálogo). De
ahí se arma la **matriz de cestas** (ticket x producto, booleana) para la minería
de patrones.

Código: `cases/masters.py::build_master_b` y `build_baskets_b`.

## Modelos y validación (dos enfoques complementarios)

1. **Segmentación de tiendas.** Se construye un perfil de compra por tienda (mezcla
   de categorías + estadísticos de cesta) y se agrupa con **K-Means**. Se **compara
   K-Means vs. Aglomerativo** por *silhouette* y se elige el mejor; además se hace
   un barrido para seleccionar `k` (no a dedo).
2. **Reglas de asociación.** Con **FP-Growth** se extraen reglas de co-compra por
   cluster (support / confidence / lift / conviction), filtrando por `lift > 1` y un
   soporte mínimo para descartar ruido.
3. **Grafo de co-compra (complementario).** Se modela un grafo (networkx) donde los
   nodos son productos y las aristas su co-ocurrencia; la **centralidad** revela los
   productos «hub» que conectan la red, una vista alternativa a las reglas.

No hay partición train/test porque es aprendizaje **no supervisado**: se usa todo
el histórico de cestas.

## Métricas (lectura, no definición)

- **Silhouette** ~0.32: estructura moderada pero clara; los segmentos son
  separables. La `k` está justificada por los datos, no elegida a mano.
- **Lift** de los combos entre 5 y 9: se compran juntos varias veces más de lo
  esperado por azar, señal robusta (con soporte suficiente, no coincidencia).

## Decisión de negocio

Por cada cluster se eligen los combos de mayor lift y se propone un precio con
descuento configurable. El lift esperado prioriza qué combos activar por segmento.

## Cómo ejecutarlo

```bash
uv run kedro run --pipeline caso_b
# o:  from tostao_ml.cases.caso_b import run_case_b
```

Archivos clave: `cases/caso_b.py` (perfiles, clustering, reglas, grafo, combos),
`pipelines/caso_b/`.

## Contexto de ejecución y relación con el proyecto

**Entorno.** El mismo entorno del proyecto (Python 3.13 + `uv`) **más el extra
`caso_b`** para las reglas de asociación y el grafo:
`uv sync --extra caso_b` (instala `mlxtend`, `efficient-apriori`, `networkx`). Ver
[requisitos y stack](../README.md#requisitos-mínimos-y-entorno).

**Cómo encaja en el flujo (resumen).**

- **Registro.** `pipeline_registry.py` registra este pipeline como `caso_b` y lo
  añade al `__default__`.
- **Pipeline.** `pipelines/caso_b/` encadena `build_master_b` (cruce + matriz de
  cestas) y el nodo de segmentación/combos; entradas y salidas son nombres del
  catálogo (`conf/base/catalog*.yml`).
- **Lógica.** Los nodos delegan en `cases/masters.py` y en
  `cases/caso_b.py::run_case_b`, que **reutiliza el framework** (`models` para
  K-Means, `viz`, `narrate`) y añade lo específico del caso (FP-Growth, grafo). Al
  ser no supervisado no hay artefacto de modelo persistido: la salida son las reglas
  y los combos.
- **Reportes y notebooks.** Completo en `cases/reporting.py`; ejecutivo en
  `cases/executive.py::executive_b`; exploración en `notebooks/caso_b/`.

**Relación con los otros casos.** Comparte framework y patrón master → modelo →
reporte con los demás. Ver [Caso A](caso_a.md) · [Caso C](caso_c.md) y el
[README general](../README.md).
