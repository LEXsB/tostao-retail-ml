# Data Card — Fuentes y tablas maestras

Describo las fuentes crudas provistas y cómo las cruzo en una **tabla maestra por
caso**. La cobertura de cada cruce se calcula en el pipeline (integridad
referencial) y se reporta como % de filas del hecho con match.

## Origen y gobierno

- **Ruta canónica:** `data/01_raw/` (SOLO LECTURA). Nunca se sobreescribe; toda
  transformación se escribe aguas abajo (`02_intermediate` … `08_reporting`).
- **Respaldo inmutable:** `data.zip`.
- **Versionado:** los datos de entrada se versionan (Git para el crudo pequeño;
  DVC/Kedro para artefactos derivados). Ver `dvc.yaml`.

## Caso A — `01_supply_optimization` (5 fuentes)

| Fuente | Grano | Campos clave |
|--------|-------|--------------|
| `ventas_historicas` | fecha × tienda × producto | unidades_vendidas |
| `catalogo_productos` | producto | costo_unitario, precio_venta, costo_almacenamiento_semanal |
| `maestro_tiendas` | tienda | ciudad, tamaño_m2 |
| `inventario_actual` | tienda × producto | stock_actual |
| `ground_truth_trends` | tienda × producto | trend_type |

**Master A** = ventas ⨝ catálogo ⨝ tiendas ⨝ inventario ⨝ tendencias, con
derivadas (ingreso, margen). Cobertura de cruces: 100 %.

## Caso B — `02_product_bundles` (3 fuentes)

| Fuente | Grano | Campos clave |
|--------|-------|--------------|
| `tickets` | ticket | fecha, id_tienda, id_cliente |
| `detalle_tickets` | ticket × producto | cantidad, precio_unitario |
| `catalogo_productos` | producto | categoria, subcategoria |

**Master B** = detalle ⨝ tickets ⨝ catálogo (línea de ticket enriquecida).
Cobertura: 100 %.

## Caso C — `03_aov_drivers` (4 fuentes)

| Fuente | Grano | Campos clave |
|--------|-------|--------------|
| `transacciones_resumen` | ticket | timestamp, total_venta, total_articulos |
| `clientes_loyalty` | cliente | edad, segmento, fecha_registro |
| `variables_exogenas` | fecha × tienda | clima, competitor_price_index, indice_trafico |
| `promociones_activas` | rango-fecha × tienda × producto | tipo_descuento |

**Master C** = transacciones ⨝ loyalty ⨝ exógenas ⨝ intensidad de promociones
(conteo activo por fecha-tienda). Cobertura: loyalty y exógenas 100 %,
promociones ≈ 98 %.

## Calidad y advertencias

- Los identificadores de tienda **difieren de formato entre casos**
  (`STORE_01` vs `TOSTAO_15`); los cruces se hacen dentro de cada caso, por lo
  que no afecta a las master tables.
- Validación de esquema (tipos, rangos, unicidad) con Pandera en ingesta; el
  pipeline falla rápido ante datos inválidos.
- `total_venta` (Caso C) está en una escala reducida; se modela tal cual y se
  winsoriza para el GLM.
