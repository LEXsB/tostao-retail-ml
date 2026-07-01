# tostao-retail-ml

Plataforma de ciencia de datos y machine learning de grado producción para una
cadena de retail. Un **framework reutilizable** resuelve tres casos de negocio
sobre un mismo núcleo: abastecimiento (A), combos (B) y AOV (C).

## Principios

- **Reutilización primero:** toda la lógica (carga, validación, EDA, features,
  modelos, HPO, evaluación, visualización, narración y reporte) vive en clases y
  funciones compartidas; cada caso añade el mínimo código.
- **Configuración sobre código:** parámetros, umbrales y espacios de búsqueda en
  YAML versionados.
- **Reproducibilidad total:** semillas fijas, entorno bloqueado, datos y modelos
  versionados.
- **Analítica narrada:** cada paso emite una mini-conclusión con cifras reales.

## Puesta en marcha

```bash
uv sync --extra caso_b --extra serving --group dev
uv run kedro run                 # ejecuta los tres casos
uv run python scripts/build_unified_report.py
uv run uvicorn serving.api.main:app --reload
```

Consulta [Arquitectura](arquitectura.md) y [Casos de uso](casos.md).
