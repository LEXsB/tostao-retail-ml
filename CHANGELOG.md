# Changelog

Todas las modificaciones notables de este proyecto se documentan aquí.
El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y el versionado semántico ([SemVer](https://semver.org/lang/es/)).

## [Sin publicar]

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
