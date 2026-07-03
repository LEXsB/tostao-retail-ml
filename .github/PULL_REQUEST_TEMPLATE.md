## Qué y por qué

<!-- Descripción breve del cambio y su motivación de negocio/ingeniería. -->

## Tipo de cambio

- [ ] feat — nueva funcionalidad
- [ ] fix — corrección
- [ ] refactor — mejora interna sin cambio de comportamiento
- [ ] docs / test / chore

## Checklist

- [ ] `ruff` (check + format) + `mypy` pasan localmente (`uv run pre-commit run --all-files`).
- [ ] Pruebas añadidas/actualizadas y `uv run pytest` en verde.
- [ ] Pipelines de humo ejecutan (`uv run kedro run --pipeline <...>`).
- [ ] Sin lógica duplicada: lo reutilizable subió al `framework/`.

## Notas de reutilización

<!-- ¿Qué utilidad nueva quedó en framework/ para que la reutilicen otros casos? -->
