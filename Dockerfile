# =========================================================================== #
# Imagen base del proyecto (pipelines Kedro + batch) — multi-stage con uv
# =========================================================================== #
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra caso_b --extra serving --extra observability

# --------------------------------------------------------------------------- #
FROM python:3.13-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    KEDRO_DISABLE_TELEMETRY=true \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY conf ./conf
COPY data/01_raw ./data/01_raw
COPY scripts ./scripts
COPY pyproject.toml ./

USER appuser

# Por defecto ejecuta el flujo unificado de los tres casos.
CMD ["kedro", "run"]
