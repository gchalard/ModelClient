# Generic ModelRunner runtime image (no model artifacts baked in).
# Mount a model directory at runtime or build a child image that adds model files/adapters.

FROM python:3.13-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv

WORKDIR /app

# Lockfile resolves wkmeans from git — git must be installed above.
COPY pyproject.toml uv.lock README.md /app/
# Hatchling needs package sources present before editable install.
COPY src /app/src

RUN uv sync --frozen --no-dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_MANIFEST_PATH=/app/model/manifest.yaml

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "modelrunner.main:app", "--host", "0.0.0.0", "--port", "8000"]
