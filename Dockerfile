# Imagen de la API FastAPI. Multi-arch: funciona en arm64 (Raspberry Pi) y amd64.
FROM python:3.11-slim-bookworm

# uv para instalar dependencias rapido.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Solo las dependencias de la API (capa cacheable). Instala el componente api
# como paquete para resolver sus dependencias declaradas en su pyproject.toml.
COPY src/api/pyproject.toml /app/src/api/pyproject.toml
RUN uv pip install --system \
        fastapi "uvicorn[standard]" prometheus-client pandas numpy scikit-learn \
        joblib pyarrow mlflow pydantic pydantic-settings pyyaml python-dotenv click

# Codigo y configuracion.
COPY src /app/src
COPY config.yaml /app/config.yaml

EXPOSE 8000

# Sirve la app. Los artefactos/mlflow.db se montan como volumen (ver docker-compose).
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
