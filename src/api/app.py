"""API FastAPI para servir el modelo + metricas Prometheus (cap. 3: MNIST)."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any


import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.api.model_loader import load_model
from src.api.schemas import DigitImage, HealthResponse, Prediction
from src.common.config import load_config
from src.common.logging_utils import get_logger
from src.common.paths import PROJECT_ROOT

logger = get_logger("api")

# --- Metricas Prometheus ----------------------------------------------------
PREDICTIONS = Counter("edge_predictions_total", "Total de predicciones servidas")
PRED_ERRORS = Counter("edge_prediction_errors_total", "Total de predicciones fallidas")
LATENCY = Histogram("edge_prediction_latency_seconds", "Latencia de prediccion (s)")
LAST_VALUE = Gauge("edge_last_prediction_value", "Ultimo valor predicho (digito)")

_state: dict[str, Any] = {}


def _load_feature_names(config) -> list[str]:
    path = PROJECT_ROOT / config.preprocessing.processed_dir / "feature_names.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return [f"pixel{i + 1}" for i in range(784)]  # fallback al esquema MNIST de openml


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    _state["handle"] = load_model(
        config.api.model_uri, config.mlflow.tracking_uri, config.mlflow.registered_model_name
    )
    _state["feature_names"] = _load_feature_names(config)
    yield
    _state.clear()


app = FastAPI(title="edge_mlops - MNIST Classifier API", version="0.3.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    handle = _state.get("handle")
    return HealthResponse(
        status="ok" if handle else "degraded",
        model_loaded=handle is not None,
        model_name=handle.name if handle else "none",
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction)
def predict(image: DigitImage) -> Prediction:
    handle = _state.get("handle")
    if handle is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    feature_names: list[str] = _state["feature_names"]
    if len(image.pixels) != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Se esperaban {len(feature_names)} pixeles, llegaron {len(image.pixels)}.",
        )

    start = time.perf_counter()
    try:
        row = pd.DataFrame([dict(zip(feature_names, image.pixels))])
        label = str(handle.model.predict(row)[0])
    except Exception as exc:  # noqa: BLE001
        PRED_ERRORS.inc()
        logger.error("Fallo la prediccion: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        LATENCY.observe(time.perf_counter() - start)

    PREDICTIONS.inc()
    try:
        LAST_VALUE.set(float(label))
    except ValueError:
        pass
    return Prediction(predicted_label=label, model_stage=handle.stage, model_name=handle.name)


if __name__ == "__main__":
    pass
