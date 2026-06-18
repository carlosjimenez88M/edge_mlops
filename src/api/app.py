"""API FastAPI para servir el modelo + metricas Prometheus."""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.api.model_loader import ModelHandle, load_model
from src.api.schemas import HealthResponse, HousingFeatures, Prediction
from src.common.config import load_config
from src.common.logging_utils import get_logger
from src.data_preprocessing.transforms import engineer_features

logger = get_logger("api")

# --- Metricas Prometheus ----------------------------------------------------
PREDICTIONS = Counter("edge_predictions_total", "Total de predicciones servidas")
PRED_ERRORS = Counter("edge_prediction_errors_total", "Total de predicciones fallidas")
LATENCY = Histogram("edge_prediction_latency_seconds", "Latencia de prediccion (s)")
LAST_VALUE = Gauge("edge_last_prediction_value", "Ultimo valor predicho")

_state: dict[str, ModelHandle] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    _state["handle"] = load_model(
        config.api.model_uri, config.mlflow.tracking_uri, config.mlflow.registered_model_name
    )
    yield
    _state.clear()


app = FastAPI(title="edge_mlops - California Housing API", version="0.2.0", lifespan=lifespan)


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
def predict(features: HousingFeatures) -> Prediction:
    handle = _state.get("handle")
    if handle is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    start = time.perf_counter()
    try:
        row = engineer_features(pd.DataFrame([features.model_dump()]))
        value = float(handle.model.predict(row)[0])
    except Exception as exc:  # noqa: BLE001
        PRED_ERRORS.inc()
        logger.error("Fallo la prediccion: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        LATENCY.observe(time.perf_counter() - start)

    PREDICTIONS.inc()
    LAST_VALUE.set(value)
    return Prediction(
        predicted_median_house_value=value,
        model_stage=handle.stage,
        model_name=handle.name,
    )
