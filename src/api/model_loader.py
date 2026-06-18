"""Carga del modelo para servir: MLflow Registry con fallback a joblib local."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.logging_utils import get_logger
from src.common.paths import ARTIFACTS_DIR

logger = get_logger("api.model_loader")


class ModelHandle:
    """Envuelve el modelo cargado y de donde viene."""

    def __init__(self, model: Any, name: str, stage: str) -> None:
        self.model = model
        self.name = name
        self.stage = stage


def load_model(model_uri: str, tracking_uri: str, model_name: str) -> ModelHandle:
    """Intenta cargar desde el Registry; si falla, usa el joblib del ultimo entrenamiento."""
    import mlflow

    from src.common.mlflow_utils import resolve_tracking_uri

    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    stage = model_uri.rsplit("/", 1)[-1] if model_uri.startswith("models:/") else "uri"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Modelo cargado desde MLflow: %s", model_uri)
        return ModelHandle(model, model_name, stage)
    except Exception as exc:  # noqa: BLE001 - fallback intencional para el demo
        logger.warning("No se pudo cargar desde MLflow (%s). Usando joblib local.", exc)

    best_info_path = ARTIFACTS_DIR / "best_model.json"
    if not best_info_path.exists():
        raise RuntimeError("No hay modelo en el Registry ni best_model.json local.")
    info = json.loads(best_info_path.read_text(encoding="utf-8"))
    import joblib

    model = joblib.load(Path(info["joblib_path"]))
    logger.info("Modelo cargado desde joblib: %s", info["joblib_path"])
    return ModelHandle(model, model_name, "local-joblib")
