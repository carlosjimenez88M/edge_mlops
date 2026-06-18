"""Utilidades comunes de MLflow (tracking y registro)."""

from __future__ import annotations

import mlflow

from src.common.config import Config
from src.common.logging_utils import get_logger
from src.common.paths import PROJECT_ROOT

logger = get_logger(__name__)


def resolve_tracking_uri(tracking_uri: str) -> str:
    """Convierte URIs sqlite/file RELATIVAS en absolutas respecto a la raiz.

    Necesario porque MLflow ejecuta cada componente desde su propio directorio;
    una ruta relativa apuntaria a una BD distinta por componente.
    """
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        rel = tracking_uri[len("sqlite:///"):]
        return f"sqlite:///{(PROJECT_ROOT / rel).resolve()}"
    if tracking_uri.startswith("file:") and "://" not in tracking_uri:
        rel = tracking_uri[len("file:"):]
        return f"file:{(PROJECT_ROOT / rel).resolve()}"
    return tracking_uri


def setup_mlflow(config: Config) -> None:
    """Configura el tracking URI (absoluto) y el experimento."""
    tracking_uri = resolve_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.project.experiment_name)
    logger.info(
        "MLflow tracking en %s | experimento '%s'",
        tracking_uri,
        config.project.experiment_name,
    )


def get_best_run(experiment_name: str, metric: str, ascending: bool = True):
    """Devuelve el mejor run del experimento ordenado por una metrica.

    `ascending=True` para metricas donde menos es mejor (rmse, mae).
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    return runs[0] if runs else None
