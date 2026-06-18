"""Logica de registro y promocion (champion-challenger) en MLflow Registry."""

from __future__ import annotations

import pandas as pd
from mlflow.tracking import MlflowClient

from src.model_competition.models import compute_metrics, is_better


def metric_of_stage_model(
    client: MlflowClient,
    model_name: str,
    stage: str,
    task: str,
    metric: str,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> float | None:
    """Valor de `metric` del modelo actualmente en `stage`, o None si no hay."""
    import mlflow.sklearn

    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    return compute_metrics(task, y_test, model.predict(x_test))[metric]


def promote_if_better(
    client: MlflowClient,
    model_name: str,
    version: str,
    stage: str,
    metric: str,
    challenger_score: float,
    champion_score: float | None,
) -> bool:
    """Promueve el challenger al stage si mejora (o si no hay campeon)."""
    if champion_score is None or is_better(challenger_score, champion_score, metric):
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )
        return True
    return False
