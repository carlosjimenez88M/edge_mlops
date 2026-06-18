"""Logica de registro y promocion (champion-challenger) en MLflow Registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error


def rmse_of_stage_model(
    client: MlflowClient, model_name: str, stage: str, x_test: pd.DataFrame, y_test: pd.Series
) -> float | None:
    """RMSE del modelo actualmente en `stage`, o None si no hay ninguno."""
    import mlflow.sklearn

    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    return float(np.sqrt(mean_squared_error(y_test, model.predict(x_test))))


def promote_if_better(
    client: MlflowClient,
    model_name: str,
    version: str,
    stage: str,
    challenger_rmse: float,
    champion_rmse: float | None,
) -> bool:
    """Promueve el challenger al stage si mejora (o si no hay campeon)."""
    if champion_rmse is None or challenger_rmse < champion_rmse:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )
        return True
    return False
