"""Componente: data_load.

Descarga el dataset crudo, lo guarda como artefacto parquet y registra
metadatos en MLflow. Punto de entrada del MLproject.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite `from src.common ...` aunque MLflow ejecute desde este directorio.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import click
import mlflow

from src.common.config import load_config
from src.common.logging_utils import get_logger, success
from src.common.mlflow_utils import setup_mlflow
from src.common.paths import PROJECT_ROOT
from src.data_load.loader import load_dataset, save_raw

logger = get_logger("data_load")


@click.command()
@click.option("--config", "config_path", default=str(PROJECT_ROOT / "config.yaml"))
def main(config_path: str) -> None:
    config = load_config(config_path)
    setup_mlflow(config)

    logger.info("Descargando dataset '%s'...", config.data.source)
    with mlflow.start_run(run_name="data_load"):
        df = load_dataset(
            config.data.source,
            config.data.target,
            sample_size=config.data.sample_size,
            random_state=config.project.random_state,
        )

        if df.isna().any().any():
            logger.warning("El dataset crudo contiene valores nulos.")

        raw_path = PROJECT_ROOT / config.data.raw_path
        save_raw(df, raw_path)

        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_cols", df.shape[1])
        mlflow.log_param("target", config.data.target)
        mlflow.log_artifact(str(raw_path), artifact_path="raw_data")

        success(logger, f"Datos guardados en {raw_path} ({len(df)} filas, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
