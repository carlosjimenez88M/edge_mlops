"""Componente: data_preprocessing (feature engineering + split + persistencia)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import click
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import load_config
from src.common.logging_utils import get_logger, success
from src.common.mlflow_utils import setup_mlflow
from src.common.paths import PROJECT_ROOT
from src.data_preprocessing.transforms import engineer_features

logger = get_logger("data_preprocessing")


@click.command()
@click.option("--config", "config_path", default=str(PROJECT_ROOT / "config.yaml"))
def main(config_path: str) -> None:
    config = load_config(config_path)
    setup_mlflow(config)

    raw_path = PROJECT_ROOT / config.data.raw_path
    if not raw_path.exists():
        logger.error("No existe %s. Corre data_load primero.", raw_path)
        raise SystemExit(1)

    df = pd.read_parquet(raw_path)

    with mlflow.start_run(run_name="data_preprocessing"):
        if config.preprocessing.add_engineered_features:
            df = engineer_features(df)
            logger.info("Features de ingenieria agregadas. Columnas: %d", df.shape[1])

        target = config.data.target
        # En clasificacion estratificamos por la clase para conservar proporciones.
        stratify = df[target] if config.project.task == "classification" else None
        train_df, test_df = train_test_split(
            df,
            test_size=config.data.test_size,
            random_state=config.project.random_state,
            stratify=stratify,
        )

        processed_dir = PROJECT_ROOT / config.preprocessing.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        train_path = processed_dir / "train.parquet"
        test_path = processed_dir / "test.parquet"
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

        feature_names = [c for c in df.columns if c != target]
        (processed_dir / "feature_names.json").write_text(
            json.dumps(feature_names, indent=2), encoding="utf-8"
        )

        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_test", len(test_df))
        mlflow.log_artifact(str(train_path), artifact_path="processed")
        mlflow.log_artifact(str(test_path), artifact_path="processed")

        success(
            logger,
            f"Splits guardados: {len(train_df)} train / {len(test_df)} test "
            f"con {len(feature_names)} features.",
        )


if __name__ == "__main__":
    main()
