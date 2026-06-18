"""Componente: data_validation (EDA + data drift)."""

from __future__ import annotations

import json


import click
import mlflow
import pandas as pd

from src.common.config import load_config
from src.common.logging_utils import get_logger, success
from src.common.mlflow_utils import setup_mlflow
from src.common.paths import ARTIFACTS_DIR, PROJECT_ROOT
from src.data_validation.drift import compute_drift
from src.data_validation.eda import basic_profile, save_histograms

logger = get_logger("data_validation")


@click.command()
@click.option("--config", "config_path", default=str(PROJECT_ROOT / "config.yaml"))
def main(config_path: str) -> None:
    config = load_config(config_path)
    setup_mlflow(config)

    raw_path = PROJECT_ROOT / config.data.raw_path
    if not raw_path.exists():
        logger.error("No existe el dataset crudo en %s. Corre data_load primero.", raw_path)
        raise SystemExit(1)

    df = pd.read_parquet(raw_path)
    logger.info("Validando %d filas...", len(df))

    with mlflow.start_run(run_name="data_validation"):
        profile = basic_profile(df, config.data.target)
        drift = compute_drift(
            df,
            target=config.data.target,
            reference_fraction=config.validation.reference_fraction,
            threshold=config.validation.drift_threshold,
        )

        report = {"profile": profile, "drift": drift}
        report_path = PROJECT_ROOT / config.validation.report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        hist_path = save_histograms(df, ARTIFACTS_DIR / "histograms.png")

        mlflow.log_metric("n_missing", profile["n_missing"])
        mlflow.log_metric("share_drifted", drift["share_drifted"])
        mlflow.log_artifact(str(report_path), artifact_path="validation")
        mlflow.log_artifact(str(hist_path), artifact_path="validation")

        # Semaforo: verde si todo bien, amarillo si hay drift.
        if profile["n_missing"] > 0:
            logger.warning("Hay %d valores nulos en el dataset.", profile["n_missing"])
        if drift["n_drifted"] > 0:
            logger.warning(
                "Drift detectado en %d/%d features: %s",
                drift["n_drifted"],
                drift["n_features"],
                ", ".join(drift["drifted_features"]),
            )
        else:
            success(logger, "Sin drift significativo entre referencia y actual.")

        success(logger, f"Reporte de validacion guardado en {report_path}")


if __name__ == "__main__":
    main()
