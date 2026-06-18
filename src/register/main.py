"""Componente: register.

Registra el mejor modelo (campeon de la competencia o afinado por el sweep)
en el MLflow Model Registry, aplica un gate de calidad (min_r2) y lo promueve
al stage configurado solo si supera al campeon actual (champion-challenger).
"""

from __future__ import annotations

import json


import click
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.common.config import load_config
from src.common.logging_utils import get_logger, success
from src.common.mlflow_utils import setup_mlflow
from src.common.paths import ARTIFACTS_DIR, PROJECT_ROOT
from src.register.registry import promote_if_better, rmse_of_stage_model

logger = get_logger("register")


@click.command()
@click.option("--config", "config_path", default=str(PROJECT_ROOT / "config.yaml"))
def main(config_path: str) -> None:
    config = load_config(config_path)
    setup_mlflow(config)

    best_info_path = ARTIFACTS_DIR / "best_model.json"
    if not best_info_path.exists():
        logger.error("No existe best_model.json. Corre model_competition (y/o sweep) primero.")
        raise SystemExit(1)
    best_info = json.loads(best_info_path.read_text(encoding="utf-8"))

    r2 = best_info["test_metrics"]["r2"]
    if r2 < config.registry.min_r2:
        logger.error(
            "Gate de calidad NO superado: R2=%.4f < min_r2=%.4f. No se registra.",
            r2,
            config.registry.min_r2,
        )
        raise SystemExit(1)
    logger.info("Gate de calidad superado (R2=%.4f >= %.4f).", r2, config.registry.min_r2)

    model_name = config.mlflow.registered_model_name
    result = mlflow.register_model(model_uri=best_info["model_uri"], name=model_name)
    logger.info("Registrado %s version %s", model_name, result.version)

    if not config.registry.promote or config.registry.stage == "None":
        success(logger, f"Modelo registrado como version {result.version} (sin promocion).")
        return

    # Champion-challenger sobre el test set.
    test = pd.read_parquet(PROJECT_ROOT / config.preprocessing.processed_dir / "test.parquet")
    target = config.data.target
    x_test, y_test = test.drop(columns=[target]), test[target]
    challenger_rmse = best_info["test_metrics"]["rmse"]

    client = MlflowClient()
    champion_rmse = rmse_of_stage_model(client, model_name, config.registry.stage, x_test, y_test)

    promoted = promote_if_better(
        client, model_name, result.version, config.registry.stage, challenger_rmse, champion_rmse
    )
    if promoted:
        success(
            logger,
            f"Version {result.version} promovida a {config.registry.stage} "
            f"(challenger RMSE={challenger_rmse:.4f} vs champion={champion_rmse}).",
        )
    else:
        logger.warning(
            "El challenger (RMSE=%.4f) NO supera al campeon (RMSE=%.4f). No se promueve.",
            challenger_rmse,
            champion_rmse,
        )


if __name__ == "__main__":
    main()
