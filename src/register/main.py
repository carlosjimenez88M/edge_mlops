"""Componente: register.

Registra el mejor modelo (campeon de la competencia o afinado por el sweep)
en el MLflow Model Registry, aplica un gate de calidad (min_r2) y lo promueve
al stage configurado solo si supera al campeon actual (champion-challenger).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import click
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.common.config import load_config
from src.common.logging_utils import get_logger, success
from src.common.mlflow_utils import setup_mlflow
from src.common.paths import ARTIFACTS_DIR, PROJECT_ROOT
from src.register.registry import metric_of_stage_model, promote_if_better

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

    gate_metric = config.registry.gate_metric
    gate_min = config.registry.gate_min
    score = best_info["test_metrics"][gate_metric]
    if score < gate_min:
        logger.error(
            "Gate de calidad NO superado: %s=%.4f < %.4f. No se registra.", gate_metric, score, gate_min
        )
        raise SystemExit(1)
    logger.info("Gate de calidad superado (%s=%.4f >= %.4f).", gate_metric, score, gate_min)

    model_name = config.mlflow.registered_model_name
    result = mlflow.register_model(model_uri=best_info["model_uri"], name=model_name)
    logger.info("Registrado %s version %s", model_name, result.version)

    if not config.registry.promote or config.registry.stage == "None":
        success(logger, f"Modelo registrado como version {result.version} (sin promocion).")
        return

    # Champion-challenger sobre el test set, usando la metrica de seleccion.
    task = config.project.task
    metric = config.competition.metric
    test = pd.read_parquet(PROJECT_ROOT / config.preprocessing.processed_dir / "test.parquet")
    target = config.data.target
    x_test, y_test = test.drop(columns=[target]), test[target]
    challenger = best_info["test_metrics"][metric]

    client = MlflowClient()
    champion = metric_of_stage_model(
        client, model_name, config.registry.stage, task, metric, x_test, y_test
    )

    promoted = promote_if_better(
        client, model_name, result.version, config.registry.stage, metric, challenger, champion
    )
    if promoted:
        success(
            logger,
            f"Version {result.version} promovida a {config.registry.stage} "
            f"(challenger {metric}={challenger:.4f} vs champion={champion}).",
        )
    else:
        logger.warning(
            "El challenger (%s=%.4f) NO supera al campeon (%.4f). No se promueve.",
            metric, challenger, champion,
        )


if __name__ == "__main__":
    main()
