"""Componente: sweep (Weights & Biases).

Lee el modelo ganador de la competencia y lanza un sweep de exactamente
`sweep.count` intentos (60) para encontrar los mejores hiperparametros.
Reentrena con la mejor combinacion, lo registra en MLflow y actualiza
best_model.json para que el componente `register` use el modelo afinado.

Se controla con `sweep.enabled` en config.yaml: si esta en false, el paso
se omite limpiamente y la pipeline continua con el ganador de la competencia.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import click
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger, success
from src.common.mlflow_utils import setup_mlflow
from src.common.paths import ARTIFACTS_DIR, PROJECT_ROOT
from src.data_preprocessing.transforms import build_preprocessor
from src.model_competition.models import build_estimator, compute_metrics
from src.sweep.search_space import build_sweep_config

logger = get_logger("sweep")


def _load_train_val(config: Config):
    processed_dir = PROJECT_ROOT / config.preprocessing.processed_dir
    train = pd.read_parquet(processed_dir / "train.parquet")
    target = config.data.target
    x = train.drop(columns=[target])
    y = train[target]
    return train_test_split(x, y, test_size=0.2, random_state=config.project.random_state)


@click.command()
@click.option("--config", "config_path", default=str(PROJECT_ROOT / "config.yaml"))
def main(config_path: str) -> None:
    config = load_config(config_path)

    forced = os.getenv("EDGE_FORCE_SWEEP") == "1"
    if not config.sweep.enabled and not forced:
        logger.warning("sweep.enabled=false -> se omite el sweep. Se usa el ganador de la competencia.")
        return

    best_info_path = ARTIFACTS_DIR / "best_model.json"
    if not best_info_path.exists():
        logger.error("No existe best_model.json. Corre model_competition primero.")
        raise SystemExit(1)
    best_info = json.loads(best_info_path.read_text(encoding="utf-8"))
    model_name: str = best_info["model_name"]

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("wandb_api_key") or os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.error("No se encontro wandb_api_key en .env. Imposible correr el sweep.")
        raise SystemExit(1)
    os.environ["WANDB_API_KEY"] = api_key

    import wandb

    if model_name == "linear_regression":
        logger.warning("El ganador (linear_regression) no tiene hiperparametros que tunear. Se omite el sweep.")
        return

    x_train, x_val, y_train, y_val = _load_train_val(config)
    numeric_features = list(x_train.select_dtypes("number").columns)
    task = config.project.task
    base_metric = config.competition.metric  # metrica que se optimiza en el sweep

    def train_one() -> None:
        with wandb.init() as run:
            params = {k: (None if v == "None" else v) for k, v in dict(run.config).items()}
            estimator = build_estimator(model_name, task, config.project.random_state)
            estimator.set_params(**params)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(numeric_features, config.preprocessing.numeric_strategy)),
                    ("model", estimator),
                ]
            )
            pipeline.fit(x_train, y_train)
            score = compute_metrics(task, y_val, pipeline.predict(x_val))[base_metric]
            wandb.log({config.sweep.metric.name: score})

    sweep_cfg = build_sweep_config(
        model_name, config.sweep.method, config.sweep.metric.name, config.sweep.metric.goal
    )
    sweep_id = wandb.sweep(sweep_cfg, project=config.sweep.project, entity=config.sweep.entity)
    logger.info("Sweep %s creado. Lanzando %d intentos sobre '%s'...", sweep_id, config.sweep.count, model_name)
    wandb.agent(sweep_id, function=train_one, count=config.sweep.count)

    # Recupera la mejor corrida del sweep.
    api = wandb.Api()
    entity = config.sweep.entity or api.default_entity
    sweep = api.sweep(f"{entity}/{config.sweep.project}/{sweep_id}")
    best_run = sweep.best_run()
    best_params = {k: v for k, v in best_run.config.items()}
    logger.info("Mejores hiperparametros: %s", best_params)

    # Reentrena en TODO el train con los mejores parametros y registra en MLflow.
    setup_mlflow(config)
    full = pd.read_parquet(PROJECT_ROOT / config.preprocessing.processed_dir / "train.parquet")
    test = pd.read_parquet(PROJECT_ROOT / config.preprocessing.processed_dir / "test.parquet")
    target = config.data.target
    x_full, y_full = full.drop(columns=[target]), full[target]
    x_test, y_test = test.drop(columns=[target]), test[target]

    estimator = build_estimator(model_name, task, config.project.random_state)
    estimator.set_params(**best_params)
    tuned = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, config.preprocessing.numeric_strategy)),
            ("model", estimator),
        ]
    )
    tuned.fit(x_full, y_full)
    tuned_metrics = compute_metrics(task, y_test, tuned.predict(x_test))

    with mlflow.start_run(run_name="sweep_best") as run:
        mlflow.log_params(best_params)
        for k, v in tuned_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.sklearn.log_model(tuned, name="tuned_model", serialization_format="cloudpickle")
        tuned_path = ARTIFACTS_DIR / "tuned_model.joblib"
        joblib.dump(tuned, tuned_path)

        best_info.update(
            {
                "model_name": model_name,
                "tuned": True,
                "best_params": best_params,
                "test_metrics": {k: round(v, 5) for k, v in tuned_metrics.items()},
                "run_id": run.info.run_id,
                "model_uri": f"runs:/{run.info.run_id}/tuned_model",
                "joblib_path": str(tuned_path),
            }
        )
        best_info_path.write_text(json.dumps(best_info, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(best_info_path))

    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in tuned_metrics.items())
    success(logger, f"Sweep terminado. Modelo afinado: {metrics_str}")


if __name__ == "__main__":
    main()
