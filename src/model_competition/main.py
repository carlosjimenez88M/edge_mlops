"""Componente: model_competition.

Entrena varios modelos con el mismo preprocesamiento, los compara con
validacion cruzada + holdout y selecciona el campeon. Registra cada
candidato como un run anidado en MLflow y persiste el ganador.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import click
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.common.config import load_config
from src.common.logging_utils import get_logger, success
from src.common.mlflow_utils import setup_mlflow
from src.common.paths import ARTIFACTS_DIR, PROJECT_ROOT
from src.data_preprocessing.transforms import build_preprocessor
from src.model_competition.models import (
    CV_SCORING,
    METRIC_LOWER_IS_BETTER,
    build_estimator,
    compute_metrics,
    is_better,
)

logger = get_logger("model_competition")


def _load_splits(config) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    processed_dir = PROJECT_ROOT / config.preprocessing.processed_dir
    train = pd.read_parquet(processed_dir / "train.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    target = config.data.target
    return (
        train.drop(columns=[target]),
        train[target],
        test.drop(columns=[target]),
        test[target],
    )


@click.command()
@click.option("--config", "config_path", default=str(PROJECT_ROOT / "config.yaml"))
def main(config_path: str) -> None:
    config = load_config(config_path)
    setup_mlflow(config)

    x_train, y_train, x_test, y_test = _load_splits(config)
    numeric_features = list(x_train.select_dtypes("number").columns)
    task = config.project.task
    metric = config.competition.metric
    lower_better = METRIC_LOWER_IS_BETTER[metric]
    scoring = CV_SCORING[metric]
    secondary = "r2" if task == "regression" else "f1_macro"

    logger.info(
        "Competencia de %d modelos (%s) | metrica de seleccion: %s",
        len(config.competition.models), task, metric,
    )

    best_name: str | None = None
    best_score = np.inf if lower_better else -np.inf
    leaderboard: list[dict] = []

    with mlflow.start_run(run_name="model_competition") as parent:
        for name in config.competition.models:
            with mlflow.start_run(run_name=name, nested=True):
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", build_preprocessor(numeric_features, config.preprocessing.numeric_strategy)),
                        ("model", build_estimator(name, task, config.project.random_state)),
                    ]
                )
                cv_scores = cross_val_score(
                    pipeline, x_train, y_train, cv=config.competition.cv_folds, scoring=scoring
                )
                cv_metric = float(-cv_scores.mean() if scoring.startswith("neg_") else cv_scores.mean())

                pipeline.fit(x_train, y_train)
                test_metrics = compute_metrics(task, y_test, pipeline.predict(x_test))

                mlflow.log_param("model", name)
                mlflow.log_metric(f"cv_{metric}", cv_metric)
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

                leaderboard.append({"model": name, f"cv_{metric}": round(cv_metric, 5), **{f"test_{k}": round(v, 5) for k, v in test_metrics.items()}})
                logger.info(
                    "  %-20s cv_%s=%.4f  test_%s=%.4f  test_%s=%.4f",
                    name, metric, cv_metric, metric, test_metrics[metric], secondary, test_metrics[secondary],
                )

                if best_name is None or is_better(cv_metric, best_score, metric):
                    best_score, best_name = cv_metric, name

        # Reentrena el ganador en TODO el train y lo persiste como campeon.
        winner = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, config.preprocessing.numeric_strategy)),
                ("model", build_estimator(best_name, task, config.project.random_state)),
            ]
        )
        winner.fit(x_train, y_train)
        winner_metrics = compute_metrics(task, y_test, winner.predict(x_test))

        model_path = ARTIFACTS_DIR / "winner_model.joblib"
        joblib.dump(winner, model_path)
        mlflow.sklearn.log_model(winner, name="winner_model", serialization_format="cloudpickle")

        best_info = {
            "model_name": best_name,
            "selection_metric": metric,
            f"cv_{metric}": round(best_score, 5),
            "test_metrics": {k: round(v, 5) for k, v in winner_metrics.items()},
            "run_id": parent.info.run_id,
            "model_uri": f"runs:/{parent.info.run_id}/winner_model",
            "joblib_path": str(model_path),
            "numeric_features": numeric_features,
        }
        (ARTIFACTS_DIR / "best_model.json").write_text(json.dumps(best_info, indent=2), encoding="utf-8")
        (ARTIFACTS_DIR / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(ARTIFACTS_DIR / "best_model.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "leaderboard.json"))
        mlflow.log_param("winner", best_name)

    _log_competition_to_wandb(config, task, leaderboard, best_name, winner_metrics)

    success(
        logger,
        f"Campeon: {best_name} (test {metric}={winner_metrics[metric]:.4f}, "
        f"{secondary}={winner_metrics[secondary]:.4f})",
    )


def _log_competition_to_wandb(config, task, leaderboard, best_name, winner_metrics) -> None:
    """Registra la competencia en Weights & Biases (si hay API key disponible).

    No-op silencioso si no hay clave, para que las corridas locales sin W&B
    no fallen. En CI la clave llega por el secret WANDB_API_KEY.
    """
    from src.common.wandb_utils import get_wandb_api_key

    if get_wandb_api_key() is None:
        logger.warning("Sin WANDB_API_KEY -> la competencia se registra solo en MLflow.")
        return

    import wandb

    run = wandb.init(
        project=config.sweep.project,
        entity=config.sweep.entity,
        name=f"competition-{config.project.experiment_name}",
        job_type="model_competition",
        config={"task": task, "models": config.competition.models, "metric": config.competition.metric},
        reinit=True,
    )
    columns = list(leaderboard[0].keys())
    table = wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in leaderboard])
    wandb.log({"leaderboard": table})
    run.summary["winner_model"] = best_name
    for k, v in winner_metrics.items():
        run.summary[f"test_{k}"] = v
    wandb.finish()
    logger.info("Competencia registrada en W&B (proyecto '%s').", config.sweep.project)


if __name__ == "__main__":
    main()
