"""Catalogo de modelos y metricas para la competencia (regresion y clasificacion)."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.tree import DecisionTreeRegressor

# Fabricas de estimadores por nombre (reciben random_state).
REGRESSION_MODELS: dict[str, Callable[[int], object]] = {
    "linear_regression": lambda rs: LinearRegression(),
    "decision_tree": lambda rs: DecisionTreeRegressor(random_state=rs),
    "random_forest": lambda rs: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=rs),
    "gradient_boosting": lambda rs: GradientBoostingRegressor(random_state=rs),
}

CLASSIFICATION_MODELS: dict[str, Callable[[int], object]] = {
    "sgd_classifier": lambda rs: SGDClassifier(random_state=rs),
    "random_forest": lambda rs: RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=rs
    ),
    "logistic_regression": lambda rs: LogisticRegression(max_iter=200, n_jobs=-1),
}

# Direccion de optimizacion: True => menos es mejor.
METRIC_LOWER_IS_BETTER: dict[str, bool] = {
    "rmse": True,
    "mae": True,
    "r2": False,
    "accuracy": False,
    "f1_macro": False,
    "precision_macro": False,
    "recall_macro": False,
}

# Nombre de scoring de sklearn para validacion cruzada.
CV_SCORING: dict[str, str] = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
}


def registry_for(task: str) -> dict[str, Callable[[int], object]]:
    return REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS


def build_estimator(name: str, task: str, random_state: int) -> object:
    registry = registry_for(task)
    if name not in registry:
        raise KeyError(f"Modelo '{name}' no esta en el catalogo de {task}: {list(registry)}")
    return registry[name](random_state)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return (
        regression_metrics(y_true, y_pred)
        if task == "regression"
        else classification_metrics(y_true, y_pred)
    )


def is_better(candidate: float, current_best: float, metric: str) -> bool:
    """Indica si `candidate` mejora a `current_best` segun la metrica."""
    if METRIC_LOWER_IS_BETTER[metric]:
        return candidate < current_best
    return candidate > current_best
