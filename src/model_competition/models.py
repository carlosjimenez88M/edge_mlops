"""Catalogo de modelos candidatos y utilidades de metricas para la competencia."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# Fabrica de estimadores por nombre. Cada uno recibe el random_state.
MODEL_REGISTRY: dict[str, Callable[[int], object]] = {
    "linear_regression": lambda rs: LinearRegression(),
    "decision_tree": lambda rs: DecisionTreeRegressor(random_state=rs),
    "random_forest": lambda rs: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=rs),
    "gradient_boosting": lambda rs: GradientBoostingRegressor(random_state=rs),
}

# Direccion de optimizacion: True => menos es mejor.
METRIC_LOWER_IS_BETTER: dict[str, bool] = {"rmse": True, "mae": True, "r2": False}


def build_estimator(name: str, random_state: int) -> object:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Modelo '{name}' no esta en el catalogo: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](random_state)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula rmse, mae y r2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def is_better(candidate: float, current_best: float, metric: str) -> bool:
    """Indica si `candidate` mejora a `current_best` segun la metrica."""
    if METRIC_LOWER_IS_BETTER[metric]:
        return candidate < current_best
    return candidate > current_best
