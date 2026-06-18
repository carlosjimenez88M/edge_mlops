"""Espacios de busqueda de hiperparametros por familia de modelo (W&B sweep)."""

from __future__ import annotations

from typing import Any

# Espacios inspirados en el reporte de W&B de Random Forest para precios de vivienda.
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": {"distribution": "int_uniform", "min": 100, "max": 600},
        "max_depth": {"distribution": "int_uniform", "min": 4, "max": 40},
        "min_samples_split": {"distribution": "int_uniform", "min": 2, "max": 20},
        "min_samples_leaf": {"distribution": "int_uniform", "min": 1, "max": 10},
        "max_features": {"values": ["sqrt", "log2", 1.0, 0.5]},
    },
    "gradient_boosting": {
        "n_estimators": {"distribution": "int_uniform", "min": 100, "max": 600},
        "learning_rate": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.3},
        "max_depth": {"distribution": "int_uniform", "min": 2, "max": 8},
        "subsample": {"distribution": "uniform", "min": 0.6, "max": 1.0},
    },
    "decision_tree": {
        "max_depth": {"distribution": "int_uniform", "min": 3, "max": 40},
        "min_samples_split": {"distribution": "int_uniform", "min": 2, "max": 40},
        "min_samples_leaf": {"distribution": "int_uniform", "min": 1, "max": 20},
    },
    # LinearRegression no tiene hiperparametros relevantes que tunear.
    "linear_regression": {},
}


def build_sweep_config(model_name: str, method: str, metric_name: str, goal: str) -> dict[str, Any]:
    """Arma el dict de configuracion que espera `wandb.sweep`."""
    return {
        "method": method,
        "metric": {"name": metric_name, "goal": goal},
        "parameters": SEARCH_SPACES.get(model_name, {}),
    }
