"""Pruebas del catalogo de modelos y metricas de la competencia."""

from __future__ import annotations

import numpy as np
import pytest

from src.model_competition.models import (
    MODEL_REGISTRY,
    build_estimator,
    is_better,
    regression_metrics,
)


def test_registry_has_expected_models() -> None:
    for name in ("linear_regression", "random_forest", "gradient_boosting", "decision_tree"):
        assert name in MODEL_REGISTRY
        assert build_estimator(name, 42) is not None


def test_build_estimator_unknown_raises() -> None:
    with pytest.raises(KeyError):
        build_estimator("xgboost_super", 42)


def test_regression_metrics() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y_true, y_true)
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_is_better_direction() -> None:
    assert is_better(0.5, 1.0, "rmse")        # menos es mejor
    assert not is_better(1.0, 0.5, "rmse")
    assert is_better(0.9, 0.5, "r2")          # mas es mejor
