"""Pruebas del catalogo de modelos y metricas (regresion y clasificacion)."""

from __future__ import annotations

import numpy as np
import pytest

from src.model_competition.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    build_estimator,
    compute_metrics,
    is_better,
)


def test_regression_registry() -> None:
    for name in ("linear_regression", "random_forest", "gradient_boosting", "decision_tree"):
        assert name in REGRESSION_MODELS
        assert build_estimator(name, "regression", 42) is not None


def test_classification_registry() -> None:
    for name in ("sgd_classifier", "random_forest", "logistic_regression"):
        assert name in CLASSIFICATION_MODELS
        assert build_estimator(name, "classification", 42) is not None


def test_build_estimator_unknown_raises() -> None:
    with pytest.raises(KeyError):
        build_estimator("xgboost_super", "classification", 42)


def test_regression_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0])
    m = compute_metrics("regression", y, y)
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_classification_metrics() -> None:
    y = np.array(["0", "1", "2", "1"])
    m = compute_metrics("classification", y, y)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_is_better_direction() -> None:
    assert is_better(0.5, 1.0, "rmse")          # menos es mejor
    assert not is_better(1.0, 0.5, "rmse")
    assert is_better(0.95, 0.90, "accuracy")    # mas es mejor
    assert not is_better(0.80, 0.90, "accuracy")
