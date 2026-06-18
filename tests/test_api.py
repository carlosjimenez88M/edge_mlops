"""Pruebas de la API con un modelo dummy (sin MLflow ni red)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.api.app import app
from src.api.model_loader import ModelHandle

N_PIXELS = 784


class _DummyModel:
    def predict(self, df):  # noqa: ANN001
        return ["5"] * len(df)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        app_module, "load_model", lambda *a, **k: ModelHandle(_DummyModel(), "dummy", "test")
    )
    monkeypatch.setattr(
        app_module, "_load_feature_names", lambda config: [f"pixel{i + 1}" for i in range(N_PIXELS)]
    )
    with TestClient(app) as c:
        yield c


def test_predict_and_metrics(client) -> None:
    resp = client.post("/predict", json={"pixels": [0.0] * N_PIXELS})
    assert resp.status_code == 200
    assert resp.json()["predicted_label"] == "5"

    assert client.get("/health").status_code == 200
    assert b"edge_predictions_total" in client.get("/metrics").content


def test_predict_validation_error(client) -> None:
    # Longitud incorrecta de pixeles -> 422 por el esquema Pydantic.
    resp = client.post("/predict", json={"pixels": [0.0, 1.0, 2.0]})
    assert resp.status_code == 422
