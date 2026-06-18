"""Pruebas de la API con un modelo dummy (sin MLflow ni red)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.api.app import app
from src.api.model_loader import ModelHandle


class _DummyModel:
    def predict(self, df):  # noqa: ANN001
        return [4.2] * len(df)


@pytest.fixture
def client(monkeypatch):
    # Sustituye la carga real (MLflow/joblib) por un modelo dummy.
    monkeypatch.setattr(
        app_module, "load_model", lambda *a, **k: ModelHandle(_DummyModel(), "dummy", "test")
    )
    with TestClient(app) as c:
        yield c


def test_predict_and_metrics(client) -> None:
    payload = {
        "MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
        "Population": 322, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["predicted_median_house_value"] == 4.2

    assert client.get("/health").status_code == 200
    assert b"edge_predictions_total" in client.get("/metrics").content


def test_predict_validation_error(client) -> None:
    resp = client.post("/predict", json={"MedInc": "no-es-numero"})
    assert resp.status_code == 422
