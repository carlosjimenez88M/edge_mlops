"""Pruebas de feature engineering y preprocesador."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_preprocessing.transforms import build_preprocessor, engineer_features


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MedInc": [8.3, 7.2],
            "HouseAge": [41.0, 21.0],
            "AveRooms": [6.98, 6.24],
            "AveBedrms": [1.02, 0.97],
            "Population": [322.0, 2401.0],
            "AveOccup": [2.55, 2.11],
            "Latitude": [37.88, 37.86],
            "Longitude": [-122.23, -122.22],
            "MedHouseVal": [4.526, 3.585],
        }
    )


def test_engineer_features_adds_columns() -> None:
    out = engineer_features(_sample())
    for col in ("bedrooms_per_room", "rooms_per_person", "households"):
        assert col in out.columns


def test_build_preprocessor_transforms() -> None:
    df = engineer_features(_sample())
    features = [c for c in df.columns if c != "MedHouseVal"]
    pre = build_preprocessor(features)
    transformed = pre.fit_transform(df[features])
    assert transformed.shape[0] == 2
    assert not np.isnan(transformed).any()
