"""Pruebas de la deteccion de drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_validation.drift import compute_drift


def test_no_drift_on_homogeneous_data() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(size=1000), "target": rng.normal(size=1000)})
    report = compute_drift(df, target="target", reference_fraction=0.5, threshold=0.05)
    assert report["n_drifted"] == 0


def test_drift_detected_on_shifted_data() -> None:
    first = np.zeros(500)
    second = np.ones(500) * 50.0
    df = pd.DataFrame({"f1": np.concatenate([first, second]), "target": np.arange(1000.0)})
    report = compute_drift(df, target="target", reference_fraction=0.5, threshold=0.05)
    assert "f1" in report["drifted_features"]
