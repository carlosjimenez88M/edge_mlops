"""Deteccion de data drift mediante el test de Kolmogorov-Smirnov.

Se parte el dataset en una mitad "referencia" y otra "actual" (simulando
produccion) y por cada feature numerica se compara su distribucion. Un
p-value por debajo del umbral indica que la distribucion cambio (drift).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import ks_2samp


def compute_drift(
    df: pd.DataFrame,
    target: str,
    reference_fraction: float,
    threshold: float,
) -> dict[str, Any]:
    """Calcula drift por feature comparando referencia vs actual."""
    features = df.select_dtypes("number").drop(columns=[target], errors="ignore")
    split = int(len(features) * reference_fraction)
    reference = features.iloc[:split]
    current = features.iloc[split:]

    per_feature: dict[str, dict[str, float | bool]] = {}
    drifted: list[str] = []
    for col in features.columns:
        stat, p_value = ks_2samp(reference[col].dropna(), current[col].dropna())
        is_drift = bool(p_value < threshold)
        per_feature[col] = {
            "ks_statistic": round(float(stat), 5),
            "p_value": round(float(p_value), 5),
            "drift": is_drift,
        }
        if is_drift:
            drifted.append(col)

    n_features = len(features.columns)
    return {
        "threshold": threshold,
        "n_features": n_features,
        "n_drifted": len(drifted),
        "drifted_features": drifted,
        "share_drifted": round(len(drifted) / n_features, 4) if n_features else 0.0,
        "per_feature": per_feature,
    }
