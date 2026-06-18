"""Transformaciones de datos: ingenieria de features y preprocesador sklearn.

`build_preprocessor` se importa tambien desde model_competition y sweep para
que el escalado/imputacion vivan DENTRO del Pipeline y se ajusten en cada
fold de validacion cruzada (sin fuga de datos).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features derivadas tipicas del capitulo 2 de Geron."""
    out = df.copy()
    if {"AveBedrms", "AveRooms"}.issubset(out.columns):
        out["bedrooms_per_room"] = out["AveBedrms"] / out["AveRooms"]
    if {"AveRooms", "AveOccup"}.issubset(out.columns):
        out["rooms_per_person"] = out["AveRooms"] / out["AveOccup"]
    if {"Population", "AveOccup"}.issubset(out.columns):
        out["households"] = out["Population"] / out["AveOccup"]
    return out.replace([float("inf"), float("-inf")], pd.NA)


def build_preprocessor(numeric_features: list[str], strategy: str = "median") -> ColumnTransformer:
    """Construye un preprocesador (imputacion + escalado) para columnas numericas."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_features)],
        remainder="drop",
    )
