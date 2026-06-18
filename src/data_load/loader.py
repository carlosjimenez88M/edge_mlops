"""Funciones de carga de datos para el capitulo 2 (California Housing)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_california_housing(target_name: str) -> pd.DataFrame:
    """Descarga el dataset California Housing como un unico DataFrame.

    Incluye las features y la columna objetivo con el nombre indicado.
    """
    bunch = fetch_california_housing(as_frame=True)
    df: pd.DataFrame = bunch.frame.copy()
    # sklearn nombra el target 'MedHouseVal'; renombramos si el usuario pidio otro.
    if target_name != "MedHouseVal" and "MedHouseVal" in df.columns:
        df = df.rename(columns={"MedHouseVal": target_name})
    return df


def save_raw(df: pd.DataFrame, path: Path | str) -> Path:
    """Guarda el DataFrame crudo en parquet y devuelve la ruta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
