"""Funciones de carga de datos (cap. 2: California Housing | cap. 3: MNIST)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml


def load_california_housing(target_name: str) -> pd.DataFrame:
    """Descarga California Housing como un unico DataFrame (features + target)."""
    bunch = fetch_california_housing(as_frame=True)
    df: pd.DataFrame = bunch.frame.copy()
    if target_name != "MedHouseVal" and "MedHouseVal" in df.columns:
        df = df.rename(columns={"MedHouseVal": target_name})
    return df


def load_mnist(
    target_name: str, sample_size: int | None = None, random_state: int = 42
) -> pd.DataFrame:
    """Descarga MNIST (784 features) como DataFrame. Submuestrea si se pide.

    `sample_size` es clave en la Raspberry Pi: 70k imagenes son demasiado para
    entrenar varios modelos con validacion cruzada en CPU de borde.
    """
    bunch = fetch_openml("mnist_784", version=1, as_frame=True, parser="auto")
    df: pd.DataFrame = bunch.frame.copy()
    if target_name != "class" and "class" in df.columns:
        df = df.rename(columns={"class": target_name})
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df


def load_dataset(
    source: str, target_name: str, sample_size: int | None = None, random_state: int = 42
) -> pd.DataFrame:
    """Despacha al cargador segun la fuente declarada en config."""
    if source == "sklearn_california_housing":
        return load_california_housing(target_name)
    if source == "openml_mnist":
        return load_mnist(target_name, sample_size=sample_size, random_state=random_state)
    raise ValueError(f"Fuente de datos no soportada: {source}")


def save_raw(df: pd.DataFrame, path: Path | str) -> Path:
    """Guarda el DataFrame crudo en parquet y devuelve la ruta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
