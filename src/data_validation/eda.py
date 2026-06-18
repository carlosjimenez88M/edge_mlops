"""Analisis exploratorio basico (EDA) para el reporte de validacion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla (necesario en la Raspberry/CI)
import matplotlib.pyplot as plt
import pandas as pd


def basic_profile(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Devuelve un perfil resumido del dataset."""
    numeric = df.select_dtypes("number")
    corr_with_target = (
        numeric.corr()[target].drop(labels=[target]).sort_values(ascending=False).round(4)
        if target in numeric.columns
        else pd.Series(dtype=float)
    )
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_missing": int(df.isna().sum().sum()),
        "columns": list(df.columns),
        "target": target,
        "correlation_with_target": corr_with_target.to_dict(),
    }


def save_histograms(df: pd.DataFrame, out_path: Path | str, max_cols: int = 25) -> Path:
    """Guarda una grilla de histogramas de las columnas numericas.

    Para datasets de alta dimension (p. ej. MNIST con 784 pixeles) se grafican
    solo las primeras `max_cols` columnas para no generar una figura inmanejable.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes("number")
    if numeric.shape[1] > max_cols:
        numeric = numeric.iloc[:, :max_cols]
    numeric.hist(figsize=(14, 10), bins=40)
    plt.tight_layout()
    plt.savefig(out_path, dpi=90)
    plt.close("all")
    return out_path
