"""Utilidades de Weights & Biases compartidas por los componentes.

Centraliza la obtención de la API key (desde .env en local o desde la variable
de entorno en CI) y la limpia, porque W&B rechaza claves con espacios o saltos
de línea (algo habitual al pegarlas en un secreto de GitHub).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.common.paths import PROJECT_ROOT


def get_wandb_api_key() -> str | None:
    """Devuelve la API key de W&B ya limpia, o None si no hay.

    Orden de búsqueda: .env (local) -> variable de entorno (CI, minúscula o
    mayúscula). Si la encuentra, la deja normalizada en WANDB_API_KEY.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    raw = os.getenv("wandb_api_key") or os.getenv("WANDB_API_KEY") or ""
    key = raw.strip()  # W&B falla si la clave tiene espacios/saltos de línea
    if not key:
        return None
    os.environ["WANDB_API_KEY"] = key
    return key
