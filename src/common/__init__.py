"""Utilidades compartidas por todos los componentes de la plataforma."""

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger, success
from src.common.paths import (
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    ensure_dirs,
)

__all__ = [
    "Config",
    "load_config",
    "get_logger",
    "success",
    "ensure_dirs",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "ARTIFACTS_DIR",
]
