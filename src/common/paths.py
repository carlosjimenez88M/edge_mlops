"""Rutas centralizadas del proyecto.

Todo se resuelve relativo a la raiz del repo para que los componentes
funcionen igual cuando MLflow los ejecuta desde su propio directorio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# src/common/paths.py -> raiz del repo son 2 niveles arriba.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
CONFIG_PATH: Final[Path] = PROJECT_ROOT / "config.yaml"
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"


def ensure_dirs() -> None:
    """Crea los directorios de trabajo si no existen."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
