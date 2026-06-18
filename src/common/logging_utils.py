"""Logging a color para toda la plataforma.

Convencion de colores pedida:
    * VERDE   -> todo bien (INFO / exito)
    * AMARILLO -> advertencia (WARNING)
    * ROJO    -> algo fallo (ERROR / CRITICAL)

No usa dependencias externas: solo codigos ANSI. Si la salida no es una
terminal (p. ej. logs de GitHub Actions redirigidos a archivo) se desactiva
el color automaticamente para no ensuciar los registros.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

# --- Codigos ANSI -----------------------------------------------------------
_RESET: Final = "\033[0m"
_BOLD: Final = "\033[1m"
_COLORS: Final[dict[int, str]] = {
    logging.DEBUG: "\033[36m",      # cian
    logging.INFO: "\033[32m",       # verde  -> todo bien
    logging.WARNING: "\033[33m",    # amarillo -> warning
    logging.ERROR: "\033[31m",      # rojo   -> fallo
    logging.CRITICAL: "\033[41m",   # fondo rojo
}

_FORMAT: Final = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT: Final = "%H:%M:%S"


class ColorFormatter(logging.Formatter):
    """Formatter que pinta el nivel y el mensaje segun la severidad."""

    def __init__(self, *, use_color: bool) -> None:
        super().__init__(fmt=_FORMAT, datefmt=_DATEFMT)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.use_color:
            return text
        color = _COLORS.get(record.levelno, "")
        return f"{color}{text}{_RESET}"


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Devuelve un logger configurado con colores (idempotente)."""
    logger = logging.getLogger(name)
    if getattr(logger, "_edge_configured", False):
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler(stream=sys.stdout)
    use_color = sys.stdout.isatty()
    handler.setFormatter(ColorFormatter(use_color=use_color))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    logger._edge_configured = True  # type: ignore[attr-defined]
    return logger


def success(logger: logging.Logger, message: str) -> None:
    """Log explicito de exito (verde + check). Azucar sobre INFO."""
    logger.info("%s %s", _BOLD + "✓" + _RESET if sys.stdout.isatty() else "[OK]", message)
