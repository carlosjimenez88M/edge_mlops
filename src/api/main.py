"""Punto de entrada de la API (uvicorn). Lee host/puerto de config.yaml."""

from __future__ import annotations


import uvicorn

from src.common.config import load_config


def main() -> None:
    config = load_config()
    uvicorn.run("src.api.app:app", host=config.api.host, port=config.api.port, reload=False)


if __name__ == "__main__":
    main()
