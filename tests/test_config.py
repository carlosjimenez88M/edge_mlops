"""Pruebas de carga y validacion de configuracion."""

from __future__ import annotations

import pytest

from src.common.config import Config, load_config
from src.common.paths import CONFIG_PATH


def test_load_default_config() -> None:
    config = load_config(CONFIG_PATH)
    assert isinstance(config, Config)
    assert config.project.task == "classification"
    assert config.sweep.count == 60
    assert config.data.source == "openml_mnist"
    assert "model_competition" in config.orchestrator.steps


def test_invalid_config_raises(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: {name: x}", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(bad)


def test_missing_config_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
