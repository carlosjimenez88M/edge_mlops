"""Esquema de configuracion validado con Pydantic.

`config.yaml` es la unica fuente de verdad de la corrida. Se carga y valida
aqui; si algo falta o tiene el tipo equivocado, falla temprano y en rojo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common.paths import CONFIG_PATH


class ProjectConfig(BaseModel):
    name: str
    chapter: str
    experiment_name: str
    task: Literal["regression", "classification"]
    random_state: int = 42


class MLflowConfig(BaseModel):
    tracking_uri: str = "file:./mlruns"
    registered_model_name: str


class DataConfig(BaseModel):
    source: Literal["sklearn_california_housing"]
    target: str
    test_size: float = Field(0.2, gt=0, lt=1)
    raw_path: str


class ValidationConfig(BaseModel):
    drift_threshold: float = Field(0.05, gt=0, lt=1)
    reference_fraction: float = Field(0.5, gt=0, lt=1)
    report_path: str


class PreprocessingConfig(BaseModel):
    numeric_strategy: Literal["median", "mean"] = "median"
    add_engineered_features: bool = True
    processed_dir: str


class CompetitionConfig(BaseModel):
    metric: Literal["rmse", "mae", "r2"] = "rmse"
    cv_folds: int = Field(5, ge=2)
    models: list[str]

    @field_validator("models")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("competition.models no puede estar vacio")
        return v


class SweepMetricConfig(BaseModel):
    name: str
    goal: Literal["minimize", "maximize"]


class SweepConfig(BaseModel):
    enabled: bool = False
    project: str
    entity: str | None = None
    count: int = Field(60, ge=1)
    method: Literal["bayes", "random", "grid"] = "bayes"
    metric: SweepMetricConfig


class RegisterConfig(BaseModel):
    promote: bool = True
    stage: Literal["Staging", "Production", "None"] = "Staging"
    min_r2: float = 0.6


class OrchestratorConfig(BaseModel):
    env_manager: Literal["local", "virtualenv", "uv"] = "local"
    steps: list[str]


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_uri: str


class Config(BaseModel):
    """Configuracion raiz de una corrida completa."""

    # populate_by_name permite usar config.registry; la clave YAML sigue siendo 'register'.
    model_config = ConfigDict(populate_by_name=True)

    project: ProjectConfig
    mlflow: MLflowConfig
    data: DataConfig
    validation: ValidationConfig
    preprocessing: PreprocessingConfig
    competition: CompetitionConfig
    sweep: SweepConfig
    registry: RegisterConfig = Field(alias="register")
    orchestrator: OrchestratorConfig
    api: ApiConfig


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    """Lee y valida `config.yaml`. Lanza si la ruta no existe o es invalida."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de configuracion: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
