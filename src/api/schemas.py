"""Esquemas Pydantic de entrada/salida de la API (cap. 3: MNIST)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

N_PIXELS = 784  # MNIST: imagen 28x28 aplanada


class DigitImage(BaseModel):
    """Una imagen MNIST como vector de 784 intensidades de pixel (0-255)."""

    pixels: list[float] = Field(
        ...,
        description="Vector de 784 pixeles (28x28) con valores 0-255.",
        min_length=N_PIXELS,
        max_length=N_PIXELS,
    )

    @field_validator("pixels")
    @classmethod
    def _check_range(cls, v: list[float]) -> list[float]:
        if any(p < 0 or p > 255 for p in v):
            raise ValueError("Cada pixel debe estar en el rango [0, 255].")
        return v


class Prediction(BaseModel):
    predicted_label: str = Field(..., description="Digito predicho (0-9).")
    model_stage: str
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
