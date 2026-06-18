"""Esquemas Pydantic de entrada/salida de la API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HousingFeatures(BaseModel):
    """Features base del dataset California Housing (un distrito)."""

    MedInc: float = Field(
        ..., description="Ingreso mediano del bloque (decenas de miles USD)", examples=[8.3]
    )
    HouseAge: float = Field(..., description="Edad mediana de las casas", examples=[41.0])
    AveRooms: float = Field(..., description="Promedio de habitaciones por hogar", examples=[6.98])
    AveBedrms: float = Field(..., description="Promedio de dormitorios por hogar", examples=[1.02])
    Population: float = Field(..., description="Poblacion del bloque", examples=[322.0])
    AveOccup: float = Field(..., description="Ocupantes promedio por hogar", examples=[2.55])
    Latitude: float = Field(..., examples=[37.88])
    Longitude: float = Field(..., examples=[-122.23])


class Prediction(BaseModel):
    predicted_median_house_value: float = Field(
        ..., description="Valor mediano de vivienda predicho (en cientos de miles de USD)"
    )
    model_stage: str
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
