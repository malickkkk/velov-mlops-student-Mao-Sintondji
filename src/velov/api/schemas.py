"""Contrat d'entrée / sortie de l'API (validé par Pydantic, publié dans OpenAPI)."""

from __future__ import annotations

from datetime import UTC

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # un champ inconnu est une erreur, pas un champ ignoré
        json_schema_extra={
            "examples": [
                {
                    "station_id": 3,
                    "timestamp": "2026-10-06T08:00:00+02:00",
                    "capacity": 20,
                    "bikes_available": 12,
                    "temperature": 14.5,
                    "is_raining": False,
                }
            ]
        },
    )

    station_id: int = Field(..., ge=1, description="Identifiant de la station")
    # AwareDatetime et non datetime : datetime accepte aussi "2026-10-06T08:00:00", sans fuseau,
    # donc ambigu. Ici, un timestamp sans fuseau est refusé (422).
    timestamp: AwareDatetime = Field(..., description="Instant de l'observation, avec fuseau (ISO 8601)")
    capacity: int = Field(..., gt=0, le=100, description="Nombre total de bornes")
    bikes_available: int = Field(..., ge=0, description="Vélos disponibles à l'instant t")
    temperature: float = Field(..., ge=-30, le=50, description="Température en °C")
    is_raining: bool

    @field_validator("timestamp")
    @classmethod
    def to_utc(cls, value: AwareDatetime) -> AwareDatetime:
        return value.astimezone(UTC)  # convention du système : tout instant est normalisé en UTC

    @model_validator(mode="after")
    def bikes_within_capacity(self) -> PredictionRequest:
        if self.bikes_available > self.capacity:
            raise ValueError("bikes_available ne peut pas dépasser capacity")
        return self


class PredictionResponse(BaseModel):
    station_id: int
    target_timestamp: AwareDatetime = Field(..., description="Instant prédit (t + 1 h), en UTC")
    predicted_bikes: float = Field(..., ge=0)
    model_version: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]