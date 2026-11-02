"""Public request schemas for the PredictGuard API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: int | None = Field(default=None, ge=1, le=100)
    cycle: int | None = Field(default=None, ge=1)
    window: list[dict[str, float]] | None = Field(default=None, min_length=1, max_length=30)

    @model_validator(mode="after")
    def check_source(self) -> "SourceInput":
        if (self.unit_id is None) == (self.window is None):
            raise ValueError("Indica unit_id o window, exactamente uno.")
        if self.unit_id is None and self.cycle is not None:
            raise ValueError("cycle requiere unit_id.")
        return self


class PredictionRequest(SourceInput):
    model: Literal["mlp", "lstm"]


class CompareRequest(SourceInput):
    pass


class CostInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intervention_q: float = Field(default=10000, ge=0, le=1e9)
    failure_q: float = Field(default=50000, ge=0, le=1e9)
    unused_cycle_q: float = Field(default=100, ge=0, le=1e9)
    mlp_inference_q: float = Field(default=0.01, ge=0, le=1e6)
    lstm_inference_q: float = Field(default=0.05, ge=0, le=1e6)


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calendar_age: int = Field(default=125, ge=1, le=1000)
    mlp_threshold: int = Field(default=15, ge=1, le=1000)
    lstm_threshold: int = Field(default=15, ge=1, le=1000)
    lead_time_cycles: int = Field(default=5, ge=1, le=100)
    costs: CostInput = Field(default_factory=CostInput)
