from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    steps: int = Field(default=30, ge=1, le=90, description="Days to forecast")


class ForecastResponse(BaseModel):
    forecast_dates: List[str]
    forecast_values: List[float]
    model_metrics: Dict[str, float]


class ChurnRequest(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    payment_method: str
    paperless_billing: str
    senior_citizen: int


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    error_rate: float
    requests_by_endpoint: Dict[str, int]
