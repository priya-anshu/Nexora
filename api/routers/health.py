from __future__ import annotations

import time

from fastapi import APIRouter, Request

from api.schemas import HealthResponse, MetricsResponse


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    start_time = getattr(request.app.state, "start_time", time.time())
    models_loaded = getattr(
        request.app.state,
        "models_loaded",
        {"forecast": False, "churn": False},
    )
    status = "ok" if all(models_loaded.values()) else "degraded"
    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        uptime_seconds=round(time.time() - start_time, 4),
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    metrics_logger = request.app.state.metrics_logger
    return MetricsResponse(**metrics_logger.get_metrics())
