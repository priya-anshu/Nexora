from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas import ForecastRequest, ForecastResponse


router = APIRouter(tags=["forecast"])


def _get_predictor(request: Request):
    predictor = getattr(request.app.state, "forecast_predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Forecast model is not loaded.")
    return predictor


@router.post("/predict", response_model=ForecastResponse)
def predict_forecast(payload: ForecastRequest, request: Request) -> ForecastResponse:
    predictor = _get_predictor(request)
    try:
        forecast = predictor.predict(steps=payload.steps)
        model_info = predictor.get_model_info()
        return ForecastResponse(
            forecast_dates=[timestamp.date().isoformat() for timestamp in forecast.index],
            forecast_values=[float(value) for value in forecast.values],
            model_metrics=model_info.get("metrics", {}),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {exc}") from exc


@router.get("/history")
def forecast_history(request: Request) -> dict[str, list]:
    predictor = _get_predictor(request)
    history = predictor.get_history(days=60)
    return {
        "history_dates": [timestamp.date().isoformat() for timestamp in history.index],
        "history_values": [float(value) for value in history.values],
    }


@router.get("/model-info")
def forecast_model_info(request: Request) -> dict[str, object]:
    predictor = _get_predictor(request)
    return predictor.get_model_info()
