from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api.routers import churn, forecast, health
from src import MONITORING_DIR, UI_DIR
from src.inference.churn_predictor import ChurnPredictor
from src.inference.forecast_predictor import ForecastPredictor
from src.monitoring.logger import metrics_logger
from src.training.train_churn_model import train_churn_model
from src.training.train_forecaster import train_forecaster


LOGGER = logging.getLogger(__name__)
METRICS_STORE_PATH = MONITORING_DIR / "metrics_store.json"


def _load_forecast_predictor() -> ForecastPredictor:
    try:
        return ForecastPredictor()
    except FileNotFoundError:
        LOGGER.warning("Forecast artifacts missing. Training ARIMA model at startup.")
        train_forecaster()
        return ForecastPredictor()


def _load_churn_predictor() -> ChurnPredictor:
    try:
        return ChurnPredictor()
    except FileNotFoundError:
        LOGGER.warning("Churn artifacts missing. Training neural network at startup.")
        train_churn_model()
        return ChurnPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app.state.start_time = time.time()
    app.state.metrics_logger = metrics_logger
    app.state.models_loaded = {"forecast": False, "churn": False}
    METRICS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_STORE_PATH.touch(exist_ok=True)

    try:
        app.state.forecast_predictor = _load_forecast_predictor()
        app.state.models_loaded["forecast"] = True
    except Exception as exc:
        LOGGER.exception("Unable to load forecast predictor: %s", exc)
        app.state.forecast_predictor = None

    try:
        app.state.churn_predictor = _load_churn_predictor()
        app.state.models_loaded["churn"] = True
    except Exception as exc:
        LOGGER.exception("Unable to load churn predictor: %s", exc)
        app.state.churn_predictor = None

    yield


app = FastAPI(title="DS Project API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000
        app.state.metrics_logger.log_request(
            request.url.path,
            latency_ms,
            error=response.status_code >= 400,
        )
        app.state.metrics_logger.persist_to_file(METRICS_STORE_PATH)
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
        return response
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        app.state.metrics_logger.log_request(request.url.path, latency_ms, error=True)
        app.state.metrics_logger.persist_to_file(METRICS_STORE_PATH)
        LOGGER.exception("Unhandled request error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


app.include_router(forecast.router, prefix="/forecast")
app.include_router(churn.router, prefix="/churn")
app.include_router(health.router)


@app.get("/", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")
