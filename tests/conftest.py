from __future__ import annotations

import pytest

from src.training.train_churn_model import CHURN_MODEL_DIR, train_churn_model
from src.training.train_forecaster import FORECAST_MODEL_PATH, train_forecaster


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_models():
    if not FORECAST_MODEL_PATH.exists():
        train_forecaster()

    if not (CHURN_MODEL_DIR / "model.keras").exists():
        train_churn_model()
