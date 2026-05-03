from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src import PROCESSED_DATA_DIR
from src.data_processing.sales_preprocessor import load_processed_revenue_series
from src.models.time_series_model import ARIMAForecaster
from src.training.train_forecaster import FORECAST_METADATA_PATH, FORECAST_MODEL_PATH


class ForecastPredictor:
    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else FORECAST_MODEL_PATH
        self.metadata_path = (
            Path(metadata_path) if metadata_path is not None else FORECAST_METADATA_PATH
        )
        self.model = ARIMAForecaster.load(self.model_path)
        self.revenue_series = load_processed_revenue_series()
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, object]:
        if not self.metadata_path.exists():
            return {
                "order": list(self.model.order),
                "last_train_date": self.model.train_end_date.date().isoformat()
                if self.model.train_end_date is not None
                else None,
                "metrics": self.model.last_metrics,
            }
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def predict(self, steps: int = 30) -> pd.Series:
        forecast = self.model.predict(steps=steps).round(2)
        return forecast

    def get_history(self, days: int = 60) -> pd.Series:
        return self.revenue_series.tail(days).round(2)

    def get_model_info(self) -> Dict[str, object]:
        return {
            "model_name": "ARIMA",
            "order": self.metadata.get("order", list(self.model.order)),
            "last_train_date": self.metadata.get("last_train_date"),
            "metrics": self.metadata.get("metrics", self.model.last_metrics),
            "adf_p_value": self.metadata.get("adf_p_value"),
            "stationary": self.metadata.get("stationary"),
        }
