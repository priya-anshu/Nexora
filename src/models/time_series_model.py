from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> Tuple[bool, float]:
    clean_series = series.dropna().astype(float)
    statistic, p_value, *_ = adfuller(clean_series)
    return bool(p_value < alpha), float(p_value)


class ARIMAForecaster:
    def __init__(self, order: tuple[int, int, int] = (5, 1, 0)) -> None:
        self.order = order
        self.model_fit = None
        self.train_end_date: pd.Timestamp | None = None
        self.last_metrics: Dict[str, float] = {}

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("ARIMAForecaster expects a pandas Series with a DatetimeIndex.")

        training_series = series.astype(float).copy()
        if training_series.index.freq is None:
            training_series = training_series.asfreq("D")

        self.model_fit = ARIMA(training_series, order=self.order).fit()
        self.train_end_date = pd.Timestamp(training_series.index.max())
        return self

    def predict(self, steps: int = 30) -> pd.Series:
        if self.model_fit is None or self.train_end_date is None:
            raise RuntimeError("The ARIMA model must be fitted before prediction.")

        raw_forecast = self.model_fit.forecast(steps=steps)
        future_index = pd.date_range(
            start=self.train_end_date + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )
        forecast = pd.Series(np.asarray(raw_forecast), index=future_index, name="forecast")
        return forecast.clip(lower=0.01)

    def evaluate(self, test_series: pd.Series) -> Dict[str, float]:
        actual = test_series.astype(float)
        predicted = self.predict(len(actual)).reindex(actual.index, method="nearest")

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        denominator = np.where(actual == 0, np.nan, actual)
        mape = np.nanmean(np.abs((actual - predicted) / denominator)) * 100

        self.last_metrics = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "MAPE": round(float(np.nan_to_num(mape, nan=0.0)), 4),
        }
        return self.last_metrics

    def save(self, path: str | Path) -> Path:
        if self.model_fit is None:
            raise RuntimeError("Cannot save an ARIMA model that has not been fitted.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "order": self.order,
            "model_fit": self.model_fit,
            "train_end_date": self.train_end_date.isoformat() if self.train_end_date else None,
            "last_metrics": self.last_metrics,
        }
        joblib.dump(payload, save_path)
        return save_path

    @classmethod
    def load(cls, path: str | Path) -> "ARIMAForecaster":
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"ARIMA model not found at {model_path}")

        payload = joblib.load(model_path)
        instance = cls(order=tuple(payload["order"]))
        instance.model_fit = payload["model_fit"]
        instance.train_end_date = (
            pd.Timestamp(payload["train_end_date"])
            if payload.get("train_end_date")
            else None
        )
        instance.last_metrics = payload.get("last_metrics", {})
        return instance
