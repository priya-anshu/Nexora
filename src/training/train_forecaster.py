from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src import PROCESSED_DATA_DIR
from src.data_processing.sales_preprocessor import preprocess_sales_data
from src.models.time_series_model import ARIMAForecaster, check_stationarity


FORECAST_MODEL_PATH = PROCESSED_DATA_DIR / "arima_model.pkl"
FORECAST_METADATA_PATH = PROCESSED_DATA_DIR / "forecast_model_info.json"


def train_forecaster() -> Dict[str, object]:
    revenue_series, _ = preprocess_sales_data(save_processed=True)
    if len(revenue_series) <= 30:
        raise ValueError("Sales series must contain more than 30 days to create a holdout set.")

    train_series = revenue_series.iloc[:-30]
    test_series = revenue_series.iloc[-30:]
    is_stationary, p_value = check_stationarity(train_series)
    model_order = (5, 0, 0) if is_stationary else (5, 1, 0)

    forecaster = ARIMAForecaster(order=model_order)
    forecaster.fit(train_series)
    metrics = forecaster.evaluate(test_series)
    forecaster.save(FORECAST_MODEL_PATH)

    summary = {
        "model_name": "ARIMA",
        "order": list(model_order),
        "stationary": is_stationary,
        "adf_p_value": round(float(p_value), 6),
        "train_points": int(len(train_series)),
        "test_points": int(len(test_series)),
        "last_train_date": train_series.index.max().date().isoformat(),
        "last_test_date": test_series.index.max().date().isoformat(),
        "metrics": metrics,
    }
    FORECAST_METADATA_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Forecast training summary")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_forecaster()
