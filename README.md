# Advanced Data Science Project — Time Series & Churn Prediction

## Project Overview
This repository is a production-oriented MVP that combines two real business workflows in one deployable service:

- Daily supermarket revenue forecasting from `supermarket_sales.csv`
- Customer churn scoring from `customer_churn.csv`

The forecasting path aggregates transaction-level sales into a daily revenue series and trains an ARIMA model for the next 30 days and beyond. The churn path preprocesses a small imbalanced tabular dataset and trains a dense neural network with class weighting and early stopping to limit overfitting.

## Architecture
Data flows through a layered pipeline:

1. Raw CSV files in `data/`
2. Preprocessing modules in `src/data_processing/`
3. Trained models and metadata saved under `data/processed/`
4. Inference helpers in `src/inference/`
5. FastAPI endpoints in `api/`
6. Monitoring via `src/monitoring/logger.py` and `monitoring/metrics_store.json`
7. Browser UI served from `ui/index.html`
8. Containerized execution with Docker and Docker Compose

## Repository Structure
The repository follows the requested MVP layout, including:

- `src/` for preprocessing, model code, training, inference, and monitoring
- `api/` for FastAPI schemas and routers
- `ui/` for the single-page HTML dashboard
- `tests/` for unit and inference checks
- `scripts/` for one-command training and test execution
- `monitoring/` for persisted request metrics

## Setup Instructions
### Local (without Docker)
```bash
pip install -r requirements.txt
python scripts/train_all_models.py
uvicorn api.main:app --reload
```

### With Docker
```bash
docker-compose up --build
```

The API is available at `http://localhost:8000` and the dashboard is served from the root route `/`.

## API Endpoints
| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Serves the minimal web dashboard |
| `POST` | `/forecast/predict` | Forecasts future daily revenue for 1 to 90 days |
| `GET` | `/forecast/history` | Returns the last 60 days of actual daily revenue |
| `GET` | `/forecast/model-info` | Returns ARIMA configuration and evaluation metrics |
| `POST` | `/churn/predict` | Predicts churn probability and risk for one customer |
| `POST` | `/churn/batch-predict` | Predicts churn for multiple customers in one request |
| `GET` | `/churn/model-info` | Returns neural network architecture and training metrics |
| `GET` | `/health` | Reports service health and model load status |
| `GET` | `/metrics` | Reports request count, latency, error rate, and endpoint usage |

## Model Details
### Time Series Model (ARIMA)
- Primary model: ARIMA
- Default order family: `(5, d, 0)` with `d` chosen from the ADF stationarity test
- Features: Daily aggregated revenue from transaction totals
- Saved artifacts: processed daily revenue CSV, ARIMA model pickle, model metadata JSON
- Metrics: `MAE`, `RMSE`, `MAPE`

### Churn Prediction Model (Neural Network)
- Architecture: Dense neural network `64 -> 32 -> 1`
- Regularization: Dropout and early stopping
- Dataset size: 500 rows
- Target: Binary churn with approximately 10.6% positive rate
- Class imbalance handling: balanced class weights
- Saved artifacts: scaler, encoder, model directory, metrics JSON
- Metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`

## Monitoring
Each request is measured by middleware in the FastAPI app. The monitoring layer tracks:

- Total requests
- Total errors
- Average latency in milliseconds
- Endpoint-level request counts

Snapshots are appended as JSON lines to `monitoring/metrics_store.json`, which makes the file simple to inspect or tail during runtime.

## Business Impact
Revenue forecasting helps teams estimate near-term demand, align staffing, and improve inventory planning with a forward-looking daily sales view. Churn prediction helps customer success and retention teams prioritize outreach before high-risk customers leave, which can reduce preventable revenue loss.

## Testing
Run the test suite with:

```bash
pytest tests/ -v
```

Or use:

```bash
bash scripts/run_tests.sh
```

## Notes
- The sales dataset in this workspace uses ISO-style dates such as `2023-08-08`, so the sales preprocessor accepts both ISO and `MM/DD/YYYY`.
- Models are loaded once during FastAPI startup and reused across requests.
- If pre-trained artifacts are missing, the training scripts can regenerate them from the raw CSV files.
