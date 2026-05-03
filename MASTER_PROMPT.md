# MASTER PROMPT — Month 5 Advanced Data Science Project (MVP)
## For: OpenAI Codex / GPT-4 Codex Agent

---

## ROLE & OBJECTIVE

You are an expert ML engineer and full-stack data scientist. Your task is to build a **production-ready, containerized, end-to-end data science project** at MVP level. The project uses **two real datasets** (described below) and implements **Time Series Forecasting + Customer Churn Prediction** as the specialization domain, combining classical ML with a neural network component.

You must produce a fully working GitHub-ready repository with clean code, a FastAPI backend, a minimal web UI, Docker containerization, and monitoring hooks. Every file must be complete and runnable — no pseudocode, no placeholder TODOs.

---

## DATASETS (already provided — do not generate synthetic data)

### 1. `supermarket_sales.csv` — 2,000 rows × 14 columns
| Column | Type | Description |
|---|---|---|
| Invoice_ID | str | Unique transaction ID |
| Branch | str | A / B / C |
| City | str | Mandalay / Naypyitaw / Yangon |
| Customer_Type | str | Member / Normal |
| Gender | str | Male / Female |
| Product_Line | str | 6 categories |
| Unit_Price | float | Item price |
| Quantity | int | Units sold (1–9) |
| Tax | float | 5% tax amount |
| Total | float | Total sale value |
| Date | str | MM/DD/YYYY |
| Time | str | HH:MM |
| Payment | str | Cash / Credit card / Ewallet |
| Rating | float | Customer satisfaction (4–10) |

**Use for: Time Series Sales Forecasting** — aggregate daily `Total` revenue and forecast next 30 days.

### 2. `customer_churn.csv` — 500 rows × 9 columns
| Column | Type | Description |
|---|---|---|
| CustomerID | str | Unique customer ID |
| Tenure | int | Months as customer |
| MonthlyCharges | int | Monthly bill |
| TotalCharges | int | Cumulative charges |
| Contract | str | Month-to-month / One year / Two year |
| PaymentMethod | str | Payment type |
| PaperlessBilling | str | Yes / No |
| SeniorCitizen | int | 0 or 1 |
| Churn | int | 0 = stayed, 1 = churned (target) — ~10.6% churn rate |

**Use for: Customer Churn Prediction** — binary classification neural network.

---

## SPECIALIZATION CHOSEN

**Time Series + Tabular Neural Network (hybrid project)**
- Model A: ARIMA + Prophet for sales revenue forecasting (supermarket_sales.csv)
- Model B: Dense Neural Network (Keras/TensorFlow) for churn classification (customer_churn.csv)

This satisfies the "Time Series Specialization" requirement while also incorporating a neural network model.

---

## COMPLETE PROJECT REQUIREMENTS

### Technical Requirements Checklist (all must be implemented)
- [ ] Complete data preprocessing pipeline for both datasets
- [ ] Time Series model (ARIMA or Prophet) for sales forecasting
- [ ] Neural network model (Keras Dense NN) for churn prediction
- [ ] Containerized deployment with Docker + docker-compose
- [ ] FastAPI REST API with all required endpoints
- [ ] Minimal web UI (HTML/JS or Streamlit) for model serving
- [ ] Model monitoring: request logging, latency tracking, error rate
- [ ] Comprehensive error handling and logging throughout
- [ ] Unit tests for preprocessing and prediction functions
- [ ] Full documentation in README.md

---

## EXACT REPOSITORY STRUCTURE TO BUILD

```
project-root/
├── README.md                          # Full setup + documentation
├── requirements.txt                   # All Python dependencies
├── Dockerfile                         # Single container for API
├── docker-compose.yml                 # Multi-service setup
├── .env.example                       # Environment variable template
│
├── data/
│   ├── supermarket_sales.csv          # Place provided dataset here
│   ├── customer_churn.csv             # Place provided dataset here
│   └── processed/                     # Auto-created at runtime
│
├── notebooks/
│   ├── 01_eda_sales.ipynb             # EDA for supermarket sales
│   └── 02_eda_churn.ipynb             # EDA for churn data
│
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── sales_preprocessor.py      # Time series preprocessing
│   │   └── churn_preprocessor.py     # Tabular preprocessing + encoding
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── time_series_model.py       # ARIMA forecasting
│   │   └── churn_model.py             # Keras Dense NN
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_forecaster.py        # Train + save ARIMA model
│   │   └── train_churn_model.py       # Train + save Keras model
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── forecast_predictor.py      # Load + run forecasting
│   │   └── churn_predictor.py         # Load + run churn prediction
│   │
│   └── monitoring/
│       ├── __init__.py
│       └── logger.py                  # Request/response logging, metrics
│
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app entrypoint
│   ├── routers/
│   │   ├── forecast.py                # /forecast endpoints
│   │   ├── churn.py                   # /churn endpoints
│   │   └── health.py                  # /health + /metrics endpoints
│   └── schemas.py                     # Pydantic request/response models
│
├── ui/
│   └── index.html                     # Simple HTML/JS dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_sales_preprocessor.py
│   ├── test_churn_preprocessor.py
│   ├── test_forecast_predictor.py
│   └── test_churn_predictor.py
│
├── scripts/
│   ├── train_all_models.py            # One-command model training
│   └── run_tests.sh                   # Run full test suite
│
└── monitoring/
    └── metrics_store.json             # Persisted metrics (append-only)
```

---

## FILE-BY-FILE IMPLEMENTATION INSTRUCTIONS

### `src/data_processing/sales_preprocessor.py`
```
- Load supermarket_sales.csv
- Parse Date column (MM/DD/YYYY) to datetime
- Aggregate Total revenue by date → daily time series
- Fill missing dates with forward fill
- Return: pd.Series with DatetimeIndex, name='revenue'
- Also return: feature-engineered DataFrame with day_of_week, month, is_weekend
- Save processed series to data/processed/daily_revenue.csv
```

### `src/data_processing/churn_preprocessor.py`
```
- Load customer_churn.csv
- Drop CustomerID (non-predictive)
- Encode: Contract (ordinal: Month-to-month=0, One year=1, Two year=2)
- Encode: PaymentMethod (one-hot or label encode)
- Encode: PaperlessBilling (Yes=1, No=0)
- Scale numeric features (Tenure, MonthlyCharges, TotalCharges) with StandardScaler
- Target: Churn column (already 0/1)
- Train/test split: 80/20 stratified on Churn
- Save scaler and encoder to data/processed/ as .pkl files
- Return: X_train, X_test, y_train, y_test
```

### `src/models/time_series_model.py`
```
- Class: ARIMAForecaster
  - __init__(order=(5,1,0))
  - fit(series: pd.Series) → saves fitted model
  - predict(steps: int = 30) → returns pd.Series of forecasted values with dates
  - evaluate(test_series) → returns dict: {MAE, RMSE, MAPE}
  - save(path) / load(path) using pickle or joblib
- Also include: check_stationarity(series) using adfuller → returns bool + p_value
```

### `src/models/churn_model.py`
```
- Class: ChurnNeuralNetwork
  - __init__(input_dim: int)
  - build_model() → Keras Sequential:
      Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.2) → Dense(1, sigmoid)
      Compile: optimizer=adam, loss=binary_crossentropy, metrics=[accuracy, AUC]
  - fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
      Use EarlyStopping(patience=10, restore_best_weights=True)
  - predict_proba(X) → np.array of probabilities
  - predict(X, threshold=0.5) → np.array of 0/1 labels
  - evaluate(X_test, y_test) → dict: {accuracy, precision, recall, f1, roc_auc}
  - save(path) / load(path) using model.save() and load_model()
```

### `src/training/train_forecaster.py`
```
- Load and preprocess sales data
- Split: last 30 days as test, rest as train
- Fit ARIMAForecaster on train
- Evaluate on test set, print metrics
- Save model to data/processed/arima_model.pkl
- Print: training summary + evaluation metrics
```

### `src/training/train_churn_model.py`
```
- Load and preprocess churn data
- Train ChurnNeuralNetwork
- Print training history summary
- Evaluate on test set, print classification report
- Save model to data/processed/churn_nn_model/ (Keras SavedModel format)
- Save: confusion matrix values, final metrics to data/processed/churn_metrics.json
```

### `api/schemas.py`
```python
# Pydantic models:

class ForecastRequest(BaseModel):
    steps: int = Field(default=30, ge=1, le=90, description="Days to forecast")

class ForecastResponse(BaseModel):
    forecast_dates: List[str]
    forecast_values: List[float]
    model_metrics: Dict[str, float]  # MAE, RMSE, MAPE from last evaluation

class ChurnRequest(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str  # "Month-to-month" | "One year" | "Two year"
    payment_method: str
    paperless_billing: str  # "Yes" | "No"
    senior_citizen: int  # 0 or 1

class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int  # 0 or 1
    risk_level: str  # "Low" | "Medium" | "High"

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float

class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    error_rate: float
    requests_by_endpoint: Dict[str, int]
```

### `api/routers/forecast.py`
```
Endpoints:
- POST /forecast/predict  → ForecastRequest → ForecastResponse
- GET  /forecast/history  → returns last 60 days of actual daily revenue
- GET  /forecast/model-info → returns model order, last train date, metrics
```

### `api/routers/churn.py`
```
Endpoints:
- POST /churn/predict       → ChurnRequest → ChurnResponse
- POST /churn/batch-predict → List[ChurnRequest] → List[ChurnResponse]
- GET  /churn/model-info    → returns architecture summary, training metrics
```

### `api/routers/health.py`
```
Endpoints:
- GET /health   → HealthResponse
- GET /metrics  → MetricsResponse
```

### `api/main.py`
```
- FastAPI app with title="DS Project API", version="1.0.0"
- Include all routers with prefixes: /forecast, /churn, /health
- On startup: load both models, initialize metrics store
- CORS middleware: allow all origins (for UI)
- Add request timing middleware that logs latency to monitoring/logger.py
- Serve ui/index.html at GET / 
```

### `src/monitoring/logger.py`
```
- Class: MetricsLogger
  - Tracks: total_requests (int), total_errors (int), latencies (list of ms), 
    requests_per_endpoint (dict)
  - Methods: log_request(endpoint, latency_ms, error=False)
  - get_metrics() → dict matching MetricsResponse schema
  - persist_to_file(path) → append JSON line to monitoring/metrics_store.json
- Use threading.Lock for thread safety
- Singleton pattern (module-level instance)
```

### `ui/index.html`
```
Single-page HTML with vanilla JS and inline CSS. Two sections:
1. Sales Forecast Panel:
   - Input: number of days to forecast (slider 1-90)
   - Button: "Generate Forecast"
   - Output: simple table of date + predicted revenue
   
2. Churn Prediction Panel:
   - Inputs: all ChurnRequest fields (dropdowns + number inputs)
   - Button: "Predict Churn Risk"
   - Output: probability bar + risk level badge (color-coded)

3. System Status Panel (auto-refreshes every 30s):
   - Shows /health and /metrics data
   
Use fetch() to call the API at http://localhost:8000
Style: dark theme, minimal, functional. No external CDN needed.
```

### `Dockerfile`
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/train_all_models.py  # Pre-train on build
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`
```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./monitoring:/app/monitoring
    environment:
      - MODEL_PATH=data/processed
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### `requirements.txt`
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
tensorflow==2.16.1
statsmodels==0.14.2
joblib==1.4.2
python-multipart==0.0.9
pytest==8.2.0
httpx==0.27.0
```

---

## TESTS — `tests/` directory

### `tests/test_sales_preprocessor.py`
```
- test_load_returns_series: assert output is pd.Series with DatetimeIndex
- test_no_missing_dates: assert no gaps in daily index after preprocessing
- test_all_values_positive: assert all revenue values > 0
```

### `tests/test_churn_preprocessor.py`
```
- test_output_shape: assert X_train has correct number of features
- test_stratified_split: assert churn rate in train ≈ churn rate in test (±5%)
- test_no_nulls: assert no NaN in output arrays
```

### `tests/test_forecast_predictor.py`
```
- test_predict_returns_correct_length: assert len(output) == steps
- test_predict_returns_positive_values: assert all forecast values > 0
- test_forecast_dates_are_future: assert all dates > last training date
```

### `tests/test_churn_predictor.py`
```
- test_predict_proba_in_range: assert 0 <= all probabilities <= 1
- test_predict_returns_binary: assert all predictions in {0, 1}
- test_batch_predict_length: assert len(batch output) == len(batch input)
```

---

## `scripts/train_all_models.py`
```python
# Entry point: trains both models sequentially
# 1. Preprocess sales data → train ARIMA → save
# 2. Preprocess churn data → train NN → save
# Print final metrics for both models
# Exit with code 0 on success, 1 on failure
```

---

## README.md MUST INCLUDE

```markdown
# Advanced Data Science Project — Time Series & Churn Prediction

## Project Overview
[Describe both models, datasets used, business problem]

## Architecture
[Describe: Data → Preprocessing → Models → FastAPI → Docker → UI]

## Setup Instructions
### Local (without Docker)
pip install -r requirements.txt
python scripts/train_all_models.py
uvicorn api.main:app --reload

### With Docker
docker-compose up --build

## API Endpoints
[Table of all endpoints with method, path, description]

## Model Details
### Time Series Model (ARIMA)
- Order: (5,1,0)
- Features: Daily aggregated revenue
- Metrics: MAE, RMSE, MAPE

### Churn Prediction Model (Neural Network)
- Architecture: Dense NN (64→32→1)
- Dataset: 500 rows, 8 features
- Target: Binary churn (10.6% positive rate)
- Metrics: Accuracy, F1, ROC-AUC

## Monitoring
[Describe what gets logged and how to view metrics]

## Business Impact
[Describe how sales forecasting aids inventory planning]
[Describe how churn prediction reduces customer loss]
```

---

## CONSTRAINTS & GUARDRAILS

1. **No external API calls** — all models run locally, no OpenAI/paid APIs
2. **No synthetic data** — use only the two provided CSVs
3. **Dataset size awareness** — churn dataset is only 500 rows; use EarlyStopping to prevent overfitting; do NOT use complex architectures
4. **Class imbalance** — churn rate is ~10.6%; use `class_weight='balanced'` in Keras or handle via sample weights
5. **ARIMA stationarity** — run ADF test first; if non-stationary, use d=1 in ARIMA order
6. **Error handling** — all API endpoints must return proper HTTP status codes (422 for validation errors, 500 for model errors, with descriptive messages)
7. **Model loading** — models must be loaded once at startup (not per request) using FastAPI lifespan events
8. **No hardcoded paths** — use `pathlib.Path` and relative paths from project root
9. **Thread safety** — MetricsLogger must use threading.Lock
10. **Docker build must succeed** — training is part of the Docker build; if training fails, the build fails

---

## EXPECTED OUTPUT METRICS (approximate targets)

| Model | Metric | Expected Range |
|---|---|---|
| ARIMA | MAE | < 50 (revenue units) |
| ARIMA | MAPE | < 20% |
| Churn NN | Accuracy | > 85% |
| Churn NN | ROC-AUC | > 0.75 |
| API | Response Time | < 300ms |
| API | /health | Always 200 OK |

---

## EXECUTION ORDER

Build files in this exact order to avoid import errors:

1. `requirements.txt`
2. `src/data_processing/sales_preprocessor.py`
3. `src/data_processing/churn_preprocessor.py`
4. `src/models/time_series_model.py`
5. `src/models/churn_model.py`
6. `src/training/train_forecaster.py`
7. `src/training/train_churn_model.py`
8. `src/inference/forecast_predictor.py`
9. `src/inference/churn_predictor.py`
10. `src/monitoring/logger.py`
11. `api/schemas.py`
12. `api/routers/health.py`
13. `api/routers/forecast.py`
14. `api/routers/churn.py`
15. `api/main.py`
16. `ui/index.html`
17. `scripts/train_all_models.py`
18. `tests/` (all test files)
19. `Dockerfile`
20. `docker-compose.yml`
21. `README.md`

---

## FINAL VALIDATION CHECKLIST

Before finishing, verify:
- [ ] `python scripts/train_all_models.py` runs without errors
- [ ] `pytest tests/ -v` — all tests pass
- [ ] `uvicorn api.main:app` starts without errors
- [ ] `curl http://localhost:8000/health` returns `{"status": "ok", ...}`
- [ ] `curl -X POST http://localhost:8000/forecast/predict -d '{"steps": 30}'` returns forecast
- [ ] `docker-compose up --build` completes successfully
- [ ] README.md contains all required sections
