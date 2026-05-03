import numpy as np

from src.inference.churn_predictor import ChurnPredictor


SAMPLE_RECORD = {
    "tenure": 21,
    "monthly_charges": 113.0,
    "total_charges": 1753.0,
    "contract": "Month-to-month",
    "payment_method": "Electronic Check",
    "paperless_billing": "Yes",
    "senior_citizen": 1,
}

BATCH_RECORDS = [
    SAMPLE_RECORD,
    {
        "tenure": 6,
        "monthly_charges": 64.0,
        "total_charges": 1540.0,
        "contract": "One year",
        "payment_method": "Credit Card",
        "paperless_billing": "No",
        "senior_citizen": 1,
    },
]


def test_predict_proba_in_range():
    predictor = ChurnPredictor()
    probabilities = predictor.predict_proba([SAMPLE_RECORD])
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_predict_returns_binary():
    predictor = ChurnPredictor()
    predictions = predictor.predict([SAMPLE_RECORD])
    assert set(predictions.tolist()).issubset({0, 1})


def test_batch_predict_length():
    predictor = ChurnPredictor()
    batch_output = predictor.batch_predict(BATCH_RECORDS)
    assert len(batch_output) == len(BATCH_RECORDS)
