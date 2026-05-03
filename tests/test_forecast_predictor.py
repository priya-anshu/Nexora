from src.inference.forecast_predictor import ForecastPredictor


def test_predict_returns_correct_length():
    predictor = ForecastPredictor()
    forecast = predictor.predict(steps=14)
    assert len(forecast) == 14


def test_predict_returns_positive_values():
    predictor = ForecastPredictor()
    forecast = predictor.predict(steps=14)
    assert (forecast > 0).all()


def test_forecast_dates_are_future():
    predictor = ForecastPredictor()
    forecast = predictor.predict(steps=7)
    assert (forecast.index > predictor.model.train_end_date).all()
