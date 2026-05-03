import pandas as pd

from src.data_processing.sales_preprocessor import preprocess_sales_data


def test_load_returns_series():
    revenue_series, features_df = preprocess_sales_data(save_processed=False)
    assert isinstance(revenue_series, pd.Series)
    assert isinstance(revenue_series.index, pd.DatetimeIndex)
    assert revenue_series.name == "revenue"
    assert {"day_of_week", "month", "is_weekend"}.issubset(features_df.columns)


def test_no_missing_dates():
    revenue_series, _ = preprocess_sales_data(save_processed=False)
    expected_index = pd.date_range(
        start=revenue_series.index.min(),
        end=revenue_series.index.max(),
        freq="D",
    )
    assert revenue_series.index.equals(expected_index)


def test_all_values_positive():
    revenue_series, _ = preprocess_sales_data(save_processed=False)
    assert (revenue_series > 0).all()
