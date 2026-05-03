from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src import DATA_DIR, PROCESSED_DATA_DIR


LOGGER = logging.getLogger(__name__)
RAW_SALES_PATH = DATA_DIR / "supermarket_sales.csv"
PROCESSED_REVENUE_PATH = PROCESSED_DATA_DIR / "daily_revenue.csv"


def _parse_sales_dates(date_series: pd.Series) -> pd.Series:
    """Accept either the prompt's MM/DD/YYYY format or ISO dates from the file."""
    parsed = pd.to_datetime(date_series, format="%m/%d/%Y", errors="coerce")
    missing_mask = parsed.isna()
    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(
            date_series.loc[missing_mask],
            errors="coerce",
        )
    if parsed.isna().any():
        invalid_values = date_series.loc[parsed.isna()].astype(str).unique().tolist()
        raise ValueError(f"Unable to parse sales dates: {invalid_values}")
    return parsed


def load_sales_dataframe(path: str | Path | None = None) -> pd.DataFrame:
    sales_path = Path(path) if path is not None else RAW_SALES_PATH
    if not sales_path.exists():
        raise FileNotFoundError(f"Sales dataset not found at {sales_path}")
    return pd.read_csv(sales_path)


def preprocess_sales_data(
    path: str | Path | None = None,
    save_processed: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    sales_df = load_sales_dataframe(path)

    required_columns = {"Date", "Total"}
    missing_columns = required_columns.difference(sales_df.columns)
    if missing_columns:
        raise ValueError(f"Sales dataset is missing columns: {sorted(missing_columns)}")

    sales_df = sales_df.copy()
    sales_df["Date"] = _parse_sales_dates(sales_df["Date"])
    sales_df["Total"] = pd.to_numeric(sales_df["Total"], errors="coerce")
    if sales_df["Total"].isna().any():
        raise ValueError("Sales dataset contains non-numeric values in the Total column.")

    daily_revenue = (
        sales_df.groupby("Date", as_index=True)["Total"]
        .sum()
        .sort_index()
        .astype(float)
    )

    full_date_index = pd.date_range(
        start=daily_revenue.index.min(),
        end=daily_revenue.index.max(),
        freq="D",
    )
    daily_revenue = daily_revenue.reindex(full_date_index).ffill().bfill()
    daily_revenue.index.name = "date"
    daily_revenue.name = "revenue"

    features_df = pd.DataFrame(index=daily_revenue.index)
    features_df["revenue"] = daily_revenue
    features_df["day_of_week"] = features_df.index.dayofweek
    features_df["month"] = features_df.index.month
    features_df["is_weekend"] = (features_df.index.dayofweek >= 5).astype(int)

    if save_processed:
        PROCESSED_REVENUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        daily_revenue.to_frame().reset_index().to_csv(PROCESSED_REVENUE_PATH, index=False)
        LOGGER.info("Saved processed daily revenue to %s", PROCESSED_REVENUE_PATH)

    return daily_revenue, features_df


def load_processed_revenue_series(path: str | Path | None = None) -> pd.Series:
    processed_path = Path(path) if path is not None else PROCESSED_REVENUE_PATH
    if not processed_path.exists():
        return preprocess_sales_data(save_processed=True)[0]

    revenue_df = pd.read_csv(processed_path)
    if "date" not in revenue_df.columns or "revenue" not in revenue_df.columns:
        raise ValueError(f"Processed revenue file is malformed: {processed_path}")

    revenue_df["date"] = pd.to_datetime(revenue_df["date"], errors="raise")
    revenue_series = revenue_df.set_index("date")["revenue"].astype(float)
    revenue_series.index = pd.DatetimeIndex(revenue_series.index, freq="D")
    revenue_series.name = "revenue"
    return revenue_series
