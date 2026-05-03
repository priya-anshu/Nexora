from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import DATA_DIR, PROCESSED_DATA_DIR


RAW_CHURN_PATH = DATA_DIR / "customer_churn.csv"
SCALER_PATH = PROCESSED_DATA_DIR / "churn_scaler.pkl"
ENCODER_PATH = PROCESSED_DATA_DIR / "payment_encoder.pkl"
FEATURE_METADATA_PATH = PROCESSED_DATA_DIR / "churn_feature_metadata.json"

NUMERIC_COLUMNS = ["Tenure", "MonthlyCharges", "TotalCharges"]
CONTRACT_COLUMN = "Contract"
PAYMENT_COLUMN = "PaymentMethod"
PAPERLESS_COLUMN = "PaperlessBilling"
SENIOR_COLUMN = "SeniorCitizen"
TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["CustomerID"]

CONTRACT_MAPPING = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}
PAPERLESS_MAPPING = {
    "No": 0,
    "Yes": 1,
}


def load_churn_dataframe(path: str | Path | None = None) -> pd.DataFrame:
    churn_path = Path(path) if path is not None else RAW_CHURN_PATH
    if not churn_path.exists():
        raise FileNotFoundError(f"Churn dataset not found at {churn_path}")
    return pd.read_csv(churn_path)


def _clean_churn_dataframe(churn_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        *NUMERIC_COLUMNS,
        CONTRACT_COLUMN,
        PAYMENT_COLUMN,
        PAPERLESS_COLUMN,
        SENIOR_COLUMN,
        TARGET_COLUMN,
        *DROP_COLUMNS,
    }
    missing_columns = required_columns.difference(churn_df.columns)
    if missing_columns:
        raise ValueError(f"Churn dataset is missing columns: {sorted(missing_columns)}")

    cleaned_df = churn_df.drop(columns=DROP_COLUMNS).copy()
    cleaned_df[NUMERIC_COLUMNS] = cleaned_df[NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    cleaned_df[SENIOR_COLUMN] = pd.to_numeric(cleaned_df[SENIOR_COLUMN], errors="coerce")
    cleaned_df[TARGET_COLUMN] = pd.to_numeric(cleaned_df[TARGET_COLUMN], errors="coerce")

    cleaned_df[CONTRACT_COLUMN] = cleaned_df[CONTRACT_COLUMN].astype(str).str.strip()
    cleaned_df[PAYMENT_COLUMN] = cleaned_df[PAYMENT_COLUMN].astype(str).str.strip()
    cleaned_df[PAPERLESS_COLUMN] = cleaned_df[PAPERLESS_COLUMN].astype(str).str.strip()

    cleaned_df["contract_encoded"] = cleaned_df[CONTRACT_COLUMN].map(CONTRACT_MAPPING)
    cleaned_df["paperless_encoded"] = cleaned_df[PAPERLESS_COLUMN].map(PAPERLESS_MAPPING)

    if cleaned_df["contract_encoded"].isna().any():
        invalid_contracts = cleaned_df.loc[
            cleaned_df["contract_encoded"].isna(),
            CONTRACT_COLUMN,
        ].unique()
        raise ValueError(f"Unknown contract values: {invalid_contracts.tolist()}")

    if cleaned_df["paperless_encoded"].isna().any():
        invalid_paperless = cleaned_df.loc[
            cleaned_df["paperless_encoded"].isna(),
            PAPERLESS_COLUMN,
        ].unique()
        raise ValueError(f"Unknown paperless billing values: {invalid_paperless.tolist()}")

    if cleaned_df.isna().any().any():
        missing_columns = cleaned_df.columns[cleaned_df.isna().any()].tolist()
        raise ValueError(f"Churn dataset contains nulls after cleaning in: {missing_columns}")

    return cleaned_df


def transform_churn_features(
    features_df: pd.DataFrame,
    scaler: StandardScaler,
    encoder: OneHotEncoder,
) -> Tuple[np.ndarray, List[str]]:
    transformed_df = features_df.copy()
    transformed_df[NUMERIC_COLUMNS] = transformed_df[NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    transformed_df[SENIOR_COLUMN] = pd.to_numeric(
        transformed_df[SENIOR_COLUMN],
        errors="coerce",
    )

    transformed_df[CONTRACT_COLUMN] = transformed_df[CONTRACT_COLUMN].astype(str).str.strip()
    transformed_df[PAYMENT_COLUMN] = transformed_df[PAYMENT_COLUMN].astype(str).str.strip()
    transformed_df[PAPERLESS_COLUMN] = transformed_df[PAPERLESS_COLUMN].astype(str).str.strip()
    transformed_df["contract_encoded"] = transformed_df[CONTRACT_COLUMN].map(CONTRACT_MAPPING)
    transformed_df["paperless_encoded"] = transformed_df[PAPERLESS_COLUMN].map(PAPERLESS_MAPPING)

    if transformed_df["contract_encoded"].isna().any():
        invalid_contracts = transformed_df.loc[
            transformed_df["contract_encoded"].isna(),
            CONTRACT_COLUMN,
        ].unique()
        raise ValueError(f"Unknown contract values for inference: {invalid_contracts.tolist()}")

    if transformed_df["paperless_encoded"].isna().any():
        invalid_paperless = transformed_df.loc[
            transformed_df["paperless_encoded"].isna(),
            PAPERLESS_COLUMN,
        ].unique()
        raise ValueError(
            f"Unknown paperless billing values for inference: {invalid_paperless.tolist()}"
        )

    scaled_numeric = scaler.transform(transformed_df[NUMERIC_COLUMNS])
    encoded_payment = encoder.transform(transformed_df[[PAYMENT_COLUMN]])
    feature_columns = [
        *NUMERIC_COLUMNS,
        "contract_encoded",
        *encoder.get_feature_names_out([PAYMENT_COLUMN]).tolist(),
        "paperless_encoded",
        SENIOR_COLUMN,
    ]
    combined_features = np.hstack(
        [
            scaled_numeric,
            transformed_df[["contract_encoded"]].to_numpy(dtype=float),
            encoded_payment.astype(float),
            transformed_df[["paperless_encoded", SENIOR_COLUMN]].to_numpy(dtype=float),
        ]
    )

    if np.isnan(combined_features).any():
        raise ValueError("Transformed churn feature matrix contains NaN values.")

    return combined_features, feature_columns


def preprocess_churn_data(
    path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    churn_df = load_churn_dataframe(path)
    cleaned_df = _clean_churn_dataframe(churn_df)

    feature_df = cleaned_df.drop(columns=[TARGET_COLUMN])
    target = cleaned_df[TARGET_COLUMN].astype(int).to_numpy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        feature_df,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    scaler.fit(X_train_df[NUMERIC_COLUMNS])
    encoder.fit(X_train_df[[PAYMENT_COLUMN]])

    X_train, feature_columns = transform_churn_features(X_train_df, scaler, encoder)
    X_test, _ = transform_churn_features(X_test_df, scaler, encoder)

    if save_artifacts:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        FEATURE_METADATA_PATH.write_text(
            json.dumps(
                {
                    "numeric_columns": NUMERIC_COLUMNS,
                    "feature_columns": feature_columns,
                    "contract_mapping": CONTRACT_MAPPING,
                    "paperless_mapping": PAPERLESS_MAPPING,
                    "payment_categories": encoder.categories_[0].tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return X_train, X_test, y_train.astype(int), y_test.astype(int)


def load_preprocessing_artifacts(
    scaler_path: str | Path | None = None,
    encoder_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> Tuple[StandardScaler, OneHotEncoder, Dict[str, object]]:
    resolved_scaler_path = Path(scaler_path) if scaler_path is not None else SCALER_PATH
    resolved_encoder_path = Path(encoder_path) if encoder_path is not None else ENCODER_PATH
    resolved_metadata_path = (
        Path(metadata_path) if metadata_path is not None else FEATURE_METADATA_PATH
    )

    if not resolved_scaler_path.exists() or not resolved_encoder_path.exists():
        preprocess_churn_data(save_artifacts=True)

    scaler = joblib.load(resolved_scaler_path)
    encoder = joblib.load(resolved_encoder_path)
    metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    return scaler, encoder, metadata


def build_inference_frame(records: Iterable[Dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    rename_map = {
        "tenure": "Tenure",
        "monthly_charges": "MonthlyCharges",
        "total_charges": "TotalCharges",
        "contract": "Contract",
        "payment_method": "PaymentMethod",
        "paperless_billing": "PaperlessBilling",
        "senior_citizen": "SeniorCitizen",
    }
    return frame.rename(columns=rename_map)
