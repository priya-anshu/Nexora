from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.models.churn_model import ChurnNeuralNetwork
from src.data_processing.churn_preprocessor import (
    FEATURE_METADATA_PATH,
    build_inference_frame,
    load_preprocessing_artifacts,
    transform_churn_features,
)
from src.training.train_churn_model import CHURN_METRICS_PATH, CHURN_MODEL_DIR


class ChurnPredictor:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        metrics_path: str | Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else CHURN_MODEL_DIR
        self.metrics_path = Path(metrics_path) if metrics_path is not None else CHURN_METRICS_PATH
        self.model = ChurnNeuralNetwork.load(self.model_dir)
        self.scaler, self.encoder, self.feature_metadata = load_preprocessing_artifacts(
            metadata_path=FEATURE_METADATA_PATH
        )
        self.training_metrics = self._load_metrics()

    def _load_metrics(self) -> Dict[str, object]:
        if not self.metrics_path.exists():
            return {}
        return json.loads(self.metrics_path.read_text(encoding="utf-8"))

    @staticmethod
    def _canonicalize_value(value: object, allowed_values: Iterable[str]) -> str:
        allowed_map = {str(item).strip().lower(): str(item) for item in allowed_values}
        normalized = str(value).strip().lower()
        if normalized not in allowed_map:
            raise ValueError(
                f"Unsupported categorical value '{value}'. Allowed values: {list(allowed_map.values())}"
            )
        return allowed_map[normalized]

    def _normalize_records(self, records: List[Dict[str, object]]) -> List[Dict[str, object]]:
        normalized_records: List[Dict[str, object]] = []
        allowed_contracts = self.feature_metadata.get("contract_mapping", {}).keys()
        allowed_paperless = self.feature_metadata.get("paperless_mapping", {}).keys()
        allowed_payments = self.feature_metadata.get("payment_categories", [])

        for record in records:
            normalized_records.append(
                {
                    "tenure": int(record["tenure"]),
                    "monthly_charges": float(record["monthly_charges"]),
                    "total_charges": float(record["total_charges"]),
                    "contract": self._canonicalize_value(record["contract"], allowed_contracts),
                    "payment_method": self._canonicalize_value(
                        record["payment_method"],
                        allowed_payments,
                    ),
                    "paperless_billing": self._canonicalize_value(
                        record["paperless_billing"],
                        allowed_paperless,
                    ),
                    "senior_citizen": int(record["senior_citizen"]),
                }
            )
        return normalized_records

    def preprocess(self, records: List[Dict[str, object]]) -> np.ndarray:
        normalized_records = self._normalize_records(records)
        inference_frame = build_inference_frame(normalized_records)
        transformed_features, _ = transform_churn_features(
            inference_frame,
            self.scaler,
            self.encoder,
        )
        return transformed_features

    def predict_proba(self, records: List[Dict[str, object]]) -> np.ndarray:
        features = self.preprocess(records)
        return self.model.predict_proba(features)

    def predict(self, records: List[Dict[str, object]], threshold: float = 0.5) -> np.ndarray:
        features = self.preprocess(records)
        return self.model.predict(features, threshold=threshold)

    @staticmethod
    def risk_level(probability: float) -> str:
        if probability < 0.33:
            return "Low"
        if probability < 0.66:
            return "Medium"
        return "High"

    def predict_one(self, record: Dict[str, object]) -> Dict[str, object]:
        probability = float(self.predict_proba([record])[0])
        prediction = int(probability >= 0.5)
        return {
            "churn_probability": round(probability, 4),
            "churn_prediction": prediction,
            "risk_level": self.risk_level(probability),
        }

    def batch_predict(self, records: List[Dict[str, object]]) -> List[Dict[str, object]]:
        probabilities = self.predict_proba(records)
        predictions = (probabilities >= 0.5).astype(int)
        return [
            {
                "churn_probability": round(float(probability), 4),
                "churn_prediction": int(prediction),
                "risk_level": self.risk_level(float(probability)),
            }
            for probability, prediction in zip(probabilities, predictions)
        ]

    def get_model_info(self) -> Dict[str, object]:
        metrics = self.training_metrics.get("metrics", {})
        return {
            "model_name": "Dense Neural Network",
            "architecture_summary": self.training_metrics.get(
                "architecture_summary",
                self.model.architecture_summary(),
            ),
            "input_dim": self.training_metrics.get("input_dim", self.model.input_dim),
            "metrics": metrics,
            "history_summary": self.training_metrics.get("history_summary", {}),
            "confusion_matrix": self.training_metrics.get("confusion_matrix", {}),
        }
