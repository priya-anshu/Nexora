from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import load_model


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")
tf.keras.utils.set_random_seed(42)


class ChurnNeuralNetwork:
    def __init__(self, input_dim: int, build: bool = True) -> None:
        self.input_dim = input_dim
        self.model = self.build_model() if build else None
        self.history: Dict[str, list[float]] = {}

    def build_model(self) -> Sequential:
        model = Sequential(
            [
                Input(shape=(self.input_dim,)),
                Dense(64, activation="relu"),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 16,
        validation_split: float = 0.2,
    ) -> Dict[str, list[float]]:
        if self.model is None:
            self.model = self.build_model()

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weight_map = {
            int(label): float(weight)
            for label, weight in zip(np.unique(y_train), class_weights)
        }
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            class_weight=class_weight_map,
            verbose=0,
        )
        self.history = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        return self.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("The churn neural network must be loaded or trained first.")
        probabilities = self.model.predict(X, verbose=0).reshape(-1)
        return probabilities.astype(float)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        probabilities = self.predict_proba(X_test)
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(
                float(precision_score(y_test, predictions, zero_division=0)),
                4,
            ),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        }
        return metrics

    def architecture_summary(self) -> str:
        if self.model is None:
            return "Model not built."
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda line: buffer.write(f"{line}\n"))
        return buffer.getvalue().strip()

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Cannot save a churn model before it is trained or loaded.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        keras_path = save_dir / "model.keras"
        metadata_path = save_dir / "metadata.json"

        self.model.save(keras_path)
        if hasattr(self.model, "export"):
            try:
                self.model.export(save_dir / "saved_model")
            except Exception:
                # The native Keras format is the reliable loading target for this MVP.
                pass

        metadata_path.write_text(
            json.dumps(
                {
                    "input_dim": self.input_dim,
                    "history": self.history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return save_dir

    @classmethod
    def load(cls, path: str | Path) -> "ChurnNeuralNetwork":
        model_dir = Path(path)
        keras_path = model_dir / "model.keras" if model_dir.is_dir() else model_dir
        metadata_path = model_dir / "metadata.json"

        if not keras_path.exists():
            raise FileNotFoundError(f"Churn model not found at {keras_path}")

        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        model = cls(input_dim=int(metadata.get("input_dim", 1)), build=False)
        model.model = load_model(keras_path)
        model.input_dim = int(model.model.input_shape[-1])
        model.history = metadata.get("history", {})
        return model
