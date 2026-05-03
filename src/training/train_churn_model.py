from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.models.churn_model import ChurnNeuralNetwork
from sklearn.metrics import classification_report, confusion_matrix

from src import PROCESSED_DATA_DIR
from src.data_processing.churn_preprocessor import preprocess_churn_data


CHURN_MODEL_DIR = PROCESSED_DATA_DIR / "churn_nn_model"
CHURN_METRICS_PATH = PROCESSED_DATA_DIR / "churn_metrics.json"


def train_churn_model(epochs: int = 50, batch_size: int = 16) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = preprocess_churn_data(save_artifacts=True)

    classifier = ChurnNeuralNetwork(input_dim=X_train.shape[1])
    history = classifier.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
    )
    metrics = classifier.evaluate(X_test, y_test)
    predictions = classifier.predict(X_test)

    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    classifier.save(CHURN_MODEL_DIR)

    history_summary = {
        "epochs_ran": len(history.get("loss", [])),
        "best_val_loss": round(float(min(history.get("val_loss", [0.0]))), 4),
        "final_train_loss": round(float(history.get("loss", [0.0])[-1]), 4),
        "final_val_auc": round(float(history.get("val_auc", [0.0])[-1]), 4),
    }
    payload = {
        "metrics": metrics,
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "classification_report": report,
        "history_summary": history_summary,
        "architecture_summary": classifier.architecture_summary(),
        "input_dim": int(X_train.shape[1]),
    }
    CHURN_METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Churn training history summary")
    print(json.dumps(history_summary, indent=2))
    print("Churn evaluation metrics")
    print(json.dumps(metrics, indent=2))
    print("Classification report")
    print(classification_report(y_test, predictions, zero_division=0))
    return payload


if __name__ == "__main__":
    train_churn_model()
