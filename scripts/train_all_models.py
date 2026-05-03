from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_churn_model import train_churn_model
from src.training.train_forecaster import train_forecaster


LOGGER = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        forecast_summary = train_forecaster()
        churn_summary = train_churn_model()
        print("Training pipeline completed successfully.")
        print(
            f"Forecast metrics: {forecast_summary['metrics']} | "
            f"Churn metrics: {churn_summary['metrics']}"
        )
        return 0
    except Exception:
        LOGGER.exception("The training pipeline failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
