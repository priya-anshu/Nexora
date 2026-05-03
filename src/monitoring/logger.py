from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict


class MetricsLogger:
    def __init__(self) -> None:
        self._lock = Lock()
        self.total_requests = 0
        self.total_errors = 0
        self.latencies: list[float] = []
        self.requests_per_endpoint: Dict[str, int] = {}

    def reset(self) -> None:
        with self._lock:
            self.total_requests = 0
            self.total_errors = 0
            self.latencies = []
            self.requests_per_endpoint = {}

    def log_request(self, endpoint: str, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self.total_requests += 1
            if error:
                self.total_errors += 1
            self.latencies.append(float(latency_ms))
            self.requests_per_endpoint[endpoint] = self.requests_per_endpoint.get(endpoint, 0) + 1

    def _snapshot_unlocked(self) -> Dict[str, object]:
        avg_latency_ms = (
            sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        )
        error_rate = (
            self.total_errors / self.total_requests if self.total_requests else 0.0
        )
        return {
            "total_requests": int(self.total_requests),
            "avg_latency_ms": round(float(avg_latency_ms), 4),
            "error_rate": round(float(error_rate), 4),
            "requests_by_endpoint": dict(self.requests_per_endpoint),
        }

    def get_metrics(self) -> Dict[str, object]:
        with self._lock:
            return self._snapshot_unlocked()

    def persist_to_file(self, path: str | Path) -> Path:
        metrics_path = Path(path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **self._snapshot_unlocked(),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        return metrics_path


metrics_logger = MetricsLogger()
