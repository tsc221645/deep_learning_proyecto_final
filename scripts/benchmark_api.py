"""Measure batch-one API latency on the current host; no retraining."""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

from api.main import create_app


def main() -> None:
    logging.getLogger("predictguard.api").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    root = Path(__file__).resolve().parents[1]
    with TestClient(create_app(root)) as client:
        for _ in range(10):
            assert client.post("/compare", json={"unit_id": 1}).status_code == 200
        durations = []
        for _ in range(100):
            started = time.perf_counter()
            response = client.post("/compare", json={"unit_id": 1})
            assert response.status_code == 200
            durations.append((time.perf_counter() - started) * 1000)
        try:
            import psutil
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        except ImportError:
            rss_mb = None
        summary = {
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "method": "in-process FastAPI TestClient POST /compare, batch one",
            "warmup_requests": 10, "measurement_requests": 100,
            "p50_ms": float(np.percentile(durations, 50)),
            "p95_ms": float(np.percentile(durations, 95)),
            "throughput_requests_per_second": 100 / (sum(durations) / 1000),
            "max_ms": max(durations),
            "process_rss_mb": rss_mb,
            "device": str(client.app.state.runtime.device),
            "cpu": platform.processor(), "os": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "checkpoint_bytes": {name: (root / item["path"]).stat().st_size for name, item in
                                 client.app.state.runtime.selected["files"].items() if name in ("mlp", "lstm")},
        }
    output = root / "artifacts/metrics/api_performance.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
