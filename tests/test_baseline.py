import json

import numpy as np

from predictguard.baseline import run_median_baseline


def test_median_baseline_reports_validation_without_loading_test(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "preprocessing_manifest.json").write_text(
        json.dumps(
            {
                "features": {"ordered_columns": ["cycle", "sensor_2"]},
                "windows": {"size": 30},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        processed_dir / "train.npz",
        X=np.zeros((4, 2, 2), dtype=np.float32),
        X_flat=np.zeros((4, 4), dtype=np.float32),
        y=np.array([0, 10, 20, 30], dtype=np.float32),
        unit_ids=np.array([1, 1, 2, 2]),
        end_cycles=np.array([2, 3, 2, 3]),
        lengths=np.full(4, 2),
    )
    np.savez_compressed(
        processed_dir / "validation.npz",
        X=np.zeros((3, 2, 2), dtype=np.float32),
        X_flat=np.zeros((3, 4), dtype=np.float32),
        y=np.array([0, 10, 100], dtype=np.float32),
        unit_ids=np.array([3, 3, 4]),
        end_cycles=np.array([2, 3, 2]),
        lengths=np.full(3, 2),
    )

    metrics = run_median_baseline(
        processed_dir,
        tmp_path / "metrics",
        tmp_path / "experiments" / "results.csv",
    )

    assert metrics["baseline_prediction_cycles"] == 15
    assert metrics["validation_metrics"]["mae_cycles"] == 35
    assert metrics["evaluation"]["official_test_loaded"] is False
    assert not (processed_dir / "test.npz").exists()
    predictions = (tmp_path / "metrics" / "baseline_validation_predictions.csv")
    assert predictions.is_file()
    results_path = tmp_path / "experiments" / "results.csv"
    assert results_path.is_file()

    run_median_baseline(
        processed_dir,
        tmp_path / "metrics",
        results_path,
    )
    assert results_path.read_text(encoding="utf-8").count("baseline-median-train-rul") == 1
