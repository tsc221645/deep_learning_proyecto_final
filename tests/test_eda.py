import numpy as np
import pandas as pd

from predictguard.data.eda import run_eda


def _write_synthetic_fd001(data_dir):
    data_dir.mkdir(parents=True)
    train_rows = []
    test_rows = []
    for engine_id in range(1, 101):
        for cycle in range(1, 33):
            train_rows.append(
                [engine_id, cycle, 0, 0, 0]
                + [cycle if sensor == 2 else 1 for sensor in range(1, 22)]
            )
        for cycle in range(1, 32):
            test_rows.append(
                [engine_id, cycle, 0, 0, 0]
                + [cycle if sensor == 2 else 1 for sensor in range(1, 22)]
            )

    pd.DataFrame(train_rows).to_csv(
        data_dir / "train_FD001.txt", sep=" ", header=False, index=False
    )
    pd.DataFrame(test_rows).to_csv(
        data_dir / "test_FD001.txt", sep=" ", header=False, index=False
    )
    pd.Series(np.arange(100)).to_csv(
        data_dir / "RUL_FD001.txt", header=False, index=False
    )
    (data_dir / "readme.txt").write_text("Synthetic test fixture\n", encoding="utf-8")


def test_run_eda_creates_outputs_without_changing_raw_data(tmp_path):
    data_dir = tmp_path / "data" / "raw"
    _write_synthetic_fd001(data_dir)
    original_hashes = {
        path.name: path.read_bytes()
        for path in data_dir.iterdir()
        if path.is_file()
    }

    outputs = run_eda(
        data_dir,
        tmp_path / "artifacts" / "figures",
        tmp_path / "artifacts" / "metrics",
        tmp_path / "docs" / "eda_findings.md",
        window_size=30,
    )

    assert outputs["window_info"]["train_windows"] == 300
    assert outputs["window_info"]["train_window_endpoints_with_rul_le_30"] == 300
    assert outputs["sensor_summary"].set_index("sensor").loc["sensor_1", "constant"]
    assert (
        outputs["sensor_summary"]
        .set_index("sensor")
        .loc["sensor_2", "median_abs_unit_spearman_cycle"]
        == 1
    )
    assert outputs["sequence_diagnostics"]["train"]["non_consecutive_cycle_steps"] == 0
    assert outputs["sequence_diagnostics"]["test"]["non_increasing_cycle_steps"] == 0
    assert len(list(outputs["figure_dir"].glob("eda_*.png"))) == 7
    assert (tmp_path / "artifacts" / "metrics" / "eda_column_summary.csv").is_file()
    assert "test oficial" in outputs["report_path"].read_text(encoding="utf-8")
    assert original_hashes == {
        path.name: path.read_bytes()
        for path in data_dir.iterdir()
        if path.is_file()
    }


def test_run_eda_rejects_non_positive_window_size(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="window_size"):
        run_eda(
            tmp_path / "missing",
            tmp_path / "figures",
            tmp_path / "metrics",
            tmp_path / "report.md",
            window_size=0,
        )
