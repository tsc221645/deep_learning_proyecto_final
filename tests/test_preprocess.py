import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from predictguard.data.preprocess import preprocess_fd001
from predictguard.data.validate import COLUMN_NAMES


def _write_synthetic_fd001(data_dir, validation_value=1_000_000):
    data_dir.mkdir(parents=True)
    engine_ids = np.arange(1, 101)
    _, validation_ids = train_test_split(
        engine_ids, test_size=0.2, random_state=42, shuffle=True
    )
    validation_ids = set(validation_ids.tolist())
    train_rows = []
    test_rows = []
    for engine_id in engine_ids:
        sensor_value = (
            validation_value if engine_id in validation_ids else engine_id
        )
        for cycle in range(1, 33):
            train_rows.append(
                [engine_id, cycle, 0, 0, 0, 0, sensor_value + cycle]
                + [0] * 19
            )
        for cycle in range(1, 21):
            test_rows.append(
                [engine_id, cycle, 0, 0, 0, 0, 10_000_000 + engine_id + cycle]
                + [0] * 19
            )
    pd.DataFrame(train_rows, columns=COLUMN_NAMES).to_csv(
        data_dir / "train_FD001.txt", sep=" ", header=False, index=False
    )
    pd.DataFrame(test_rows, columns=COLUMN_NAMES).to_csv(
        data_dir / "test_FD001.txt", sep=" ", header=False, index=False
    )
    pd.Series(np.arange(100)).to_csv(
        data_dir / "RUL_FD001.txt", header=False, index=False
    )
    (data_dir / "readme.txt").write_text("Synthetic test fixture\n", encoding="utf-8")
    return validation_ids


def _config():
    return {
        "seed": 42,
        "data": {"subset": "FD001"},
        "split": {
            "validation_fraction": 0.2,
            "random_state": 42,
            "group_column": "unit_id",
        },
        "window": {"size": 30},
        "features": {
            "include_cycle": True,
            "include_operating_settings": False,
            "sensors": "all",
            "exclude_unit_id": True,
        },
        "preprocessing": {
            "constant_variance_threshold": 1e-12,
            "drop_constant_sensors": True,
            "train_window_stride": 1,
        },
    }


def test_preprocess_splits_by_engine_and_fits_scaler_on_train_only(tmp_path):
    data_dir = tmp_path / "data" / "raw"
    validation_ids = _write_synthetic_fd001(data_dir)
    output_dir = tmp_path / "artifacts" / "processed" / "fd001"

    manifest = preprocess_fd001(data_dir, output_dir, _config())

    fit_ids = json.loads((output_dir / "train_unit_ids.json").read_text())
    saved_validation_ids = json.loads(
        (output_dir / "val_unit_ids.json").read_text()
    )
    assert len(fit_ids) == 80
    assert len(saved_validation_ids) == 20
    assert set(fit_ids).isdisjoint(saved_validation_ids)
    assert set(saved_validation_ids) == validation_ids
    assert set(fit_ids).union(saved_validation_ids) == set(range(1, 101))
    assert manifest["split"]["overlap"] == []

    scaler = joblib.load(output_dir / "scaler.joblib")
    assert manifest["scaler"]["fit_unit_ids"] == fit_ids
    assert "unit_id" not in manifest["features"]["ordered_columns"]
    assert manifest["features"]["ordered_columns"] == ["cycle", "sensor_2"]
    assert manifest["features"]["dropped_constant_sensors"] == [
        f"sensor_{index}" for index in range(1, 22) if index != 2
    ]
    fit_frame = pd.read_csv(
        data_dir / "train_FD001.txt",
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )
    fit_rows = fit_frame.loc[fit_frame["unit_id"].isin(fit_ids)]
    expected_sensor_mean = fit_rows["sensor_2"].mean()
    assert scaler.mean_[1] == expected_sensor_mean
    assert scaler.mean_[1] < 1000

    with np.load(output_dir / "train.npz") as train:
        assert train["X"].shape == (80 * 3, 30, 2)
        assert train["X_flat"].shape == (80 * 3, 60)
        assert not (train["y"] < 0).any()
        assert set(np.unique(train["unit_ids"])) == set(fit_ids)
    with np.load(output_dir / "validation.npz") as validation:
        assert validation["X"].shape == (20 * 3, 30, 2)
        assert set(np.unique(validation["unit_ids"])) == set(saved_validation_ids)
    with np.load(output_dir / "test.npz") as test:
        assert test["X"].shape == (100, 30, 2)
        np.testing.assert_array_equal(test["lengths"], np.full(100, 20))
        np.testing.assert_array_equal(test["end_cycles"], np.full(100, 20))
        np.testing.assert_array_equal(test["y"], np.arange(100))
        expected_padding = np.repeat(test["X"][:, 9:10], 10, axis=1)
        np.testing.assert_array_equal(test["X"][:, :10], expected_padding)

    report = tmp_path / "docs" / "preprocessing_report.md"
    assert report.is_file()
    report_text = " ".join(report.read_text(encoding="utf-8").split())
    assert "ajustó con 2,560 filas" in report_text
    assert (output_dir / "preprocessing_manifest.json").is_file()


def test_preprocess_refuses_to_replace_unmanaged_outputs(tmp_path):
    data_dir = tmp_path / "data" / "raw"
    _write_synthetic_fd001(data_dir)
    output_dir = tmp_path / "artifacts" / "processed" / "fd001"
    preprocess_fd001(data_dir, output_dir, _config())
    sentinel = output_dir / "user_notes.txt"
    sentinel.write_text("keep", encoding="utf-8")

    try:
        preprocess_fd001(data_dir, output_dir, _config())
    except FileExistsError as error:
        assert "no administrados" in str(error)
    else:
        raise AssertionError("Preprocessing must not overwrite unknown files.")

    assert sentinel.read_text(encoding="utf-8") == "keep"
