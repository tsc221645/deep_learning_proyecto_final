import json

import numpy as np
import pytest
import torch

from predictguard.models.mlp import (
    MLPRegressor,
    load_mlp_checkpoint,
    predict_rul,
)
from predictguard.train import run_mlp_training


def test_mlp_forward_checks_feature_shape():
    model = MLPRegressor(input_dim=6, hidden_dims=(4, 2), dropout=0.2)

    assert model(torch.zeros((3, 6))).shape == (3,)
    with pytest.raises(ValueError, match="Se esperaba un tensor"):
        model(torch.zeros((3, 5)))


def test_predict_rul_accepts_one_window_and_rejects_invalid_values():
    model = MLPRegressor(input_dim=4, hidden_dims=(3,), dropout=0.0)

    prediction = predict_rul(np.zeros((2, 2), dtype=np.float32), model, "cpu")

    assert isinstance(prediction, float)
    assert prediction >= 0
    with pytest.raises(ValueError, match="valores finitos"):
        predict_rul(np.array([[np.nan, 0], [0, 0]], dtype=np.float32), model)
    with pytest.raises(ValueError, match="contiene 3 valores"):
        predict_rul(np.zeros((1, 3), dtype=np.float32), model)


def test_mlp_training_saves_validation_outputs_and_never_loads_test(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "preprocessing_manifest.json").write_text(
        json.dumps(
            {
                "features": {"ordered_columns": ["sensor_1", "sensor_2"]},
                "windows": {"size": 2},
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(42)
    for split_name, sample_count in (("train", 12), ("validation", 6)):
        np.savez_compressed(
            processed_dir / f"{split_name}.npz",
            X_flat=rng.normal(size=(sample_count, 4)).astype(np.float32),
            y=rng.uniform(0, 8, size=sample_count).astype(np.float32),
            unit_ids=np.arange(sample_count, dtype=np.int64),
            end_cycles=np.arange(10, 10 + sample_count, dtype=np.int64),
        )

    summary = run_mlp_training(
        processed_dir,
        tmp_path / "metrics",
        tmp_path / "models",
        tmp_path / "figures",
        tmp_path / "experiments" / "results.csv",
        seeds=[7],
        training_config={
            "max_epochs": 3,
            "batch_size": 4,
            "early_stopping": {
                "monitor": "validation_mae_cycles",
                "patience": 2,
                "min_delta_cycles": 0.0,
                "restore_best_checkpoint": True,
            },
            "loss": {"name": "huber", "delta_cycles": 1.0},
            "optimizer": {"name": "adam", "learning_rate": 0.001},
        },
        model_config={"hidden_dims": [8, 4], "dropout": 0.2},
        device_name="cpu",
    )

    assert summary["selected_seed"] == 7
    assert summary["evaluation"]["official_test_loaded"] is False
    assert not (processed_dir / "test.npz").exists()
    assert (tmp_path / "models" / "mlp_best.pt").is_file()
    assert (tmp_path / "models" / "mlp_seed_7.pt").is_file()
    assert (tmp_path / "metrics" / "mlp_training_history.csv").is_file()
    assert (tmp_path / "metrics" / "mlp_training_history.json").is_file()
    assert (tmp_path / "metrics" / "mlp_training_summary.json").is_file()
    assert (tmp_path / "metrics" / "mlp_validation_predictions.csv").is_file()
    assert (tmp_path / "figures" / "mlp_learning_curves.png").is_file()
    assert (tmp_path / "figures" / "mlp_validation_residuals.png").is_file()
    assert summary["selected_validation_metrics"]["sample_count"] == 6
    model, checkpoint = load_mlp_checkpoint(tmp_path / "models" / "mlp_best.pt")
    assert checkpoint["seed"] == 7
    assert predict_rul(np.zeros((2, 2), dtype=np.float32), model) >= 0
    with pytest.raises(ValueError, match="window debe tener forma"):
        predict_rul(np.zeros((1, 4), dtype=np.float32), model)
