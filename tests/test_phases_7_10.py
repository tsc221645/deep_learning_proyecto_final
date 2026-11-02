import numpy as np
import pandas as pd
import pytest
import torch
import json
import yaml
from pathlib import Path

from predictguard.costs import Costs, trajectory_cost
from predictguard.models.lstm import LSTMRegressor
from predictguard.policy import simulate
from predictguard.policy import run as run_policy
from predictguard.evaluate_final import freeze


def test_lstm_ignores_left_padding_and_checks_shape():
    torch.manual_seed(42)
    model = LSTMRegressor(2, hidden_size=4, num_layers=1, dropout=0).eval()
    first = torch.tensor([[[99., 99.], [99., 99.], [1., 2.], [3., 4.]]])
    second = torch.tensor([[[-99., -99.], [-99., -99.], [1., 2.], [3., 4.]]])
    with torch.inference_mode():
        assert torch.allclose(model(first, torch.tensor([2])), model(second, torch.tensor([2])))
    with pytest.raises(ValueError):
        model(first, torch.tensor([5]))


def test_cost_does_not_charge_intervention_after_lead_time():
    costs = Costs(intervention_q=10, failure_q=100, unused_cycle_q=2, inference_q=1)
    early = trajectory_cost(failure_cycle=20, intervention_cycle=15, evaluated_cycles=3,
                            costs=costs, lead_time=5)
    late = trajectory_cost(failure_cycle=20, intervention_cycle=17, evaluated_cycles=3,
                           costs=costs, lead_time=5)
    assert early["total_q"] == 23
    assert late["total_q"] == 103


def test_policy_uses_first_threshold_crossing_and_complete_rul():
    frame = pd.DataFrame({"unit_id": [1, 1, 1], "end_cycle": [10, 11, 12],
                          "rul_actual_cycles": [10, 9, 8],
                          "rul_predicted_cycles": [7, 5, 3]})
    outcome = simulate(frame, "mlp", 5, Costs(10, 100, 1, 0), 2).iloc[0]
    assert outcome["intervention_cycle"] == 11
    assert outcome["unused_life_q"] == 9
    assert outcome["evaluated_cycles"] == 2


def test_business_outputs_use_validation_only(tmp_path):
    metrics = tmp_path / "artifacts" / "metrics"
    metrics.mkdir(parents=True)
    frame = pd.DataFrame({"unit_id": np.repeat(np.arange(1, 21), 2),
                          "end_cycle": np.tile([10, 11], 20),
                          "rul_actual_cycles": np.tile([10, 9], 20),
                          "rul_predicted_cycles": np.tile([10, 9], 20)})
    frame.to_csv(metrics / "mlp_validation_predictions.csv", index=False)
    frame.to_csv(metrics / "lstm_validation_predictions.csv", index=False)
    config = tmp_path / "business.yaml"
    config.write_text(yaml.safe_dump({"lstm_run": "lstm", "lead_time_cycles": 2,
                                      "thresholds_cycles": [5, 10, 15],
                                      "costs_q": {"intervention_q": 10, "failure_q": 100,
                                                  "unused_cycle_q": 1, "inference_q": 0.1}}))
    result = run_policy(tmp_path, config)
    assert result["test_official_used"] is False
    assert (tmp_path / "artifacts" / "business" / "cost_curve.csv").is_file()


def test_freeze_does_not_read_test(tmp_path):
    models = tmp_path / "artifacts" / "models"
    metrics = tmp_path / "artifacts" / "metrics"
    processed = tmp_path / "artifacts" / "processed" / "fd001"
    for directory in (models, metrics, processed):
        directory.mkdir(parents=True)
    (models / "mlp_best.pt").write_bytes(b"mlp")
    (models / "lstm_best.pt").write_bytes(b"lstm")
    features = ["sensor_2"]
    for model in ("mlp", "lstm"):
        (metrics / f"{model}_training_summary.json").write_text(json.dumps({
            "feature_columns": features, "window_size": 30,
            "selected_checkpoint": f"{model}_best.pt",
            "selected_validation_metrics": {"mae_cycles": 10.0}}))
    (processed / "preprocessing_manifest.json").write_text(json.dumps({
        "features": {"ordered_columns": features}}))
    assert freeze(tmp_path).is_file()
