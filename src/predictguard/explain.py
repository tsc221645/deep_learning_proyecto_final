"""Validation-only feature permutation sensitivity for frozen models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predictguard.evaluate_final import sha256
from predictguard.experiment_log import atomic_write_csv
from predictguard.models.lstm import load_lstm_checkpoint
from predictguard.models.mlp import load_mlp_checkpoint
from predictguard.train_lstm import read_split


def run(root: Path, seed: int = 42) -> pd.DataFrame:
    selected = json.loads((root / "artifacts" / "experiments" / "selected.json").read_text(encoding="utf-8"))
    for item in selected["files"].values():
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError("Un artefacto congelado cambió.")
    validation = read_split(root / "artifacts" / "processed" / "fd001" / "validation.npz")
    if validation["X"].shape[2] != len(selected["features"]):
        raise ValueError("El número de features no coincide.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"mlp": load_mlp_checkpoint(root / selected["files"]["mlp"]["path"], device)[0],
              "lstm": load_lstm_checkpoint(root / selected["files"]["lstm"]["path"], device)[0]}
    def predict(model_name: str, x: np.ndarray) -> np.ndarray:
        output = []
        model = models[model_name]
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(x), 256):
                batch = torch.from_numpy(x[start:start + 256]).to(device)
                if model_name == "mlp":
                    raw = model(batch.flatten(1))
                else:
                    lengths = torch.from_numpy(validation["lengths"][start:start + 256]).to(device)
                    raw = model(batch, lengths)
                output.append(raw.clamp_min(0).cpu().numpy())
        return np.concatenate(output)
    labels = validation["y"]
    baseline = {name: float(np.abs(predict(name, validation["X"]) - labels).mean()) for name in models}
    rng = np.random.default_rng(seed)
    rows = []
    for index, feature in enumerate(selected["features"]):
        permuted = validation["X"].copy()
        # Shuffle complete feature trajectories between validation windows.
        # This is predictive sensitivity, not a causal sensor explanation.
        permuted[:, :, index] = permuted[rng.permutation(len(permuted)), :, index]
        for name in models:
            mae = float(np.abs(predict(name, permuted) - labels).mean())
            rows.append({"model": name, "feature": feature, "baseline_mae_cycles": baseline[name],
                         "permuted_mae_cycles": mae, "mae_increase_cycles": mae - baseline[name],
                         "split": "validation"})
    frame = pd.DataFrame(rows)
    atomic_write_csv(root / "artifacts" / "metrics" / "validation_feature_sensitivity.csv", frame)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(run(args.root.resolve()).sort_values("mae_increase_cycles", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
