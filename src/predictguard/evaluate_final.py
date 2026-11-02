"""Freeze validation choices, then evaluate terminal FD001 test once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predictguard.experiment_log import atomic_write_csv
from predictguard.metrics import regression_metrics
from predictguard.models.lstm import load_lstm_checkpoint
from predictguard.models.mlp import load_mlp_checkpoint
from predictguard.train import _atomic_write_json
from predictguard.train_lstm import read_split


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(root: Path, lstm_run: str = "lstm") -> Path:
    """Select using validation artifacts only; do not open test data."""
    models = root / "artifacts" / "models"
    metrics = root / "artifacts" / "metrics"
    processed = root / "artifacts" / "processed" / "fd001"
    mlp = json.loads((metrics / "mlp_training_summary.json").read_text(encoding="utf-8"))
    lstm = json.loads((metrics / f"{lstm_run}_training_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((processed / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    if mlp["feature_columns"] != lstm["feature_columns"] or mlp["window_size"] != lstm["window_size"]:
        raise ValueError("MLP y LSTM no comparten features y ventana.")
    if mlp["feature_columns"] != manifest["features"]["ordered_columns"]:
        raise ValueError("Los modelos no coinciden con el preprocesamiento.")
    files = {"mlp": models / mlp["selected_checkpoint"],
             "lstm": models / lstm["selected_checkpoint"],
             "preprocessing": processed / "preprocessing_manifest.json"}
    payload = {"selection_basis": "validation_mae_cycles", "lstm_run": lstm_run,
               "features": mlp["feature_columns"], "window_size": mlp["window_size"],
               "validation_mae_cycles": {"mlp": mlp["selected_validation_metrics"]["mae_cycles"],
                                         "lstm": lstm["selected_validation_metrics"]["mae_cycles"]},
               "files": {name: {"path": str(path.relative_to(root)).replace('\\', '/'),
                                 "sha256": sha256(path)} for name, path in files.items()}}
    path = root / "artifacts" / "experiments" / "selected.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError("selected.json ya está congelado; conserva la evaluación original.")
        return path
    _atomic_write_json(path, payload)
    return path


def evaluate(root: Path, selected_path: Path, *, bootstrap_repeats: int = 2000) -> dict:
    out = root / "artifacts" / "metrics"
    if (out / "test_predictions.csv").exists():
        raise FileExistsError("El test oficial ya fue evaluado. Conserva los resultados existentes.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    for item in selected["files"].values():
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError(f"Cambió un artefacto congelado: {item['path']}")
    processed = root / "artifacts" / "processed" / "fd001"
    test = read_split(processed / "test.npz")
    if len(test["y"]) != 100 or len(np.unique(test["unit_ids"])) != 100:
        raise ValueError("Se esperaba una predicción terminal por cada uno de 100 motores.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp, _ = load_mlp_checkpoint(root / selected["files"]["mlp"]["path"], device)
    lstm, _ = load_lstm_checkpoint(root / selected["files"]["lstm"]["path"], device)
    mlp.eval(); lstm.eval()
    with torch.inference_mode():
        x = torch.from_numpy(test["X"]).to(device)
        lengths = torch.from_numpy(test["lengths"]).to(device)
        mlp_pred = mlp(x.flatten(1)).clamp_min(0).cpu().numpy()
        lstm_pred = lstm(x, lengths).clamp_min(0).cpu().numpy()
    frame = pd.DataFrame({"unit_id": test["unit_ids"], "end_cycle": test["end_cycles"],
                          "rul_actual_cycles": test["y"], "mlp_predicted_cycles": mlp_pred,
                          "lstm_predicted_cycles": lstm_pred})
    for model in ("mlp", "lstm"):
        frame[f"{model}_residual_cycles"] = frame[f"{model}_predicted_cycles"] - frame["rul_actual_cycles"]
    metrics = {model: regression_metrics(test["y"], frame[f"{model}_predicted_cycles"].to_numpy())
               for model in ("mlp", "lstm")}
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(frame), size=(bootstrap_repeats, len(frame)))
    errors = {model: np.abs(frame[f"{model}_residual_cycles"].to_numpy()) for model in ("mlp", "lstm")}
    boot = {model: errors[model][indices].mean(axis=1) for model in errors}
    diff = boot["lstm"] - boot["mlp"]
    intervals = {model: np.quantile(boot[model], [0.025, 0.975]).tolist() for model in boot}
    intervals["lstm_minus_mlp_mae"] = np.quantile(diff, [0.025, 0.975]).tolist()
    result = {"metrics": metrics, "bootstrap_95pct_mae_cycles": intervals,
              "bootstrap_unit": "engine", "bootstrap_repeats": bootstrap_repeats,
              "test_policy": "terminal observation only; no tuning on test"}
    atomic_write_csv(out / "test_predictions.csv", frame)
    _atomic_write_json(out / "test_metrics.json", result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = root / "artifacts" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in ("mlp", "lstm"):
        axes[0].scatter(frame["rul_actual_cycles"], frame[f"{model}_predicted_cycles"],
                        label=model.upper(), alpha=0.55, s=16)
        axes[1].scatter(frame["rul_actual_cycles"], frame[f"{model}_residual_cycles"],
                        label=model.upper(), alpha=0.55, s=16)
    axes[0].set(xlabel="RUL real", ylabel="RUL predicho", title="Test terminal")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="RUL real", ylabel="Predicho - real", title="Residuales")
    for axis in axes:
        axis.legend(); axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "test_comparison.png", dpi=150)
    plt.close(figure)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--lstm-run", default="lstm")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "freeze":
        print(freeze(root, args.lstm_run))
    else:
        print(json.dumps(evaluate(root, root / "artifacts" / "experiments" / "selected.json"), indent=2))


if __name__ == "__main__":
    main()
