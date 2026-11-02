"""Train LSTM candidates using the same frozen engines and metrics as the MLP."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from predictguard.experiment_log import atomic_write_csv, save_experiment_results
from predictguard.metrics import regression_metrics
from predictguard.models.lstm import LSTMRegressor, load_lstm_checkpoint, predict_rul
from predictguard.train import _atomic_save_checkpoint, _atomic_write_json, _load_config, _seed_everything


def read_split(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path}. Ejecuta preprocess primero.")
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "y", "unit_ids", "end_cycles", "lengths"}
        if not required <= set(archive.files):
            raise ValueError(f"{path}: faltan campos {required - set(archive.files)}")
        data = {key: archive[key].copy() for key in required}
    if (data["X"].ndim != 3 or data["X"].shape[0] != len(data["y"])
            or any(data[key].shape != data["y"].shape for key in ("unit_ids", "end_cycles", "lengths"))
            or data["X"].dtype != np.float32 or data["y"].dtype != np.float32
            or not np.isfinite(data["X"]).all() or not np.isfinite(data["y"]).all()
            or (data["y"] < 0).any()):
        raise ValueError(f"{path}: datos inválidos.")
    return data


def evaluate(model: LSTMRegressor, data: dict[str, np.ndarray],
             device: torch.device, batch_size: int,
             loss_fn: nn.Module) -> tuple[dict, float, np.ndarray]:
    model.eval()
    predictions = []
    weighted_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(data["y"]), batch_size):
            stop = start + batch_size
            x = torch.from_numpy(data["X"][start:stop]).to(device)
            lengths = torch.from_numpy(data["lengths"][start:stop]).to(device)
            y = torch.from_numpy(data["y"][start:stop]).to(device)
            raw = model(x, lengths)
            weighted_loss += float(loss_fn(raw, y).item()) * len(y)
            predictions.append(raw.clamp_min(0).cpu().numpy())
    predicted = np.concatenate(predictions).astype(np.float64)
    return regression_metrics(data["y"], predicted), weighted_loss / len(data["y"]), predicted


def run(config_path: Path, *, run_id: str = "lstm", hidden_size: int | None = None,
        num_layers: int | None = None, dropout: float | None = None,
        learning_rate: float | None = None, seeds: list[int] | None = None) -> dict:
    if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
        raise ValueError("run_id debe usar letras, números, guion o guion bajo.")
    config = _load_config(config_path)
    root = config_path.resolve().parent.parent
    processed = Path(config["preprocessing"]["output_root"])
    if not processed.is_absolute():
        processed = root / processed
    manifest = json.loads((processed / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    features = manifest["features"]["ordered_columns"]
    window_size = int(manifest["windows"]["size"])
    train, val = read_split(processed / "train.npz"), read_split(processed / "validation.npz")
    if train["X"].shape[1:] != (window_size, len(features)) or val["X"].shape[1:] != train["X"].shape[1:]:
        raise ValueError("Las ventanas no coinciden con el manifiesto.")
    if set(train["unit_ids"]) & set(val["unit_ids"]):
        raise ValueError("Hay fuga de motores entre train y validación.")
    model_config = dict(config["models"]["lstm"])
    for key, value in (("hidden_size", hidden_size), ("num_layers", num_layers), ("dropout", dropout)):
        if value is not None:
            model_config[key] = value
    training = config["training"]
    seeds = list(training["seeds"] if seeds is None else seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Se requiere al menos una semilla sin duplicados.")
    batch_size = int(training["batch_size"])
    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping"]["patience"])
    min_delta = float(training["early_stopping"]["min_delta_cycles"])
    lr = float(training["optimizer"]["learning_rate"] if learning_rate is None else learning_rate)
    if (batch_size < 1 or max_epochs < 1 or patience < 1 or lr <= 0 or min_delta < 0
            or training["early_stopping"]["monitor"] != "validation_mae_cycles"
            or training["loss"]["name"] != "huber" or training["optimizer"]["name"] != "adam"):
        raise ValueError("Protocolo de entrenamiento inválido.")
    device_name = config.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    loss_fn = nn.SmoothL1Loss(beta=float(training["loss"]["delta_cycles"]))
    models_dir = root / "artifacts" / "models"
    metrics_dir = root / "artifacts" / "metrics"
    histories, results = [], []
    for seed in seeds:
        _seed_everything(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(train["X"]),
                                          torch.from_numpy(train["lengths"]),
                                          torch.from_numpy(train["y"])),
                            batch_size=batch_size, shuffle=True, generator=generator,
                            pin_memory=device.type == "cuda")
        model = LSTMRegressor(len(features), **model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        checkpoint = models_dir / f"{run_id}_seed_{seed}.pt"
        best_mae, best_epoch, no_improve = float("inf"), 0, 0
        started = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for x, lengths, y in loader:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x, lengths), y)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * len(y)
            train_metrics, _, _ = evaluate(model, train, device, batch_size * 4, loss_fn)
            val_metrics, val_loss, _ = evaluate(model, val, device, batch_size * 4, loss_fn)
            histories.append({"run_id": run_id, "seed": seed, "epoch": epoch,
                              "train_loss": train_loss / len(train["y"]), "validation_loss": val_loss,
                              "train_mae_cycles": train_metrics["mae_cycles"],
                              "validation_mae_cycles": val_metrics["mae_cycles"]})
            if val_metrics["mae_cycles"] < best_mae - min_delta:
                best_mae, best_epoch, no_improve = val_metrics["mae_cycles"], epoch, 0
                payload = {"state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                           "model": model_config, "feature_columns": features, "window_size": window_size,
                           "seed": seed, "best_epoch": epoch, "validation_metrics": val_metrics,
                           "preprocessing_manifest": manifest["source_file_sha256"]}
                _atomic_save_checkpoint(checkpoint, payload)
                best_train, best_val = train_metrics, val_metrics
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        results.append({"seed": seed, "best_epoch": best_epoch, "epochs_trained": epoch,
                        "train_seconds": time.perf_counter() - started,
                        "training_metrics": best_train, "validation_metrics": best_val,
                        "checkpoint": checkpoint.name})
        print(f"{run_id} seed {seed}: época {best_epoch}, MAE validación {best_mae:.2f}")
    selected = min(results, key=lambda row: row["validation_metrics"]["mae_cycles"])
    model, payload = load_lstm_checkpoint(models_dir / selected["checkpoint"], device)
    _atomic_save_checkpoint(models_dir / f"{run_id}_best.pt", payload)
    _, _, predictions = evaluate(model, val, device, batch_size * 4, loss_fn)
    prediction_frame = pd.DataFrame({"unit_id": val["unit_ids"], "end_cycle": val["end_cycles"],
                                     "rul_actual_cycles": val["y"], "rul_predicted_cycles": predictions,
                                     "residual_cycles": predictions - val["y"]})
    atomic_write_csv(metrics_dir / f"{run_id}_validation_predictions.csv", prediction_frame)
    atomic_write_csv(metrics_dir / f"{run_id}_training_history.csv", pd.DataFrame(histories))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 5))
    for seed in seeds:
        rows = [row for row in histories if row["seed"] == seed]
        axis.plot([row["epoch"] for row in rows],
                  [row["validation_mae_cycles"] for row in rows], label=f"seed {seed}")
    axis.set(xlabel="Época", ylabel="MAE validación (ciclos)", title=f"{run_id}: convergencia")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figures_dir = root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures_dir / f"{run_id}_learning_curves.png", dpi=150)
    plt.close(figure)
    for _ in range(20):
        predict_rul(val["X"][0], model)
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        predict_rul(val["X"][0], model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
               "model": "LSTMRegressor", "device": str(device), "device_name":
               torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "torch_version": torch.__version__, "window_size": window_size,
               "feature_columns": features, "model_config": model_config,
               "training_config": training, "validation_metrics_by_seed": results,
               "selected_seed": selected["seed"], "selected_checkpoint": f"{run_id}_best.pt",
               "selected_validation_metrics": selected["validation_metrics"],
               "inference_latency_batch_1_ms": {"p50": float(np.percentile(latencies, 50)),
                                                "p95": float(np.percentile(latencies, 95))},
               "official_test_loaded": False}
    _atomic_write_json(metrics_dir / f"{run_id}_training_summary.json", summary)
    rows = []
    for result in results:
        val_metrics, train_metrics = result["validation_metrics"], result["training_metrics"]
        rows.append({"experiment_id": f"{run_id}-seed-{result['seed']}", "model": "lstm",
                     "seed": result["seed"], "features": json.dumps(features), "window_size": window_size,
                     "learning_rate": lr, "epochs": result["best_epoch"], "batch_size": batch_size,
                     "train_mae_cycles": train_metrics["mae_cycles"],
                     "validation_mae_cycles": val_metrics["mae_cycles"],
                     "validation_rmse_cycles": val_metrics["rmse_cycles"],
                     "validation_nasa_phm_score_sum": val_metrics["nasa_phm_score_sum"],
                     "validation_nasa_phm_score_mean": val_metrics["nasa_phm_score_mean"],
                     "validation_bias_cycles": val_metrics["bias_cycles"],
                     "validation_absolute_error_p50_cycles": val_metrics["absolute_error_p50_cycles"],
                     "validation_absolute_error_p90_cycles": val_metrics["absolute_error_p90_cycles"],
                     "train_seconds": result["train_seconds"],
                     "inference_p50_ms": summary["inference_latency_batch_1_ms"]["p50"] if result is selected else "",
                     "inference_p95_ms": summary["inference_latency_batch_1_ms"]["p95"] if result is selected else "",
                     "status": "selected" if result is selected else "complete", "split": "frozen_group_split",
                     "notes": "Validación por motores; test oficial reservado."})
    save_experiment_results(root / "artifacts" / "experiments" / "results.csv", rows,
                            replace_ids=set(), replace_prefixes=(f"{run_id}-seed-",))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--run-id", default="lstm")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    run(args.config, run_id=args.run_id, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
