"""Train and evaluate the MLP on the frozen FD001 validation split."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from predictguard.experiment_log import atomic_write_csv, save_experiment_results
from predictguard.metrics import regression_metrics
from predictguard.models.mlp import (
    MLPRegressor,
    load_mlp_checkpoint,
    predict_rul,
)

HISTORY_COLUMNS = [
    "seed",
    "epoch",
    "train_loss",
    "validation_loss",
    "train_mae_cycles",
    "validation_mae_cycles",
    "validation_rmse_cycles",
    "validation_nasa_phm_score_mean",
    "validation_bias_cycles",
    "validation_absolute_error_p50_cycles",
    "validation_absolute_error_p90_cycles",
    "learning_rate",
]


def _read_split(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Falta el artefacto de preprocesamiento: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"X_flat", "y", "unit_ids", "end_cycles"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path.name} no contiene los campos: {sorted(missing)}")
        split = {name: archive[name].copy() for name in required}

    features = split["X_flat"]
    labels = split["y"]
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"{path.name}: X_flat debe ser una matriz no vacía.")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(f"{path.name}: y debe coincidir con las ventanas.")
    for metadata_name in ("unit_ids", "end_cycles"):
        if split[metadata_name].shape != labels.shape:
            raise ValueError(
                f"{path.name}: {metadata_name} debe coincidir con las ventanas."
            )
    if not np.isfinite(features).all():
        raise ValueError(f"{path.name}: X_flat contiene valores no finitos.")
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError(f"{path.name}: y debe ser finito y no negativo.")
    if features.dtype != np.float32 or labels.dtype != np.float32:
        raise ValueError(f"{path.name}: X_flat y deben tener dtype float32.")
    return split


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            torch.save(payload, file)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _evaluate(
    model: MLPRegressor,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    loss_function: nn.Module,
) -> tuple[dict[str, float | int], float, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    weighted_loss = 0.0
    with torch.inference_mode():
        for start in range(0, labels.size, batch_size):
            stop = min(start + batch_size, labels.size)
            feature_batch = torch.from_numpy(features[start:stop]).to(device)
            label_batch = torch.from_numpy(labels[start:stop]).to(device)
            raw_predictions = model(feature_batch)
            loss = loss_function(raw_predictions, label_batch)
            weighted_loss += float(loss.item()) * (stop - start)
            predictions.append(
                raw_predictions.clamp_min(0.0).cpu().numpy()
            )

    predicted = np.concatenate(predictions).astype(np.float64, copy=False)
    metrics = regression_metrics(labels, predicted)
    return metrics, weighted_loss / labels.size, predicted


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_payload(
    model: MLPRegressor,
    *,
    seed: int,
    epoch: int,
    validation_metrics: dict[str, float | int],
    model_config: dict[str, Any],
    feature_columns: list[str],
    window_size: int,
) -> dict[str, Any]:
    return {
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "model": model_config,
        "input_dim": model.input_dim,
        "feature_columns": feature_columns,
        "window_size": window_size,
        "seed": seed,
        "best_epoch": epoch,
        "validation_metrics": validation_metrics,
    }


def _measure_latency(
    model: MLPRegressor,
    window: np.ndarray,
    device: torch.device,
    *,
    warmup: int = 20,
    repetitions: int = 100,
) -> tuple[float, float]:
    for _ in range(warmup):
        predict_rul(window, model, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    durations: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        predict_rul(window, model, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - started) * 1000)
    return (
        float(np.percentile(durations, 50)),
        float(np.percentile(durations, 95)),
    )


def _save_plots(history: list[dict[str, Any]], predictions: pd.DataFrame, figures_dir: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    learning_curve_path = figures_dir / "mlp_learning_curves.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    for seed in sorted({int(row["seed"]) for row in history}):
        seed_rows = [row for row in history if int(row["seed"]) == seed]
        epochs = [int(row["epoch"]) for row in seed_rows]
        axis.plot(
            epochs,
            [float(row["train_mae_cycles"]) for row in seed_rows],
            alpha=0.65,
            linestyle="--",
            label=f"Train {seed}",
        )
        axis.plot(
            epochs,
            [float(row["validation_mae_cycles"]) for row in seed_rows],
            label=f"Validación {seed}",
        )
    axis.set(title="MLP: MAE por época", xlabel="Época", ylabel="MAE (ciclos)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    figure.savefig(learning_curve_path, dpi=150)
    plt.close(figure)

    residual_path = figures_dir / "mlp_validation_residuals.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        predictions["rul_actual_cycles"],
        predictions["residual_cycles"],
        s=12,
        alpha=0.45,
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        title="MLP: residuales en validación",
        xlabel="RUL real (ciclos)",
        ylabel="Predicción - real (ciclos)",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(residual_path, dpi=150)
    plt.close(figure)


def run_mlp_training(
    processed_dir: Path,
    metrics_dir: Path,
    models_dir: Path,
    figures_dir: Path,
    results_path: Path,
    *,
    seeds: list[int],
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train configured seeds and select the checkpoint with lowest validation MAE."""
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("training.seeds debe ser una lista de enteros >= 0.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds no puede contener valores duplicados.")

    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / "preprocessing_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta el manifiesto de preprocesamiento: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_columns = manifest["features"]["ordered_columns"]
    window_size = int(manifest["windows"]["size"])

    train = _read_split(processed_dir / "train.npz")
    validation = _read_split(processed_dir / "validation.npz")
    input_dim = window_size * len(feature_columns)
    if train["X_flat"].shape[1] != input_dim or validation["X_flat"].shape[1] != input_dim:
        raise ValueError(
            "La dimensión de X_flat no coincide con ventana * número de features."
        )
    if validation["X_flat"].shape[1] != train["X_flat"].shape[1]:
        raise ValueError("Train y validación deben tener la misma cantidad de features.")

    max_epochs = int(training_config["max_epochs"])
    batch_size = int(training_config["batch_size"])
    patience = int(training_config["early_stopping"]["patience"])
    min_delta = float(training_config["early_stopping"]["min_delta_cycles"])
    restore_best = training_config["early_stopping"]["restore_best_checkpoint"]
    if training_config["early_stopping"]["monitor"] != "validation_mae_cycles":
        raise ValueError("Early stopping debe monitorear validation_mae_cycles.")
    if restore_best is not True:
        raise ValueError("early_stopping.restore_best_checkpoint debe ser true.")
    loss_config = training_config["loss"]
    if loss_config["name"] != "huber":
        raise ValueError("Este entrenador requiere loss.name = huber.")
    delta = float(loss_config["delta_cycles"])
    optimizer_config = training_config["optimizer"]
    if optimizer_config["name"] != "adam":
        raise ValueError("Este entrenador requiere optimizer.name = adam.")
    learning_rate = float(optimizer_config["learning_rate"])
    hidden_dims = [int(width) for width in model_config["hidden_dims"]]
    dropout = float(model_config["dropout"])
    if max_epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("Épocas, batch_size y paciencia deben ser positivos.")
    if (
        not np.isfinite([min_delta, delta, learning_rate]).all()
        or min_delta < 0
        or delta <= 0
        or learning_rate <= 0
    ):
        raise ValueError("min_delta debe ser >= 0; delta y learning rate, positivos.")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")

    metrics_dir = Path(metrics_dir)
    models_dir = Path(models_dir)
    history_rows: list[dict[str, Any]] = []
    seed_results: list[dict[str, Any]] = []
    loss_function = nn.SmoothL1Loss(beta=delta)

    for seed in seeds:
        _seed_everything(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        train_dataset = TensorDataset(
            torch.from_numpy(train["X_flat"]),
            torch.from_numpy(train["y"]),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model = MLPRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            window_size=window_size,
            feature_count=len(feature_columns),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        checkpoint_path = models_dir / f"mlp_seed_{seed}.pt"
        best_validation_mae = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_train_metrics: dict[str, float | int] | None = None
        best_validation_metrics: dict[str, float | int] | None = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            model.train()
            weighted_train_loss = 0.0
            for feature_batch, label_batch in train_loader:
                feature_batch = feature_batch.to(device, non_blocking=True)
                label_batch = label_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                raw_predictions = model(feature_batch)
                loss = loss_function(raw_predictions, label_batch)
                loss.backward()
                optimizer.step()
                weighted_train_loss += float(loss.item()) * label_batch.shape[0]

            train_metrics, _, _ = _evaluate(
                model,
                train["X_flat"],
                train["y"],
                device=device,
                batch_size=batch_size * 4,
                loss_function=loss_function,
            )
            validation_metrics, validation_loss, _ = _evaluate(
                model,
                validation["X_flat"],
                validation["y"],
                device=device,
                batch_size=batch_size * 4,
                loss_function=loss_function,
            )
            epoch_row = {
                "seed": seed,
                "epoch": epoch,
                "train_loss": weighted_train_loss / train["y"].size,
                "validation_loss": validation_loss,
                "train_mae_cycles": train_metrics["mae_cycles"],
                "validation_mae_cycles": validation_metrics["mae_cycles"],
                "validation_rmse_cycles": validation_metrics["rmse_cycles"],
                "validation_nasa_phm_score_mean": validation_metrics[
                    "nasa_phm_score_mean"
                ],
                "validation_bias_cycles": validation_metrics["bias_cycles"],
                "validation_absolute_error_p50_cycles": validation_metrics[
                    "absolute_error_p50_cycles"
                ],
                "validation_absolute_error_p90_cycles": validation_metrics[
                    "absolute_error_p90_cycles"
                ],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history_rows.append(epoch_row)

            if float(validation_metrics["mae_cycles"]) < best_validation_mae - min_delta:
                best_validation_mae = float(validation_metrics["mae_cycles"])
                best_epoch = epoch
                epochs_without_improvement = 0
                best_train_metrics = train_metrics
                best_validation_metrics = validation_metrics
                _atomic_save_checkpoint(
                    checkpoint_path,
                    _checkpoint_payload(
                        model,
                        seed=seed,
                        epoch=epoch,
                        validation_metrics=validation_metrics,
                        model_config={
                            "hidden_dims": hidden_dims,
                            "dropout": dropout,
                        },
                        feature_columns=feature_columns,
                        window_size=window_size,
                    ),
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        if best_epoch == 0 or best_validation_metrics is None or best_train_metrics is None:
            raise RuntimeError(f"No se generó un checkpoint válido para la semilla {seed}.")
        restored_model, _ = load_mlp_checkpoint(checkpoint_path, device)
        model.load_state_dict(restored_model.state_dict())
        seed_results.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "epochs_trained": epoch,
                "train_seconds": train_seconds,
                "training_metrics": best_train_metrics,
                "validation_metrics": best_validation_metrics,
                "checkpoint": checkpoint_path.name,
            }
        )
        print(
            f"Seed {seed}: mejor época {best_epoch}/{epoch}, "
            f"MAE validación {best_validation_mae:.2f} ciclos"
        )

    best_result = min(
        seed_results,
        key=lambda result: (
            float(result["validation_metrics"]["mae_cycles"]),
            seeds.index(int(result["seed"])),
        ),
    )
    selected_seed = int(best_result["seed"])
    selected_checkpoint = models_dir / f"mlp_seed_{selected_seed}.pt"
    selected_model, selected_payload = load_mlp_checkpoint(selected_checkpoint, device)
    best_checkpoint_path = models_dir / "mlp_best.pt"
    _atomic_save_checkpoint(best_checkpoint_path, selected_payload)

    actual = validation["y"].astype(np.float64, copy=False)
    _, _, selected_predictions = _evaluate(
        selected_model,
        validation["X_flat"],
        validation["y"],
        device=device,
        batch_size=batch_size * 4,
        loss_function=loss_function,
    )
    predictions = np.asarray(selected_predictions, dtype=np.float64)
    prediction_frame = pd.DataFrame(
        {
            "unit_id": validation["unit_ids"],
            "end_cycle": validation["end_cycles"],
            "rul_actual_cycles": actual,
            "rul_predicted_cycles": predictions,
            "residual_cycles": predictions - actual,
        }
    )
    prediction_path = metrics_dir / "mlp_validation_predictions.csv"
    history_path = metrics_dir / "mlp_training_history.csv"
    summary_path = metrics_dir / "mlp_training_summary.json"
    atomic_write_csv(history_path, pd.DataFrame(history_rows, columns=HISTORY_COLUMNS))
    atomic_write_csv(prediction_path, prediction_frame)

    validation_windows = validation["X_flat"].reshape(
        -1, window_size, len(feature_columns)
    )
    inference_p50, inference_p95 = _measure_latency(
        selected_model,
        validation_windows[0],
        device,
    )
    metric_names = (
        "mae_cycles",
        "rmse_cycles",
        "nasa_phm_score_mean",
        "bias_cycles",
    )
    aggregate = {
        metric: {
            "mean": float(
                np.mean(
                    [
                        float(result["validation_metrics"][metric])
                        for result in seed_results
                    ]
                )
            ),
            "std": float(
                np.std(
                    [
                        float(result["validation_metrics"][metric])
                        for result in seed_results
                    ],
                    ddof=0,
                )
            ),
        }
        for metric in metric_names
    }
    device_label = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    summary = {
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "model": "MLPRegressor",
        "device": str(device),
        "device_name": device_label,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "training_config": {
            "seeds": seeds,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "early_stopping": training_config["early_stopping"],
            "loss": loss_config,
            "optimizer": training_config["optimizer"],
        },
        "model_config": {"hidden_dims": hidden_dims, "dropout": dropout},
        "window_size": window_size,
        "feature_columns": feature_columns,
        "training_window_count": int(train["y"].size),
        "validation_window_count": int(validation["y"].size),
        "selected_seed": selected_seed,
        "selected_best_epoch": int(best_result["best_epoch"]),
        "selected_checkpoint": best_checkpoint_path.name,
        "selected_validation_metrics": best_result["validation_metrics"],
        "validation_metrics_by_seed": seed_results,
        "validation_metrics_aggregate": aggregate,
        "inference_latency_batch_1_ms": {
            "p50": inference_p50,
            "p95": inference_p95,
            "warmup_runs": 20,
            "measurement_runs": 100,
        },
        "evaluation": {
            "split": "frozen validation engines",
            "pooled_overlapping_windows": True,
            "official_test_loaded": False,
        },
    }
    _atomic_write_json(summary_path, summary)
    _atomic_write_json(
        metrics_dir / "mlp_training_history.json",
        {"history": history_rows, "summary": summary},
    )

    result_rows: list[dict[str, Any]] = []
    for result in seed_results:
        seed = int(result["seed"])
        train_metrics = result["training_metrics"]
        val_metrics = result["validation_metrics"]
        is_selected = seed == selected_seed
        result_rows.append(
            {
                "experiment_id": f"mlp-seed-{seed}",
                "model": "mlp",
                "seed": seed,
                "features": json.dumps(feature_columns, separators=(",", ":")),
                "window_size": window_size,
                "learning_rate": learning_rate,
                "epochs": result["best_epoch"],
                "batch_size": batch_size,
                "train_mae_cycles": train_metrics["mae_cycles"],
                "validation_mae_cycles": val_metrics["mae_cycles"],
                "validation_rmse_cycles": val_metrics["rmse_cycles"],
                "validation_nasa_phm_score_sum": val_metrics[
                    "nasa_phm_score_sum"
                ],
                "validation_nasa_phm_score_mean": val_metrics[
                    "nasa_phm_score_mean"
                ],
                "validation_bias_cycles": val_metrics["bias_cycles"],
                "validation_absolute_error_p50_cycles": val_metrics[
                    "absolute_error_p50_cycles"
                ],
                "validation_absolute_error_p90_cycles": val_metrics[
                    "absolute_error_p90_cycles"
                ],
                "train_seconds": result["train_seconds"],
                "inference_p50_ms": inference_p50 if is_selected else "",
                "inference_p95_ms": inference_p95 if is_selected else "",
                "status": "selected" if is_selected else "complete",
                "split": "frozen_group_split",
                "notes": (
                    "Validation metrics pool complete overlapping windows. "
                    "Checkpoint selected by validation MAE; official test "
                    "is not loaded."
                ),
            }
        )
    save_experiment_results(
        Path(results_path),
        result_rows,
        replace_ids=set(),
        replace_prefixes=("mlp-seed-",),
    )

    _save_plots(history_rows, prediction_frame, Path(figures_dir))
    return summary


def _load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuración YAML inválida: {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar el MLP de RUL.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Ruta a la configuración del proyecto.",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    repo_root = Path(args.config).resolve().parent.parent
    processed_root = Path(
        config.get("preprocessing", {}).get(
            "output_root", "artifacts/processed/fd001"
        )
    )
    if not processed_root.is_absolute():
        processed_root = repo_root / processed_root
    summary = run_mlp_training(
        processed_root,
        repo_root / "artifacts" / "metrics",
        repo_root / "artifacts" / "models",
        repo_root / "artifacts" / "figures",
        repo_root / "artifacts" / "experiments" / "results.csv",
        seeds=config["training"]["seeds"],
        training_config=config["training"],
        model_config=config["models"]["mlp"],
        device_name=config.get("device", "auto"),
    )
    aggregate_mae = summary["validation_metrics_aggregate"]["mae_cycles"]
    latency = summary["inference_latency_batch_1_ms"]
    print(
        f"Checkpoint seleccionado: seed {summary['selected_seed']}, "
        f"época {summary['selected_best_epoch']}"
    )
    print(
        f"MAE validación promedio: {aggregate_mae['mean']:.2f} "
        f"+/- {aggregate_mae['std']:.2f} ciclos"
    )
    print(
        f"Latencia batch 1: p50 {latency['p50']:.3f} ms, "
        f"p95 {latency['p95']:.3f} ms"
    )
    print("Test oficial: reservado, no se cargó.")


if __name__ == "__main__":
    main()
