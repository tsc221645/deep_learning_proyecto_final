"""Run the training-median RUL baseline on the frozen validation split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from predictguard.metrics import regression_metrics

EXPERIMENT_COLUMNS = [
    "experiment_id",
    "model",
    "seed",
    "features",
    "window_size",
    "learning_rate",
    "epochs",
    "batch_size",
    "train_mae_cycles",
    "validation_mae_cycles",
    "validation_rmse_cycles",
    "validation_nasa_phm_score_sum",
    "validation_nasa_phm_score_mean",
    "validation_bias_cycles",
    "validation_absolute_error_p50_cycles",
    "validation_absolute_error_p90_cycles",
    "train_seconds",
    "inference_p50_ms",
    "inference_p95_ms",
    "status",
    "split",
    "notes",
]
EXPERIMENT_ID = "baseline-median-train-rul"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Falta el artefacto de preprocesamiento: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "X_flat", "y", "unit_ids", "end_cycles", "lengths"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path.name} no contiene los campos: {sorted(missing)}")
        return {name: archive[name].copy() for name in required}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            frame.to_csv(file, index=False, quoting=csv.QUOTE_MINIMAL)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _save_experiment_result(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) != EXPERIMENT_COLUMNS:
            raise ValueError(
                f"El esquema de resultados existente no coincide con el "
                f"protocolo: {path}"
            )
        existing = existing.loc[existing["experiment_id"] != EXPERIMENT_ID]
        results = pd.concat(
            [existing, pd.DataFrame([row], columns=EXPERIMENT_COLUMNS)],
            ignore_index=True,
        )
    else:
        results = pd.DataFrame([row], columns=EXPERIMENT_COLUMNS)
    _atomic_write_csv(path, results)


def run_median_baseline(
    processed_dir: Path,
    metrics_dir: Path,
    results_path: Path,
) -> dict[str, Any]:
    """Fit on training labels and report only the held-out validation split."""
    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / "preprocessing_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta el manifiesto de preprocesamiento: {manifest_path}")
    preprocessing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = _load_npz(processed_dir / "train.npz")
    validation = _load_npz(processed_dir / "validation.npz")

    training_labels = np.asarray(train["y"], dtype=np.float64)
    validation_labels = np.asarray(validation["y"], dtype=np.float64)
    if training_labels.ndim != 1 or training_labels.size == 0:
        raise ValueError("Las etiquetas de train deben ser un vector no vacío.")
    if not np.isfinite(training_labels).all() or (training_labels < 0).any():
        raise ValueError("Las etiquetas de train deben ser finitas y no negativas.")
    if not np.isfinite(validation_labels).all() or (validation_labels < 0).any():
        raise ValueError("Las etiquetas de validación deben ser finitas y no negativas.")

    baseline_prediction = float(np.median(training_labels))
    train_predictions = np.full(
        training_labels.shape, baseline_prediction, dtype=np.float64
    )
    validation_predictions = np.full(
        validation_labels.shape, baseline_prediction, dtype=np.float64
    )
    train_metrics = regression_metrics(training_labels, train_predictions)
    validation_metrics = regression_metrics(
        validation_labels, validation_predictions
    )

    feature_columns = preprocessing_manifest["features"]["ordered_columns"]
    window_size = int(preprocessing_manifest["windows"]["size"])
    metrics_dir = Path(metrics_dir)
    prediction_frame = pd.DataFrame(
        {
            "unit_id": validation["unit_ids"],
            "end_cycle": validation["end_cycles"],
            "rul_actual_cycles": validation_labels,
            "rul_predicted_cycles": validation_predictions,
            "residual_cycles": validation_predictions - validation_labels,
        }
    )
    prediction_path = metrics_dir / "baseline_validation_predictions.csv"
    metrics_path = metrics_dir / "baseline_validation_metrics.json"
    experiment_row: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "model": "median_train_rul",
        "seed": "",
        "features": json.dumps(feature_columns, separators=(",", ":")),
        "window_size": window_size,
        "learning_rate": "",
        "epochs": "",
        "batch_size": "",
        "train_mae_cycles": train_metrics["mae_cycles"],
        "validation_mae_cycles": validation_metrics["mae_cycles"],
        "validation_rmse_cycles": validation_metrics["rmse_cycles"],
        "validation_nasa_phm_score_sum": validation_metrics[
            "nasa_phm_score_sum"
        ],
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
        "train_seconds": "",
        "inference_p50_ms": "",
        "inference_p95_ms": "",
        "status": "complete",
        "split": "frozen_group_split",
        "notes": (
            "Prediction is the median of training-window labels. Validation "
            "metrics pool complete overlapping windows; bootstrap by engine "
            "if uncertainty is needed. Official test is not loaded."
        ),
    }
    timestamp = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    metrics_payload = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": timestamp,
        "baseline_prediction_cycles": baseline_prediction,
        "training_label_count": int(training_labels.size),
        "validation_window_count": int(validation_labels.size),
        "training_metrics_in_sample": train_metrics,
        "validation_metrics": validation_metrics,
        "evaluation": {
            "split": "frozen validation engines",
            "unit_ids": sorted(
                int(unit_id) for unit_id in np.unique(validation["unit_ids"])
            ),
            "pooled_overlapping_windows": True,
            "official_test_loaded": False,
        },
        "features": feature_columns,
        "window_size": window_size,
    }

    _atomic_write_csv(prediction_path, prediction_frame)
    _atomic_write_json(metrics_path, metrics_payload)
    _save_experiment_result(Path(results_path), experiment_row)
    return metrics_payload


def _load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuración YAML inválida: {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecutar el baseline de mediana para RUL."
    )
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
    metrics = run_median_baseline(
        processed_root,
        repo_root / "artifacts" / "metrics",
        repo_root / "artifacts" / "experiments" / "results.csv",
    )
    validation_metrics = metrics["validation_metrics"]
    print(f"Predicción constante de RUL: {metrics['baseline_prediction_cycles']:.1f} ciclos")
    print(f"Validación MAE: {validation_metrics['mae_cycles']:.2f} ciclos")
    print(f"Validación RMSE: {validation_metrics['rmse_cycles']:.2f} ciclos")
    print(f"Score PHM08 medio: {validation_metrics['nasa_phm_score_mean']:.4f}")
    print("Test oficial: reservado, no se cargó.")


if __name__ == "__main__":
    main()
