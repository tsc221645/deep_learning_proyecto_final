"""Shared experiment result schema and idempotent CSV updates."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

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


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
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


def save_experiment_results(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    replace_ids: set[str],
    replace_prefixes: tuple[str, ...] = (),
) -> None:
    """Replace matching experiment IDs while preserving all other results."""
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) != EXPERIMENT_COLUMNS:
            raise ValueError(
                f"El esquema de resultados existente no coincide con el "
                f"protocolo: {path}"
            )
        experiment_ids = existing["experiment_id"].astype(str)
        replace_mask = experiment_ids.isin(replace_ids)
        for prefix in replace_prefixes:
            replace_mask |= experiment_ids.str.startswith(prefix)
        existing = existing.loc[~replace_mask]
        results = pd.concat(
            [existing, pd.DataFrame(rows, columns=EXPERIMENT_COLUMNS)],
            ignore_index=True,
        )
    else:
        results = pd.DataFrame(rows, columns=EXPERIMENT_COLUMNS)
    atomic_write_csv(path, results)
