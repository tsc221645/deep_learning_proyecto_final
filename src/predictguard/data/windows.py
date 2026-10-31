"""Leakage-safe fixed-length windows for C-MAPSS trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

WindowMode = Literal["all", "last"]


def build_windows(
    features: np.ndarray,
    labels: np.ndarray | None,
    unit_ids: np.ndarray,
    cycles: np.ndarray,
    *,
    window_size: int,
    mode: WindowMode,
    stride: int = 1,
    labels_at_end: Mapping[int, float] | None = None,
) -> dict[str, np.ndarray]:
    """Build per-engine sliding windows without crossing trajectory boundaries."""
    features = np.asarray(features, dtype=np.float32)
    unit_ids = np.asarray(unit_ids, dtype=np.int64)
    cycles = np.asarray(cycles, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("features debe tener forma (registros, variables).")
    if len(features) != len(unit_ids) or len(features) != len(cycles):
        raise ValueError("features, unit_ids y cycles deben tener igual longitud.")
    if not len(features):
        raise ValueError("No se pueden generar ventanas sin registros.")
    if not np.isfinite(features).all():
        raise ValueError("features contiene valores no finitos.")
    if window_size < 1:
        raise ValueError("window_size debe ser mayor que cero.")
    if stride < 1:
        raise ValueError("stride debe ser mayor que cero.")
    if mode not in ("all", "last"):
        raise ValueError("mode debe ser 'all' o 'last'.")
    if mode == "all" and stride > window_size:
        raise ValueError("stride no puede superar window_size.")

    labels_array: np.ndarray | None = None
    if labels is not None:
        labels_array = np.asarray(labels, dtype=np.float32)
        if labels_array.shape != (len(features),):
            raise ValueError("labels debe contener una etiqueta por registro.")
        if not np.isfinite(labels_array).all():
            raise ValueError("labels contiene valores no finitos.")
    if mode == "all" and labels_array is None:
        raise ValueError("mode='all' requiere etiquetas para todos los registros.")
    if mode == "last" and labels_at_end is None:
        raise ValueError("mode='last' requiere labels_at_end por motor.")

    x_windows: list[np.ndarray] = []
    y_windows: list[float] = []
    window_units: list[int] = []
    end_cycles: list[int] = []
    sequence_lengths: list[int] = []

    unique_unit_ids = np.unique(unit_ids)
    if not np.array_equal(unit_ids, np.sort(unit_ids, kind="stable")):
        raise ValueError("Los registros deben estar ordenados por unit_id.")

    for unit_id in unique_unit_ids:
        engine_indices = np.flatnonzero(unit_ids == unit_id)
        engine_cycles = cycles[engine_indices]
        if len(engine_cycles) > 1 and not np.equal(np.diff(engine_cycles), 1).all():
            raise ValueError(f"El motor {unit_id} no tiene ciclos consecutivos.")
        engine_features = features[engine_indices]
        if mode == "all":
            assert labels_array is not None
            engine_labels = labels_array[engine_indices]
            for end_position in range(window_size - 1, len(engine_indices), stride):
                start_position = end_position - window_size + 1
                x_windows.append(
                    engine_features[start_position : end_position + 1].copy()
                )
                y_windows.append(float(engine_labels[end_position]))
                window_units.append(int(unit_id))
                end_cycles.append(int(engine_cycles[end_position]))
                sequence_lengths.append(window_size)
            continue

        assert labels_at_end is not None
        if int(unit_id) not in labels_at_end:
            raise ValueError(f"Falta la etiqueta terminal del motor {unit_id}.")
        observed_length = min(len(engine_indices), window_size)
        observed_window = engine_features[-observed_length:]
        if observed_length < window_size:
            padding = np.repeat(observed_window[:1], window_size - observed_length, axis=0)
            observed_window = np.concatenate((padding, observed_window), axis=0)
        x_windows.append(observed_window.copy())
        y_windows.append(float(labels_at_end[int(unit_id)]))
        window_units.append(int(unit_id))
        end_cycles.append(int(engine_cycles[-1]))
        sequence_lengths.append(observed_length)

    if not x_windows:
        raise ValueError(
            f"No se pudieron formar ventanas completas de {window_size} ciclos."
        )

    x_array = np.stack(x_windows).astype(np.float32, copy=False)
    y_array = np.asarray(y_windows, dtype=np.float32)
    if mode == "all" and (y_array < 0).any():
        raise ValueError("Las etiquetas de train/validación no pueden ser negativas.")
    return {
        "X": x_array,
        "X_flat": x_array.reshape(len(x_array), -1).copy(),
        "y": y_array,
        "unit_ids": np.asarray(window_units, dtype=np.int64),
        "end_cycles": np.asarray(end_cycles, dtype=np.int64),
        "lengths": np.asarray(sequence_lengths, dtype=np.int64),
    }
