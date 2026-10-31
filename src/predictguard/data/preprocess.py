"""Prepare leakage-safe FD001 splits, scaling, labels, and windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from predictguard.data.validate import (
    COLUMN_NAMES,
    RUL_FILE,
    TEST_FILE,
    TRAIN_FILE,
    resolve_data_dir,
    validate_fd001,
)
from predictguard.data.windows import build_windows

MANAGED_OUTPUT_FILES = (
    "train.npz",
    "validation.npz",
    "test.npz",
    "scaler.joblib",
    "preprocessing_manifest.json",
    "train_unit_ids.json",
    "val_unit_ids.json",
)


def _read_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(
        data_dir / TRAIN_FILE, sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    test = pd.read_csv(
        data_dir / TEST_FILE, sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    rul = pd.read_csv(data_dir / RUL_FILE, sep=r"\s+", header=None).iloc[:, 0]
    return train, test, rul.to_numpy(dtype=np.float32)


def _resolve_output_dir(config_path: Path, config: dict[str, Any]) -> Path:
    repo_root = Path(config_path).resolve().parent.parent
    output_root = Path(config.get("preprocessing", {}).get("output_root", "artifacts/processed/fd001"))
    if output_root.is_absolute():
        return output_root
    return repo_root / output_root


def _select_features(
    config: dict[str, Any], fit_frame: pd.DataFrame
) -> tuple[list[str], list[str], dict[str, float]]:
    data_config = config.get("data", {})
    feature_config = config.get("features", {})
    preprocessing_config = config.get("preprocessing", {})
    if data_config.get("subset") != "FD001":
        raise ValueError("Este preprocesador solo admite el subconjunto FD001.")
    if feature_config.get("exclude_unit_id", True) is not True:
        raise ValueError("unit_id es identificador y no puede ser feature.")

    selected: list[str] = []
    if feature_config.get("include_cycle", True):
        selected.append("cycle")
    if feature_config.get("include_operating_settings", True):
        selected.extend(["setting_1", "setting_2", "setting_3"])

    configured_sensors = feature_config.get("sensors", "all")
    if configured_sensors == "all":
        selected.extend(f"sensor_{index}" for index in range(1, 22))
    elif (
        isinstance(configured_sensors, list)
        and configured_sensors
        and all(
            isinstance(sensor, str)
            and sensor.startswith("sensor_")
            and sensor[7:].isdigit()
            and 1 <= int(sensor[7:]) <= 21
            for sensor in configured_sensors
        )
    ):
        selected.extend(configured_sensors)
    else:
        raise ValueError("features.sensors debe ser 'all' o una lista sensor_1..sensor_21.")

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("La configuración no seleccionó features.")
    missing_columns = sorted(set(selected).difference(fit_frame.columns))
    if missing_columns:
        raise ValueError(f"Columnas de features no disponibles: {missing_columns}")

    threshold = float(
        preprocessing_config.get("constant_variance_threshold", 1e-12)
    )
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("constant_variance_threshold debe ser finito y >= 0.")
    fit_variances = fit_frame[selected].var(ddof=0)
    drop_constant_sensors = preprocessing_config.get(
        "drop_constant_sensors", True
    )
    dropped = (
        [
            sensor
            for sensor in selected
            if sensor.startswith("sensor_")
            and fit_variances[sensor] <= threshold
        ]
        if drop_constant_sensors
        else []
    )
    retained = [feature for feature in selected if feature not in dropped]
    if not retained:
        raise ValueError("La selección de features eliminó todas las variables.")
    variances = {
        feature: float(fit_variances[feature]) for feature in selected
    }
    return retained, dropped, variances


def _make_targets(frame: pd.DataFrame) -> np.ndarray:
    last_cycle = frame.groupby("unit_id", sort=False)["cycle"].transform("max")
    targets = (last_cycle - frame["cycle"]).to_numpy(dtype=np.float32)
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("RUL de train debe ser finito y no negativo.")
    return targets


def _transform_partition(
    frame: pd.DataFrame,
    targets: np.ndarray,
    feature_columns: list[str],
    scaler: StandardScaler,
    window_size: int,
    stride: int,
) -> dict[str, np.ndarray]:
    scaled_features = scaler.transform(frame[feature_columns]).astype(np.float32)
    return build_windows(
        scaled_features,
        targets,
        frame["unit_id"].to_numpy(dtype=np.int64),
        frame["cycle"].to_numpy(dtype=np.int64),
        window_size=window_size,
        mode="all",
        stride=stride,
    )


def _write_json(path: Path, content: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_preprocessing_report(
    report_path: Path, manifest: dict[str, Any]
) -> None:
    split = manifest["split"]
    features = manifest["features"]
    windows = manifest["windows"]
    text = f"""# Preprocesamiento: NASA C-MAPSS FD001

## Partición por motor

- Semilla: {manifest["seed"]}.
- Motores para ajuste: {split["train_engines"]}.
- Motores para validación: {split["validation_engines"]}.
- Intersección train/validación: vacía.
- El conjunto oficial de 100 motores de test permanece reservado.
- IDs congelados: `artifacts/processed/fd001/train_unit_ids.json` y
  `artifacts/processed/fd001/val_unit_ids.json`.

## Etiquetas y escalado

- Train y validación usan `RUL = max_cycle_del_motor - cycle`. Las etiquetas
  corresponden al ciclo final de cada ventana y son no negativas.
- Test usa exclusivamente el RUL oficial del último ciclo observado por motor.
  No se inventan etiquetas para los ciclos parciales de test.
- `StandardScaler` se ajustó con {manifest["scaler"]["fit_rows"]:,} filas de los
  motores de ajuste. No se usaron datos de validación ni test para ajustar el
  escalador.
- `unit_id` se conserva como metadato de las ventanas y no es una feature.
- Orden de features: {", ".join(f"`{name}`" for name in features["ordered_columns"])}.
- Sensores constantes excluidos según la varianza de ajuste
  (umbral <= {features["constant_variance_threshold"]}):
  {", ".join(f"`{name}`" for name in features["dropped_constant_sensors"]) or "ninguno"}.
- Sensores no constantes, aunque casi planos, se retienen para evitar descartes
  basados en todo el conjunto antes de experimentar.

## Ventanas

- Tamaño: {windows["size"]} ciclos. Paso train/validación: {windows["train_stride"]}.
- Train: {windows["train"]["count"]:,} ventanas de
  {windows["train"]["unit_ids"]} motores.
- Validación: {windows["validation"]["count"]:,} ventanas de
  {windows["validation"]["unit_ids"]} motores.
- Test: {windows["test"]["count"]:,} ventanas, una por cada uno de los
  {windows["test"]["unit_ids"]} motores. Cada ventana termina en el último ciclo
  disponible de ese motor.
- Train/validación generan ventanas completas. Para trayectorias más cortas que
  la ventana, test repite a la izquierda la primera observación disponible y
  guarda la longitud real en `lengths`. No se usan observaciones futuras.
- Cada archivo `.npz` contiene `X` con forma
  `(muestras, ventana, features)`, `X_flat` para MLP, `y`, `unit_ids`,
  `end_cycles` y `lengths`.

## Artefactos

- `artifacts/processed/fd001/train.npz`
- `artifacts/processed/fd001/validation.npz`
- `artifacts/processed/fd001/test.npz`
- `artifacts/processed/fd001/scaler.joblib`
- `artifacts/processed/fd001/preprocessing_manifest.json`

Los hashes de los archivos fuente, parámetros del escalador, columnas y
partición quedan registrados en el manifiesto de preprocesamiento.
"""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def _atomic_save_outputs(output_dir: Path, outputs: dict[str, Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown_existing = [
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in MANAGED_OUTPUT_FILES
    ]
    if unknown_existing:
        raise FileExistsError(
            f"El directorio contiene archivos no administrados: {unknown_existing}"
        )

    for name in outputs:
        if name not in MANAGED_OUTPUT_FILES:
            raise ValueError(f"Archivo de salida no administrado: {name}")

    staged_files: list[tuple[Path, Path]] = []
    temporary_directory: str | None = None
    try:
        temporary_directory = tempfile.mkdtemp(
            prefix=".predictguard-preprocess-", dir=output_dir.parent
        )
        staging_dir = Path(temporary_directory)
        for name, write_output in outputs.items():
            staged_path = staging_dir / name
            write_output(staged_path)
            staged_files.append((staged_path, output_dir / name))

        for staged_path, destination in staged_files:
            os.replace(staged_path, destination)
    finally:
        if temporary_directory is not None:
            for path in Path(temporary_directory).iterdir():
                path.unlink()
            Path(temporary_directory).rmdir()


def preprocess_fd001(
    data_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Split train engines, fit a training-only scaler, and save window arrays."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    source_manifest = validate_fd001(data_dir)
    train, test, test_rul = _read_frames(data_dir)

    split_config = config.get("split", {})
    validation_fraction = float(split_config.get("validation_fraction", 0.2))
    random_state = int(split_config.get("random_state", config.get("seed", 42)))
    group_column = split_config.get("group_column", "unit_id")
    if group_column != "unit_id":
        raise ValueError("La partición debe agruparse por unit_id.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction debe estar entre 0 y 1.")

    engine_ids = np.sort(train["unit_id"].unique())
    fit_ids, validation_ids = train_test_split(
        engine_ids,
        test_size=validation_fraction,
        random_state=random_state,
        shuffle=True,
    )
    fit_ids = np.sort(fit_ids)
    validation_ids = np.sort(validation_ids)
    if set(fit_ids).intersection(validation_ids):
        raise AssertionError("Los motores de ajuste y validación se intersectan.")
    if set(fit_ids).union(validation_ids) != set(engine_ids):
        raise AssertionError("La partición no cubre todos los motores de train.")

    fit_frame = train.loc[train["unit_id"].isin(fit_ids)].copy()
    validation_frame = train.loc[train["unit_id"].isin(validation_ids)].copy()
    test_frame = test.copy()
    for frame in (fit_frame, validation_frame, test_frame):
        frame.sort_values(["unit_id", "cycle"], inplace=True, ignore_index=True)

    feature_columns, dropped_sensors, fit_variances = _select_features(
        config, fit_frame
    )
    scaler = StandardScaler()
    scaler.fit(fit_frame[feature_columns])
    window_config = config.get("window", {})
    window_size = int(window_config.get("size", 30))
    stride = int(
        config.get("preprocessing", {}).get("train_window_stride", 1)
    )
    if window_size < 1:
        raise ValueError("window.size debe ser mayor que cero.")
    if stride < 1 or stride > window_size:
        raise ValueError("train_window_stride debe estar entre 1 y window.size.")

    fit_targets = _make_targets(fit_frame)
    validation_targets = _make_targets(validation_frame)
    fit_windows = _transform_partition(
        fit_frame,
        fit_targets,
        feature_columns,
        scaler,
        window_size,
        stride,
    )
    validation_windows = _transform_partition(
        validation_frame,
        validation_targets,
        feature_columns,
        scaler,
        window_size,
        stride,
    )

    test_scaled = scaler.transform(test_frame[feature_columns]).astype(np.float32)
    test_engine_order = sorted(test_frame["unit_id"].unique().tolist())
    if test_engine_order != list(range(1, len(test_rul) + 1)):
        raise ValueError("El RUL oficial no se alinea con los motores de test.")
    test_windows = build_windows(
        test_scaled,
        labels=None,
        unit_ids=test_frame["unit_id"].to_numpy(dtype=np.int64),
        cycles=test_frame["cycle"].to_numpy(dtype=np.int64),
        window_size=window_size,
        mode="last",
        labels_at_end={
            unit_id: float(test_rul[unit_id - 1]) for unit_id in test_engine_order
        },
    )
    if len(test_windows["X"]) != len(test_engine_order):
        raise AssertionError("Test debe producir una ventana por motor.")

    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("El escalador ajustado contiene parámetros no finitos.")
    if any(feature in feature_columns for feature in ("unit_id",)):
        raise AssertionError("unit_id no puede ser una feature.")
    if (fit_windows["y"] < 0).any() or (validation_windows["y"] < 0).any():
        raise AssertionError("Las etiquetas de train y validación no pueden ser negativas.")

    preprocessing_manifest = {
        "schema_version": 1,
        "dataset": "NASA C-MAPSS FD001",
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source_file_sha256": {
            filename: metadata["sha256"]
            for filename, metadata in source_manifest["files"].items()
        },
        "seed": int(config.get("seed", random_state)),
        "split": {
            "group_column": group_column,
            "validation_fraction": validation_fraction,
            "random_state": random_state,
            "train_engines": len(fit_ids),
            "validation_engines": len(validation_ids),
            "train_unit_ids_file": "train_unit_ids.json",
            "validation_unit_ids_file": "val_unit_ids.json",
            "overlap": [],
        },
        "features": {
            "ordered_columns": feature_columns,
            "excluded_identifier": "unit_id",
            "dropped_constant_sensors": dropped_sensors,
            "constant_variance_threshold": float(
                config.get("preprocessing", {}).get(
                    "constant_variance_threshold", 1e-12
                )
            ),
            "fit_subset_population_variance": fit_variances,
        },
        "scaler": {
            "type": "StandardScaler",
            "fit_unit_ids": fit_ids.tolist(),
            "fit_rows": int(len(fit_frame)),
            "feature_names": feature_columns,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "artifact": "scaler.joblib",
        },
        "labels": {
            "training": "max_cycle_per_engine - cycle",
            "test": "official RUL for each engine's last observed cycle only",
        },
        "windows": {
            "size": window_size,
            "train_stride": stride,
            "train_validation_policy": "complete sliding windows only",
            "test_policy": "one window ending at each engine's last observed cycle",
            "short_sequence_policy": "left-pad by repeating its first observed feature row; record valid length",
            "train": {
                "count": len(fit_windows["y"]),
                "unit_ids": int(np.unique(fit_windows["unit_ids"]).size),
            },
            "validation": {
                "count": len(validation_windows["y"]),
                "unit_ids": int(np.unique(validation_windows["unit_ids"]).size),
            },
            "test": {
                "count": len(test_windows["y"]),
                "unit_ids": int(np.unique(test_windows["unit_ids"]).size),
            },
        },
        "artifacts": {
            "train": "train.npz",
            "validation": "validation.npz",
            "test": "test.npz",
        },
    }

    def write_npz(windows: dict[str, np.ndarray]):
        return lambda path: np.savez_compressed(path, **windows)

    def write_scaler(path: Path):
        joblib.dump(scaler, path)

    outputs = {
        "train.npz": write_npz(fit_windows),
        "validation.npz": write_npz(validation_windows),
        "test.npz": write_npz(test_windows),
        "scaler.joblib": write_scaler,
        "preprocessing_manifest.json": lambda path: _write_json(
            path, preprocessing_manifest
        ),
        "train_unit_ids.json": lambda path: _write_json(
            path, fit_ids.tolist()
        ),
        "val_unit_ids.json": lambda path: _write_json(
            path, validation_ids.tolist()
        ),
    }
    _atomic_save_outputs(output_dir, outputs)
    report_path = output_dir.parents[2] / "docs" / "preprocessing_report.md"
    _write_preprocessing_report(report_path, preprocessing_manifest)
    return preprocessing_manifest


def _load_config(config_path: Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuración YAML inválida: {config_path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preparar splits y ventanas de NASA C-MAPSS FD001."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Ruta a la configuración del proyecto.",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    data_dir = resolve_data_dir(args.config)
    output_dir = _resolve_output_dir(args.config, config)
    manifest = preprocess_fd001(data_dir, output_dir, config)
    print(json.dumps(manifest["windows"], ensure_ascii=False, indent=2))
    print(f"Preprocesamiento: {output_dir}")
    print(f"Features: {', '.join(manifest['features']['ordered_columns'])}")
    print(
        "Sensores constantes excluidos: "
        + (
            ", ".join(manifest["features"]["dropped_constant_sensors"])
            or "ninguno"
        )
    )


if __name__ == "__main__":
    main()
