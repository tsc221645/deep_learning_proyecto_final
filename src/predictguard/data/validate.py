"""Validation and provenance reporting for NASA C-MAPSS FD001."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

CATALOG_URL = "https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data"
DOWNLOAD_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
DATASET_CITATION = (
    'A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation '
    'Data Set", NASA Prognostics Data Repository, NASA Ames Research Center.'
)
PAPER_CITATION = (
    "A. Saxena, K. Goebel, D. Simon, and N. Eklund (2008). "
    '"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", '
    "Proceedings of the 1st International Conference on Prognostics and Health "
    "Management."
)
LICENSE_NOTE = (
    "No explicit data license was found in the included readme or the cited "
    "NASA repository page. Do not infer a license."
)

COLUMN_NAMES = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{index}" for index in range(1, 22)]
)
TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"
README_FILE = "readme.txt"
REQUIRED_FILES = (TRAIN_FILE, TEST_FILE, RUL_FILE)
EXPECTED_ENGINES = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sensor_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Archivo requerido no encontrado: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"El archivo está vacío: {path}")

    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64)
    if frame.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"{path.name}: se esperaban {len(COLUMN_NAMES)} columnas y "
            f"se encontraron {frame.shape[1]}."
        )
    if frame.empty:
        raise ValueError(f"El archivo no contiene registros: {path}")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path.name}: contiene valores faltantes o no finitos.")

    frame.columns = COLUMN_NAMES
    for column in ("unit_id", "cycle"):
        values = frame[column].to_numpy()
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{path.name}: {column} debe contener enteros.")
        if (values < 1).any():
            raise ValueError(f"{path.name}: {column} debe ser mayor o igual a 1.")

    frame[["unit_id", "cycle"]] = frame[["unit_id", "cycle"]].astype(np.int64)
    if frame.duplicated(["unit_id", "cycle"]).any():
        raise ValueError(f"{path.name}: hay ciclos duplicados para un motor.")

    return frame


def _validate_engine_ids(frame: pd.DataFrame, path: Path) -> list[int]:
    engine_ids = frame["unit_id"].drop_duplicates().tolist()
    expected_ids = list(range(1, EXPECTED_ENGINES + 1))
    if sorted(engine_ids) != expected_ids:
        raise ValueError(
            f"{path.name}: se esperaban los motores 1-{EXPECTED_ENGINES}; "
            f"se encontraron {len(engine_ids)} motores."
        )
    return engine_ids


def _file_metadata(
    path: Path, *, rows: int | None = None, columns: list[str] | None = None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        metadata["rows"] = rows
    if columns is not None:
        metadata["columns"] = columns
    return metadata


def validate_fd001(
    data_dir: Path,
    *,
    archive_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate FD001 files and return a reproducible provenance manifest."""
    data_dir = Path(data_dir)
    train_path = data_dir / TRAIN_FILE
    test_path = data_dir / TEST_FILE
    rul_path = data_dir / RUL_FILE

    train = _read_sensor_table(train_path)
    test = _read_sensor_table(test_path)
    _validate_engine_ids(train, train_path)
    test_engine_ids = _validate_engine_ids(test, test_path)

    expected_test_order = list(range(1, EXPECTED_ENGINES + 1))
    if test_engine_ids != expected_test_order:
        raise ValueError(
            f"{TEST_FILE}: el orden de motores debe corresponder al orden "
            "de las etiquetas en RUL_FD001.txt (1-100)."
        )

    if not rul_path.is_file():
        raise FileNotFoundError(f"Archivo requerido no encontrado: {rul_path}")
    if rul_path.stat().st_size == 0:
        raise ValueError(f"El archivo está vacío: {rul_path}")
    rul_frame = pd.read_csv(rul_path, sep=r"\s+", header=None, dtype=np.float64)
    if rul_frame.shape != (EXPECTED_ENGINES, 1):
        raise ValueError(
            f"{RUL_FILE}: se esperaban {EXPECTED_ENGINES} valores RUL en "
            f"una columna y se encontró la forma {rul_frame.shape}."
        )
    rul_values = rul_frame.iloc[:, 0].to_numpy()
    if not np.isfinite(rul_values).all():
        raise ValueError(f"{RUL_FILE}: contiene valores faltantes o no finitos.")
    if not np.equal(rul_values, np.floor(rul_values)).all():
        raise ValueError(f"{RUL_FILE}: los valores RUL deben ser enteros.")
    if (rul_values < 0).any():
        raise ValueError(f"{RUL_FILE}: el RUL no puede ser negativo.")

    files = {
        TRAIN_FILE: _file_metadata(
            train_path, rows=len(train), columns=COLUMN_NAMES.copy()
        ),
        TEST_FILE: _file_metadata(
            test_path, rows=len(test), columns=COLUMN_NAMES.copy()
        ),
        RUL_FILE: _file_metadata(
            rul_path, rows=len(rul_frame), columns=["rul"]
        ),
    }
    readme_path = data_dir / README_FILE
    if readme_path.is_file():
        files[README_FILE] = _file_metadata(readme_path)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "NASA C-MAPSS FD001",
        "manifest_created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source": {
            "catalog_url": CATALOG_URL,
            "download_url": DOWNLOAD_URL,
            "dataset_citation": DATASET_CITATION,
            "paper_citation": PAPER_CITATION,
            "license": LICENSE_NOTE,
        },
        "archive": archive_metadata,
        "files": files,
        "validation": {
            "train_rows": len(train),
            "train_trajectories": train["unit_id"].nunique(),
            "train_cycle_range": [
                int(train["cycle"].min()),
                int(train["cycle"].max()),
            ],
            "test_rows": len(test),
            "test_trajectories": test["unit_id"].nunique(),
            "test_cycle_range": [
                int(test["cycle"].min()),
                int(test["cycle"].max()),
            ],
            "test_rul_values": len(rul_values),
            "test_rul_range": [
                int(rul_values.min()),
                int(rul_values.max()),
            ],
            "test_rul_order": "unit_id 1-100",
            "columns_per_sensor_file": len(COLUMN_NAMES),
            "sensor_columns": COLUMN_NAMES.copy(),
        },
    }
    if archive_metadata is not None:
        manifest["source"]["accessed_at"] = archive_metadata["downloaded_at"]
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest atomically without replacing an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"El manifiesto ya existe: {path}. Valida los datos sin "
            "sobrescribirlo o revisa el archivo manualmente."
        )
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
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def resolve_data_dir(config_path: Path) -> Path:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError(f"Configuración sin la sección 'data': {config_path}")

    configured_root = Path(config["data"]["root"])
    if configured_root.is_absolute():
        return configured_root
    return config_path.parent.parent / configured_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Validar los datos C-MAPSS FD001.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Ruta a la configuración del proyecto.",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.config)
    manifest = validate_fd001(data_dir)
    manifest_path = data_dir / "data_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("files") != manifest["files"]:
            raise FileExistsError(
                f"El manifiesto existente no coincide con los datos actuales: "
                f"{manifest_path}. No se sobrescribió."
            )
        manifest = existing_manifest
    else:
        write_manifest(manifest_path, manifest)

    print(json.dumps(manifest["validation"], ensure_ascii=False, indent=2))
    print(f"Manifiesto: {manifest_path}")


if __name__ == "__main__":
    main()
