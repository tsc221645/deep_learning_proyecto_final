"""Download the official NASA C-MAPSS FD001 source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from predictguard.data.validate import (
    DOWNLOAD_URL,
    README_FILE,
    REQUIRED_FILES,
    resolve_data_dir,
    validate_fd001,
)

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = 50 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def _download_archive(url: str, destination: Path, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PredictGuard/0.1 (NASA C-MAPSS data pipeline)"},
    )
    digest = hashlib.sha256()
    size_bytes = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with destination.open("xb") as archive_file:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > MAX_ARCHIVE_BYTES:
                    raise ValueError(
                        f"El archivo descargado excede el límite de "
                        f"{MAX_ARCHIVE_BYTES} bytes."
                    )
                digest.update(chunk)
                archive_file.write(chunk)

    if size_bytes == 0:
        raise ValueError("La descarga del archivo C-MAPSS está vacía.")
    return {"size_bytes": size_bytes, "sha256": digest.hexdigest()}


def _extract_fd001(
    archive_path: Path, staging_dir: Path
) -> list[str]:
    expected_names = (*REQUIRED_FILES, README_FILE)
    matches: dict[str, list[zipfile.ZipInfo]] = {
        name: [] for name in expected_names
    }

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            basename = PurePosixPath(info.filename.replace("\\", "/")).name
            if basename in matches and not info.is_dir():
                matches[basename].append(info)

        missing = [name for name in REQUIRED_FILES if not matches[name]]
        if missing:
            raise ValueError(
                "El ZIP no contiene los archivos FD001 requeridos: "
                + ", ".join(missing)
            )

        duplicates = [
            name for name, entries in matches.items() if len(entries) > 1
        ]
        if duplicates:
            raise ValueError(
                "El ZIP contiene nombres de archivo ambiguos: "
                + ", ".join(duplicates)
            )

        extracted_names: list[str] = []
        for name in expected_names:
            if not matches[name]:
                continue
            info = matches[name][0]
            if info.flag_bits & 0x1:
                raise ValueError(f"El archivo del ZIP está cifrado: {name}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(
                    f"El archivo del ZIP excede el límite permitido: {name}"
                )
            target = staging_dir / name
            with archive.open(info) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=DOWNLOAD_CHUNK_BYTES)
            if target.stat().st_size != info.file_size:
                raise ValueError(f"El tamaño extraído no coincide: {name}")
            extracted_names.append(name)

    return extracted_names


def download_fd001(
    data_dir: Path,
    *,
    url: str = DOWNLOAD_URL,
    timeout: int = 60,
) -> dict[str, Any]:
    """Download, validate, and install FD001 without overwriting local files."""
    if timeout <= 0:
        raise ValueError("El timeout debe ser mayor que cero.")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "data_manifest.json"

    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_manifest = validate_fd001(data_dir)
        if existing_manifest.get("files") != current_manifest["files"]:
            raise FileExistsError(
                f"El manifiesto existente no coincide con los datos actuales: "
                f"{manifest_path}. No se descargó ni sobrescribió ningún archivo."
            )
        return existing_manifest

    existing_files = [
        name for name in (*REQUIRED_FILES, README_FILE) if (data_dir / name).exists()
    ]
    if existing_files:
        raise FileExistsError(
            "Ya existen archivos en el destino y no se sobrescribirán: "
            + ", ".join(existing_files)
            + ". Ejecuta primero el validador para generar el manifiesto."
        )

    temporary_parent = data_dir.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".predictguard-download-", dir=temporary_parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        archive_path = temporary_root / "CMAPSSData.zip"
        archive_metadata = _download_archive(url, archive_path, timeout)
        downloaded_at = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        archive_metadata.update(
            {"url": url, "downloaded_at": downloaded_at}
        )

        staging_dir = temporary_root / "extracted"
        staging_dir.mkdir()
        extracted_names = _extract_fd001(archive_path, staging_dir)
        manifest = validate_fd001(
            staging_dir, archive_metadata=archive_metadata
        )
        manifest_path_in_staging = staging_dir / "data_manifest.json"
        with manifest_path_in_staging.open(
            "x", encoding="utf-8", newline="\n"
        ) as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
            manifest_file.write("\n")

        for name in extracted_names:
            (staging_dir / name).replace(data_dir / name)
        manifest_path_in_staging.replace(manifest_path)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descargar y validar NASA C-MAPSS FD001."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Ruta a la configuración del proyecto.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Tiempo máximo en segundos para cada operación de descarga.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout debe ser mayor que cero.")

    data_dir = resolve_data_dir(args.config)
    manifest = download_fd001(data_dir, timeout=args.timeout)
    print(json.dumps(manifest["validation"], ensure_ascii=False, indent=2))
    print(f"Manifiesto: {data_dir / 'data_manifest.json'}")


if __name__ == "__main__":
    main()
