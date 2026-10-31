import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from predictguard.data.download import download_fd001
from predictguard.data.validate import COLUMN_NAMES, validate_fd001


def _write_fd001_files(data_dir):
    data_dir.mkdir(parents=True)
    values = np.zeros((100, len(COLUMN_NAMES)), dtype=np.int64)
    values[:, 0] = np.arange(1, 101)
    values[:, 1] = 1

    np.savetxt(data_dir / "train_FD001.txt", values, fmt="%d")
    np.savetxt(data_dir / "test_FD001.txt", values, fmt="%d")
    np.savetxt(data_dir / "RUL_FD001.txt", np.arange(100), fmt="%d")
    (data_dir / "readme.txt").write_text("Dataset readme\n", encoding="utf-8")


def test_validate_fd001_reports_shapes_and_hashes(tmp_path):
    data_dir = tmp_path / "raw"
    _write_fd001_files(data_dir)

    manifest = validate_fd001(data_dir)

    assert manifest["validation"]["train_rows"] == 100
    assert manifest["validation"]["test_rows"] == 100
    assert manifest["validation"]["test_rul_values"] == 100
    assert manifest["validation"]["columns_per_sensor_file"] == 26
    assert manifest["validation"]["test_rul_order"] == "unit_id 1-100"
    assert manifest["files"]["train_FD001.txt"]["sha256"] == hashlib.sha256(
        (data_dir / "train_FD001.txt").read_bytes()
    ).hexdigest()
    assert "readme.txt" in manifest["files"]


def test_validate_fd001_rejects_unexpected_column_count(tmp_path):
    data_dir = tmp_path / "raw"
    _write_fd001_files(data_dir)
    (data_dir / "test_FD001.txt").write_text("1 1 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="26 columnas"):
        validate_fd001(data_dir)


def test_validate_fd001_rejects_misaligned_rul_count(tmp_path):
    data_dir = tmp_path / "raw"
    _write_fd001_files(data_dir)
    (data_dir / "RUL_FD001.txt").write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="100 valores RUL"):
        validate_fd001(data_dir)


def test_download_extracts_only_fd001_and_records_archive_hash(
    tmp_path, monkeypatch
):
    archive_bytes = io.BytesIO()
    values = np.zeros((100, len(COLUMN_NAMES)), dtype=np.int64)
    values[:, 0] = np.arange(1, 101)
    values[:, 1] = 1
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        for filename, rows in (
            ("train_FD001.txt", values),
            ("test_FD001.txt", values),
            ("RUL_FD001.txt", np.arange(100).reshape(-1, 1)),
        ):
            content = io.StringIO()
            np.savetxt(content, rows, fmt="%d")
            archive.writestr(f"dataset/{filename}", content.getvalue())
        archive.writestr("dataset/readme.txt", "Dataset readme\n")
        archive.writestr("dataset/train_FD002.txt", "ignored")
    expected_archive = archive_bytes.getvalue()
    archive_hash = hashlib.sha256(expected_archive).hexdigest()

    def fake_urlopen(_request, timeout):
        assert timeout == 60
        return io.BytesIO(expected_archive)

    monkeypatch.setattr(
        "predictguard.data.download.urllib.request.urlopen", fake_urlopen
    )
    data_dir = tmp_path / "data" / "raw"

    manifest = download_fd001(data_dir)

    assert manifest["archive"]["sha256"] == archive_hash
    assert manifest["archive"]["size_bytes"] == len(expected_archive)
    assert json.loads((data_dir / "data_manifest.json").read_text()) == manifest
    assert (data_dir / "readme.txt").is_file()
    assert not (data_dir / "train_FD002.txt").exists()


def test_download_does_not_overwrite_existing_files(tmp_path):
    data_dir = tmp_path / "raw"
    _write_fd001_files(data_dir)
    original = (data_dir / "train_FD001.txt").read_bytes()

    with pytest.raises(FileExistsError, match="no se sobrescribirán"):
        download_fd001(data_dir)

    assert (data_dir / "train_FD001.txt").read_bytes() == original
