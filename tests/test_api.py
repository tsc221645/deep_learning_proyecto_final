"""Integration and contract checks against the frozen Kaggle artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ROOT)) as http:
        yield http


def test_health_models_and_frozen_hashes(client):
    assert client.get("/health").json()["models_loaded"] == ["mlp", "lstm"]
    models = client.get("/models").json()
    assert {item["name"] for item in models} == {"mlp", "lstm"}
    selected = json.loads((ROOT / "artifacts/experiments/selected.json").read_text(encoding="utf-8"))
    for item in selected["files"].values():
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_engine_prediction_matches_custom_window(client):
    fleet = client.get("/engines").json()
    assert len(fleet) == 100
    history = client.get("/engines/1/history").json()
    assert "rul_actual_cycles" not in history["records"][0]
    endpoint = client.post("/compare", json={"unit_id": 1})
    assert endpoint.status_code == 200
    direct = client.post("/compare", json={"window": history["records"][-30:]})
    assert direct.status_code == 200
    for name in ("mlp", "lstm"):
        expected = endpoint.json()["predictions"][name]["rul_predicted"]
        assert expected >= 0
        assert direct.json()["predictions"][name]["rul_predicted"] == pytest.approx(expected, abs=1e-4)
        assert fleet[0]["predictions"][name] == pytest.approx(expected, abs=1e-4)


def test_bad_requests_and_missing_motor(client):
    assert client.get("/engines/999/history").status_code == 404
    assert client.post("/predict", json={"model": "cnn", "unit_id": 1}).status_code == 422
    assert client.post("/predict", json={"model": "mlp", "unit_id": 1, "window": [{"cycle": 1}]}).status_code == 422
    assert client.post("/predict", json={"model": "mlp", "unit_id": 1, "cycle": 9999}).status_code == 404
    assert client.post("/compare", json={"window": [{"cycle": 1}]}).status_code == 422
    assert client.post("/compare", json={"unit_id": 1, "unexpected": 1}).status_code == 422


def test_simulation_and_report_figures(client):
    response = client.post("/simulate-maintenance", json={})
    assert response.status_code == 200
    result = response.json()
    assert result["data_split"] == "validation complete trajectories"
    assert [item["policy"] for item in result["policies"]] == ["calendar", "mlp", "lstm"]
    assert all(item["engine_count"] == 20 for item in result["policies"])
    assert client.get("/evaluation/summary").json()["test"]["metrics"]["lstm"]["sample_count"] == 100
    assert client.get("/evaluation/figures/test_comparison.png").status_code == 200
    assert client.get("/evaluation/figures/unknown.png").status_code == 404
