"""Assistant contract: tools are grounded in the frozen models; no key falls back safely."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ROOT)) as http:
        yield http


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def events(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_status_without_key_is_offline(client):
    status = client.get("/assistant/status").json()
    assert status["available"] is False and status["mode"] == "offline"


def test_offline_chat_is_labelled_and_grounded(client):
    body = {"messages": [{"role": "user", "content": "¿Cómo está el motor?"}], "unit_id": 1, "threshold": 15}
    out = events(client.post("/assistant/chat", json=body))
    assert out[0]["type"] == "notice" and out[-1] == {"type": "done", "source": "offline"}
    text = "".join(e["delta"] for e in out if e["type"] == "text")
    rul = client.post("/compare", json={"unit_id": 1}).json()["predictions"]["mlp"]["rul_predicted"]
    assert f"{round(rul, 1)}" in text


def test_tools_match_api(client):
    assistant = client.app.state.assistant
    fleet = assistant.fleet_overview(15, 3)
    assert fleet["motores_totales"] == 100 and sum(fleet["conteo_por_estado"].values()) == 100
    assert len(fleet["mas_urgentes"]) == 3
    diagnosis = assistant.engine_diagnosis(1, 15)
    live = client.post("/compare", json={"unit_id": 1}).json()["predictions"]
    assert diagnosis["rul_lstm"] == round(live["lstm"]["rul_predicted"], 1)
    assert len(diagnosis["sensores_que_mas_cambiaron"]) == 4
    assert "error" in assistant.engine_diagnosis(999, 15)
    assert assistant.simulate_maintenance()["policies"][0]["policy"] == "calendar"
    assert "error" in assistant.simulate_maintenance(calendar_age=-5)
    assert "mlp" in assistant.model_metrics()["test"]


def test_tool_loop_and_cache(client, monkeypatch):
    assistant = client.app.state.assistant
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    calls = []

    class FakeStream:
        def __init__(self, final, chunks):
            self.final, self.text_stream = final, iter(chunks)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def get_final_message(self): return self.final

    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=50, cache_creation_input_tokens=0)
    tool = SimpleNamespace(type="tool_use", id="t1", name="get_engine_diagnosis", input={"unit_id": 1})
    rounds = [FakeStream(SimpleNamespace(stop_reason="tool_use", content=[tool], usage=usage), []),
              FakeStream(SimpleNamespace(stop_reason="end_turn", content=[], usage=usage), ["Listo ", "ya."])]

    def stream(**kwargs):
        calls.append(kwargs["messages"][-1])
        return rounds[len(calls) - 1]

    assistant._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    body = {"messages": [{"role": "user", "content": "diagnóstico"}], "unit_id": 1, "threshold": 15}
    out = events(client.post("/assistant/chat", json=body))
    assert [e["type"] for e in out] == ["tool", "text", "text", "done"] and out[-1]["source"] == "claude"
    assert out[-1]["usage"]["input_tokens"] == 200 and out[-1]["usage"]["api_calls"] == 2
    assert out[-1]["usage"]["estimated_cost_usd"] > 0
    result = json.loads(calls[1]["content"][0]["content"])
    assert result["unit_id"] == 1
    again = events(client.post("/assistant/chat", json=body))
    assert again[-1]["source"] == "cache" and len(calls) == 2


def test_api_failure_falls_back(client, monkeypatch):
    assistant = client.app.state.assistant
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    def boom(**kwargs): raise RuntimeError("sin red")
    assistant._client = SimpleNamespace(messages=SimpleNamespace(stream=boom))
    body = {"messages": [{"role": "user", "content": "hola otra"}], "unit_id": 2, "threshold": 15}
    out = events(client.post("/assistant/chat", json=body))
    assert out[-1] == {"type": "done", "source": "offline"} and any(e["type"] == "notice" for e in out)


def test_rejects_bad_payload(client):
    assert client.post("/assistant/chat", json={"messages": [], "threshold": 15}).status_code == 422
