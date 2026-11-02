"""Load frozen models and serve reproducible FD001 inference."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from predictguard.costs import Costs
from predictguard.data.validate import COLUMN_NAMES
from predictguard.models.lstm import load_lstm_checkpoint
from predictguard.models.mlp import load_mlp_checkpoint
from predictguard.policy import simulate

from api.schemas import SimulationRequest, SourceInput


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_file_matches(path: Path, expected_sha256: str) -> bool:
    """Allow Git's CRLF checkout of NASA's LF text without changing values."""
    if file_sha256(path) == expected_sha256:
        return True
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected_sha256


class Runtime:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        selected = json.loads((self.root / "artifacts/experiments/selected.json").read_text(encoding="utf-8"))
        for item in selected["files"].values():
            path = (self.root / item["path"]).resolve()
            if not path.is_relative_to(self.root) or not raw_file_matches(path, item["sha256"]):
                raise ValueError(f"Artefacto congelado inválido: {item['path']}")
        self.selected = selected
        self.manifest = json.loads((self.root / selected["files"]["preprocessing"]["path"]).read_text(encoding="utf-8"))
        self.features = list(selected["features"])
        self.window_size = int(selected["window_size"])
        if self.features != self.manifest["features"]["ordered_columns"] or self.window_size != self.manifest["windows"]["size"]:
            raise ValueError("Features o ventana incompatibles con el manifiesto.")
        scaler_path = self.root / "artifacts/processed/fd001/scaler.joblib"
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Falta {scaler_path}; cópialo desde Kaggle antes de iniciar la API.")
        self.scaler = joblib.load(scaler_path)
        if list(self.scaler.feature_names_in_) != self.features:
            raise ValueError("El orden de features del scaler no coincide.")
        if not np.allclose(self.scaler.mean_, self.manifest["scaler"]["mean"], rtol=1e-6, atol=1e-8):
            raise ValueError("El scaler no coincide con el manifiesto.")
        if not np.allclose(self.scaler.scale_, self.manifest["scaler"]["scale"], rtol=1e-6, atol=1e-8):
            raise ValueError("La escala no coincide con el manifiesto.")
        threads = int(os.getenv("PREDICTGUARD_TORCH_THREADS", "2"))
        torch.set_num_threads(max(1, threads))
        device_name = os.getenv("PREDICTGUARD_DEVICE", "cpu")
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA pero no hay GPU disponible.")
        self.models = {
            "mlp": load_mlp_checkpoint(self.root / selected["files"]["mlp"]["path"], self.device)[0],
            "lstm": load_lstm_checkpoint(self.root / selected["files"]["lstm"]["path"], self.device)[0],
        }
        for model in self.models.values():
            model.eval()
        test_path = self.root / "data/raw/test_FD001.txt"
        if not raw_file_matches(test_path, self.manifest["source_file_sha256"]["test_FD001.txt"]):
            raise ValueError("El test FD001 no coincide con el manifiesto.")
        self.test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
        if self.test.shape[1] != 26 or self.test.unit_id.nunique() != 100:
            raise ValueError("Los datos FD001 de demo no tienen el esquema esperado.")
        self.test.sort_values(["unit_id", "cycle"], inplace=True, ignore_index=True)
        self.groups = {int(unit): frame.reset_index(drop=True) for unit, frame in self.test.groupby("unit_id")}
        self.test_metrics = json.loads((self.root / "artifacts/metrics/test_metrics.json").read_text(encoding="utf-8"))
        self.business_summary = json.loads((self.root / "artifacts/business/summary.json").read_text(encoding="utf-8"))
        self.business_config = yaml.safe_load((self.root / "configs/business.yaml").read_text(encoding="utf-8"))
        self.validation_predictions = {
            "mlp": pd.read_csv(self.root / "artifacts/metrics/mlp_validation_predictions.csv"),
            "lstm": pd.read_csv(self.root / "artifacts/metrics/lstm_validation_predictions.csv"),
        }
        # Precompute the overview once. Neither official test labels nor
        # historical labels are exposed as if they were available per cycle.
        self.fleet = []
        for unit, frame in self.groups.items():
            window, length, _ = self.engine_window(unit, int(frame.cycle.iloc[-1]))
            predictions = {name: self._predict_scaled(name, window, length)[0] for name in self.models}
            self.fleet.append({"unit_id": unit, "last_observed_cycle": int(frame.cycle.iloc[-1]),
                               "predictions": predictions, "unit": "cycles"})

    def engine_window(self, unit_id: int, cycle: int | None) -> tuple[np.ndarray, int, int]:
        frame = self.groups.get(unit_id)
        if frame is None:
            raise KeyError(f"Motor {unit_id} no encontrado.")
        if cycle is None:
            cycle = int(frame.cycle.iloc[-1])
        observed = frame[frame.cycle <= cycle]
        if observed.empty or int(observed.cycle.iloc[-1]) != cycle:
            raise KeyError(f"El motor {unit_id} no tiene el ciclo {cycle} observado.")
        observed = observed.tail(self.window_size)
        values = self.scaler.transform(observed[self.features]).astype(np.float32)
        length = len(values)
        if length < self.window_size:
            padding = np.repeat(values[:1], self.window_size - length, axis=0)
            values = np.concatenate((padding, values), axis=0)
        return values, length, cycle

    def custom_window(self, rows: list[dict[str, float]]) -> tuple[np.ndarray, int, int]:
        if not 1 <= len(rows) <= self.window_size:
            raise ValueError(f"window debe tener entre 1 y {self.window_size} ciclos.")
        expected = set(self.features)
        for index, row in enumerate(rows):
            if set(row) != expected:
                raise ValueError(f"Fila {index}: se requieren exactamente las features {self.features}.")
        frame = pd.DataFrame(rows, columns=self.features)
        values = frame.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("La ventana contiene valores no finitos.")
        cycles = frame["cycle"].to_numpy(dtype=np.float64)
        if not np.equal(cycles, np.floor(cycles)).all() or (np.diff(cycles) != 1).any():
            raise ValueError("Los ciclos de la ventana deben ser enteros consecutivos.")
        scaled = self.scaler.transform(frame).astype(np.float32)
        length = len(scaled)
        if length < self.window_size:
            scaled = np.concatenate((np.repeat(scaled[:1], self.window_size - length, axis=0), scaled), axis=0)
        return scaled, length, int(cycles[-1])

    def input_window(self, source: SourceInput) -> tuple[np.ndarray, int, int, int | None]:
        if source.unit_id is not None:
            window, length, cycle = self.engine_window(source.unit_id, source.cycle)
            return window, length, cycle, source.unit_id
        assert source.window is not None
        window, length, cycle = self.custom_window(source.window)
        return window, length, cycle, None

    def _predict_scaled(self, name: str, window: np.ndarray, length: int) -> tuple[float, float]:
        started = time.perf_counter()
        model = self.models[name]
        x = torch.from_numpy(window[None]).to(self.device)
        with torch.inference_mode():
            if name == "mlp":
                raw = model(x.flatten(1))
            else:
                raw = model(x, torch.tensor([length], device=self.device))
            prediction = float(raw.clamp_min(0).item())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return prediction, (time.perf_counter() - started) * 1000

    def predict(self, name: str, source: SourceInput) -> dict[str, Any]:
        window, length, cycle, unit_id = self.input_window(source)
        prediction, latency = self._predict_scaled(name, window, length)
        return {"model": name, "model_version": self.selected["files"][name]["sha256"][:12],
                "unit_id": unit_id, "observed_cycle": cycle, "rul_predicted": prediction,
                "unit": "cycles", "latency_ms": latency,
                "warnings": ["Datos simulados; no utilizar para mantenimiento real."]}

    def history(self, unit_id: int) -> dict[str, Any]:
        frame = self.groups.get(unit_id)
        if frame is None:
            raise KeyError(f"Motor {unit_id} no encontrado.")
        return {"unit_id": unit_id, "last_observed_cycle": int(frame.cycle.iloc[-1]),
                "features": self.features, "records": frame[self.features].to_dict(orient="records"),
                "rul_truth_policy": "Solo existe una etiqueta oficial para el último ciclo observado."}

    def simulate(self, request: SimulationRequest) -> dict[str, Any]:
        costs = request.costs
        rows = []
        for policy, threshold, inference_q in (
            ("calendar", request.calendar_age, 0.0),
            ("mlp", request.mlp_threshold, costs.mlp_inference_q),
            ("lstm", request.lstm_threshold, costs.lstm_inference_q),
        ):
            frame = self.validation_predictions["mlp" if policy == "calendar" else policy]
            outcome = simulate(frame, policy, threshold,
                               Costs(costs.intervention_q, costs.failure_q, costs.unused_cycle_q, inference_q),
                               request.lead_time_cycles)
            rows.append({"policy": policy, "threshold_cycles": threshold,
                         "mean_cost_q_per_engine": float(outcome.total_q.mean()),
                         "total_cost_q": float(outcome.total_q.sum()),
                         "failures": int((outcome.outcome == "failure").sum()),
                         "engine_count": len(outcome)})
        calendar = rows[0]["mean_cost_q_per_engine"]
        for row in rows:
            row["savings_vs_calendar_q_per_engine"] = calendar - row["mean_cost_q_per_engine"]
            row["savings_vs_calendar_pct"] = 100 * (calendar - row["mean_cost_q_per_engine"]) / calendar if calendar > 0 else None
        return {"data_split": "validation complete trajectories", "currency": "hypothetical GTQ",
                "costs_are_hypothetical": True, "thresholds_selected_on_validation": True,
                "policies": rows}
