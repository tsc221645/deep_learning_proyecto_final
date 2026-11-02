"""FastAPI entry point for PredictGuard."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from api.schemas import CompareRequest, PredictionRequest, SimulationRequest
from api.service import Runtime

logger = logging.getLogger("predictguard.api")
logging.basicConfig(level=os.getenv("PREDICTGUARD_LOG_LEVEL", "INFO"))


def create_app(root: Path | None = None) -> FastAPI:
    repo_root = (root or Path(os.getenv("PREDICTGUARD_ROOT", Path(__file__).resolve().parent.parent))).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = Runtime(repo_root)
        logger.info(json.dumps({"event": "models_loaded", "device": str(app.state.runtime.device),
                                "models": list(app.state.runtime.models)}))
        yield

    app = FastAPI(title="PredictGuard API", version="0.2.0", lifespan=lifespan,
                  description="Predicción de RUL sobre motores simulados NASA FD001; no apta para mantenimiento real.")

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        started = time.perf_counter()
        size = int(request.headers.get("content-length", "0") or 0)
        if size > 1_000_000:
            return JSONResponse(status_code=413, content={"detail": "Solicitud demasiado grande."})
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            logger.info(json.dumps({"event": "request", "method": request.method,
                                    "path": request.url.path, "status": status,
                                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                                    "request_bytes": size}))

    def runtime(request: Request) -> Runtime:
        return request.app.state.runtime

    @app.get("/health")
    def health(request: Request):
        service = runtime(request)
        return {"status": "ok", "models_loaded": list(service.models), "device": str(service.device)}

    @app.get("/models")
    def models(request: Request):
        service = runtime(request)
        return [{"name": name, "version": service.selected["files"][name]["sha256"][:12],
                 "window_size": service.window_size, "features": service.features,
                 "validation_mae_cycles": service.selected["validation_mae_cycles"][name],
                 "test_mae_cycles": service.test_metrics["metrics"][name]["mae_cycles"],
                 "checkpoint_bytes": (service.root / service.selected["files"][name]["path"]).stat().st_size}
                for name in ("mlp", "lstm")]

    @app.get("/engines")
    def engines(request: Request):
        return runtime(request).fleet

    @app.get("/engines/{unit_id}/history")
    def engine_history(unit_id: int, request: Request):
        try:
            return runtime(request).history(unit_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/predict")
    def predict(payload: PredictionRequest, request: Request):
        try:
            return runtime(request).predict(payload.model, payload)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/compare")
    def compare(payload: CompareRequest, request: Request):
        service = runtime(request)
        try:
            window, length, cycle, unit_id = service.input_window(payload)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        predictions = {}
        for name in ("mlp", "lstm"):
            value, latency = service._predict_scaled(name, window, length)
            predictions[name] = {"rul_predicted": value, "latency_ms": latency,
                                 "model_version": service.selected["files"][name]["sha256"][:12]}
        return {"unit_id": unit_id, "observed_cycle": cycle, "unit": "cycles",
                "predictions": predictions,
                "warnings": ["Datos simulados; no utilizar para mantenimiento real."]}

    @app.post("/simulate-maintenance")
    def simulate_maintenance(payload: SimulationRequest, request: Request):
        return runtime(request).simulate(payload)

    @app.get("/evaluation/summary")
    def evaluation_summary(request: Request):
        service = runtime(request)
        return {"test": service.test_metrics, "business_validation": service.business_summary,
                "dataset": "NASA C-MAPSS FD001", "time_unit": "cycles",
                "note": "Test: solo una etiqueta terminal por motor. Costos: simulación sobre validación."}

    @app.get("/evaluation/figures/{name}")
    def evaluation_figure(name: str, request: Request):
        allowed = {"mlp_learning_curves.png", "lstm_learning_curves.png",
                   "test_comparison.png", "business_cost_curve.png"}
        if name not in allowed:
            raise HTTPException(status_code=404, detail="Figura no disponible.")
        path = runtime(request).root / "artifacts" / "figures" / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Figura no disponible.")
        return FileResponse(path, media_type="image/png")

    return app


app = create_app()
