"""Asistente de mantenimiento: un LLM (Claude) que consulta los modelos del proyecto.

El LLM no predice nada por sí mismo. Obtiene cada cifra llamando a herramientas que
leen los resultados reales de MLP/LSTM, el simulador de costos y las métricas
congeladas. Así la respuesta es una explicación fundamentada, no una invención.

Flujo: pregunta del usuario -> contexto de la app -> Claude (+ herramientas) ->
respuesta en streaming -> interfaz de chat. Si no hay conexión o llave, se responde
con un resumen determinista y se avisa claramente de que no proviene del LLM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from api.schemas import SimulationRequest

logger = logging.getLogger("predictguard.assistant")

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOOL_ROUNDS = 6
MAX_HISTORY = 12
MAX_MESSAGE_CHARS = 4000
CACHE_SIZE = 128
RATE_LIMIT_PER_MINUTE = 30

SENSOR_INFO = {
    "sensor_2": ("T24", "temperatura a la salida del compresor de baja presión (LPC)"),
    "sensor_3": ("T30", "temperatura a la salida del compresor de alta presión (HPC)"),
    "sensor_4": ("T50", "temperatura a la salida de la turbina de baja presión (LPT)"),
    "sensor_6": ("P15", "presión en el ducto de derivación"),
    "sensor_7": ("P30", "presión a la salida del HPC"),
    "sensor_8": ("Nf", "velocidad física del ventilador"),
    "sensor_9": ("Nc", "velocidad física del núcleo"),
    "sensor_11": ("Ps30", "presión estática a la salida del HPC"),
    "sensor_12": ("phi", "relación flujo de combustible / Ps30"),
    "sensor_13": ("NRf", "velocidad corregida del ventilador"),
    "sensor_14": ("NRc", "velocidad corregida del núcleo"),
    "sensor_15": ("BPR", "relación de derivación"),
    "sensor_17": ("htBleed", "entalpía del aire sangrado"),
    "sensor_20": ("W31", "refrigerante sangrado para la turbina de alta presión"),
    "sensor_21": ("W32", "refrigerante sangrado para la turbina de baja presión"),
}

SYSTEM_PROMPT = """Eres el Asistente de Mantenimiento de PredictGuard, una aplicación académica que \
estima la vida útil restante (RUL, en ciclos) de motores turbofan simulados del dataset NASA C-MAPSS FD001 \
con dos redes neuronales: una MLP y una LSTM. Hablas en español, claro y directo, para alguien que puede \
no ser experto en deep learning.

REGLAS
1. Nunca inventes cifras. Todo número (RUL, ciclos, costos, métricas, ranking) debe salir de una herramienta. \
Si no tienes el dato, llama a la herramienta; si ninguna lo da, dilo.
2. Los datos son simulados y los costos en quetzales son hipotéticos. No es una recomendación de mantenimiento \
real: dilo una vez, de forma breve, cuando des una recomendación.
3. Sé honesto con la incertidumbre. En test ambos modelos tienden a sobreestimar la vida restante (sesgo \
positivo), lo cual es el error más riesgoso. Si MLP y LSTM discrepan más de 15 ciclos, señálalo.
4. Si el usuario pregunta por "el motor actual" o "este motor", usa el motor del contexto de la pantalla.
5. Responde breve (idealmente menos de 180 palabras) con Markdown simple: **negritas**, listas con guiones (puedes usar tablas pequeñas para comparar varios motores) y, \
cuando aplique, una sección final "**Qué haría falta**" con 2 a 4 pasos concretos (por ejemplo inspeccionar el \
componente señalado, confirmar con el otro modelo, esperar más ciclos de datos, ajustar el umbral). Para \
preguntas conceptuales no uses esa sección.
6. Sé útil con preguntas sobre la app o conceptos, pero no inventes funciones que la app no tiene. Si la \
pregunta no tiene relación con el proyecto, redirígela amablemente.

GLOSARIO
- Ciclo: una unidad operativa del motor (un vuelo completo). Todo está en ciclos.
- RUL: ciclos que le quedan al motor antes de fallar.
- Umbral: RUL por debajo del cual se interviene. Estado: "bajo umbral" si RUL <= umbral, "vigilar" si \
RUL <= 2 x umbral, "sobre umbral" en otro caso. Se usa la menor de las dos estimaciones (MLP, LSTM).
- Calendario: política de referencia que interviene a una edad fija (125 ciclos) sin mirar el estado del motor.
- MAE: error absoluto medio en ciclos. RMSE: raíz del error cuadrático medio, castiga más los errores grandes.
- Sesgo: error medio con signo; positivo significa que el modelo sobreestima la vida restante (optimista).
- Score NASA (PHM08): suma de errores exponenciales, asimétrica: penaliza más sobreestimar la vida (el motor \
falla antes de lo previsto) que subestimarla. Menor es mejor.
- MLP: red de capas densas que ve la ventana de 30 ciclos aplanada, sin memoria temporal. LSTM: red recurrente \
con memoria que lee la ventana ciclo a ciclo.
- Simulador de costos: compara calendario vs. MLP vs. LSTM con costos hipotéticos (intervención, falla, ciclo \
desperdiciado) sobre 20 motores de validación. El ahorro es exploratorio y probablemente optimista.
- Las variables son sensores del motor; los cambios fuertes en ellos suelen acompañar el desgaste, pero un \
cambio en un sensor no prueba por sí solo qué componente falla.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_fleet_overview",
        "description": "Resumen de la flota de 100 motores de test: cuántos están bajo el umbral, en vigilancia o "
                       "sobre el umbral, y los motores más urgentes con la RUL estimada por MLP y LSTM.",
        "input_schema": {"type": "object", "properties": {
            "threshold": {"type": "integer", "description": "Umbral de atención en ciclos (por defecto el de la pantalla)."},
            "top_n": {"type": "integer", "description": "Cuántos motores urgentes listar (1-15, por defecto 8)."}}},
    },
    {
        "name": "get_engine_diagnosis",
        "description": "Diagnóstico de un motor: ciclos observados, RUL estimada por MLP y LSTM, estado frente al "
                       "umbral, posición en la flota, acuerdo entre modelos y los sensores que más cambiaron "
                       "(desviaciones estándar respecto del inicio de su vida).",
        "input_schema": {"type": "object", "properties": {
            "unit_id": {"type": "integer", "description": "Número de motor, 1 a 100."},
            "threshold": {"type": "integer", "description": "Umbral de atención en ciclos."}},
            "required": ["unit_id"]},
    },
    {
        "name": "simulate_maintenance",
        "description": "Ejecuta el simulador de costos (calendario vs MLP vs LSTM) sobre los motores de validación. "
                       "Todos los parámetros son opcionales; por defecto usa los valores base del proyecto. Los "
                       "costos son hipotéticos en quetzales.",
        "input_schema": {"type": "object", "properties": {
            "calendar_age": {"type": "integer"}, "mlp_threshold": {"type": "integer"},
            "lstm_threshold": {"type": "integer"}, "lead_time_cycles": {"type": "integer"},
            "intervention_q": {"type": "number"}, "failure_q": {"type": "number"},
            "unused_cycle_q": {"type": "number"}}},
    },
    {
        "name": "get_model_metrics",
        "description": "Métricas oficiales de test de MLP y LSTM (MAE, RMSE, score NASA, sesgo, percentiles de error) "
                       "con los intervalos bootstrap, más las validaciones del modelo seleccionado.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# USD por millón de tokens: (entrada, salida, lectura de caché, escritura de caché).
# Sonnet 5.5: $2 / $10, caché de lectura $0.20. Escritura de caché = 1.25 x entrada (tarifa estándar).
# Haiku 5.5: $0.10 / $0.50. Fuente: tabla de modelos de la documentación de Claude (octubre 2026).
PRICES_USD_PER_MTOK = {
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

TOOL_LABELS = {
    "get_fleet_overview": "Revisando la flota",
    "get_engine_diagnosis": "Diagnosticando el motor",
    "simulate_maintenance": "Corriendo el simulador de costos",
    "get_model_metrics": "Consultando las métricas de los modelos",
}


def _load_dotenv(root: Path) -> None:
    """Carga ANTHROPIC_* desde .env (ignorado por git) sin pisar variables ya definidas."""
    path = root / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("ANTHROPIC_") and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _event(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


class Assistant:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        _load_dotenv(runtime.root)
        self.model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._client: Any = None
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()

    # ---- estado -----------------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def status(self) -> dict[str, Any]:
        return {"available": self.available, "model": self.model if self.available else None,
                "mode": "claude" if self.available else "offline"}

    def client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(timeout=60.0, max_retries=2)
        return self._client

    def _allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._hits and now - self._hits[0] > 60:
                self._hits.popleft()
            if len(self._hits) >= RATE_LIMIT_PER_MINUTE:
                return False
            self._hits.append(now)
            return True

    # ---- herramientas (fuente de verdad de las cifras) -------------------------
    @staticmethod
    def _level(rul: float, threshold: int) -> str:
        if rul <= threshold:
            return "bajo umbral"
        if rul <= threshold * 2:
            return "vigilar"
        return "sobre umbral"

    def _queue(self) -> list[dict[str, Any]]:
        return sorted(self.runtime.fleet, key=lambda e: min(e["predictions"].values()))

    def fleet_overview(self, threshold: int = 15, top_n: int = 8) -> dict[str, Any]:
        threshold = max(1, int(threshold))
        top_n = max(1, min(int(top_n), 15))
        queue = self._queue()
        counts = {"bajo umbral": 0, "vigilar": 0, "sobre umbral": 0}
        for engine in queue:
            counts[self._level(min(engine["predictions"].values()), threshold)] += 1
        urgent = [{
            "unit_id": e["unit_id"], "ciclos_observados": e["last_observed_cycle"],
            "rul_mlp": round(e["predictions"]["mlp"], 1), "rul_lstm": round(e["predictions"]["lstm"], 1),
            "estado": self._level(min(e["predictions"].values()), threshold)} for e in queue[:top_n]]
        return {"umbral_ciclos": threshold, "motores_totales": len(queue), "conteo_por_estado": counts,
                "mas_urgentes": urgent, "unidad": "ciclos",
                "nota": "RUL estimada en el último ciclo observado de cada motor de test."}

    def engine_diagnosis(self, unit_id: int, threshold: int = 15) -> dict[str, Any]:
        rt = self.runtime
        frame = rt.groups.get(int(unit_id))
        if frame is None:
            return {"error": f"Motor {unit_id} no encontrado. Usa un número entre 1 y 100."}
        entry = next(e for e in rt.fleet if e["unit_id"] == int(unit_id))
        queue = self._queue()
        position = next(i for i, e in enumerate(queue) if e["unit_id"] == int(unit_id)) + 1
        mlp, lstm = entry["predictions"]["mlp"], entry["predictions"]["lstm"]
        worst = min(mlp, lstm)
        scaled = rt.scaler.transform(frame[rt.features])
        span = min(10, max(1, len(scaled) // 3))
        drift = scaled[-span:].mean(axis=0) - scaled[:span].mean(axis=0)
        ranking = np.argsort(-np.abs(drift))
        sensors = []
        for index in ranking:
            key = rt.features[int(index)]
            if key not in SENSOR_INFO:
                continue
            code, name = SENSOR_INFO[key]
            sensors.append({"sensor": code, "mide": name, "cambio_en_desviaciones_estandar": round(float(drift[index]), 2),
                            "sube_o_baja": "sube" if drift[index] > 0 else "baja"})
            if len(sensors) == 4:
                break
        gap = abs(mlp - lstm)
        return {"unit_id": int(unit_id), "ciclos_observados": entry["last_observed_cycle"],
                "rul_mlp": round(mlp, 1), "rul_lstm": round(lstm, 1), "rul_mas_baja": round(worst, 1),
                "umbral_ciclos": int(threshold), "estado": self._level(worst, int(threshold)),
                "prioridad_en_flota": f"{position} de {len(queue)} (1 = más urgente)",
                "diferencia_entre_modelos": round(gap, 1),
                "acuerdo_entre_modelos": "coinciden" if gap <= 15 else "discrepan: tratar con cautela",
                "sensores_que_mas_cambiaron": sensors,
                "nota": "Los cambios de sensores describen el desgaste observado; no identifican por sí solos "
                        "qué componente falla. No existe RUL real por ciclo para comparar."}

    def simulate_maintenance(self, **params: Any) -> dict[str, Any]:
        costs = {k: params[k] for k in ("intervention_q", "failure_q", "unused_cycle_q") if params.get(k) is not None}
        base = {k: params[k] for k in ("calendar_age", "mlp_threshold", "lstm_threshold", "lead_time_cycles")
                if params.get(k) is not None}
        try:
            request = SimulationRequest(**base, costs=costs)
        except Exception as error:  # parámetros fuera de rango
            return {"error": f"Parámetros inválidos: {error}"}
        result = self.runtime.simulate(request)
        for row in result["policies"]:
            row["mean_cost_q_per_engine"] = round(row["mean_cost_q_per_engine"], 2)
            row["total_cost_q"] = round(row["total_cost_q"], 2)
            row["savings_vs_calendar_q_per_engine"] = round(row["savings_vs_calendar_q_per_engine"], 2)
            if row["savings_vs_calendar_pct"] is not None:
                row["savings_vs_calendar_pct"] = round(row["savings_vs_calendar_pct"], 1)
        result["parametros"] = {**base, **costs}
        result["advertencia"] = ("Costos hipotéticos; umbrales elegidos sobre los mismos 20 motores de validación, "
                                 "por lo que el ahorro es exploratorio y probablemente optimista.")
        return result

    def model_metrics(self) -> dict[str, Any]:
        rt = self.runtime
        return {"test": rt.test_metrics["metrics"], "bootstrap_mae_95": rt.test_metrics["bootstrap_95pct_mae_cycles"],
                "validation_mae_cycles": rt.selected["validation_mae_cycles"], "unidad": "ciclos",
                "nota": "Test: una observación terminal por motor (100 motores). La diferencia MLP vs LSTM en MAE "
                        "tiene un intervalo bootstrap que incluye 0: no es concluyente. El score NASA sí favorece "
                        "con claridad a la LSTM."}

    def run_tool(self, name: str, args: dict[str, Any], default_threshold: int) -> dict[str, Any]:
        try:
            if name == "get_fleet_overview":
                return self.fleet_overview(args.get("threshold", default_threshold), args.get("top_n", 8))
            if name == "get_engine_diagnosis":
                return self.engine_diagnosis(args["unit_id"], args.get("threshold", default_threshold))
            if name == "simulate_maintenance":
                return self.simulate_maintenance(**args)
            if name == "get_model_metrics":
                return self.model_metrics()
        except Exception as error:
            logger.exception("tool_failed")
            return {"error": f"La herramienta falló: {error}"}
        return {"error": f"Herramienta desconocida: {name}"}

    # ---- respuesta sin LLM (respaldo) ------------------------------------------
    def offline_answer(self, unit_id: int | None, threshold: int) -> str:
        if unit_id:
            d = self.engine_diagnosis(unit_id, threshold)
            if "error" not in d:
                sensors = ", ".join(f"{s['sensor']} ({s['sube_o_baja']} {abs(s['cambio_en_desviaciones_estandar'])} σ)"
                                    for s in d["sensores_que_mas_cambiaron"][:3])
                return (f"**Motor #{d['unit_id']}** · estado: **{d['estado']}** (umbral {threshold} ciclos)\n\n"
                        f"- RUL estimada: MLP {d['rul_mlp']} · LSTM {d['rul_lstm']} ciclos ({d['acuerdo_entre_modelos']}).\n"
                        f"- Prioridad en la flota: {d['prioridad_en_flota']}.\n"
                        f"- Sensores que más cambiaron: {sensors}.\n\n"
                        "**Qué haría falta:** confirmar con una inspección el componente asociado a esos sensores y "
                        "contrastar ambas estimaciones antes de decidir.")
        o = self.fleet_overview(threshold, 5)
        lines = "\n".join(f"- Motor #{e['unit_id']}: MLP {e['rul_mlp']} · LSTM {e['rul_lstm']} ciclos ({e['estado']})"
                          for e in o["mas_urgentes"])
        c = o["conteo_por_estado"]
        return (f"**Flota** (umbral {threshold} ciclos): {c['bajo umbral']} bajo umbral, {c['vigilar']} en vigilancia, "
                f"{c['sobre umbral']} sobre el umbral.\n\nLos más urgentes:\n{lines}")

    # ---- chat con streaming ----------------------------------------------------
    @staticmethod
    def _sanitize(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        clean = [{"role": m["role"], "content": m["content"][:MAX_MESSAGE_CHARS]}
                 for m in messages if m["role"] in ("user", "assistant") and m["content"].strip()]
        clean = clean[-MAX_HISTORY:]
        while clean and clean[0]["role"] != "user":
            clean.pop(0)
        return clean

    def _cache_key(self, messages: list[dict[str, str]], unit_id: int | None, threshold: int) -> str:
        raw = json.dumps([messages, unit_id, threshold, self.model], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def estimate_cost_usd(self, usage: dict[str, int]) -> float | None:
        """Costo estimado en USD con tarifas públicas por millón de tokens (ver PRICES_USD_PER_MTOK)."""
        prices = PRICES_USD_PER_MTOK.get(self.model)
        if prices is None:
            return None
        p_in, p_out, p_read, p_write = prices
        return round((usage["input_tokens"] * p_in + usage["output_tokens"] * p_out
                      + usage["cache_read_input_tokens"] * p_read
                      + usage["cache_creation_input_tokens"] * p_write) / 1_000_000, 6)

    def chat(self, messages: list[dict[str, str]], unit_id: int | None, threshold: int, *,
             use_tools: bool = True, internal: bool = False) -> Iterator[str]:
        """Genera eventos NDJSON. `internal=True` (evaluación) omite caché y límite de uso."""
        messages = self._sanitize(messages)
        if not messages or messages[-1]["role"] != "user":
            yield _event(type="error", message="Escribe una pregunta para empezar.")
            return
        if not self.available:
            yield _event(type="notice", message="Modo sin conexión: no hay ANTHROPIC_API_KEY configurada. "
                                                "Esta respuesta es un resumen predefinido, no de un LLM.")
            yield _event(type="text", delta=self.offline_answer(unit_id, threshold))
            yield _event(type="done", source="offline")
            return

        key = self._cache_key(messages, unit_id, threshold)
        cached = None if internal else self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            for event in cached:
                yield _event(**event)
            yield _event(type="done", source="cache", model=self.model)
            return
        if not internal and not self._allow():
            yield _event(type="error", message="Demasiadas consultas seguidas. Espera unos segundos.")
            return

        recorded: list[dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0, "api_calls": 0}
        started = time.perf_counter()
        try:
            for event in self._claude_turns(messages, unit_id, threshold, usage, use_tools):
                if event["type"] in ("text", "tool"):
                    recorded.append(event)
                yield _event(**event)
        except Exception as error:
            logger.exception("assistant_failed")
            kind = type(error).__name__
            yield _event(type="notice", message=f"No se pudo contactar a Claude ({kind}). Se muestra un resumen "
                                                "predefinido, no generado por el LLM.")
            yield _event(type="text", delta="\n\n" + self.offline_answer(unit_id, threshold))
            yield _event(type="done", source="offline")
            return
        if not internal:
            with self._lock:
                self._cache[key] = recorded
                while len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
        latency_ms = round((time.perf_counter() - started) * 1000)
        cost = self.estimate_cost_usd(usage)
        logger.info(json.dumps({"event": "assistant_reply", "model": self.model, "latency_ms": latency_ms,
                                "tool_calls": sum(1 for e in recorded if e["type"] == "tool"),
                                "estimated_cost_usd": cost, **usage}))
        yield _event(type="done", source="claude", model=self.model,
                     usage={**usage, "latency_ms": latency_ms, "estimated_cost_usd": cost})

    def _claude_turns(self, messages: list[dict[str, str]], unit_id: int | None, threshold: int,
                      usage: dict[str, int], use_tools: bool = True) -> Iterator[dict[str, Any]]:
        screen = (f"Contexto de la pantalla: umbral de atención = {threshold} ciclos. "
                  + (f"El usuario está viendo el motor #{unit_id}." if unit_id else "El usuario no ha abierto un motor."))
        system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
                  {"type": "text", "text": screen}]
        # Nota: Claude Sonnet 5.5 no admite temperature/top_p distintos del valor por defecto (400),
        # por eso no se fijan; la determinación se acota con max_tokens y reglas en el prompt.
        extra = {"tools": TOOLS} if use_tools else {}
        convo: list[dict[str, Any]] = [dict(m) for m in messages]
        for _ in range(MAX_TOOL_ROUNDS):
            with self.client().messages.stream(model=self.model, max_tokens=1500, system=system,
                                               messages=convo, **extra) as stream:
                for delta in stream.text_stream:
                    yield {"type": "text", "delta": delta}
                final = stream.get_final_message()
            usage["api_calls"] += 1
            for field in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                usage[field] += int(getattr(final.usage, field, 0) or 0)
            if final.stop_reason != "tool_use":
                return
            convo.append({"role": "assistant", "content": final.content})
            results = []
            for block in final.content:
                if block.type != "tool_use":
                    continue
                yield {"type": "tool", "name": block.name, "label": TOOL_LABELS.get(block.name, block.name),
                       "input": block.input}
                output = self.run_tool(block.name, dict(block.input), threshold)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(output, ensure_ascii=False)})
            convo.append({"role": "user", "content": results})
        yield {"type": "text", "delta": "\n\n_Se alcanzó el máximo de consultas internas; reformula la pregunta._"}
