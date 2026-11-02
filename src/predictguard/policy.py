"""Compare calendar, MLP, and LSTM policies on complete validation engines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from predictguard.costs import Costs, trajectory_cost
from predictguard.experiment_log import atomic_write_csv
from predictguard.train import _atomic_write_json


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"unit_id", "end_cycle", "rul_actual_cycles", "rul_predicted_cycles"}
    if not required <= set(frame.columns) or frame.duplicated(["unit_id", "end_cycle"]).any():
        raise ValueError(f"Predicciones de validación inválidas: {path}")
    return frame.sort_values(["unit_id", "end_cycle"])


def simulate(frame: pd.DataFrame, policy: str, threshold: int,
             costs: Costs, lead_time: int) -> pd.DataFrame:
    rows = []
    for unit_id, group in frame.groupby("unit_id", sort=True):
        failure_cycles = group["end_cycle"].to_numpy() + group["rul_actual_cycles"].to_numpy()
        if not np.allclose(failure_cycles, failure_cycles[0]):
            raise ValueError(f"RUL inconsistente para el motor {unit_id}.")
        failure_cycle = int(round(failure_cycles[0]))
        if policy == "calendar":
            intervention_cycle = threshold if threshold <= failure_cycle else None
            evaluations = 0
        else:
            signals = group[group["rul_predicted_cycles"] <= threshold]
            intervention_cycle = int(signals.iloc[0]["end_cycle"]) if len(signals) else None
            evaluations = int((group["end_cycle"] <= (intervention_cycle or failure_cycle)).sum())
        item = trajectory_cost(failure_cycle=failure_cycle, intervention_cycle=intervention_cycle,
                               evaluated_cycles=evaluations, costs=costs, lead_time=lead_time)
        rows.append({"unit_id": int(unit_id), "policy": policy, "threshold_cycles": threshold,
                     "failure_cycle": failure_cycle, "intervention_cycle": intervention_cycle,
                     "evaluated_cycles": evaluations, **item})
    return pd.DataFrame(rows)


def run(root: Path, config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    costs = Costs(**config["costs_q"])
    inference_by_policy = config.get("inference_q_by_policy", {})
    if any(float(value) < 0 for value in inference_by_policy.values()):
        raise ValueError("Los costos de inferencia deben ser no negativos.")
    lead_time = int(config["lead_time_cycles"])
    thresholds = list(map(int, config["thresholds_cycles"]))
    if not thresholds or min(thresholds) < 1 or len(set(thresholds)) != len(thresholds):
        raise ValueError("La lista de umbrales es inválida.")
    metrics = root / "artifacts" / "metrics"
    frames = {"mlp": _load_predictions(metrics / "mlp_validation_predictions.csv"),
              "lstm": _load_predictions(metrics / f"{config['lstm_run']}_validation_predictions.csv")}
    for policy in frames:
        if not frames[policy][["unit_id", "end_cycle", "rul_actual_cycles"]].reset_index(drop=True).equals(
            frames["mlp"][["unit_id", "end_cycle", "rul_actual_cycles"]].reset_index(drop=True)):
            raise ValueError("Las políticas no usan las mismas ventanas de validación.")
    if len(frames["mlp"]["unit_id"].unique()) != 20:
        raise ValueError("Se esperaban los 20 motores completos de validación.")
    curves = []
    details = {}
    for scenario, scenario_costs in [("base", costs)] + [
        (f"{key}_{factor:+.0%}", Costs(**{**costs.__dict__, key: getattr(costs, key) * (1 + factor)}))
        for key in ("intervention_q", "failure_q", "unused_cycle_q")
        for factor in (-0.5, -0.2, 0.2, 0.5)
    ]:
        for policy in ("calendar", "mlp", "lstm"):
            frame = frames["mlp"] if policy == "calendar" else frames[policy]
            policy_costs = Costs(**{**scenario_costs.__dict__,
                                    "inference_q": float(inference_by_policy.get(policy, scenario_costs.inference_q))})
            for threshold in thresholds:
                outcomes = simulate(frame, policy, threshold, policy_costs, lead_time)
                curves.append({"scenario": scenario, "policy": policy,
                               "threshold_cycles": threshold,
                               "mean_cost_q_per_engine": float(outcomes["total_q"].mean()),
                               "median_cost_q_per_engine": float(outcomes["total_q"].median()),
                               "total_cost_q": float(outcomes["total_q"].sum()),
                               "failures": int((outcomes["outcome"] == "failure").sum())})
                if scenario == "base":
                    details[(policy, threshold)] = outcomes
    curve = pd.DataFrame(curves)
    selected = curve.sort_values(["mean_cost_q_per_engine", "threshold_cycles"]).groupby(
        ["scenario", "policy"], sort=False).first().reset_index()
    baseline = selected[selected["policy"] == "calendar"].set_index("scenario")["mean_cost_q_per_engine"]
    selected["calendar_cost_q_per_engine"] = selected["scenario"].map(baseline)
    selected["savings_q_per_engine"] = selected["calendar_cost_q_per_engine"] - selected["mean_cost_q_per_engine"]
    selected["savings_pct"] = np.where(selected["calendar_cost_q_per_engine"] > 0,
                                       100 * selected["savings_q_per_engine"] / selected["calendar_cost_q_per_engine"], np.nan)
    out = root / "artifacts" / "business"
    atomic_write_csv(out / "cost_curve.csv", curve)
    atomic_write_csv(out / "sensitivity.csv", selected)
    best_base = selected[selected["scenario"] == "base"]
    detail_frames = [details[(row.policy, int(row.threshold_cycles))] for row in best_base.itertuples()]
    atomic_write_csv(out / "engine_outcomes.csv", pd.concat(detail_frames, ignore_index=True))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 5))
    for policy in ("calendar", "mlp", "lstm"):
        rows = curve[(curve["scenario"] == "base") & (curve["policy"] == policy)]
        axis.plot(rows["threshold_cycles"], rows["mean_cost_q_per_engine"], label=policy.upper())
    axis.set(xlabel="Umbral (ciclos)", ylabel="Costo hipotético medio (Q/motor)",
             title="Sensibilidad al umbral en validación")
    axis.legend(); axis.grid(alpha=0.25)
    figure.tight_layout()
    figures = root / "artifacts" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "business_cost_curve.png", dpi=150)
    plt.close(figure)
    summary = {"data_split": "complete validation engines", "currency": "hypothetical GTQ",
               "threshold_selection": "minimum mean cost on validation; exploratory in-sample estimate",
               "lead_time_cycles": lead_time, "costs_q": costs.__dict__,
               "inference_q_by_policy": inference_by_policy,
               "selected_base": best_base.to_dict(orient="records"),
               "test_official_used": False}
    _atomic_write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/business.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.config), indent=2))


if __name__ == "__main__":
    main()
