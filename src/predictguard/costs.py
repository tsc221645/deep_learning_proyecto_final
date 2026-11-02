"""Hypothetical maintenance costs in quetzales for simulated FD001 engines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    intervention_q: float
    failure_q: float
    unused_cycle_q: float
    inference_q: float = 0.0

    def __post_init__(self) -> None:
        if min(self.intervention_q, self.failure_q, self.unused_cycle_q, self.inference_q) < 0:
            raise ValueError("Los costos deben ser no negativos.")


def trajectory_cost(*, failure_cycle: int, intervention_cycle: int | None,
                    evaluated_cycles: int, costs: Costs, lead_time: int = 1) -> dict[str, float | str]:
    """One intervention avoids failure only when enough cycles remain for action."""
    if failure_cycle < 1 or evaluated_cycles < 0 or lead_time < 1:
        raise ValueError("Ciclos o tiempo de acción inválidos.")
    if intervention_cycle is not None and not 1 <= intervention_cycle <= failure_cycle:
        raise ValueError("El ciclo de intervención está fuera de la trayectoria.")
    timely = intervention_cycle is not None and failure_cycle - intervention_cycle >= lead_time
    intervention = costs.intervention_q if timely else 0.0
    failure = 0.0 if timely else costs.failure_q
    unused = costs.unused_cycle_q * (failure_cycle - intervention_cycle) if timely else 0.0
    inference = costs.inference_q * evaluated_cycles
    return {"outcome": "intervention" if timely else "failure",
            "intervention_q": intervention, "failure_q": failure,
            "unused_life_q": unused, "inference_q": inference,
            "total_q": intervention + failure + unused + inference}
