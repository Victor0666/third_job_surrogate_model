"""Feasibility-first fuzzy episode and multi-seed metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .env_adapter import CEWSEnvAdapter


METRIC_FIELDS = (
    "deadline_violation_rate",
    "max_fuzzy_lateness",
    "mean_fuzzy_lateness",
    "fuzzy_energy_score",
)


def comparison_key(metrics: dict) -> tuple[float, float, float, float]:
    return tuple(float(metrics[name]) for name in METRIC_FIELDS)


def episode_metrics(adapter: CEWSEnvAdapter) -> dict:
    env = adapter.env
    lateness = []
    safety_margins = []
    rows = []
    for workflow in env.workflows:
        workflow_id = int(workflow.workflow_id)
        task_ids = adapter.workflow_task_ids(workflow_id)
        modal = max(
            (float(env.task_end_time[task_id]) for task_id in task_ids),
            default=float(workflow.arrival_time),
        )
        lower = max(
            (
                float(
                    env.shadow_task_end_time["optimistic"][task_id]
                )
                for task_id in task_ids
            ),
            default=float(workflow.arrival_time),
        )
        upper = max(
            (
                float(
                    env.shadow_task_end_time["pessimistic"][task_id]
                )
                for task_id in task_ids
            ),
            default=float(workflow.arrival_time),
        )
        if lower > modal + 1e-8 or modal > upper + 1e-8:
            raise ValueError("workflow fuzzy timeline order is invalid")
        risk = (
            (1.0 - float(env.fuzzy_deadline_eta)) * modal
            + float(env.fuzzy_deadline_eta) * upper
        )
        deadline = float(workflow.deadline)
        late = max(0.0, risk - deadline)
        lateness.append(late)
        safety_margins.append(deadline - risk)
        rows.append(
            {
                "workflow_id": workflow_id,
                "arrival_time": float(workflow.arrival_time),
                "deadline": deadline,
                "fuzzy_finish_lower": lower,
                "fuzzy_finish_modal": modal,
                "fuzzy_finish_upper": upper,
                "fuzzy_risk_finish": risk,
                "fuzzy_lateness": late,
                "feasible": bool(late <= 0.0),
            }
        )
    energy = env.get_fuzzy_energy_summary()
    count = len(lateness)
    violations = sum(value > 0.0 for value in lateness)
    result = {
        "seed": int(adapter.seed),
        "workflow_count": count,
        "deadline_violation_count": int(violations),
        "deadline_violation_rate": float(violations / max(count, 1)),
        "max_fuzzy_lateness": float(max(lateness, default=0.0)),
        "mean_fuzzy_lateness": float(np.mean(lateness))
        if lateness
        else 0.0,
        "feasible_workflow_ratio": float(
            (count - violations) / max(count, 1)
        ),
        "minimum_fuzzy_safety_margin": float(
            min(safety_margins, default=0.0)
        ),
        "fuzzy_energy_mean": float(
            energy["fuzzy_total_energy_mean"]
        ),
        "fuzzy_energy_std": float(
            energy["fuzzy_total_energy_std"]
        ),
        "fuzzy_energy_score": float(
            energy["fuzzy_total_energy_score"]
        ),
        "modal_energy": float(energy["energy_modal"]),
        "makespan": float(env.current_time),
        "assignment_count": int(adapter.assignment_count),
        "no_legal_vm_advance_count": int(
            adapter.no_legal_vm_advances
        ),
        "instance_fingerprint": adapter.instance_fingerprint(),
        "workflow_metrics": rows,
    }
    result["comparison_key"] = list(comparison_key(result))
    return result


def aggregate_seed_metrics(seed_metrics: Iterable[dict]) -> dict:
    metrics = list(seed_metrics)
    if not metrics:
        raise ValueError("at least one seed metric is required")
    rates = [float(row["deadline_violation_rate"]) for row in metrics]
    max_late = [float(row["max_fuzzy_lateness"]) for row in metrics]
    mean_late = [float(row["mean_fuzzy_lateness"]) for row in metrics]
    energies = [float(row["fuzzy_energy_score"]) for row in metrics]
    feasible = [rate <= 0.0 for rate in rates]
    aggregate = {
        "seed_count": len(metrics),
        "all_seed_feasible": bool(all(feasible)),
        "feasible_seed_rate": float(np.mean(feasible)),
        "worst_seed_violation": float(max(rates)),
        "worst_seed_lateness": float(max(max_late)),
        "deadline_violation_rate": float(np.mean(rates)),
        "max_fuzzy_lateness": float(max(max_late)),
        "mean_fuzzy_lateness": float(np.mean(mean_late)),
        "fuzzy_energy_score": float(np.mean(energies)),
        "fuzzy_energy_mean": float(
            np.mean([row["fuzzy_energy_mean"] for row in metrics])
        ),
        "fuzzy_energy_std": float(
            np.mean([row["fuzzy_energy_std"] for row in metrics])
        ),
        "seed_metrics": metrics,
    }
    aggregate["comparison_key"] = list(comparison_key(aggregate))
    return aggregate
