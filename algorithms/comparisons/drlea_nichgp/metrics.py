"""DRL-EA diagnostics over the same episode evaluator used by Safe-HRL."""

from typing import Iterable

from algorithms.llm_safe_hrl.hrl_mix.safe_metrics import (
    build_episode_metric_record, aggregate_safe_metric_records,
)
from algorithms.llm_safe_hrl.hrl_mix.model_selection import aggregate_seed_feasibility_metrics
from .env_adapter import CEWSEnvAdapter


METRIC_FIELDS = (
    "deadline_violation_rate", "max_fuzzy_lateness", "mean_fuzzy_lateness", "fuzzy_energy_score",
)


def comparison_key(metrics: dict) -> tuple[float, float, float, float]:
    return tuple(float(metrics[name]) for name in METRIC_FIELDS)


def episode_metrics(adapter: CEWSEnvAdapter) -> dict:
    env = adapter.env
    result = build_episode_metric_record(env, seed=adapter.seed, scheduling_time_seconds=0.0)
    if not result["evaluation_completed"]:
        raise ValueError("DRL-EA evaluation did not complete all expected workflows")
    rows = []
    for workflow in env.workflows:
        finish = env._workflow_finish_tfn(workflow.workflow_id)
        risk = env.fuzzy_deadline_measure(finish)
        late = max(0.0, risk - workflow.deadline)
        rows.append({
            "workflow_id": int(workflow.workflow_id), "arrival_time": float(workflow.arrival_time),
            "deadline": float(workflow.deadline), "fuzzy_finish_lower": float(finish.lower),
            "fuzzy_finish_modal": float(finish.modal), "fuzzy_finish_upper": float(finish.upper),
            "fuzzy_risk_finish": float(risk), "fuzzy_lateness": float(late), "feasible": late == 0.0,
        })
    result.update({
        "workflow_count": result["completed_workflow_count"], "makespan": float(env.current_time),
        "assignment_count": int(adapter.assignment_count),
        "no_legal_vm_advance_count": int(adapter.no_legal_vm_advances),
        "instance_fingerprint": adapter.instance_fingerprint(), "workflow_metrics": rows,
        "comparison_key": list(comparison_key(result)),
    })
    return result


def aggregate_seed_metrics(seed_metrics: Iterable[dict]) -> dict:
    records = list(seed_metrics)
    result = aggregate_safe_metric_records(records)
    result.update(aggregate_seed_feasibility_metrics(records))
    result.update(seed_count=len(records), seed_metrics=records, comparison_key=list(comparison_key(result)))
    return result
