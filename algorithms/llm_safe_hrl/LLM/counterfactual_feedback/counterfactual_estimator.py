"""Read-only local one-step counterfactual estimator using environment APIs."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import time
from typing import Any

import numpy as np

from .cache import (
    LOCAL_ESTIMATOR_VERSION,
    CounterfactualCache,
    CounterfactualCacheKey,
)
from .schemas import (
    CounterfactualComparison,
    CounterfactualConfig,
    CounterfactualOutcome,
    DecisionTrace,
    canonical_hash,
)


ESTIMATOR_VERSION = LOCAL_ESTIMATOR_VERSION


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in sorted(vars(value).items())
            if isinstance(item, (str, int, float, bool, type(None), list, tuple, dict, np.ndarray, np.generic))
        }
    return repr(value)


def environment_state_fingerprint(environment) -> str:
    """Hash every scheduling field touched by ``assign_task``.

    The fingerprint covers task state, VM and fuzzy timelines, event queue,
    energy/load records, assignment maps/history, ready IDs, and mutable task
    scheduling attributes. It deliberately excludes caches populated by pure
    prediction helpers because those do not alter scheduling semantics.
    """
    names = (
        "task_state",
        "task_end_time",
        "vm_available_at",
        "event_heap",
        "shadow_vm_available_at",
        "shadow_task_start_time",
        "shadow_task_end_time",
        "_records",
        "shadow_records",
        "assignment_history",
        "task_assigned_vm",
        "task_assigned_host",
        "ready_task_ids",
        "total_energy",
        "current_time",
        "completed_workflows",
        "next_arrival_idx",
    )
    payload = {name: _plain(getattr(environment, name, None)) for name in names}
    task_rows = []
    for workflow in getattr(environment, "workflows", []):
        for task in getattr(workflow, "tasks", []):
            task_rows.append(
                {
                    name: _plain(getattr(task, name, None))
                    for name in (
                        "task_id",
                        "state",
                        "ready_time",
                        "assigned_server_id",
                        "assigned_vm_pc",
                        "assigned_vm_id",
                        "start_processing_time",
                        "end_processing_time",
                    )
                }
            )
    payload["task_objects"] = task_rows
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _successor_evidence(environment, task_id: int, critical_ids: set[int]) -> tuple[int, int]:
    workflow_id = int(environment.task_meta[task_id][0])
    local_to_global = {
        int(local_id): int(global_id)
        for global_id, (owner, local_id) in enumerate(environment.task_meta)
        if int(owner) == workflow_id
    }
    releasable = []
    for child_local in environment.task_children[task_id]:
        child_id = local_to_global.get(int(child_local))
        if child_id is None or environment.task_state[child_id] in {"Running", "Finished"}:
            continue
        other_parents = [parent for parent in environment.task_global_parents[child_id] if int(parent) != int(task_id)]
        if all(environment.task_state[parent] == "Finished" for parent in other_parents):
            releasable.append(child_id)
    return len(releasable), sum(child in critical_ids for child in releasable)


def _host_load(environment, host_id: int, vm_id: int, start_time: float) -> tuple[float, float]:
    active_pc = 0.0
    for vm_index in environment.host_to_vm_indices[host_id]:
        if float(environment.vm_available_at[vm_index]) > start_time + 1e-12:
            active_pc += float(environment.vms[environment.vm_ids[vm_index]].pc)
    total_pc = max(float(environment.hosts[host_id].total_pc), 1e-12)
    before = float(np.clip(active_pc / total_pc, 0.0, 1.0))
    after = float(np.clip((active_pc + float(environment.vms[vm_id].pc)) / total_pc, 0.0, 1.0))
    return before, after


class LocalCounterfactualEstimator:
    """Compare ready tasks under the exact current state and fixed VM rule."""

    def __init__(
        self,
        config: CounterfactualConfig,
        cache: CounterfactualCache | None = None,
        resource_config_hash: str = "",
    ):
        self.config = config
        self.cache = cache or CounterfactualCache(enabled=False)
        self.resource_config_hash = str(resource_config_hash)

    def estimate_outcome(self, environment, task_id: int) -> CounterfactualOutcome:
        # The counterfactual must use the same candidate set as the real
        # decision: idle VMs only, chosen host first. Ranking over busy VMs
        # would report an assignment the evaluator can never make.
        host_id, vm_id, details = environment.select_host_then_vm_deterministic(
            task_id,
            candidate_vm_ids=environment.idle_feasible_vm_ids(task_id),
        )
        workflow_id = int(environment.task_meta[task_id][0])
        deadline = float(environment.workflows[workflow_id].deadline)
        if hasattr(environment, "predict_task_vm_action_risk"):
            risk = environment.predict_task_vm_action_risk(task_id, vm_id)
            optimistic = float(risk["optimistic_finish"])
            modal = float(risk["modal_finish"])
            pessimistic = float(risk["pessimistic_finish"])
            safe_deadline = float(risk["task_safe_deadline"])
            violation = float(risk["predicted_violation_amount"])
            margin = float(risk["safety_margin"])
            remaining = float(risk["remaining_critical_path_risk"])
            workflow_optimistic = optimistic + float(
                risk.get("remaining_critical_path_lower", remaining)
            )
            workflow_modal = modal + float(
                risk.get("remaining_critical_path_modal", remaining)
            )
            workflow_pessimistic = pessimistic + float(
                risk.get("remaining_critical_path_upper", remaining)
            )
            ddl_risk = float(risk["risk_finish"]) + remaining
            critical_ids = set(map(int, risk.get("critical_path_task_ids_modal", [])))
            evidence_quality = "high"
        else:
            optimistic = modal = pessimistic = float(details["predicted_finish_time"])
            workflow_optimistic = workflow_modal = workflow_pessimistic = modal
            ddl_risk = workflow_modal
            safe_deadline = deadline
            violation = max(0.0, ddl_risk - deadline)
            margin = deadline - ddl_risk
            remaining = 0.0
            critical_ids = set()
            evidence_quality = "medium"
        start_time = modal - float(details["exec_time"]) - float(details["comm_time"])
        before, after = _host_load(environment, host_id, vm_id, start_time)
        released, critical_released = _successor_evidence(environment, task_id, critical_ids)
        fuzzy_energy = float(
            details.get(
                "fuzzy_incremental_energy_score",
                environment.estimate_incremental_energy_score(task_id, vm_id)
                if getattr(environment, "fuzzy_enabled", False)
                else details["incremental_energy"],
            )
        )
        width = max(0.0, pessimistic - optimistic)
        # The environment already subtracts the remaining critical path when
        # constructing task_safe_deadline. Reuse its margin so that the path is
        # not deducted a second time.
        joint_risk = width if margin <= self.config.slack_threshold else 0.0
        host_vm_finishes = [
            float(environment.vm_available_at[index])
            for index in environment.host_to_vm_indices[host_id]
        ]
        active_extension = max(0.0, modal - max(host_vm_finishes, default=float(environment.current_time)))
        return CounterfactualOutcome(
            task_id=int(task_id),
            workflow_id=workflow_id,
            vm_id=int(vm_id),
            host_id=host_id,
            execution_time=float(details["exec_time"]),
            communication_time=float(details["comm_time"]),
            queue_time=float(details["queue_time"]),
            predicted_finish=float(details["predicted_finish_time"]),
            optimistic_finish=optimistic,
            modal_finish=modal,
            pessimistic_finish=pessimistic,
            workflow_optimistic_finish=workflow_optimistic,
            workflow_modal_finish=workflow_modal,
            workflow_pessimistic_finish=workflow_pessimistic,
            ddl_risk=ddl_risk,
            task_safe_deadline=safe_deadline,
            predicted_violation=violation,
            safety_margin=margin,
            marginal_fuzzy_energy=fuzzy_energy,
            host_load_before=before,
            host_load_after=after,
            host_load_delta=after - before,
            released_successors=released,
            released_critical_successors=critical_released,
            remaining_critical_path=remaining,
            uncertainty_width=width,
            uncertainty_low_slack_risk=joint_risk,
            server_active_time_extension=active_extension,
            evidence_quality=evidence_quality,
        )

    def compare(
        self,
        environment,
        trace: DecisionTrace,
        alternative_task_id: int,
    ) -> CounterfactualComparison:
        ready_evidence = set(trace.ready_task_ids) or {
            int(item.task_id) for item in trace.candidate_tasks
        }
        if alternative_task_id not in ready_evidence:
            raise ValueError("counterfactual alternative must be ready in the traced state")
        before = environment_state_fingerprint(environment)
        cache_key = CounterfactualCacheKey(
            structure_hash=trace.structure_hash,
            frozen_rule_hash=trace.frozen_rule_hash,
            scenario_id=trace.scenario_id,
            seed=trace.seed,
            decision_state_hash=before,
            selected_task_id=trace.selected_task_id,
            alternative_task_id=int(alternative_task_id),
            counterfactual_config_hash=self.config.config_hash,
            resource_config_hash=self.resource_config_hash,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            cached.pop("state_immutable", None)
            cached["selected_outcome"] = CounterfactualOutcome(**cached["selected_outcome"])
            cached["alternative_outcome"] = CounterfactualOutcome(**cached["alternative_outcome"])
            return CounterfactualComparison(**cached)
        started = time.perf_counter()
        selected = self.estimate_outcome(environment, trace.selected_task_id)
        alternative = self.estimate_outcome(environment, int(alternative_task_id))
        after = environment_state_fingerprint(environment)
        if self.config.assert_state_immutability and before != after:
            raise RuntimeError("Local counterfactual estimator modified scheduling state")
        cost_deltas = {
            "ddl_risk_delta": selected.ddl_risk - alternative.ddl_risk,
            "predicted_violation_delta": selected.predicted_violation - alternative.predicted_violation,
            "marginal_fuzzy_energy_delta": selected.marginal_fuzzy_energy - alternative.marginal_fuzzy_energy,
            "communication_delta": selected.communication_time - alternative.communication_time,
            "predicted_finish_delta": selected.predicted_finish - alternative.predicted_finish,
            "pessimistic_finish_delta": selected.pessimistic_finish - alternative.pessimistic_finish,
            "remaining_critical_path_delta": selected.remaining_critical_path - alternative.remaining_critical_path,
            "uncertainty_risk_delta": selected.uncertainty_low_slack_risk - alternative.uncertainty_low_slack_risk,
        }
        ddl_regret = max(cost_deltas["predicted_violation_delta"], cost_deltas["ddl_risk_delta"])
        energy_regret = cost_deltas["marginal_fuzzy_energy_delta"]
        if ddl_regret >= 0.0 and energy_regret >= 0.0 and (ddl_regret > 0.0 or energy_regret > 0.0):
            dominance = "alternative_dominates"
        elif ddl_regret <= 0.0 and energy_regret <= 0.0 and (ddl_regret < 0.0 or energy_regret < 0.0):
            dominance = "selected_dominates"
        else:
            dominance = "tradeoff_or_equal"
        comparison = CounterfactualComparison(
            decision_id=trace.decision_id,
            selected_task_id=trace.selected_task_id,
            alternative_task_id=int(alternative_task_id),
            selected_outcome=selected,
            alternative_outcome=alternative,
            safety_margin_delta=alternative.safety_margin - selected.safety_margin,
            released_successor_delta=alternative.released_successors - selected.released_successors,
            released_critical_successor_delta=(
                alternative.released_critical_successors - selected.released_critical_successors
            ),
            evidence_quality=("high" if selected.evidence_quality == alternative.evidence_quality == "high" else "medium"),
            estimator_type="local_one_step",
            dominance=dominance,
            elapsed_ms=float((time.perf_counter() - started) * 1000.0),
            state_fingerprint_before=before,
            state_fingerprint_after=after,
            **cost_deltas,
        )
        payload = comparison.to_dict()
        payload["state_immutable"] = before == after
        self.cache.put(cache_key, payload)
        return comparison
