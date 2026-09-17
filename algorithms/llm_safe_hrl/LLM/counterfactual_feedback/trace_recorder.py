"""Deterministic decision tracing and representative alternative selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import CandidateTaskSnapshot, CounterfactualConfig, DecisionTrace


FEATURE_NAMES = (
    "min_exec_time",
    "min_comm_time",
    "min_incremental_energy",
    "slack",
    "upward_rank",
    "remaining_work",
    "ready_wait_time",
    "uncertainty",
)


def _host_load_summary(environment) -> dict[str, float]:
    loads = []
    for host_id in environment.host_ids:
        active = 0.0
        for vm_index in environment.host_to_vm_indices[host_id]:
            if float(environment.vm_available_at[vm_index]) > float(environment.current_time) + 1e-12:
                active += float(environment.vms[environment.vm_ids[vm_index]].pc)
        loads.append(active / max(float(environment.hosts[host_id].total_pc), 1e-12))
    return {
        "mean": float(np.mean(loads)) if loads else 0.0,
        "maximum": float(max(loads, default=0.0)),
        "active_host_ratio": float(np.mean(np.asarray(loads) > 0.0)) if loads else 0.0,
    }


def _workflow_id(environment, task_id: int) -> int:
    return int(environment.task_meta[int(task_id)][0])


def _workflow_type(environment, task_id: int) -> str:
    workflow_id = _workflow_id(environment, task_id)
    workflow = environment.workflows[workflow_id]
    dax_path = getattr(workflow, "dax_path", None)
    if dax_path:
        stem = Path(str(dax_path)).stem
        return stem.split("_", 1)[0] or stem
    return f"workflow_{workflow_id}"


def _potential_release_count(environment, task_id: int) -> int:
    workflow_id = _workflow_id(environment, task_id)
    local_to_global = {
        int(local_id): int(global_id)
        for global_id, (owner, local_id) in enumerate(environment.task_meta)
        if int(owner) == workflow_id
    }
    count = 0
    for child_local in environment.task_children[task_id]:
        child_id = local_to_global.get(int(child_local))
        if child_id is None or environment.task_state[child_id] in {"Running", "Finished"}:
            continue
        other_parents = [
            parent
            for parent in environment.task_global_parents[child_id]
            if int(parent) != int(task_id)
        ]
        if all(environment.task_state[parent] == "Finished" for parent in other_parents):
            count += 1
    return count


@dataclass
class RecordedDecision:
    trace: DecisionTrace
    alternative_task_ids: list[int]


class TraceRecorder:
    """Records bounded traces without changing task, Host, or VM choices."""

    def __init__(self, config: CounterfactualConfig):
        self.config = config
        self.traced_count = 0
        self._previous_selected_risk: float | None = None

    def _ordered_ids(self, ready_ids: list[int], scores: np.ndarray) -> list[int]:
        order = np.lexsort((np.arange(len(scores), dtype=int), scores))
        return [int(ready_ids[index]) for index in order]

    def select_alternatives(
        self,
        ready_ids: list[int],
        selected_task_id: int,
        scores: np.ndarray,
        features: dict[str, np.ndarray],
    ) -> list[int]:
        """Choose ready alternatives in a fixed evidence-oriented order."""
        if selected_task_id not in ready_ids:
            raise ValueError("selected task must belong to the ready set")
        ordered = self._ordered_ids(ready_ids, scores)
        proposals: list[int] = []
        if self.config.include_rule_runner_up and len(ordered) > 1:
            proposals.append(ordered[1])
        selectors = (
            (self.config.include_min_slack, "slack", np.argmin),
            (self.config.include_max_upward_rank, "upward_rank", np.argmax),
            (self.config.include_min_energy, "min_incremental_energy", np.argmin),
            (self.config.include_max_uncertainty, "uncertainty", np.argmax),
        )
        for enabled, name, selector in selectors:
            if enabled:
                proposals.append(int(ready_ids[int(selector(features[name]))]))
        result = []
        for task_id in proposals:
            if task_id == selected_task_id or task_id not in ready_ids or task_id in result:
                continue
            result.append(task_id)
            if len(result) >= self.config.max_alternatives_per_decision:
                break
        return result

    def record(
        self,
        environment,
        ready_tasks,
        selection_details: dict[str, Any],
        metadata: dict[str, Any],
        decision_index: int,
        *,
        store_full_ready_features: bool = False,
    ) -> RecordedDecision | None:
        if self.traced_count >= self.config.max_traced_decisions_per_run:
            return None
        ready_ids = [int(environment._task_id(value)) for value in ready_tasks]
        selected = int(selection_details["selected_task_id"])
        scores = np.asarray(selection_details["scores"], dtype=float)
        features = {
            name: np.asarray(selection_details["features"][name], dtype=float)
            for name in FEATURE_NAMES
        }
        ordered = self._ordered_ids(ready_ids, scores)
        selected_index = ready_ids.index(selected)
        selected_rank = ordered.index(selected) + 1
        runner_margin = (
            float(abs(scores[ready_ids.index(ordered[1])] - scores[selected_index]))
            if len(ordered) > 1 else 0.0
        )
        relative_margin = runner_margin / max(
            abs(float(scores[selected_index])),
            abs(float(scores[ready_ids.index(ordered[1])])) if len(ordered) > 1 else 0.0,
            1.0,
        )
        min_slack_index = int(np.argmin(features["slack"]))
        max_rank_index = int(np.argmax(features["upward_rank"]))
        reasons = []
        if float(features["slack"][min_slack_index]) <= self.config.slack_threshold:
            reasons.append("low_or_negative_slack_present")
        if selected_index != min_slack_index:
            reasons.append("selected_not_minimum_slack")
        if selected_index != max_rank_index:
            reasons.append("selected_not_maximum_upward_rank")
        if len(ready_ids) > 1 and relative_margin <= self.config.score_margin_threshold:
            reasons.append("close_rule_scores")
        if len(ready_ids) >= self.config.queue_congestion_threshold:
            reasons.append("ready_queue_congestion")
        if float(np.max(features["uncertainty"])) >= self.config.uncertainty_threshold:
            reasons.append("high_uncertainty")

        min_energy_index = int(np.argmin(features["min_incremental_energy"]))
        selected_energy = float(features["min_incremental_energy"][selected_index])
        min_energy = float(features["min_incremental_energy"][min_energy_index])
        relative_energy_gap = abs(selected_energy - min_energy) / max(
            abs(selected_energy), abs(min_energy), 1.0
        )
        if (
            min_energy_index != selected_index
            and relative_energy_gap <= self.config.score_margin_threshold
            and abs(
                float(features["slack"][selected_index])
                - float(features["slack"][min_energy_index])
            ) > self.config.slack_threshold
        ):
            reasons.append("small_energy_gap_possible_ddl_risk_difference")

        release_counts = [
            _potential_release_count(environment, task_id)
            for task_id in ready_ids
        ]
        if max(release_counts, default=0) > min(release_counts, default=0):
            reasons.append("successor_release_difference")

        # Same candidate set as the evaluator: idle VMs only, host chosen first.
        _selected_host, _selected_vm, vm_details = (
            environment.select_host_then_vm_deterministic(
                selected,
                candidate_vm_ids=environment.idle_feasible_vm_ids(selected),
            )
        )
        current_risk = float(vm_details.get("deadline_violation", 0.0))
        if (
            self._previous_selected_risk is not None
            and current_risk > self._previous_selected_risk
            + max(abs(self._previous_selected_risk), 1.0) * 0.1
        ):
            reasons.append("ddl_risk_jump")
        self._previous_selected_risk = current_risk
        if selected_index != min_slack_index and selected_index != max_rank_index:
            reasons.append("critical_path_competition")

        criticality_score = 0.0
        minimum_slack = float(features["slack"][min_slack_index])
        if "low_or_negative_slack_present" in reasons:
            criticality_score += 4.0 + min(4.0, max(0.0, -minimum_slack))
        if "selected_not_minimum_slack" in reasons:
            slack_gap = float(features["slack"][selected_index]) - minimum_slack
            criticality_score += 2.0 + min(3.0, max(0.0, slack_gap))
        if "selected_not_maximum_upward_rank" in reasons:
            criticality_score += 1.5
        if "close_rule_scores" in reasons:
            criticality_score += 1.0 + max(
                0.0,
                self.config.score_margin_threshold - relative_margin,
            )
        if "ready_queue_congestion" in reasons:
            criticality_score += min(
                2.0,
                len(ready_ids) / max(self.config.queue_congestion_threshold, 1),
            )
        if "high_uncertainty" in reasons:
            criticality_score += min(
                3.0,
                float(np.max(features["uncertainty"]))
                / max(self.config.uncertainty_threshold, 1e-12),
            )
        if "small_energy_gap_possible_ddl_risk_difference" in reasons:
            criticality_score += 1.5
        if "successor_release_difference" in reasons:
            criticality_score += 1.0 + min(
                2.0,
                max(release_counts, default=0) - min(release_counts, default=0),
            )
        if "ddl_risk_jump" in reasons:
            criticality_score += 3.0
        if "critical_path_competition" in reasons:
            criticality_score += 1.0

        is_critical = bool(reasons)
        alternatives = self.select_alternatives(ready_ids, selected, scores, features) if is_critical else []
        detailed_ids = [selected, *alternatives]
        if self.config.store_full_ready_features or store_full_ready_features:
            detailed_ids = list(ready_ids)
        upward_cut = float(np.quantile(features["upward_rank"], 0.75))
        snapshots = []
        for task_id in detailed_ids:
            index = ready_ids.index(task_id)
            feature_values = {name: float(features[name][index]) for name in FEATURE_NAMES}
            snapshots.append(
                CandidateTaskSnapshot(
                    task_id=task_id,
                    features=feature_values,
                    rule_score=float(scores[index]),
                    rule_rank=ordered.index(task_id) + 1,
                    workflow_id=_workflow_id(environment, task_id),
                    low_or_negative_slack=feature_values["slack"] <= self.config.slack_threshold,
                    high_upward_rank=feature_values["upward_rank"] >= upward_cut,
                    uncertainty_level=feature_values["uncertainty"],
                )
            )
        run_id = str(metadata["run_id"])
        decision_id = f"{run_id}:d{int(decision_index):06d}"
        trace = DecisionTrace(
            run_id=run_id,
            decision_id=decision_id,
            structure_hash=str(metadata["structure_hash"]),
            frozen_rule_hash=str(metadata["frozen_rule_hash"]),
            scenario_id=str(metadata["scenario_id"]),
            seed=int(metadata["seed"]),
            decision_index=int(decision_index),
            current_time=float(environment.current_time),
            queue_size=len(ready_ids),
            ready_task_ids=list(ready_ids) if self.config.store_full_ready_ids else [],
            selected_task_id=selected,
            selected_task_rank=selected_rank,
            selected_task_score=float(scores[selected_index]),
            score_margin_to_runner_up=runner_margin,
            minimum_slack=float(np.min(features["slack"])),
            maximum_uncertainty=float(np.max(features["uncertainty"])),
            host_load_summary=_host_load_summary(environment),
            workflow_id=_workflow_id(environment, selected),
            parameter_hash=str(metadata.get("parameter_hash", "")),
            is_critical=is_critical,
            critical_reasons=reasons if is_critical else [],
            criticality_score=float(criticality_score),
            candidate_tasks=snapshots,
            workflow_type=_workflow_type(environment, selected),
        )
        self.traced_count += 1
        return RecordedDecision(trace=trace, alternative_task_ids=alternatives)
