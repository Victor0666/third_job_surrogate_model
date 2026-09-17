# -*- coding: utf-8 -*-
"""Stable metrics schema for safe-HRL training, validation, and final test.

This module only observes completed episodes and persists reports. It does not
change rewards, costs, actions, policies, fuzzy definitions, or networks.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


SAFE_METRICS_SCHEMA_VERSION = 1
SAFE_METRIC_SOURCES = ("training", "validation", "final_test")

# CSV keeps one stable aggregate schema for all three sources. Nested per-seed
# evidence remains in JSONL/JSON and is never collapsed into the CSV cell set.
SAFE_METRIC_CSV_FIELDS = (
    "metrics_schema_version",
    "metric_source",
    "record_kind",
    "global_step",
    "episode",
    "evaluation_seed_count",
    "completed_workflow_count",
    "deadline_violation_count",
    "fuzzy_ddl_violation_rate",
    "feasible_workflow_ratio",
    "feasible_episode_count",
    "feasible_episode_ratio",
    "all_seed_feasible",
    "feasible_seed_rate",
    "all_seed_evaluation_completed",
    "completed_evaluation_seed_rate",
    "mean_fuzzy_lateness",
    "max_fuzzy_lateness",
    "minimum_fuzzy_safety_margin",
    "worst_seed_violation_rate",
    "worst_seed_fuzzy_lateness",
    "worst_seed_minimum_fuzzy_safety_margin",
    "shield_record_count",
    "shield_intervention_count",
    "shield_intervention_rate",
    "no_safe_action_count",
    "no_safe_action_rate",
    "fallback_count",
    "fallback_rate",
    "proposed_executed_action_mismatch_count",
    "proposed_executed_action_mismatch_rate",
    "q_c_prediction_error",
    "q_c_prediction_error_sample_count",
    "lambda_current",
    "lambda_trajectory",
    "fuzzy_energy_mean",
    "fuzzy_energy_std",
    "fuzzy_energy_score",
    "fuzzy_energy_score_across_seed_std",
    "modal_energy",
    "scheduling_time_seconds",
    "mean_seed_scheduling_time_seconds",
    "worst_seed_scheduling_time_seconds",
    "worst_seed_fuzzy_energy_score",
    "convergence_speed",
    "evaluations_to_convergence",
    "convergence_episode",
    "convergence_global_step",
    "heuristic_selection_count",
    "selected_llm_heuristic_count",
    "selected_llm_heuristic_frequency",
    "llm_associated_shield_record_count",
    "llm_associated_shield_intervention_count",
    "llm_associated_shield_rate",
    "energy_improvement_from_llm",
    "convergence_acceleration",
    "strict_ddl_feasibility_with_llm",
    "strict_ddl_feasibility_without_llm",
    "strict_ddl_feasibility_improvement",
)


def _finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _count(value: Any, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(float(numerator) / max(float(denominator), 1.0))


def _control_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    record_count = len(records)
    intervention_count = 0
    no_safe_count = 0
    fallback_count = 0
    mismatch_count = 0
    for raw in records:
        record = dict(raw)
        intervention_count += int(
            bool(record.get("shield_intervened", False))
        )
        fallback_triggered = bool(
            record.get("fallback_triggered", False)
        )
        reason = str(record.get("fallback_reason", ""))
        modification_reason = str(
            record.get("modification_reason", "")
        )
        no_safe_count += int(
            fallback_triggered
            and (
                reason == "empty_safe_action_set"
                or modification_reason == "no_safe_action_fallback"
                or not reason
            )
        )
        fallback_count += int(
            bool(
                record.get(
                    "fallback_applied",
                    fallback_triggered,
                )
            )
        )
        proposed = record.get(
            "proposed_action",
            record.get("rl_proposed_action"),
        )
        executed = record.get("executed_action")
        mismatch_count += int(
            proposed is not None
            and executed is not None
            and int(proposed) != int(executed)
        )
    return {
        "shield_record_count": int(record_count),
        "shield_intervention_count": int(intervention_count),
        "shield_intervention_rate": _rate(
            intervention_count,
            record_count,
        ),
        "no_safe_action_count": int(no_safe_count),
        "no_safe_action_rate": _rate(
            no_safe_count,
            record_count,
        ),
        "fallback_count": int(fallback_count),
        "fallback_rate": _rate(
            fallback_count,
            record_count,
        ),
        "proposed_executed_action_mismatch_count": int(
            mismatch_count
        ),
        "proposed_executed_action_mismatch_rate": _rate(
            mismatch_count,
            record_count,
        ),
    }


def _workflow_metrics(env: Any) -> dict[str, Any]:
    finish_ids = sorted(
        int(workflow_id)
        for workflow_id in getattr(env, "wf_finish_time", {})
    )
    workflows = getattr(env, "workflows", None)
    finish_getter = getattr(env, "_workflow_finish_tfn", None)
    risk_measure = getattr(env, "fuzzy_deadline_measure", None)

    lateness_values: list[float] = []
    safety_margins: list[float] = []
    if (
        finish_ids
        and workflows is not None
        and callable(finish_getter)
        and callable(risk_measure)
    ):
        for workflow_id in finish_ids:
            finish_tfn = finish_getter(workflow_id)
            risk_finish = float(risk_measure(finish_tfn))
            deadline = float(workflows[workflow_id].deadline)
            margin = deadline - risk_finish
            safety_margins.append(float(margin))
            lateness_values.append(max(0.0, -margin))
        completed_count = len(finish_ids)
        violation_count = int(
            sum(value > 0.0 for value in lateness_values)
        )
        exact = True
    else:
        completed_count = int(
            getattr(
                env,
                "_safety_cumulative_completed_workflow_count",
                0,
            )
        )
        violation_count = int(
            getattr(
                env,
                "_safety_cumulative_deadline_violation_count",
                0,
            )
        )
        total_lateness = max(
            0.0,
            float(
                getattr(
                    env,
                    "_safety_cumulative_fuzzy_lateness_cost",
                    0.0,
                )
            ),
        )
        lateness_values = (
            [total_lateness] if completed_count > 0 else []
        )
        safety_margins = (
            [-total_lateness] if completed_count > 0 else []
        )
        exact = False

    expected_count = getattr(env, "workflows_per_episode", None)
    if expected_count is None:
        expected_count = (
            len(workflows)
            if workflows is not None
            else completed_count
        )
    expected_count = int(expected_count)
    lateness_sum = float(sum(lateness_values))
    max_lateness = float(max(lateness_values, default=0.0))
    minimum_margin = float(min(safety_margins, default=0.0))
    evaluation_completed = bool(
        getattr(env, "done_flag", False)
        and completed_count == expected_count
    )
    episode_feasible = bool(
        evaluation_completed
        and completed_count > 0
        and violation_count == 0
        and max_lateness == 0.0
    )
    return {
        "completed_workflow_count": int(completed_count),
        "expected_workflow_count": expected_count,
        "deadline_violation_count": int(violation_count),
        "fuzzy_ddl_violation_rate": _rate(
            violation_count,
            completed_count,
        ),
        "feasible_workflow_count": int(
            max(completed_count - violation_count, 0)
        ),
        "feasible_workflow_ratio": _rate(
            completed_count - violation_count,
            completed_count,
        ),
        "fuzzy_lateness_sum": lateness_sum,
        "mean_fuzzy_lateness": _rate(
            lateness_sum,
            completed_count,
        ),
        "max_fuzzy_lateness": max_lateness,
        "minimum_fuzzy_safety_margin": minimum_margin,
        "evaluation_completed": evaluation_completed,
        "episode_feasible": episode_feasible,
        "exact_fuzzy_timeline_reconstruction": exact,
    }


def _heuristic_metrics(
    phase_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selection_count = len(phase_records)
    llm_records = [
        dict(record)
        for record in phase_records
        if str(record.get("heuristic_source", ""))
        == "seevo_llm"
    ]
    llm_count = len(llm_records)
    llm_shield_records = int(
        sum(
            int(record.get("heuristic_shield_record_count", 0))
            for record in llm_records
        )
    )

    def _llm_shield_interventions(
        record: Mapping[str, Any],
    ) -> int:
        detailed = record.get(
            "subsequent_safety_interventions"
        )
        if isinstance(detailed, (list, tuple)):
            return int(
                sum(
                    bool(
                        dict(item).get(
                            "shield_intervened",
                            False,
                        )
                    )
                    for item in detailed
                )
            )
        # Compatibility for old phase records without detailed audits.
        return int(
            record.get(
                "heuristic_shield_intervention_count",
                0,
            )
        )

    llm_interventions = int(
        sum(
            _llm_shield_interventions(record)
            for record in llm_records
        )
    )
    return {
        "heuristic_selection_count": int(selection_count),
        "selected_llm_heuristic_count": int(llm_count),
        "selected_llm_heuristic_frequency": _rate(
            llm_count,
            selection_count,
        ),
        "llm_associated_shield_record_count": (
            llm_shield_records
        ),
        "llm_associated_shield_intervention_count": (
            llm_interventions
        ),
        "llm_associated_shield_rate": _rate(
            llm_interventions,
            llm_shield_records,
        ),
    }


def build_episode_metric_record(
    env: Any,
    *,
    seed: int,
    scheduling_time_seconds: float,
    phase_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one completed episode/seed record from existing environment APIs."""
    scheduling_time = _finite(
        scheduling_time_seconds,
        "scheduling_time_seconds",
        minimum=0.0,
    )
    fuzzy_getter = getattr(env, "get_fuzzy_energy_summary", None)
    fuzzy = (
        fuzzy_getter()
        if callable(fuzzy_getter)
        else {
            "fuzzy_total_energy_mean": float(
                getattr(env, "total_energy", 0.0)
            ),
            "fuzzy_total_energy_std": 0.0,
            "fuzzy_total_energy_score": float(
                getattr(env, "total_energy", 0.0)
            ),
            "energy_modal": float(
                getattr(env, "total_energy", 0.0)
            ),
        }
    )
    shield_getter = getattr(
        env,
        "get_safety_shield_records",
        None,
    )
    shield_records = (
        shield_getter() if callable(shield_getter) else []
    )
    record = {
        "seed": int(seed),
        **_workflow_metrics(env),
        **_control_metrics(shield_records),
        **_heuristic_metrics(phase_records),
        "fuzzy_energy_mean": _finite(
            fuzzy["fuzzy_total_energy_mean"],
            "fuzzy_energy_mean",
            minimum=0.0,
        ),
        "fuzzy_energy_std": _finite(
            fuzzy["fuzzy_total_energy_std"],
            "fuzzy_energy_std",
            minimum=0.0,
        ),
        "fuzzy_energy_score": _finite(
            fuzzy["fuzzy_total_energy_score"],
            "fuzzy_energy_score",
            minimum=0.0,
        ),
        "modal_energy": _finite(
            fuzzy.get(
                "energy_modal",
                fuzzy.get("fuzzy_total_energy_modal", 0.0),
            ),
            "modal_energy",
            minimum=0.0,
        ),
        "scheduling_time_seconds": scheduling_time,
        "safety_cost": _finite(
            getattr(env, "_safety_cumulative_cost", 0.0),
            "safety_cost",
            minimum=0.0,
        ),
    }
    # Compatibility aliases used by model selection and earlier tests.
    record.update(
        {
            "deadline_violation_rate": record[
                "fuzzy_ddl_violation_rate"
            ],
            "seed_feasible": record["episode_feasible"],
        }
    )
    return record


def aggregate_safe_metric_records(
    records: Sequence[Mapping[str, Any]],
    *,
    q_c_prediction_error: float = 0.0,
    q_c_prediction_error_sample_count: int = 0,
    lambda_current: float = 0.0,
) -> dict[str, Any]:
    """Aggregate episode/seed records without hiding worst-seed outcomes."""
    if not records:
        raise ValueError("safe metric aggregation requires records")
    rows = [dict(record) for record in records]
    completed = sum(
        _count(
            row["completed_workflow_count"],
            "completed_workflow_count",
        )
        for row in rows
    )
    violations = sum(
        _count(
            row["deadline_violation_count"],
            "deadline_violation_count",
        )
        for row in rows
    )
    feasible_workflows = sum(
        _count(
            row["feasible_workflow_count"],
            "feasible_workflow_count",
        )
        for row in rows
    )
    lateness_sum = sum(
        _finite(
            row["fuzzy_lateness_sum"],
            "fuzzy_lateness_sum",
            minimum=0.0,
        )
        for row in rows
    )
    feasible_episodes = sum(
        bool(row["episode_feasible"]) for row in rows
    )
    completed_episodes = sum(
        bool(row["evaluation_completed"]) for row in rows
    )

    shield_records = sum(
        _count(row["shield_record_count"], "shield_record_count")
        for row in rows
    )
    shield_interventions = sum(
        _count(
            row["shield_intervention_count"],
            "shield_intervention_count",
        )
        for row in rows
    )
    no_safe_actions = sum(
        _count(
            row["no_safe_action_count"],
            "no_safe_action_count",
        )
        for row in rows
    )
    fallbacks = sum(
        _count(row["fallback_count"], "fallback_count")
        for row in rows
    )
    mismatches = sum(
        _count(
            row["proposed_executed_action_mismatch_count"],
            "proposed_executed_action_mismatch_count",
        )
        for row in rows
    )
    heuristic_selections = sum(
        _count(
            row["heuristic_selection_count"],
            "heuristic_selection_count",
        )
        for row in rows
    )
    llm_selections = sum(
        _count(
            row["selected_llm_heuristic_count"],
            "selected_llm_heuristic_count",
        )
        for row in rows
    )
    llm_shield_records = sum(
        _count(
            row["llm_associated_shield_record_count"],
            "llm_associated_shield_record_count",
        )
        for row in rows
    )
    llm_interventions = sum(
        _count(
            row["llm_associated_shield_intervention_count"],
            "llm_associated_shield_intervention_count",
        )
        for row in rows
    )

    energy_scores = [
        _finite(
            row["fuzzy_energy_score"],
            "fuzzy_energy_score",
            minimum=0.0,
        )
        for row in rows
    ]
    schedule_times = [
        _finite(
            row["scheduling_time_seconds"],
            "scheduling_time_seconds",
            minimum=0.0,
        )
        for row in rows
    ]
    seed_count = len(rows)
    report = {
        "metrics_schema_version": SAFE_METRICS_SCHEMA_VERSION,
        "record_kind": "aggregate",
        "evaluation_seed_count": seed_count,
        "completed_workflow_count": int(completed),
        "deadline_violation_count": int(violations),
        "fuzzy_ddl_violation_rate": _rate(
            violations,
            completed,
        ),
        "feasible_workflow_ratio": _rate(
            feasible_workflows,
            completed,
        ),
        "feasible_episode_count": int(feasible_episodes),
        "feasible_episode_ratio": _rate(
            feasible_episodes,
            seed_count,
        ),
        "all_seed_feasible": bool(
            feasible_episodes == seed_count
        ),
        "feasible_seed_rate": _rate(
            feasible_episodes,
            seed_count,
        ),
        "all_seed_evaluation_completed": bool(
            completed_episodes == seed_count
        ),
        "completed_evaluation_seed_rate": _rate(
            completed_episodes,
            seed_count,
        ),
        "mean_fuzzy_lateness": _rate(
            lateness_sum,
            completed,
        ),
        "max_fuzzy_lateness": float(
            max(row["max_fuzzy_lateness"] for row in rows)
        ),
        "minimum_fuzzy_safety_margin": float(
            min(
                row["minimum_fuzzy_safety_margin"]
                for row in rows
            )
        ),
        "worst_seed_violation_rate": float(
            max(row["fuzzy_ddl_violation_rate"] for row in rows)
        ),
        "worst_seed_fuzzy_lateness": float(
            max(row["max_fuzzy_lateness"] for row in rows)
        ),
        "worst_seed_minimum_fuzzy_safety_margin": float(
            min(
                row["minimum_fuzzy_safety_margin"]
                for row in rows
            )
        ),
        "shield_record_count": int(shield_records),
        "shield_intervention_count": int(
            shield_interventions
        ),
        "shield_intervention_rate": _rate(
            shield_interventions,
            shield_records,
        ),
        "no_safe_action_count": int(no_safe_actions),
        "no_safe_action_rate": _rate(
            no_safe_actions,
            shield_records,
        ),
        "fallback_count": int(fallbacks),
        "fallback_rate": _rate(
            fallbacks,
            shield_records,
        ),
        "proposed_executed_action_mismatch_count": int(
            mismatches
        ),
        "proposed_executed_action_mismatch_rate": _rate(
            mismatches,
            shield_records,
        ),
        "q_c_prediction_error": _finite(
            q_c_prediction_error,
            "q_c_prediction_error",
            minimum=0.0,
        ),
        "q_c_prediction_error_sample_count": _count(
            q_c_prediction_error_sample_count,
            "q_c_prediction_error_sample_count",
        ),
        "lambda_current": _finite(
            lambda_current,
            "lambda_current",
            minimum=0.0,
        ),
        "lambda_trajectory": [],
        "fuzzy_energy_mean": float(
            statistics.fmean(
                float(row["fuzzy_energy_mean"]) for row in rows
            )
        ),
        "fuzzy_energy_std": float(
            statistics.fmean(
                float(row["fuzzy_energy_std"]) for row in rows
            )
        ),
        "fuzzy_energy_score": float(
            statistics.fmean(energy_scores)
        ),
        "fuzzy_energy_score_across_seed_std": float(
            statistics.pstdev(energy_scores)
            if len(energy_scores) > 1
            else 0.0
        ),
        "modal_energy": float(
            statistics.fmean(
                float(row["modal_energy"]) for row in rows
            )
        ),
        "scheduling_time_seconds": float(sum(schedule_times)),
        "mean_seed_scheduling_time_seconds": float(
            statistics.fmean(schedule_times)
        ),
        "worst_seed_scheduling_time_seconds": float(
            max(schedule_times)
        ),
        "worst_seed_fuzzy_energy_score": float(
            max(energy_scores)
        ),
        "convergence_speed": 0.0,
        "evaluations_to_convergence": None,
        "convergence_episode": None,
        "convergence_global_step": None,
        "heuristic_selection_count": int(
            heuristic_selections
        ),
        "selected_llm_heuristic_count": int(
            llm_selections
        ),
        "selected_llm_heuristic_frequency": _rate(
            llm_selections,
            heuristic_selections,
        ),
        "llm_associated_shield_record_count": int(
            llm_shield_records
        ),
        "llm_associated_shield_intervention_count": int(
            llm_interventions
        ),
        "llm_associated_shield_rate": _rate(
            llm_interventions,
            llm_shield_records,
        ),
        # Paired-comparison-only fields must remain null for one run.
        "energy_improvement_from_llm": None,
        "convergence_acceleration": None,
        "strict_ddl_feasibility_with_llm": None,
        "strict_ddl_feasibility_without_llm": None,
        "strict_ddl_feasibility_improvement": None,
        "safety_cost": float(
            statistics.fmean(
                float(row.get("safety_cost", 0.0))
                for row in rows
            )
        ),
        "per_seed_metrics": rows,
    }
    # Compatibility aliases retained for model selection and stage metrics.
    report.update(
        {
            "deadline_violation_rate": report[
                "fuzzy_ddl_violation_rate"
            ],
            "max_fuzzy_lateness": report[
                "max_fuzzy_lateness"
            ],
            "worst_seed_violation": report[
                "worst_seed_violation_rate"
            ],
            "worst_seed_lateness": report[
                "worst_seed_fuzzy_lateness"
            ],
            "validation_seed_count": seed_count,
            "per_seed_safety_metrics": rows,
            "zero_violation_pass": bool(
                report["fuzzy_ddl_violation_rate"] == 0.0
            ),
        }
    )
    return report


def compute_llm_assistance_metrics(
    *,
    with_llm_report: Mapping[str, Any],
    without_llm_report: Mapping[str, Any],
    strict_ddl: bool,
) -> dict[str, Any]:
    """Compute paired LLM effects; positive values mean improvement.

    Comparative values are only meaningful for the same seed set and a
    strict-DDL evaluation. The caller must state the latter explicitly.
    """
    if not bool(strict_ddl):
        raise ValueError(
            "strict-DDL LLM comparison requires strict_ddl=True"
        )

    def _seed_signature(report: Mapping[str, Any]) -> tuple[int, ...]:
        rows = report.get(
            "per_seed_metrics",
            report.get("per_seed_safety_metrics", ()),
        )
        return tuple(
            sorted(int(dict(row)["seed"]) for row in rows)
        )

    assisted_seed_count = _count(
        with_llm_report["evaluation_seed_count"],
        "with_llm.evaluation_seed_count",
    )
    baseline_seed_count = _count(
        without_llm_report["evaluation_seed_count"],
        "without_llm.evaluation_seed_count",
    )
    if assisted_seed_count != baseline_seed_count:
        raise ValueError(
            "LLM comparison requires equal evaluation seed counts"
        )
    assisted_seeds = _seed_signature(with_llm_report)
    baseline_seeds = _seed_signature(without_llm_report)
    if (
        assisted_seeds
        and baseline_seeds
        and assisted_seeds != baseline_seeds
    ):
        raise ValueError(
            "LLM comparison requires identical evaluation seeds"
        )
    assisted_energy = _finite(
        with_llm_report["fuzzy_energy_score"],
        "with_llm.fuzzy_energy_score",
        minimum=0.0,
    )
    baseline_energy = _finite(
        without_llm_report["fuzzy_energy_score"],
        "without_llm.fuzzy_energy_score",
        minimum=0.0,
    )
    energy_improvement = (
        float((baseline_energy - assisted_energy) / baseline_energy)
        if baseline_energy > 0.0
        else None
    )

    assisted_evaluations = with_llm_report.get(
        "evaluations_to_convergence"
    )
    baseline_evaluations = without_llm_report.get(
        "evaluations_to_convergence"
    )
    if (
        assisted_evaluations is None
        or baseline_evaluations is None
        or int(baseline_evaluations) <= 0
    ):
        convergence_acceleration = None
    else:
        convergence_acceleration = float(
            (
                int(baseline_evaluations)
                - int(assisted_evaluations)
            )
            / int(baseline_evaluations)
        )

    with_feasibility = _finite(
        with_llm_report["feasible_episode_ratio"],
        "with_llm.feasible_episode_ratio",
        minimum=0.0,
    )
    without_feasibility = _finite(
        without_llm_report["feasible_episode_ratio"],
        "without_llm.feasible_episode_ratio",
        minimum=0.0,
    )
    return {
        "selected_llm_heuristic_frequency": float(
            with_llm_report.get(
                "selected_llm_heuristic_frequency",
                0.0,
            )
        ),
        "energy_improvement_from_llm": energy_improvement,
        "convergence_acceleration": convergence_acceleration,
        "llm_associated_shield_rate": float(
            with_llm_report.get(
                "llm_associated_shield_rate",
                0.0,
            )
        ),
        "strict_ddl_feasibility_with_llm": with_feasibility,
        "strict_ddl_feasibility_without_llm": (
            without_feasibility
        ),
        "strict_ddl_feasibility_improvement": float(
            with_feasibility - without_feasibility
        ),
    }


class SafeMetricStore:
    """Persist source-separated CSV and JSONL with a stable field schema."""

    def __init__(
        self,
        output_directory: str | os.PathLike[str],
        *,
        convergence_window: int = 5,
    ):
        self.output_directory = Path(output_directory).resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.convergence_window = int(convergence_window)
        if self.convergence_window <= 0:
            raise ValueError("convergence_window must be positive")
        self._lambda_trajectories = {
            source: [] for source in SAFE_METRIC_SOURCES
        }
        self._validation_feasibility: list[bool] = []
        self._convergence = {
            "evaluations_to_convergence": None,
            "convergence_episode": None,
            "convergence_global_step": None,
        }
        self._load_existing_series()

    def _jsonl_path(self, source: str) -> Path:
        return self.output_directory / f"{source}_metrics.jsonl"

    def _csv_path(self, source: str) -> Path:
        return self.output_directory / f"{source}_metrics.csv"

    def _load_existing_series(self) -> None:
        for source in SAFE_METRIC_SOURCES:
            path = self._jsonl_path(source)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if (
                        int(
                            payload.get(
                                "metrics_schema_version",
                                -1,
                            )
                        )
                        != SAFE_METRICS_SCHEMA_VERSION
                    ):
                        raise ValueError(
                            "existing safe metrics schema mismatch"
                        )
                    self._lambda_trajectories[source].append(
                        float(payload["lambda_current"])
                    )
                    if source == "validation":
                        self._validation_feasibility.append(
                            bool(payload["all_seed_feasible"])
                        )
                        if (
                            self._convergence[
                                "evaluations_to_convergence"
                            ]
                            is None
                            and len(self._validation_feasibility)
                            >= self.convergence_window
                            and all(
                                self._validation_feasibility[
                                    -self.convergence_window :
                                ]
                            )
                        ):
                            self._convergence = {
                                "evaluations_to_convergence": len(
                                    self._validation_feasibility
                                ),
                                "convergence_episode": payload.get(
                                    "episode"
                                ),
                                "convergence_global_step": (
                                    payload.get("global_step")
                                ),
                            }

    @staticmethod
    def _validate_source(source: str) -> str:
        source = str(source)
        if source not in SAFE_METRIC_SOURCES:
            raise ValueError(
                f"metric_source must be one of {SAFE_METRIC_SOURCES}"
            )
        return source

    def append(
        self,
        source: str,
        report: Mapping[str, Any],
        *,
        global_step: int | None,
        episode: int | None,
    ) -> dict[str, Any]:
        source = self._validate_source(source)
        result = dict(report)
        result.update(
            {
                "metrics_schema_version": (
                    SAFE_METRICS_SCHEMA_VERSION
                ),
                "metric_source": source,
                "record_kind": "aggregate",
                "global_step": (
                    None if global_step is None else int(global_step)
                ),
                "episode": (
                    None if episode is None else int(episode)
                ),
            }
        )
        missing = [
            field
            for field in SAFE_METRIC_CSV_FIELDS
            if field not in result
            and field
            not in {
                "metric_source",
                "global_step",
                "episode",
            }
        ]
        if missing:
            raise ValueError(
                f"safe metric report is missing fields: {missing}"
            )

        self._lambda_trajectories[source].append(
            _finite(
                result["lambda_current"],
                "lambda_current",
                minimum=0.0,
            )
        )
        if source == "validation":
            self._validation_feasibility.append(
                bool(result["all_seed_feasible"])
            )
            if (
                self._convergence[
                    "evaluations_to_convergence"
                ]
                is None
                and len(self._validation_feasibility)
                >= self.convergence_window
                and all(
                    self._validation_feasibility[
                        -self.convergence_window :
                    ]
                )
            ):
                self._convergence = {
                    "evaluations_to_convergence": len(
                        self._validation_feasibility
                    ),
                    "convergence_episode": result["episode"],
                    "convergence_global_step": result[
                        "global_step"
                    ],
                }

        evaluations = self._convergence[
            "evaluations_to_convergence"
        ]
        result["lambda_trajectory"] = list(
            self._lambda_trajectories[source]
        )
        result.update(self._convergence)
        result["convergence_speed"] = (
            0.0
            if evaluations is None
            else float(1.0 / max(int(evaluations), 1))
        )

        jsonl_path = self._jsonl_path(source)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            json.dump(
                result,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            handle.write("\n")

        csv_path = self._csv_path(source)
        exists = csv_path.exists() and csv_path.stat().st_size > 0
        if exists:
            with csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                existing_header = next(csv.reader(handle), [])
            if existing_header != list(SAFE_METRIC_CSV_FIELDS):
                raise ValueError(
                    "existing safe metrics CSV schema mismatch"
                )
        with csv_path.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(SAFE_METRIC_CSV_FIELDS),
                extrasaction="ignore",
            )
            if not exists:
                writer.writeheader()
            row = {
                key: (
                    ""
                    if result.get(key) is None
                    else (
                        json.dumps(
                            result[key],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        if isinstance(result.get(key), (list, dict))
                        else result[key]
                    )
                )
                for key in SAFE_METRIC_CSV_FIELDS
            }
            writer.writerow(row)

        if source == "final_test":
            final_path = (
                self.output_directory
                / "final_test_metrics.json"
            )
            temporary = final_path.with_suffix(".json.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    result,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                handle.write("\n")
            os.replace(temporary, final_path)
        return result

    def save_llm_comparison(
        self,
        *,
        with_llm_report: Mapping[str, Any],
        without_llm_report: Mapping[str, Any],
        strict_ddl: bool,
    ) -> dict[str, Any]:
        """Persist one paired, same-seed LLM ablation report."""
        metrics = compute_llm_assistance_metrics(
            with_llm_report=with_llm_report,
            without_llm_report=without_llm_report,
            strict_ddl=strict_ddl,
        )
        payload = {
            "metrics_schema_version": (
                SAFE_METRICS_SCHEMA_VERSION
            ),
            "comparison_kind": "paired_llm_ablation",
            "strict_ddl": bool(strict_ddl),
            "evaluation_seed_count": int(
                with_llm_report["evaluation_seed_count"]
            ),
            **metrics,
        }
        target = (
            self.output_directory
            / "llm_comparison_metrics.json"
        )
        temporary = target.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
        os.replace(temporary, target)
        return payload


__all__ = [
    "SAFE_METRIC_CSV_FIELDS",
    "SAFE_METRIC_SOURCES",
    "SAFE_METRICS_SCHEMA_VERSION",
    "SafeMetricStore",
    "aggregate_safe_metric_records",
    "build_episode_metric_record",
    "compute_llm_assistance_metrics",
]
