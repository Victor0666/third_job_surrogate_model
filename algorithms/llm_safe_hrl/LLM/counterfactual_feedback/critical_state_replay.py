"""Deterministic feature-level replay for archived ready-task states."""

from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np

from rule_optimization import validate_frozen_rule_source

from .critical_state_schemas import (
    ARCHIVE_VERSION,
    FEATURE_SCHEMA_VERSION,
    REPLAY_VERSION,
    RULE_INTERFACE_VERSION,
    CriticalStateRecord,
    CriticalStateReplayConfig,
    CriticalStateReplaySummary,
    ReplayResult,
)
from .schemas import canonical_hash
from .trace_recorder import FEATURE_NAMES


STRUCTURAL_ACTIONS_BY_RISK = {
    "NEGATIVE_SLACK": "add_conditional_ddl_protection_gate",
    "MULTI_NEAR_ZERO_SLACK": "normalize_slack_by_workflow_deadline_budget",
    "READY_QUEUE_CONGESTION": "add_successor_release_interaction",
    "CRITICAL_PATH_STARVATION": "add_upward_rank_remaining_work_interaction",
    "UNCERTAINTY_SPIKE": "add_uncertainty_ddl_interaction",
    "RESOURCE_BOTTLENECK": "add_host_load_conditional_gate",
    "DDL_ENERGY_CONFLICT": "add_small_energy_gap_high_ddl_risk_gate",
    "COUNTERFACTUAL_DOMINATED_CHOICE": "add_conditional_ddl_protection_gate",
    "PESSIMISTIC_TIMELINE_RISK": "add_pessimistic_risk_gate",
    "SUCCESSOR_RELEASE_BLOCKING": "add_successor_release_interaction",
}


def load_frozen_priority_rule(path: str | Path) -> Callable:
    """Load a safety-validated frozen rule without dynamic string evaluation."""
    source_path = Path(path).resolve()
    source = source_path.read_text(encoding="utf-8")
    validate_frozen_rule_source(source)
    module_name = "critical_state_replay_" + canonical_hash(
        {"path": str(source_path), "source": source}
    )[:20]
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen replay rule: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "get_task_priority_v2", None)
    if not callable(function):
        raise ValueError("frozen replay rule does not define get_task_priority_v2")
    return function


class CriticalStateReplayCache:
    """Small successful-result cache isolated by archive, rule, and schemas."""

    def __init__(self, enabled: bool, path: str | Path | None = None):
        self.enabled = bool(enabled)
        self.path = Path(path).resolve() if path else None
        self.items = {}
        self.hits = 0
        self.misses = 0
        if self.enabled and self.path and self.path.is_file():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                self.items = value

    def key(
        self,
        *,
        archive_hash: str,
        state_id: str,
        structure_hash: str,
        frozen_rule_hash: str,
        config_hash: str,
        generation: int,
    ) -> str:
        return canonical_hash(
            {
                "archive_hash": archive_hash,
                "state_id": state_id,
                "replay_structure_hash": structure_hash,
                "replay_frozen_rule_hash": frozen_rule_hash,
                "replay_config_hash": config_hash,
                "generation": int(generation),
                "rule_interface_version": RULE_INTERFACE_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "replay_version": REPLAY_VERSION,
            }
        )

    def get(self, key: str) -> ReplayResult | None:
        if not self.enabled or key not in self.items:
            self.misses += 1
            return None
        self.hits += 1
        return ReplayResult(**json.loads(json.dumps(self.items[key])))

    def put(self, key: str, result: ReplayResult) -> None:
        if not self.enabled or not result.output_valid:
            return
        self.items[key] = result.to_dict()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self.items, ensure_ascii=True, allow_nan=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.path)


class CriticalStateReplayer:
    """Execute the unchanged eight-array rule interface on archived features."""

    def __init__(
        self,
        config: CriticalStateReplayConfig,
        *,
        cache: CriticalStateReplayCache | None = None,
    ):
        self.config = config
        self.cache = cache or CriticalStateReplayCache(False)

    @staticmethod
    def _arrays(record: CriticalStateRecord) -> list[np.ndarray]:
        return [
            np.asarray(
                [record.ready_task_features[str(task_id)][name] for task_id in record.ready_task_ids],
                dtype=float,
            )
            for name in FEATURE_NAMES
        ]

    @staticmethod
    def _invalid_result(
        record: CriticalStateRecord,
        *,
        structure_hash: str,
        frozen_rule_hash: str,
        generation: int,
        reason: str,
        latency: float,
    ) -> ReplayResult:
        payload = {
            "state_id": record.state_id,
            "replay_rule_structure_hash": structure_hash,
            "replay_frozen_rule_hash": frozen_rule_hash,
            "generation": int(generation),
            "selected_task_id": None,
            "selected_task_rank": None,
            "selected_task_score": None,
            "preferred_task_ranks": {},
            "dominated_task_ranks": {},
            "replay_status": "INVALID",
            "verified_coverage": float(record.verified_coverage),
            "repeated_historical_error": False,
            "ddl_ordering_correct": None,
            "uncertainty_ordering_correct": None,
            "energy_ordering_reasonable": None,
            "deterministic_output": False,
            "output_valid": False,
            "evidence_used": {"error": reason},
            "replay_latency": float(latency),
            "state_status_before": record.status,
        }
        stable_payload = {key: value for key, value in payload.items() if key != "replay_latency"}
        return ReplayResult(**payload, result_hash=canonical_hash(stable_payload))

    def replay(
        self,
        record: CriticalStateRecord,
        priority_function: Callable,
        *,
        structure_hash: str,
        frozen_rule_hash: str,
        generation: int,
        archive_hash: str,
    ) -> ReplayResult:
        record.validate()
        cache_key = self.cache.key(
            archive_hash=archive_hash,
            state_id=record.state_id,
            structure_hash=structure_hash,
            frozen_rule_hash=frozen_rule_hash,
            config_hash=self.config.config_hash,
            generation=generation,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        started = time.perf_counter()
        archive_before = canonical_hash(record.to_dict())
        try:
            first_inputs = self._arrays(record)
            first_input_values = [value.copy() for value in first_inputs]
            first = np.asarray(priority_function(*first_inputs), dtype=float).reshape(-1)
            inputs_unchanged = all(
                np.array_equal(before, after, equal_nan=True)
                for before, after in zip(first_input_values, first_inputs)
            )
            second_inputs = self._arrays(record)
            second = np.asarray(priority_function(*second_inputs), dtype=float).reshape(-1)
            if (
                first.shape != (len(record.ready_task_ids),)
                or second.shape != first.shape
                or not np.all(np.isfinite(first))
                or not np.all(np.isfinite(second))
                or not inputs_unchanged
            ):
                raise ValueError("rule output shape/finite/input-immutability contract failed")
            deterministic = bool(np.array_equal(first, second))
            if not deterministic:
                raise ValueError("rule output is not deterministic")
        except Exception as exc:
            return self._invalid_result(
                record,
                structure_hash=structure_hash,
                frozen_rule_hash=frozen_rule_hash,
                generation=generation,
                reason=f"{type(exc).__name__}: {exc}",
                latency=(time.perf_counter() - started) * 1000.0,
            )
        if canonical_hash(record.to_dict()) != archive_before:
            return self._invalid_result(
                record,
                structure_hash=structure_hash,
                frozen_rule_hash=frozen_rule_hash,
                generation=generation,
                reason="archived state was modified during replay",
                latency=(time.perf_counter() - started) * 1000.0,
            )

        order = np.argsort(first, kind="stable").tolist()
        ranked_ids = [record.ready_task_ids[index] for index in order]
        ranks = {task_id: rank + 1 for rank, task_id in enumerate(ranked_ids)}
        selected_task = int(ranked_ids[0])
        preferred_ranks = {
            str(task_id): ranks[task_id]
            for task_id in record.preferred_task_ids
            if task_id in ranks
        }
        dominated_ranks = {
            str(task_id): ranks[task_id]
            for task_id in record.dominated_task_ids
            if task_id in ranks
        }
        preferred_before_dominated = bool(
            preferred_ranks
            and dominated_ranks
            and min(preferred_ranks.values()) < min(dominated_ranks.values())
        )
        dominated_before_preferred = bool(
            preferred_ranks
            and dominated_ranks
            and min(dominated_ranks.values()) < min(preferred_ranks.values())
        )
        if selected_task in record.unverified_task_ids:
            status = "PARTIAL_SUCCESS" if preferred_before_dominated else "UNVERIFIED"
        elif selected_task in record.preferred_task_ids and preferred_before_dominated:
            status = "VERIFIED_SUCCESS"
        elif selected_task in record.dominated_task_ids or dominated_before_preferred:
            status = "VERIFIED_FAILURE"
        else:
            status = "UNVERIFIED"
        ddl_correct = (
            True if status in {"VERIFIED_SUCCESS", "PARTIAL_SUCCESS"}
            else False if status == "VERIFIED_FAILURE" else None
        )
        uncertainty_correct = (
            ddl_correct
            if record.risk_category in {"UNCERTAINTY_SPIKE", "PESSIMISTIC_TIMELINE_RISK"}
            or "PESSIMISTIC_TIMELINE_RISK" in record.auxiliary_risk_categories
            else None
        )
        energy_reasonable = None
        if ddl_correct is True and record.risk_category == "DDL_ENERGY_CONFLICT":
            energy_reasonable = selected_task in record.preferred_task_ids
        latency = (time.perf_counter() - started) * 1000.0
        payload = {
            "state_id": record.state_id,
            "replay_rule_structure_hash": structure_hash,
            "replay_frozen_rule_hash": frozen_rule_hash,
            "generation": int(generation),
            "selected_task_id": selected_task,
            "selected_task_rank": 1,
            "selected_task_score": float(first[order[0]]),
            "preferred_task_ranks": preferred_ranks,
            "dominated_task_ranks": dominated_ranks,
            "replay_status": status,
            "verified_coverage": float(record.verified_coverage),
            "repeated_historical_error": status == "VERIFIED_FAILURE",
            "ddl_ordering_correct": ddl_correct,
            "uncertainty_ordering_correct": uncertainty_correct,
            "energy_ordering_reasonable": energy_reasonable,
            "deterministic_output": deterministic,
            "output_valid": True,
            "evidence_used": {
                "risk_category": record.risk_category,
                "diagnosis_codes": record.diagnosis_codes,
                "preferred_task_ids": record.preferred_task_ids,
                "dominated_task_ids": record.dominated_task_ids,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "rule_interface_version": RULE_INTERFACE_VERSION,
            },
            "replay_latency": float(latency),
            "state_status_before": record.status,
        }
        stable_payload = {key: value for key, value in payload.items() if key != "replay_latency"}
        result = ReplayResult(**payload, result_hash=canonical_hash(stable_payload))
        self.cache.put(cache_key, result)
        return result


def aggregate_replay_results(
    results: Iterable[ReplayResult],
    records: Iterable[CriticalStateRecord],
    *,
    structure_hash: str,
    frozen_rule_hash: str,
    generation: int,
    archive_hash: str,
    config: CriticalStateReplayConfig,
) -> CriticalStateReplaySummary:
    """Aggregate current results against a pre-replay archive snapshot.

    Persistent counts and current-rule outcomes remain separate, preventing
    another rule replayed in the same generation from contaminating feedback.
    No DDL/energy weighted sum is used.
    """
    result_rows = list(results)
    record_map = {record.state_id: record for record in records}
    failures = [row for row in result_rows if row.replay_status == "VERIFIED_FAILURE"]
    successes = [row for row in result_rows if row.replay_status == "VERIFIED_SUCCESS"]
    unverified = [
        row for row in result_rows
        if row.replay_status in {"UNVERIFIED", "PARTIAL_SUCCESS", "INVALID"}
    ]
    failure_categories = Counter(record_map[row.state_id].risk_category for row in failures)
    success_categories = Counter(record_map[row.state_id].risk_category for row in successes)
    failure_scenarios = Counter(record_map[row.state_id].source_scenario_id for row in failures)
    failure_workflows = Counter(record_map[row.state_id].workflow_type for row in failures)
    low_coverage_failures = [
        row
        for row in failures
        if row.verified_coverage < config.min_verified_candidate_coverage
    ]
    hard_patterns = []
    cross_generation = []
    high_actions = []
    medium_actions = []
    for category, count in sorted(failure_categories.items()):
        category_results = [
            row
            for row in failures
            if record_map[row.state_id].risk_category == category
        ]
        category_records = [record_map[row.state_id] for row in category_results]
        covered_results = [
            row
            for row in category_results
            if row.verified_coverage >= config.min_verified_candidate_coverage
        ]
        covered_records = [record_map[row.state_id] for row in covered_results]
        seeds = {seed for record in covered_records for seed in record.source_seeds}
        scenarios = {
            value for record in covered_records for value in record.source_scenarios
        }
        generations = {
            *[value for record in covered_records for value in record.failure_generations],
            int(generation),
        }
        structures = {
            *[value for record in covered_records for value in record.replay_structure_hashes],
            str(structure_hash),
        }
        historical_repetition = any(
            record.failure_count + 1 >= config.min_repeated_failure_count
            for record in covered_records
        )
        repeated = (
            len(covered_results) >= config.min_repeated_failure_count
            or historical_repetition
        )
        high = repeated and (
            len(seeds) >= config.high_confidence_seed_count
            or len(scenarios) >= config.high_confidence_scenario_count
            or len(generations) >= 2
            or len(structures) >= 2
        )
        pattern = {
            "risk_category": category,
            "failure_count": count,
            "state_ids": [record.state_id for record in category_records[:8]],
            "seed_count": len(seeds),
            "scenario_count": len(scenarios),
            "generation_count": len(generations),
            "structure_count": len(structures),
            "coverage_sufficient_count": len(covered_results),
            "minimum_verified_coverage": float(
                min((row.verified_coverage for row in category_results), default=0.0)
            ),
            "confidence": "high" if high else ("medium" if repeated else "low"),
        }
        if any(
            record.status == "HARD"
            or record.consecutive_failure_count + 1 >= config.hard_failure_threshold
            for record in category_records
        ):
            hard_patterns.append(pattern)
        if len(generations) >= 2:
            cross_generation.append({**pattern, "failure_generations": sorted(generations)})
        action = {
            "action": STRUCTURAL_ACTIONS_BY_RISK[category],
            "risk_category": category,
            "failure_count": len(covered_results),
            "evidence_state_ids": [
                record_map[row.state_id].state_id for row in covered_results[:8]
            ],
        }
        if high:
            high_actions.append(action)
        elif repeated:
            medium_actions.append(action)
    regressions = [
        {
            "state_id": row.state_id,
            "risk_category": record_map[row.state_id].risk_category,
            "suggested_structural_action": STRUCTURAL_ACTIONS_BY_RISK[
                record_map[row.state_id].risk_category
            ],
        }
        for row in failures
        if row.state_status_before == "RESOLVED"
    ]
    failure_examples = [
        {
            "state_id": row.state_id,
            "risk_category": record_map[row.state_id].risk_category,
            "scenario_id": record_map[row.state_id].source_scenario_id,
            "workflow_type": record_map[row.state_id].workflow_type,
            "selected_task_id": row.selected_task_id,
            "preferred_task_ranks": row.preferred_task_ranks,
            "dominated_task_ranks": row.dominated_task_ranks,
            "verified_coverage": float(row.verified_coverage),
            "coverage_sufficient": bool(
                row.verified_coverage >= config.min_verified_candidate_coverage
            ),
        }
        for row in failures[: config.max_failure_examples]
    ]
    success_examples = [
        {
            "state_id": row.state_id,
            "risk_category": record_map[row.state_id].risk_category,
            "scenario_id": record_map[row.state_id].source_scenario_id,
            "selected_task_id": row.selected_task_id,
        }
        for row in successes[: config.max_success_examples]
    ]
    base = {
        "structure_hash": structure_hash,
        "frozen_rule_hash": frozen_rule_hash,
        "generation": int(generation),
        "replayed_state_count": len(result_rows),
        "verified_state_count": len(failures) + len(successes),
        "unverified_state_count": len(unverified),
        "verified_success_count": len(successes),
        "verified_failure_count": len(failures),
        "repeated_historical_error_count": sum(row.repeated_historical_error for row in failures),
        "failure_by_risk_category": dict(sorted(failure_categories.items())),
        "success_by_risk_category": dict(sorted(success_categories.items())),
        "failure_by_scenario": dict(sorted(failure_scenarios.items())),
        "failure_by_workflow_type": dict(sorted(failure_workflows.items())),
        "hard_state_failure_patterns": hard_patterns,
        "resolved_state_regressions": regressions,
        "cross_generation_failure_patterns": cross_generation,
        "high_confidence_structural_actions": high_actions,
        "medium_confidence_actions": medium_actions,
        "representative_failures": failure_examples,
        "representative_successes": success_examples,
        "limitations": [
            "feature-level replay is a deterministic stress test, not a full dynamic simulation",
            "unverified and partial outcomes are never counted as success or failure",
            "energy is inspected only after DDL ordering is not worse",
        ] + (
            [
                f"{len(low_coverage_failures)} verified failures were excluded from "
                "structural-action confidence because candidate coverage was below "
                f"{config.min_verified_candidate_coverage:.3f}"
            ]
            if low_coverage_failures else []
        ),
        "archive_hash": archive_hash,
        "replay_config_hash": config.config_hash,
        "used_test_seed": False,
        "archive_version": ARCHIVE_VERSION,
    }
    return CriticalStateReplaySummary(**base, summary_hash=canonical_hash(base))


def compact_replay_feedback(summary: dict, max_chars: int) -> dict:
    """Keep persistent failures and regressions before lower-priority evidence."""
    keys = (
        "structure_hash", "generation", "replayed_state_count", "verified_state_count",
        "unverified_state_count", "verified_success_count", "verified_failure_count",
        "repeated_historical_error_count", "hard_state_failure_patterns",
        "resolved_state_regressions", "cross_generation_failure_patterns",
        "high_confidence_structural_actions", "medium_confidence_actions",
        "representative_failures", "representative_successes", "limitations",
    )
    result = {key: json.loads(json.dumps(summary.get(key))) for key in keys if key in summary}
    drop_order = (
        "representative_successes", "limitations", "medium_confidence_actions",
        "cross_generation_failure_patterns", "representative_failures",
    )
    while len(json.dumps(result, ensure_ascii=True, sort_keys=True)) > max_chars:
        changed = False
        for key in drop_order:
            value = result.get(key)
            if isinstance(value, list) and value:
                value.pop()
                changed = True
                break
        if not changed:
            result = {
                "structure_hash": summary.get("structure_hash"),
                "high_confidence_structural_actions": summary.get(
                    "high_confidence_structural_actions", []
                )[:1],
                "limitations": ["critical-state replay feedback truncated"],
            }
            break
    payload_chars = len(json.dumps(result, ensure_ascii=True, sort_keys=True))
    result["prompt_payload_chars"] = payload_chars
    if len(json.dumps(result, ensure_ascii=True, sort_keys=True)) > max_chars:
        result.pop("prompt_payload_chars", None)
    return result


def strict_replay_gate_triggered(
    results: Iterable[ReplayResult],
    records: Iterable[CriticalStateRecord],
    config: CriticalStateReplayConfig,
) -> bool:
    """Return true only for verified repeated DDL failures on HARD states."""
    if not config.strict_replay_gate:
        return False
    record_rows = list(records)
    record_map = {record.state_id: record for record in record_rows}
    result_rows = list(results)
    failing_state_ids = {
        result.state_id
        for result in result_rows
        if result.replay_status == "VERIFIED_FAILURE"
    }
    for result in result_rows:
        record = record_map[result.state_id]
        peer_records = [
            row
            for row in record_rows
            if row.state_id in failing_state_ids
            and row.semantic_signature == record.semantic_signature
        ]
        seeds = {seed for row in peer_records for seed in row.source_seeds}
        scenarios = {
            scenario for row in peer_records for scenario in row.source_scenarios
        }
        repeated_source = (
            len(seeds) >= config.high_confidence_seed_count
            or len(scenarios) >= config.high_confidence_scenario_count
        )
        hard = (
            record.status == "HARD"
            or record.consecutive_failure_count + 1 >= config.hard_failure_threshold
        )
        if (
            result.replay_status == "VERIFIED_FAILURE"
            and hard
            and repeated_source
            and result.verified_coverage >= config.min_verified_candidate_coverage
            and bool(record.ddl_evidence.get("max_counterfactual_ddl_regret", 0.0) > 0.0)
        ):
            return True
    return False
