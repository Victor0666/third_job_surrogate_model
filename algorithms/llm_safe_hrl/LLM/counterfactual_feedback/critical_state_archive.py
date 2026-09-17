"""Persistent, diverse archive built from existing counterfactual evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .schemas import CounterfactualComparison, DecisionTrace, DiagnosticReport, canonical_hash
from .trace_recorder import FEATURE_NAMES
from .critical_state_schemas import (
    ARCHIVE_VERSION,
    CONFIDENCE_ORDER,
    EVIDENCE_QUALITY_ORDER,
    RISK_CATEGORIES,
    CriticalStateRecord,
    CriticalStateReplayConfig,
    ReplayResult,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round(value: float, precision: int) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("critical-state signature contains non-finite values")
    return round(number, precision)


def _feature_rows(trace: DecisionTrace) -> list[dict]:
    snapshots = {int(row.task_id): row for row in trace.candidate_tasks}
    if set(snapshots) != set(map(int, trace.ready_task_ids)):
        raise ValueError("critical trace does not contain the complete ready-task feature set")
    rows = []
    for task_id in trace.ready_task_ids:
        snapshot = snapshots[int(task_id)]
        if set(snapshot.features) != set(FEATURE_NAMES):
            raise ValueError("critical trace has an incomplete eight-feature snapshot")
        rows.append(
            {
                "task_id": int(task_id),
                "features": {name: float(snapshot.features[name]) for name in FEATURE_NAMES},
                "score": float(snapshot.rule_score),
                "rank": int(snapshot.rule_rank),
            }
        )
    return rows


def state_signatures(
    trace: DecisionTrace,
    risk_category: str,
    precision: int,
    *,
    comparisons: Iterable[CounterfactualComparison] = (),
    reports: Iterable[DiagnosticReport] = (),
    config_hash: str = "",
) -> tuple[str, str]:
    """Return exact and task-ID-anonymous semantic hashes for a decision state.

    Exact signatures include rule-dependent ranking and stable counterfactual
    evidence. Runtime latency is deliberately excluded. Semantic signatures
    remain task-ID and rule anonymous so compatible historical states can be
    clustered across structures and seeds.
    """
    rows = _feature_rows(trace)
    stable_comparisons = sorted(
        (
            {
                "selected_task_id": int(row.selected_task_id),
                "alternative_task_id": int(row.alternative_task_id),
                "ddl_risk_delta": _round(row.ddl_risk_delta, max(precision, 8)),
                "predicted_violation_delta": _round(
                    row.predicted_violation_delta, max(precision, 8)
                ),
                "safety_margin_delta": _round(
                    row.safety_margin_delta, max(precision, 8)
                ),
                "marginal_fuzzy_energy_delta": _round(
                    row.marginal_fuzzy_energy_delta, max(precision, 8)
                ),
                "pessimistic_finish_delta": _round(
                    row.pessimistic_finish_delta, max(precision, 8)
                ),
                "released_critical_successor_delta": int(
                    row.released_critical_successor_delta
                ),
                "uncertainty_risk_delta": _round(
                    row.uncertainty_risk_delta, max(precision, 8)
                ),
                "evidence_quality": str(row.evidence_quality),
                "estimator_type": str(row.estimator_type),
                "state_fingerprint_before": str(row.state_fingerprint_before),
                "state_fingerprint_after": str(row.state_fingerprint_after),
                "dominance": str(row.dominance),
            }
            for row in comparisons
        ),
        key=lambda value: (
            value["selected_task_id"],
            value["alternative_task_id"],
            canonical_hash(value),
        ),
    )
    stable_reports = sorted(
        (
            {
                "agent_name": str(row.agent_name),
                "diagnosis_code": str(row.diagnosis_code),
                "severity": str(row.severity),
                "confidence": str(row.confidence),
                "preferred_task": (
                    None if row.preferred_task is None else int(row.preferred_task)
                ),
                "rejected_task": (
                    None if row.rejected_task is None else int(row.rejected_task)
                ),
                "quantitative_deltas": row.quantitative_deltas,
            }
            for row in reports
        ),
        key=lambda value: (
            value["agent_name"],
            value["diagnosis_code"],
            canonical_hash(value),
        ),
    )
    exact_payload = {
        "signature_schema": "critical_state_exact_signature_v2",
        "feature_schema": "cews_ready_task_features_v1",
        "config_hash": str(config_hash),
        "scenario": trace.scenario_id,
        "workflow_type": trace.workflow_type,
        "risk_category": risk_category,
        "selected": {
            "task_id": int(trace.selected_task_id),
            "rank": int(trace.selected_task_rank),
            "score": _round(trace.selected_task_score, max(precision, 8)),
        },
        "criticality": {
            "score": _round(trace.criticality_score, max(precision, 8)),
            "reasons": sorted(map(str, trace.critical_reasons)),
        },
        "ready": [
            {
                "task_id": row["task_id"],
                "score": _round(row["score"], max(precision, 8)),
                "rank": int(row["rank"]),
                "features": {
                    name: _round(row["features"][name], max(precision, 8))
                    for name in FEATURE_NAMES
                },
            }
            for row in rows
        ],
        "load": {
            key: _round(value, max(precision, 8))
            for key, value in sorted(trace.host_load_summary.items())
        },
        "counterfactual_evidence": stable_comparisons,
        "diagnostic_evidence": stable_reports,
    }

    matrix = np.asarray(
        [[row["features"][name] for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )
    lower = np.min(matrix, axis=0)
    span = np.max(matrix, axis=0) - lower
    normalized = np.divide(
        matrix - lower,
        np.where(span > 1e-12, span, 1.0),
    )
    anonymous_rows = sorted(
        [tuple(round(float(value), precision) for value in row) for row in normalized]
    )
    slack = matrix[:, FEATURE_NAMES.index("slack")]
    upward = matrix[:, FEATURE_NAMES.index("upward_rank")]
    uncertainty = matrix[:, FEATURE_NAMES.index("uncertainty")]
    semantic_payload = {
        "feature_schema": "cews_ready_task_features_v1",
        "scenario": trace.scenario_id,
        "workflow_type": trace.workflow_type,
        "risk_category": risk_category,
        "ready_count": len(rows),
        "anonymous_normalized_features": anonymous_rows,
        "slack_distribution": [
            round(float(np.min(slack)), precision),
            round(float(np.median(slack)), precision),
            round(float(np.max(slack)), precision),
            int(np.sum(slack < 0.0)),
        ],
        "upward_rank_distribution": sorted(
            round(float(value), precision) for value in upward
        ),
        "uncertainty_distribution": [
            round(float(np.min(uncertainty)), precision),
            round(float(np.median(uncertainty)), precision),
            round(float(np.max(uncertainty)), precision),
        ],
        "load_bin": int(
            min(9, max(0, math.floor(float(trace.host_load_summary.get("maximum", 0.0)) * 10.0)))
        ),
    }
    return canonical_hash(exact_payload), canonical_hash(semantic_payload)


def _risk_categories(
    trace: DecisionTrace,
    comparisons: list[CounterfactualComparison],
    reports: list[DiagnosticReport],
    config: CriticalStateReplayConfig,
) -> list[str]:
    categories = set()
    rows = _feature_rows(trace)
    slacks = [row["features"]["slack"] for row in rows]
    if min(slacks) < 0.0:
        categories.add("NEGATIVE_SLACK")
    if sum(abs(value) <= config.near_zero_slack_threshold for value in slacks) >= 2:
        categories.add("MULTI_NEAR_ZERO_SLACK")
    if trace.queue_size >= 4 or "ready_queue_congestion" in trace.critical_reasons:
        categories.add("READY_QUEUE_CONGESTION")
    if any(
        reason in trace.critical_reasons
        for reason in ("selected_not_maximum_upward_rank", "critical_path_competition")
    ):
        categories.add("CRITICAL_PATH_STARVATION")
    if "high_uncertainty" in trace.critical_reasons:
        categories.add("UNCERTAINTY_SPIKE")
    if float(trace.host_load_summary.get("maximum", 0.0)) >= config.resource_bottleneck_threshold:
        categories.add("RESOURCE_BOTTLENECK")
    if any(
        row.marginal_fuzzy_energy_delta < 0.0
        and (row.predicted_violation_delta > 1e-9 or row.safety_margin_delta > 1e-9)
        for row in comparisons
    ):
        categories.add("DDL_ENERGY_CONFLICT")
    if any(
        row.predicted_violation_delta > 1e-9
        or row.safety_margin_delta > 1e-9
        for row in comparisons
    ):
        categories.add("COUNTERFACTUAL_DOMINATED_CHOICE")
    if any(
        "modal_safe_pessimistic_violation" in row.diagnosis_code
        for row in reports
    ):
        categories.add("PESSIMISTIC_TIMELINE_RISK")
    if any(row.released_critical_successor_delta > 0 for row in comparisons):
        categories.add("SUCCESSOR_RELEASE_BLOCKING")
    order = {name: index for index, name in enumerate(RISK_CATEGORIES)}
    return sorted(categories, key=lambda name: order[name])


def build_critical_state_record(
    trace: DecisionTrace,
    comparisons: Iterable[CounterfactualComparison],
    reports: Iterable[DiagnosticReport],
    *,
    generation: int,
    config: CriticalStateReplayConfig,
) -> CriticalStateRecord | None:
    """Admit only difficult states with immutable, sufficient failure evidence."""
    rows = [row for row in comparisons if row.decision_id == trace.decision_id]
    diagnostic_rows = [row for row in reports if row.decision_id == trace.decision_id]
    categories = _risk_categories(trace, rows, diagnostic_rows, config)
    feature_rows = _feature_rows(trace)
    slacks = [row["features"]["slack"] for row in feature_rows]
    if not trace.is_critical or not categories:
        return None
    if not rows or any(
        row.state_fingerprint_before != row.state_fingerprint_after
        or row.estimator_type != "local_one_step"
        for row in rows
    ):
        return None
    quality_rows = [
        row for row in rows
        if EVIDENCE_QUALITY_ORDER.get(row.evidence_quality, -1)
        >= EVIDENCE_QUALITY_ORDER[config.min_counterfactual_evidence_quality]
    ]
    if not quality_rows:
        return None
    quality_rows = sorted(
        quality_rows,
        key=lambda row: (
            -float(max(row.predicted_violation_delta, 0.0)),
            -float(max(row.safety_margin_delta, 0.0)),
            -int(max(row.released_critical_successor_delta, 0)),
            -float(max(row.pessimistic_finish_delta, 0.0)),
            -float(max(row.uncertainty_risk_delta, 0.0)),
            int(row.alternative_task_id),
        ),
    )[: config.candidate_outcome_expansion_cap]
    substantive = [
        row for row in diagnostic_rows
        if row.diagnosis_code != "insufficient_evidence"
        and CONFIDENCE_ORDER.get(row.confidence, -1)
        >= CONFIDENCE_ORDER[config.min_admission_confidence]
    ]
    quality_by_alternative = {
        int(row.alternative_task_id): row for row in quality_rows
    }
    supported_reports = [
        row for row in substantive
        if row.preferred_task is not None
        and int(row.preferred_task) in quality_by_alternative
        and quality_by_alternative[int(row.preferred_task)].predicted_violation_delta >= -1e-9
    ]
    material_rows = [
        row for row in quality_rows
        if row.predicted_violation_delta > 1e-9
        or (
            row.predicted_violation_delta >= -1e-9
            and (
                row.safety_margin_delta > 1e-9
                or row.released_critical_successor_delta > 0
                or row.pessimistic_finish_delta > 1e-9
                or row.uncertainty_risk_delta > 1e-9
            )
        )
    ]
    if not supported_reports and not material_rows:
        return None

    preferred = {
        int(row.preferred_task)
        for row in supported_reports
    }
    preferred.update(int(row.alternative_task_id) for row in material_rows)
    dominated = {
        int(row.rejected_task)
        for row in supported_reports
        if row.rejected_task is not None
    }
    if preferred:
        dominated.add(int(trace.selected_task_id))
    if not preferred and not dominated:
        return None
    evidence_rows = [
        row for row in quality_rows
        if int(row.alternative_task_id) in preferred
    ]

    ready_ids = [int(row["task_id"]) for row in feature_rows]
    verified_ids = preferred | dominated
    coverage = len(verified_ids.intersection(ready_ids)) / max(len(ready_ids), 1)
    primary_order = (
        "COUNTERFACTUAL_DOMINATED_CHOICE",
        "PESSIMISTIC_TIMELINE_RISK",
        "SUCCESSOR_RELEASE_BLOCKING",
        "DDL_ENERGY_CONFLICT",
        "NEGATIVE_SLACK",
        "CRITICAL_PATH_STARVATION",
        "UNCERTAINTY_SPIKE",
        "RESOURCE_BOTTLENECK",
        "READY_QUEUE_CONGESTION",
        "MULTI_NEAR_ZERO_SLACK",
    )
    primary = next(name for name in primary_order if name in categories)
    exact, semantic = state_signatures(
        trace,
        primary,
        config.semantic_signature_precision,
        comparisons=quality_rows,
        reports=supported_reports,
        config_hash=config.config_hash,
    )
    confidence = max(
        (row.confidence for row in supported_reports),
        key=lambda value: CONFIDENCE_ORDER[value],
        default="medium",
    )
    report_by_agent = defaultdict(list)
    for row in supported_reports:
        report_by_agent[row.agent_name].append(row.to_dict())
    max_ddl_regret = max(
        max(row.predicted_violation_delta, row.safety_margin_delta, 0.0)
        for row in evidence_rows
    )
    energy_regret = max(
        (row.marginal_fuzzy_energy_delta for row in evidence_rows),
        default=0.0,
    )
    ddl_severe = float(
        min(slacks) < 0.0
        or max_ddl_regret > 1e-9
        or any(row.selected_outcome.safety_margin < 0.0 for row in quality_rows)
    )
    priority = [
        ddl_severe,
        float(CONFIDENCE_ORDER[confidence]),
        1.0,
        0.0,
        float(max_ddl_regret),
        1.0,
        float(generation + 1),
        float(max(0.0, energy_regret)),
    ]
    state_id = canonical_hash(
        {"exact_signature": exact, "source_run_id": trace.run_id, "decision": trace.decision_index}
    )
    record = CriticalStateRecord(
        state_id=state_id,
        state_signature=exact,
        exact_signature=exact,
        semantic_signature=semantic,
        source_structure_hash=trace.structure_hash,
        source_frozen_rule_hash=trace.frozen_rule_hash,
        source_parameter_hash=trace.parameter_hash,
        source_generation=int(generation),
        source_scenario_id=trace.scenario_id,
        source_seed=int(trace.seed),
        source_run_id=trace.run_id,
        source_decision_index=int(trace.decision_index),
        current_time=float(trace.current_time),
        workflow_type=trace.workflow_type,
        workflow_id=int(trace.workflow_id),
        ready_task_ids=ready_ids,
        ready_task_features={str(row["task_id"]): row["features"] for row in feature_rows},
        ready_task_scores={str(row["task_id"]): row["score"] for row in feature_rows},
        selected_task_id=int(trace.selected_task_id),
        selected_task_rank=int(trace.selected_task_rank),
        selected_task_score=float(trace.selected_task_score),
        criticality_score=float(trace.criticality_score),
        critical_reasons=list(trace.critical_reasons),
        counterfactual_comparisons=[row.to_dict() for row in quality_rows],
        preferred_task_ids=sorted(preferred.intersection(ready_ids)),
        dominated_task_ids=sorted(dominated.intersection(ready_ids)),
        unverified_task_ids=sorted(set(ready_ids) - verified_ids),
        ddl_evidence={
            "reports": report_by_agent.get("ddl_diagnostic_agent", []),
            "max_counterfactual_ddl_regret": float(max_ddl_regret),
            "negative_slack": min(slacks) < 0.0,
        },
        energy_evidence={
            "reports": report_by_agent.get("energy_diagnostic_agent", []),
            "max_energy_regret_within_ddl_class": float(max(0.0, energy_regret)),
        },
        uncertainty_evidence={
            "reports": report_by_agent.get("uncertainty_diagnostic_agent", []),
            "max_uncertainty_regret": float(
                max((row.uncertainty_risk_delta for row in evidence_rows), default=0.0)
            ),
        },
        diagnosis_codes=sorted({row.diagnosis_code for row in supported_reports}),
        confidence=confidence,
        risk_category=primary,
        auxiliary_risk_categories=[name for name in categories if name != primary],
        archive_priority=priority,
        replay_count=0,
        failure_count=0,
        success_count=0,
        unresolved_count=0,
        consecutive_success_count=0,
        consecutive_failure_count=0,
        created_generation=int(generation),
        last_replayed_generation=None,
        status="NEW",
        content_hash="",
        config_hash=config.config_hash,
        verified_coverage=float(coverage),
        source_seeds=[int(trace.seed)],
        source_scenarios=[trace.scenario_id],
        source_structures=[trace.structure_hash],
    )
    record.refresh_hash()
    record.validate()
    return record


class CriticalStateArchive:
    """Cross-generation archive with exact deduplication and semantic clusters."""

    def __init__(
        self,
        config: CriticalStateReplayConfig,
        *,
        path: str | Path,
        train_seeds: Iterable[int],
        validation_seeds: Iterable[int],
        test_seeds: Iterable[int],
    ):
        self.config = config
        self.path = Path(path).resolve()
        self.train_seeds = {int(seed) for seed in train_seeds}
        self.validation_seeds = {int(seed) for seed in validation_seeds}
        self.test_seeds = {int(seed) for seed in test_seeds}
        self.created_at = _now()
        self.last_updated_generation = -1
        self.records: dict[str, CriticalStateRecord] = {}
        self.clusters: dict[str, list[str]] = {}
        self.eviction_log: list[dict] = []

    @classmethod
    def load_or_create(
        cls,
        config: CriticalStateReplayConfig,
        *,
        path: str | Path,
        train_seeds: Iterable[int],
        validation_seeds: Iterable[int],
        test_seeds: Iterable[int],
    ) -> "CriticalStateArchive":
        archive = cls(
            config,
            path=path,
            train_seeds=train_seeds,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
        )
        if not archive.path.is_file():
            return archive
        payload = json.loads(archive.path.read_text(encoding="utf-8"))
        stored_hash = payload.pop("archive_hash", "")
        if canonical_hash(payload) != stored_hash:
            raise ValueError("critical-state archive hash mismatch")
        if payload.get("archive_version") != ARCHIVE_VERSION:
            raise ValueError("unsupported critical-state archive version")
        if payload.get("config_hash") != config.config_hash:
            raise ValueError("critical-state archive configuration mismatch")
        archive.created_at = str(payload["created_at"])
        archive.last_updated_generation = int(payload["last_updated_generation"])
        archive.records = {
            row["state_id"]: CriticalStateRecord.from_dict(row)
            for row in payload.get("state_records", [])
        }
        archive.clusters = {
            str(key): list(value) for key, value in payload.get("cluster_metadata", {}).items()
        }
        archive.eviction_log = list(payload.get("eviction_log", []))
        archive.validate()
        return archive

    def _priority_key(self, record: CriticalStateRecord) -> tuple:
        ddl_severity = float(record.archive_priority[0])
        confidence = CONFIDENCE_ORDER[record.confidence]
        repeated = max(len(record.source_seeds), len(record.source_scenarios))
        ddl_regret = float(record.ddl_evidence.get("max_counterfactual_ddl_regret", 0.0))
        novelty = 1.0 / max(len(self.clusters.get(record.semantic_signature, [])), 1)
        recency = (
            self.last_updated_generation + 1
            if record.last_replayed_generation is None
            else max(0, self.last_updated_generation - record.last_replayed_generation)
        )
        energy = float(record.energy_evidence.get("max_energy_regret_within_ddl_class", 0.0))
        return (
            ddl_severity,
            confidence,
            ddl_regret,
            record.consecutive_failure_count,
            repeated,
            novelty,
            recency,
            energy,
            record.state_id,
        )

    def _refresh_priority(self, record: CriticalStateRecord) -> None:
        record.archive_priority = [float(value) for value in self._priority_key(record)[:-1]]
        record.refresh_hash()

    def admit(self, record: CriticalStateRecord) -> tuple[bool, str]:
        record.validate()
        if record.source_seed in self.test_seeds:
            raise ValueError("Final test seeds are forbidden in the critical-state archive")
        if record.source_seed in self.validation_seeds and not self.config.allow_validation_archive:
            return False, "validation_seed_read_only"
        if self.train_seeds and record.source_seed not in self.train_seeds:
            return False, "seed_not_in_training_archive_split"
        if CONFIDENCE_ORDER[record.confidence] < CONFIDENCE_ORDER[self.config.min_admission_confidence]:
            return False, "insufficient_confidence"
        return True, "admitted"

    def add_or_merge(self, record: CriticalStateRecord) -> tuple[CriticalStateRecord | None, str]:
        admitted, reason = self.admit(record)
        if not admitted:
            return None, reason
        if not self.config.semantic_similarity_enabled:
            record.semantic_signature = record.exact_signature
            record.refresh_hash()
        exact_match = next(
            (
                row
                for row in self.records.values()
                if row.exact_signature == record.exact_signature
                and self._exact_records_compatible(row, record)
            ),
            None,
        )
        if exact_match is not None and self.config.exact_duplicate_merge:
            exact_match.merged_evidence_count += 1
            exact_match.source_seeds = sorted(set(exact_match.source_seeds + record.source_seeds))
            exact_match.source_scenarios = sorted(set(exact_match.source_scenarios + record.source_scenarios))
            exact_match.source_structures = sorted(set(exact_match.source_structures + record.source_structures))
            exact_match.diagnosis_codes = sorted(set(exact_match.diagnosis_codes + record.diagnosis_codes))
            exact_match.critical_reasons = sorted(
                set(exact_match.critical_reasons + record.critical_reasons)
            )
            exact_match.auxiliary_risk_categories = sorted(
                set(
                    exact_match.auxiliary_risk_categories
                    + record.auxiliary_risk_categories
                )
            )
            exact_match.counterfactual_comparisons = self._merge_evidence_rows(
                exact_match.counterfactual_comparisons,
                record.counterfactual_comparisons,
            )
            exact_match.ddl_evidence = self._merge_evidence_mapping(
                exact_match.ddl_evidence,
                record.ddl_evidence,
            )
            exact_match.energy_evidence = self._merge_evidence_mapping(
                exact_match.energy_evidence,
                record.energy_evidence,
            )
            exact_match.uncertainty_evidence = self._merge_evidence_mapping(
                exact_match.uncertainty_evidence,
                record.uncertainty_evidence,
            )
            exact_match.confidence = max(
                (exact_match.confidence, record.confidence),
                key=lambda value: CONFIDENCE_ORDER[value],
            )
            exact_match.preferred_task_ids = sorted(
                set(exact_match.preferred_task_ids + record.preferred_task_ids)
            )
            exact_match.dominated_task_ids = sorted(
                set(exact_match.dominated_task_ids + record.dominated_task_ids)
                - set(exact_match.preferred_task_ids)
            )
            exact_match.unverified_task_ids = sorted(
                set(exact_match.ready_task_ids)
                - set(exact_match.preferred_task_ids)
                - set(exact_match.dominated_task_ids)
            )
            exact_match.verified_coverage = (
                len(
                    set(exact_match.preferred_task_ids)
                    | set(exact_match.dominated_task_ids)
                )
                / max(len(exact_match.ready_task_ids), 1)
            )
            if exact_match.status in {"RESOLVED", "DORMANT"}:
                exact_match.status = "ACTIVE"
                exact_match.consecutive_success_count = 0
            self._refresh_priority(exact_match)
            self.last_updated_generation = max(self.last_updated_generation, record.source_generation)
            return exact_match, "merged_exact_duplicate"
        self.records[record.state_id] = record
        self.clusters.setdefault(record.semantic_signature, []).append(record.state_id)
        self.clusters[record.semantic_signature] = sorted(set(self.clusters[record.semantic_signature]))
        for state_id in self.clusters[record.semantic_signature]:
            if state_id == record.state_id:
                continue
            similar = self.records.get(state_id)
            if similar is not None and similar.status == "RESOLVED":
                similar.status = "ACTIVE"
                similar.consecutive_success_count = 0
                self._refresh_priority(similar)
        self.last_updated_generation = max(self.last_updated_generation, record.source_generation)
        self.evict()
        return record, "added"

    @staticmethod
    def _exact_records_compatible(
        first: CriticalStateRecord,
        second: CriticalStateRecord,
    ) -> bool:
        """Reject forged or legacy exact collisions with incompatible evidence."""
        fields = (
            "config_hash",
            "source_scenario_id",
            "workflow_type",
            "risk_category",
            "ready_task_ids",
            "ready_task_features",
            "ready_task_scores",
            "selected_task_id",
            "selected_task_rank",
            "selected_task_score",
            "criticality_score",
            "preferred_task_ids",
            "dominated_task_ids",
        )
        return all(getattr(first, name) == getattr(second, name) for name in fields)

    @staticmethod
    def _merge_evidence_rows(first: list[dict], second: list[dict]) -> list[dict]:
        rows = {canonical_hash(row): row for row in [*first, *second]}
        return [rows[key] for key in sorted(rows)]

    @classmethod
    def _merge_evidence_mapping(cls, first: dict, second: dict) -> dict:
        merged = json.loads(json.dumps(first))
        for key, value in second.items():
            if key == "reports" and isinstance(value, list):
                merged[key] = cls._merge_evidence_rows(
                    list(merged.get(key, [])),
                    value,
                )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = max(float(merged.get(key, value)), float(value))
            elif isinstance(value, bool):
                merged[key] = bool(merged.get(key, False) or value)
            elif key not in merged:
                merged[key] = value
        return merged

    def update_after_replay(
        self,
        result: ReplayResult,
        *,
        replay_structure_hash: str,
    ) -> CriticalStateRecord:
        record = self.records[result.state_id]
        record.replay_count += 1
        record.last_replayed_generation = int(result.generation)
        record.replay_structure_hashes = sorted(
            set(record.replay_structure_hashes + [str(replay_structure_hash)])
        )
        outcome = {
            "generation": int(result.generation),
            "structure_hash": str(replay_structure_hash),
            "replay_status": result.replay_status,
            "result_hash": result.result_hash,
        }
        record.replay_outcomes.append(outcome)
        del record.replay_outcomes[:-30]
        if result.replay_status == "VERIFIED_FAILURE":
            record.failure_count += 1
            record.unresolved_count += 1
            record.consecutive_failure_count += 1
            record.consecutive_success_count = 0
            record.failure_generations = sorted(set(record.failure_generations + [result.generation]))
            record.status = "HARD" if (
                record.status == "HARD"
                or record.consecutive_failure_count >= self.config.hard_failure_threshold
            ) else "ACTIVE"
        elif result.replay_status == "VERIFIED_SUCCESS":
            record.success_count += 1
            record.consecutive_success_count += 1
            record.consecutive_failure_count = 0
            record.success_generations = sorted(set(record.success_generations + [result.generation]))
            stable_sources = (
                len(record.replay_structure_hashes) >= 2
                or len(record.success_generations) >= 2
            )
            record.status = (
                "RESOLVED"
                if record.consecutive_success_count >= self.config.resolved_success_threshold
                and stable_sources
                else "ACTIVE"
            )
        elif result.replay_status in {"PARTIAL_SUCCESS", "UNVERIFIED"}:
            record.unresolved_count += 1
            record.consecutive_success_count = 0
            if record.status not in {"HARD", "RESOLVED"}:
                record.status = "ACTIVE"
        else:
            record.status = "INVALID"
        self.last_updated_generation = max(self.last_updated_generation, result.generation)
        self._refresh_priority(record)
        return record

    def _diverse_take(
        self,
        pool: list[CriticalStateRecord],
        limit: int,
        selected: list[CriticalStateRecord],
    ) -> list[CriticalStateRecord]:
        if limit <= 0:
            return []
        ordered = sorted(pool, key=self._priority_key, reverse=True)
        chosen = []
        cluster_counts = Counter(row.semantic_signature for row in selected)
        represented = {
            (
                row.risk_category,
                row.source_scenario_id,
                row.workflow_type,
                row.source_seed,
                int(float(row.archive_priority[0]) > 0.0),
            )
            for row in selected
        }
        for diversity_pass in (True, False):
            for row in ordered:
                if row in chosen or row in selected:
                    continue
                if cluster_counts[row.semantic_signature] >= self.config.max_states_per_semantic_cluster:
                    continue
                group = (
                    row.risk_category,
                    row.source_scenario_id,
                    row.workflow_type,
                    row.source_seed,
                    int(float(row.archive_priority[0]) > 0.0),
                )
                if diversity_pass and group in represented:
                    continue
                chosen.append(row)
                cluster_counts[row.semantic_signature] += 1
                represented.add(group)
                if len(chosen) >= limit:
                    return chosen
        return chosen

    def sample_for_replay(self) -> list[CriticalStateRecord]:
        """Deterministic HARD/NEW/ACTIVE/RESOLVED stratified selection."""
        total = self.config.max_states_per_generation
        selected: list[CriticalStateRecord] = []
        by_status = defaultdict(list)
        for record in self.records.values():
            if record.status not in {"INVALID", "EVICTED"}:
                by_status[record.status].append(record)
        selected += self._diverse_take(
            by_status["HARD"],
            min(self.config.max_hard_states_per_generation, total),
            selected,
        )
        selected += self._diverse_take(
            by_status["NEW"],
            min(self.config.max_new_states_per_generation, total - len(selected)),
            selected,
        )
        resolved_limit = min(
            len(by_status["RESOLVED"]),
            max(1, int(round(total * self.config.resolved_replay_fraction)))
            if self.config.resolved_replay_fraction > 0.0 else 0,
        )
        active_pool = by_status["ACTIVE"] + by_status["DORMANT"]
        active_budget = max(0, total - len(selected) - resolved_limit)
        selected += self._diverse_take(active_pool, active_budget, selected)
        selected += self._diverse_take(
            by_status["RESOLVED"],
            min(resolved_limit, total - len(selected)),
            selected,
        )
        if len(selected) < total:
            remaining = [
                row for row in self.records.values()
                if row.status not in {"INVALID", "EVICTED"}
            ]
            selected += self._diverse_take(remaining, total - len(selected), selected)
        return selected[:total]

    def _remove(self, record: CriticalStateRecord, reason: str) -> None:
        record.status = "EVICTED"
        record.eviction_reason = reason
        record.refresh_hash()
        self.eviction_log.append(
            {"state_id": record.state_id, "reason": reason, "generation": self.last_updated_generation}
        )
        del self.records[record.state_id]
        cluster = self.clusters.get(record.semantic_signature, [])
        self.clusters[record.semantic_signature] = [value for value in cluster if value != record.state_id]
        if not self.clusters[record.semantic_signature]:
            del self.clusters[record.semantic_signature]

    def _eviction_key(self, record: CriticalStateRecord) -> tuple:
        status_order = {"INVALID": 0, "DORMANT": 1, "RESOLVED": 2, "NEW": 3, "ACTIVE": 4, "HARD": 5}
        return (
            status_order.get(record.status, 6),
            self._priority_key(record),
            record.last_replayed_generation if record.last_replayed_generation is not None else -1,
            record.state_id,
        )

    def _evict_excess(self, rows: list[CriticalStateRecord], excess: int, reason: str) -> None:
        if excess <= 0:
            return
        candidates = {row.state_id for row in rows}
        while excess > 0:
            available = [
                self.records[state_id]
                for state_id in candidates
                if state_id in self.records
            ]
            if not available:
                break
            category_counts = Counter(row.risk_category for row in self.records.values())
            scenario_counts = Counter(row.source_scenario_id for row in self.records.values())
            workflow_counts = Counter(row.workflow_type for row in self.records.values())
            seed_counts = Counter(row.source_seed for row in self.records.values())
            severity_counts = Counter(
                int(float(row.archive_priority[0]) > 0.0)
                for row in self.records.values()
            )
            ordered = sorted(available, key=self._eviction_key)
            diversity_safe = [
                row for row in ordered
                if category_counts[row.risk_category] > 1
                and scenario_counts[row.source_scenario_id] > 1
                and workflow_counts[row.workflow_type] > 1
                and seed_counts[row.source_seed] > 1
                and severity_counts[int(float(row.archive_priority[0]) > 0.0)] > 1
            ]
            selected = (diversity_safe or ordered)[0]
            self._remove(selected, reason)
            excess -= 1

    def evict(self) -> None:
        for signature, state_ids in list(self.clusters.items()):
            rows = [
                self.records[state_id]
                for state_id in state_ids
                if state_id in self.records
            ]
            self._evict_excess(
                rows,
                len(rows) - self.config.max_states_per_semantic_cluster,
                f"semantic_cluster_capacity:{signature}",
            )
        for category in RISK_CATEGORIES:
            rows = [row for row in self.records.values() if row.risk_category == category]
            self._evict_excess(
                rows,
                len(rows) - self.config.per_category_capacity,
                "per_category_capacity",
            )
        for scenario, rows in list(self._group_records("source_scenario_id").items()):
            self._evict_excess(
                rows,
                len(rows) - self.config.max_states_per_scenario,
                f"scenario_capacity:{scenario}",
            )
        for workflow, rows in list(self._group_records("workflow_type").items()):
            self._evict_excess(
                rows,
                len(rows) - self.config.max_states_per_workflow_type,
                f"workflow_capacity:{workflow}",
            )
        self._evict_excess(
            list(self.records.values()),
            len(self.records) - self.config.global_capacity,
            "global_capacity",
        )

    def _group_records(self, field_name: str) -> dict[str, list[CriticalStateRecord]]:
        result = defaultdict(list)
        for record in self.records.values():
            result[str(getattr(record, field_name))].append(record)
        return result

    def compact(self, current_generation: int) -> None:
        for record in self.records.values():
            last = record.last_replayed_generation
            if (
                record.status in {"ACTIVE", "RESOLVED"}
                and last is not None
                and current_generation - last >= self.config.dormant_generation_threshold
            ):
                record.status = "DORMANT"
                self._refresh_priority(record)
        self.last_updated_generation = max(self.last_updated_generation, int(current_generation))
        self.evict()

    def statistics(self) -> dict:
        return {
            "state_count": len(self.records),
            "cluster_count": len(self.clusters),
            "status_counts": dict(sorted(Counter(row.status for row in self.records.values()).items())),
            "risk_category_counts": dict(
                sorted(Counter(row.risk_category for row in self.records.values()).items())
            ),
            "scenario_counts": dict(
                sorted(Counter(row.source_scenario_id for row in self.records.values()).items())
            ),
            "workflow_type_counts": dict(
                sorted(Counter(row.workflow_type for row in self.records.values()).items())
            ),
            "seed_counts": dict(
                sorted(Counter(row.source_seed for row in self.records.values()).items())
            ),
            "ddl_severity_counts": dict(
                sorted(
                    Counter(
                        int(float(row.archive_priority[0]) > 0.0)
                        for row in self.records.values()
                    ).items()
                )
            ),
            "eviction_count": len(self.eviction_log),
        }

    def _payload(self) -> dict:
        return {
            "archive_version": ARCHIVE_VERSION,
            "created_at": self.created_at,
            "last_updated_generation": self.last_updated_generation,
            "global_capacity": self.config.global_capacity,
            "per_category_capacity": self.config.per_category_capacity,
            "state_records": [
                self.records[state_id].to_dict() for state_id in sorted(self.records)
            ],
            "cluster_metadata": {
                key: sorted(value) for key, value in sorted(self.clusters.items())
            },
            "archive_statistics": self.statistics(),
            "eviction_log": list(self.eviction_log[-200:]),
            "config_hash": self.config.config_hash,
            "used_test_seed": False,
        }

    @property
    def archive_hash(self) -> str:
        return canonical_hash(self._payload())

    def validate(self) -> None:
        for record in self.records.values():
            record.validate()
            if record.source_seed in self.test_seeds:
                raise ValueError("critical-state archive contains a final test seed")
        memberships = Counter(
            state_id for values in self.clusters.values() for state_id in values
        )
        if set(memberships) != set(self.records) or any(count != 1 for count in memberships.values()):
            raise ValueError("critical-state semantic cluster metadata is inconsistent")
        for signature, state_ids in self.clusters.items():
            if any(self.records[state_id].semantic_signature != signature for state_id in state_ids):
                raise ValueError("critical-state record is assigned to the wrong semantic cluster")

    def save(self, *, statistics_root: str | Path | None = None) -> dict:
        self.validate()
        payload = self._payload()
        payload["archive_hash"] = canonical_hash(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        states_root = self.path.parent.parent / "states"
        states_root.mkdir(parents=True, exist_ok=True)
        for record in self.records.values():
            (states_root / f"{record.state_id}.json").write_text(
                json.dumps(record.to_dict(), ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
        manifest_path = self.path.with_name("critical_state_archive_manifest.json")
        manifest = {
            "archive_version": ARCHIVE_VERSION,
            "archive_path": str(self.path),
            "archive_hash": payload["archive_hash"],
            "config_hash": self.config.config_hash,
            "state_count": len(self.records),
            "cluster_count": len(self.clusters),
            "used_test_seed": False,
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        if statistics_root is not None:
            statistics_path = Path(statistics_root) / f"archive_generation_{self.last_updated_generation}.json"
            statistics_path.parent.mkdir(parents=True, exist_ok=True)
            statistics_path.write_text(
                json.dumps(self.statistics(), ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )
        return {**manifest, "manifest_path": str(manifest_path)}
