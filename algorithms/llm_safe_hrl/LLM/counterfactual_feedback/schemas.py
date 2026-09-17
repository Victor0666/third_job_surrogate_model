"""Serializable schemas and bounded configuration for counterfactual feedback."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def canonical_json(payload: Any) -> str:
    """Return stable JSON used by all counterfactual hashes."""
    return json.dumps(
        payload,
        default=_json_default,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class Serializable:
    """Dataclass mixin with an audit-friendly dictionary representation."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateTaskSnapshot(Serializable):
    task_id: int
    features: dict[str, float]
    rule_score: float
    rule_rank: int
    workflow_id: int
    low_or_negative_slack: bool
    high_upward_rank: bool
    uncertainty_level: float


@dataclass(frozen=True)
class DecisionTrace(Serializable):
    run_id: str
    decision_id: str
    structure_hash: str
    frozen_rule_hash: str
    scenario_id: str
    seed: int
    decision_index: int
    current_time: float
    queue_size: int
    ready_task_ids: list[int]
    selected_task_id: int
    selected_task_rank: int
    selected_task_score: float
    score_margin_to_runner_up: float
    minimum_slack: float
    maximum_uncertainty: float
    host_load_summary: dict[str, float]
    workflow_id: int
    parameter_hash: str
    is_critical: bool
    critical_reasons: list[str]
    criticality_score: float = 0.0
    candidate_tasks: list[CandidateTaskSnapshot] = field(default_factory=list)
    workflow_type: str = "unknown"


@dataclass(frozen=True)
class CounterfactualOutcome(Serializable):
    task_id: int
    workflow_id: int
    vm_id: int
    host_id: int
    execution_time: float
    communication_time: float
    queue_time: float
    predicted_finish: float
    optimistic_finish: float
    modal_finish: float
    pessimistic_finish: float
    workflow_optimistic_finish: float
    workflow_modal_finish: float
    workflow_pessimistic_finish: float
    ddl_risk: float
    task_safe_deadline: float
    predicted_violation: float
    safety_margin: float
    marginal_fuzzy_energy: float
    host_load_before: float
    host_load_after: float
    host_load_delta: float
    released_successors: int
    released_critical_successors: int
    remaining_critical_path: float
    uncertainty_width: float
    uncertainty_low_slack_risk: float
    server_active_time_extension: float
    evidence_quality: str


@dataclass(frozen=True)
class CounterfactualComparison(Serializable):
    """Selected-versus-alternative evidence.

    Every delta is a regret: a positive number means that the selected task is
    worse than the alternative for that measure. For safety margins and
    released successors, whose natural direction is larger-is-better, the
    subtraction is reversed to preserve this single sign convention.
    """

    decision_id: str
    selected_task_id: int
    alternative_task_id: int
    selected_outcome: CounterfactualOutcome
    alternative_outcome: CounterfactualOutcome
    ddl_risk_delta: float
    predicted_violation_delta: float
    safety_margin_delta: float
    marginal_fuzzy_energy_delta: float
    communication_delta: float
    predicted_finish_delta: float
    pessimistic_finish_delta: float
    released_successor_delta: int
    released_critical_successor_delta: int
    remaining_critical_path_delta: float
    uncertainty_risk_delta: float
    evidence_quality: str
    estimator_type: str
    dominance: str
    elapsed_ms: float
    state_fingerprint_before: str
    state_fingerprint_after: str


@dataclass(frozen=True)
class DiagnosticReport(Serializable):
    agent_name: str
    decision_id: str
    scenario_id: str
    seed: int
    evidence: dict[str, Any]
    diagnosis_code: str
    severity: str
    confidence: str
    state_conditions: dict[str, Any]
    preferred_task: int | None
    rejected_task: int | None
    quantitative_deltas: dict[str, float]
    suggested_structural_actions: list[str]
    limitations: list[str]


@dataclass(frozen=True)
class AggregatedFeedback(Serializable):
    summary_version: str
    structure_hash: str
    frozen_rule_hash: str
    analyzed_runs: int
    analyzed_decisions: int
    compared_alternatives: int
    repeated_failure_patterns: list[dict[str, Any]]
    ddl_patterns: list[dict[str, Any]]
    energy_patterns: list[dict[str, Any]]
    uncertainty_patterns: list[dict[str, Any]]
    cross_agent_consensus: list[dict[str, Any]]
    cross_agent_conflicts: list[dict[str, Any]]
    high_confidence_structural_actions: list[dict[str, Any]]
    medium_confidence_actions: list[dict[str, Any]]
    insufficient_evidence_items: list[dict[str, Any]]
    representative_cases: list[dict[str, Any]]
    limitations: list[str]
    config_hash: str
    diagnostics_hash: str


@dataclass(frozen=True)
class CounterfactualConfig(Serializable):
    enabled: bool = True
    run_after_parameter_optimization: bool = True
    max_structures_per_generation: int = 2
    include_top_feasible: int = 1
    include_infeasible_low_energy: int = 1
    scenario_ids: tuple[str, ...] = ("SS", "MS")
    train_seed_count: int = 2
    validation_seed_count: int = 0
    max_traced_decisions_per_run: int = 200
    max_critical_decisions_per_run: int = 12
    store_full_ready_ids: bool = True
    store_full_ready_features: bool = False
    max_alternatives_per_decision: int = 3
    include_rule_runner_up: bool = True
    include_min_slack: bool = True
    include_max_upward_rank: bool = True
    include_min_energy: bool = True
    include_max_uncertainty: bool = True
    slack_threshold: float = 0.0
    uncertainty_threshold: float = 1.0
    score_margin_threshold: float = 0.05
    queue_congestion_threshold: int = 4
    estimator_mode: str = "local_one_step"
    bounded_replay_enabled: bool = False
    bounded_replay_steps: int = 0
    assert_state_immutability: bool = True
    min_repeated_evidence: int = 2
    high_confidence_seed_count: int = 2
    high_confidence_scenario_count: int = 2
    max_representative_cases: int = 4
    max_feedback_chars: int = 12000
    cache_enabled: bool = True
    cache_path: str = "counterfactual_feedback/cache.json"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CounterfactualConfig":
        raw = dict(value or {})
        trace = dict(raw.pop("trace", {}) or {})
        alternatives = dict(raw.pop("alternatives", {}) or {})
        criticality = dict(raw.pop("criticality", {}) or {})
        estimator = dict(raw.pop("estimator", {}) or {})
        aggregation = dict(raw.pop("aggregation", {}) or {})
        cache = dict(raw.pop("cache", {}) or {})
        aliases = {
            "max_traced_decisions_per_run": trace.get("max_traced_decisions_per_run"),
            "max_critical_decisions_per_run": trace.get("max_critical_decisions_per_run"),
            "store_full_ready_ids": trace.get("store_full_ready_ids"),
            "store_full_ready_features": trace.get("store_full_ready_features"),
            "max_alternatives_per_decision": alternatives.get("max_per_decision"),
            "include_rule_runner_up": alternatives.get("include_rule_runner_up"),
            "include_min_slack": alternatives.get("include_min_slack"),
            "include_max_upward_rank": alternatives.get("include_max_upward_rank"),
            "include_min_energy": alternatives.get("include_min_energy"),
            "include_max_uncertainty": alternatives.get("include_max_uncertainty"),
            "slack_threshold": criticality.get("slack_threshold"),
            "uncertainty_threshold": criticality.get("uncertainty_threshold"),
            "score_margin_threshold": criticality.get("score_margin_threshold"),
            "queue_congestion_threshold": criticality.get("queue_congestion_threshold"),
            "estimator_mode": estimator.get("mode"),
            "bounded_replay_enabled": estimator.get("bounded_replay_enabled"),
            "bounded_replay_steps": estimator.get("bounded_replay_steps"),
            "assert_state_immutability": estimator.get("assert_state_immutability"),
            "min_repeated_evidence": aggregation.get("min_repeated_evidence"),
            "high_confidence_seed_count": aggregation.get("high_confidence_seed_count"),
            "high_confidence_scenario_count": aggregation.get("high_confidence_scenario_count"),
            "max_representative_cases": aggregation.get("max_representative_cases"),
            "max_feedback_chars": aggregation.get("max_feedback_chars"),
            "cache_enabled": cache.get("enabled"),
            "cache_path": cache.get("path"),
        }
        for name, item in aliases.items():
            if item is not None:
                raw[name] = item
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown counterfactual configuration fields: {unknown}")
        if "scenario_ids" in raw:
            raw["scenario_ids"] = tuple(str(v).upper() for v in raw["scenario_ids"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "max_structures_per_generation": self.max_structures_per_generation,
            "max_traced_decisions_per_run": self.max_traced_decisions_per_run,
            "max_critical_decisions_per_run": self.max_critical_decisions_per_run,
            "max_alternatives_per_decision": self.max_alternatives_per_decision,
            "max_representative_cases": self.max_representative_cases,
            "max_feedback_chars": self.max_feedback_chars,
        }
        for name, number in positive.items():
            if int(number) <= 0:
                raise ValueError(f"counterfactual_feedback.{name} must be positive")
        if self.max_critical_decisions_per_run > self.max_traced_decisions_per_run:
            raise ValueError("critical decision limit cannot exceed trace limit")
        if self.estimator_mode != "local_one_step":
            raise ValueError("Only estimator.mode='local_one_step' is implemented")
        if self.bounded_replay_enabled or self.bounded_replay_steps:
            raise ValueError("bounded replay is unavailable without a safe environment snapshot API")
        if not self.scenario_ids:
            raise ValueError("counterfactual scenario_ids cannot be empty")
        if self.train_seed_count < 0 or self.validation_seed_count < 0:
            raise ValueError("counterfactual seed counts cannot be negative")

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())


def reject_test_seeds(analysis_seeds: list[int], test_seeds: list[int]) -> None:
    """Enforce the experimental split before any trace is produced."""
    overlap = sorted(set(map(int, analysis_seeds)).intersection(map(int, test_seeds)))
    if overlap:
        raise ValueError(
            "Final test seeds are forbidden in counterfactual feedback: "
            + ",".join(map(str, overlap))
        )
