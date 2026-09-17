"""Serializable schemas for persistent critical-state replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

from .schemas import canonical_hash


ARCHIVE_VERSION = "critical_state_archive_v1"
REPLAY_VERSION = "critical_state_feature_replay_v1"
FEATURE_SCHEMA_VERSION = "cews_ready_task_features_v1"
RULE_INTERFACE_VERSION = "get_task_priority_v2_v1"

RISK_CATEGORIES = (
    "NEGATIVE_SLACK",
    "MULTI_NEAR_ZERO_SLACK",
    "READY_QUEUE_CONGESTION",
    "CRITICAL_PATH_STARVATION",
    "UNCERTAINTY_SPIKE",
    "RESOURCE_BOTTLENECK",
    "DDL_ENERGY_CONFLICT",
    "COUNTERFACTUAL_DOMINATED_CHOICE",
    "PESSIMISTIC_TIMELINE_RISK",
    "SUCCESSOR_RELEASE_BLOCKING",
)
STATE_STATUSES = ("NEW", "ACTIVE", "HARD", "RESOLVED", "DORMANT", "EVICTED", "INVALID")
REPLAY_STATUSES = (
    "VERIFIED_SUCCESS",
    "VERIFIED_FAILURE",
    "PARTIAL_SUCCESS",
    "UNVERIFIED",
    "INVALID",
)
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
EVIDENCE_QUALITY_ORDER = {"low": 0, "medium": 1, "high": 2}
READY_TASK_FEATURE_NAMES = (
    "min_exec_time",
    "min_comm_time",
    "min_incremental_energy",
    "slack",
    "upward_rank",
    "remaining_work",
    "ready_wait_time",
    "uncertainty",
)


def _finite_tree(value: Any, path: str = "value") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _finite_tree(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


@dataclass(frozen=True)
class CriticalStateReplayConfig:
    """Central bounded configuration for offline archive and feature replay."""

    enabled: bool = True
    run_after_counterfactual_feedback: bool = True
    strict_replay_gate: bool = False
    global_capacity: int = 300
    per_category_capacity: int = 50
    max_states_per_scenario: int = 120
    max_states_per_workflow_type: int = 120
    exact_duplicate_merge: bool = True
    semantic_similarity_enabled: bool = True
    semantic_signature_precision: int = 4
    min_admission_confidence: str = "medium"
    min_counterfactual_evidence_quality: str = "medium"
    resolved_success_threshold: int = 3
    hard_failure_threshold: int = 2
    dormant_generation_threshold: int = 8
    allow_validation_archive: bool = False
    near_zero_slack_threshold: float = 1.0
    resource_bottleneck_threshold: float = 0.85
    max_states_per_generation: int = 40
    max_hard_states_per_generation: int = 20
    max_new_states_per_generation: int = 10
    resolved_replay_fraction: float = 0.1
    max_states_per_semantic_cluster: int = 3
    candidate_outcome_expansion_cap: int = 6
    min_verified_candidate_coverage: float = 0.5
    deterministic_sampling_seed: int = 0
    min_repeated_failure_count: int = 2
    high_confidence_seed_count: int = 2
    high_confidence_scenario_count: int = 2
    max_failure_examples: int = 5
    max_success_examples: int = 3
    max_feedback_chars: int = 10000
    cache_enabled: bool = True
    archive_path: str = "critical_state_replay/archive/critical_state_archive.json"
    cache_path: str = "critical_state_replay/replay_cache.json"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CriticalStateReplayConfig":
        raw = dict(value or {})
        archive = dict(raw.pop("archive", {}) or {})
        replay = dict(raw.pop("replay", {}) or {})
        feedback = dict(raw.pop("feedback", {}) or {})
        cache = dict(raw.pop("cache", {}) or {})
        raw.update(archive)
        raw.update(replay)
        raw.update(feedback)
        if "enabled" in cache:
            raw["cache_enabled"] = cache["enabled"]
        if "path" in cache:
            raw["cache_path"] = cache["path"]
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown critical_state_replay fields: {unknown}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        positive = (
            "global_capacity", "per_category_capacity", "max_states_per_scenario",
            "max_states_per_workflow_type", "resolved_success_threshold",
            "hard_failure_threshold", "dormant_generation_threshold",
            "max_states_per_generation", "max_hard_states_per_generation",
            "max_new_states_per_generation", "max_states_per_semantic_cluster",
            "candidate_outcome_expansion_cap", "min_repeated_failure_count",
            "high_confidence_seed_count", "high_confidence_scenario_count",
            "max_failure_examples", "max_success_examples", "max_feedback_chars",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"critical_state_replay.{name} must be positive")
        if self.min_admission_confidence not in CONFIDENCE_ORDER:
            raise ValueError("invalid min_admission_confidence")
        if self.min_counterfactual_evidence_quality not in EVIDENCE_QUALITY_ORDER:
            raise ValueError("invalid min_counterfactual_evidence_quality")
        if not 0.0 <= self.resolved_replay_fraction <= 1.0:
            raise ValueError("resolved_replay_fraction must be in [0, 1]")
        if not 0.0 <= self.min_verified_candidate_coverage <= 1.0:
            raise ValueError("min_verified_candidate_coverage must be in [0, 1]")
        if self.max_hard_states_per_generation > self.max_states_per_generation:
            raise ValueError("hard replay budget exceeds total replay budget")
        if self.max_new_states_per_generation > self.max_states_per_generation:
            raise ValueError("new replay budget exceeds total replay budget")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass
class CriticalStateRecord:
    """Portable decision state; it contains values only, never environment objects."""

    state_id: str
    state_signature: str
    exact_signature: str
    semantic_signature: str
    source_structure_hash: str
    source_frozen_rule_hash: str
    source_parameter_hash: str
    source_generation: int
    source_scenario_id: str
    source_seed: int
    source_run_id: str
    source_decision_index: int
    current_time: float
    workflow_type: str
    workflow_id: int
    ready_task_ids: list[int]
    ready_task_features: dict[str, dict[str, float]]
    ready_task_scores: dict[str, float]
    selected_task_id: int
    selected_task_rank: int
    selected_task_score: float
    criticality_score: float
    critical_reasons: list[str]
    counterfactual_comparisons: list[dict[str, Any]]
    preferred_task_ids: list[int]
    dominated_task_ids: list[int]
    unverified_task_ids: list[int]
    ddl_evidence: dict[str, Any]
    energy_evidence: dict[str, Any]
    uncertainty_evidence: dict[str, Any]
    diagnosis_codes: list[str]
    confidence: str
    risk_category: str
    auxiliary_risk_categories: list[str]
    archive_priority: list[float]
    replay_count: int
    failure_count: int
    success_count: int
    unresolved_count: int
    consecutive_success_count: int
    consecutive_failure_count: int
    created_generation: int
    last_replayed_generation: int | None
    status: str
    content_hash: str
    config_hash: str
    verified_coverage: float = 0.0
    merged_evidence_count: int = 1
    source_seeds: list[int] = field(default_factory=list)
    source_scenarios: list[str] = field(default_factory=list)
    source_structures: list[str] = field(default_factory=list)
    replay_structure_hashes: list[str] = field(default_factory=list)
    success_generations: list[int] = field(default_factory=list)
    failure_generations: list[int] = field(default_factory=list)
    replay_outcomes: list[dict[str, Any]] = field(default_factory=list)
    eviction_reason: str | None = None

    def validate(self) -> None:
        if self.risk_category not in RISK_CATEGORIES:
            raise ValueError(f"unsupported risk_category: {self.risk_category}")
        if self.status not in STATE_STATUSES:
            raise ValueError(f"unsupported critical-state status: {self.status}")
        if self.confidence not in CONFIDENCE_ORDER:
            raise ValueError(f"unsupported confidence: {self.confidence}")
        if (
            not self.ready_task_ids
            or len(self.ready_task_ids) != len(set(self.ready_task_ids))
            or self.selected_task_id not in self.ready_task_ids
        ):
            raise ValueError("critical state must contain its selected ready task")
        if set(self.ready_task_features) != {str(task_id) for task_id in self.ready_task_ids}:
            raise ValueError("ready_task_features must cover the complete ready set")
        if set(self.ready_task_scores) != {str(task_id) for task_id in self.ready_task_ids}:
            raise ValueError("ready_task_scores must cover the complete ready set")
        required_features = set(READY_TASK_FEATURE_NAMES)
        if any(set(values) != required_features for values in self.ready_task_features.values()):
            raise ValueError("each ready task must contain the exact eight-feature schema")
        if not set(self.preferred_task_ids + self.dominated_task_ids):
            raise ValueError("critical state lacks verified preferred/dominated evidence")
        ready_set = set(self.ready_task_ids)
        preferred = set(self.preferred_task_ids)
        dominated = set(self.dominated_task_ids)
        unverified = set(self.unverified_task_ids)
        if preferred & dominated or not (preferred | dominated | unverified).issubset(ready_set):
            raise ValueError("critical-state evidence task sets are inconsistent")
        if not 0.0 <= float(self.verified_coverage) <= 1.0:
            raise ValueError("verified_coverage must be in [0, 1]")
        if not 1 <= int(self.selected_task_rank) <= len(self.ready_task_ids):
            raise ValueError("selected_task_rank is outside the ready set")
        counters = (
            self.replay_count,
            self.failure_count,
            self.success_count,
            self.unresolved_count,
            self.consecutive_success_count,
            self.consecutive_failure_count,
        )
        if any(isinstance(value, bool) or int(value) < 0 for value in counters):
            raise ValueError("critical-state replay counters must be non-negative integers")
        _finite_tree(self.payload(include_hash=False), "critical_state")
        expected = canonical_hash(self.payload(include_hash=False))
        if self.content_hash and self.content_hash != expected:
            raise ValueError("critical state content_hash mismatch")

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_hash:
            value.pop("content_hash", None)
        return value

    def refresh_hash(self) -> str:
        self.content_hash = canonical_hash(self.payload(include_hash=False))
        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        self.refresh_hash()
        return self.payload()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CriticalStateRecord":
        record = cls(**dict(value))
        record.validate()
        return record


@dataclass(frozen=True)
class ReplayResult:
    """Deterministic feature-level replay result for one archived state."""

    state_id: str
    replay_rule_structure_hash: str
    replay_frozen_rule_hash: str
    generation: int
    selected_task_id: int | None
    selected_task_rank: int | None
    selected_task_score: float | None
    preferred_task_ranks: dict[str, int]
    dominated_task_ranks: dict[str, int]
    replay_status: str
    verified_coverage: float
    repeated_historical_error: bool
    ddl_ordering_correct: bool | None
    uncertainty_ordering_correct: bool | None
    energy_ordering_reasonable: bool | None
    deterministic_output: bool
    output_valid: bool
    evidence_used: dict[str, Any]
    replay_latency: float
    state_status_before: str
    result_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CriticalStateReplaySummary:
    """Bounded aggregate feedback for one frozen structure and generation."""

    structure_hash: str
    frozen_rule_hash: str
    generation: int
    replayed_state_count: int
    verified_state_count: int
    unverified_state_count: int
    verified_success_count: int
    verified_failure_count: int
    repeated_historical_error_count: int
    failure_by_risk_category: dict[str, int]
    success_by_risk_category: dict[str, int]
    failure_by_scenario: dict[str, int]
    failure_by_workflow_type: dict[str, int]
    hard_state_failure_patterns: list[dict[str, Any]]
    resolved_state_regressions: list[dict[str, Any]]
    cross_generation_failure_patterns: list[dict[str, Any]]
    high_confidence_structural_actions: list[dict[str, Any]]
    medium_confidence_actions: list[dict[str, Any]]
    representative_failures: list[dict[str, Any]]
    representative_successes: list[dict[str, Any]]
    limitations: list[str]
    archive_hash: str
    replay_config_hash: str
    used_test_seed: bool
    archive_version: str
    summary_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
