"""Offline counterfactual mechanism feedback for frozen SeEvo rules."""

from .cache import CounterfactualCache, CounterfactualCacheKey
from .counterfactual_estimator import (
    ESTIMATOR_VERSION,
    LocalCounterfactualEstimator,
    environment_state_fingerprint,
)
from .diagnostic_agents import (
    DEFAULT_AGENTS,
    DDLDiagnosticAgent,
    DiagnosticAgent,
    EnergyDiagnosticAgent,
    UncertaintyDiagnosticAgent,
)
from .feedback_aggregator import FeedbackAggregator, compact_feedback_for_prompt
from .critical_state_archive import (
    CriticalStateArchive,
    build_critical_state_record,
    state_signatures,
)
from .critical_state_replay import (
    CriticalStateReplayCache,
    CriticalStateReplayer,
    aggregate_replay_results,
    compact_replay_feedback,
    load_frozen_priority_rule,
    strict_replay_gate_triggered,
)
from .critical_state_schemas import (
    ARCHIVE_VERSION,
    FEATURE_SCHEMA_VERSION,
    REPLAY_VERSION,
    RISK_CATEGORIES,
    CriticalStateRecord,
    CriticalStateReplayConfig,
    CriticalStateReplaySummary,
    ReplayResult,
)
from .pipeline import CounterfactualRunSession, load_jsonl
from .schemas import (
    AggregatedFeedback,
    CandidateTaskSnapshot,
    CounterfactualComparison,
    CounterfactualConfig,
    CounterfactualOutcome,
    DecisionTrace,
    DiagnosticReport,
    canonical_hash,
    reject_test_seeds,
)
from .trace_recorder import FEATURE_NAMES, TraceRecorder

__all__ = [
    "AggregatedFeedback",
    "CandidateTaskSnapshot",
    "CriticalStateArchive",
    "CriticalStateRecord",
    "CriticalStateReplayCache",
    "CriticalStateReplayConfig",
    "CriticalStateReplayer",
    "CriticalStateReplaySummary",
    "CounterfactualCache",
    "CounterfactualCacheKey",
    "CounterfactualComparison",
    "CounterfactualConfig",
    "CounterfactualOutcome",
    "CounterfactualRunSession",
    "DDLDiagnosticAgent",
    "DEFAULT_AGENTS",
    "DecisionTrace",
    "DiagnosticAgent",
    "DiagnosticReport",
    "ESTIMATOR_VERSION",
    "EnergyDiagnosticAgent",
    "FEATURE_NAMES",
    "FeedbackAggregator",
    "FEATURE_SCHEMA_VERSION",
    "LocalCounterfactualEstimator",
    "TraceRecorder",
    "UncertaintyDiagnosticAgent",
    "canonical_hash",
    "compact_replay_feedback",
    "compact_feedback_for_prompt",
    "environment_state_fingerprint",
    "load_jsonl",
    "reject_test_seeds",
    "ReplayResult",
    "REPLAY_VERSION",
    "RISK_CATEGORIES",
    "ARCHIVE_VERSION",
    "aggregate_replay_results",
    "build_critical_state_record",
    "load_frozen_priority_rule",
    "state_signatures",
    "strict_replay_gate_triggered",
]
