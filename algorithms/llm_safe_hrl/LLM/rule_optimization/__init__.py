"""Offline structure-parameter co-evolution for SeEvo rules."""

from .cmaes_optimizer import (
    CMAESOptimizer,
    DiagnosticReplayGateConfig,
    OptimizerConfig,
    OptimizationResult,
    constraint_priority_key,
    rank_fitness,
)
from .evaluation_cache import (
    EvaluationCache,
    EvaluationCacheKey,
    aggregate_seed_evaluations,
)
from .parameter_diagnostics import (
    accumulate_cross_generation_diagnostics,
    generate_parameter_diagnostics,
)
from .parameter_schema import (
    PARAMETER_CONTAINER_NAME,
    PARAMETER_SCHEMA_NAME,
    RULE_METADATA_NAME,
    ParameterDefinition,
    ParameterSchema,
    RuleCandidate,
    RuleValidationError,
    canonical_json_sha256,
    extract_rule_metadata,
    freeze_rule_source,
    parse_rule_candidate,
    validate_frozen_rule_source,
)

__all__ = [
    "CMAESOptimizer",
    "DiagnosticReplayGateConfig",
    "EvaluationCache",
    "EvaluationCacheKey",
    "OptimizerConfig",
    "OptimizationResult",
    "PARAMETER_CONTAINER_NAME",
    "PARAMETER_SCHEMA_NAME",
    "RULE_METADATA_NAME",
    "ParameterDefinition",
    "ParameterSchema",
    "RuleCandidate",
    "RuleValidationError",
    "canonical_json_sha256",
    "accumulate_cross_generation_diagnostics",
    "aggregate_seed_evaluations",
    "constraint_priority_key",
    "extract_rule_metadata",
    "freeze_rule_source",
    "generate_parameter_diagnostics",
    "parse_rule_candidate",
    "rank_fitness",
    "validate_frozen_rule_source",
]
