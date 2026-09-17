"""Manager 可选择的传统/SeEvo ready-task 启发式库。

LLM 规则只接收 ``get_task_priority_v2`` 的八个数值数组并返回 task
priority scores。本模块不提供 Host/VM 对象，也不执行任何资源选择。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping
import uuid
import warnings

import numpy as np

try:
    from scenario_registry import validate_protocol_identity
except ModuleNotFoundError:  # Package-style imports used by some test runners.
    from algorithms.llm_safe_hrl.scenario_registry import (
        validate_protocol_identity,
    )

from LLM.rule_optimization import (
    extract_rule_metadata,
    validate_frozen_rule_source,
)

from base.heuristic_admission import (
    ADMISSION_SCOPE_MODE,
    ADMISSION_MANIFEST_SCHEMA_VERSION,
    ADMISSION_RECORD_SCHEMA_VERSION,
    CEWS_EVALUATOR_PROTOCOL_VERSION,
    RESOURCE_DOMAIN_ALLOWED_SCENARIOS,
    admission_policy_from_config,
    canonical_json_sha256,
    evaluate_admission_result,
    file_sha256,
    normalize_admission_scope,
    parse_evaluation_report,
    record_sha256,
)
from base.topk_schema import (
    SAFE_ADMISSION_MODE,
    TOPK_MANIFEST_SCHEMA_VERSION,
    TOPK_RECORD_SCHEMA_VERSION,
    TOPK_SELECTION_MODE,
    TOPK_SELECTION_POLICY_VERSION,
)

LEGACY_RULE_WEIGHT_MODE = "legacy_rule_weight_mode"
HEURISTIC_SELECTION_MODE = "heuristic_selection_mode"
MANAGER_HEURISTIC_MODES = frozenset(
    {LEGACY_RULE_WEIGHT_MODE, HEURISTIC_SELECTION_MODE}
)
HEURISTIC_LIBRARY_SCHEMA_VERSION = (
    ADMISSION_MANIFEST_SCHEMA_VERSION
)
HEURISTIC_MANAGER_OBSERVATION_SCHEMA_VERSION = (
    "safe_manager_heuristic_selection_v4"
)

HEURISTIC_ADMISSION_CONTEXT_FIELDS = (
    "workflows_per_instance",
    "arrival_lambda",
    "horizon",
    "deadline_mode",
    "deadline_alpha_small",
    "deadline_alpha_large",
    "deadline_alpha_small_prob",
    "num_cloud_hosts",
    "num_edge_hosts",
    "cloud_vms_per_host",
    "edge_vms_per_host",
    "cloud_pc_tiers",
    "edge_pc_tiers",
    "cloud_bw_tiers",
    "edge_bw_tiers",
    "fuzzy_delta1",
    "fuzzy_delta2",
    "fuzzy_deadline_eta",
    "fuzzy_energy_lambda",
    "fuzzy_resource_seed_mode",
    "fuzzy_resource_seed_offset",
)
HEURISTIC_RUNTIME_DOMAIN_FIELDS = (
    "scenario_code",
    "task_code",
    "resource_code",
    "workflow_families",
)

TRADITIONAL_HEURISTICS = (
    ("traditional_fcfs", "FCFS", 0),
    ("traditional_sjf", "SJF", 1),
    ("traditional_mcf", "MCF", 2),
    ("traditional_hur", "HUR", 3),
    ("traditional_edf", "EDF", 4),
)

HEURISTIC_RECENT_FEATURE_SCHEMA = (
    {
        "name": "recent_energy_performance",
        "low": -1.0,
        "high": 1.0,
        "normalization": (
            "x / (1 + abs(x)), where x is mean recent "
            "total_performance_reward"
        ),
    },
    {
        "name": "recent_safety_performance",
        "low": 0.0,
        "high": 1.0,
        "normalization": (
            "1 / (1 + mean recent non-negative safety_cost); "
            "0 before the rule has observations"
        ),
    },
    {
        "name": "recent_shield_intervention_rate",
        "low": 0.0,
        "high": 1.0,
        "normalization": "recent intervention count / shield record count",
    },
    {
        "name": "heuristic_available",
        "low": 0.0,
        "high": 1.0,
        "normalization": "binary Manager action availability mask",
    },
)


@dataclass(frozen=True)
class ManagerHeuristic:
    """一个稳定 Manager 动作槽及其执行/准入元数据。"""

    heuristic_id: str
    display_name: str
    source: str
    version: str
    admitted: bool
    availability_reason: str
    selected_for_manager: bool = False
    traditional_feature_index: int | None = None
    priority_rule: Callable | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    metadata: Mapping = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @property
    def available(self) -> bool:
        return bool(
            self.admitted or self.selected_for_manager
        )

    @property
    def is_llm_rule(self) -> bool:
        return self.source == "seevo_llm"

    def public_metadata(self) -> dict:
        return {
            "heuristic_id": self.heuristic_id,
            "display_name": self.display_name,
            "heuristic_source": self.source,
            "llm_rule_version": (
                self.version if self.is_llm_rule else ""
            ),
            "rule_version": self.version,
            "admitted": bool(self.admitted),
            "selected_for_manager": bool(
                self.selected_for_manager
            ),
            "available": bool(self.available),
            "availability_basis": self.metadata.get(
                "availability_basis",
                "safe_admission" if self.admitted else "",
            ),
            "selection_rank": self.metadata.get(
                "selection_rank"
            ),
            "selection_status": self.metadata.get(
                "selection_status",
                "",
            ),
            "availability_reason": self.availability_reason,
            "traditional_feature_index": (
                self.traditional_feature_index
            ),
            "evaluation": dict(
                self.metadata.get("evaluation", {})
            ),
            "evaluation_context": dict(
                self.metadata.get("evaluation_context", {})
            ),
            "admission_scope": dict(
                self.metadata.get("admission_scope", {})
            ),
            "source_file": self.metadata.get("source_file", ""),
            "source_hash": self.metadata.get("source_hash", ""),
            "seevo_iteration": self.metadata.get(
                "seevo_iteration"
            ),
            "admission_status": self.metadata.get(
                "admission_status",
                "",
            ),
            "rejection_reasons": list(
                self.metadata.get("rejection_reasons", [])
            ),
        }


def _traditional_heuristics(
    admission_scope: Mapping | None = None,
) -> list[ManagerHeuristic]:
    metadata = MappingProxyType(
        {
            "admission_scope": (
                dict(admission_scope)
                if admission_scope is not None
                else {}
            )
        }
    )
    return [
        ManagerHeuristic(
            heuristic_id=heuristic_id,
            display_name=display_name,
            source="traditional",
            version="builtin_v1",
            admitted=True,
            availability_reason="builtin_traditional_rule",
            traditional_feature_index=feature_index,
            metadata=metadata,
        )
        for heuristic_id, display_name, feature_index
        in TRADITIONAL_HEURISTICS
    ]


def _safe_artifact_path(
    manifest_path: Path,
    artifact_value,
    trusted_root_name: str,
    artifact_label: str,
) -> Path:
    manifest_root = manifest_path.parent.resolve()
    trusted_root = (
        manifest_root / str(trusted_root_name)
    ).resolve()
    try:
        trusted_root.relative_to(manifest_root)
    except ValueError as exc:
        raise ValueError(
            f"{artifact_label} trusted root must remain under "
            f"{manifest_root}"
        ) from exc
    artifact = (
        manifest_root / str(artifact_value)
    ).resolve()
    try:
        artifact.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(
            f"{artifact_label} must remain under trusted root "
            f"{trusted_root}"
        ) from exc
    return artifact


def _context_values_match(expected, actual) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if not isinstance(
            actual,
            (int, float, np.integer, np.floating),
        ):
            return False
        return bool(
            np.isclose(
                float(expected),
                float(actual),
                rtol=0.0,
                atol=1e-12,
            )
        )
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return False
        return len(expected) == len(actual) and all(
            _context_values_match(left, right)
            for left, right in zip(expected, actual)
        )
    return str(expected).strip().lower() == str(
        actual
    ).strip().lower()


def _admission_context_reasons(
    entry: Mapping,
    runtime_context: Mapping | None,
) -> list[str]:
    evaluation_context = entry.get("evaluation_context")
    if not isinstance(evaluation_context, Mapping):
        return ["missing_evaluation_context"]
    missing = [
        field
        for field in HEURISTIC_ADMISSION_CONTEXT_FIELDS
        if field not in evaluation_context
    ]
    reasons = [
        f"evaluation_context_missing:{field}"
        for field in missing
    ]
    if runtime_context is None:
        return reasons
    runtime_missing = [
        field
        for field in HEURISTIC_ADMISSION_CONTEXT_FIELDS
        if field not in runtime_context
    ]
    if runtime_missing:
        raise ValueError(
            "runtime heuristic admission context is incomplete: "
            + ", ".join(runtime_missing)
        )
    reasons.extend(
        f"evaluation_context_mismatch:{field}"
        for field in HEURISTIC_ADMISSION_CONTEXT_FIELDS
        if field in evaluation_context
        and not _context_values_match(
            evaluation_context[field],
            runtime_context[field],
        )
    )
    return reasons


def _admission_scope_reasons(
    entry: Mapping,
    runtime_context: Mapping | None,
    manifest_admission_scope: Mapping,
) -> list[str]:
    raw_scope = entry.get("admission_scope")
    if not isinstance(raw_scope, Mapping):
        return ["missing_admission_scope"]
    try:
        scope = normalize_admission_scope(raw_scope)
    except Exception as exc:
        return [
            f"invalid_admission_scope:{type(exc).__name__}:{exc}"
        ]
    reasons = []
    if canonical_json_sha256(
        scope
    ) != canonical_json_sha256(manifest_admission_scope):
        reasons.append("manifest_admission_scope_mismatch")
    if runtime_context is None:
        return reasons
    missing = [
        field
        for field in HEURISTIC_RUNTIME_DOMAIN_FIELDS
        if field not in runtime_context
    ]
    if missing:
        raise ValueError(
            "runtime heuristic domain context is incomplete: "
            + ", ".join(missing)
        )
    scenario_code = str(
        runtime_context["scenario_code"]
    ).strip().upper()
    task_code = str(
        runtime_context["task_code"]
    ).strip().upper()
    resource_code = str(
        runtime_context["resource_code"]
    ).strip().upper()
    if (
        len(scenario_code) != 2
        or task_code not in RESOURCE_DOMAIN_ALLOWED_SCENARIOS
        or resource_code not in RESOURCE_DOMAIN_ALLOWED_SCENARIOS
        or scenario_code != task_code + resource_code
    ):
        reasons.append("runtime_domain_context_inconsistent")
    if resource_code != scope["resource_code"]:
        reasons.append("admission_scope_mismatch:resource_code")
    if scenario_code not in scope["allowed_scenarios"]:
        reasons.append("admission_scope_mismatch:scenario_code")
    try:
        runtime_family_scope = normalize_admission_scope(
            {
                "mode": ADMISSION_SCOPE_MODE,
                "resource_code": scope["resource_code"],
                "allowed_scenarios": scope[
                    "allowed_scenarios"
                ],
                "workflow_families": runtime_context[
                    "workflow_families"
                ],
            }
        )
        if (
            runtime_family_scope["workflow_families"]
            != scope["workflow_families"]
        ):
            reasons.append(
                "admission_scope_mismatch:workflow_families"
            )
    except Exception:
        reasons.append(
            "admission_scope_mismatch:workflow_families"
        )
    return list(dict.fromkeys(reasons))


def _admission_reasons(
    entry: Mapping,
    runtime_context: Mapping | None,
    manifest_admission_policy: Mapping,
    manifest_admission_scope: Mapping,
) -> list[str]:
    evaluation = entry.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return ["missing_evaluation"]
    reasons = _admission_scope_reasons(
        entry,
        runtime_context,
        manifest_admission_scope,
    )
    reasons.extend(
        _admission_context_reasons(
            entry,
            runtime_context,
        )
    )
    if (
        entry.get("record_schema_version")
        != ADMISSION_RECORD_SCHEMA_VERSION
    ):
        reasons.append("admission_record_schema_mismatch")
    if entry.get("source_kind") != "seevo_generated":
        reasons.append("untrusted_source_kind")
    if not str(entry.get("source_file", "")).strip():
        reasons.append("missing_source_file")
    source_hash = str(entry.get("source_hash", "")).strip()
    if len(source_hash) != 64:
        reasons.append("invalid_source_hash")
    iteration = entry.get("seevo_iteration")
    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, (int, np.integer))
        or int(iteration) < 0
    ):
        reasons.append("invalid_seevo_iteration")
    if entry.get("function_name") != "get_task_priority_v2":
        reasons.append("invalid_priority_function_name")
    if entry.get("record_sha256") != record_sha256(entry):
        reasons.append("admission_record_hash_mismatch")

    admitted_flag = bool(entry.get("admitted", False))
    expected_status = "admitted" if admitted_flag else "rejected"
    if entry.get("admission_status") != expected_status:
        reasons.append("admission_status_mismatch")
    if (
        "passed_final_admission" in entry
        and bool(entry.get("passed_final_admission")) != admitted_flag
    ):
        reasons.append("final_admission_flag_mismatch")
    persisted_reasons = entry.get("rejection_reasons")
    if not isinstance(persisted_reasons, list):
        reasons.append("invalid_rejection_reasons")
        persisted_reasons = []
    if admitted_flag and persisted_reasons:
        reasons.append("admitted_record_has_rejection_reasons")
    if not admitted_flag:
        reasons.append("manifest_not_admitted")
        reasons.extend(
            f"record_rejection:{reason}"
            for reason in persisted_reasons
        )

    policy = entry.get("admission_policy")
    if not isinstance(policy, Mapping):
        reasons.append("missing_admission_policy")
    else:
        try:
            normalized_policy = admission_policy_from_config(
                {"admission": dict(policy)}
            )
            if canonical_json_sha256(
                normalized_policy
            ) != canonical_json_sha256(
                manifest_admission_policy
            ):
                reasons.append(
                    "manifest_admission_policy_mismatch"
                )
            reasons.extend(
                evaluate_admission_result(
                    evaluation,
                    normalized_policy,
                    expected_workflows_per_seed=int(
                        entry.get(
                            "evaluation_context",
                            {},
                        ).get(
                            "workflows_per_instance",
                            -1,
                        )
                    ),
                )
            )
        except Exception as exc:
            reasons.append(
                f"invalid_admission_policy:{type(exc).__name__}:{exc}"
            )

    result_hash = str(
        entry.get("evaluation_result_sha256", "")
    )
    if result_hash != canonical_json_sha256(evaluation):
        reasons.append("evaluation_result_hash_mismatch")
    optimization_metadata_present = bool(
        entry.get("optimization_metadata_present", False)
    )
    optimization_field_map = {
        "structure_hash": "structure_hash",
        "parameter_schema_hash": "parameter_schema_hash",
        "best_parameter_hash": "best_parameter_hash",
        "optimizer_config_hash": "optimizer_config_hash",
        "optimizer_seed": "optimizer_seed",
        "frozen_rule_hash": "frozen_rule_hash",
        "parameter_diagnostics_hash": "parameter_diagnostics_hash",
        "parameter_training_seeds": "training_seeds",
        "parameter_validation_seeds": "validation_seeds",
    }
    if optimization_metadata_present:
        for record_field, evaluation_field in optimization_field_map.items():
            if not _context_values_match(
                entry.get(record_field),
                evaluation.get(evaluation_field),
            ):
                reasons.append(
                    f"record_optimization_field_mismatch:{record_field}"
                )
        if entry.get("frozen_rule_hash") != entry.get("source_hash"):
            reasons.append("frozen_rule_hash_mismatch")
    elif any(entry.get(field) for field in optimization_field_map):
        reasons.append("partial_optimization_metadata")
    flattened_fields = {
        "evaluation_seeds": evaluation.get("seeds"),
        "fuzzy_energy_mean": evaluation.get(
            "fuzzy_total_energy_mean"
        ),
        "fuzzy_energy_std": evaluation.get(
            "fuzzy_total_energy_std"
        ),
        "fuzzy_energy_score": evaluation.get(
            "fuzzy_total_energy_score"
        ),
        "deadline_violation_rate": evaluation.get(
            "max_deadline_violation_rate_across_seeds"
        ),
        "max_fuzzy_lateness": evaluation.get(
            "max_fuzzy_lateness"
        ),
        "feasible_seed_rate": evaluation.get(
            "feasible_seed_rate"
        ),
        "objective_cv_across_seeds": evaluation.get(
            "objective_cv_across_seeds"
        ),
    }
    reasons.extend(
        f"record_evaluation_field_mismatch:{field}"
        for field, expected in flattened_fields.items()
        if not _context_values_match(
            entry.get(field),
            expected,
        )
    )

    seeds = entry.get("evaluation_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            for seed in seeds
        )
    ):
        reasons.append("missing_evaluation_seeds")
    return list(dict.fromkeys(reasons))


def _topk_selection_key(evaluation: Mapping) -> tuple:
    """Return the deterministic feasibility-first key used by Top-K export."""
    numeric_fields = (
        "max_deadline_violation_rate_across_seeds",
        "max_fuzzy_lateness",
        "fuzzy_total_energy_score",
        "objective_cv_across_seeds",
    )
    values = []
    for field in numeric_fields:
        value = evaluation.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(float(value))
        ):
            raise ValueError(f"invalid_evaluation_metric:{field}")
        values.append(float(value))
    source_hash = str(
        evaluation.get("candidate_sha256", "")
    ).strip().lower()
    if len(source_hash) != 64:
        raise ValueError("invalid_evaluation_candidate_hash")
    return (
        0 if bool(evaluation.get("constraint_feasible", False)) else 1,
        *values,
        source_hash,
    )


def _topk_reasons(
    entry: Mapping,
    runtime_context: Mapping | None,
    manifest_admission_scope: Mapping,
    manifest_protocol: Mapping,
) -> list[str]:
    """Validate Top-K provenance without applying admission thresholds."""
    evaluation = entry.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return ["missing_evaluation"]
    reasons = _admission_scope_reasons(
        entry,
        runtime_context,
        manifest_admission_scope,
    )
    reasons.extend(
        _admission_context_reasons(
            entry,
            runtime_context,
        )
    )
    if (
        entry.get("record_schema_version")
        != TOPK_RECORD_SCHEMA_VERSION
    ):
        reasons.append("topk_record_schema_mismatch")
    if entry.get("source_kind") != "seevo_generated":
        reasons.append("untrusted_source_kind")
    if not str(entry.get("source_file", "")).strip():
        reasons.append("missing_source_file")
    source_hash = str(
        entry.get("source_hash", "")
    ).strip().lower()
    if len(source_hash) != 64:
        reasons.append("invalid_source_hash")
    iteration = entry.get("seevo_iteration")
    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, (int, np.integer))
        or int(iteration) < 0
    ):
        reasons.append("invalid_seevo_iteration")
    if entry.get("function_name") != "get_task_priority_v2":
        reasons.append("invalid_priority_function_name")
    if entry.get("record_sha256") != record_sha256(entry):
        reasons.append("topk_record_hash_mismatch")
    if bool(entry.get("admitted", False)):
        reasons.append("topk_record_must_not_claim_admission")
    if entry.get("admission_status") != "not_applied":
        reasons.append("topk_admission_status_mismatch")
    if bool(entry.get("passed_final_admission", False)):
        reasons.append("topk_final_admission_flag_mismatch")
    if not bool(entry.get("selected_for_manager", False)):
        reasons.append("topk_record_not_selected")
    if entry.get("selection_status") != "selected":
        reasons.append("topk_selection_status_mismatch")
    rank = entry.get("selection_rank")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, (int, np.integer))
        or int(rank) < 1
    ):
        reasons.append("invalid_selection_rank")

    if (
        str(entry.get("evaluation_result_sha256", ""))
        != canonical_json_sha256(evaluation)
    ):
        reasons.append("evaluation_result_hash_mismatch")
    if not bool(evaluation.get("interface_valid", False)):
        reasons.append("candidate_interface_invalid")
    if not bool(
        evaluation.get("all_evaluation_seeds_completed", False)
    ):
        reasons.append("evaluation_seeds_incomplete")
    if (
        evaluation.get("evaluator_protocol_version")
        != CEWS_EVALUATOR_PROTOCOL_VERSION
    ):
        reasons.append("evaluator_protocol_version_mismatch")

    try:
        expected_key = _topk_selection_key(evaluation)
        persisted_key = entry.get("selection_key")
        if (
            not isinstance(persisted_key, list)
            or not _context_values_match(
                list(expected_key),
                persisted_key,
            )
        ):
            reasons.append("topk_selection_key_mismatch")
    except ValueError as exc:
        reasons.append(str(exc))

    seeds = evaluation.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            for seed in seeds
        )
    ):
        reasons.append("missing_evaluation_seeds")
        seeds = []
    else:
        expected_seeds = {
            int(seed)
            for seed in manifest_protocol.get(
                "llm_train_seeds",
                [],
            )
        }
        actual_seeds = {int(seed) for seed in seeds}
        final_test_seeds = {
            int(seed)
            for seed in manifest_protocol.get(
                "final_test_seeds",
                [],
            )
        }
        if expected_seeds and actual_seeds != expected_seeds:
            reasons.append("topk_evaluation_seed_set_mismatch")
        if actual_seeds.intersection(final_test_seeds):
            reasons.append("final_test_seed_leakage")

    scenario_id = str(
        evaluation.get("scenario_id", "")
    ).strip().upper()
    training_scenarios = {
        str(value).strip().upper()
        for value in manifest_protocol.get(
            "training_scenarios",
            [],
        )
    }
    if not scenario_id or scenario_id not in training_scenarios:
        reasons.append("topk_evaluation_scenario_mismatch")

    optimization_field_map = {
        "structure_hash": "structure_hash",
        "parameter_schema_hash": "parameter_schema_hash",
        "best_parameter_hash": "best_parameter_hash",
        "optimizer_config_hash": "optimizer_config_hash",
        "optimizer_seed": "optimizer_seed",
        "frozen_rule_hash": "frozen_rule_hash",
        "parameter_diagnostics_hash": "parameter_diagnostics_hash",
        "parameter_training_seeds": "training_seeds",
        "parameter_validation_seeds": "validation_seeds",
    }
    if bool(entry.get("optimization_metadata_present", False)):
        for record_field, evaluation_field in optimization_field_map.items():
            if not _context_values_match(
                entry.get(record_field),
                evaluation.get(evaluation_field),
            ):
                reasons.append(
                    f"record_optimization_field_mismatch:{record_field}"
                )
        if str(entry.get("frozen_rule_hash", "")).lower() != source_hash:
            reasons.append("frozen_rule_hash_mismatch")
    else:
        reasons.append("missing_optimization_metadata")

    flattened_fields = {
        "evaluation_seeds": evaluation.get("seeds"),
        "fuzzy_energy_mean": evaluation.get(
            "fuzzy_total_energy_mean"
        ),
        "fuzzy_energy_std": evaluation.get(
            "fuzzy_total_energy_std"
        ),
        "fuzzy_energy_score": evaluation.get(
            "fuzzy_total_energy_score"
        ),
        "deadline_violation_rate": evaluation.get(
            "max_deadline_violation_rate_across_seeds"
        ),
        "max_fuzzy_lateness": evaluation.get(
            "max_fuzzy_lateness"
        ),
        "feasible_seed_rate": evaluation.get(
            "feasible_seed_rate"
        ),
        "objective_cv_across_seeds": evaluation.get(
            "objective_cv_across_seeds"
        ),
    }
    reasons.extend(
        f"record_evaluation_field_mismatch:{field}"
        for field, expected in flattened_fields.items()
        if not _context_values_match(
            entry.get(field),
            expected,
        )
    )
    return list(dict.fromkeys(reasons))


def _load_priority_rule(path: Path, function_name: str):
    digest = hashlib.sha256(
        str(path).encode("utf-8")
    ).hexdigest()[:12]
    module_name = (
        f"safe_manager_heuristic_{digest}_{uuid.uuid4().hex}"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot create import specification for {path}"
        )
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(
            f"candidate does not define callable {function_name}"
        )
    signature = inspect.signature(function)
    try:
        signature.bind(
            *[np.zeros(3, dtype=float) for _ in range(8)]
        )
    except TypeError as exc:
        raise ValueError(
            "get_task_priority_v2 must accept the eight established "
            "ready-task feature arrays"
        ) from exc
    _probe_priority_rule(function)
    return function


def _probe_priority_rule(function: Callable) -> None:
    """探测 N=1/N=3、有限输出和输入不变性。"""
    for count in (1, 3):
        inputs = [
            np.linspace(
                float(index) - 1.0,
                float(index) + 1.0,
                count,
                dtype=float,
            )
            for index in range(8)
        ]
        originals = [value.copy() for value in inputs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores = function(*inputs)
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (count,):
            raise ValueError(
                "get_task_priority_v2 probe returned shape "
                f"{scores.shape}; expected {(count,)}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "get_task_priority_v2 probe returned NaN or infinity"
            )
        if any(
            not np.array_equal(current, original)
            for current, original in zip(inputs, originals)
        ):
            raise ValueError(
                "get_task_priority_v2 modified an input array"
            )


def _llm_heuristic(
    entry: Mapping,
    *,
    manifest_path: Path,
    runtime_context: Mapping | None,
    trusted_source_root: str,
    trusted_report_root: str,
    manifest_admission_policy: Mapping,
    manifest_admission_scope: Mapping,
    manifest_protocol: Mapping,
    selection_mode: str,
) -> ManagerHeuristic:
    heuristic_id = str(entry.get("heuristic_id", "")).strip()
    version = str(entry.get("version", "")).strip()
    display_name = str(
        entry.get("display_name", heuristic_id)
    ).strip()
    if not heuristic_id:
        raise ValueError("LLM heuristic_id must not be empty")
    if not version:
        raise ValueError(
            f"LLM heuristic {heuristic_id} has no version"
        )

    if selection_mode == TOPK_SELECTION_MODE:
        reasons = _topk_reasons(
            entry,
            runtime_context,
            manifest_admission_scope,
            manifest_protocol,
        )
    else:
        reasons = _admission_reasons(
            entry,
            runtime_context,
            manifest_admission_policy,
            manifest_admission_scope,
        )
    function = None
    candidate_path = None
    if not reasons:
        try:
            candidate_path = _safe_artifact_path(
                manifest_path,
                entry.get("source_file", ""),
                trusted_source_root,
                "candidate source",
            )
            if not candidate_path.is_file():
                raise FileNotFoundError(candidate_path)
            expected_hash = str(
                entry.get("source_hash", "")
            ).strip().lower()
            actual_hash = file_sha256(candidate_path)
            if not expected_hash or actual_hash != expected_hash:
                raise ValueError("candidate_source_hash_mismatch")
            if entry.get("optimization_metadata_present", False):
                source_text = candidate_path.read_text(encoding="utf-8")
                validate_frozen_rule_source(
                    source_text,
                    require_metadata=True,
                )
                source_metadata = extract_rule_metadata(source_text)
                for field in (
                    "structure_hash",
                    "parameter_schema_hash",
                    "best_parameter_hash",
                    "optimizer_config_hash",
                    "parameter_diagnostics_hash",
                ):
                    if source_metadata.get(field) != entry.get(field):
                        raise ValueError(
                            f"frozen_rule_metadata_mismatch:{field}"
                        )

            report_path = _safe_artifact_path(
                manifest_path,
                entry.get("evaluation_report_file", ""),
                trusted_report_root,
                "evaluation report",
            )
            if not report_path.is_file():
                raise FileNotFoundError(report_path)
            report_hash = file_sha256(report_path)
            if report_hash != str(
                entry.get("evaluation_report_hash", "")
            ).strip().lower():
                raise ValueError("evaluation_report_hash_mismatch")
            report_result = parse_evaluation_report(report_path)
            if canonical_json_sha256(report_result) != str(
                entry.get("evaluation_result_sha256", "")
            ):
                raise ValueError("evaluation_report_result_mismatch")

            function_name = str(
                entry.get(
                    "function_name",
                    "get_task_priority_v2",
                )
            )
            if function_name != "get_task_priority_v2":
                raise ValueError(
                    "unsupported_priority_function_name"
                )
            function = _load_priority_rule(
                candidate_path,
                function_name,
            )
        except Exception as exc:
            reasons.append(
                f"candidate_validation_failed:{type(exc).__name__}:"
                f"{exc}"
            )

    validated = not reasons
    is_topk = selection_mode == TOPK_SELECTION_MODE
    admitted = validated and not is_topk
    selected_for_manager = validated and is_topk
    metadata = dict(entry)
    metadata["availability_basis"] = (
        TOPK_SELECTION_MODE
        if is_topk
        else SAFE_ADMISSION_MODE
    )
    metadata["selected_for_manager"] = bool(
        selected_for_manager
    )
    if candidate_path is not None:
        metadata["resolved_source_file"] = str(
            candidate_path
        )
    return ManagerHeuristic(
        heuristic_id=heuristic_id,
        display_name=display_name,
        source="seevo_llm",
        version=version,
        admitted=admitted,
        selected_for_manager=selected_for_manager,
        availability_reason=(
            "top_k_selected"
            if selected_for_manager
            else (
                "safe_admission_passed"
                if admitted
                else ";".join(reasons)
            )
        ),
        priority_rule=function,
        metadata=MappingProxyType(metadata),
    )


def load_manager_heuristic_library(
    manifest_path: str | Path,
    *,
    runtime_context: Mapping | None = None,
    expected_protocol_identity=None,
    include_traditional: bool = True,
) -> tuple[ManagerHeuristic, ...]:
    """加载五个传统规则及 manifest 中所有稳定 LLM 动作槽。

    ``include_traditional=False`` 时完全不构造 FCFS/SJF/MCF/HUR/EDF 动作槽，
    Manager 的动作空间只剩 manifest 中的 LLM 规则。这里选择直接移除而不是把
    传统槽标记为不可用：动作维度必须真实反映可选集合，留下永远被 mask 掉的
    空槽既浪费网络输出，也让任何绕过 mask 的代码路径重新选到传统规则。

    动作集合的变化会经由 ``heuristic_action_schema_version`` 体现，因而与旧
    checkpoint 的不兼容是显式失败而不是静默错位。
    """
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"safe heuristic manifest not found: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(
            "safe heuristic manifest must be a JSON object"
        )
    selection_mode = str(
        payload.get(
            "selection_mode",
            SAFE_ADMISSION_MODE,
        )
    ).strip().lower()
    if selection_mode not in {
        SAFE_ADMISSION_MODE,
        TOPK_SELECTION_MODE,
    }:
        raise ValueError(
            "unsupported heuristic selection mode: "
            f"{selection_mode}"
        )
    expected_schema = (
        TOPK_MANIFEST_SCHEMA_VERSION
        if selection_mode == TOPK_SELECTION_MODE
        else HEURISTIC_LIBRARY_SCHEMA_VERSION
    )
    if payload.get("schema_version") != expected_schema:
        raise ValueError(
            "safe heuristic manifest schema mismatch: "
            f"mode={selection_mode}, "
            f"expected={expected_schema}, "
            f"actual={payload.get('schema_version')}"
        )
    entries = payload.get("llm_rules")
    if not isinstance(entries, list):
        raise ValueError(
            "safe heuristic manifest llm_rules must be a list"
        )
    if not str(payload.get("manifest_id", "")).strip():
        raise ValueError("safe heuristic manifest_id is required")
    if not str(payload.get("manifest_version", "")).strip():
        raise ValueError(
            "safe heuristic manifest_version is required"
        )
    manifest_protocol = payload.get("experiment_protocol")
    if expected_protocol_identity is not None:
        if not isinstance(manifest_protocol, Mapping):
            raise ValueError(
                "safe heuristic manifest is missing experiment_protocol"
            )
        manifest_protocol = validate_protocol_identity(
            expected_protocol_identity,
            manifest_protocol,
            artifact_name="safe heuristic manifest",
        )
    elif manifest_protocol is not None:
        manifest_protocol = validate_protocol_identity(
            manifest_protocol,
            manifest_protocol,
            artifact_name="safe heuristic manifest",
        )
    if (
        selection_mode == TOPK_SELECTION_MODE
        and not isinstance(manifest_protocol, Mapping)
    ):
        raise ValueError(
            "Top-K heuristic manifest is missing experiment_protocol"
        )
    if manifest_protocol is not None:
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(
                    "safe heuristic manifest rule entry must be a mapping"
                )
            entry_protocol = entry.get("experiment_protocol")
            if not isinstance(entry_protocol, Mapping):
                raise ValueError(
                    "protocol-bound heuristic rule is missing "
                    "experiment_protocol"
                )
            validate_protocol_identity(
                manifest_protocol,
                entry_protocol,
                artifact_name=(
                    "heuristic rule "
                    + str(entry.get("heuristic_id", "<unknown>"))
                ),
            )
    trusted_source_root = str(
        payload.get("trusted_source_root", "")
    ).strip()
    trusted_report_root = str(
        payload.get("trusted_report_root", "")
    ).strip()
    if selection_mode == SAFE_ADMISSION_MODE:
        manifest_admission_policy = admission_policy_from_config(
            {"admission": payload.get("admission_policy")}
        )
    else:
        manifest_admission_policy = {}
        selection_policy = payload.get("selection_policy")
        if not isinstance(selection_policy, Mapping):
            raise ValueError(
                "Top-K heuristic manifest is missing selection_policy"
            )
        if (
            selection_policy.get("policy_version")
            != TOPK_SELECTION_POLICY_VERSION
        ):
            raise ValueError(
                "Top-K selection policy version mismatch"
            )
        expected_ranking_fields = [
            "constraint_feasible",
            "max_deadline_violation_rate_across_seeds",
            "max_fuzzy_lateness",
            "fuzzy_total_energy_score",
            "objective_cv_across_seeds",
            "candidate_sha256",
        ]
        if (
            selection_policy.get("ranking_fields")
            != expected_ranking_fields
        ):
            raise ValueError(
                "Top-K ranking fields do not match the supported policy"
            )
        if (
            selection_policy.get("unique_structure_first")
            is not True
        ):
            raise ValueError(
                "Top-K policy must enable unique_structure_first"
            )
        if selection_policy.get("hard_energy_threshold") is not None:
            raise ValueError(
                "Top-K policy must not declare a hard energy threshold"
            )
        requested_k = int(
            selection_policy.get("requested_k", 0)
        )
        selected_count = int(
            selection_policy.get("selected_count", -1)
        )
        if requested_k < 1:
            raise ValueError(
                "Top-K requested_k must be positive"
            )
        if (
            selected_count != requested_k
            or selected_count != len(entries)
        ):
            raise ValueError(
                "Top-K requested_k, selected_count and llm_rules "
                "length must match"
            )
        ranks = [
            int(entry.get("selection_rank", -1))
            for entry in entries
        ]
        if ranks != list(range(1, len(entries) + 1)):
            raise ValueError(
                "Top-K selection ranks must be consecutive and "
                "start from 1"
            )
        frozen_hashes = [
            str(entry.get("frozen_rule_hash", "")).lower()
            for entry in entries
        ]
        if len(set(frozen_hashes)) != len(frozen_hashes):
            raise ValueError(
                "Top-K frozen rule hashes must be unique"
            )
    manifest_admission_scope = normalize_admission_scope(
        payload.get("admission_scope")
    )
    for label, root_name in (
        ("trusted_source_root", trusted_source_root),
        ("trusted_report_root", trusted_report_root),
    ):
        root_path = Path(root_name)
        if (
            not root_name
            or root_path.is_absolute()
            or ".." in root_path.parts
            or root_name == "."
        ):
            raise ValueError(
                f"{label} must be a relative child directory"
            )
    heuristics = (
        _traditional_heuristics(manifest_admission_scope)
        if include_traditional
        else []
    )
    heuristics.extend(
        _llm_heuristic(
            entry,
            manifest_path=path,
            runtime_context=runtime_context,
            trusted_source_root=trusted_source_root,
            trusted_report_root=trusted_report_root,
            manifest_admission_policy=manifest_admission_policy,
            manifest_admission_scope=manifest_admission_scope,
            manifest_protocol=(
                manifest_protocol
                if isinstance(manifest_protocol, Mapping)
                else {}
            ),
            selection_mode=selection_mode,
        )
        for entry in entries
    )
    if not heuristics:
        # 只可能发生在 include_traditional=False 且 manifest 没有 LLM 规则时。
        # 单独报错，避免退化成下游“动作空间为空”这类难以定位的现象。
        raise ValueError(
            "LLM-only Manager heuristic library is empty: "
            f"{path} declares no LLM rule"
        )
    identifiers = [
        heuristic.heuristic_id for heuristic in heuristics
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "safe heuristic IDs must be unique and stable"
        )
    return tuple(heuristics)


def heuristic_availability_mask(
    heuristics,
) -> np.ndarray:
    return np.asarray(
        [heuristic.available for heuristic in heuristics],
        dtype=np.float32,
    )


def heuristic_action_schema_version(heuristics) -> str:
    """把稳定 ID、版本和准入 mask 固化进 Manager checkpoint schema。"""
    payload = [
        {
            "heuristic_id": heuristic.heuristic_id,
            "source": heuristic.source,
            "version": heuristic.version,
            "available": bool(heuristic.available),
            "admission_scope": dict(
                heuristic.metadata.get(
                    "admission_scope",
                    {},
                )
            ),
        }
        for heuristic in heuristics
    ]
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"{HEURISTIC_MANAGER_OBSERVATION_SCHEMA_VERSION}:"
        f"{digest}"
    )


__all__ = [
    "HEURISTIC_LIBRARY_SCHEMA_VERSION",
    "HEURISTIC_ADMISSION_CONTEXT_FIELDS",
    "HEURISTIC_RUNTIME_DOMAIN_FIELDS",
    "HEURISTIC_MANAGER_OBSERVATION_SCHEMA_VERSION",
    "HEURISTIC_RECENT_FEATURE_SCHEMA",
    "HEURISTIC_SELECTION_MODE",
    "LEGACY_RULE_WEIGHT_MODE",
    "MANAGER_HEURISTIC_MODES",
    "ManagerHeuristic",
    "TRADITIONAL_HEURISTICS",
    "heuristic_action_schema_version",
    "heuristic_availability_mask",
    "load_manager_heuristic_library",
]
