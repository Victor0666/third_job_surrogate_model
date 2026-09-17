# -*- coding: utf-8 -*-
"""Reproducible comparison/ablation experiment manifests.

This module is an experiment-planning layer only.  It validates shared data
and information-access contracts, resolves method/ablation configurations,
hashes all fixed inputs, and writes deterministic manifests.  It does not
train agents, execute untrusted heuristic source files, or alter scheduling
logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from algorithms.llm_safe_hrl.paths import PROJECT_ROOT
from algorithms.llm_safe_hrl.scenario_registry import SCENARIO_REGISTRY
from common.read_xml_opt_Tsize import poisson_arrival_times
from hrl_mix.safe_metrics import SAFE_METRICS_SCHEMA_VERSION


EXPERIMENT_MATRIX_SCHEMA_VERSION = 1
EXPERIMENT_MANIFEST_SCHEMA_VERSION = 1

REQUIRED_METHOD_IDS = (
    'fuzzy_irws',
    'fuzzy_marl',
    'fuzzy_pd3qn',
    'drlea_nichgp',
    "original_hrl",
    "original_hrl_plus_llm",
    "safe_hrl_without_llm",
    "llm_augmented_safe_hrl",
    "llm_safe_hrl_without_curriculum",
    "seevo_best_heuristic_only",
    "edf_baseline",
    "fcfs_fcfs",
    "fcfs_fixed",
)

REQUIRED_ABLATION_IDS = (
    "no_safety_shield",
    "no_safety_value_network",
    "fixed_lambda",
    "no_task_level_safety_deadline",
    "no_fallback_controller",
    "no_llm_demonstration_pretraining",
    "single_seed_training",
)

GET_TASK_PRIORITY_V2_INPUTS = (
    "min_exec_time",
    "min_comm_time",
    "min_incremental_energy",
    "slack",
    "upward_rank",
    "remaining_work",
    "ready_wait_time",
    "uncertainty",
)

REQUIRED_METRIC_FIELDS = frozenset(
    {
        "fuzzy_ddl_violation_rate",
        "feasible_workflow_ratio",
        "feasible_episode_ratio",
        "all_seed_feasible",
        "feasible_seed_rate",
        "mean_fuzzy_lateness",
        "max_fuzzy_lateness",
        "minimum_fuzzy_safety_margin",
        "worst_seed_violation_rate",
        "worst_seed_fuzzy_lateness",
        "shield_intervention_count",
        "shield_intervention_rate",
        "no_safe_action_count",
        "no_safe_action_rate",
        "fallback_count",
        "fallback_rate",
        "proposed_executed_action_mismatch_rate",
        "q_c_prediction_error",
        "lambda_trajectory",
        "fuzzy_energy_mean",
        "fuzzy_energy_std",
        "fuzzy_energy_score",
        "modal_energy",
        "scheduling_time_seconds",
        "convergence_speed",
        "selected_llm_heuristic_frequency",
        "energy_improvement_from_llm",
        "convergence_acceleration",
        "llm_associated_shield_rate",
        "strict_ddl_feasibility_with_llm",
        "strict_ddl_feasibility_without_llm",
    }
)

COMPONENT_KEYS = frozenset(
    {
        "policy_family",
        "manager_policy",
        "manager_observation",
        "task_ordering",
        "heuristic_candidate_set",
        "fixed_heuristic_id",
        "llm_enabled",
        "llm_rule_mode",
        "llm_host_vm_access",
        "host_policy",
        "vm_policy",
        "safe_rl",
        "safety_cost_recording",
        "safety_value_network",
        "safety_shield",
        "task_level_safety_deadline",
        "fallback_controller",
        "empty_safe_action_behavior",
        "lagrange_mode",
        "lambda_value",
        "demonstration_pretraining",
        "llm_demonstration_pretraining",
        "training_seed_mode",
        "information_contract_id",
    }
)

# These adapters are already reachable from the current public training path.
# Other configurations remain valid planned experiments, but the generated
# manifest marks them fail-closed until a dedicated adapter is implemented.
IMPLEMENTED_EXECUTION_ADAPTERS = frozenset(
    {
        'fuzzy_irws_runner',
        'fuzzy_marl_runner',
        'fuzzy_pd3qn_runner',
        'drlea_nichgp_pipeline',
        "fcfs_fcfs_evaluator",
        "fcfs_fixed_evaluator",
        "original_hrl_train_runner",
        "safe_hrl_train_runner",
    }
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _resolve_repo_artifact(
    config_path: Path,
    value: Any,
    name: str,
) -> tuple[str, Path]:
    raw_path = Path(str(value))
    resolved = (
        raw_path
        if raw_path.is_absolute()
        else config_path.parent / raw_path
    ).resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{name} must remain under project root"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    return relative.as_posix(), resolved


def _load_deadline_cache_evidence(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        isinstance(payload, Mapping)
        and isinstance(payload.get("data"), list)
    ):
        records = payload["data"]
    elif (
        isinstance(payload, Mapping)
        and isinstance(payload.get("seed_to_record"), Mapping)
    ):
        records = list(payload["seed_to_record"].values())
    else:
        raise ValueError(
            "deadline cache must contain data or seed_to_record"
        )
    seeds = sorted(int(record["seed"]) for record in records)
    return {
        "meta": dict(payload.get("meta", {})),
        "available_seeds": seeds,
        "seed_count": len(seeds),
    }


def _validate_case_splits(
    case_splits: Mapping[str, Any],
    *,
    available_deadline_seeds: set[int],
) -> dict[str, list[dict[str, Any]]]:
    required_splits = ("training", "validation", "final_test")
    unknown = set(case_splits).difference(required_splits)
    if unknown:
        raise ValueError(
            f"unknown case split keys: {sorted(unknown)}"
        )
    result: dict[str, list[dict[str, Any]]] = {}
    global_case_ids: set[str] = set()
    role_seed_sets: dict[str, set[int]] = {}
    required_case_fields = {
        "case_id",
        "episode_seed",
        "workflow_seed",
        "arrival_seed",
        "resource_seed",
    }
    for split in required_splits:
        rows = case_splits.get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{split} cases must be a non-empty list")
        normalized = []
        split_roles: set[int] = set()
        episode_seeds: set[int] = set()
        for index, raw in enumerate(rows):
            row = dict(_mapping(raw, f"{split}[{index}]"))
            missing = required_case_fields.difference(row)
            if missing:
                raise ValueError(
                    f"{split}[{index}] missing fields: "
                    f"{sorted(missing)}"
                )
            case_id = str(row["case_id"]).strip()
            if not case_id or case_id in global_case_ids:
                raise ValueError(
                    f"duplicate or empty case_id: {case_id!r}"
                )
            global_case_ids.add(case_id)
            values = {
                name: int(row[name])
                for name in required_case_fields
                if name != "case_id"
            }
            if any(value < 0 for value in values.values()):
                raise ValueError("case seeds must be non-negative")
            # The current environment couples DAX choice, workload payload,
            # arrival process, and deadline-cache lookup to random_seed.
            if not (
                values["episode_seed"]
                == values["workflow_seed"]
                == values["arrival_seed"]
            ):
                raise ValueError(
                    "current environment requires episode_seed == "
                    "workflow_seed == arrival_seed"
                )
            if values["episode_seed"] in episode_seeds:
                raise ValueError(
                    f"duplicate {split} episode seed"
                )
            episode_seeds.add(values["episode_seed"])
            if (
                values["episode_seed"]
                not in available_deadline_seeds
            ):
                raise ValueError(
                    "deadline cache does not contain episode seed "
                    f"{values['episode_seed']}"
                )
            split_roles.update(values.values())
            normalized.append({"case_id": case_id, **values})
        role_seed_sets[split] = split_roles
        result[split] = normalized

    for index, left in enumerate(required_splits):
        for right in required_splits[index + 1 :]:
            overlap = role_seed_sets[left].intersection(
                role_seed_sets[right]
            )
            if overlap:
                raise ValueError(
                    f"seed leakage between {left} and {right}: "
                    f"{sorted(overlap)}"
                )
    return result


def _validate_information_contracts(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for contract_id, raw in payload.items():
        contract = dict(
            _mapping(raw, f"information_contracts.{contract_id}")
        )
        for field in (
            "manager_inputs",
            "host_inputs",
            "vm_inputs",
            "llm_rule_inputs",
            "llm_rule_outputs",
        ):
            if not isinstance(contract.get(field), list):
                raise ValueError(
                    f"{contract_id}.{field} must be a list"
                )
        llm_inputs = tuple(contract["llm_rule_inputs"])
        if llm_inputs and llm_inputs != GET_TASK_PRIORITY_V2_INPUTS:
            raise ValueError(
                f"{contract_id} changes get_task_priority_v2 inputs"
            )
        if contract["llm_rule_outputs"] not in (
            [],
            ["ready_task_priority_scores"],
        ):
            raise ValueError(
                f"{contract_id} LLM output may only be task priorities"
            )
        resource_inputs = [
            str(value).lower()
            for value in (
                contract["host_inputs"] + contract["vm_inputs"]
            )
        ]
        if any("llm" in value for value in resource_inputs):
            raise ValueError(
                f"{contract_id} leaks LLM information to Host/VM"
            )
        contracts[str(contract_id)] = contract
    if not contracts:
        raise ValueError("information_contracts must not be empty")
    return contracts


def _validate_components(
    components: Mapping[str, Any],
    *,
    name: str,
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result = dict(components)
    missing = COMPONENT_KEYS.difference(result)
    extra = set(result).difference(COMPONENT_KEYS)
    if missing or extra:
        raise ValueError(
            f"{name} component schema mismatch; missing="
            f"{sorted(missing)}, extra={sorted(extra)}"
        )
    contract_id = str(result["information_contract_id"])
    if contract_id not in contracts:
        raise ValueError(
            f"{name} references unknown information contract"
        )
    for field in (
        "llm_enabled",
        "llm_host_vm_access",
        "safe_rl",
        "safety_cost_recording",
        "safety_value_network",
        "safety_shield",
        "task_level_safety_deadline",
        "fallback_controller",
        "demonstration_pretraining",
        "llm_demonstration_pretraining",
    ):
        if not isinstance(result[field], bool):
            raise ValueError(f"{name}.{field} must be boolean")
    if result["llm_host_vm_access"]:
        raise ValueError(
            f"{name} must not expose LLM outputs to Host/VM"
        )
    contract = contracts[contract_id]
    contract_has_llm = bool(contract["llm_rule_inputs"])
    if bool(result["llm_enabled"]) != contract_has_llm:
        raise ValueError(
            f"{name} LLM flag and information contract disagree"
        )
    if result["llm_enabled"]:
        if result["llm_rule_mode"] == "none":
            raise ValueError(f"{name} enables no LLM rule mode")
    else:
        if result["llm_rule_mode"] != "none":
            raise ValueError(
                f"{name} without LLM must use llm_rule_mode=none"
            )
        if result["llm_demonstration_pretraining"]:
            raise ValueError(
                f"{name} cannot use LLM demonstrations"
            )
    if not result["safe_rl"]:
        forbidden = (
            "safety_cost_recording",
            "safety_value_network",
            "safety_shield",
            "task_level_safety_deadline",
            "fallback_controller",
        )
        enabled = [
            field for field in forbidden if result[field]
        ]
        if enabled or result["lagrange_mode"] != "none":
            raise ValueError(
                f"{name} legacy HRL receives safe-only information: "
                f"{enabled}"
            )
        if (
            result["policy_family"] == "hrl"
            and result["manager_observation"] != "legacy"
        ):
            raise ValueError(
                f"{name} legacy mode must use legacy observation"
            )
    if (
        not result["safety_value_network"]
        and result["lagrange_mode"] not in {"none", "performance_only"}
    ):
        raise ValueError(
            f"{name} cannot use lambda without Q_c"
        )
    if not result["safety_shield"] and result["fallback_controller"]:
        raise ValueError(
            f"{name} fallback cannot be active without shield"
        )
    if result["fallback_controller"]:
        if (
            result["empty_safe_action_behavior"]
            != "deterministic_fixed_vm_fallback"
        ):
            raise ValueError(
                f"{name} changes the fixed VM fallback semantics"
            )
    if result["llm_demonstration_pretraining"] and not result[
        "demonstration_pretraining"
    ]:
        raise ValueError(
            f"{name} LLM demonstration flag requires pretraining"
        )
    if result["training_seed_mode"] not in {
        "multi_seed",
        "single_seed",
        "not_applicable",
    }:
        raise ValueError(f"{name} invalid training_seed_mode")
    if (
        result["policy_family"] == "fixed_heuristic"
        and result["training_seed_mode"] != "not_applicable"
    ):
        raise ValueError(
            f"{name} fixed heuristic baseline must not train"
        )
    lambda_value = result["lambda_value"]
    if lambda_value is not None:
        result["lambda_value"] = _finite(
            lambda_value,
            f"{name}.lambda_value",
            minimum=0.0,
        )
    return result


def _validate_methods(
    payload: Sequence[Any],
    *,
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payload):
        method = dict(_mapping(raw, f"methods[{index}]"))
        method_id = str(method.get("method_id", "")).strip()
        if not method_id or method_id in methods:
            raise ValueError(f"duplicate method_id: {method_id}")
        if "shared_overrides" in method:
            raise ValueError(
                f"{method_id} may not override shared protocol"
            )
        for field in (
            "description",
            "execution_adapter",
            "components",
        ):
            if field not in method:
                raise ValueError(
                    f"{method_id} missing {field}"
                )
        method["components"] = _validate_components(
            _mapping(method["components"], method_id),
            name=method_id,
            contracts=contracts,
        )
        runtime_overrides = dict(
            _mapping(
                method.get("runtime_overrides", {}),
                f"{method_id}.runtime_overrides",
            )
        )
        if set(runtime_overrides).difference({"curriculum_enabled"}):
            raise ValueError(f"{method_id} has unknown runtime override")
        if "curriculum_enabled" in runtime_overrides and not isinstance(
            runtime_overrides["curriculum_enabled"], bool
        ):
            raise ValueError("curriculum_enabled override must be boolean")
        method["runtime_overrides"] = runtime_overrides
        method["output_namespace"] = str(
            method.get("output_namespace", method_id)
        )
        methods[method_id] = method
    missing = set(REQUIRED_METHOD_IDS).difference(methods)
    if missing:
        raise ValueError(
            f"missing required methods: {sorted(missing)}"
        )
    return methods


def _validate_ablations(
    payload: Sequence[Any],
    *,
    methods: Mapping[str, Mapping[str, Any]],
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    ablations: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payload):
        ablation = dict(_mapping(raw, f"ablations[{index}]"))
        ablation_id = str(
            ablation.get("ablation_id", "")
        ).strip()
        if not ablation_id or ablation_id in ablations:
            raise ValueError(
                f"duplicate ablation_id: {ablation_id}"
            )
        base_method_id = str(
            ablation.get("base_method_id", "")
        )
        if base_method_id not in methods:
            raise ValueError(
                f"{ablation_id} references unknown base method"
            )
        if "shared_overrides" in ablation:
            raise ValueError(
                f"{ablation_id} may not override shared protocol"
            )
        overrides = dict(
            _mapping(
                ablation.get("component_overrides"),
                f"{ablation_id}.component_overrides",
            )
        )
        unknown_overrides = set(overrides).difference(
            COMPONENT_KEYS
        )
        if unknown_overrides:
            raise ValueError(
                f"{ablation_id} unknown component overrides: "
                f"{sorted(unknown_overrides)}"
            )
        effective = dict(methods[base_method_id]["components"])
        effective.update(overrides)
        effective = _validate_components(
            effective,
            name=ablation_id,
            contracts=contracts,
        )
        disabled = ablation.get("disabled_components")
        if not isinstance(disabled, list) or not disabled:
            raise ValueError(
                f"{ablation_id} must document disabled_components"
            )
        for disabled_index, raw_change in enumerate(disabled):
            change = dict(
                _mapping(
                    raw_change,
                    f"{ablation_id}.disabled_components"
                    f"[{disabled_index}]",
                )
            )
            component = str(change.get("component", ""))
            if component not in COMPONENT_KEYS:
                raise ValueError(
                    f"{ablation_id} documents unknown component "
                    f"{component}"
                )
            if "from" not in change or "to" not in change:
                raise ValueError(
                    f"{ablation_id} disabled component needs from/to"
                )
            if (
                methods[base_method_id]["components"][component]
                != change["from"]
                or effective[component] != change["to"]
            ):
                raise ValueError(
                    f"{ablation_id} disabled component declaration "
                    f"does not match effective override: {component}"
                )
            if not str(change.get("effect", "")).strip():
                raise ValueError(
                    f"{ablation_id} must explain the effect of "
                    f"disabling {component}"
                )
        ablation["effective_components"] = effective
        ablations[ablation_id] = ablation
    missing = set(REQUIRED_ABLATION_IDS).difference(ablations)
    if missing:
        raise ValueError(
            f"missing required ablations: {sorted(missing)}"
        )
    return ablations


def load_experiment_matrix(
    config_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load and strictly validate an experiment matrix without execution."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw = dict(_mapping(raw, "experiment matrix"))
    if int(raw.get("schema_version", -1)) != (
        EXPERIMENT_MATRIX_SCHEMA_VERSION
    ):
        raise ValueError("unsupported experiment matrix schema")
    experiment_id = str(raw.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("experiment_id must be non-empty")

    shared = dict(
        _mapping(raw.get("shared_protocol"), "shared_protocol")
    )
    scenario = str(shared.get("scenario", "")).upper()
    if scenario != "SS":
        raise ValueError(
            "stage-17 reference matrix currently freezes scenario SS"
        )
    registry_topology = SCENARIO_REGISTRY[scenario].resource_mapping()
    configured_topology = shared.get("resource_topology")
    if configured_topology is not None and dict(
        _mapping(configured_topology, "shared.resource_topology")
    ) != registry_topology:
        raise ValueError(
            "shared.resource_topology conflicts with Scenario Registry"
        )
    if str(shared.get("ddl_setting", "")).lower() != "tight":
        raise ValueError("reference comparison must use Tight DDL")
    fuzzy = dict(_mapping(shared.get("fuzzy"), "shared.fuzzy"))
    if float(fuzzy.get("energy_lambda", math.nan)) != 1.0:
        raise ValueError("fuzzy energy lambda_E is frozen at 1.0")
    if float(fuzzy.get("deadline_eta", math.nan)) != 0.95:
        raise ValueError("fuzzy deadline eta is frozen at 0.95")
    if fuzzy.get("deadline_is_deterministic") is not True:
        raise ValueError("workflow deadline must remain deterministic")
    delta1 = _finite(
        fuzzy.get("delta1"),
        "fuzzy.delta1",
        minimum=0.0,
    )
    delta2 = _finite(
        fuzzy.get("delta2"),
        "fuzzy.delta2",
        minimum=0.0,
    )
    if not delta1 <= 1.0 <= delta2:
        raise ValueError(
            "fuzzy uncertainty must satisfy delta1 <= 1 <= delta2"
        )

    workload = dict(
        _mapping(shared.get("workload"), "shared.workload")
    )
    workload["workflows_per_episode"] = _positive_int(
        workload.get("workflows_per_episode"),
        "workflows_per_episode",
    )
    workload["arrival_lambda"] = _finite(
        workload.get("arrival_lambda"),
        "arrival_lambda",
        minimum=1e-12,
    )
    workload["horizon"] = _finite(
        workload.get("horizon"),
        "horizon",
        minimum=1e-12,
    )

    workflow_files = shared.get("workflow_files")
    if not isinstance(workflow_files, list) or not workflow_files:
        raise ValueError("workflow_files must be a non-empty list")
    artifacts: list[dict[str, Any]] = []
    resolved_workflows = []
    for index, value in enumerate(workflow_files):
        relative, resolved = _resolve_repo_artifact(
            path,
            value,
            f"workflow_files[{index}]",
        )
        resolved_workflows.append(resolved)
        artifacts.append(
            {
                "artifact_role": "workflow_instance",
                "path": relative,
                "sha256": _file_sha256(resolved),
            }
        )
    deadline_relative, deadline_path = _resolve_repo_artifact(
        path,
        shared.get("deadline_cache_path"),
        "deadline_cache_path",
    )
    artifacts.append(
        {
            "artifact_role": "deadline_cache",
            "path": deadline_relative,
            "sha256": _file_sha256(deadline_path),
        }
    )
    heuristic_raw = Path(str(shared.get("heuristic_library_manifest_path")))
    heuristic_path = (
        heuristic_raw
        if heuristic_raw.is_absolute()
        else (path.parent / heuristic_raw)
    ).resolve()
    try:
        heuristic_relative = heuristic_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(
            "heuristic_library_manifest_path must stay inside the project"
        ) from exc
    heuristic_manifest = {}
    if heuristic_path.is_file():
        artifacts.append(
            {
                "artifact_role": "heuristic_library_manifest",
                "path": heuristic_relative,
                "sha256": _file_sha256(heuristic_path),
            }
        )
        with heuristic_path.open("r", encoding="utf-8") as handle:
            heuristic_manifest = json.load(handle)
    formal_library = False
    protocol_metadata = heuristic_manifest.get("experiment_protocol")
    if isinstance(protocol_metadata, Mapping):
        formal_library = (
            str(protocol_metadata.get("protocol", "")).lower() == "single"
            and str(protocol_metadata.get("source_scenario", "")).upper()
            == str(shared.get("scenario", "")).upper()
            and tuple(protocol_metadata.get("training_scenarios", ()))
            == (str(shared.get("scenario", "")).upper(),)
            and tuple(protocol_metadata.get("llm_train_seeds", ())) == (1, 2, 3)
            and tuple(protocol_metadata.get("llm_validation_seeds", ())) == (4, 5)
            and tuple(protocol_metadata.get("final_test_seeds", ()))
            == tuple(range(201, 231))
        )
    admitted_llm = (
        sorted(
            str(record["heuristic_id"])
            for record in heuristic_manifest.get("llm_rules", [])
            if record.get("admitted") is True
            and str(record.get("admission_status", "")).lower()
            == "admitted"
        )
        if formal_library
        else []
    )
    heuristic_library_status = (
        "formal"
        if formal_library
        else ("incompatible" if heuristic_path.is_file() else "missing_formal")
    )
    if heuristic_path.is_file():
        artifacts[-1]["artifact_protocol_status"] = heuristic_library_status

    deadline_evidence = _load_deadline_cache_evidence(
        deadline_path
    )
    case_splits = _validate_case_splits(
        _mapping(shared.get("case_splits"), "case_splits"),
        available_deadline_seeds=set(
            deadline_evidence["available_seeds"]
        ),
    )
    metrics = dict(
        _mapping(
            shared.get("evaluation_metrics"),
            "evaluation_metrics",
        )
    )
    if int(metrics.get("schema_version", -1)) != (
        SAFE_METRICS_SCHEMA_VERSION
    ):
        raise ValueError("safe metrics schema version mismatch")
    metric_fields = metrics.get("fields")
    if not isinstance(metric_fields, list):
        raise ValueError("evaluation metric fields must be a list")
    missing_metrics = REQUIRED_METRIC_FIELDS.difference(
        str(value) for value in metric_fields
    )
    if missing_metrics:
        raise ValueError(
            f"shared metrics missing: {sorted(missing_metrics)}"
        )

    contracts = _validate_information_contracts(
        _mapping(
            raw.get("information_contracts"),
            "information_contracts",
        )
    )
    methods = _validate_methods(
        raw.get("methods", []),
        contracts=contracts,
    )
    ablations = _validate_ablations(
        raw.get("ablations", []),
        methods=methods,
        contracts=contracts,
    )
    smoke = dict(
        _mapping(raw.get("smoke_profile"), "smoke_profile")
    )
    smoke["case_limit_per_split"] = _positive_int(
        smoke.get("case_limit_per_split"),
        "smoke.case_limit_per_split",
    )
    smoke["workflows_per_episode"] = _positive_int(
        smoke.get("workflows_per_episode"),
        "smoke.workflows_per_episode",
    )
    smoke["episodes_per_learning_method"] = _positive_int(
        smoke.get("episodes_per_learning_method"),
        "smoke.episodes_per_learning_method",
    )
    if smoke.get("execute_training") is not False:
        raise ValueError(
            "stage-17 configuration smoke must not execute training"
        )

    normalized_shared = dict(shared)
    normalized_shared["resource_topology"] = registry_topology
    normalized_shared["workload"] = workload
    normalized_shared["fuzzy"] = fuzzy
    normalized_shared["case_splits"] = case_splits
    return {
        "schema_version": EXPERIMENT_MATRIX_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "manifest_version": str(
            raw.get("manifest_version", "")
        ),
        "source_path": str(path),
        "source_sha256": _file_sha256(path),
        "shared_protocol": normalized_shared,
        "shared_protocol_sha256": _json_sha256(
            normalized_shared
        ),
        "information_contracts": contracts,
        "methods": methods,
        "ablations": ablations,
        "smoke_profile": smoke,
        "artifact_integrity": artifacts,
        "heuristic_library_status": heuristic_library_status,
        "expected_heuristic_library_path": heuristic_relative,
        "deadline_cache_evidence": deadline_evidence,
        "admitted_llm_heuristic_ids": admitted_llm,
        "_workflow_paths": resolved_workflows,
    }


def _materialize_case_fixture(
    case: Mapping[str, Any],
    *,
    dax_names: Sequence[str],
    workload: Mapping[str, Any],
    fuzzy: Mapping[str, Any],
    workflows_per_episode: int,
) -> dict[str, Any]:
    count = int(workflows_per_episode)
    arrivals = poisson_arrival_times(
        float(workload["arrival_lambda"]),
        n_max=count,
        seed=int(case["arrival_seed"]),
    )
    workflow_rng = np.random.RandomState(
        int(case["workflow_seed"])
    )
    dax_sequence = [
        str(dax_names[int(workflow_rng.randint(len(dax_names)))])
        for _ in range(count)
    ]
    alpha_small = float(workload["deadline_alpha_small"])
    alpha_large = float(workload["deadline_alpha_large"])
    alpha_probability = float(
        workload["deadline_alpha_small_probability"]
    )
    deadline_alphas = []
    for workflow_id in range(count):
        local_seed = (
            int(case["episode_seed"]) * 1000003
            + workflow_id * 9176
            + 20260316
        ) & 0xFFFFFFFF
        value = float(np.random.RandomState(local_seed).rand())
        deadline_alphas.append(
            alpha_small
            if value < alpha_probability
            else alpha_large
        )
    fixture = {
        **dict(case),
        "workflows_per_episode": count,
        "arrival_times": [float(value) for value in arrivals],
        "dax_sequence": dax_sequence,
        "workload_payload_seeds": [
            int(case["workflow_seed"]) + index
            for index in range(count)
        ],
        "deadline_alpha_sequence": deadline_alphas,
        "fuzzy_resource_seed": int(case["resource_seed"]),
        "fuzzy_delta1": float(fuzzy["delta1"]),
        "fuzzy_delta2": float(fuzzy["delta2"]),
    }
    fixture["fixture_sha256"] = _json_sha256(fixture)
    return fixture


def _execution_readiness(
    run: Mapping[str, Any],
    *,
    admitted_llm_ids: Sequence[str],
    pretraining_manifest_exists: bool,
) -> dict[str, Any]:
    reasons = []
    adapter = str(run["execution_adapter"])
    if adapter not in IMPLEMENTED_EXECUTION_ADAPTERS:
        reasons.append(
            f"execution_adapter_not_implemented:{adapter}"
        )
    components = run["effective_components"]
    if components["llm_enabled"] and not admitted_llm_ids:
        reasons.append("no_admitted_llm_heuristic")
    if (
        components["demonstration_pretraining"]
        and not pretraining_manifest_exists
    ):
        reasons.append(
            "demonstration_dataset_manifest_missing"
        )
    return {
        "execution_ready": not reasons,
        "blocking_reasons": reasons,
        "configuration_valid": True,
    }


def build_experiment_manifest(
    config_path: str | os.PathLike[str],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Build a deterministic manifest; no scheduling or training is run."""
    matrix = load_experiment_matrix(config_path)
    shared = matrix["shared_protocol"]
    cases = shared["case_splits"]
    active_cases = {
        split: (
            rows[: matrix["smoke_profile"]["case_limit_per_split"]]
            if smoke
            else list(rows)
        )
        for split, rows in cases.items()
    }
    active_workflows = (
        matrix["smoke_profile"]["workflows_per_episode"]
        if smoke
        else int(shared["workload"]["workflows_per_episode"])
    )
    dax_names = [
        path.name for path in matrix["_workflow_paths"]
    ]
    fixtures = {
        split: [
            _materialize_case_fixture(
                row,
                dax_names=dax_names,
                workload=shared["workload"],
                fuzzy=shared["fuzzy"],
                workflows_per_episode=active_workflows,
            )
            for row in rows
        ]
        for split, rows in active_cases.items()
    }
    evaluation_case_ids = [
        row["case_id"]
        for split in ("validation", "final_test")
        for row in active_cases[split]
    ]
    pretraining_path_value = shared.get(
        "demonstration_dataset_manifest_path"
    )
    if pretraining_path_value:
        candidate = Path(str(pretraining_path_value))
        if not candidate.is_absolute():
            candidate = (
                Path(matrix["source_path"]).parent / candidate
            )
        pretraining_exists = candidate.resolve().is_file()
    else:
        pretraining_exists = False

    runs = []
    for method_id in REQUIRED_METHOD_IDS:
        method = matrix["methods"][method_id]
        components = dict(method["components"])
        training_ids = (
            []
            if components["training_seed_mode"]
            == "not_applicable"
            else [
                row["case_id"]
                for row in active_cases["training"]
            ]
        )
        run = {
            "run_id": method_id,
            "run_kind": "comparison_method",
            "base_method_id": method_id,
            "description": method["description"],
            "execution_adapter": method[
                "execution_adapter"
            ],
            "runtime_overrides": dict(method["runtime_overrides"]),
            "output_namespace": method["output_namespace"],
            "effective_components": components,
            "disabled_components": [],
            "shared_protocol_sha256": matrix[
                "shared_protocol_sha256"
            ],
            "information_contract_id": components[
                "information_contract_id"
            ],
            "information_contract_sha256": _json_sha256(
                matrix["information_contracts"][
                    components["information_contract_id"]
                ]
            ),
            "training_case_ids": training_ids,
            "evaluation_case_ids": list(
                evaluation_case_ids
            ),
            "uses_information_outside_declared_contract": False,
            "llm_host_vm_access": False,
        }
        run.update(
            _execution_readiness(
                run,
                admitted_llm_ids=matrix[
                    "admitted_llm_heuristic_ids"
                ],
                pretraining_manifest_exists=(
                    pretraining_exists
                ),
            )
        )
        runs.append(run)

    for ablation_id in REQUIRED_ABLATION_IDS:
        ablation = matrix["ablations"][ablation_id]
        components = dict(
            ablation["effective_components"]
        )
        training_rows = list(active_cases["training"])
        if components["training_seed_mode"] == "single_seed":
            training_rows = training_rows[:1]
        run = {
            "run_id": ablation_id,
            "run_kind": "safety_ablation",
            "base_method_id": ablation["base_method_id"],
            "description": ablation["description"],
            "execution_adapter": ablation[
                "execution_adapter"
            ],
            "effective_components": components,
            "disabled_components": list(
                ablation["disabled_components"]
            ),
            "shared_protocol_sha256": matrix[
                "shared_protocol_sha256"
            ],
            "information_contract_id": components[
                "information_contract_id"
            ],
            "information_contract_sha256": _json_sha256(
                matrix["information_contracts"][
                    components["information_contract_id"]
                ]
            ),
            "training_case_ids": [
                row["case_id"] for row in training_rows
            ],
            "evaluation_case_ids": list(
                evaluation_case_ids
            ),
            "uses_information_outside_declared_contract": False,
            "llm_host_vm_access": False,
            "intentional_protocol_deviation": (
                "training_seed_subset_only"
                if ablation_id == "single_seed_training"
                else None
            ),
        }
        run.update(
            _execution_readiness(
                run,
                admitted_llm_ids=matrix[
                    "admitted_llm_heuristic_ids"
                ],
                pretraining_manifest_exists=(
                    pretraining_exists
                ),
            )
        )
        runs.append(run)

    if len({run["shared_protocol_sha256"] for run in runs}) != 1:
        raise RuntimeError("methods diverged from shared protocol")
    if any(
        run["evaluation_case_ids"] != evaluation_case_ids
        for run in runs
    ):
        raise RuntimeError(
            "methods do not share identical evaluation cases"
        )

    manifest = {
        "manifest_schema_version": (
            EXPERIMENT_MANIFEST_SCHEMA_VERSION
        ),
        "generator": {
            "module": "hrl_mix.experiment_matrix",
            "code_sha256": _file_sha256(Path(__file__)),
            "python_version": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy_version": str(np.__version__),
        },
        "experiment_id": matrix["experiment_id"],
        "manifest_version": matrix["manifest_version"],
        "profile_kind": "smoke" if smoke else "full_plan",
        "source_config_path": (
            Path(matrix["source_path"])
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "source_config_sha256": matrix["source_sha256"],
        "shared_protocol_sha256": matrix[
            "shared_protocol_sha256"
        ],
        "shared_protocol": shared,
        "artifact_integrity": matrix["artifact_integrity"],
        "case_fixtures": fixtures,
        "active_workflows_per_episode": active_workflows,
        "admitted_llm_heuristic_ids": matrix[
            "admitted_llm_heuristic_ids"
        ],
        "runs": runs,
        "smoke_validation": {
            "configuration_resolution_passed": True,
            "method_count": len(REQUIRED_METHOD_IDS),
            "ablation_count": len(REQUIRED_ABLATION_IDS),
            "case_limit_per_split": (
                matrix["smoke_profile"][
                    "case_limit_per_split"
                ]
                if smoke
                else None
            ),
            "episodes_per_learning_method": (
                matrix["smoke_profile"][
                    "episodes_per_learning_method"
                ]
                if smoke
                else None
            ),
            "training_executed": False,
            "formal_experiment_executed": False,
            "all_runs_share_evaluation_cases": True,
            "all_runs_share_protocol_hash": True,
            "unready_run_ids": [
                run["run_id"]
                for run in runs
                if not run["execution_ready"]
            ],
        },
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    return manifest


def write_experiment_manifest(
    manifest: Mapping[str, Any],
    output_path: str | os.PathLike[str],
) -> str:
    """Atomically write a validated manifest."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            dict(manifest),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
    os.replace(temporary, path)
    return str(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the safe-HRL comparison/ablation matrix and "
            "write a deterministic experiment manifest. No training "
            "is executed."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Resolve only the configured tiny case subset. This is a "
            "configuration smoke test and still executes no training."
        ),
    )
    args = parser.parse_args(argv)
    manifest = build_experiment_manifest(
        args.config,
        smoke=args.smoke,
    )
    output = write_experiment_manifest(
        manifest,
        args.output,
    )
    print(
        "EXPERIMENT_MANIFEST "
        f"profile={manifest['profile_kind']} "
        f"runs={len(manifest['runs'])} "
        f"ready={sum(run['execution_ready'] for run in manifest['runs'])} "
        f"path={output} "
        f"sha256={manifest['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPERIMENT_MANIFEST_SCHEMA_VERSION",
    "EXPERIMENT_MATRIX_SCHEMA_VERSION",
    "GET_TASK_PRIORITY_V2_INPUTS",
    "REQUIRED_ABLATION_IDS",
    "REQUIRED_METHOD_IDS",
    "build_experiment_manifest",
    "load_experiment_matrix",
    "write_experiment_manifest",
]
