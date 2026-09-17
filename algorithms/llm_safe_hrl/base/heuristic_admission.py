"""SeEvo ready-task 启发式的离线安全准入与版本记录。

本模块只读取候选文件字节、CEWS ``RESULT_JSON`` 报告和 YAML 配置，不导入、
不执行候选 Python。真正的函数加载由 Manager 规则库在全部准入、哈希和上下文
检查通过后执行。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Mapping

try:
    from scenario_registry import (
        assert_no_final_test_seed,
        validate_protocol_identity,
    )
except ModuleNotFoundError:  # Package-style imports used by some test runners.
    from algorithms.llm_safe_hrl.scenario_registry import (
        assert_no_final_test_seed,
        validate_protocol_identity,
    )


ADMISSION_MANIFEST_SCHEMA_VERSION = 3
ADMISSION_RECORD_SCHEMA_VERSION = 2
# v3: CEWS 评价器只从空闲 VM 中选择，并按 Host -> VM 的层次执行同一条固定
# 规则。v2 及更早的报告由 busy-VM 排队产生，调度结果与 deadline 缓存、各比较
# 算法及 Safe-HRL 运行时不可比，必须重新评价。
CEWS_EVALUATOR_PROTOCOL_VERSION = 3
DEFAULT_MANIFEST_ID = "cews_safe_manager_heuristics"
DEFAULT_MANIFEST_VERSION = "2026-07-31.resource-domain.v1"
DEFAULT_TRUSTED_SOURCE_ROOT = "generated"
DEFAULT_TRUSTED_REPORT_ROOT = "admission_reports"
ADMISSION_SCOPE_MODE = "resource_task_domain"
RESOURCE_DOMAIN_ALLOWED_SCENARIOS = {
    "S": ("SS", "MS", "LS"),
    "M": ("SM", "MM", "LM"),
    "L": ("SL", "ML", "LL"),
}
ADMISSION_WORKFLOW_FAMILIES = (
    "CyberShake",
    "Epigenomics",
    "Ligo",
    "Montage",
    "Sipht",
)
_WORKFLOW_FAMILY_BY_CASEFOLD = {
    family.casefold(): family
    for family in ADMISSION_WORKFLOW_FAMILIES
}

_CANDIDATE_NAME_PATTERN = re.compile(
    r"^candidate_iter(?P<iteration>\d+)_ind(?P<individual>\d+)\.py$"
)


def canonical_json_sha256(value) -> str:
    """返回不受字典键顺序影响的 JSON SHA-256。"""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_evaluation_report(path: str | Path) -> dict:
    """读取纯 JSON 或包含 ``RESULT_JSON=`` 的 CEWS stdout 报告。"""
    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
    else:
        matching = [
            line[len("RESULT_JSON=") :]
            for line in text.splitlines()
            if line.startswith("RESULT_JSON=")
        ]
        if not matching:
            raise ValueError(
                f"evaluation report has no RESULT_JSON record: {report_path}"
            )
        payload = json.loads(matching[-1])
    if not isinstance(payload, dict):
        raise ValueError("CEWS evaluation result must be a JSON object")
    return payload


def _require_mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _normalize_scenario_code(value, name: str) -> str:
    scenario = str(value).strip().upper()
    if (
        len(scenario) != 2
        or scenario[0] not in RESOURCE_DOMAIN_ALLOWED_SCENARIOS
        or scenario[1] not in RESOURCE_DOMAIN_ALLOWED_SCENARIOS
    ):
        raise ValueError(
            f"{name} must be one of SS, SM, SL, MS, MM, ML, "
            "LS, LM, LL"
        )
    return scenario


def workflow_families_from_dax_files(dax_files) -> list[str]:
    """Return the stable workflow-family set without binding DAX sizes."""
    if not isinstance(dax_files, (list, tuple)) or not dax_files:
        raise ValueError("dax_files must be a non-empty list")
    found: set[str] = set()
    for dax_file in dax_files:
        stem = Path(str(dax_file)).stem
        raw_family = stem.split("_", 1)[0].strip()
        family = _WORKFLOW_FAMILY_BY_CASEFOLD.get(
            raw_family.casefold()
        )
        if family is None:
            raise ValueError(
                f"unsupported workflow family in DAX file: {dax_file}"
            )
        found.add(family)
    return [
        family
        for family in ADMISSION_WORKFLOW_FAMILIES
        if family in found
    ]


def normalize_admission_scope(value: Mapping) -> dict:
    """Validate and canonicalize the versioned resource-domain scope."""
    scope = _require_mapping(value, "admission_scope")
    required = (
        "mode",
        "resource_code",
        "allowed_scenarios",
        "workflow_families",
    )
    missing = [key for key in required if key not in scope]
    if missing:
        raise ValueError(
            "admission_scope is missing: " + ", ".join(missing)
        )
    mode = str(scope["mode"]).strip().lower()
    if mode != ADMISSION_SCOPE_MODE:
        raise ValueError(
            "admission_scope.mode must be resource_task_domain"
        )
    resource_code = str(scope["resource_code"]).strip().upper()
    if resource_code not in RESOURCE_DOMAIN_ALLOWED_SCENARIOS:
        raise ValueError(
            "admission_scope.resource_code must be S, M, or L"
        )
    raw_scenarios = scope["allowed_scenarios"]
    if not isinstance(raw_scenarios, (list, tuple)):
        raise ValueError(
            "admission_scope.allowed_scenarios must be a list"
        )
    scenarios = [
        _normalize_scenario_code(
            item,
            "admission_scope.allowed_scenarios item",
        )
        for item in raw_scenarios
    ]
    expected_scenarios = list(
        RESOURCE_DOMAIN_ALLOWED_SCENARIOS[resource_code]
    )
    if (
        len(scenarios) != len(set(scenarios))
        or set(scenarios) != set(expected_scenarios)
    ):
        raise ValueError(
            "admission_scope.allowed_scenarios does not match "
            f"resource domain {resource_code}: {expected_scenarios}"
        )
    raw_families = scope["workflow_families"]
    if not isinstance(raw_families, (list, tuple)):
        raise ValueError(
            "admission_scope.workflow_families must be a list"
        )
    families = []
    for raw_family in raw_families:
        family = _WORKFLOW_FAMILY_BY_CASEFOLD.get(
            str(raw_family).strip().casefold()
        )
        if family is None:
            raise ValueError(
                "admission_scope has unsupported workflow family: "
                f"{raw_family}"
            )
        families.append(family)
    if (
        len(families) != len(set(families))
        or set(families) != set(ADMISSION_WORKFLOW_FAMILIES)
    ):
        raise ValueError(
            "admission_scope.workflow_families must contain exactly "
            + ", ".join(ADMISSION_WORKFLOW_FAMILIES)
        )
    return {
        "mode": ADMISSION_SCOPE_MODE,
        "resource_code": resource_code,
        "allowed_scenarios": expected_scenarios,
        "workflow_families": list(ADMISSION_WORKFLOW_FAMILIES),
    }


def admission_scope_from_config(config: Mapping) -> dict:
    """Read a declared scope and bind it to the offline evaluation domain."""
    scope = normalize_admission_scope(
        config.get("admission_scope")
    )
    dataset = _require_mapping(config.get("dataset"), "dataset")
    scenario = _normalize_scenario_code(
        dataset.get("scenario"),
        "dataset.scenario",
    )
    if scenario not in scope["allowed_scenarios"]:
        raise ValueError(
            "dataset.scenario is outside admission_scope"
        )
    if scenario[1] != scope["resource_code"]:
        raise ValueError(
            "dataset.scenario resource code does not match "
            "admission_scope.resource_code"
        )
    evaluated_families = workflow_families_from_dax_files(
        dataset.get("dax_files")
    )
    if evaluated_families != scope["workflow_families"]:
        raise ValueError(
            "dataset DAX workflow families do not match "
            "admission_scope.workflow_families"
        )
    return scope


def evaluation_context_from_config(config: Mapping) -> dict:
    """提取会改变准入语义的工作流、资源、DDL 与模糊参数。"""
    dataset = _require_mapping(config.get("dataset"), "dataset")
    resources = _require_mapping(
        config.get("resources"),
        "resources",
    )
    fuzzy = _require_mapping(config.get("fuzzy"), "fuzzy")
    return {
        "workflow_families": workflow_families_from_dax_files(
            dataset.get("dax_files")
        ),
        "workflows_per_instance": int(
            dataset.get(
                "workflows_per_instance",
                config.get("problem_size"),
            )
        ),
        "arrival_lambda": float(dataset["arrival_lambda"]),
        "horizon": float(dataset["horizon"]),
        "deadline_mode": str(dataset["deadline_mode"]),
        "deadline_alpha_small": float(
            dataset["deadline_alpha_small"]
        ),
        "deadline_alpha_large": float(
            dataset["deadline_alpha_large"]
        ),
        "deadline_alpha_small_prob": float(
            dataset["deadline_alpha_small_prob"]
        ),
        "num_cloud_hosts": int(resources["num_cloud_hosts"]),
        "num_edge_hosts": int(resources["num_edge_hosts"]),
        "cloud_vms_per_host": [
            int(value)
            for value in resources["cloud_vms_per_host"]
        ],
        "edge_vms_per_host": [
            int(value)
            for value in resources["edge_vms_per_host"]
        ],
        "cloud_pc_tiers": [
            float(value) for value in resources["cloud_pc_tiers"]
        ],
        "edge_pc_tiers": [
            float(value) for value in resources["edge_pc_tiers"]
        ],
        "cloud_bw_tiers": [
            float(value) for value in resources["cloud_bw_tiers"]
        ],
        "edge_bw_tiers": [
            float(value) for value in resources["edge_bw_tiers"]
        ],
        "fuzzy_delta1": float(fuzzy["delta1"]),
        "fuzzy_delta2": float(fuzzy["delta2"]),
        "fuzzy_deadline_eta": float(fuzzy["deadline_eta"]),
        "fuzzy_energy_lambda": float(
            fuzzy["energy_uncertainty_weight"]
        ),
        "fuzzy_resource_seed_mode": str(
            fuzzy["resource_seed_mode"]
        ),
        "fuzzy_resource_seed_offset": int(
            fuzzy.get("resource_seed_offset", 0)
        ),
    }


def admission_policy_from_config(config: Mapping) -> dict:
    """读取显式准入阈值；缺字段时拒绝使用隐含安全默认值。"""
    admission = _require_mapping(
        config.get("admission"),
        "admission",
    )
    required = (
        "policy_version",
        "required_evaluation_seeds",
        "minimum_evaluation_seed_count",
        "deadline_violation_rate_max",
        "max_fuzzy_lateness_max",
        "feasible_seed_rate_min",
        "fuzzy_energy_score_max",
        "objective_cv_max",
    )
    missing = [
        key for key in required if key not in admission
    ]
    if missing:
        raise ValueError(
            "admission config is missing: " + ", ".join(missing)
        )
    seeds = admission["required_evaluation_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seeds
        )
    ):
        raise ValueError(
            "required_evaluation_seeds must be a non-empty int list"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "required_evaluation_seeds must not contain duplicates"
        )
    assert_no_final_test_seed(seeds, "heuristic admission")
    policy = {
        "policy_version": str(admission["policy_version"]),
        "required_evaluation_seeds": [
            int(seed) for seed in seeds
        ],
        "minimum_evaluation_seed_count": int(
            admission["minimum_evaluation_seed_count"]
        ),
        "deadline_violation_rate_max": float(
            admission["deadline_violation_rate_max"]
        ),
        "max_fuzzy_lateness_max": float(
            admission["max_fuzzy_lateness_max"]
        ),
        "feasible_seed_rate_min": float(
            admission["feasible_seed_rate_min"]
        ),
        "fuzzy_energy_score_max": float(
            admission["fuzzy_energy_score_max"]
        ),
        "objective_cv_max": float(
            admission["objective_cv_max"]
        ),
    }
    numeric = [
        value
        for key, value in policy.items()
        if key not in {"policy_version", "required_evaluation_seeds"}
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("admission thresholds must be finite")
    if policy["minimum_evaluation_seed_count"] <= 0:
        raise ValueError(
            "minimum_evaluation_seed_count must be positive"
        )
    if (
        len(policy["required_evaluation_seeds"])
        < policy["minimum_evaluation_seed_count"]
    ):
        raise ValueError(
            "required seeds do not meet minimum seed count"
        )
    if policy["deadline_violation_rate_max"] != 0.0:
        raise ValueError(
            "safe heuristic admission requires zero violation rate"
        )
    if policy["max_fuzzy_lateness_max"] != 0.0:
        raise ValueError(
            "safe heuristic admission requires zero maximum lateness"
        )
    if policy["feasible_seed_rate_min"] != 1.0:
        raise ValueError(
            "safe heuristic admission requires feasible_seed_rate 1"
        )
    if policy["fuzzy_energy_score_max"] < 0.0:
        raise ValueError(
            "fuzzy_energy_score_max must be non-negative"
        )
    if policy["objective_cv_max"] < 0.0:
        raise ValueError("objective_cv_max must be non-negative")
    return policy


def _finite_float(value, field_name: str, reasons: list[str]):
    try:
        number = float(value)
    except (TypeError, ValueError):
        reasons.append(f"invalid_metric:{field_name}")
        return None
    if not math.isfinite(number):
        reasons.append(f"non_finite_metric:{field_name}")
        return None
    return number


def evaluate_admission_result(
    result: Mapping,
    policy: Mapping,
    *,
    expected_workflows_per_seed: int,
) -> list[str]:
    """按显式策略评价 CEWS 报告，不改变 CEWS 自身比较原则。"""
    policy = admission_policy_from_config(
        {"admission": dict(policy)}
    )
    reasons: list[str] = []
    if (
        result.get("evaluator_protocol_version")
        != CEWS_EVALUATOR_PROTOCOL_VERSION
    ):
        reasons.append("unsupported_evaluator_protocol")
    if result.get("function_name") != "get_task_priority_v2":
        reasons.append("invalid_priority_function_name")
    if not bool(result.get("interface_valid", False)):
        reasons.append("invalid_priority_interface")
    if not bool(
        result.get("all_evaluation_seeds_completed", False)
    ):
        reasons.append("not_all_evaluation_seeds_completed")

    raw_seeds = result.get("seeds")
    if (
        not isinstance(raw_seeds, list)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in raw_seeds
        )
    ):
        reasons.append("invalid_evaluation_seeds")
        seeds = []
    else:
        seeds = [int(seed) for seed in raw_seeds]
    required_seeds = [
        int(seed)
        for seed in policy["required_evaluation_seeds"]
    ]
    if seeds != required_seeds:
        reasons.append("evaluation_seed_set_mismatch")
    if len(seeds) < int(policy["minimum_evaluation_seed_count"]):
        reasons.append("insufficient_evaluation_seed_count")
    if int(result.get("completed_seed_count", -1)) != len(seeds):
        reasons.append("completed_seed_count_mismatch")

    per_seed = result.get("per_seed_metrics")
    if not isinstance(per_seed, list) or len(per_seed) != len(seeds):
        reasons.append("missing_per_seed_metrics")
        per_seed = []
    for expected_seed, metrics in zip(seeds, per_seed):
        if not isinstance(metrics, Mapping):
            reasons.append(f"invalid_seed_metrics:{expected_seed}")
            continue
        if int(metrics.get("seed", -1)) != expected_seed:
            reasons.append(f"seed_metrics_order_mismatch:{expected_seed}")
        if int(metrics.get("completed_workflows", -1)) != int(
            expected_workflows_per_seed
        ):
            reasons.append(
                f"seed_incomplete_workflows:{expected_seed}"
            )
        seed_violation = _finite_float(
            metrics.get("deadline_violation_rate"),
            f"seed_{expected_seed}_deadline_violation_rate",
            reasons,
        )
        if seed_violation is not None and seed_violation > float(
            policy["deadline_violation_rate_max"]
        ) + 1e-12:
            reasons.append(
                f"seed_nonzero_deadline_violation:{expected_seed}"
            )
        seed_lateness = _finite_float(
            metrics.get("max_fuzzy_lateness"),
            f"seed_{expected_seed}_max_fuzzy_lateness",
            reasons,
        )
        if seed_lateness is not None and seed_lateness > float(
            policy["max_fuzzy_lateness_max"]
        ) + 1e-9:
            reasons.append(
                f"seed_nonzero_fuzzy_lateness:{expected_seed}"
            )
        if not bool(metrics.get("constraint_feasible", False)):
            reasons.append(f"seed_not_feasible:{expected_seed}")

    feasible_rate = _finite_float(
        result.get("feasible_seed_rate"),
        "feasible_seed_rate",
        reasons,
    )
    if feasible_rate is not None and feasible_rate < float(
        policy["feasible_seed_rate_min"]
    ) - 1e-12:
        reasons.append("feasible_seed_rate_below_minimum")
    maximum_violation = _finite_float(
        result.get("max_deadline_violation_rate_across_seeds"),
        "max_deadline_violation_rate_across_seeds",
        reasons,
    )
    if maximum_violation is not None and maximum_violation > float(
        policy["deadline_violation_rate_max"]
    ) + 1e-12:
        reasons.append("nonzero_deadline_violation_rate")
    maximum_lateness = _finite_float(
        result.get("max_fuzzy_lateness"),
        "max_fuzzy_lateness",
        reasons,
    )
    if maximum_lateness is not None and maximum_lateness > float(
        policy["max_fuzzy_lateness_max"]
    ) + 1e-9:
        reasons.append("nonzero_max_fuzzy_lateness")
    energy_score = _finite_float(
        result.get("fuzzy_total_energy_score"),
        "fuzzy_total_energy_score",
        reasons,
    )
    if energy_score is not None and energy_score > float(
        policy["fuzzy_energy_score_max"]
    ) + 1e-9:
        reasons.append("fuzzy_energy_score_above_threshold")
    objective_cv = _finite_float(
        result.get("objective_cv_across_seeds"),
        "objective_cv_across_seeds",
        reasons,
    )
    if objective_cv is not None and objective_cv > float(
        policy["objective_cv_max"]
    ) + 1e-12:
        reasons.append("objective_stability_below_requirement")
    if not bool(result.get("constraint_feasible", False)):
        reasons.append("aggregate_constraint_not_feasible")
    return list(dict.fromkeys(reasons))


def _relative_trusted_path(
    path: Path,
    *,
    manifest_path: Path,
    trusted_root_name: str,
    label: str,
) -> str:
    manifest_root = manifest_path.parent.resolve()
    trusted_root = (
        manifest_root / trusted_root_name
    ).resolve()
    try:
        trusted_root.relative_to(manifest_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} trusted root must remain under {manifest_root}"
        ) from exc
    resolved = path.resolve()
    try:
        resolved.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain under {trusted_root}"
        ) from exc
    return resolved.relative_to(manifest_root).as_posix()


def record_sha256(record: Mapping) -> str:
    """计算不含自校验字段的准入记录指纹。"""
    payload = dict(record)
    payload.pop("record_sha256", None)
    return canonical_json_sha256(payload)


def _normalized_experiment_protocol(
    value,
    *,
    artifact_name: str,
) -> dict | None:
    """Normalize protocol metadata through the canonical registry."""
    if value is None:
        return None
    return validate_protocol_identity(
        value,
        value,
        artifact_name=artifact_name,
    )


def _protocol_from_evaluation_config(config: Mapping) -> dict | None:
    nested = config.get("experiment_protocol")
    if nested is not None:
        return _normalized_experiment_protocol(
            nested,
            artifact_name="evaluation config protocol",
        )
    fields = (
        "protocol",
        "source_scenario",
        "training_scenarios",
        "test_scenarios",
        "llm_train_seeds",
        "llm_validation_seeds",
        "safe_hrl_train_seeds",
        "safe_hrl_validation_seeds",
        "final_test_seeds",
    )
    present = [field for field in fields if field in config]
    if not present:
        return None
    if len(present) != len(fields):
        raise ValueError(
            "evaluation config has incomplete experiment protocol"
        )
    return _normalized_experiment_protocol(
        {field: config[field] for field in fields},
        artifact_name="evaluation config protocol",
    )


def build_admission_record(
    *,
    heuristic_id: str,
    source_path: str | Path,
    evaluation_report_path: str | Path,
    evaluation_config: Mapping,
    manifest_path: str | Path,
    seevo_iteration: int,
    seevo_individual: int,
    display_name: str | None = None,
    trusted_source_root: str = DEFAULT_TRUSTED_SOURCE_ROOT,
    trusted_report_root: str = DEFAULT_TRUSTED_REPORT_ROOT,
    experiment_protocol=None,
) -> dict:
    """从可信目录中的候选字节和评价报告构造不可执行的准入记录。"""
    manifest = Path(manifest_path).resolve()
    source = Path(source_path).resolve()
    report = Path(evaluation_report_path).resolve()
    source_file = _relative_trusted_path(
        source,
        manifest_path=manifest,
        trusted_root_name=trusted_source_root,
        label="candidate source",
    )
    report_file = _relative_trusted_path(
        report,
        manifest_path=manifest,
        trusted_root_name=trusted_report_root,
        label="evaluation report",
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    if not report.is_file():
        raise FileNotFoundError(report)

    identifier = str(heuristic_id).strip()
    if not identifier:
        raise ValueError("heuristic_id must not be empty")
    iteration = int(seevo_iteration)
    individual = int(seevo_individual)
    if iteration < 0 or individual < 0:
        raise ValueError(
            "SeEvo iteration and individual must be non-negative"
        )
    filename_match = _CANDIDATE_NAME_PATTERN.fullmatch(
        source.name
    )
    provenance_reasons = []
    if filename_match is None:
        provenance_reasons.append(
            "unrecognized_seevo_candidate_filename"
        )
    else:
        if int(filename_match.group("iteration")) != iteration:
            provenance_reasons.append(
                "seevo_iteration_filename_mismatch"
            )
        if int(filename_match.group("individual")) != individual:
            provenance_reasons.append(
                "seevo_individual_filename_mismatch"
            )

    result = parse_evaluation_report(report)
    policy = admission_policy_from_config(evaluation_config)
    admission_scope = admission_scope_from_config(
        evaluation_config
    )
    context = evaluation_context_from_config(evaluation_config)
    source_hash = file_sha256(source)
    report_hash = file_sha256(report)
    config_hash = canonical_json_sha256(evaluation_config)
    if result.get("candidate_sha256") != source_hash:
        provenance_reasons.append(
            "evaluation_candidate_hash_mismatch"
        )
    if result.get("evaluation_config_sha256") != config_hash:
        provenance_reasons.append(
            "evaluation_config_hash_mismatch"
        )

    optimization_hash_fields = (
        "structure_hash",
        "parameter_schema_hash",
        "best_parameter_hash",
        "optimizer_config_hash",
        "frozen_rule_hash",
        "parameter_diagnostics_hash",
    )
    optimization_metadata_present = any(
        result.get(field) for field in optimization_hash_fields
    )
    if optimization_metadata_present:
        for field in optimization_hash_fields:
            value = str(result.get(field, "")).strip().lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                provenance_reasons.append(
                    f"invalid_optimization_hash:{field}"
                )
        if result.get("frozen_rule_hash") != source_hash:
            provenance_reasons.append("frozen_rule_hash_mismatch")
        best_parameters = result.get("best_parameters")
        if not isinstance(best_parameters, Mapping):
            provenance_reasons.append("missing_best_parameters")
        elif canonical_json_sha256(best_parameters) != result.get(
            "best_parameter_hash"
        ):
            provenance_reasons.append("best_parameter_hash_mismatch")
        training_seeds = result.get("training_seeds")
        validation_seeds = result.get("validation_seeds")
        if not isinstance(training_seeds, list) or not training_seeds:
            provenance_reasons.append("missing_parameter_training_seeds")
        if not isinstance(validation_seeds, list):
            provenance_reasons.append("invalid_parameter_validation_seeds")
        if isinstance(training_seeds, list) and isinstance(validation_seeds, list):
            if set(training_seeds).intersection(validation_seeds):
                provenance_reasons.append(
                    "parameter_training_validation_seed_overlap"
                )
            final_test_seeds = set(
                evaluation_config.get("dataset", {}).get("test_seeds", [])
            )
            if final_test_seeds.intersection(training_seeds) or final_test_seeds.intersection(
                validation_seeds
            ):
                provenance_reasons.append(
                    "parameter_optimization_uses_final_test_seed"
                )

    counterfactual_hash_fields = (
        "counterfactual_feedback_hash",
        "counterfactual_config_hash",
    )
    counterfactual_metadata_present = any(
        result.get(field) for field in counterfactual_hash_fields
    )
    if counterfactual_metadata_present:
        for field in counterfactual_hash_fields:
            value = str(result.get(field, "")).strip().lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef"
                for character in value
            ):
                provenance_reasons.append(
                    f"invalid_counterfactual_hash:{field}"
                )
        analyzed_seeds = result.get(
            "counterfactual_analyzed_seeds"
        )
        analyzed_scenarios = result.get(
            "counterfactual_analyzed_scenarios"
        )
        if not isinstance(analyzed_seeds, list) or not analyzed_seeds:
            provenance_reasons.append(
                "missing_counterfactual_analyzed_seeds"
            )
        if not isinstance(analyzed_scenarios, list) or not analyzed_scenarios:
            provenance_reasons.append(
                "missing_counterfactual_analyzed_scenarios"
            )
        final_test_seeds = set(
            evaluation_config.get("dataset", {}).get("test_seeds", [])
        )
        if isinstance(analyzed_seeds, list) and final_test_seeds.intersection(
            analyzed_seeds
        ):
            provenance_reasons.append(
                "counterfactual_feedback_uses_final_test_seed"
            )
        if bool(result.get("counterfactual_used_test_seed", False)):
            provenance_reasons.append(
                "counterfactual_manifest_marks_test_seed_usage"
            )

    critical_state_hash_fields = (
        "critical_state_archive_hash",
        "critical_state_replay_hash",
        "critical_state_config_hash",
    )
    critical_state_metadata_present = any(
        result.get(field) for field in critical_state_hash_fields
    )
    if critical_state_metadata_present:
        for field in critical_state_hash_fields:
            value = str(result.get(field, "")).strip().lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                provenance_reasons.append(f"invalid_critical_state_hash:{field}")
        if result.get("critical_state_archive_version") != "critical_state_archive_v1":
            provenance_reasons.append("invalid_critical_state_archive_version")
        for field in (
            "critical_state_replayed_state_count",
            "critical_state_verified_failure_count",
        ):
            value = result.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                provenance_reasons.append(f"invalid_critical_state_count:{field}")
        if bool(result.get("critical_state_used_test_seed", False)):
            provenance_reasons.append("critical_state_replay_uses_final_test_seed")

    evaluation_reasons = evaluate_admission_result(
        result,
        policy,
        expected_workflows_per_seed=int(
            context["workflows_per_instance"]
        ),
    )
    rejection_reasons = list(
        dict.fromkeys(
            [*provenance_reasons, *evaluation_reasons]
        )
    )
    admitted = not rejection_reasons
    result_hash = canonical_json_sha256(result)
    version = (
        f"seevo.iter{iteration}.ind{individual}."
        f"src{source_hash[:12]}.eval{result_hash[:12]}."
        f"policy-{policy['policy_version']}"
    )
    record = {
        "record_schema_version": ADMISSION_RECORD_SCHEMA_VERSION,
        "heuristic_id": identifier,
        "display_name": (
            str(display_name)
            if display_name is not None
            else identifier
        ),
        "source_kind": "seevo_generated",
        "source_file": source_file,
        "source_hash": source_hash,
        "seevo_iteration": iteration,
        "seevo_individual": individual,
        "function_name": "get_task_priority_v2",
        "version": version,
        "evaluation_report_file": report_file,
        "evaluation_report_hash": report_hash,
        "evaluation_result_sha256": result_hash,
        "evaluation_config_sha256": config_hash,
        "structure_hash": result.get("structure_hash", ""),
        "parameter_schema_hash": result.get(
            "parameter_schema_hash", ""
        ),
        "best_parameter_hash": result.get(
            "best_parameter_hash", ""
        ),
        "optimizer_config_hash": result.get(
            "optimizer_config_hash", ""
        ),
        "optimizer_seed": result.get("optimizer_seed"),
        "frozen_rule_hash": result.get("frozen_rule_hash", ""),
        "parameter_diagnostics_hash": result.get(
            "parameter_diagnostics_hash", ""
        ),
        "parameter_training_seeds": result.get(
            "training_seeds", []
        ),
        "parameter_validation_seeds": result.get(
            "validation_seeds", []
        ),
        "counterfactual_feedback_hash": result.get(
            "counterfactual_feedback_hash", ""
        ),
        "counterfactual_config_hash": result.get(
            "counterfactual_config_hash", ""
        ),
        "counterfactual_analyzed_seeds": result.get(
            "counterfactual_analyzed_seeds", []
        ),
        "counterfactual_analyzed_scenarios": result.get(
            "counterfactual_analyzed_scenarios", []
        ),
        "counterfactual_used_test_seed": bool(
            result.get("counterfactual_used_test_seed", False)
        ),
        "counterfactual_metadata_present": counterfactual_metadata_present,
        "critical_state_archive_hash": result.get(
            "critical_state_archive_hash", ""
        ),
        "critical_state_replay_hash": result.get(
            "critical_state_replay_hash", ""
        ),
        "critical_state_config_hash": result.get(
            "critical_state_config_hash", ""
        ),
        "critical_state_replayed_state_count": result.get(
            "critical_state_replayed_state_count", 0
        ),
        "critical_state_verified_failure_count": result.get(
            "critical_state_verified_failure_count", 0
        ),
        "critical_state_used_test_seed": bool(
            result.get("critical_state_used_test_seed", False)
        ),
        "critical_state_archive_version": result.get(
            "critical_state_archive_version", ""
        ),
        "critical_state_metadata_present": critical_state_metadata_present,
        "optimization_metadata_present": optimization_metadata_present,
        "evaluation_context": context,
        "admission_scope": admission_scope,
        "admission_policy": policy,
        "evaluation_seeds": [
            int(seed) for seed in result.get("seeds", [])
        ],
        "fuzzy_energy_mean": result.get(
            "fuzzy_total_energy_mean"
        ),
        "fuzzy_energy_std": result.get(
            "fuzzy_total_energy_std"
        ),
        "fuzzy_energy_score": result.get(
            "fuzzy_total_energy_score"
        ),
        "deadline_violation_rate": result.get(
            "max_deadline_violation_rate_across_seeds"
        ),
        "max_fuzzy_lateness": result.get(
            "max_fuzzy_lateness"
        ),
        "feasible_seed_rate": result.get(
            "feasible_seed_rate"
        ),
        "objective_cv_across_seeds": result.get(
            "objective_cv_across_seeds"
        ),
        "all_evaluation_seeds_completed": bool(
            result.get(
                "all_evaluation_seeds_completed",
                False,
            )
        ),
        "evaluation": result,
        "admitted": admitted,
        "admission_status": (
            "admitted" if admitted else "rejected"
        ),
        "passed_final_admission": admitted,
        "rejection_reasons": rejection_reasons,
        "rejection_reason": (
            ""
            if admitted
            else ";".join(rejection_reasons)
        ),
    }
    protocol_identity = (
        _normalized_experiment_protocol(
            experiment_protocol,
            artifact_name="admission record",
        )
        if experiment_protocol is not None
        else _protocol_from_evaluation_config(evaluation_config)
    )
    if protocol_identity is not None:
        record["experiment_protocol"] = protocol_identity
    record["record_sha256"] = record_sha256(record)
    logging.info(
        "Heuristic admission id=%s structure=%s frozen_hash=%s admitted=%s reasons=%s",
        identifier,
        record.get("structure_hash", ""),
        record.get("frozen_rule_hash", source_hash),
        admitted,
        rejection_reasons,
    )
    return record


def append_admission_record(
    manifest_path: str | Path,
    record: Mapping,
    *,
    manifest_version: str = DEFAULT_MANIFEST_VERSION,
    trusted_source_root: str = DEFAULT_TRUSTED_SOURCE_ROOT,
    trusted_report_root: str = DEFAULT_TRUSTED_REPORT_ROOT,
    expected_protocol_identity=None,
) -> dict:
    """追加新版本记录；拒绝覆盖相同 ID 或相同版本。"""
    path = Path(manifest_path)
    if (
        record.get("record_schema_version")
        != ADMISSION_RECORD_SCHEMA_VERSION
    ):
        raise ValueError("admission record schema mismatch")
    record_scope = normalize_admission_scope(
        record.get("admission_scope")
    )
    record_policy = admission_policy_from_config(
        {"admission": record.get("admission_policy")}
    )
    record_protocol = _normalized_experiment_protocol(
        record.get("experiment_protocol"),
        artifact_name="admission record",
    )
    expected_protocol = _normalized_experiment_protocol(
        expected_protocol_identity,
        artifact_name="expected admission protocol",
    )
    if expected_protocol is not None:
        if record_protocol is None:
            raise ValueError(
                "admission record is missing experiment_protocol"
            )
        validate_protocol_identity(
            expected_protocol,
            record_protocol,
            artifact_name="admission record",
        )
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version")
            != ADMISSION_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                "safe heuristic manifest schema mismatch"
            )
        manifest_policy = admission_policy_from_config(
            {"admission": payload.get("admission_policy")}
        )
        manifest_scope = normalize_admission_scope(
            payload.get("admission_scope")
        )
        manifest_protocol = _normalized_experiment_protocol(
            payload.get("experiment_protocol"),
            artifact_name="safe heuristic manifest",
        )
        if (manifest_protocol is None) != (record_protocol is None):
            raise ValueError(
                "record experiment_protocol does not match manifest"
            )
        if manifest_protocol is not None:
            validate_protocol_identity(
                manifest_protocol,
                record_protocol,
                artifact_name="admission record",
            )
        if expected_protocol is not None:
            if manifest_protocol is None:
                raise ValueError(
                    "safe heuristic manifest is missing experiment_protocol"
                )
            validate_protocol_identity(
                expected_protocol,
                manifest_protocol,
                artifact_name="safe heuristic manifest",
            )
        if canonical_json_sha256(
            manifest_policy
        ) != canonical_json_sha256(record_policy):
            raise ValueError(
                "record admission policy does not match manifest"
            )
        if canonical_json_sha256(
            manifest_scope
        ) != canonical_json_sha256(record_scope):
            raise ValueError(
                "record admission scope does not match manifest"
            )
    else:
        payload = {
            "schema_version": ADMISSION_MANIFEST_SCHEMA_VERSION,
            "manifest_id": (
                f"{DEFAULT_MANIFEST_ID}_res"
                f"{record_scope['resource_code']}"
            ),
            "manifest_version": str(manifest_version),
            "manifest_revision": 0,
            "trusted_source_root": trusted_source_root,
            "trusted_report_root": trusted_report_root,
            "admission_scope": record_scope,
            "admission_policy": record_policy,
            "llm_rules": [],
        }
        if record_protocol is not None:
            payload["experiment_protocol"] = record_protocol
    rules = payload.get("llm_rules")
    if not isinstance(rules, list):
        raise ValueError("manifest llm_rules must be a list")
    identifier = record.get("heuristic_id")
    version = record.get("version")
    if any(item.get("heuristic_id") == identifier for item in rules):
        raise ValueError(
            f"duplicate heuristic_id is not allowed: {identifier}"
        )
    if any(item.get("version") == version for item in rules):
        raise ValueError(
            f"duplicate heuristic version is not allowed: {version}"
        )
    if record.get("record_sha256") != record_sha256(record):
        raise ValueError("admission record SHA-256 mismatch")
    rules.append(dict(record))
    payload["manifest_revision"] = int(
        payload.get("manifest_revision", 0)
    ) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


__all__ = [
    "ADMISSION_MANIFEST_SCHEMA_VERSION",
    "ADMISSION_RECORD_SCHEMA_VERSION",
    "ADMISSION_SCOPE_MODE",
    "ADMISSION_WORKFLOW_FAMILIES",
    "CEWS_EVALUATOR_PROTOCOL_VERSION",
    "DEFAULT_MANIFEST_ID",
    "DEFAULT_MANIFEST_VERSION",
    "DEFAULT_TRUSTED_REPORT_ROOT",
    "DEFAULT_TRUSTED_SOURCE_ROOT",
    "RESOURCE_DOMAIN_ALLOWED_SCENARIOS",
    "admission_scope_from_config",
    "admission_policy_from_config",
    "append_admission_record",
    "build_admission_record",
    "canonical_json_sha256",
    "evaluate_admission_result",
    "evaluation_context_from_config",
    "file_sha256",
    "normalize_admission_scope",
    "parse_evaluation_report",
    "record_sha256",
    "workflow_families_from_dax_files",
]
