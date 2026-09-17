"""Export a parameterized Top-K SeEvo library from persisted LLM results.

The exporter never changes the original run manifest or admission library.
For Single protocol experiments the source scenario (SS/SM/SL), deadline
setting (T/M/L), execution ID and K are explicit command-line inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Mapping

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.llm_safe_hrl.base.heuristic_admission import (  # noqa: E402
    CEWS_EVALUATOR_PROTOCOL_VERSION,
    admission_scope_from_config,
    canonical_json_sha256,
    evaluation_context_from_config,
    file_sha256,
    parse_evaluation_report,
    record_sha256,
)
from algorithms.llm_safe_hrl.base.topk_schema import (  # noqa: E402
    TOPK_MANIFEST_SCHEMA_VERSION,
    TOPK_RECORD_SCHEMA_VERSION,
    TOPK_SELECTION_MODE,
    TOPK_SELECTION_POLICY_VERSION,
)
from algorithms.llm_safe_hrl.LLM.rule_optimization import (  # noqa: E402
    extract_rule_metadata,
    validate_frozen_rule_source,
)
from algorithms.llm_safe_hrl.run_context import (  # noqa: E402
    resolve_deadline_setting,
    validate_execution_identifier,
)


SINGLE_SOURCE_SCENARIOS = ("SS", "SM", "SL")
DEADLINE_CODES = ("T", "M", "L")
_CANDIDATE_PATTERN = re.compile(
    r"^candidate_iter(?P<iteration>\d+)"
    r"_ind(?P<individual>\d+)\.py$"
)
_OPTIMIZATION_METADATA_FIELDS = (
    "structure_hash",
    "parameter_schema_hash",
    "best_parameter_hash",
    "optimizer_config_hash",
    "parameter_diagnostics_hash",
)


def resolve_topk_paths(
    *,
    project_root: str | Path,
    source_scenario: str,
    ddl: str,
    execution_id: str,
    run_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve scenario-specific artifact paths without machine constants."""
    root = Path(project_root).expanduser().resolve()
    source = str(source_scenario).strip().upper()
    deadline = resolve_deadline_setting(ddl).code
    execution = validate_execution_identifier(
        execution_id,
        "execution_id",
    )
    if source not in SINGLE_SOURCE_SCENARIOS:
        raise ValueError(
            "source_scenario must be SS, SM, or SL"
        )
    artifact = (
        Path(run_dir).expanduser().resolve()
        if run_dir is not None
        else (
            root
            / "out"
            / "main_single"
            / source
            / deadline
            / execution
        ).resolve()
    )
    runtime = (
        Path(runtime_dir).expanduser().resolve()
        if runtime_dir is not None
        else (
            root
            / "algorithms"
            / "llm_safe_hrl"
            / "LLM"
            / "outputs"
            / "formal"
            / f"{source}_{deadline}"
            / execution
        ).resolve()
    )
    return artifact, runtime


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite(metrics: Mapping, name: str) -> float:
    value = metrics.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"invalid metric: {name}")
    return float(value)


def topk_ranking_key(metrics: Mapping) -> tuple:
    """Feasibility first; energy is ranked but is not a hard threshold."""
    source_hash = str(
        metrics.get("candidate_sha256", "")
    ).strip().lower()
    if len(source_hash) != 64:
        raise ValueError("invalid candidate_sha256")
    return (
        0 if bool(metrics.get("constraint_feasible", False)) else 1,
        _finite(
            metrics,
            "max_deadline_violation_rate_across_seeds",
        ),
        _finite(metrics, "max_fuzzy_lateness"),
        _finite(metrics, "fuzzy_total_energy_score"),
        _finite(metrics, "objective_cv_across_seeds"),
        source_hash,
    )


def _validate_requested_run(
    run_manifest: Mapping,
    *,
    source_scenario: str,
    ddl: str,
    execution_id: str,
) -> Mapping:
    protocol = run_manifest.get("experiment_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(
            "run_manifest is missing experiment_protocol"
        )
    expected_source = str(source_scenario).strip().upper()
    if str(protocol.get("protocol", "")).lower() != "single":
        raise ValueError(
            "Top-K exporter currently requires protocol=single"
        )
    if (
        str(protocol.get("source_scenario", "")).upper()
        != expected_source
    ):
        raise ValueError(
            "command source_scenario does not match run_manifest"
        )
    expected_deadline = resolve_deadline_setting(ddl).identity()
    if run_manifest.get("deadline_setting") != expected_deadline:
        raise ValueError(
            "command ddl does not match run_manifest"
        )
    if (
        str(run_manifest.get("execution_id", ""))
        != str(execution_id)
    ):
        raise ValueError(
            "command execution_id does not match run_manifest"
        )
    return protocol


def _source_run_state(
    run_manifest: Mapping,
    *,
    allow_failed_final_admission: bool,
    allow_partial_run: bool,
) -> str:
    status = str(
        run_manifest.get("status", "")
    ).strip().upper()
    error = str(run_manifest.get("error", ""))
    if status == "COMPLETED":
        return "COMPLETED"
    failed_final_admission = (
        status == "FAILED"
        and "best frozen rule failed final admission"
        in error.lower()
    )
    if (
        failed_final_admission
        and allow_failed_final_admission
    ):
        return "FAILED_FINAL_ADMISSION"
    if status in {"FAILED", "RUNNING"} and allow_partial_run:
        return (
            "FAILED_PARTIAL_RUN"
            if status == "FAILED"
            else "RUNNING_SNAPSHOT"
        )
    raise ValueError(
        "LLM run is not COMPLETED. Use "
        "--allow-failed-final-admission only when evolution "
        "finished and final admission failed; use "
        "--allow-partial-run only for an explicitly accepted "
        "partial or immutable running snapshot."
    )


def _candidate_from_report(
    *,
    stdout_path: Path,
    run_dir: Path,
    expected_scenario: str,
    expected_seeds: set[int],
    final_test_seeds: set[int],
) -> dict:
    metrics = parse_evaluation_report(stdout_path)
    if not bool(metrics.get("interface_valid", False)):
        raise ValueError("candidate interface is invalid")
    if not bool(
        metrics.get("all_evaluation_seeds_completed", False)
    ):
        raise ValueError("evaluation seeds are incomplete")
    if (
        metrics.get("evaluator_protocol_version")
        != CEWS_EVALUATOR_PROTOCOL_VERSION
    ):
        raise ValueError("evaluator protocol version mismatch")
    if metrics.get("function_name") != "get_task_priority_v2":
        raise ValueError("unsupported priority function")
    if (
        str(metrics.get("scenario_id", "")).strip().upper()
        != expected_scenario
    ):
        raise ValueError("evaluation scenario mismatch")

    seeds = metrics.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seeds
        )
    ):
        raise ValueError("invalid evaluation seed list")
    actual_seeds = {int(seed) for seed in seeds}
    if expected_seeds and actual_seeds != expected_seeds:
        raise ValueError("evaluation seed set mismatch")
    if actual_seeds.intersection(final_test_seeds):
        raise RuntimeError(
            f"final-test seed leakage in {stdout_path}"
        )
    if int(metrics.get("evaluation_seed_count", -1)) != len(seeds):
        raise ValueError("evaluation_seed_count mismatch")
    if int(metrics.get("completed_seed_count", -1)) != len(seeds):
        raise ValueError("completed_seed_count mismatch")

    source_name = Path(
        str(metrics.get("candidate_source_file", ""))
    ).name
    match = _CANDIDATE_PATTERN.fullmatch(source_name)
    if match is None:
        raise ValueError("invalid candidate source name")
    source_path = run_dir / "generated" / source_name
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_hash = file_sha256(source_path)
    if source_hash != str(
        metrics.get("candidate_sha256", "")
    ).strip().lower():
        raise ValueError("candidate source hash mismatch")
    if source_hash != str(
        metrics.get("frozen_rule_hash", "")
    ).strip().lower():
        raise ValueError("frozen rule hash mismatch")

    source_text = source_path.read_text(encoding="utf-8")
    validate_frozen_rule_source(
        source_text,
        require_metadata=True,
    )
    source_metadata = extract_rule_metadata(source_text)
    for field in _OPTIMIZATION_METADATA_FIELDS:
        if source_metadata.get(field) != metrics.get(field):
            raise ValueError(
                f"frozen rule metadata mismatch: {field}"
            )

    ranking_key = topk_ranking_key(metrics)
    json.dumps(metrics, allow_nan=False)
    return {
        "metrics": metrics,
        "source_path": source_path,
        "source_hash": source_hash,
        "iteration": int(match.group("iteration")),
        "individual": int(match.group("individual")),
        "ranking_key": ranking_key,
        "stdout_path": stdout_path,
    }


def load_candidate_pool(
    *,
    run_dir: Path,
    runtime_dir: Path,
    expected_scenario: str,
    expected_seeds: set[int],
    final_test_seeds: set[int],
) -> tuple[list[dict], list[dict]]:
    """Load successful persisted evaluations and de-duplicate frozen rules."""
    if not runtime_dir.is_dir():
        raise FileNotFoundError(
            f"runtime output directory not found: {runtime_dir}"
        )
    reports = sorted(
        runtime_dir.glob("problem_iter*_stdout.txt")
    )
    if not reports:
        raise FileNotFoundError(
            f"no candidate stdout reports found in {runtime_dir}"
        )

    by_frozen_hash: dict[str, dict] = {}
    rejected = []
    for stdout_path in reports:
        try:
            candidate = _candidate_from_report(
                stdout_path=stdout_path,
                run_dir=run_dir,
                expected_scenario=expected_scenario,
                expected_seeds=expected_seeds,
                final_test_seeds=final_test_seeds,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            rejected.append(
                {
                    "report": stdout_path.name,
                    "reason": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            continue

        frozen_hash = str(
            candidate["metrics"]["frozen_rule_hash"]
        ).lower()
        previous = by_frozen_hash.get(frozen_hash)
        if (
            previous is None
            or candidate["ranking_key"]
            < previous["ranking_key"]
        ):
            by_frozen_hash[frozen_hash] = candidate

    candidates = sorted(
        by_frozen_hash.values(),
        key=lambda item: item["ranking_key"],
    )
    if not candidates:
        raise ValueError(
            "no technically valid LLM candidates were found"
        )
    return candidates, rejected


def select_topk(
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """Select ranked candidates, preferring unique structures first."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    selected = []
    deferred = []
    used_structures = set()
    for candidate in candidates:
        structure_hash = str(
            candidate["metrics"].get("structure_hash", "")
        ).strip().lower()
        if len(structure_hash) != 64:
            continue
        if structure_hash in used_structures:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        used_structures.add(structure_hash)
        if len(selected) == top_k:
            return selected
    for candidate in deferred:
        selected.append(candidate)
        if len(selected) == top_k:
            return selected
    raise ValueError(
        f"only {len(selected)} valid candidates are available, "
        f"but top_k={top_k}"
    )


def _build_record(
    candidate: Mapping,
    *,
    rank: int,
    run_dir: Path,
    report_file: Path,
    evaluation_context: Mapping,
    admission_scope: Mapping,
    experiment_protocol: Mapping,
) -> dict:
    metrics = dict(candidate["metrics"])
    source_path = Path(candidate["source_path"])
    source_hash = str(candidate["source_hash"])
    record = {
        "record_schema_version": TOPK_RECORD_SCHEMA_VERSION,
        "heuristic_id": f"seevo_topk_{source_hash[:16]}",
        "display_name": f"SeEvo Top-K #{rank}",
        "version": (
            f"seevo.topk.v1.rank{rank}.{source_hash[:12]}"
        ),
        "source_kind": "seevo_generated",
        "source_file": source_path.relative_to(
            run_dir
        ).as_posix(),
        "source_hash": source_hash,
        "function_name": "get_task_priority_v2",
        "seevo_iteration": int(candidate["iteration"]),
        "seevo_individual": int(candidate["individual"]),
        "evaluation_report_file": report_file.relative_to(
            run_dir
        ).as_posix(),
        "evaluation_report_hash": file_sha256(report_file),
        "evaluation_result_sha256": canonical_json_sha256(
            metrics
        ),
        "evaluation": metrics,
        "evaluation_context": dict(evaluation_context),
        "admission_scope": dict(admission_scope),
        "experiment_protocol": dict(experiment_protocol),
        "optimization_metadata_present": True,
        "structure_hash": metrics.get("structure_hash"),
        "parameter_schema_hash": metrics.get(
            "parameter_schema_hash"
        ),
        "best_parameter_hash": metrics.get(
            "best_parameter_hash"
        ),
        "optimizer_config_hash": metrics.get(
            "optimizer_config_hash"
        ),
        "optimizer_seed": metrics.get("optimizer_seed"),
        "frozen_rule_hash": metrics.get("frozen_rule_hash"),
        "parameter_diagnostics_hash": metrics.get(
            "parameter_diagnostics_hash"
        ),
        "parameter_training_seeds": metrics.get(
            "training_seeds"
        ),
        "parameter_validation_seeds": metrics.get(
            "validation_seeds"
        ),
        "evaluation_seeds": metrics.get("seeds"),
        "fuzzy_energy_mean": metrics.get(
            "fuzzy_total_energy_mean"
        ),
        "fuzzy_energy_std": metrics.get(
            "fuzzy_total_energy_std"
        ),
        "fuzzy_energy_score": metrics.get(
            "fuzzy_total_energy_score"
        ),
        "deadline_violation_rate": metrics.get(
            "max_deadline_violation_rate_across_seeds"
        ),
        "max_fuzzy_lateness": metrics.get(
            "max_fuzzy_lateness"
        ),
        "feasible_seed_rate": metrics.get(
            "feasible_seed_rate"
        ),
        "objective_cv_across_seeds": metrics.get(
            "objective_cv_across_seeds"
        ),
        "admitted": False,
        "admission_status": "not_applied",
        "passed_final_admission": False,
        "selected_for_manager": True,
        "selection_status": "selected",
        "selection_rank": rank,
        "selection_key": list(candidate["ranking_key"]),
        "availability_basis": TOPK_SELECTION_MODE,
        "rejection_reasons": [],
    }
    record["record_sha256"] = record_sha256(record)
    return record


def export_topk_library(
    *,
    source_scenario: str,
    ddl: str,
    execution_id: str,
    top_k: int,
    project_root: str | Path = PROJECT_ROOT,
    run_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    allow_failed_final_admission: bool = False,
    allow_partial_run: bool = False,
) -> Path:
    """Create a Top-K library without mutating original LLM artifacts."""
    source = str(source_scenario).strip().upper()
    deadline = resolve_deadline_setting(ddl).code
    execution = validate_execution_identifier(
        execution_id,
        "execution_id",
    )
    artifact_root, runtime_root = resolve_topk_paths(
        project_root=project_root,
        source_scenario=source,
        ddl=deadline,
        execution_id=execution,
        run_dir=run_dir,
        runtime_dir=runtime_dir,
    )
    run_manifest_path = artifact_root / "run_manifest.json"
    admission_config_path = (
        artifact_root / "effective_admission_config.yaml"
    )
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    if not admission_config_path.is_file():
        raise FileNotFoundError(admission_config_path)

    run_manifest = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    protocol = _validate_requested_run(
        run_manifest,
        source_scenario=source,
        ddl=deadline,
        execution_id=execution,
    )
    source_run_status = _source_run_state(
        run_manifest,
        allow_failed_final_admission=(
            allow_failed_final_admission
        ),
        allow_partial_run=allow_partial_run,
    )

    admission_config = OmegaConf.to_container(
        OmegaConf.load(admission_config_path),
        resolve=True,
    )
    evaluation_context = evaluation_context_from_config(
        admission_config
    )
    admission_scope = admission_scope_from_config(
        admission_config
    )
    expected_seeds = {
        int(seed)
        for seed in protocol.get("llm_train_seeds", [])
    }
    final_test_seeds = {
        int(seed)
        for seed in protocol.get("final_test_seeds", [])
    }
    candidates, rejected = load_candidate_pool(
        run_dir=artifact_root,
        runtime_dir=runtime_root,
        expected_scenario=source,
        expected_seeds=expected_seeds,
        final_test_seeds=final_test_seeds,
    )
    selected = select_topk(candidates, int(top_k))

    report_root_name = f"topk_reports_k{top_k}"
    report_root = artifact_root / report_root_name
    output_path = (
        artifact_root
        / f"topk_heuristic_library_k{top_k}.json"
    )
    audit_path = (
        artifact_root
        / f"topk_selection_audit_k{top_k}.json"
    )
    if output_path.exists() or audit_path.exists():
        raise FileExistsError(
            "refusing to overwrite an existing Top-K export: "
            f"{output_path}"
        )

    records = []
    for rank, candidate in enumerate(selected, start=1):
        report_file = (
            report_root
            / f"{Path(candidate['source_path']).stem}.json"
        )
        _atomic_write_json(
            report_file,
            candidate["metrics"],
        )
        records.append(
            _build_record(
                candidate,
                rank=rank,
                run_dir=artifact_root,
                report_file=report_file,
                evaluation_context=evaluation_context,
                admission_scope=admission_scope,
                experiment_protocol=protocol,
            )
        )

    manifest = {
        "schema_version": TOPK_MANIFEST_SCHEMA_VERSION,
        "manifest_id": (
            f"cews_topk_{source}_{deadline}_{execution}"
        ),
        "manifest_version": (
            "2026-09-10.feasibility-first-topk.v1"
        ),
        "selection_mode": TOPK_SELECTION_MODE,
        "selection_policy": {
            "policy_version": TOPK_SELECTION_POLICY_VERSION,
            "requested_k": int(top_k),
            "selected_count": len(records),
            "unique_structure_first": True,
            "hard_energy_threshold": None,
            "ranking_fields": [
                "constraint_feasible",
                "max_deadline_violation_rate_across_seeds",
                "max_fuzzy_lateness",
                "fuzzy_total_energy_score",
                "objective_cv_across_seeds",
                "candidate_sha256",
            ],
        },
        "source_run": {
            "run_manifest": "run_manifest.json",
            "source_status": source_run_status,
            "original_status": run_manifest.get("status"),
            "original_error": run_manifest.get("error"),
            "source_scenario": source,
            "ddl": deadline,
            "execution_id": execution,
        },
        "trusted_source_root": "generated",
        "trusted_report_root": report_root_name,
        "admission_scope": admission_scope,
        "experiment_protocol": protocol,
        "deadline_setting": run_manifest.get(
            "deadline_setting"
        ),
        "llm_rules": records,
    }
    audit = {
        "source_scenario": source,
        "ddl": deadline,
        "execution_id": execution,
        "source_run_status": source_run_status,
        "top_k": int(top_k),
        "stdout_report_count": len(
            list(
                runtime_root.glob(
                    "problem_iter*_stdout.txt"
                )
            )
        ),
        "valid_unique_candidate_count": len(candidates),
        "selected": [
            {
                "selection_rank": rank,
                "source_file": record["source_file"],
                "source_hash": record["source_hash"],
                "structure_hash": record["structure_hash"],
                "selection_key": record["selection_key"],
            }
            for rank, record in enumerate(records, start=1)
        ],
        "rejected_reports": rejected,
    }
    _atomic_write_json(output_path, manifest)
    _atomic_write_json(audit_path, audit)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a scenario/DDL-specific Top-K SeEvo "
            "heuristic library."
        )
    )
    parser.add_argument(
        "--source-scenario",
        required=True,
        type=str.upper,
        choices=SINGLE_SOURCE_SCENARIOS,
        help="Single-protocol source scenario: SS, SM, or SL.",
    )
    parser.add_argument(
        "--ddl",
        required=True,
        type=str.upper,
        choices=DEADLINE_CODES,
        help="Deadline setting: T, M, or L.",
    )
    parser.add_argument(
        "--execution-id",
        required=True,
        help="Existing LLM execution directory name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of LLM heuristics to export.",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root used to derive default paths.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional artifact directory override. Scenario and "
            "DDL are still verified against run_manifest.json."
        ),
    )
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help=(
            "Optional LLM stdout directory override, useful after "
            "moving a run between machines."
        ),
    )
    parser.add_argument(
        "--allow-failed-final-admission",
        action="store_true",
        help=(
            "Accept a run whose evolution completed but whose final "
            "safe-admission check failed."
        ),
    )
    parser.add_argument(
        "--allow-partial-run",
        action="store_true",
        help=(
            "Explicitly accept an interrupted run or immutable "
            "snapshot; the exported provenance records this."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    output_path = export_topk_library(
        source_scenario=args.source_scenario,
        ddl=args.ddl,
        execution_id=args.execution_id,
        top_k=args.top_k,
        project_root=args.project_root,
        run_dir=args.run_dir,
        runtime_dir=args.runtime_dir,
        allow_failed_final_admission=(
            args.allow_failed_final_admission
        ),
        allow_partial_run=args.allow_partial_run,
    )
    print(f"Top-K heuristic library: {output_path}")


if __name__ == "__main__":
    main()
