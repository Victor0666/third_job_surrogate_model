"""Shared runner for the two deterministic FCFS baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_paths import PROJECT_ROOT
from algorithms.comparisons.fuzzy_common.evaluation import evaluate_policy
from algorithms.comparisons.fuzzy_common.protocol import (
    load_protocol_config,
    protocol_from_config,
)
from algorithms.comparisons.fcfs.policies import (
    METHOD_DISPLAY_NAMES,
    POLICY_TYPES,
    make_policy,
)
from algorithms.llm_safe_hrl.hrl_mix.train_config import (
    parse_deadline_cache_overrides,
    validate_single_deadline_cache_paths,
)


ROOT_DIR = str(PROJECT_ROOT)
DEFAULT_PROTOCOL_CONFIG = (
    PROJECT_ROOT / "algorithms" / "comparisons" / "config"
    / "fuzzy_baselines.json"
)
FORMAL_TEST_SEEDS = tuple(range(201, 231))
SINGLE_GENERALIZATION_GROUPS = {
    "SS": ("SS", "MS", "LS"),
    "SM": ("SM", "MM", "LM"),
    "SL": ("SL", "ML", "LL"),
}

def build_fcfs_protocol(
    scenario="SS",
    ddl="T",
    *,
    config_path=DEFAULT_PROTOCOL_CONFIG,
    workflows_per_episode=None,
    deadline_cache_path=None,
    deadline_cache_paths=None,
):
    """Load the common fuzzy protocol and enforce the formal test split."""
    protocol = protocol_from_config(
        load_protocol_config(config_path),
        scenario=scenario,
        ddl=ddl,
        workflows_per_episode=workflows_per_episode,
        deadline_cache_path=deadline_cache_path,
        deadline_cache_paths=deadline_cache_paths,
    )
    if protocol.test_seeds != FORMAL_TEST_SEEDS:
        raise ValueError(
            "FCFS formal test seeds must be exactly 201-230"
        )
    return protocol


def default_output_path(method_id, protocol):
    """Return the method-specific formal result path."""
    policy = make_policy(method_id)
    return (
        PROJECT_ROOT
        / "out"
        / "comparisons"
        / policy.method_id
        / f"{protocol.scenario}_{protocol.ddl_setting[0]}"
        / "final_test_metrics.json"
    )


def run_formal_evaluation(
    method_id,
    scenario="SS",
    ddl="T",
    *,
    config_path=DEFAULT_PROTOCOL_CONFIG,
    output_path=None,
    workflows_per_episode=None,
    deadline_cache_path=None,
    deadline_cache_paths=None,
):
    """Evaluate one explicit FCFS method on the shared fuzzy environment."""
    policy = make_policy(method_id)
    scenario = str(scenario).strip().upper()
    cache_paths = parse_deadline_cache_overrides(deadline_cache_paths)
    if deadline_cache_path is not None:
        cache_paths[scenario] = str(deadline_cache_path)
    cache_paths = validate_single_deadline_cache_paths(
        "single",
        cache_paths,
        required_scenarios=(scenario,),
    )
    protocol = build_fcfs_protocol(
        scenario,
        ddl,
        config_path=config_path,
        workflows_per_episode=workflows_per_episode,
        deadline_cache_path=cache_paths[scenario],
        deadline_cache_paths=cache_paths,
    )
    result = evaluate_policy(protocol, policy, split="final_test")
    payload = {
        "method_id": policy.method_id,
        "display_name": policy.display_name,
        "task_order": (
            "(ready_time, workflow_arrival_time, workflow_id, task_id)"
        ),
        "resource_rule": (
            "earliest_available_then_vm_id"
            if policy.method_id == "fcfs_fcfs"
            else "shared_select_vm_deterministic"
        ),
        "deadline_cache_paths": dict(protocol.deadline_cache_paths),
        "protocol": protocol.to_manifest(),
        "seed_records": list(result.records),
        "aggregate": result.aggregate,
        "model_selection": result.model_selection.to_dict(),
    }
    destination = (
        Path(output_path)
        if output_path
        else default_output_path(policy.method_id, protocol)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    payload["output_path"] = str(destination.resolve())
    return payload


def run_single_generalization(
    method_id,
    source_scenario="SS",
    ddl="T",
    *,
    config_path=DEFAULT_PROTOCOL_CONFIG,
    output_root=None,
    workflows_per_episode=None,
    deadline_cache_paths=None,
):
    """Evaluate one frozen FCFS method across its Single scenario group."""
    source = str(source_scenario).strip().upper()
    if source not in SINGLE_GENERALIZATION_GROUPS:
        raise ValueError("Single FCFS source scenario must be SS, SM, or SL")
    cache_paths = validate_single_deadline_cache_paths(
        "single",
        deadline_cache_paths,
        source_scenario=source,
        required_scenarios=SINGLE_GENERALIZATION_GROUPS[source],
    )
    policy = make_policy(method_id)
    scenario_results = {}
    for scenario in SINGLE_GENERALIZATION_GROUPS[source]:
        protocol = build_fcfs_protocol(
            scenario,
            ddl,
            config_path=config_path,
            workflows_per_episode=workflows_per_episode,
            deadline_cache_path=cache_paths[scenario],
            deadline_cache_paths=cache_paths,
        )
        result = evaluate_policy(protocol, policy, split="final_test")
        scenario_results[scenario] = {
            "aggregate": result.aggregate,
            "seed_records": list(result.records),
        }
    root = Path(output_root) if output_root else (
        PROJECT_ROOT / "out" / "comparisons" / policy.method_id
        / "main_single" / source
    )
    destination = root / "generalization_metrics.json"
    payload = {
        "method_id": policy.method_id,
        "display_name": policy.display_name,
        "protocol": "single",
        "source_scenario": source,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "test_scenarios": list(SINGLE_GENERALIZATION_GROUPS[source]),
        "final_test_seeds": list(FORMAL_TEST_SEEDS),
        "deadline_cache_paths": dict(cache_paths),
        "scenario_results": scenario_results,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    payload["output_path"] = str(destination.resolve())
    return payload

def main(argv=None, *, fixed_method_id=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_method_id is None:
        parser.add_argument(
            "--method", required=True, choices=sorted(POLICY_TYPES)
        )
    parser.add_argument("--scenario", default="SS")
    parser.add_argument("--ddl", default="T")
    parser.add_argument("--config", default=str(DEFAULT_PROTOCOL_CONFIG))
    parser.add_argument("--output")
    parser.add_argument("--single-source", choices=("SS", "SM", "SL"))
    parser.add_argument(
        "--deadline-cache",
        action="append",
        default=None,
        metavar="SCENARIO=PATH",
    )
    args = parser.parse_args(argv)
    method_id = (
        str(fixed_method_id)
        if fixed_method_id is not None
        else str(args.method)
    )
    deadline_cache_paths = parse_deadline_cache_overrides(
        args.deadline_cache
    )
    payload = (
        run_single_generalization(
            method_id,
            args.single_source,
            args.ddl,
            config_path=args.config,
            output_root=args.output,
            deadline_cache_paths=deadline_cache_paths,
        )
        if args.single_source
        else run_formal_evaluation(
            method_id,
            args.scenario,
            args.ddl,
            config_path=args.config,
            output_path=args.output,
            deadline_cache_paths=deadline_cache_paths,
        )
    )
    print(payload["output_path"])


def main_for_method(method_id, argv=None):
    """Run a dedicated FCFS entry point without an ambiguous default."""
    if method_id not in METHOD_DISPLAY_NAMES:
        raise ValueError(f"unknown FCFS method_id: {method_id}")
    return main(list(argv or []), fixed_method_id=method_id)


if __name__ == "__main__":
    main()
