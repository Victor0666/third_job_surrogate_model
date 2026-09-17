"""Generate exact-workload-mix FCFS deadline-reference caches.

This offline data-preparation runner reuses the shared fuzzy environment and
the existing deterministic FCFS policies. It supports both FCFS-FCFS and
FCFS-Fixed for diagnostic comparison while keeping the workflow instances,
resource configuration, fuzzy environment, and random seeds unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Iterable

from algorithms.comparisons.fcfs.policies import (
    POLICY_TYPES,
    make_policy,
)

from algorithms.comparisons.fcfs.train_fcfs import (
    DEFAULT_PROTOCOL_CONFIG,
    build_fcfs_protocol,
)
from algorithms.comparisons.fuzzy_common.environment import FuzzyBaselineEnv
from algorithms.comparisons.fuzzy_common.evaluation import run_episode
from algorithms.llm_safe_hrl.scenario_registry import (
    FINAL_TEST_SEEDS,
    SAFE_HRL_TRAIN_SEEDS,
    SAFE_HRL_VALIDATION_SEEDS,
    SCENARIO_REGISTRY,
    workload_category_counts,
)


FORMAL_ENVIRONMENT_SEEDS = tuple(sorted(set(
    SAFE_HRL_TRAIN_SEEDS
    + SAFE_HRL_VALIDATION_SEEDS
    + FINAL_TEST_SEEDS
)))


def _generate_seed_record(args) -> dict:
    (
        scenario,
        ddl,
        config_path,
        workflows_per_episode,
        seed,
        policy_id,
    ) = args
    protocol = build_fcfs_protocol(
        scenario,
        ddl,
        config_path=config_path,
        workflows_per_episode=workflows_per_episode,
    )
    kwargs = protocol.environment_kwargs(
        int(seed), require_deadline_cache=False
    )
    kwargs.update({
        "deadline_mode": "none",
        "deadline_cache_path": None,
        "deadline_cache_strict": False,
        "fuzzy_use_deadline_constraint": False,
    })
    env = FuzzyBaselineEnv(**kwargs)

    policy = make_policy(policy_id)
    run_episode(
        env,
        policy,
        seed=int(seed),
        training=False,
    )
    names = list(env.episode_dax_sequence)
    if len(names) != int(workflows_per_episode):
        raise RuntimeError("generated FCFS episode has wrong DAX count")
    makespans = []
    for workflow_id, workflow in enumerate(env.workflows):
        finish = float(env.wf_finish_time[int(workflow_id)])
        makespans.append(finish - float(workflow.arrival_time))
    if len(makespans) != len(names):
        raise RuntimeError("generated FCFS makespan count does not match DAX count")
    return {
        "seed": int(seed),
        "wf_makespans": makespans,
        "wf_dax_names": names,
        "arrival_times": [float(value) for value in env.arrival_times],
    }


def generate_exact_deadline_cache(
    scenario: str,
    *,
    seeds: Iterable[int] = FORMAL_ENVIRONMENT_SEEDS,
    ddl: str = "T",
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
    workflows_per_episode: int = 50,
    workers: int = 1,
    output_path: str | Path | None = None,
    policy_id: str = "fcfs_fcfs",
) -> Path:
    """Run one deterministic FCFS policy and write an exact-mix cache."""
    policy_id = str(policy_id).strip().lower()
    if policy_id not in POLICY_TYPES:
        raise ValueError(
            f"unknown policy_id {policy_id!r}; "
            f"expected one of {sorted(POLICY_TYPES)}"
        )
    if policy_id != "fcfs_fcfs" and output_path is None:
        raise ValueError(
            "FCFS-Fixed cache generation requires an explicit --output "
            "to avoid overwriting the formal FCFS-FCFS cache."
        )

    scenario_id = str(scenario).strip().upper()
    spec = SCENARIO_REGISTRY[scenario_id]

    seed_values = tuple(sorted({int(value) for value in seeds}))
    if not seed_values:
        raise ValueError("deadline-cache generation requires at least one seed")

    jobs = [
        (
            scenario_id,
            str(ddl),
            str(Path(config_path).resolve()),
            int(workflows_per_episode),
            seed,
            policy_id,
        )
        for seed in seed_values
    ]
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            records = list(executor.map(_generate_seed_record, jobs))
    else:
        records = [_generate_seed_record(job) for job in jobs]
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else spec.deadline_cache_path()
    )
    payload = {
        "meta": {
            "schema_version": "exact_workload_mix_v1",
            # "policy": "FCFS_task + VM_earliest_available_then_id",
            "policy_id": policy_id,
            "policy": (
                "FCFS_task + VM_earliest_available_then_id"
                if policy_id == "fcfs_fcfs"
                else "FCFS_task + shared_select_vm_deterministic"
            ),
            "scenario": scenario_id,
            "task_scale": spec.task_code,
            "resource_scale": spec.resource_code,
            "category_counts": workload_category_counts(
                spec.task_code, int(workflows_per_episode)
            ),
            "workflows_per_episode": int(workflows_per_episode),
            "environment_seeds": list(seed_values),
            "resource": spec.resource_scale.scale_mapping(),
        },
        "data": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=tuple(SCENARIO_REGISTRY))
    parser.add_argument("--ddl", default="T")
    parser.add_argument("--config", default=str(DEFAULT_PROTOCOL_CONFIG))
    parser.add_argument("--workflows", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--policy",default="fcfs_fcfs", choices=sorted(POLICY_TYPES),)
    args = parser.parse_args(argv)
    print(generate_exact_deadline_cache(
        args.scenario,
        ddl=args.ddl,
        config_path=args.config,
        workflows_per_episode=args.workflows,
        workers=args.workers,
        output_path=args.output,
        policy_id=args.policy,
    ))


if __name__ == "__main__":
    main()
