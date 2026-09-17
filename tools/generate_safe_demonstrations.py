"""Generate versioned safe-HRL demonstrations from admitted heuristics.

Example:
    python -m tools.generate_safe_demonstrations ^
      --manifest out/demonstrations/manifest.json ^
      --heuristic traditional_edf --generate-split train ^
      --train-workflow-seeds 1 --train-resource-seeds 1 ^
      --validation-workflow-seeds 2 --validation-resource-seeds 2 ^
      --final-test-workflow-seeds 3 --final-test-resource-seeds 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from base.hrl_env import CloudWorkflowEnv_VMAgents
from base.safe_demonstration import (
    DemonstrationSafetyStandard,
    StrictSeedSplit,
    append_demonstration_episode,
)
from hrl_mix.safe_demonstrations import (
    DemonstrationGenerationOptions,
    generate_safe_demonstration_episode,
)
from hrl_mix.train_config import build_train_config
from hrl_mix.train_utils import apply_env_scales


def _seed_values(text: str) -> tuple[int, ...]:
    values = tuple(
        int(value.strip())
        for value in str(text).split(",")
        if value.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError(
            "seed list must not be empty"
        )
    return values


def _environment_kwargs(cfg, workflow_seed, resource_seed):
    return {
        "dax_paths": cfg.dax_list,
        "horizon": cfg.horizon,
        "arrival_lambda": cfg.arrival_lambda,
        "random_seed": int(workflow_seed),
        "max_ready_tasks": cfg.max_ready_tasks,
        "normalize": cfg.normalize_obs,
        "workflows_per_episode": cfg.workflows_per_episode,
        "num_cloud_hosts": cfg.num_cloud_hosts,
        "num_edge_hosts": cfg.num_edge_hosts,
        "cloud_vms_per_host": cfg.cloud_vms_per_host,
        "edge_vms_per_host": cfg.edge_vms_per_host,
        "cloud_pc_tiers": cfg.cloud_pc_tiers,
        "edge_pc_tiers": cfg.edge_pc_tiers,
        "cloud_bw_tiers": cfg.cloud_bw_tiers,
        "edge_bw_tiers": cfg.edge_bw_tiers,
        "deadline_mode": "cache_fcfs",
        "deadline_cache_path": cfg.deadline_cache_path,
        "deadline_cache_strict": True,
        "deadline_alpha_small": cfg.deadline_alpha_small,
        "deadline_alpha_large": cfg.deadline_alpha_large,
        "deadline_alpha_small_prob": (
            cfg.deadline_alpha_small_prob
        ),
        "manager_alpha_delay": cfg.manager_alpha_delay,
        "manager_delay_mode": cfg.manager_delay_mode,
        "safe_rl_enabled": True,
        "safe_rl_process_risk_aggregation": (
            cfg.safe_rl.process_risk_aggregation
        ),
        "safe_rl_shield_enabled": True,
        "safe_rl_fallback_controller": (
            cfg.safe_rl.shield.fallback_controller
        ),
        "safe_rl_state_enabled": True,
        "safe_rl_state_high_uncertainty_threshold": (
            cfg.safe_rl.state.high_uncertainty_threshold
        ),
        "safe_rl_state_recent_record_window": (
            cfg.safe_rl.state.recent_record_window
        ),
        "manager_mode": "heuristic_selection_mode",
        "manager_heuristic_library_path": (
            cfg.safe_rl.manager_heuristics.library_manifest_path
        ),
        "manager_heuristic_recent_window": (
            cfg.safe_rl.manager_heuristics.recent_window
        ),
        "fuzzy_enabled": True,
        "fuzzy_energy_uncertainty_weight": 1.0,
        "fuzzy_deadline_eta": 0.95,
        "fuzzy_resource_seed": int(resource_seed),
        "fuzzy_use_deadline_constraint": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate real safe-HRL replay-compatible "
            "demonstration episodes."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--heuristic",
        required=True,
        help=(
            "Traditional heuristic ID or an admitted SeEvo "
            "heuristic ID."
        ),
    )
    parser.add_argument(
        "--generate-split",
        choices=("train", "validation", "final_test"),
        required=True,
    )
    parser.add_argument("--scenario", default="SS")
    parser.add_argument("--ddl", default="T")
    for split in ("train", "validation", "final-test"):
        option = split.replace("-", "_")
        parser.add_argument(
            f"--{split}-workflow-seeds",
            type=_seed_values,
            required=True,
            dest=f"{option}_workflow_seeds",
        )
        parser.add_argument(
            f"--{split}-resource-seeds",
            type=_seed_values,
            required=True,
            dest=f"{option}_resource_seeds",
        )
    args = parser.parse_args(argv)

    seed_split = StrictSeedSplit(
        train_workflow_seeds=args.train_workflow_seeds,
        validation_workflow_seeds=(
            args.validation_workflow_seeds
        ),
        final_test_workflow_seeds=(
            args.final_test_workflow_seeds
        ),
        train_resource_seeds=args.train_resource_seeds,
        validation_resource_seeds=(
            args.validation_resource_seeds
        ),
        final_test_resource_seeds=(
            args.final_test_resource_seeds
        ),
    )
    selected = str(args.generate_split)
    workflow_seeds = getattr(
        seed_split, f"{selected}_workflow_seeds"
    )
    resource_seeds = getattr(
        seed_split, f"{selected}_resource_seeds"
    )
    if len(workflow_seeds) != len(resource_seeds):
        raise ValueError(
            "selected split workflow/resource seed lists must "
            "have equal lengths"
        )

    cfg = build_train_config(
        scenario=args.scenario,
        ddl=args.ddl,
        max_episodes=1,
        safe_rl_enabled=True,
        safe_rl_shield_enabled=True,
        safe_rl_state_enabled=True,
        safe_rl_heuristic_manager_enabled=True,
    )
    standard = DemonstrationSafetyStandard()
    for workflow_seed, resource_seed in zip(
        workflow_seeds, resource_seeds
    ):
        env = CloudWorkflowEnv_VMAgents(
            **_environment_kwargs(
                cfg, workflow_seed, resource_seed
            )
        )
        apply_env_scales(env, cfg)
        episode = generate_safe_demonstration_episode(
            env,
            DemonstrationGenerationOptions(
                heuristic_id=args.heuristic,
                workflow_seed=workflow_seed,
                resource_seed=resource_seed,
                split=selected,
            ),
            safety_standard=standard,
        )
        manifest = append_demonstration_episode(
            Path(args.manifest),
            episode,
            seed_split=seed_split,
        )
        print(
            f"{episode.episode_id}: "
            f"safe={episode.safe_demonstration} "
            f"trajectories={episode.trajectory_count['total']} "
            f"energy={episode.episode_metrics['fuzzy_energy_score']:.6f} "
            f"manifest_revision={manifest['manifest_revision']}"
        )


if __name__ == "__main__":
    main()
