"""Command-line entry point for fuzzy IRWS, MARL and PD3QN baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from algorithms.comparisons.fuzzy_common.evaluation import make_environment
from algorithms.comparisons.fuzzy_common.protocol import (
    FuzzyComparisonProtocol,
    load_protocol_config,
    protocol_from_config,
)
from algorithms.comparisons.fuzzy_common.training import (
    seed_everything,
    train_baseline,
)
from algorithms.comparisons.irws import IRWSPolicy
from algorithms.comparisons.marl import MARLPolicy
from algorithms.comparisons.pd3qn import PD3QNPolicy
from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
from hrl_mix.train_config import (
    parse_deadline_cache_overrides,
    validate_single_deadline_cache_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "config" / "fuzzy_baselines.json"
)


def _policy(
    method: str,
    env,
    device: str,
    method_config: dict[str, Any] | None = None,
):
    factories = {
        "irws": IRWSPolicy,
        "marl": MARLPolicy,
        "pd3qn": PD3QNPolicy,
    }
    key = str(method).strip().lower()
    if key not in factories:
        raise ValueError("method must be irws, marl or pd3qn")
    return factories[key](
        env,
        device=device,
        **dict(method_config or {}),
    )


build_policy = _policy


def _smoke_protocol(
    protocol: FuzzyComparisonProtocol,
    smoke: dict[str, Any],
) -> FuzzyComparisonProtocol:
    return FuzzyComparisonProtocol(
        scenario=protocol.scenario,
        ddl_setting=protocol.ddl_setting,
        train_seeds=tuple(smoke.get("train_seeds", (1,))),
        validation_seeds=tuple(smoke.get("validation_seeds", (101,))),
        test_seeds=tuple(smoke.get("final_test_seeds", (201,))),
        workflows_per_episode=int(smoke.get("workflows_per_episode", 1)),
        fuzzy_delta1=protocol.fuzzy_delta1,
        fuzzy_delta2=protocol.fuzzy_delta2,
        fuzzy_energy_uncertainty_weight=(
            protocol.fuzzy_energy_uncertainty_weight
        ),
        fuzzy_deadline_eta=protocol.fuzzy_deadline_eta,
        protocol_mode=protocol.protocol_mode,
        source_scenario=protocol.source_scenario,
        resource_scale=protocol.resource_scale,
        training_scenarios=protocol.training_scenarios,
        test_scenarios=protocol.test_scenarios,
        deadline_cache_path=protocol.deadline_cache_path,
        deadline_cache_paths=protocol.deadline_cache_paths,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train one fuzzy comparison baseline",
    )
    parser.add_argument("--method", required=True, choices=("irws", "marl", "pd3qn"))
    parser.add_argument("--scenario", default="SS")
    parser.add_argument("--protocol", choices=("single", "multi"), default="single")
    parser.add_argument("--resource-scale", choices=("S", "M", "L"), default=None)
    parser.add_argument("--ddl", default="T")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--workflows-per-episode", type=int, default=None)
    parser.add_argument(
        "--deadline-cache",
        action="append",
        default=None,
        help="Single cache override as SCENARIO=PATH; one plain source path is allowed.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer-seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.deadline_cache is not None and args.protocol != "single":
        parser.error(
            "--deadline-cache override is currently supported only "
            "for --protocol single"
        )
    if not args.smoke:
        if args.episodes not in (None, 600):
            parser.error("formal baseline training requires exactly 600 episodes")
        if args.validation_interval not in (None, 25):
            parser.error("formal baseline validation interval must be 25 episodes")
        if args.output is not None:
            parser.error("formal Single/Multi output directory is protocol-managed")

    config = load_protocol_config(args.config)
    experiment_context = resolve_experiment_protocol(
        args.protocol,
        source_scenario=(args.scenario if args.protocol == "single" else None),
        resource_scale=args.resource_scale,
    )
    deadline_cache_paths = parse_deadline_cache_overrides(
        args.deadline_cache,
        default_scenario=experiment_context.source_scenario,
    )
    if not args.smoke:
        deadline_cache_paths = validate_single_deadline_cache_paths(
            args.protocol,
            deadline_cache_paths,
            source_scenario=experiment_context.source_scenario,
            required_scenarios=experiment_context.test_scenarios,
        )
    protocol = protocol_from_config(
        config,
        scenario=experiment_context.training_scenarios[0],
        ddl=args.ddl,
        workflows_per_episode=args.workflows_per_episode,
        experiment_context=experiment_context,
        deadline_cache_path=deadline_cache_paths.get(
            experiment_context.training_scenarios[0]
        ),
        deadline_cache_paths=deadline_cache_paths,
    )
    training = dict(config.get("training", {}))
    if args.smoke:
        smoke = dict(config.get("smoke", {}))
        protocol = _smoke_protocol(protocol, smoke)
        episodes = int(args.episodes or smoke.get("episodes", 1))
        validation_interval = int(args.validation_interval or 1)
    else:
        episodes = int(args.episodes or training.get("episodes", 600))
        validation_interval = int(
            args.validation_interval
            or training.get("validation_interval", 25)
        )

    default_output = (
        PROJECT_ROOT
        / "out"
        / "fuzzy_comparisons"
        / args.method
        / (
            "main_single"
            if protocol.protocol_mode == "single"
            else "enhancement_multi"
        )
        / (
            protocol.source_scenario
            if protocol.protocol_mode == "single"
            else protocol.resource_scale
        )
        / protocol.ddl_setting.lower()
    )
    source_cache = deadline_cache_paths.get(protocol.source_scenario)
    if source_cache is not None:
        default_output = (
            default_output
            / "deadline_cache_override"
            / Path(source_cache).stem
        )

    output = (
        args.output.resolve()
        if args.output is not None
        else default_output.resolve()
    )
    seed_everything(int(args.optimizer_seed))
    prototype = make_environment(
        protocol,
        protocol.train_seeds[0],
        reward_config=dict(config.get("reward", {})),
    )
    algorithm_parameters = dict(config.get("algorithm_parameters", {}))
    policy = _policy(
        args.method,
        prototype,
        args.device,
        dict(algorithm_parameters.get(args.method, {})),
    )
    result = train_baseline(
        protocol,
        policy,
        output_dir=output,
        episodes=episodes,
        validation_interval=validation_interval,
        reward_config=dict(config.get("reward", {})),
        max_assignment_steps=int(
            training.get("max_assignment_steps", 1_000_000)
        ),
        optimizer_seed=int(args.optimizer_seed),
    )
    print(
        json.dumps(
            {
                "method": args.method,
                "scenario": protocol.scenario,
                "ddl": protocol.ddl_setting,
                "checkpoint": result.checkpoint_path,
                "manifest": result.manifest_path,
                "best_validation": result.best_validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
