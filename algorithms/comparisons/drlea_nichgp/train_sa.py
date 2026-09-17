"""CLI for stage 3: train the four-action sequencing agent."""

from __future__ import annotations

import argparse

from .config import build_config, protocol_artifact_identity
from .checkpointing import portable_path
from .niching_gp import load_rules
from .routing_agent import RoutingAgent
from .sequencing_agent import train_sequencing
from algorithms.llm_safe_hrl.hrl_mix.train_config import (
    parse_deadline_cache_overrides,
    validate_single_deadline_cache_paths,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="SS")
    result.add_argument("--ddl", default="T")
    result.add_argument("--protocol", choices=("single", "multi"), default="single")
    result.add_argument("--resource-scale", choices=("S", "M", "L"), default=None)
    result.add_argument(
        "--algorithm-seed", dest="algorithm_seed", type=int, default=0
    )
    result.add_argument("--ra-checkpoint", required=True)
    result.add_argument("--rules-file", required=True)
    result.add_argument("--episodes", type=int)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--deadline-cache", action="append")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.smoke and args.episodes not in (None, 300):
        parser().error("formal DRL-EA training requires exactly 300 episodes")
    deadline_cache_paths = parse_deadline_cache_overrides(
        args.deadline_cache,
        default_scenario=args.scenario,
    )
    if not args.smoke:
        deadline_cache_paths = validate_single_deadline_cache_paths(
            args.protocol,
            deadline_cache_paths,
            source_scenario=(args.scenario if args.protocol == "single" else None),
        )
    config = build_config(
        args.scenario,
        args.ddl,
        args.algorithm_seed,
        sa_episodes=args.episodes or 300,
        smoke=args.smoke,
        deadline_cache_paths=deadline_cache_paths,
        protocol=args.protocol,
        source_scenario=(args.scenario if args.protocol == "single" else None),
        resource_scale=args.resource_scale,
    )
    identity = protocol_artifact_identity(config)
    routing = RoutingAgent.load(
        args.ra_checkpoint,
        expected_protocol_identity=identity,
    )
    rules = load_rules(
        args.rules_file,
        expected_protocol_identity=identity,
    )
    _agent, path, _history = train_sequencing(
        config, routing, rules
    )
    print(portable_path(path))


if __name__ == "__main__":
    main()
