"""CLI for stage 1: train the flat global-VM routing agent."""

from __future__ import annotations

import argparse

from .config import build_config
from .checkpointing import portable_path
from .routing_agent import train_routing
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
    result.add_argument("--episodes", type=int)
    result.add_argument("--reward-mode", default="deadline_energy")
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
        ra_episodes=args.episodes or 300,
        reward_mode=args.reward_mode,
        smoke=args.smoke,
        deadline_cache_paths=deadline_cache_paths,
        protocol=args.protocol,
        source_scenario=(args.scenario if args.protocol == "single" else None),
        resource_scale=args.resource_scale,
    )
    _agent, path, _history = train_routing(config)
    print(portable_path(path))


if __name__ == "__main__":
    main()
