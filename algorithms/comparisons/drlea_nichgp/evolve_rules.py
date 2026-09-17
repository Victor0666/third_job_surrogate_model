"""CLI for stage 2: collect situations and evolve four Niching-GP rules."""

from __future__ import annotations

import argparse

from .config import build_config, protocol_artifact_identity
from .checkpointing import portable_path
from .decision_situations import collect_decision_situations
from .niching_gp import evolve_niching_gp
from .routing_agent import RoutingAgent
from algorithms.llm_safe_hrl.hrl_mix.train_config import (
    parse_deadline_cache_overrides,
    validate_single_deadline_cache_paths,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="SS")
    result.add_argument("--ddl", default="T")
    result.add_argument(
        "--algorithm-seed", dest="algorithm_seed", type=int, default=0
    )
    result.add_argument("--ra-checkpoint", required=True)
    result.add_argument("--population", type=int)
    result.add_argument("--generations", type=int)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--deadline-cache", action="append")
    result.add_argument("--workers", type=int, default=1)
    result.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu"
    )
    result.add_argument("--threads-per-worker", type=int, default=1)
    result.add_argument("--fitness-cache")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.workers < 1:
        parser().error("--workers must be at least 1")
    if args.threads_per_worker < 1:
        parser().error("--threads-per-worker must be at least 1")
    if args.workers > 1 and args.device != "cpu":
        parser().error("parallel GP evaluation requires --device cpu")
    deadline_cache_paths = parse_deadline_cache_overrides(
        args.deadline_cache,
        default_scenario=args.scenario,
    )
    if not args.smoke:
        deadline_cache_paths = validate_single_deadline_cache_paths(
            "single",
            deadline_cache_paths,
            source_scenario=args.scenario,
        )
    config = build_config(
        args.scenario,
        args.ddl,
        args.algorithm_seed,
        smoke=args.smoke,
        deadline_cache_paths=deadline_cache_paths,
    )
    if args.population or args.generations:
        from dataclasses import replace

        config = replace(
            config,
            gp=replace(
                config.gp,
                population_size=args.population
                or config.gp.population_size,
                generations=args.generations
                or config.gp.generations,
            ),
        )
    routing = RoutingAgent.load(
        args.ra_checkpoint,
        device=args.device,
        expected_protocol_identity=protocol_artifact_identity(config),
    )
    situations = collect_decision_situations(
        config,
        routing,
        seeds=config.train_seeds,
    )
    _rules, path, _history = evolve_niching_gp(
        config,
        routing,
        situations,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        fitness_cache_path=args.fitness_cache,
    )
    print(portable_path(path))


if __name__ == "__main__":
    main()
