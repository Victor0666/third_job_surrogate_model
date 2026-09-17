"""Run RA training, Niching GP, SA training, and final evaluation."""

from __future__ import annotations

import argparse
import time

from .config import build_config
from algorithms.llm_safe_hrl.hrl_mix.train_config import (
    parse_deadline_cache_overrides,
    validate_single_deadline_cache_paths,
)
from .post_ra_pipeline import run_after_ra
from .routing_agent import train_routing


def run_pipeline(
    config,
    *,
    gp_workers: int = 1,
    gp_device: str = "cpu",
    threads_per_worker: int = 1,
    fitness_cache_path=None,
) -> dict:
    started = time.perf_counter()
    started_cpu = time.process_time()
    _routing, ra_path, _ra_history = train_routing(config)
    return run_after_ra(
        config,
        ra_path,
        gp_workers=gp_workers,
        gp_device=gp_device,
        threads_per_worker=threads_per_worker,
        fitness_cache_path=fitness_cache_path,
        started=started,
        started_cpu=started_cpu,
        manifest_stage="complete_pipeline",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="SS")
    result.add_argument("--protocol", choices=("single", "multi"), default="single")
    result.add_argument("--resource-scale", choices=("S", "M", "L"), default=None)
    result.add_argument("--ddl", default="T")
    result.add_argument(
        "--algorithm-seed", dest="algorithm_seed", type=int, default=0
    )
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
    if args.workers < 1:
        parser().error("--workers must be at least 1")
    if args.threads_per_worker < 1:
        parser().error("--threads-per-worker must be at least 1")
    if args.workers > 1 and args.device != "cpu":
        parser().error("parallel GP evaluation requires --device cpu")
    result = run_pipeline(
        build_config(
            args.scenario,
            args.ddl,
            args.algorithm_seed,
            smoke=args.smoke,
            deadline_cache_paths=deadline_cache_paths,
            protocol=args.protocol,
            source_scenario=(args.scenario if args.protocol == "single" else None),
            resource_scale=args.resource_scale,
        ),
        gp_workers=args.workers,
        gp_device=args.device,
        threads_per_worker=args.threads_per_worker,
        fitness_cache_path=args.fitness_cache,
    )
    print(result["output_dir"])
    print(tuple(result["evaluation"]["comparison_key"]))


if __name__ == "__main__":
    main()
