"""Run Niching-GP, SA training and evaluation from an existing best RA."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .post_ra_pipeline import run_after_ra


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ra-checkpoint", required=True)
    result.add_argument(
        "--config",
        help="defaults to config.json beside --ra-checkpoint",
    )
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
    checkpoint = Path(args.ra_checkpoint)
    config_path = Path(args.config) if args.config else (
        checkpoint.parent / "config.json"
    )
    config = load_config(config_path)
    result = run_after_ra(
        config,
        checkpoint,
        gp_workers=args.workers,
        gp_device=args.device,
        threads_per_worker=args.threads_per_worker,
        fitness_cache_path=args.fitness_cache,
    )
    print(result["output_dir"])
    print(tuple(result["evaluation"]["comparison_key"]))


if __name__ == "__main__":
    main()
