"""Run all formal main-single GP/SA pipelines from trained RA checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from project_paths import PROJECT_ROOT

from .checkpointing import file_sha256, read_json, write_json
from .config import load_config, protocol_artifact_identity
from .gp_fitness_cache import (
    evaluation_identity,
    routing_online_hash,
)
from .niching_gp import load_rules
from .post_ra_pipeline import run_after_ra
from .routing_agent import RoutingAgent
from .sequencing_agent import SequencingAgent


DEFAULT_CONFIGS = tuple(
    f"{scenario}:{ddl}"
    for scenario in ("SS", "SM", "SL")
    for ddl in ("T", "M", "L")
)


def _parse_configs(values) -> tuple[tuple[str, str], ...]:
    values = values or DEFAULT_CONFIGS
    result = []
    for value in values:
        parts = str(value).strip().upper().split(":")
        if (
            len(parts) != 2
            or parts[0] not in {"SS", "SM", "SL"}
            or parts[1] not in {"T", "M", "L"}
        ):
            raise ValueError(
                "matrix configs must use SOURCE:DDL, for example SS:T"
            )
        item = (parts[0], parts[1])
        if item not in result:
            result.append(item)
    return tuple(result)


def _completed_artifacts_are_valid(
    config,
    ra_path: Path,
    device: str,
) -> bool:
    output = config.output_dir
    required = (
        output / "rules.json",
        output / "sa.pt",
        output / "sa_manifest.json",
        output / "manifest.json",
        output / "generalization_metrics.json",
    )
    if not all(path.is_file() for path in required):
        return False
    identity = protocol_artifact_identity(config)
    routing = RoutingAgent.load(
        ra_path,
        device=device,
        expected_protocol_identity=identity,
    )
    try:
        rules_payload = read_json(output / "rules.json")
        expected_fingerprint = evaluation_identity(
            config, routing, device
        )["sha256"]
        if (
            rules_payload.get("evaluation_fingerprint")
            != expected_fingerprint
        ):
            return False
        load_rules(
            output / "rules.json",
            expected_protocol_identity=identity,
        )
        sa_manifest = read_json(output / "sa_manifest.json")
        if (
            sa_manifest.get("ra_online_sha256")
            != routing_online_hash(routing)
            or sa_manifest.get("rules_file_sha256")
            != file_sha256(output / "rules.json")
        ):
            return False
        SequencingAgent.load(
            output / "sa.pt",
            device="cpu",
            expected_protocol_identity=identity,
        )
    except (
        EOFError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--configs",
        action="append",
        help="repeat SOURCE:DDL; defaults to all nine configurations",
    )
    result.add_argument("--workers", type=int, default=1)
    result.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu"
    )
    result.add_argument("--threads-per-worker", type=int, default=1)
    result.add_argument("--force", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.workers < 1:
        parser().error("--workers must be at least 1")
    if args.threads_per_worker < 1:
        parser().error("--threads-per-worker must be at least 1")
    if args.workers > 1 and args.device != "cpu":
        parser().error("parallel GP evaluation requires --device cpu")
    try:
        selected = _parse_configs(args.configs)
    except ValueError as exc:
        parser().error(str(exc))
    root = (
        PROJECT_ROOT
        / "out"
        / "comparisons"
        / "drlea_nichgp"
        / "main_single"
    )
    status_path = root / "gp_matrix_status.json"
    status = {
        "worker_count": int(args.workers),
        "device": args.device,
        "configs": {},
    }
    for scenario, ddl in selected:
        key = f"{scenario}:{ddl}"
        directory = root / scenario / f"{ddl}_a0"
        config_path = directory / "config.json"
        ra_path = directory / "ra.pt"
        started = time.perf_counter()
        if not config_path.is_file() or not ra_path.is_file():
            status["configs"][key] = {
                "status": "missing_ra_artifact",
                "config": str(config_path),
                "ra_checkpoint": str(ra_path),
            }
            write_json(status_path, status)
            continue
        try:
            config = load_config(config_path)
            if (
                not args.force
                and _completed_artifacts_are_valid(
                    config, ra_path, args.device
                )
            ):
                status["configs"][key] = {
                    "status": "skipped_valid_complete",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            else:
                result = run_after_ra(
                    config,
                    ra_path,
                    gp_workers=args.workers,
                    gp_device=args.device,
                    threads_per_worker=args.threads_per_worker,
                )
                status["configs"][key] = {
                    "status": "completed",
                    "elapsed_seconds": time.perf_counter() - started,
                    "rules_file": result["rules_file"],
                    "sa_checkpoint": result["sa_checkpoint"],
                    "comparison_key": result["evaluation"][
                        "comparison_key"
                    ],
                }
        except Exception as exc:
            status["configs"][key] = {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        write_json(status_path, status)
    print(status_path)


if __name__ == "__main__":
    main()
