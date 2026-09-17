"""Frozen 30-seed evaluation for trained fuzzy comparison baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from algorithms.comparisons.fuzzy_common.evaluation import (
    evaluate_policy,
    make_environment,
)
from algorithms.comparisons.fuzzy_common.protocol import (
    load_protocol_config,
    protocol_from_config,
)
from algorithms.comparisons.run_fuzzy_baseline import (
    DEFAULT_CONFIG,
    build_policy,
)
from algorithms.llm_safe_hrl.scenario_registry import (
    resolve_experiment_protocol,
)
from hrl_mix.train_config import (
    parse_deadline_cache_overrides,
    validate_single_deadline_cache_paths,
)


METHODS = ("irws", "marl", "pd3qn")
SUMMARY_FIELDS = (
    "method",
    "source_scenario",
    "test_scenario",
    "protocol",
    "ddl",
    "checkpoint",
    "optimizer_seed",
    "evaluation_seed_count",
    "deadline_violation_rate",
    "max_fuzzy_lateness",
    "mean_fuzzy_lateness",
    "fuzzy_energy_mean",
    "fuzzy_energy_std",
    "fuzzy_energy_score",
    "feasible_seed_rate",
    "mean_seed_scheduling_time_seconds",
)
SEED_FIELDS = (
    "method",
    "source_scenario",
    "test_scenario",
    "protocol",
    "ddl",
    "seed",
)


def _resolve_device(requested: str) -> str:
    value = str(requested).strip().lower()
    if value == "cuda":
        import torch

        if not torch.cuda.is_available():
            return "cpu"
    return value


def _optimizer_seed(checkpoint: Path) -> int | None:
    manifest_path = checkpoint.parent / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    value = payload.get("optimizer_seed")
    if value is None:
        value = (
            payload.get("config_snapshot", {})
            .get("config", {})
            .get("optimizer_seed")
        )
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        if isinstance(value, set):
            value = sorted(value)
        return json.dumps(value, ensure_ascii=False)
    return value


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        for key in row:
            if key not in result:
                result.append(key)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _csv_value(row.get(key)) for key in fieldnames}
            )


def _write_outputs(payload: dict[str, Any], output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    summary_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for scenario, result in payload["scenario_results"].items():
        aggregate = dict(result["aggregate"])
        summary = {
            "method": payload["method"],
            "source_scenario": payload["source_scenario"],
            "test_scenario": scenario,
            "protocol": payload["protocol"],
            "ddl": payload["ddl"],
            "checkpoint": payload["checkpoint"],
            "optimizer_seed": payload["optimizer_seed"],
            "evaluation_seed_count": len(result["seed_records"]),
        }
        for field in SUMMARY_FIELDS[8:]:
            summary[field] = aggregate.get(field)
        for key, value in aggregate.items():
            summary.setdefault(key, value)
        summary_rows.append(summary)

        for record in result["seed_records"]:
            row = {
                "method": payload["method"],
                "source_scenario": payload["source_scenario"],
                "test_scenario": scenario,
                "protocol": payload["protocol"],
                "ddl": payload["ddl"],
                "seed": record.get("seed"),
            }
            for key, value in record.items():
                row.setdefault(key, value)
            seed_rows.append(row)

    summary_path = output_path.with_name(
        f"{output_path.stem}_summary.csv"
    )
    seed_path = output_path.with_name(
        f"{output_path.stem}_seed_records.csv"
    )
    _write_csv(summary_path, summary_rows)
    _write_csv(seed_path, seed_rows)
    return summary_path, seed_path


def evaluate_checkpoint(
    *,
    method: str,
    checkpoint: str | Path,
    source_scenario: str,
    protocol_name: str,
    ddl: str,
    device: str,
    deadline_cache_paths,
    output: str | Path | None = None,
) -> dict[str, Any]:
    method_id = str(method).strip().lower()
    if method_id not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if str(protocol_name).strip().lower() != "single":
        raise ValueError("frozen fuzzy baseline evaluation supports only single")

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    context = resolve_experiment_protocol(
        "single",
        source_scenario=source_scenario,
    )
    cache_paths = parse_deadline_cache_overrides(
        deadline_cache_paths,
        default_scenario=context.source_scenario,
    )
    cache_paths = validate_single_deadline_cache_paths(
        "single",
        cache_paths,
        source_scenario=context.source_scenario,
        required_scenarios=context.test_scenarios,
    )

    config = load_protocol_config(DEFAULT_CONFIG)
    protocol = protocol_from_config(
        config,
        scenario=context.training_scenarios[0],
        ddl=ddl,
        experiment_context=context,
        deadline_cache_path=cache_paths[context.training_scenarios[0]],
        deadline_cache_paths=cache_paths,
    )
    if tuple(protocol.test_seeds) != tuple(context.final_test_seeds):
        raise ValueError("formal final-test seeds must come from the Single protocol")

    reward_config = dict(config.get("reward", {}))
    training_config = dict(config.get("training", {}))
    prototype = make_environment(
        protocol,
        protocol.train_seeds[0],
        reward_config=reward_config,
    )
    policy = build_policy(
        method_id,
        prototype,
        _resolve_device(device),
        dict(config.get("algorithm_parameters", {}).get(method_id, {})),
    )
    policy.load(str(checkpoint_path))

    scenario_results: dict[str, Any] = {}
    for scenario in context.test_scenarios:
        result = evaluate_policy(
            protocol.for_scenario(scenario),
            policy,
            split="final_test",
            reward_config=reward_config,
            max_assignment_steps=int(
                training_config.get("max_assignment_steps", 1_000_000)
            ),
        )
        records = list(result.records)
        if len(records) != len(protocol.test_seeds):
            raise RuntimeError(
                f"scenario {scenario} produced {len(records)} seed records; "
                f"expected {len(protocol.test_seeds)}"
            )
        scenario_results[scenario] = {
            "aggregate": dict(result.aggregate),
            "seed_records": records,
        }

    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else checkpoint_path.parent / "final_test_metrics_reval.json"
    )
    payload = {
        "method": method_id,
        "source_scenario": context.source_scenario,
        "protocol": "single",
        "ddl": str(ddl),
        "checkpoint": str(checkpoint_path),
        "optimizer_seed": _optimizer_seed(checkpoint_path),
        "training_during_test": False,
        "checkpoint_reselection": False,
        "test_seeds": list(protocol.test_seeds),
        "deadline_cache_paths": dict(cache_paths),
        "scenario_results": scenario_results,
    }
    summary_path, seed_path = _write_outputs(payload, output_path)
    payload["output_files"] = {
        "json": str(output_path),
        "summary_csv": str(summary_path),
        "seed_records_csv": str(seed_path),
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--protocol", choices=("single",), default="single")
    parser.add_argument("--ddl", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--deadline-cache",
        action="append",
        required=True,
        metavar="SCENARIO=PATH",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_checkpoint(
        method=args.method,
        checkpoint=args.checkpoint,
        source_scenario=args.scenario,
        protocol_name=args.protocol,
        ddl=args.ddl,
        device=args.device,
        deadline_cache_paths=args.deadline_cache,
        output=args.output,
    )
    print(json.dumps(payload["output_files"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
