"""CLI and API for final frozen RA + GP + SA evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from .checkpointing import (
    experiment_manifest,
    portable_path,
    prepare_output,
    write_csv,
    write_json,
)
from .config import (
    build_config,
    config_for_scenario,
    ensure_disjoint_seeds,
    protocol_artifact_identity,
)
from .niching_gp import load_rules
from .routing_agent import RoutingAgent
from .sequencing_agent import (
    SequencingAgent,
    evaluate_sequencing,
)

from algorithms.comparisons.fuzzy_common.evaluation import (
    aggregate_paper_final_metrics,
)
from algorithms.llm_safe_hrl.hrl_mix.train_config import (
    parse_deadline_cache_overrides,
    validate_single_deadline_cache_paths,
)


def evaluate_frozen(
    config,
    routing,
    sequencing,
    rules,
    seeds,
    *,
    output_path: str | Path | None = None,
) -> dict:
    ensure_disjoint_seeds(
        config.train_seeds,
        config.validation_seeds,
        tuple(int(seed) for seed in seeds),
    )
    evaluation_seeds = tuple(int(seed) for seed in seeds)
    if config.protocol != "legacy" and evaluation_seeds != tuple(config.test_seeds):
        raise ValueError(
            "formal DRL-EA final evaluation must use the configured paper final-test seeds"
        )
    started = time.perf_counter()
    started_cpu = time.process_time()
    if config.protocol == "legacy":
        output = prepare_output(config)
    else:
        output = config.output_dir / "generalization" / config.scenario
        output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_sequencing(
        config, routing, rules, sequencing, seeds
    )
    metrics.update(
        aggregate_paper_final_metrics(metrics["seed_metrics"])
    )
    metrics["final_metrics_evaluator"] = (
        "algorithms.comparisons.fuzzy_common.evaluation."
        "aggregate_paper_final_metrics"
    )
    path = Path(output_path or output / "eval.json")
    write_json(path, metrics)
    write_csv(output / "eval.csv", metrics["seed_metrics"])
    workflow_rows = [
        {
            "seed": row["seed"],
            **workflow,
        }
        for row in metrics["seed_metrics"]
        for workflow in row["workflow_metrics"]
    ]
    write_csv(output / "workflows.csv", workflow_rows)
    write_json(
        output / "eval_manifest.json",
        experiment_manifest(
            config,
            stage="evaluation",
            elapsed_seconds=time.perf_counter() - started,
            extra={
                "evaluation_seeds": [int(seed) for seed in seeds],
                "comparison_key": metrics["comparison_key"],
                "all_seed_feasible": metrics[
                    "all_seed_feasible"
                ],
                "feasible_seed_rate": metrics[
                    "feasible_seed_rate"
                ],
                "worst_seed_violation": metrics[
                    "worst_seed_violation"
                ],
                "worst_seed_lateness": metrics[
                    "worst_seed_lateness"
                ],
                "compute_device": str(sequencing.device),
                "process_cpu_seconds": float(
                    time.process_time() - started_cpu
                ),
            },
        ),
    )
    return metrics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="SS")
    result.add_argument("--protocol", choices=("single", "multi"), default="single")
    result.add_argument("--resource-scale", choices=("S", "M", "L"), default=None)
    result.add_argument("--ddl", default="T")
    result.add_argument(
        "--algorithm-seed", dest="algorithm_seed", type=int, default=0
    )
    result.add_argument("--ra-checkpoint", required=True)
    result.add_argument("--sa-checkpoint", required=True)
    result.add_argument("--rules-file", required=True)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--deadline-cache", action="append")
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
    config = build_config(
        args.scenario,
        args.ddl,
        args.algorithm_seed,
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
    sequencing = SequencingAgent.load(
        args.sa_checkpoint,
        expected_protocol_identity=identity,
    )
    rules = load_rules(
        args.rules_file,
        expected_protocol_identity=identity,
    )
    scenario_results = {}
    for scenario in config.test_scenarios:
        target = config_for_scenario(config, scenario)
        scenario_results[scenario] = evaluate_frozen(
            target,
            routing,
            sequencing,
            rules,
            config.test_seeds,
        )
    report_path = config.output_dir / "generalization_metrics.json"
    write_json(
        report_path,
        {
            "protocol_identity": identity,
            "training_during_generalization_test": False,
            "checkpoint_reselection_during_generalization_test": False,
            "scenario_results": scenario_results,
        },
    )
    print(portable_path(report_path))
    print(tuple(scenario_results[config.test_scenarios[0]]["comparison_key"]))


if __name__ == "__main__":
    main()
