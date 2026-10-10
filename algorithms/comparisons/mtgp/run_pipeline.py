"""Train MTGP, select on validation, then evaluate frozen trees on final seeds."""

import argparse
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import numpy as np

from common.scheduling_transport import COMMUNICATION_MODEL_VERSION
from algorithms.comparisons.fuzzy_common.protocol import load_protocol_config, protocol_from_config
from algorithms.comparisons.run_fuzzy_baseline import DEFAULT_CONFIG
from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
from algorithms.llm_safe_hrl.hrl_mix.train_config import (
    parse_deadline_cache_overrides, validate_single_deadline_cache_paths,
)
from project_paths import PROJECT_ROOT
from algorithms.comparisons.drlea_nichgp.checkpointing import write_csv
from .core import RulePair, train, evaluate


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def input_identity(protocol):
    """Bind frozen trees to both the protocol and actual input file contents."""
    files = {}
    for scenario in set(protocol.training_scenarios + protocol.test_scenarios):
        manifest = protocol.for_scenario(scenario).environment_manifest()
        for value in [*manifest["workflow_files"], manifest["deadline_cache_path"]]:
            path = Path(value)
            files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "communication_model_version": COMMUNICATION_MODEL_VERSION,
        "protocol_hash": protocol.protocol_hash, "input_sha256": files,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for root in (PROJECT_ROOT / "common", PROJECT_ROOT / "algorithms/llm_safe_hrl/base",
                         PROJECT_ROOT / "algorithms/comparisons/mtgp")
            for path in sorted(root.rglob("*.py"))
        },
        "evaluation_source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (PROJECT_ROOT / "algorithms/llm_safe_hrl/hrl_mix/safe_metrics.py",
                         PROJECT_ROOT / "algorithms/llm_safe_hrl/hrl_mix/model_selection.py",
                         PROJECT_ROOT / "algorithms/llm_safe_hrl/hrl_mix/train_config.py",
                         PROJECT_ROOT / "algorithms/llm_safe_hrl/scenario_registry.py",
                         PROJECT_ROOT / "algorithms/comparisons/fuzzy_common/protocol.py",
                         PROJECT_ROOT / "algorithms/comparisons/fuzzy_common/environment.py",
                         PROJECT_ROOT / "algorithms/comparisons/fuzzy_common/evaluation.py",
                         PROJECT_ROOT / "algorithms/comparisons/drlea_nichgp/gp_primitives.py")
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="SS")
    parser.add_argument("--ddl", default="T")
    parser.add_argument("--protocol", choices=("single", "multi"), default="single")
    parser.add_argument("--resource-scale", choices=("S", "M", "L"))
    parser.add_argument("--algorithm-seed", type=int, default=0)
    parser.add_argument("--deadline-cache", action="append")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rules-file", type=Path, help="Evaluate an existing frozen tree pair, without training")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Continue the generation checkpoint in the output directory")
    args = parser.parse_args(argv)
    if args.algorithm_seed < 0:
        parser.error("algorithm seed must be non-negative")
    if args.workers < 1 or args.threads_per_worker < 1:
        parser.error("workers and threads-per-worker must be positive")
    if args.resume and args.rules_file:
        parser.error("use --resume or --rules-file, not both")
    context = resolve_experiment_protocol(
        args.protocol, source_scenario=args.scenario if args.protocol == "single" else None,
        resource_scale=args.resource_scale,
    )
    caches = parse_deadline_cache_overrides(args.deadline_cache, default_scenario=context.source_scenario)
    if not args.smoke:
        caches = validate_single_deadline_cache_paths(
            args.protocol, caches, source_scenario=context.source_scenario,
            required_scenarios=context.test_scenarios,
        )
    protocol = protocol_from_config(
        load_protocol_config(DEFAULT_CONFIG), scenario=context.training_scenarios[0], ddl=args.ddl,
        experiment_context=context, deadline_cache_paths=caches,
        deadline_cache_path=caches.get(context.training_scenarios[0]),
    )
    if args.smoke:
        protocol = replace(protocol, workflows_per_episode=2, train_seeds=(1,),
                           validation_seeds=(101,), test_seeds=(201,),
                           test_scenarios=(protocol.scenario,))
    group = context.source_scenario if args.protocol == "single" else context.resource_scale
    output = args.output or (PROJECT_ROOT / "out" / "comparisons" / "mtgp" / "combined_v1"
                            / ("smoke" if args.smoke else args.protocol) / group
                            / f"{args.ddl}_a{args.algorithm_seed}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = input_identity(protocol)
    settings = {"population_size": 8, "generations": 2, "elite_size": 2} if args.smoke else {
        "population_size": 1000, "generations": 51, "elite_size": 10,
    }
    rules_path = output / "rules.json"
    if args.rules_file or (args.resume and rules_path.exists()):
        payload = json.loads((args.rules_file or rules_path).read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("frozen MTGP rules do not match current protocol, model or input files")
        if args.resume and (payload.get("algorithm_seed") != args.algorithm_seed or payload.get("gp_settings") != settings):
            raise ValueError("completed MTGP run does not match requested training settings")
        pair = RulePair.from_dict(payload["rules"])
    else:
        if rules_path.exists():
            raise FileExistsError(f"refusing to overwrite trained MTGP rules: {rules_path}")
        pair, validation, history, counts = train(
            protocol, algorithm_seed=args.algorithm_seed, workers=args.workers,
            threads_per_worker=args.threads_per_worker, checkpoint_path=output / "training_checkpoint.json",
            resume=args.resume, identity=identity, **settings,
        )
        _write(rules_path, {
            "method_id": "mtgp", "identity": identity, "rules": pair.to_dict(),
            "validation": validation, "algorithm_seed": args.algorithm_seed,
            "gp_settings": settings, "evaluation_counts": counts,
            "fitness_fields": ["deadline_violation_rate", "max_fuzzy_lateness", "mean_fuzzy_lateness", "fuzzy_energy_score"],
            "deadline_reference": "shared_frozen_input_cache",
        })
        _write(output / "history.json", history)
    final, records = evaluate(protocol, pair, protocol.test_scenarios, protocol.test_seeds)
    _write(output / "eval.json", {
        "method_id": "mtgp", "identity": identity, "aggregate": final, "records": records,
        "evaluator": "algorithms.llm_safe_hrl.hrl_mix.safe_metrics.build_episode_metric_record",
    })
    write_csv(output / "eval.csv", records)
    print(f"MTGP frozen final evaluation: {final['comparison_key']}\nOutput: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
