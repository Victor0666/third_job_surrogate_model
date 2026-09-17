"""Shared GP, SA and frozen-evaluation pipeline after RA training."""

from __future__ import annotations

from pathlib import Path
import time

from .checkpointing import (
    experiment_manifest,
    file_sha256,
    portable_path,
    prepare_output,
    read_json,
    warn_if_source_changed,
    write_json,
)
from .gp_fitness_cache import routing_online_hash
from .config import (
    config_for_scenario,
    protocol_artifact_identity,
)
from .decision_situations import collect_decision_situations
from .evaluate import evaluate_frozen
from .niching_gp import evolve_niching_gp, load_rules
from .routing_agent import RoutingAgent
from .sequencing_agent import SequencingAgent, train_sequencing


def run_after_ra(
    config,
    ra_checkpoint: str | Path,
    *,
    gp_workers: int = 1,
    gp_device: str = "cpu",
    threads_per_worker: int = 1,
    fitness_cache_path: str | Path | None = None,
    started: float | None = None,
    started_cpu: float | None = None,
    manifest_stage: str = "post_ra_pipeline",
) -> dict:
    """Run every frozen-RA stage from one validated best checkpoint."""

    started = time.perf_counter() if started is None else float(started)
    started_cpu = (
        time.process_time()
        if started_cpu is None
        else float(started_cpu)
    )
    output = prepare_output(config)
    identity = protocol_artifact_identity(config)
    ra_path = Path(ra_checkpoint)
    # 只告警不失败：已经训练好的 RA 记的是旧口径指纹，硬失败会把几天的算力
    # 直接作废。这里的作用是让"stage-1 与 stage-2 代码不同源"这件事显式暴露。
    ra_source_status = warn_if_source_changed(
        ra_path.parent / "ra_manifest.json",
        label="stage-1 RA checkpoint",
    )
    routing = RoutingAgent.load(
        ra_path,
        device=gp_device,
        expected_protocol_identity=identity,
    )
    situations = collect_decision_situations(
        config,
        routing,
        seeds=config.train_seeds,
    )
    write_json(
        output / "situations.json",
        [item.to_dict() for item in situations],
    )
    rules, rules_path, _gp_history = evolve_niching_gp(
        config,
        routing,
        situations,
        workers=gp_workers,
        threads_per_worker=threads_per_worker,
        fitness_cache_path=fitness_cache_path,
    )
    sequencing, sa_path, _sa_history = train_sequencing(
        config, routing, rules
    )
    ra_hash = routing_online_hash(routing)
    rules_hash = file_sha256(rules_path)
    sa_manifest_path = output / "sa_manifest.json"
    sa_manifest = read_json(sa_manifest_path)
    sa_manifest.update(
        {
            "ra_online_sha256": ra_hash,
            "rules_file_sha256": rules_hash,
        }
    )
    write_json(sa_manifest_path, sa_manifest)
    routing = RoutingAgent.load(
        ra_path,
        device=gp_device,
        expected_protocol_identity=identity,
    )
    sequencing = SequencingAgent.load(
        sa_path,
        expected_protocol_identity=identity,
    )
    rules = load_rules(
        rules_path,
        expected_protocol_identity=identity,
    )
    scenario_evaluations = {}
    for scenario in config.test_scenarios:
        evaluation_config = config_for_scenario(config, scenario)
        scenario_evaluations[scenario] = evaluate_frozen(
            evaluation_config,
            routing,
            sequencing,
            rules,
            config.test_seeds,
            output_path=(
                output / "eval.json"
                if (
                    config.protocol == "legacy"
                    and len(config.test_scenarios) == 1
                )
                else output / f"eval_{scenario}.json"
            ),
        )
    evaluation = scenario_evaluations[config.test_scenarios[0]]
    write_json(
        output / "generalization_metrics.json",
        {
            "protocol": config.protocol,
            "source_scenario": config.source_scenario,
            "training_scenarios": list(config.training_scenarios),
            "test_scenarios": list(config.test_scenarios),
            "training_during_generalization_test": False,
            "checkpoint_reselection_during_generalization_test": False,
            "scenario_results": scenario_evaluations,
        },
    )
    gp_manifest = read_json(output / "gp_manifest.json")
    manifest = experiment_manifest(
        config,
        stage=manifest_stage,
        elapsed_seconds=time.perf_counter() - started,
        failures=int(gp_manifest["failure_count"]),
        invalid_individuals=int(
            gp_manifest["invalid_individual_count"]
        ),
        extra={
            "ra_checkpoint": portable_path(ra_path),
            "sa_checkpoint": portable_path(sa_path),
            "rules_file": portable_path(rules_path),
            "ra_online_sha256": ra_hash,
            "rules_file_sha256": rules_hash,
            "instance_fingerprints": [
                row["instance_fingerprint"]
                for row in evaluation["seed_metrics"]
            ],
            "comparison_key": evaluation["comparison_key"],
            "generalization_scenarios": list(config.test_scenarios),
            "generalization_metrics": "generalization_metrics.json",
            "routing_inference_device": str(routing.device),
            "compute_device": str(sequencing.device),
            "gp_worker_count": int(max(1, gp_workers)),
            "ra_source_hash_status": ra_source_status,
            "process_cpu_seconds": float(
                time.process_time() - started_cpu
            ),
        },
    )
    write_json(output / "manifest.json", manifest)
    return {
        "output_dir": portable_path(output),
        "ra_checkpoint": portable_path(ra_path),
        "rules_file": portable_path(rules_path),
        "sa_checkpoint": portable_path(sa_path),
        "evaluation": evaluation,
        "scenario_evaluations": scenario_evaluations,
    }
