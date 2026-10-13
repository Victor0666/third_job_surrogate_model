"""Collect fresh batch cases and evaluate joint ICL using a verified MADDPG teacher."""
import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from algorithms.comparisons.dtecn_llm import run as runner
from algorithms.comparisons.dtecn_llm.maddpg import MADDPG
from tools.dtecn_batch_policy import BatchPolicy, BATCH_PROMPT
from tools.evaluate_dtecn_llm_with_csv import evaluate_with_csv, load_compatible_manifest
from tools.dtecn_evaluation_llm import EvaluationLLMClient as LLMClient, evaluation_llm_config


def load_batch_manifest(root, expected, expected_teacher, teacher_identity):
    saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if saved.get("identity", {}).get("case_schema") == 2:
        return runner.load_manifest(root, expected)
    # Only the published v1 collector/logging versions can migrate; all other fields remain exact.
    legacy = deepcopy(expected_teacher)
    legacy.update(decision_mode="fcfs_joint_batch_v1", teacher_identity=teacher_identity,
        teacher_checkpoint_sha256=expected["teacher_checkpoint_sha256"],
        prompt_sha256=hashlib.sha256(BATCH_PROMPT.encode()).hexdigest(),
        batch_source_hashes={
            str(Path(__file__).resolve()): "68a54ac46bfce29b989eaefb6242543ae85db6b9626b2550dafb899f935722b7",
            str(Path(__file__).with_name("dtecn_batch_policy.py").resolve()): expected["batch_policy_sha256"],
            str(Path(__file__).with_name("evaluate_dtecn_llm_with_csv.py").resolve()):
                "5926b798c248d83ddd2a5f66fdef18327c0e2d943be49751874a15fbb4fa6f65"})
    manifest = load_compatible_manifest(root, legacy)
    print("Verified legacy batch cases; original manifest preserved", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "validate", "evaluate"))
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--protocol", choices=("single", "multi"), default="single")
    parser.add_argument("--scenario", default="SS")
    parser.add_argument("--resource-scale", choices=("S", "M", "L"))
    parser.add_argument("--ddl", default="T")
    parser.add_argument("--optimizer-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path, default=runner.CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--deadline-cache", action="append")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    context = runner.resolve_experiment_protocol(args.protocol,
        source_scenario=args.scenario if args.protocol == "single" else None, resource_scale=args.resource_scale)
    caches = runner.parse_deadline_cache_overrides(args.deadline_cache, default_scenario=context.source_scenario)
    if caches:
        caches = runner.validate_single_deadline_cache_paths(args.protocol, caches,
            source_scenario=context.source_scenario, required_scenarios=context.test_scenarios)
    protocol = runner.protocol_from_config({}, scenario=context.training_scenarios[0], ddl=args.ddl,
        experiment_context=context, deadline_cache_paths=caches)
    random.seed(args.optimizer_seed)
    np.random.seed(args.optimizer_seed)
    torch.manual_seed(args.optimizer_seed)
    if args.device == "cpu":
        torch.set_num_threads(1)
    teacher = args.teacher_run.resolve()
    teacher_manifest = json.loads((teacher / "manifest.json").read_text(encoding="utf-8"))
    # LLM flags apply only to evaluation; every non-LLM training field remains checked.
    config["llm"] = deepcopy(teacher_manifest["identity"]["config"]["llm"])
    expected_teacher = runner.identity(protocol, config, args.optimizer_seed, False)
    verified_teacher = load_compatible_manifest(teacher, expected_teacher)
    expected = deepcopy(expected_teacher)
    expected.update(case_schema=2, decision_mode="fcfs_joint_batch_v1",
        collector_sha256=runner.file_hash(Path(__file__)),
        teacher_checkpoint_sha256=runner.file_hash(teacher / "maddpg.pt"),
        teacher_identity=verified_teacher["identity"],
        prompt_sha256=hashlib.sha256(BATCH_PROMPT.encode()).hexdigest(),
        batch_policy_sha256=runner.file_hash(Path(__file__).with_name("dtecn_batch_policy.py")))
    root = teacher / "batch_v1"
    if args.stage == "collect":
        if root.exists():
            raise FileExistsError("preserve existing batch collection; inspect/archive explicitly")
        env = runner.environment(protocol, protocol.scenario, protocol.train_seeds[0])
        agent = MADDPG(protocol.workflows_per_episode, env.observation_dim, env.num_vms, config["maddpg"], args.device)
        agent.load(teacher / "maddpg.pt")
        root.mkdir()
        shutil.copyfile(teacher / "maddpg.pt", root / "maddpg.pt")
        manifest = deepcopy(verified_teacher)
        manifest.update(identity=expected, artifacts={"maddpg.pt": runner.file_hash(root / "maddpg.pt")})
        policy = BatchPolicy(config, agent=agent, collect=True)
        started = runner.time.perf_counter()
        steps = episodes = 0
        for scenario in protocol.training_scenarios:
            for seed in protocol.train_seeds:
                protocol.assert_not_test_seed(seed, "batch collection")
                record = runner.episode(protocol, scenario, seed, policy)
                steps += record["assignment_steps"]
                episodes += 1
        with (root / "cases.jsonl").open("x", encoding="utf-8") as stream:
            for case in policy.collected:
                stream.write(json.dumps(case, allow_nan=False) + "\n")
        manifest["artifacts"]["cases.jsonl"] = runner.file_hash(root / "cases.jsonl")
        manifest.update(case_count=len(policy.collected), collection_assignment_steps=steps,
                        collection_episodes=episodes, collection_seconds=runner.time.perf_counter() - started)
        runner.write_json(root / "manifest.json", manifest)
    else:
        manifest = load_batch_manifest(root, expected, expected_teacher, verified_teacher["identity"])
        cases = runner.load_cases(root, protocol)
        if any("assignments" not in case or "tasks" not in case["state"] for case in cases):
            raise ValueError("batch cases required; regenerate with batch collect")
        config["llm"] = evaluation_llm_config(config["llm"], args.model,
            args.reasoning_effort, args.max_tokens, args.thinking)
        client = LLMClient(config["llm"])
        effective = deepcopy(client.service_identity)
        for key in ("sdk_version", "returned_models", "system_fingerprints"):
            effective.pop(key, None)
        evaluation_identity = dict(case_identity=manifest["identity"],
            cases_sha256=runner.file_hash(root / "cases.jsonl"), llm_config=config["llm"],
            service_settings=effective,
            evaluation_source_hashes={str(Path(__file__).resolve()): runner.file_hash(Path(__file__)),
                str(Path(__file__).with_name("dtecn_evaluation_llm.py").resolve()):
                    runner.file_hash(Path(__file__).with_name("dtecn_evaluation_llm.py")),
                str(Path(__file__).with_name("evaluate_dtecn_llm_with_csv.py").resolve()):
                    runner.file_hash(Path(__file__).with_name("evaluate_dtecn_llm_with_csv.py"))})
        digest = hashlib.sha256(json.dumps(evaluation_identity, sort_keys=True).encode()).hexdigest()[:16]
        root = root / "evaluations" / digest
        root.mkdir(parents=True, exist_ok=True)
        settings = root / "evaluation_config.json"
        if settings.exists():
            if json.loads(settings.read_text(encoding="utf-8")) != evaluation_identity:
                raise ValueError("evaluation output identity collision")
        else:
            runner.write_json(settings, evaluation_identity)
        indices = np.linspace(0, len(cases) - 1, min(config["llm"]["initial_example_count"], len(cases)), dtype=int)
        split = "validation" if args.stage == "validate" else "final_test"
        output = root / f"icl_{split}.json"
        trace = root / f"icl_{split}_decisions.jsonl"
        batch_trace = root / f"icl_{split}_batches.jsonl"
        if any(path.exists() for path in (output, output.with_suffix(".csv"), trace, trace.with_suffix(".csv"), batch_trace)):
            raise FileExistsError("preserve completed or partial batch evaluation; inspect before restarting")
        with batch_trace.open("x", encoding="utf-8") as stream:
            def batch_callback(case):
                stream.write(json.dumps(case, allow_nan=False) + "\n")
                stream.flush()
            policy = BatchPolicy(config, client=client, initial_cases=[cases[i] for i in indices], batch_callback=batch_callback)
            result = evaluate_with_csv(protocol, policy, split, trace)
        result.update(identity=evaluation_identity, cases_sha256=evaluation_identity["cases_sha256"],
                      checkpoint_sha256=expected["teacher_checkpoint_sha256"], service_identity=client.service_identity,
                      preparation_cost={k: manifest[k] for k in (
                          "training_steps", "training_episodes", "training_seconds", "actor_updates", "critic_updates",
                          "validation_count", "validation_assignment_steps", "trainable_actor_parameters",
                          "trainable_critic_parameters", "collection_seconds", "collection_assignment_steps", "collection_episodes")})
        runner.write_json(output, result)
        rows = [dict(scenario=scenario, **value["aggregate"]) for scenario, value in result["scenario_results"].items()]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with output.with_suffix(".csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()} for row in rows)
    print(json.dumps(dict(stage=args.stage, output=str(root), decision_mode="fcfs_joint_batch_v1")))


if __name__ == "__main__":
    main()
