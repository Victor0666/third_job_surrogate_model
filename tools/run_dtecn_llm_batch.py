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
from algorithms.comparisons.dtecn_llm.policy import LLMClient
from tools.dtecn_batch_policy import BatchPolicy, BATCH_PROMPT
from tools.evaluate_dtecn_llm_with_csv import evaluate_with_csv, load_compatible_manifest


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
    parser.add_argument("--deadline-cache", action="append")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for value, key in ((args.model, "model"), (args.reasoning_effort, "reasoning_effort"), (args.max_tokens, "max_tokens")):
        if value is not None:
            config["llm"][key] = value
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
    expected_teacher = runner.identity(protocol, config, args.optimizer_seed, False)
    verified_teacher = load_compatible_manifest(teacher, expected_teacher)
    expected = deepcopy(expected_teacher)
    expected.update(decision_mode="fcfs_joint_batch_v1",
        teacher_checkpoint_sha256=runner.file_hash(teacher / "maddpg.pt"),
        teacher_identity=verified_teacher["identity"],
        prompt_sha256=hashlib.sha256(BATCH_PROMPT.encode()).hexdigest(),
        batch_source_hashes={str(Path(__file__).resolve()): runner.file_hash(Path(__file__)),
                            str(Path(__file__).with_name("dtecn_batch_policy.py").resolve()):
                                runner.file_hash(Path(__file__).with_name("dtecn_batch_policy.py")),
                            str(Path(__file__).with_name("evaluate_dtecn_llm_with_csv.py").resolve()):
                                runner.file_hash(Path(__file__).with_name("evaluate_dtecn_llm_with_csv.py"))})
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
        manifest = runner.load_manifest(root, expected)
        cases = runner.load_cases(root, protocol)
        if any("assignments" not in case or "tasks" not in case["state"] for case in cases):
            raise ValueError("batch cases required; regenerate with batch collect")
        indices = np.linspace(0, len(cases) - 1, min(config["llm"]["initial_example_count"], len(cases)), dtype=int)
        split = "validation" if args.stage == "validate" else "final_test"
        output = root / f"icl_{split}.json"
        trace = root / f"icl_{split}_decisions.jsonl"
        batch_trace = root / f"icl_{split}_batches.jsonl"
        if any(path.exists() for path in (output, output.with_suffix(".csv"), trace, trace.with_suffix(".csv"), batch_trace)):
            raise FileExistsError("preserve completed or partial batch evaluation; inspect before restarting")
        client = LLMClient(config["llm"])
        with batch_trace.open("x", encoding="utf-8") as stream:
            def batch_callback(case):
                stream.write(json.dumps(case, allow_nan=False) + "\n")
                stream.flush()
            policy = BatchPolicy(config, client=client, initial_cases=[cases[i] for i in indices], batch_callback=batch_callback)
            result = evaluate_with_csv(protocol, policy, split, trace)
        result.update(identity=expected, cases_sha256=runner.file_hash(root / "cases.jsonl"),
                      checkpoint_sha256=runner.file_hash(root / "maddpg.pt"), service_identity=client.service_identity,
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
