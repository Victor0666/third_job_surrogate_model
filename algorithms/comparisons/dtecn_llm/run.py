"""Train MADDPG, collect training cases, validate/evaluate online ICL separately."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from algorithms.comparisons.fuzzy_common.evaluation import run_episode, aggregate_paper_final_metrics
from algorithms.comparisons.fuzzy_common.protocol import protocol_from_config
from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
from algorithms.llm_safe_hrl.hrl_mix.model_selection import aggregate_seed_feasibility_metrics, FeasibilityFirstModelMetrics
from algorithms.llm_safe_hrl.hrl_mix.safe_metrics import aggregate_safe_metric_records
from algorithms.llm_safe_hrl.hrl_mix.train_config import parse_deadline_cache_overrides, validate_single_deadline_cache_paths
from .environment import DtecnEnv
from .maddpg import MADDPG
from .policy import DtecnPolicy, LLMClient, PROMPT

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("config.json")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(protocol, scenario, seed):
    return DtecnEnv(**protocol.for_scenario(scenario).environment_kwargs(seed))


def identity(protocol, config, optimizer_seed, smoke):
    files = {str(p.resolve()): file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    shared = [ROOT / "common", ROOT / "algorithms/llm_safe_hrl/base"]
    for directory in shared:
        for path in directory.rglob("*.py"):
            files[str(path.resolve())] = file_hash(path)
    for relative in ("algorithms/llm_safe_hrl/scenario_registry.py",
                     "algorithms/llm_safe_hrl/hrl_mix/train_config.py",
                     "algorithms/llm_safe_hrl/hrl_mix/safe_metrics.py",
                     "algorithms/llm_safe_hrl/hrl_mix/model_selection.py",
                     "algorithms/comparisons/fuzzy_common/protocol.py",
                     "algorithms/comparisons/fuzzy_common/environment.py",
                     "algorithms/comparisons/fuzzy_common/evaluation.py",
                     "algorithms/llm_safe_hrl/LLM/utils/utils.py"):
        path = ROOT / relative
        files[str(path.resolve())] = file_hash(path)
    for scenario in set(protocol.training_scenarios + protocol.test_scenarios):
        kwargs = protocol.for_scenario(scenario).environment_kwargs(protocol.train_seeds[0])
        for path in list(kwargs["dax_paths"]) + [kwargs["deadline_cache_path"]]:
            files[str(Path(path).resolve())] = file_hash(path)
    result = dict(schema=3, method="dtecn_llm", protocol=protocol.to_manifest(),
                config=config, optimizer_seed=optimizer_seed, smoke=smoke,
                source_and_input_hashes=files, prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest())
    return json.loads(json.dumps(result, allow_nan=False))


def load_manifest(root, expected):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("identity", {}).get("schema") != 3 or manifest["identity"] != expected:
        raise ValueError("artifact protocol/config/source/input identity mismatch")
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or file_hash(root / name) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    return manifest


def episode(protocol, scenario, seed, policy, training=False):
    env = environment(protocol, scenario, seed)
    record = run_episode(env, policy, seed=seed, training=training)
    record["decision_policy"] = "online_icl" if policy.client is not None else "maddpg_case_generator"
    record.update(policy.stats)
    record["fallback_rate"] = policy.stats["fallbacks"] / max(1, record["assignment_steps"])
    return record


def evaluate(protocol, policy, split, trace_path=None):
    seeds = protocol.validation_seeds if split == "validation" else protocol.test_seeds
    scenarios = protocol.training_scenarios if split == "validation" else protocol.test_scenarios
    result = {}
    for scenario in scenarios:
        rows = []
        for seed in seeds:
            rows.append(episode(protocol, scenario, seed, policy))
            if trace_path is not None:
                with trace_path.open("a", encoding="utf-8") as trace:
                    for decision in policy.decision_records:
                        trace.write(json.dumps(dict(seed=seed, scenario=scenario, **decision), allow_nan=False) + "\n")
        aggregate = aggregate_safe_metric_records(rows)
        aggregate.update(aggregate_paper_final_metrics(rows))
        for field in policy.stats:
            aggregate[field] = sum(row[field] for row in rows)
        aggregate["fallback_rate"] = aggregate["fallbacks"] / max(1, sum(r["assignment_steps"] for r in rows))
        aggregate["mean_request_seconds"] = aggregate["llm_seconds"] / max(1, aggregate["requests"])
        aggregate["token_usage_complete"] = aggregate["usage_missing"] == 0
        aggregate["total_tokens"] = aggregate["reported_tokens"] if aggregate["token_usage_complete"] else None
        result[scenario] = dict(seed_records=rows, aggregate=aggregate)
    records = [r for value in result.values() for r in value["seed_records"]]
    selection = FeasibilityFirstModelMetrics.from_mapping(aggregate_seed_feasibility_metrics(records))
    return dict(split=split, scenario_results=result, selection=selection.to_dict(),
                used_for_selection=(split == "validation"), cross_seed_case_updates=False)


def train(protocol, config, agent, root, expected):
    policy = DtecnPolicy(config, agent=agent)
    steps, episode_index, next_validation = 0, 0, config["training"]["validation_interval"]
    validation_count, validation_steps = 0, 0
    incumbent = None
    started = time.perf_counter()
    with (root / "training_metrics.jsonl").open("w", encoding="utf-8") as log:
        while steps < config["training"]["time_steps"]:
            scenario = protocol.training_scenarios[episode_index % len(protocol.training_scenarios)]
            seed = protocol.train_seeds[(episode_index // len(protocol.training_scenarios)) % len(protocol.train_seeds)]
            record = episode(protocol, scenario, seed, policy, training=True)
            episode_index += 1
            steps += record["assignment_steps"]
            log.write(json.dumps(dict(episode=episode_index, steps=steps, **record), allow_nan=False) + "\n")
            if steps < next_validation and steps < config["training"]["time_steps"]:
                continue
            validation = evaluate(protocol, policy, "validation")
            validation_count += 1
            validation_steps += sum(row["assignment_steps"] for value in validation["scenario_results"].values() for row in value["seed_records"])
            key = tuple(validation["selection"]["comparison_key"])
            with (root / "validation_metrics.jsonl").open("a", encoding="utf-8") as v:
                v.write(json.dumps(dict(steps=steps, **validation), allow_nan=False) + "\n")
            if incumbent is None or key < incumbent:
                incumbent = key
                agent.save(root / "maddpg.pt")
                write_json(root / "best_validation.json", validation)
            next_validation = (steps // config["training"]["validation_interval"] + 1) * config["training"]["validation_interval"]
    manifest = dict(identity=expected, artifacts={"maddpg.pt": file_hash(root / "maddpg.pt")},
                    training_steps=steps, training_episodes=episode_index,
                    actor_updates=agent.actor_updates, critic_updates=agent.critic_updates,
                    validation_count=validation_count, validation_assignment_steps=validation_steps,
                    trainable_actor_parameters=sum(p.numel() for p in agent.actors.parameters()),
                    trainable_critic_parameters=sum(p.numel() for p in agent.critics.parameters()),
                    training_seconds=time.perf_counter() - started, final_test_used=False)
    write_json(root / "manifest.json", manifest)


def collect(protocol, config, agent, root, manifest):
    started = time.perf_counter()
    policy = DtecnPolicy(config, agent=agent, collect=True)
    collection_steps, collection_episodes = 0, 0
    for scenario in protocol.training_scenarios:
        for seed in protocol.train_seeds:
            protocol.assert_not_test_seed(seed, "case collection")
            record = episode(protocol, scenario, seed, policy)
            collection_steps += record["assignment_steps"]
            collection_episodes += 1
    with (root / "cases.jsonl").open("w", encoding="utf-8") as out:
        for case in policy.collected:
            out.write(json.dumps(case, allow_nan=False) + "\n")
    manifest["artifacts"]["cases.jsonl"] = file_hash(root / "cases.jsonl")
    manifest["case_count"] = len(policy.collected)
    manifest["collection_seconds"] = time.perf_counter() - started
    manifest["collection_assignment_steps"] = collection_steps
    manifest["collection_episodes"] = collection_episodes
    write_json(root / "manifest.json", manifest)


def load_cases(root, protocol):
    cases = [json.loads(line) for line in (root / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    if not cases:
        raise ValueError("empty initial case library")
    for case in cases:
        if case["seed"] not in protocol.train_seeds or case["scenario"] not in protocol.training_scenarios:
            raise ValueError("case seed/scenario leakage")
        protocol.assert_not_test_seed(case["seed"], "initial cases")
    # Fixed evenly spaced training cases cover the collection without reward selection.
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "collect", "validate", "evaluate", "smoke"))
    parser.add_argument("--protocol", choices=("single", "multi"), default="single")
    parser.add_argument("--scenario", default="SS")
    parser.add_argument("--resource-scale", choices=("S", "M", "L"))
    parser.add_argument("--ddl", default="T")
    parser.add_argument("--deadline-cache", action="append")
    parser.add_argument("--optimizer-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--model", help="Exact API model ID; use the same ID for LLM-Safe-HRL")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--max-tokens", type=int, help="Total output budget, including reasoning for thinking models")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != 3:
        raise ValueError("config schema 3 required for 512/256 networks; old artifacts must be retrained")
    for argument, key in ((args.model, "model"), (args.reasoning_effort, "reasoning_effort"), (args.max_tokens, "max_tokens")):
        if argument is not None:
            config["llm"][key] = argument
    if config["llm"]["reasoning_effort"] not in ("low", "medium", "high", "xhigh", "max"):
        raise ValueError("invalid reasoning effort")
    if type(config["llm"]["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be boolean")
    for group in ("training", "maddpg", "objective", "llm"):
        for name, value in config[group].items():
            if not isinstance(value, bool) and isinstance(value, (int, float)) and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"invalid positive finite setting {group}.{name}")
    if not 0 < config["maddpg"]["gamma"] <= 1 or not 0 < config["maddpg"]["tau"] <= 1:
        raise ValueError("gamma/tau outside (0,1]")
    for group, names in (("training", ("time_steps", "validation_interval")),
                         ("maddpg", ("batch_size", "buffer_size")),
                         ("llm", ("max_tokens", "prompt_char_budget", "initial_example_count", "online_example_count"))):
        if any(type(config[group][name]) is not int for name in names):
            raise ValueError(f"{group} counts must be positive integers")
    if config["maddpg"]["buffer_size"] < config["maddpg"]["batch_size"]:
        raise ValueError("replay capacity cannot be smaller than batch size")
    if not 0 < config["llm"]["top_p"] <= 1 or not 0 < config["maddpg"]["epsilon"] <= 1:
        raise ValueError("top_p/epsilon outside (0,1]")
    context = resolve_experiment_protocol(args.protocol, source_scenario=args.scenario if args.protocol == "single" else None,
                                          resource_scale=args.resource_scale)
    caches = parse_deadline_cache_overrides(args.deadline_cache, default_scenario=context.source_scenario)
    if caches:
        caches = validate_single_deadline_cache_paths(args.protocol, caches, source_scenario=context.source_scenario,
                                                       required_scenarios=context.test_scenarios)
    protocol = protocol_from_config({}, scenario=context.training_scenarios[0], ddl=args.ddl,
                                    experiment_context=context, deadline_cache_paths=caches)
    smoke = args.stage == "smoke"
    if smoke:
        # Small mixed-workload episodes do not match the formal 50-workflow DDL cache.
        protocol = replace(protocol, workflows_per_episode=1, train_seeds=(1,), validation_seeds=(101,),
                           test_seeds=(201,), test_scenarios=(protocol.scenario,))
        config["training"] = dict(time_steps=1, validation_interval=1)
        config["maddpg"]["batch_size"] = 2
    torch.manual_seed(args.optimizer_seed)
    np.random.seed(args.optimizer_seed)
    random.seed(args.optimizer_seed)
    if args.device == "cpu":
        torch.set_num_threads(1)
    root = ROOT / "out/comparisons/dtecn_llm/combined_v1/artifacts_v3" / args.protocol / (context.source_scenario or context.resource_scale) / protocol.ddl_setting.lower() / f"s{args.optimizer_seed}"
    if caches:
        digest = hashlib.sha256(json.dumps(caches, sort_keys=True).encode()).hexdigest()[:10]
        root = root / digest
    if smoke:
        root = root / "smoke"
    if any(v is not None for v in (args.model, args.reasoning_effort, args.max_tokens)):
        root = root / hashlib.sha256(json.dumps(config["llm"], sort_keys=True).encode()).hexdigest()[:10]
    root.mkdir(parents=True, exist_ok=True)
    expected = identity(protocol, config, args.optimizer_seed, smoke)
    agent = None
    if args.stage in ("train", "collect", "smoke"):
        prototype = environment(protocol, protocol.scenario, protocol.train_seeds[0])
        agent = MADDPG(protocol.workflows_per_episode, prototype.observation_dim, prototype.num_vms, config["maddpg"], args.device)
    if args.stage in ("train", "smoke"):
        if (root / "manifest.json").exists():
            raise FileExistsError("preserve existing run; choose a different optimizer seed or archive it explicitly")
        train(protocol, config, agent, root, expected)
    manifest = load_manifest(root, expected)
    if agent is not None:
        agent.load(root / "maddpg.pt")
    if args.stage in ("collect", "smoke"):
        if "cases.jsonl" in manifest["artifacts"]:
            raise FileExistsError("case library already frozen")
        collect(protocol, config, agent, root, manifest)
    if args.stage in ("validate", "evaluate", "smoke"):
        cases = load_cases(root, protocol)
        limit = config["llm"]["initial_example_count"]
        indices = np.linspace(0, len(cases) - 1, min(limit, len(cases)), dtype=int)
        initial = [cases[i] for i in indices]
        if smoke:
            def client(prompt):
                state = json.loads(prompt.split("\nCurrent state: ", 1)[1])
                return json.dumps({"resource_preferences": [0.5] * len(state["vm_ids"])}), None
        else:
            client = LLMClient(config["llm"])
        policy = DtecnPolicy(config, client=client, initial_cases=initial)
        split = "validation" if args.stage == "validate" else "final_test"
        output = root / f"icl_{split}.json"
        if output.exists():
            raise FileExistsError("preserve completed ICL evaluation")
        trace_path = root / f"icl_{split}_decisions.jsonl"
        if trace_path.exists():
            raise FileExistsError("preserve partial evaluation trace; inspect it before restarting")
        result = evaluate(protocol, policy, split, trace_path)
        result.update(identity=expected, cases_sha256=file_hash(root / "cases.jsonl"),
                      checkpoint_sha256=file_hash(root / "maddpg.pt"), mock_llm=smoke,
                      service_identity=getattr(client, "service_identity", {"mock": True}),
                      preparation_cost={k: manifest[k] for k in (
                          "training_steps", "training_episodes", "training_seconds", "actor_updates", "critic_updates",
                          "validation_count", "validation_assignment_steps", "trainable_actor_parameters",
                          "trainable_critic_parameters", "collection_seconds", "collection_assignment_steps", "collection_episodes")})
        write_json(output, result)
        rows = [dict(scenario=scenario, **value["aggregate"]) for scenario, value in result["scenario_results"].items()]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()} for row in rows)
    print(json.dumps(dict(stage=args.stage, output=str(root), smoke=smoke), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
