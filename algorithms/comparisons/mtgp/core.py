"""Routing before transfer, sequencing after arrival; no RL or LLM policies."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import time

import numpy as np

from algorithms.comparisons.drlea_nichgp.gp_primitives import (
    GPProgram, OPERATORS, program_depth,
)
from algorithms.comparisons.fuzzy_common.evaluation import make_environment
from algorithms.llm_safe_hrl.hrl_mix.safe_metrics import (
    build_episode_metric_record, aggregate_safe_metric_records,
)
from algorithms.llm_safe_hrl.hrl_mix.model_selection import (
    aggregate_seed_feasibility_metrics, MODEL_COMPARISON_FIELDS,
)


TERMINALS = (
    "NIQ", "WIQ", "MRT", "IN_COMM", "OUT_COMM", "PT", "TTIQ",
    "TIS", "TWT", "NTR", "TIME_TO_DDL", "DELTA_FUZZY_ENERGY",
)
TREE_SCHEMA = "mtgp12_v1"


def evaluate_tree(program, terminals):
    """Use the author's exact-zero protected division, without feature clipping."""
    values = dict(zip(TERMINALS, terminals))
    stack = []
    for token in reversed(program.tokens):
        if token not in OPERATORS:
            stack.append(float(values[token]))
            continue
        left, right = stack.pop(), stack.pop()
        if token == "add":
            value = left + right
        elif token == "sub":
            value = left - right
        elif token == "mul":
            value = left * right
        elif token == "pdiv":
            value = 1.0 if right == 0.0 else left / right
        elif token == "min":
            value = min(left, right)
        else:
            value = max(left, right)
        if not math.isfinite(value):
            raise ValueError("non-finite MTGP expression")
        stack.append(value)
    if len(stack) != 1 or not math.isfinite(stack[0]):
        raise ValueError("invalid MTGP expression")
    return stack[0]


@lru_cache(maxsize=1024)
def _instructions(program):
    return tuple(TERMINALS.index(token) if token in TERMINALS else token
                 for token in reversed(program.tokens))


def evaluate_tree_batch(program, matrix):
    """Elementwise float64 operations in exactly the scalar tree's order."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(TERMINALS) or not np.isfinite(matrix).all():
        raise ValueError("non-finite MTGP terminal matrix")
    stack = []
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for token in _instructions(program):
            if isinstance(token, int):
                stack.append(matrix[:, token])
                continue
            left, right = stack.pop(), stack.pop()
            if token == "add":
                value = left + right
            elif token == "sub":
                value = left - right
            elif token == "mul":
                value = left * right
            elif token == "pdiv":
                value = np.ones_like(left)
                np.divide(left, right, out=value, where=right != 0.0)
            elif token == "min":
                value = np.where(left <= right, left, right)
            elif token == "max":
                value = np.where(left >= right, left, right)
            else:
                raise ValueError("invalid MTGP expression")
            if not np.isfinite(value).all():
                raise ValueError("non-finite MTGP expression")
            stack.append(value)
    if len(stack) != 1:
        raise ValueError("invalid MTGP expression")
    return stack[0]


@dataclass(frozen=True)
class RulePair:
    sequencing: GPProgram
    routing: GPProgram

    def to_dict(self):
        return {"sequencing": self.sequencing.to_dict(), "routing": self.routing.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if any(len(value[name]["tokens"]) > 255 for name in ("sequencing", "routing")):
            raise ValueError("MTGP tree exceeds maximum node count")
        trees = [GPProgram.from_dict(value[name]) for name in ("sequencing", "routing")]
        for tree in trees:
            if tree.version != TREE_SCHEMA or tree.terminal_names != TERMINALS:
                raise ValueError("MTGP terminal schema mismatch")
            if program_depth(tree) + 1 > 8:
                raise ValueError("MTGP tree exceeds maximum depth")
        return cls(*trees)


def task_terminals(env, task_id, vm_id):
    index = env.vm_ids.index(vm_id)
    queue = env.vm_waiting_queues[index]
    components = env.estimate_task_duration_components_scenario(task_id, vm_id, "modal")
    queued = [env.estimate_task_duration_components_scenario(task, vm_id, "modal") for task in queue]
    workflow_id, _ = env.task_meta[task_id]
    workflow = env.workflows[workflow_id]
    transfer = env._input_transfers.get(task_id)
    wait = 0.0 if transfer is None else max(0.0, env.current_time - transfer["arrivals"]["modal"])
    values = (
        len(queue), sum(row["execution_time"] for row in queued), env.vm_available_at[index],
        components["input_communication_time"], components["output_communication_time"],
        components["execution_time"], sum(row["total_duration"] for row in queued),
        env.current_time - workflow.arrival_time, wait, env.wf_remaining_tasks[workflow_id],
        workflow.deadline - env.current_time, env.estimate_incremental_energy_score(task_id, vm_id),
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("non-finite MTGP terminal")
    return values


def terminal_matrix(env, tasks, vms, program):
    """Only used terminals; queue totals are shared within one decision."""
    needed = frozenset(token for token in program.tokens if token in TERMINALS)
    rows = np.zeros((len(tasks), len(TERMINALS)), dtype=np.float64)
    queue_totals = {}
    for row, (task, vm) in enumerate(zip(tasks, vms)):
        index = env._vm_id_and_index(vm)[1]
        workflow_id, _ = env.task_meta[task]
        workflow = env.workflows[workflow_id]
        values = {}
        if needed.intersection(("NIQ", "WIQ", "TTIQ")):
            if vm not in queue_totals:
                queue = env.vm_waiting_queues[index]
                queued = [env._task_duration_components_ref(item, vm, "modal") for item in queue]
                # Keep the original summation order, including floating-point rounding.
                queue_totals[vm] = (len(queue), sum(item["execution_time"] for item in queued),
                                    sum(item["total_duration"] for item in queued))
            values.update(zip(("NIQ", "WIQ", "TTIQ"), queue_totals[vm]))
        if needed.intersection(("IN_COMM", "OUT_COMM", "PT")):
            components = env._task_duration_components_ref(task, vm, "modal")
            values.update(IN_COMM=components["input_communication_time"],
                          OUT_COMM=components["output_communication_time"], PT=components["execution_time"])
        if "MRT" in needed:
            values["MRT"] = env.vm_available_at[index]
        if "TIS" in needed:
            values["TIS"] = env.current_time - workflow.arrival_time
        if "TWT" in needed:
            transfer = env._input_transfers.get(task)
            values["TWT"] = 0.0 if transfer is None else max(0.0, env.current_time - transfer["arrivals"]["modal"])
        if "NTR" in needed:
            values["NTR"] = env.wf_remaining_tasks[workflow_id]
        if "TIME_TO_DDL" in needed:
            values["TIME_TO_DDL"] = workflow.deadline - env.current_time
        if "DELTA_FUZZY_ENERGY" in needed:
            values["DELTA_FUZZY_ENERGY"] = env.estimate_incremental_energy_score(task, vm)
        for name in needed:
            rows[row, TERMINALS.index(name)] = values[name]
    if not np.isfinite(rows).all():
        raise ValueError("non-finite MTGP terminal")
    return rows


def choose_task(env, pair, vm_id, queue):
    scores = evaluate_tree_batch(pair.sequencing, terminal_matrix(env, queue, [vm_id] * len(queue), pair.sequencing))
    index = min(range(len(queue)), key=lambda index: (
        scores[index], env.workflows[env.task_meta[queue[index]][0]].arrival_time,
        env.task_meta[queue[index]], queue[index],
    ))
    return queue[index]


def run_episode(protocol, pair, seed):
    started = time.perf_counter()
    env = make_environment(protocol, seed)
    env.reset(seed=seed)
    env.vm_queue_selector = lambda vm, queue: choose_task(env, pair, vm, queue)
    decisions = 0
    while not env.done_flag:
        ready = sorted(env.get_ready_tasks(), key=lambda task: (env.task_ready_time[task], env.task_meta[task], task))
        for task in ready:
            feasible = env.get_feasible_vms(task)
            if not feasible:
                raise ValueError("MTGP task has no feasible VM")
            scores = evaluate_tree_batch(pair.routing, terminal_matrix(env, [task] * len(feasible), feasible, pair.routing))
            vm = feasible[min(range(len(feasible)), key=lambda index: (scores[index], feasible[index]))]
            env.route_task_to_vm(task, vm)
            decisions += 1
        env.advance_to_next_resource_event(compute_energy_reward=False)
        if decisions > 1_000_000:
            raise RuntimeError("MTGP exceeded the assignment limit")
    record = build_episode_metric_record(
        env, seed=seed, scheduling_time_seconds=time.perf_counter() - started,
    )
    if not record["evaluation_completed"]:
        raise ValueError("MTGP episode did not complete every expected workflow")
    record.update(comparison_method_id="mtgp", assignment_steps=decisions, scenario=protocol.scenario)
    return record


def evaluate(protocol, pair, scenarios, seeds):
    records = [run_episode(protocol.for_scenario(scenario), pair, seed)
               for scenario in scenarios for seed in seeds]
    aggregate = aggregate_safe_metric_records(records)
    aggregate.update(aggregate_seed_feasibility_metrics(records))
    aggregate["comparison_key"] = [aggregate[name] for name in MODEL_COMPARISON_FIELDS]
    return aggregate, records


def random_tree(rng, depth, full):
    def grow(level):
        if level >= depth or (not full and rng.random() < len(TERMINALS) / (len(TERMINALS) + len(OPERATORS))):
            return [str(rng.choice(TERMINALS))]
        return [str(rng.choice(OPERATORS)), *grow(level + 1), *grow(level + 1)]
    return GPProgram(tuple(grow(1)), TERMINALS, TREE_SCHEMA)


def _node(rng, tree):
    internal = [index for index, token in enumerate(tree.tokens) if token in OPERATORS]
    leaves = [index for index, token in enumerate(tree.tokens) if token not in OPERATORS]
    candidates = internal if rng.random() < 0.9 else leaves
    return int(rng.choice(candidates or leaves or internal))


def _subtree_end(tokens, start):
    needed, index = 1, start
    while needed:
        needed += 1 if tokens[index] in OPERATORS else -1
        index += 1
    return index


def _replace(tree, start, replacement):
    tokens = tree.tokens[:start] + replacement + tree.tokens[_subtree_end(tree.tokens, start):]
    child = GPProgram(tokens, TERMINALS, TREE_SCHEMA)
    return child if program_depth(child) + 1 <= 8 else tree


def crossover(first, second, rng):
    """Subtree crossover in one tree, whole-tree exchange in the other."""
    a, b = [first.sequencing, first.routing], [second.sequencing, second.routing]
    index = int(rng.integers(2))
    left, right = a[index], b[index]
    p, q = _node(rng, left), _node(rng, right)
    a[index] = _replace(left, p, right.tokens[q:_subtree_end(right.tokens, q)])
    b[index] = _replace(right, q, left.tokens[p:_subtree_end(left.tokens, p)])
    a[1 - index], b[1 - index] = b[1 - index], a[1 - index]
    return RulePair(*a), RulePair(*b)


def mutate(pair, rng):
    trees = [pair.sequencing, pair.routing]
    index = int(rng.integers(2))
    trees[index] = _replace(trees[index], _node(rng, trees[index]), random_tree(rng, 4, False).tokens)
    return RulePair(*trees)


def _fitness_cache_key(pair, scenario, seed):
    return (scenario, int(seed), pair.sequencing.tokens, pair.routing.tokens)


def train(protocol, *, algorithm_seed=0, population_size=1000, generations=51, elite_size=10,
          workers=1, threads_per_worker=1, checkpoint_path=None, resume=False, identity=None):
    from .parallel import EvaluationPool
    from .checkpoint import SCHEMA_VERSION, write_checkpoint, read_checkpoint
    if population_size < 2 or generations < 1 or not 0 < elite_size < population_size or algorithm_seed < 0:
        raise ValueError("invalid MTGP population/generation/elite/seed settings")
    if resume and checkpoint_path is None:
        raise ValueError("resume requires a checkpoint path")
    if identity is None:
        from .run_pipeline import input_identity
        identity = input_identity(protocol)
    settings = dict(algorithm_seed=algorithm_seed, population_size=population_size,
                    generations=generations, elite_size=elite_size)
    rng = np.random.default_rng(algorithm_seed)
    cache, archive, history = {}, {}, []
    evaluation_count = invalid_count = simulated_count = cache_hits = next_generation = 0
    if resume:
        state = read_checkpoint(checkpoint_path, identity, settings)
        population = [RulePair.from_dict(value) for value in state["population"]]
        for value in state["archive"]:
            pair = RulePair.from_dict(value)
            archive[(pair.sequencing.tokens, pair.routing.tokens)] = pair
        for row in state["fitness_cache"]:
            if row["scenario"] not in protocol.training_scenarios or row["seed"] not in protocol.train_seeds:
                raise ValueError("MTGP cache contains a non-training instance")
            pair = RulePair.from_dict({name: {"tokens": row[name], "terminal_names": TERMINALS, "version": TREE_SCHEMA}
                                      for name in ("sequencing", "routing")})
            cache[_fitness_cache_key(pair, row["scenario"], row["seed"])] = None if row["fitness"] is None else tuple(row["fitness"])
        rng.bit_generator.state = state["rng_state"]
        history, next_generation = state["history"], state["next_generation"]
        counts = state["counts"]
        evaluation_count, invalid_count = counts["training_episode_count"], counts["invalid_individual_count"]
        simulated_count, cache_hits = counts["simulated_training_episode_count"], counts["training_cache_hits"]
        if any(not isinstance(value, int) or value < 0 for value in (evaluation_count, invalid_count, simulated_count, cache_hits)):
            raise ValueError("invalid MTGP checkpoint counters")
        if evaluation_count != next_generation * population_size or simulated_count + cache_hits != evaluation_count:
            raise ValueError("inconsistent MTGP checkpoint counters")
    else:
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            raise FileExistsError("MTGP checkpoint already exists; use --resume or a new output directory")
        population = [RulePair(*(random_tree(rng, int(rng.integers(2, 7)), i % 2 == 0)
                                 for _ in range(2))) for i in range(population_size)]
    instances = [(scenario, seed) for scenario in protocol.training_scenarios for seed in protocol.train_seeds]
    with EvaluationPool(workers, threads_per_worker) as pool:
        for generation in range(next_generation, generations):
            scenario, seed = instances[generation % len(instances)]
            view = protocol.for_scenario(scenario)
            keys = [_fitness_cache_key(pair, scenario, seed) for pair in population]
            missing = {}
            for key, pair in zip(keys, population):
                if key not in cache:
                    missing.setdefault(key, pair)
            records = pool.evaluate([(view, pair, seed) for pair in missing.values()])
            for key, record in zip(missing, records):
                cache[key] = None if record is None else tuple(record[name] for name in MODEL_COMPARISON_FIELDS)
            simulated_count += len(missing)
            cache_hits += population_size - len(missing)
            fitness = [(math.inf,) * 4 if cache[key] is None else cache[key] for key in keys]
            invalid_count += sum(cache[key] is None for key in keys)
            evaluation_count += population_size
            ranking = sorted(range(population_size), key=lambda index: fitness[index])
            best = population[ranking[0]]
            if not all(math.isfinite(value) for value in fitness[ranking[0]]):
                raise ValueError("all MTGP individuals are invalid")
            archive[(best.sequencing.tokens, best.routing.tokens)] = best
            history.append({"generation": generation, "scenario": scenario, "seed": seed,
                            "comparison_key": list(fitness[ranking[0]]), "training_episode_count": evaluation_count,
                            "simulated_training_episode_count": simulated_count, "training_cache_hits": cache_hits})
            print(f"MTGP generation {generation + 1}/{generations}: {fitness[ranking[0]]} "
                  f"(simulated={len(missing)}, cache_hits={population_size - len(missing)})", flush=True)
            if generation + 1 < generations:
                def tournament():
                    indices = rng.integers(population_size, size=7)
                    return population[min(indices, key=lambda index: fitness[index])]
                children = [population[index] for index in ranking[:elite_size]]
                while len(children) < population_size:
                    probability = rng.random()
                    if probability < 0.8:
                        children.extend(crossover(tournament(), tournament(), rng))
                    elif probability < 0.95:
                        children.append(mutate(tournament(), rng))
                    else:
                        children.append(tournament())
                population = children[:population_size]
            if checkpoint_path is not None:
                write_checkpoint(checkpoint_path, {
                    "schema_version": SCHEMA_VERSION, "identity": identity, "settings": settings,
                    "next_generation": generation + 1, "rng_state": rng.bit_generator.state,
                    "population": [pair.to_dict() for pair in population],
                    "archive": [pair.to_dict() for pair in archive.values()], "history": history,
                    "fitness_cache": [{"scenario": key[0], "seed": key[1], "sequencing": key[2],
                                       "routing": key[3], "fitness": value} for key, value in cache.items()],
                    "counts": {"training_episode_count": evaluation_count, "invalid_individual_count": invalid_count,
                               "simulated_training_episode_count": simulated_count, "training_cache_hits": cache_hits},
                })
        best_pair, best_metrics = None, None
        validation_count = 0
        for pair in archive.values():
            jobs = [(protocol.for_scenario(scenario), pair, seed) for scenario in protocol.training_scenarios
                    for seed in protocol.validation_seeds]
            records = pool.evaluate(jobs)
            validation_count += len(jobs)
            invalid_count += sum(record is None for record in records)
            if any(record is None for record in records):
                continue
            metrics = aggregate_safe_metric_records(records)
            metrics.update(aggregate_seed_feasibility_metrics(records))
            metrics["comparison_key"] = [metrics[name] for name in MODEL_COMPARISON_FIELDS]
            if best_metrics is None or metrics["comparison_key"] < best_metrics["comparison_key"]:
                best_pair, best_metrics = pair, metrics
    if best_pair is None:
        raise ValueError("no MTGP candidate is valid on every validation instance")
    return best_pair, best_metrics, history, {
        "training_episode_count": evaluation_count, "validation_episode_count": validation_count,
        "invalid_individual_count": invalid_count, "simulated_training_episode_count": simulated_count,
        "training_cache_hits": cache_hits, "workers": workers, "threads_per_worker": threads_per_worker,
    }
