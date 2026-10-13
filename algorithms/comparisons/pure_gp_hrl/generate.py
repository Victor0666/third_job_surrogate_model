"""Evolve pure GP Top-K rules using direct simulation, without parameter optimization."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .core import PureGPProgram as GPProgram, random_program, crossover, mutate, frozen_source, terminals_for_objective
from algorithms.llm_safe_hrl.LLM.export_topk import _build_record, select_topk, topk_ranking_key
from algorithms.llm_safe_hrl.LLM.problems.cews_task_constructive.eval import evaluate_candidate
from algorithms.llm_safe_hrl.base.heuristic_admission import (
    file_sha256, record_sha256, evaluation_context_from_config, admission_scope_from_config,
)
from algorithms.llm_safe_hrl.base.llm_objective import (
    objective_mode, objective_identity, ENERGY_POLICY, ENERGY_RANKING_FIELDS,
)
from algorithms.llm_safe_hrl.base.manager_heuristics import load_manager_heuristic_library
from algorithms.llm_safe_hrl.base.topk_schema import (
    TOPK_MANIFEST_SCHEMA_VERSION, TOPK_SELECTION_MODE, TOPK_SELECTION_POLICY_VERSION,
)
from algorithms.llm_safe_hrl.run_context import resolve_deadline_setting
from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol, apply_scenario_to_problem_config
from hrl_mix.train_config import ROOT_DIR, validate_single_deadline_cache_paths


class EvaluationBudgetExceeded(RuntimeError):
    """Stop offline evolution when the requested simulation budget ends."""


@contextmanager
def evaluation_pool(workers):
    if workers < 1:
        raise ValueError('workers must be positive')
    if workers == 1:
        yield None
        return
    names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = '1'
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            yield executor
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def experiment_settings(path, scenario, ddl, caches, *, top_k=None, objective=None):
    path = Path(path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    mode = objective_mode(objective if objective is not None else config['objective_mode'])
    k = top_k if top_k is not None else config['top_k']
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('top-k must be a positive integer')
    context = resolve_experiment_protocol('single', source_scenario=scenario)
    deadline = resolve_deadline_setting(ddl)
    caches = validate_single_deadline_cache_paths('single', caches,
                                                source_scenario=context.source_scenario)
    config = apply_scenario_to_problem_config(config, context.source_scenario)
    for name in config['dataset']['dax_files']:
        if not (ROOT_DIR / 'data/dax' / name).is_file():
            raise FileNotFoundError(f'missing scenario DAX file: {name}')
    for cache in caches.values():
        if not Path(cache).is_file():
            raise FileNotFoundError(f'missing deadline cache: {cache}')
    config['top_k'], config['objective_mode'] = k, mode
    # The shared evaluator retains this wire key; no LLM configuration is loaded.
    config['llm_objective'] = mode
    config['experiment_protocol'] = context.identity()
    config['dataset'].update(train_seeds=list(context.llm_train_seeds),
                             validation_seeds=list(context.llm_validation_seeds),
                             test_seeds=list(context.final_test_seeds),
                             deadline_cache_path=caches[context.source_scenario])
    config['dataset'].update({key: value for key, value in deadline.identity().items()
                              if key.startswith('deadline_')})
    config['deadline_cache_paths'] = caches
    config['admission_scope'] = dict(mode='resource_task_domain',
        resource_code=context.source_scenario[1], allowed_scenarios=list(context.test_scenarios),
        workflow_families=evaluation_context_from_config(config)['workflow_families'])
    admission_scope_from_config(config)
    if mode == 'energy_only' and not config['fuzzy']['enabled']:
        raise ValueError('energy_only requires fuzzy energy evaluation')
    return context, deadline, k, mode, config

class RuleEvaluator:
    def __init__(self, directory, config, context, search_config, episode_budget=None):
        self.directory, self.config, self.context = Path(directory), config, context
        self.search_config, self.episode_budget = search_config, episode_budget
        self.counts = dict(training_episode_count=0, validation_episode_count=0,
                           attempted_episode_count=0, cache_hits=0)
        self.cache = {}

    def reserve(self, path, seeds):
        seeds = tuple(int(seed) for seed in seeds)
        if not seeds or set(seeds) & set(self.context.final_test_seeds):
            raise ValueError('offline search may not consume final-test seeds')
        if not set(seeds) <= set((*self.context.llm_train_seeds, *self.context.llm_validation_seeds)):
            raise ValueError('offline search seeds outside training/validation split')
        key = (file_sha256(path), seeds)
        if key in self.cache:
            self.counts['cache_hits'] += 1
            return key, dict(self.cache[key])
        if (self.episode_budget is not None and
                self.counts['attempted_episode_count'] + len(seeds) > self.episode_budget):
            raise EvaluationBudgetExceeded()
        self.counts['attempted_episode_count'] += len(seeds)
        return key, None

    def complete(self, key, metrics):
        seeds = key[1]
        self.counts['training_episode_count'] += len(set(seeds) & set(self.context.llm_train_seeds))
        self.counts['validation_episode_count'] += len(set(seeds) & set(self.context.llm_validation_seeds))
        self.cache[key] = dict(metrics)
        return metrics

    def exact(self, path, seeds):
        key, metrics = self.reserve(path, seeds)
        if metrics is None:
            metrics = self.complete(key, evaluate_candidate(path, self.config, key[1]))
        return metrics

    def source_record(self, program, generation, individual):
        source = frozen_source(program, self.config['llm_objective'], self.search_config, self.context)
        path = self.directory / 'generated' / f'gp_g{generation}_i{individual}.py'
        path.write_text(source, encoding='utf-8')
        return dict(source_path=path, source_hash=file_sha256(path),
                    iteration=generation, individual=individual,
                    program=program.to_dict())

    def prepare(self, program, generation, individual):
        record = self.source_record(program, generation, individual)
        metrics = self.exact(record['source_path'], self.context.llm_train_seeds)
        return dict(record, metrics=metrics, ranking_key=topk_ranking_key(metrics))

    def population(self, programs, generation, archive, executor=None, workers=1):
        pending, tokens = deque(), set()
        index, exhausted = 0, False
        while index < len(programs) or pending:
            while not exhausted and index < len(programs) and len(pending) < workers:
                program = programs[index]
                # Wait for an earlier identical candidate before deciding whether to reuse or retry it.
                if program.tokens in tokens:
                    break
                record, key, future = None, None, None
                try:
                    if program.tokens in archive:
                        record = archive[program.tokens]
                    elif executor is None:
                        record = self.prepare(program, generation, index)
                    else:
                        record = self.source_record(program, generation, index)
                        key, metrics = self.reserve(record['source_path'], self.context.llm_train_seeds)
                        if metrics is None:
                            future = executor.submit(evaluate_candidate, record['source_path'], self.config, key[1])
                        else:
                            record.update(metrics=metrics, ranking_key=topk_ranking_key(metrics))
                except EvaluationBudgetExceeded:
                    exhausted = True
                    break
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    record = exc
                pending.append((index, program, record, key, future))
                tokens.add(program.tokens)
                index += 1
            if not pending:
                if exhausted:
                    raise EvaluationBudgetExceeded()
                break
            individual, program, record, key, future = pending.popleft()
            tokens.remove(program.tokens)
            if future is not None:
                try:
                    metrics = self.complete(key, future.result())
                    record.update(metrics=metrics, ranking_key=topk_ranking_key(metrics))
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    record = exc
            if not isinstance(record, Exception):
                archive[program.tokens] = record
            yield individual, program, record
        if exhausted:
            raise EvaluationBudgetExceeded()


def export_library(directory, config_path, context, deadline, k, mode, config,
                   candidates, budget, smoke=False):
    ranked = sorted(candidates, key=lambda candidate: candidate['ranking_key'])
    selected = select_topk(ranked, k, llm_objective=mode)
    records = []
    for rank, candidate in enumerate(selected, 1):
        report = directory / 'reports' / f"{candidate['source_path'].stem}.json"
        write_json(report, candidate['metrics'])
        record = _build_record(candidate, rank=rank, run_dir=directory, report_file=report,
                               evaluation_context=evaluation_context_from_config(config),
                               admission_scope=admission_scope_from_config(config),
                               experiment_protocol=context.identity())
        record.update(heuristic_id=f"pure_gp_topk_{candidate['source_hash'][:16]}",
                      display_name=f'Pure GP Top-K #{rank}', version=f'pure_gp.topk.v1.rank{rank}',
                      generator='pure_gp', gp_generation=candidate['iteration'],
                      gp_individual=candidate['individual'], gp_program=candidate['program'],
                      cmaes_enabled=False, parameter_optimization_enabled=False)
        # shortcut: production requires this legacy wire tag, remove when its schema supports GP.
        record['source_kind'] = 'seevo_generated'
        record['record_sha256'] = record_sha256(record)
        records.append(record)
    manifest = dict(schema_version=TOPK_MANIFEST_SCHEMA_VERSION,
                    manifest_id=f'pure_gp_hrl_{directory.name}', manifest_version='pure_gp_hrl.topk.v1',
                    method_id='pure_gp_hrl', generator='pure_gp', llm_enabled=False, smoke_only=smoke,
                    cmaes_enabled=False, parameter_optimization_enabled=False,
                    compatibility_source_kind='seevo_generated',
                    selection_mode=TOPK_SELECTION_MODE,
                    selection_policy=dict(policy_version=TOPK_SELECTION_POLICY_VERSION,
                        requested_k=k, selected_count=k, unique_structure_first=True,
                        hard_energy_threshold=None, ranking_fields=[
                            'constraint_feasible', 'max_deadline_violation_rate_across_seeds',
                            'max_fuzzy_lateness', 'fuzzy_total_energy_score',
                            'objective_cv_across_seeds', 'candidate_sha256']),
                    trusted_source_root='generated', trusted_report_root='reports',
                    admission_scope=admission_scope_from_config(config),
                    experiment_protocol=context.identity(), deadline_setting=deadline.identity(),
                    experiment_config=str(Path(config_path).resolve()),
                    experiment_config_sha256=file_sha256(config_path),
                    search_budget=budget, llm_rules=records)
    if mode == 'energy_only':
        manifest.update(objective_identity())
        manifest['selection_policy'].update(policy_version=ENERGY_POLICY,
                                             ranking_fields=ENERGY_RANKING_FIELDS,
                                             unique_structure_first=False)
    else:
        manifest['selection_policy']['policy_version'] = TOPK_SELECTION_POLICY_VERSION
    path = directory / f'topk_heuristic_library_k{k}.json'
    write_json(path, manifest)
    library = load_manager_heuristic_library(path, include_traditional=False,
                                              expected_protocol_identity=context.identity())
    if len(library) != k or any(not rule.available for rule in library):
        raise ValueError(f'exported GP rules failed production loader: {[r.public_metadata() for r in library]}')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('experiment.yaml'))
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--ddl', choices=('T', 'M', 'L'), required=True)
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--objective', choices=('original', 'energy_only'))
    parser.add_argument('--deadline-cache', action='append', required=True)
    parser.add_argument('--algorithm-seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1,
                        help='CPU candidate evaluation processes; each worker uses one numeric-library thread')
    parser.add_argument('--population-size', type=int, default=50)
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--episode-budget', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    if min(args.population_size, args.generations, args.max_depth) < 1 or args.population_size < 2:
        parser.error('population-size >= 2, generations/max-depth >= 1 required')
    if args.episode_budget is not None and args.episode_budget < 1:
        parser.error('episode-budget must be positive')
    if args.algorithm_seed < 0:
        parser.error('algorithm-seed must be non-negative')
    context, deadline, k, mode, config = experiment_settings(
        args.config, args.scenario, args.ddl, args.deadline_cache,
        top_k=args.top_k, objective=args.objective)
    if args.smoke:
        args.population_size, args.generations, k = 4, 2, 1
        config['top_k'] = k
        config['dataset']['workflows_per_instance'] = config['problem_size'] = 2
    elif args.population_size < k:
        parser.error('population-size must be at least top-k')
    directory = (args.output_dir or ROOT_DIR / 'out/ablations/pure_gp_hrl/rules'
                 / context.source_scenario / deadline.code
                 / f"{datetime.now():%Y%m%d_%H%M%S_%f}_a{args.algorithm_seed}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('generated', 'reports'):
        (directory / name).mkdir()
    config_path = directory / 'effective_experiment_config.yaml'
    OmegaConf.save(OmegaConf.create(config), config_path)
    search_config = dict(method='pure_gp', algorithm_seed=args.algorithm_seed,
                         population_size=args.population_size, generations=args.generations,
                         max_depth=args.max_depth, constant_range=[-4., 4.],
                         constant_leaf_probability=.25, tournament_size=3, elite_count=1,
                         crossover_probability=.80, mutation_probability=.15,
                         reproduction_probability=.05, cmaes_enabled=False,
                         parameter_optimization_enabled=False)
    evaluator = RuleEvaluator(directory, config, context, search_config, args.episode_budget)
    rng = np.random.default_rng(args.algorithm_seed)
    terminals = terminals_for_objective(mode)
    population = [random_program(rng, terminals, max_depth=min(args.max_depth, 2 + i % 5),
                                 force_operator=i % 2 == 0) for i in range(args.population_size)]
    archive, failures, history = {}, [], []
    status = 'COMPLETED'
    try:
        with evaluation_pool(args.workers) as executor:
            for generation in range(args.generations):
                scored = []
                for individual, program, record in evaluator.population(
                        population, generation, archive, executor, args.workers):
                    if isinstance(record, Exception):
                        failures.append(dict(generation=generation, individual=individual,
                                             error=f'{type(record).__name__}: {record}'))
                        continue
                    scored.append(record)
                if not scored:
                    raise RuntimeError('no valid GP individuals; inspect search_history.json')
                scored.sort(key=lambda row: row['ranking_key'])
                history.append(dict(generation=generation, best_key=scored[0]['ranking_key'],
                                    unique_rules=len(archive), **evaluator.counts))
                write_json(directory / 'search_history.json', dict(history=history, failures=failures))
                print(f"generation={generation} valid={len(scored)} unique={len(archive)} "
                      f"episodes={evaluator.counts['attempted_episode_count']} "
                      f"best={scored[0]['ranking_key']}", flush=True)
                if generation + 1 == args.generations:
                    break

                def tournament():
                    indices = rng.integers(len(scored), size=3)
                    winner = min((scored[index] for index in indices), key=lambda row: row['ranking_key'])
                    return GPProgram.from_dict(winner['program'])

                population = [GPProgram.from_dict(scored[0]['program'])]
                while len(population) < args.population_size:
                    parent = tournament()
                    draw = rng.random()
                    child = (crossover(parent, tournament(), rng, args.max_depth)[0] if draw < .8 else
                             mutate(parent, rng, args.max_depth) if draw < .95 else parent)
                    population.append(child)
    except EvaluationBudgetExceeded:
        status = 'EPISODE_BUDGET_REACHED'
    except Exception:
        status = 'FAILED'
        raise
    finally:
        write_json(directory / 'search_history.json', dict(status=status, history=history,
                                                          failures=failures, **evaluator.counts))
    budget = dict(**evaluator.counts, episode_budget=args.episode_budget,
                  population_size=args.population_size, generations=args.generations,
                  completed_generations=len(history), max_depth=args.max_depth,
                  algorithm_seed=args.algorithm_seed, status=status,
                  evaluation_workers=args.workers, threads_per_worker=1, cache_scope='per_run_in_memory',
                  search_config=search_config, cmaes_enabled=False,
                  parameter_optimization_enabled=False, surrogate_enabled=False,
                  counterfactual_feedback_enabled=False, critical_state_replay_enabled=False)
    path = export_library(directory, config_path, context, deadline, k,
                          mode, config, list(archive.values()), budget, smoke=args.smoke)
    print(path)
    return path


if __name__ == '__main__':
    main()
