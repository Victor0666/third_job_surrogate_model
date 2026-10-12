"""Evolve GP Top-K rules; reuse CEWS evaluation and offline CMA-ES only."""
import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from algorithms.comparisons.drlea_nichgp.gp_primitives import GPProgram, random_program, crossover, mutate
from algorithms.llm_safe_hrl.LLM.export_topk import _build_record, select_topk, topk_ranking_key
from algorithms.llm_safe_hrl.LLM.protocol_config import apply_seevo_scenario_config
from algorithms.llm_safe_hrl.LLM.problems.cews_task_constructive.eval import evaluate_candidate
from algorithms.llm_safe_hrl.LLM.rule_optimization import (
    CMAESOptimizer, OptimizerConfig, parse_rule_candidate, freeze_rule_source,
    generate_parameter_diagnostics, canonical_json_sha256,
)
from algorithms.llm_safe_hrl.base.heuristic_admission import (
    file_sha256, record_sha256, evaluation_context_from_config, admission_scope_from_config,
)
from algorithms.llm_safe_hrl.base.llm_objective import (
    validate_objective_identity, objective_identity, ENERGY_POLICY, ENERGY_RANKING_FIELDS,
)
from algorithms.llm_safe_hrl.base.manager_heuristics import load_manager_heuristic_library
from algorithms.llm_safe_hrl.base.topk_schema import (
    TOPK_MANIFEST_SCHEMA_VERSION, TOPK_SELECTION_MODE, TOPK_SELECTION_POLICY_VERSION,
)
from algorithms.llm_safe_hrl.run_context import resolve_deadline_setting
from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
from hrl_mix.train_config import ROOT_DIR, validate_single_deadline_cache_paths
from .core import parameterized_source, terminals_for_objective


class EvaluationBudgetExceeded(BaseException):
    """Escape CMA's candidate-error handler when the simulation budget ends."""


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def reference_settings(path, caches):
    path = Path(path).resolve()
    reference = json.loads(path.read_text(encoding='utf-8'))
    mode = validate_objective_identity(reference)
    if reference.get('selection_mode') != TOPK_SELECTION_MODE:
        raise ValueError('reference must be a Top-K library')
    identity = reference['experiment_protocol']
    context = resolve_experiment_protocol('single', source_scenario=identity['source_scenario'])
    if context.identity() != identity:
        raise ValueError('reference does not match the formal Single seed protocol')
    deadline = resolve_deadline_setting(reference['deadline_setting']['ddl_code'])
    if deadline.identity() != reference['deadline_setting']:
        raise ValueError('reference deadline identity mismatch')
    k = int(reference['selection_policy']['requested_k'])
    if k < 1 or len(reference['llm_rules']) != k:
        raise ValueError('reference must contain exactly K rules')
    caches = validate_single_deadline_cache_paths('single', caches,
                                                source_scenario=context.source_scenario)
    template = path.parent / 'effective_admission_config.yaml'
    if not template.is_file():
        raise FileNotFoundError(f'reference evaluation config not found: {template}')
    config = OmegaConf.to_container(OmegaConf.load(template), resolve=True)
    config = apply_seevo_scenario_config(config, context.source_scenario, caches, require_files=True)
    config['llm_objective'] = mode
    config['experiment_protocol'] = identity
    config['dataset'].update(train_seeds=list(context.llm_train_seeds),
                             validation_seeds=list(context.llm_validation_seeds),
                             test_seeds=list(context.final_test_seeds))
    config['dataset'].update({key: value for key, value in deadline.identity().items()
                              if key.startswith('deadline_')})
    if any(evaluation_context_from_config(config) != rule['evaluation_context']
           for rule in reference['llm_rules']):
        raise ValueError('current simulator inputs differ from reference evaluation context')
    # Historical Linux paths are never used; explicit current-machine caches win.
    config['deadline_cache_paths'] = caches
    defaults = OmegaConf.to_container(OmegaConf.load(
        ROOT_DIR / 'algorithms/llm_safe_hrl/LLM/cfg/config.yaml'), resolve=False)
    runtime = (ROOT_DIR / 'algorithms/llm_safe_hrl/LLM/outputs/formal'
               / f'{context.source_scenario}_{deadline.code}' / path.parent.name / '.hydra/config.yaml')
    if runtime.is_file():
        defaults = OmegaConf.to_container(OmegaConf.load(runtime), resolve=False)
    optimizer = OptimizerConfig.from_mapping(defaults['parameter_optimization'])
    optimizer = replace(optimizer, enabled=True, llm_objective=mode,
                        auto_admission_enabled=False, admission_required=False,
                        max_parallel_evaluations=1, scenario_ids=context.training_scenarios,
                        diagnostic_replay_gate=replace(optimizer.diagnostic_replay_gate, enabled=False))
    return reference, context, deadline, k, mode, config, optimizer


class RuleEvaluator:
    def __init__(self, directory, config, context, optimizer, episode_budget=None):
        self.directory, self.config, self.context = Path(directory), config, context
        self.optimizer, self.episode_budget = optimizer, episode_budget
        self.counts = dict(training_episode_count=0, validation_episode_count=0,
                           attempted_episode_count=0, cache_hits=0)
        self.cache = {}
        (self.directory / 'trials').mkdir(exist_ok=True)

    def exact(self, path, seeds):
        seeds = tuple(int(seed) for seed in seeds)
        if not seeds or set(seeds) & set(self.context.final_test_seeds):
            raise ValueError('offline search may not consume final-test seeds')
        if not set(seeds) <= set((*self.context.llm_train_seeds, *self.context.llm_validation_seeds)):
            raise ValueError('offline search seeds outside training/validation split')
        key = (file_sha256(path), seeds)
        if self.optimizer.cache_enabled and key in self.cache:
            self.counts['cache_hits'] += 1
            return dict(self.cache[key])
        if (self.episode_budget is not None and
                self.counts['attempted_episode_count'] + len(seeds) > self.episode_budget):
            raise EvaluationBudgetExceeded()
        self.counts['attempted_episode_count'] += len(seeds)
        metrics = evaluate_candidate(path, self.config, seeds)
        self.counts['training_episode_count'] += len(set(seeds) & set(self.context.llm_train_seeds))
        self.counts['validation_episode_count'] += len(set(seeds) & set(self.context.llm_validation_seeds))
        if self.optimizer.cache_enabled:
            self.cache[key] = dict(metrics)
        return metrics

    def prepare(self, program, generation, individual):
        source = parameterized_source(program, self.optimizer.llm_objective)
        candidate = parse_rule_candidate(source, max_parameters=self.optimizer.max_parameters,
                                         max_branches=self.optimizer.max_branches,
                                         max_ast_depth=self.optimizer.max_ast_depth,
                                         max_interactions=self.optimizer.max_interactions)
        schema = candidate.parameter_schema
        def evaluate(parameters, stage, seeds):
            frozen = freeze_rule_source(source, schema, parameters)
            digest = hashlib.sha256(frozen.encode('utf-8')).hexdigest()
            # Immutable paths prevent Python's timestamp/size bytecode cache from reusing another vector.
            temporary = self.directory / 'trials' / f'{digest[:24]}.py'
            if temporary.exists() and temporary.read_text(encoding='utf-8') != frozen:
                raise ValueError('parameter-trial hash collision')
            if not temporary.exists():
                temporary.write_text(frozen, encoding='utf-8')
            return self.exact(temporary, seeds)

        result = CMAESOptimizer(self.optimizer).optimize(
            schema, evaluate, train_seeds=self.context.llm_train_seeds,
            validation_seeds=self.context.llm_validation_seeds,
            final_test_seeds=self.context.final_test_seeds)
        diagnostics = generate_parameter_diagnostics(
            schema, result, boundary_epsilon=self.optimizer.boundary_epsilon,
            sensitivity_epsilon=self.optimizer.sensitivity_epsilon,
            correlation_threshold=self.optimizer.correlation_threshold)
        metadata = dict(structure_hash=candidate.structure_hash,
                        parameter_schema_hash=schema.schema_hash,
                        best_parameter_hash=canonical_json_sha256(result.best_parameters),
                        best_parameters=result.best_parameters,
                        optimizer_config_hash=canonical_json_sha256(self.optimizer.as_dict()),
                        parameter_diagnostics_hash=canonical_json_sha256(diagnostics),
                        optimizer_seed=self.optimizer.optimizer_seed,
                        training_seeds=list(self.context.llm_train_seeds),
                        validation_seeds=list(self.context.llm_validation_seeds))
        path = self.directory / 'generated' / f'gp_g{generation}_i{individual}.py'
        path.write_text(freeze_rule_source(source, schema, result.best_parameters, metadata=metadata),
                        encoding='utf-8')
        metrics = self.exact(path, self.context.llm_train_seeds)
        record = dict(metrics=metrics, source_path=path, source_hash=file_sha256(path),
                      iteration=generation, individual=individual,
                      ranking_key=topk_ranking_key(metrics), program=program.to_dict())
        write_json(self.directory / 'optimization' / f'{path.stem}.json',
                   dict(optimization=result.as_dict(), diagnostics=diagnostics,
                        gp_program=program.to_dict()))
        return record


def export_library(directory, reference_path, reference, context, deadline, k, mode, config,
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
        record.update(heuristic_id=f"gp_topk_{candidate['source_hash'][:16]}",
                      display_name=f'GP Top-K #{rank}', version=f'gp.topk.v1.rank{rank}',
                      generator='gp', gp_generation=candidate['iteration'],
                      gp_individual=candidate['individual'], gp_program=candidate['program'])
        # shortcut: production requires this legacy wire tag, remove when its schema supports GP.
        record['source_kind'] = 'seevo_generated'
        record['record_sha256'] = record_sha256(record)
        records.append(record)
    manifest = dict(schema_version=TOPK_MANIFEST_SCHEMA_VERSION,
                    manifest_id=f'gp_hrl_{directory.name}', manifest_version='gp_hrl.topk.v1',
                    method_id='gp_hrl', generator='gp', llm_enabled=False, smoke_only=smoke,
                    compatibility_source_kind='seevo_generated',
                    selection_mode=TOPK_SELECTION_MODE,
                    selection_policy={**reference['selection_policy'], 'requested_k': k,
                                      'selected_count': k},
                    trusted_source_root='generated', trusted_report_root='reports',
                    admission_scope=admission_scope_from_config(config),
                    experiment_protocol=context.identity(), deadline_setting=deadline.identity(),
                    reference_library=str(Path(reference_path).resolve()),
                    reference_library_sha256=file_sha256(reference_path),
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
    parser.add_argument('--reference-library', type=Path, required=True)
    parser.add_argument('--deadline-cache', action='append', required=True)
    parser.add_argument('--algorithm-seed', type=int, default=0)
    parser.add_argument('--population-size', type=int, default=50)
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--episode-budget', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if min(args.population_size, args.generations, args.max_depth) < 1 or args.population_size < 2:
        parser.error('population-size >= 2, generations/max-depth >= 1 required')
    if args.episode_budget is not None and args.episode_budget < 1:
        parser.error('episode-budget must be positive')
    if args.algorithm_seed < 0:
        parser.error('algorithm-seed must be non-negative')
    reference, context, deadline, k, mode, config, optimizer = reference_settings(
        args.reference_library, args.deadline_cache)
    # pycma interprets seed=0 as a time-based seed; use an explicit positive seed.
    optimizer = replace(optimizer, optimizer_seed=args.algorithm_seed + 1)
    if args.smoke:
        args.population_size, args.generations, k = 4, 2, 1
        config['dataset']['workflows_per_instance'] = config['problem_size'] = 2
        optimizer = replace(optimizer, population_size=4, max_generations=1,
                            diagnostic_perturbations=False, stage_seed_counts=dict(quick=1, refine=1, confirm=1))
    elif args.population_size < k:
        parser.error('population-size must be at least reference K')
    directory = (args.output_dir or ROOT_DIR / 'out/ablations/gp_hrl/rules'
                 / context.source_scenario / deadline.code
                 / f"{datetime.now():%Y%m%d_%H%M%S_%f}_a{args.algorithm_seed}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('generated', 'reports', 'optimization'):
        (directory / name).mkdir()
    OmegaConf.save(OmegaConf.create(config), directory / 'effective_admission_config.yaml')
    evaluator = RuleEvaluator(directory, config, context, optimizer, args.episode_budget)
    rng = np.random.default_rng(args.algorithm_seed)
    terminals = terminals_for_objective(mode)
    population = [random_program(rng, terminals, max_depth=min(args.max_depth, 2 + i % 5),
                                 force_operator=i % 2 == 0) for i in range(args.population_size)]
    archive, failures, history = {}, [], []
    status = 'COMPLETED'
    try:
        for generation in range(args.generations):
            scored = []
            for individual, program in enumerate(population):
                if program.tokens not in archive:
                    try:
                        record = evaluator.prepare(program, generation, individual)
                        archive[program.tokens] = record
                    except (ValueError, FloatingPointError, OverflowError) as exc:
                        failures.append(dict(generation=generation, individual=individual,
                                             error=f'{type(exc).__name__}: {exc}'))
                        continue
                scored.append(archive[program.tokens])
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
                  evaluation_workers=1, cache_scope='per_run_in_memory',
                  optimizer_config=optimizer.as_dict(), surrogate_enabled=False,
                  counterfactual_feedback_enabled=False, critical_state_replay_enabled=False)
    path = export_library(directory, args.reference_library, reference, context, deadline, k,
                          mode, config, list(archive.values()), budget, smoke=args.smoke)
    print(path)
    return path


if __name__ == '__main__':
    main()
