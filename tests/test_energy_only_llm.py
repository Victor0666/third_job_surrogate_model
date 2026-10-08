"""Energy-only contracts and original-mode regression, without LLM API calls."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from algorithms.llm_safe_hrl.paths import LLM_ROOT
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))
from seevo import SeEvo, individual_comparison_key, parameter_feedback_summary
from algorithms.llm_safe_hrl.base.llm_objective import (
    ENERGY_POLICY, ENERGY_RANKING_FIELDS, energy_fitness, objective_identity,
    validate_energy_only_source, validate_objective_identity,
)
from algorithms.llm_safe_hrl.LLM.export_topk import (
    _build_record, _candidate_from_report, export_topk_library, select_topk, topk_ranking_key,
)
from algorithms.llm_safe_hrl.LLM.rule_optimization.cmaes_optimizer import (
    CMAESOptimizer, OptimizerConfig, constraint_priority_key, rank_fitness,
)
from algorithms.llm_safe_hrl.LLM.rule_optimization.parameter_schema import parse_rule_candidate, freeze_rule_source
from algorithms.llm_safe_hrl.LLM.protocol_config import configure_seevo_protocol
from algorithms.llm_safe_hrl.base.heuristic_admission import file_sha256, record_sha256
from algorithms.llm_safe_hrl.base.manager_heuristics import load_manager_heuristic_library
from algorithms.llm_safe_hrl.base.safe_demonstration import _validate_manager_heuristic_identity
from hrl_mix.train_config import build_train_config
from hrl_mix.model_selection import build_heuristic_library_version
from tests.test_topk_heuristic_selection import _write_topk_library, _context, _scope


def caches():
    root = Path(__file__).resolve().parents[1] / 'data/deadlines/fcfs/exact_mix_v1'
    return {scenario: str(root / f'fcfs_{task}Task_smallRes_exactmix_formal38.json')
            for scenario, task in [('SS', 'small'), ('MS', 'med'), ('LS', 'large')]}


def metrics(energy=300, violation=.2, mode='energy_only'):
    return dict(fuzzy_total_energy_score=energy, objective=energy,
                constraint_feasible=violation == 0, constraint_violation=violation,
                deadline_violation_rate=violation, max_deadline_violation_rate_across_seeds=violation,
                total_lateness=100, max_fuzzy_lateness=100, objective_cv_across_seeds=.1,
                candidate_sha256='a' * 64, llm_objective=mode)


def individual(m):
    return dict(metrics=m, obj=m['objective'], exec_success=True)


def energy_library(tmp_path):
    path = _write_topk_library(tmp_path)
    original_hash = file_sha256(path)
    payload = json.loads(path.read_text())
    payload.update(objective_identity())
    payload['selection_policy'].update(policy_version=ENERGY_POLICY,
        ranking_fields=ENERGY_RANKING_FIELDS, unique_structure_first=False)
    record = payload['llm_rules'][0]
    evaluation = record['evaluation']
    evaluation.update(objective_identity())
    evaluation.update(constraint_feasible=False, deadline_violation_rate=.5,
        max_deadline_violation_rate_across_seeds=.5, max_fuzzy_lateness=100., feasible_seed_rate=0.)
    report = tmp_path / record['evaluation_report_file']
    report.write_text(json.dumps(evaluation), encoding='utf-8')
    from algorithms.llm_safe_hrl.base.heuristic_admission import canonical_json_sha256
    record.update(objective_identity())
    record['evaluation_report_hash'] = file_sha256(report)
    record['evaluation_result_sha256'] = canonical_json_sha256(evaluation)
    record.update(deadline_violation_rate=.5, max_fuzzy_lateness=100., feasible_seed_rate=0.)
    record['selection_key'] = list(topk_ranking_key(evaluation))
    record['record_sha256'] = record_sha256(record)
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path, original_hash


def test_energy_ranking_and_original_regression():
    low = metrics(300, .2)
    high = metrics(400, 0)
    assert individual_comparison_key(individual(low)) < individual_comparison_key(individual(high))
    assert topk_ranking_key(low) < topk_ranking_key(high)
    low['llm_objective'] = high['llm_objective'] = 'original'
    assert individual_comparison_key(individual(high)) < individual_comparison_key(individual(low))
    assert topk_ranking_key(high) < topk_ranking_key(low)


@pytest.mark.parametrize('field', ['deadline_violation_rate', 'constraint_violation',
    'max_deadline_violation_rate_across_seeds', 'max_fuzzy_lateness', 'total_lateness',
    'objective_cv_across_seeds', 'constraint_feasible'])
def test_safety_and_robustness_do_not_change_fitness(field):
    a = metrics()
    b = dict(a, **{field: 0})
    assert energy_fitness(a) == energy_fitness(b)
    assert topk_ranking_key(a) == topk_ranking_key(b)
    assert individual_comparison_key(individual(a)) == individual_comparison_key(individual(b))
    assert rank_fitness([a, b], llm_objective='energy_only') == [300., 300.]


@pytest.mark.parametrize('expression', ['slack', 'uncertainty', 'min_exec_time + slack', 'locals()["slack"]'])
def test_forbidden_features(expression):
    source = ('def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, '
              'upward_rank, remaining_work, ready_wait_time, uncertainty):\n    return ' + expression)
    with pytest.raises(ValueError, match='energy_only'):
        validate_energy_only_source(source)


def test_renamed_safety_slot_and_allowed_features():
    with pytest.raises(ValueError, match='secret'):
        validate_energy_only_source('def get_task_priority_v2(a,b,c,secret,e,f,g,h):\n    return secret')
    source = (LLM_ROOT / 'prompts/cews_task_constructive_energy_only/parameterized_seed_func.txt').read_text()
    source = source.split('```python')[1].split('```')[0]
    validate_energy_only_source(source)
    candidate = parse_rule_candidate(source)
    frozen = freeze_rule_source(source, candidate.parameter_schema, candidate.parameter_schema.initial_values)
    validate_energy_only_source(frozen)


def test_cmaes_uses_energy_scalar_and_selects_low_energy():
    assert rank_fitness([metrics(300, .5), metrics(400, 0)], llm_objective='energy_only') == [300., 400.]
    assert constraint_priority_key(metrics(300, .5), llm_objective='energy_only') < constraint_priority_key(metrics(400, 0), llm_objective='energy_only')
    source = (LLM_ROOT / 'prompts/cews_task_constructive_energy_only/parameterized_seed_func.txt').read_text().split('```python')[1].split('```')[0]
    candidate = parse_rule_candidate(source)
    cfg = OptimizerConfig(enabled=True, llm_objective='energy_only', population_size=4,
        max_generations=1, optimizer_seed=17, diagnostic_perturbations=False)
    result = CMAESOptimizer(cfg).optimize(candidate.parameter_schema,
        lambda params, stage, seeds: metrics(100 + params['energy_weight'] * 100,
                                            .5 if params['energy_weight'] < 1 else 0),
        train_seeds=[1, 2, 3], validation_seeds=[4, 5])
    assert result.best_metrics['fuzzy_total_energy_score'] == min(row['metrics']['fuzzy_total_energy_score'] for row in result.history)
    assert 'llm_objective' not in OptimizerConfig().as_dict()


def test_unsafe_valid_rule_loads_and_hash_demo_mismatch(tmp_path):
    path, old_hash = energy_library(tmp_path)
    rules = load_manager_heuristic_library(path, runtime_context=dict(_context(), scenario_code="SS", task_code="S", resource_code="S"), include_traditional=False)
    assert len(rules) == 1 and rules[0].available
    assert not rules[0].admitted
    from algorithms.llm_safe_hrl.base.heuristic_admission import admission_policy_from_config, evaluate_admission_result
    admission_cfg = OmegaConf.to_container(OmegaConf.load(LLM_ROOT / 'cfg/problem/cews_task_constructive_hrl_ss_admission.yaml'))
    original_reasons = evaluate_admission_result(json.loads(path.read_text())['llm_rules'][0]['evaluation'],
        admission_policy_from_config(admission_cfg), expected_workflows_per_seed=50)
    assert 'aggregate_constraint_not_feasible' in original_reasons
    assert file_sha256(path) != old_hash
    payload = json.loads(path.read_text())
    validate_objective_identity(payload, 'energy_only')
    with pytest.raises(ValueError, match='llm_objective mismatch'):
        validate_objective_identity(payload, 'original')
    ids = [rules[0].heuristic_id]
    with pytest.raises(ValueError, match='hash mismatch'):
        _validate_manager_heuristic_identity(dict(manager_heuristic_ids=ids,
            manager_heuristic_manifest_sha256=old_hash), ids, file_sha256(path))
    version = build_heuristic_library_version(SimpleNamespace(manager_mode='heuristic_selection_mode',
        manager_heuristics=rules), manifest_path=path)
    assert version['llm_objective'] == 'energy_only'
    assert version['manifest_sha256'] == file_sha256(path)


def test_topk_is_lowest_energy_even_same_structure():
    candidates = [dict(metrics=dict(structure_hash='a'*64), ranking_key=(energy, str(energy))) for energy in range(20)]
    assert [x['ranking_key'][0] for x in select_topk(candidates[::-1], 10, llm_objective='energy_only')] == list(range(10))


def test_feedback_excludes_safety_evidence():
    ind = individual(metrics())
    ind.update(parameter_diagnostics={'safety_margin': 123}, counterfactual_feedback={'deadline': 3},
               critical_state_replay_summary={'slack': 10})
    assert json.loads(parameter_feedback_summary(ind)) == dict(llm_objective='energy_only', mean_fuzzy_energy_score=300.)


@pytest.mark.parametrize('surrogate_enabled', [True, False])
def test_config_protocol_budget_and_isolation(tmp_path, surrogate_enabled):
    with initialize_config_dir(version_base=None, config_dir=str(LLM_ROOT / 'cfg')):
        cfg = compose(config_name='config', overrides=['problem=cews_task_constructive_energy_only'])
        original = compose(config_name='config')
    for key in ('max_fe', 'pop_size', 'init_pop_size', 'candidate_generation'):
        assert cfg[key] == original[key]
    OmegaConf.update(cfg, 'deadline_cache_paths', caches(), force_add=True)
    cfg.surrogate.enabled = surrogate_enabled
    cfg.execution_id = 'energy_test'
    cfg.experiment_key = 'SS_T'
    cfg.runtime_output_root = str(tmp_path / 'runtime')
    # Patch context roots to keep this integration test's artifacts in tmp_path.
    from unittest.mock import patch
    from algorithms.llm_safe_hrl.run_context import ExperimentRunContext
    with (patch.object(ExperimentRunContext, 'artifact_output_root', new=property(lambda self: tmp_path / 'main_energy_only/SS/T/energy_test')),
          patch.object(ExperimentRunContext, 'checkpoint_root', new=property(lambda self: tmp_path / 'checkpoints/energy_test'))):
        ctx = configure_seevo_protocol(cfg, llm_root=LLM_ROOT)
        manifest = json.loads((ctx.artifact_output_root / 'run_manifest.json').read_text())
        algorithm = SeEvo.__new__(SeEvo)
        algorithm.cfg, algorithm.root_dir = cfg, str(LLM_ROOT)
        algorithm.llm_objective = 'energy_only'
        algorithm.init_prompt()
        assert 'smooth-risk change' not in algorithm.user_reflector_st_prompt
        assert 'Only mean fuzzy energy' in algorithm.user_reflector_st_prompt
        assert 'slack_weight' not in algorithm.generation_reference_func
    assert manifest['llm_objective'] == 'energy_only'
    assert cfg.parameter_optimization.llm_objective == 'energy_only'
    assert not cfg.counterfactual_feedback.enabled and not cfg.critical_state_replay.enabled
    assert cfg.timeout == 300
    assert cfg.surrogate.enabled == surrogate_enabled
    assert cfg.surrogate.llm_objective == 'energy_only'
    assert not cfg.parameter_optimization.auto_admission_enabled
    assert not cfg.parameter_optimization.diagnostic_replay_gate.enabled


def test_energy_manifest_auto_isolates_safe_training(tmp_path):
    path, _ = energy_library(tmp_path)
    cfg = build_train_config(protocol='single', source_scenario='SS', ddl='T',
        safe_rl_enabled=True, safe_rl_shield_enabled=True, safe_rl_state_enabled=True,
        safe_rl_dynamic_lambda_enabled=True, safe_rl_heuristic_manager_enabled=True,
        deadline_cache_paths=caches(),
        manager_heuristic_llm_only=True, manager_heuristic_manifest=str(path))
    assert 'main_energy_only' in str(cfg.save_dir)
    assert cfg.safe_rl.enabled
    assert cfg.safe_rl.safety_cost_definition == "actual_deadline_violation"
    assert cfg.safe_rl.shield_semantics == "monitor_only"
    assert cfg.safe_rl.lagrangian.cost_budget == .02


def test_real_evaluator_energy_identity_and_invalid_outputs(tmp_path):
    from problems.cews_task_constructive.eval import evaluate_candidate, load_problem_config
    cfg = load_problem_config()
    cfg['llm_objective'] = 'energy_only'
    cfg['dataset'].update(workflows_per_instance=1, deadline_mode='none')
    source = ('import numpy as np\n'
              'def get_task_priority_v2(min_exec_time,min_comm_time,min_incremental_energy,slack,'
              'upward_rank,remaining_work,ready_wait_time,uncertainty):\n'
              '    return np.asarray(min_incremental_energy, dtype=float)\n')
    candidate = tmp_path / 'candidate.py'
    candidate.write_text(source, encoding='utf-8')
    result = evaluate_candidate(candidate, cfg, [1])
    assert result['all_evaluation_seeds_completed']
    assert result['objective'] == result['fuzzy_total_energy_score']
    validate_objective_identity(result, 'energy_only')
    original = evaluate_candidate(candidate, dict(cfg, llm_objective='original'), [1])
    assert original['objective'] == result['objective']
    for expression in ('np.array([np.nan])', 'np.array([np.inf])', 'np.zeros((2, 2))'):
        candidate.write_text(source.replace('np.asarray(min_incremental_energy, dtype=float)', expression), encoding='utf-8')
        with pytest.raises(ValueError):
            evaluate_candidate(candidate, cfg, [1])


def test_full_top10_export_and_runtime_loading(tmp_path):
    from algorithms.llm_safe_hrl.base.heuristic_admission import evaluation_context_from_config
    from algorithms.llm_safe_hrl.LLM.protocol_config import apply_seevo_scenario_config
    from algorithms.llm_safe_hrl.run_context import resolve_deadline_setting
    from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
    path, _ = energy_library(tmp_path)
    payload = json.loads(path.read_text())
    template = payload['llm_rules'][0]
    original_source = (tmp_path / template['source_file']).read_text()
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    for index in range(10):
        source = tmp_path / 'generated' / f'candidate_iter2_ind{index}.py'
        source.write_text(original_source + f'\n# distinct frozen source {index}\n', encoding='utf-8')
        report = copy.deepcopy(template['evaluation'])
        report.update(candidate_source_file=source.name, candidate_sha256=file_sha256(source),
            frozen_rule_hash=file_sha256(source), fuzzy_total_energy_score=100. + index,
            objective=100. + index)
        (runtime / f'problem_iter2_ind{index}_stdout.txt').write_text('RESULT_JSON=' + json.dumps(report), encoding='utf-8')
    admission = OmegaConf.to_container(OmegaConf.load(LLM_ROOT / 'cfg/problem/cews_task_constructive_hrl_ss_admission.yaml'))
    admission = apply_seevo_scenario_config(admission, 'SS', caches(), require_files=True)
    admission['llm_objective'] = 'energy_only'
    OmegaConf.save(OmegaConf.create(admission), tmp_path / 'effective_admission_config.yaml')
    run = dict(status='COMPLETED', llm_objective='energy_only', execution_id='energy_export',
        deadline_setting=resolve_deadline_setting('T').identity(),
        experiment_protocol=resolve_experiment_protocol('single', source_scenario='SS').identity())
    (tmp_path / 'run_manifest.json').write_text(json.dumps(run), encoding='utf-8')
    exported = export_topk_library(source_scenario='SS', ddl='T', execution_id='energy_export',
        top_k=10, run_dir=tmp_path, runtime_dir=runtime, llm_objective='energy_only')
    library = json.loads(exported.read_text())
    validate_objective_identity(library, 'energy_only')
    assert [x['fuzzy_energy_score'] for x in library['llm_rules']] == list(range(100, 110))
    assert len({x['heuristic_id'] for x in library['llm_rules']}) == 10
    rules = load_manager_heuristic_library(exported, include_traditional=False)
    assert all(rule.available for rule in rules)
    with pytest.raises(ValueError, match='objective mismatch'):
        export_topk_library(source_scenario='SS', ddl='T', execution_id='energy_export',
            top_k=10, run_dir=tmp_path, runtime_dir=runtime, llm_objective='original')
