"""Energy-only surrogate gates use real labels and ignore safety diagnostics."""
from dataclasses import replace
import sys

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))
from algorithms.llm_safe_hrl.LLM.rule_optimization.parameter_schema import parse_rule_candidate
from algorithms.llm_safe_hrl.LLM.rule_optimization.cmaes_optimizer import CMAESOptimizer, OptimizerConfig
from algorithms.llm_safe_hrl.LLM.surrogate.config import (
    ActiveLearningConfig, ParameterGateConfig, StructureGateConfig, SurrogateConfig, WarmupConfig,
)
from algorithms.llm_safe_hrl.LLM.surrogate.dataset import SurrogateContext
from algorithms.llm_safe_hrl.LLM.surrogate.features import extract_parameter_features
from algorithms.llm_safe_hrl.LLM.surrogate.manager import SurrogateManager
from algorithms.llm_safe_hrl.LLM.surrogate.models import ExtraTreesMetricModel, SurrogatePrediction
from algorithms.llm_safe_hrl.LLM.utils.utils import extract_code_from_generator


def candidate():
    source = (LLM_ROOT / 'prompts/cews_task_constructive_energy_only/parameterized_seed_func.txt').read_text()
    return parse_rule_candidate(extract_code_from_generator(source))


def context():
    return SurrogateContext('v1', 'sim', 'eval', 'res', 'protocol', 'SS', 'cews', 'T')


def metrics(energy, violation):
    return dict(fuzzy_total_energy_score=energy, constraint_feasible=violation == 0,
                deadline_violation_rate=violation, total_lateness=violation * 100,
                objective_std_across_seeds=violation * 10)


def test_energy_features_and_regressors_ignore_safety():
    rule = candidate()
    parameters = rule.parameter_schema.values_dict(rule.parameter_schema.initial_values)
    a = extract_parameter_features(rule, parameters, metrics(50, 0), llm_objective='energy_only')
    b = extract_parameter_features(rule, parameters, metrics(50, 1), llm_objective='energy_only')
    np.testing.assert_array_equal(a, b)
    model = ExtraTreesMetricModel(llm_objective='energy_only')
    # No safety labels are required to fit or validate this model.
    model.fit([{'features': [float(i)], 'label': {'energy': float(i)}} for i in range(80)])
    assert model.healthy
    assert set(model.regressors) == {'energy'}
    assert model.validation_metrics['promising_recall'] >= .8
    prediction = model.predict([10], robustness=1000)
    assert set(prediction.uncertainty) == {'energy'}
    assert 'deadline_violation_rate' not in prediction.metrics
    assert 'objective_std_across_seeds' not in prediction.metrics


def test_real_energy_structure_and_parameter_gates(tmp_path):
    cfg = SurrogateConfig(enabled=True, llm_objective='energy_only',
        retrain_every_exact_evaluations=1000,
        warmup=WarmupConfig(min_structures=1, min_exact_evaluations=1),
        structure_gate=StructureGateConfig(.25, .25, 1),
        parameter_gate=ParameterGateConfig(1, 1),
        active_learning=ActiveLearningConfig(0, 0))
    manager = SurrogateManager(cfg, context(), artifact_root=tmp_path)
    rule = candidate()
    parameters = rule.parameter_schema.values_dict(rule.parameter_schema.initial_values)
    # Warmup falls back to all exact candidates, including unsafe low-energy ones.
    assert not manager.select_parameters(rule, [parameters], [metrics(10, 1)], generation=0)['gate_used']
    for energy in range(1, 81):
        observation = metrics(float(energy), 1 if energy < 40 else 0)
        manager.record_parameter_exact(rule, parameters, observation, observation)
        manager.record_structure_exact(rule, parameters, observation, observation)
    manager.retrain(force=True)
    observations = [metrics(10, 1), metrics(70, 0), metrics(50, 0), metrics(30, 1)]
    for observation in observations:
        observation['objective_std_across_seeds'] = 'unused diagnostic'
    for decision in (
        manager.select_parameters(rule, [parameters] * 4, observations, generation=0),
        manager.select_structures([rule] * 4, [parameters] * 4, observations, iteration=0),
    ):
        assert decision['gate_used'], decision
        assert decision['selected_indices'] == [0]  # Unsafe but lowest energy.
    assert manager._priority_key(observations[0]) < manager._priority_key(observations[1])
    assert all(set(row['label']) == {'energy'} for row in manager.dataset.records('parameter'))
    manager._save_checkpoint()
    loaded = SurrogateManager(cfg, context(), artifact_root=tmp_path)
    assert loaded.healthy
    assert set(loaded.parameter_model.regressors) == {'energy'}
    original = SurrogateManager(replace(cfg, llm_objective='original'), context(), artifact_root=tmp_path)
    assert original.dataset.count() == 0
    assert original.disabled_reason == 'checkpoint_context_mismatch'


def test_energy_prediction_failure_keeps_all_exact(tmp_path):
    cfg = SurrogateConfig(enabled=True, llm_objective='energy_only',
                          warmup=WarmupConfig(1, 1), parameter_gate=ParameterGateConfig(1, 1))
    manager = SurrogateManager(cfg, context(), artifact_root=tmp_path)
    rule = candidate()
    params = rule.parameter_schema.values_dict(rule.parameter_schema.initial_values)
    manager.record_parameter_exact(rule, params, metrics(10, 1), metrics(10, 1))
    manager.parameter_model.healthy = True
    manager.parameter_model.predict = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('failed'))
    decision = manager.select_parameters(rule, [params] * 3, [metrics(10, 1)] * 3, generation=0)
    assert not decision['gate_used']
    assert decision['selected_indices'] == [0, 1, 2]


def test_energy_cma_skips_seed_calls_but_keeps_best_and_elites_exact():
    rule = candidate()
    executed = set()

    class Manager:
        enabled = True

        def select_parameters(self, candidate, maps, quick, **kwargs):
            prediction = SurrogatePrediction(
                {'llm_objective': 'energy_only', 'fuzzy_total_energy_score': 1e6, 'objective': 1e6},
                {'energy': 0.0})
            return dict(selected_indices=[0, 1, 2], predictions=[prediction] * len(maps),
                        reasons={}, gate_used=True)

        def record_parameter_exact(self, *args, **kwargs):
            pass

    def batch(maps, stage, seeds):
        results = []
        for params in maps:
            for seed in seeds:
                executed.add((tuple(sorted(params.items())), seed))
            result = metrics(params['energy_weight'] ** 2 + params['work_weight'] ** 2, 1)
            result['llm_objective'] = 'energy_only'
            results.append(result)
        return results

    result = CMAESOptimizer(OptimizerConfig(
        enabled=True, llm_objective='energy_only', optimizer_seed=4,
        population_size=6, max_generations=1, diagnostic_perturbations=False,
        elite_fraction=.5, stage_seed_counts={'quick': 1, 'refine': 3, 'confirm': 0},
    )).optimize(rule.parameter_schema, lambda params, stage, seeds: batch([params], stage, seeds)[0],
               batch_evaluator=batch, train_seeds=[1, 2, 3], final_test_seeds=[201],
               surrogate_manager=Manager(), surrogate_candidate=rule)
    assert len(executed) == 12  # Full exact evaluation would need 18 unique seed calls.
    assert result.surrogate_stats['predicted_refine_vectors'] == 3
    assert result.surrogate_stats['estimated_seed_calls_avoided'] == 6
    assert all(row['evaluation_source'] == 'exact' for row in result.elite_samples)
    assert tuple(sorted(result.best_parameters.items())) in {params for params, _ in executed}
    assert all(seed != 201 for _, seed in executed)
