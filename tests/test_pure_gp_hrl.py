"""Pure GP constants, direct evaluation and production Top-K compatibility."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from algorithms.comparisons.pure_gp_hrl.core import (
    PureGPProgram, random_program, crossover, mutate, frozen_source, terminals_for_objective,
)
from algorithms.comparisons.pure_gp_hrl.generate import RuleEvaluator, export_library, EvaluationBudgetExceeded
from algorithms.comparisons.pure_gp_hrl.train import validate_gp_library, main as train_main, build_gp_config
from algorithms.comparisons.drlea_nichgp.gp_primitives import program_depth
from algorithms.llm_safe_hrl.LLM.rule_optimization import CMAESOptimizer, extract_rule_metadata
from algorithms.llm_safe_hrl.LLM.export_topk import topk_ranking_key
from algorithms.llm_safe_hrl.base.heuristic_admission import file_sha256
from algorithms.llm_safe_hrl.base.manager_heuristics import load_manager_heuristic_library
from algorithms.llm_safe_hrl.base.llm_objective import validate_energy_only_source
from algorithms.llm_safe_hrl.run_context import resolve_deadline_setting
from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
from hrl_mix.train_config import ROOT_DIR
from tests.test_topk_heuristic_selection import _context, _scope, _metrics


CONTEXT = resolve_experiment_protocol('single', source_scenario='SS')
SEARCH = {'algorithm_seed': 0, 'cmaes_enabled': False, 'parameter_optimization_enabled': False}


def compile_rule(tokens, mode='original'):
    program = PureGPProgram(tuple(tokens), terminals_for_objective(mode))
    source = frozen_source(program, mode, SEARCH, CONTEXT)
    namespace = {}
    exec(source, namespace)
    return program, source, namespace['get_task_priority_v2']


class PureGPTests(unittest.TestCase):
    def test_constants_compile_scalar_equivalence_empty_and_no_input_mutation(self):
        inputs = [np.array([-2., 0., 2.]) + i for i in range(8)]
        inputs[1] = np.array([-1., 0., 1.])
        before = [x.copy() for x in inputs]
        for tokens in [('mul', .7, 'min_exec_time'),
                       ('pdiv', 2., 'min_comm_time'),
                       ('pdiv', 'min_exec_time', 0.),
                       ('add', 'pdiv', 2., 0., 'min_exec_time')]:
            program, source, rule = compile_rule(tokens)
            expected = [program([x[i] for x in inputs]) for i in range(3)]
            np.testing.assert_allclose(rule(*inputs), expected, rtol=0, atol=0)
            self.assertEqual(rule(*[np.array([]) for _ in inputs]).shape, (0,))
            self.assertEqual(rule(*[x[:1] for x in inputs]).shape, (1,))
            self.assertNotIn('PARAMS', source)
            self.assertNotIn('PARAMETER_SCHEMA', source)
            self.assertEqual(PureGPProgram.from_dict(program.to_dict()), program)
        for current, original in zip(inputs, before):
            np.testing.assert_array_equal(current, original)

    def test_rng_reproducibility_constants_and_depth_limits(self):
        names = terminals_for_objective('original')
        first = np.random.default_rng(9)
        second = np.random.default_rng(9)
        saw_constant, constants_changed = False, False
        for _ in range(60):
            a = random_program(first, names, max_depth=4)
            b = random_program(second, names, max_depth=4)
            self.assertEqual(a, b)
            saw_constant |= any(isinstance(token, float) for token in a.tokens)
            child = mutate(a, first, 4)
            twin = mutate(b, second, 4)
            self.assertEqual(child, twin)
            self.assertLessEqual(program_depth(child), 4)
            constants_changed |= ({token for token in a.tokens if isinstance(token, float)} !=
                                  {token for token in child.tokens if isinstance(token, float)})
            for crossed in crossover(a, child, first, 4):
                self.assertLessEqual(program_depth(crossed), 4)
            crossover(b, twin, second, 4)
        self.assertTrue(saw_constant)
        self.assertTrue(constants_changed)

    def test_bad_constants_constant_only_rules_and_forbidden_features_fail(self):
        for tokens in [(float('nan'),), ('mul', 5., 'min_exec_time'), (1.,), (True,), ('add', 'min_exec_time')]:
            with self.assertRaises((ValueError, IndexError)):
                compile_rule(tokens)
        _, source, rule = compile_rule(('mul', .8, 'min_incremental_energy'), 'energy_only')
        validate_energy_only_source(source)
        inputs = [np.array([2.]) for _ in range(8)]
        inputs[3] = inputs[7] = np.array([np.nan])
        np.testing.assert_array_equal(rule(*inputs), [1.6])
        with self.assertRaises(ValueError):
            compile_rule(('slack',), 'energy_only')

    def test_direct_simulation_no_cma_budget_and_test_seed_guard(self):
        program, _, _ = compile_rule(('mul', .5, 'min_exec_time'))
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as raw:
            directory = Path(raw)
            (directory / 'generated').mkdir()
            evaluator = RuleEvaluator(directory, {'llm_objective': 'original'}, CONTEXT, SEARCH, 3)

            def fake_evaluate(path, config, seeds):
                metadata = extract_rule_metadata(Path(path).read_text(encoding='utf-8'))
                return _metrics(Path(path), metadata)

            with patch.object(CMAESOptimizer, 'optimize', side_effect=AssertionError('CMA must never run')), \
                 patch('algorithms.comparisons.pure_gp_hrl.generate.evaluate_candidate', side_effect=fake_evaluate) as evaluate:
                candidate = evaluator.prepare(program, 0, 0)
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(evaluate.call_args.args[2], CONTEXT.llm_train_seeds)
                evaluator.exact(candidate['source_path'], [1, 2, 3])
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(evaluator.counts['training_episode_count'], 3)
                self.assertEqual(evaluator.counts['validation_episode_count'], 0)
                with self.assertRaises(EvaluationBudgetExceeded):
                    evaluator.exact(candidate['source_path'], [4, 5])
                with self.assertRaisesRegex(ValueError, 'final-test'):
                    evaluator.exact(candidate['source_path'], [201])

    def test_export_real_loader_and_variant_isolation(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as raw:
            directory = Path(raw)
            (directory / 'generated').mkdir()
            (directory / 'reports').mkdir()
            candidates = []
            for index, tokens in enumerate([('mul', .5, 'min_exec_time'), ('min_comm_time',)]):
                program, source, _ = compile_rule(tokens)
                path = directory / 'generated' / f'rule{index}.py'
                path.write_text(source, encoding='utf-8')
                metrics = _metrics(path, extract_rule_metadata(source))
                metrics['fuzzy_total_energy_score'] -= index * 10000
                candidates.append(dict(metrics=metrics, source_path=path, source_hash=file_sha256(path),
                                       iteration=0, individual=index, ranking_key=topk_ranking_key(metrics),
                                       program=program.to_dict()))
            reference_path = directory / 'reference.json'
            reference = {'selection_policy': dict(policy_version='feasibility_first_topk_v1',
                unique_structure_first=True, hard_energy_threshold=None,
                ranking_fields=['constraint_feasible', 'max_deadline_violation_rate_across_seeds',
                                'max_fuzzy_lateness', 'fuzzy_total_energy_score',
                                'objective_cv_across_seeds', 'candidate_sha256'])}
            reference_path.write_text(json.dumps(reference), encoding='utf-8')
            c = _context()
            config = dict(dataset={key: c[key] for key in ('workflows_per_instance', 'arrival_lambda',
                'horizon', 'deadline_mode', 'deadline_alpha_small', 'deadline_alpha_large',
                'deadline_alpha_small_prob')}, resources={key: c[key] for key in ('num_cloud_hosts',
                'num_edge_hosts', 'cloud_vms_per_host', 'edge_vms_per_host', 'cloud_pc_tiers',
                'edge_pc_tiers', 'cloud_bw_tiers', 'edge_bw_tiers')},
                fuzzy=dict(delta1=.75, delta2=1.2, deadline_eta=.95, energy_uncertainty_weight=1.,
                           resource_seed_mode='episode_seed', resource_seed_offset=0), admission_scope=_scope())
            config['dataset'].update(scenario='SS', dax_files=['CyberShake_30.xml', 'Epigenomics_24.xml',
                'Ligo_30.xml', 'Montage_25.xml', 'Sipht_29.xml'])
            path = export_library(directory, reference_path, reference, CONTEXT,
                                  resolve_deadline_setting('T'), 1, 'original', config, candidates, SEARCH)
            payload, _ = validate_gp_library(path)
            self.assertEqual(payload['llm_rules'][0]['source_hash'], candidates[1]['source_hash'])
            self.assertTrue(load_manager_heuristic_library(path, include_traditional=False)[0].available)
            self.assertFalse(payload['cmaes_enabled'])
            self.assertFalse(payload['parameter_optimization_enabled'])
            payload['method_id'] = 'gp_hrl'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_gp_library(path)

    def test_training_keeps_original_three_agents_and_independent_namespace(self):
        with patch('algorithms.comparisons.pure_gp_hrl.train.validate_gp_library',
                   return_value=({'deadline_setting': {'ddl_code': 'T'}}, CONTEXT)), \
             patch('algorithms.comparisons.pure_gp_hrl.train.validate_single_deadline_cache_paths', return_value={}), \
             patch('algorithms.comparisons.pure_gp_hrl.train.train') as runner:
            train_main(['--gp-library', 'pure.json', '--deadline-cache', 'SS=cache'])
        kwargs = runner.call_args.kwargs
        self.assertNotIn('manager_factory', kwargs)
        self.assertNotIn('environment_class', kwargs)
        self.assertIs(kwargs['config_builder'], build_gp_config)
        self.assertTrue(kwargs['safe_rl_shield_enabled'])
        self.assertTrue(kwargs['manager_heuristic_llm_only'])


if __name__ == '__main__':
    unittest.main()
