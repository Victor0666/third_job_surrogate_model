"""GP-HRL contract, budget and production-loader integration checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from algorithms.comparisons.drlea_nichgp.gp_primitives import GPProgram, OPERATORS
from algorithms.comparisons.gp_hrl.core import parameterized_source, terminals_for_objective
from algorithms.comparisons.gp_hrl.generate import (
    RuleEvaluator, EvaluationBudgetExceeded, export_library,
)
from algorithms.comparisons.gp_hrl.train import (
    validate_gp_library, build_gp_config, save_gp_checkpoint, main as train_main,
)
from algorithms.llm_safe_hrl.LLM.export_topk import topk_ranking_key
from algorithms.llm_safe_hrl.LLM.rule_optimization import (
    parse_rule_candidate, freeze_rule_source, canonical_json_sha256, OptimizerConfig,
)
from algorithms.llm_safe_hrl.base.heuristic_admission import file_sha256
from algorithms.llm_safe_hrl.base.llm_objective import validate_energy_only_source
from algorithms.llm_safe_hrl.base.manager_heuristics import load_manager_heuristic_library
from algorithms.llm_safe_hrl.run_context import resolve_deadline_setting
from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
from hrl_mix.train_config import ROOT_DIR
from tests.test_topk_heuristic_selection import _context, _scope, _metrics


def compile_rule(program, mode='original'):
    source = parameterized_source(program, mode)
    candidate = parse_rule_candidate(source)
    schema = candidate.parameter_schema
    parameters = schema.values_dict(schema.initial_values)
    metadata = dict(structure_hash=candidate.structure_hash, parameter_schema_hash=schema.schema_hash,
                    best_parameter_hash=canonical_json_sha256(parameters), best_parameters=parameters,
                    optimizer_config_hash=canonical_json_sha256({'test': True}),
                    parameter_diagnostics_hash=canonical_json_sha256({'status': 'not_run'}),
                    optimizer_seed=0, training_seeds=[1, 2, 3], validation_seeds=[4, 5])
    frozen = freeze_rule_source(source, schema, parameters, metadata=metadata)
    namespace = {}
    exec(frozen, namespace)
    return frozen, metadata, namespace['get_task_priority_v2']


class GPHRLTests(unittest.TestCase):
    def test_training_keeps_original_manager_environment_and_safe_flags(self):
        context = resolve_experiment_protocol('single', source_scenario='SL')
        payload = dict(deadline_setting={'ddl_code': 'L'})
        with patch('algorithms.comparisons.gp_hrl.train.validate_gp_library', return_value=(payload, context)), \
             patch('algorithms.comparisons.gp_hrl.train.validate_single_deadline_cache_paths', return_value={'SL': 'cache'}), \
             patch('algorithms.comparisons.gp_hrl.train.train') as runner:
            train_main(['--gp-library', 'gp.json', '--deadline-cache', 'SL=cache', '--optimizer-seed', '2'])
        kwargs = runner.call_args.kwargs
        self.assertNotIn('manager_factory', kwargs)
        self.assertNotIn('environment_class', kwargs)
        self.assertEqual(kwargs['config_builder'], build_gp_config)
        self.assertEqual(kwargs['checkpoint_writer'], save_gp_checkpoint)
        self.assertEqual(kwargs['source_scenario'], 'SL')
        self.assertEqual(kwargs['ddl'], 'L')
        self.assertEqual(kwargs['optimizer_seed'], 2)
        self.assertEqual(kwargs['max_episodes'], 600)
        for flag in ('safe_rl_enabled', 'safe_rl_shield_enabled', 'safe_rl_state_enabled',
                     'safe_rl_dynamic_lambda_enabled', 'safe_rl_heuristic_manager_enabled',
                     'manager_heuristic_llm_only'):
            self.assertTrue(kwargs[flag])

    def test_compiler_matches_scalar_gp_and_never_mutates_inputs(self):
        arguments = terminals_for_objective('original')
        inputs = [np.array([-2., 0., 1e-13, 2., 5.]) + i for i in range(8)]
        inputs[1] = np.array([-2., 0., 1e-13, 2., 5.])
        before = [x.copy() for x in inputs]
        for operator in OPERATORS:
            program = GPProgram((operator, arguments[0], arguments[1]), arguments)
            _, _, rule = compile_rule(program)
            expected = [program([x[i] for x in inputs]) for i in range(len(inputs[0]))]
            np.testing.assert_allclose(rule(*inputs), expected, rtol=0, atol=0)
            self.assertEqual(rule(*[np.array([]) for _ in inputs]).shape, (0,))
            self.assertEqual(rule(*[x[:1] for x in inputs]).shape, (1,))
        for current, original in zip(inputs, before):
            np.testing.assert_array_equal(current, original)

    def test_energy_only_cannot_read_safety_features(self):
        names = terminals_for_objective('energy_only')
        self.assertNotIn('slack', names)
        self.assertNotIn('uncertainty', names)
        program = GPProgram(('min_incremental_energy',), names)
        source, _, rule = compile_rule(program, 'energy_only')
        validate_energy_only_source(source)
        inputs = [np.array([2.]) for _ in range(8)]
        inputs[3] = inputs[7] = np.array([np.nan])
        np.testing.assert_array_equal(rule(*inputs), [2.])
        with self.assertRaisesRegex(ValueError, 'outside'):
            parameterized_source(GPProgram(('slack',), ('slack',)), 'energy_only')

    def test_nonfinite_gp_is_not_silently_repaired(self):
        program = GPProgram(('pdiv', 'min_exec_time', 'min_comm_time'), terminals_for_objective('original'))
        _, _, rule = compile_rule(program)
        inputs = [np.array([np.nan]) for _ in range(8)]
        self.assertTrue(np.isnan(rule(*inputs)[0]))

    def test_budget_cache_and_test_seed_guard(self):
        context = resolve_experiment_protocol('single', source_scenario='SS')
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as raw:
            directory = Path(raw)
            source = directory / 'rule.py'
            source.write_text('x=1', encoding='utf-8')
            evaluator = RuleEvaluator(directory, {}, context, OptimizerConfig(), episode_budget=3)
            with patch('algorithms.comparisons.gp_hrl.generate.evaluate_candidate', return_value={'ok': True}) as evaluate:
                evaluator.exact(source, [1, 2, 3])
                evaluator.exact(source, [1, 2, 3])
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(evaluator.counts['training_episode_count'], 3)
                self.assertEqual(evaluator.counts['cache_hits'], 1)
                with self.assertRaises(EvaluationBudgetExceeded):
                    evaluator.exact(source, [4])
                with self.assertRaisesRegex(ValueError, 'final-test'):
                    evaluator.exact(source, [201])

    def test_export_load_tamper_smoke_and_isolated_training(self):
        context = resolve_experiment_protocol('single', source_scenario='SS')
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as raw:
            directory = Path(raw)
            (directory / 'generated').mkdir()
            (directory / 'reports').mkdir()
            frozen, metadata, _ = compile_rule(GPProgram(('min_exec_time',), terminals_for_objective('original')))
            source = directory / 'generated/gp_g0_i0.py'
            source.write_text(frozen, encoding='utf-8')
            metrics = _metrics(source, metadata)
            candidate = dict(metrics=metrics, source_path=source, source_hash=file_sha256(source),
                             iteration=0, individual=0, ranking_key=topk_ranking_key(metrics),
                             program={'tokens': ['min_exec_time']})
            better_source = directory / 'generated/gp_g0_i1.py'
            better_frozen, better_metadata, _ = compile_rule(GPProgram(('min_comm_time',), terminals_for_objective('original')))
            better_source.write_text(better_frozen, encoding='utf-8')
            better_metrics = _metrics(better_source, better_metadata)
            better_metrics['fuzzy_total_energy_score'] = 380000.0
            better = dict(metrics=better_metrics, source_path=better_source,
                          source_hash=file_sha256(better_source), iteration=0, individual=1,
                          ranking_key=topk_ranking_key(better_metrics), program={'tokens': ['min_comm_time']})
            reference_path = directory / 'reference.json'
            reference = {'selection_policy': dict(policy_version='feasibility_first_topk_v1',
                         unique_structure_first=True, hard_energy_threshold=None,
                         ranking_fields=['constraint_feasible', 'max_deadline_violation_rate_across_seeds',
                         'max_fuzzy_lateness', 'fuzzy_total_energy_score', 'objective_cv_across_seeds',
                         'candidate_sha256'])}
            reference_path.write_text(json.dumps(reference), encoding='utf-8')
            # Derive the complete CEWS context using the same fixture as existing Top-K tests.
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
            path = export_library(directory, reference_path, reference, context,
                resolve_deadline_setting('T'), 1, 'original', config, [candidate, better], {})
            payload, _ = validate_gp_library(path, source_scenario='SS', ddl='T')
            library = load_manager_heuristic_library(path, include_traditional=False)
            self.assertEqual(len(library), 1)
            self.assertTrue(library[0].available)
            self.assertFalse(library[0].admitted)
            self.assertEqual(payload['generator'], 'gp')
            self.assertEqual(payload['llm_rules'][0]['source_hash'], file_sha256(better_source))
            with self.assertRaisesRegex(ValueError, 'DDL'):
                validate_gp_library(path, ddl='L')
            with patch('algorithms.comparisons.gp_hrl.train.build_train_config') as builder:
                builder.return_value.save_dir = str(directory)
                build_gp_config(manager_heuristic_manifest=str(path), source_scenario='SS', ddl='T')
                self.assertEqual(builder.call_args.kwargs['output_namespace'], 'ablations/gp_hrl')
            (directory / 'best_checkpoint_manifest.json').write_text('{"selection_key": [1]}', encoding='utf-8')
            with patch('algorithms.comparisons.gp_hrl.train.save_best_checkpoint_bundle'):
                save_gp_checkpoint(directory)
            checkpoint = json.loads((directory / 'best_checkpoint_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(checkpoint['selection_key'], [1])
            self.assertEqual(checkpoint['ablation'], 'gp_hrl')
            payload['smoke_only'] = True
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'smoke'):
                validate_gp_library(path)
            payload['smoke_only'] = False
            path.write_text(json.dumps(payload), encoding='utf-8')
            better_source.write_text(better_frozen + '\n# tampered\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                validate_gp_library(path)


if __name__ == '__main__':
    unittest.main()
