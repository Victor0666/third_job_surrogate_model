from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from algorithms.llm_safe_hrl.base.heuristic_admission import (
    CEWS_EVALUATOR_PROTOCOL_VERSION,
)
from algorithms.llm_safe_hrl.paths import LLM_ROOT
from algorithms.llm_safe_hrl.scenario_registry import (
    apply_scenario_to_problem_config,
    resolve_experiment_protocol,
)
from algorithms.llm_safe_hrl.LLM.protocol_config import (
    apply_seevo_scenario_config,
)

if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from seevo import (
    SeEvo,
    individual_comparison_key,
    parameter_feedback_summary,
)

from LLM.rule_optimization import (
    CMAESOptimizer,
    EvaluationCache,
    EvaluationCacheKey,
    OptimizationResult,
    OptimizerConfig,
    RuleValidationError,
    accumulate_cross_generation_diagnostics,
    constraint_priority_key,
    canonical_json_sha256,
    freeze_rule_source,
    generate_parameter_diagnostics,
    parse_rule_candidate,
    rank_fitness,
)

try:
    import cma as _cma  # noqa: F401
except ImportError:
    CMA_AVAILABLE = False
else:
    CMA_AVAILABLE = True


def _parameter(name, initial=1.0, lower=0.0, upper=2.0):
    return {
        "name": name,
        "initial_value": initial,
        "lower_bound": lower,
        "upper_bound": upper,
        "semantic_description": f"continuous {name}",
    }


def _source(parameters=None, expression=None):
    parameters = parameters or [_parameter("weight"), _parameter("epsilon", 0.1, 0.01, 1.0)]
    expression = expression or (
        'PARAMS["weight"] * slack / '
        '(np.mean(np.abs(slack)) + PARAMS["epsilon"])'
    )
    return (
        "import numpy as np\n"
        f"PARAMETER_SCHEMA = {parameters!r}\n".replace(
            f"{parameters!r}",
            repr({"schema_version": "rule_parameters_v1", "parameters": parameters}),
        )
        + "def get_task_priority_v2(\n"
        "    min_exec_time, min_comm_time, min_incremental_energy, slack,\n"
        "    upward_rank, remaining_work, ready_wait_time, uncertainty\n"
        "):\n"
        f"    return {expression}\n"
    )


def _metrics(energy, *, feasible=True, violation=0.0, tardiness=0.0, seed=1):
    return {
        "constraint_feasible": feasible,
        "deadline_violation_rate": violation,
        "max_deadline_violation_rate_across_seeds": violation,
        "total_lateness": tardiness,
        "constraint_secondary_violation": tardiness,
        "objective": energy,
        "fuzzy_total_energy_score": energy,
        "objective_std_across_seeds": 0.0,
        "objective_cv_across_seeds": 0.0,
        "per_seed_metrics": [
            {
                "seed": seed,
                "scenario_id": "SS",
                "constraint_feasible": feasible,
                "deadline_violation_rate": violation,
                "total_lateness": tardiness,
                "fuzzy_total_energy_score": energy,
                "objective": energy,
            }
        ],
    }


class ParameterSchemaTests(unittest.TestCase):
    def test_valid_schema_and_structure_hash_ignore_numeric_schema_changes(self):
        first = parse_rule_candidate(_source())
        changed = [_parameter("weight", 1.5, -1.0, 4.0), _parameter("epsilon", 0.2, 0.01, 2.0)]
        second = parse_rule_candidate(_source(changed))
        self.assertEqual(first.parameter_schema.names, ("weight", "epsilon"))
        self.assertEqual(first.structure_hash, second.structure_hash)

    def test_missing_field_invalid_bounds_duplicate_and_parameter_limit(self):
        missing = _parameter("weight")
        missing.pop("semantic_description")
        cases = [
            [missing, _parameter("epsilon", 0.1, 0.01, 1.0)],
            [_parameter("weight", 3.0, 0.0, 2.0), _parameter("epsilon", 0.1, 0.01, 1.0)],
            [_parameter("weight"), _parameter("weight", 0.5)],
        ]
        for parameters in cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(RuleValidationError):
                    parse_rule_candidate(_source(parameters))
        with self.assertRaises(RuleValidationError):
            parse_rule_candidate(
                _source([_parameter(f"p{index}") for index in range(4)], 'PARAMS["p0"] * slack'),
                max_parameters=3,
            )

    def test_hidden_undeclared_numeric_constant_is_rejected(self):
        with self.assertRaisesRegex(RuleValidationError, "hidden numeric constant"):
            parse_rule_candidate(
                _source(expression='PARAMS["weight"] * slack + 0.5 + PARAMS["epsilon"]')
            )

    def test_parameterized_rule_rejects_loop_and_external_io(self):
        loop_source = _source().replace(
            "    return ",
            "    for _ in slack:\n        pass\n    return ",
        )
        with self.assertRaisesRegex(RuleValidationError, "forbidden rule construct"):
            parse_rule_candidate(loop_source)
        io_source = _source().replace(
            "    return ",
            '    open("x", "w")\n    return ',
        )
        with self.assertRaisesRegex(RuleValidationError, "forbidden call"):
            parse_rule_candidate(io_source)

    def test_legacy_rule_has_no_optimizable_parameters(self):
        legacy = (
            "def get_task_priority_v2(a,b,c,d,e,f,g,h):\n"
            "    return d * 0.25\n"
        )
        candidate = parse_rule_candidate(legacy)
        self.assertIsNone(candidate.parameter_schema)
        self.assertEqual(candidate.parameterized_rule_source, legacy.strip())


class CandidateGenerationRecoveryTests(unittest.TestCase):
    def _algorithm(self):
        algorithm = object.__new__(SeEvo)
        algorithm.cfg = SimpleNamespace(
            model="mock-model",
            temperature=0.2,
            candidate_generation=SimpleNamespace(
                validation_retries=2,
                repair_temperature=0.0,
                use_validated_reference_fallback=True,
            ),
        )
        algorithm.problem = "cews_task_constructive"
        algorithm.iteration = 4
        algorithm.parameter_optimizer_config = OptimizerConfig(enabled=False)
        algorithm.candidate_repair_prompt = "repair this error: {validation_error}"
        algorithm.generation_fallback_individual = None
        return algorithm

    def test_error_directed_retry_repairs_static_validation_failure(self):
        algorithm = self._algorithm()
        invalid_source = _source(
            expression=(
                'PARAMS["weight"] * slack + 0.5 + PARAMS["epsilon"]'
            )
        )
        valid_response = "```python\n" + _source() + "```"
        with tempfile.TemporaryDirectory() as directory:
            previous_cwd = os.getcwd()
            os.chdir(directory)
            try:
                with mock.patch(
                    "seevo.multi_chat_completion",
                    return_value=[valid_response],
                ) as completion:
                    individuals = algorithm._responses_to_validated_individuals(
                        ["```python\n" + invalid_source + "```"],
                        [[{"role": "user", "content": "generate"}]],
                    )
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(len(individuals), 1)
        self.assertNotIn("candidate_validation_error", individuals[0])
        self.assertEqual(individuals[0]["candidate_generation_attempts"], 2)
        self.assertIn(
            "hidden numeric constant",
            individuals[0]["candidate_validation_history"][0],
        )
        repair_messages = completion.call_args.args[0][0]
        self.assertIn("hidden numeric constant", repair_messages[-1]["content"])

    def test_parameterized_generation_reference_matches_strict_parser(self):
        reference_path = (
            LLM_ROOT
            / "prompts"
            / "cews_task_constructive"
            / "parameterized_seed_func.txt"
        )
        raw = reference_path.read_text(encoding="utf-8")
        source = raw.split("```python", 1)[1].rsplit("```", 1)[0].strip()
        candidate = parse_rule_candidate(
            source,
            max_parameters=12,
            max_branches=6,
            max_ast_depth=18,
            max_interactions=8,
        )
        self.assertIsNotNone(candidate.parameter_schema)
        self.assertEqual(len(candidate.parameter_schema.parameters), 8)

    def test_validated_reference_fallback_is_evaluated_and_cached(self):
        algorithm = self._algorithm()
        reference_path = (
            LLM_ROOT
            / "prompts"
            / "cews_task_constructive"
            / "parameterized_seed_func.txt"
        )
        algorithm.generation_reference_func = reference_path.read_text(
            encoding="utf-8"
        )
        algorithm.case_num = [1]

        def evaluate(population, _case_num):
            population[0].update(exec_success=True, obj=7.0, metrics={})
            return population

        algorithm.evaluate_population = mock.Mock(side_effect=evaluate)
        with tempfile.TemporaryDirectory() as directory:
            previous_cwd = os.getcwd()
            os.chdir(directory)
            try:
                first = algorithm._validated_reference_fallback()
                second = algorithm._validated_reference_fallback()
            finally:
                os.chdir(previous_cwd)

        self.assertIs(first, second)
        self.assertTrue(first["is_generation_reference_fallback"])
        algorithm.evaluate_population.assert_called_once()


class FreezeTests(unittest.TestCase):
    def test_freezing_preserves_signature_values_and_determinism(self):
        candidate = parse_rule_candidate(_source())
        frozen = freeze_rule_source(
            candidate.parameterized_rule_source,
            candidate.parameter_schema,
            {"weight": 1.75, "epsilon": 0.25},
            metadata={"structure_hash": candidate.structure_hash},
        )
        tree = ast.parse(frozen)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        self.assertEqual(
            [argument.arg for argument in function.args.args],
            [
                "min_exec_time",
                "min_comm_time",
                "min_incremental_energy",
                "slack",
                "upward_rank",
                "remaining_work",
                "ready_wait_time",
                "uncertainty",
            ],
        )
        self.assertNotIn("PARAMS", frozen)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.py"
            path.write_text(frozen, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("frozen_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            inputs = [np.asarray([1.0, -2.0]) for _ in range(8)]
            first = module.get_task_priority_v2(*inputs)
            second = module.get_task_priority_v2(*inputs)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_allclose(first, 1.75 * inputs[3] / 1.75)


class ConstraintRankingTests(unittest.TestCase):
    def test_feasibility_violation_tardiness_energy_and_robustness_order(self):
        feasible = _metrics(100.0)
        infeasible_low_energy = _metrics(
            1.0, feasible=False, violation=0.01, tardiness=1.0
        )
        self.assertLess(
            constraint_priority_key(feasible),
            constraint_priority_key(infeasible_low_energy),
        )
        fewer_violations = _metrics(
            1000.0, feasible=False, violation=0.01, tardiness=10.0
        )
        more_violations = _metrics(
            0.0, feasible=False, violation=0.02, tardiness=0.0
        )
        self.assertLess(
            constraint_priority_key(fewer_violations),
            constraint_priority_key(more_violations),
        )
        less_tardy = _metrics(
            1000.0, feasible=False, violation=0.01, tardiness=2.0
        )
        self.assertLess(
            constraint_priority_key(less_tardy),
            constraint_priority_key(fewer_violations),
        )
        self.assertEqual(rank_fitness([infeasible_low_energy, feasible]), [1.0, 0.0])

        stable = dict(feasible, objective_std_across_seeds=0.1)
        unstable = dict(feasible, objective_std_across_seeds=1.0)
        self.assertLess(
            constraint_priority_key(stable),
            constraint_priority_key(unstable),
        )

    def test_near_equal_energy_uses_robustness_in_cma_key(self):
        unstable = dict(_metrics(100.0), objective_std_across_seeds=4.0)
        stable = dict(_metrics(100.004), objective_std_across_seeds=0.1)
        self.assertLess(
            constraint_priority_key(stable, performance_tolerance=0.01),
            constraint_priority_key(unstable, performance_tolerance=0.01),
        )

    def test_near_equal_performance_uses_complexity_after_robustness(self):
        simple = {
            "obj": 100.1,
            "exec_success": True,
            "metrics": _metrics(100.1),
            "performance_tolerance": 1.0,
            "complexity": {
                "branch_count": 0,
                "interaction_count": 0,
                "ast_node_count": 20,
            },
        }
        complex_rule = {
            "obj": 100.0,
            "exec_success": True,
            "metrics": _metrics(100.0),
            "performance_tolerance": 1.0,
            "complexity": {
                "branch_count": 3,
                "interaction_count": 4,
                "ast_node_count": 100,
            },
        }
        self.assertLess(
            individual_comparison_key(simple),
            individual_comparison_key(complex_rule),
        )


@unittest.skipUnless(CMA_AVAILABLE, "cma is not installed")
class CMAESSmokeTests(unittest.TestCase):
    def _run(self):
        candidate = parse_rule_candidate(_source())
        config = OptimizerConfig(
            enabled=True,
            optimizer_seed=17,
            population_size=4,
            max_generations=3,
            initial_sigma=0.2,
            stage_seed_counts={"quick": 1, "refine": 1, "confirm": 1},
            elite_fraction=0.25,
            early_stop_patience=3,
            diagnostic_perturbations=False,
        )

        def evaluator(parameters, stage, seeds):
            del stage
            energy = (
                (parameters["weight"] - 0.3) ** 2
                + (parameters["epsilon"] - 0.2) ** 2
            )
            return _metrics(energy, seed=int(seeds[0]))

        return CMAESOptimizer(config).optimize(
            candidate.parameter_schema,
            evaluator,
            train_seeds=[1],
            validation_seeds=[2],
            final_test_seeds=[100],
        )

    def test_ask_tell_is_reproducible(self):
        first = self._run()
        second = self._run()
        self.assertEqual(first.best_parameters, second.best_parameters)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.generations, 3)

    def test_final_candidate_is_re_evaluated_stage_only(self):
        candidate = parse_rule_candidate(_source())
        config = OptimizerConfig(
            enabled=True,
            optimizer_seed=11,
            population_size=4,
            max_generations=2,
            initial_sigma=0.1,
            stage_seed_counts={"quick": 1, "refine": 2, "confirm": 0},
            elite_fraction=0.25,
            early_stop_patience=9,
            diagnostic_perturbations=False,
        )

        def evaluator(parameters, stage, seeds):
            base = 1.0 if stage == "quick" else 100.0
            return _metrics(base + parameters["weight"], seed=int(seeds[0]))

        result = CMAESOptimizer(config).optimize(
            candidate.parameter_schema,
            evaluator,
            train_seeds=[1, 2],
            validation_seeds=[],
            final_test_seeds=[100],
        )
        self.assertEqual(result.final_stage, "refine")
        self.assertGreaterEqual(result.best_metrics["objective"], 100.0)
        self.assertTrue(
            all(item["stage"] == "refine" for item in result.elite_samples)
        )

    def test_single_generation_runs_refine_not_quick(self):
        candidate = parse_rule_candidate(_source())
        config = OptimizerConfig(
            enabled=True,
            optimizer_seed=5,
            population_size=3,
            max_generations=1,
            stage_seed_counts={"quick": 1, "refine": 2, "confirm": 0},
            diagnostic_perturbations=False,
        )
        result = CMAESOptimizer(config).optimize(
            candidate.parameter_schema,
            lambda parameters, stage, seeds: _metrics(
                parameters["weight"], seed=int(seeds[0])
            ),
            train_seeds=[1, 2],
            final_test_seeds=[100],
        )
        self.assertEqual(result.final_stage, "refine")
        self.assertEqual({item["stage"] for item in result.history}, {"refine"})

    def test_generation_confirm_and_diagnostics_use_batch_evaluator(self):
        candidate = parse_rule_candidate(_source())
        config = OptimizerConfig(
            enabled=True,
            optimizer_seed=5,
            population_size=4,
            max_generations=1,
            stage_seed_counts={"quick": 1, "refine": 2, "confirm": 1},
            elite_fraction=0.25,
            diagnostic_perturbations=True,
        )
        scalar_calls = []
        batch_calls = []

        def scalar_evaluator(parameters, stage, seeds):
            scalar_calls.append((parameters, stage, list(seeds)))
            return _metrics(parameters["weight"], seed=int(seeds[0]))

        def batch_evaluator(parameter_maps, stage, seeds):
            batch_calls.append((stage, len(parameter_maps), list(seeds)))
            return [
                _metrics(parameters["weight"], seed=int(seeds[0]))
                for parameters in parameter_maps
            ]

        result = CMAESOptimizer(config).optimize(
            candidate.parameter_schema,
            scalar_evaluator,
            batch_evaluator=batch_evaluator,
            train_seeds=[1, 2],
            validation_seeds=[3],
            final_test_seeds=[100],
        )

        self.assertFalse(scalar_calls)
        self.assertEqual(
            batch_calls,
            [
                ("refine", 4, [1, 2]),
                ("confirm", 1, [3]),
                ("diagnostic_baseline", 1, [1, 2]),
                ("diagnostic", 4, [1, 2]),
            ],
        )
        self.assertEqual(result.evaluations, 9)


class DiagnosticTests(unittest.TestCase):
    def test_boundary_inactivity_correlation_confidence_and_scenario_format(self):
        parameters = [
            _parameter("boundary", 1.0, 0.0, 1.0),
            _parameter("x", 0.5, 0.0, 1.0),
            _parameter("y", 0.5, 0.0, 1.0),
            _parameter("z", 0.0, -1.0, 1.0),
        ]
        candidate = parse_rule_candidate(
            _source(
                parameters,
                'PARAMS["boundary"] * slack + PARAMS["x"] * upward_rank '
                '+ PARAMS["y"] * remaining_work + PARAMS["z"]',
            )
        )
        history = []
        for index in range(6):
            values = {
                "boundary": 0.99,
                "x": 0.2 + index * 0.1,
                "y": 0.2 + index * 0.1,
                "z": 0.0,
            }
            history.append(
                {
                    "parameters": values,
                    "metrics": _metrics(10.0 + index),
                    "comparison_key": list(constraint_priority_key(_metrics(10.0 + index))),
                    "rank": float(index),
                }
            )
        optimization = OptimizationResult(
            best_parameters=history[0]["parameters"],
            best_metrics=_metrics(10.0),
            history=history,
            elite_samples=history,
            local_perturbations=[
                {
                    "parameter": "boundary",
                    "direction": 1,
                    "delta": 0.01,
                    "metrics": _metrics(
                        9.0,
                        feasible=False,
                        violation=0.1,
                        tardiness=3.0,
                    ),
                }
            ],
            generations=1,
            evaluations=6,
            stop_reason="test",
            stage_seeds={"quick": [1], "refine": [1], "confirm": []},
        )
        diagnostics = generate_parameter_diagnostics(
            candidate.parameter_schema,
            optimization,
            correlation_threshold=0.8,
        )
        self.assertGreater(
            diagnostics["boundary_analysis"][0]["evidence"]["upper_boundary_rate"],
            0.9,
        )
        z_item = next(
            item for item in diagnostics["inactivity_analysis"] if item["parameter"] == "z"
        )
        self.assertTrue(z_item["evidence"]["inactive"])
        self.assertIn("matrix", diagnostics["correlation_analysis"]["evidence"])
        self.assertTrue(
            diagnostics["correlation_analysis"]["evidence"]["flagged_pairs"]
        )
        self.assertEqual(
            diagnostics["scenario_sensitivity"][0]["confidence"], "low"
        )
        self.assertTrue(
            diagnostics["fragility_analysis"]["evidence"]["rule_fragile"]
        )

    def test_insufficient_elites_lower_correlation_confidence(self):
        candidate = parse_rule_candidate(_source())
        entry = {
            "parameters": {"weight": 1.0, "epsilon": 0.1},
            "metrics": _metrics(1.0),
            "comparison_key": list(constraint_priority_key(_metrics(1.0))),
            "rank": 0.0,
        }
        result = OptimizationResult(
            best_parameters=entry["parameters"],
            best_metrics=entry["metrics"],
            history=[entry],
            elite_samples=[entry],
            local_perturbations=[],
            generations=1,
            evaluations=1,
            stop_reason="test",
            stage_seeds={"quick": [1], "refine": [1], "confirm": []},
        )
        diagnostics = generate_parameter_diagnostics(candidate.parameter_schema, result)
        self.assertTrue(
            diagnostics["correlation_analysis"]["evidence"]["insufficient_samples"]
        )
        self.assertEqual(diagnostics["correlation_analysis"]["confidence"], "low")

    def test_multi_scenario_effects_are_compared(self):
        candidate = parse_rule_candidate(_source())
        entry = {
            "parameters": {"weight": 1.0, "epsilon": 0.1},
            "metrics": _metrics(10.0),
            "comparison_key": list(constraint_priority_key(_metrics(10.0))),
            "rank": 0.0,
        }
        baseline = _metrics(10.0)
        baseline["per_seed_metrics"] = [
            {**baseline["per_seed_metrics"][0], "scenario_id": "SS"},
            {**baseline["per_seed_metrics"][0], "scenario_id": "MS"},
        ]
        perturbed = _metrics(11.0)
        perturbed["per_seed_metrics"] = [
            {**perturbed["per_seed_metrics"][0], "scenario_id": "SS"},
            {
                **perturbed["per_seed_metrics"][0],
                "scenario_id": "MS",
                "constraint_feasible": False,
                "deadline_violation_rate": 0.2,
            },
        ]
        result = OptimizationResult(
            best_parameters=entry["parameters"],
            best_metrics=baseline,
            history=[entry],
            elite_samples=[entry],
            local_perturbations=[
                {
                    "parameter": "weight",
                    "direction": 1,
                    "delta": 0.02,
                    "metrics": perturbed,
                }
            ],
            generations=1,
            evaluations=2,
            stop_reason="test",
            stage_seeds={"quick": [1], "refine": [1], "confirm": []},
        )
        diagnostics = generate_parameter_diagnostics(
            candidate.parameter_schema,
            result,
        )
        weight = next(
            item
            for item in diagnostics["scenario_sensitivity"]
            if item["parameter"] == "weight"
        )
        self.assertFalse(weight["evidence"]["insufficient_scenarios"])
        self.assertTrue(weight["evidence"]["scenario_sensitive"])
        self.assertEqual(len(weight["evidence"]["scenario_effects"]), 2)

    def test_cross_generation_evidence_requires_repetition(self):
        diagnostics = {
            "boundary_analysis": [
                {
                    "parameter": "weight",
                    "evidence": {"persistent_boundary_contact": True},
                    "suggested_structural_action": ["replace_linear_term"],
                }
            ],
            "inactivity_analysis": [],
            "correlation_analysis": {"evidence": {"flagged_pairs": []}},
            "scenario_sensitivity": [],
            "fragility_analysis": {"evidence": {"parameter_results": []}},
        }
        summary = accumulate_cross_generation_diagnostics(
            [
                {"iteration": iteration, "diagnostics": diagnostics}
                for iteration in range(3)
            ]
        )
        self.assertTrue(summary["explicit_statistics_available"])
        self.assertTrue(summary["strong_structural_change_supported"])
        self.assertEqual(summary["stable_signals"][0]["occurrences"], 3)
        self.assertEqual(summary["stable_signals"][0]["confidence"], "high")


class EvaluationCacheTests(unittest.TestCase):
    def test_hit_config_isolation_and_exception_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EvaluationCache(path=Path(directory) / "cache.json")
            key = EvaluationCacheKey.create(
                structure_hash="a" * 64,
                parameter_vector=[0.1, 0.2],
                seed=1,
                scenario_id="SS",
                evaluation_config_hash="b" * 64,
                resource_config_hash="c" * 64,
            )
            changed = EvaluationCacheKey.create(
                structure_hash="a" * 64,
                parameter_vector=[0.1, 0.2],
                seed=1,
                scenario_id="SS",
                evaluation_config_hash="d" * 64,
                resource_config_hash="c" * 64,
            )
            cache.put(key, _metrics(1.0))
            self.assertEqual(cache.get(key)["objective"], 1.0)
            self.assertIsNone(cache.get(changed))
            cache.put(changed, {"error": "boom"}, successful=False)
            self.assertIsNone(cache.get(changed))
            cache.put_many([(changed, _metrics(2.0))])
            self.assertEqual(cache.get(changed)["objective"], 2.0)
            self.assertGreater(cache.hit_rate, 0.0)


class StagedEvaluationConfigurationTests(unittest.TestCase):
    def test_default_configuration_is_multi_seed_and_has_confirm(self):
        config = OmegaConf.load(LLM_ROOT / "cfg" / "config.yaml")
        problem = OmegaConf.load(
            LLM_ROOT / "cfg" / "problem" / "cews_task_constructive.yaml"
        )
        train = list(problem.dataset.train_seeds)
        validation = list(problem.dataset.validation_seeds)
        tests = list(problem.dataset.test_seeds)
        self.assertGreaterEqual(len(train), config.parameter_optimization.stage_seed_counts.refine)
        self.assertTrue(validation)
        self.assertFalse(set(train) & set(validation))
        self.assertFalse((set(train) | set(validation)) & set(tests))
        context = resolve_experiment_protocol(
            config.protocol,
            source_scenario=config.source_scenario,
            resource_scale=config.resource_scale,
            train_seeds=train,
            validation_seeds=validation,
        )
        self.assertEqual(context.training_scenarios, ("SS",))
        self.assertEqual(list(config.parameter_optimization.scenario_ids), [])
        self.assertGreaterEqual(
            config.parameter_optimization.max_parallel_evaluations,
            2,
        )
        self.assertEqual(
            config.parameter_optimization.max_parallel_evaluations,
            28,
        )

    def test_parameter_map_runs_same_seed_grid_for_each_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            algorithm = object.__new__(SeEvo)
            algorithm.generated_dir = directory
            algorithm.parameter_optimizer_config = OptimizerConfig(
                enabled=True,
                scenario_ids=("SS", "MS"),
                cache_enabled=False,
            )
            algorithm.parameter_evaluation_cache = EvaluationCache(enabled=False)
            algorithm.parameter_evaluation_count = 0
            algorithm.cfg = SimpleNamespace(
                problem={
                    "dataset": {"scenario": "SS"},
                    "resources": {},
                    "fuzzy": {},
                },
                timeout=10,
            )
            algorithm._evaluation_command = MethodType(
                lambda self, candidate_path, case_num, dataset_mode=None,
                scenario_id=None, config_path=None: [
                    "fake-eval",
                    str(scenario_id),
                    str(case_num[0]),
                ],
                algorithm,
            )
            candidate = parse_rule_candidate(_source())

            def fake_run(command, **_kwargs):
                scenario_id = command[1]
                seed = int(command[2])
                metrics = _metrics(10.0 + seed, seed=seed)
                metrics["scenario_id"] = scenario_id
                metrics["per_seed_metrics"][0]["scenario_id"] = scenario_id
                return SimpleNamespace(
                    returncode=0,
                    stdout="RESULT_JSON=" + json.dumps(metrics),
                    stderr="",
                )

            with mock.patch(
                "seevo.apply_seevo_scenario_config",
                side_effect=lambda config, scenario, cache_paths, **_kwargs: (
                    apply_seevo_scenario_config(
                        config,
                        scenario,
                        cache_paths,
                        require_files=False,
                    )
                ),
            ), mock.patch("seevo.subprocess.run", side_effect=fake_run) as run:
                result = algorithm._evaluate_parameter_map(
                    candidate,
                    {"weight": 1.0, "epsilon": 0.1},
                    "refine",
                    [1, 2],
                )
            self.assertEqual(run.call_count, 4)
            self.assertEqual(result["scenario_ids"], ["SS", "MS"])
            self.assertEqual(result["evaluation_context_count"], 4)
            self.assertEqual(
                {row["scenario_id"] for row in result["per_seed_metrics"]},
                {"SS", "MS"},
            )

    def test_parameter_contexts_run_in_bounded_parallel_and_keep_order(self):
        with tempfile.TemporaryDirectory() as directory:
            algorithm = object.__new__(SeEvo)
            algorithm.generated_dir = directory
            algorithm.parameter_optimizer_config = OptimizerConfig(
                enabled=True,
                scenario_ids=("SS", "MS"),
                cache_enabled=False,
                max_parallel_evaluations=2,
            )
            algorithm.parameter_evaluation_cache = EvaluationCache(enabled=False)
            algorithm.parameter_evaluation_count = 0
            algorithm.cfg = SimpleNamespace(
                problem={
                    "dataset": {"scenario": "SS"},
                    "resources": {},
                    "fuzzy": {},
                },
                timeout=10,
            )
            algorithm._evaluation_command = MethodType(
                lambda self, candidate_path, case_num, dataset_mode=None,
                scenario_id=None, config_path=None: [
                    "fake-eval",
                    str(scenario_id),
                    str(case_num[0]),
                ],
                algorithm,
            )
            candidate = parse_rule_candidate(_source())
            lock = threading.Lock()
            activity = {"active": 0, "maximum": 0}

            def fake_run(command, **_kwargs):
                with lock:
                    activity["active"] += 1
                    activity["maximum"] = max(
                        activity["maximum"], activity["active"]
                    )
                time.sleep(0.03)
                with lock:
                    activity["active"] -= 1
                scenario_id = command[1]
                seed = int(command[2])
                metrics = _metrics(10.0 + seed, seed=seed)
                metrics["scenario_id"] = scenario_id
                metrics["per_seed_metrics"][0]["scenario_id"] = scenario_id
                return SimpleNamespace(
                    returncode=0,
                    stdout="RESULT_JSON=" + json.dumps(metrics),
                    stderr="",
                )

            with mock.patch(
                "seevo.apply_seevo_scenario_config",
                side_effect=lambda config, scenario, cache_paths, **_kwargs: (
                    apply_seevo_scenario_config(
                        config,
                        scenario,
                        cache_paths,
                        require_files=False,
                    )
                ),
            ), mock.patch("seevo.subprocess.run", side_effect=fake_run) as run:
                result = algorithm._evaluate_parameter_map(
                    candidate,
                    {"weight": 1.0, "epsilon": 0.1},
                    "refine",
                    [1, 2],
                )

            self.assertEqual(run.call_count, 4)
            self.assertEqual(activity["maximum"], 2)
            self.assertEqual(
                [
                    (row["scenario_id"], row["seed"])
                    for row in result["per_seed_metrics"]
                ],
                [("SS", 1), ("SS", 2), ("MS", 1), ("MS", 2)],
            )
            self.assertEqual(algorithm.parameter_evaluation_count, 4)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["env"]["OMP_NUM_THREADS"], "1")

    def test_parameter_batch_flattens_vectors_into_one_worker_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            algorithm = object.__new__(SeEvo)
            algorithm.generated_dir = directory
            algorithm.parameter_optimizer_config = OptimizerConfig(
                enabled=True,
                scenario_ids=("SS", "MS"),
                cache_enabled=False,
                max_parallel_evaluations=28,
            )
            algorithm.parameter_evaluation_cache = EvaluationCache(enabled=False)
            algorithm.parameter_evaluation_count = 0
            algorithm.cfg = SimpleNamespace(
                problem={
                    "dataset": {"scenario": "SS"},
                    "resources": {},
                    "fuzzy": {},
                },
                timeout=10,
            )
            algorithm._evaluation_command = MethodType(
                lambda self, candidate_path, case_num, dataset_mode=None,
                scenario_id=None, config_path=None: [
                    "fake-eval",
                    str(scenario_id),
                    str(case_num[0]),
                    str(candidate_path),
                ],
                algorithm,
            )
            candidate = parse_rule_candidate(_source())
            lock = threading.Lock()
            activity = {"active": 0, "maximum": 0}

            def fake_run(command, **_kwargs):
                with lock:
                    activity["active"] += 1
                    activity["maximum"] = max(
                        activity["maximum"], activity["active"]
                    )
                time.sleep(0.03)
                with lock:
                    activity["active"] -= 1
                scenario_id = command[1]
                seed = int(command[2])
                metrics = _metrics(10.0 + seed, seed=seed)
                metrics["scenario_id"] = scenario_id
                metrics["per_seed_metrics"][0]["scenario_id"] = scenario_id
                return SimpleNamespace(
                    returncode=0,
                    stdout="RESULT_JSON=" + json.dumps(metrics),
                    stderr="",
                )

            parameter_maps = [
                {"weight": weight, "epsilon": 0.1}
                for weight in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
            ]
            with mock.patch(
                "seevo.apply_seevo_scenario_config",
                side_effect=lambda config, scenario, cache_paths, **_kwargs: (
                    apply_seevo_scenario_config(
                        config,
                        scenario,
                        cache_paths,
                        require_files=False,
                    )
                ),
            ), mock.patch("seevo.subprocess.run", side_effect=fake_run) as run:
                results = algorithm._evaluate_parameter_maps(
                    candidate,
                    parameter_maps,
                    "refine",
                    [1, 2],
                )

            self.assertEqual(run.call_count, 28)
            self.assertEqual(activity["maximum"], 28)
            self.assertEqual(len(results), 7)
            self.assertEqual(algorithm.parameter_evaluation_count, 28)
            for result in results:
                self.assertEqual(
                    [
                        (row["scenario_id"], row["seed"])
                        for row in result["per_seed_metrics"]
                    ],
                    [("SS", 1), ("SS", 2), ("MS", 1), ("MS", 2)],
                )


class StructureDeduplicationTests(unittest.TestCase):
    def test_same_structure_is_evaluated_once_per_population(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout_paths = [Path(directory) / f"result{index}.txt" for index in range(2)]
            for path in stdout_paths:
                path.write_text(
                    'RESULT_JSON={"objective": 1.0}\n',
                    encoding="utf-8",
                )
            algorithm = object.__new__(SeEvo)
            algorithm.problem = "cews_task_constructive"
            algorithm.iteration = 0
            algorithm.cfg = SimpleNamespace(timeout=1)
            algorithm.obj_type = "min"
            algorithm.parameter_optimizer_config = OptimizerConfig(enabled=False)
            calls = []

            def fake_run(individual, response_id, cases):
                del individual, cases
                calls.append(response_id)
                return SimpleNamespace(
                    communicate=lambda timeout: (None, None),
                    kill=lambda: None,
                )

            algorithm._run_code = fake_run
            source = "def get_task_priority_v2(*args):\n    return [0]\n"
            population = [
                {
                    "code": source,
                    "stdout_filepath": str(stdout_paths[index]),
                    "response_id": index,
                }
                for index in range(2)
            ]
            result = algorithm.evaluate_population(population, [0])
            self.assertEqual(calls, [0])
            self.assertTrue(result[0]["exec_success"])
            self.assertFalse(result[1]["exec_success"])
            self.assertIn("duplicate_structure_hash", result[1])


class TinyEndToEndTests(unittest.TestCase):
    @unittest.skipUnless(CMA_AVAILABLE, "cma is not installed")
    def test_template_optimization_diagnostics_and_freeze(self):
        candidate = parse_rule_candidate(_source())
        config = OptimizerConfig(
            enabled=True,
            optimizer_seed=3,
            population_size=3,
            max_generations=1,
            stage_seed_counts={"quick": 1, "refine": 1, "confirm": 0},
            elite_fraction=0.5,
            diagnostic_perturbations=True,
            diagnostic_seed_count=1,
        )

        def evaluator(parameters, stage, seeds):
            del stage
            return _metrics(
                (parameters["weight"] - 0.5) ** 2
                + (parameters["epsilon"] - 0.2) ** 2,
                seed=int(seeds[0]),
            )

        optimized = CMAESOptimizer(config).optimize(
            candidate.parameter_schema,
            evaluator,
            train_seeds=[7],
            final_test_seeds=[100],
        )
        diagnostics = generate_parameter_diagnostics(
            candidate.parameter_schema,
            optimized,
        )
        frozen = freeze_rule_source(
            candidate.parameterized_rule_source,
            candidate.parameter_schema,
            optimized.best_parameters,
            metadata={
                "structure_hash": candidate.structure_hash,
                "parameter_diagnostics_hash": "f" * 64,
            },
        )
        self.assertTrue(optimized.best_parameters)
        self.assertEqual(diagnostics["schema_version"], "parameter_diagnostics_v1")
        self.assertNotIn("PARAMS", frozen)
        json.dumps(diagnostics, allow_nan=False)

    @unittest.skipUnless(CMA_AVAILABLE, "cma is not installed")
    def test_seevo_prepares_frozen_rule_without_real_llm_or_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            algorithm = object.__new__(SeEvo)
            algorithm.problem = "cews_task_constructive"
            algorithm.mode = "train"
            algorithm.case_num = [7]
            algorithm.iteration = 2
            algorithm.generated_dir = directory
            algorithm.parameter_optimizer_config = OptimizerConfig(
                enabled=True,
                optimizer_seed=9,
                population_size=3,
                max_generations=1,
                stage_seed_counts={"quick": 1, "refine": 1, "confirm": 0},
                elite_fraction=0.5,
                diagnostic_perturbations=True,
                diagnostic_seed_count=1,
                cache_enabled=False,
            )
            algorithm.parameter_evaluation_cache = EvaluationCache(enabled=False)
            algorithm.parameter_warm_starts = {}
            algorithm.parameter_diagnostic_history = {}
            algorithm.parameter_evaluation_count = 0
            algorithm.cfg = SimpleNamespace(
                problem={
                    "dataset": {
                        "scenario": "SS",
                        "train_seeds": [7],
                        "validation_seeds": [],
                        "test_seeds": [100],
                    },
                    "resources": {},
                    "fuzzy": {},
                },
                timeout=10,
            )

            def synthetic_evaluation(self, candidate, parameters, stage, seeds):
                del self, candidate, stage
                return _metrics(
                    (parameters["weight"] - 0.4) ** 2
                    + (parameters["epsilon"] - 0.2) ** 2,
                    seed=int(seeds[0]),
                )

            algorithm._evaluate_parameter_map = MethodType(
                synthetic_evaluation,
                algorithm,
            )
            individual = {
                "code": _source(),
                "response_id": 4,
                "stdout_filepath": str(Path(directory) / "stdout.txt"),
            }
            prepared = algorithm._prepare_individual_for_evaluation(individual)
            self.assertTrue(prepared["best_parameters"])
            self.assertIn("RULE_METADATA", prepared["code"])
            self.assertNotIn("PARAMS", prepared["code"])
            self.assertTrue(Path(prepared["parameter_diagnostics_path"]).is_file())
            self.assertEqual(
                prepared["rule_candidate"]["structure_hash"],
                prepared["structure_hash"],
            )

    @unittest.skipUnless(CMA_AVAILABLE, "cma is not installed")
    def test_fixed_llm_feedback_changes_structure_and_reoptimizes(self):
        with tempfile.TemporaryDirectory() as directory:
            algorithm = object.__new__(SeEvo)
            algorithm.problem = "cews_task_constructive"
            algorithm.mode = "train"
            algorithm.case_num = [7, 8]
            algorithm.iteration = 1
            algorithm.generated_dir = directory
            algorithm.parameter_optimizer_config = OptimizerConfig(
                enabled=True,
                optimizer_seed=12,
                population_size=3,
                max_generations=1,
                stage_seed_counts={"quick": 1, "refine": 2, "confirm": 0},
                elite_fraction=0.5,
                diagnostic_perturbations=True,
                cache_enabled=False,
            )
            algorithm.parameter_evaluation_cache = EvaluationCache(enabled=False)
            algorithm.parameter_warm_starts = {}
            algorithm.parameter_diagnostic_history = {}
            algorithm.parameter_evaluation_count = 0
            algorithm.cfg = SimpleNamespace(
                problem={
                    "dataset": {
                        "scenario": "SS",
                        "train_seeds": [7, 8],
                        "validation_seeds": [],
                        "test_seeds": [100],
                    },
                    "resources": {},
                    "fuzzy": {},
                },
                timeout=10,
            )

            def synthetic_evaluation(self, candidate, parameters, stage, seeds):
                del self, candidate, stage
                return _metrics(
                    (parameters["weight"] - 0.4) ** 2
                    + (parameters["epsilon"] - 0.2) ** 2,
                    seed=int(seeds[0]),
                )

            algorithm._evaluate_parameter_map = MethodType(
                synthetic_evaluation,
                algorithm,
            )
            first_response = "```python\n" + _source() + "```"
            first = algorithm.response_to_individual(
                first_response,
                1,
                str(Path(directory) / "first_response"),
            )
            first = algorithm._prepare_individual_for_evaluation(first)
            first["counterfactual_feedback"] = {
                "summary_version": "counterfactual_feedback_v1",
                "structure_hash": first["structure_hash"],
                "high_confidence_structural_actions": [
                    {
                        "action": "add_upward_rank_remaining_work_interaction",
                        "evidence_count": 2,
                    }
                ],
                "representative_cases": [
                    {"decision_id": "SS:7:d000003", "diagnosis": "critical_path_blocked"}
                ],
                "cross_agent_conflicts": [],
                "limitations": ["local one-step evidence is not causal proof"],
            }
            first["counterfactual_max_feedback_chars"] = 2000
            first["critical_state_replay_summary"] = {
                "structure_hash": first["structure_hash"],
                "generation": 1,
                "replayed_state_count": 2,
                "verified_state_count": 2,
                "unverified_state_count": 0,
                "verified_success_count": 0,
                "verified_failure_count": 2,
                "repeated_historical_error_count": 2,
                "hard_state_failure_patterns": [
                    {"risk_category": "CRITICAL_PATH_STARVATION", "failure_count": 2}
                ],
                "resolved_state_regressions": [],
                "cross_generation_failure_patterns": [],
                "high_confidence_structural_actions": [
                    {"action": "add_upward_rank_remaining_work_interaction"}
                ],
                "medium_confidence_actions": [],
                "representative_failures": [{"state_id": "state-1"}],
                "representative_successes": [],
                "limitations": ["feature replay is not full simulation"],
            }
            first["critical_state_max_feedback_chars"] = 2000
            feedback = json.loads(parameter_feedback_summary(first))
            self.assertIn("parameter_diagnostics", feedback)
            self.assertIn("counterfactual_mechanism_feedback", feedback)
            self.assertEqual(
                feedback["counterfactual_mechanism_feedback"]
                ["high_confidence_structural_actions"][0]["action"],
                "add_upward_rank_remaining_work_interaction",
            )
            self.assertIn("critical_state_replay_feedback", feedback)
            self.assertEqual(
                feedback["critical_state_replay_feedback"]
                ["high_confidence_structural_actions"][0]["action"],
                "add_upward_rank_remaining_work_interaction",
            )

            algorithm.iteration = 2
            changed_source = _source(
                expression=(
                    'PARAMS["weight"] * slack / '
                    '(np.mean(np.abs(slack)) + PARAMS["epsilon"]) '
                    '+ PARAMS["epsilon"] * upward_rank'
                )
            )
            second = algorithm.response_to_individual(
                "```python\n" + changed_source + "```",
                2,
                str(Path(directory) / "second_response"),
            )
            second = algorithm._prepare_individual_for_evaluation(second)
            self.assertNotEqual(first["structure_hash"], second["structure_hash"])
            self.assertTrue(second["best_parameters"])
            self.assertEqual(
                second["parameter_optimization"]["final_stage"],
                "refine",
            )
            self.assertIn(
                "cross_generation_evidence",
                second["parameter_diagnostics"],
            )

    def test_evolve_invokes_final_admission_in_train_mode(self):
        algorithm = object.__new__(SeEvo)
        algorithm.mode = "train"
        algorithm.function_evals = 0
        algorithm.cfg = SimpleNamespace(max_fe=0)
        algorithm.best_code_overall = "frozen"
        algorithm.best_code_path_overall = "candidate.py"
        algorithm._finalize_best_rule_admission = mock.Mock(return_value={})
        result = algorithm.evolve()
        algorithm._finalize_best_rule_admission.assert_called_once_with()
        self.assertEqual(result, ("frozen", "candidate.py"))

    def test_final_admission_runs_all_scenarios_and_registers_frozen_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem_dir = root / "problems" / "cews_task_constructive"
            generated_dir = problem_dir / "generated"
            generated_dir.mkdir(parents=True)
            manifest_path = problem_dir / "safe_heuristic_library_resS.json"
            context = resolve_experiment_protocol(
                "multi",
                source_scenario=None,
                resource_scale="S",
                train_seeds=(1, 2, 3),
                validation_seeds=(4, 5),
            )
            admission_template_path = (
                LLM_ROOT
                / "cfg"
                / "problem"
                / "cews_task_constructive_hrl_ss_admission.yaml"
            )
            admission_cfg = apply_scenario_to_problem_config(
                OmegaConf.to_container(
                    OmegaConf.load(admission_template_path),
                    resolve=True,
                ),
                "SS",
                require_files=True,
            )
            admission_cfg["experiment_protocol"] = context.identity()
            admission_cfg["admission_evaluation_scenarios"] = list(
                context.training_scenarios
            )
            admission_config_path = root / "effective_admission_config.yaml"
            OmegaConf.save(
                OmegaConf.create(admission_cfg),
                admission_config_path,
                resolve=True,
            )
            candidate = parse_rule_candidate(_source())
            best_parameters = candidate.parameter_schema.values_dict(
                candidate.parameter_schema.initial_values
            )
            parameter_hash = canonical_json_sha256(best_parameters)
            frozen = freeze_rule_source(
                candidate.parameterized_rule_source,
                candidate.parameter_schema,
                best_parameters,
                metadata={
                    "structure_hash": candidate.structure_hash,
                    "parameter_schema_hash": candidate.parameter_schema.schema_hash,
                    "best_parameter_hash": parameter_hash,
                    "best_parameters": best_parameters,
                    "optimizer_config_hash": "b" * 64,
                    "parameter_diagnostics_hash": "c" * 64,
                    "optimizer_seed": 7,
                    "training_seeds": [1, 2, 3],
                    "validation_seeds": [4, 5],
                },
            )
            source_path = generated_dir / "candidate_iter1_ind2.py"
            source_path.write_bytes(frozen.encode("utf-8"))
            source_hash = hashlib.sha256(frozen.encode("utf-8")).hexdigest()

            algorithm = object.__new__(SeEvo)
            algorithm.root_dir = str(root)
            algorithm.problem_dir = str(
                LLM_ROOT / "problems" / "cews_task_constructive"
            )
            algorithm.problem = "cews_task_constructive"
            algorithm.mode = "train"
            algorithm.cfg = SimpleNamespace(timeout=10)
            algorithm.experiment_protocol_context = context
            algorithm.experiment_protocol_identity = context.identity()
            algorithm.parameter_optimizer_config = OptimizerConfig(
                enabled=True,
                auto_admission_enabled=True,
                admission_required=True,
                admission_config_path=str(admission_config_path),
                admission_manifest_path=str(manifest_path),
            )
            algorithm.elitist = {
                "parameter_schema": candidate.parameter_schema.as_dict(),
                "code_path": str(source_path),
                "structure_hash": candidate.structure_hash,
                "best_parameter_hash": parameter_hash,
                "frozen_rule_hash": source_hash,
                "candidate_iteration": 1,
                "candidate_individual": 2,
            }

            def fake_run(command, **_kwargs):
                scenario_id = command[command.index("--scenario") + 1]
                seed = int(command[command.index("--cases") + 1])
                metrics = _metrics(100.0, seed=seed)
                metrics.update(
                    {
                        "completed_workflows": 50,
                        "max_fuzzy_lateness": 0.0,
                        "fuzzy_total_energy_mean": 100.0,
                        "fuzzy_total_energy_std": 0.0,
                        "candidate_source_file": source_path.name,
                        "candidate_sha256": source_hash,
                        "evaluation_config_sha256": "d" * 64,
                        "interface_valid": True,
                        "function_name": "get_task_priority_v2",
                        "evaluator_protocol_version": (
                            CEWS_EVALUATOR_PROTOCOL_VERSION
                        ),
                        "scenario_id": scenario_id,
                        "structure_hash": candidate.structure_hash,
                        "parameter_schema_hash": candidate.parameter_schema.schema_hash,
                        "best_parameter_hash": parameter_hash,
                        "best_parameters": best_parameters,
                        "optimizer_config_hash": "b" * 64,
                        "optimizer_seed": 7,
                        "frozen_rule_hash": source_hash,
                        "parameter_diagnostics_hash": "c" * 64,
                        "training_seeds": [1, 2, 3],
                        "validation_seeds": [4, 5],
                    }
                )
                metrics["per_seed_metrics"][0].update(
                    {
                        "scenario_id": scenario_id,
                        "completed_workflows": 50,
                        "max_fuzzy_lateness": 0.0,
                    }
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="RESULT_JSON=" + json.dumps(metrics),
                    stderr="",
                )

            with mock.patch("seevo.subprocess.run", side_effect=fake_run) as run:
                record = algorithm._finalize_best_rule_admission()
            self.assertTrue(record["admitted"])
            self.assertEqual(run.call_count, 9)
            self.assertEqual(
                record["evaluation"]["scenario_ids"],
                ["SS", "MS", "LS"],
            )
            self.assertEqual(len(record["evaluation"]["per_seed_metrics"]), 3)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["llm_rules"]), 1)


if __name__ == "__main__":
    unittest.main()
