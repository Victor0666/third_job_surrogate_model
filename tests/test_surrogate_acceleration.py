from __future__ import annotations

import builtins
import json
import hashlib
import os
from pathlib import Path
import pickle
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from rule_optimization import (
    CMAESOptimizer,
    DiagnosticReplayGateConfig,
    OptimizerConfig,
    freeze_rule_source,
    parse_rule_candidate,
)
from seevo import SeEvo
from surrogate import (
    SurrogateConfig,
    SurrogateContext,
    SurrogateDataset,
    SurrogateManager,
    SurrogatePrediction,
    DecisionTraceCapture,
    ReplayVerdict,
    extract_parameter_features,
    extract_structure_features,
    replay_frozen_rule,
)
from surrogate.config import (
    ActiveLearningConfig,
    FailSafeConfig,
    ParameterGateConfig,
    StructureGateConfig,
    WarmupConfig,
)
from surrogate.features import PARAMETER_FEATURE_NAMES, STRUCTURE_FEATURE_NAMES
from surrogate.models import ExtraTreesMetricModel


def _source() -> str:
    return '''import numpy as np
PARAMETER_SCHEMA = {"schema_version": "rule_parameters_v1", "parameters": [
    {"name": "weight", "initial_value": 1.0, "lower_bound": 0.1,
     "upper_bound": 2.0, "semantic_description": "slack weight",
     "parameter_type": "float", "transform": "identity"},
    {"name": "epsilon", "initial_value": 0.1, "lower_bound": 0.01,
     "upper_bound": 1.0, "semantic_description": "stability",
     "parameter_type": "float", "transform": "log"},
]}
def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy,
        slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    return PARAMS["weight"] * slack / (np.abs(slack) + PARAMS["epsilon"])
'''


def _metrics(energy: float, feasible: bool = True, violation: float = 0.0) -> dict:
    return {
        "constraint_feasible": feasible,
        "deadline_violation_rate": violation,
        "max_deadline_violation_rate_across_seeds": violation,
        "constraint_violation": violation,
        "total_lateness": violation * 10.0,
        "constraint_secondary_violation": violation * 10.0,
        "objective": float(energy),
        "fuzzy_total_energy_score": float(energy),
        "objective_std_across_seeds": 0.1,
        "objective_cv_across_seeds": 0.01,
        "per_seed_metrics": [],
    }


def _context(tag: str = "a") -> SurrogateContext:
    return SurrogateContext(
        schema_version="llm_surrogate_v1",
        simulator_fingerprint="sim-" + tag,
        evaluation_config_hash="eval-" + tag,
        resource_config_hash="resource-" + tag,
        protocol_identity="protocol-" + tag,
        scenario="SS",
        domain="cews",
        ddl="T",
    )


def _prediction(
    energy: float,
    *,
    feasible: bool = True,
    uncertainty: float = 0.1,
) -> SurrogatePrediction:
    return SurrogatePrediction(
        _metrics(energy, feasible=feasible, violation=0.0 if feasible else 1.0),
        {"violation": uncertainty, "lateness": uncertainty, "energy": uncertainty},
    )


class _FakeModel:
    healthy = True

    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.index = 0

    def predict(self, _features, **_kwargs):
        prediction = self.predictions[self.index % len(self.predictions)]
        self.index += 1
        return prediction


class FeatureAndDatasetTests(unittest.TestCase):
    def test_features_are_fixed_finite_and_candidate_source_is_never_executed(self):
        candidate = parse_rule_candidate(_source())
        marker = Path("SURROGATE_FEATURE_EXECUTION_MARKER")
        dangerous = SimpleNamespace(
            parameterized_rule_source=(
                "open('SURROGATE_FEATURE_EXECUTION_MARKER', 'w')\n" + _source()
            ),
            complexity=candidate.complexity,
            parameter_schema=candidate.parameter_schema,
        )
        try:
            structure = extract_structure_features(dangerous)
            parameters = extract_parameter_features(
                dangerous,
                {"weight": 1.0, "epsilon": 0.1},
                _metrics(float("nan")),
            )
            self.assertEqual(structure.shape, (len(STRUCTURE_FEATURE_NAMES),))
            self.assertEqual(parameters.shape, (len(PARAMETER_FEATURE_NAMES),))
            self.assertTrue(np.isfinite(structure).all())
            self.assertTrue(np.isfinite(parameters).all())
            self.assertFalse(marker.exists())
        finally:
            marker.unlink(missing_ok=True)

    def test_dataset_accepts_only_exact_labels_and_filters_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            first = SurrogateDataset(path, _context("a"))
            first.add_exact("parameter", [1.0, 2.0], {
                "constraint_feasible": True, "violation": 0.0,
                "lateness": 0.0, "energy": 3.0,
            })
            with self.assertRaisesRegex(ValueError, "predictions"):
                first.add_exact("parameter", [1.0, 2.0], {}, label_source="surrogate")
            second = SurrogateDataset(path, _context("b"))
            self.assertEqual(second.count(), 0)
            self.assertEqual(second.ignored_context_records, 1)

    def test_dataset_concurrent_appends_remain_complete_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            dataset = SurrogateDataset(path, _context())
            def append(index):
                dataset.add_exact("parameter", [float(index)], {
                    "constraint_feasible": bool(index % 2),
                    "violation": float(index % 2),
                    "lateness": float(index),
                    "energy": float(index + 1),
                })
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(append, range(40)))
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 40)
            self.assertEqual(dataset.count(), 40)


class ModelAndManagerTests(unittest.TestCase):
    def _config(self, directory: str, **overrides) -> SurrogateConfig:
        values = dict(
            enabled=True,
            retrain_every_exact_evaluations=100,
            dataset_path=str(Path(directory) / "labels.jsonl"),
            checkpoint_path=str(Path(directory) / "model.pkl"),
            warmup=WarmupConfig(min_structures=1, min_exact_evaluations=1),
            structure_gate=StructureGateConfig(
                initial_full_cma_fraction=0.5,
                mature_full_cma_fraction=0.5,
                min_true_structures=2,
            ),
            parameter_gate=ParameterGateConfig(
                min_exact_pairs=1,
                min_true_candidates_per_generation=3,
            ),
            active_learning=ActiveLearningConfig(0.2, 0.1),
            fail_safe=FailSafeConfig(
                min_promising_recall=0.90,
                audit_window=10,
                min_audit_samples=2,
            ),
        )
        values.update(overrides)
        return SurrogateConfig(**values)

    def test_disabled_default_and_missing_sklearn_are_fail_open(self):
        self.assertFalse(SurrogateConfig().enabled)
        model = ExtraTreesMetricModel()
        with mock.patch.dict(
            sys.modules,
            {
                "sklearn": None,
                "sklearn.ensemble": None,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "scikit-learn"):
                model.fit(
                    [
                        {"features": [float(index)], "label": {
                            "constraint_feasible": bool(index % 2),
                            "violation": float(not index % 2),
                            "lateness": float(not index % 2),
                            "energy": float(index + 1),
                        }}
                        for index in range(10)
                    ]
                )

    def test_conservative_prediction_uses_regression_upper_bounds(self):
        class RegTree:
            def __init__(self, value): self.value = value
            def predict(self, row): return np.asarray([self.value])
        model = ExtraTreesMetricModel(conservative_sigma=1.0)
        model.regressors = {
            "violation": SimpleNamespace(estimators_=[RegTree(1.0), RegTree(3.0)]),
            "lateness": SimpleNamespace(estimators_=[RegTree(2.0), RegTree(4.0)]),
            "energy": SimpleNamespace(estimators_=[RegTree(10.0), RegTree(14.0)]),
        }
        model.residual_scale = {"violation": 0.5, "lateness": 0.5, "energy": 1.0}
        model.healthy = True
        prediction = model.predict([0.0])
        self.assertFalse(prediction.metrics["constraint_feasible"])
        self.assertGreater(prediction.metrics["constraint_violation"], 2.0)
        self.assertGreater(prediction.metrics["objective"], 12.0)

    def test_real_extratrees_fit_predict_and_pickle_round_trip(self):
        records = []
        for index in range(60):
            feasible = bool(index % 2)
            records.append(
                {
                    "features": [float(index % 2), float(index) / 59.0],
                    "label": {
                        "constraint_feasible": feasible,
                        "violation": 0.0 if feasible else 1.0,
                        "lateness": 0.0 if feasible else 10.0,
                        "energy": 20.0 + float(index) / 59.0,
                    },
                }
            )
        model = ExtraTreesMetricModel(random_seed=17, conservative_sigma=1.0)
        model.fit(records)
        self.assertTrue(model.healthy, model.failure_reason)
        self.assertGreaterEqual(model.validation_metrics["promising_recall"], 0.90)
        self.assertEqual(model.mode, "ranking_only")
        self.assertEqual(set(model.regressors), {"violation", "lateness", "energy"})
        prediction = model.predict([1.0, 0.5], robustness=0.2)
        values = [
            prediction.total_uncertainty,
            prediction.metrics["constraint_violation"],
            prediction.metrics["total_lateness"],
            prediction.metrics["objective"],
        ]
        self.assertTrue(np.isfinite(values).all())
        restored = pickle.loads(pickle.dumps(model))
        restored_prediction = restored.predict([1.0, 0.5], robustness=0.2)
        self.assertEqual(prediction.metrics, restored_prediction.metrics)
        self.assertEqual(prediction.uncertainty, restored_prediction.uncertainty)

    def test_all_infeasible_records_train_ranking_only_model(self):
        records = []
        for index in range(60):
            value = float(index) / 59.0
            records.append(
                {
                    "features": [value, value * value],
                    "label": {
                        "constraint_feasible": False,
                        "violation": 1.0 - 0.8 * value,
                        "lateness": 20.0 - 10.0 * value,
                        "energy": 30.0 + value,
                    },
                }
            )
        model = ExtraTreesMetricModel(random_seed=17, conservative_sigma=1.0)
        model.fit(records)
        self.assertTrue(model.healthy, model.failure_reason)
        self.assertEqual(model.mode, "ranking_only")
        self.assertNotIn("feasible_recall", model.validation_metrics)
        prediction = model.predict([0.5, 0.25])
        self.assertFalse(prediction.metrics["constraint_feasible"])
        self.assertTrue(
            np.isfinite(
                [
                    prediction.metrics["constraint_violation"],
                    prediction.metrics["total_lateness"],
                    prediction.metrics["objective"],
                ]
            ).all()
        )

    def test_all_feasible_records_also_train_ranking_only_model(self):
        records = [
            {
                "features": [float(index), float(index % 3)],
                "label": {
                    "constraint_feasible": True,
                    "violation": 0.0,
                    "lateness": 0.0,
                    "energy": float(100 - index),
                },
            }
            for index in range(30)
        ]
        model = ExtraTreesMetricModel(random_seed=9)
        model.fit(records)
        self.assertTrue(model.healthy, model.failure_reason)
        self.assertEqual(model.mode, "ranking_only")
        self.assertEqual(set(model.regressors), set(model.TARGETS))

    def test_ranking_only_model_is_validated_without_feasible_recall(self):
        class RankingOnlyModel:
            healthy = False
            mode = "ranking_only"
            failure_reason = "not_trained"
            validation_metrics = {}

            def fit(self, _records):
                self.healthy = True
                self.failure_reason = ""
                self.validation_metrics = {
                    "promising_recall": 1.0,
                    "validation_samples": 10.0,
                }

        with tempfile.TemporaryDirectory() as directory:
            manager = SurrogateManager(self._config(directory), _context())
            manager.structure_model = RankingOnlyModel()
            manager.parameter_model = RankingOnlyModel()
            with mock.patch.object(manager, "_save_checkpoint"):
                manager.retrain(force=True)
            stats = manager.stats()
        self.assertTrue(stats["models"]["structure"]["healthy"])
        self.assertEqual(
            stats["models"]["parameter"]["mode"],
            "ranking_only",
        )
        self.assertTrue(stats["gates"]["structure"]["healthy"])
        self.assertTrue(stats["gates"]["parameter"]["healthy"])

    def test_mature_structure_gate_keeps_bounded_active_learning_mix(self):
        candidate = parse_rule_candidate(_source())
        candidates = [candidate] * 8
        params = [{"weight": 1.0, "epsilon": 0.1}] * 8
        quick = [_metrics(10.0 + index) for index in range(8)]
        predictions = [
            _prediction(
                10.0 + index,
                uncertainty=9.0 if index == 7 else 0.1,
            )
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as directory:
            manager = SurrogateManager(self._config(directory), _context())
            manager.structure_model = _FakeModel(predictions)
            manager._model_ready = lambda kind: True
            first = manager.select_structures(candidates, params, quick, iteration=3)
            manager.structure_model.index = 0
            second = manager.select_structures(candidates, params, quick, iteration=3)
        self.assertTrue(first["gate_used"])
        self.assertEqual(first["selected_indices"], second["selected_indices"])
        self.assertEqual(len(first["selected_indices"]), 4)
        all_reasons = {reason for reasons in first["reasons"].values() for reason in reasons}
        self.assertIn("predicted_promising", all_reasons)
        self.assertIn("high_uncertainty", all_reasons)
        self.assertIn("deterministic_random", all_reasons)

    def test_warmup_and_nonfinite_predictions_both_release_every_candidate(self):
        candidate = parse_rule_candidate(_source())
        with tempfile.TemporaryDirectory() as directory:
            manager = SurrogateManager(self._config(directory), _context())
            warmup = manager.select_structures(
                [candidate, candidate],
                [{"weight": 1.0, "epsilon": 0.1}] * 2,
                [_metrics(1.0), _metrics(2.0)],
                iteration=0,
            )
            self.assertEqual(warmup["selected_indices"], [0, 1])
            manager.parameter_model = _FakeModel([
                _prediction(float("nan")), _prediction(2.0)
            ])
            manager._model_ready = lambda kind: True
            decision = manager.select_parameters(
                candidate,
                [{"weight": 0.5, "epsilon": 0.1}, {"weight": 1.5, "epsilon": 0.1}],
                [_metrics(1.0), _metrics(2.0)],
                generation=0,
            )
            self.assertFalse(decision["gate_used"])
            self.assertEqual(decision["selected_indices"], [0, 1])

    def test_ddl_priority_and_promising_recall_failure_disable_gate(self):
        candidate = parse_rule_candidate(_source())
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                directory,
                parameter_gate=ParameterGateConfig(1, 1),
            )
            manager = SurrogateManager(config, _context())
            manager.parameter_model = _FakeModel([
                _prediction(0.01, feasible=False),
                _prediction(100.0, feasible=True),
            ])
            manager._model_ready = lambda kind: manager.healthy
            decision = manager.select_parameters(
                candidate,
                [{"weight": 0.5, "epsilon": 0.1}, {"weight": 1.5, "epsilon": 0.1}],
                [_metrics(1.0), _metrics(1.0)],
                generation=1,
            )
            self.assertEqual(decision["selected_indices"], [1])
            manager.record_audit(promising_found=False)
            manager.record_audit(promising_found=False)
            manager.finish_audit("parameter")
            self.assertIn(
                "recall_below_threshold",
                manager.gate_disabled_reasons["parameter"],
            )
            self.assertEqual(manager.gate_disabled_reasons["structure"], "")

    def test_parameter_recall_audit_triggers_across_short_cma_runs(self):
        candidate = parse_rule_candidate(_source())
        with tempfile.TemporaryDirectory() as directory:
            manager = SurrogateManager(self._config(directory), _context())
            manager.parameter_model = _FakeModel(
                [_prediction(float(index)) for index in range(6)]
            )
            manager._model_ready = lambda kind: True
            decision = manager.select_parameters(
                candidate,
                [
                    {"weight": 0.5 + index * 0.1, "epsilon": 0.1}
                    for index in range(6)
                ],
                [_metrics(float(index)) for index in range(6)],
                iteration=9,
                generation=0,
            )
        self.assertTrue(decision["audit_generation"])
        self.assertEqual(decision["selected_indices"], list(range(6)))
        self.assertEqual(len(decision["would_select_indices"]), 3)

    def test_checkpoint_round_trip_and_context_mismatch_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            first = SurrogateManager(config, _context("a"))
            first.structure_model = _FakeModel([_prediction(1.0)])
            first.parameter_model = _FakeModel([_prediction(2.0)])
            first._save_checkpoint()
            loaded = SurrogateManager(config, _context("a"))
            self.assertIsInstance(loaded.structure_model, _FakeModel)
            mismatch = SurrogateManager(config, _context("b"))
            self.assertEqual(mismatch.disabled_reason, "checkpoint_context_mismatch")


class ParameterGateIntegrationTests(unittest.TestCase):
    class Manager:
        enabled = True
        def __init__(self, fail=False):
            self.fail = fail
            self.labels = []
        def select_parameters(self, candidate, maps, quick, generation, iteration=0):
            if self.fail:
                raise RuntimeError("model failed")
            predictions = [_prediction(100.0 + index) for index in range(len(maps))]
            return {
                "selected_indices": [0, 1, 2],
                "predictions": predictions,
                "reasons": {0: ["predicted_promising"], 1: ["high_uncertainty"],
                            2: ["deterministic_random"]},
                "gate_used": True,
                "failure_reason": "",
            }
        def record_parameter_exact(self, candidate, params, quick, refine, metadata):
            self.labels.append((params, quick, refine, metadata))

    def _run(self, manager):
        candidate = parse_rule_candidate(_source())
        schema = candidate.parameter_schema
        executed = set()

        def batch(parameter_maps, stage, seeds):
            results = []
            for params in parameter_maps:
                energy_rows = []
                for seed in seeds:
                    key = (round(params["weight"], 12), round(params["epsilon"], 12), int(seed))
                    if key not in executed:
                        executed.add(key)
                    energy_rows.append((params["weight"] - 0.75) ** 2 + seed * 0.001)
                results.append(_metrics(sum(energy_rows) / len(energy_rows)))
            return results

        config = OptimizerConfig(
            enabled=True,
            optimizer_seed=4,
            population_size=6,
            max_generations=1,
            stage_seed_counts={"quick": 1, "refine": 3, "confirm": 0},
            diagnostic_perturbations=False,
            elite_fraction=0.5,
        )
        result = CMAESOptimizer(config).optimize(
            schema,
            lambda params, stage, seeds: batch([params], stage, seeds)[0],
            batch_evaluator=batch,
            train_seeds=[1, 2, 3],
            final_test_seeds=[201, 202],
            surrogate_manager=manager,
            surrogate_candidate=candidate,
        )
        return result, executed, manager

    def test_six_by_three_refine_uses_twelve_true_seed_calls(self):
        result, executed, manager = self._run(self.Manager())
        self.assertEqual(len(executed), 12)
        self.assertEqual(result.surrogate_stats["predicted_refine_vectors"], 3)
        self.assertEqual(result.surrogate_stats["estimated_true_seed_calls"], 12)
        self.assertEqual(len(manager.labels), 3)
        self.assertNotIn(201, {key[2] for key in executed})
        self.assertNotIn(202, {key[2] for key in executed})
        surrogate_rows = [
            item for item in result.history
            if item.get("evaluation_source") == "surrogate"
        ]
        self.assertTrue(surrogate_rows)
        self.assertTrue(all(item["surrogate_uncertainty"] for item in surrogate_rows))

    def test_disabled_surrogate_preserves_six_by_three_exact_calls(self):
        result, executed, _ = self._run(None)
        self.assertEqual(len(executed), 18)
        self.assertEqual(result.surrogate_stats, {})
        self.assertTrue(all("evaluation_source" not in item for item in result.history))

    def test_surrogate_only_vectors_never_become_best_or_elite(self):
        result, _, _ = self._run(self.Manager())
        exact_parameter_sets = {
            tuple(sorted(item["parameters"].items()))
            for item in result.history
            if item.get("evaluation_source") == "exact"
        }
        self.assertIn(tuple(sorted(result.best_parameters.items())), exact_parameter_sets)
        for elite in result.elite_samples:
            self.assertEqual(elite.get("evaluation_source"), "exact")

    def test_model_exception_falls_back_to_all_exact_refine(self):
        result, executed, _ = self._run(self.Manager(fail=True))
        self.assertEqual(len(executed), 18)
        self.assertEqual(result.surrogate_stats["predicted_refine_vectors"], 0)
        self.assertEqual(result.surrogate_stats["fallback_generations"], 1)

    def test_confirm_and_diagnostics_remain_exact(self):
        candidate = parse_rule_candidate(_source())
        manager = self.Manager()
        def batch(parameter_maps, stage, seeds):
            return [_metrics(params["weight"]) for params in parameter_maps]
        result = CMAESOptimizer(OptimizerConfig(
            enabled=True,
            optimizer_seed=7,
            population_size=6,
            max_generations=1,
            stage_seed_counts={"quick": 1, "refine": 3, "confirm": 1},
            diagnostic_perturbations=True,
            elite_fraction=0.5,
        )).optimize(
            candidate.parameter_schema,
            lambda params, stage, seeds: batch([params], stage, seeds)[0],
            batch_evaluator=batch,
            train_seeds=[1, 2, 3],
            validation_seeds=[4],
            final_test_seeds=[201],
            surrogate_manager=manager,
            surrogate_candidate=candidate,
        )
        self.assertEqual(result.final_stage, "confirm")
        self.assertTrue(result.elite_samples)
        self.assertTrue(all(item["evaluation_source"] == "exact" for item in result.elite_samples))
        self.assertTrue(result.local_perturbations)
        self.assertTrue(all(item["evaluation_source"] == "exact" for item in result.local_perturbations))


class StructureWiringTests(unittest.TestCase):
    def test_screened_structure_does_not_enter_prepare(self):
        candidate = parse_rule_candidate(_source())
        algorithm = object.__new__(SeEvo)
        algorithm.iteration = 2
        algorithm.parameter_optimizer_config = OptimizerConfig(
            enabled=True,
            max_parallel_evaluations=1,
            stage_seed_counts={"quick": 1, "refine": 3, "confirm": 0},
        )
        algorithm.case_num = [1, 2, 3]
        algorithm.cfg = SimpleNamespace(problem={
            "dataset": {"train_seeds": [1, 2, 3], "validation_seeds": [4],
                        "test_seeds": [201, 202]},
        })
        class Manager:
            enabled = True
            def select_structures(self, candidates, parameter_maps, quick_metrics, iteration):
                return {
                    "selected_indices": [0, 2],
                    "reasons": {0: ["predicted_promising"], 2: ["high_uncertainty"]},
                    "gate_used": True,
                    "failure_reason": "",
                }
        algorithm.surrogate_manager = Manager()
        algorithm._evaluate_parameter_map = MethodType(
            lambda self, candidate, params, stage, seeds: _metrics(params["weight"]),
            algorithm,
        )
        population = [
            {"rule_candidate_object": candidate, "code": _source()}
            for _ in range(3)
        ]
        selected = algorithm._apply_structure_surrogate_gate(population, [0, 1, 2])
        self.assertEqual(selected, [0, 2])
        self.assertTrue(population[1]["surrogate_screened_out"])
        self.assertFalse(population[1]["exec_success"])
        self.assertEqual(population[1]["obj"], float("inf"))

    def test_disabled_manager_preserves_ids_and_performs_no_quick_evaluation(self):
        algorithm = object.__new__(SeEvo)
        algorithm.surrogate_manager = SimpleNamespace(enabled=False)
        algorithm._evaluate_parameter_map = mock.Mock(side_effect=AssertionError)
        population = [{}, {}, {}]
        self.assertEqual(
            algorithm._apply_structure_surrogate_gate(population, [0, 1, 2]),
            [0, 1, 2],
        )
        algorithm._evaluate_parameter_map.assert_not_called()


class CandidateRepairParallelTests(unittest.TestCase):
    def test_candidates_repair_concurrently_but_return_in_response_order(self):
        algorithm = object.__new__(SeEvo)
        algorithm.iteration = 3
        algorithm.problem = "cews_task_constructive"
        algorithm.candidate_repair_prompt = "repair: {validation_error}"
        algorithm.cfg = SimpleNamespace(
            model="mock",
            candidate_generation=SimpleNamespace(
                validation_retries=1,
                repair_temperature=0.0,
                candidate_repair_workers=2,
            ),
        )

        def parse(self, response, response_id, file_name=None):
            del self, file_name
            if response.startswith("bad"):
                return {
                    "code": None,
                    "response_id": response_id,
                    "candidate_validation_error": "invalid",
                }
            return {"code": response, "response_id": response_id}

        algorithm.response_to_individual = MethodType(parse, algorithm)
        lock = threading.Lock()
        activity = {"active": 0, "maximum": 0}

        def completion(_messages, _count, _model, _temperature):
            with lock:
                activity["active"] += 1
                activity["maximum"] = max(activity["maximum"], activity["active"])
            time.sleep(0.03)
            with lock:
                activity["active"] -= 1
            return ["fixed"]

        with mock.patch("seevo.multi_chat_completion", side_effect=completion):
            individuals = algorithm._responses_to_validated_individuals(
                [f"bad-{index}" for index in range(4)],
                [[{"role": "user", "content": "generate"}]],
            )
        self.assertEqual([row["response_id"] for row in individuals], list(range(4)))
        self.assertEqual(activity["maximum"], 2)
        self.assertTrue(all(row["candidate_generation_attempts"] == 2 for row in individuals))


class IndependentRecallTests(unittest.TestCase):
    def test_gate_failures_are_isolated_and_stable_audits_back_off(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SurrogateConfig(
                enabled=True,
                dataset_path=str(Path(directory) / "labels.jsonl"),
                checkpoint_path=str(Path(directory) / "model.pkl"),
                warmup=WarmupConfig(1, 1),
                structure_gate=StructureGateConfig(0.5, 0.5, 1),
                parameter_gate=ParameterGateConfig(1, 1),
                active_learning=ActiveLearningConfig(0.0, 0.0),
                fail_safe=FailSafeConfig(
                    min_promising_recall=0.9,
                    audit_window=20,
                    min_audit_samples=2,
                    audit_every_generations=2,
                    stable_audit_every_generations=6,
                    stable_audit_rounds=2,
                ),
            )
            manager = SurrogateManager(config, _context())
            for _ in range(2):
                manager.record_audit(
                    gate="structure",
                    promising_found=True,
                )
            manager.finish_audit("structure")
            manager.finish_audit("structure")
            self.assertEqual(manager.stats()["gates"]["structure"]["audit_interval"], 6)

            for _ in range(2):
                manager.record_audit(
                    gate="parameter",
                    promising_found=False,
                )
            manager.finish_audit("parameter")
            self.assertTrue(manager.gate_healthy("structure"))
            self.assertFalse(manager.gate_healthy("parameter"))


class ExactReplayTests(unittest.TestCase):
    @staticmethod
    def _rule(path: Path, source: str) -> Path:
        path.write_text(source, encoding="utf-8")
        return path

    def test_trace_replay_distinguishes_identical_divergent_and_rule_error(self):
        candidate = parse_rule_candidate(_source())
        frozen = freeze_rule_source(
            candidate.parameterized_rule_source,
            candidate.parameter_schema,
            {"weight": 1.0, "epsilon": 0.1},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_rule = self._rule(root / "baseline.py", frozen)
            divergent_rule = self._rule(
                root / "divergent.py",
                "import numpy as np\n"
                "def get_task_priority_v2(min_exec_time, min_comm_time, "
                "min_incremental_energy, slack, upward_rank, remaining_work, "
                "ready_wait_time, uncertainty):\n    return -slack\n",
            )
            invalid_rule = self._rule(
                root / "invalid.py",
                "import numpy as np\n"
                "def get_task_priority_v2(min_exec_time, min_comm_time, "
                "min_incremental_energy, slack, upward_rank, remaining_work, "
                "ready_wait_time, uncertainty):\n    return slack / (slack - slack)\n",
            )
            trace_path = root / "SS_seed1.npz"
            capture = DecisionTraceCapture(
                trace_path,
                baseline_frozen_rule_hash=hashlib.sha256(
                    frozen.encode("utf-8")
                ).hexdigest(),
                evaluation_config_sha256="a" * 64,
                scenario_id="SS",
                seed=1,
                max_mb=1,
            )
            features = {
                name: np.asarray([-2.0, 1.0], dtype=np.float64)
                for name in (
                    "min_exec_time", "min_comm_time", "min_incremental_energy",
                    "slack", "upward_rank", "remaining_work",
                    "ready_wait_time", "uncertainty",
                )
            }
            capture.record(
                [10, 11],
                {
                    "selected_task_id": 10,
                    "features": features,
                },
            )
            capture.finalize(expected_workflows=1, completed_workflows=1)

            identical = replay_frozen_rule([trace_path], baseline_rule)[("SS", 1)]
            divergent = replay_frozen_rule([trace_path], divergent_rule)[("SS", 1)]
            invalid = replay_frozen_rule([trace_path], invalid_rule)[("SS", 1)]
            self.assertTrue(identical.identical)
            self.assertEqual(identical.status, "ok")
            self.assertFalse(divergent.identical)
            self.assertEqual(divergent.first_divergence_index, 0)
            self.assertEqual(invalid.status, "rule_error")

    def test_cma_skips_only_vectors_identical_in_every_context(self):
        candidate = parse_rule_candidate(_source())
        calls = []

        def evaluator(parameters, stage, seeds):
            del parameters
            calls.append((stage, tuple(seeds)))
            return _metrics(10.0)

        def trace_capture(parameters, stage, seeds):
            del parameters
            calls.append((stage, tuple(seeds)))
            return _metrics(10.0)

        def gate(_parameters, contexts):
            return {
                context: ReplayVerdict(
                    scenario_id=context[0],
                    seed=context[1],
                    identical=True,
                    divergence_count_along_baseline=0,
                    first_divergence_index=None,
                    baseline_frozen_rule_hash="a" * 64,
                    candidate_frozen_rule_hash="b" * 64,
                    trace_sha256="c" * 64,
                )
                for context in contexts
            }

        config = OptimizerConfig(
            enabled=True,
            optimizer_seed=7,
            population_size=3,
            max_generations=1,
            stage_seed_counts={"quick": 1, "refine": 2, "confirm": 0},
            diagnostic_seed_count=2,
            diagnostic_perturbations=True,
            diagnostic_replay_gate=DiagnosticReplayGateConfig(
                enabled=True,
                replay_workers=2,
            ),
        )
        result = CMAESOptimizer(config).optimize(
            candidate.parameter_schema,
            evaluator,
            train_seeds=[1, 2],
            replay_gate=gate,
            trace_capture_evaluator=trace_capture,
            diagnostic_scenario_ids=["SS"],
        )
        self.assertEqual(result.replay_gate_stats["skipped_vector_count"], 4)
        self.assertEqual(result.replay_gate_stats["real_vector_count"], 0)
        self.assertTrue(
            all(
                item["evaluation_source"] == "exact_identical"
                for item in result.local_perturbations
            )
        )
        self.assertNotIn("diagnostic", {stage for stage, _ in calls})

        disabled_calls = []

        def disabled_evaluator(parameters, stage, seeds):
            del parameters
            disabled_calls.append((stage, tuple(seeds)))
            return _metrics(10.0)

        disabled = CMAESOptimizer(
            OptimizerConfig(
                enabled=True,
                optimizer_seed=7,
                population_size=3,
                max_generations=1,
                stage_seed_counts={"quick": 1, "refine": 2, "confirm": 0},
                diagnostic_seed_count=2,
                diagnostic_perturbations=True,
            )
        ).optimize(
            candidate.parameter_schema,
            disabled_evaluator,
            train_seeds=[1, 2],
            diagnostic_scenario_ids=["SS"],
        )
        self.assertEqual(result.best_parameters, disabled.best_parameters)
        self.assertEqual(result.best_metrics, disabled.best_metrics)
        self.assertEqual(
            [item["metrics"]["objective"] for item in result.local_perturbations],
            [item["metrics"]["objective"] for item in disabled.local_perturbations],
        )
        self.assertEqual(
            sum(stage == "diagnostic" for stage, _ in disabled_calls),
            4,
        )

    def test_one_divergent_context_forces_the_whole_vector_to_real_evaluation(self):
        candidate = parse_rule_candidate(_source())
        diagnostic_calls = []

        def evaluator(parameters, stage, seeds):
            del parameters
            if stage == "diagnostic":
                diagnostic_calls.append(tuple(seeds))
            return _metrics(10.0)

        def gate(_parameters, contexts):
            verdicts = {}
            for index, context in enumerate(contexts):
                verdicts[context] = ReplayVerdict(
                    scenario_id=context[0],
                    seed=context[1],
                    identical=index != 0,
                    divergence_count_along_baseline=int(index == 0),
                    first_divergence_index=0 if index == 0 else None,
                    baseline_frozen_rule_hash="a" * 64,
                    candidate_frozen_rule_hash="b" * 64,
                    trace_sha256="c" * 64,
                )
            return verdicts

        result = CMAESOptimizer(
            OptimizerConfig(
                enabled=True,
                optimizer_seed=7,
                population_size=3,
                max_generations=1,
                stage_seed_counts={"quick": 1, "refine": 2, "confirm": 0},
                diagnostic_seed_count=2,
                diagnostic_perturbations=True,
                diagnostic_replay_gate=DiagnosticReplayGateConfig(
                    enabled=True,
                    replay_workers=2,
                ),
            )
        ).optimize(
            candidate.parameter_schema,
            evaluator,
            train_seeds=[1, 2],
            trace_capture_evaluator=evaluator,
            replay_gate=gate,
            diagnostic_scenario_ids=["SS"],
        )

        self.assertEqual(result.replay_gate_stats["skipped_vector_count"], 0)
        self.assertEqual(result.replay_gate_stats["real_vector_count"], 4)
        self.assertEqual(len(diagnostic_calls), 4)
        self.assertTrue(
            all(
                item["evaluation_source"] == "real"
                for item in result.local_perturbations
            )
        )


if __name__ == "__main__":
    unittest.main()
