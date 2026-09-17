"""Fast tests for offline trajectory counterfactual mechanism feedback."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from base.hrl_env import HrlHeftEnv
from counterfactual_feedback import (
    CounterfactualCache,
    CounterfactualCacheKey,
    CounterfactualConfig,
    CounterfactualRunSession,
    DDLDiagnosticAgent,
    DiagnosticReport,
    EnergyDiagnosticAgent,
    FeedbackAggregator,
    LocalCounterfactualEstimator,
    TraceRecorder,
    UncertaintyDiagnosticAgent,
    compact_feedback_for_prompt,
    environment_state_fingerprint,
    load_jsonl,
    reject_test_seeds,
)
from seevo import SeEvo, parameter_feedback_summary
from rule_optimization import freeze_rule_source, parse_rule_candidate
from problems.cews_task_constructive.eval import (
    evaluate_candidate,
    load_problem_config,
)


FEATURES = {
    "min_exec_time": np.array([2.0, 2.5, 1.0]),
    "min_comm_time": np.array([1.0, 0.5, 0.2]),
    "min_incremental_energy": np.array([3.0, 3.2, 1.0]),
    "slack": np.array([5.0, -1.0, 10.0]),
    "upward_rank": np.array([2.0, 10.0, 1.0]),
    "remaining_work": np.array([10.0, 30.0, 5.0]),
    "ready_wait_time": np.array([0.0, 3.0, 1.0]),
    "uncertainty": np.array([4.0, 1.0, 8.0]),
}


class FakeEnvironment:
    """Read-only predictor surface matching the production estimator calls."""

    def __init__(self):
        self.current_time = 5.0
        self.ready_task_ids = [0, 1, 2]
        self.task_state = ["Ready", "Ready", "Ready", "unReady"]
        self.task_meta = [(0, 0), (0, 1), (0, 2), (0, 3)]
        self.task_end_time = [0.0] * 4
        self.task_children = [[], [3], [], []]
        self.task_global_parents = [[], [], [], [1]]
        self.vm_ids = [0, 1]
        self.vm_available_at = np.array([7.0, 5.0])
        self.shadow_vm_available_at = {
            "optimistic": np.array([6.0, 5.0]),
            "pessimistic": np.array([8.0, 5.0]),
        }
        self.shadow_task_start_time = {
            "optimistic": [0.0] * 4,
            "pessimistic": [0.0] * 4,
        }
        self.shadow_task_end_time = {
            "optimistic": [0.0] * 4,
            "pessimistic": [0.0] * 4,
        }
        self.host_ids = [0, 1]
        self.host_to_vm_indices = {0: [0], 1: [1]}
        self.hosts = {
            0: SimpleNamespace(total_pc=10.0, server_type="edge"),
            1: SimpleNamespace(total_pc=10.0, server_type="cloud"),
        }
        self.vms = {
            0: SimpleNamespace(pc=4.0, host_id=0),
            1: SimpleNamespace(pc=3.0, host_id=1),
        }
        tasks = [
            SimpleNamespace(
                task_id=index,
                state=self.task_state[index],
                ready_time=0.0,
                assigned_server_id=None,
                assigned_vm_pc=None,
                assigned_vm_id=None,
                start_processing_time=None,
                end_processing_time=None,
            )
            for index in range(4)
        ]
        self.workflows = [SimpleNamespace(deadline=10.0, tasks=tasks)]
        self.event_heap = [(7.0, "finish", 99, 0)]
        self._records = []
        self.shadow_records = {"optimistic": [], "pessimistic": []}
        self.assignment_history = []
        self.task_assigned_vm = {}
        self.task_assigned_host = {}
        self.total_energy = 11.0
        self.completed_workflows = 0
        self.next_arrival_idx = 1
        self.fuzzy_enabled = True

    @staticmethod
    def _task_id(task):
        return int(task)

    def idle_feasible_vm_ids(self, task):
        """Offer every VM as a candidate.

        This stub is a fixed oracle for the downstream diagnostic logic, not a
        model of the VM policy, so it deliberately does not re-implement the
        idle filter; keeping every VM candidate preserves the distinct per-task
        outcomes the agent assertions below depend on. The production idle-only
        contract is covered against the real environment in
        tests/test_cews_task_constructive.py.
        """
        del task
        return list(self.vm_ids)

    def select_host_then_vm_deterministic(self, task, candidate_vm_ids=None):
        del candidate_vm_ids
        vm_id, details = self.select_vm_deterministic(task)
        return int(self.vms[vm_id].host_id), int(vm_id), details

    def select_vm_deterministic(self, task):
        task = int(task)
        vm_id = 0 if task in {0, 2} else 1
        modal = {0: 9.0, 1: 7.0, 2: 8.0}[task]
        energy = {0: 3.0, 1: 3.2, 2: 1.0}[task]
        return vm_id, {
            "vm_id": vm_id,
            "exec_time": {0: 2.0, 1: 1.5, 2: 1.0}[task],
            "comm_time": {0: 1.0, 1: 0.5, 2: 0.2}[task],
            "queue_time": 1.0 if vm_id == 0 else 0.0,
            "predicted_finish_time": modal,
            "incremental_energy": energy,
            "fuzzy_incremental_energy_score": energy,
            "fuzzy_finish_risk": {0: 12.0, 1: 8.0, 2: 9.0}[task],
        }

    def predict_task_vm_action_risk(self, task, vm):
        del vm
        task = int(task)
        values = {
            0: (6.0, 9.0, 14.0, 12.0, 10.0, -2.0, 2.0, []),
            1: (6.0, 7.0, 8.0, 8.0, 10.0, 2.0, 8.0, [3]),
            2: (4.0, 8.0, 15.0, 9.0, 10.0, 1.0, 1.0, []),
        }
        optimistic, modal, pessimistic, risk, safe, margin, remaining, critical = values[task]
        return {
            "optimistic_finish": optimistic,
            "modal_finish": modal,
            "pessimistic_finish": pessimistic,
            "risk_finish": risk,
            "task_safe_deadline": safe,
            "predicted_violation_amount": max(0.0, -margin),
            "safety_margin": margin,
            "remaining_critical_path_risk": remaining,
            "critical_path_task_ids_modal": critical,
        }

    def estimate_incremental_energy_score(self, task, vm):
        del vm
        return {0: 3.0, 1: 3.2, 2: 1.0}[int(task)]


def selection_details():
    return {
        "selected_task_id": 0,
        "selected_index": 0,
        "ready_task_ids": [0, 1, 2],
        "features": {name: values.copy() for name, values in FEATURES.items()},
        "scores": np.array([0.0, 0.01, 0.2]),
        "ranked_task_ids": [0, 1, 2],
        "score_direction": "lower_is_higher_priority",
    }


def metadata(seed=7, scenario="SS"):
    return {
        "run_id": "smoke",
        "structure_hash": "a" * 64,
        "frozen_rule_hash": "b" * 64,
        "parameter_hash": "c" * 64,
        "scenario_id": scenario,
        "seed": seed,
    }


class TraceAndAlternativeTests(unittest.TestCase):
    def test_priority_details_preserve_actual_ascending_selection(self):
        environment = object.__new__(HrlHeftEnv)
        environment._task_id = lambda task: int(task)
        environment.build_task_features = lambda ready: {
            name: np.asarray(values) for name, values in FEATURES.items()
        }
        selected, details = environment.select_task_with_priority_rule(
            [10, 11, 12],
            lambda *arrays: np.array([2.0, -1.0, 0.0]),
            return_details=True,
        )
        self.assertEqual(selected, 11)
        self.assertEqual(details["ranked_task_ids"], [11, 12, 10])
        self.assertEqual(details["score_direction"], "lower_is_higher_priority")

    def test_trace_fields_criticality_and_alternatives_are_deterministic(self):
        config = CounterfactualConfig(max_alternatives_per_decision=3)
        recorder = TraceRecorder(config)
        environment = FakeEnvironment()
        first = recorder.record(environment, [0, 1, 2], selection_details(), metadata(), 0)
        second = TraceRecorder(config).record(
            environment, [0, 1, 2], selection_details(), metadata(), 0
        )
        self.assertTrue(first.trace.is_critical)
        self.assertEqual(first.alternative_task_ids, [1, 2])
        self.assertEqual(first.alternative_task_ids, second.alternative_task_ids)
        self.assertEqual(first.trace.ready_task_ids, [0, 1, 2])
        self.assertEqual(first.trace.selected_task_rank, 1)
        self.assertGreater(first.trace.criticality_score, 0.0)
        json.dumps(first.trace.to_dict(), allow_nan=False)

    def test_planning_pass_selects_most_critical_not_first_trigger(self):
        config = CounterfactualConfig(
            max_traced_decisions_per_run=3,
            max_critical_decisions_per_run=1,
            max_alternatives_per_decision=2,
            cache_enabled=False,
        )
        session = CounterfactualRunSession(
            config,
            metadata(),
            PROJECT_ROOT / "tests" / "artifacts" / "counterfactual_feedback_smoke",
            resource_config_hash="resource",
            final_test_seeds=[100],
            planning_only=True,
        )
        environment = FakeEnvironment()
        early = selection_details()
        late = selection_details()
        late["features"]["slack"] = np.array([10.0, -100.0, 10.0])
        late["features"]["uncertainty"] = np.array([4.0, 100.0, 8.0])
        session.observe_decision(environment, [0, 1, 2], early, 0)
        session.observe_decision(environment, [0, 1, 2], late, 1)
        self.assertEqual(session.selected_critical_decision_indices(), [1])


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.config = CounterfactualConfig(max_alternatives_per_decision=2)
        self.environment = FakeEnvironment()
        self.recorded = TraceRecorder(self.config).record(
            self.environment, [0, 1, 2], selection_details(), metadata(), 0
        )

    def test_local_estimator_is_reproducible_and_state_immutable(self):
        estimator = LocalCounterfactualEstimator(self.config)
        before = environment_state_fingerprint(self.environment)
        first = estimator.compare(self.environment, self.recorded.trace, 1)
        second = estimator.compare(self.environment, self.recorded.trace, 1)
        after = environment_state_fingerprint(self.environment)
        self.assertEqual(before, after)
        self.assertEqual(first.state_fingerprint_before, first.state_fingerprint_after)
        first_payload = first.to_dict()
        second_payload = second.to_dict()
        first_payload.pop("elapsed_ms")
        second_payload.pop("elapsed_ms")
        self.assertEqual(first_payload, second_payload)
        self.assertEqual(first.selected_outcome.vm_id, 0)
        self.assertEqual(first.alternative_outcome.vm_id, 1)
        self.assertGreater(first.predicted_violation_delta, 0.0)
        self.assertLess(first.marginal_fuzzy_energy_delta, 0.0)
        self.assertGreater(first.safety_margin_delta, 0.0)
        self.assertGreater(first.released_successor_delta, 0)
        self.assertLess(first.remaining_critical_path_delta, 0.0)

    def test_cache_key_isolated_by_state_and_config(self):
        cache = CounterfactualCache(enabled=True)
        base = CounterfactualCacheKey(
            structure_hash="a",
            frozen_rule_hash="b",
            scenario_id="SS",
            seed=1,
            decision_state_hash="state1",
            selected_task_id=0,
            alternative_task_id=1,
            counterfactual_config_hash="cfg1",
            resource_config_hash="res1",
        )
        cache.put(base, {"state_immutable": True, "value": 3})
        self.assertEqual(cache.get(base)["value"], 3)
        changed = replace(base, counterfactual_config_hash="cfg2")
        self.assertIsNone(cache.get(changed))
        cache.put(replace(base, alternative_task_id=2), {"value": 9})
        self.assertIsNone(cache.get(replace(base, alternative_task_id=2)))

    def test_uncertainty_joint_risk_uses_environment_safety_margin_once(self):
        comparison = LocalCounterfactualEstimator(self.config).compare(
            self.environment,
            self.recorded.trace,
            2,
        )
        alternative = comparison.alternative_outcome
        self.assertEqual(alternative.safety_margin, 1.0)
        self.assertEqual(alternative.remaining_critical_path, 1.0)
        self.assertEqual(alternative.uncertainty_low_slack_risk, 0.0)


class DiagnosticAgentTests(unittest.TestCase):
    def setUp(self):
        config = CounterfactualConfig(max_alternatives_per_decision=2)
        environment = FakeEnvironment()
        self.trace = TraceRecorder(config).record(
            environment, [0, 1, 2], selection_details(), metadata(), 0
        ).trace
        estimator = LocalCounterfactualEstimator(config)
        self.comparison = estimator.compare(environment, self.trace, 1)

    def test_agents_use_distinct_quantitative_evidence(self):
        ddl = DDLDiagnosticAgent().analyze(self.trace, [self.comparison])
        energy = EnergyDiagnosticAgent().analyze(self.trace, [self.comparison])
        uncertainty = UncertaintyDiagnosticAgent().analyze(self.trace, [self.comparison])
        self.assertIn("low_slack_task_deferred", ddl.diagnosis_code)
        self.assertEqual(energy.diagnosis_code, "small_energy_saving_large_ddl_regret")
        self.assertEqual(uncertainty.diagnosis_code, "modal_safe_pessimistic_violation")
        self.assertIn("selected_slack", ddl.evidence)
        self.assertIn("selected_energy", energy.evidence)
        self.assertIn("selected_pessimistic_finish", uncertainty.evidence)
        self.assertEqual({ddl.preferred_task, energy.preferred_task, uncertainty.preferred_task}, {1})

    def test_insufficient_evidence_is_explicit(self):
        for agent in (DDLDiagnosticAgent(), EnergyDiagnosticAgent(), UncertaintyDiagnosticAgent()):
            report = agent.analyze(self.trace, [])
            self.assertEqual(report.diagnosis_code, "insufficient_evidence")
            self.assertEqual(report.confidence, "low")

    def test_energy_agent_detects_avoidable_energy_only_after_ddl_is_comparable(self):
        selected = replace(
            self.comparison.selected_outcome,
            marginal_fuzzy_energy=10.0,
        )
        alternative = replace(
            self.comparison.alternative_outcome,
            marginal_fuzzy_energy=2.0,
        )
        comparable = replace(
            self.comparison,
            selected_outcome=selected,
            alternative_outcome=alternative,
            marginal_fuzzy_energy_delta=8.0,
            predicted_violation_delta=0.0,
            safety_margin_delta=0.0,
        )
        report = EnergyDiagnosticAgent().analyze(self.trace, [comparable])
        self.assertEqual(
            report.diagnosis_code,
            "avoidable_marginal_energy_or_load_cost",
        )
        self.assertEqual(report.preferred_task, comparable.alternative_task_id)

    def test_uncertainty_agent_does_not_recommend_a_worse_alternative(self):
        worse_alternative = replace(
            self.comparison.alternative_outcome,
            pessimistic_finish=20.0,
            safety_margin=-10.0,
            uncertainty_low_slack_risk=20.0,
        )
        worse = replace(
            self.comparison,
            alternative_outcome=worse_alternative,
            pessimistic_finish_delta=-6.0,
            safety_margin_delta=-8.0,
            uncertainty_risk_delta=-12.0,
        )
        report = UncertaintyDiagnosticAgent().analyze(self.trace, [worse])
        self.assertEqual(report.diagnosis_code, "modal_safe_pessimistic_violation")
        self.assertIsNone(report.preferred_task)
        self.assertIsNone(report.rejected_task)


class AggregatorTests(unittest.TestCase):
    def setUp(self):
        self.config = CounterfactualConfig(
            min_repeated_evidence=2,
            high_confidence_seed_count=2,
            high_confidence_scenario_count=2,
            max_representative_cases=1,
            max_feedback_chars=900,
        )
        environment = FakeEnvironment()
        recorded = TraceRecorder(self.config).record(
            environment, [0, 1, 2], selection_details(), metadata(), 0
        )
        self.trace = recorded.trace
        self.comparison = LocalCounterfactualEstimator(self.config).compare(
            environment, self.trace, 1
        )
        self.reports = [
            agent.analyze(self.trace, [self.comparison])
            for agent in (
                DDLDiagnosticAgent(),
                EnergyDiagnosticAgent(),
                UncertaintyDiagnosticAgent(),
            )
        ]

    def test_repeated_cross_seed_evidence_raises_confidence_without_weighted_sum(self):
        second_trace = replace(
            self.trace,
            run_id="smoke2",
            decision_id="smoke2:d000000",
            seed=8,
        )
        second_comparison = replace(
            self.comparison,
            decision_id=second_trace.decision_id,
        )
        second_reports = [
            replace(row, decision_id=second_trace.decision_id, seed=8)
            for row in self.reports
        ]
        feedback = FeedbackAggregator(self.config).aggregate(
            [*self.reports, *second_reports],
            [self.trace, second_trace],
            [self.comparison, second_comparison],
            structure_hash="a" * 64,
            frozen_rule_hash="b" * 64,
        ).to_dict()
        self.assertTrue(feedback["high_confidence_structural_actions"])
        self.assertEqual(len(feedback["representative_cases"]), 1)
        self.assertIn("energy is considered only after DDL", " ".join(feedback["limitations"]))
        compact = compact_feedback_for_prompt(feedback, self.config.max_feedback_chars)
        self.assertLessEqual(len(json.dumps(compact, sort_keys=True)), self.config.max_feedback_chars)

    def test_repeated_cross_scenario_evidence_raises_confidence(self):
        second_trace = replace(
            self.trace,
            run_id="smoke-ms",
            decision_id="smoke-ms:d000000",
            scenario_id="MS",
        )
        second_comparison = replace(
            self.comparison,
            decision_id=second_trace.decision_id,
        )
        second_reports = [
            replace(
                row,
                decision_id=second_trace.decision_id,
                scenario_id="MS",
            )
            for row in self.reports
        ]
        feedback = FeedbackAggregator(self.config).aggregate(
            [*self.reports, *second_reports],
            [self.trace, second_trace],
            [self.comparison, second_comparison],
            structure_hash="a" * 64,
            frozen_rule_hash="b" * 64,
        ).to_dict()
        self.assertTrue(feedback["high_confidence_structural_actions"])
        self.assertEqual(feedback["analyzed_runs"], 2)

    def test_single_case_cannot_trigger_high_confidence_and_conflicts_are_recorded(self):
        conflicting_energy = replace(
            self.reports[1],
            preferred_task=self.trace.selected_task_id,
            rejected_task=self.comparison.alternative_task_id,
        )
        feedback = FeedbackAggregator(self.config).aggregate(
            [self.reports[0], conflicting_energy],
            [self.trace],
            [self.comparison],
            structure_hash="a" * 64,
            frozen_rule_hash="b" * 64,
        ).to_dict()
        self.assertFalse(feedback["high_confidence_structural_actions"])
        self.assertTrue(feedback["cross_agent_conflicts"])


class IsolationAndIntegrationTests(unittest.TestCase):
    def test_final_test_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Final test seeds"):
            reject_test_seeds([7, 201], [201])
        with self.assertRaisesRegex(ValueError, "Final test seeds"):
            CounterfactualRunSession(
                CounterfactualConfig(cache_enabled=False),
                metadata(seed=201),
                PROJECT_ROOT / "tests" / "artifacts" / "counterfactual_feedback_smoke",
                resource_config_hash="resource",
                final_test_seeds=[201],
            )
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            candidate.write_text(
                "def get_task_priority_v2(a, b, c, d, e, f, g, h):\n"
                "    return a\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Final test seeds"):
                evaluate_candidate(
                    candidate,
                    load_problem_config(),
                    [201],
                    counterfactual_options={
                        "config": CounterfactualConfig(cache_enabled=False),
                        "metadata": metadata(seed=201),
                        "output_dir": directory,
                    },
                )

    def test_joint_feedback_enters_existing_reflection_payload(self):
        individual = {
            "structure_hash": "a" * 64,
            "parameter_schema_hash": "p" * 64,
            "best_parameter_hash": "q" * 64,
            "best_parameters": {"weight": 1.0},
            "parameter_diagnostics": {
                "correlation_analysis": {"evidence": {"matrix": [[1.0]]}},
            },
            "counterfactual_feedback": {
                "summary_version": "counterfactual_feedback_v1",
                "structure_hash": "a" * 64,
                "high_confidence_structural_actions": [
                    {"action": "add_uncertainty_ddl_interaction"}
                ],
                "representative_cases": [{"decision_id": "d1"}],
                "limitations": ["local evidence"],
            },
            "counterfactual_max_feedback_chars": 2000,
        }
        payload = json.loads(parameter_feedback_summary(individual))
        self.assertIn("parameter_diagnostics", payload)
        self.assertIn("counterfactual_mechanism_feedback", payload)
        self.assertNotIn(
            "matrix",
            payload["parameter_diagnostics"]["correlation_analysis"]["evidence"],
        )

    def test_seevo_selects_only_post_cma_frozen_elite_for_analysis(self):
        algorithm = object.__new__(SeEvo)
        algorithm.counterfactual_config = CounterfactualConfig(
            max_structures_per_generation=1,
            include_top_feasible=1,
            include_infeasible_low_energy=1,
        )
        analyzed = []

        def fake_analysis(self, individual):
            del self
            analyzed.append(individual["structure_hash"])
            individual["counterfactual_feedback_hash"] = "f" * 64

        algorithm._run_counterfactual_analysis = MethodType(
            fake_analysis,
            algorithm,
        )
        optimized = {
            "exec_success": True,
            "parameter_schema": {"parameters": []},
            "frozen_rule_hash": "b" * 64,
            "structure_hash": "a" * 64,
            "code_path": "frozen.py",
            "obj": 5.0,
            "metrics": {"constraint_feasible": True, "objective": 5.0},
        }
        legacy = {
            "exec_success": True,
            "parameter_schema": None,
            "structure_hash": "c" * 64,
            "code_path": "legacy.py",
            "obj": 1.0,
            "metrics": {"constraint_feasible": True, "objective": 1.0},
        }
        algorithm._attach_counterfactual_feedback([legacy, optimized])
        self.assertEqual(analyzed, ["a" * 64])
        self.assertNotIn("counterfactual_feedback_hash", legacy)

    def test_pipeline_writes_small_end_to_end_artifacts(self):
        artifact_root = (
            PROJECT_ROOT / "tests" / "artifacts" / "counterfactual_feedback_smoke"
        )
        config = CounterfactualConfig(
            max_traced_decisions_per_run=2,
            max_critical_decisions_per_run=1,
            max_alternatives_per_decision=2,
            cache_enabled=False,
            max_representative_cases=2,
        )
        environment = FakeEnvironment()
        session = CounterfactualRunSession(
            config,
            metadata(),
            artifact_root,
            resource_config_hash="resource",
            final_test_seeds=[100],
        )
        session.observe_decision(
            environment,
            [0, 1, 2],
            selection_details(),
            0,
        )
        manifest = session.finalize()
        feedback = FeedbackAggregator(config).aggregate(
            session.reports,
            session.traces,
            session.comparisons,
            structure_hash="a" * 64,
            frozen_rule_hash="b" * 64,
        ).to_dict()
        summary_path = artifact_root / "summaries" / "smoke_feedback.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(feedback, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        self.assertTrue(Path(manifest["trace_path"]).is_file())
        self.assertTrue(Path(manifest["comparison_path"]).is_file())
        self.assertTrue(Path(manifest["diagnostics_path"]).is_file())
        self.assertTrue(Path(manifest["manifest_path"]).is_file())
        self.assertTrue(summary_path.is_file())
        self.assertEqual(len(load_jsonl(manifest["comparison_path"])), 2)
        self.assertFalse(manifest["used_test_seed"])

    def test_real_environment_rule_produces_read_only_counterfactual_trace(self):
        config = load_problem_config()
        config["dataset"] = dict(config["dataset"])
        config["dataset"]["workflows_per_instance"] = 1
        artifact_root = (
            PROJECT_ROOT / "tests" / "artifacts" / "counterfactual_feedback_real"
        )
        cf_config = CounterfactualConfig(
            max_traced_decisions_per_run=10,
            max_critical_decisions_per_run=2,
            max_alternatives_per_decision=2,
            cache_enabled=False,
        )
        parameterized_source = '''import numpy as np
PARAMETER_SCHEMA = {
    "schema_version": "rule_parameters_v1",
    "parameters": [{
        "name": "weight",
        "initial_value": 1.0,
        "lower_bound": 0.1,
        "upper_bound": 2.0,
        "semantic_description": "deadline slack priority weight"
    }]
}
def get_task_priority_v2(
    min_exec_time, min_comm_time, min_incremental_energy, slack,
    upward_rank, remaining_work, ready_wait_time, uncertainty
):
    return PARAMS["weight"] * np.asarray(slack, dtype=float)
'''
        parsed = parse_rule_candidate(parameterized_source)
        frozen_source = freeze_rule_source(
            parsed.parameterized_rule_source,
            parsed.parameter_schema,
            {"weight": 1.0},
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        candidate = artifact_root / "frozen_rule.py"
        candidate.write_text(frozen_source, encoding="utf-8")
        self.assertNotIn("PARAMS", frozen_source)
        result = evaluate_candidate(
            candidate,
            config,
                [1],
            counterfactual_options={
                "config": cf_config,
                "metadata": metadata(),
                "output_dir": artifact_root,
            },
        )
        manifest = result["counterfactual_manifests"][0]
        traces = load_jsonl(manifest["trace_path"])
        comparisons = load_jsonl(manifest["comparison_path"])
        with Path(manifest["diagnostics_path"]).open(
            "r", encoding="utf-8"
        ) as handle:
            diagnostic_rows = json.load(handle)
        real_feedback = FeedbackAggregator(cf_config).aggregate(
            [DiagnosticReport(**row) for row in diagnostic_rows],
            [SeEvo._trace_from_dict(row) for row in traces],
            [SeEvo._comparison_from_dict(row) for row in comparisons],
            structure_hash="a" * 64,
            frozen_rule_hash="b" * 64,
        ).to_dict()
        real_summary_path = artifact_root / "summaries" / "real_feedback.json"
        real_summary_path.parent.mkdir(parents=True, exist_ok=True)
        real_summary_path.write_text(
            json.dumps(real_feedback, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        self.assertTrue(traces)
        self.assertTrue(any(row["is_critical"] for row in traces))
        self.assertTrue(comparisons)
        self.assertTrue(
            all(
                row["state_fingerprint_before"] == row["state_fingerprint_after"]
                for row in comparisons
            )
        )
        self.assertEqual(result["evaluation_seed_count"], 1)
        self.assertTrue(real_summary_path.is_file())

    def test_online_manager_does_not_import_counterfactual_package(self):
        manager_source = (
            PROJECT_ROOT / "algorithms" / "llm_safe_hrl" / "base" / "manager_heuristics.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("counterfactual_feedback", manager_source)
        self.assertNotIn("CMAESOptimizer", manager_source)


if __name__ == "__main__":
    unittest.main()
