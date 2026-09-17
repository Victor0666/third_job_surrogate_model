"""Fast deterministic tests for cross-generation critical-state replay."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

import sys

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from counterfactual_feedback import (
    ARCHIVE_VERSION,
    CriticalStateArchive,
    CounterfactualConfig,
    CriticalStateReplayCache,
    CriticalStateReplayConfig,
    CriticalStateReplayer,
    DDLDiagnosticAgent,
    DiagnosticReport,
    EnergyDiagnosticAgent,
    LocalCounterfactualEstimator,
    TraceRecorder,
    UncertaintyDiagnosticAgent,
    aggregate_replay_results,
    build_critical_state_record,
    compact_replay_feedback,
    load_frozen_priority_rule,
    load_jsonl,
    state_signatures,
    strict_replay_gate_triggered,
)
from seevo import SeEvo, parameter_feedback_summary
from problems.cews_task_constructive.eval import evaluate_candidate, load_problem_config
from tests.test_counterfactual_feedback import (
    FakeEnvironment,
    metadata,
    selection_details,
)


def make_evidence(*, seed=7, scenario="SS", generation=0):
    config = CriticalStateReplayConfig(cache_enabled=False)
    cf_config = CounterfactualConfig(
        store_full_ready_features=True,
        max_alternatives_per_decision=2,
        cache_enabled=False,
    )
    environment = FakeEnvironment()
    trace = TraceRecorder(cf_config).record(
        environment,
        [0, 1, 2],
        selection_details(),
        metadata(seed=seed, scenario=scenario),
        0,
        store_full_ready_features=True,
    ).trace
    estimator = LocalCounterfactualEstimator(cf_config)
    comparisons = [estimator.compare(environment, trace, task_id) for task_id in (1, 2)]
    reports = [
        agent.analyze(trace, comparisons)
        for agent in (
            DDLDiagnosticAgent(),
            EnergyDiagnosticAgent(),
            UncertaintyDiagnosticAgent(),
        )
    ]
    record = build_critical_state_record(
        trace,
        comparisons,
        reports,
        generation=generation,
        config=config,
    )
    if record is None:
        raise AssertionError("test evidence was not admitted")
    return config, trace, comparisons, reports, record


def clone_record(record, suffix, *, seed=None, scenario=None, category=None):
    value = copy.deepcopy(record)
    value.state_id = value.state_id[:-len(suffix)] + suffix if len(suffix) < len(value.state_id) else suffix
    value.exact_signature = value.exact_signature[:-len(suffix)] + suffix
    value.state_signature = value.exact_signature
    value.semantic_signature = value.semantic_signature[:-len(suffix)] + suffix
    if seed is not None:
        value.source_seed = int(seed)
        value.source_seeds = [int(seed)]
    if scenario is not None:
        value.source_scenario_id = str(scenario)
        value.source_scenarios = [str(scenario)]
    if category is not None:
        value.risk_category = str(category)
    value.content_hash = ""
    value.refresh_hash()
    return value


def make_archive(config, path, *, train=(7, 8, 9), validation=(3,), test=(100,)):
    return CriticalStateArchive(
        config,
        path=path,
        train_seeds=train,
        validation_seeds=validation,
        test_seeds=test,
    )


def replay(record, scores, *, generation=1, structure="d" * 64):
    def rule(*_features):
        return np.asarray(scores, dtype=float)

    return CriticalStateReplayer(
        CriticalStateReplayConfig(cache_enabled=False)
    ).replay(
        record,
        rule,
        structure_hash=structure,
        frozen_rule_hash="e" * 64,
        generation=generation,
        archive_hash="f" * 64,
    )


class CriticalStateSchemaAdmissionTests(unittest.TestCase):
    def test_record_is_serializable_hashed_and_rejects_nonfinite_values(self):
        _config, _trace, _comparisons, _reports, record = make_evidence()
        payload = record.to_dict()
        self.assertEqual(payload["content_hash"], record.content_hash)
        self.assertEqual(record, type(record).from_dict(payload))
        json.dumps(payload, allow_nan=False)
        broken = copy.deepcopy(record)
        broken.ready_task_features["0"]["slack"] = float("nan")
        broken.content_hash = ""
        with self.assertRaisesRegex(ValueError, "non-finite"):
            broken.validate()

    def test_admission_requires_failure_evidence_immutability_and_quality(self):
        config, trace, comparisons, reports, _record = make_evidence()
        neutral = [
            replace(
                row,
                predicted_violation_delta=0.0,
                safety_margin_delta=0.0,
                released_critical_successor_delta=0,
                pessimistic_finish_delta=0.0,
                uncertainty_risk_delta=0.0,
            )
            for row in comparisons
        ]
        insufficient = [
            replace(
                row,
                diagnosis_code="insufficient_evidence",
                confidence="low",
                preferred_task=None,
                rejected_task=None,
            )
            for row in reports
        ]
        self.assertIsNone(
            build_critical_state_record(
                trace, neutral, insufficient, generation=0, config=config
            )
        )
        mutated = [replace(comparisons[0], state_fingerprint_after="changed")]
        self.assertIsNone(
            build_critical_state_record(
                trace, mutated, reports, generation=0, config=config
            )
        )
        low_quality = [replace(row, evidence_quality="low") for row in comparisons]
        self.assertIsNone(
            build_critical_state_record(
                trace, low_quality, reports, generation=0, config=config
            )
        )

    def test_energy_or_uncertainty_improvement_cannot_override_worse_ddl(self):
        config, trace, comparisons, reports, _record = make_evidence()
        unsafe = [
            replace(
                row,
                predicted_violation_delta=-1.0,
                safety_margin_delta=-1.0,
                released_critical_successor_delta=0,
                pessimistic_finish_delta=2.0,
                uncertainty_risk_delta=2.0,
                marginal_fuzzy_energy_delta=2.0,
            )
            for row in comparisons
        ]
        promoted = [
            replace(
                row,
                diagnosis_code="local_energy_or_uncertainty_gain",
                confidence="high",
                preferred_task=unsafe[0].alternative_task_id,
                rejected_task=trace.selected_task_id,
            )
            for row in reports
        ]
        self.assertIsNone(
            build_critical_state_record(
                trace, unsafe, promoted, generation=1, config=config
            )
        )

    def test_seed_split_is_fail_closed_and_validation_is_read_only(self):
        config, _trace, _comparisons, _reports, record = make_evidence()
        with tempfile.TemporaryDirectory() as directory:
            archive = make_archive(config, Path(directory) / "archive.json")
            stored, reason = archive.add_or_merge(record)
            self.assertIsNotNone(stored)
            self.assertEqual(reason, "added")
            validation = clone_record(record, "3", seed=3)
            stored, reason = archive.add_or_merge(validation)
            self.assertIsNone(stored)
            self.assertEqual(reason, "validation_seed_read_only")
            test = clone_record(record, "1", seed=100)
            with self.assertRaisesRegex(ValueError, "Final test seeds"):
                archive.add_or_merge(test)


class SignatureArchiveTests(unittest.TestCase):
    def test_exact_signature_tracks_scores_risk_evidence_and_config(self):
        config, trace, _comparisons, _reports, record = make_evidence()
        comparisons = [
            SeEvo._comparison_from_dict(row)
            for row in record.counterfactual_comparisons
        ]
        reports = [
            DiagnosticReport(**row)
            for evidence in (
                record.ddl_evidence,
                record.energy_evidence,
                record.uncertainty_evidence,
            )
            for row in evidence.get("reports", [])
        ]
        exact, semantic = state_signatures(
            trace,
            record.risk_category,
            config.semantic_signature_precision,
            comparisons=comparisons,
            reports=reports,
            config_hash=config.config_hash,
        )
        self.assertEqual(exact, record.exact_signature)

        score_changed = replace(
            trace,
            candidate_tasks=[
                replace(row, rule_score=row.rule_score + 1.0)
                for row in trace.candidate_tasks
            ],
        )
        risk_changed = replace(
            trace,
            criticality_score=trace.criticality_score + 1.0,
            critical_reasons=[*trace.critical_reasons, "ddl_risk_jump"],
        )
        evidence_changed = [
            replace(
                comparisons[0],
                predicted_violation_delta=comparisons[0].predicted_violation_delta + 1.0,
            ),
            *comparisons[1:],
        ]
        for changed_trace, changed_comparisons, changed_config in (
            (score_changed, comparisons, config.config_hash),
            (risk_changed, comparisons, config.config_hash),
            (trace, evidence_changed, config.config_hash),
            (trace, comparisons, "f" * 64),
        ):
            changed_exact, changed_semantic = state_signatures(
                changed_trace,
                record.risk_category,
                config.semantic_signature_precision,
                comparisons=changed_comparisons,
                reports=reports,
                config_hash=changed_config,
            )
            self.assertNotEqual(changed_exact, exact)
            self.assertEqual(changed_semantic, semantic)

    def test_semantic_signature_ignores_task_ids_but_separates_risk(self):
        config, trace, _comparisons, _reports, record = make_evidence()
        remapped = replace(
            trace,
            ready_task_ids=[10, 11, 12],
            selected_task_id=10,
            candidate_tasks=[
                replace(row, task_id=row.task_id + 10)
                for row in trace.candidate_tasks
            ],
        )
        exact, semantic = state_signatures(
            remapped, record.risk_category, config.semantic_signature_precision
        )
        self.assertNotEqual(exact, record.exact_signature)
        self.assertEqual(semantic, record.semantic_signature)
        self.assertNotEqual(
            state_signatures(trace, "UNCERTAINTY_SPIKE", config.semantic_signature_precision)[1],
            semantic,
        )

    def test_exact_duplicate_merges_without_overwriting_semantic_evidence(self):
        config, _trace, _comparisons, _reports, record = make_evidence()
        with tempfile.TemporaryDirectory() as directory:
            archive = make_archive(config, Path(directory) / "archive.json")
            archive.add_or_merge(record)
            duplicate = copy.deepcopy(record)
            duplicate.state_id = "9" * 64
            duplicate.source_seed = 8
            duplicate.source_seeds = [8]
            duplicate.content_hash = ""
            duplicate.refresh_hash()
            merged, reason = archive.add_or_merge(duplicate)
            self.assertEqual(reason, "merged_exact_duplicate")
            self.assertEqual(len(archive.records), 1)
            self.assertEqual(merged.source_seeds, [7, 8])
            distinct = clone_record(record, "8", seed=8)
            distinct.semantic_signature = record.semantic_signature
            distinct.content_hash = ""
            distinct.refresh_hash()
            archive.add_or_merge(distinct)
            self.assertEqual(len(archive.clusters[record.semantic_signature]), 2)
            archive.save()
            restored = CriticalStateArchive.load_or_create(
                config,
                path=Path(directory) / "archive.json",
                train_seeds=[7, 8, 9],
                validation_seeds=[17],
                test_seeds=[99],
            )
            self.assertEqual(restored.archive_hash, archive.archive_hash)
            self.assertEqual(restored.statistics(), archive.statistics())

    def test_exact_merge_accumulates_auditable_evidence(self):
        config, _trace, _comparisons, _reports, record = make_evidence()
        with tempfile.TemporaryDirectory() as directory:
            archive = make_archive(config, Path(directory) / "archive.json")
            archive.add_or_merge(record)
            duplicate = copy.deepcopy(record)
            duplicate.state_id = "9" * 64
            duplicate.source_seed = 8
            duplicate.source_seeds = [8]
            duplicate.ddl_evidence["reports"] = [
                *duplicate.ddl_evidence.get("reports", []),
                {"agent_name": "ddl_diagnostic_agent", "diagnosis_code": "repeat"},
            ]
            duplicate.ddl_evidence["max_counterfactual_ddl_regret"] = (
                float(record.ddl_evidence["max_counterfactual_ddl_regret"]) + 1.0
            )
            duplicate.content_hash = ""
            duplicate.refresh_hash()
            merged, reason = archive.add_or_merge(duplicate)
            self.assertEqual(reason, "merged_exact_duplicate")
            self.assertEqual(merged.source_seeds, [7, 8])
            self.assertTrue(
                any(
                    row.get("diagnosis_code") == "repeat"
                    for row in merged.ddl_evidence["reports"]
                )
            )
            self.assertEqual(
                merged.ddl_evidence["max_counterfactual_ddl_regret"],
                duplicate.ddl_evidence["max_counterfactual_ddl_regret"],
            )

    def test_semantic_cluster_capacity_is_enforced_in_archive(self):
        base_config, _trace, _comparisons, _reports, record = make_evidence()
        config = replace(
            base_config,
            global_capacity=20,
            per_category_capacity=20,
            max_states_per_scenario=20,
            max_states_per_workflow_type=20,
            max_states_per_semantic_cluster=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            archive = make_archive(config, Path(directory) / "archive.json")
            for index in range(4):
                row = clone_record(record, str(index + 1), seed=7 + index % 3)
                row.semantic_signature = record.semantic_signature
                row.content_hash = ""
                row.refresh_hash()
                archive.add_or_merge(row)
            self.assertEqual(len(archive.clusters[record.semantic_signature]), 2)
            self.assertTrue(
                any(
                    row["reason"].startswith("semantic_cluster_capacity:")
                    for row in archive.eviction_log
                )
            )

    def test_capacity_preserves_hard_and_group_diversity(self):
        base_config, _trace, _comparisons, _reports, record = make_evidence()
        config = replace(
            base_config,
            global_capacity=3,
            per_category_capacity=2,
            max_states_per_scenario=3,
            max_states_per_workflow_type=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            archive = make_archive(config, Path(directory) / "archive.json")
            hard = clone_record(record, "1", seed=7, scenario="SS")
            hard.status = "HARD"
            hard.content_hash = ""
            hard.refresh_hash()
            archive.add_or_merge(hard)
            archive.add_or_merge(clone_record(record, "2", seed=8, scenario="MS"))
            archive.add_or_merge(
                clone_record(record, "3", seed=9, scenario="LS", category="UNCERTAINTY_SPIKE")
            )
            archive.add_or_merge(clone_record(record, "4", seed=7, scenario="SS"))
            self.assertLessEqual(len(archive.records), 3)
            self.assertIn(hard.state_id, archive.records)
            self.assertGreaterEqual(len({row.source_scenario_id for row in archive.records.values()}), 2)
            self.assertGreaterEqual(len({row.risk_category for row in archive.records.values()}), 2)


class DynamicReplayTests(unittest.TestCase):
    def test_replay_statuses_direction_determinism_and_input_immutability(self):
        _config, _trace, _comparisons, _reports, record = make_evidence()
        record.preferred_task_ids = [1]
        record.dominated_task_ids = [0]
        record.unverified_task_ids = [2]
        record.verified_coverage = 2.0 / 3.0
        record.content_hash = ""
        record.refresh_hash()
        before = json.dumps(record.to_dict(), sort_keys=True)
        success = replay(record, [2.0, 0.0, 3.0])
        failure = replay(record, [0.0, 2.0, 3.0])
        partial = replay(record, [2.0, 1.0, 0.0])
        unverified = replay(record, [1.0, 2.0, 0.0])
        self.assertEqual(success.replay_status, "VERIFIED_SUCCESS")
        self.assertEqual(failure.replay_status, "VERIFIED_FAILURE")
        self.assertEqual(partial.replay_status, "PARTIAL_SUCCESS")
        self.assertEqual(unverified.replay_status, "UNVERIFIED")
        self.assertTrue(success.deterministic_output)
        self.assertEqual(before, json.dumps(record.to_dict(), sort_keys=True))
        self.assertFalse(failure.ddl_ordering_correct)

    def test_invalid_output_and_mutating_rule_are_rejected(self):
        _config, _trace, _comparisons, _reports, record = make_evidence()
        replayer = CriticalStateReplayer(CriticalStateReplayConfig(cache_enabled=False))
        cases = (
            lambda *_values: np.array([0.0, np.nan, 1.0]),
            lambda *_values: np.array([0.0, 1.0]),
        )
        for function in cases:
            result = replayer.replay(
                record,
                function,
                structure_hash="d" * 64,
                frozen_rule_hash="e" * 64,
                generation=1,
                archive_hash="f" * 64,
            )
            self.assertEqual(result.replay_status, "INVALID")

        def mutating_rule(first, *_values):
            first[0] = 999.0
            return np.array([0.0, 1.0, 2.0])

        result = replayer.replay(
            record,
            mutating_rule,
            structure_hash="d" * 64,
            frozen_rule_hash="e" * 64,
            generation=1,
            archive_hash="f" * 64,
        )
        self.assertEqual(result.replay_status, "INVALID")

    def test_dynamic_hard_resolved_regression_and_resolved_sampling(self):
        config, _trace, _comparisons, _reports, record = make_evidence()
        with tempfile.TemporaryDirectory() as directory:
            archive = make_archive(config, Path(directory) / "archive.json")
            archive.add_or_merge(record)
            first_failure = replay(record, [0.0, 2.0, 3.0], generation=1, structure="1" * 64)
            archive.update_after_replay(first_failure, replay_structure_hash="1" * 64)
            self.assertNotEqual(record.status, "HARD")
            second_failure = replay(record, [0.0, 2.0, 3.0], generation=2, structure="2" * 64)
            archive.update_after_replay(second_failure, replay_structure_hash="2" * 64)
            self.assertEqual(record.status, "HARD")
            for generation, structure in ((3, "3" * 64), (4, "4" * 64), (5, "5" * 64)):
                result = replay(record, [2.0, 0.0, 3.0], generation=generation, structure=structure)
                archive.update_after_replay(result, replay_structure_hash=structure)
            self.assertEqual(record.status, "RESOLVED")
            self.assertIn(record.state_id, [row.state_id for row in archive.sample_for_replay()])
            regression = replay(record, [0.0, 2.0, 3.0], generation=6, structure="6" * 64)
            self.assertEqual(regression.state_status_before, "RESOLVED")
            archive.update_after_replay(regression, replay_structure_hash="6" * 64)
            self.assertEqual(record.status, "ACTIVE")
            archive.compact(6 + config.dormant_generation_threshold)
            self.assertEqual(record.status, "DORMANT")

    def test_unverified_outcome_does_not_erase_hard_failure_history(self):
        config, _trace, _comparisons, _reports, record = make_evidence()
        record.preferred_task_ids = [1]
        record.dominated_task_ids = [0]
        record.unverified_task_ids = [2]
        record.verified_coverage = 2.0 / 3.0
        record.status = "HARD"
        record.failure_count = 2
        record.consecutive_failure_count = 2
        record.content_hash = ""
        record.refresh_hash()
        with tempfile.TemporaryDirectory() as directory:
            archive = make_archive(config, Path(directory) / "archive.json")
            archive.add_or_merge(record)
            unresolved = replay(
                record, [1.0, 2.0, 0.0], generation=3, structure="3" * 64
            )
            self.assertEqual(unresolved.replay_status, "UNVERIFIED")
            archive.update_after_replay(
                unresolved, replay_structure_hash="3" * 64
            )
            self.assertEqual(record.status, "HARD")
            self.assertEqual(record.consecutive_failure_count, 2)
            failure = replay(
                record, [0.0, 2.0, 3.0], generation=4, structure="4" * 64
            )
            archive.update_after_replay(failure, replay_structure_hash="4" * 64)
            self.assertEqual(record.status, "HARD")
            self.assertEqual(record.consecutive_failure_count, 3)

    def test_cache_isolated_by_generation_and_result_hash_excludes_latency(self):
        config, _trace, _comparisons, _reports, record = make_evidence()
        with tempfile.TemporaryDirectory() as directory:
            cache = CriticalStateReplayCache(True, Path(directory) / "cache.json")
            replayer = CriticalStateReplayer(config, cache=cache)

            def priority(*_values):
                return np.asarray([2.0, 0.0, 3.0])

            first = replayer.replay(
                record,
                priority,
                structure_hash="d" * 64,
                frozen_rule_hash="e" * 64,
                generation=1,
                archive_hash="f" * 64,
            )
            second = replayer.replay(
                record,
                priority,
                structure_hash="d" * 64,
                frozen_rule_hash="e" * 64,
                generation=2,
                archive_hash="f" * 64,
            )
            repeated = replayer.replay(
                record,
                priority,
                structure_hash="d" * 64,
                frozen_rule_hash="e" * 64,
                generation=2,
                archive_hash="f" * 64,
            )
            self.assertEqual((first.generation, second.generation), (1, 2))
            self.assertEqual(second.result_hash, repeated.result_hash)
            self.assertEqual(cache.hits, 1)

    def test_stratified_sampling_is_bounded_and_reproducible(self):
        base_config, _trace, _comparisons, _reports, record = make_evidence()
        config = replace(
            base_config,
            max_states_per_generation=4,
            max_hard_states_per_generation=1,
            max_new_states_per_generation=1,
            resolved_replay_fraction=0.25,
        )
        with tempfile.TemporaryDirectory() as directory:
            archive = make_archive(config, Path(directory) / "archive.json")
            statuses = ("HARD", "NEW", "ACTIVE", "RESOLVED", "ACTIVE", "NEW")
            categories = (
                "NEGATIVE_SLACK", "UNCERTAINTY_SPIKE", "READY_QUEUE_CONGESTION",
                "DDL_ENERGY_CONFLICT", "SUCCESSOR_RELEASE_BLOCKING", "RESOURCE_BOTTLENECK",
            )
            for index, (status, category) in enumerate(zip(statuses, categories), start=1):
                row = clone_record(record, str(index), seed=7 + index % 3, category=category)
                row.status = status
                row.content_hash = ""
                row.refresh_hash()
                archive.add_or_merge(row)
            first = [row.state_id for row in archive.sample_for_replay()]
            second = [row.state_id for row in archive.sample_for_replay()]
            self.assertEqual(first, second)
            self.assertLessEqual(len(first), 4)
            selected_statuses = {archive.records[state_id].status for state_id in first}
            self.assertIn("HARD", selected_statuses)
            self.assertIn("NEW", selected_statuses)
            self.assertIn("RESOLVED", selected_statuses)


class ReplayAggregationIntegrationTests(unittest.TestCase):
    def test_low_candidate_coverage_cannot_create_structural_action(self):
        config, _trace, _comparisons, _reports, record = make_evidence()
        record.preferred_task_ids = [1]
        record.dominated_task_ids = [0]
        record.unverified_task_ids = [2]
        record.verified_coverage = 0.1
        record.content_hash = ""
        record.refresh_hash()
        second = clone_record(record, "8", seed=8)
        first_result = replay(record, [0.0, 2.0, 3.0])
        second_result = replay(second, [0.0, 2.0, 3.0])
        summary = aggregate_replay_results(
            [first_result, second_result],
            [record, second],
            structure_hash="d" * 64,
            frozen_rule_hash="e" * 64,
            generation=1,
            archive_hash="f" * 64,
            config=config,
        ).to_dict()
        self.assertFalse(summary["high_confidence_structural_actions"])
        self.assertFalse(summary["medium_confidence_actions"])
        self.assertTrue(
            any("candidate coverage" in row for row in summary["limitations"])
        )

    def test_strict_gate_requires_repeated_seed_or_scenario_evidence(self):
        base_config, _trace, _comparisons, _reports, record = make_evidence()
        config = replace(base_config, strict_replay_gate=True)
        record.preferred_task_ids = [1]
        record.dominated_task_ids = [0]
        record.unverified_task_ids = [2]
        record.verified_coverage = 2.0 / 3.0
        record.failure_count = 1
        record.consecutive_failure_count = 1
        record.failure_generations = [1]
        record.replay_structure_hashes = ["1" * 64]
        record.status = "ACTIVE"
        record.content_hash = ""
        record.refresh_hash()
        current = replay(
            record, [0.0, 2.0, 3.0], generation=2, structure="2" * 64
        )
        self.assertFalse(strict_replay_gate_triggered([current], [record], config))

        second = clone_record(record, "8", seed=8)
        second.semantic_signature = record.semantic_signature
        second.content_hash = ""
        second.refresh_hash()
        peer = replay(
            second, [0.0, 2.0, 3.0], generation=2, structure="2" * 64
        )
        self.assertTrue(
            strict_replay_gate_triggered(
                [current, peer], [record, second], config
            )
        )

    def test_aggregation_requires_repeated_cross_source_evidence(self):
        base_config, _trace, _comparisons, _reports, record = make_evidence()
        config = replace(
            base_config,
            min_repeated_failure_count=2,
            high_confidence_seed_count=2,
            max_failure_examples=1,
            max_feedback_chars=900,
        )
        second = clone_record(record, "8", seed=8)
        first_result = replay(record, [0.0, 2.0, 3.0])
        second_result = replace(
            replay(second, [0.0, 2.0, 3.0]),
            state_id=second.state_id,
        )
        single = aggregate_replay_results(
            [first_result],
            [record],
            structure_hash="d" * 64,
            frozen_rule_hash="e" * 64,
            generation=1,
            archive_hash="f" * 64,
            config=config,
        ).to_dict()
        self.assertFalse(single["high_confidence_structural_actions"])
        repeated = aggregate_replay_results(
            [first_result, second_result],
            [record, second],
            structure_hash="d" * 64,
            frozen_rule_hash="e" * 64,
            generation=1,
            archive_hash="f" * 64,
            config=config,
        ).to_dict()
        self.assertTrue(repeated["high_confidence_structural_actions"])
        self.assertEqual(len(repeated["representative_failures"]), 1)
        compact = compact_replay_feedback(repeated, config.max_feedback_chars)
        self.assertLessEqual(len(json.dumps(compact, sort_keys=True)), config.max_feedback_chars)
        self.assertNotIn("weight", json.dumps(repeated).lower())

    def test_hard_cross_generation_failure_is_strong_without_peer_contamination(self):
        config, _trace, _comparisons, _reports, record = make_evidence()
        record.failure_count = 1
        record.consecutive_failure_count = 1
        record.failure_generations = [1]
        record.replay_structure_hashes = ["1" * 64]
        record.status = "ACTIVE"
        record.content_hash = ""
        record.refresh_hash()
        current = replay(record, [0.0, 2.0, 3.0], generation=2, structure="2" * 64)
        summary = aggregate_replay_results(
            [current],
            [record],
            structure_hash="2" * 64,
            frozen_rule_hash="e" * 64,
            generation=2,
            archive_hash="f" * 64,
            config=config,
        ).to_dict()
        self.assertTrue(summary["hard_state_failure_patterns"])
        self.assertTrue(summary["cross_generation_failure_patterns"])
        self.assertTrue(summary["high_confidence_structural_actions"])

    def test_small_offline_seevo_attachment_writes_archive_replay_and_summary(self):
        config, trace, comparisons, reports, record = make_evidence(generation=0)
        artifact_root = PROJECT_ROOT / "tests" / "artifacts" / "critical_state_replay_smoke"
        archive_path = artifact_root / "archive" / "critical_state_archive.json"
        archive = make_archive(config, archive_path)
        archive.add_or_merge(record)

        candidate = artifact_root / "frozen_rule.py"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(
            "import numpy as np\n"
            "def get_task_priority_v2(\n"
            "    min_exec_time, min_comm_time, min_incremental_energy, slack,\n"
            "    upward_rank, remaining_work, ready_wait_time, uncertainty\n"
            "):\n"
            "    return np.asarray(slack, dtype=float)\n",
            encoding="utf-8",
        )
        loaded_rule = load_frozen_priority_rule(candidate)
        self.assertEqual(int(np.argmin(loaded_rule(*[
            np.asarray([record.ready_task_features[str(task_id)][name] for task_id in record.ready_task_ids])
            for name in ("min_exec_time", "min_comm_time", "min_incremental_energy", "slack", "upward_rank", "remaining_work", "ready_wait_time", "uncertainty")
        ]))), 1)

        run_root = artifact_root / "innovation2"
        trace_path = run_root / "trace.jsonl"
        comparison_path = run_root / "comparison.jsonl"
        diagnostics_path = run_root / "diagnostics.json"
        run_manifest_path = run_root / "manifest.json"
        run_root.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
        comparison_path.write_text(
            "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in comparisons),
            encoding="utf-8",
        )
        diagnostics_path.write_text(
            json.dumps([row.to_dict() for row in reports], sort_keys=True),
            encoding="utf-8",
        )
        run_manifest_path.write_text(
            json.dumps(
                {
                    "used_test_seed": False,
                    "trace_path": str(trace_path),
                    "comparison_path": str(comparison_path),
                    "diagnostics_path": str(diagnostics_path),
                }
            ),
            encoding="utf-8",
        )
        combined_path = run_root / "combined_manifest.json"
        combined_path.write_text(
            json.dumps({"used_test_seed": False, "run_manifests": [str(run_manifest_path)]}),
            encoding="utf-8",
        )
        individual = {
            "exec_success": True,
            "parameter_schema": {"parameters": []},
            "structure_hash": "d" * 64,
            "frozen_rule_hash": "e" * 64,
            "code_path": str(candidate),
            "counterfactual_trace_manifest": str(combined_path),
            "parameter_diagnostics": {"fragility_analysis": {"confidence": "low"}},
            "counterfactual_feedback": {"high_confidence_structural_actions": []},
            "rule_candidate": {},
        }
        algorithm = object.__new__(SeEvo)
        algorithm.iteration = 1
        algorithm.generated_dir = str(artifact_root)
        algorithm.critical_state_replay_config = config
        algorithm.critical_state_archive = archive
        algorithm._attach_critical_state_replay([individual])

        self.assertTrue(archive_path.is_file())
        self.assertTrue(Path(individual["critical_state_archive_manifest"]).is_file())
        self.assertTrue(Path(individual["critical_state_replay_path"]).is_file())
        self.assertTrue(Path(individual["critical_state_replay_summary_path"]).is_file())
        self.assertEqual(individual["critical_state_replay_summary"]["used_test_seed"], False)
        self.assertTrue(individual["replay_success_state_ids"])
        prompt = json.loads(parameter_feedback_summary(individual))
        self.assertIn("critical_state_replay_feedback", prompt)
        self.assertIn("joint_evolution_focus", prompt)

    def test_real_environment_trace_is_archived_and_replayed_by_next_rule(self):
        problem_config = load_problem_config()
        problem_config["dataset"] = dict(problem_config["dataset"])
        problem_config["dataset"]["workflows_per_instance"] = 1
        scenario = str(problem_config["dataset"].get("scenario", "SS"))
        artifact_root = (
            PROJECT_ROOT / "tests" / "artifacts" / "critical_state_replay_real"
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        replay_config = CriticalStateReplayConfig(
            cache_enabled=False,
            archive_path="critical_state_replay/archive/critical_state_archive.json",
        )
        counterfactual_config = CounterfactualConfig(
            store_full_ready_features=True,
            max_traced_decisions_per_run=30,
            max_critical_decisions_per_run=6,
            max_alternatives_per_decision=3,
            cache_enabled=False,
        )

        def evaluate_rule(name, structure_hash, source, parameter_hash):
            candidate = artifact_root / f"{name}_frozen_rule.py"
            candidate.write_text(source, encoding="utf-8")
            frozen_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
            output_root = artifact_root / "innovation2" / name
            result = evaluate_candidate(
                candidate,
                problem_config,
                [1],
                counterfactual_options={
                    "config": counterfactual_config,
                    "metadata": {
                        "run_id": f"critical-state-real-{name}",
                        "structure_hash": structure_hash,
                        "frozen_rule_hash": frozen_hash,
                        "parameter_hash": parameter_hash,
                        "scenario_id": scenario,
                        "seed": 1,
                        "archive_ready_features": True,
                    },
                    "output_dir": output_root,
                },
            )
            manifests = result["counterfactual_manifests"]
            combined_path = output_root / "combined_manifest.json"
            combined_path.write_text(
                json.dumps(
                    {
                        "used_test_seed": False,
                        "run_manifests": [row["manifest_path"] for row in manifests],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return candidate, frozen_hash, combined_path

        signature = (
            "min_exec_time, min_comm_time, min_incremental_energy, slack, "
            "upward_rank, remaining_work, ready_wait_time, uncertainty"
        )
        bad_rule, bad_hash, bad_manifest = evaluate_rule(
            "historical_bad",
            "a" * 64,
            "import numpy as np\n"
            f"def get_task_priority_v2({signature}):\n"
            "    return -np.asarray(slack, dtype=float)\n",
            "b" * 64,
        )
        archive_path = (
            artifact_root
            / "critical_state_replay"
            / "archive"
            / "critical_state_archive.json"
        )
        archive = make_archive(
            replay_config,
            archive_path,
            train=(1,),
            validation=(2,),
            test=(100,),
        )
        algorithm = object.__new__(SeEvo)
        algorithm.iteration = 0
        algorithm.generated_dir = str(artifact_root)
        algorithm.critical_state_replay_config = replay_config
        algorithm.critical_state_archive = archive
        historical = {
            "exec_success": True,
            "parameter_schema": {"parameters": []},
            "structure_hash": "a" * 64,
            "frozen_rule_hash": bad_hash,
            "code_path": str(bad_rule),
            "counterfactual_trace_manifest": str(bad_manifest),
            "rule_candidate": {},
        }
        algorithm._attach_critical_state_replay([historical])
        self.assertTrue(archive.records)
        self.assertTrue(archive_path.is_file())
        self.assertTrue(all(row.source_seed == 1 for row in archive.records.values()))
        restored_algorithm = object.__new__(SeEvo)
        restored_algorithm.generated_dir = str(artifact_root)
        restored_algorithm.critical_state_replay_config = replay_config
        restored_algorithm.critical_state_archive = None
        restored_algorithm._optimization_seed_sets = lambda: ([1], [2], [100])
        restored_archive = restored_algorithm._critical_state_archive_instance()
        self.assertEqual(restored_archive.archive_hash, archive.archive_hash)
        self.assertEqual(
            set(restored_archive.records),
            set(archive.records),
        )

        improved_rule, improved_hash, improved_manifest = evaluate_rule(
            "next_improved",
            "d" * 64,
            "import numpy as np\n"
            f"def get_task_priority_v2({signature}):\n"
            "    return np.asarray(slack, dtype=float)\n",
            "e" * 64,
        )
        algorithm.iteration = 1
        improved = {
            "exec_success": True,
            "parameter_schema": {"parameters": []},
            "structure_hash": "d" * 64,
            "frozen_rule_hash": improved_hash,
            "code_path": str(improved_rule),
            "counterfactual_trace_manifest": str(improved_manifest),
            "parameter_diagnostics": {
                "schema_version": "parameter_diagnostics_v1"
            },
            "counterfactual_feedback": {},
            "rule_candidate": {},
        }
        algorithm._attach_critical_state_replay([improved])
        replay_rows = load_jsonl(improved["critical_state_replay_path"])
        self.assertTrue(replay_rows)
        self.assertTrue(all(row["output_valid"] for row in replay_rows))
        self.assertGreater(
            improved["critical_state_replay_summary"]["replayed_state_count"],
            0,
        )
        self.assertFalse(improved["critical_state_replay_summary"]["used_test_seed"])
        self.assertTrue(Path(improved["critical_state_replay_summary_path"]).is_file())
        self.assertIn(
            "critical_state_replay_feedback",
            json.loads(parameter_feedback_summary(improved)),
        )

    def test_online_manager_has_no_critical_state_dependency(self):
        source = (
            PROJECT_ROOT / "algorithms" / "llm_safe_hrl" / "base" / "manager_heuristics.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("CriticalStateArchive", source)
        self.assertNotIn("critical_state_replay", source)


if __name__ == "__main__":
    unittest.main()
