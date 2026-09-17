# -*- coding: utf-8 -*-
"""Stage 16 aggregation and persistence tests for safe-HRL metrics."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from hrl_mix.safe_metrics import (
    SAFE_METRIC_CSV_FIELDS,
    SAFE_METRICS_SCHEMA_VERSION,
    SafeMetricStore,
    aggregate_safe_metric_records,
    build_episode_metric_record,
    compute_llm_assistance_metrics,
)
from hrl_mix.train_config import SafeMetricsConfig, SafeRLConfig


class _MetricEnvironment:
    def __init__(
        self,
        *,
        deadlines,
        fuzzy_finishes,
        energy_mean,
        energy_std,
        modal_energy,
        shield_records=(),
        safety_cost=0.0,
    ):
        self.done_flag = True
        self.workflows_per_episode = len(deadlines)
        self.workflows = [
            SimpleNamespace(deadline=float(value))
            for value in deadlines
        ]
        self.wf_finish_time = {
            index: float(values[1])
            for index, values in enumerate(fuzzy_finishes)
        }
        self._finishes = tuple(fuzzy_finishes)
        self._shield_records = list(shield_records)
        self._safety_cumulative_cost = float(safety_cost)
        self._energy = {
            "fuzzy_total_energy_mean": float(energy_mean),
            "fuzzy_total_energy_std": float(energy_std),
            "fuzzy_total_energy_score": float(
                energy_mean + energy_std
            ),
            "energy_modal": float(modal_energy),
        }

    def _workflow_finish_tfn(self, workflow_id):
        lower, modal, upper = self._finishes[workflow_id]
        return SimpleNamespace(
            lower=float(lower),
            modal=float(modal),
            upper=float(upper),
        )

    @staticmethod
    def fuzzy_deadline_measure(value):
        return float(0.05 * value.modal + 0.95 * value.upper)

    def get_fuzzy_energy_summary(self):
        return dict(self._energy)

    def get_safety_shield_records(self):
        return [dict(record) for record in self._shield_records]


def _safe_report(*, seed=1, score=10.0):
    env = _MetricEnvironment(
        deadlines=(10.0,),
        fuzzy_finishes=((7.0, 8.0, 8.0),),
        energy_mean=score - 1.0,
        energy_std=1.0,
        modal_energy=score - 2.0,
    )
    row = build_episode_metric_record(
        env,
        seed=seed,
        scheduling_time_seconds=1.0,
    )
    return aggregate_safe_metric_records([row])


class SafeMetricAggregationTests(unittest.TestCase):
    def setUp(self):
        self.first_env = _MetricEnvironment(
            deadlines=(10.0, 10.0),
            fuzzy_finishes=(
                (7.0, 8.0, 8.0),
                (10.0, 12.0, 12.0),
            ),
            energy_mean=90.0,
            energy_std=2.0,
            modal_energy=88.0,
            safety_cost=2.0,
            shield_records=(
                {
                    "shield_intervened": False,
                    "fallback_triggered": False,
                    "fallback_applied": False,
                    "proposed_action": 0,
                    "executed_action": 0,
                },
                {
                    "shield_intervened": True,
                    "fallback_triggered": True,
                    "fallback_applied": True,
                    "fallback_reason": (
                        "empty_safe_action_set"
                    ),
                    "modification_reason": (
                        "no_safe_action_fallback"
                    ),
                    "proposed_action": 1,
                    "executed_action": 0,
                },
            ),
        )
        self.second_env = _MetricEnvironment(
            deadlines=(10.0, 10.0),
            fuzzy_finishes=(
                (6.0, 7.0, 7.0),
                (8.0, 9.0, 9.0),
            ),
            energy_mean=110.0,
            energy_std=4.0,
            modal_energy=108.0,
            shield_records=(
                {
                    "shield_intervened": False,
                    "fallback_triggered": False,
                    "fallback_applied": False,
                    "proposed_action": 0,
                    "executed_action": 0,
                },
            ),
        )

    def _records(self):
        first = build_episode_metric_record(
            self.first_env,
            seed=11,
            scheduling_time_seconds=1.0,
            phase_records=(
                {
                    "heuristic_source": "seevo_llm",
                    "heuristic_shield_record_count": 2,
                    # The legacy aggregate deliberately counts both
                    # records; stage 16 must use the detailed, exact
                    # shield_intervened flag and count only one.
                    "heuristic_shield_intervention_count": 2,
                    "subsequent_safety_interventions": [
                        {"shield_intervened": True},
                        {
                            "shield_intervened": False,
                            "fallback_triggered": True,
                        },
                    ],
                },
                {
                    "heuristic_source": "traditional",
                    "heuristic_shield_record_count": 1,
                    "heuristic_shield_intervention_count": 0,
                },
            ),
        )
        second = build_episode_metric_record(
            self.second_env,
            seed=22,
            scheduling_time_seconds=2.0,
            phase_records=(
                {
                    "heuristic_source": "traditional",
                    "heuristic_shield_record_count": 1,
                    "heuristic_shield_intervention_count": 0,
                },
            ),
        )
        return first, second

    def test_aggregates_all_required_metric_families(self):
        report = aggregate_safe_metric_records(
            self._records(),
            q_c_prediction_error=0.25,
            q_c_prediction_error_sample_count=8,
            lambda_current=3.0,
        )

        self.assertEqual(report["evaluation_seed_count"], 2)
        self.assertEqual(report["completed_workflow_count"], 4)
        self.assertEqual(report["deadline_violation_count"], 1)
        self.assertAlmostEqual(
            report["fuzzy_ddl_violation_rate"],
            0.25,
        )
        self.assertAlmostEqual(
            report["feasible_workflow_ratio"],
            0.75,
        )
        self.assertAlmostEqual(
            report["feasible_episode_ratio"],
            0.5,
        )
        self.assertFalse(report["all_seed_feasible"])
        self.assertEqual(report["feasible_seed_rate"], 0.5)
        self.assertAlmostEqual(
            report["mean_fuzzy_lateness"],
            0.5,
        )
        self.assertAlmostEqual(
            report["max_fuzzy_lateness"],
            2.0,
        )
        self.assertAlmostEqual(
            report["minimum_fuzzy_safety_margin"],
            -2.0,
        )

        self.assertEqual(report["shield_record_count"], 3)
        self.assertEqual(report["shield_intervention_count"], 1)
        self.assertAlmostEqual(
            report["shield_intervention_rate"],
            1.0 / 3.0,
        )
        self.assertEqual(report["no_safe_action_count"], 1)
        self.assertEqual(report["fallback_count"], 1)
        self.assertEqual(
            report[
                "proposed_executed_action_mismatch_count"
            ],
            1,
        )
        self.assertEqual(report["q_c_prediction_error"], 0.25)
        self.assertEqual(
            report["q_c_prediction_error_sample_count"],
            8,
        )
        self.assertEqual(report["lambda_current"], 3.0)

        self.assertEqual(report["fuzzy_energy_mean"], 100.0)
        self.assertEqual(report["fuzzy_energy_std"], 3.0)
        self.assertEqual(report["fuzzy_energy_score"], 103.0)
        self.assertEqual(report["modal_energy"], 98.0)
        self.assertEqual(report["scheduling_time_seconds"], 3.0)
        self.assertEqual(
            report["mean_seed_scheduling_time_seconds"],
            1.5,
        )
        self.assertAlmostEqual(
            report["selected_llm_heuristic_frequency"],
            1.0 / 3.0,
        )
        self.assertEqual(
            report["llm_associated_shield_rate"],
            0.5,
        )

    def test_worst_seed_is_preserved_instead_of_averaged_away(self):
        report = aggregate_safe_metric_records(self._records())
        self.assertEqual(
            report["worst_seed_violation_rate"],
            0.5,
        )
        self.assertAlmostEqual(
            report["worst_seed_fuzzy_lateness"],
            2.0,
        )
        self.assertAlmostEqual(
            report[
                "worst_seed_minimum_fuzzy_safety_margin"
            ],
            -2.0,
        )
        self.assertNotEqual(
            report["worst_seed_violation_rate"],
            report["fuzzy_ddl_violation_rate"],
        )

    def test_risk_measure_uses_modal_upper_eta_point_ninety_five(self):
        env = _MetricEnvironment(
            deadlines=(10.0,),
            fuzzy_finishes=((1.0, 2.0, 12.0),),
            energy_mean=1.0,
            energy_std=0.0,
            modal_energy=1.0,
        )
        record = build_episode_metric_record(
            env,
            seed=1,
            scheduling_time_seconds=0.0,
        )
        expected_risk = 0.05 * 2.0 + 0.95 * 12.0
        self.assertAlmostEqual(
            record["minimum_fuzzy_safety_margin"],
            10.0 - expected_risk,
        )

    def test_single_run_does_not_invent_paired_llm_effects(self):
        report = aggregate_safe_metric_records(self._records())
        self.assertIsNone(report["energy_improvement_from_llm"])
        self.assertIsNone(report["convergence_acceleration"])
        self.assertIsNone(
            report["strict_ddl_feasibility_with_llm"]
        )


class SafeMetricPersistenceTests(unittest.TestCase):
    def test_sources_use_separate_files_and_stable_csv_header(self):
        report = _safe_report()
        with tempfile.TemporaryDirectory() as directory:
            store = SafeMetricStore(
                directory,
                convergence_window=2,
            )
            store.append(
                "training",
                report,
                global_step=10,
                episode=1,
            )
            first_validation = store.append(
                "validation",
                report,
                global_step=10,
                episode=1,
            )
            second_validation = store.append(
                "validation",
                report,
                global_step=20,
                episode=2,
            )
            final = store.append(
                "final_test",
                report,
                global_step=None,
                episode=None,
            )

            root = Path(directory)
            for source in (
                "training",
                "validation",
                "final_test",
            ):
                csv_path = root / f"{source}_metrics.csv"
                jsonl_path = root / f"{source}_metrics.jsonl"
                self.assertTrue(csv_path.exists())
                self.assertTrue(jsonl_path.exists())
                with csv_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    header = next(csv.reader(handle))
                self.assertEqual(
                    header,
                    list(SAFE_METRIC_CSV_FIELDS),
                )

            self.assertIsNone(
                first_validation["evaluations_to_convergence"]
            )
            self.assertEqual(
                second_validation["evaluations_to_convergence"],
                2,
            )
            self.assertEqual(
                second_validation["convergence_speed"],
                0.5,
            )
            self.assertEqual(
                second_validation["lambda_trajectory"],
                [0.0, 0.0],
            )
            final_path = root / "final_test_metrics.json"
            payload = json.loads(
                final_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["metrics_schema_version"],
                SAFE_METRICS_SCHEMA_VERSION,
            )
            self.assertEqual(
                payload["metric_source"],
                "final_test",
            )
            self.assertEqual(
                final["metric_source"],
                "final_test",
            )

    def test_unknown_source_and_schema_incomplete_report_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SafeMetricStore(directory)
            with self.assertRaises(ValueError):
                store.append(
                    "test",
                    _safe_report(),
                    global_step=0,
                    episode=0,
                )
            with self.assertRaises(ValueError):
                store.append(
                    "training",
                    {},
                    global_step=0,
                    episode=0,
                )

    def test_existing_csv_schema_mismatch_is_not_silently_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_metrics.csv"
            path.write_text("old_field\n1\n", encoding="utf-8")
            store = SafeMetricStore(directory)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                store.append(
                    "training",
                    _safe_report(),
                    global_step=0,
                    episode=0,
                )

    def test_paired_llm_metrics_require_same_seeds_and_strict_ddl(self):
        assisted = _safe_report(seed=1, score=8.0)
        baseline = _safe_report(seed=1, score=10.0)
        assisted.update(
            {
                "evaluations_to_convergence": 4,
                "selected_llm_heuristic_frequency": 0.25,
                "llm_associated_shield_rate": 0.1,
            }
        )
        baseline["evaluations_to_convergence"] = 8
        metrics = compute_llm_assistance_metrics(
            with_llm_report=assisted,
            without_llm_report=baseline,
            strict_ddl=True,
        )
        self.assertAlmostEqual(
            metrics["energy_improvement_from_llm"],
            0.2,
        )
        self.assertAlmostEqual(
            metrics["convergence_acceleration"],
            0.5,
        )
        self.assertEqual(
            metrics["strict_ddl_feasibility_with_llm"],
            1.0,
        )
        self.assertEqual(
            metrics["strict_ddl_feasibility_without_llm"],
            1.0,
        )

        with self.assertRaises(ValueError):
            compute_llm_assistance_metrics(
                with_llm_report=assisted,
                without_llm_report=baseline,
                strict_ddl=False,
            )
        different_seed = _safe_report(seed=2, score=10.0)
        with self.assertRaises(ValueError):
            compute_llm_assistance_metrics(
                with_llm_report=assisted,
                without_llm_report=different_seed,
                strict_ddl=True,
            )

    def test_paired_llm_report_is_persisted_explicitly(self):
        assisted = _safe_report(seed=1, score=8.0)
        baseline = _safe_report(seed=1, score=10.0)
        with tempfile.TemporaryDirectory() as directory:
            store = SafeMetricStore(directory)
            result = store.save_llm_comparison(
                with_llm_report=assisted,
                without_llm_report=baseline,
                strict_ddl=True,
            )
            self.assertEqual(
                result["comparison_kind"],
                "paired_llm_ablation",
            )
            self.assertTrue(
                (
                    Path(directory)
                    / "llm_comparison_metrics.json"
                ).exists()
            )


class SafeMetricConfigurationTests(unittest.TestCase):
    def test_default_safe_rl_stays_disabled_with_metrics_configured(self):
        config = SafeRLConfig()
        self.assertFalse(config.enabled)
        self.assertTrue(config.metrics.enabled)
        self.assertEqual(config.metrics.schema_version, 1)
        self.assertEqual(config.metrics.convergence_window, 5)

    def test_metrics_config_rejects_unsafe_or_invalid_paths(self):
        with self.assertRaises(ValueError):
            SafeMetricsConfig(output_subdir="../escape")
        with self.assertRaises(ValueError):
            SafeMetricsConfig(convergence_window=0)
        with self.assertRaises(ValueError):
            SafeMetricsConfig(schema_version=2)


class FinalTestSeedIsolationTests(unittest.TestCase):
    def test_final_test_rejects_seed_overlap_before_evaluation(self):
        try:
            from hrl_mix.train_eval import (
                evaluate_and_save_safe_hrl_final_test,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("PyTorch is not installed")
            raise

        with self.assertRaises(ValueError):
            evaluate_and_save_safe_hrl_final_test(
                object,
                {},
                object(),
                object(),
                object(),
                training_seeds=(1, 2),
                validation_seeds=(3,),
                final_test_seeds=(2, 4),
                metrics_output_directory="unused",
            )

    def test_final_test_report_uses_only_explicit_withheld_seeds(self):
        try:
            import hrl_mix.train_eval as train_eval
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("PyTorch is not installed")
            raise

        report = _safe_report(seed=30)
        fake_result = (1.0, 2.0, 3.0, 4.0, report)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                train_eval,
                "evaluate_hrl_three_layer_multi_seed",
                return_value=fake_result,
            ) as evaluation:
                result = (
                    train_eval
                    .evaluate_and_save_safe_hrl_final_test(
                        object,
                        {},
                        object(),
                        object(),
                        object(),
                        training_seeds=(10,),
                        validation_seeds=(20,),
                        final_test_seeds=(30,),
                        metrics_output_directory=directory,
                    )
                )
            self.assertEqual(
                evaluation.call_args.args[5],
                (30,),
            )
            self.assertEqual(
                result[-1]["seed_split"]["final_test"],
                [30],
            )
            self.assertTrue(
                (
                    Path(directory)
                    / "final_test_metrics.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
