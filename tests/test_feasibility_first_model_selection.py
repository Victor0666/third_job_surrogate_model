# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from common.resource_opt import TriangularFuzzyNumber
from hrl_mix.model_selection import (
    FeasibilityFirstModelMetrics,
    aggregate_seed_feasibility_metrics,
    build_config_snapshot,
    build_heuristic_library_version,
    build_replay_metadata,
    is_better_model,
    save_best_checkpoint_bundle,
)
from hrl_mix.train_config import SafeRLConfig


def _metrics(**overrides):
    values = {
        "deadline_violation_rate": 0.0,
        "max_fuzzy_lateness": 0.0,
        "mean_fuzzy_lateness": 0.0,
        "fuzzy_energy_score": 10.0,
        "all_seed_feasible": True,
        "feasible_seed_rate": 1.0,
        "worst_seed_violation": 0.0,
        "worst_seed_lateness": 0.0,
        "validation_seed_count": 3,
    }
    values.update(overrides)
    return FeasibilityFirstModelMetrics(**values)


class FeasibilityFirstOrderingTests(unittest.TestCase):
    def test_zero_violation_beats_any_nonzero_violation(self):
        safe = _metrics(fuzzy_energy_score=1000.0)
        unsafe = _metrics(
            deadline_violation_rate=0.01,
            max_fuzzy_lateness=0.1,
            mean_fuzzy_lateness=0.01,
            fuzzy_energy_score=1.0,
            all_seed_feasible=False,
            feasible_seed_rate=2 / 3,
            worst_seed_violation=0.03,
            worst_seed_lateness=0.1,
        )
        self.assertTrue(is_better_model(safe, unsafe))
        self.assertFalse(is_better_model(unsafe, safe))

    def test_lower_violation_rate_has_priority(self):
        lower = _metrics(
            deadline_violation_rate=0.1,
            max_fuzzy_lateness=100.0,
            mean_fuzzy_lateness=50.0,
            all_seed_feasible=False,
            feasible_seed_rate=0.0,
            worst_seed_violation=0.2,
            worst_seed_lateness=100.0,
        )
        higher = _metrics(
            deadline_violation_rate=0.2,
            max_fuzzy_lateness=1.0,
            mean_fuzzy_lateness=0.1,
            fuzzy_energy_score=1.0,
            all_seed_feasible=False,
            feasible_seed_rate=0.0,
            worst_seed_violation=0.3,
            worst_seed_lateness=1.0,
        )
        self.assertTrue(is_better_model(lower, higher))

    def test_max_lateness_breaks_violation_rate_tie(self):
        lower = _metrics(
            deadline_violation_rate=0.1,
            max_fuzzy_lateness=2.0,
            mean_fuzzy_lateness=1.9,
            all_seed_feasible=False,
            feasible_seed_rate=0.0,
            worst_seed_violation=0.1,
            worst_seed_lateness=2.0,
        )
        higher = _metrics(
            deadline_violation_rate=0.1,
            max_fuzzy_lateness=3.0,
            mean_fuzzy_lateness=0.1,
            fuzzy_energy_score=1.0,
            all_seed_feasible=False,
            feasible_seed_rate=0.0,
            worst_seed_violation=0.1,
            worst_seed_lateness=3.0,
        )
        self.assertTrue(is_better_model(lower, higher))

    def test_mean_lateness_breaks_max_lateness_tie(self):
        lower = _metrics(
            deadline_violation_rate=0.1,
            max_fuzzy_lateness=2.0,
            mean_fuzzy_lateness=0.2,
            all_seed_feasible=False,
            feasible_seed_rate=0.0,
            worst_seed_violation=0.1,
            worst_seed_lateness=2.0,
        )
        higher = _metrics(
            deadline_violation_rate=0.1,
            max_fuzzy_lateness=2.0,
            mean_fuzzy_lateness=0.3,
            fuzzy_energy_score=1.0,
            all_seed_feasible=False,
            feasible_seed_rate=0.0,
            worst_seed_violation=0.1,
            worst_seed_lateness=2.0,
        )
        self.assertTrue(is_better_model(lower, higher))

    def test_energy_breaks_fully_feasible_tie(self):
        self.assertTrue(
            is_better_model(
                _metrics(fuzzy_energy_score=9.0),
                _metrics(fuzzy_energy_score=10.0),
            )
        )

    def test_equal_key_does_not_overwrite_incumbent(self):
        incumbent = _metrics()
        self.assertFalse(is_better_model(_metrics(), incumbent))


class MultiSeedMetricTests(unittest.TestCase):
    def test_aggregate_reports_required_seed_fields(self):
        aggregate = aggregate_seed_feasibility_metrics(
            [
                {
                    "seed": 1,
                    "deadline_violation_count": 1,
                    "completed_workflow_count": 2,
                    "fuzzy_lateness_sum": 2.0,
                    "max_fuzzy_lateness": 2.0,
                    "fuzzy_energy_score": 10.0,
                    "evaluation_completed": True,
                },
                {
                    "seed": 2,
                    "deadline_violation_count": 0,
                    "completed_workflow_count": 2,
                    "fuzzy_lateness_sum": 0.0,
                    "max_fuzzy_lateness": 0.0,
                    "fuzzy_energy_score": 20.0,
                    "evaluation_completed": True,
                },
            ]
        )
        self.assertFalse(aggregate["all_seed_feasible"])
        self.assertEqual(aggregate["feasible_seed_rate"], 0.5)
        self.assertEqual(aggregate["worst_seed_violation"], 0.5)
        self.assertEqual(aggregate["worst_seed_lateness"], 2.0)
        self.assertEqual(aggregate["deadline_violation_rate"], 0.25)
        self.assertEqual(aggregate["mean_fuzzy_lateness"], 0.5)
        self.assertEqual(aggregate["fuzzy_energy_score"], 15.0)

    def test_train_eval_reconstructs_fuzzy_lateness_per_seed(self):
        try:
            from hrl_mix.train_eval import (
                evaluate_hrl_three_layer_multi_seed,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest(
                    "train_eval imports the PyTorch training utilities"
                )
            raise

        class FinishedEnvironment:
            def __init__(self, **kwargs):
                self.random_seed = int(kwargs["random_seed"])
                self.done_flag = True
                self.total_energy = float(
                    10 * self.random_seed
                )
                self.workflows_per_episode = 2
                self.workflows = [
                    SimpleNamespace(deadline=10.0),
                    SimpleNamespace(deadline=10.0),
                ]
                self.wf_finish_time = {0: 0.0, 1: 0.0}

            def reset(self):
                risks = (
                    [9.0, 12.0]
                    if self.random_seed == 1
                    else [8.0, 9.0]
                )
                self.risks = risks
                self._safety_cumulative_deadline_violation_count = (
                    int(sum(value > 10.0 for value in risks))
                )
                self._safety_cumulative_completed_workflow_count = 2
                self._safety_cumulative_cost = float(
                    sum(max(0.0, value - 10.0) for value in risks)
                )

            @staticmethod
            def get_manager_state():
                return [0.0]

            @staticmethod
            def get_manager_action_mask():
                return [1.0]

            @staticmethod
            def apply_manager_delta(_delta):
                return None

            def _workflow_finish_tfn(self, workflow_id):
                value = self.risks[workflow_id]
                return TriangularFuzzyNumber(
                    value,
                    value,
                    value,
                )

            @staticmethod
            def fuzzy_deadline_measure(finish_tfn):
                return float(finish_tfn.upper)

            def get_fuzzy_energy_summary(self):
                return {
                    "fuzzy_total_energy_mean": self.total_energy,
                    "fuzzy_total_energy_std": 0.0,
                    "fuzzy_total_energy_score": self.total_energy,
                }

        class FixedAgent:
            @staticmethod
            def select_action(
                _state,
                _mask,
                deterministic,
                count_step,
            ):
                if not deterministic or count_step:
                    raise AssertionError(
                        "evaluation must be deterministic"
                    )
                return 0

        result = evaluate_hrl_three_layer_multi_seed(
            FinishedEnvironment,
            {},
            FixedAgent(),
            FixedAgent(),
            FixedAgent(),
            seeds=(1, 2),
            return_safety_metrics=True,
        )
        safety = result[-1]
        self.assertEqual(safety["deadline_violation_rate"], 0.25)
        self.assertEqual(safety["max_fuzzy_lateness"], 2.0)
        self.assertEqual(safety["mean_fuzzy_lateness"], 0.5)
        self.assertFalse(safety["all_seed_feasible"])
        self.assertEqual(safety["feasible_seed_rate"], 0.5)
        self.assertEqual(safety["worst_seed_violation"], 0.5)
        self.assertEqual(safety["worst_seed_lateness"], 2.0)
        self.assertEqual(safety["fuzzy_energy_score"], 15.0)


class CheckpointBundleTests(unittest.TestCase):
    class Agent:
        safe_rl_enabled = True

        def __init__(self, name):
            self.name = name

        def save(self, path, *, lagrange_controller_state=None):
            Path(path).write_text(
                json.dumps(
                    {
                        "name": self.name,
                        "lagrange": lagrange_controller_state,
                    }
                ),
                encoding="utf-8",
            )

    class Lagrange:
        def state_dict(self):
            return {
                "state_version": 1,
                "current_lambda": 2.5,
            }

    def test_bundle_binds_all_required_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            agents = {
                layer: self.Agent(layer)
                for layer in ("manager", "host", "vm")
            }
            manifest = save_best_checkpoint_bundle(
                directory,
                agents=agents,
                lagrange_controller=self.Lagrange(),
                model_metrics=_metrics(),
                curriculum_state={
                    "stage_id": "stage_3",
                    "stage_index": 0,
                },
                replay_metadata={
                    layer: {
                        "transition_count": 3,
                        "replay_transitions_embedded": False,
                    }
                    for layer in agents
                },
                heuristic_library_version={
                    "manifest_version": "stage12.v1",
                },
                config_snapshot={
                    "config_snapshot_schema_version": 1,
                    "config": {
                        "safe_rl": {"enabled": True},
                        "optimizer_seed": 0,
                    },
                },
            )
            payload = json.loads(
                Path(manifest).read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(payload["agent_checkpoints"]),
                {"manager", "host", "vm"},
            )
            for layer in agents:
                contents = payload[
                    "agent_checkpoint_contents"
                ][layer]
                self.assertEqual(
                    contents["q_r"],
                    {
                        "online_key": "online",
                        "target_key": "target",
                        "optimizer_key": "optim",
                    },
                )
                self.assertEqual(
                    contents["q_c"],
                    {
                        "online_key": "q_c_online",
                        "target_key": "q_c_target",
                        "optimizer_key": "q_c_optim",
                    },
                )
            self.assertEqual(
                payload["lagrange_controller_state"][
                    "current_lambda"
                ],
                2.5,
            )
            self.assertEqual(
                payload["curriculum_state"]["stage_id"],
                "stage_3",
            )
            self.assertEqual(
                payload["heuristic_library_version"][
                    "manifest_version"
                ],
                "stage12.v1",
            )
            self.assertIn("config_snapshot", payload)
            self.assertIn("replay_metadata", payload)

    def test_default_requires_all_validation_seeds_feasible(self):
        config = SafeRLConfig()
        self.assertFalse(config.enabled)
        self.assertTrue(config.model_selection.enabled)
        self.assertTrue(
            config.model_selection
            .require_all_validation_seeds_feasible
        )

    def test_runtime_metadata_is_versioned_without_replay_payload(self):
        config_snapshot = build_config_snapshot(SafeRLConfig())
        self.assertEqual(
            config_snapshot["config_snapshot_schema_version"],
            1,
        )
        self.assertEqual(len(config_snapshot["sha256"]), 64)

        agent = SimpleNamespace(
            safe_rl_enabled=True,
            buffer=[],
            buffer_size=100,
            input_dim=11,
            output_dim=4,
            use_per=False,
            priorities=[],
        )
        replay = build_replay_metadata(agent)
        self.assertEqual(replay["transition_count"], 0)
        self.assertEqual(replay["capacity"], 100)
        self.assertFalse(replay["replay_transitions_embedded"])

        heuristic = build_heuristic_library_version(
            SimpleNamespace(
                manager_mode="legacy_rule_weight_mode",
                manager_heuristic_schema_version=None,
            ),
            manifest_path=None,
        )
        self.assertEqual(
            heuristic["manifest_version"],
            "legacy_v1",
        )


if __name__ == "__main__":
    unittest.main()
