# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hrl_mix.safe_training_pipeline import (
    SafeStageMetricsLogger,
    SafeTrainingController,
    StageMetrics,
    apply_curriculum_to_env_kwargs,
    load_safe_training_plan,
    q_c_prediction_error_from_agents,
    read_pipeline_checkpoint,
    restore_pipeline_agents,
    save_pipeline_checkpoint,
    validate_preparation_artifacts,
)
from hrl_mix.train_config import build_train_config


def _curriculum(level, arrival, uncertainty, episode_count):
    ddl_values = {
        "loose": (3.0, 4.0),
        "medium": (2.5, 3.5),
        "tight": (2.0, 3.0),
    }
    arrival_values = {
        "low": 0.01,
        "medium": 0.03,
        "high": 0.05,
    }
    uncertainty_values = {
        "low": (0.9, 1.1),
        "medium": (0.8, 1.15),
        "high": (0.75, 1.2),
    }
    alpha_small, alpha_large = ddl_values[level]
    delta1, delta2 = uncertainty_values[uncertainty]
    return {
        "deadline": {
            "level": level,
            "alpha_small": alpha_small,
            "alpha_large": alpha_large,
            "alpha_small_probability": 0.8,
        },
        "arrival": {
            "level": arrival,
            "poisson_lambda": arrival_values[arrival],
        },
        "uncertainty": {
            "level": uncertainty,
            "fuzzy_delta1": delta1,
            "fuzzy_delta2": delta2,
        },
        "resource": {
            "scale": f"{level}_capacity",
            "capacity_scale": {
                "loose": 1.0,
                "medium": 0.9,
                "tight": 0.8,
            }[level],
            "topology": "inherit",
        },
        "workflow": {
            "scale": level,
            "workflows_per_episode": episode_count,
        },
    }


def _plan_payload(*, manifest_name="dataset.json"):
    return {
        "schema_version": 1,
        "pipeline_id": "unit_safe_pipeline",
        "checkpoint_interval_episodes": 1,
        "metrics_path": "metrics.jsonl",
        "seed_split": {
            "training": [1, 2],
            "validation": [11, 12],
            "final_test": [21, 22],
        },
        "preparation_stages": [
            {
                "stage_id": "stage_1",
                "stage_type": "demonstration_generation",
                "execution": "external",
                "artifact_manifest_path": manifest_name,
                "require_completed_artifact": True,
            },
            {
                "stage_id": "stage_2",
                "stage_type": "offline_pretraining",
                "offline_pretraining": {
                    "dataset_manifest_path": manifest_name,
                    "epochs": 1,
                    "batch_size": 2,
                    "performance_learning_rate": 0.001,
                    "safety_learning_rate": 0.001,
                    "train_q_r": True,
                    "train_q_c": True,
                    "behavior_cloning_enabled": False,
                    "behavior_cloning_weight": 0.1,
                    "sync_targets_after_pretraining": True,
                    "random_seed": 7,
                },
            },
        ],
        "online_stages": [
            {
                "stage_id": "stage_3",
                "stage_type": "shield_online_training",
                "training_seed_mode": "primary",
                "curriculum": _curriculum(
                    "loose", "low", "low", 2
                ),
                "transition": {
                    "mode": "fixed_episodes",
                    "episodes": 120,
                },
            },
            {
                "stage_id": "stage_4_loose",
                "stage_type": "curriculum_training",
                "training_seed_mode": "primary",
                "curriculum": _curriculum(
                    "loose", "low", "low", 3
                ),
                "transition": {
                    "mode": "fixed_episodes",
                    "episodes": 120,
                },
            },
            {
                "stage_id": "stage_4_medium",
                "stage_type": "curriculum_training",
                "training_seed_mode": "round_robin",
                "curriculum": _curriculum(
                    "medium", "medium", "medium", 4
                ),
                "transition": {
                    "mode": "fixed_episodes",
                    "episodes": 120,
                },
            },
            {
                "stage_id": "stage_4_tight",
                "stage_type": "curriculum_training",
                "training_seed_mode": "round_robin",
                "curriculum": _curriculum(
                    "tight", "high", "high", 5
                ),
                "transition": {
                    "mode": "fixed_episodes",
                    "episodes": 120,
                },
            },
            {
                "stage_id": "stage_5",
                "stage_type": "cross_seed_robust_training",
                "training_seed_mode": "round_robin",
                "curriculum": _curriculum(
                    "tight", "high", "high", 5
                ),
                "transition": {
                    "mode": "fixed_episodes",
                    "episodes": 120,
                },
            },
        ],
    }


def _write_plan(directory: Path, payload=None) -> Path:
    path = directory / "plan.json"
    path.write_text(
        json.dumps(
            payload or _plan_payload(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _safe_metrics(**overrides):
    values = {
        "fuzzy_energy_score": 10.0,
        "safety_cost": 0.0,
        "violation_rate": 0.0,
        "shield_intervention_rate": 0.1,
        "fallback_rate": 0.0,
        "lagrange_multiplier": 1.0,
        "q_c_prediction_error": 0.2,
        "q_c_prediction_error_sample_count": 3,
    }
    values.update(overrides)
    return StageMetrics(**values)


class _DummyAgent:
    safe_rl_enabled = True

    def __init__(self, name):
        self.name = name
        self.loaded = None
        self.lagrange_multiplier = 0.0

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

    def load(self, path):
        self.loaded = json.loads(
            Path(path).read_text(encoding="utf-8")
        )


class _DummyLagrange:
    def __init__(self):
        self.current_lambda = 2.0
        self.restored = False

    def state_dict(self):
        return {
            "state_version": 1,
            "current_lambda": self.current_lambda,
        }

    def load_state_dict(self, state, *, strict=True):
        self.current_lambda = float(state["current_lambda"])
        self.restored = bool(strict)


class SafeTrainingPipelineTests(unittest.TestCase):
    def test_plan_requires_disjoint_seed_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = _plan_payload()
            payload["seed_split"]["final_test"] = [2, 22]
            path = _write_plan(root, payload)
            with self.assertRaisesRegex(
                ValueError, "seed leakage"
            ):
                load_safe_training_plan(path)

    def test_plan_explicitly_contains_all_five_stage_types(self):
        with tempfile.TemporaryDirectory() as td:
            plan = load_safe_training_plan(
                _write_plan(Path(td))
            )
            types = [
                stage["stage_type"]
                for stage in plan.preparation_stages
            ] + [
                stage.stage_type for stage in plan.online_stages
            ]
            self.assertIn("demonstration_generation", types)
            self.assertIn("offline_pretraining", types)
            self.assertIn("shield_online_training", types)
            self.assertIn("curriculum_training", types)
            self.assertIn("cross_seed_robust_training", types)

    def test_missing_demonstration_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            plan = load_safe_training_plan(
                _write_plan(Path(td))
            )
            with self.assertRaises(FileNotFoundError):
                validate_preparation_artifacts(plan)
            Path(plan.demonstration_manifest_path).write_text(
                "{}",
                encoding="utf-8",
            )
            validate_preparation_artifacts(plan)

    def test_validation_is_observation_only_and_fixed_episode_driven(self):
        with tempfile.TemporaryDirectory() as td:
            controller = SafeTrainingController(
                load_safe_training_plan(_write_plan(Path(td)))
            )
            event = controller.observe_validation(_safe_metrics())
            self.assertFalse(event["transitioned"])
            self.assertEqual(controller.total_episode_count, 0)
            for _ in range(119):
                self.assertFalse(controller.record_episode()["transitioned"])
            transition = controller.record_episode()
            self.assertTrue(transition["transitioned"])
            self.assertEqual(
                controller.current_stage.stage_id,
                "stage_4_loose",
            )
    def test_final_test_metrics_cannot_advance_curriculum(self):
        with tempfile.TemporaryDirectory() as td:
            controller = SafeTrainingController(
                load_safe_training_plan(
                    _write_plan(Path(td))
                )
            )
            with self.assertRaisesRegex(
                ValueError, "formal validation and final test are read-only"
            ):
                controller.observe_validation(
                    _safe_metrics(),
                    source="final_test",
                )
            self.assertEqual(controller.total_episode_count, 0)

    def test_curriculum_applies_all_dimensions_without_topology_change(self):
        with tempfile.TemporaryDirectory() as td:
            controller = SafeTrainingController(
                load_safe_training_plan(
                    _write_plan(Path(td))
                )
            )
            base = {
                "random_seed": 99,
                "arrival_lambda": 0.9,
                "workflows_per_episode": 99,
                "deadline_alpha_small": 9.0,
                "deadline_alpha_large": 10.0,
                "deadline_alpha_small_prob": 0.5,
                "fuzzy_delta1": 0.5,
                "fuzzy_delta2": 1.5,
                "num_cloud_hosts": 2,
                "num_edge_hosts": 1,
                "cloud_vms_per_host": (2, 2),
                "edge_vms_per_host": (1,),
                "cloud_pc_tiers": (1.0, 2.0),
                "edge_pc_tiers": (1.0, 2.0),
                "cloud_bw_tiers": (10.0, 20.0),
                "edge_bw_tiers": (10.0, 20.0),
            }
            result = apply_curriculum_to_env_kwargs(
                base,
                controller.current_stage.curriculum,
                training_seed=1,
            )
            self.assertEqual(result["random_seed"], 1)
            self.assertEqual(result["arrival_lambda"], 0.01)
            self.assertEqual(result["workflows_per_episode"], 2)
            self.assertEqual(result["fuzzy_delta1"], 0.9)
            self.assertEqual(result["fuzzy_delta2"], 1.1)
            self.assertEqual(result["num_cloud_hosts"], 2)
            self.assertEqual(
                result["cloud_vms_per_host"], (2, 2)
            )
            self.assertEqual(
                result["cloud_pc_tiers"], (1.0, 2.0)
            )

    def test_checkpoint_restores_current_curriculum_and_agents(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = load_safe_training_plan(_write_plan(root))
            controller = SafeTrainingController(plan)
            for _ in range(120):
                controller.record_episode()
            agents = {
                layer: _DummyAgent(layer)
                for layer in ("manager", "host", "vm")
            }
            lagrange = _DummyLagrange()
            checkpoint = save_pipeline_checkpoint(
                root / "checkpoint",
                controller=controller,
                agents=agents,
                lagrange_controller=lagrange,
                global_step=7,
                next_episode=120,
                best_model_metrics={
                    "deadline_violation_rate": 0.0,
                    "max_fuzzy_lateness": 0.0,
                    "mean_fuzzy_lateness": 0.0,
                    "fuzzy_energy_score": 12.0,
                    "all_seed_feasible": True,
                    "feasible_seed_rate": 1.0,
                    "worst_seed_violation": 0.0,
                    "worst_seed_lateness": 0.0,
                    "validation_seed_count": 2,
                },
                replay_metadata={
                    layer: {
                        "transition_count": 4,
                        "replay_transitions_embedded": False,
                    }
                    for layer in agents
                },
                heuristic_library_version={
                    "manifest_version": "legacy_v1",
                },
                config_snapshot={
                    "config_snapshot_schema_version": 1,
                    "config": {
                        "safe_rl": {"enabled": True},
                        "optimizer_seed": 0,
                    },
                },
            )

            restored_controller = SafeTrainingController(plan)
            payload = read_pipeline_checkpoint(
                checkpoint,
                controller=restored_controller,
            )
            restored_agents = {
                layer: _DummyAgent(f"new_{layer}")
                for layer in ("manager", "host", "vm")
            }
            restored_lagrange = _DummyLagrange()
            restored_lagrange.current_lambda = 0.0
            restore_pipeline_agents(
                payload,
                agents=restored_agents,
                lagrange_controller=restored_lagrange,
            )
            self.assertEqual(
                restored_controller.current_stage.stage_id,
                "stage_4_loose",
            )
            self.assertEqual(
                restored_controller.total_episode_count, 120
            )
            self.assertTrue(restored_lagrange.restored)
            self.assertEqual(restored_lagrange.current_lambda, 2.0)
            self.assertTrue(
                all(
                    agent.loaded is not None
                    for agent in restored_agents.values()
                )
            )
            self.assertEqual(
                payload["best_model_metrics"][
                    "fuzzy_energy_score"
                ],
                12.0,
            )
            self.assertEqual(
                set(payload["replay_metadata"]),
                {"manager", "host", "vm"},
            )
            self.assertEqual(
                payload["curriculum_stage"]["stage_id"],
                "stage_4_loose",
            )

    def test_pre_feasibility_checkpoint_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = load_safe_training_plan(_write_plan(root))
            controller = SafeTrainingController(plan)
            old_checkpoint = root / "old_checkpoint.json"
            old_checkpoint.write_text(
                json.dumps(
                    {
                        "checkpoint_schema_version": 1,
                        "pipeline_id": plan.pipeline_id,
                        "plan_hash": plan.plan_hash,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "unsupported safe training checkpoint schema",
            ):
                read_pipeline_checkpoint(
                    old_checkpoint,
                    controller=controller,
                )

    def test_q_c_prediction_error_uses_updated_layers_only(self):
        class Agent:
            def __init__(self, value):
                self.last_update_info = value

        value, count = q_c_prediction_error_from_agents(
            [
                Agent({"safety_loss": 0.2}),
                Agent({"safety_loss": 0.4}),
                Agent(None),
            ]
        )
        self.assertAlmostEqual(value, 0.3)
        self.assertEqual(count, 2)

    def test_cross_seed_stage_cycles_training_seeds_only(self):
        with tempfile.TemporaryDirectory() as td:
            controller = SafeTrainingController(
                load_safe_training_plan(_write_plan(Path(td)))
            )
            for _ in range(120 * 4):
                controller.record_episode()
            self.assertEqual(
                controller.current_stage.stage_type,
                "cross_seed_robust_training",
            )
            used = []
            for _ in range(3):
                used.append(controller.training_seed_for_next_episode())
                controller.record_episode()
            self.assertEqual(used, [1, 2, 1])
            self.assertNotIn(11, used)
            self.assertNotIn(21, used)
    def test_short_five_stage_flow_smoke_and_metrics_log(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = load_safe_training_plan(_write_plan(root))
            controller = SafeTrainingController(plan)
            logger = SafeStageMetricsLogger(plan.metrics_path)
            visited = []
            for episode in range(600):
                if episode % 120 == 0:
                    visited.append(controller.current_stage.stage_id)
                controller.observe_validation(_safe_metrics())
                event = controller.record_episode()
                if episode % 120 == 119:
                    logger.log(
                        controller=controller,
                        metrics=_safe_metrics(),
                        transition_event=event,
                        global_step=episode,
                        episode=episode,
                    )
            self.assertEqual(
                visited,
                ["stage_3", "stage_4_loose", "stage_4_medium", "stage_4_tight", "stage_5"],
            )
            self.assertTrue(controller.completed)
            records = [
                json.loads(line)
                for line in Path(plan.metrics_path).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(records), 5)
            self.assertTrue(
                all(not record["final_test_consumed"] for record in records)
            )
    def test_default_train_config_keeps_pipeline_disabled(self):
        with mock.patch("hrl_mix.train_config.os.makedirs"):
            config = build_train_config(
                scenario="SS",
                ddl="T",
                max_episodes=1,
            )
        self.assertFalse(config.safe_rl.enabled)
        self.assertFalse(
            config.safe_rl.training_pipeline.enabled
        )

    def test_pipeline_plan_drives_offline_stage_config(self):
        with tempfile.TemporaryDirectory() as td:
            plan_path = _write_plan(Path(td))
            with mock.patch(
                "hrl_mix.train_config.os.makedirs"
            ):
                config = build_train_config(
                    scenario="SS",
                    ddl="T",
                    max_episodes=1,
                    safe_rl_enabled=True,
                    safe_rl_shield_enabled=True,
                    safe_rl_state_enabled=True,
                    safe_rl_dynamic_lambda_enabled=True,
                    safe_rl_heuristic_manager_enabled=True,
                    safe_rl_training_pipeline_plan=str(
                        plan_path
                    ),
                )
            self.assertTrue(
                config.safe_rl.training_pipeline.enabled
            )
            self.assertEqual(
                config.safe_rl.offline_pretraining.epochs, 1
            )
            self.assertEqual(
                config.safe_rl.offline_pretraining.batch_size, 2
            )
            self.assertEqual(
                Path(
                    config.safe_rl.offline_pretraining
                    .dataset_manifest_path
                ),
                Path(td) / "dataset.json",
            )


if __name__ == "__main__":
    unittest.main()
