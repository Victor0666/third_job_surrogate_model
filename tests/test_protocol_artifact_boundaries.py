"""Protocol identity binding at offline artifact load/save boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from base.heuristic_admission import append_admission_record, record_sha256
from base.hrl_env import HrlFcfsCacheEnv
from base.manager_heuristics import (
    HEURISTIC_SELECTION_MODE,
    load_manager_heuristic_library,
)
from hrl_mix.model_selection import (
    FeasibilityFirstModelMetrics,
    protocol_identity_from_config_snapshot,
    read_best_checkpoint_manifest,
    save_best_checkpoint_bundle,
)
from hrl_mix.safe_training_pipeline import (
    SafeTrainingController,
    load_safe_training_plan,
    read_pipeline_checkpoint,
    save_pipeline_checkpoint,
)
from algorithms.llm_safe_hrl.scenario_registry import (
    resolve_experiment_protocol,
)


class _Agent:
    safe_rl_enabled = True

    def save(self, path, *, lagrange_controller_state=None):
        Path(path).write_text("{}", encoding="utf-8")


class _Lagrange:
    def state_dict(self):
        return {"current_lambda": 1.0}


def _identity_config(context):
    return {
        "config_snapshot_schema_version": 1,
        "sha256": "unused-in-boundary-test",
        "config": {
            **context.identity(),
            "optimizer_seed": 0,
        },
    }


def _runtime_identity_config(context):
    """Mirror TrainConfig: nested identity plus partial flat runtime fields."""
    identity = context.identity()
    return {
        "config_snapshot_schema_version": 1,
        "sha256": "unused-in-boundary-test",
        "config": {
            "experiment_protocol": copy.deepcopy(identity),
            "protocol": identity["protocol"],
            "source_scenario": identity["source_scenario"],
            "resource_scale": context.resource_scale,
            "training_scenarios": tuple(identity["training_scenarios"]),
            "test_scenarios": tuple(identity["test_scenarios"]),
            "train_seeds": tuple(identity["safe_hrl_train_seeds"]),
            "validation_seeds": tuple(
                identity["safe_hrl_validation_seeds"]
            ),
            "final_test_seeds": tuple(identity["final_test_seeds"]),
            "optimizer_seed": 0,
        },
    }


def _metrics():
    return FeasibilityFirstModelMetrics(
        deadline_violation_rate=0.0,
        max_fuzzy_lateness=0.0,
        mean_fuzzy_lateness=0.0,
        fuzzy_energy_score=10.0,
        all_seed_feasible=True,
        feasible_seed_rate=1.0,
        worst_seed_violation=0.0,
        worst_seed_lateness=0.0,
        validation_seed_count=2,
    )


def _plan(directory: Path) -> Path:
    manifest = directory / "demo.json"
    manifest.write_text("{}", encoding="utf-8")
    curriculum = {
        "deadline": {
            "level": "loose",
            "alpha_small": 2.0,
            "alpha_large": 3.0,
            "alpha_small_probability": 0.8,
        },
        "arrival": {"level": "low", "poisson_lambda": 0.01},
        "uncertainty": {
            "level": "low",
            "fuzzy_delta1": 0.9,
            "fuzzy_delta2": 1.1,
        },
        "resource": {
            "scale": "small",
            "capacity_scale": 1.0,
            "topology": "inherit",
        },
        "workflow": {"scale": "small", "workflows_per_episode": 1},
    }
    payload = {
        "schema_version": 1,
        "pipeline_id": "protocol_boundary",
        "checkpoint_interval_episodes": 1,
        "metrics_path": "metrics.jsonl",
        "seed_split": {
            "training": [1],
            "validation": [4],
            "final_test": [201],
        },
        "preparation_stages": [
            {
                "stage_id": "demo",
                "stage_type": "demonstration_generation",
                "execution": "external",
                "artifact_manifest_path": manifest.name,
                "require_completed_artifact": False,
            },
            {
                "stage_id": "offline",
                "stage_type": "offline_pretraining",
                "offline_pretraining": {
                    "dataset_manifest_path": manifest.name,
                    "epochs": 1,
                    "batch_size": 1,
                    "performance_learning_rate": 0.001,
                    "safety_learning_rate": 0.001,
                    "train_q_r": True,
                    "train_q_c": True,
                    "behavior_cloning_enabled": False,
                    "behavior_cloning_weight": 0.1,
                    "sync_targets_after_pretraining": True,
                    "random_seed": 1,
                },
            },
        ],
        "online_stages": [
            {
                "stage_id": "online",
                "stage_type": "shield_online_training",
                "training_seed_mode": "primary",
                "curriculum": curriculum,
                "transition": {"mode": "fixed_episodes", "episodes": 200},
            },
            {
                "stage_id": "curriculum",
                "stage_type": "curriculum_training",
                "training_seed_mode": "primary",
                "curriculum": curriculum,
                "transition": {"mode": "fixed_episodes", "episodes": 200},
            },
            {
                "stage_id": "robust",
                "stage_type": "cross_seed_robust_training",
                "training_seed_mode": "round_robin",
                "curriculum": curriculum,
                "transition": {"mode": "fixed_episodes", "episodes": 200},
            },
        ],
    }
    path = directory / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ProtocolArtifactBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.single = resolve_experiment_protocol(
            "single", source_scenario="SS"
        )
        self.multi = resolve_experiment_protocol(
            "multi", source_scenario=None, resource_scale="S"
        )

    def test_runtime_config_snapshot_prefers_nested_protocol_identity(self):
        snapshot = _runtime_identity_config(self.single)
        self.assertEqual(
            protocol_identity_from_config_snapshot(snapshot),
            self.single.identity(),
        )
        with tempfile.TemporaryDirectory() as directory:
            agents = {
                name: _Agent()
                for name in ("manager", "host", "vm")
            }
            manifest = save_best_checkpoint_bundle(
                directory,
                agents=agents,
                lagrange_controller=_Lagrange(),
                model_metrics=_metrics(),
                curriculum_state={},
                replay_metadata={name: {} for name in agents},
                heuristic_library_version={
                    "manifest_version": "v1",
                    "experiment_protocol": self.single.identity(),
                },
                config_snapshot=snapshot,
                experiment_protocol=self.single.identity(),
            )
            payload = read_best_checkpoint_manifest(
                manifest,
                expected_protocol_identity=self.single,
            )
            self.assertEqual(
                payload["experiment_protocol"],
                self.single.identity(),
            )

    def test_flat_partial_protocol_identity_remains_invalid(self):
        snapshot = _runtime_identity_config(self.single)
        snapshot["config"].pop("experiment_protocol")
        with self.assertRaisesRegex(
            ValueError,
            "incomplete protocol identity",
        ):
            protocol_identity_from_config_snapshot(snapshot)

    def test_protocol_failure_precedes_agent_checkpoint_writes(self):
        snapshot = _runtime_identity_config(self.single)
        snapshot["config"]["experiment_protocol"].pop(
            "llm_validation_seeds"
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "invalid_bundle"
            agents = {
                name: _Agent()
                for name in ("manager", "host", "vm")
            }
            with self.assertRaises(ValueError):
                save_best_checkpoint_bundle(
                    target,
                    agents=agents,
                    lagrange_controller=_Lagrange(),
                    model_metrics=_metrics(),
                    curriculum_state={},
                    replay_metadata={name: {} for name in agents},
                    heuristic_library_version={
                        "manifest_version": "v1",
                    },
                    config_snapshot=snapshot,
                    experiment_protocol=self.single.identity(),
                )
            self.assertFalse(target.exists())

    def test_best_checkpoint_rejects_cross_protocol_load(self):
        with tempfile.TemporaryDirectory() as directory:
            agents = {name: _Agent() for name in ("manager", "host", "vm")}
            manifest = save_best_checkpoint_bundle(
                directory,
                agents=agents,
                lagrange_controller=_Lagrange(),
                model_metrics=_metrics(),
                curriculum_state={},
                replay_metadata={name: {} for name in agents},
                heuristic_library_version={"manifest_version": "v1"},
                config_snapshot=_identity_config(self.single),
            )
            payload = read_best_checkpoint_manifest(
                manifest,
                expected_protocol_identity=self.single,
            )
            self.assertEqual(payload["experiment_protocol"], self.single.identity())
            with self.assertRaisesRegex(ValueError, "protocol identity mismatch"):
                read_best_checkpoint_manifest(
                    manifest,
                    expected_protocol_identity=self.multi,
                )

    def test_legacy_best_checkpoint_is_readable_only_without_expectation(self):
        with tempfile.TemporaryDirectory() as directory:
            agents = {name: _Agent() for name in ("manager", "host", "vm")}
            manifest = save_best_checkpoint_bundle(
                directory,
                agents=agents,
                lagrange_controller=_Lagrange(),
                model_metrics=_metrics(),
                curriculum_state={},
                replay_metadata={name: {} for name in agents},
                heuristic_library_version={"manifest_version": "legacy"},
                config_snapshot={
                    "config_snapshot_schema_version": 1,
                    "config": {
                        "safe_rl": {"enabled": True},
                        "optimizer_seed": 0,
                    },
                },
            )
            self.assertNotIn(
                "experiment_protocol", read_best_checkpoint_manifest(manifest)
            )
            with self.assertRaisesRegex(ValueError, "missing experiment_protocol"):
                read_best_checkpoint_manifest(
                    manifest,
                    expected_protocol_identity=self.single,
                )

    def test_pipeline_checkpoint_rejects_cross_protocol_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = load_safe_training_plan(_plan(root))
            controller = SafeTrainingController(plan)
            agents = {name: _Agent() for name in ("manager", "host", "vm")}
            checkpoint = save_pipeline_checkpoint(
                root / "checkpoint",
                controller=controller,
                agents=agents,
                lagrange_controller=_Lagrange(),
                global_step=0,
                next_episode=0,
                best_model_metrics=None,
                replay_metadata={name: {} for name in agents},
                heuristic_library_version={"manifest_version": "v1"},
                config_snapshot=_identity_config(self.single),
            )
            read_pipeline_checkpoint(
                checkpoint,
                controller=SafeTrainingController(plan),
                expected_protocol_identity=self.single,
            )
            with self.assertRaisesRegex(ValueError, "protocol identity mismatch"):
                read_pipeline_checkpoint(
                    checkpoint,
                    controller=SafeTrainingController(plan),
                    expected_protocol_identity=self.multi,
                )

    def test_protocol_bound_library_and_record_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "library.json"
            record = {
                "record_schema_version": 2,
                "heuristic_id": "rule",
                "version": "v1",
                "admission_scope": {
                    "mode": "resource_task_domain",
                    "resource_code": "S",
                    "allowed_scenarios": ["SS", "MS", "LS"],
                    "workflow_families": [
                        "CyberShake", "Epigenomics", "Ligo", "Montage", "Sipht"
                    ],
                },
                "admission_policy": {
                    "policy_version": "test",
                    "required_evaluation_seeds": [1],
                    "minimum_evaluation_seed_count": 1,
                    "deadline_violation_rate_max": 0.0,
                    "max_fuzzy_lateness_max": 0.0,
                    "feasible_seed_rate_min": 1.0,
                    "fuzzy_energy_score_max": 1.0,
                    "objective_cv_max": 0.1,
                },
                "experiment_protocol": self.single.identity(),
            }
            record["record_sha256"] = record_sha256(record)
            payload = append_admission_record(
                manifest,
                record,
                expected_protocol_identity=self.single,
            )
            self.assertEqual(payload["experiment_protocol"], self.single.identity())
            self.assertEqual(
                len(
                    load_manager_heuristic_library(
                        manifest,
                        expected_protocol_identity=self.single,
                    )
                ),
                6,
            )
            different = copy.deepcopy(record)
            different["heuristic_id"] = "other"
            different["version"] = "v2"
            different["experiment_protocol"] = self.multi.identity()
            different["record_sha256"] = record_sha256(different)
            with self.assertRaisesRegex(ValueError, "protocol identity mismatch"):
                append_admission_record(manifest, different)
            with self.assertRaisesRegex(ValueError, "protocol identity mismatch"):
                load_manager_heuristic_library(
                    manifest,
                    expected_protocol_identity=self.multi,
                )
            legacy_manifest = root / "legacy_library.json"
            legacy_record = copy.deepcopy(record)
            legacy_record.pop("experiment_protocol")
            legacy_record["record_sha256"] = record_sha256(legacy_record)
            append_admission_record(legacy_manifest, legacy_record)
            load_manager_heuristic_library(legacy_manifest)
            with self.assertRaisesRegex(ValueError, "missing experiment_protocol"):
                load_manager_heuristic_library(
                    legacy_manifest,
                    expected_protocol_identity=self.single,
                )

    def test_environment_passes_expected_identity_to_library_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "library.json"
            record = {
                "record_schema_version": 2,
                "heuristic_id": "rule",
                "version": "v1",
                "admission_scope": {
                    "mode": "resource_task_domain",
                    "resource_code": "S",
                    "allowed_scenarios": ["SS", "MS", "LS"],
                    "workflow_families": [
                        "CyberShake", "Epigenomics", "Ligo", "Montage", "Sipht"
                    ],
                },
                "admission_policy": {
                    "policy_version": "test",
                    "required_evaluation_seeds": [1],
                    "minimum_evaluation_seed_count": 1,
                    "deadline_violation_rate_max": 0.0,
                    "max_fuzzy_lateness_max": 0.0,
                    "feasible_seed_rate_min": 1.0,
                    "fuzzy_energy_score_max": 1.0,
                    "objective_cv_max": 0.1,
                },
                "experiment_protocol": self.single.identity(),
            }
            record["record_sha256"] = record_sha256(record)
            append_admission_record(manifest, record)
            dax = (
                Path(__file__).resolve().parents[1]
                / "data"
                / "dax"
                / "Montage_25.xml"
            )
            common = {
                "dax_paths": [str(dax)],
                "deadline_mode": "none",
                "workflows_per_episode": 1,
                "manager_mode": HEURISTIC_SELECTION_MODE,
                "manager_heuristic_library_path": str(manifest),
                "fuzzy_enabled": True,
                "safe_rl_enabled": True,
                "safe_rl_shield_enabled": True,
                "safe_rl_state_enabled": True,
                "scenario_code": "SS",
                "task_code": "S",
                "resource_code": "S",
                "workflow_families": [
                    "CyberShake", "Epigenomics", "Ligo", "Montage", "Sipht"
                ],
            }
            HrlFcfsCacheEnv(
                **common,
                experiment_protocol_identity=self.single.identity(),
            )
            with self.assertRaisesRegex(ValueError, "protocol identity mismatch"):
                HrlFcfsCacheEnv(
                    **common,
                    experiment_protocol_identity=self.multi.identity(),
                )


if __name__ == "__main__":
    unittest.main()
