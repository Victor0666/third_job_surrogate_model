"""Fast acceptance tests for the Manager + Global Safe-VM Worker topology."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from base.hrl_env import HrlFcfsCacheEnv
from hrl_mix.model_selection import (
    FeasibilityFirstModelMetrics,
    read_best_checkpoint_manifest,
    save_best_checkpoint_bundle,
)
from hrl_mix.train_config import build_train_config
from algorithms.llm_safe_hrl.scenario_registry import (
    FINAL_TEST_SEEDS,
    RESOURCE_SCALE_REGISTRY,
    SAFE_HRL_VALIDATION_SEEDS,
    resolve_experiment_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
DAX = ROOT / "data" / "dax" / "Montage_25.xml"


def _environment(resource_code="S"):
    resource = RESOURCE_SCALE_REGISTRY[resource_code]
    environment = HrlFcfsCacheEnv(
        dax_paths=[str(DAX)],
        deadline_mode="none",
        workflows_per_episode=1,
        horizon=1e6,
        arrival_lambda=0.03,
        random_seed=7,
        max_ready_tasks=32,
        fuzzy_enabled=True,
        safe_rl_enabled=True,
        safe_rl_shield_enabled=True,
        safe_rl_state_enabled=True,
        scenario_code=f"S{resource_code}",
        task_code="S",
        resource_code=resource_code,
        **resource.resource_mapping(),
    )
    environment.reset()
    return environment


def _metrics(*, feasible, energy, worst_violation=0.0):
    return FeasibilityFirstModelMetrics(
        deadline_violation_rate=(0.0 if feasible else worst_violation),
        max_fuzzy_lateness=(0.0 if feasible else 2.0),
        mean_fuzzy_lateness=(0.0 if feasible else 1.0),
        fuzzy_energy_score=float(energy),
        all_seed_feasible=bool(feasible),
        feasible_seed_rate=(1.0 if feasible else 0.8),
        worst_seed_violation=(0.0 if feasible else worst_violation),
        worst_seed_lateness=(0.0 if feasible else 2.0),
        validation_seed_count=3,
    )


class GlobalWorkerEnvironmentTests(unittest.TestCase):
    def test_host_load_and_running_ratio_have_distinct_semantics(self):
        environment = _environment("S")
        host_id = environment.host_ids[0]
        host_indices = environment.host_to_vm_indices[host_id]
        selected = max(
            host_indices,
            key=lambda index: environment.vms[
                environment.vm_ids[index]
            ].pc,
        )
        environment.vm_available_at[:] = environment.current_time
        environment.vm_available_at[selected] = environment.current_time + 10.0
        host_load, running_ratio = (
            environment._host_runtime_load_and_running_ratio(host_id)
        )
        expected_load = min(
            1.0,
            float(environment.vms[environment.vm_ids[selected]].pc)
            / float(environment.hosts[host_id].total_pc),
        )
        self.assertAlmostEqual(host_load, expected_load)
        self.assertAlmostEqual(running_ratio, 1.0 / len(host_indices))
        self.assertNotAlmostEqual(host_load, running_ratio)

    def test_resource_action_dims_match_actual_vm_counts(self):
        for resource_code, expected in (("S", 25), ("M", 50), ("L", 75)):
            environment = _environment(resource_code)
            self.assertEqual(environment.num_vms, expected)
            self.assertEqual(environment.global_vm_act_dim, expected)

    def test_same_resource_generalization_keeps_action_dim(self):
        for source in ("SS", "SM", "SL"):
            protocol = resolve_experiment_protocol(
                "single", source_scenario=source
            )
            dimensions = {
                RESOURCE_SCALE_REGISTRY[scenario[1]].total_vms
                for scenario in protocol.test_scenarios
            }
            self.assertEqual(
                dimensions,
                {RESOURCE_SCALE_REGISTRY[source[1]].total_vms},
            )

    def test_global_mask_shape_and_vm_to_host_assignment(self):
        environment = _environment("S")
        state, available = environment.get_global_vm_state_for_current_task()
        self.assertTrue(available)
        for key in (
            "mask",
            "legal_action_mask",
            "safety_action_mask",
            "final_action_mask",
        ):
            self.assertEqual(np.asarray(state[key]).shape, (environment.num_vms,))
        action = int(np.flatnonzero(state["mask"] > 0.5)[0])
        expected_vm_id = environment.vm_ids[action]
        expected_host_id = int(environment.vms[expected_vm_id].host_id)
        reward, info = environment.global_vm_assign(action)
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(info["invalid"], 0)
        self.assertEqual(info["vm_global_idx"], action)
        self.assertEqual(info["host_id"], expected_host_id)
        self.assertEqual(info["worker_action_type"], "global_vm_index")

    def test_empty_safe_set_uses_hard_legal_global_fallback(self):
        environment = _environment("S")
        state, available = environment.get_global_vm_state_for_current_task()
        self.assertTrue(available)
        context = environment._current_safety_shield_context
        masks = context["vm_masks_global"]
        legal_actions = np.flatnonzero(masks["legal_action_mask"] > 0.5)
        fallback_action = int(legal_actions[-1])
        masks["safety_action_mask"][:] = 0.0
        masks["safe_legal_action_mask"][:] = 0.0
        masks["final_action_mask"][:] = 0.0
        candidates = []
        for action in legal_actions:
            action = int(action)
            vm_id = int(environment.vm_ids[action])
            metric = context["global_vm_metrics"][action]
            candidates.append(
                {
                    "vm_id": vm_id,
                    "host_id": int(environment.vms[vm_id].host_id),
                    "vm_global_index": action,
                    "predicted_violation_amount": float(
                        metric["predicted_violation_amount"]
                    ),
                    "fuzzy_marginal_energy": float(
                        metric["fuzzy_marginal_energy"]
                    ),
                    "risk_finish": float(metric["predicted_risk"]),
                }
            )
        record = environment.safety_fallback_controller.select(
            candidates, safe_action_count=0
        )
        fallback_action = int(record["selected_candidate"]["vm_global_index"])
        context["fallback_record"] = record
        context["fallback_vm_global_index"] = fallback_action
        proposed = int(legal_actions[0])
        reward, info = environment.global_vm_assign(proposed)
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(info["vm_global_idx"], fallback_action)
        self.assertTrue(info["fallback_applied"])
        self.assertEqual(
            info["vm_shield_decision"]["modification_reason"],
            "no_safe_action_fallback",
        )


class GlobalWorkerConfigTests(unittest.TestCase):
    def test_protocol_seeds_and_network_sizes_are_frozen(self):
        self.assertEqual(SAFE_HRL_VALIDATION_SEEDS, (101, 102, 103))
        self.assertEqual(FINAL_TEST_SEEDS, tuple(range(201, 231)))
        with patch("hrl_mix.train_config.os.makedirs"):
            config = build_train_config(
                protocol="single",
                source_scenario="SS",
                require_deadline_cache=False,
            )
        self.assertEqual(config.worker_agent.hidden_dims, (512, 256))
        self.assertEqual(config.manager_agent.hidden_dims, (512, 256))
        self.assertEqual(config.worker_agent.gamma, 0.99)
        self.assertEqual(config.manager_agent.gamma, 0.95)
        self.assertEqual(config.safe_rl.safety_discount, 0.99)
        self.assertFalse(hasattr(config, "host_agent"))


class CheckpointTopologyTests(unittest.TestCase):
    class Agent:
        safe_rl_enabled = True
        input_dim = 3
        output_dim = 2
        hidden_dims = (512, 256)
        gamma = 0.99
        safety_discount = 0.99
        observation_schema_version = "test_v1"

        def __init__(self, layer):
            self.layer = layer
            self.gamma = 0.95 if layer == "manager" else 0.99

        def save(self, path, *, lagrange_controller_state=None):
            Path(path).write_text(
                json.dumps(
                    {
                        "layer": self.layer,
                        "lagrange": lagrange_controller_state,
                    }
                ),
                encoding="utf-8",
            )

    class Lagrange:
        @staticmethod
        def state_dict():
            return {"state_version": 1, "current_lambda": 1.0}

    def _save(self, directory, metrics):
        agents = {
            layer: self.Agent(layer) for layer in ("manager", "worker")
        }
        return save_best_checkpoint_bundle(
            directory,
            agents=agents,
            lagrange_controller=self.Lagrange(),
            model_metrics=metrics,
            curriculum_state={"stage_id": "test"},
            replay_metadata={layer: {} for layer in agents},
            heuristic_library_version={"manifest_version": "test"},
            config_snapshot={
                "config_snapshot_schema_version": 1,
                "config": {"optimizer_seed": 0},
            },
        )

    def test_checkpoint_has_no_host_and_feasible_manifest_is_preferred(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            fallback = self._save(
                directory,
                _metrics(feasible=False, energy=1.0, worst_violation=0.1),
            )
            feasible = self._save(
                directory,
                _metrics(feasible=True, energy=100.0),
            )
            self.assertTrue(Path(fallback).name.startswith("best_fallback"))
            self.assertTrue(Path(feasible).name.startswith("best_feasible"))
            loaded = read_best_checkpoint_manifest(directory)
            self.assertEqual(loaded["selection_track"], "best_feasible")
            self.assertEqual(
                set(loaded["agent_checkpoints"]), {"manager", "worker"}
            )
            self.assertEqual(
                loaded["agent_architectures"]["manager"]["performance_gamma"],
                0.95,
            )
            self.assertEqual(
                loaded["agent_architectures"]["worker"]["performance_gamma"],
                0.99,
            )
            self.assertEqual(
                loaded["agent_architectures"]["worker"]["safety_gamma"],
                0.99,
            )

    def test_legacy_host_checkpoint_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            manifest = Path(
                self._save(directory, _metrics(feasible=True, energy=10.0))
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["agent_checkpoints"] = {
                "manager": "manager.pth",
                "host": "host.pth",
                "vm": "vm.pth",
            }
            legacy = Path(directory) / "legacy_host_manifest.json"
            legacy.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "layer set mismatch"):
                read_best_checkpoint_manifest(legacy)


if __name__ == "__main__":
    unittest.main()
