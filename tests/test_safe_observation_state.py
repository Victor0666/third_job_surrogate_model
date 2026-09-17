"""阶段 6 三层安全 observation 维度、schema 和有限性测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from base.hrl_env import HrlFcfsCacheEnv
from hrl_mix.train_config import SafeRLConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_environment(
    *,
    safe_rl_enabled,
    safe_rl_state_enabled,
):
    environment = HrlFcfsCacheEnv(
        dax_paths=[
            str(
                PROJECT_ROOT
                / "data"
                / "dax"
                / "Montage_25.xml"
            )
        ],
        deadline_mode="none",
        workflows_per_episode=1,
        horizon=1e6,
        arrival_lambda=0.03,
        random_seed=0,
        max_ready_tasks=32,
        num_cloud_hosts=1,
        num_edge_hosts=1,
        cloud_vms_per_host=(2,),
        edge_vms_per_host=(2,),
        fuzzy_enabled=bool(safe_rl_enabled),
        safe_rl_enabled=bool(safe_rl_enabled),
        safe_rl_state_enabled=bool(safe_rl_state_enabled),
    )
    environment.reset()
    return environment


def _assert_extension_in_schema(
    testcase,
    extension,
    schema,
):
    values = np.asarray(extension, dtype=np.float64).reshape(
        -1,
        len(schema),
    )
    testcase.assertTrue(np.all(np.isfinite(values)))
    for index, feature in enumerate(schema):
        testcase.assertTrue(
            np.all(values[:, index] >= feature["low"] - 1e-6),
            feature["name"],
        )
        testcase.assertTrue(
            np.all(values[:, index] <= feature["high"] + 1e-6),
            feature["name"],
        )


class SafeObservationCompatibilityTests(unittest.TestCase):
    def test_default_and_safe_rl_without_state_keep_legacy_dimensions(self):
        config = SafeRLConfig()
        self.assertFalse(config.enabled)
        self.assertFalse(config.state.enabled)
        self.assertEqual(config.safety_discount, 0.95)
        self.assertEqual(config.safety_learning_rate, 3e-4)
        self.assertEqual(config.safety_loss_weight, 1.0)
        self.assertEqual(
            config.initial_lagrange_multiplier,
            1.0,
        )

        legacy_environment = _make_environment(
            safe_rl_enabled=False,
            safe_rl_state_enabled=False,
        )
        safe_without_state = _make_environment(
            safe_rl_enabled=True,
            safe_rl_state_enabled=False,
        )
        for environment in (
            legacy_environment,
            safe_without_state,
        ):
            self.assertEqual(environment.manager_obs_dim, 15)
            self.assertEqual(
                environment.host_obs_dim,
                environment.host_legacy_obs_dim,
            )
            self.assertEqual(
                environment.vm_obs_dim,
                environment.vm_legacy_obs_dim,
            )
            self.assertEqual(
                environment.get_manager_state().shape,
                (15,),
            )

    def test_safe_state_requires_safe_rl(self):
        with self.assertRaisesRegex(
            ValueError,
            "safe_rl_state_enabled=True requires",
        ):
            _make_environment(
                safe_rl_enabled=False,
                safe_rl_state_enabled=True,
            )


class SafeObservationSchemaTests(unittest.TestCase):
    def setUp(self):
        self.environment = _make_environment(
            safe_rl_enabled=True,
            safe_rl_state_enabled=True,
        )

    def test_dimensions_follow_schema_for_all_three_layers(self):
        environment = self.environment
        schemas = environment.get_observation_schema()
        self.assertEqual(environment.manager_obs_dim, 15 + 11)
        self.assertEqual(
            environment.host_obs_dim,
            environment.host_legacy_obs_dim
            + environment.num_hosts * 6,
        )
        self.assertEqual(
            environment.vm_obs_dim,
            environment.vm_legacy_obs_dim
            + environment.max_vms_per_host * 12,
        )
        for layer in ("manager", "host", "vm"):
            self.assertEqual(
                schemas[layer]["total_dim"],
                getattr(environment, f"{layer}_obs_dim"),
            )
            self.assertTrue(schemas[layer]["safe_state_enabled"])
            self.assertGreater(
                schemas[layer]["safe_extension_dim"],
                0,
            )
        host_feature_names = {
            feature["name"]
            for feature in schemas["host"]["safe_features"]
        }
        self.assertIn(
            "minimum_fuzzy_safety_margin_after_host_norm",
            host_feature_names,
        )

    def test_safe_features_are_finite_and_inside_declared_ranges(self):
        environment = self.environment
        manager_state = environment.get_manager_state()
        manager_schema = environment.get_observation_schema(
            "manager"
        )
        _assert_extension_in_schema(
            self,
            manager_state[manager_schema["legacy_dim"] :],
            manager_schema["safe_features"],
        )

        host_state, has_host = (
            environment.get_host_state_for_next_assignment()
        )
        self.assertTrue(has_host)
        host_schema = environment.get_observation_schema("host")
        _assert_extension_in_schema(
            self,
            host_state["obs"][host_schema["legacy_dim"] :],
            host_schema["safe_features"],
        )
        host_extension = host_state["obs"][
            host_schema["legacy_dim"] :
        ].reshape(environment.num_hosts, 6)
        host_names = [
            feature["name"]
            for feature in host_schema["safe_features"]
        ]
        minimum_margin_index = host_names.index(
            "minimum_fuzzy_safety_margin_after_host_norm"
        )
        budget = environment._workflow_budget_for_task(
            environment._cur_tid
        )
        host_predictions = {
            int(row["host_id"]): row
            for row in environment._current_safety_shield_context[
                "prediction"
            ]["host_predictions"]
        }
        for host_slot, host_id in enumerate(environment.host_ids):
            candidate_rows = host_predictions[host_id][
                "candidate_vm_predictions"
            ]
            expected_minimum_margin = (
                np.clip(
                    min(
                        float(row["safety_margin"])
                        for row in candidate_rows
                    )
                    / budget,
                    -1.0,
                    1.0,
                )
                if candidate_rows
                else 0.0
            )
            self.assertAlmostEqual(
                float(
                    host_extension[
                        host_slot,
                        minimum_margin_index,
                    ]
                ),
                float(expected_minimum_margin),
            )

        proposed_host = int(np.argmax(host_state["mask"]))
        environment.host_select(proposed_host)
        vm_state, has_vm = (
            environment.get_vm_state_for_current_task()
        )
        self.assertTrue(has_vm)
        vm_schema = environment.get_observation_schema("vm")
        vm_extension = vm_state["obs"][vm_schema["legacy_dim"] :]
        _assert_extension_in_schema(
            self,
            vm_extension,
            vm_schema["safe_features"],
        )
        vm_rows = vm_extension.reshape(
            environment.max_vms_per_host,
            12,
        )
        self.assertTrue(
            np.all(vm_rows[:, 0] <= vm_rows[:, 1] + 1e-6)
        )
        self.assertTrue(
            np.all(vm_rows[:, 1] <= vm_rows[:, 2] + 1e-6)
        )

    def test_schema_contains_no_continuous_id_features(self):
        for layer in ("manager", "host", "vm"):
            feature_names = {
                feature["name"]
                for feature in self.environment.get_observation_schema(
                    layer
                )["safe_features"]
            }
            self.assertFalse(
                any(
                    name.endswith("_id")
                    or "task_id" in name
                    or "host_id" in name
                    or "vm_id" in name
                    for name in feature_names
                )
            )

    def test_manager_recent_control_rates_use_bounded_window(self):
        environment = self.environment
        environment._safety_shield_records = [
            {
                "shield_intervened": True,
                "fallback_triggered": True,
            },
            {
                "shield_intervened": True,
                "fallback_triggered": False,
            },
            {
                "shield_intervened": False,
                "fallback_triggered": False,
            },
            {
                "shield_intervened": False,
                "fallback_triggered": False,
            },
        ]
        state = environment.get_manager_state()
        schema = environment.get_observation_schema("manager")
        extension = state[schema["legacy_dim"] :]
        names = [
            feature["name"] for feature in schema["safe_features"]
        ]
        intervention_index = names.index(
            "recent_shield_intervention_rate"
        )
        fallback_index = names.index(
            "recent_fallback_trigger_rate"
        )
        self.assertAlmostEqual(
            float(extension[intervention_index]),
            0.5,
        )
        self.assertAlmostEqual(
            float(extension[fallback_index]),
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
