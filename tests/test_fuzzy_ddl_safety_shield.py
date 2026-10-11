"""阶段 4 Host/VM 模糊 DDL safety shield 测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from base.hrl_env import HrlFcfsCacheEnv
from base.safety_shield import FuzzyDDLSafetyShield
from hrl_mix.train_config import SafeRLConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _prediction(vm_id, *, safe, risk, margin):
    return {
        "vm_id": int(vm_id),
        "is_predicted_safe": bool(safe),
        "risk_finish": float(risk),
        "safety_margin": float(margin),
        "predicted_violation_amount": float(max(0.0, -margin)),
    }


def _metrics(*values):
    return [
        {
            "predicted_risk": float(risk),
            "safety_margin": float(margin),
            "predicted_violation_amount": float(max(0.0, -margin)),
        }
        for risk, margin in values
    ]


class VMSafetyMaskTests(unittest.TestCase):
    """验证一个/多个安全 VM 以及硬合法性冲突。"""

    def test_one_safe_vm_is_the_only_final_action(self):
        shield = FuzzyDDLSafetyShield(enabled=True)
        masks = shield.build_vm_masks(
            [1, 1, 1],
            [10, 11, 12],
            [
                _prediction(10, safe=False, risk=12.0, margin=-2.0),
                _prediction(11, safe=True, risk=9.0, margin=1.0),
                _prediction(12, safe=False, risk=13.0, margin=-3.0),
            ],
        )
        np.testing.assert_array_equal(
            masks["safety_action_mask"],
            [0, 1, 0],
        )
        np.testing.assert_array_equal(
            masks["final_action_mask"],
            [0, 1, 0],
        )

    def test_multiple_safe_vms_are_all_preserved(self):
        shield = FuzzyDDLSafetyShield(enabled=True)
        masks = shield.build_vm_masks(
            [1, 1, 1],
            [10, 11, 12],
            [
                _prediction(10, safe=True, risk=8.0, margin=2.0),
                _prediction(11, safe=True, risk=9.0, margin=1.0),
                _prediction(12, safe=False, risk=12.0, margin=-2.0),
            ],
        )
        np.testing.assert_array_equal(
            masks["final_action_mask"],
            [1, 1, 0],
        )
        self.assertEqual(masks["safe_action_count"], 2)

    def test_legal_and_safety_masks_remain_separate_on_conflict(self):
        shield = FuzzyDDLSafetyShield(enabled=True)
        masks = shield.combine_masks(
            legal_action_mask=[1, 0, 1],
            safety_action_mask=[0, 1, 1],
        )
        np.testing.assert_array_equal(
            masks["legal_action_mask"],
            [1, 0, 1],
        )
        np.testing.assert_array_equal(
            masks["safety_action_mask"],
            [0, 1, 1],
        )
        np.testing.assert_array_equal(
            masks["final_action_mask"],
            [0, 0, 1],
        )


class HostSafetyMaskTests(unittest.TestCase):
    """验证 Host mask 完全由内部硬合法且安全的 VM 聚合。"""

    def test_host_without_safe_vm_is_masked(self):
        shield = FuzzyDDLSafetyShield(enabled=True)
        masks = shield.build_host_masks(
            legal_action_mask=[1, 1],
            safe_legal_vm_count_by_host=[0, 2],
        )
        np.testing.assert_array_equal(
            masks["safety_action_mask"],
            [0, 1],
        )
        np.testing.assert_array_equal(
            masks["final_action_mask"],
            [0, 1],
        )

    def test_all_hosts_without_safe_vm_require_empty_safe_set(self):
        shield = FuzzyDDLSafetyShield(enabled=True)
        masks = shield.build_host_masks(
            legal_action_mask=[1, 1],
            safe_legal_vm_count_by_host=[0, 0],
        )
        np.testing.assert_array_equal(
            masks["final_action_mask"],
            [0, 0],
        )
        decision = shield.resolve_action(
            0,
            masks,
            action_metrics=_metrics((12.0, -2.0), (11.0, -1.0)),
            fallback_action=1,
            layer="host",
        )
        self.assertEqual(decision["executed_action"], 1)
        self.assertTrue(decision["action_modified"])
        self.assertTrue(decision["fallback_applied"])
        self.assertEqual(
            decision["modification_reason"],
            "no_safe_action_fallback",
        )


class ShieldActionResolutionTests(unittest.TestCase):
    """验证关闭、接受、修正和回退的动作记录。"""

    def test_disabled_shield_keeps_legacy_action_and_legal_final_mask(self):
        shield = FuzzyDDLSafetyShield(enabled=False)
        masks = shield.combine_masks(
            legal_action_mask=[1, 1],
            safety_action_mask=[0, 1],
        )
        np.testing.assert_array_equal(
            masks["final_action_mask"],
            [1, 1],
        )
        decision = shield.resolve_action(
            0,
            masks,
            action_metrics=_metrics((12.0, -2.0), (9.0, 1.0)),
            layer="vm",
        )
        self.assertEqual(decision["executed_action"], 0)
        self.assertFalse(decision["action_modified"])
        self.assertEqual(
            decision["modification_reason"],
            "shield_disabled",
        )

    def test_safe_proposed_action_is_not_modified(self):
        shield = FuzzyDDLSafetyShield(enabled=True)
        masks = shield.combine_masks([1, 1], [1, 0])
        decision = shield.resolve_action(
            0,
            masks,
            action_metrics=_metrics((9.0, 1.0), (12.0, -2.0)),
            layer="vm",
        )
        self.assertEqual(decision["rl_proposed_action"], 0)
        self.assertEqual(decision["executed_action"], 0)
        self.assertFalse(decision["action_modified"])
        self.assertTrue(decision["proposal_accepted"])

    def test_unsafe_proposed_action_is_corrected_to_best_safe_action(self):
        shield = FuzzyDDLSafetyShield(enabled=True)
        masks = shield.combine_masks([1, 1, 1], [1, 0, 1])
        decision = shield.resolve_action(
            1,
            masks,
            action_metrics=_metrics(
                (9.0, 1.0),
                (12.0, -2.0),
                (8.0, 2.0),
            ),
            layer="vm",
        )
        self.assertEqual(decision["rl_proposed_action"], 1)
        self.assertEqual(decision["executed_action"], 2)
        self.assertTrue(decision["action_modified"])
        self.assertFalse(decision["fallback_applied"])
        self.assertEqual(
            decision["modification_reason"],
            "proposed_action_predicted_unsafe",
        )
        self.assertEqual(decision["predicted_risk"], 8.0)
        self.assertEqual(decision["safety_margin"], 2.0)
        self.assertTrue(
            {
                "rl_proposed_action",
                "executed_action",
                "action_modified",
                "modification_reason",
                "predicted_risk",
                "safety_margin",
                "legal_action_mask",
                "safety_action_mask",
                "final_action_mask",
            }.issubset(decision)
        )


class SafetyShieldEnvironmentIntegrationTests(unittest.TestCase):
    """验证空安全集由固定 VM 规则接管且 Manager 权重不变。"""

    def test_all_unsafe_actions_use_fixed_vm_fallback_and_are_recorded(self):
        config = SafeRLConfig()
        self.assertFalse(config.shield.enabled)
        self.assertEqual(
            config.shield.fallback_controller,
            "fixed_vm_rule",
        )

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
            num_edge_hosts=0,
            cloud_vms_per_host=(2,),
            edge_vms_per_host=(1,),
            fuzzy_enabled=True,
            safe_rl_enabled=True,
            safe_rl_shield_enabled=True,
        )
        environment.reset()
        manager_weights_before = environment.combo_weights.copy()
        impossible_deadline = float(environment.current_time - 1.0)
        for workflow in environment.workflows:
            workflow.deadline = impossible_deadline
        environment.task_baseline_finish = [
            impossible_deadline
            for _ in environment.task_baseline_finish
        ]

        host_state, has_next = (
            environment.get_host_state_for_next_assignment()
        )
        self.assertTrue(has_next)
        self.assertGreater(
            float(np.sum(host_state["legal_action_mask"])),
            0.0,
        )
        self.assertEqual(
            float(np.sum(host_state["final_action_mask"])),
            0.0,
        )
        self.assertTrue(host_state["safety_fallback_required"])
        np.testing.assert_array_equal(
            host_state["mask"],
            host_state["legal_action_mask"],
        )
        fallback_record = dict(
            environment._current_safety_shield_context[
                "fallback_record"
            ]
        )
        self.assertTrue(fallback_record["fallback_triggered"])
        self.assertEqual(
            fallback_record["fallback_reason"],
            "empty_safe_action_set",
        )
        self.assertEqual(fallback_record["candidate_count"], 2)
        self.assertGreater(
            fallback_record["minimum_violation"],
            0.0,
        )

        proposed_host = int(np.argmax(host_state["mask"]))
        fallback_selection = {
            "action": proposed_host,
            "proposed_action": None,
            "selected_by_agent": False,
            "selection_type": "fallback_action",
            "policy_selection_type": "fallback_action",
        }
        environment.host_select(
            proposed_host,
            action_selection=fallback_selection,
        )
        vm_state, has_vm = environment.get_vm_state_for_current_task()
        self.assertTrue(has_vm)
        self.assertEqual(
            float(np.sum(vm_state["final_action_mask"])),
            0.0,
        )
        self.assertTrue(vm_state["safety_fallback_required"])

        proposed_vm = int(np.argmax(vm_state["mask"]))
        _, _, info = environment.vm_assign(
            proposed_vm,
            action_selection={
                **fallback_selection,
                "action": proposed_vm,
            },
        )
        for field in (
            "energy_reward",
            "completion_reward",
            "waiting_reward",
            "utilization_reward",
            "communication_reward",
            "total_performance_reward",
        ):
            self.assertIn(field, info)
        self.assertAlmostEqual(
            info["total_performance_reward"],
            sum(
                info[field]
                for field in (
                    "energy_reward",
                    "completion_reward",
                    "waiting_reward",
                    "utilization_reward",
                    "communication_reward",
                )
            ),
        )
        self.assertFalse(
            info["performance_reward_safety_cost_included"]
        )
        self.assertEqual(info["manager_phase_id"], 0)
        host_decision = info["host_shield_decision"]
        vm_decision = info["vm_shield_decision"]
        self.assertTrue(host_decision["fallback_applied"])
        self.assertTrue(vm_decision["fallback_applied"])
        self.assertTrue(host_decision["shield_intervened"])
        self.assertTrue(vm_decision["shield_intervened"])
        for decision in (host_decision, vm_decision):
            self.assertEqual(
                decision["action_source"],
                "fallback_action",
            )
            self.assertIsNone(decision["proposed_action"])
            self.assertFalse(decision["selected_by_agent"])
            self.assertTrue(decision["fallback_triggered"])
            self.assertEqual(
                decision["fallback_reason"],
                "empty_safe_action_set",
            )
            self.assertEqual(decision["candidate_count"], 2)
            self.assertEqual(
                decision["minimum_violation"],
                fallback_record["minimum_violation"],
            )
            self.assertEqual(
                decision["selected_host"],
                fallback_record["selected_host"],
            )
            self.assertEqual(
                decision["selected_vm"],
                fallback_record["selected_vm"],
            )
            self.assertEqual(
                decision["tie_break_stage"],
                fallback_record["tie_break_stage"],
            )
        self.assertEqual(
            host_decision["action_modified"],
            host_decision["rl_proposed_action"]
            != host_decision["executed_action"],
        )
        self.assertEqual(
            vm_decision["action_modified"],
            vm_decision["rl_proposed_action"]
            != vm_decision["executed_action"],
        )
        self.assertEqual(
            info["modification_reason"],
            "no_safe_action_fallback",
        )
        self.assertIn(info["vm_global_idx"], [0, 1])
        self.assertEqual(info["hard_constraint_violation"], 0)
        self.assertTrue(info["fallback_triggered"])
        self.assertEqual(
            info["selected_vm"],
            fallback_record["selected_vm"],
        )

        _, phase_info = environment.finish_phase_and_advance()
        self.assertEqual(phase_info["manager_phase_id"], 0)
        self.assertEqual(environment._manager_phase_id, 1)
        self.assertEqual(phase_info["shield_record_count"], 2)
        self.assertEqual(
            phase_info["shield_intervention_count"],
            2,
        )
        self.assertEqual(phase_info["shield_fallback_count"], 2)
        self.assertEqual(phase_info["fallback_action_count"], 2)
        np.testing.assert_array_equal(
            environment.combo_weights,
            manager_weights_before,
        )


if __name__ == "__main__":
    unittest.main()
