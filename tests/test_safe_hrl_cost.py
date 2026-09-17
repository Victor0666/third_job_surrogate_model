"""阶段 1 CMDP reward-cost 环境接口回归测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from base.hrl_env import HrlFcfsCacheEnv, HrlHeftEnv
from common.resource_opt import TriangularFuzzyNumber
from hrl_mix.train_config import SafeRLConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safety_stub(
    *,
    deadline: float,
    finish_tfn: TriangularFuzzyNumber | None,
    current_time: float = 0.0,
    predicted_tfn: TriangularFuzzyNumber | None = None,
):
    """构造只包含安全代价结算状态的最小环境桩。"""
    environment = object.__new__(HrlHeftEnv)
    environment.safe_rl_enabled = True
    environment.fuzzy_enabled = True
    environment.fuzzy_deadline_eta = 0.95
    environment.current_time = float(current_time)
    environment.workflows = [
        SimpleNamespace(deadline=float(deadline), arrival_time=0.0)
    ]
    environment.wf_finish_time = (
        {0: float(finish_tfn.modal)} if finish_tfn is not None else {}
    )
    environment._safety_accounted_workflow_ids = set()
    environment._safety_cumulative_cost = 0.0
    environment._safety_cumulative_deadline_violation_count = 0
    environment._safety_cumulative_completed_workflow_count = 0
    environment._safety_cumulative_fuzzy_lateness_cost = 0.0
    environment._safety_cumulative_process_risk_cost = 0.0
    if finish_tfn is not None:
        environment._workflow_finish_tfn = (
            lambda workflow_id: finish_tfn
        )
    if predicted_tfn is not None:
        environment.predict_workflow_finish_tfn = (
            lambda workflow_id: predicted_tfn
        )
    return environment


class SafeWorkflowCostTests(unittest.TestCase):
    """验证模糊 DDL cost 定义、过程风险与完成事件去重。"""

    def test_completed_workflow_on_time_has_zero_cost(self):
        environment = _safety_stub(
            deadline=10.0,
            finish_tfn=TriangularFuzzyNumber(7.0, 8.0, 9.0),
        )
        info = environment.get_safety_diagnostics()
        self.assertEqual(info["deadline_violation_cost"], 0.0)
        self.assertEqual(info["fuzzy_lateness_cost"], 0.0)
        self.assertEqual(info["process_risk_cost"], 0.0)
        self.assertEqual(info["safety_cost"], 0.0)
        self.assertEqual(info["completed_workflow_count"], 1)

    def test_completed_risk_finish_after_deadline_has_expected_cost(self):
        environment = _safety_stub(
            deadline=10.0,
            finish_tfn=TriangularFuzzyNumber(9.0, 10.0, 12.0),
        )
        info = environment.get_safety_diagnostics()
        # R(T) = 0.05 * 10 + 0.95 * 12 = 11.9。
        self.assertAlmostEqual(info["deadline_violation_cost"], 1.0)
        self.assertAlmostEqual(info["fuzzy_lateness_cost"], 1.9)
        self.assertAlmostEqual(info["safety_cost"], 2.9)
        self.assertEqual(info["deadline_violation_count"], 1)

    def test_unfinished_workflow_risk_after_deadline_is_one(self):
        environment = _safety_stub(
            deadline=10.0,
            finish_tfn=None,
            current_time=5.0,
            predicted_tfn=TriangularFuzzyNumber(9.0, 10.0, 12.0),
        )
        info = environment.get_safety_diagnostics()
        self.assertEqual(info["completed_workflow_count"], 0)
        self.assertEqual(info["deadline_violation_cost"], 0.0)
        self.assertEqual(info["fuzzy_lateness_cost"], 0.0)
        self.assertEqual(info["process_risk_cost"], 1.0)
        self.assertEqual(info["safety_cost"], 1.0)
        self.assertEqual(info["predicted_deadline_violation_count"], 1)

    def test_completed_workflow_is_not_charged_twice(self):
        environment = _safety_stub(
            deadline=10.0,
            finish_tfn=TriangularFuzzyNumber(9.0, 10.0, 12.0),
        )
        first = environment.get_safety_diagnostics()
        second = environment.get_safety_diagnostics()
        self.assertEqual(first["completed_workflow_count"], 1)
        self.assertEqual(first["deadline_violation_cost"], 1.0)
        self.assertEqual(second["completed_workflow_count"], 0)
        self.assertEqual(second["deadline_violation_cost"], 0.0)
        self.assertEqual(second["fuzzy_lateness_cost"], 0.0)
        self.assertEqual(second["safety_cost"], 0.0)
        self.assertEqual(
            second["cumulative_completed_workflow_count"],
            1,
        )


class SafeRLBackwardCompatibilityTests(unittest.TestCase):
    """验证默认配置与旧 Host/VM/Manager 返回接口仍可运行。"""

    def test_safe_rl_config_is_disabled_by_default(self):
        config = SafeRLConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.fuzzy_energy_uncertainty_weight, 1.0)
        self.assertEqual(config.fuzzy_deadline_eta, 0.95)
        self.assertEqual(
            config.replay.transition_schema_version,
            1,
        )
        self.assertEqual(config.replay.near_boundary_margin, 1.0)
        self.assertFalse(config.replay.combined_per_priority)
        self.assertEqual(config.replay.performance_td_weight, 1.0)
        self.assertEqual(config.replay.safety_td_weight, 1.0)

    def test_legacy_environment_return_shapes_still_run(self):
        dax_path = PROJECT_ROOT / "data" / "dax" / "Montage_25.xml"
        environment = HrlFcfsCacheEnv(
            dax_paths=[str(dax_path)],
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
        )
        self.assertFalse(environment.safe_rl_enabled)
        environment.reset()

        host_state, has_next = (
            environment.get_host_state_for_next_assignment()
        )
        self.assertTrue(has_next)
        host_action = int(np.argmax(host_state["mask"]))
        environment.host_select(host_action)
        vm_state, has_vm = environment.get_vm_state_for_current_task()
        self.assertTrue(has_vm)
        vm_action = int(np.argmax(vm_state["mask"]))

        assignment_result = environment.vm_assign(vm_action)
        self.assertEqual(len(assignment_result), 3)
        self.assertIsInstance(assignment_result[0], float)
        self.assertIsInstance(assignment_result[1], float)
        self.assertEqual(assignment_result[2]["safety_cost"], 0.0)

        phase_result = environment.finish_phase_and_advance()
        self.assertEqual(len(phase_result), 2)
        self.assertIsInstance(phase_result[0], float)
        self.assertEqual(phase_result[1]["safety_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
