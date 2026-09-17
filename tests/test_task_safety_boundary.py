"""阶段 3 任务级动态安全边界与候选动作风险测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from base.hrl_env import HrlHeftEnv
from common.resource_opt import TriangularFuzzyNumber


def _constant_tfn(value):
    value = float(value)
    return TriangularFuzzyNumber(value, value, value)


def _task_boundary_environment(
    *,
    children,
    workloads,
    deadline=100.0,
    input_bits=None,
    output_bits=None,
    parent_in_bits=None,
    vm_specs=None,
    current_time=0.0,
):
    """构造任务级边界预测所需的最小真实数据结构。"""
    task_count = len(workloads)
    input_bits = list(input_bits or [0.0] * task_count)
    output_bits = list(output_bits or [0.0] * task_count)
    parent_in_bits = list(
        parent_in_bits or [{} for _ in range(task_count)]
    )
    if vm_specs is None:
        vm_specs = [
            {
                "vm_id": 0,
                "host_id": 0,
                "pc": _constant_tfn(1.0),
                "bw": _constant_tfn(1.0),
            }
        ]

    environment = object.__new__(HrlHeftEnv)
    environment.safe_rl_enabled = True
    environment.fuzzy_enabled = True
    environment.fuzzy_deadline_eta = 0.95
    environment.current_time = float(current_time)
    environment.task_meta = [(0, task_id) for task_id in range(task_count)]
    environment.task_state = ["unReady"] * task_count
    environment.task_state[0] = "Ready"
    environment.ready_task_ids = [0]
    environment.task_children = [list(value) for value in children]
    environment.task_global_parents = [[] for _ in range(task_count)]
    for parent_id, child_ids in enumerate(children):
        for child_id in child_ids:
            environment.task_global_parents[int(child_id)].append(parent_id)
    environment.task_mi = [float(value) for value in workloads]
    environment.task_in_bits = [float(value) for value in input_bits]
    environment.task_out_bits = [float(value) for value in output_bits]
    environment.task_up_rank = [0.0] * task_count
    environment.task_end_time = [0.0] * task_count
    environment.shadow_task_end_time = {
        "optimistic": [0.0] * task_count,
        "pessimistic": [0.0] * task_count,
    }

    tasks = []
    for task_id in range(task_count):
        parent_bits = dict(parent_in_bits[task_id])
        tasks.append(
            SimpleNamespace(
                task_id=task_id,
                parent_in_bits=parent_bits,
                from_parents_bits=float(sum(parent_bits.values())),
                ext_in_bits=max(
                    0.0,
                    float(input_bits[task_id])
                    - float(sum(parent_bits.values())),
                ),
                out_file_size_sum_bits=float(output_bits[task_id]),
            )
        )
    environment.workflows = [
        SimpleNamespace(
            workflow_id=0,
            tasks=tasks,
            arrival_time=0.0,
            deadline=float(deadline),
        )
    ]

    environment.vm_ids = [
        int(specification["vm_id"]) for specification in vm_specs
    ]
    environment.vms = {
        int(specification["vm_id"]): SimpleNamespace(
            vm_id=int(specification["vm_id"]),
            host_id=int(specification["host_id"]),
            pc=specification["pc"],
            bw=specification["bw"],
        )
        for specification in vm_specs
    }
    environment.host_ids = sorted(
        {int(specification["host_id"]) for specification in vm_specs}
    )
    environment.host_to_vm_indices = {
        host_id: [
            vm_index
            for vm_index, vm_id in enumerate(environment.vm_ids)
            if environment.vms[vm_id].host_id == host_id
        ]
        for host_id in environment.host_ids
    }
    environment.vm_available_at = np.zeros(
        len(environment.vm_ids),
        dtype=float,
    )
    environment.shadow_vm_available_at = {
        "optimistic": np.zeros(len(environment.vm_ids), dtype=float),
        "pessimistic": np.zeros(len(environment.vm_ids), dtype=float),
    }
    environment.task_assigned_vm = {}
    environment.task_assigned_host = {}
    return environment


class RemainingCriticalPathTests(unittest.TestCase):
    """验证当前任务自身与完成后剩余关键路径严格分离。"""

    def test_empty_remaining_critical_path(self):
        environment = _task_boundary_environment(
            children=[[]],
            workloads=[5.0],
        )
        remaining = environment.estimate_task_remaining_critical_path(0)
        self.assertEqual(remaining["remaining_critical_path_lower"], 0.0)
        self.assertEqual(remaining["remaining_critical_path_modal"], 0.0)
        self.assertEqual(remaining["remaining_critical_path_upper"], 0.0)
        self.assertEqual(remaining["remaining_critical_path_risk"], 0.0)
        self.assertTrue(remaining["excludes_current_task"])

    def test_single_successor_uses_successor_duration_only(self):
        environment = _task_boundary_environment(
            children=[[1], []],
            workloads=[10.0, 3.0],
            deadline=20.0,
        )
        remaining = environment.estimate_task_remaining_critical_path(0)
        self.assertEqual(remaining["remaining_critical_path_modal"], 3.0)
        self.assertEqual(
            remaining["critical_path_task_ids_modal"],
            [1],
        )
        action = environment.predict_task_vm_action_risk(0, 0)
        self.assertEqual(
            action["current_task_execution_time_modal"],
            10.0,
        )
        self.assertEqual(
            action["remaining_execution_time_modal"],
            3.0,
        )
        self.assertEqual(action["task_safe_deadline"], 17.0)
        self.assertEqual(action["safety_margin"], 7.0)

    def test_multi_branch_dag_uses_longest_successor_path(self):
        environment = _task_boundary_environment(
            children=[[1, 2], [3], [], []],
            workloads=[1.0, 2.0, 4.0, 3.0],
        )
        remaining = environment.estimate_task_remaining_critical_path(0)
        # 分支 0->1->3 为 2+3=5，分支 0->2 为 4。
        self.assertEqual(remaining["remaining_critical_path_modal"], 5.0)
        self.assertEqual(
            remaining["critical_path_task_ids_modal"],
            [1, 3],
        )


class CandidateActionRiskTests(unittest.TestCase):
    """验证通信、排队、VM 风险和 Host 聚合均为只读预测。"""

    def test_zero_communication_time(self):
        environment = _task_boundary_environment(
            children=[[]],
            workloads=[1.0],
            input_bits=[0.0],
            output_bits=[0.0],
        )
        prediction = environment.predict_task_vm_action_risk(0, 0)
        self.assertEqual(
            prediction["current_task_communication_time_optimistic"],
            0.0,
        )
        self.assertEqual(
            prediction["current_task_communication_time_modal"],
            0.0,
        )
        self.assertEqual(
            prediction["current_task_communication_time_pessimistic"],
            0.0,
        )

    def test_cross_host_parent_data_is_detected_and_transferred(self):
        environment = _task_boundary_environment(
            children=[[1], []],
            workloads=[0.0, 0.0],
            input_bits=[0.0, 1_000_000.0],
            parent_in_bits=[{}, {0: 1_000_000.0}],
            vm_specs=[
                {
                    "vm_id": 0,
                    "host_id": 0,
                    "pc": _constant_tfn(1.0),
                    "bw": _constant_tfn(1.0),
                },
                {
                    "vm_id": 1,
                    "host_id": 1,
                    "pc": _constant_tfn(1.0),
                    "bw": _constant_tfn(1.0),
                },
            ],
            current_time=1.0,
        )
        environment.task_state = ["Finished", "Ready"]
        environment.ready_task_ids = [1]
        environment.task_end_time[0] = 1.0
        environment.shadow_task_end_time["optimistic"][0] = 1.0
        environment.shadow_task_end_time["pessimistic"][0] = 1.0
        environment.task_assigned_vm[0] = 0
        environment.task_assigned_host[0] = 0

        prediction = environment.predict_task_vm_action_risk(1, 1)
        self.assertEqual(
            prediction["cross_host_parent_bits"],
            1_000_000.0,
        )
        self.assertEqual(
            prediction["runtime_transferred_parent_bits"],
            1_000_000.0,
        )
        self.assertEqual(
            prediction["current_task_communication_time_modal"],
            1.0,
        )

    def test_vm_queue_is_included_in_all_finish_scenarios(self):
        environment = _task_boundary_environment(
            children=[[]],
            workloads=[1.0],
        )
        environment.vm_available_at[0] = 5.0
        environment.shadow_vm_available_at["optimistic"][0] = 4.0
        environment.shadow_vm_available_at["pessimistic"][0] = 6.0

        prediction = environment.predict_task_vm_action_risk(0, 0)
        self.assertEqual(prediction["optimistic_finish"], 5.0)
        self.assertEqual(prediction["modal_finish"], 6.0)
        self.assertEqual(prediction["pessimistic_finish"], 7.0)
        self.assertEqual(prediction["vm_queue_delay_modal"], 5.0)
        self.assertFalse(prediction["is_currently_selectable"])

    def test_safe_boundary_and_unsafe_vm_feed_host_aggregation(self):
        environment = _task_boundary_environment(
            children=[[]],
            workloads=[10.0],
            deadline=10.0,
            vm_specs=[
                {
                    "vm_id": 0,
                    "host_id": 0,
                    "pc": _constant_tfn(1.0),
                    "bw": _constant_tfn(1.0),
                },
                {
                    "vm_id": 1,
                    "host_id": 0,
                    "pc": _constant_tfn(0.5),
                    "bw": _constant_tfn(1.0),
                },
            ],
        )
        result = environment.get_task_action_risk_predictions(0)
        vm_by_id = {
            prediction["vm_id"]: prediction
            for prediction in result["vm_predictions"]
        }
        self.assertTrue(vm_by_id[0]["is_predicted_safe"])
        self.assertTrue(vm_by_id[0]["is_at_safety_boundary"])
        self.assertEqual(vm_by_id[0]["predicted_violation_amount"], 0.0)
        self.assertFalse(vm_by_id[1]["is_predicted_safe"])
        self.assertEqual(
            vm_by_id[1]["predicted_violation_amount"],
            10.0,
        )

        host = result["host_predictions"][0]
        self.assertEqual(host["candidate_vm_count"], 2)
        self.assertEqual(host["safe_vm_count"], 1)
        self.assertTrue(host["is_predicted_safe"])
        self.assertEqual(host["best_diagnostic_vm_id"], 0)
        self.assertIs(
            host["candidate_vm_predictions"][0],
            vm_by_id[0],
        )
        self.assertFalse(result["action_mask_applied"])
        self.assertFalse(result["action_selection_changed"])

        environment.safe_rl_enabled = False
        disabled = environment.get_task_action_risk_predictions(0)
        self.assertFalse(disabled["prediction_available"])
        self.assertEqual(disabled["vm_predictions"], [])
        self.assertEqual(disabled["host_predictions"], [])
        self.assertFalse(disabled["action_mask_applied"])


if __name__ == "__main__":
    unittest.main()
