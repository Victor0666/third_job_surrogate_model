"""CEWS 三角模糊资源、三场景能耗和模糊 DDL 约束回归测试。"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from base.heuristic_admission import CEWS_EVALUATOR_PROTOCOL_VERSION
from base.hrl_env import HrlHeftEnv
from common.resource_opt import Host, TriangularFuzzyNumber
from common.workflow_opt import LoadRecord, energy_from_records
from problems.cews_task_constructive.eval import (
    _add_objective_and_constraints,
    build_environment,
    evaluate_candidate,
    load_problem_config,
)
from problems.cews_task_constructive.reference import get_task_priority_v2


REFERENCE_PATH = (
    LLM_ROOT / "problems" / "cews_task_constructive" / "reference.py"
)


def _small_config(*, fuzzy_enabled=True):
    """返回只含一个工作流的独立配置，避免测试之间共享可变 YAML 字典。"""
    config = copy.deepcopy(load_problem_config())
    config["dataset"]["workflows_per_instance"] = 1
    config["problem_size"] = 1
    config["fuzzy"]["enabled"] = bool(fuzzy_enabled)
    return config


def _run_small_environment(seed=1):
    """用 reference 规则运行一个完整工作流，并返回结束后的真实环境。"""
    config = _small_config(fuzzy_enabled=True)
    environment = build_environment(config, seed)
    environment.reset()
    while not environment.done_flag:
        ready_tasks = environment.get_ready_tasks()
        if ready_tasks:
            task = environment.select_task_with_priority_rule(
                ready_tasks, get_task_priority_v2
            )
            vm_id, _ = environment.select_vm_deterministic(task)
            environment.assign_task(task, vm_id)
        else:
            environment.advance_to_next_event()
    return environment


def _fuzzy_vm_policy_environment(deadline=10.0):
    """构造只验证 fuzzy VM 字典序所需的最小桩环境。"""
    environment = object.__new__(HrlHeftEnv)
    environment.fuzzy_enabled = True
    environment.task_state = ["Ready"]
    environment.task_meta = [(0, 0)]
    environment.task_baseline_finish = [float(deadline)]
    environment.current_time = 0.0
    environment.vm_ids = [0, 1]
    environment.vms = {
        0: SimpleNamespace(vm_id=0),
        1: SimpleNamespace(vm_id=1),
    }
    environment.vm_available_at = np.array([0.0, 0.0], dtype=float)
    environment.get_feasible_vms = lambda task: [0, 1]
    environment.estimate_exec_time = lambda task, vm: 1.0
    environment.estimate_comm_time = lambda task, vm: 0.0
    environment.estimate_incremental_energy = lambda task, vm: {
        0: 1.0,
        1: 2.0,
    }[int(vm)]
    environment.fuzzy_deadline_measure = lambda finish: (
        finish.upper - 0.05 * (finish.upper - finish.modal)
    )
    return environment


class TriangularFuzzyNumberTests(unittest.TestCase):
    """验证显式模糊运算和论文风险统计公式。"""

    def test_fuzzy_energy_score_formula(self):
        fuzzy_energy = TriangularFuzzyNumber(80.0, 100.0, 120.0)
        expected_mean = (80.0 + 2.0 * 100.0 + 120.0) / 4.0
        expected_std = math.sqrt(
            (
                2.0 * (80.0 - 100.0) ** 2
                + (80.0 - 120.0) ** 2
                + 2.0 * (100.0 - 120.0) ** 2
            )
            / 80.0
        )
        self.assertAlmostEqual(fuzzy_energy.mean(), expected_mean)
        self.assertAlmostEqual(fuzzy_energy.std(), expected_std)
        self.assertAlmostEqual(
            fuzzy_energy.score(1.25),
            expected_mean + 1.25 * expected_std,
        )

    def test_explicit_fuzzy_operations_and_types(self):
        left = TriangularFuzzyNumber(1.0, 2.0, 3.0)
        right = TriangularFuzzyNumber(4.0, 5.0, 6.0)
        self.assertEqual(left.component("modal"), 2.0)
        self.assertEqual(left.fuzzy_add(right).as_tuple(), (5.0, 7.0, 9.0))
        self.assertEqual(left.fuzzy_sub(right).as_tuple(), (-5.0, -3.0, -1.0))
        self.assertEqual(left.fuzzy_scale(2.0).as_tuple(), (2.0, 4.0, 6.0))
        self.assertEqual(
            TriangularFuzzyNumber.fuzzy_max(left, right).as_tuple(),
            (4.0, 5.0, 6.0),
        )
        with self.assertRaises(ValueError):
            left.component("mean")
        with self.assertRaises(TypeError):
            left.fuzzy_add(1.0)
        with self.assertRaises(ValueError):
            left.fuzzy_scale(-1.0)


class FuzzyEnergyReplayTests(unittest.TestCase):
    """验证三场景 Host 负载积分与资源 seed 的确定性。"""

    def test_degenerate_tfn_matches_modal_energy(self):
        host = Host(
            host_id=0,
            total_pc=TriangularFuzzyNumber(10.0, 10.0, 10.0),
            power_model=lambda load: 20.0 + 30.0 * load,
        )
        records = [LoadRecord(0.0, 5.0, 0, 5.0)]
        environment = object.__new__(HrlHeftEnv)
        environment.hosts = {0: host}
        environment._records = list(records)
        environment.shadow_records = {
            "optimistic": list(records),
            "pessimistic": list(records),
        }
        environment.fuzzy_enabled = True
        environment.fuzzy_energy_uncertainty_weight = 1.0

        old_modal_energy = energy_from_records(records, {0: host})
        summary = environment.get_fuzzy_energy_summary()
        self.assertEqual(
            summary["fuzzy_total_energy_lower"], old_modal_energy
        )
        self.assertEqual(
            summary["fuzzy_total_energy_modal"], old_modal_energy
        )
        self.assertEqual(
            summary["fuzzy_total_energy_upper"], old_modal_energy
        )
        self.assertEqual(summary["fuzzy_total_energy_std"], 0.0)
        self.assertEqual(
            summary["fuzzy_total_energy_score"], old_modal_energy
        )

    def test_same_seed_same_fuzzy_resources(self):
        config = _small_config(fuzzy_enabled=True)
        first = build_environment(config, 7)
        second = build_environment(config, 7)
        first_values = [
            (vm.pc.as_tuple(), vm.bw.as_tuple())
            for vm in first.vms.values()
        ]
        second_values = [
            (vm.pc.as_tuple(), vm.bw.as_tuple())
            for vm in second.vms.values()
        ]
        self.assertEqual(first_values, second_values)

    def test_different_seed_changes_fuzzy_bounds(self):
        config = _small_config(fuzzy_enabled=True)
        first = build_environment(config, 7)
        second = build_environment(config, 8)
        first_values = [
            (vm.pc.as_tuple(), vm.bw.as_tuple())
            for vm in first.vms.values()
        ]
        second_values = [
            (vm.pc.as_tuple(), vm.bw.as_tuple())
            for vm in second.vms.values()
        ]
        self.assertTrue(
            any(a != b for a, b in zip(first_values, second_values))
        )
        self.assertEqual(
            [vm.pc.modal for vm in first.vms.values()],
            [vm.pc.modal for vm in second.vms.values()],
        )
        self.assertEqual(
            [vm.bw.modal for vm in first.vms.values()],
            [vm.bw.modal for vm in second.vms.values()],
        )

    def test_shadow_finish_order(self):
        environment = _run_small_environment(seed=1)
        self.assertGreater(len(environment.assignment_history), 0)
        for assignment in environment.assignment_history:
            self.assertLessEqual(
                assignment["finish_time_optimistic"],
                assignment["finish_time_modal"] + 1e-8,
            )
            self.assertLessEqual(
                assignment["finish_time_modal"],
                assignment["finish_time_pessimistic"] + 1e-8,
            )


class FuzzyDeadlineAndObjectiveTests(unittest.TestCase):
    """验证 eta 风险完成时刻、纯模糊能耗目标和 feasibility-first 输入。"""

    def test_fuzzy_deadline_measure(self):
        environment = object.__new__(HrlHeftEnv)
        environment.fuzzy_deadline_eta = 0.95
        finish = TriangularFuzzyNumber(80.0, 100.0, 120.0)
        self.assertAlmostEqual(
            environment.fuzzy_deadline_measure(finish),
            119.0,
        )

    def test_objective_is_only_fuzzy_energy(self):
        config = {
            "objective": {"metric": "total_energy"},
            "constraints": {"deadline_violation_rate_max": 0.0},
            "fuzzy": {"enabled": True},
        }
        base_metrics = {
            "total_energy": 90.0,
            "fuzzy_total_energy_score": 100.0,
            "deadline_violation_rate": 0.0,
            "total_lateness": 0.0,
        }
        late_metrics = dict(
            base_metrics,
            deadline_violation_rate=0.5,
            total_lateness=500.0,
        )
        feasible = _add_objective_and_constraints(base_metrics, config)
        infeasible = _add_objective_and_constraints(late_metrics, config)
        self.assertEqual(feasible["objective"], 100.0)
        self.assertEqual(infeasible["objective"], 100.0)
        self.assertTrue(feasible["constraint_feasible"])
        self.assertFalse(infeasible["constraint_feasible"])


class FuzzyVMSelectionTests(unittest.TestCase):
    """验证 fuzzy 模式固定 VM 规则仍先保证截止期可行性。"""

    def test_fuzzy_vm_selection_prefers_feasible(self):
        environment = _fuzzy_vm_policy_environment(deadline=10.0)
        finishes = {
            0: TriangularFuzzyNumber(5.0, 6.0, 20.0),
            1: TriangularFuzzyNumber(5.0, 6.0, 8.0),
        }
        environment.estimate_task_finish_tfn = (
            lambda task, vm: finishes[int(vm)]
        )
        environment.estimate_incremental_energy_score = lambda task, vm: {
            0: 1.0,
            1: 100.0,
        }[int(vm)]
        selected, details = environment.select_vm_deterministic(0)
        self.assertEqual(selected, 1)
        self.assertEqual(details["deadline_violation"], 0.0)

    def test_fuzzy_vm_selection_uses_energy_within_feasible_set(self):
        environment = _fuzzy_vm_policy_environment(deadline=10.0)
        finishes = {
            0: TriangularFuzzyNumber(4.0, 5.0, 7.0),
            1: TriangularFuzzyNumber(3.0, 4.0, 6.0),
        }
        environment.estimate_task_finish_tfn = (
            lambda task, vm: finishes[int(vm)]
        )
        environment.estimate_incremental_energy_score = lambda task, vm: {
            0: 2.0,
            1: 10.0,
        }[int(vm)]
        selected, _ = environment.select_vm_deterministic(0)
        self.assertEqual(selected, 0)

    def test_modal_mode_backward_compatible(self):
        environment = _fuzzy_vm_policy_environment(deadline=10.0)
        environment.fuzzy_enabled = False
        environment.estimate_exec_time = lambda task, vm: {
            0: 4.0,
            1: 5.0,
        }[int(vm)]
        environment.estimate_incremental_energy = lambda task, vm: {
            0: 10.0,
            1: 2.0,
        }[int(vm)]
        selected, _ = environment.select_vm_deterministic(0)
        self.assertEqual(selected, 1)

        metrics = {
            "total_energy": 123.0,
            "deadline_violation_rate": 0.0,
            "total_lateness": 0.0,
        }
        config = {
            "objective": {"metric": "total_energy"},
            "constraints": {"deadline_violation_rate_max": 0.0},
            "fuzzy": {"enabled": False},
        }
        result = _add_objective_and_constraints(metrics, config)
        self.assertEqual(result["objective"], 123.0)


class FuzzyResultProtocolTests(unittest.TestCase):
    """用 reference.py 完成最小真实评价并核对全部模糊字段。"""

    def test_result_json_has_all_fuzzy_fields(self):
        config = _small_config(fuzzy_enabled=True)
        result = evaluate_candidate(
            REFERENCE_PATH,
            config,
            seeds=[1],
        )
        required = {
            "fuzzy_total_energy_lower",
            "fuzzy_total_energy_modal",
            "fuzzy_total_energy_upper",
            "fuzzy_total_energy_mean",
            "fuzzy_total_energy_std",
            "fuzzy_total_energy_score",
            "energy_optimistic",
            "energy_pessimistic",
            "modal_deadline_violation_rate",
            "modal_total_lateness",
            "modal_average_lateness",

            # 跨 seed 诊断字段，只用于反思与分析，不进入 objective。
            "objective_std_across_seeds",
            "objective_max_across_seeds",
            "objective_cv_across_seeds",
            "fuzzy_energy_std_mean_across_seeds",
            "feasible_seed_rate",
            "max_deadline_violation_rate_across_seeds",
            "max_fuzzy_lateness",
            "evaluation_seed_count",
            "completed_seed_count",
        }
        self.assertTrue(required.issubset(result))
        for key in required:
            self.assertTrue(math.isfinite(float(result[key])), key)
        # 与 eval.py 的 RESULT_JSON 协议一致：严格 JSON 禁止 NaN/Infinity。
        payload = json.dumps(result, allow_nan=False)
        parsed = json.loads(payload)
        self.assertEqual(
            parsed["objective"],
            parsed["fuzzy_total_energy_score"],
        )
        self.assertEqual(parsed["energy"], parsed["objective"])
        self.assertEqual(parsed["modal_energy"], parsed["total_energy"])

        # 当前测试只使用 seeds=[1]，因此跨 seed 标准差必须为 0。
        self.assertEqual(
            parsed["objective_std_across_seeds"],
            0.0,
        )

        # 单 seed 下，最差 seed 的目标就是该 seed 的 objective。
        self.assertEqual(
            parsed["objective_max_across_seeds"],
            parsed["objective"],
        )

        # 单 seed 下，seed 内模糊能耗标准差的均值等于该 seed 自身的标准差。
        self.assertEqual(
            parsed["fuzzy_energy_std_mean_across_seeds"],
            parsed["fuzzy_total_energy_std"],
        )

        # constraint_feasible 表示所有 seed 均可行；
        # feasible_seed_rate=1.0 与“所有 seed 可行”等价。
        self.assertEqual(
            parsed["constraint_feasible"],
            parsed["feasible_seed_rate"] == 1.0,
        )
        self.assertEqual(
            parsed["evaluator_protocol_version"],
            CEWS_EVALUATOR_PROTOCOL_VERSION,
        )
        self.assertTrue(parsed["interface_valid"])
        self.assertEqual(
            parsed["function_name"],
            "get_task_priority_v2",
        )
        self.assertTrue(parsed["all_evaluation_seeds_completed"])
        self.assertEqual(parsed["evaluation_seed_count"], 1)
        self.assertEqual(parsed["completed_seed_count"], 1)
        self.assertEqual(parsed["seeds"], [1])
        self.assertEqual(len(parsed["candidate_sha256"]), 64)
        self.assertEqual(len(parsed["evaluation_config_sha256"]), 64)
        self.assertEqual(len(parsed["per_seed_metrics"]), 1)
        self.assertEqual(parsed["per_seed_metrics"][0]["seed"], 1)


if __name__ == "__main__":
    unittest.main()
