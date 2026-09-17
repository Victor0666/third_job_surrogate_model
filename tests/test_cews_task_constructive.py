"""cews_task_constructive 的接口、确定性、并发隔离和结果协议回归测试。

多数测试使用最小桩环境，只验证本问题新加的边界，不重复执行昂贵的完整仿真；
完整 DAX/Host/VM/能耗链路由纯评价命令另行覆盖。
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from base.hrl_env import (
    HrlHeftEnv,
    NoFeasibleVMError,
    validate_task_priority_scores,
)
from common.resource_opt import create_cluster
from problems.cews_task_constructive.eval import (
    _add_objective_and_constraints,
    build_environment,
    load_priority_function,
    load_problem_config,
    run_instance,
)
from problems.cews_task_constructive.reference import get_task_priority_v2
# from seevo import SeEvo, individual_comparison_key, parse_result_json
from seevo import (
    SeEvo,
    individual_comparison_key,
    individual_performance_summary,
    parse_result_json,
)
from utils.utils import block_until_running
from surrogate import DecisionTraceCapture


FEATURE_NAMES = (
    "min_exec_time",
    "min_comm_time",
    "min_incremental_energy",
    "slack",
    "upward_rank",
    "remaining_work",
    "ready_wait_time",
    "uncertainty",
)


def _priority_only_environment(feature_count: int):
    """构造只含任务选择所需接口的轻量环境，隔离 VM 与事件逻辑。"""
    # 绕过庞大的真实环境构造器，直接复用待测类方法；所有特征长度保持一致。
    environment = object.__new__(HrlHeftEnv)
    environment._task_id = lambda task: int(task)
    environment.build_task_features = lambda ready: {
        name: np.arange(feature_count, dtype=float) for name in FEATURE_NAMES
    }
    return environment


def _vm_policy_environment(deadline: float):
    """构造一个 ready task、两台 VM 的最小确定性 VM 选择环境。"""
    environment = object.__new__(HrlHeftEnv)
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
    return environment


class TaskPriorityTests(unittest.TestCase):
    """验证候选分数接口、较小分数语义和空 ready 集事件推进。"""

    def test_single_ready_task(self):
        """N=1 时仍返回唯一任务，不能因 squeeze/标量处理破坏形状。"""
        environment = _priority_only_environment(1)
        selected = environment.select_task_with_priority_rule(
            [17], lambda *features: np.array([3.0])
        )
        self.assertEqual(selected, 17)

    def test_multiple_ready_tasks(self):
        """多个 ready tasks 中应选择分数最小项对应的原任务 ID。"""
        environment = _priority_only_environment(3)
        selected = environment.select_task_with_priority_rule(
            [5, 6, 7], lambda *features: np.array([2.0, -1.0, 0.0])
        )
        self.assertEqual(selected, 6)

    def test_wrong_priority_length_is_rejected(self):
        """候选返回长度与 ready 数不一致时必须立即报错，不能静默截断。"""
        with self.assertRaisesRegex(ValueError, "expected"):
            validate_task_priority_scores([1.0], 2)

    def test_nan_priority_is_rejected(self):
        """NaN 不能进入 np.argmin，否则任务选择语义会依赖 NumPy 异常行为。"""
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_task_priority_scores([0.0, np.nan], 2)

    def test_no_ready_task_advances_event(self):
        """空 ready 集不调用 LLM，而应委托现有环境推进下一离散事件。"""
        environment = object.__new__(HrlHeftEnv)
        environment._advance_until_decision_energy_only = mock.Mock(return_value=-0.5)
        reward = environment.advance_to_next_event()
        self.assertEqual(reward, -0.5)
        environment._advance_until_decision_energy_only.assert_called_once_with()


class DeterministicVMPolicyTests(unittest.TestCase):
    """验证 LLM 选任务后固定 VM 策略的硬约束与排序优先级。"""

    def test_all_vms_infeasible(self):
        """没有任何可行 VM 时抛出专用异常，不能制造虚假 VM 或丢弃任务。"""
        environment = _vm_policy_environment(deadline=10.0)
        environment.get_feasible_vms = lambda task: []
        with self.assertRaises(NoFeasibleVMError):
            environment.select_vm_deterministic(0)

    def test_on_time_vm_uses_energy_then_finish(self):
        """存在按时 VM 时能耗优先，即使低能耗 VM 的执行时间略长。"""
        environment = _vm_policy_environment(deadline=10.0)
        exec_time = {0: 4.0, 1: 5.0}
        energy = {0: 10.0, 1: 2.0}
        environment.estimate_exec_time = lambda task, vm: exec_time[int(vm)]
        environment.estimate_comm_time = lambda task, vm: 0.0
        environment.estimate_incremental_energy = lambda task, vm: energy[int(vm)]
        vm_id, details = environment.select_vm_deterministic(0)
        self.assertEqual(vm_id, 1)
        self.assertEqual(details["deadline_violation"], 0.0)

    def test_all_late_vms_use_violation_before_energy(self):
        """全部延期时先最小化违反量，不能让极低能耗掩盖更严重延期。"""
        environment = _vm_policy_environment(deadline=3.0)
        exec_time = {0: 7.0, 1: 5.0}
        energy = {0: 1.0, 1: 100.0}
        environment.estimate_exec_time = lambda task, vm: exec_time[int(vm)]
        environment.estimate_comm_time = lambda task, vm: 0.0
        environment.estimate_incremental_energy = lambda task, vm: energy[int(vm)]
        vm_id, details = environment.select_vm_deterministic(0)
        self.assertEqual(vm_id, 1)
        self.assertEqual(details["deadline_violation"], 2.0)

    def test_cloud_and_edge_vms_exist(self):
        """真实资源构造器能同时提供 cloud 与 edge VM，覆盖两类分配路径。"""
        hosts, vms = create_cluster(
            num_cloud_hosts=1,
            num_edge_hosts=1,
            cloud_vms_per_host=(1,),
            edge_vms_per_host=(1,),
        )
        server_types = {hosts[vm.host_id].server_type for vm in vms.values()}
        self.assertEqual(server_types, {"cloud", "edge"})


def _small_environment(seed=1, workflows=3):
    """构造小规模真实环境，覆盖完整 Host/VM/能耗链路。

    默认三个工作流：任务数超过 VM 数，因而会出现真实资源竞争、多 Host 参与
    以及“全部 VM 忙”的等待步；单个工作流的任务数不超过 VM 数，这些分支都不会
    被触发。
    """
    config = copy.deepcopy(load_problem_config())
    config["dataset"]["workflows_per_instance"] = int(workflows)
    config["problem_size"] = int(workflows)
    environment = build_environment(config, seed)
    environment.reset()
    return environment


def _late_host_environment(deadline=10.0):
    """构造“整台 Host 全部延期、但全局仍存在按时 VM”的最小桩环境。

    这是全局分组与逐 Host 分组唯一会分歧的情形，而实测表明它在 SS 规模的
    真实 workload 中不会自然出现（3/10/20 个工作流共 919 个决策步里一次都没有
    命中），因此必须直接构造，否则等价性测试对该错误没有鉴别力。

    Host 0 的两台 VM 全部延期但边际能耗很低，Host 1 上有一台按时但昂贵的 VM。
    全局规则必须先筛出按时组，因而只能选 Host 1 上那台贵的。
    """
    environment = object.__new__(HrlHeftEnv)
    environment.fuzzy_enabled = False
    environment.task_state = ["Ready"]
    environment.task_meta = [(0, 0)]
    environment.task_baseline_finish = [float(deadline)]
    environment.current_time = 0.0
    environment.vm_ids = [0, 1, 2, 3]
    environment.vms = {
        0: SimpleNamespace(vm_id=0, host_id=0),
        1: SimpleNamespace(vm_id=1, host_id=0),
        2: SimpleNamespace(vm_id=2, host_id=1),
        3: SimpleNamespace(vm_id=3, host_id=1),
    }
    environment.vm_available_at = np.zeros(4, dtype=float)
    environment.get_feasible_vms = lambda task: [0, 1, 2, 3]
    # 完成时刻只由 exec_time 决定：0/1 与 3 延期，只有 2 按时。
    environment.estimate_exec_time = lambda task, vm: {
        0: 20.0,
        1: 21.0,
        2: 5.0,
        3: 30.0,
    }[int(vm)]
    environment.estimate_comm_time = lambda task, vm: 0.0
    # 能耗与按时性刻意反向：若分组出错，低能耗的延期 VM 就会被选中。
    environment.estimate_incremental_energy = lambda task, vm: {
        0: 1.0,
        1: 2.0,
        2: 100.0,
        3: 3.0,
    }[int(vm)]
    return environment


def _select_with_per_host_grouping(environment, task):
    """错误实现：在每台 Host 内部各自做“按时优先”分组。

    只在测试中存在，用来证明上面的边界用例确实能区分正确与错误的实现。
    """
    from base.safety_fallback import select_vm_candidate_by_fixed_rule_order

    task_id = environment._task_id(task)
    deadline = environment._task_deadline(task_id)
    candidates = environment._score_vm_candidates(
        task_id, environment.get_feasible_vms(task_id), deadline
    )
    by_host = {}
    for candidate in candidates:
        host_id = int(environment.vms[int(candidate["vm_id"])].host_id)
        by_host.setdefault(host_id, []).append(candidate)
    host_best = []
    for host_id in sorted(by_host):
        survivors, order_keys = environment._fixed_rule_group_and_keys(
            by_host[host_id], deadline
        )
        host_best.append(
            select_vm_candidate_by_fixed_rule_order(survivors, order_keys)
        )
    _, global_keys = environment._fixed_rule_group_and_keys(
        candidates, deadline
    )
    return int(
        select_vm_candidate_by_fixed_rule_order(host_best, global_keys)["vm_id"]
    )


class _BusyThenIdleEnvironment:
    """ready 集非空但全部 VM 忙的桩环境，用于验证等待而非排队。"""

    def __init__(self, busy_steps):
        self.busy_steps = int(busy_steps)
        self.resource_advances = 0
        self.decision_advances = 0
        self.candidate_calls = []
        self.assigned = []
        self.done_flag = False
        self.current_time = 0.0
        self.completed_workflows = 0
        self.event_heap = [(1.0, "finish")]
        self.next_arrival_idx = 0

    def reset(self):
        return None

    def get_ready_tasks(self):
        return [] if self.done_flag else [0]

    def idle_feasible_vm_ids(self, task):
        del task
        # 前 busy_steps 步没有任何空闲 VM；之后 VM 0 空出来，VM 1 仍在忙。
        return [] if self.resource_advances < self.busy_steps else [0]

    def select_task_with_priority_rule(
        self,
        ready_tasks,
        priority_function,
        return_details=False,
    ):
        del priority_function
        if return_details:
            values = np.asarray([0.0], dtype=float)
            return int(ready_tasks[0]), {
                "selected_task_id": int(ready_tasks[0]),
                "scores": values,
                "features": {
                    name: values.copy()
                    for name in (
                        "min_exec_time", "min_comm_time",
                        "min_incremental_energy", "slack", "upward_rank",
                        "remaining_work", "ready_wait_time", "uncertainty",
                    )
                },
            }
        return int(ready_tasks[0])

    def _task_id(self, task):
        return int(task)

    def select_host_then_vm_deterministic(self, task, candidate_vm_ids=None):
        del task
        self.candidate_calls.append(list(candidate_vm_ids))
        return 0, int(candidate_vm_ids[0]), {"host_id": 0}

    def assign_task(self, task, vm):
        self.assigned.append((int(task), int(vm)))
        self.done_flag = True
        self.completed_workflows = 1

    def advance_to_next_resource_event(self):
        self.resource_advances += 1
        self.current_time += 1.0
        return 0.0

    def advance_to_next_event(self):
        self.decision_advances += 1
        self.current_time += 1.0
        return 0.0


class IdleOnlyHostThenVmTests(unittest.TestCase):
    """验证评价器只从空闲 VM 中选择，且 Host->VM 两级与扁平选择等价。"""

    def test_idle_judgement_matches_comparison_baseline_mask(self):
        """空闲判据必须与比较算法的 global_vm_action_mask 完全一致。"""
        from algorithms.comparisons.fuzzy_common.environment import (
            FuzzyBaselineEnv,
        )

        environment = _small_environment()
        ready_tasks = environment.get_ready_tasks()
        self.assertTrue(ready_tasks)
        task = int(ready_tasks[0])
        # 先在全部 VM 空闲的初始时刻比较，再分配一个任务制造忙 VM 后重比。
        for _ in range(2):
            environment._cur_tid = task
            mask = FuzzyBaselineEnv.global_vm_action_mask(environment)
            expected = {
                int(vm_id)
                for index, vm_id in enumerate(environment.vm_ids)
                if mask[index] > 0.5
            }
            self.assertEqual(
                set(environment.idle_feasible_vm_ids(task)),
                expected,
            )
            idle = environment.idle_feasible_vm_ids(task)
            _, vm_id, _ = environment.select_host_then_vm_deterministic(
                task, candidate_vm_ids=idle
            )
            environment.assign_task(task, vm_id)
            ready_tasks = environment.get_ready_tasks()
            while not ready_tasks and not environment.done_flag:
                environment.advance_to_next_event()
                ready_tasks = environment.get_ready_tasks()
            if not ready_tasks:
                break
            task = int(ready_tasks[0])

    def test_host_then_vm_matches_flat_selection_over_full_episode(self):
        """两级分解必须与扁平固定规则逐位一致，覆盖按时与全延期两个分支。"""
        environment = _small_environment()
        checked = 0
        waits = 0
        hosts_used = set()
        winner_off_lowest_host = 0
        while not environment.done_flag:
            ready_tasks = environment.get_ready_tasks()
            idle = (
                environment.idle_feasible_vm_ids(ready_tasks[0])
                if ready_tasks
                else []
            )
            if ready_tasks and idle:
                task = environment.select_task_with_priority_rule(
                    ready_tasks, get_task_priority_v2
                )
                flat_vm, flat_details = environment.select_vm_deterministic(
                    task, candidate_vm_ids=idle
                )
                host_id, vm_id, details = (
                    environment.select_host_then_vm_deterministic(
                        task, candidate_vm_ids=idle
                    )
                )
                self.assertEqual(vm_id, flat_vm)
                self.assertEqual(
                    host_id, int(environment.vms[vm_id].host_id)
                )
                self.assertEqual(int(details["host_id"]), host_id)
                self.assertEqual(
                    details["incremental_energy"],
                    flat_details["incremental_energy"],
                )
                self.assertEqual(
                    details["predicted_finish_time"],
                    flat_details["predicted_finish_time"],
                )
                # 被选中的 VM 必须确实空闲，queue_time 因而恒为 0。
                self.assertIn(vm_id, idle)
                self.assertEqual(details["queue_time"], 0.0)
                candidate_hosts = sorted(
                    {int(environment.vms[x].host_id) for x in idle}
                )
                if len(candidate_hosts) > 1 and host_id != candidate_hosts[0]:
                    winner_off_lowest_host += 1
                hosts_used.add(host_id)
                checked += 1
                environment.assign_task(task, vm_id)
            elif ready_tasks:
                waits += 1
                environment.advance_to_next_resource_event()
            else:
                environment.advance_to_next_event()
        # 覆盖度断言：若哪天工作流规模退化成无竞争，本测试会立即暴露而不是空过。
        self.assertGreater(checked, 0)
        self.assertGreater(waits, 0, "未覆盖“全部 VM 忙”的等待分支")
        self.assertGreater(len(hosts_used), 1, "未覆盖多 Host 竞争")
        self.assertGreater(
            winner_off_lowest_host,
            0,
            "Host 排序从未生效，等价性断言可能是平凡的",
        )

    def test_host_then_vm_matches_flat_selection_without_candidate_filter(self):
        """不限制候选（含忙 VM）时两级分解同样必须与扁平选择一致。"""
        environment = _small_environment()
        ready_tasks = environment.get_ready_tasks()
        self.assertTrue(ready_tasks)
        task = int(ready_tasks[0])
        flat_vm, _ = environment.select_vm_deterministic(task)
        host_id, vm_id, _ = environment.select_host_then_vm_deterministic(task)
        self.assertEqual(vm_id, flat_vm)
        self.assertEqual(host_id, int(environment.vms[vm_id].host_id))

    def test_hosts_without_idle_vm_are_excluded(self):
        """没有空闲 VM 的 Host 不参与选择，被选 Host 必须含所选 VM。"""
        environment = _small_environment()
        ready_tasks = environment.get_ready_tasks()
        task = int(ready_tasks[0])
        idle = environment.idle_feasible_vm_ids(task)
        # 只保留单台 Host 上的空闲 VM，其余 Host 应完全不参与竞争。
        target_host = int(environment.vms[idle[0]].host_id)
        restricted = [
            vm_id
            for vm_id in idle
            if int(environment.vms[vm_id].host_id) == target_host
        ]
        host_id, vm_id, details = (
            environment.select_host_then_vm_deterministic(
                task, candidate_vm_ids=restricted
            )
        )
        self.assertEqual(host_id, target_host)
        self.assertIn(vm_id, restricted)
        self.assertEqual(int(details["host_count"]), 1)

    def test_grouping_is_global_when_a_whole_host_is_late(self):
        """整台 Host 全延期时，分组仍必须在全局候选上一次性完成。"""
        environment = _late_host_environment(deadline=10.0)
        flat_vm, _ = environment.select_vm_deterministic(0)
        host_id, vm_id, details = (
            environment.select_host_then_vm_deterministic(0)
        )

        # 全局按时组只有 VM 2，即便它的边际能耗是最贵的。
        self.assertEqual(flat_vm, 2)
        self.assertEqual(vm_id, flat_vm)
        self.assertEqual(host_id, 1)
        # Host 0 的 VM 全部延期，被全局分组淘汰，因而根本不参与 Host 排名。
        self.assertEqual(int(details["host_count"]), 1)

        # 逐 Host 分组会让 Host 0 的低能耗延期 VM 复活并赢下比较，
        # 证明本用例确实能区分两种实现。
        self.assertEqual(_select_with_per_host_grouping(environment, 0), 0)

    def test_no_feasible_vm_still_raises(self):
        """候选集为空时仍抛 NoFeasibleVMError，不能退化成静默跳过。"""
        environment = _small_environment()
        task = int(environment.get_ready_tasks()[0])
        with self.assertRaises(NoFeasibleVMError):
            environment.select_host_then_vm_deterministic(
                task, candidate_vm_ids=[]
            )


class BusyVmWaitingTests(unittest.TestCase):
    """验证 ready 集非空但全部 VM 忙时等待资源事件而不是排队。"""

    def _run(self, environment, decision_trace_sink=None):
        config = {
            "safety": {"max_decisions": 1000, "max_stalled_steps": 10},
            "dataset": {"workflows_per_instance": 1},
            "problem_size": 1,
        }
        module = "problems.cews_task_constructive.eval"
        with mock.patch(f"{module}._schedule_metrics", return_value={}), \
                mock.patch(
                    f"{module}._add_objective_and_constraints",
                    side_effect=lambda metrics, cfg: dict(metrics),
                ):
            return run_instance(
                lambda *features: np.zeros(1),
                config,
                1,
                environment_factory=lambda cfg, seed: environment,
                decision_trace_sink=decision_trace_sink,
            )

    def test_waits_for_resource_event_without_triggering_stall_guard(self):
        """连续 20 步无空闲 VM 也不能触发 max_stalled_steps=10 的停滞保护。"""
        environment = _BusyThenIdleEnvironment(busy_steps=20)
        self._run(environment)
        self.assertEqual(environment.resource_advances, 20)
        # ready 集始终非空，因此决策点推进接口一次都不应被调用。
        self.assertEqual(environment.decision_advances, 0)
        self.assertEqual(environment.assigned, [(0, 0)])

    def test_only_idle_vms_are_offered_as_candidates(self):
        """固定规则收到的候选必须只有空闲 VM，忙 VM 不得进入候选集。"""
        environment = _BusyThenIdleEnvironment(busy_steps=3)
        self._run(environment)
        self.assertEqual(environment.candidate_calls, [[0]])

    def test_complete_trace_uses_the_formal_event_loop(self):
        class Sink:
            def __init__(self):
                self.records = []
                self.finalized = None

            def record(self, ready_ids, details):
                self.records.append((list(ready_ids), details["selected_task_id"]))

            def finalize(self, **kwargs):
                self.finalized = kwargs

        environment = _BusyThenIdleEnvironment(busy_steps=3)
        sink = Sink()
        self._run(environment, sink)
        self.assertEqual(environment.resource_advances, 3)
        self.assertEqual(sink.records, [([0], 0)])
        self.assertEqual(
            sink.finalized,
            {"expected_workflows": 1, "completed_workflows": 1},
        )


class DecisionTraceCaptureTests(unittest.TestCase):
    def test_real_small_environment_metrics_are_unchanged_by_capture(self):
        config = copy.deepcopy(load_problem_config())
        config["dataset"]["workflows_per_instance"] = 3
        config["problem_size"] = 3
        baseline = run_instance(get_task_priority_v2, config, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SS_seed1.npz"
            sink = DecisionTraceCapture(
                path,
                baseline_frozen_rule_hash="a" * 64,
                evaluation_config_sha256="b" * 64,
                scenario_id="SS",
                seed=1,
                max_mb=16,
            )
            captured = run_instance(
                get_task_priority_v2,
                config,
                1,
                decision_trace_sink=sink,
            )
            self.assertEqual(captured, baseline)
            with np.load(path, allow_pickle=False) as trace:
                self.assertTrue(bool(trace["complete"].item()))
                self.assertEqual(
                    int(trace["decision_count"].item()),
                    len(trace["selected_indices"]),
                )


class CandidateIsolationTests(unittest.TestCase):
    """验证并发候选模块不会共享文件内容或 Python 模块缓存。"""

    def test_two_candidates_load_concurrently_without_cross_read(self):
        """并发反复加载两个常量不同的规则，结果必须始终与各自文件对应。"""
        template = """import numpy as np
def get_task_priority_v2(a, b, c, d, e, f, g, h):
    return np.full(np.asarray(a).shape, {value}.0)
"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            first = Path(directory) / "candidate_iter1_ind0.py"
            second = Path(directory) / "candidate_iter1_ind1.py"
            first.write_text(template.format(value=11), encoding="utf-8")
            second.write_text(template.format(value=22), encoding="utf-8")

            def load_value(path):
                function = load_priority_function(path)
                inputs = [np.array([1.0, 2.0])] * 8
                return float(function(*inputs)[0])

            # 交错提交同一路径，增加模块缓存或共享文件竞态被触发的机会。
            paths = [first, second] * 10
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                values = list(executor.map(load_value, paths))
            self.assertEqual(values[0::2], [11.0] * 10)
            self.assertEqual(values[1::2], [22.0] * 10)

    def test_seevo_writes_distinct_candidate_modules(self):
        """SeEvo 应按迭代号/个体号写独立文件，并把对应路径传给 eval.py。"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            problem_dir = root / "problems" / "cews_task_constructive"
            generated_dir = problem_dir / "generated"
            generated_dir.mkdir(parents=True)
            (problem_dir / "eval.py").write_text("", encoding="utf-8")
            config_dir = root / "cfg" / "problem"
            config_dir.mkdir(parents=True)
            (config_dir / "cews_task_constructive.yaml").write_text("{}", encoding="utf-8")
            effective_config = root / "effective_cews_task_constructive_config.yaml"
            effective_config.write_text(
                "problem_size: 5\ndataset:\n  workflows_per_instance: 5\n",
                encoding="utf-8",
            )

            algorithm = object.__new__(SeEvo)
            algorithm.iteration = 4
            algorithm.generated_dir = str(generated_dir)
            algorithm.problem_dir = str(problem_dir)
            algorithm.root_dir = str(root)
            algorithm.problem = "cews_task_constructive"
            algorithm.mode = "train"
            algorithm.effective_problem_config_path = str(effective_config)
            individuals = [
                {
                    "code": "VALUE = 1",
                    "stdout_filepath": str(root / "stdout0.txt"),
                    "response_id": 0,
                },
                {
                    "code": "VALUE = 2",
                    "stdout_filepath": str(root / "stdout1.txt"),
                    "response_id": 1,
                },
            ]

            # 不真正启动评价子进程，只检查生成文件和 Popen 命令参数。
            with mock.patch("seevo.subprocess.Popen") as popen, mock.patch(
                "seevo.block_until_running"
            ):
                popen.return_value = SimpleNamespace()
                algorithm._run_code(individuals[0], 0, [0])
                algorithm._run_code(individuals[1], 1, [0])

            first_path = Path(individuals[0]["code_path"])
            second_path = Path(individuals[1]["code_path"])
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(first_path.read_text(encoding="utf-8").strip(), "VALUE = 1")
            self.assertEqual(second_path.read_text(encoding="utf-8").strip(), "VALUE = 2")
            commands = [call.args[0] for call in popen.call_args_list]
            self.assertIn(str(first_path), commands[0])
            self.assertIn(str(second_path), commands[1])
            self.assertIn(str(effective_config), commands[0])
            self.assertIn(str(effective_config), commands[1])
            # Windows 下必须显式强制评价子进程使用 UTF-8，否则中文绝对路径可能
            # 按 GBK 写入 stdout，导致父进程监视日志时解码失败。
            for call in popen.call_args_list:
                self.assertEqual(call.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
                self.assertEqual(call.kwargs["env"]["PYTHONUTF8"], "1")

    def test_startup_monitor_tolerates_legacy_gbk_log(self):
        """启动监视器应能读取旧 GBK 日志，不能把编码问题误判成 seed 无效。"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            stdout_path = Path(directory) / "legacy_gbk_stdout.txt"
            stdout_path.write_bytes(
                "[cews-eval] 候选评价已经启动".encode("gbk")
            )
            block_until_running(str(stdout_path))


class ResultProtocolTests(unittest.TestCase):
    """验证 RESULT_JSON 定位、完整指标保存及无效候选惩罚。"""

    def test_result_json_is_parsed_from_last_matching_line(self):
        """stdout 含多条结果时必须选择最后一条，而不是依赖固定行号。"""
        payload = {"objective": 1.25, "energy": 100.0, "violation_rate": 0.1}
        stdout = "log line\nRESULT_JSON={\"objective\": 99}\nmore\nRESULT_JSON=" + json.dumps(payload)
        self.assertEqual(parse_result_json(stdout), payload)

    def test_evaluate_population_stores_metrics(self):
        """合法结果应同时写入用于选择的 obj 和用于分析的完整 metrics。"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            stdout_path = Path(directory) / "valid.txt"
            stdout_path.write_text(
                'progress\nRESULT_JSON={"objective": 2.5, "energy": 7.0}\n',
                encoding="utf-8",
            )
            algorithm = object.__new__(SeEvo)
            algorithm.iteration = 0
            algorithm.cfg = SimpleNamespace(timeout=1)
            algorithm.obj_type = "min"
            # 用假进程复用真实 evaluate_population 解析逻辑，避免启动 Python 子进程。
            algorithm._run_code = lambda individual, response_id, cases: SimpleNamespace(
                communicate=lambda timeout: (None, None),
                kill=lambda: None,
            )
            population = [{
                "code": "def get_task_priority_v2(*args): return [0]",
                "stdout_filepath": str(stdout_path),
                "response_id": 0,
            }]
            result = algorithm.evaluate_population(population, [0])
            self.assertTrue(result[0]["exec_success"])
            self.assertEqual(result[0]["obj"], 2.5)
            self.assertEqual(result[0]["metrics"]["energy"], 7.0)

    def test_individual_performance_summary_contains_fuzzy_components(self):
        """反思摘要应包含模糊能耗分解、约束及跨 seed 诊断信息。"""
        individual = {
            "obj": 120.0,
            "metrics": {
                "constraint_feasible": True,
                "fuzzy_total_energy_score": 120.0,
                "fuzzy_total_energy_mean": 110.0,
                "fuzzy_total_energy_std": 10.0,
                "total_energy": 105.0,
                "deadline_violation_rate": 0.0,
                "total_lateness": 0.0,
                "objective_std_across_seeds": 3.0,
                "objective_max_across_seeds": 125.0,
                "feasible_seed_rate": 1.0,
            },
        }

        text = individual_performance_summary(individual)

        self.assertIn(
            "fuzzy_energy_score=120.0000",
            text,
        )
        self.assertIn(
            "fuzzy_energy_mean=110.0000",
            text,
        )
        self.assertIn(
            "fuzzy_energy_std=10.0000",
            text,
        )
        self.assertIn(
            "modal_energy=105.0000",
            text,
        )
        self.assertIn(
            "cross_seed_objective_std=3.0000",
            text,
        )
        self.assertIn(
            "worst_seed_objective=125.0000",
            text,
        )
        self.assertIn(
            "feasible_seed_rate=1.0000",
            text,
        )
        self.assertIn(
            "DDL_feasible=True",
            text,
        )

    def test_bad_candidate_output_is_marked_infinite(self):
        """缺少 RESULT_JSON 的候选必须 exec_success=False 且 obj=正无穷。"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            stdout_path = Path(directory) / "invalid.txt"
            stdout_path.write_text("candidate failed without result\n", encoding="utf-8")
            algorithm = object.__new__(SeEvo)
            algorithm.iteration = 0
            algorithm.cfg = SimpleNamespace(timeout=1)
            algorithm.obj_type = "min"
            algorithm._run_code = lambda individual, response_id, cases: SimpleNamespace(
                communicate=lambda timeout: (None, None),
                kill=lambda: None,
            )
            population = [{
                "code": "this is invalid candidate code",
                "stdout_filepath": str(stdout_path),
                "response_id": 0,
            }]
            result = algorithm.evaluate_population(population, [0])
            self.assertFalse(result[0]["exec_success"])
            self.assertEqual(result[0]["obj"], float("inf"))


class ConstrainedEnergyObjectiveTests(unittest.TestCase):
    """验证 objective 是纯能耗，并由独立比较键实施 DDL 约束。"""

    @staticmethod
    def _metrics(energy, violation_rate, total_lateness):
        return {
            "total_energy": float(energy),
            "deadline_violation_rate": float(violation_rate),
            "total_lateness": float(total_lateness),
        }

    def test_objective_equals_total_energy_without_deadline_penalty(self):
        """即使存在延期，objective 也必须严格等于焦耳能耗，不能加权 DDL。"""
        config = {
            "objective": {"metric": "total_energy"},
            "constraints": {"deadline_violation_rate_max": 0.0},
        }
        result = _add_objective_and_constraints(
            self._metrics(energy=123.5, violation_rate=0.4, total_lateness=80.0),
            config,
        )
        self.assertEqual(result["objective"], 123.5)
        self.assertFalse(result["constraint_feasible"])
        self.assertEqual(result["constraint_violation"], 0.4)

    def test_feasible_rule_beats_lower_energy_infeasible_rule(self):
        """可行性是硬约束：可行规则能耗较高时仍必须排在违约规则之前。"""
        feasible = {
            "obj": 200.0,
            "exec_success": True,
            "metrics": {
                "constraint_feasible": True,
                "deadline_violation_rate": 0.0,
                "total_lateness": 0.0,
            },
        }
        infeasible = {
            "obj": 100.0,
            "exec_success": True,
            "metrics": {
                "constraint_feasible": False,
                "constraint_violation": 0.1,
                "constraint_secondary_violation": 1.0,
                "deadline_violation_rate": 0.1,
                "total_lateness": 1.0,
            },
        }
        self.assertLess(
            individual_comparison_key(feasible),
            individual_comparison_key(infeasible),
        )

    def test_infeasible_rules_reduce_violation_before_energy(self):
        """均不可行时先降低违反率和总延期，最后才用纯能耗打破平局。"""
        fewer_violations = {
            "obj": 300.0,
            "exec_success": True,
            "metrics": {
                "constraint_feasible": False,
                "constraint_violation": 0.1,
                "constraint_secondary_violation": 50.0,
            },
        }
        lower_energy_but_more_violations = {
            "obj": 10.0,
            "exec_success": True,
            "metrics": {
                "constraint_feasible": False,
                "constraint_violation": 0.2,
                "constraint_secondary_violation": 1.0,
            },
        }
        self.assertLess(
            individual_comparison_key(fewer_violations),
            individual_comparison_key(lower_energy_but_more_violations),
        )


if __name__ == "__main__":
    unittest.main()
