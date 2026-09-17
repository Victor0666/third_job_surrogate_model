"""验证评估并行化的语义回归测试。

验证 episode 是 ``(env_kwargs, seed, 三个 agent 权重)`` 的纯函数：agent 只被
``deterministic=True, count_step=False`` 的只读推理调用，环境由 seed 完全决定，
汇总只发生在 :func:`hrl_mix.train_eval.aggregate_seed_results`。因此把 seed
分发到工作进程后结果必须逐位不变。本测试锁住：

1. 拆分不改变结果——单 seed 执行 + 顺序聚合与原来的单函数实现完全一致；
2. 聚合顺序由入参顺序决定，与并行完成顺序无关；
3. ``--validation-workers 1`` 严格串行，根本不建进程池；
4. 审计比较器确实能抓出偏差，且不会被墙钟字段的正常差异误伤。
"""

from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
import sys
import unittest

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import numpy as np

from hrl_mix.train_eval import (
    aggregate_seed_results,
    assert_seed_results_identical,
    evaluate_hrl_three_layer_multi_seed,
    evaluation_ctor_kwargs,
)
from hrl_mix.validation_parallel import (
    ValidationEvaluationPool,
    resolve_worker_count,
)


class _StubEnv:
    """最小可评估环境：一个 phase 一次分配，指标只由 seed 决定。"""

    safe_rl_enabled = False

    def __init__(self, random_seed, phases=3, **_ignored):
        self.random_seed = int(random_seed)
        self._phases_total = int(phases)
        self._phase = 0
        self.done_flag = False
        self.total_energy = 0.0
        self._assigned = False

    def reset(self):
        self._phase = 0
        self.done_flag = False
        self.total_energy = 0.0

    def get_manager_state(self):
        return np.zeros(3, dtype=np.float32)

    def get_manager_action_mask(self):
        return np.ones(2, dtype=np.float32)

    def apply_manager_action(self, action):
        del action

    def get_host_state_for_next_assignment(self):
        if self._assigned:
            return None, False
        # 非 safe 模式下层级 mask 走历史 "mask" 字段。
        return {
            "obs": np.full(3, float(self.random_seed), dtype=np.float32),
            "mask": np.ones(2, dtype=np.float32),
        }, True

    def host_select(self, action):
        del action

    def get_vm_state_for_current_task(self):
        return {
            "obs": np.full(3, float(self.random_seed), dtype=np.float32),
            "mask": np.ones(2, dtype=np.float32),
        }, True

    def vm_assign(self, action):
        del action
        self._assigned = True
        # 奖励刻意用不可交换的小数，聚合顺序一变就会露馅。
        r_host = 0.1 * self.random_seed + 0.007 * self._phase
        r_vm = 0.3 * self.random_seed - 0.011 * self._phase
        return r_host, r_vm, {}

    def finish_phase_and_advance(self):
        self._assigned = False
        self._phase += 1
        self.total_energy += 1.0 / (self.random_seed + self._phase)
        self.done_flag = self._phase >= self._phases_total
        return 0.05 * self.random_seed, {"phase": self._phase}


class _StubAgent:
    def __init__(self):
        self.calls = 0

    def select_action(self, state, mask, deterministic=False, count_step=True):
        del mask
        assert deterministic is True
        assert count_step is False
        self.calls += 1
        return int(abs(float(np.sum(state)))) % 2

    def select_action_with_info(
        self, state, mask, deterministic=False, count_step=True
    ):
        del mask
        assert deterministic is True
        assert count_step is False
        self.calls += 1
        action = int(abs(float(np.sum(state)))) % 2
        # select_layer_action_with_info 会读取 selection_type 来补齐审计字段。
        return {
            "action": action,
            "proposed_action": action,
            "selected_by_agent": True,
            "selection_type": "greedy_action",
            "random_exploration": False,
        }


def _legacy_multi_seed(env_cls, env_kwargs, vm, host, mgr, seeds):
    """改动前的实现：单函数内循环 seed 并即时累积。"""
    base_ctor_kwargs = evaluation_ctor_kwargs(env_kwargs)
    vm_list, host_list, mgr_list, energy_list = [], [], [], []
    for sd in seeds:
        ctor_kwargs = dict(base_ctor_kwargs)
        ctor_kwargs["random_seed"] = int(sd)
        env = env_cls(**ctor_kwargs)
        env.reset()
        s = env.get_manager_state()
        mgr.select_action(
            s,
            env.get_manager_action_mask(),
            deterministic=True,
            count_step=False,
        )
        done = False
        phases = 0
        ret_mgr = ret_vm = ret_host = 0.0
        while not done:
            vm_rewards, host_rewards = [], []
            while True:
                st_host, has_next = env.get_host_state_for_next_assignment()
                if not has_next:
                    break
                host.select_action_with_info(
                    st_host["obs"],
                    st_host["mask"],
                    deterministic=True,
                    count_step=False,
                )
                env.host_select(0)
                st_vm, ok_vm = env.get_vm_state_for_current_task()
                if not ok_vm:
                    break
                vm.select_action_with_info(
                    st_vm["obs"],
                    st_vm["mask"],
                    deterministic=True,
                    count_step=False,
                )
                r_host, r_vm, _ = env.vm_assign(0)
                host_rewards.append(float(r_host))
                vm_rewards.append(float(r_vm))
            r_manager_raw, _ = env.finish_phase_and_advance()
            ret_mgr += float(r_manager_raw)
            ret_vm += float(np.mean(vm_rewards)) if vm_rewards else 0.0
            ret_host += float(np.mean(host_rewards)) if host_rewards else 0.0
            phases += 1
            done = bool(env.done_flag)
        vm_list.append(ret_vm / max(phases, 1))
        host_list.append(ret_host / max(phases, 1))
        mgr_list.append(ret_mgr / max(phases, 1))
        energy_list.append(float(env.total_energy))
    return (
        float(np.mean(vm_list)),
        float(np.mean(host_list)),
        float(np.mean(mgr_list)),
        float(np.mean(energy_list)),
    )


class RefactorExactnessTests(unittest.TestCase):
    def test_split_and_aggregate_matches_the_previous_implementation(self):
        seeds = (101, 102, 103)
        env_kwargs = {"phases": 3, "energy_reward_scale": 1.0}
        expected = _legacy_multi_seed(
            _StubEnv, env_kwargs, _StubAgent(), _StubAgent(), _StubAgent(), seeds
        )
        got = evaluate_hrl_three_layer_multi_seed(
            _StubEnv,
            env_kwargs,
            _StubAgent(),
            _StubAgent(),
            _StubAgent(),
            seeds=seeds,
        )
        # 浮点加法不满足结合律，assertEqual 而非 assertAlmostEqual。
        self.assertEqual(got, expected)

    def test_scale_keys_are_stripped_from_ctor_kwargs(self):
        stripped = evaluation_ctor_kwargs(
            {
                "phases": 2,
                "energy_reward_scale": 1.0,
                "task_baseline_norm": 2.0,
                "energy_norm_per_mi_ref": 3.0,
                "alpha_delay_host": 4.0,
                "alpha_delay_vm": 5.0,
            }
        )
        self.assertEqual(stripped, {"phases": 2})


class _RecordingPool:
    """把作业按乱序完成，再按入参顺序写回——模拟真实的池行为。"""

    audit_enabled = False

    def __init__(self, agents):
        self._agents = agents
        self.jobs_seen = []

    def evaluate_jobs(self, jobs, *, return_safety_metrics):
        from hrl_mix.train_eval import evaluate_one_seed

        jobs = list(jobs)
        self.jobs_seen = [seed for _, seed in jobs]
        order = list(range(len(jobs)))
        order.reverse()  # 完成顺序刻意与入参顺序相反
        results = [None] * len(jobs)
        for index in order:
            env_kwargs, seed = jobs[index]
            results[index] = evaluate_one_seed(
                _StubEnv,
                env_kwargs,
                *self._agents,
                seed,
                return_safety_metrics=return_safety_metrics,
            )
        return results

    def evaluate_seeds(self, env_kwargs, seeds, *, return_safety_metrics):
        return self.evaluate_jobs(
            [(env_kwargs, seed) for seed in seeds],
            return_safety_metrics=return_safety_metrics,
        )


class PoolOrderingTests(unittest.TestCase):
    def test_out_of_order_completion_still_aggregates_in_seed_order(self):
        seeds = (101, 102, 103)
        env_kwargs = {"phases": 3}
        serial = evaluate_hrl_three_layer_multi_seed(
            _StubEnv,
            env_kwargs,
            _StubAgent(),
            _StubAgent(),
            _StubAgent(),
            seeds=seeds,
        )
        pool = _RecordingPool((_StubAgent(), _StubAgent(), _StubAgent()))
        pooled = evaluate_hrl_three_layer_multi_seed(
            _StubEnv,
            env_kwargs,
            _StubAgent(),
            _StubAgent(),
            _StubAgent(),
            seeds=seeds,
            evaluation_pool=pool,
        )
        self.assertEqual(pooled, serial)
        self.assertEqual(pool.jobs_seen, [101, 102, 103])

    def test_aggregation_is_order_sensitive(self):
        # 顺序真的会改变结果，所以上一个用例不是空断言。
        # 大数吸收小数：正序先加满 1.0，逆序两个小量先相加得以幸存。
        rows = [
            (0.1, 0.0, 0.0, 1.0, None),
            (0.2, 0.0, 0.0, 1e-16, None),
            (0.3, 0.0, 0.0, 1e-16, None),
        ]
        forward = aggregate_seed_results(rows, return_safety_metrics=False)
        backward = aggregate_seed_results(
            list(reversed(rows)), return_safety_metrics=False
        )
        self.assertNotEqual(forward[3], backward[3])


class WorkerCountTests(unittest.TestCase):
    def test_one_worker_means_strictly_serial(self):
        self.assertEqual(resolve_worker_count(1, job_count=8), 1)

    def test_auto_is_capped_by_the_job_count(self):
        self.assertEqual(resolve_worker_count(0, job_count=2), 2)
        self.assertEqual(resolve_worker_count(64, job_count=3), 3)

    def test_negative_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_worker_count(-1, job_count=4)


class _DeadExecutor:
    def map(self, *_args, **_kwargs):
        raise BrokenProcessPool("worker died")


class BrokenPoolDiagnosticTests(unittest.TestCase):
    """裸的 BrokenProcessPool 只说"worker 没了"，不说为什么。

    spawn 模式下最常见的死因是入口脚本没有 ``__main__`` 守卫——每个 worker
    重新 import 时会把整个训练再跑一遍。这条信息必须出现在报错里，否则排查
    只能从头猜。
    """

    def _dead_pool(self):
        pool = object.__new__(ValidationEvaluationPool)
        pool._executor = _DeadExecutor()
        pool._weights_version = 1
        pool._weights_path = "unused.pt"
        pool._workers = 3
        pool.audit_enabled = False
        return pool

    def test_broken_pool_is_translated_to_an_actionable_error(self):
        pool = self._dead_pool()
        with self.assertRaises(RuntimeError) as caught:
            pool.evaluate_jobs(
                [({}, 101)], return_safety_metrics=False
            )
        message = str(caught.exception)
        self.assertIn("__main__", message)
        self.assertIn("--validation-workers 1", message)
        # 原始异常必须保留，否则真正的死因（比如 OOM）就看不见了。
        self.assertIsInstance(caught.exception.__cause__, BrokenProcessPool)

    def test_publish_weights_is_still_required_first(self):
        pool = self._dead_pool()
        pool._weights_version = 0
        with self.assertRaises(RuntimeError) as caught:
            pool.evaluate_jobs([({}, 101)], return_safety_metrics=False)
        self.assertIn("publish_weights", str(caught.exception))


class AuditComparatorTests(unittest.TestCase):
    def _row(self, energy, *, lateness=0.5, scheduling=1.0):
        return (
            0.1,
            0.2,
            0.3,
            energy,
            {
                "max_fuzzy_lateness": lateness,
                "scheduling_time_seconds": scheduling,
            },
        )

    def test_identical_rows_pass(self):
        assert_seed_results_identical(
            [self._row(1.0)], [self._row(1.0)]
        )

    def test_wall_clock_difference_is_ignored(self):
        assert_seed_results_identical(
            [self._row(1.0, scheduling=1.0)],
            [self._row(1.0, scheduling=9.9)],
        )

    def test_one_ulp_of_energy_is_caught(self):
        assert_seed_results_identical  # 引用一下，避免误删
        with self.assertRaises(AssertionError):
            assert_seed_results_identical(
                [self._row(1.0)],
                [self._row(np.nextafter(1.0, 2.0))],
            )

    def test_metric_difference_is_caught(self):
        with self.assertRaises(AssertionError):
            assert_seed_results_identical(
                [self._row(1.0, lateness=0.5)],
                [self._row(1.0, lateness=0.6)],
            )

    def test_seed_count_mismatch_is_caught(self):
        with self.assertRaises(AssertionError):
            assert_seed_results_identical(
                [self._row(1.0)], [self._row(1.0), self._row(2.0)]
            )


if __name__ == "__main__":
    unittest.main()
