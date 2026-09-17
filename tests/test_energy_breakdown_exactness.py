"""能耗单趟积分与旧多趟实现的逐位一致性回归测试。

`energy_from_records_with_breakdown` 用一趟积分同时给出总能耗与分主机能耗，
取代了 `_energy_reward_to_current_time_with_breakdown` 里"一次全量 + 每主机
一次"的多趟调用。本测试把改动前的实现逐字保留为 `_legacy_energy_from_records`，
并断言两者**精确相等**（`assertEqual` 对浮点即逐位比较），而不是近似相等。

之所以必须是逐位而非数学等价：MARL / PD3QN / IRWS 的训练奖励里有
`energy_delta = max(0.0, after - before)` 这样的大数相减，任何舍入差异都会被
放大；这四个基线的权重已经冻结，仿真器的数值口径不能漂移。
"""

from __future__ import annotations

from collections import defaultdict
import math
import sys
import unittest

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from base.hrl_env import _clip01_numpy_exact
from common.resource_opt import Host, TriangularFuzzyNumber
from common.workflow_opt import (
    LoadRecord,
    energy_from_records,
    energy_from_records_with_breakdown,
)


def _float_bits(value):
    """返回浮点数的原始位模式，用于区分 +0.0 与 -0.0。"""
    import struct

    return struct.pack(">d", value).hex()


def _legacy_energy_from_records(records, hosts, total_pc_component="modal"):
    """改动前 `energy_from_records` 的逐字副本，作为对照基准。"""
    if total_pc_component not in {"lower", "modal", "upper"}:
        raise ValueError(
            "total_pc_component must be 'lower', 'modal', or 'upper'."
        )

    srv_records = defaultdict(list)
    for r in records:
        srv_records[r.server_id].append(r)

    total_energy = 0.0
    very_small = 1e-9

    for srv_id, recs in srv_records.items():
        host = hosts[srv_id]
        times = set()
        for r in recs:
            times.add(r.start_time)
            times.add(r.end_time)
        timeline = sorted(times)

        for t0, t1 in zip(timeline, timeline[1:]):
            if (t1 - t0) < very_small:
                continue
            vm_pc_sum = 0.0
            for r in recs:
                if r.start_time <= t0 and r.end_time >= t1:
                    vm_pc_sum += r.vm_pc

            total_pc = host.total_pc.component(total_pc_component)
            load_ratio = vm_pc_sum / max(total_pc, very_small)
            power = host.power(load_ratio)
            total_energy += power * (t1 - t0)

    return total_energy


def _legacy_breakdown(records, hosts, host_ids, total_pc_component="modal"):
    """改动前 `_energy_reward_to_current_time_with_breakdown` 的分主机口径。

    旧代码把记录按 `int(server_id)` 分桶，再对每个非空桶单独调用一次
    `energy_from_records(桶, {srv: host})`；空桶直接记 0.0。
    """
    by_host = {int(h): [] for h in host_ids}
    for r in records:
        by_host[int(r.server_id)].append(r)

    result = {}
    for h in host_ids:
        bucket = by_host[int(h)]
        if len(bucket) == 0:
            result[int(h)] = 0.0
        else:
            result[int(h)] = _legacy_energy_from_records(
                bucket,
                {int(h): hosts[int(h)]},
                total_pc_component,
            )
    return result


def _make_hosts(host_count):
    """构造功率曲线非线性的主机，放大潜在的浮点结合差异。"""
    hosts = {}
    for host_id in range(host_count):
        base = 90.0 + 7.5 * host_id
        hosts[host_id] = Host(
            host_id=host_id,
            total_pc=TriangularFuzzyNumber(
                8.0 + host_id * 1.3,
                11.0 + host_id * 1.7,
                14.5 + host_id * 2.1,
            ),
            power_model=(
                lambda load, base=base: base
                + 83.7 * load
                - 21.3 * load * load
            ),
        )
    return hosts


def _random_records(rng, host_count, record_count):
    """生成含重叠区间、重复端点、零长与极短区间的记录集。"""
    records = []
    for _ in range(record_count):
        server_id = int(rng.randint(0, host_count))
        start = float(rng.choice([0.0, 1.0, 2.5, 3.0, 7.25])) + float(
            rng.randint(0, 4)
        )
        style = int(rng.randint(0, 4))
        if style == 0:
            # 零长记录：不产生任何区间。
            end = start
        elif style == 1:
            # 短于 very_small 的区间，必须被两侧同样跳过。
            end = start + 1e-12
        else:
            end = start + float(rng.uniform(0.05, 6.0))
        vm_pc = float(rng.uniform(0.4, 5.5))
        records.append(LoadRecord(start, end, server_id, vm_pc))
    return records


class EnergyBreakdownExactnessTests(unittest.TestCase):
    """单趟积分必须与旧多趟实现逐位相同。"""

    components = ("lower", "modal", "upper")

    def test_total_matches_legacy_bitwise(self):
        rng = np.random.RandomState(20260826)
        for trial in range(40):
            host_count = int(rng.randint(1, 5))
            hosts = _make_hosts(host_count)
            records = _random_records(
                rng, host_count, int(rng.randint(0, 60))
            )
            for component in self.components:
                with self.subTest(trial=trial, component=component):
                    expected = _legacy_energy_from_records(
                        records, hosts, component
                    )
                    actual = energy_from_records(records, hosts, component)
                    self.assertEqual(actual, expected)

    def test_per_host_matches_legacy_bitwise(self):
        rng = np.random.RandomState(3141592)
        for trial in range(40):
            host_count = int(rng.randint(1, 5))
            hosts = _make_hosts(host_count)
            host_ids = list(range(host_count))
            records = _random_records(
                rng, host_count, int(rng.randint(0, 60))
            )
            for component in self.components:
                with self.subTest(trial=trial, component=component):
                    expected = _legacy_breakdown(
                        records, hosts, host_ids, component
                    )
                    _, by_server = energy_from_records_with_breakdown(
                        records, hosts, component
                    )
                    actual = {
                        int(h): float(by_server.get(int(h), 0.0))
                        for h in host_ids
                    }
                    self.assertEqual(actual, expected)

    def test_breakdown_total_matches_own_total(self):
        """同一趟返回的总能耗必须与单独调用总能耗完全一致。"""
        rng = np.random.RandomState(2718281)
        for trial in range(20):
            host_count = int(rng.randint(1, 5))
            hosts = _make_hosts(host_count)
            records = _random_records(
                rng, host_count, int(rng.randint(0, 60))
            )
            with self.subTest(trial=trial):
                total, _ = energy_from_records_with_breakdown(records, hosts)
                self.assertEqual(total, energy_from_records(records, hosts))

    def test_hosts_without_records_report_zero(self):
        hosts = _make_hosts(3)
        records = [LoadRecord(0.0, 4.0, 1, 3.0)]
        total, by_server = energy_from_records_with_breakdown(records, hosts)
        self.assertEqual(float(by_server.get(0, 0.0)), 0.0)
        self.assertEqual(float(by_server.get(2, 0.0)), 0.0)
        self.assertEqual(by_server[1], total)

    def test_empty_records_are_zero(self):
        hosts = _make_hosts(2)
        total, by_server = energy_from_records_with_breakdown([], hosts)
        self.assertEqual(total, 0.0)
        self.assertEqual(by_server, {})

    def test_invalid_component_still_rejected(self):
        hosts = _make_hosts(1)
        with self.assertRaises(ValueError):
            energy_from_records_with_breakdown([], hosts, "mean")
        with self.assertRaises(ValueError):
            energy_from_records([], hosts, "mean")


class ScalarClipExactnessTests(unittest.TestCase):
    """标量截断必须与 `float(np.clip(x, 0.0, 1.0))` 逐位相同。"""

    def _assert_same(self, value):
        expected = float(np.clip(value, 0.0, 1.0))
        actual = _clip01_numpy_exact(value)
        if math.isnan(expected):
            self.assertTrue(
                math.isnan(actual),
                msg=f"NaN 未按 numpy 语义传播: {value!r} -> {actual!r}",
            )
            return
        self.assertEqual(
            _float_bits(actual),
            _float_bits(expected),
            msg=f"位模式不一致: {value!r} -> {actual!r} != {expected!r}",
        )
        self.assertIs(type(actual), float)

    def test_edge_values_match_numpy_bitwise(self):
        for value in (
            0.0,
            -0.0,          # numpy 的 clip 返回 +0.0；朴素实现会漏掉符号
            1.0,
            0.5,
            -1e-300,
            1e-300,
            1.0000000000000002,
            0.9999999999999999,
            1e308,
            -1e308,
            float("inf"),
            float("-inf"),
            float("nan"),
        ):
            with self.subTest(value=value):
                self._assert_same(value)

    def test_random_values_match_numpy_bitwise(self):
        rng = np.random.RandomState(11235)
        samples = list(rng.uniform(-2.0, 3.0, 5000))
        samples += list(rng.uniform(-1e-9, 1e-9, 2000))
        for value in samples:
            self._assert_same(float(value))


if __name__ == "__main__":
    unittest.main()
