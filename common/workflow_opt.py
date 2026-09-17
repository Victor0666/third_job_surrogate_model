# -*- coding: utf-8 -*-
"""
workflow_opt.py
改动要点：
1) Workflow 的秩计算改用真实的数据量字段：
   - 输入：ext_in_bits + from_parents_bits
   - 输出：out_file_size_sum_bits
   - 计算：workload_mi / avg_proc_speed
2) 新增：执行时间组件估计函数 exec_time_components()
   - 根据 VM 的 pc（MI/s）与 bw（Mb/s）计算 (tau_in, tau_comp, tau_out, tau_exec)
   - 支持“与父任务同 VM / 同 Host 时通信视为 0”这样的近似策略
"""

import networkx as nx
from bisect import insort
from collections import defaultdict
from dataclasses import dataclass

BITS_PER_MB = 1024.0 * 1024.0 * 8.0

@dataclass
class LoadRecord:
    start_time: float
    end_time: float
    server_id: int
    vm_pc: float  # MI/s

class Workflow:
    def __init__(self, workflow_id, graph, tasks, arrival_time, source_device_id, deadline=None):
        self.workflow_id = workflow_id
        self.graph = graph
        self.tasks = tasks
        self.arrival_time = arrival_time
        self.source_device_id = source_device_id
        self.deadline = deadline
        self.finish_time = None
        self.is_arrived = False

    def _all_parents_finished(self, task):
        for pred in self.graph.predecessors(task.task_id):
            if self.tasks[pred].state != "Finished":
                return False
        return True

    def get_ready_tasks(self):
        ready = []
        for t in self.tasks:
            if t.state == "unReady" and self._all_parents_finished(t):
                t.state = "Ready"
            if t.state == "Ready":
                ready.append(t)
        return ready

    def compute_upward_ranks(self, avg_proc_speed_mi_s: float, avg_tx_speed_Mb_s: float):
        """向上秩：上传 + 计算 + 下载 + 后继最大秩"""
        for t in self.tasks:
            t.upward_rank = 0.0

        topo = list(nx.topological_sort(self.graph))
        avg_tx_bps = max(avg_tx_speed_Mb_s, 1e-9) * 1e6
        for tid in reversed(topo):
            t = self.tasks[tid]
            upload = (t.ext_in_bits + t.from_parents_bits) / avg_tx_bps
            download = t.out_file_size_sum_bits / avg_tx_bps
            comp = t.workload_mi / max(avg_proc_speed_mi_s, 1e-9)
            self_time = upload + comp + download

            succ_rank = 0.0
            for s in self.graph.successors(tid):
                succ_rank = max(succ_rank, self.tasks[s].upward_rank or 0.0)
            t.upward_rank = self_time + succ_rank

    def compute_downward_ranks(self, avg_proc_speed_mi_s: float, avg_tx_speed_Mb_s: float):
        """向下秩：上传 + 计算 + 下载 + 前驱最大秩"""
        for t in self.tasks:
            t.downward_rank = 0.0

        topo = list(nx.topological_sort(self.graph))
        avg_tx_bps = max(avg_tx_speed_Mb_s, 1e-9) * 1e6
        for tid in topo:
            t = self.tasks[tid]
            upload = (t.ext_in_bits + t.from_parents_bits) / avg_tx_bps
            download = t.out_file_size_sum_bits / avg_tx_bps
            comp = t.workload_mi / max(avg_proc_speed_mi_s, 1e-9)
            self_time = upload + comp + download

            pred_rank = 0.0
            for p in self.graph.predecessors(tid):
                pred_rank = max(pred_rank, self.tasks[p].downward_rank or 0.0)
            t.downward_rank = self_time + pred_rank


def exec_time_components(
    task,
    vm_pc_mi_s: float,
    vm_bw_Mb_s: float,
    *,
    parents_loc: dict | None = None,      # {parent_task_id: (parent_vm_id, parent_host_id)}
    child_vm_id: int | None = None,
    child_host_id: int | None = None,
    zero_if_same_vm: bool = True,
    zero_if_same_host: bool = True,
):
    """
    计算 (tau_in, tau_comp, tau_out, tau_exec)
    - task.workload_mi：MI
    - 带宽换算：bits/s = vm_bw_Mb_s * 1e6
    - 若提供 parents_loc / child_{vm,host}，可在“同 VM/同 Host”时将父→子传输记为 0。
      否则默认将 task.from_parents_bits 全部通过网络传输。
    """
    bps = max(vm_bw_Mb_s, 1e-9) * 1e6

    # 来自父任务的数据量（考虑同机/同VM零传输的近似）
    from_parents_bits_eff = 0.0
    if parents_loc is None or (child_vm_id is None and child_host_id is None):
        from_parents_bits_eff = task.from_parents_bits
    else:
        for p, bits in task.parent_in_bits.items():
            vmh = parents_loc.get(p)
            if vmh is None:
                from_parents_bits_eff += bits
                continue
            p_vm, p_host = vmh
            same_vm = (child_vm_id is not None and p_vm == child_vm_id)
            same_host = (child_host_id is not None and p_host == child_host_id)
            if (same_vm and zero_if_same_vm) or (same_host and zero_if_same_host):
                # 近似：同 VM / 同 Host 视为 0 成本
                pass
            else:
                from_parents_bits_eff += bits

    tau_in = (task.ext_in_bits + from_parents_bits_eff) / bps
    tau_comp = task.workload_mi / max(vm_pc_mi_s, 1e-9)
    tau_out = task.out_file_size_sum_bits / bps
    tau_exec = tau_in + tau_comp + tau_out
    return tau_in, tau_comp, tau_out, tau_exec


def energy_from_records_with_breakdown(
    records,
    hosts,
    total_pc_component="modal",
):
    """单趟积分，同时返回总能量与分主机能量（W·s）。

    ``total_pc_component`` 指定计算 Host 负载率时使用总处理能力三角模糊数的
    ``lower``、``modal`` 或 ``upper`` 分量。LoadRecord.vm_pc 已经是当前影子
    场景的普通浮点处理能力，因此分子不再做隐式模糊数转换。

    默认值 ``modal`` 与旧调用完全一致；SPECpower 功率函数本身仍是确定性的，
    能耗差异只来自三场景资源性能、任务时长及其并发负载时间线。

    两个返回值都与历史实现逐位相同，而非仅数学等价：

    * ``total_energy`` 从 0.0 起按 ``srv_records`` 的插入序逐主机、逐区间累加，
      与旧的一次全量 :func:`energy_from_records` 调用是同一串浮点加法。
    * ``energy_by_server[srv]`` 各自从 0.0 起只累加该主机的区间项，与旧的
      ``energy_from_records(该主机记录, {srv: host})`` 单主机调用是同一串加法。

    因此调用方可以用一趟积分同时取代"一次全量 + 每主机一次"的多趟调用。
    没有任何记录的主机不会出现在字典里，调用方应按 0.0 处理——这与旧实现对空
    记录列表直接返回 0.0 的行为一致。
    """
    if total_pc_component not in {"lower", "modal", "upper"}:
        raise ValueError(
            "total_pc_component must be 'lower', 'modal', or 'upper'."
        )

    srv_records = defaultdict(list)
    for r in records:
        srv_records[r.server_id].append(r)

    total_energy = 0.0
    energy_by_server = {}
    very_small = 1e-9

    for srv_id, recs in srv_records.items():
        host = hosts[srv_id]

        # 端点即事件点：记录在 [t0, t1) 上活跃 <=> start <= t0 且 end >= t1。
        # 由于 end 本身也是时间线上的点，且 t1 是 t0 之后紧邻的点，
        # ``end >= t1`` 与 ``end > t0`` 完全等价，于是"区间过滤"可以退化成
        # 在每个事件点上增删的活跃集扫描——省掉每个区间对全部记录的重扫。
        times = set()
        starts_at = defaultdict(list)
        ends_at = defaultdict(list)
        for index, r in enumerate(recs):
            times.add(r.start_time)
            times.add(r.end_time)
            starts_at[r.start_time].append(index)
            ends_at[r.end_time].append(index)
        timeline = sorted(times)

        # 循环不变量提前求值：component 与 max 的结果在整台主机上恒定，
        # 提出去只是少算几十万次，取值逐位不变。
        total_pc = host.total_pc.component(total_pc_component)
        total_pc_guarded = max(total_pc, very_small)
        power_of = host.power

        active = []
        server_energy = 0.0
        for position in range(len(timeline) - 1):
            t0 = timeline[position]
            t1 = timeline[position + 1]

            # 先加后删：零长记录（start == end）在同一点被加入又立刻移除，
            # 与旧实现里 ``end >= t1`` 判否、从不计入的行为一致。
            # 这两步必须在 very_small 跳过判断之前完成，否则被跳过的极短区间
            # 会让活跃集与时间线脱节。
            for index in starts_at.get(t0, ()):
                insort(active, index)
            for index in ends_at.get(t0, ()):
                active.remove(index)

            if (t1 - t0) < very_small:
                continue

            # active 按 recs 下标升序，因此累加顺序与旧的 ``for r in recs``
            # 顺序过滤逐位相同——浮点加法不满足结合律，顺序不能变。
            vm_pc_sum = 0.0
            for index in active:
                vm_pc_sum += recs[index].vm_pc

            load_ratio = vm_pc_sum / total_pc_guarded
            power = power_of(load_ratio)
            interval_energy = power * (t1 - t0)
            total_energy += interval_energy
            server_energy += interval_energy

        energy_by_server[srv_id] = server_energy

    return total_energy, energy_by_server


def energy_from_records(records, hosts, total_pc_component="modal"):
    """按主机时间分片积分能量（W·s）。

    保留原有签名与返回值语义；实际积分委托给
    :func:`energy_from_records_with_breakdown`，两者的总能量逐位相同。
    """
    total_energy, _ = energy_from_records_with_breakdown(
        records,
        hosts,
        total_pc_component,
    )
    return total_energy
