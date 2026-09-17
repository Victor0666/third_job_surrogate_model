# -*- coding: utf-8 -*-
"""
read_xml_opt_Tsize.py
改动要点：
1) 任务工作量统一改为 MI 单位：Task.workload_mi（MI）。
2) 为每个子任务建立“来自各父任务的输入量”映射：
   - Task.parent_in_bits: {parent_task_id -> bits}
   - Task.from_parents_bits: float（合计）
   - Task.ext_in_bits: float（外部输入 = in_sum - 来自父任务）
3) 保留随机 I/O 文件生成；最终统一到 bits。
"""

import random
import math
import networkx as nx
from xml.etree import ElementTree as ET
from dataclasses import dataclass, field


# -----------------------------
# 基本数据结构
# -----------------------------

@dataclass
class DataFile:
    name: str
    size_bits: float
    source_task_id: int | None  # 由哪个任务生成；若为外部输入则为 None


@dataclass
class Task:
    task_id: int
    parents: list = field(default_factory=list)     # 父任务编号
    children: list = field(default_factory=list)    # 子任务编号

    # 任务负载（统一用 MI）
    workload_mi: float = 0.0

    # 输入/输出文件
    in_files: list = field(default_factory=list)
    out_files: list = field(default_factory=list)

    # 输入/输出总大小（bits）
    in_file_size_sum_bits: float = 0.0
    out_file_size_sum_bits: float = 0.0

    # —— 父→子 I/O 映射（由本模块在构图后填充）——
    parent_in_bits: dict = field(default_factory=dict)  # {parent_id: bits}
    from_parents_bits: float = 0.0                      # sum(parent_in_bits.values())
    ext_in_bits: float = 0.0                            # max(in_sum - from_parents, 0)

    # 运行期（可选）
    state: str = "unReady"
    assigned_server_id: int | None = None
    assigned_vm_pc: float | None = None
    ready_time: float | None = None
    arrival_time: float | None = None
    start_processing_time: float | None = None
    end_processing_time: float | None = None
    finish_time: float | None = None

    # 排序用的两个“秩”
    upward_rank: float | None = None
    downward_rank: float | None = None


def _read_dax_to_graph(xml_path):
    """将 DAX 解析为 nx.DiGraph（节点为 0..N-1 的整数）。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'a_dax': 'http://pegasus.isi.edu/schema/DAX'}

    id_map = {}
    G = nx.DiGraph()

    # 注册所有 job
    for idx, job in enumerate(root.findall('a_dax:job', ns)):
        job_id = job.get('id')
        id_map[job_id] = idx
        G.add_node(idx)

    # 建立父子边
    for child in root.findall('a_dax:child', ns):
        child_ref = child.get('ref')
        if child_ref not in id_map:
            continue
        c = id_map[child_ref]
        for parent in child.findall('a_dax:parent', ns):
            p_ref = parent.get('ref')
            if p_ref in id_map:
                G.add_edge(id_map[p_ref], c)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("解析到的图不是 DAG。")
    return G, id_map


def _randomize_payloads(
    G,
    seed=None,
    length_mi_range=(60, 90),    # 任务计算量（MI）
    io_file_count_range=(1, 3),      # 输入/输出文件个数范围
    file_size_mb_range=(512, 1024),  # 每个文件大小范围（MB）
):
    """为图中的每个任务填充随机计算量与输入/输出文件大小（统一到 MI & bits）。"""
    rng = random.Random(seed)
    tasks = [Task(tid) for tid in G.nodes]

    for t in tasks:
        # 工作量：MI
        mi = rng.uniform(*length_mi_range)
        t.workload_mi = max(abs(mi), 1e-6)

        # 随机生成输入文件
        t.in_files.clear()
        num_in = rng.randint(*io_file_count_range)
        for i in range(num_in):
            mb = rng.uniform(*file_size_mb_range)
            bits = mb * 1024 * 1024 * 8.0
            t.in_files.append(DataFile(name=f"input_{t.task_id}_{i}", size_bits=bits, source_task_id=None))

        # 随机生成输出文件
        t.out_files.clear()
        num_out = rng.randint(*io_file_count_range)
        for i in range(num_out):
            mb = rng.uniform(*file_size_mb_range)
            bits = mb * 1024 * 1024 * 8.0
            t.out_files.append(DataFile(name=f"output_{t.task_id}_{i}", size_bits=bits, source_task_id=t.task_id))

        # 预先记录输入/输出总大小
        t.in_file_size_sum_bits = sum(f.size_bits for f in t.in_files)
        t.out_file_size_sum_bits = sum(f.size_bits for f in t.out_files)

    # 恢复父子列表
    for u, v in G.edges:
        tasks[u].children.append(v)
        tasks[v].parents.append(u)

    # 父→子 I/O 映射：将父任务的输出按出度均分给其每个后继（简化近似）
    for u in G.nodes:
        succs = list(G.successors(u))
        if len(succs) == 0:
            continue
        per_child_bits = tasks[u].out_file_size_sum_bits / float(len(succs))
        for v in succs:
            tasks[v].parent_in_bits[u] = tasks[v].parent_in_bits.get(u, 0.0) + per_child_bits

    # 计算每个子任务的“来自父任务的数据量”和“外部输入数据量”
    for t in tasks:
        t.from_parents_bits = sum(t.parent_in_bits.values()) if t.parent_in_bits else 0.0
        t.ext_in_bits = max(t.in_file_size_sum_bits - t.from_parents_bits, 0.0)

    return tasks


def load_workflow_from_dax(
    xml_path,
    seed=None,
    randomize_payloads=True,
    # length_mi_range=(2000, 3000),
    length_mi_range=(60, 90),
    io_file_count_range=(1, 3),
    file_size_mb_range=(512, 1024),
):
    """高层接口：解析 DAG + 可选随机负载 + 建立父→子 I/O 映射。"""
    G, id_map = _read_dax_to_graph(xml_path)
    if randomize_payloads:
        tasks = _randomize_payloads(
            G, seed=seed,
            length_mi_range=length_mi_range,
            io_file_count_range=io_file_count_range,
            file_size_mb_range=file_size_mb_range,
        )
    else:
        tasks = [Task(tid) for tid in G.nodes]
        for u, v in G.edges:
            tasks[u].children.append(v)
            tasks[v].parents.append(u)
        # 没有随机 I/O 时，parent_in_bits/ext_in_bits 保持为 0
    return G, tasks


def poisson_arrival_times(lmbda, horizon=None, seed=None, n_max=None, start_time=0.0):
    """
    生成泊松过程到达时刻。
    两种用法二选一：
      - 指定 n_max：返回前 n_max 个到达时刻（推荐）
      - 指定 horizon：返回 [start_time, start_time + horizon) 内所有到达时刻（可能很多）
    """
    if (horizon is None) and (n_max is None):
        raise ValueError("poisson_arrival_times: must a_set either horizon or n_max")

    import numpy as np
    rng = np.random.default_rng(seed)

    if n_max is not None:
        gaps = rng.exponential(1.0 / max(lmbda, 1e-12), size=int(n_max))
        return (start_time + np.cumsum(gaps)).tolist()

    # 旧逻辑：按 horizon 生成（可能非常多，谨慎使用）
    t = start_time
    arr = []
    # 保护：如果期望数量过大，直接抛错避免 OOM
    expected = lmbda * float(horizon)
    if expected > 2_000_000:
        raise MemoryError(f"poisson_arrival_times: expected arrivals ~{expected:.0f} is too large. "
                          "Use n_max instead of horizon.")
    while t < start_time + horizon:
        t += rng.exponential(1.0 / max(lmbda, 1e-12))
        if t < start_time + horizon:
            arr.append(t)
    return arr
