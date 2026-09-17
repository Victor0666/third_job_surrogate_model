# -*- coding: utf-8 -*-
"""
hrl_env.py

三层智能体环境（阶段式 HRL + HostAgent + VMAgent）：

L1 Manager（阶段级）【动态权重增量控制】：
- 维护 5 维 raw 权重 w_raw（FCFS/SJF/MCF/HUR/EDF）
- 每个 phase 开始时，Manager 选择一个离散动作 a ∈ {0..242}
  对应 Δw ∈ {-0.1, 0, +0.1}^5（共 3^5=243 种组合）
- 更新：w_raw ← clip(w_raw + Δw, min=0)；若全 0 回退为全 1
- combo_weights = normalize(w_raw) 用于 ready tasks 打分排序
- 提供 get_manager_action_mask()：若某维 w_raw==0，则禁止该维再减 0.1 的动作

Deadline 逻辑：
1) 工作流级 deadline:
   wf.deadline = wf.arrival_time + 1.5 * HEFT_makespan
2) 任务级 deadline（task_baseline_finish）：
   根据 wf_deadline 做 Latest-Finish 反推
3) 工作流级 deadline 用途：
   - EDF 特征
   - Manager state 的 slack 统计
   - workflow lateness 统计

【本版本关键修改】
中层/下层 reward 改为与 MARL 一致的口径：

- HostAgent reward:
    r_host = (1-alpha_delay_host) * r_energy_total + alpha_delay_host * r_delay

- VMAgent reward:
    r_vm   = (1-alpha_delay_vm)   * r_energy_host  + alpha_delay_vm   * r_delay

其中：
- r_delay        = - clip( max(0, finish_time - task_deadline) / task_baseline_norm , 0, 1 )
- r_energy_total = - ΔE_total * energy_reward_scale
- r_energy_host  = - ΔE_host  * energy_reward_scale

注意：
- 若 assignment 后 phase 内还能立刻继续分配，则不推进时间，故 ΔE_total=ΔE_host=0
- 若 assignment 后 phase 内不能继续分配，则立即推进到下一个决策点，
  并将这段真实能耗记到该 assignment 的 lower-layer reward 上
- Manager reward 保持原逻辑不变

上述为 ``safe_rl_enabled=False`` 的兼容路径。安全模式下，三层学习统一消费
每次 assignment 前后风险调整模糊总能耗
``mean + lambda_E * std`` 的负增量（保留正比例 ``energy_reward_scale``），
并把 completion/waiting/utilization/communication shaping 分字段记录。
``safety_cost`` 始终独立，不进入 ``total_performance_reward``。
"""

import heapq
import json
import numpy as np

import os
import re
from collections import deque
from functools import lru_cache

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces

from common.workflow_opt import (
    LoadRecord,
    energy_from_records,
    energy_from_records_with_breakdown,
    Workflow,
)
from common.read_xml_opt_Tsize import load_workflow_from_dax, poisson_arrival_times
from common.resource_opt import TriangularFuzzyNumber, create_cluster
from base.safety_fallback import (
    DeterministicFuzzyDDLFallbackController,
    select_vm_candidate_by_fixed_rule_order,
)
from base.safety_observation import (
    HOST_SAFETY_FEATURE_SCHEMA,
    MANAGER_SAFETY_FEATURE_SCHEMA,
    SAFE_OBSERVATION_SCHEMA_VERSION,
    VM_SAFETY_FEATURE_SCHEMA,
    get_safety_feature_schema,
)
from base.manager_heuristics import (
    HEURISTIC_RECENT_FEATURE_SCHEMA,
    HEURISTIC_SELECTION_MODE,
    LEGACY_RULE_WEIGHT_MODE,
    MANAGER_HEURISTIC_MODES,
    heuristic_action_schema_version,
    heuristic_availability_mask,
    load_manager_heuristic_library,
)
from base.heuristic_admission import (
    workflow_families_from_dax_files,
)
from base.safety_shield import FuzzyDDLSafetyShield
try:
    from scenario_registry import deterministic_workload_sequence
except ModuleNotFoundError:
    from algorithms.llm_safe_hrl.scenario_registry import (
        deterministic_workload_sequence,
    )


def _safe_div(a, b, eps=1e-9):
    """执行带最小分母保护的浮点除法"""
    return float(a) / float(b if abs(b) > eps else (eps if b >= 0 else -eps))


def _clip01(x: float) -> float:
    """将输入值裁剪到闭区间 [0, 1]"""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _clip01_numpy_exact(x: float) -> float:
    """标量版 ``float(np.clip(x, 0.0, 1.0))``，逐位等价。

    与 :func:`_clip01` 的唯一区别是负零：numpy 的 clip 等价于
    ``minimum(maximum(x, 0.0), 1.0)``，``maximum(-0.0, 0.0)`` 给出 **正零**，
    而朴素的 ``if x < 0.0`` 判断对 ``-0.0`` 不成立，会把 ``-0.0`` 原样返回。
    多出来的 ``x == 0.0`` 分支就是为了消掉这个符号差异。

    NaN 与 ±inf 也与 numpy 一致：NaN 两个比较都为假因而原样返回，``-inf``
    截到 0.0，``+inf`` 截到 1.0。
    """
    if x < 0.0 or x == 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _simulator_cache_audit_enabled() -> bool:
    """``SIM_EXACTNESS_AUDIT=1`` 时，缓存命中也重算一遍并断言精确相等。

    仿真器缓存的正确性依据是"被缓存的量在 episode 内恒定"，这是一个必须能被
    验证而不是只能被论证的前提：MARL / PD3QN / IRWS 的训练奖励里含有
    ``max(0.0, after - before)`` 这样的大数相减，任何舍入偏差都会被放大，而这
    三个基线的权重已经冻结。开启本开关跑一遍完整 episode，是"逐位一致"这个
    声明的直接证据。
    """
    return os.environ.get("SIM_EXACTNESS_AUDIT", "") == "1"


class NoFeasibleVMError(RuntimeError):
    """ready task 找不到满足基本处理能力与带宽条件的 VM 时抛出。

    该异常表示环境无法构造合法调度动作，不是普通的候选分数优劣。评价子进程
    会保留 traceback，SeEvo 随后将对应个体标记为无效并赋予无穷大目标值。
    """


def validate_task_priority_scores(scores, ready_count: int) -> np.ndarray:
    """把候选返回值转换并严格验证为长度 ``ready_count`` 的有限一维数组。

    不做 flatten/squeeze 等“自动修复”，因为标量、(N, 1) 或错误长度通常表示
    LLM 生成的接口不符合契约。尽早抛出明确异常可避免错误分数与任务下标错配。
    """
    # dtype=float 同时拒绝不能解释为数值的对象；后续选择只处理统一浮点类型。
    values = np.asarray(scores, dtype=float)
    expected_shape = (int(ready_count),)
    if values.shape != expected_shape:
        raise ValueError(
            f"Task-priority rule returned shape {values.shape}; "
            f"expected {expected_shape}."
        )
    # np.argmin 对 NaN 的行为不适合作为调度语义，因此 NaN 和正负无穷一律无效。
    if not np.all(np.isfinite(values)):
        raise ValueError("Task-priority rule returned NaN or infinite values.")
    return values


MANAGER_STEP = 0.1


def _build_manager_action_table(step: float = 0.1) -> np.ndarray:
    """生成 Manager 五维权重的全部增减动作组合"""
    vals = (-step, 0.0, step)
    table = []
    for d0 in vals:
        for d1 in vals:
            for d2 in vals:
                for d3 in vals:
                    for d4 in vals:
                        table.append([d0, d1, d2, d3, d4])
    return np.asarray(table, dtype=np.float32)


MANAGER_ACTION_TABLE = _build_manager_action_table(MANAGER_STEP)


@lru_cache(maxsize=None)
def _infer_task_count_from_dax_path(dax_path: str) -> int:
    """优先从 DAX 文件名推断任务数，失败时再解析工作流文件"""
    base = os.path.basename(dax_path)
    m = re.search(r"_(\d+)\.xml$", base)
    if m:
        return int(m.group(1))
    try:
        G, tasks = load_workflow_from_dax(dax_path, seed=0, randomize_payloads=False)
        return int(len(tasks))
    except Exception:
        return 0


def _auto_max_ready_tasks(dax_paths, workflows_per_episode: int) -> int:
    """根据最大工作流规模和每轮工作流数估算就绪任务槽位上限"""
    nmax = 0
    for p in dax_paths:
        nmax = max(nmax, _infer_task_count_from_dax_path(str(p)))
    if nmax <= 0:
        nmax = 100
    if workflows_per_episode is None:
        return int(nmax)
    return int(workflows_per_episode * nmax)


def _topo_sort_local(tasks):
    """对单个工作流的局部任务执行拓扑排序并返回后继表"""
    n = len(tasks)
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for t in tasks:
        i = int(t.task_id)
        for c in getattr(t, "children", []):
            succ[i].append(int(c))
        indeg[i] = len(getattr(t, "parents", []))
    q = [i for i in range(n) if indeg[i] == 0]
    topo = []
    while q:
        u = q.pop(0)
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != n:
        topo = list(range(n))
    return topo, succ


def _heft_makespan_and_finish_times(tasks, vm_pc, vm_bw_bps):
    """使用 HEFT 估计工作流完工时间、任务完成时间和平均执行时长"""
    n = len(tasks)
    if n == 0:
        return 0.0, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    topo, succ = _topo_sort_local(tasks)

    mi = np.array([float(t.workload_mi) for t in tasks], dtype=np.float64)
    in_bits = np.array(
        [
            float((getattr(t, "ext_in_bits", 0.0) or 0.0) + (getattr(t, "from_parents_bits", 0.0) or 0.0))
            for t in tasks
        ],
        dtype=np.float64,
    )
    out_bits = np.array([float(getattr(t, "out_file_size_sum_bits", 0.0) or 0.0) for t in tasks], dtype=np.float64)

    bw = vm_bw_bps.reshape(1, -1)
    pc = vm_pc.reshape(1, -1)
    dur = (
        in_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
        + mi.reshape(-1, 1) / np.maximum(pc, 1e-12)
        + out_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
    )

    avg_dur = dur.mean(axis=1)

    rank_u = np.zeros((n,), dtype=np.float64)
    for u in reversed(topo):
        if len(succ[u]) == 0:
            rank_u[u] = avg_dur[u]
        else:
            rank_u[u] = avg_dur[u] + float(np.max(rank_u[succ[u]]))

    indeg = np.array([len(getattr(tasks[i], "parents", [])) for i in range(n)], dtype=np.int32)
    ready = []
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(ready, (-rank_u[i], i))

    Pn = vm_pc.size
    vm_ready = np.zeros((Pn,), dtype=np.float64)
    finish = np.zeros((n,), dtype=np.float64)
    start = np.zeros((n,), dtype=np.float64)

    parents = [list(getattr(tasks[i], "parents", [])) for i in range(n)]

    scheduled = 0
    while ready:
        _neg_r, u = heapq.heappop(ready)

        if len(parents[u]) == 0:
            ready_t = 0.0
        else:
            ready_t = float(np.max(finish[np.array(parents[u], dtype=np.int32)]))

        st = np.maximum(vm_ready, ready_t)
        ft = st + dur[u]
        p_star = int(np.argmin(ft))
        start[u] = float(st[p_star])
        finish[u] = float(ft[p_star])
        vm_ready[p_star] = finish[u]

        scheduled += 1
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, (-rank_u[v], v))

    if scheduled != n:
        ms = float(np.max(avg_dur)) if n > 0 else 0.0
        return ms, finish, avg_dur

    makespan = float(np.max(finish))
    return makespan, finish, avg_dur


class HrlHeftEnv(gym.Env):
    """基于 HEFT 期限估计的三层分阶段调度环境"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dax_paths,
        horizon=36000.0,
        arrival_lambda=0.1,
        random_seed=0,
        max_ready_tasks=32,
        normalize=True,
        num_cloud_hosts=2,
        num_edge_hosts=1,
        cloud_vms_per_host=(9, 8),
        edge_vms_per_host=(8,),
        cloud_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        edge_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        cloud_bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        edge_bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        dax_probs=None,
        state_include_vm=True,
        state_include_host=True,
        state_include_queue=True,
        combo_weights=None,
        workflows_per_episode=8,
        fuzzy_enabled=False,
        fuzzy_delta1=0.75,
        fuzzy_delta2=1.2,
        fuzzy_energy_uncertainty_weight=1.0,
        fuzzy_deadline_eta=0.95,
        fuzzy_resource_seed=None,
        fuzzy_use_deadline_constraint=True,
        safe_rl_enabled=False,
        safe_rl_process_risk_aggregation="mean",
        safe_rl_shield_enabled=False,
        safe_rl_fallback_controller="fixed_vm_rule",
        safe_rl_state_enabled=False,
        safe_rl_state_high_uncertainty_threshold=0.2,
        safe_rl_state_recent_record_window=100,
        manager_mode=LEGACY_RULE_WEIGHT_MODE,
        manager_heuristic_library_path=None,
        manager_heuristic_llm_only=False,
        experiment_protocol_identity=None,
        manager_heuristic_recent_window=20,
        scenario_code=None,
        task_code=None,
        resource_code=None,
        workflow_families=None,
    ):
        """初始化云边资源、动作空间、观测空间和运行期缓存"""
        super().__init__()

        # 基础随机状态与工作流到达参数
        self.rng = np.random.RandomState(random_seed)
        self.horizon = float(horizon)
        self.arrival_lambda = float(arrival_lambda)
        self.dax_paths = list(dax_paths)
        self.random_seed = int(random_seed)
        self.normalize = bool(normalize)
        # CEWS 模糊评价开关。默认关闭可保证所有旧 HRL/FCFS 调用仍使用原 modal
        # 时间、能耗和 VM 排序；即使资源保存为 TFN，旧运算也只读取 modal。
        self.fuzzy_enabled = bool(fuzzy_enabled)
        self.fuzzy_delta1 = float(fuzzy_delta1)
        self.fuzzy_delta2 = float(fuzzy_delta2)
        self.fuzzy_energy_uncertainty_weight = float(
            fuzzy_energy_uncertainty_weight
        )
        self.fuzzy_deadline_eta = float(fuzzy_deadline_eta)
        self.fuzzy_use_deadline_constraint = bool(
            fuzzy_use_deadline_constraint
        )
        # 阶段 1 仅启用 CMDP 所需的 reward-cost 输出与统计。默认关闭时，
        # vm_assign()/finish_phase_and_advance() 的旧 reward 和返回元组保持不变。
        self.safe_rl_enabled = bool(safe_rl_enabled)
        self.safe_rl_process_risk_aggregation = str(
            safe_rl_process_risk_aggregation
        ).strip().lower()
        self.safe_rl_shield_enabled = bool(
            safe_rl_shield_enabled
        )
        self.safe_rl_fallback_controller = str(
            safe_rl_fallback_controller
        ).strip().lower()
        self.safe_rl_state_enabled = bool(
            safe_rl_state_enabled
        )
        self.safe_rl_state_high_uncertainty_threshold = float(
            safe_rl_state_high_uncertainty_threshold
        )
        self.safe_rl_state_recent_record_window = int(
            safe_rl_state_recent_record_window
        )
        self.manager_mode = str(manager_mode).strip().lower()
        self.manager_heuristic_library_path = (
            None
            if manager_heuristic_library_path is None
            else str(manager_heuristic_library_path)
        )
        # 为真时 Manager 动作空间只含 manifest 中的 LLM 规则，
        # 五个传统规则连动作槽都不会构造。
        self.manager_heuristic_llm_only = bool(
            manager_heuristic_llm_only
        )
        self.experiment_protocol_identity = (
            None
            if experiment_protocol_identity is None
            else dict(experiment_protocol_identity)
        )
        self.manager_heuristic_recent_window = int(
            manager_heuristic_recent_window
        )
        self.scenario_code = (
            ""
            if scenario_code is None
            else str(scenario_code).strip().upper()
        )
        self.task_code = (
            ""
            if task_code is None
            else str(task_code).strip().upper()
        )
        self.resource_code = (
            ""
            if resource_code is None
            else str(resource_code).strip().upper()
        )
        if self.scenario_code:
            if not self.task_code and len(self.scenario_code) == 2:
                self.task_code = self.scenario_code[0]
            if (
                not self.resource_code
                and len(self.scenario_code) == 2
            ):
                self.resource_code = self.scenario_code[1]
        elif self.task_code and self.resource_code:
            self.scenario_code = (
                self.task_code + self.resource_code
            )
        if (
            workflow_families is None
            and self.manager_mode == HEURISTIC_SELECTION_MODE
        ):
            self.workflow_families = tuple(
                workflow_families_from_dax_files(
                    self.dax_paths
                )
            )
        elif workflow_families is None:
            self.workflow_families = tuple()
        else:
            self.workflow_families = tuple(
                str(value).strip()
                for value in workflow_families
            )
        if self.manager_mode not in MANAGER_HEURISTIC_MODES:
            raise ValueError(
                "manager_mode must be legacy_rule_weight_mode or "
                "heuristic_selection_mode"
            )
        if (
            self.safe_rl_enabled
            and self.safe_rl_process_risk_aggregation != "mean"
        ):
            raise ValueError(
                "safe_rl_process_risk_aggregation currently supports "
                "only 'mean'"
            )
        if (
            not np.isfinite(self.fuzzy_energy_uncertainty_weight)
            or self.fuzzy_energy_uncertainty_weight < 0.0
        ):
            raise ValueError(
                "fuzzy_energy_uncertainty_weight 必须是有限非负数"
            )
        if (
            not np.isfinite(self.fuzzy_deadline_eta)
            or not 0.0 <= self.fuzzy_deadline_eta <= 1.0
        ):
            raise ValueError("fuzzy_deadline_eta 必须位于 [0, 1]")
        if self.safe_rl_enabled and not self.fuzzy_enabled:
            raise ValueError(
                "safe_rl_enabled=True requires fuzzy_enabled=True so that "
                "the fuzzy energy objective and fuzzy deadline cost use the "
                "same optimistic/modal/pessimistic schedule."
            )
        if self.safe_rl_shield_enabled and not self.safe_rl_enabled:
            raise ValueError(
                "safe_rl_shield_enabled=True requires "
                "safe_rl_enabled=True"
            )
        if self.safe_rl_state_enabled and not self.safe_rl_enabled:
            raise ValueError(
                "safe_rl_state_enabled=True requires "
                "safe_rl_enabled=True"
            )
        if (
            not np.isfinite(
                self.safe_rl_state_high_uncertainty_threshold
            )
            or self.safe_rl_state_high_uncertainty_threshold < 0.0
        ):
            raise ValueError(
                "safe_rl_state_high_uncertainty_threshold must be "
                "finite and non-negative"
            )
        if self.safe_rl_state_recent_record_window <= 0:
            raise ValueError(
                "safe_rl_state_recent_record_window must be positive"
            )
        if self.manager_heuristic_recent_window <= 0:
            raise ValueError(
                "manager_heuristic_recent_window must be positive"
            )
        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            if not self.safe_rl_enabled:
                raise ValueError(
                    "heuristic_selection_mode requires "
                    "safe_rl_enabled=True"
                )
            if not self.safe_rl_shield_enabled:
                raise ValueError(
                    "heuristic_selection_mode requires "
                    "safe_rl_shield_enabled=True"
                )
            if not self.safe_rl_state_enabled:
                raise ValueError(
                    "heuristic_selection_mode requires "
                    "safe_rl_state_enabled=True"
                )
            if not self.manager_heuristic_library_path:
                raise ValueError(
                    "heuristic_selection_mode requires a safe "
                    "heuristic library manifest"
                )
        if self.safe_rl_fallback_controller != "fixed_vm_rule":
            raise ValueError(
                "safe_rl_fallback_controller currently supports only "
                "'fixed_vm_rule'"
            )
        self.safety_shield = FuzzyDDLSafetyShield(
            enabled=self.safe_rl_shield_enabled
        )
        self.safety_fallback_controller = (
            DeterministicFuzzyDDLFallbackController(
                enabled=self.safe_rl_shield_enabled
            )
        )
        self.fuzzy_resource_seed = int(
            self.random_seed
            if fuzzy_resource_seed is None
            else fuzzy_resource_seed
        )

        # 下层智能体的能耗与延迟奖励参数
        self.energy_reward_scale = 1e-3
        self.task_baseline_norm = 300.0
        self.energy_norm_per_mi_ref = 20.0  # 保留但本版 lower reward 不再使用
        self.alpha_delay_host = 0.35
        self.alpha_delay_vm = 0.60

        self.workflows_per_episode = int(workflows_per_episode) if workflows_per_episode is not None else None

        self.dax_probs = None
        if dax_probs is not None:
            probs = np.asarray(dax_probs, dtype=float).copy()
            assert len(probs) == len(self.dax_paths), "dax_probs 长度需与 dax_paths 一致"
            probs = np.clip(probs, 0.0, None)
            if probs.sum() <= 0:
                probs[:] = 1.0
            self.dax_probs = probs / probs.sum()
        self._episode_dax_paths = self._build_episode_dax_paths()

        # 分别使用云端和边缘端配置创建主机及其 VM
        self.hosts, self.vms = create_cluster(
            num_cloud_hosts=num_cloud_hosts,
            num_edge_hosts=num_edge_hosts,
            cloud_vms_per_host=cloud_vms_per_host,
            edge_vms_per_host=edge_vms_per_host,
            cloud_pc_tiers=cloud_pc_tiers,
            edge_pc_tiers=edge_pc_tiers,
            cloud_bw_tiers=cloud_bw_tiers,
            edge_bw_tiers=edge_bw_tiers,
            fuzzy_delta1=self.fuzzy_delta1,
            fuzzy_delta2=self.fuzzy_delta2,
            fuzzy_seed=self.fuzzy_resource_seed,
        )
        self.host_ids = sorted(self.hosts.keys())
        self.vm_ids = sorted(self.vms.keys())
        # 三场景时长只依赖 (task_id, vm_id, scenario)：vm.pc/vm.bw 在 create_cluster
        # 之后不再改写，task_mi/in_bits/out_bits 只在 reset 时重建并在工作流到达时
        # 追加（既有条目从不被修改）。因此缓存在 episode 内恒定，reset 时清空即可。
        self._scenario_duration_cache = {}
        self._scenario_duration_cache_audit = _simulator_cache_audit_enabled()
        self.num_hosts = len(self.hosts)
        self.num_vms = len(self.vm_ids)
        self.vm_host = np.array([self.vms[vid].host_id for vid in self.vm_ids], dtype=np.int32)

        self._heft_vm_pc = np.array([float(self.vms[vid].pc) for vid in self.vm_ids], dtype=np.float64)
        self._heft_vm_bw_bps = np.array([float(self.vms[vid].bw) * 1e6 for vid in self.vm_ids], dtype=np.float64)

        # 建立主机到全局 VM 索引的映射，供分层动作空间使用
        self.host_to_vm_indices = {h: [] for h in self.host_ids}
        for j, vid in enumerate(self.vm_ids):
            h = self.vms[vid].host_id
            self.host_to_vm_indices[h].append(j)
        self.max_vms_per_host = max(len(vs) for vs in self.host_to_vm_indices.values()) if self.num_hosts > 0 else 0

        if (
            (max_ready_tasks is None)
            or (isinstance(max_ready_tasks, (int, float)) and max_ready_tasks <= 0)
            or (isinstance(max_ready_tasks, str) and max_ready_tasks.lower() == "auto")
        ):
            self.max_ready = _auto_max_ready_tasks(self.dax_paths, self.workflows_per_episode)
        else:
            self.max_ready = int(max_ready_tasks)

        self.max_ready = max(1, int(self.max_ready))
        self.task_heur_dim = 5

        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            # Resource-domain scope replaces exact DAX/seed binding. Workload,
            # topology, DDL, fuzzy settings and artifact hashes remain exact.
            self.manager_heuristic_runtime_context = {
                "scenario_code": self.scenario_code,
                "task_code": self.task_code,
                "resource_code": self.resource_code,
                "workflow_families": list(
                    self.workflow_families
                ),
                "workflows_per_instance": (
                    None
                    if self.workflows_per_episode is None
                    else int(self.workflows_per_episode)
                ),
                "arrival_lambda": float(self.arrival_lambda),
                "horizon": float(self.horizon),
                "deadline_mode": str(
                    getattr(self, "deadline_mode", "heft")
                ),
                "deadline_alpha_small": float(
                    getattr(
                        self,
                        "deadline_alpha_small",
                        np.nan,
                    )
                ),
                "deadline_alpha_large": float(
                    getattr(
                        self,
                        "deadline_alpha_large",
                        np.nan,
                    )
                ),
                "deadline_alpha_small_prob": float(
                    getattr(
                        self,
                        "deadline_alpha_small_prob",
                        np.nan,
                    )
                ),
                "num_cloud_hosts": int(num_cloud_hosts),
                "num_edge_hosts": int(num_edge_hosts),
                "cloud_vms_per_host": [
                    int(value)
                    for value in cloud_vms_per_host
                ],
                "edge_vms_per_host": [
                    int(value)
                    for value in edge_vms_per_host
                ],
                "cloud_pc_tiers": [
                    float(value) for value in cloud_pc_tiers
                ],
                "edge_pc_tiers": [
                    float(value) for value in edge_pc_tiers
                ],
                "cloud_bw_tiers": [
                    float(value) for value in cloud_bw_tiers
                ],
                "edge_bw_tiers": [
                    float(value) for value in edge_bw_tiers
                ],
                "fuzzy_delta1": float(self.fuzzy_delta1),
                "fuzzy_delta2": float(self.fuzzy_delta2),
                "fuzzy_deadline_eta": float(
                    self.fuzzy_deadline_eta
                ),
                "fuzzy_energy_lambda": float(
                    self.fuzzy_energy_uncertainty_weight
                ),
                "fuzzy_resource_seed_mode": "episode_seed",
                "fuzzy_resource_seed_offset": int(
                    self.fuzzy_resource_seed
                    - self.random_seed
                ),
            }
            self.manager_heuristics = (
                load_manager_heuristic_library(
                    self.manager_heuristic_library_path,
                    runtime_context=(
                        self.manager_heuristic_runtime_context
                    ),
                    expected_protocol_identity=(
                        self.experiment_protocol_identity
                    ),
                    include_traditional=(
                        not self.manager_heuristic_llm_only
                    ),
                )
            )
            availability = heuristic_availability_mask(
                self.manager_heuristics
            )
            if not np.any(availability > 0.5):
                # LLM-only 模式下传统规则不再兜底；安全准入和 Top-K
                # 技术验证均可能令某个 LLM 动作不可用。
                raise ValueError(
                    "heuristic selection has no available Manager action"
                    + (
                        " (llm_only mode: every LLM rule failed "
                        "admission or Top-K artifact validation)"
                        if self.manager_heuristic_llm_only
                        else ""
                    )
                )
            self.selected_heuristic_index = int(
                np.flatnonzero(availability > 0.5)[0]
            )
            self.manager_heuristic_schema_version = (
                heuristic_action_schema_version(
                    self.manager_heuristics
                )
            )
        else:
            self.manager_heuristic_runtime_context = None
            self.manager_heuristics = tuple()
            self.selected_heuristic_index = None
            self.manager_heuristic_schema_version = None
        self.manager_action_dim = (
            len(self.manager_heuristics)
            if self.manager_mode == HEURISTIC_SELECTION_MODE
            else int(MANAGER_ACTION_TABLE.shape[0])
        )
        self._heuristic_recent_metrics = {
            heuristic.heuristic_id: {
                "performance_reward": deque(
                    maxlen=self.manager_heuristic_recent_window
                ),
                "safety_cost": deque(
                    maxlen=self.manager_heuristic_recent_window
                ),
                "shield_intervention_rate": deque(
                    maxlen=self.manager_heuristic_recent_window
                ),
            }
            for heuristic in self.manager_heuristics
        }
        self._phase_ready_task_ordering = []

        self.state_include_vm = bool(state_include_vm)
        self.state_include_host = bool(state_include_host)
        self.state_include_queue = bool(state_include_queue)

        # Manager 原始权重对应 FCFS、SJF、MCF、HUR、EDF 五种启发式
        self.manager_raw_w = np.ones(self.task_heur_dim, dtype=np.float32)
        if combo_weights is not None:
            w = np.asarray(combo_weights, dtype=float).reshape(-1)
            assert w.size == self.task_heur_dim, "combo_weights 维度需等于 5（FCFS/SJF/MCF/HUR/EDF）"
            w = np.clip(w, 0.0, None)
            if np.allclose(w.sum(), 0.0):
                w = np.ones_like(w)
            self.manager_raw_w = w.astype(np.float32)

        self.combo_weights = (self.manager_raw_w / max(float(self.manager_raw_w.sum()), 1e-9)).astype(np.float32)

        # 观测由任务特征、全局队列特征、主机特征和 VM 特征组成
        self.block1_dim = 12
        self.block2_dim = 10
        self.vm_feat_dim = 5
        self.host_feat_dim = 5

        self.manager_system_obs_dim = 10
        self.manager_legacy_obs_dim = (
            self.manager_system_obs_dim + self.task_heur_dim
        )
        self.manager_safety_feature_dim = len(
            MANAGER_SAFETY_FEATURE_SCHEMA
        )
        self.manager_heuristic_feature_dim = len(
            HEURISTIC_RECENT_FEATURE_SCHEMA
        )
        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            self.manager_obs_dim = (
                self.manager_system_obs_dim
                + self.manager_safety_feature_dim
                + len(self.manager_heuristics)
                * self.manager_heuristic_feature_dim
            )
        else:
            self.manager_obs_dim = self.manager_legacy_obs_dim + (
                self.manager_safety_feature_dim
                if self.safe_rl_state_enabled
                else 0
            )

        self.host_legacy_obs_dim = (
            self.block1_dim
            + self.block2_dim
            + self.host_feat_dim * self.num_hosts
        )
        self.host_safety_feature_dim = len(
            HOST_SAFETY_FEATURE_SCHEMA
        )
        self.host_obs_dim = self.host_legacy_obs_dim + (
            self.host_safety_feature_dim * self.num_hosts
            if self.safe_rl_state_enabled
            else 0
        )
        self.host_act_dim = self.num_hosts

        self.vm_legacy_obs_dim = (
            self.block1_dim
            + self.block2_dim
            + self.host_feat_dim
            + self.vm_feat_dim * self.max_vms_per_host
        )
        self.vm_safety_feature_dim = len(
            VM_SAFETY_FEATURE_SCHEMA
        )
        self.vm_obs_dim = self.vm_legacy_obs_dim + (
            self.vm_safety_feature_dim * self.max_vms_per_host
            if self.safe_rl_state_enabled
            else 0
        )
        self.vm_act_dim = self.max_vms_per_host

        self.observation_space = spaces.Dict({
            "host_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.host_obs_dim,), dtype=np.float32),
            "host_mask": spaces.Box(low=0.0, high=1.0, shape=(self.host_act_dim,), dtype=np.float32),
            "vm_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.vm_obs_dim,), dtype=np.float32),
            "vm_mask": spaces.Box(low=0.0, high=1.0, shape=(self.vm_act_dim,), dtype=np.float32),
        })

        # 能耗记录和分主机增量能耗缓存
        self.total_energy = 0.0
        self.lifetime_energy = 0.0
        self._records = []
        self._energy_cache = 0.0
        self._energy_cache_by_host = {int(h): 0.0 for h in self.host_ids}

        # 防止连续空分配或无事件推进导致环境停滞
        self._fuse_zero_assign_limit = 50
        self._fuse_no_event_limit = 10

        self._reset_internal_buffers()

    def _normalize_manager_raw(self):
        """将 Manager 原始权重归一化并处理全零退化情况"""
        s = float(np.sum(self.manager_raw_w))
        if s <= 1e-12:
            self.manager_raw_w[:] = 1.0
            s = float(np.sum(self.manager_raw_w))
        self.combo_weights = (self.manager_raw_w / max(s, 1e-9)).astype(np.float32)

    def set_manager_combo(self, weights):
        """直接设置 Manager 启发式组合权重并重启当前阶段"""
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        assert w.size == self.task_heur_dim, "combo 权重长度应为 5"
        w = np.clip(w, 0.0, None)
        if np.allclose(w.sum(), 0.0):
            w = np.ones_like(w)
        self.manager_raw_w = w.astype(np.float32)
        self._normalize_manager_raw()

        self._phase_started = False
        self._phase_tasks = []

    def apply_manager_delta(self, delta_vec):
        """应用 Manager 权重增量并重启当前阶段"""
        if self.manager_mode != LEGACY_RULE_WEIGHT_MODE:
            raise RuntimeError(
                "Manager weight deltas are unavailable in "
                "heuristic_selection_mode"
            )
        d = np.asarray(delta_vec, dtype=np.float32).reshape(-1)
        assert d.size == self.task_heur_dim, "delta_vec 长度必须为 5"
        self.manager_raw_w = np.maximum(0.0, self.manager_raw_w + d).astype(np.float32)
        self._normalize_manager_raw()

        self._phase_started = False
        self._phase_tasks = []

    def apply_manager_heuristic(self, heuristic_index: int):
        """把 Manager 动作解释为候选启发式的稳定索引。"""
        if self.manager_mode != HEURISTIC_SELECTION_MODE:
            raise RuntimeError(
                "Manager heuristic indices are available only in "
                "heuristic_selection_mode"
            )
        index = int(heuristic_index)
        mask = self.get_manager_action_mask()
        if index < 0 or index >= mask.size:
            raise ValueError("Manager heuristic action is out of range")
        if mask[index] <= 0.5:
            heuristic = self.manager_heuristics[index]
            raise ValueError(
                "Manager selected a non-admitted heuristic: "
                f"{heuristic.heuristic_id} "
                f"({heuristic.availability_reason})"
            )
        self.selected_heuristic_index = index
        self._phase_started = False
        self._phase_tasks = []
        self._phase_ready_task_ordering = []

    def apply_manager_action(self, action_index: int):
        """按显式 Manager 模式解释动作，禁止权重/规则索引混用。"""
        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            self.apply_manager_heuristic(action_index)
            return
        index = int(action_index)
        if index < 0 or index >= MANAGER_ACTION_TABLE.shape[0]:
            raise ValueError("Manager legacy action is out of range")
        self.apply_manager_delta(MANAGER_ACTION_TABLE[index])

    def get_manager_action_mask(self) -> np.ndarray:
        """返回当前模式的 Manager 可用动作 mask。"""
        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            return heuristic_availability_mask(
                self.manager_heuristics
            )
        w = self.manager_raw_w.reshape(1, -1)
        delta = MANAGER_ACTION_TABLE
        invalid = (w <= 1e-12) & (delta < 0.0)
        mask = (~invalid.any(axis=1)).astype(np.float32)
        return mask

    def get_manager_action_semantics(self) -> dict:
        """返回动作含义，供训练日志、测试和 checkpoint 审计。"""
        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            return {
                "manager_mode": self.manager_mode,
                "action_dim": int(self.manager_action_dim),
                "action_type": "heuristic_index",
                "heuristics": [
                    heuristic.public_metadata()
                    for heuristic in self.manager_heuristics
                ],
                "available_heuristic_mask": (
                    self.get_manager_action_mask().tolist()
                ),
            }
        return {
            "manager_mode": self.manager_mode,
            "action_dim": int(self.manager_action_dim),
            "action_type": "five_rule_weight_delta",
            "traditional_rules": [
                "FCFS",
                "SJF",
                "MCF",
                "HUR",
                "EDF",
            ],
        }

    def _selected_manager_heuristic(self):
        if self.manager_mode != HEURISTIC_SELECTION_MODE:
            return None
        if self.selected_heuristic_index is None:
            raise RuntimeError("Manager heuristic has not been selected")
        return self.manager_heuristics[
            int(self.selected_heuristic_index)
        ]

    def get_observation_schema(self, layer=None):
        """返回三层 observation 的维度、扩展顺序和归一化范围。"""
        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            manager_meta = {
                "legacy_dim": int(self.manager_system_obs_dim),
                "repeat_count": 1,
                "safe_feature_dim_per_item": int(
                    self.manager_safety_feature_dim
                ),
                "total_dim": int(self.manager_obs_dim),
                "safe_features": get_safety_feature_schema(
                    "manager"
                ),
                "heuristic_features": [
                    dict(feature)
                    for feature in HEURISTIC_RECENT_FEATURE_SCHEMA
                ],
                "heuristic_count": len(
                    self.manager_heuristics
                ),
                "heuristic_ids": [
                    heuristic.heuristic_id
                    for heuristic in self.manager_heuristics
                ],
                "heuristic_feature_dim_per_item": int(
                    self.manager_heuristic_feature_dim
                ),
                "action_dim": int(self.manager_action_dim),
                "action_type": "heuristic_index",
                "layout": (
                    "10 global system features + 11 global safety "
                    "features + one 4-feature recent-performance block "
                    "per heuristic in Manager action order"
                ),
            }
        else:
            manager_meta = {
                "legacy_dim": int(self.manager_legacy_obs_dim),
                "repeat_count": 1,
                "safe_feature_dim_per_item": int(
                    self.manager_safety_feature_dim
                ),
                "total_dim": int(self.manager_obs_dim),
                "safe_features": get_safety_feature_schema(
                    "manager"
                ),
                "heuristic_features": [],
                "heuristic_count": 0,
                "heuristic_ids": [],
                "heuristic_feature_dim_per_item": 0,
                "action_dim": int(self.manager_action_dim),
                "action_type": "five_rule_weight_delta",
                "layout": (
                    "legacy block + one global safety block"
                ),
            }
        layer_meta = {
            "manager": manager_meta,
            "host": {
                "legacy_dim": int(self.host_legacy_obs_dim),
                "repeat_count": int(self.num_hosts),
                "safe_feature_dim_per_item": int(
                    self.host_safety_feature_dim
                ),
                "total_dim": int(self.host_obs_dim),
                "safe_features": get_safety_feature_schema("host"),
                "layout": (
                    "legacy block + Host safety blocks in action-slot "
                    "order; no Host ID feature"
                ),
            },
            "vm": {
                "legacy_dim": int(self.vm_legacy_obs_dim),
                "repeat_count": int(self.max_vms_per_host),
                "safe_feature_dim_per_item": int(
                    self.vm_safety_feature_dim
                ),
                "total_dim": int(self.vm_obs_dim),
                "safe_features": get_safety_feature_schema("vm"),
                "layout": (
                    "legacy block + VM safety blocks in action-slot "
                    "order; padding slots are zero and no VM ID feature"
                ),
            },
        }
        for layer_name, value in layer_meta.items():
            if (
                layer_name == "manager"
                and self.manager_mode == HEURISTIC_SELECTION_MODE
            ):
                safe_extension_dim = (
                    self.manager_safety_feature_dim
                    + len(self.manager_heuristics)
                    * self.manager_heuristic_feature_dim
                )
                schema_version = (
                    self.manager_heuristic_schema_version
                )
            else:
                safe_extension_dim = int(
                    value["repeat_count"]
                    * value["safe_feature_dim_per_item"]
                    if self.safe_rl_state_enabled
                    else 0
                )
                schema_version = SAFE_OBSERVATION_SCHEMA_VERSION
            value.update(
                {
                    "schema_version": schema_version,
                    "safe_state_enabled": bool(
                        self.safe_rl_state_enabled
                    ),
                    "safe_extension_dim": int(
                        safe_extension_dim
                    ),
                }
            )
        if layer is None:
            return layer_meta
        key = str(layer).strip().lower()
        if key not in layer_meta:
            raise ValueError(
                "layer must be 'manager', 'host', or 'vm'"
            )
        return layer_meta[key]

    @staticmethod
    def _validated_safety_feature_vector(
        values,
        schema,
        *,
        context: str,
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.size != len(schema):
            raise RuntimeError(
                f"{context} safety feature dimension mismatch: "
                f"{vector.size} != {len(schema)}"
            )
        if not np.all(np.isfinite(vector)):
            raise RuntimeError(
                f"{context} safety features contain NaN or infinity"
            )
        for index, feature in enumerate(schema):
            low = float(feature["low"])
            high = float(feature["high"])
            if (
                float(vector[index]) < low - 1e-6
                or float(vector[index]) > high + 1e-6
            ):
                raise RuntimeError(
                    f"{context} safety feature "
                    f"{feature['name']}={float(vector[index])} "
                    f"is outside [{low}, {high}]"
                )
        return vector

    def _workflow_budget_for_task(self, task_id: int) -> float:
        workflow_id, _ = self.task_meta[int(task_id)]
        workflow = self.workflows[workflow_id]
        return max(
            float(workflow.deadline)
            - float(workflow.arrival_time),
            1e-9,
        )

    def _platform_congestion(self, server_type: str) -> float:
        now = float(self.current_time)
        indices = [
            index
            for index, vm_id in enumerate(self.vm_ids)
            if str(
                self.hosts[self.vms[vm_id].host_id].server_type
            ).lower()
            == str(server_type).lower()
        ]
        if not indices:
            return 0.0
        busy = sum(
            self.vm_available_at[index] > now + 1e-9
            for index in indices
        )
        return _clip01(float(busy) / float(len(indices)))

    @staticmethod
    def _prediction_relative_duration_uncertainty(row) -> float:
        optimistic = float(
            row["current_task_execution_time_optimistic"]
            + row["current_task_communication_time_optimistic"]
        )
        modal = float(
            row["current_task_execution_time_modal"]
            + row["current_task_communication_time_modal"]
        )
        pessimistic = float(
            row["current_task_execution_time_pessimistic"]
            + row["current_task_communication_time_pessimistic"]
        )
        return max(
            0.0,
            (pessimistic - optimistic) / max(modal, 1e-9),
        )

    def _build_manager_safety_observation(self) -> np.ndarray:
        margin_info = self.get_dynamic_fuzzy_safety_margins()
        margin_rows = margin_info["workflow_safety_margins"]
        normalized_margins = [
            float(row["normalized_fuzzy_safety_margin"])
            for row in margin_rows
        ]
        minimum_margin = (
            float(np.min(normalized_margins))
            if normalized_margins
            else 0.0
        )
        mean_margin = (
            float(np.mean(normalized_margins))
            if normalized_margins
            else 0.0
        )

        high_uncertainty_flags = []
        task_safe_host_ratios = []
        legal_host_safe_vm_ratios = []
        for task_id in list(self.ready_task_ids):
            prediction = self.get_task_action_risk_predictions(
                task_id
            )
            vm_rows = list(prediction["vm_predictions"])
            selectable_rows = [
                row
                for row in vm_rows
                if bool(row["is_currently_selectable"])
            ]
            uncertainty_rows = selectable_rows or vm_rows
            if uncertainty_rows:
                best_uncertainty = min(
                    self._prediction_relative_duration_uncertainty(
                        row
                    )
                    for row in uncertainty_rows
                )
                high_uncertainty_flags.append(
                    best_uncertainty
                    >= self.safe_rl_state_high_uncertainty_threshold
                )

            legal_hosts = [
                row
                for row in prediction["host_predictions"]
                if bool(row["is_hard_action_legal"])
            ]
            if legal_hosts:
                task_safe_host_ratios.append(
                    sum(
                        bool(row["is_predicted_safe"])
                        for row in legal_hosts
                    )
                    / len(legal_hosts)
                )
                legal_host_safe_vm_ratios.extend(
                    float(row["safe_vm_ratio"])
                    for row in legal_hosts
                )

        recent_records = self._safety_shield_records[
            -self.safe_rl_state_recent_record_window :
        ]
        if recent_records:
            recent_intervention_rate = float(
                np.mean(
                    [
                        bool(record.get("shield_intervened", False))
                        for record in recent_records
                    ]
                )
            )
            recent_fallback_rate = float(
                np.mean(
                    [
                        bool(record.get("fallback_triggered", False))
                        for record in recent_records
                    ]
                )
            )
        else:
            recent_intervention_rate = 0.0
            recent_fallback_rate = 0.0

        values = [
            float(np.clip(minimum_margin, -1.0, 1.0)),
            float(np.clip(mean_margin, -1.0, 1.0)),
            _clip01(margin_info["risk_workflow_ratio"]),
            _clip01(margin_info["predicted_violation_rate"]),
            (
                _clip01(float(np.mean(high_uncertainty_flags)))
                if high_uncertainty_flags
                else 0.0
            ),
            self._platform_congestion("cloud"),
            self._platform_congestion("edge"),
            (
                _clip01(float(np.mean(task_safe_host_ratios)))
                if task_safe_host_ratios
                else 0.0
            ),
            (
                _clip01(
                    float(np.mean(legal_host_safe_vm_ratios))
                )
                if legal_host_safe_vm_ratios
                else 0.0
            ),
            _clip01(recent_intervention_rate),
            _clip01(recent_fallback_rate),
        ]
        return self._validated_safety_feature_vector(
            values,
            MANAGER_SAFETY_FEATURE_SCHEMA,
            context="Manager",
        )

    def _build_manager_heuristic_observation(self) -> np.ndarray:
        """按 Manager 动作顺序输出规则近期表现和当前准入 mask。"""
        if self.manager_mode != HEURISTIC_SELECTION_MODE:
            return np.zeros(0, dtype=np.float32)
        availability = self.get_manager_action_mask()
        blocks = []
        for index, heuristic in enumerate(
            self.manager_heuristics
        ):
            history = self._heuristic_recent_metrics[
                heuristic.heuristic_id
            ]
            reward_values = list(
                history["performance_reward"]
            )
            cost_values = list(history["safety_cost"])
            shield_values = list(
                history["shield_intervention_rate"]
            )
            if reward_values:
                mean_reward = float(np.mean(reward_values))
                energy_performance = float(
                    mean_reward / (1.0 + abs(mean_reward))
                )
                safety_performance = float(
                    1.0
                    / (
                        1.0
                        + max(0.0, float(np.mean(cost_values)))
                    )
                )
                shield_rate = _clip01(
                    float(np.mean(shield_values))
                )
            else:
                # 0 表示尚无该规则的在线表现证据，不伪装成安全满分。
                energy_performance = 0.0
                safety_performance = 0.0
                shield_rate = 0.0
            blocks.extend(
                [
                    float(
                        np.clip(
                            energy_performance,
                            -1.0,
                            1.0,
                        )
                    ),
                    _clip01(safety_performance),
                    _clip01(shield_rate),
                    float(availability[index] > 0.5),
                ]
            )
        vector = np.asarray(blocks, dtype=np.float32)
        expected = (
            len(self.manager_heuristics)
            * self.manager_heuristic_feature_dim
        )
        if vector.size != expected:
            raise RuntimeError(
                "Manager heuristic observation dimension mismatch"
            )
        if not np.all(np.isfinite(vector)):
            raise RuntimeError(
                "Manager heuristic observation contains NaN or "
                "infinity"
            )
        reshaped = vector.reshape(
            len(self.manager_heuristics),
            self.manager_heuristic_feature_dim,
        )
        for feature_index, feature in enumerate(
            HEURISTIC_RECENT_FEATURE_SCHEMA
        ):
            values = reshaped[:, feature_index]
            if (
                np.any(values < float(feature["low"]) - 1e-6)
                or np.any(
                    values > float(feature["high"]) + 1e-6
                )
            ):
                raise RuntimeError(
                    "Manager heuristic feature outside declared "
                    f"range: {feature['name']}"
                )
        return vector

    def get_manager_state(self):
        """构造当前 Manager 模式所需的全局与规则状态。"""
        now = getattr(self, "current_time", 0.0)

        ready_cnt = float(len(self.ready_task_ids))
        waiting_cnt = float(sum(1 for s in self.task_state if s == "unReady"))
        running_cnt = float(sum(1 for s in self.task_state if s == "Running"))

        waits = [max(0.0, now - self.task_ready_time[tid]) for tid in self.ready_task_ids]
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        max_wait = float(np.max(waits)) if len(waits) > 0 else 0.0

        ready_ratio = ready_cnt / max(1.0, self.max_ready)
        waiting_ratio = waiting_cnt / max(1.0, len(self.task_state) if len(self.task_state) > 0 else 1.0)
        running_ratio = running_cnt / max(1.0, self.num_vms)

        idle_vms = float(np.sum(self._vm_idle_mask()))
        idle_vm_ratio = idle_vms / max(1.0, self.num_vms)

        host_load_list, host_running_ratio_list = [], []
        for h in self.host_ids:
            active_pc = 0.0
            running = 0
            cnt = 0
            soonest = None
            for j, vid in enumerate(self.vm_ids):
                if self.vms[vid].host_id == h:
                    cnt += 1
                    if self.vm_available_at[j] > now + 1e-12:
                        active_pc += self.vms[vid].pc
                        running += 1
                    if soonest is None or self.vm_available_at[j] < soonest:
                        soonest = self.vm_available_at[j]
            host_load_list.append(_safe_div(active_pc, max(self.hosts[h].total_pc, 1e-9)))
            host_running_ratio_list.append(_safe_div(running, max(cnt, 1)))

        host_load_mean = float(np.mean(host_load_list)) if host_load_list else 0.0
        host_running_ratio_mean = float(np.mean(host_running_ratio_list)) if host_running_ratio_list else 0.0

        pc_mean = float(np.mean([self.vms[v].pc for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_mean = float(np.mean([self.vms[v].bw for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_bps = bw_mean * 1e6

        slacks = []
        for tid in self.ready_task_ids:
            exp_t = (
                self.task_in_bits[tid] / max(bw_bps, 1e-9)
                + self.task_mi[tid] / max(pc_mean, 1.0)
                + self.task_out_bits[tid] / max(bw_bps, 1e-9)
            )
            wf_idx, _ = self.task_meta[tid]
            wf = self.workflows[wf_idx]
            dl = getattr(wf, "deadline", None)
            if dl is None:
                continue
            slacks.append(dl - (now + exp_t))

        if len(slacks) > 0:
            min_slack = float(np.min(slacks))
            avg_slack = float(np.mean(slacks))
            min_slack_norm = min_slack / max(self.horizon, 1.0)
            avg_slack_norm = avg_slack / max(self.horizon, 1.0)
        else:
            min_slack_norm = 0.0
            avg_slack_norm = 0.0

        vec = np.array([
            ready_ratio, waiting_ratio, running_ratio,
            avg_wait / max(self.horizon, 1.0),
            max_wait / max(self.horizon, 1.0),
            idle_vm_ratio, host_load_mean, host_running_ratio_mean,
            min_slack_norm, avg_slack_norm
        ], dtype=np.float32)

        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            state = np.concatenate(
                [
                    vec,
                    self._build_manager_safety_observation(),
                    self._build_manager_heuristic_observation(),
                ],
                axis=0,
            ).astype(np.float32)
            if state.size != self.manager_obs_dim:
                raise RuntimeError(
                    "Heuristic Manager observation dimension does "
                    "not match schema"
                )
            if not np.all(np.isfinite(state)):
                raise RuntimeError(
                    "Heuristic Manager observation contains NaN or "
                    "infinity"
                )
            return state

        w_norm = self.combo_weights.astype(np.float32)
        legacy_state = np.concatenate([vec, w_norm], axis=0).astype(
            np.float32
        )
        if not self.safe_rl_state_enabled:
            return legacy_state
        safety_state = self._build_manager_safety_observation()
        state = np.concatenate(
            [legacy_state, safety_state],
            axis=0,
        ).astype(np.float32)
        if state.size != self.manager_obs_dim:
            raise RuntimeError(
                "Manager observation dimension does not match schema"
            )
        return state

    # ------------------------------------------------------------------
    # Constructive cloud-edge scheduling interface used by SeEvo.
    # Tasks are represented by the environment's global integer task IDs and
    # VMs by their stable VM IDs. The existing HRL interfaces remain unchanged.
    # ------------------------------------------------------------------
    def _task_id(self, task) -> int:
        if isinstance(task, (int, np.integer)):
            task_id = int(task)
        else:
            task_id = None
            for gid, (wf_id, local_id) in enumerate(self.task_meta):
                candidate = self.workflows[wf_id].tasks[local_id]
                if candidate is task:
                    task_id = gid
                    break
            if task_id is None:
                raise ValueError("Task is not owned by this environment.")
        if task_id < 0 or task_id >= len(self.task_state):
            raise ValueError(f"Task ID out of range: {task_id}")
        return task_id

    def _vm_id_and_index(self, vm):
        vm_id = int(vm.vm_id) if hasattr(vm, "vm_id") else int(vm)
        if vm_id not in self.vms:
            raise ValueError(f"Unknown VM ID: {vm_id}")
        try:
            vm_index = self.vm_ids.index(vm_id)
        except ValueError as exc:
            raise ValueError(f"VM ID {vm_id} is not active in this environment.") from exc
        return vm_id, int(vm_index)

    def _task_deadline(self, task) -> float:
        """返回任务绝对子截止期；缺失时回退到所属工作流的绝对截止期。"""
        task_id = self._task_id(task)
        # task_baseline_finish 在环境初始化时保存任务级 deadline/baseline 口径。
        if task_id < len(self.task_baseline_finish):
            deadline = float(self.task_baseline_finish[task_id])
            if np.isfinite(deadline):
                return deadline
        wf_id, _ = self.task_meta[task_id]
        # 某些旧数据没有任务子截止期，使用工作流截止期仍能保持规则可运行。
        workflow_deadline = getattr(self.workflows[wf_id], "deadline", None)
        if workflow_deadline is None or not np.isfinite(float(workflow_deadline)):
            return float("inf")
        return float(workflow_deadline)

    def get_ready_tasks(self):
        """只读返回当前全部 ready task 的全局整数 ID。

        ``ready_task_ids`` 是环境维护的集合/列表；再次检查 task_state 可以过滤
        刚完成状态迁移但尚未从容器清理的条目。该函数不改变 ready 集顺序或状态。
        """
        return [
            int(task_id) for task_id in self.ready_task_ids
            if self.task_state[int(task_id)] == "Ready"
        ]

    def get_feasible_vms(self, task):
        """返回处理能力和带宽均为有限正数的稳定 VM ID 列表。

        三角模糊处理能力/带宽的 lower、modal、upper 三点必须全部合法。本方法只
        回答“这台 VM 能不能跑这个任务”，不回答“现在能不能派上去”：正在忙的 VM
        仍然可行，其等待代价由确定性选择器的 queue_time 显式计入。

        是否允许排到 busy VM 后面由调用方决定，两种口径都存在：Safe-HRL 运行时
        接受排队；CEWS 评价器与反事实估计器不接受，它们用 ``idle_feasible_vm_ids``
        把结果收窄到当前空闲的子集，以与各比较算法的 ``global_vm_action_mask``
        保持同一口径。
        """
        # _task_id 同时验证整数 ID 或 Task 对象可映射到当前环境任务。
        self._task_id(task)
        feasible = []
        for vm_id in self.vm_ids:
            vm = self.vms[vm_id]
            pc_values = (float(vm.pc.lower), float(vm.pc.modal), float(vm.pc.upper))
            bw_values = (float(vm.bw.lower), float(vm.bw.modal), float(vm.bw.upper))
            # 任一点非有限或非正都可能使时间估计出现除零、负时长或无穷。
            if all(np.isfinite(value) and value > 0.0 for value in pc_values + bw_values):
                feasible.append(int(vm_id))
        return feasible

    def estimate_exec_time(self, task, vm) -> float:
        """用 VM 模态处理能力估计纯计算时长，单位秒，不修改环境状态。

        计算式为 task_mi / vm.pc.modal。``float(vm.pc)`` 沿用三角模糊数的模态值
        转换语义；1e-12 仅是防御性下界，正常 VM 已由 get_feasible_vms 验证。
        """
        task_id = self._task_id(task)
        vm_id, _ = self._vm_id_and_index(vm)
        value = float(self.task_mi[task_id]) / max(float(self.vms[vm_id].pc), 1e-12)
        return float(value)

    def estimate_comm_time(self, task, vm) -> float:
        """按现有模拟器模型估计数据通信时长，单位秒，不修改状态。

        现有模型把任务输入（其中已包含前驱数据）和输出数据都计入选定 VM 的
        带宽开销。这里复用完全相同的 ``(in_bits + out_bits) / bandwidth_bps``
        口径，使特征、预测完成时刻和真实分配时长保持一致。VM 带宽配置单位为
        Mbps，因此乘 1e6 转为 bit/s。
        """
        task_id = self._task_id(task)
        vm_id, _ = self._vm_id_and_index(vm)
        data_bits = float(self.task_in_bits[task_id]) + float(self.task_out_bits[task_id])
        bandwidth_bps = max(float(self.vms[vm_id].bw) * 1e6, 1e-12)
        return float(data_bits / bandwidth_bps)

    def _resource_component_for_scenario(self, fuzzy_number, scenario):
        """把三场景名称映射到资源 TFN 分量。

        optimistic 使用最大处理能力/带宽 ``upper``，modal 使用 ``modal``，
        pessimistic 使用最小能力/带宽 ``lower``。资源仅在集群创建时随机模糊化
        一次；本函数只是读取固定分量，不进行任何运行时采样。
        """
        if not isinstance(fuzzy_number, TriangularFuzzyNumber):
            raise TypeError(
                "scenario resource must be a TriangularFuzzyNumber"
            )
        component_by_scenario = {
            "optimistic": "upper",
            "modal": "modal",
            "pessimistic": "lower",
        }
        try:
            component_name = component_by_scenario[str(scenario)]
        except KeyError as exc:
            raise ValueError(
                "scenario must be 'optimistic', 'modal', or 'pessimistic'"
            ) from exc
        return fuzzy_number.component(component_name)

    def estimate_task_duration_components_scenario(
        self,
        task_id,
        vm_id,
        scenario,
    ) -> dict:
        """返回现有运行时口径下的场景执行/通信时长分量。

        当前 HRL 运行时无论父任务位于何处，都会通过目标 VM 带宽计入全部输入
        与输出。本接口严格复用该口径，不单独启用尚未接入运行时的 locality
        近似，避免安全预测比真实调度更乐观。

        结果按 ``(task_id, vm_id, scenario)`` 记忆化。返回值只由这三者决定，
        缓存命中返回的是同一函数在同一状态下算出的同一批浮点数，因此与不加
        缓存时逐位相同。``SIM_EXACTNESS_AUDIT=1`` 会在每次命中时重算并断言
        精确相等。每次返回副本，避免调用方就地修改污染缓存。
        """
        task_id = self._task_id(task_id)
        vm_id, _ = self._vm_id_and_index(vm_id)
        cache_key = (task_id, vm_id, str(scenario))
        # 缓存字典懒建：测试会用 object.__new__(HrlHeftEnv) 造只填了几个字段的
        # 替身，不走 __init__。这里绝不能用类属性做默认值，否则多个环境实例会
        # 共用同一份缓存，把别的集群的时长读回来。
        cache = getattr(self, "_scenario_duration_cache", None)
        if cache is None:
            cache = {}
            self._scenario_duration_cache = cache
        cached = cache.get(cache_key)
        if cached is None:
            cached = self._compute_task_duration_components_scenario(
                task_id,
                vm_id,
                scenario,
            )
            cache[cache_key] = cached
        elif getattr(self, "_scenario_duration_cache_audit", False):
            actual = self._compute_task_duration_components_scenario(
                task_id,
                vm_id,
                scenario,
            )
            if actual != cached:
                raise AssertionError(
                    "scenario duration cache changed for "
                    f"task_id={task_id}, vm_id={vm_id}, "
                    f"scenario={scenario}: {cached} -> {actual}"
                )
        return dict(cached)

    def _compute_task_duration_components_scenario(
        self,
        task_id,
        vm_id,
        scenario,
    ) -> dict:
        """在已归一化的 ``(task_id, vm_id)`` 上做实际计算，不查缓存。"""
        vm = self.vms[vm_id]
        pc = self._resource_component_for_scenario(vm.pc, scenario)
        bw = self._resource_component_for_scenario(vm.bw, scenario)
        input_bits = float(self.task_in_bits[task_id])
        output_bits = float(self.task_out_bits[task_id])
        workload_mi = float(self.task_mi[task_id])
        input_communication_time = input_bits / max(bw * 1e6, 1e-12)
        execution_time = workload_mi / max(pc, 1e-12)
        output_communication_time = output_bits / max(
            bw * 1e6,
            1e-12,
        )
        communication_time = (
            input_communication_time + output_communication_time
        )
        duration = execution_time + communication_time
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError(
                f"Invalid {scenario} duration for task_id={task_id}, "
                f"vm_id={vm_id}: {duration}"
            )
        return {
            "input_communication_time": float(
                input_communication_time
            ),
            "execution_time": float(execution_time),
            "output_communication_time": float(
                output_communication_time
            ),
            "communication_time": float(communication_time),
            "total_duration": float(duration),
        }

    def estimate_task_duration_scenario(self, task_id, vm_id, scenario) -> float:
        """计算指定任务在某个固定资源分量场景中的总时长。"""
        return float(
            self.estimate_task_duration_components_scenario(
                task_id,
                vm_id,
                scenario,
            )["total_duration"]
        )

    def _task_start_time_scenario(self, task_id, vm_index, scenario) -> float:
        """按固定映射重放规则计算任务在某个场景中的开始时刻。"""
        task_id = self._task_id(task_id)
        wf_id, _ = self.task_meta[task_id]
        arrival_time = float(self.workflows[wf_id].arrival_time)
        parents = self.task_global_parents[task_id]

        if scenario == "modal":
            vm_available = float(self.vm_available_at[vm_index])
            parent_finish = max(
                (float(self.task_end_time[parent_id]) for parent_id in parents),
                default=arrival_time,
            )
            # modal 事件驱动环境只会在任务 ready 后做预测；current_time 因而同时
            # 包含到达和父任务完成约束，保留它可与旧预测逻辑完全一致。
            return float(
                max(
                    float(self.current_time),
                    arrival_time,
                    vm_available,
                    parent_finish,
                )
            )

        if scenario not in {"optimistic", "pessimistic"}:
            raise ValueError(
                "scenario must be 'optimistic', 'modal', or 'pessimistic'"
            )
        vm_available = float(self.shadow_vm_available_at[scenario][vm_index])
        parent_finish = max(
            (
                float(self.shadow_task_end_time[scenario][parent_id])
                for parent_id in parents
            ),
            default=arrival_time,
        )
        return float(max(arrival_time, vm_available, parent_finish))

    def estimate_task_finish_tfn(self, task, vm) -> TriangularFuzzyNumber:
        """返回相同任务/VM 映射在三场景中的预计完成时刻。

        lower、modal、upper 分别对应 optimistic、modal、pessimistic。若三点
        传播违反自然次序，超过 1e-8 容差时抛出包含任务和 VM 信息的明确异常；
        容差内的浮点反转仅贴合到 modal，不对真实传播结果做排序掩盖。
        """
        task_id = self._task_id(task)
        vm_id, vm_index = self._vm_id_and_index(vm)
        finishes = {}
        for scenario in ("optimistic", "modal", "pessimistic"):
            start = self._task_start_time_scenario(
                task_id, vm_index, scenario
            )
            finishes[scenario] = (
                start
                + self.estimate_task_duration_scenario(
                    task_id, vm_id, scenario
                )
            )

        lower = float(finishes["optimistic"])
        modal = float(finishes["modal"])
        upper = float(finishes["pessimistic"])
        tolerance = 1e-8
        if lower > modal + tolerance or modal > upper + tolerance:
            raise ValueError(
                "Fuzzy finish propagation order violated for "
                f"task_id={task_id}, vm_id={vm_id}: "
                f"optimistic={lower}, modal={modal}, pessimistic={upper}"
            )
        if lower > modal:
            lower = modal
        if upper < modal:
            upper = modal
        return TriangularFuzzyNumber(lower, modal, upper)

    def fuzzy_deadline_measure(self, finish_tfn, eta=None) -> float:
        """计算三角模糊完成时刻的 eta 风险测度。"""
        if not isinstance(finish_tfn, TriangularFuzzyNumber):
            raise TypeError(
                "finish_tfn must be a TriangularFuzzyNumber"
            )
        eta = self.fuzzy_deadline_eta if eta is None else float(eta)
        if not np.isfinite(eta) or not 0.0 <= eta <= 1.0:
            raise ValueError("eta 必须位于 [0, 1]")
        return float(
            finish_tfn.upper
            - (1.0 - eta) * (finish_tfn.upper - finish_tfn.modal)
        )

    @staticmethod
    def _completed_workflow_safety_cost(
        risk_finish_time: float,
        deadline: float,
    ) -> tuple[float, float]:
        """返回一次工作流完成事件的违反指示和模糊超期秒数。"""
        risk_finish_time = float(risk_finish_time)
        deadline = float(deadline)
        if not np.isfinite(risk_finish_time) or not np.isfinite(deadline):
            raise ValueError(
                "risk_finish_time and deadline must both be finite"
            )
        lateness = max(0.0, risk_finish_time - deadline)
        violation = 1.0 if lateness > 0.0 else 0.0
        return float(violation), float(lateness)

    @staticmethod
    def _normalized_process_risk(
        current_time: float,
        predicted_risk_finish_time: float,
        deadline: float,
    ) -> float:
        """把未完成工作流的预测完成位置归一化到 ``[0, 1]``。

        定义为 ``clip((R_hat-current)/(D-current), 0, 1)``。因此预测完成
        时刻等于/超过确定截止期时取 1；预测完成越靠近当前时刻越接近 0。
        若当前时刻已到截止期但工作流仍未完成，则直接取 1。
        """
        current_time = float(current_time)
        predicted_risk_finish_time = float(predicted_risk_finish_time)
        deadline = float(deadline)
        if not all(
            np.isfinite(value)
            for value in (current_time, predicted_risk_finish_time, deadline)
        ):
            raise ValueError(
                "current_time, predicted_risk_finish_time and deadline "
                "must all be finite"
            )
        remaining_deadline = deadline - current_time
        if remaining_deadline <= 1e-12:
            return 1.0
        return float(
            np.clip(
                (predicted_risk_finish_time - current_time)
                / remaining_deadline,
                0.0,
                1.0,
            )
        )

    def _empty_safety_info(self) -> dict:
        """返回字段稳定的零安全代价信息。"""
        return {
            "safety_cost": 0.0,
            "deadline_violation_cost": 0.0,
            "fuzzy_lateness_cost": 0.0,
            "process_risk_cost": 0.0,
            "deadline_violation_count": 0,
            "completed_workflow_count": 0,
            "unfinished_workflow_count": 0,
            "predicted_deadline_violation_count": 0,
            "max_process_risk": 0.0,
            "min_fuzzy_safety_margin": 0.0,
            "minimum_safety_margin": 0.0,
            "minimum_fuzzy_safety_margin": 0.0,
            "mean_safety_margin": 0.0,
            "mean_fuzzy_safety_margin": 0.0,
            "risk_workflow_ratio": 0.0,
            "predicted_violation_rate": 0.0,
            "cumulative_safety_cost": float(
                getattr(self, "_safety_cumulative_cost", 0.0)
            ),
            "cumulative_safety_transition_count": int(
                getattr(
                    self,
                    "_safety_cumulative_transition_count",
                    0,
                )
            ),
            "cumulative_mean_safety_cost": float(
                getattr(self, "_safety_cumulative_cost", 0.0)
                / max(
                    int(
                        getattr(
                            self,
                            "_safety_cumulative_transition_count",
                            0,
                        )
                    ),
                    1,
                )
            ),
            "cumulative_deadline_violation_count": int(
                getattr(
                    self,
                    "_safety_cumulative_deadline_violation_count",
                    0,
                )
            ),
            "cumulative_completed_workflow_count": int(
                getattr(
                    self,
                    "_safety_cumulative_completed_workflow_count",
                    0,
                )
            ),
            "cumulative_fuzzy_lateness_cost": float(
                getattr(
                    self,
                    "_safety_cumulative_fuzzy_lateness_cost",
                    0.0,
                )
            ),
            "cumulative_process_risk_cost": float(
                getattr(
                    self,
                    "_safety_cumulative_process_risk_cost",
                    0.0,
                )
            ),
            "safe_rl_enabled": bool(
                getattr(self, "safe_rl_enabled", False)
            ),
            "safety_cost_aggregation": (
                "deadline_violation_cost + fuzzy_lateness_cost "
                "+ process_risk_cost"
            ),
            "process_risk_aggregation": (
                f"{getattr(self, 'safe_rl_process_risk_aggregation', 'mean')}"
                "_over_unfinished_workflows"
            ),
            "workflow_safety": [],
            "workflow_safety_margins": [],
        }

    def _workflow_task_ids(self, workflow_id: int) -> list[int]:
        """返回属于指定工作流的全局任务编号。"""
        workflow_id = int(workflow_id)
        return [
            int(task_id)
            for task_id, meta in enumerate(self.task_meta)
            if int(meta[0]) == workflow_id
        ]

    def _known_task_finish_for_scenario(
        self,
        task_id: int,
        scenario: str,
    ) -> float:
        """返回已调度任务在指定场景中已经确定的完成时刻。"""
        if scenario == "modal":
            return float(self.task_end_time[task_id])
        return float(self.shadow_task_end_time[scenario][task_id])

    def _workflow_finish_tfn(self, workflow_id: int) -> TriangularFuzzyNumber:
        """从已分配任务时间线取得已完成工作流的模糊完成时刻。"""
        task_ids = self._workflow_task_ids(workflow_id)
        if not task_ids:
            raise ValueError(
                f"workflow_id={workflow_id} has no task in the environment"
            )
        lower = max(
            self._known_task_finish_for_scenario(tid, "optimistic")
            for tid in task_ids
        )
        modal = max(
            self._known_task_finish_for_scenario(tid, "modal")
            for tid in task_ids
        )
        upper = max(
            self._known_task_finish_for_scenario(tid, "pessimistic")
            for tid in task_ids
        )
        tolerance = 1e-8
        if lower > modal + tolerance or modal > upper + tolerance:
            raise ValueError(
                "Completed workflow fuzzy finish order violated for "
                f"workflow_id={workflow_id}: "
                f"optimistic={lower}, modal={modal}, pessimistic={upper}"
            )
        return TriangularFuzzyNumber(
            min(float(lower), float(modal)),
            float(modal),
            max(float(modal), float(upper)),
        )

    def _predict_workflow_finish_scenario(
        self,
        workflow_id: int,
        scenario: str,
    ) -> float:
        """按当前排队状态预测未完成工作流的场景完成时刻。

        已经 Running/Finished 的任务使用真实已提交时间线。尚未调度任务按 DAG
        拓扑依赖，使用当前各 VM 可用时刻和该场景资源能力计算最早可行完成时刻；
        预测不预占后续 VM，因而只作为阶段 1 的过程风险诊断，不是安全屏蔽器。
        """
        if scenario not in {"optimistic", "modal", "pessimistic"}:
            raise ValueError(
                "scenario must be 'optimistic', 'modal', or 'pessimistic'"
            )
        workflow_id = int(workflow_id)
        task_ids = self._workflow_task_ids(workflow_id)
        if not task_ids:
            raise ValueError(
                f"workflow_id={workflow_id} has no task in the environment"
            )
        task_id_set = set(task_ids)
        finish_by_task = {}
        pending = set()
        for task_id in task_ids:
            if self.task_state[task_id] in {"Running", "Finished"}:
                finish_by_task[task_id] = self._known_task_finish_for_scenario(
                    task_id,
                    scenario,
                )
            else:
                pending.add(task_id)

        if scenario == "modal":
            vm_available = self.vm_available_at
        else:
            vm_available = self.shadow_vm_available_at[scenario]

        workflow = self.workflows[workflow_id]
        arrival_time = float(workflow.arrival_time)
        while pending:
            progressed = False
            for task_id in sorted(pending):
                parents = [
                    int(parent_id)
                    for parent_id in self.task_global_parents[task_id]
                    if int(parent_id) in task_id_set
                ]
                if any(parent_id not in finish_by_task for parent_id in parents):
                    continue
                parent_finish = max(
                    (finish_by_task[parent_id] for parent_id in parents),
                    default=arrival_time,
                )
                candidate_finishes = []
                for vm_id in self.get_feasible_vms(task_id):
                    _, vm_index = self._vm_id_and_index(vm_id)
                    start_time = max(
                        float(self.current_time),
                        arrival_time,
                        float(parent_finish),
                        float(vm_available[vm_index]),
                    )
                    candidate_finishes.append(
                        start_time
                        + self.estimate_task_duration_scenario(
                            task_id,
                            vm_id,
                            scenario,
                        )
                    )
                if not candidate_finishes:
                    raise NoFeasibleVMError(
                        f"Task {task_id} has no feasible VM."
                    )
                finish_by_task[task_id] = float(min(candidate_finishes))
                pending.remove(task_id)
                progressed = True
            if not progressed:
                raise ValueError(
                    "Cannot predict workflow finish because its task graph "
                    f"is cyclic or incomplete: workflow_id={workflow_id}"
                )
        return float(max(finish_by_task.values()))

    def predict_workflow_finish_tfn(
        self,
        workflow_id: int,
    ) -> TriangularFuzzyNumber:
        """返回未完成工作流在当前调度状态下的预测模糊完成时刻。"""
        lower = self._predict_workflow_finish_scenario(
            workflow_id,
            "optimistic",
        )
        modal = self._predict_workflow_finish_scenario(
            workflow_id,
            "modal",
        )
        upper = self._predict_workflow_finish_scenario(
            workflow_id,
            "pessimistic",
        )
        tolerance = 1e-8
        if lower > modal + tolerance or modal > upper + tolerance:
            raise ValueError(
                "Predicted workflow fuzzy finish order violated for "
                f"workflow_id={workflow_id}: "
                f"optimistic={lower}, modal={modal}, pessimistic={upper}"
            )
        return TriangularFuzzyNumber(
            min(float(lower), float(modal)),
            float(modal),
            max(float(modal), float(upper)),
        )

    def _remaining_critical_path_components_scenario(
        self,
        task_id: int,
        scenario: str,
    ) -> dict:
        """估计当前任务完成后的场景剩余关键路径，不包含当前任务自身。"""
        task_id = self._task_id(task_id)
        if scenario not in {"optimistic", "modal", "pessimistic"}:
            raise ValueError(
                "scenario must be 'optimistic', 'modal', or 'pessimistic'"
            )
        workflow_id, _ = self.task_meta[task_id]
        local_to_global = {
            int(local_id): int(global_id)
            for global_id, (wf_id, local_id) in enumerate(self.task_meta)
            if int(wf_id) == int(workflow_id)
        }
        memo = {}
        visiting = set()

        def path_after(global_id: int) -> dict:
            if global_id in memo:
                return memo[global_id]
            if global_id in visiting:
                raise ValueError(
                    "Cannot estimate remaining critical path for a cyclic "
                    f"workflow: workflow_id={workflow_id}"
                )
            visiting.add(global_id)
            branches = []
            for child_local_id in self.task_children[global_id]:
                child_id = local_to_global.get(int(child_local_id))
                if child_id is None:
                    continue
                if self.task_state[child_id] == "Finished":
                    own = {
                        "execution_time": 0.0,
                        "communication_time": 0.0,
                        "total_duration": 0.0,
                        "vm_id": None,
                    }
                else:
                    vm_options = []
                    for vm_id in self.get_feasible_vms(child_id):
                        components = (
                            self.estimate_task_duration_components_scenario(
                                child_id,
                                vm_id,
                                scenario,
                            )
                        )
                        vm_options.append(
                            {
                                **components,
                                "vm_id": int(vm_id),
                            }
                        )
                    if not vm_options:
                        raise NoFeasibleVMError(
                            f"Task {child_id} has no feasible VM."
                        )
                    own = min(
                        vm_options,
                        key=lambda value: (
                            float(value["total_duration"]),
                            int(value["vm_id"]),
                        ),
                    )
                tail = path_after(child_id)
                branches.append(
                    {
                        "execution_time": float(
                            own["execution_time"]
                            + tail["execution_time"]
                        ),
                        "communication_time": float(
                            own["communication_time"]
                            + tail["communication_time"]
                        ),
                        "total_duration": float(
                            own["total_duration"]
                            + tail["total_duration"]
                        ),
                        "critical_path_task_ids": [
                            int(child_id),
                            *tail["critical_path_task_ids"],
                        ],
                        "selected_vm_ids": [
                            own["vm_id"],
                            *tail["selected_vm_ids"],
                        ],
                    }
                )
            visiting.remove(global_id)
            if branches:
                result = max(
                    branches,
                    key=lambda value: (
                        float(value["total_duration"]),
                        -int(value["critical_path_task_ids"][0]),
                    ),
                )
            else:
                result = {
                    "execution_time": 0.0,
                    "communication_time": 0.0,
                    "total_duration": 0.0,
                    "critical_path_task_ids": [],
                    "selected_vm_ids": [],
                }
            memo[global_id] = result
            return result

        return dict(path_after(task_id))

    def estimate_task_remaining_critical_path(self, task) -> dict:
        """返回当前任务完成后剩余关键路径的三场景模糊时间预测。"""
        task_id = self._task_id(task)
        by_scenario = {
            scenario: self._remaining_critical_path_components_scenario(
                task_id,
                scenario,
            )
            for scenario in ("optimistic", "modal", "pessimistic")
        }
        lower = float(by_scenario["optimistic"]["total_duration"])
        modal = float(by_scenario["modal"]["total_duration"])
        upper = float(by_scenario["pessimistic"]["total_duration"])
        tolerance = 1e-8
        if lower > modal + tolerance or modal > upper + tolerance:
            raise ValueError(
                "Remaining critical path fuzzy order violated for "
                f"task_id={task_id}: optimistic={lower}, modal={modal}, "
                f"pessimistic={upper}"
            )
        remaining_tfn = TriangularFuzzyNumber(
            min(lower, modal),
            modal,
            max(modal, upper),
        )
        risk_time = self.fuzzy_deadline_measure(remaining_tfn)
        return {
            "task_id": int(task_id),
            "excludes_current_task": True,
            "remaining_critical_path_lower": float(
                remaining_tfn.lower
            ),
            "remaining_critical_path_modal": float(
                remaining_tfn.modal
            ),
            "remaining_critical_path_upper": float(
                remaining_tfn.upper
            ),
            "remaining_critical_path_risk": float(risk_time),
            "remaining_execution_time_optimistic": float(
                by_scenario["optimistic"]["execution_time"]
            ),
            "remaining_execution_time_modal": float(
                by_scenario["modal"]["execution_time"]
            ),
            "remaining_execution_time_pessimistic": float(
                by_scenario["pessimistic"]["execution_time"]
            ),
            "remaining_communication_time_optimistic": float(
                by_scenario["optimistic"]["communication_time"]
            ),
            "remaining_communication_time_modal": float(
                by_scenario["modal"]["communication_time"]
            ),
            "remaining_communication_time_pessimistic": float(
                by_scenario["pessimistic"]["communication_time"]
            ),
            "critical_path_task_ids_optimistic": list(
                by_scenario["optimistic"]["critical_path_task_ids"]
            ),
            "critical_path_task_ids_modal": list(
                by_scenario["modal"]["critical_path_task_ids"]
            ),
            "critical_path_task_ids_pessimistic": list(
                by_scenario["pessimistic"]["critical_path_task_ids"]
            ),
            "upward_rank_reference": float(
                self.task_up_rank[task_id]
                if task_id < len(self.task_up_rank)
                else 0.0
            ),
            "prediction_is_safety_guarantee": False,
        }

    def _task_parent_placement_diagnostics(
        self,
        task_id: int,
        vm_id: int,
    ) -> dict:
        """按候选 VM 统计父数据位置；运行时仍传输全部父数据。"""
        task_id = self._task_id(task_id)
        vm_id, _ = self._vm_id_and_index(vm_id)
        workflow_id, local_id = self.task_meta[task_id]
        task_obj = self.workflows[workflow_id].tasks[local_id]
        candidate_host_id = int(self.vms[vm_id].host_id)
        local_to_global = {
            int(candidate_local_id): int(global_id)
            for global_id, (wf_id, candidate_local_id) in enumerate(
                self.task_meta
            )
            if int(wf_id) == int(workflow_id)
        }
        same_vm_bits = 0.0
        same_host_bits = 0.0
        cross_host_bits = 0.0
        unknown_parent_location_bits = 0.0
        parent_in_bits = dict(
            getattr(task_obj, "parent_in_bits", {}) or {}
        )
        for parent_local_id, bits in parent_in_bits.items():
            bits = float(bits)
            parent_global_id = local_to_global.get(int(parent_local_id))
            if parent_global_id is None:
                unknown_parent_location_bits += bits
                continue
            parent_vm_id = self.task_assigned_vm.get(parent_global_id)
            parent_host_id = self.task_assigned_host.get(parent_global_id)
            if parent_host_id is None:
                unknown_parent_location_bits += bits
            elif parent_vm_id == vm_id:
                same_vm_bits += bits
            elif int(parent_host_id) == candidate_host_id:
                same_host_bits += bits
            else:
                cross_host_bits += bits
        parent_bits_total = float(sum(parent_in_bits.values()))
        external_input_bits = float(
            getattr(task_obj, "ext_in_bits", 0.0) or 0.0
        )
        return {
            "candidate_host_id": int(candidate_host_id),
            "parent_input_bits": float(parent_bits_total),
            "external_input_bits": float(external_input_bits),
            "same_vm_parent_bits": float(same_vm_bits),
            "same_host_parent_bits": float(same_host_bits),
            "cross_host_parent_bits": float(cross_host_bits),
            "unknown_parent_location_bits": float(
                unknown_parent_location_bits
            ),
            "runtime_transferred_parent_bits": float(
                parent_bits_total
            ),
            "runtime_communication_model": (
                "all task input/output uses candidate VM bandwidth; "
                "parent locality is diagnostic only"
            ),
        }

    def predict_task_vm_action_risk(
        self,
        task,
        vm,
        *,
        remaining_prediction=None,
    ) -> dict:
        """只读预测当前任务分配到候选 VM 后的任务级安全风险。"""
        task_id = self._task_id(task)
        vm_id, vm_index = self._vm_id_and_index(vm)
        if self.task_state[task_id] == "Finished":
            raise ValueError(
                f"Cannot predict an action for finished task {task_id}."
            )
        if remaining_prediction is None:
            remaining_prediction = (
                self.estimate_task_remaining_critical_path(task_id)
            )
        scenario_components = {
            scenario: self.estimate_task_duration_components_scenario(
                task_id,
                vm_id,
                scenario,
            )
            for scenario in ("optimistic", "modal", "pessimistic")
        }
        finish_tfn = self.estimate_task_finish_tfn(task_id, vm_id)
        risk_finish = self.fuzzy_deadline_measure(finish_tfn)
        workflow_id, _ = self.task_meta[task_id]
        workflow_deadline = float(self.workflows[workflow_id].deadline)
        task_safe_deadline = float(
            workflow_deadline
            - remaining_prediction["remaining_critical_path_risk"]
        )
        safety_margin = float(task_safe_deadline - risk_finish)
        boundary_tolerance = 1e-9
        predicted_violation_amount = max(0.0, -safety_margin)
        is_at_boundary = abs(safety_margin) <= boundary_tolerance
        is_predicted_safe = safety_margin >= -boundary_tolerance

        if self.vm_available_at[vm_index] <= self.current_time + 1e-9:
            is_currently_selectable = True
        else:
            is_currently_selectable = False
        vm_available_by_scenario = {
            "optimistic": float(
                self.shadow_vm_available_at["optimistic"][vm_index]
            ),
            "modal": float(self.vm_available_at[vm_index]),
            "pessimistic": float(
                self.shadow_vm_available_at["pessimistic"][vm_index]
            ),
        }
        queue_delay_by_scenario = {
            scenario: max(
                0.0,
                available_at - float(self.current_time),
            )
            for scenario, available_at in vm_available_by_scenario.items()
        }
        placement = self._task_parent_placement_diagnostics(
            task_id,
            vm_id,
        )
        return {
            "task_id": int(task_id),
            "workflow_id": int(workflow_id),
            "vm_id": int(vm_id),
            "host_id": int(self.vms[vm_id].host_id),
            "optimistic_finish": float(finish_tfn.lower),
            "modal_finish": float(finish_tfn.modal),
            "pessimistic_finish": float(finish_tfn.upper),
            "risk_finish": float(risk_finish),
            "task_safe_deadline": float(task_safe_deadline),
            "safety_margin": float(safety_margin),
            "predicted_violation_amount": float(
                predicted_violation_amount
            ),
            "is_predicted_safe": bool(is_predicted_safe),
            "is_at_safety_boundary": bool(is_at_boundary),
            "is_currently_selectable": bool(
                is_currently_selectable
            ),
            "current_task_execution_time_optimistic": float(
                scenario_components["optimistic"]["execution_time"]
            ),
            "current_task_execution_time_modal": float(
                scenario_components["modal"]["execution_time"]
            ),
            "current_task_execution_time_pessimistic": float(
                scenario_components["pessimistic"]["execution_time"]
            ),
            "current_task_communication_time_optimistic": float(
                scenario_components["optimistic"][
                    "communication_time"
                ]
            ),
            "current_task_communication_time_modal": float(
                scenario_components["modal"]["communication_time"]
            ),
            "current_task_communication_time_pessimistic": float(
                scenario_components["pessimistic"][
                    "communication_time"
                ]
            ),
            "vm_queue_delay_optimistic": float(
                queue_delay_by_scenario["optimistic"]
            ),
            "vm_queue_delay_modal": float(
                queue_delay_by_scenario["modal"]
            ),
            "vm_queue_delay_pessimistic": float(
                queue_delay_by_scenario["pessimistic"]
            ),
            **remaining_prediction,
            **placement,
            "prediction_only": True,
            "action_mask_applied": False,
            "prediction_is_safety_guarantee": False,
        }

    def predict_task_host_action_risk(
        self,
        task,
        host_id: int,
        *,
        vm_predictions=None,
    ) -> dict:
        """只通过 Host 内当前可选 VM 的预测聚合 Host 动作风险。"""
        task_id = self._task_id(task)
        host_id = int(host_id)
        if host_id not in self.host_ids:
            raise ValueError(f"Unknown host_id: {host_id}")
        internal_vm_ids = {
            int(self.vm_ids[vm_index])
            for vm_index in self.host_to_vm_indices[host_id]
        }
        if vm_predictions is None:
            remaining = self.estimate_task_remaining_critical_path(task_id)
            vm_predictions = [
                self.predict_task_vm_action_risk(
                    task_id,
                    vm_id,
                    remaining_prediction=remaining,
                )
                for vm_id in sorted(internal_vm_ids)
                if vm_id in self.get_feasible_vms(task_id)
            ]
        internal_predictions = [
            prediction
            for prediction in vm_predictions
            if int(prediction["vm_id"]) in internal_vm_ids
        ]
        selectable_predictions = [
            prediction
            for prediction in internal_predictions
            if bool(prediction["is_currently_selectable"])
        ]
        safe_predictions = [
            prediction
            for prediction in selectable_predictions
            if bool(prediction["is_predicted_safe"])
        ]
        if selectable_predictions:
            best = min(
                selectable_predictions,
                key=lambda prediction: (
                    float(
                        prediction["predicted_violation_amount"]
                    ),
                    float(prediction["risk_finish"]),
                    int(prediction["vm_id"]),
                ),
            )
            minimum_violation = min(
                float(value["predicted_violation_amount"])
                for value in selectable_predictions
            )
            maximum_margin = max(
                float(value["safety_margin"])
                for value in selectable_predictions
            )
            minimum_risk_finish = min(
                float(value["risk_finish"])
                for value in selectable_predictions
            )
            best_vm_id = int(best["vm_id"])
        else:
            minimum_violation = 0.0
            maximum_margin = 0.0
            minimum_risk_finish = 0.0
            best_vm_id = None
        return {
            "task_id": int(task_id),
            "host_id": int(host_id),
            "is_hard_action_legal": bool(selectable_predictions),
            "candidate_vm_count": int(len(selectable_predictions)),
            "safe_vm_count": int(len(safe_predictions)),
            "safe_vm_ratio": float(
                len(safe_predictions) / len(selectable_predictions)
                if selectable_predictions
                else 0.0
            ),
            "is_predicted_safe": bool(safe_predictions),
            "minimum_predicted_violation_amount": float(
                minimum_violation
            ),
            "maximum_safety_margin": float(maximum_margin),
            "minimum_risk_finish": float(minimum_risk_finish),
            "best_diagnostic_vm_id": best_vm_id,
            "candidate_vm_predictions": selectable_predictions,
            "all_internal_vm_predictions": internal_predictions,
            "aggregation_rule": (
                "aggregate currently selectable VM predictions; no "
                "independent Host finish-time model"
            ),
            "prediction_only": True,
            "action_mask_applied": False,
        }

    def get_task_action_risk_predictions(self, task=None) -> dict:
        """输出当前任务全部 VM/Host 候选动作风险，不执行或屏蔽动作。"""
        if not getattr(self, "safe_rl_enabled", False):
            return {
                "safe_rl_enabled": False,
                "prediction_available": False,
                "vm_predictions": [],
                "host_predictions": [],
                "action_mask_applied": False,
            }
        if task is None:
            task = getattr(self, "_cur_tid", None)
        if task is None:
            return {
                "safe_rl_enabled": True,
                "prediction_available": False,
                "vm_predictions": [],
                "host_predictions": [],
                "action_mask_applied": False,
            }
        task_id = self._task_id(task)
        remaining = self.estimate_task_remaining_critical_path(task_id)
        vm_predictions = [
            self.predict_task_vm_action_risk(
                task_id,
                vm_id,
                remaining_prediction=remaining,
            )
            for vm_id in self.get_feasible_vms(task_id)
        ]
        host_predictions = [
            self.predict_task_host_action_risk(
                task_id,
                host_id,
                vm_predictions=vm_predictions,
            )
            for host_id in self.host_ids
        ]
        return {
            "safe_rl_enabled": True,
            "prediction_available": True,
            "task_id": int(task_id),
            "remaining_critical_path": remaining,
            "vm_predictions": vm_predictions,
            "host_predictions": host_predictions,
            "prediction_only": True,
            "action_mask_applied": False,
            "action_selection_changed": False,
            "prediction_is_safety_guarantee": False,
        }

    @staticmethod
    def _normalize_fuzzy_safety_margin(
        margin: float,
        arrival_time: float,
        deadline: float,
    ) -> float:
        """按工作流确定性 DDL 总预算把裕量归一化到 ``[-1, 1]``。"""
        margin = float(margin)
        arrival_time = float(arrival_time)
        deadline = float(deadline)
        if not all(
            np.isfinite(value)
            for value in (margin, arrival_time, deadline)
        ):
            raise ValueError(
                "margin, arrival_time and deadline must all be finite"
            )
        deadline_budget = max(deadline - arrival_time, 1e-12)
        return float(np.clip(margin / deadline_budget, -1.0, 1.0))

    def get_dynamic_fuzzy_safety_margins(self) -> dict:
        """返回所有未完成工作流的统一动态模糊安全裕量。

        该接口只读取当前任务状态、DAG、VM 可用时刻与已有三条模糊时间线，
        不修改任务顺序、动作 mask、Host/VM 选择或事件队列。返回值是模型预测，
        不是实际完成后的安全保证。
        """
        base = {
            "safe_rl_enabled": bool(
                getattr(self, "safe_rl_enabled", False)
            ),
            "current_time": float(getattr(self, "current_time", 0.0)),
            "deadline_eta": float(
                getattr(self, "fuzzy_deadline_eta", 0.95)
            ),
            "unfinished_workflow_count": 0,
            "active_workflow_count": 0,
            "risk_workflow_count": 0,
            "predicted_violation_count": 0,
            "minimum_safety_margin": 0.0,
            "minimum_fuzzy_safety_margin": 0.0,
            "mean_safety_margin": 0.0,
            "mean_fuzzy_safety_margin": 0.0,
            "risk_workflow_ratio": 0.0,
            "predicted_violation_rate": 0.0,
            "workflow_safety_margins": [],
            "prediction_model": (
                "current DAG state + existing optimistic/modal/"
                "pessimistic VM timelines"
            ),
            "prediction_is_safety_guarantee": False,
            "safety_margin_definition": (
                "deadline - ((1 - eta) * modal + eta * upper)"
            ),
            "normalized_safety_margin_definition": (
                "clip(fuzzy_safety_margin / "
                "(deadline - arrival_time), -1, 1)"
            ),
            "safety_status_definition": (
                "positive: remaining margin; zero: safety boundary; "
                "negative: predicted deadline violation"
            ),
        }
        if not getattr(self, "safe_rl_enabled", False):
            return base

        boundary_tolerance = 1e-9
        rows = []
        completed_ids = set(
            int(workflow_id)
            for workflow_id in getattr(self, "wf_finish_time", {})
        )
        ready_ids = set(
            int(task_id)
            for task_id in getattr(self, "ready_task_ids", [])
        )
        for workflow_id, workflow in enumerate(self.workflows):
            if workflow_id in completed_ids:
                continue

            predicted_tfn = self.predict_workflow_finish_tfn(workflow_id)
            fuzzy_finish_risk = self.fuzzy_deadline_measure(predicted_tfn)
            deadline = float(workflow.deadline)
            arrival_time = float(workflow.arrival_time)
            margin = float(deadline - fuzzy_finish_risk)
            normalized_margin = self._normalize_fuzzy_safety_margin(
                margin,
                arrival_time,
                deadline,
            )
            is_boundary = abs(margin) <= boundary_tolerance
            is_predicted_violation = margin < -boundary_tolerance
            # 位于边界或已经预测越界都属于风险工作流；只有严格负裕量计入
            # predicted_violation_rate。
            is_predicted_at_risk = margin <= boundary_tolerance
            if is_predicted_violation:
                safety_status = "predicted_violation"
            elif is_boundary:
                safety_status = "safety_boundary"
            else:
                safety_status = "positive_margin"

            task_ids = (
                self._workflow_task_ids(workflow_id)
                if hasattr(self, "task_meta")
                else []
            )
            task_states = getattr(self, "task_state", [])
            task_workloads = getattr(self, "task_mi", [])
            task_upward_ranks = getattr(self, "task_up_rank", [])
            finished_task_count = sum(
                task_id < len(task_states)
                and task_states[task_id] == "Finished"
                for task_id in task_ids
            )
            running_task_count = sum(
                task_id < len(task_states)
                and task_states[task_id] == "Running"
                for task_id in task_ids
            )
            ready_task_count = sum(
                task_id in ready_ids
                and task_id < len(task_states)
                and task_states[task_id] == "Ready"
                for task_id in task_ids
            )
            remaining_task_ids = [
                task_id
                for task_id in task_ids
                if task_id >= len(task_states)
                or task_states[task_id] != "Finished"
            ]
            remaining_workload_mi = float(
                sum(
                    task_workloads[task_id]
                    for task_id in remaining_task_ids
                    if task_id < len(task_workloads)
                )
            )
            max_remaining_upward_rank = float(
                max(
                    (
                        task_upward_ranks[task_id]
                        for task_id in remaining_task_ids
                        if task_id < len(task_upward_ranks)
                    ),
                    default=0.0,
                )
            )
            process_risk = self._normalized_process_risk(
                self.current_time,
                fuzzy_finish_risk,
                deadline,
            )

            rows.append(
                {
                    "workflow_id": int(workflow_id),
                    "completed": False,
                    "fuzzy_finish_estimate_lower": float(
                        predicted_tfn.lower
                    ),
                    "fuzzy_finish_estimate_modal": float(
                        predicted_tfn.modal
                    ),
                    "fuzzy_finish_estimate_upper": float(
                        predicted_tfn.upper
                    ),
                    "fuzzy_finish_risk": float(fuzzy_finish_risk),
                    "deadline": float(deadline),
                    "fuzzy_safety_margin": float(margin),
                    "normalized_fuzzy_safety_margin": float(
                        normalized_margin
                    ),
                    "is_predicted_at_risk": bool(
                        is_predicted_at_risk
                    ),
                    "is_at_safety_boundary": bool(is_boundary),
                    "is_predicted_violation": bool(
                        is_predicted_violation
                    ),
                    "safety_status": safety_status,
                    "finished_task_count": int(finished_task_count),
                    "running_task_count": int(running_task_count),
                    "ready_task_count": int(ready_task_count),
                    "remaining_task_count": int(
                        len(remaining_task_ids)
                    ),
                    "remaining_workload_mi": float(
                        remaining_workload_mi
                    ),
                    "max_remaining_upward_rank": float(
                        max_remaining_upward_rank
                    ),
                    "process_risk_cost": float(process_risk),
                    # 阶段 1 字段别名继续保留，避免破坏现有 info 消费代码。
                    "fuzzy_finish_lower": float(predicted_tfn.lower),
                    "fuzzy_finish_modal": float(predicted_tfn.modal),
                    "fuzzy_finish_upper": float(predicted_tfn.upper),
                    "risk_finish_time": float(fuzzy_finish_risk),
                    "deadline_violation_cost": 0.0,
                    "fuzzy_lateness_cost": 0.0,
                }
            )

        margins = [float(row["fuzzy_safety_margin"]) for row in rows]
        risk_count = sum(
            bool(row["is_predicted_at_risk"]) for row in rows
        )
        violation_count = sum(
            bool(row["is_predicted_violation"]) for row in rows
        )
        workflow_count = len(rows)
        minimum_margin = float(min(margins)) if margins else 0.0
        mean_margin = float(np.mean(margins)) if margins else 0.0
        base.update(
            {
                "unfinished_workflow_count": int(workflow_count),
                "active_workflow_count": int(workflow_count),
                "risk_workflow_count": int(risk_count),
                "predicted_violation_count": int(violation_count),
                "minimum_safety_margin": float(minimum_margin),
                "minimum_fuzzy_safety_margin": float(minimum_margin),
                "mean_safety_margin": float(mean_margin),
                "mean_fuzzy_safety_margin": float(mean_margin),
                "risk_workflow_ratio": float(
                    risk_count / workflow_count
                    if workflow_count > 0
                    else 0.0
                ),
                "predicted_violation_rate": float(
                    violation_count / workflow_count
                    if workflow_count > 0
                    else 0.0
                ),
                "workflow_safety_margins": rows,
            }
        )
        return base

    def get_safety_diagnostics(self) -> dict:
        """返回本次转换的独立安全代价和可累计诊断指标。

        完成代价只对尚未进入 ``_safety_accounted_workflow_ids`` 的工作流结算，
        因而同一完成事件不会在后续 VM/Manager 转换中重复计费。过程风险是
        当前状态快照，在每个实际转换上重新计算；多个层级的 cost 不应跨层相加。
        """
        if not getattr(self, "safe_rl_enabled", False):
            return self._empty_safety_info()

        info = self._empty_safety_info()
        accounted = self._safety_accounted_workflow_ids
        completed_ids = sorted(
            int(workflow_id)
            for workflow_id in self.wf_finish_time
            if int(workflow_id) not in accounted
        )

        violation_cost = 0.0
        lateness_cost = 0.0
        workflow_safety = []
        for workflow_id in completed_ids:
            workflow = self.workflows[workflow_id]
            deadline = float(workflow.deadline)
            finish_tfn = self._workflow_finish_tfn(workflow_id)
            risk_finish = self.fuzzy_deadline_measure(finish_tfn)
            violation, lateness = self._completed_workflow_safety_cost(
                risk_finish,
                deadline,
            )
            violation_cost += violation
            lateness_cost += lateness
            workflow_safety.append(
                {
                    "workflow_id": int(workflow_id),
                    "completed": True,
                    "deadline": deadline,
                    "fuzzy_finish_lower": float(finish_tfn.lower),
                    "fuzzy_finish_modal": float(finish_tfn.modal),
                    "fuzzy_finish_upper": float(finish_tfn.upper),
                    "risk_finish_time": float(risk_finish),
                    "fuzzy_safety_margin": float(deadline - risk_finish),
                    "deadline_violation_cost": float(violation),
                    "fuzzy_lateness_cost": float(lateness),
                    "process_risk_cost": 0.0,
                }
            )
            accounted.add(workflow_id)

        margin_summary = self.get_dynamic_fuzzy_safety_margins()
        unfinished_rows = list(
            margin_summary["workflow_safety_margins"]
        )

        process_risks = [
            float(row["process_risk_cost"]) for row in unfinished_rows
        ]
        process_risk_cost = (
            float(np.mean(process_risks)) if process_risks else 0.0
        )
        all_rows = workflow_safety + unfinished_rows
        safety_cost = (
            float(violation_cost)
            + float(lateness_cost)
            + float(process_risk_cost)
        )

        completed_count = len(completed_ids)
        violation_count = int(violation_cost)
        self._safety_cumulative_cost += float(safety_cost)
        self._safety_cumulative_transition_count = int(
            getattr(
                self,
                "_safety_cumulative_transition_count",
                0,
            )
        ) + 1
        self._safety_cumulative_deadline_violation_count += violation_count
        self._safety_cumulative_completed_workflow_count += completed_count
        self._safety_cumulative_fuzzy_lateness_cost += float(lateness_cost)
        self._safety_cumulative_process_risk_cost += float(
            process_risk_cost
        )

        info.update(
            {
                "safety_cost": float(safety_cost),
                "deadline_violation_cost": float(violation_cost),
                "fuzzy_lateness_cost": float(lateness_cost),
                "process_risk_cost": float(process_risk_cost),
                "deadline_violation_count": int(violation_count),
                "completed_workflow_count": int(completed_count),
                "unfinished_workflow_count": int(len(unfinished_rows)),
                "predicted_deadline_violation_count": int(
                    margin_summary["predicted_violation_count"]
                ),
                "max_process_risk": float(
                    max(process_risks) if process_risks else 0.0
                ),
                "min_fuzzy_safety_margin": float(
                    margin_summary["minimum_fuzzy_safety_margin"]
                ),
                "minimum_safety_margin": float(
                    margin_summary["minimum_safety_margin"]
                ),
                "minimum_fuzzy_safety_margin": float(
                    margin_summary["minimum_fuzzy_safety_margin"]
                ),
                "mean_safety_margin": float(
                    margin_summary["mean_safety_margin"]
                ),
                "mean_fuzzy_safety_margin": float(
                    margin_summary["mean_fuzzy_safety_margin"]
                ),
                "risk_workflow_ratio": float(
                    margin_summary["risk_workflow_ratio"]
                ),
                "predicted_violation_rate": float(
                    margin_summary["predicted_violation_rate"]
                ),
                "cumulative_safety_cost": float(
                    self._safety_cumulative_cost
                ),
                "cumulative_safety_transition_count": int(
                    self._safety_cumulative_transition_count
                ),
                "cumulative_mean_safety_cost": float(
                    self._safety_cumulative_cost
                    / max(
                        self._safety_cumulative_transition_count,
                        1,
                    )
                ),
                "cumulative_deadline_violation_count": int(
                    self._safety_cumulative_deadline_violation_count
                ),
                "cumulative_completed_workflow_count": int(
                    self._safety_cumulative_completed_workflow_count
                ),
                "cumulative_fuzzy_lateness_cost": float(
                    self._safety_cumulative_fuzzy_lateness_cost
                ),
                "cumulative_process_risk_cost": float(
                    self._safety_cumulative_process_risk_cost
                ),
                "workflow_safety": all_rows,
                "workflow_safety_margins": unfinished_rows,
            }
        )
        return info

    def _safe_performance_reward_breakdown(
        self,
        energy_before=None,
    ) -> dict:
        """按动作前后模糊能耗 objective 快照构造性能奖励。

        当前没有额外 shaping；五个 shaping 字段显式为 0。保留正的
        ``energy_reward_scale`` 只改变学习数值尺度，不改变最终能耗目标。
        safety cost 不在这里读取，也不参与总性能奖励。
        """
        if not getattr(self, "safe_rl_enabled", False):
            score = float(
                getattr(self, "_safe_fuzzy_energy_score", 0.0)
            )
            return {
                "energy_reward": 0.0,
                "completion_reward": 0.0,
                "waiting_reward": 0.0,
                "utilization_reward": 0.0,
                "communication_reward": 0.0,
                "total_performance_reward": 0.0,
                "raw_energy_reward": 0.0,
                "performance_energy_delta": 0.0,
                "fuzzy_energy_score_before": score,
                "fuzzy_energy_score_after": score,
                "performance_energy_score": score,
                "performance_reward_scale": float(
                    self.energy_reward_scale
                ),
                "performance_reward_safety_cost_included": False,
            }

        current = self.get_fuzzy_energy_summary()
        current_score = float(
            current["fuzzy_total_energy_score"]
        )
        if energy_before is None:
            previous_score = float(
                getattr(self, "_safe_fuzzy_energy_score", 0.0)
            )
        elif isinstance(energy_before, dict):
            previous_score = float(
                energy_before["fuzzy_total_energy_score"]
            )
        else:
            previous_score = float(energy_before)
        if not np.isfinite(previous_score):
            raise ValueError(
                "fuzzy energy score before action must be finite"
            )

        delta = float(current_score - previous_score)
        raw_energy_reward = -delta
        energy_reward = (
            raw_energy_reward * float(self.energy_reward_scale)
        )
        # 本阶段选择不启用 shaping，避免悄悄改变既有安全实验目标。
        components = {
            "energy_reward": float(energy_reward),
            "completion_reward": 0.0,
            "waiting_reward": 0.0,
            "utilization_reward": 0.0,
            "communication_reward": 0.0,
        }
        total = float(sum(components.values()))
        self._safe_fuzzy_energy_score = current_score
        return {
            **components,
            "total_performance_reward": total,
            "raw_energy_reward": float(raw_energy_reward),
            "performance_energy_delta": delta,
            "fuzzy_energy_score_before": float(previous_score),
            "fuzzy_energy_score_after": float(current_score),
            "performance_energy_score": float(current_score),
            "fuzzy_energy_mean_after": float(
                current["fuzzy_total_energy_mean"]
            ),
            "fuzzy_energy_std_after": float(
                current["fuzzy_total_energy_std"]
            ),
            "performance_reward_scale": float(
                self.energy_reward_scale
            ),
            "performance_reward_safety_cost_included": False,
            "performance_reward_definition": (
                "- delta(fuzzy energy mean + "
                f"{self.fuzzy_energy_uncertainty_weight:.1f} * "
                "fuzzy energy std) * performance_reward_scale"
            ),
        }

    def _safe_performance_reward_delta(
        self,
        energy_before=None,
    ) -> tuple[float, float, float]:
        """兼容旧三元组接口；内部使用显式 reward 分解。"""
        breakdown = self._safe_performance_reward_breakdown(
            energy_before
        )
        return (
            float(breakdown["total_performance_reward"]),
            float(breakdown["performance_energy_delta"]),
            float(breakdown["performance_energy_score"]),
        )

    def _empty_safe_performance_reward_info(self) -> dict:
        """返回没有执行调度动作时的零性能奖励分解。"""
        if not getattr(self, "safe_rl_enabled", False):
            return {}
        current = self.get_fuzzy_energy_summary()
        breakdown = self._safe_performance_reward_breakdown(
            current
        )
        return {
            "performance_reward": 0.0,
            "performance_reward_host": 0.0,
            "performance_reward_vm": 0.0,
            **breakdown,
        }

    def estimate_incremental_energy(self, task, vm) -> float:
        """估计把任务追加到该 VM 后产生的 Host 边际能耗，单位焦耳。

        若 VM 忙，任务从 vm_available_at 开始执行；能耗代理接收 Host、预计开始
        时刻、执行加通信总时长和 VM 模态处理能力，复用已有能耗模型而非新公式。
        此函数只预测，不写入时间线或累计能耗。
        """
        task_id = self._task_id(task)
        vm_id, vm_index = self._vm_id_and_index(vm)
        vm_obj = self.vms[vm_id]
        queue_time = max(0.0, float(self.vm_available_at[vm_index]) - float(self.current_time))
        start_time = float(self.current_time) + queue_time
        duration = self.estimate_exec_time(task_id, vm_id) + self.estimate_comm_time(task_id, vm_id)
        value = self._estimate_marginal_energy_proxy(
            int(vm_obj.host_id),
            start_time,
            duration,
            float(vm_obj.pc),
            vm_available_array=self.vm_available_at,
            pc_component="modal",
            total_pc_component="modal",
        )
        return float(value)

    def estimate_incremental_energy_tfn(self, task, vm):
        """返回同一 task-VM 映射在三场景中的边际能耗包络。

        三个原始边际能耗不假定单调：并发负载与 SPECpower 分段曲线可能使
        optimistic 能耗高于 modal 或 pessimistic。因而 lower/upper 使用三者
        min/max，modal 分量始终保留原确定性场景能耗。
        """
        task_id = self._task_id(task)
        vm_id, vm_index = self._vm_id_and_index(vm)
        vm_obj = self.vms[vm_id]
        component_by_scenario = {
            "optimistic": "upper",
            "modal": "modal",
            "pessimistic": "lower",
        }
        energies = {}
        for scenario, component_name in component_by_scenario.items():
            start_time = self._task_start_time_scenario(
                task_id, vm_index, scenario
            )
            duration = self.estimate_task_duration_scenario(
                task_id, vm_id, scenario
            )
            vm_available_array = (
                self.vm_available_at
                if scenario == "modal"
                else self.shadow_vm_available_at[scenario]
            )
            energies[scenario] = self._estimate_marginal_energy_proxy(
                int(vm_obj.host_id),
                start_time,
                duration,
                vm_obj.pc.component(component_name),
                vm_available_array=vm_available_array,
                pc_component=component_name,
                total_pc_component=component_name,
            )

        raw_values = [
            float(energies["optimistic"]),
            float(energies["modal"]),
            float(energies["pessimistic"]),
        ]
        return TriangularFuzzyNumber(
            min(raw_values),
            float(energies["modal"]),
            max(raw_values),
        )

    def estimate_incremental_energy_score(self, task, vm) -> float:
        """返回风险调整后的模糊边际能耗，单位仍解释为风险调整焦耳。"""
        return float(
            self.estimate_incremental_energy_tfn(task, vm).score(
                self.fuzzy_energy_uncertainty_weight
            )
        )

    def calculate_task_slack(self, task) -> float:
        """计算 ``任务绝对子截止期 - 最早可行预测完成时刻``，单位秒。

        对每个可行 VM 同时计入当前时刻、排队、执行和通信，取其中最早完成值。
        返回负数表示即使采用当前最快的可行方案也预计超期。这里只提供任务特征，
        不会因此提前选择该 VM；真正 VM 仍由固定策略在选定任务后决定。
        """
        task_id = self._task_id(task)
        feasible_vms = self.get_feasible_vms(task_id)
        if not feasible_vms:
            raise NoFeasibleVMError(f"Task {task_id} has no feasible VM.")
        finish_times = []
        for vm_id in feasible_vms:
            if self.fuzzy_enabled:
                finish_times.append(
                    self.fuzzy_deadline_measure(
                        self.estimate_task_finish_tfn(task_id, vm_id)
                    )
                )
            else:
                _, vm_index = self._vm_id_and_index(vm_id)
                queue_time = max(
                    0.0,
                    float(self.vm_available_at[vm_index])
                    - float(self.current_time),
                )
                finish_times.append(
                    float(self.current_time)
                    + queue_time
                    + self.estimate_exec_time(task_id, vm_id)
                    + self.estimate_comm_time(task_id, vm_id)
                )
        return float(self._task_deadline(task_id) - min(finish_times))

    def calculate_upward_rank(self, task) -> float:
        """返回环境预计算的 HEFT upward rank，表示剩余关键路径重要程度。"""
        task_id = self._task_id(task)
        return float(self.task_up_rank[task_id])

    def calculate_remaining_work(self, task) -> float:
        """返回当前任务及其所有后继节点的去重计算工作量总和，单位 MI。

        DAG 可能包含汇合分支，同一后继可由多条路径到达，因此用 visited 去重。
        工作流 DAG 在单次 episode 内不会改变，结果可以按全局 task_id 缓存。
        """
        task_id = self._task_id(task)
        cached = self._remaining_work_cache.get(task_id)
        if cached is not None:
            return float(cached)

        workflow_id, _ = self.task_meta[task_id]
        # task_children 保存工作流内局部 ID；遍历前建立局部到全局 ID 的映射。
        local_to_global = {
            local_id: gid
            for gid, (wf_id, local_id) in enumerate(self.task_meta)
            if wf_id == workflow_id
        }
        stack = [task_id]
        visited = set()
        work_mi = 0.0
        # 使用显式栈进行深度优先遍历，避免递归深度受大型 DAG 影响。
        while stack:
            gid = int(stack.pop())
            if gid in visited:
                continue
            visited.add(gid)
            work_mi += float(self.task_mi[gid])
            for child_local_id in self.task_children[gid]:
                child_gid = local_to_global.get(int(child_local_id))
                if child_gid is not None:
                    stack.append(int(child_gid))
        self._remaining_work_cache[task_id] = float(work_mi)
        return float(work_mi)

    def calculate_uncertainty(self, task, vm) -> float:
        """返回模糊“计算+传输”总时长的标准差，作为该 task-VM 风险量。

        VM 的处理能力和带宽均为三角模糊数；processing_time/transfer_time 会把
        不确定性传播到时长，``std`` 将其压缩为非负标量。值越大表示预计时长
        对处理能力或带宽波动越敏感。
        """
        task_id = self._task_id(task)
        vm_id, _ = self._vm_id_and_index(vm)
        vm_obj = self.vms[vm_id]
        fuzzy_duration = vm_obj.total_duration(
            float(self.task_mi[task_id]),
            float(self.task_in_bits[task_id]),
            float(self.task_out_bits[task_id]),
        )
        return float(fuzzy_duration.std())

    def build_task_features(self, ready_tasks):
        """按 ready_tasks 当前顺序构造 LLM 规则消费的八个一维特征数组。

        每个任务遍历全部可行 VM：执行时间、通信时间和增量能耗分别取最小值。
        fuzzy_enabled=true 时，min_incremental_energy 是三场景边际能耗 TFN 的
        ``mean + uncertainty_weight * std`` 最小值；否则保持旧 modal 焦耳值。
        slack 在 fuzzy 模式下基于 eta 风险完成时刻，modal 模式保持旧预测完成时刻。
        uncertainty 统一取可行 VM 模糊时长标准差的最小值，使其与前三个
        “best feasible VM”特征口径一致。这里可能分别由不同 VM 产生各项最小值，
        它们只用于任务级排序，绝不意味着 LLM 已选择某台 VM。

        返回字典中的每个数组形状都是 (N,)，且下标 i 始终对应 ready_tasks[i]。
        """
        ready_ids = [self._task_id(task) for task in ready_tasks]
        names = (
            "min_exec_time",
            "min_comm_time",
            "min_incremental_energy",
            "slack",
            "upward_rank",
            "remaining_work",
            "ready_wait_time",
            "uncertainty",
        )
        # 先按任务逐项追加，最后统一转换为 float NumPy 数组。
        values = {name: [] for name in names}
        for task_id in ready_ids:
            feasible_vms = self.get_feasible_vms(task_id)
            if not feasible_vms:
                raise NoFeasibleVMError(f"Task {task_id} has no feasible VM.")
            exec_times = [self.estimate_exec_time(task_id, vm_id) for vm_id in feasible_vms]
            comm_times = [self.estimate_comm_time(task_id, vm_id) for vm_id in feasible_vms]
            if self.fuzzy_enabled:
                energies = [
                    self.estimate_incremental_energy_score(task_id, vm_id)
                    for vm_id in feasible_vms
                ]
            else:
                energies = [
                    self.estimate_incremental_energy(task_id, vm_id)
                    for vm_id in feasible_vms
                ]
            risks = [self.calculate_uncertainty(task_id, vm_id) for vm_id in feasible_vms]

            values["min_exec_time"].append(min(exec_times))
            values["min_comm_time"].append(min(comm_times))
            values["min_incremental_energy"].append(min(energies))
            values["slack"].append(self.calculate_task_slack(task_id))
            values["upward_rank"].append(self.calculate_upward_rank(task_id))
            values["remaining_work"].append(self.calculate_remaining_work(task_id))
            # 等待时间从任务首次进入 ready 集的时间算起，并截断为非负数。
            values["ready_wait_time"].append(
                max(0.0, float(self.current_time) - float(self.task_ready_time[task_id]))
            )
            values["uncertainty"].append(min(risks))

        return {name: np.asarray(values[name], dtype=float) for name in names}

    def select_task_with_priority_rule(
        self,
        ready_tasks,
        priority_rule,
        *,
        return_details=False,
    ):
        """调用 LLM 生成规则并从 ready 集中选择唯一一个最小分数任务。

        调用顺序严格对应 Prompt 定义的八个参数。候选输出必须通过形状和有限性
        验证；``np.argmin`` 在分数完全相同时稳定选择 ready_tasks 中最靠前者。
        本函数不查询候选提供的 VM 信息，也不执行任何资源分配。
        """
        ready_ids = [self._task_id(task) for task in ready_tasks]
        if not ready_ids:
            raise ValueError("Cannot select a task from an empty ready set.")
        features = self.build_task_features(ready_ids)
        # 只把数值特征传给候选，隔离 Task/VM 可变对象和环境内部状态。
        scores = priority_rule(
            features["min_exec_time"],
            features["min_comm_time"],
            features["min_incremental_energy"],
            features["slack"],
            features["upward_rank"],
            features["remaining_work"],
            features["ready_wait_time"],
            features["uncertainty"],
        )
        scores = validate_task_priority_scores(scores, len(ready_ids))
        selected_index = int(np.argmin(scores))
        selected_task_id = int(ready_ids[selected_index])
        if not return_details:
            return selected_task_id
        order = np.lexsort(
            (np.arange(len(ready_ids), dtype=np.int64), scores)
        )
        return selected_task_id, {
            "selected_task_id": selected_task_id,
            "selected_index": selected_index,
            "ready_task_ids": list(ready_ids),
            "features": {
                name: np.asarray(values, dtype=float).copy()
                for name, values in features.items()
            },
            "scores": np.asarray(scores, dtype=float).copy(),
            "ranked_task_ids": [
                int(ready_ids[index]) for index in order
            ],
            "score_direction": "lower_is_higher_priority",
        }

    def idle_feasible_vm_ids(self, task):
        """返回当前时刻既可行又空闲的稳定 VM ID 列表。

        判据与比较算法使用的 ``global_vm_action_mask`` 完全一致：VM 必须通过
        ``get_feasible_vms`` 的处理能力/带宽校验，且 ``vm_available_at`` 不晚于
        当前时刻。该方法只读，不修改任何环境状态，也不选择任务、Host 或 VM。

        本接口供不允许 busy-VM 排队的调用方使用（CEWS 评价器、反事实估计器），
        避免它们各自重复实现“空闲且可行”的判定而产生口径漂移。
        """
        task_id = self._task_id(task)
        index_by_vm_id = {
            int(vm_id): index
            for index, vm_id in enumerate(self.vm_ids)
        }
        now = float(self.current_time)
        return [
            int(vm_id)
            for vm_id in self.get_feasible_vms(task_id)
            if float(self.vm_available_at[index_by_vm_id[int(vm_id)]])
            <= now + 1e-9
        ]

    def _score_vm_candidates(self, task_id, vm_ids, deadline):
        """为给定 VM 集合计算固定规则排序所需的全部预测量。

        由 ``select_vm_deterministic`` 与 ``select_host_then_vm_deterministic``
        共用，保证两者的候选打分逐位相同。本方法只预测，不修改环境状态。
        """
        # 保存完整预测详情，既供排序使用，也便于测试和实验诊断。
        candidates = []
        for vm_id in vm_ids:
            _, vm_index = self._vm_id_and_index(vm_id)
            exec_time = self.estimate_exec_time(task_id, vm_id)
            comm_time = self.estimate_comm_time(task_id, vm_id)
            queue_time = max(0.0, float(self.vm_available_at[vm_index]) - float(self.current_time))
            predicted_finish_time = (
                float(self.current_time) + queue_time + exec_time + comm_time
            )
            incremental_energy = self.estimate_incremental_energy(task_id, vm_id)
            candidate = {
                "vm_id": int(vm_id),
                "exec_time": float(exec_time),
                "comm_time": float(comm_time),
                "queue_time": float(queue_time),
                "predicted_finish_time": float(predicted_finish_time),
                "incremental_energy": float(incremental_energy),
            }
            if getattr(self, "fuzzy_enabled", False):
                finish_tfn = self.estimate_task_finish_tfn(task_id, vm_id)
                finish_risk = self.fuzzy_deadline_measure(finish_tfn)
                fuzzy_energy_score = self.estimate_incremental_energy_score(
                    task_id, vm_id
                )
                candidate.update({
                    "predicted_finish_optimistic": float(finish_tfn.lower),
                    "predicted_finish_modal": float(finish_tfn.modal),
                    "predicted_finish_pessimistic": float(finish_tfn.upper),
                    "fuzzy_finish_risk": float(finish_risk),
                    "fuzzy_incremental_energy_score": float(
                        fuzzy_energy_score
                    ),
                    "deadline_violation": float(
                        max(0.0, finish_risk - deadline)
                    ),
                })
            else:
                candidate["deadline_violation"] = float(
                    max(0.0, predicted_finish_time - deadline)
                )
            candidates.append(candidate)
        return candidates

    def _fixed_rule_group_and_keys(self, candidates, deadline):
        """执行“按时优先”硬分组，并返回该组对应的固定排序键顺序。

        先分组再排序，可确保“按时优先”是硬规则，而不是一个可被能耗抵消的权重。
        分组必须在全部候选上一次性完成：若按 Host 分别分组，某台 Host 内全部
        延期而全局仍存在按时 VM 时，结果会与全局固定规则不一致。
        """
        if getattr(self, "fuzzy_enabled", False):
            on_time = [
                item
                for item in candidates
                if item["fuzzy_finish_risk"] <= deadline
            ]
            if on_time:
                return on_time, (
                    "fuzzy_incremental_energy_score",
                    "fuzzy_finish_risk",
                    "predicted_finish_time",
                )
            return candidates, (
                "deadline_violation",
                "fuzzy_incremental_energy_score",
                "fuzzy_finish_risk",
                "predicted_finish_time",
            )
        on_time = [
            item
            for item in candidates
            if item["predicted_finish_time"] <= deadline
        ]
        if on_time:
            return on_time, (
                "incremental_energy",
                "predicted_finish_time",
            )
        return candidates, (
            "deadline_violation",
            "incremental_energy",
            "predicted_finish_time",
        )

    def _fixed_rule_candidate_set(self, task, candidate_vm_ids):
        """解析候选 VM、打分并完成全局分组，返回 ``(存活候选, 排序键)``。

        ``select_vm_deterministic`` 与 ``select_host_then_vm_deterministic``
        共用本方法，因而两者面对的候选集合、打分与分组完全一致。
        """
        task_id = self._task_id(task)
        feasible_vms = self.get_feasible_vms(task_id)
        if candidate_vm_ids is not None:
            allowed = {
                int(vm_id) for vm_id in candidate_vm_ids
            }
            feasible_vms = [
                vm_id for vm_id in feasible_vms
                if int(vm_id) in allowed
            ]
        if not feasible_vms:
            raise NoFeasibleVMError(f"Task {task_id} has no feasible VM.")

        deadline = self._task_deadline(task_id)
        candidates = self._score_vm_candidates(
            task_id,
            feasible_vms,
            deadline,
        )
        return self._fixed_rule_group_and_keys(candidates, deadline)

    def select_vm_deterministic(
        self,
        task,
        candidate_vm_ids=None,
    ):
        """对已选任务使用固定 deadline/energy 策略选择一个可行 VM。

        每台 VM 都计算 exec_time、comm_time、queue_time、predicted_finish_time、
        incremental_energy 和 deadline_violation。排序规则为：

        modal 模式：
        1. 若存在按时 VM：最小增量能耗 -> 最早完成 -> 最小 vm_id；
        2. 若全部延期：最小违反量 -> 最小增量能耗 -> 最早完成 -> 最小 vm_id。

        fuzzy 模式把“按时”改为 eta 风险完成时刻满足子截止期，并使用风险调整
        模糊边际能耗。两种模式均由环境统一执行，LLM 不能改变 VM 选择口径。

        最后使用 vm_id 可保证完全相同代价下结果仍确定，不依赖字典遍历偶然顺序。
        ``candidate_vm_ids`` 是向后兼容的可选候选子集，仅供 safety shield 在
        当前硬合法（空闲）VM 中调用同一固定规则；省略时 CEWS 原接口和语义不变。
        """
        survivors, order_keys = self._fixed_rule_candidate_set(
            task,
            candidate_vm_ids,
        )
        selected = select_vm_candidate_by_fixed_rule_order(
            survivors,
            order_keys,
        )
        return int(selected["vm_id"]), dict(selected)

    def select_host_then_vm_deterministic(
        self,
        task,
        candidate_vm_ids=None,
    ):
        """按 Manager -> Host -> VM 的层次形态执行同一条固定规则。

        返回 ``(host_id, vm_id, details)``。候选打分、按时/延期分组与排序键
        全部复用 ``select_vm_deterministic`` 的实现，随后把存活候选按 Host
        分桶：先在每台 Host 内选出最优 VM，再在这些 Host 最优之间选出胜者。

        因为固定排序键以 ``vm_id`` 兜底而构成全序，而全序上的 argmin 对划分
        可分解（``min`` 于并集等于各子集 ``min`` 的 ``min``），本方法返回的
        VM 与 ``select_vm_deterministic`` 在相同候选集合下逐位相同。层次形态
        只让 Host 这一层显式化——边际能耗本就定义在 Host 的功率曲线上——而不
        改变任何数值结果，也不引入新的 Host 级排序准则。

        没有空闲/可行 VM 的 Host 不会形成分桶，因而天然被排除在 Host 选择之外。
        """
        survivors, order_keys = self._fixed_rule_candidate_set(
            task,
            candidate_vm_ids,
        )

        candidates_by_host = {}
        for item in survivors:
            host_id = int(self.vms[int(item["vm_id"])].host_id)
            candidates_by_host.setdefault(host_id, []).append(item)

        host_best = []
        for host_id in sorted(candidates_by_host):
            best = dict(
                select_vm_candidate_by_fixed_rule_order(
                    candidates_by_host[host_id],
                    order_keys,
                )
            )
            best["host_id"] = int(host_id)
            best["host_candidate_count"] = len(
                candidates_by_host[host_id]
            )
            host_best.append(best)

        selected = dict(
            select_vm_candidate_by_fixed_rule_order(
                host_best,
                order_keys,
            )
        )
        selected["host_count"] = len(host_best)
        return (
            int(selected["host_id"]),
            int(selected["vm_id"]),
            selected,
        )

    def assign_task(self, task, vm):
        """把一个 ready task 真正分配给可行 VM，并返回分配前预测详情。

        这是 CEWS 建设式循环中实际修改环境状态的入口。执行前再次验证任务仍在
        ready 集、VM 仍可行，防止选择与分配之间状态变化造成非法动作。底层
        ``_assign_task_to_specific_vm`` 负责事件、VM 可用时刻、任务状态和能耗账本。
        """
        task_id = self._task_id(task)
        vm_id, vm_index = self._vm_id_and_index(vm)
        if self.task_state[task_id] != "Ready" or task_id not in self.ready_task_ids:
            raise ValueError(f"Task {task_id} is not currently ready.")
        if vm_id not in self.get_feasible_vms(task_id):
            raise NoFeasibleVMError(f"VM {vm_id} is infeasible for task {task_id}.")

        queue_time = max(0.0, float(self.vm_available_at[vm_index]) - float(self.current_time))
        exec_time = self.estimate_exec_time(task_id, vm_id)
        comm_time = self.estimate_comm_time(task_id, vm_id)
        predicted_finish_time = float(self.current_time) + queue_time + exec_time + comm_time
        incremental_energy = self.estimate_incremental_energy(task_id, vm_id)
        # 所有预测值必须在状态修改前计算，否则 vm_available_at 已更新会重复排队。
        self._assign_task_to_specific_vm(task_id, vm_index)
        return {
            "task_id": int(task_id),
            "vm_id": int(vm_id),
            "queue_time": float(queue_time),
            "exec_time": float(exec_time),
            "comm_time": float(comm_time),
            "predicted_finish_time": float(predicted_finish_time),
            "incremental_energy": float(incremental_energy),
        }

    def advance_to_next_event(self):
        """无 ready task 时推进至下一决策点或终止，并返回环境原有能耗奖励。

        下一事件可能是运行任务完成，也可能是新工作流到达。评价器不会在空 ready
        集上调用 LLM，而是持续推进事件，直到出现可分配任务或全部工作流完成。
        """
        return self._advance_until_decision_energy_only()

    def advance_to_next_resource_event(self):
        """即使 ready 集非空，也推进到下一次到达或任务完成事件。

        该公开问题级接口供不允许 busy-VM 排队的独立比较算法使用：当所有 VM
        均忙时，比较算法不能产生一个非法 RA transition，因此必须等待最近资源
        事件。方法只复用环境已有事件、三条影子时间线与能耗账本；它不选择任务、
        Host 或 VM。现有 ``advance_to_next_event`` 及主算法调用语义保持不变。
        """
        if self.done_flag:
            return 0.0

        next_times = []
        if self.event_heap:
            next_times.append(float(self.event_heap[0][0]))
        if not self._no_more_arrivals():
            next_times.append(
                float(self.arrival_times[self.next_arrival_idx])
            )
        if not next_times:
            if self._episode_should_end():
                self.done_flag = True
                return 0.0
            raise RuntimeError(
                "No future resource or arrival event is available while "
                "the episode is not complete."
            )

        next_time = min(next_times)
        if next_time <= self.current_time + 1e-12:
            next_time = self.current_time + 1e-9
        self.current_time = float(next_time)
        reward = float(self._energy_reward_to_current_time())
        self._process_finish_events_at_current_time()
        self._add_workflow_if_arrived()
        if self._episode_should_end() or self.current_time >= self.horizon:
            self.done_flag = True
        return reward

    def get_task_safety_action_masks(self, task) -> dict:
        """构造当前任务的全局 VM 与 Host 三类动作掩码。

        VM safety mask 直接来自阶段 3 的任务级风险预测；Host safety mask 只在
        Host 内至少存在一个“硬合法且预测安全”的 VM 时为 1。该接口只读，不
        修改动作、任务状态、时间线或 Manager 权重。
        """
        task_id = self._task_id(task)
        feasible_vm_ids = set(self.get_feasible_vms(task_id))
        global_vm_legal = np.asarray(
            [
                1.0
                if (
                    vm_id in feasible_vm_ids
                    and self.vm_available_at[vm_index]
                    <= self.current_time + 1e-9
                )
                else 0.0
                for vm_index, vm_id in enumerate(self.vm_ids)
            ],
            dtype=np.float32,
        )
        host_legal = np.asarray(
            [
                1.0
                if any(
                    global_vm_legal[vm_index] > 0.5
                    for vm_index in self.host_to_vm_indices[host_id]
                )
                else 0.0
                for host_id in self.host_ids
            ],
            dtype=np.float32,
        )

        if getattr(self, "safe_rl_enabled", False):
            prediction = self.get_task_action_risk_predictions(task_id)
            vm_predictions = list(prediction["vm_predictions"])
        else:
            prediction = {
                "safe_rl_enabled": False,
                "prediction_available": False,
                "vm_predictions": [],
                "host_predictions": [],
            }
            vm_predictions = []

        if vm_predictions:
            vm_masks = self.safety_shield.build_vm_masks(
                global_vm_legal,
                self.vm_ids,
                vm_predictions,
            )
        else:
            # 未启用 safe_rl 时不运行模糊预测；全 1 safety mask 表示没有附加
            # DDL 限制，final mask 因而严格退化为旧 legal mask。
            vm_masks = self.safety_shield.combine_masks(
                global_vm_legal,
                np.ones_like(global_vm_legal),
            )
            vm_masks["action_vm_ids"] = [
                int(vm_id) for vm_id in self.vm_ids
            ]

        safe_legal_global_vm = vm_masks[
            "safe_legal_action_mask"
        ]
        safe_vm_count_by_host = [
            sum(
                safe_legal_global_vm[vm_index] > 0.5
                for vm_index in self.host_to_vm_indices[host_id]
            )
            for host_id in self.host_ids
        ]
        host_masks = self.safety_shield.build_host_masks(
            host_legal,
            safe_vm_count_by_host,
        )

        vm_prediction_by_id = {
            int(row["vm_id"]): row for row in vm_predictions
        }
        global_vm_metrics = []
        for vm_id in self.vm_ids:
            row = vm_prediction_by_id.get(int(vm_id), {})
            global_vm_metrics.append(
                {
                    "predicted_risk": float(
                        row.get("risk_finish", 0.0)
                    ),
                    "safety_margin": float(
                        row.get("safety_margin", 0.0)
                    ),
                    "predicted_violation_amount": float(
                        row.get(
                            "predicted_violation_amount",
                            0.0,
                        )
                    ),
                }
            )

        host_prediction_by_id = {
            int(row["host_id"]): row
            for row in prediction.get("host_predictions", [])
        }
        host_metrics = []
        for host_id in self.host_ids:
            row = host_prediction_by_id.get(int(host_id), {})
            host_metrics.append(
                {
                    "predicted_risk": float(
                        row.get("minimum_risk_finish", 0.0)
                    ),
                    "safety_margin": float(
                        row.get("maximum_safety_margin", 0.0)
                    ),
                    "predicted_violation_amount": float(
                        row.get(
                            "minimum_predicted_violation_amount",
                            0.0,
                        )
                    ),
                }
            )

        fallback_candidates = []
        if (
            self.safe_rl_shield_enabled
            and host_masks["safe_action_count"] == 0
        ):
            # 只在真实空安全集时计算模糊边际能耗，避免正常安全动作路径引入
            # 额外排序或改变 RL 的选择语义。
            for vm_index, vm_id in enumerate(self.vm_ids):
                if global_vm_legal[vm_index] <= 0.5:
                    continue
                row = vm_prediction_by_id.get(int(vm_id))
                if row is None:
                    continue
                host_id = int(self.vms[int(vm_id)].host_id)
                fallback_candidates.append(
                    {
                        "vm_id": int(vm_id),
                        "host_id": host_id,
                        "vm_global_index": int(vm_index),
                        "host_action": int(
                            self.host_ids.index(host_id)
                        ),
                        "predicted_violation_amount": float(
                            row["predicted_violation_amount"]
                        ),
                        "fuzzy_marginal_energy": float(
                            self.estimate_incremental_energy_score(
                                task_id,
                                vm_id,
                            )
                        ),
                        "risk_finish": float(row["risk_finish"]),
                    }
                )

        fallback_record = self.safety_fallback_controller.select(
            fallback_candidates,
            safe_action_count=int(host_masks["safe_action_count"]),
            fallback_reason="empty_safe_action_set",
        )
        fallback_vm_id = fallback_record["selected_vm"]
        fallback_vm_global_index = None
        fallback_host_action = None
        if fallback_record["fallback_triggered"]:
            selected_candidate = fallback_record[
                "selected_candidate"
            ]
            fallback_vm_global_index = int(
                selected_candidate["vm_global_index"]
            )
            fallback_host_action = int(
                selected_candidate["host_action"]
            )

        return {
            "task_id": int(task_id),
            "safety_shield_enabled": bool(
                self.safe_rl_shield_enabled
            ),
            "prediction": prediction,
            "vm_masks_global": vm_masks,
            "host_masks": host_masks,
            "global_vm_metrics": global_vm_metrics,
            "host_metrics": host_metrics,
            "fallback_vm_id": fallback_vm_id,
            "fallback_vm_global_index": fallback_vm_global_index,
            "fallback_host_action": fallback_host_action,
            "fallback_record": dict(fallback_record),
            "fallback_required": bool(
                fallback_record["fallback_triggered"]
            ),
        }

    def _cross_platform_communication_risk(
        self,
        task_id: int,
        candidate_host_id: int,
    ) -> float:
        workflow_id, local_id = self.task_meta[int(task_id)]
        task_obj = self.workflows[workflow_id].tasks[local_id]
        parent_bits = dict(
            getattr(task_obj, "parent_in_bits", {}) or {}
        )
        total_bits = float(sum(parent_bits.values()))
        if total_bits <= 0.0:
            return 0.0

        candidate_type = str(
            self.hosts[int(candidate_host_id)].server_type
        ).lower()
        cross_platform_bits = 0.0
        for parent_id in self.task_global_parents[int(task_id)]:
            parent_local_id = int(self.task_meta[parent_id][1])
            bits = float(
                parent_bits.get(
                    parent_local_id,
                    parent_bits.get(str(parent_local_id), 0.0),
                )
            )
            parent_host_id = self.task_assigned_host.get(
                int(parent_id)
            )
            if parent_host_id is None:
                continue
            parent_type = str(
                self.hosts[int(parent_host_id)].server_type
            ).lower()
            if parent_type != candidate_type:
                cross_platform_bits += bits
        return _clip01(cross_platform_bits / total_bits)

    def _build_host_safety_observation(
        self,
        task_id: int,
        context,
    ) -> np.ndarray:
        prediction = context["prediction"]
        host_prediction_by_id = {
            int(row["host_id"]): row
            for row in prediction["host_predictions"]
        }
        budget = self._workflow_budget_for_task(task_id)
        remaining_risk = float(
            prediction["remaining_critical_path"][
                "remaining_critical_path_risk"
            ]
        )
        critical_path_pressure = _clip01(
            remaining_risk / budget
        )
        now = float(self.current_time)
        rows = []
        for host_id in self.host_ids:
            host_prediction = host_prediction_by_id.get(
                int(host_id),
                {},
            )
            safe_count = int(
                host_prediction.get("safe_vm_count", 0)
            )
            selectable_count = int(
                host_prediction.get("candidate_vm_count", 0)
            )
            vm_indices = self.host_to_vm_indices[host_id]
            queue_delays = [
                max(0.0, float(self.vm_available_at[index]) - now)
                for index in vm_indices
            ]
            queue_risk = _clip01(
                (
                    float(np.mean(queue_delays))
                    if queue_delays
                    else 0.0
                )
                / max(self.horizon, 1e-9)
            )
            selectable_predictions = list(
                host_prediction.get(
                    "candidate_vm_predictions",
                    [],
                )
            )
            minimum_margin = (
                min(
                    float(row["safety_margin"])
                    for row in selectable_predictions
                )
                if selectable_predictions
                else 0.0
            )
            busy_count = sum(
                self.vm_available_at[index] > now + 1e-9
                for index in vm_indices
            )
            projected_busy_ratio = _clip01(
                (busy_count + 1.0) / max(len(vm_indices), 1)
            )
            values = [
                _clip01(
                    safe_count / max(self.max_vms_per_host, 1)
                ),
                _clip01(
                    safe_count / max(selectable_count, 1)
                    if selectable_count > 0
                    else 0.0
                ),
                queue_risk,
                float(
                    np.clip(minimum_margin / budget, -1.0, 1.0)
                ),
                self._cross_platform_communication_risk(
                    task_id,
                    host_id,
                ),
                _clip01(
                    projected_busy_ratio
                    * critical_path_pressure
                ),
            ]
            rows.append(
                self._validated_safety_feature_vector(
                    values,
                    HOST_SAFETY_FEATURE_SCHEMA,
                    context=f"Host action slot {len(rows)}",
                )
            )
        return np.concatenate(rows).astype(np.float32)

    @staticmethod
    def _relative_fuzzy_resource_uncertainty(
        fuzzy_value,
    ) -> float:
        return _clip01(
            (
                float(fuzzy_value.upper)
                - float(fuzzy_value.lower)
            )
            / max(float(fuzzy_value.modal), 1e-9)
        )

    def _build_vm_safety_observation(
        self,
        task_id: int,
        context,
    ) -> np.ndarray:
        predictions_by_vm = {
            int(row["vm_id"]): row
            for row in context["prediction"]["vm_predictions"]
        }
        now = float(self.current_time)
        budget = self._workflow_budget_for_task(task_id)
        horizon = max(float(self.horizon), 1e-9)
        energy_reference = max(
            float(self.task_mi[int(task_id)])
            * float(self.energy_norm_per_mi_ref),
            1e-9,
        )
        rows = []
        for vm_id in context["current_action_vm_ids"]:
            if vm_id is None:
                rows.append(
                    np.zeros(
                        self.vm_safety_feature_dim,
                        dtype=np.float32,
                    )
                )
                continue
            prediction = predictions_by_vm.get(int(vm_id))
            if prediction is None:
                rows.append(
                    np.zeros(
                        self.vm_safety_feature_dim,
                        dtype=np.float32,
                    )
                )
                continue
            vm_index = self.vm_ids.index(int(vm_id))
            vm = self.vms[int(vm_id)]
            energy = max(
                0.0,
                float(
                    self.estimate_incremental_energy_score(
                        task_id,
                        vm_id,
                    )
                ),
            )
            margin = float(prediction["safety_margin"])

            def finish_distance(value):
                return _clip01(
                    max(0.0, float(value) - now) / budget
                )

            values = [
                finish_distance(prediction["optimistic_finish"]),
                finish_distance(prediction["modal_finish"]),
                finish_distance(prediction["pessimistic_finish"]),
                finish_distance(prediction["risk_finish"]),
                float(
                    np.clip(
                        (
                            float(prediction["task_safe_deadline"])
                            - now
                        )
                        / budget,
                        -1.0,
                        1.0,
                    )
                ),
                float(np.clip(margin / horizon, -1.0, 1.0)),
                float(np.clip(margin / budget, -1.0, 1.0)),
                _clip01(energy / (energy + energy_reference)),
                _clip01(
                    max(
                        0.0,
                        float(self.vm_available_at[vm_index]) - now,
                    )
                    / horizon
                ),
                self._relative_fuzzy_resource_uncertainty(
                    vm.pc
                ),
                self._relative_fuzzy_resource_uncertainty(
                    vm.bw
                ),
                _clip01(
                    float(
                        prediction[
                            "remaining_critical_path_risk"
                        ]
                    )
                    / budget
                ),
            ]
            rows.append(
                self._validated_safety_feature_vector(
                    values,
                    VM_SAFETY_FEATURE_SCHEMA,
                    context=f"VM action slot {len(rows)}",
                )
            )
        return np.concatenate(rows).astype(np.float32)

    @staticmethod
    def _policy_mask_from_shield_bundle(mask_bundle) -> np.ndarray:
        """安全集合为空时返回 legal mask，让旧 Agent 产生可审计的提议。

        环境随后不会直接执行该提议，而是由 shield 调用固定 VM 回退控制器。
        真实空安全集合始终保存在 ``final_action_mask``，不会被伪装成安全动作。
        """
        final_mask = np.asarray(
            mask_bundle["final_action_mask"],
            dtype=np.float32,
        )
        if np.sum(final_mask) > 0.0:
            return final_mask.copy()
        return np.asarray(
            mask_bundle["legal_action_mask"],
            dtype=np.float32,
        ).copy()

    def _layer_state_with_safety_masks(
        self,
        obs,
        mask_bundle,
    ) -> dict:
        policy_mask = self._policy_mask_from_shield_bundle(
            mask_bundle
        )
        return {
            "obs": np.asarray(obs, dtype=np.float32),
            # 兼容字段：安全集合非空时为 final；空集合时为 legal，以便取得
            # RL proposed action 后显式进入确定性回退。
            "mask": policy_mask,
            "policy_action_mask": policy_mask.copy(),
            "legal_action_mask": np.asarray(
                mask_bundle["legal_action_mask"],
                dtype=np.float32,
            ).copy(),
            "safety_action_mask": np.asarray(
                mask_bundle["safety_action_mask"],
                dtype=np.float32,
            ).copy(),
            "final_action_mask": np.asarray(
                mask_bundle["final_action_mask"],
                dtype=np.float32,
            ).copy(),
            "safety_fallback_required": bool(
                self.safe_rl_shield_enabled
                and np.sum(mask_bundle["final_action_mask"]) <= 0.0
            ),
            "safety_shield_enabled": bool(
                self.safe_rl_shield_enabled
            ),
            "safe_state_enabled": bool(
                self.safe_rl_state_enabled
            ),
            "observation_schema_version": (
                SAFE_OBSERVATION_SCHEMA_VERSION
                if self.safe_rl_state_enabled
                else "legacy_observation"
            ),
        }

    def _record_safety_shield_decision(self, decision) -> dict:
        record = {
            **dict(decision),
            "task_id": (
                None
                if self._cur_tid is None
                else int(self._cur_tid)
            ),
            "current_time": float(self.current_time),
        }
        self._safety_shield_records.append(record)
        self._phase_safety_shield_records.append(record)
        return record

    def _attach_action_selection(
        self,
        decision: dict,
        action_selection=None,
    ) -> dict:
        """把策略探索类型与 shield 最终执行结果合并到环境记录。"""
        selection = dict(action_selection or {})
        selected_by_agent = bool(
            selection.get("selected_by_agent", True)
        )
        proposed_action = (
            selection.get(
                "proposed_action",
                decision.get("rl_proposed_action"),
            )
            if selected_by_agent
            else None
        )
        policy_type = str(
            selection.get(
                "policy_selection_type",
                selection.get(
                    "selection_type",
                    "unclassified_rl_action",
                ),
            )
        )
        fallback = bool(
            decision.get("fallback_applied", False)
            or not selected_by_agent
        )
        modified = bool(
            decision.get("action_modified", False)
            or (
                proposed_action is not None
                and int(proposed_action)
                != int(decision["executed_action"])
            )
        )
        if fallback:
            action_source = "fallback_action"
        elif (
            modified
            or bool(decision.get("shield_intervened", False))
        ):
            action_source = "shield_correction"
        else:
            action_source = policy_type
        decision.update(
            {
                "proposed_action": (
                    int(proposed_action)
                    if proposed_action is not None
                    else None
                ),
                "selected_by_agent": selected_by_agent,
                "policy_selection_type": policy_type,
                "action_source": action_source,
                "selection_epsilon": selection.get("epsilon"),
                "action_modified": modified,
            }
        )
        return decision

    def _attach_fallback_record(
        self,
        decision: dict,
        context,
    ) -> dict:
        fallback_record = dict(
            (context or {}).get("fallback_record", {})
        )
        decision["fallback_controller_record"] = fallback_record
        decision.update(
            self.safety_fallback_controller.record_fields(
                fallback_record
            )
        )
        return decision

    def get_safety_shield_records(self) -> list[dict]:
        """返回 episode 内 shield 决策记录的浅拷贝。"""
        return [dict(record) for record in self._safety_shield_records]

    def get_safety_shield_diagnostics(self) -> dict:
        """返回 episode 内 shield 干预与回退聚合，不改变 Manager 动作。"""
        return self._safety_shield_diagnostics()

    def _safety_shield_diagnostics(self, records=None) -> dict:
        selected_records = (
            self._safety_shield_records
            if records is None
            else list(records)
        )
        return {
            "safety_shield_enabled": bool(
                self.safe_rl_shield_enabled
            ),
            "shield_record_count": int(len(selected_records)),
            "shield_intervention_count": int(
                sum(
                    bool(record.get("shield_intervened", False))
                    for record in selected_records
                )
            ),
            "shield_fallback_count": int(
                sum(
                    bool(record.get("fallback_applied", False))
                    for record in selected_records
                )
            ),
            "host_action_modified_count": int(
                sum(
                    record.get("layer") == "host"
                    and bool(record.get("action_modified", False))
                    for record in selected_records
                )
            ),
            "vm_action_modified_count": int(
                sum(
                    record.get("layer") == "vm"
                    and bool(record.get("action_modified", False))
                    for record in selected_records
                )
            ),
            "random_safe_exploration_count": int(
                sum(
                    record.get("action_source")
                    == "random_safe_exploration"
                    for record in selected_records
                )
            ),
            "greedy_safe_action_count": int(
                sum(
                    record.get("action_source")
                    == "greedy_safe_action"
                    for record in selected_records
                )
            ),
            "shield_correction_count": int(
                sum(
                    record.get("action_source")
                    == "shield_correction"
                    for record in selected_records
                )
            ),
            "fallback_action_count": int(
                sum(
                    record.get("action_source")
                    == "fallback_action"
                    for record in selected_records
                )
            ),
        }

    def _attach_safety_shield_info(
        self,
        info: dict,
        vm_decision=None,
    ) -> dict:
        host_decision = dict(
            self._current_host_shield_decision or {}
        )
        vm_decision = dict(vm_decision or {})
        active = vm_decision or host_decision
        info.update(
            {
                "safety_shield_enabled": bool(
                    self.safe_rl_shield_enabled
                ),
                "manager_phase_id": int(
                    getattr(self, "_manager_phase_id", 0)
                ),
                "host_shield_decision": host_decision,
                "vm_shield_decision": vm_decision,
                "rl_proposed_action": active.get(
                    "rl_proposed_action"
                ),
                "proposed_action": active.get(
                    "proposed_action",
                    active.get("rl_proposed_action"),
                ),
                "executed_action": active.get("executed_action"),
                "action_source": active.get(
                    "action_source",
                    "unclassified_action",
                ),
                "policy_selection_type": active.get(
                    "policy_selection_type",
                    "unclassified_action",
                ),
                "selected_by_agent": bool(
                    active.get("selected_by_agent", True)
                ),
                "action_modified": bool(
                    active.get("action_modified", False)
                ),
                "shield_intervened": bool(
                    active.get("shield_intervened", False)
                ),
                "fallback_applied": bool(
                    active.get("fallback_applied", False)
                ),
                "fallback_triggered": bool(
                    active.get("fallback_triggered", False)
                ),
                "fallback_reason": active.get(
                    "fallback_reason",
                    "fallback_not_triggered",
                ),
                "candidate_count": int(
                    active.get("candidate_count", 0) or 0
                ),
                "minimum_violation": float(
                    active.get("minimum_violation", 0.0) or 0.0
                ),
                "selected_host": active.get("selected_host"),
                "selected_vm": active.get("selected_vm"),
                "tie_break_stage": active.get(
                    "tie_break_stage",
                    "not_triggered",
                ),
                "modification_reason": active.get(
                    "modification_reason",
                    "shield_not_applied",
                ),
                "predicted_risk": float(
                    active.get("predicted_risk", 0.0)
                ),
                "safety_margin": float(
                    active.get("safety_margin", 0.0)
                ),
                "legal_action_mask": list(
                    active.get("legal_action_mask", [])
                ),
                "safety_action_mask": list(
                    active.get("safety_action_mask", [])
                ),
                "final_action_mask": list(
                    active.get("final_action_mask", [])
                ),
            }
        )
        info.update(
            self._manager_heuristic_audit_info(
                self._phase_safety_shield_records
            )
        )
        return info

    def get_host_state_for_next_assignment(self):
        """取出阶段内下一个任务并构造 HostAgent 状态与动作掩码"""
        if self.done_flag:
            return {
                "obs": np.zeros(self.host_obs_dim, np.float32),
                "mask": np.zeros(self.host_act_dim, np.float32),
            }, False

        if not self._phase_started:
            self._phase_prepare_tasks()
            self._phase_started = True
            self._phase_assign_cnt = 0
            self._phase_assigned_tids = []
            self._phase_size_sum_mi = 0.0

            self._phase_preadvanced = False
            self._phase_preadvance_r_energy_total = 0.0
            self._phase_preadvance_r_energy_by_host = {int(h): 0.0 for h in self.host_ids}
            self._phase_performance_reward = 0.0
            self._phase_performance_energy_delta = 0.0
            self._phase_heuristic_safety_cost = 0.0
            self._phase_safety_shield_records = []

        if not self._has_decision_point():
            return {
                "obs": np.zeros(self.host_obs_dim, np.float32),
                "mask": np.zeros(self.host_act_dim, np.float32),
            }, False

        if len(self._phase_tasks) == 0:
            return {
                "obs": np.zeros(self.host_obs_dim, np.float32),
                "mask": np.zeros(self.host_act_dim, np.float32),
            }, False

        self._cur_tid = int(self._phase_tasks.pop(0))
        self._cur_host_id = None
        self._current_host_shield_decision = None

        obs, legal_mask = self._build_host_obs_for_task(self._cur_tid)
        if np.sum(legal_mask) <= 0.0:
            self._phase_tasks.insert(0, self._cur_tid)
            self._cur_tid = None
            return {
                "obs": np.zeros(self.host_obs_dim, np.float32),
                "mask": np.zeros(self.host_act_dim, np.float32),
            }, False

        self._current_safety_shield_context = (
            self.get_task_safety_action_masks(self._cur_tid)
        )
        host_masks = self._current_safety_shield_context[
            "host_masks"
        ]
        # 防御性确认 shield 的 legal 口径与原 Host mask 完全一致。
        if not np.array_equal(
            np.asarray(host_masks["legal_action_mask"]),
            np.asarray(legal_mask, dtype=np.float32),
        ):
            raise RuntimeError(
                "Host legal mask diverged from safety shield context"
            )
        if self.safe_rl_state_enabled:
            host_safety_obs = (
                self._build_host_safety_observation(
                    self._cur_tid,
                    self._current_safety_shield_context,
                )
            )
            obs = np.concatenate(
                [obs, host_safety_obs],
                axis=0,
            ).astype(np.float32)
        if int(obs.size) != int(self.host_obs_dim):
            raise RuntimeError(
                "Host observation dimension does not match schema"
            )
        state = self._layer_state_with_safety_masks(
            obs,
            host_masks,
        )
        state["fallback_action"] = (
            None
            if self._current_safety_shield_context[
                "fallback_host_action"
            ]
            is None
            else int(
                self._current_safety_shield_context[
                    "fallback_host_action"
                ]
            )
        )
        return state, True

    def host_select(
        self,
        host_index: int,
        action_selection=None,
    ):
        """记录 HostAgent 选择并校验目标主机是否存在空闲 VM"""
        assert hasattr(self, "_cur_tid") and self._cur_tid is not None, "请先调用 get_host_state_for_next_assignment()"
        proposed_hi = int(host_index)
        context = self._current_safety_shield_context
        if context is None:
            raise RuntimeError(
                "Missing safety shield context; call "
                "get_host_state_for_next_assignment() first"
            )
        decision = self.safety_shield.resolve_action(
            proposed_hi,
            context["host_masks"],
            action_metrics=context["host_metrics"],
            fallback_action=context["fallback_host_action"],
            layer="host",
        )
        self._attach_fallback_record(decision, context)
        self._attach_action_selection(
            decision,
            action_selection,
        )
        hi = int(decision["executed_action"])
        if hi < 0 or hi >= self.num_hosts:
            raise ValueError(f"host_index out of range: {hi}")
        host_id = self.host_ids[hi]

        has_idle = False
        now = self.current_time
        for j in self.host_to_vm_indices[host_id]:
            if self.vm_available_at[j] <= now + 1e-9:
                has_idle = True
                break
        if not has_idle:
            raise ValueError(f"Selected host {host_id} has no idle VM at time {now:.6f}")

        self._cur_host_id = host_id
        decision.update(
            {
                "executed_host_id": int(host_id),
                "rl_proposed_host_action": int(proposed_hi),
                "executed_host_action": int(hi),
            }
        )
        if self.safe_rl_enabled:
            decision = self._record_safety_shield_decision(
                decision
            )
        self._current_host_shield_decision = dict(decision)

    def get_vm_state_for_current_task(self):
        """为当前任务和已选主机构造 VMAgent 状态与动作掩码"""
        if self.done_flag:
            return {
                "obs": np.zeros(self.vm_obs_dim, np.float32),
                "mask": np.zeros(self.vm_act_dim, np.float32),
            }, False

        if not hasattr(self, "_cur_tid") or self._cur_tid is None:
            return {
                "obs": np.zeros(self.vm_obs_dim, np.float32),
                "mask": np.zeros(self.vm_act_dim, np.float32),
            }, False

        if self._cur_host_id is None:
            return {
                "obs": np.zeros(self.vm_obs_dim, np.float32),
                "mask": np.zeros(self.vm_act_dim, np.float32),
            }, False

        obs, legal_mask = self._build_vm_obs_for_task_host(
            self._cur_tid,
            self._cur_host_id,
        )
        if np.sum(legal_mask) <= 0.0:
            return {
                "obs": np.zeros(self.vm_obs_dim, np.float32),
                "mask": np.zeros(self.vm_act_dim, np.float32),
            }, False

        context = self._current_safety_shield_context
        if context is None:
            raise RuntimeError("Missing safety shield context")
        vm_indices = self.host_to_vm_indices[self._cur_host_id]
        action_vm_ids = [
            (
                int(self.vm_ids[vm_indices[slot]])
                if slot < len(vm_indices)
                else None
            )
            for slot in range(self.vm_act_dim)
        ]
        vm_predictions = context["prediction"].get(
            "vm_predictions",
            [],
        )
        if vm_predictions:
            vm_masks = self.safety_shield.build_vm_masks(
                legal_mask,
                action_vm_ids,
                vm_predictions,
            )
        else:
            vm_masks = self.safety_shield.combine_masks(
                legal_mask,
                np.ones_like(legal_mask),
            )
            vm_masks["action_vm_ids"] = action_vm_ids

        fallback_vm_slot = None
        fallback_global_index = context[
            "fallback_vm_global_index"
        ]
        if (
            fallback_global_index is not None
            and int(fallback_global_index) in vm_indices
        ):
            fallback_vm_slot = vm_indices.index(
                int(fallback_global_index)
            )
        local_metrics = [
            (
                context["global_vm_metrics"][vm_indices[slot]]
                if slot < len(vm_indices)
                else {}
            )
            for slot in range(self.vm_act_dim)
        ]
        context["current_vm_masks"] = vm_masks
        context["current_vm_metrics"] = local_metrics
        context["fallback_vm_slot"] = fallback_vm_slot
        context["current_action_vm_ids"] = action_vm_ids

        if self.safe_rl_state_enabled:
            vm_safety_obs = self._build_vm_safety_observation(
                self._cur_tid,
                context,
            )
            obs = np.concatenate(
                [obs, vm_safety_obs],
                axis=0,
            ).astype(np.float32)
        if int(obs.size) != int(self.vm_obs_dim):
            raise RuntimeError(
                "VM observation dimension does not match schema"
            )
        state = self._layer_state_with_safety_masks(
            obs,
            vm_masks,
        )
        state["fallback_action"] = (
            None
            if fallback_vm_slot is None
            else int(fallback_vm_slot)
        )
        return state, True

    def _phase_has_immediate_next_assignment(self) -> bool:
        """判断当前阶段是否还能在不推进时间的情况下继续分配"""
        if self.done_flag:
            return False
        if len(self._phase_tasks) == 0:
            return False
        return bool(self._has_decision_point())

    def vm_assign(
        self,
        vm_slot_index: int,
        action_selection=None,
    ):
        """将当前任务分配到指定 VM 槽位并计算 Host 与 VM 奖励"""
        assert hasattr(self, "_cur_tid") and self._cur_tid is not None, "请先调用 get_host_state_for_next_assignment()"
        assert self._cur_host_id is not None, "请先调用 host_select()"

        tid = int(self._cur_tid)
        host_id = int(self._cur_host_id)

        proposed_slot = int(vm_slot_index)
        context = self._current_safety_shield_context
        if context is None:
            raise RuntimeError("Missing safety shield context")
        if "current_vm_masks" not in context:
            _, vm_state_available = (
                self.get_vm_state_for_current_task()
            )
            if not vm_state_available:
                raise RuntimeError(
                    "Current host has no hard-legal VM action"
                )
        vm_decision = self.safety_shield.resolve_action(
            proposed_slot,
            context["current_vm_masks"],
            action_metrics=context["current_vm_metrics"],
            fallback_action=context["fallback_vm_slot"],
            layer="vm",
        )
        self._attach_fallback_record(vm_decision, context)
        self._attach_action_selection(
            vm_decision,
            action_selection,
        )
        slot = int(vm_decision["executed_action"])
        vm_decision.update(
            {
                "rl_proposed_vm_action": int(proposed_slot),
                "executed_vm_action": int(slot),
            }
        )
        if self.safe_rl_enabled:
            vm_decision = self._record_safety_shield_decision(
                vm_decision
            )
        if slot < 0 or slot >= self.max_vms_per_host:
            r_host = -1.0
            r_vm = -1.0
            info = {
                "invalid": 1,
                "reason": "vm_slot_out_of_range",
                "performance_reward": 0.0,
                "performance_reward_host": 0.0,
                "performance_reward_vm": 0.0,
                "hard_constraint_violation": 1,
            }
            info.update(
                self._empty_safe_performance_reward_info()
            )
            info.update(self._empty_safety_info())
            self._attach_safety_shield_info(
                info,
                vm_decision,
            )
            self._cur_tid = None
            self._cur_host_id = None
            self._current_safety_shield_context = None
            return float(r_host), float(r_vm), info

        vm_list = self.host_to_vm_indices[host_id]
        if slot >= len(vm_list):
            r_host = -1.0
            r_vm = -1.0
            info = {
                "invalid": 1,
                "reason": "vm_slot_padding",
                "performance_reward": 0.0,
                "performance_reward_host": 0.0,
                "performance_reward_vm": 0.0,
                "hard_constraint_violation": 1,
            }
            info.update(
                self._empty_safe_performance_reward_info()
            )
            info.update(self._empty_safety_info())
            self._attach_safety_shield_info(
                info,
                vm_decision,
            )
            self._cur_tid = None
            self._cur_host_id = None
            self._current_safety_shield_context = None
            return float(r_host), float(r_vm), info

        vm_global_idx = int(vm_list[slot])

        now = self.current_time
        if self.vm_available_at[vm_global_idx] > now + 1e-9:
            r_host = -1.0
            r_vm = -1.0
            info = {
                "invalid": 1,
                "reason": "vm_not_idle",
                "performance_reward": 0.0,
                "performance_reward_host": 0.0,
                "performance_reward_vm": 0.0,
                "hard_constraint_violation": 1,
            }
            info.update(
                self._empty_safe_performance_reward_info()
            )
            info.update(self._empty_safety_info())
            self._attach_safety_shield_info(
                info,
                vm_decision,
            )
            self._cur_tid = None
            self._cur_host_id = None
            self._current_safety_shield_context = None
            return float(r_host), float(r_vm), info

        vm_id = self.vm_ids[vm_global_idx]
        vm_decision["executed_vm_id"] = int(vm_id)
        bw_bps = self.vms[vm_id].bw * 1e6
        pc_mi_s = self.vms[vm_id].pc

        in_bits = self.task_in_bits[tid]
        out_bits = self.task_out_bits[tid]
        mi = self.task_mi[tid]

        t_upload = in_bits / max(bw_bps, 1e-9)
        t_compute = mi / max(pc_mi_s, 1e-9)
        t_download = out_bits / max(bw_bps, 1e-9)
        duration = float(t_upload + t_compute + t_download)

        start_time = max(now, float(self.vm_available_at[vm_global_idx]))
        finish_time = start_time + duration

        if tid < len(self.task_baseline_finish):
            deadline = float(self.task_baseline_finish[tid])
        else:
            deadline = float(finish_time)
        lateness = max(0.0, float(finish_time - deadline))
        delay_norm = _clip01(lateness / max(self.task_baseline_norm, 1e-9))
        r_delay = -float(delay_norm)

        energy_before_action = (
            self.get_fuzzy_energy_summary()
            if self.safe_rl_enabled
            else None
        )
        self._assign_task_to_specific_vm(tid, vm_global_idx)
        performance_breakdown = (
            self._safe_performance_reward_breakdown(
                energy_before_action
            )
        )
        performance_reward = float(
            performance_breakdown["total_performance_reward"]
        )
        performance_energy_delta = float(
            performance_breakdown["performance_energy_delta"]
        )
        performance_energy_score = float(
            performance_breakdown["performance_energy_score"]
        )
        if self.safe_rl_enabled:
            self._phase_performance_reward += float(performance_reward)
            self._phase_performance_energy_delta += float(
                performance_energy_delta
            )

        self._phase_assign_cnt += 1
        self._phase_assigned_tids.append(tid)
        self._phase_size_sum_mi += float(mi)

        if self._phase_has_immediate_next_assignment():
            r_energy_total = 0.0
            r_energy_host = 0.0
        else:
            r_energy_total, r_energy_by_host = self._advance_until_decision_energy_only(return_breakdown=True)
            r_energy_host = float(r_energy_by_host.get(host_id, 0.0))

            self._phase_preadvanced = True
            self._phase_preadvance_r_energy_total = float(r_energy_total)
            self._phase_preadvance_r_energy_by_host = {
                int(h): float(r_energy_by_host.get(h, 0.0)) for h in self.host_ids
            }

        r_host = (1.0 - float(self.alpha_delay_host)) * float(r_energy_total) + float(self.alpha_delay_host) * float(r_delay)
        r_vm = (1.0 - float(self.alpha_delay_vm)) * float(r_energy_host) + float(self.alpha_delay_vm) * float(r_delay)

        info = {
            "invalid": 0,
            "task_id": tid,
            "host_id": host_id,
            "vm_global_idx": vm_global_idx,
            "start_time": float(start_time),
            "finish_time": float(finish_time),
            "deadline": float(deadline),
            "lateness": float(lateness),
            "delay_norm": float(delay_norm),
            "r_delay": float(r_delay),
            "r_energy_total": float(r_energy_total),
            "r_energy_host": float(r_energy_host),
            "lower_reward_mode": "marl_consistent",
            "hard_constraint_violation": 0,
            "reward_host": float(r_host),
            "reward_vm": float(r_vm),
            "legacy_reward_host": float(r_host),
            "legacy_reward_vm": float(r_vm),
        }
        if self.safe_rl_enabled:
            info.update(
                {
                    "performance_reward": float(performance_reward),
                    "performance_reward_host": float(performance_reward),
                    "performance_reward_vm": float(performance_reward),
                    "performance_energy_delta": float(
                        performance_energy_delta
                    ),
                    "performance_energy_score": float(
                        performance_energy_score
                    ),
                    **performance_breakdown,
                }
            )
        else:
            # 关闭 safe_rl 时只提供诊断别名；训练仍消费原返回的混合 reward。
            info.update(
                {
                    "performance_reward": float(r_vm),
                    "performance_reward_host": float(r_host),
                    "performance_reward_vm": float(r_vm),
                    "performance_energy_delta": 0.0,
                    "performance_energy_score": 0.0,
                    "performance_reward_definition": "legacy_mixed_reward",
                }
            )
        info.update(self.get_safety_diagnostics())
        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            self._phase_heuristic_safety_cost += float(
                info.get("safety_cost", 0.0)
            )
        self._attach_safety_shield_info(
            info,
            vm_decision,
        )

        self._cur_tid = None
        self._cur_host_id = None
        self._current_safety_shield_context = None

        return float(r_host), float(r_vm), info

    def finish_phase_and_advance(self):
        """结束当前分配阶段、推进仿真时间并计算 Manager 基础奖励"""
        if self._phase_assign_cnt == 0 and self._has_decision_point():
            self._fuse_zero_assign_streak += 1
            if self._fuse_zero_assign_streak >= self._fuse_zero_assign_limit:
                self._fuse_zero_assign_streak = 0
                if len(self.ready_task_ids) > 0 and np.any(self._vm_idle_mask()):
                    forced_tid = int(self.ready_task_ids[0])
                    forced_vm = int(self._vm_serial_order()[0])
                    if self._vm_idle_mask()[forced_vm]:
                        forced_energy_before = (
                            self.get_fuzzy_energy_summary()
                            if self.safe_rl_enabled
                            else None
                        )
                        self._assign_task_to_specific_vm(forced_tid, forced_vm)
                        (
                            forced_performance_reward,
                            forced_performance_energy_delta,
                            _,
                        ) = self._safe_performance_reward_delta(
                            forced_energy_before
                        )
                        if self.safe_rl_enabled:
                            self._phase_performance_reward += float(
                                forced_performance_reward
                            )
                            self._phase_performance_energy_delta += float(
                                forced_performance_energy_delta
                            )
                        self._phase_assign_cnt += 1
                        self._phase_assigned_tids.append(forced_tid)
                        self._phase_size_sum_mi += float(self.task_mi[forced_tid])
        else:
            self._fuse_zero_assign_streak = 0

        if self._phase_preadvanced:
            r_energy_phase = float(self._phase_preadvance_r_energy_total)
        else:
            r_energy_phase = float(self._advance_until_decision_energy_only())

        if self._phase_assign_cnt > 0 and self._phase_size_sum_mi > 0:
            r_manager_raw = r_energy_phase / float(self._phase_size_sum_mi)
        else:
            r_manager_raw = 0.0

        info = {
            "current_time": self.current_time,
            "manager_phase_id": int(self._manager_phase_id),
            "assign_cnt": int(self._phase_assign_cnt),
            "phase_size_sum_mi": float(self._phase_size_sum_mi),
            "r_energy_phase": float(r_energy_phase),
            "phase_assigned_tids": list(self._phase_assigned_tids),
        }
        if self.safe_rl_enabled:
            phase_energy_reward = float(
                self._phase_performance_reward
            )
            phase_energy_delta = float(
                self._phase_performance_energy_delta
            )
            phase_score_after = float(
                self._safe_fuzzy_energy_score
            )
            info.update(
                {
                    "performance_reward": phase_energy_reward,
                    "energy_reward": phase_energy_reward,
                    "completion_reward": 0.0,
                    "waiting_reward": 0.0,
                    "utilization_reward": 0.0,
                    "communication_reward": 0.0,
                    "total_performance_reward": (
                        phase_energy_reward
                    ),
                    "raw_energy_reward": -phase_energy_delta,
                    "performance_energy_delta": phase_energy_delta,
                    "fuzzy_energy_score_before": (
                        phase_score_after - phase_energy_delta
                    ),
                    "fuzzy_energy_score_after": phase_score_after,
                    "performance_energy_score": phase_score_after,
                    "performance_reward_scale": float(
                        self.energy_reward_scale
                    ),
                    "performance_reward_safety_cost_included": False,
                    "performance_reward_definition": (
                        "- delta(fuzzy energy mean + "
                        f"{self.fuzzy_energy_uncertainty_weight:.1f} * "
                        "fuzzy energy std) * performance_reward_scale"
                    ),
                }
            )
        else:
            info.update(
                {
                    "performance_reward": float(r_manager_raw),
                    "performance_energy_delta": 0.0,
                    "performance_energy_score": 0.0,
                    "performance_reward_definition": "legacy_manager_reward",
                }
            )
        info.update(self.get_safety_diagnostics())
        phase_heuristic_safety_cost = float(
            getattr(
                self,
                "_phase_heuristic_safety_cost",
                0.0,
            )
            + float(info.get("safety_cost", 0.0))
        )
        phase_shield_records = [
            dict(record)
            for record in self._phase_safety_shield_records
        ]
        info.update(
            self._safety_shield_diagnostics(
                phase_shield_records
            )
        )
        info["safety_shield_records"] = phase_shield_records
        heuristic_audit = self._manager_heuristic_audit_info(
            phase_shield_records
        )
        info.update(heuristic_audit)
        info["heuristic_phase_safety_cost"] = (
            phase_heuristic_safety_cost
        )
        self._record_selected_heuristic_phase(
            performance_reward=float(
                info.get(
                    "total_performance_reward",
                    info.get("performance_reward", r_manager_raw),
                )
            ),
            safety_cost=phase_heuristic_safety_cost,
            shield_intervention_rate=float(
                heuristic_audit[
                    "heuristic_shield_intervention_rate"
                ]
            ),
        )

        self._phase_started = False
        self._phase_tasks = []
        self._phase_ready_task_ordering = []
        self._phase_assigned_tids = []
        self._phase_assign_cnt = 0
        self._phase_size_sum_mi = 0.0
        self._cur_tid = None
        self._cur_host_id = None
        self._current_safety_shield_context = None
        self._current_host_shield_decision = None

        self._phase_preadvanced = False
        self._phase_preadvance_r_energy_total = 0.0
        self._phase_preadvance_r_energy_by_host = {int(h): 0.0 for h in self.host_ids}
        self._phase_performance_reward = 0.0
        self._phase_performance_energy_delta = 0.0
        self._phase_heuristic_safety_cost = 0.0
        self._phase_safety_shield_records = []
        self._manager_phase_id += 1

        return float(r_manager_raw), info

    def reset(self, *, seed=None, options=None):
        """重置环境状态、到达序列、Manager 权重和阶段缓存"""
        super().reset(seed=seed)
        self._episode_dax_paths = self._build_episode_dax_paths()
        self._reset_internal_buffers()

        rs = np.random.RandomState(12345)
        w0 = rs.uniform(low=0.0, high=1.0, size=self.task_heur_dim).astype(np.float32)
        if float(w0.sum()) <= 1e-12:
            w0[:] = 1.0
        self.manager_raw_w = w0
        self._normalize_manager_raw()

        self.arrival_times = poisson_arrival_times(
            self.arrival_lambda,
            n_max=self.workflows_per_episode,
            seed=self.random_seed,
        )

        if self.workflows_per_episode is not None:
            n = int(self.workflows_per_episode)
            if n > 0 and len(self.arrival_times) > n:
                self.arrival_times = self.arrival_times[:n]

        self.first_arrival_time = float(self.arrival_times[0]) if self.arrival_times else 0.0
        self.next_arrival_idx = 0

        _ = self._advance_until_decision_energy_only()

        self._phase_started = False
        self._phase_tasks = []
        self._phase_assigned_tids = []
        self._phase_assign_cnt = 0
        self._phase_size_sum_mi = 0.0
        self._cur_tid = None
        self._cur_host_id = None

        self._phase_preadvanced = False
        self._phase_preadvance_r_energy_total = 0.0
        self._phase_preadvance_r_energy_by_host = {int(h): 0.0 for h in self.host_ids}
        self._phase_performance_reward = 0.0
        self._phase_performance_energy_delta = 0.0
        self._phase_heuristic_safety_cost = 0.0
        self._phase_ready_task_ordering = []

        self._fuse_zero_assign_streak = 0
        self._fuse_no_event_streak = 0

        info = {
            "current_time": getattr(self, "current_time", 0.0),
            "first_arrival_time": self.first_arrival_time,
            "wf_target": self.workflows_per_episode if self.workflows_per_episode is not None else None,
        }
        obs = {
            "host_obs": np.zeros(self.host_obs_dim, np.float32),
            "host_mask": np.zeros(self.host_act_dim, np.float32),
            "vm_obs": np.zeros(self.vm_obs_dim, np.float32),
            "vm_mask": np.zeros(self.vm_act_dim, np.float32),
        }
        return obs, info

    def render(self):
        """输出当前时间、就绪任务数、空闲 VM 数和结束标记"""
        idle_vms = int(np.sum(self._vm_idle_mask()))
        print(f"[t={self.current_time:.2f}] ready={len(self.ready_task_ids)} idleVM={idle_vms} done={self.done_flag}")

    def _reset_internal_buffers(self):
        """清空任务、事件、工作流、能耗和当前选择等运行期状态"""
        self.current_time = 0.0
        self.done_flag = False

        self.workflows = []
        self.arrival_times = []
        self.next_arrival_idx = 0

        self.task_meta = []
        self.task_state = []
        self.task_parents = []
        # 旧 task_parents 必须继续保存工作流内部局部 ID，供 HRL 完成事件逻辑使用。
        # CEWS 三场景传播另存全局父任务 ID，避免不同工作流局部 ID 冲突。
        self.task_global_parents = []
        self.task_children = []
        self.task_mi = []
        self.task_in_bits = []
        self.task_out_bits = []
        self.task_up_rank = []
        self.task_down_rank = []
        self.task_ready_time = []
        self.task_end_time = []
        # modal 场景继续复用 vm_available_at、task_end_time 和 _records；另外两个
        # 影子场景仅重放相同任务顺序与 VM 映射，不驱动离散事件堆。
        self.shadow_task_start_time = {
            "optimistic": [],
            "pessimistic": [],
        }
        self.shadow_task_end_time = {
            "optimistic": [],
            "pessimistic": [],
        }
        # 以下映射与历史记录供建设式 CEWS 评价统计实际分配结果；reset 时必须
        # 清空，防止不同 seed/episode 之间串用 VM、Host 或 cloud/edge 计数。
        self.task_assigned_vm = {}
        self.task_assigned_host = {}
        self.assignment_history = []
        # remaining_work 只依赖本 episode 的 DAG，加载新工作流集合后缓存失效。
        self._remaining_work_cache = {}
        # 三场景时长缓存同理：task_mi/in_bits/out_bits 在此重建，旧的 task_id
        # 不再对应同一任务，必须一并清空。
        self._scenario_duration_cache = {}

        self.ready_task_ids = []
        self.event_heap = []
        self.vm_available_at = np.zeros(self.num_vms, dtype=np.float64)
        self.shadow_vm_available_at = {
            "optimistic": np.zeros(self.num_vms, dtype=np.float64),
            "pessimistic": np.zeros(self.num_vms, dtype=np.float64),
        }

        self.completed_workflows = 0
        self.wf_remaining_tasks = {}

        self.task_baseline_finish = []

        self.wf_finish_time = {}
        self.wf_lateness = {}
        self.wf_timeout = {}
        self.ep_wf_lateness_sum = 0.0

        self.total_energy = 0.0
        self.lifetime_energy = 0.0
        self._records = []
        self.shadow_records = {
            "optimistic": [],
            "pessimistic": [],
        }
        self._energy_cache = 0.0
        self._energy_cache_by_host = {int(h): 0.0 for h in self.host_ids}

        # 安全完成事件采用 episode 内集合去重；累计字段只用于诊断，不进入旧
        # reward，也不写入当前 D3QN replay。
        self._safety_accounted_workflow_ids = set()
        self._safety_cumulative_cost = 0.0
        self._safety_cumulative_transition_count = 0
        self._safety_cumulative_deadline_violation_count = 0
        self._safety_cumulative_completed_workflow_count = 0
        self._safety_cumulative_fuzzy_lateness_cost = 0.0
        self._safety_cumulative_process_risk_cost = 0.0
        self._safe_fuzzy_energy_score = 0.0
        self._phase_heuristic_safety_cost = 0.0
        self._phase_ready_task_ordering = []
        self._safety_shield_records = []
        self._phase_safety_shield_records = []
        self._current_safety_shield_context = None
        self._current_host_shield_decision = None
        self._manager_phase_id = 0

        self._cur_tid = None
        self._cur_host_id = None

    def _build_episode_dax_paths(self) -> tuple[str, ...]:
        """Build the cache-compatible category sequence without env RNG."""
        if not self.task_code or self.workflows_per_episode is None:
            return tuple()
        by_name = {
            os.path.basename(str(path)).lower(): str(path)
            for path in self.dax_paths
        }
        names = deterministic_workload_sequence(
            self.task_code,
            self.random_seed,
            self.workflows_per_episode,
        )
        missing = sorted({name for name in names if name.lower() not in by_name})
        if missing:
            if getattr(self, "deadline_mode", None) == "none":
                return tuple()
            raise ValueError(
                "registered workload DAX files are missing from dax_paths: "
                f"{missing}"
            )
        return tuple(by_name[name.lower()] for name in names)

    @property
    def episode_dax_sequence(self) -> tuple[str, ...]:
        """Return immutable DAX basenames selected for this episode."""
        return tuple(os.path.basename(path) for path in self._episode_dax_paths)

    def _dax_path_for_arrival(self, arrival_index: int) -> str:
        index = int(arrival_index)
        if self._episode_dax_paths:
            return self._episode_dax_paths[index]
        if self.dax_probs is None:
            return self.dax_paths[self.rng.randint(len(self.dax_paths))]
        sampled = self.rng.choice(len(self.dax_paths), p=self.dax_probs)
        return self.dax_paths[int(sampled)]
    def _no_more_arrivals(self):
        """判断本轮预生成的工作流是否已全部到达"""
        return self.next_arrival_idx >= len(self.arrival_times)

    def _add_workflow_if_arrived(self):
        """加载当前时刻已经到达的工作流并建立任务级期限和状态"""
        added = False
        while (
            self.next_arrival_idx < len(self.arrival_times)
            and self.arrival_times[self.next_arrival_idx] <= self.current_time
        ):
            arr_t = self.arrival_times[self.next_arrival_idx]

            path = self._dax_path_for_arrival(self.next_arrival_idx)

            G, tasks = load_workflow_from_dax(
                path,
                seed=self.random_seed + self.next_arrival_idx,
                randomize_payloads=True,
            )
            wf = Workflow(
                workflow_id=len(self.workflows),
                graph=G,
                tasks=tasks,
                arrival_time=arr_t,
                source_device_id=0,
            )
            wf.is_arrived = True

            pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)

            self.wf_remaining_tasks[wf.workflow_id] = len(tasks)

            heft_ms, heft_finish_rel, avg_dur = _heft_makespan_and_finish_times(
                tasks, self._heft_vm_pc, self._heft_vm_bw_bps
            )
            alpha = 1.5
            wf_deadline_abs = float(arr_t + alpha * float(heft_ms))
            wf.deadline = wf_deadline_abs
            wf.heft_makespan = float(heft_ms)

            n_local = len(tasks)
            topo, succ = _topo_sort_local(tasks)

            P = avg_dur.reshape(-1).astype(np.float64)
            D_wf_rel = float(alpha * float(heft_ms))

            LF = np.zeros((n_local,), dtype=np.float64)
            LS = np.zeros((n_local,), dtype=np.float64)

            is_sink = np.array([len(succ[i]) == 0 for i in range(n_local)], dtype=bool)

            for i in range(n_local):
                if is_sink[i]:
                    LF[i] = D_wf_rel
                    LS[i] = LF[i] - P[i]

            for u in reversed(topo):
                if is_sink[u]:
                    continue
                if len(succ[u]) > 0:
                    min_LS_child = float(np.min(LS[np.array(succ[u], dtype=np.int32)]))
                    LF[u] = min_LS_child
                else:
                    LF[u] = D_wf_rel
                LF[u] = max(float(LF[u]), float(heft_finish_rel[u]))
                LS[u] = LF[u] - P[u]

            for u in range(n_local):
                if is_sink[u]:
                    LF[u] = max(float(LF[u]), float(heft_finish_rel[u]))
                    LS[u] = LF[u] - P[u]

            base = len(self.task_meta)

            for local_id in range(n_local):
                abs_deadline = float(arr_t + LF[local_id])
                self.task_baseline_finish.append(abs_deadline)
                # 同时把绝对子截止期写回已有 Task 对象，便于环境接口和外部分析
                # 使用统一字段；不创建新的 Task 类型或复制工作流。
                tasks[local_id].sub_deadline = abs_deadline

            for t in tasks:
                self.task_meta.append((wf.workflow_id, t.task_id))
                self.task_state.append("unReady")
                self.task_parents.append(list(t.parents))
                self.task_global_parents.append(
                    [base + int(parent_local_id) for parent_local_id in t.parents]
                )
                self.task_children.append(list(t.children))
                self.task_mi.append(float(t.workload_mi))
                in_bits_total = float((t.ext_in_bits or 0.0) + (t.from_parents_bits or 0.0))
                self.task_in_bits.append(in_bits_total)
                self.task_out_bits.append(float(t.out_file_size_sum_bits))
                self.task_up_rank.append(float(t.upward_rank or 0.0))
                self.task_down_rank.append(float(t.downward_rank or 0.0))
                self.task_ready_time.append(0.0)
                self.task_end_time.append(0.0)
                self.shadow_task_start_time["optimistic"].append(0.0)
                self.shadow_task_start_time["pessimistic"].append(0.0)
                self.shadow_task_end_time["optimistic"].append(0.0)
                self.shadow_task_end_time["pessimistic"].append(0.0)

            for local_id, t in enumerate(tasks):
                if len(t.parents) == 0:
                    gid = base + local_id
                    self.task_state[gid] = "Ready"
                    self.task_ready_time[gid] = self.current_time
                    t.state = "Ready"
                    t.ready_time = float(self.current_time)
                    self.ready_task_ids.append(gid)

            self.workflows.append(wf)
            self.next_arrival_idx += 1
            added = True
        return added

    def _vm_idle_mask(self):
        """返回表示各 VM 当前是否空闲的布尔掩码"""
        return self.vm_available_at <= self.current_time + 1e-9

    def _has_decision_point(self):
        """判断是否同时存在就绪任务和空闲 VM"""
        return (len(self.ready_task_ids) > 0) and np.any(self._vm_idle_mask())

    def _episode_should_end(self):
        """根据到达、完成、运行和就绪状态判断本轮是否结束"""
        if self.workflows_per_episode is None:
            return (
                self.current_time >= self.horizon
                and len(self.event_heap) == 0
                and self._no_more_arrivals()
            )
        else:
            all_arrived = self._no_more_arrivals()
            all_done = self.completed_workflows >= self.workflows_per_episode
            no_running = len(self.event_heap) == 0
            no_ready = len(self.ready_task_ids) == 0
            return all_arrived and all_done and no_running and no_ready

    def _energy_reward_to_current_time_with_breakdown(self):
        """累计截至当前时刻的增量能耗并返回总奖励及分主机奖励"""
        started = len(self.workflows) > 0
        if not started:
            return 0.0, {int(h): 0.0 for h in self.host_ids}

        if self.workflows_per_episode is not None:
            counting = self.completed_workflows < self.workflows_per_episode
        else:
            counting = any(s != "Finished" for s in self.task_state)

        if not counting:
            return 0.0, {int(h): 0.0 for h in self.host_ids}

        t = self.current_time
        clipped_all = []

        for r in self._records:
            if r.end_time <= 0.0:
                continue
            if r.start_time >= t:
                continue
            end = min(r.end_time, t)
            if end <= r.start_time:
                continue
            clipped_all.append(
                LoadRecord(r.start_time, end, r.server_id, r.vm_pc)
            )

        # 单趟积分同时得到总能耗与分主机能耗。两者都与旧的"一次全量 +
        # 每主机一次"多趟调用逐位相同，因为分主机记录原本就是按同一顺序
        # 追加的，函数内部也按同样顺序累加。
        E_total, E_by_server = energy_from_records_with_breakdown(
            clipped_all,
            self.hosts,
        )
        dE_total = max(0.0, E_total - self._energy_cache)
        self._energy_cache = float(E_total)

        dE_by_host = {}
        for h in self.host_ids:
            # 没有可计入记录的主机不会出现在字典里，与旧实现返回 0.0 一致。
            E_h = float(E_by_server.get(int(h), 0.0))
            prev_h = float(self._energy_cache_by_host.get(int(h), 0.0))
            dE_h = max(0.0, E_h - prev_h)
            self._energy_cache_by_host[int(h)] = E_h
            dE_by_host[int(h)] = float(dE_h)

        self.total_energy += float(dE_total)
        self.lifetime_energy += float(dE_total)

        r_total = -(float(dE_total) * float(self.energy_reward_scale))
        r_by_host = {
            int(h): -(float(dE_by_host[int(h)]) * float(self.energy_reward_scale)) for h in self.host_ids
        }
        return float(r_total), r_by_host

    def _energy_reward_to_current_time(self) -> float:
        """返回截至当前时刻的总增量能耗奖励"""
        r_total, _ = self._energy_reward_to_current_time_with_breakdown()
        return float(r_total)

    def get_fuzzy_energy_summary(self) -> dict:
        """重放三场景 Host 负载时间线并返回风险调整模糊总能耗。

        modal 能耗始终用原 ``_records`` 与 Host modal 总处理能力重新积分。启用
        fuzzy 时，optimistic/pessimistic 分别使用影子记录及 upper/lower 容量；
        关闭时两个影子值退化为 modal，从而令 std=0、score=旧 modal 能耗。
        """
        energy_modal = float(
            energy_from_records(
                self._records,
                self.hosts,
                total_pc_component="modal",
            )
        )
        if getattr(self, "fuzzy_enabled", False):
            energy_optimistic = float(
                energy_from_records(
                    self.shadow_records["optimistic"],
                    self.hosts,
                    total_pc_component="upper",
                )
            )
            energy_pessimistic = float(
                energy_from_records(
                    self.shadow_records["pessimistic"],
                    self.hosts,
                    total_pc_component="lower",
                )
            )
        else:
            energy_optimistic = energy_modal
            energy_pessimistic = energy_modal

        raw_energies = (
            energy_optimistic,
            energy_modal,
            energy_pessimistic,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in raw_energies):
            raise ValueError(
                "Fuzzy energy replay produced invalid values: "
                f"optimistic={energy_optimistic}, modal={energy_modal}, "
                f"pessimistic={energy_pessimistic}"
            )
        energy_tfn = TriangularFuzzyNumber(
            min(raw_energies),
            energy_modal,
            max(raw_energies),
        )
        return {
            "fuzzy_total_energy_lower": float(energy_tfn.lower),
            "fuzzy_total_energy_modal": float(energy_tfn.modal),
            "fuzzy_total_energy_upper": float(energy_tfn.upper),
            "fuzzy_total_energy_mean": float(energy_tfn.mean()),
            "fuzzy_total_energy_std": float(energy_tfn.std()),
            "fuzzy_total_energy_score": float(
                energy_tfn.score(
                    self.fuzzy_energy_uncertainty_weight
                )
            ),
            "energy_optimistic": float(energy_optimistic),
            "energy_modal": float(energy_modal),
            "energy_pessimistic": float(energy_pessimistic),
        }

    def _advance_until_decision_energy_only(self, return_breakdown: bool = False):
        """推进到下一个决策点并累计推进区间内的能耗奖励"""
        r_energy = 0.0
        r_energy_by_host = {int(h): 0.0 for h in self.host_ids}

        while True:
            self._add_workflow_if_arrived()
            if self._has_decision_point():
                self._fuse_no_event_streak = 0
                break

            next_times = []
            if len(self.event_heap) > 0:
                next_times.append(self.event_heap[0][0])
            if not self._no_more_arrivals():
                next_times.append(self.arrival_times[self.next_arrival_idx])

            if len(next_times) == 0:
                self._fuse_no_event_streak += 1
                if self._episode_should_end() or self._fuse_no_event_streak >= self._fuse_no_event_limit:
                    self.done_flag = True
                    break
                self.current_time += 1e-9
                r0, rb = self._energy_reward_to_current_time_with_breakdown()
                r_energy += float(r0)
                for h in self.host_ids:
                    r_energy_by_host[int(h)] += float(rb.get(int(h), 0.0))
                continue

            next_t = min(next_times)
            if next_t <= self.current_time + 1e-12:
                next_t = self.current_time + 1e-9

            self.current_time = next_t
            r0, rb = self._energy_reward_to_current_time_with_breakdown()
            r_energy += float(r0)
            for h in self.host_ids:
                r_energy_by_host[int(h)] += float(rb.get(int(h), 0.0))

            self._process_finish_events_at_current_time()

            if self._has_decision_point():
                self._fuse_no_event_streak = 0
                break
            if self._episode_should_end() or self.current_time >= self.horizon:
                self.done_flag = True
                break

        if return_breakdown:
            return float(r_energy), {int(h): float(r_energy_by_host[int(h)]) for h in self.host_ids}
        return float(r_energy)

    def _process_finish_events_at_current_time(self):
        """处理当前时刻的任务完成事件并释放满足依赖的子任务"""
        while len(self.event_heap) > 0 and self.event_heap[0][0] <= self.current_time + 1e-12:
            _, etype, task_id, vm_idx = heapq.heappop(self.event_heap)
            if etype != "finish":
                continue
            self.task_state[task_id] = "Finished"
            self.task_end_time[task_id] = self.current_time

            wf_id, local_id = self.task_meta[task_id]
            task_obj = self.workflows[wf_id].tasks[local_id]
            # 同步更新已有 Task 对象，使基于数组的高效环境状态与面向对象的
            # Workflow/Task 视图一致，便于 CEWS 指标和后续依赖任务读取完成时间。
            task_obj.state = "Finished"
            task_obj.end_processing_time = float(self.current_time)
            task_obj.finish_time = float(self.current_time)

            wf_id, _ = self.task_meta[task_id]
            if wf_id in self.wf_remaining_tasks:
                self.wf_remaining_tasks[wf_id] -= 1
                if self.wf_remaining_tasks[wf_id] == 0:
                    self.completed_workflows += 1

                    wf = self.workflows[wf_id]
                    finish_t = float(self.current_time)
                    self.wf_finish_time[wf_id] = finish_t
                    wf_finish_late = max(0.0, finish_t - float(getattr(wf, "deadline", finish_t)))
                    self.wf_lateness[wf_id] = float(wf_finish_late)
                    self.wf_timeout[wf_id] = bool(finish_t > float(getattr(wf, "deadline", finish_t)))
                    self.ep_wf_lateness_sum += float(wf_finish_late)

            wf_idx, _ = self.task_meta[task_id]
            for child_local in self.task_children[task_id]:
                child_gid = None
                for gid, meta in enumerate(self.task_meta):
                    if meta[0] == wf_idx and meta[1] == child_local:
                        child_gid = gid
                        break
                if child_gid is None:
                    continue
                if self.task_state[child_gid] == "unReady":
                    all_ok = True
                    for p_local in self.task_parents[child_gid]:
                        p_gid = None
                        for gid, meta in enumerate(self.task_meta):
                            if meta[0] == wf_idx and meta[1] == p_local:
                                p_gid = gid
                                break
                        if p_gid is None or self.task_state[p_gid] != "Finished":
                            all_ok = False
                            break
                    if all_ok:
                        self.task_state[child_gid] = "Ready"
                        self.task_ready_time[child_gid] = self.current_time
                        _, child_local_id = self.task_meta[child_gid]
                        child_obj = self.workflows[wf_idx].tasks[child_local_id]
                        child_obj.state = "Ready"
                        child_obj.ready_time = float(self.current_time)
                        self.ready_task_ids.append(child_gid)

        self.ready_task_ids = [tid for tid in self.ready_task_ids if self.task_state[tid] == "Ready"]

    def _compute_task_heuristics_for_ready(self):
        """计算所有就绪任务的五维启发式特征"""
        now = self.current_time
        sel_ids = list(self.ready_task_ids)
        if len(sel_ids) == 0:
            return [], np.zeros((0, self.task_heur_dim), dtype=np.float32)

        pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
        bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
        bw_mean_bps = bw_mean * 1e6

        time_scale = 1.0 + np.mean(np.array(self.task_mi, dtype=np.float64) / max(pc_mean, 1.0))

        feats = np.zeros((len(sel_ids), self.task_heur_dim), dtype=np.float32)
        for i, tid in enumerate(sel_ids):
            exp_t = (
                self.task_in_bits[tid] / max(bw_mean_bps, 1e-9)
                + self.task_mi[tid] / max(pc_mean, 1.0)
                + self.task_out_bits[tid] / max(bw_mean_bps, 1e-9)
            )
            fcfs = max(0.0, now - self.task_ready_time[tid])
            sjf = 1.0 / (exp_t + 1.0)
            mcf = float(len(self.task_children[tid]))
            hur = float(self.task_up_rank[tid])

            wf_idx, _ = self.task_meta[tid]
            wf = self.workflows[wf_idx]
            if getattr(wf, "deadline", None) is not None:
                slack = max(0.0, float(wf.deadline) - (now + exp_t))
                edf = 1.0 / (slack + 1.0)
            else:
                edf = 0.0

            if self.normalize:
                fcfs = fcfs / max(self.horizon, 1.0)
                hur = hur / max(time_scale, 1.0)
                mcf = mcf / max(1.0, np.mean([len(ch) for ch in self.task_children]) + 1e-6)

            feats[i] = np.array([fcfs, sjf, mcf, hur, edf], dtype=np.float32)
        return sel_ids, feats

    def _phase_prepare_tasks(self):
        """按当前 Manager 模式排序 ready tasks 并建立阶段队列。"""
        self._add_workflow_if_arrived()
        sel_ids, feats = self._compute_task_heuristics_for_ready()
        if feats.shape[0] == 0:
            self._phase_tasks = []
            self._phase_ready_task_ordering = []
            return
        stable_ids = np.asarray(sel_ids, dtype=np.int64)
        if self.manager_mode == LEGACY_RULE_WEIGHT_MODE:
            scores = (
                feats @ self.combo_weights.reshape(-1, 1)
            ).squeeze(1)
            # 保持历史 argsort 语义，避免关闭新模式时改变旧实验。
            order = np.argsort(-scores)
        else:
            heuristic = self._selected_manager_heuristic()
            if (
                self.get_manager_action_mask()[
                    int(self.selected_heuristic_index)
                ]
                <= 0.5
            ):
                raise RuntimeError(
                    "selected Manager heuristic is not admitted for "
                    "the current evaluation context and workflow seed"
                )
            if heuristic.traditional_feature_index is not None:
                scores = feats[
                    :,
                    int(heuristic.traditional_feature_index),
                ]
                # 传统五规则复用原 HRL 特征，仍保持分数越高越优先。
                order = np.lexsort((stable_ids, -scores))
            else:
                # LLM 只获得八个 ready-task 数值数组。返回值只决定 task
                # ordering，不会流入 Host/VM 的动作或 observation。
                features = self.build_task_features(sel_ids)
                scores = heuristic.priority_rule(
                    features["min_exec_time"],
                    features["min_comm_time"],
                    features["min_incremental_energy"],
                    features["slack"],
                    features["upward_rank"],
                    features["remaining_work"],
                    features["ready_wait_time"],
                    features["uncertainty"],
                )
                scores = validate_task_priority_scores(
                    scores,
                    len(sel_ids),
                )
                # get_task_priority_v2 明确定义为分数越小越优先。
                order = np.lexsort((stable_ids, scores))
        ordering = [
            int(sel_ids[index]) for index in order.tolist()
        ]
        self._phase_tasks = list(ordering)
        self._phase_ready_task_ordering = list(ordering)

    def _manager_heuristic_identity(self) -> dict:
        if self.manager_mode == HEURISTIC_SELECTION_MODE:
            heuristic = self._selected_manager_heuristic()
            return {
                "selected_heuristic_id": (
                    heuristic.heuristic_id
                ),
                "heuristic_source": heuristic.source,
                "llm_rule_version": (
                    heuristic.version
                    if heuristic.is_llm_rule
                    else ""
                ),
                "heuristic_rule_version": heuristic.version,
                "manager_mode": self.manager_mode,
                "selected_heuristic_index": int(
                    self.selected_heuristic_index
                ),
            }
        return {
            "selected_heuristic_id": "legacy_weighted_combo",
            "heuristic_source": "traditional_weighted_combo",
            "llm_rule_version": "",
            "heuristic_rule_version": "legacy_v1",
            "manager_mode": self.manager_mode,
            "selected_heuristic_index": None,
        }

    def _manager_heuristic_audit_info(
        self,
        shield_records=None,
    ) -> dict:
        records = [
            dict(record)
            for record in (shield_records or [])
        ]
        interventions = [
            record
            for record in records
            if bool(record.get("shield_intervened", False))
            or bool(record.get("action_modified", False))
            or bool(record.get("fallback_triggered", False))
        ]
        result = {
            **self._manager_heuristic_identity(),
            "ready_task_ordering": list(
                self._phase_ready_task_ordering
            ),
            "subsequent_safety_interventions": interventions,
            "heuristic_shield_record_count": len(records),
            "heuristic_shield_intervention_count": len(
                interventions
            ),
            "heuristic_shield_intervention_rate": float(
                len(interventions) / max(len(records), 1)
            ),
        }
        return result

    def _record_selected_heuristic_phase(
        self,
        *,
        performance_reward: float,
        safety_cost: float,
        shield_intervention_rate: float,
    ) -> None:
        if self.manager_mode != HEURISTIC_SELECTION_MODE:
            return
        heuristic = self._selected_manager_heuristic()
        history = self._heuristic_recent_metrics[
            heuristic.heuristic_id
        ]
        values = (
            float(performance_reward),
            float(safety_cost),
            float(shield_intervention_rate),
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Manager heuristic phase metrics must be finite"
            )
        if values[1] < 0.0:
            raise ValueError(
                "Manager heuristic safety cost must be non-negative"
            )
        history["performance_reward"].append(values[0])
        history["safety_cost"].append(values[1])
        history["shield_intervention_rate"].append(
            _clip01(values[2])
        )

    def _compute_block2(self):
        """构造描述队列规模、等待时间和期限紧迫度的全局特征块"""
        now = self.current_time
        ready_cnt = float(len(self.ready_task_ids))
        waiting_cnt = float(sum(1 for s in self.task_state if s == "unReady"))
        running_cnt = float(sum(1 for s in self.task_state if s == "Running"))

        waits = [max(0.0, now - self.task_ready_time[x]) for x in self.ready_task_ids]
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        max_wait = float(np.max(waits)) if len(waits) > 0 else 0.0

        sum_mi = float(np.sum([self.task_mi[x] for x in self.ready_task_ids])) if len(self.ready_task_ids) > 0 else 0.0
        urgent_cnt = 0

        pc_mean = float(np.mean([self.vms[v].pc for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_mean = float(np.mean([self.vms[v].bw for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_bps = bw_mean * 1e6

        slacks = []
        for x in self.ready_task_ids:
            exp_t = (
                self.task_in_bits[x] / max(bw_bps, 1e-9)
                + self.task_mi[x] / max(pc_mean, 1.0)
                + self.task_out_bits[x] / max(bw_bps, 1e-9)
            )
            wf_idx, _ = self.task_meta[x]
            wf = self.workflows[wf_idx]
            dl = getattr(wf, "deadline", None)
            if dl is not None:
                s = float(dl) - (now + exp_t)
                slacks.append(s)
                if s < 0:
                    urgent_cnt += 1

        if len(slacks) > 0:
            min_slack = float(np.min(slacks))
            avg_slack = float(np.mean(slacks))
        else:
            min_slack = 0.0
            avg_slack = 0.0

        block2 = np.array([
            ready_cnt / max(1.0, self.max_ready),
            waiting_cnt / max(1.0, len(self.task_state) if len(self.task_state) > 0 else 1.0),
            running_cnt / max(1.0, self.num_vms),
            avg_wait / max(self.horizon, 1.0),
            max_wait / max(self.horizon, 1.0),
            sum_mi / max(1.0, np.mean(self.task_mi) if len(self.task_mi) > 0 else 1.0),
            float(urgent_cnt) / max(1.0, len(self.ready_task_ids)),
            min_slack / max(self.horizon, 1.0),
            avg_slack / max(self.horizon, 1.0),
            float(len(set([self.task_meta[x][0] for x in self.ready_task_ids]))) / max(
                1.0, len(self.workflows) if len(self.workflows) > 0 else 1.0
            ),
        ], dtype=np.float32)
        return block2

    def _compute_block1(self, tid: int):
        """构造指定任务的计算量、通信量、拓扑和期限特征块"""
        now = self.current_time
        in_bits = float(self.task_in_bits[tid])
        out_bits = float(self.task_out_bits[tid])
        mi = float(self.task_mi[tid])

        io_ratio = _safe_div(in_bits + out_bits, max(mi, 1.0))
        wait_norm = (now - self.task_ready_time[tid]) / max(self.horizon, 1.0)
        children_cnt_n = float(len(self.task_children[tid])) / max(
            1.0, np.mean([len(ch) for ch in self.task_children]) + 1e-6
        )
        up_rank_n = self.task_up_rank[tid] / max(
            1.0, np.mean(self.task_up_rank) if len(self.task_up_rank) > 0 else 1.0
        )
        down_rank_n = self.task_down_rank[tid] / max(
            1.0, np.mean(self.task_down_rank) if len(self.task_down_rank) > 0 else 1.0
        )

        vm_idle = self._vm_idle_mask()
        idle_indices = np.where(vm_idle)[0].tolist()
        if len(idle_indices) == 0:
            est_best = est_mean = 0.0
        else:
            durations = []
            for j in idle_indices:
                vm_id = self.vm_ids[j]
                bw_bps_vm = self.vms[vm_id].bw * 1e6
                pc_mi_s = self.vms[vm_id].pc
                t_upload = in_bits / max(bw_bps_vm, 1e-9)
                t_compute = mi / max(pc_mi_s, 1e-9)
                t_download = out_bits / max(bw_bps_vm, 1e-9)
                durations.append(t_upload + t_compute + t_download)
            est_best = float(np.min(durations))
            est_mean = float(np.mean(durations))

        wf_idx, _ = self.task_meta[tid]
        wf = self.workflows[wf_idx]
        if getattr(wf, "deadline", None) is not None:
            slack_on_best = float(wf.deadline) - (now + est_best)
        else:
            slack_on_best = 0.0

        same_wf_ready_frac = 0.0
        if len(self.ready_task_ids) > 0:
            same = [x for x in self.ready_task_ids if self.task_meta[x][0] == wf_idx]
            same_wf_ready_frac = float(len(same)) / float(max(1, len(self.ready_task_ids)))

        block1 = np.array([
            mi / max(1.0, np.mean(self.task_mi) if len(self.task_mi) > 0 else 1.0),
            in_bits / max(1.0, np.mean(self.task_in_bits) if len(self.task_in_bits) > 0 else 1.0),
            out_bits / max(1.0, np.mean(self.task_out_bits) if len(self.task_out_bits) > 0 else 1.0),
            io_ratio,
            wait_norm,
            children_cnt_n,
            up_rank_n,
            down_rank_n,
            est_best / max(self.horizon, 1.0),
            est_mean / max(self.horizon, 1.0),
            slack_on_best / max(self.horizon, 1.0),
            same_wf_ready_frac,
        ], dtype=np.float32)
        return block1, in_bits, out_bits, mi

    def _compute_host_block(self):
        """构造所有主机的容量、负载、空闲比例和可用时间特征"""
        now = self.current_time
        host_stats = {
            "active_pc": {h: 0.0 for h in self.host_ids},
            "idle_vm": {h: 0 for h in self.host_ids},
            "soonest": {h: None for h in self.host_ids},
            "running": {h: 0 for h in self.host_ids},
            "vm_cnt": {h: 0 for h in self.host_ids},
        }
        for j, vid in enumerate(self.vm_ids):
            h = self.vms[vid].host_id
            host_stats["vm_cnt"][h] += 1
            if self.vm_available_at[j] > now + 1e-12:
                host_stats["active_pc"][h] += self.vms[vid].pc
                host_stats["running"][h] += 1
            else:
                host_stats["idle_vm"][h] += 1
            if host_stats["soonest"][h] is None or self.vm_available_at[j] < host_stats["soonest"][h]:
                host_stats["soonest"][h] = self.vm_available_at[j]

        mean_total_pc = max(1.0, float(np.mean([self.hosts[x].total_pc for x in self.host_ids]))) if self.num_hosts > 0 else 1.0
        host_block = np.zeros((self.num_hosts, self.host_feat_dim), dtype=np.float32)
        for hi, h in enumerate(self.host_ids):
            vm_cnt = max(1, host_stats["vm_cnt"][h])
            total_pc_n = self.hosts[h].total_pc / mean_total_pc
            host_load = _safe_div(host_stats["active_pc"][h], max(self.hosts[h].total_pc, 1e-9))
            host_idle_vm = float(host_stats["idle_vm"][h]) / vm_cnt
            soonest = host_stats["soonest"][h] if host_stats["soonest"][h] is not None else now
            host_soonest_delay = max(0.0, float(soonest - now))
            host_running_ratio = _safe_div(host_stats["running"][h], vm_cnt)
            host_block[hi] = np.array([
                total_pc_n,
                host_load,
                host_idle_vm,
                host_soonest_delay / max(self.horizon, 1e-9),
                host_running_ratio,
            ], dtype=np.float32)
        return host_block

    def _build_host_obs_for_task(self, tid: int):
        """拼接 HostAgent 观测并标记具有空闲 VM 的可选主机"""
        block1, _, _, _ = self._compute_block1(tid)
        block2 = self._compute_block2()
        host_block = self._compute_host_block()

        obs = np.concatenate([block1, block2, host_block.reshape(-1)], axis=0)

        mask = np.zeros((self.host_act_dim,), dtype=np.float32)
        now = self.current_time
        for hi, h in enumerate(self.host_ids):
            ok = False
            for j in self.host_to_vm_indices[h]:
                if self.vm_available_at[j] <= now + 1e-9:
                    ok = True
                    break
            if ok:
                mask[hi] = 1.0
        return obs, mask

    def _build_vm_obs_for_task_host(self, tid: int, host_id: int):
        """拼接 VMAgent 观测并标记目标主机上的空闲 VM 槽位"""
        block1, in_bits, out_bits, mi = self._compute_block1(tid)
        block2 = self._compute_block2()
        host_block_all = self._compute_host_block()
        hi = self.host_ids.index(host_id)
        host_feat = host_block_all[hi].astype(np.float32)

        mean_vm_pc = max(1.0, float(np.mean([self.vms[v].pc for v in self.vms])))
        mean_vm_bw = max(1.0, float(np.mean([self.vms[v].bw for v in self.vms])))

        vm_block = np.zeros((self.max_vms_per_host, self.vm_feat_dim), dtype=np.float32)
        mask = np.zeros((self.vm_act_dim,), dtype=np.float32)

        now = self.current_time
        vm_list = self.host_to_vm_indices[host_id]
        for slot in range(self.max_vms_per_host):
            if slot >= len(vm_list):
                continue
            j = vm_list[slot]
            vid = self.vm_ids[j]
            vm = self.vms[vid]
            idle_flag = 1.0 if self.vm_available_at[j] <= now + 1e-12 else 0.0
            avail_delay = max(0.0, float(self.vm_available_at[j] - now))
            bw_bps_vm = vm.bw * 1e6
            t_upload = in_bits / max(bw_bps_vm, 1e-9)
            t_compute = mi / max(vm.pc, 1e-9)
            t_download = out_bits / max(bw_bps_vm, 1e-9)
            pred_finish = max(now, float(self.vm_available_at[j])) + t_upload + t_compute + t_download

            vm_block[slot] = np.array([
                idle_flag,
                avail_delay / max(self.horizon, 1e-9),
                vm.pc / mean_vm_pc,
                vm.bw / mean_vm_bw,
                pred_finish / max(self.horizon, 1e-9),
            ], dtype=np.float32)

            if idle_flag > 0.5:
                mask[slot] = 1.0

        obs = np.concatenate([block1, block2, host_feat, vm_block.reshape(-1)], axis=0)
        return obs, mask

    def _estimate_marginal_energy_proxy(
        self,
        host_id: int,
        start_time: float,
        duration: float,
        added_pc: float,
        *,
        vm_available_array=None,
        pc_component="modal",
        total_pc_component="modal",
    ) -> float:
        """估算新增计算负载在给定持续时间内引起的边际能耗。

        可显式传入三场景 VM 可用时刻以及 VM/Host 处理能力分量。默认参数完全
        等同旧 modal 行为，因而原 HRL 奖励与状态逻辑无需改变。
        """
        if duration <= 0.0:
            return 0.0
        host = self.hosts[host_id]
        if pc_component not in {"lower", "modal", "upper"}:
            raise ValueError(
                "pc_component must be 'lower', 'modal', or 'upper'."
            )
        total_pc = max(
            host.total_pc.component(total_pc_component),
            1e-9,
        )
        if vm_available_array is None:
            vm_available_array = self.vm_available_at
        if len(vm_available_array) != self.num_vms:
            raise ValueError(
                "vm_available_array length does not match environment VM count"
            )

        active_pc_before = 0.0
        for j in self.host_to_vm_indices[host_id]:
            if vm_available_array[j] > start_time + 1e-12:
                vid = self.vm_ids[j]
                active_pc_before += self.vms[vid].pc.component(pc_component)

        load_before = _safe_div(active_pc_before, total_pc)
        load_after = _safe_div(active_pc_before + float(added_pc), total_pc)

        # _safe_div 已保证是 Python 标量 float，标量截断与 np.clip 逐位等价。
        # 这两行是全评估中调用最频繁的 np.clip 点位（约 120 万次）。
        load_before = _clip01_numpy_exact(load_before)
        load_after = _clip01_numpy_exact(load_after)

        P_before = float(host.power(load_before))
        P_after = float(host.power(load_after))
        dP = max(0.0, P_after - P_before)
        return float(dP * duration)

    def _assign_task_to_specific_vm(self, task_id, vm_index):
        """提交任务到指定 VM，并在启用模糊模式时同步写入两个影子时间线。

        modal 任务完成事件仍是唯一驱动环境时钟和 ready 状态的事件；optimistic
        与 pessimistic 只按相同任务顺序和 VM 映射重放，不影响 HRL 交互过程。
        """
        vm_id = self.vm_ids[vm_index]
        now = self.current_time
        bw_bps = float(self.vms[vm_id].bw) * 1e6
        pc_mi_s = float(self.vms[vm_id].pc)

        in_bits = self.task_in_bits[task_id]
        out_bits = self.task_out_bits[task_id]
        mi = self.task_mi[task_id]

        t_upload = in_bits / max(bw_bps, 1e-9)
        t_compute = mi / max(pc_mi_s, 1e-9)
        t_download = out_bits / max(bw_bps, 1e-9)

        start_time = max(now, float(self.vm_available_at[vm_index]))
        end_time = start_time + t_upload + t_compute + t_download

        self.task_state[task_id] = "Running"
        self.task_end_time[task_id] = end_time
        self.vm_available_at[vm_index] = end_time
        if task_id in self.ready_task_ids:
            self.ready_task_ids.remove(task_id)
        heapq.heappush(self.event_heap, (end_time, "finish", task_id, vm_index))

        host_id = self.vms[vm_id].host_id
        self._records.append(LoadRecord(start_time, end_time, host_id, pc_mi_s))
        shadow_times = {}
        if getattr(self, "fuzzy_enabled", False):
            for scenario, component_name in (
                ("optimistic", "upper"),
                ("pessimistic", "lower"),
            ):
                scenario_start = self._task_start_time_scenario(
                    task_id, vm_index, scenario
                )
                scenario_end = (
                    scenario_start
                    + self.estimate_task_duration_scenario(
                        task_id, vm_id, scenario
                    )
                )
                self.shadow_vm_available_at[scenario][vm_index] = scenario_end
                self.shadow_task_start_time[scenario][task_id] = scenario_start
                self.shadow_task_end_time[scenario][task_id] = scenario_end
                self.shadow_records[scenario].append(
                    LoadRecord(
                        float(scenario_start),
                        float(scenario_end),
                        int(host_id),
                        self.vms[vm_id].pc.component(component_name),
                    )
                )
                shadow_times[scenario] = (
                    float(scenario_start),
                    float(scenario_end),
                )
            tolerance = 1e-8
            optimistic_end = shadow_times["optimistic"][1]
            pessimistic_end = shadow_times["pessimistic"][1]
            if (
                optimistic_end > float(end_time) + tolerance
                or float(end_time) > pessimistic_end + tolerance
            ):
                raise ValueError(
                    "Assigned task shadow finish order violated for "
                    f"task_id={task_id}, vm_id={vm_id}: "
                    f"optimistic={optimistic_end}, modal={float(end_time)}, "
                    f"pessimistic={pessimistic_end}"
                )
        else:
            # 关闭模糊模式时不计算额外场景；诊断字段退化为 modal，避免改变旧逻辑。
            shadow_times = {
                "optimistic": (float(start_time), float(end_time)),
                "pessimistic": (float(start_time), float(end_time)),
            }
        # 保存全局 task_id 到实际 VM/Host 的映射，并记录 server_type。
        # 评价器据此统计 edge_task_ratio/cloud_task_ratio，而不是根据资源数量推测。
        self.task_assigned_vm[int(task_id)] = int(vm_id)
        self.task_assigned_host[int(task_id)] = int(host_id)
        self.assignment_history.append({
            "task_id": int(task_id),
            "vm_id": int(vm_id),
            "host_id": int(host_id),
            "server_type": str(self.hosts[host_id].server_type),
            "start_time": float(start_time),
            "finish_time": float(end_time),
            "start_time_optimistic": shadow_times["optimistic"][0],
            "finish_time_optimistic": shadow_times["optimistic"][1],
            "start_time_modal": float(start_time),
            "finish_time_modal": float(end_time),
            "start_time_pessimistic": shadow_times["pessimistic"][0],
            "finish_time_pessimistic": shadow_times["pessimistic"][1],
        })

        wf_id, local_id = self.task_meta[task_id]
        task_obj = self.workflows[wf_id].tasks[local_id]
        # 写回原 Task 实例的运行状态与分配信息，保持既有数据模型可观察。
        task_obj.state = "Running"
        task_obj.assigned_server_id = int(host_id)
        task_obj.assigned_vm_pc = float(pc_mi_s)
        task_obj.assigned_vm_id = int(vm_id)
        task_obj.start_processing_time = float(start_time)
        task_obj.end_processing_time = float(end_time)

    def _vm_serial_order(self):
        """按可用时间、处理能力和索引生成稳定的 VM 排序"""
        avail = np.round(self.vm_available_at, 12)
        neg_pc = -np.array([self.vms[vid].pc for vid in self.vm_ids], dtype=np.float64)
        idx = np.arange(self.num_vms, dtype=np.int64)
        order = np.lexsort((idx, neg_pc, avail))
        return order


# ---------------------------------------------------------------------
# 使用 FCFS 缓存设置期限的环境变体
# ---------------------------------------------------------------------
def _avg_task_dur_over_all_vms(tasks, vm_pc, vm_bw_bps):
    """
    计算每个任务在“所有 VM 上 duration 的均值”：
        dur(i,p)=in/bw + mi/pc + out/bw
    返回 avg_dur: shape=(n_tasks,)
    """
    n = len(tasks)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    mi = np.array([float(t.workload_mi) for t in tasks], dtype=np.float64)
    in_bits = np.array(
        [float((getattr(t, "ext_in_bits", 0.0) or 0.0) + (getattr(t, "from_parents_bits", 0.0) or 0.0)) for t in tasks],
        dtype=np.float64,
    )
    out_bits = np.array([float(getattr(t, "out_file_size_sum_bits", 0.0) or 0.0) for t in tasks], dtype=np.float64)

    bw = vm_bw_bps.reshape(1, -1)
    pc = vm_pc.reshape(1, -1)
    dur = (
        in_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
        + mi.reshape(-1, 1) / np.maximum(pc, 1e-12)
        + out_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
    )
    return dur.mean(axis=1)


class HrlFcfsCacheEnv(HrlHeftEnv):
    """
    deadline_mode:
      - "cache_fcfs": 从 deadline_cache_path 查表
      - "none": 不使用 cache，deadline 给一个很大的占位值（通常用于离线预计算）

    workflow deadline:
      wf.deadline = arrival_time + ddl_alpha * makespan_ref
      ddl_alpha 为每个 workflow 独立采样：
        - deadline_alpha_small_prob 概率取 deadline_alpha_small
        - 其余概率取 deadline_alpha_large

    manager reward:
      r_mgr = - [ alpha * phase_delay_cost + (1-alpha) * phase_energy_cost ]
    """

    def __init__(
        self,
        *args,
        deadline_mode: str = "cache_fcfs",
        deadline_cache_path: str = None,
        deadline_cache_strict: bool = True,
        deadline_alpha_small: float = 2.0,
        deadline_alpha_large: float = 3.0,
        deadline_alpha_small_prob: float = 0.2,
        manager_alpha_delay: float = None,
        manager_alpha_energy: float = None,   # 兼容旧代码；若传这个，则自动换算为 delay 权重
        manager_delay_mode: str = "tardiness",
        fuzzy_enabled=False,
        fuzzy_delta1=0.75,
        fuzzy_delta2=1.2,
        fuzzy_energy_uncertainty_weight=1.0,
        fuzzy_deadline_eta=0.95,
        fuzzy_resource_seed=None,
        fuzzy_use_deadline_constraint=True,
        **kwargs,
    ):
        """初始化 FCFS 期限缓存、期限系数和 Manager 奖励权重"""
        self.deadline_mode = str(deadline_mode)
        self.deadline_cache_path = deadline_cache_path
        self.deadline_cache_strict = bool(deadline_cache_strict)

        self.deadline_alpha_small = float(deadline_alpha_small)
        self.deadline_alpha_large = float(deadline_alpha_large)
        self.deadline_alpha_small_prob = float(deadline_alpha_small_prob)

        if manager_alpha_delay is not None and manager_alpha_energy is not None:
            raise ValueError("manager_alpha_delay 与 manager_alpha_energy 只能传一个")

        if manager_alpha_delay is None and manager_alpha_energy is None:
            manager_alpha_delay = 0.5
        elif manager_alpha_delay is None:
            manager_alpha_delay = 1.0 - float(manager_alpha_energy)

        self.manager_alpha_delay = float(manager_alpha_delay)
        self.manager_delay_mode = str(manager_delay_mode).lower()

        self._deadline_cache = None
        super().__init__(
            *args,
            fuzzy_enabled=fuzzy_enabled,
            fuzzy_delta1=fuzzy_delta1,
            fuzzy_delta2=fuzzy_delta2,
            fuzzy_energy_uncertainty_weight=(
                fuzzy_energy_uncertainty_weight
            ),
            fuzzy_deadline_eta=fuzzy_deadline_eta,
            fuzzy_resource_seed=fuzzy_resource_seed,
            fuzzy_use_deadline_constraint=(
                fuzzy_use_deadline_constraint
            ),
            **kwargs,
        )

        if self.deadline_mode == "cache_fcfs":
            if not self.deadline_cache_path:
                raise ValueError("deadline_mode='cache_fcfs' 但未提供 deadline_cache_path")
            self._deadline_cache = self._load_deadline_cache(self.deadline_cache_path)

        self._mgr_prev_total_energy = 0.0
        self._mgr_finished_task_ids = set()

    # ---------------- 期限缓存读写 ----------------
    @staticmethod
    def _load_deadline_cache(path: str):
        """读取期限缓存文件并转换为按随机种子索引的字典"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"deadline cache not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        cache = {}
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            for rec in obj["data"]:
                sd = int(rec["seed"])
                cache[sd] = rec
        elif isinstance(obj, dict) and "seed_to_record" in obj:
            for k, rec in obj["seed_to_record"].items():
                cache[int(k)] = rec
        else:
            raise ValueError("deadline cache json 格式不正确：必须包含 data(list) 或 seed_to_record(dict)")

        return cache

    def _cache_lookup_makespan(self, episode_seed: int, wf_id: int, dax_name: str = None) -> float:
        """按轮次种子和工作流编号查询 FCFS 参考完工时间"""
        if self._deadline_cache is None:
            raise RuntimeError("deadline cache not loaded")

        rec = self._deadline_cache.get(int(episode_seed), None)
        if rec is None:
            msg = f"[deadline-cache] 缺少 seed={episode_seed} 的记录"
            if self.deadline_cache_strict:
                raise KeyError(msg)
            print("WARNING:", msg, "-> fallback makespan_ref=0")
            return 0.0

        ms_list = rec.get("wf_makespans", None)
        if ms_list is None:
            raise ValueError(f"[deadline-cache] seed={episode_seed} record 缺少 wf_makespans")

        if wf_id >= len(ms_list):
            msg = f"[deadline-cache] seed={episode_seed} wf_id={wf_id} 越界（cache里只有 {len(ms_list)} 个workflow）"
            if self.deadline_cache_strict:
                raise IndexError(msg)
            print("WARNING:", msg, "-> fallback makespan_ref=0")
            return 0.0

        if dax_name is not None:
            names = rec.get("wf_dax_names", None)
            if isinstance(names, list) and wf_id < len(names):
                if str(names[wf_id]).casefold() != str(dax_name).casefold():
                    msg = (
                        f"[deadline-cache] seed={episode_seed} wf_id={wf_id} "
                        f"dax mismatch: runtime={dax_name} "
                        f"cache={names[wf_id]}"
                    )
                    if self.deadline_cache_strict:
                        raise ValueError(
                            msg + "; regenerate the FCFS cache with the "
                            "current exact workload-category registry"
                        )
                    print("WARNING:", msg)

        return float(ms_list[wf_id])

    # ---------------- 期限系数采样 ----------------
    def _sample_deadline_alpha_for_workflow(self, wf_id: int) -> float:
        """
        为每个 workflow 独立决定 ddlAlpha
        注意：不能使用 self.rng，否则会污染 DAX 模板抽样序列

        做法：
        - 基于 (episode_seed=random_seed, wf_id) 构造独立 RandomState
        - 再按 deadline_alpha_small_prob 进行采样
        """
        p = float(np.clip(self.deadline_alpha_small_prob, 0.0, 1.0))

        local_seed = (
                             int(self.random_seed) * 1000003
                             + int(wf_id) * 9176
                             + 20260316
                     ) & 0xFFFFFFFF

        rs = np.random.RandomState(local_seed)
        u = float(rs.rand())

        if u < p:
            return float(self.deadline_alpha_small)
        return float(self.deadline_alpha_large)

    # ---------------- 环境重置 ----------------
    def reset(self, *, seed=None, options=None):
        """重置环境并清空 Manager 的阶段能耗与完成任务缓存"""
        self.rng = np.random.RandomState(int(self.random_seed))
        ret = super().reset(seed=seed, options=options)

        self._mgr_prev_total_energy = float(getattr(self, "total_energy", 0.0))
        self._mgr_finished_task_ids = set()

        return ret

    # ---------------- phase 内新完成任务 ----------------
    def _collect_new_finished_task_ids(self):
        """收集相对上一阶段新增的已完成任务编号"""
        old_finished = set(self._mgr_finished_task_ids)
        cur_finished = set()
        new_finished = []

        n_tasks = len(getattr(self, "task_state", []))
        for tid in range(n_tasks):
            if self.task_state[tid] == "Finished":
                cur_finished.add(tid)
                if tid not in old_finished:
                    new_finished.append(tid)

        self._mgr_finished_task_ids = cur_finished
        return new_finished

    # ---------------- phase delay 计算 ----------------
    def _compute_phase_delay(self, finished_tids):
        """根据新增完成任务计算当前阶段的延期或延迟总量"""
        phase_delay = 0.0
        for tid in finished_tids:
            if tid >= len(self.task_baseline_finish):
                continue

            finish_t = float(self.task_end_time[tid])
            ddl_t = float(self.task_baseline_finish[tid])
            late = finish_t - ddl_t

            if self.manager_delay_mode == "tardiness":
                phase_delay += max(0.0, late)
            elif self.manager_delay_mode == "lateness":
                phase_delay += late
            else:
                raise ValueError(f"Unsupported manager_delay_mode: {self.manager_delay_mode}")

        return float(phase_delay)

    # ---------------- 重写 Manager reward ----------------
    def finish_phase_and_advance(self):
        """使用阶段延迟成本和能耗成本重新计算 Manager 奖励"""
        r_old, pinfo = super().finish_phase_and_advance()
        pinfo = dict(pinfo or {})

        cur_total_energy = float(getattr(self, "total_energy", 0.0))
        phase_energy = cur_total_energy - float(self._mgr_prev_total_energy)
        self._mgr_prev_total_energy = cur_total_energy

        new_finished_tids = self._collect_new_finished_task_ids()
        phase_delay = self._compute_phase_delay(new_finished_tids)

        energy_scale = float(getattr(self, "energy_reward_scale", 1.0))
        delay_norm = max(float(getattr(self, "task_baseline_norm", 1.0)), 1e-12)

        phase_energy_cost = phase_energy * energy_scale
        phase_delay_cost = phase_delay / delay_norm

        alpha_delay = float(np.clip(self.manager_alpha_delay, 0.0, 1.0))
        alpha_energy = 1.0 - alpha_delay

        # Manager 奖励为延迟成本与能耗成本的加权负值
        r_manager_new = - (alpha_delay * phase_delay_cost + alpha_energy * phase_energy_cost)

        pinfo["phase_energy"] = float(phase_energy)
        pinfo["phase_delay"] = float(phase_delay)
        pinfo["phase_energy_cost"] = float(phase_energy_cost)
        pinfo["phase_delay_cost"] = float(phase_delay_cost)
        pinfo["manager_alpha_delay"] = float(alpha_delay)
        pinfo["manager_alpha_energy"] = float(alpha_energy)
        pinfo["manager_reward_old"] = float(r_old)
        pinfo["manager_reward_new"] = float(r_manager_new)
        pinfo["manager_new_finished_task_cnt"] = int(len(new_finished_tids))
        pinfo["reward"] = float(r_manager_new)
        pinfo["legacy_reward"] = float(r_manager_new)
        if not self.safe_rl_enabled:
            pinfo["performance_reward"] = float(r_manager_new)
            pinfo["performance_reward_definition"] = (
                "legacy_manager_mixed_reward"
            )

        return float(r_manager_new), pinfo

    # ---------------- 核心：arrival 时设置 workflow deadline ----------------
    def _add_workflow_if_arrived(self):
        """加载已到达工作流并根据 FCFS 缓存设置工作流及任务期限"""
        added = False
        while (
            self.next_arrival_idx < len(self.arrival_times)
            and self.arrival_times[self.next_arrival_idx] <= self.current_time
        ):
            arr_t = self.arrival_times[self.next_arrival_idx]

            path = self._dax_path_for_arrival(self.next_arrival_idx)
            dax_name = os.path.basename(str(path))

            G, tasks = load_workflow_from_dax(
                path,
                seed=self.random_seed + self.next_arrival_idx,
                randomize_payloads=True,
            )
            wf = Workflow(
                workflow_id=len(self.workflows),
                graph=G,
                tasks=tasks,
                arrival_time=arr_t,
                source_device_id=0,
            )
            wf.is_arrived = True
            wf.dax_path = str(path)

            pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)

            self.wf_remaining_tasks[wf.workflow_id] = len(tasks)

            # 每个工作流使用独立随机源采样期限系数
            ddl_alpha = self._sample_deadline_alpha_for_workflow(wf.workflow_id)
            wf.deadline_alpha = float(ddl_alpha)

            if self.deadline_mode == "cache_fcfs":
                makespan_ref = self._cache_lookup_makespan(
                    episode_seed=int(self.random_seed),
                    wf_id=int(wf.workflow_id),
                    dax_name=dax_name,
                )
                wf.ref_makespan = float(makespan_ref)
                wf.deadline = float(arr_t + ddl_alpha * float(makespan_ref))

            elif self.deadline_mode == "none":
                wf.ref_makespan = None
                wf.deadline = float(arr_t + max(self.horizon, 1.0))

            else:
                raise ValueError(f"Unsupported deadline_mode: {self.deadline_mode}")

            # 从工作流期限反向推导各任务的最晚完成时间
            n_local = len(tasks)
            topo, succ = _topo_sort_local(tasks)

            avg_dur = _avg_task_dur_over_all_vms(tasks, self._heft_vm_pc, self._heft_vm_bw_bps)
            D_wf_rel = float(wf.deadline - arr_t)

            LF = np.zeros((n_local,), dtype=np.float64)
            LS = np.zeros((n_local,), dtype=np.float64)
            is_sink = np.array([len(succ[i]) == 0 for i in range(n_local)], dtype=bool)

            for i in range(n_local):
                if is_sink[i]:
                    LF[i] = D_wf_rel
                    LS[i] = LF[i] - avg_dur[i]

            for u in reversed(topo):
                if is_sink[u]:
                    continue
                if len(succ[u]) > 0:
                    min_LS_child = float(np.min(LS[np.array(succ[u], dtype=np.int32)]))
                    LF[u] = min_LS_child
                else:
                    LF[u] = D_wf_rel
                LS[u] = LF[u] - avg_dur[u]

            for local_id in range(n_local):
                abs_deadline = float(arr_t + LF[local_id])
                self.task_baseline_finish.append(abs_deadline)
                # FCFS-cache 环境同样写回绝对子截止期，确保训练/测试加载路径
                # 构造出的 Task 对象都满足 calculate_task_slack 的统一语义。
                tasks[local_id].sub_deadline = abs_deadline

            base = len(self.task_meta)
            for t in tasks:
                self.task_meta.append((wf.workflow_id, t.task_id))
                self.task_state.append("unReady")
                self.task_parents.append(list(t.parents))
                self.task_global_parents.append(
                    [base + int(parent_local_id) for parent_local_id in t.parents]
                )
                self.task_children.append(list(t.children))
                self.task_mi.append(float(t.workload_mi))
                in_bits_total = float((t.ext_in_bits or 0.0) + (t.from_parents_bits or 0.0))
                self.task_in_bits.append(in_bits_total)
                self.task_out_bits.append(float(t.out_file_size_sum_bits))
                self.task_up_rank.append(float(t.upward_rank or 0.0))
                self.task_down_rank.append(float(t.downward_rank or 0.0))
                self.task_ready_time.append(0.0)
                self.task_end_time.append(0.0)
                self.shadow_task_start_time["optimistic"].append(0.0)
                self.shadow_task_start_time["pessimistic"].append(0.0)
                self.shadow_task_end_time["optimistic"].append(0.0)
                self.shadow_task_end_time["pessimistic"].append(0.0)

            for local_id, t in enumerate(tasks):
                if len(t.parents) == 0:
                    gid = base + local_id
                    self.task_state[gid] = "Ready"
                    self.task_ready_time[gid] = self.current_time
                    t.state = "Ready"
                    t.ready_time = float(self.current_time)
                    self.ready_task_ids.append(gid)

            self.workflows.append(wf)
            self.next_arrival_idx += 1
            added = True

        return added


# 保留旧名称供当前 HRL Mix 训练和评估代码兼容使用
CloudWorkflowEnv_VMAgents = HrlFcfsCacheEnv

__all__ = [
    "HrlHeftEnv",
    "HrlFcfsCacheEnv",
    "CloudWorkflowEnv_VMAgents",
    "MANAGER_ACTION_TABLE",
    "NoFeasibleVMError",
    "validate_task_priority_scores",
]
