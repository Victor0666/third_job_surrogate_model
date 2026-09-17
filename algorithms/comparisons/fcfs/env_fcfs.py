# -*- coding: utf-8 -*-
"""
env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15.py

阶段式 HRL 环境（去 Top-K；本版改为启发式 FCFS + 先空闲先服务）：
- 统一能耗记账：LoadRecord + energy_from_records（与既有基线一致）
- 阶段流程：
  1) Manager 逻辑不再使用，环境内部直接按 FCFS 构建“本阶段任务顺序”
     （先到先服务：按 task_ready_time 升序；同 ready_time 先随机再排序）
  2) Worker 循环：每次仅给“队首 1 个任务”的观测/掩码，
     下层动作改为环境内部启发式：
       * 选取 earliest-available VM（vm_available_at 最小）
       * 若有多个 VM 在最早时间并列，则在这些 VM 中随机选择一台
       * 调用 worker_assign(vm_index) 完成实际分配与子 deadline 奖励计算
  3) 阶段末推进到下一个决策点，仅结算能耗差分 r_energy_phase = -ΔE×scale
     * 上层阶段奖励 r_manager_raw = r_energy_phase / sum_mi_phase（能耗强度）

Deadline 设计：
  * 对每条新到达的工作流：
      - 以“平均处理能力 + 平均带宽”为资源模型，做 HEFT-like 前向 DP，
        得到每个任务的最早完成时间 EF_i 与 makespan M_avg；
      - 工作流总 deadline：D_wf = arrival_time + alpha * M_avg
        （系数 alpha 默认 1.5，可调）。
  * 子 deadline 倒推：
      - 在局部时间轴（相对 arrival_time）上，以 D_wf_rel=alpha*M_avg 为终点，
        按逆拓扑顺序计算每个任务的最新完成时间 LF_i：
          · sink 节点：LF_i = D_wf_rel
          · 一般节点：LF_i = min_j LS_j, 其中 LS_j = LF_j - P_j
          · P_i 为任务 i 的“平均执行+通信时间”
      - 再强制 LF_i >= EF_i，保证可行性；
      - 绝对子 deadline：D_i = arrival_time + LF_i，
        存入 self.task_baseline_finish[gid]，Worker 奖励与 EDF/slack 直接使用。

软卡死保险丝：
  * 连续 zero_assign_streak ≥ Z 且有决策点：
      - 原逻辑为强制 FCFS 分配 1 个任务后推进，这里保留作为保护；
  * 无事件/就绪/运行/到达且仍未达目标工作流：
      - 连续 N 次后强制结束 episode

对外训练可用接口：
- get_worker_state_for_next_assignment() -> ({obs, mask}, has_next)
- worker_assign(vm_index) -> r_worker_task_raw
- worker_assign_first_idle() -> r_worker_task_raw  （本版新增：先空闲先服务）
- finish_phase_and_advance() -> (r_manager_raw, info)
- get_manager_state() / set_manager_combo(weights) 仍保留以兼容旧代码，但启发式版本不会再用。
"""

import heapq
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces

# === 统一能耗与工作流 ===
from common.workflow_opt import LoadRecord, energy_from_records, Workflow
from common.read_xml_opt_Tsize import load_workflow_from_dax, poisson_arrival_times
from common.resource_opt import create_cluster


def _safe_div(a, b, eps=1e-9):
    return float(a) / float(b if abs(b) > eps else (eps if b >= 0 else -eps))


class CloudWorkflowEnv_VMAgents(gym.Env):
    metadata = {"render.modes": ["human"]}

    # ---------------- 初始化 ----------------
    def __init__(
        self,
        dax_paths,
        horizon=36000.0,
        arrival_lambda=0.1,
        random_seed=0,
        # 观测/任务槽
        max_ready_tasks=32,               # 仅用于“启发式评分批处理”的上限；无 Top-K 截断
        normalize=True,
        # 资源
        num_cloud_hosts=2,
        num_edge_hosts=1,
        cloud_vms_per_host=(9, 8),
        edge_vms_per_host=(8,),
        cloud_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        edge_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        cloud_bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        edge_bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        # DAX 模板选择
        dax_probs=None,
        # 状态包含角度
        state_include_vm=True,
        state_include_host=True,
        state_include_queue=True,
        # 模板组合权重（5 维：FCFS/SJF/MCF/HUR/EDF），启发式版本中不再用于排序，仅保留接口
        combo_weights=None,
        # 每个 episode 完成的工作流个数
        workflows_per_episode=8,
    ):
        super().__init__()

        # 基础参数
        self.rng = np.random.RandomState(random_seed)
        self.horizon = float(horizon)
        self.arrival_lambda = float(arrival_lambda)
        self.dax_paths = list(dax_paths)
        self.random_seed = int(random_seed)
        self.normalize = bool(normalize)

        # 奖励尺度参数
        self.energy_reward_scale = 1e-3   # r_energy = -ΔE * scale
        self.task_baseline_beta = 0.05    # r_worker_task 原始延迟奖励的缩放
        self.task_baseline_norm = 300.0   # 用于归一和裁剪，单位：秒

        # 结束条件
        self.workflows_per_episode = int(workflows_per_episode) if workflows_per_episode is not None else None

        # DAX 模板概率
        self.dax_probs = None
        if dax_probs is not None:
            probs = np.asarray(dax_probs, dtype=float).copy()
            assert len(probs) == len(self.dax_paths), "dax_probs 长度需与 dax_paths 一致"
            probs = np.clip(probs, 0.0, None)
            if probs.sum() <= 0:
                probs[:] = 1.0
            self.dax_probs = probs / probs.sum()

        # 集群
        self.hosts, self.vms = create_cluster(
            num_cloud_hosts=num_cloud_hosts,
            num_edge_hosts=num_edge_hosts,
            cloud_vms_per_host=cloud_vms_per_host,
            edge_vms_per_host=edge_vms_per_host,
            cloud_pc_tiers=cloud_pc_tiers,
            edge_pc_tiers=edge_pc_tiers,
            cloud_bw_tiers=cloud_bw_tiers,
            edge_bw_tiers=edge_bw_tiers,
        )
        self.host_ids = sorted(self.hosts.keys())
        self.vm_ids = sorted(self.vms.keys())
        self.num_hosts = len(self.hosts)
        self.num_vms = len(self.vm_ids)
        self.vm_host = np.array([self.vms[vid].host_id for vid in self.vm_ids], dtype=np.int32)

        # 评分槽与启发式
        self.max_ready = int(max_ready_tasks)
        self.task_heur_dim = 5  # [FCFS, SJF, MCF, HUR, EDF]

        # 状态包含
        self.state_include_vm = bool(state_include_vm)
        self.state_include_host = bool(state_include_host)
        self.state_include_queue = bool(state_include_queue)

        # 模板权重（不再实际参与排序，但保留以兼容旧接口）
        if combo_weights is None:
            self.combo_weights = np.ones(self.task_heur_dim, dtype=np.float32) / float(self.task_heur_dim)
        else:
            w = np.asarray(combo_weights, dtype=float).reshape(-1)
            assert w.size == self.task_heur_dim, "combo_weights 维度需等于 5（FCFS/SJF/MCF/HUR/EDF）"
            if np.allclose(w.sum(), 0.0):
                w = np.ones_like(w)
            self.combo_weights = (w / w.sum()).astype(np.float32)

        # 观测维度（Worker 单智能体）：
        #   Block1: 当前任务特征（12）
        #   Block2: 队列上下文（10）
        #   VMBlock: 5 × num_vms
        #   HostBlock: 5 × num_hosts
        self.block1_dim = 12
        self.block2_dim = 10
        self.vm_feat_dim = 5
        self.host_feat_dim = 5
        self.worker_obs_dim = (
            self.block1_dim + self.block2_dim
            + self.vm_feat_dim * self.num_vms
            + self.host_feat_dim * self.num_hosts
        )
        self.worker_act_dim = self.num_vms  # 选择一台空闲 VM（无 No-op 动作）
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.worker_obs_dim,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(self.worker_act_dim,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(self.worker_act_dim)

        # 能耗统计（与 MARL 统一）
        self.total_energy = 0.0         # 当前 episode 累计（J）
        self.lifetime_energy = 0.0      # 跨 episode 累计（J）
        self._records = []              # List[LoadRecord]
        self._energy_cache = 0.0        # 已计入的能量缓存（裁剪到 current_time）

        # 保险丝阈值
        self._fuse_zero_assign_limit = 50
        self._fuse_no_event_limit = 10

        # 运行态缓冲
        self._reset_internal_buffers()

    # ---------------- HRL 接口 ----------------
    def set_manager_combo(self, weights):
        """
        保留旧接口：设置任务排序模板权重。
        启发式版本中，真正排序逻辑已经改为 FCFS，本函数仅更新内部记录。
        """
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        assert w.size == self.task_heur_dim, "combo 权重长度应为 5"
        if np.allclose(w.sum(), 0.0):
            w = np.ones_like(w)
        self.combo_weights = (w / w.sum()).astype(np.float32)
        # 每次 Manager 变更，标记下阶段重建队列
        self._phase_started = False
        self._phase_tasks = []

    def get_manager_state(self):
        """
        保留旧接口：返回上层状态。启发式版本不会使用，但用于兼容旧训练脚本。
        """
        now = getattr(self, "current_time", 0.0)

        ready_cnt   = float(len(self.ready_task_ids))
        waiting_cnt = float(sum(1 for s in self.task_state if s == "unReady"))
        running_cnt = float(sum(1 for s in self.task_state if s == "Running"))

        waits = [max(0.0, now - self.task_ready_time[tid]) for tid in self.ready_task_ids]
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        max_wait = float(np.max(waits)) if len(waits) > 0 else 0.0

        ready_ratio   = ready_cnt   / max(1.0, self.max_ready)
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
        bw_bps  = bw_mean * 1e6

        slacks = []
        for tid in self.ready_task_ids:
            exp_t = (
                self.task_in_bits[tid]  / max(bw_bps, 1e-9)
                + self.task_mi[tid]     / max(pc_mean, 1.0)
                + self.task_out_bits[tid]/ max(bw_bps, 1e-9)
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
        return vec

    # ------ 阶段式 Worker 接口 ------
    def get_worker_state_for_next_assignment(self):
        """
        返回：obs(1D), mask(1D), has_next: 是否还有“可立即分配”的任务（且存在空闲 VM）
        * 若本阶段尚未开始，将基于 FCFS 构建“全量就绪任务顺序”
        * 若无空闲 VM 或无任务→ has_next=False
        """
        if self.done_flag:
            return {
                "obs": np.zeros(self.worker_obs_dim, np.float32),
                "mask": np.zeros(self.worker_act_dim, np.float32),
            }, False

        # 若未开局，先构建本阶段任务顺序
        if not self._phase_started:
            self._phase_prepare_tasks()
            self._phase_started = True
            self._phase_assign_cnt = 0
            self._phase_assigned_tids = []
            self._phase_size_sum_mi = 0.0

        vm_idle = self._vm_idle_mask()
        if not np.any(vm_idle) or len(self._phase_tasks) == 0:
            return {
                "obs": np.zeros(self.worker_obs_dim, np.float32),
                "mask": np.zeros(self.worker_act_dim, np.float32),
            }, False

        # 取队首 1 个任务作为“当前任务”
        self._phase_current_tid = int(self._phase_tasks.pop(0))
        obs, mask = self._build_worker_obs_for_task(self._phase_current_tid)
        return {"obs": obs.astype(np.float32), "mask": mask.astype(np.float32)}, True

    def worker_assign(self, vm_index: int):
        """
        Worker 为“当前任务”选择一台空闲 VM。返回该任务的“原始延迟奖励”（未归一化）。

        r_worker_task_raw = beta * clip(deadline - finish_time, [-norm, +norm]) / norm
        其中：
          - beta = self.task_baseline_beta  (默认 0.05)
          - norm = self.task_baseline_norm (默认 300.0)
        """
        assert hasattr(self, "_phase_current_tid") and self._phase_current_tid is not None, \
            "请先调用 get_worker_state_for_next_assignment() 取得当前任务"

        tid = int(self._phase_current_tid)
        self._phase_current_tid = None

        vm_idx = int(vm_index)
        if vm_idx < 0 or vm_idx >= self.num_vms:
            # 非法则直接返回 0（训练侧应通过 mask 避免）
            return 0.0
        if not self._vm_idle_mask()[vm_idx]:
            # 非空闲：返回轻微惩罚（理论上 mask 会屏蔽）
            return -0.01

        # 执行分配并得到结束时间
        self._assign_task_to_specific_vm(tid, vm_idx)
        self._phase_assign_cnt += 1
        self._phase_assigned_tids.append(tid)
        self._phase_size_sum_mi += float(self.task_mi[tid])

        # 逐任务延迟奖励：基于“倒推子 deadline” + 对称裁剪
        if tid < len(self.task_baseline_finish):
            deadline = float(self.task_baseline_finish[tid])
        else:
            deadline = self.current_time
        finish_t = float(self.task_end_time[tid])

        delta = deadline - finish_t  # >0 提前完成；<0 延迟
        clip_th = float(self.task_baseline_norm)  # 例如 300 秒
        if clip_th > 0.0:
            if delta > clip_th:
                delta = clip_th
            elif delta < -clip_th:
                delta = -clip_th

        r = self.task_baseline_beta * (delta / max(self.task_baseline_norm, 1e-9))
        return float(r)

    def worker_assign_first_idle(self):
        """
        启发式版本：对当前任务采用“先空闲先服务”（最早可用 VM），
        若有多个 VM 在最早时间并列，则在这些 VM 中随机选择一台。

        返回：与 worker_assign(vm_index) 一样的 r_worker_task_raw。
        """
        assert hasattr(self, "_phase_current_tid") and self._phase_current_tid is not None, \
            "请先调用 get_worker_state_for_next_assignment() 取得当前任务"

        # 当前要分配的任务 ID（由 get_worker_state_for_next_assignment() 设置）
        tid = int(self._phase_current_tid)

        vm_idle = self._vm_idle_mask()
        idle_indices = np.where(vm_idle)[0]
        if len(idle_indices) == 0:
            # 理论上在有决策点时应当有空闲 VM，这里返回 0 作为兜底
            return 0.0

        # 找到最早可用时间（vm_available_at 最小）
        avail_times = self.vm_available_at[idle_indices]
        min_t = float(np.min(avail_times))

        # 所有在最早时间并列的 VM 都作为候选，在其中随机选一台
        eps = 1e-12
        candidate_vm_idxs = [int(idx) for idx, t in zip(idle_indices, avail_times) if abs(t - min_t) <= eps]
        if len(candidate_vm_idxs) == 0:
            # 理论上不会发生，如果发生就退化为在所有 idle 中随机
            candidate_vm_idxs = [int(i) for i in idle_indices]

        chosen_vm_idx = int(self.rng.choice(candidate_vm_idxs))

        # 复用原有 worker_assign 的逻辑（包含子deadline奖励计算）
        return float(self.worker_assign(chosen_vm_idx))

    def finish_phase_and_advance(self):
        """
        阶段末推进到下一决策点，仅结算“能耗差分”（r_energy_phase = -ΔE * scale）。
        上层阶段奖励 r_manager_raw = r_energy_phase / sum_mi_phase（能耗强度；负越好）。
        返回：(r_manager_raw, info)
        """
        # 保险丝-1：若本阶段 assign_cnt=0 且存在决策点，累计计数并在阈值触发后强制分配 1 个任务
        if self._phase_assign_cnt == 0 and self._has_decision_point():
            self._fuse_zero_assign_streak += 1
            if self._fuse_zero_assign_streak >= self._fuse_zero_assign_limit:
                self._fuse_zero_assign_streak = 0
                # 强制 FCFS：ready 中第一个任务 → 最早空闲 VM
                if len(self.ready_task_ids) > 0 and np.any(self._vm_idle_mask()):
                    forced_tid = int(self.ready_task_ids[0])
                    forced_vm = int(self._vm_serial_order()[0])
                    if self._vm_idle_mask()[forced_vm]:
                        self._assign_task_to_specific_vm(forced_tid, forced_vm)
                        self._phase_assign_cnt += 1
                        self._phase_assigned_tids.append(forced_tid)
                        self._phase_size_sum_mi += float(self.task_mi[forced_tid])
        else:
            self._fuse_zero_assign_streak = 0

        # 推进到下一决策点，仅累计能耗差分（负号）
        r_energy_phase = self._advance_until_decision_energy_only()

        # 上层阶段奖励：能耗强度（负越好）
        if self._phase_assign_cnt > 0 and self._phase_size_sum_mi > 0:
            r_manager_raw = r_energy_phase / float(self._phase_size_sum_mi)
        else:
            r_manager_raw = 0.0

        info = {
            "current_time": self.current_time,
            "assign_cnt": int(self._phase_assign_cnt),
            "phase_size_sum_mi": float(self._phase_size_sum_mi),
            "r_energy_phase": float(r_energy_phase),
            "phase_assigned_tids": list(self._phase_assigned_tids),
        }

        # 清理阶段缓存
        self._phase_started = False
        self._phase_tasks = []
        self._phase_assigned_tids = []
        self._phase_assign_cnt = 0
        self._phase_size_sum_mi = 0.0
        self._phase_current_tid = None

        return float(r_manager_raw), info

    # ---------------- Gym reset/render ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal_buffers()

        # 采用基于 n_max 的泊松到达：只生成本 episode 所需的工作流数
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

        # 推进到第一个“可能出现决策点”的时刻（只累计能耗，不给奖励）
        _ = self._advance_until_decision_energy_only()

        # 初始化阶段标记
        self._phase_started = False
        self._phase_tasks = []
        self._phase_assigned_tids = []
        self._phase_assign_cnt = 0
        self._phase_size_sum_mi = 0.0
        self._phase_current_tid = None

        # 保险丝计数
        self._fuse_zero_assign_streak = 0
        self._fuse_no_event_streak = 0

        info = {
            "current_time": getattr(self, "current_time", 0.0),
            "first_arrival_time": self.first_arrival_time,
            "wf_target": self.workflows_per_episode if self.workflows_per_episode is not None else None,
        }
        # 由于本环境的“动作/状态”在阶段接口完成，这里返回一个占位 obs/mask 方便打印维度
        obs = {
            "obs": np.zeros(self.worker_obs_dim, np.float32),
            "mask": np.zeros(self.worker_act_dim, np.float32),
        }
        return obs, info

    def render(self):
        idle_hosts = int(np.sum(self._host_idle_mask()))
        print(f"[t={self.current_time:.2f}] ready={len(self.ready_task_ids)} idleHosts={idle_hosts}")

    # ---------------- 运行态缓冲 ----------------
    def _reset_internal_buffers(self):
        self.current_time = 0.0
        self.done_flag = False

        self.workflows = []
        self.arrival_times = []
        self.next_arrival_idx = 0

        # 任务属性
        self.task_meta = []
        self.task_state = []
        self.task_parents = []
        self.task_children = []
        self.task_mi = []
        self.task_in_bits = []
        self.task_out_bits = []
        self.task_up_rank = []
        self.task_down_rank = []
        self.task_ready_time = []
        self.task_end_time = []

        self.ready_task_ids = []
        self.event_heap = []  # (time, "finish", task_id, vm_idx)
        self.vm_available_at = np.zeros(self.num_vms, dtype=np.float64)

        # 工作流完成统计
        self.completed_workflows = 0
        self.wf_remaining_tasks = {}

        # “子 deadline”数组
        self.task_baseline_finish = []

        # 能耗统计缓存
        self.total_energy = 0.0
        self._records = []
        self._energy_cache = 0.0

    # ---------------- 调度逻辑与停止条件 ----------------
    def _no_more_arrivals(self):
        return self.next_arrival_idx >= len(self.arrival_times)

    def _add_workflow_if_arrived(self):
        """
        按当前时间 current_time 检查是否有工作流到达：
        - 解析 DAX，构建 Workflow + Task 属性
        - 计算 up/down rank
        - 基于“平均执行+通信时间”估计：
            · 前向：EF_i 与 makespan M_avg
            · 工作流总 deadline: D_wf = arrival + alpha * M_avg
            · 逆向：每个任务的子 deadline（倒推 LF_i），保证 LF_i>=EF_i
        - 将子 deadline 写入 self.task_baseline_finish[gid]
        """
        added = False
        while (
            self.next_arrival_idx < len(self.arrival_times)
            and self.arrival_times[self.next_arrival_idx] <= self.current_time
        ):
            arr_t = self.arrival_times[self.next_arrival_idx]

            # 模板选择
            if self.dax_probs is None:
                idx = self.rng.randint(len(self.dax_paths))
            else:
                idx = self.rng.choice(len(self.dax_paths), p=self.dax_probs)
            path = self.dax_paths[idx]

            # 读取工作流
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

            # 预估 rank（仍用平均 pc/bw）
            pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)

            self.wf_remaining_tasks[wf.workflow_id] = len(tasks)

            # 展平并填充全局数组
            base = len(self.task_meta)
            for t in tasks:
                self.task_meta.append((wf.workflow_id, t.task_id))
                self.task_state.append("unReady")
                self.task_parents.append(list(t.parents))
                self.task_children.append(list(t.children))
                self.task_mi.append(float(t.workload_mi))
                in_bits_total = float((t.ext_in_bits or 0.0) + (t.from_parents_bits or 0.0))
                self.task_in_bits.append(in_bits_total)
                self.task_out_bits.append(float(t.out_file_size_sum_bits))
                self.task_up_rank.append(float(t.upward_rank or 0.0))
                self.task_down_rank.append(float(t.downward_rank or 0.0))
                self.task_ready_time.append(0.0)
                self.task_end_time.append(0.0)

            # === 基于平均执行+通信时间的 HEFT-like 估计 + 倒推子 deadline ===
            n_local = len(tasks)
            bw_mean_bps = bw_mean * 1e6 if n_local > 0 else 1.0

            # 建立 children（succ）与拓扑序
            indeg = [len(t.parents) for t in tasks]
            children = [[] for _ in range(n_local)]
            for i, t in enumerate(tasks):
                for p in t.parents:
                    children[p].append(i)

            q = [i for i in range(n_local) if indeg[i] == 0]
            topo = []
            while q:
                u = q.pop(0)
                topo.append(u)
                for v in children[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
            if len(topo) != n_local:
                # 若有环或异常，退化为顺序 0..n-1
                topo = list(range(n_local))

            # 1) 前向：最早完成时间 EF_i + 节点平均处理时间 P_i
            local_EF = [0.0] * n_local   # 相对 arrival_time 的最早完成时间
            P = [0.0] * n_local          # 平均执行+通信时间
            for i in topo:
                ti = tasks[i]
                in_bits_total = float((ti.ext_in_bits or 0.0) + (ti.from_parents_bits or 0.0))
                dur = (
                    in_bits_total / max(bw_mean_bps, 1e-9)
                    + float(ti.workload_mi) / max(pc_mean, 1.0)
                    + float(ti.out_file_size_sum_bits) / max(bw_mean_bps, 1e-9)
                )
                P[i] = dur
                est_start = 0.0
                if ti.parents:
                    est_start = max(local_EF[p] for p in ti.parents)
                local_EF[i] = est_start + dur

            M_avg = max(local_EF) if n_local > 0 else 0.0  # HEFT-like makespan（平均资源模型）
            alpha = 1.5
            D_wf_rel = alpha * M_avg                        # 相对到达时刻的总 DDL
            D_wf_abs = arr_t + D_wf_rel
            wf.deadline = D_wf_abs                          # 保存到 Workflow，供 EDF/slack 使用

            # 2) 逆向：最新完成时间 LF_i、子 deadline 倒推
            LF = [0.0] * n_local  # latest finish (relative)
            LS = [0.0] * n_local  # latest start  (relative)

            # sink 节点（无子任务）
            succ = children
            is_sink = [len(succ[i]) == 0 for i in range(n_local)]
            for i in range(n_local):
                if is_sink[i]:
                    LF[i] = D_wf_rel
                    LS[i] = LF[i] - P[i]

            # 逆拓扑：其余任务的 LF/LS
            for i in reversed(topo):
                if is_sink[i]:
                    continue
                if len(succ[i]) > 0:
                    min_LS_child = min(LS[j] for j in succ[i])
                    LF[i] = min_LS_child
                else:
                    # 理论上不会走到，否则按 sink 处理
                    if LF[i] <= 0.0:
                        LF[i] = D_wf_rel
                LS[i] = LF[i] - P[i]

            # 3) 保证子 deadline 不早于最早完成时间 EF_i
            for i in range(n_local):
                lf = max(LF[i], local_EF[i])
                LF[i] = lf

            # 4) 转为绝对子 deadline，并写入全局数组
            for local_id in range(n_local):
                abs_deadline = arr_t + LF[local_id]
                self.task_baseline_finish.append(abs_deadline)

            # 无父任务置 Ready
            for local_id, t in enumerate(tasks):
                if len(t.parents) == 0:
                    gid = base + local_id
                    self.task_state[gid] = "Ready"
                    self.task_ready_time[gid] = self.current_time
                    self.ready_task_ids.append(gid)

            self.workflows.append(wf)
            self.next_arrival_idx += 1
            added = True
        return added

    def _vm_idle_mask(self):
        return self.vm_available_at <= self.current_time + 1e-9

    def _host_idle_mask(self):
        idle = np.zeros(self.num_hosts, dtype=bool)
        for h in self.host_ids:
            has_idle = False
            for j, vid in enumerate(self.vm_ids):
                if self.vms[vid].host_id == h and self.vm_available_at[j] <= self.current_time + 1e-9:
                    has_idle = True
                    break
            idle[self.host_ids.index(h)] = has_idle
        return idle

    def _has_decision_point(self):
        return (len(self.ready_task_ids) > 0) and np.any(self._vm_idle_mask())

    def _episode_should_end(self):
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

    # ---------------- 时间推进（仅能耗） ----------------
    def _energy_reward_to_current_time(self) -> float:
        # ΔE = E(current_time) - cache；r_energy = -ΔE * scale
        started = len(self.workflows) > 0
        if not started:
            return 0.0
        if self.workflows_per_episode is not None:
            counting = self.completed_workflows < self.workflows_per_episode
        else:
            counting = any(s != "Finished" for s in self.task_state)
        if not counting:
            return 0.0

        t = self.current_time
        clipped = []
        for r in self._records:
            if r.end_time <= 0.0:
                continue
            if r.start_time >= t:
                continue
            end = min(r.end_time, t)
            if end <= r.start_time:
                continue
            clipped.append(LoadRecord(r.start_time, end, r.server_id, r.vm_pc))

        E = energy_from_records(clipped, self.hosts)
        dE = max(0.0, E - self._energy_cache)
        self._energy_cache = E
        self.total_energy += dE
        self.lifetime_energy += dE
        return -(dE * self.energy_reward_scale)

    def _advance_until_decision_energy_only(self) -> float:
        """
        推进到下一个决策点，仅累计并返回 r_energy（能耗差分）。
        完成事件与解锁照常执行。
        """
        r_energy = 0.0
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
                # 无事件/无到达：保险丝-2（连续多次后强制结束）
                self._fuse_no_event_streak += 1
                if self._episode_should_end() or self._fuse_no_event_streak >= self._fuse_no_event_limit:
                    self.done_flag = True
                    break
                # 为避免原地打转，轻推进一个极小步并重试
                self.current_time += 1e-9
                r_energy += self._energy_reward_to_current_time()
                continue

            next_t = min(next_times)
            if next_t <= self.current_time + 1e-12:
                next_t = self.current_time + 1e-9

            # 前进到 next_t：能耗积分 + 完成任务（状态更新/解锁）
            self.current_time = next_t
            r_energy += self._energy_reward_to_current_time()
            self._process_finish_events_at_current_time()

            if self._has_decision_point():
                self._fuse_no_event_streak = 0
                break
            if self._episode_should_end() or self.current_time >= self.horizon:
                self.done_flag = True
                break
        return r_energy

    def _process_finish_events_at_current_time(self):
        # 完成事件与解锁 ready 的逻辑
        while len(self.event_heap) > 0 and self.event_heap[0][0] <= self.current_time + 1e-12:
            _, etype, task_id, vm_idx = heapq.heappop(self.event_heap)
            if etype != "finish":
                continue
            self.task_state[task_id] = "Finished"
            self.task_end_time[task_id] = self.current_time

            # 工作流完成统计
            wf_id, _ = self.task_meta[task_id]
            if wf_id in self.wf_remaining_tasks:
                self.wf_remaining_tasks[wf_id] -= 1
                if self.wf_remaining_tasks[wf_id] == 0:
                    self.completed_workflows += 1

            # 解锁子任务
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
                        self.ready_task_ids.append(child_gid)

        # 清理不再 Ready 的
        self.ready_task_ids = [tid for tid in self.ready_task_ids if self.task_state[tid] == "Ready"]

    # ---------------- 阶段构建与观测 ----------------
    def _compute_task_heuristics_for_ready(self):
        """
        原始版本：基于多种启发式评分（FCFS/SJF/MCF/HUR/EDF）。
        启发式 FCFS 版本中不再被 _phase_prepare_tasks 使用，但保留以便需要时可以切回启发式线性组合。
        """
        now = self.current_time
        sel_ids = list(self.ready_task_ids)  # 全量

        if len(sel_ids) == 0:
            return [], np.zeros((0, self.task_heur_dim), dtype=np.float32)

        pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))   # MI/s
        bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))   # Mb/s
        bw_mean_bps = bw_mean * 1e6

        time_scale = 1.0 + np.mean(np.array(self.task_mi, dtype=np.float64) / max(pc_mean, 1.0))

        feats = np.zeros((len(sel_ids), self.task_heur_dim), dtype=np.float32)
        for i, tid in enumerate(sel_ids):
            exp_t = (
                self.task_in_bits[tid]  / max(bw_mean_bps, 1e-9)
                + self.task_mi[tid]     / max(pc_mean, 1.0)
                + self.task_out_bits[tid]/ max(bw_mean_bps, 1e-9)
            )
            fcfs = max(0.0, now - self.task_ready_time[tid])  # 等待越久越大
            sjf  = 1.0 / (exp_t + 1.0)                        # 预计总时长越短越大
            mcf  = float(len(self.task_children[tid]))        # 后继越多越大
            hur  = float(self.task_up_rank[tid])              # 向上秩
            wf_idx, _ = self.task_meta[tid]
            wf = self.workflows[wf_idx]
            if getattr(wf, "deadline", None) is not None:
                slack = max(0.0, wf.deadline - (now + exp_t))
                edf = 1.0 / (slack + 1.0)                     # 紧急程度
            else:
                edf = 0.0
            if self.normalize:
                fcfs = fcfs / max(self.horizon, 1.0)
                hur  = hur  / max(time_scale, 1.0)
                mcf  = mcf  / max(1.0, np.mean([len(ch) for ch in self.task_children]) + 1e-6)
            feats[i] = np.array([fcfs, sjf, mcf, hur, edf], dtype=np.float32)
        return sel_ids, feats

    def _phase_prepare_tasks(self):
        """
        基于 FCFS 生成“本阶段的任务顺序”（全量 ready，先到先服务；同 ready_time 随机打乱）。
        """
        # 先把当前时间点之前应到达的工作流加入进来
        self._add_workflow_if_arrived()

        # 全量 ready 任务
        ready_ids = list(self.ready_task_ids)
        if len(ready_ids) == 0:
            self._phase_tasks = []
            return

        # 为了“同一 ready_time 随机”，先随机打乱，再按照 ready_time 稳定排序
        self.rng.shuffle(ready_ids)
        ready_ids.sort(key=lambda tid: self.task_ready_time[tid])

        # 作为本阶段任务序列（队首 → 先服务）
        self._phase_tasks = [int(tid) for tid in ready_ids]

    def _build_worker_obs_for_task(self, tid: int):
        """
        拼接 Worker 单智能体观测：
        Block1(12) + Block2(10) + VMBlock(5×NV) + HostBlock(5×NH)
        并返回 mask（仅空闲 VM 为1；若存在空闲 VM，则不提供 No-op）
        """
        now = self.current_time

        # === Block2: 队列上下文（先算，复用） ===
        ready_cnt   = float(len(self.ready_task_ids))
        waiting_cnt = float(sum(1 for s in self.task_state if s == "unReady"))
        running_cnt = float(sum(1 for s in self.task_state if s == "Running"))
        waits = [max(0.0, now - self.task_ready_time[x]) for x in self.ready_task_ids]
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        max_wait = float(np.max(waits)) if len(waits) > 0 else 0.0

        sum_mi = float(np.sum([self.task_mi[x] for x in self.ready_task_ids])) if len(self.ready_task_ids) > 0 else 0.0
        urgent_cnt = 0
        pc_mean = float(np.mean([self.vms[v].pc for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_mean = float(np.mean([self.vms[v].bw for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_bps  = bw_mean * 1e6
        slacks = []
        for x in self.ready_task_ids:
            exp_t = (
                self.task_in_bits[x]  / max(bw_bps, 1e-9)
                + self.task_mi[x]     / max(pc_mean, 1.0)
                + self.task_out_bits[x]/ max(bw_bps, 1e-9)
            )
            wf_idx, _ = self.task_meta[x]
            wf = self.workflows[wf_idx]
            dl = getattr(wf, "deadline", None)
            if dl is not None:
                s = dl - (now + exp_t)
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

        # === Block1: 当前任务特征（12） ===
        in_bits  = self.task_in_bits[tid]
        out_bits = self.task_out_bits[tid]
        mi       = self.task_mi[tid]
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
                bw_bps_vm  = self.vms[vm_id].bw * 1e6
                pc_mi_s    = self.vms[vm_id].pc
                t_upload   = in_bits  / max(bw_bps_vm, 1e-9)
                t_compute  = mi       / max(pc_mi_s, 1e-9)
                t_download = out_bits / max(bw_bps_vm, 1e-9)
                durations.append(t_upload + t_compute + t_download)
            est_best = float(np.min(durations))
            est_mean = float(np.mean(durations))

        wf_idx, _ = self.task_meta[tid]
        wf = self.workflows[wf_idx]
        if getattr(wf, "deadline", None) is not None:
            slack_on_best = wf.deadline - (now + est_best)
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

        # === VMBlock + HostBlock ===
        mean_vm_pc = max(1.0, np.mean([self.vms[v].pc for v in self.vms]))
        mean_vm_bw = max(1.0, np.mean([self.vms[v].bw for v in self.vms]))

        vm_block = np.zeros((self.num_vms, self.vm_feat_dim), dtype=np.float32)
        for j, vid in enumerate(self.vm_ids):
            vm = self.vms[vid]
            idle_flag   = 1.0 if self.vm_available_at[j] <= now + 1e-12 else 0.0
            avail_delay = max(0.0, float(self.vm_available_at[j] - now))
            bw_bps_vm   = vm.bw * 1e6
            t_upload    = in_bits  / max(bw_bps_vm, 1e-9)
            t_compute   = mi       / max(vm.pc, 1e-9)
            t_download  = out_bits / max(bw_bps_vm, 1e-9)
            pred_finish = max(now, float(self.vm_available_at[j])) + t_upload + t_compute + t_download
            vm_block[j] = np.array([
                idle_flag,
                avail_delay / max(self.horizon, 1.0),
                vm.pc / mean_vm_pc,
                vm.bw / mean_vm_bw,
                pred_finish / max(self.horizon, 1.0),
            ], dtype=np.float32)

        host_stats = {
            "active_pc": {h: 0.0 for h in self.host_ids},
            "idle_vm":   {h: 0 for h in self.host_ids},
            "soonest":   {h: None for h in self.host_ids},
            "running":   {h: 0 for h in self.host_ids},
            "vm_cnt":    {h: 0 for h in self.host_ids},
        }
        for j, vid in enumerate(self.vm_ids):
            h = self.vms[vid].host_id
            host_stats["vm_cnt"][h] += 1
            if self.vm_available_at[j] > now + 1e-12:
                host_stats["active_pc"][h] += self.vms[vid].pc
                host_stats["running"][h]   += 1
            else:
                host_stats["idle_vm"][h]   += 1
            if host_stats["soonest"][h] is None or self.vm_available_at[j] < host_stats["soonest"][h]:
                host_stats["soonest"][h] = self.vm_available_at[j]

        host_block = np.zeros((self.num_hosts, self.host_feat_dim), dtype=np.float32)
        for hi, h in enumerate(self.host_ids):
            vm_cnt = max(1, host_stats["vm_cnt"][h])
            total_pc_n          = self.hosts[h].total_pc / max(
                1.0, np.mean([self.hosts[x].total_pc for x in self.host_ids])
            )
            host_load           = _safe_div(host_stats["active_pc"][h], max(self.hosts[h].total_pc, 1e-9))
            host_idle_vm        = float(host_stats["idle_vm"][h]) / vm_cnt
            host_soonest_delay  = max(0.0, float((host_stats["soonest"][h] or now) - now))
            host_running_ratio  = _safe_div(host_stats["running"][h], vm_cnt)
            host_block[hi] = np.array([
                total_pc_n,
                host_load,
                host_idle_vm,
                host_soonest_delay / max(self.horizon, 1.0),
                host_running_ratio,
            ], dtype=np.float32)

        obs = np.concatenate([block1, block2, vm_block.reshape(-1), host_block.reshape(-1)], axis=0)
        mask = np.zeros((self.worker_act_dim,), dtype=np.float32)
        vm_idle = self._vm_idle_mask()
        if np.any(vm_idle):
            mask[vm_idle] = 1.0   # 仅空闲 VM 可选；存在空闲 VM 时无 No-op
        return obs, mask

    # ---------------- 分配/观测辅助 ----------------
    def _assign_task_to_specific_vm(self, task_id, vm_index):
        vm_id = self.vm_ids[vm_index]
        now = self.current_time
        bw_bps  = self.vms[vm_id].bw * 1e6     # Mb/s -> bits/s
        pc_mi_s = self.vms[vm_id].pc           # MI/s

        in_bits  = self.task_in_bits[task_id]
        out_bits = self.task_out_bits[task_id]
        mi       = self.task_mi[task_id]

        t_upload   = in_bits  / max(bw_bps, 1e-9)
        t_compute  = mi       / max(pc_mi_s, 1e-9)
        t_download = out_bits / max(bw_bps, 1e-9)

        start_time = max(now, float(self.vm_available_at[vm_index]))
        end_time   = start_time + t_upload + t_compute + t_download

        self.task_state[task_id] = "Running"
        self.task_end_time[task_id] = end_time
        self.vm_available_at[vm_index] = end_time
        if task_id in self.ready_task_ids:
            self.ready_task_ids.remove(task_id)
        heapq.heappush(self.event_heap, (end_time, "finish", task_id, vm_index))

        # 记录能耗片段
        host_id = self.vms[vm_id].host_id
        self._records.append(LoadRecord(start_time, end_time, host_id, pc_mi_s))

    def _vm_serial_order(self):
        avail = np.round(self.vm_available_at, 12)
        neg_pc = -np.array([self.vms[vid].pc for vid in self.vm_ids], dtype=np.float64)
        idx = np.arange(self.num_vms, dtype=np.int64)
        order = np.lexsort((idx, neg_pc, avail))  # 主键=avail，次键=pc 降序，三键=索引
        return order
