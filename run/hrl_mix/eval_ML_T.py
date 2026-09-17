# -*- coding: utf-8 -*-
"""
run_hrl_routeA_medTask_largeRes_deadlineCACHE_FCFS_alpha075_dalphaMix_HVrn_Tight.py

用途：
评测“最新修改后的 HRL_MIX 代码”训练出的三层模型（Manager + HostAgent + VMAgent）

严格对齐训练脚本：
train_hrl_tri_003_alpha15_medTask_largeRes_seed5_ddlFCFS_alpha075_dalphaMix_Tight.py

对齐点：
- env:
    b_hrl_tri_mgc_ave_ddlFCFS_mgrn_Mix_ddlAlpha
    .env_fcfs
- agent:
    base.d3qn_agent.D3QNAgent
- 执行逻辑：
    phase 开始 Manager 先调权 ->
    phase 内 Host -> VM 逐任务分配 ->
    phase 末 finish_phase_and_advance() ->
    下一 phase 再调权
- deterministic=True（评测时不走 epsilon）
- 评测 seeds：默认 2..31
- deadline 逻辑：
    每个 workflow 独立按固定概率采样 ddlAlpha：
        20% -> 2.0
        80% -> 3.0
    采样方式与训练环境完全一致（由 env 内部基于 episode_seed 和 wf_id 局部采样）
- Manager reward：
    reward = -[ alpha * phase_delay_cost + (1-alpha) * phase_energy_cost ]
    其中：
        manager_alpha_delay = 0.75
        manager_delay_mode  = "tardiness"

输出：
- 每个 seed 一行 CSV
- 终端打印每次与均值：
    avg_vm_reward / avg_host_reward / avg_mgr_reward / total_energy
- task级指标：
    task_avg_lateness   = mean(finish - deadline)
    task_avg_tardiness  = mean(max(finish - deadline, 0))
    task_success_rate   = on_time_tasks / finished_tasks
- workflow级指标：
    wf_avg_lateness     = mean(workflow_finish - workflow_deadline)
    wf_avg_tardiness    = mean(max(workflow_finish - workflow_deadline, 0))
    wf_success_rate     = on_time_workflows / finished_workflows
- makespan             = env.current_time
"""

import os
import sys
import glob
import numpy as np
import torch


# ---------------------------------------------------------------------
# 0) 定位项目根目录
# ---------------------------------------------------------------------
def _find_project_root(start_dir: str, max_up: int = 7) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        if os.path.isdir(os.path.join(cur, "common")) and os.path.isdir(os.path.join(cur, "base")):
            return cur
        nxt = os.path.abspath(os.path.join(cur, ".."))
        if nxt == cur:
            break
        cur = nxt
    return os.path.abspath(start_dir)


PROJECT_DIR = _find_project_root(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from common.metrics_logger import CSVLogger
from base.d3qn_agent import D3QNAgent
from output_naming import (
    eval_identity_from_filename,
    hrl_evaluation_csv_path,
    resolve_hrl_checkpoint_dir,
)

EVAL_SCENARIO, EVAL_DDL = eval_identity_from_filename(__file__)

# ✅ 改成与你最新训练脚本一致的环境路径
from base.hrl_env import (
    CloudWorkflowEnv_VMAgents as EnvHRLMix,
    MANAGER_ACTION_TABLE,
)


# ---------------------------------------------------------------------
# 1) 通用工具
# ---------------------------------------------------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


NUM_OPTIONS = int(MANAGER_ACTION_TABLE.shape[0])  # 243


def manager_apply_action(env: EnvHRLMix, act_idx: int):
    """act_idx ∈ [0,242] -> Δw(5,) -> env.apply_manager_delta(Δw)"""
    delta = MANAGER_ACTION_TABLE[int(act_idx)]
    env.apply_manager_delta(delta)


def _sync_env_scales(env: EnvHRLMix, env_kwargs: dict):
    """与训练脚本一致：同步 Route-A 参数。"""
    env.energy_reward_scale = float(env_kwargs["energy_reward_scale"])
    env.task_baseline_norm = float(env_kwargs["task_baseline_norm"])
    env.energy_norm_per_mi_ref = float(env_kwargs["energy_norm_per_mi_ref"])
    env.alpha_delay_host = float(env_kwargs["alpha_delay_host"])
    env.alpha_delay_vm = float(env_kwargs["alpha_delay_vm"])


def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _is_finished_state(x) -> bool:
    if isinstance(x, str):
        return x.lower() == "finished"
    return False


def _safe_get_index(arr_like, idx, default=None):
    if arr_like is None:
        return default
    try:
        if isinstance(arr_like, dict):
            return arr_like.get(idx, default)
        return arr_like[idx]
    except Exception:
        return default


def _iter_workflow_objs(env):
    cand_names = [
        "all_workflows", "workflows", "workflow_list", "wf_list",
        "completed_workflows", "active_workflows",
    ]
    for name in cand_names:
        obj = getattr(env, name, None)
        if obj is None:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                wf_id = getattr(v, "wf_id", None)
                if wf_id is None:
                    wf_id = getattr(v, "id", None)
                if wf_id is None:
                    try:
                        wf_id = int(k)
                    except Exception:
                        continue
                yield int(wf_id), v
            return
        try:
            for i, v in enumerate(obj):
                if v is None:
                    continue
                wf_id = getattr(v, "wf_id", None)
                if wf_id is None:
                    wf_id = getattr(v, "id", None)
                if wf_id is None:
                    wf_id = i
                yield int(wf_id), v
            return
        except Exception:
            continue


def _extract_task_to_wf_mapping(env, n_tasks: int):
    """
    尽量从环境中恢复 global task id -> workflow id 的映射。
    """
    cand_names = [
        "task_to_wf", "task2wf", "task_wf_ids", "task_wf_map",
        "task_workflow_ids", "task_workflow_map", "global_task_to_wf",
        "global_task_to_workflow", "task_owner_wf",
    ]

    for name in cand_names:
        obj = getattr(env, name, None)
        if obj is None:
            continue

        mapping = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    mapping[int(k)] = int(v)
                except Exception:
                    pass
        else:
            try:
                if len(obj) == n_tasks:
                    for tid in range(n_tasks):
                        val = obj[tid]
                        if val is None:
                            continue
                        mapping[int(tid)] = int(val)
            except Exception:
                pass

        if len(mapping) > 0:
            return mapping

    meta_names = ["task_meta", "task_info", "task_infos", "task_records"]
    wf_key_names = ["wf_id", "workflow_id", "owner_wf", "workflow"]

    for name in meta_names:
        meta = getattr(env, name, None)
        if meta is None:
            continue
        try:
            if len(meta) != n_tasks:
                continue
            mapping = {}
            for tid in range(n_tasks):
                row = meta[tid]
                if isinstance(row, dict):
                    for k in wf_key_names:
                        if k in row:
                            mapping[int(tid)] = int(row[k])
                            break
                elif isinstance(row, (tuple, list)) and len(row) >= 1:
                    # 常见格式：(wf_id, local_task_id)
                    try:
                        mapping[int(tid)] = int(row[0])
                    except Exception:
                        pass
            if len(mapping) > 0:
                return mapping
        except Exception:
            pass

    return {}


def _extract_wf_deadlines(env):
    out = {}

    cand_names = [
        "workflow_deadlines", "wf_deadlines", "workflow_deadline",
        "wf_deadline", "workflow_baseline_finish", "wf_baseline_finish",
    ]
    for name in cand_names:
        obj = getattr(env, name, None)
        if obj is None:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                val = _to_float(v, None)
                if val is not None:
                    try:
                        out[int(k)] = val
                    except Exception:
                        pass
        else:
            try:
                for i, v in enumerate(obj):
                    val = _to_float(v, None)
                    if val is not None:
                        out[int(i)] = val
            except Exception:
                pass
        if len(out) > 0:
            return out

    for wf_id, wf in _iter_workflow_objs(env):
        for attr in ["deadline", "wf_deadline", "workflow_deadline"]:
            if hasattr(wf, attr):
                val = _to_float(getattr(wf, attr), None)
                if val is not None:
                    out[int(wf_id)] = val
                    break

    return out


def _extract_wf_finish_times(env):
    out = {}

    cand_names = [
        "workflow_finish_time", "workflow_finish_times", "wf_finish_time", "wf_finish_times",
        "workflow_end_time", "workflow_end_times", "wf_end_time", "wf_end_times",
        "workflow_completed_time", "workflow_completed_times",
    ]
    for name in cand_names:
        obj = getattr(env, name, None)
        if obj is None:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                val = _to_float(v, None)
                if val is not None:
                    try:
                        out[int(k)] = val
                    except Exception:
                        pass
        else:
            try:
                for i, v in enumerate(obj):
                    val = _to_float(v, None)
                    if val is not None:
                        out[int(i)] = val
            except Exception:
                pass
        if len(out) > 0:
            return out

    for wf_id, wf in _iter_workflow_objs(env):
        for attr in ["finish_time", "end_time", "completed_time", "completion_time"]:
            if hasattr(wf, attr):
                val = _to_float(getattr(wf, attr), None)
                if val is not None:
                    out[int(wf_id)] = val
                    break

    return out


def _extract_wf_finished_flags(env):
    out = {}

    cand_names = [
        "workflow_done", "workflow_finished", "workflow_completed",
        "wf_done", "wf_finished", "wf_completed",
    ]
    for name in cand_names:
        obj = getattr(env, name, None)
        if obj is None:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    out[int(k)] = bool(v)
                except Exception:
                    pass
        else:
            try:
                for i, v in enumerate(obj):
                    out[int(i)] = bool(v)
            except Exception:
                pass
        if len(out) > 0:
            return out

    for wf_id, wf in _iter_workflow_objs(env):
        for attr in ["done", "finished", "completed", "is_done", "is_finished", "is_completed"]:
            if hasattr(wf, attr):
                try:
                    out[int(wf_id)] = bool(getattr(wf, attr))
                    break
                except Exception:
                    pass

    return out


def compute_task_and_workflow_metrics(env: EnvHRLMix):
    """
    同时统计：
    1) task级 lateness / tardiness / success_rate
    2) workflow级 lateness / tardiness / success_rate

    task级：
        基于 env.task_baseline_finish 与 env.task_end_time
    workflow级：
        优先调用 env.get_episode_metrics()
        若无，则手工从 workflow deadline / finish time 恢复
    """
    # ---------------- task级 ----------------
    task_state = getattr(env, "task_state", [])
    task_end_time = getattr(env, "task_end_time", [])
    task_deadline = getattr(env, "task_baseline_finish", [])

    task_sum_lateness = 0.0
    task_sum_tardiness = 0.0
    task_n_finished = 0
    task_n_tardy = 0

    n_tasks_all = len(task_state)
    for tid in range(n_tasks_all):
        if not _is_finished_state(task_state[tid]):
            continue
        if tid >= len(task_deadline):
            continue

        finish_t = _to_float(_safe_get_index(task_end_time, tid), None)
        deadline_t = _to_float(_safe_get_index(task_deadline, tid), None)
        if finish_t is None or deadline_t is None:
            continue

        lateness = finish_t - deadline_t
        tardiness = max(lateness, 0.0)

        task_sum_lateness += lateness
        task_sum_tardiness += tardiness
        task_n_finished += 1
        if tardiness > 0.0:
            task_n_tardy += 1

    task_avg_lateness = task_sum_lateness / max(task_n_finished, 1)
    task_avg_tardiness = task_sum_tardiness / max(task_n_finished, 1)
    task_pct_tardy = float(task_n_tardy) / float(max(task_n_finished, 1))
    task_success_rate = 1.0 - task_pct_tardy

    # ---------------- workflow级：优先使用 env.get_episode_metrics() ----------------
    wf_sum_lateness = 0.0
    wf_sum_tardiness = 0.0
    wf_n_finished = 0
    wf_n_tardy = 0
    used_episode_metrics = False

    get_episode_metrics_fn = getattr(env, "get_episode_metrics", None)
    if callable(get_episode_metrics_fn):
        try:
            epi = get_episode_metrics_fn()
            if isinstance(epi, dict):
                wf_avg_lateness = epi.get("wf_avg_lateness", epi.get("workflow_avg_lateness", None))
                wf_avg_tardiness = epi.get("wf_avg_tardiness", epi.get("workflow_avg_tardiness", None))
                wf_success_rate = epi.get("wf_success_rate", epi.get("workflow_success_rate", None))
                wf_pct_tardy = epi.get("wf_pct_tardy", epi.get("workflow_pct_tardy", None))

                wf_total_lateness = epi.get("wf_total_lateness", epi.get("workflow_total_lateness", None))
                wf_total_tardiness = epi.get("wf_total_tardiness", epi.get("workflow_total_tardiness", None))
                wf_n_finished_ = epi.get(
                    "wf_finished",
                    epi.get("finished_workflows", epi.get("n_finished_workflows", epi.get("num_finished_workflows", None)))
                )
                wf_n_tardy_ = epi.get(
                    "wf_tardy",
                    epi.get("tardy_workflows", epi.get("n_tardy_workflows", epi.get("num_tardy_workflows", None)))
                )

                if wf_avg_lateness is not None and wf_avg_tardiness is not None:
                    wf_avg_lateness = float(wf_avg_lateness)
                    wf_avg_tardiness = float(wf_avg_tardiness)

                    if wf_n_finished_ is not None:
                        wf_n_finished = int(wf_n_finished_)
                    elif wf_total_lateness is not None and abs(wf_avg_lateness) > 1e-12:
                        wf_n_finished = int(round(float(wf_total_lateness) / wf_avg_lateness))
                    else:
                        wf_n_finished = 0

                    if wf_n_tardy_ is not None:
                        wf_n_tardy = int(wf_n_tardy_)
                    elif wf_pct_tardy is not None and wf_n_finished > 0:
                        wf_n_tardy = int(round(float(wf_pct_tardy) * wf_n_finished))
                    elif wf_success_rate is not None and wf_n_finished > 0:
                        wf_n_tardy = int(round((1.0 - float(wf_success_rate)) * wf_n_finished))
                    else:
                        wf_n_tardy = 0

                    if wf_total_lateness is not None:
                        wf_sum_lateness = float(wf_total_lateness)
                    else:
                        wf_sum_lateness = wf_avg_lateness * wf_n_finished

                    if wf_total_tardiness is not None:
                        wf_sum_tardiness = float(wf_total_tardiness)
                    else:
                        wf_sum_tardiness = wf_avg_tardiness * wf_n_finished

                    used_episode_metrics = True
        except Exception:
            pass

    # ---------------- workflow级：若环境未直接给出，则手工恢复 ----------------
    if not used_episode_metrics:
        task_to_wf = _extract_task_to_wf_mapping(env, n_tasks_all)
        wf_deadline_map = _extract_wf_deadlines(env)
        wf_finish_map = _extract_wf_finish_times(env)
        wf_finished_map = _extract_wf_finished_flags(env)

        wf_to_tasks = {}
        for tid, wf_id in task_to_wf.items():
            wf_to_tasks.setdefault(int(wf_id), []).append(int(tid))

        all_wf_ids = set(wf_deadline_map.keys()) | set(wf_finish_map.keys()) | set(wf_finished_map.keys()) | set(wf_to_tasks.keys())
        for wf_id, _wf in _iter_workflow_objs(env):
            all_wf_ids.add(int(wf_id))

        for wf_id in sorted(all_wf_ids):
            deadline_t = _to_float(wf_deadline_map.get(wf_id, None), None)
            finish_t = _to_float(wf_finish_map.get(wf_id, None), None)
            finished_flag = wf_finished_map.get(wf_id, None)

            tids = wf_to_tasks.get(wf_id, [])
            if finish_t is None and len(tids) > 0:
                all_finished = True
                fin_times = []
                for tid in tids:
                    st = _safe_get_index(task_state, tid, None)
                    et = _to_float(_safe_get_index(task_end_time, tid, None), None)
                    if (not _is_finished_state(st)) or (et is None):
                        all_finished = False
                        break
                    fin_times.append(et)
                if all_finished and len(fin_times) > 0:
                    finish_t = max(fin_times)
                    if finished_flag is None:
                        finished_flag = True

            if finished_flag is False:
                continue
            if finish_t is None or deadline_t is None:
                continue

            lateness = finish_t - deadline_t
            tardiness = max(lateness, 0.0)

            wf_sum_lateness += lateness
            wf_sum_tardiness += tardiness
            wf_n_finished += 1
            if tardiness > 0.0:
                wf_n_tardy += 1

    wf_avg_lateness = wf_sum_lateness / max(wf_n_finished, 1)
    wf_avg_tardiness = wf_sum_tardiness / max(wf_n_finished, 1)
    wf_pct_tardy = float(wf_n_tardy) / float(max(wf_n_finished, 1))
    wf_success_rate = 1.0 - wf_pct_tardy

    makespan = float(getattr(env, "current_time", 0.0))

    return {
        # task级
        "task_n_finished": int(task_n_finished),
        "task_n_tardy": int(task_n_tardy),
        "task_total_lateness": float(task_sum_lateness),
        "task_total_tardiness": float(task_sum_tardiness),
        "task_avg_lateness": float(task_avg_lateness),
        "task_avg_tardiness": float(task_avg_tardiness),
        "task_pct_tardy": float(task_pct_tardy),
        "task_success_rate": float(task_success_rate),

        # workflow级
        "wf_n_finished": int(wf_n_finished),
        "wf_n_tardy": int(wf_n_tardy),
        "wf_total_lateness": float(wf_sum_lateness),
        "wf_total_tardiness": float(wf_sum_tardiness),
        "wf_avg_lateness": float(wf_avg_lateness),
        "wf_avg_tardiness": float(wf_avg_tardiness),
        "wf_pct_tardy": float(wf_pct_tardy),
        "wf_success_rate": float(wf_success_rate),

        # 其它
        "makespan": float(makespan),
    }


def _first_existing_glob(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------
# 2) 构造与你最新 HRL_MIX 训练脚本一致的 env_kwargs（medTask + largeRes）
# ---------------------------------------------------------------------
def build_env_kwargs_hrl_mix(random_seed=1):
    dax_dir = os.path.join(PROJECT_DIR, "data", "dax")
    dax_30 = [
        os.path.join(dax_dir, "CyberShake_30.xml"),
        os.path.join(dax_dir, "Epigenomics_24.xml"),
        os.path.join(dax_dir, "Ligo_30.xml"),
        os.path.join(dax_dir, "Montage_25.xml"),
        os.path.join(dax_dir, "Sipht_29.xml"),
    ]

    # 50-scale 集合（你项目里对应“规模=50”的一组）
    dax_50 = [
        os.path.join(dax_dir, "CyberShake_50.xml"),
        os.path.join(dax_dir, "Epigenomics_47.xml"),
        os.path.join(dax_dir, "Ligo_50.xml"),
        os.path.join(dax_dir, "Montage_50.xml"),
        os.path.join(dax_dir, "Sipht_58.xml"),
    ]

    # 30% / 70% 权重：重复实现
    dax_list = list(dax_30) * 3 + list(dax_50) * 7

    horizon = 1e9
    arrival_lambda = 0.03
    max_ready_tasks = "auto"
    normalize_obs = True
    workflows_per_episode = 50

    # ===== Route-A 参数 =====
    ENERGY_REWARD_SCALE = 1e-3
    TASK_BASELINE_NORM = 300.0
    ENERGY_NORM_PER_MI_REF = 20.0
    ALPHA_DELAY_HOST = 0.75
    ALPHA_DELAY_VM = 0.75

    # ===== Manager reward 配置：与训练代码完全一致 =====
    MGR_ALPHA_DELAY = 0.75
    MGR_DELAY_MODE = "tardiness"

    # ===== workflow ddlAlpha 混合采样：与训练代码完全一致 =====
    DEADLINE_ALPHA_SMALL = 2.0
    DEADLINE_ALPHA_LARGE = 3.0
    DEADLINE_ALPHA_SMALL_PROB = 0.8

    # ===== deadline cache 路径 =====
    deadline_cache_path = os.path.join(
        PROJECT_DIR, "data", "deadlines", "fcfs",
        "fcfs_medTask_largeRes_seed0-1000.json",
    )
    if not os.path.exists(deadline_cache_path):
        alt = _first_existing_glob([
            os.path.join(PROJECT_DIR, "data", "deadlines", "fcfs", "fcfs_medTask_largeRes_seed*.json"),
            os.path.join(PROJECT_DIR, "data", "deadlines", "fcfs", "fcfs_*medTask*largeRes*.json"),
        ])
        if alt is None:
            raise FileNotFoundError(
                f"deadline cache 不存在：{deadline_cache_path}\n"
                f"也未在 data/deadlines/fcfs/ 下找到可用的 deadline_cache_fcfs_vmfirst*.json\n"
                f"请先运行预计算脚本生成 cache。"
            )
        deadline_cache_path = alt

    env_kwargs = dict(
        dax_paths=dax_list,
        horizon=float(horizon),
        arrival_lambda=float(arrival_lambda),
        random_seed=int(random_seed),
        max_ready_tasks=max_ready_tasks,
        normalize=bool(normalize_obs),
        workflows_per_episode=int(workflows_per_episode),

        # ===== largeRes：9 Hosts / 75 VMs（5 Cloud + 4 Edge）=====
        num_cloud_hosts=5,
        num_edge_hosts=4,
        cloud_vms_per_host=(9, 9, 9, 8, 8),
        edge_vms_per_host=(8, 8, 8, 8),
        cloud_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        edge_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        cloud_bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        edge_bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),

        # ===== deadline cache + per-workflow random alpha =====
        deadline_mode="cache_fcfs",
        deadline_cache_path=deadline_cache_path,
        deadline_cache_strict=True,
        deadline_alpha_small=float(DEADLINE_ALPHA_SMALL),
        deadline_alpha_large=float(DEADLINE_ALPHA_LARGE),
        deadline_alpha_small_prob=float(DEADLINE_ALPHA_SMALL_PROB),

        # ===== Manager reward：与训练一致 =====
        manager_alpha_delay=float(MGR_ALPHA_DELAY),
        manager_delay_mode=str(MGR_DELAY_MODE),

        # ===== 这些不是 env ctor 参数：用于 _sync_env_scales() =====
        energy_reward_scale=float(ENERGY_REWARD_SCALE),
        task_baseline_norm=float(TASK_BASELINE_NORM),
        energy_norm_per_mi_ref=float(ENERGY_NORM_PER_MI_REF),
        alpha_delay_host=float(ALPHA_DELAY_HOST),
        alpha_delay_vm=float(ALPHA_DELAY_VM),
    )
    return env_kwargs


# ---------------------------------------------------------------------
# 3) 单 seed 评测：严格复刻训练脚本 evaluate 的执行逻辑
# ---------------------------------------------------------------------
def evaluate_hrl_mix_one_seed(env_cls, env_kwargs, vm_agent, host_agent, manager_agent):
    """
    deterministic=True（不走 epsilon）
    返回：
      三层奖励均值、总能耗、task级指标、workflow级指标、makespan
    """
    ctor_block = {
        "energy_reward_scale", "task_baseline_norm",
        "energy_norm_per_mi_ref", "alpha_delay_host", "alpha_delay_vm"
    }
    ctor_kwargs = {k: v for k, v in env_kwargs.items() if k not in ctor_block}

    env = env_cls(**ctor_kwargs)
    _sync_env_scales(env, env_kwargs)

    env.reset()

    # ===== phase 开始：Manager 先调一次权 =====
    sH = env.get_manager_state()
    m_mask = env.get_manager_action_mask()
    m_act = manager_agent.select_action(sH, m_mask, deterministic=True, count_step=False)
    manager_apply_action(env, m_act)

    done = bool(getattr(env, "done_flag", False))
    phases = 0
    ret_mgr = 0.0
    ret_vm_phase_mean = 0.0
    ret_host_phase_mean = 0.0

    while not done:
        vm_rewards = []
        host_rewards = []

        # ===== phase 内：逐任务（Host -> VM）=====
        while True:
            st_host, has_next = env.get_host_state_for_next_assignment()
            if not has_next:
                break

            a_host = host_agent.select_action(
                st_host["obs"], st_host["mask"],
                deterministic=True, count_step=False
            )
            env.host_select(int(a_host))

            st_vm, ok_vm = env.get_vm_state_for_current_task()
            if not ok_vm:
                break

            a_vm = vm_agent.select_action(
                st_vm["obs"], st_vm["mask"],
                deterministic=True, count_step=False
            )
            r_host, r_vm, _info_task = env.vm_assign(int(a_vm))

            host_rewards.append(float(r_host))
            vm_rewards.append(float(r_vm))

        # ===== phase 末：推进 / 能耗结算 -> Manager奖励 =====
        r_manager_raw, _pinfo = env.finish_phase_and_advance()

        ret_mgr += float(r_manager_raw)
        ret_vm_phase_mean += float(np.mean(vm_rewards)) if len(vm_rewards) > 0 else 0.0
        ret_host_phase_mean += float(np.mean(host_rewards)) if len(host_rewards) > 0 else 0.0
        phases += 1

        done = bool(getattr(env, "done_flag", False))
        if done:
            break

        # ===== 下一 phase：Manager 再调一次权 =====
        sH = env.get_manager_state()
        m_mask = env.get_manager_action_mask()
        m_act = manager_agent.select_action(sH, m_mask, deterministic=True, count_step=False)
        manager_apply_action(env, m_act)

    avg_vm = ret_vm_phase_mean / max(phases, 1)
    avg_host = ret_host_phase_mean / max(phases, 1)
    avg_mgr = ret_mgr / max(phases, 1)
    total_energy = float(env.total_energy)

    metrics = compute_task_and_workflow_metrics(env)
    metrics.update({
        "eval_vm_reward": float(avg_vm),
        "eval_host_reward": float(avg_host),
        "eval_mgr_reward": float(avg_mgr),
        "eval_energy": float(total_energy),
    })
    return metrics


# ---------------------------------------------------------------------
# 4) 构建三层 agents（维度探测对齐训练脚本）并加载最新 HRL_MIX best ckpt
# ---------------------------------------------------------------------
def build_three_layer_agents_and_load(env_kwargs: dict):
    ctor_block = {
        "energy_reward_scale", "task_baseline_norm",
        "energy_norm_per_mi_ref", "alpha_delay_host", "alpha_delay_vm"
    }
    ctor_kwargs = {k: v for k, v in env_kwargs.items() if k not in ctor_block}

    tmp_env = EnvHRLMix(**ctor_kwargs)
    _sync_env_scales(tmp_env, env_kwargs)
    tmp_env.reset()

    # ---- 探测 host/vm 维度 ----
    st_host, ok = tmp_env.get_host_state_for_next_assignment()
    while (not ok) and (not tmp_env.done_flag):
        _rM, _ = tmp_env.finish_phase_and_advance()
        st_host, ok = tmp_env.get_host_state_for_next_assignment()

    host_state_dim = int(st_host["obs"].shape[0])
    host_act_dim = int(st_host["mask"].shape[0])

    a_host0 = int(np.argmax(st_host["mask"])) if np.sum(st_host["mask"]) > 0 else 0
    tmp_env.host_select(a_host0)

    st_vm, ok_vm = tmp_env.get_vm_state_for_current_task()
    if not ok_vm:
        raise RuntimeError("Probe vm_state failed: get_vm_state_for_current_task() returned ok_vm=False")

    vm_state_dim = int(st_vm["obs"].shape[0])
    vm_act_dim = int(st_vm["mask"].shape[0])

    mgr_state_dim = int(tmp_env.get_manager_state().shape[0])

    print(
        f"[HRL_MIX probe] host_obs_dim={host_state_dim}, host_act_dim={host_act_dim} | "
        f"vm_obs_dim={vm_state_dim}, vm_act_dim={vm_act_dim} | "
        f"mgr_obs_dim={mgr_state_dim}, mgr_act_dim={NUM_OPTIONS}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vm_agent = D3QNAgent(
        input_dim=vm_state_dim, output_dim=vm_act_dim,
        lr=3e-4, gamma=0.95,
        batch_size=256, buffer_size=120000,
        eps_start=1.0, eps_end=0.05, eps_decay_steps=80000,
        target_update_tau=0.005,
        grad_clip=10.0,
        hidden_dims=(1024, 1024, 512, 512, 256),
        device=device,
    )
    host_agent = D3QNAgent(
        input_dim=host_state_dim, output_dim=host_act_dim,
        lr=3e-4, gamma=0.95,
        batch_size=256, buffer_size=120000,
        eps_start=1.0, eps_end=0.05, eps_decay_steps=80000,
        target_update_tau=0.005,
        grad_clip=10.0,
        hidden_dims=(1024, 1024, 512, 512, 256),
        device=device,
    )
    manager_agent = D3QNAgent(
        input_dim=mgr_state_dim, output_dim=NUM_OPTIONS,
        lr=3e-4, gamma=0.95,
        batch_size=256, buffer_size=60000,
        eps_start=1.0, eps_end=0.05, eps_decay_steps=40000,
        target_update_tau=0.01,
        grad_clip=10.0,
        hidden_dims=(512, 512, 256, 128),
        device=device,
    )

    ckpt_dir_default = str(
        resolve_hrl_checkpoint_dir(
            PROJECT_DIR,
            EVAL_SCENARIO,
            EVAL_DDL,
            seed=1,
        )
    )
    ckpt_vm = os.path.join(ckpt_dir_default, "best_vm.pth")
    ckpt_host = os.path.join(ckpt_dir_default, "best_host.pth")
    ckpt_mgr = os.path.join(ckpt_dir_default, "best_manager.pth")

    print(f"[HRL_MIX load] ckpt_dir: {ckpt_dir_default}")
    print(f"[HRL_MIX load] vm     : {ckpt_vm}")
    vm_agent.load(ckpt_vm)
    print(f"[HRL_MIX load] host   : {ckpt_host}")
    host_agent.load(ckpt_host)
    print(f"[HRL_MIX load] manager: {ckpt_mgr}")
    manager_agent.load(ckpt_mgr)

    vm_agent.online.eval()
    vm_agent.target.eval()
    host_agent.online.eval()
    host_agent.target.eval()
    manager_agent.online.eval()
    manager_agent.target.eval()

    return vm_agent, host_agent, manager_agent


# ---------------------------------------------------------------------
# 5) 主逻辑：30 seeds 求均值
# ---------------------------------------------------------------------
def main():
    seeds = list(range(2, 32))

    env_kwargs_base = build_env_kwargs_hrl_mix(random_seed=seeds[0])

    csv_path = str(
        hrl_evaluation_csv_path(
            PROJECT_DIR,
            EVAL_SCENARIO,
            EVAL_DDL,
        )
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    logger = CSVLogger(
        filepath=csv_path,
        fieldnames=[
            "run_idx", "seed",
            "eval_vm_reward", "eval_host_reward", "eval_mgr_reward",
            "eval_energy",

            "task_n_finished", "task_n_tardy",
            "task_total_lateness", "task_total_tardiness",
            "task_avg_lateness", "task_avg_tardiness",
            "task_pct_tardy", "task_success_rate",

            "wf_n_finished", "wf_n_tardy",
            "wf_total_lateness", "wf_total_tardiness",
            "wf_avg_lateness", "wf_avg_tardiness",
            "wf_pct_tardy", "wf_success_rate",

            "makespan",
        ],
    )

    print(">>> 构建三层 agents 并加载最新 HRL_MIX best_ckpt ...")
    print(">>> ddlAlpha cfg: small=2.0, large=3.0, P(small)=0.2")
    print(">>> manager reward cfg: alpha_delay=0.75, delay_mode=tardiness")
    vm_agent, host_agent, manager_agent = build_three_layer_agents_and_load(env_kwargs_base)

    vm_list, host_list, mgr_list, energy_list = [], [], [], []
    task_late_list, task_tard_list, task_succ_list, task_pct_list = [], [], [], []
    wf_late_list, wf_tard_list, wf_succ_list, wf_pct_list = [], [], [], []
    ms_list = []

    for idx, sd in enumerate(seeds, start=1):
        print(f"\n===== Eval {idx}/{len(seeds)} | seed={sd} =====")
        set_seed(sd)

        kwargs_i = dict(env_kwargs_base)
        kwargs_i["random_seed"] = int(sd)

        out = evaluate_hrl_mix_one_seed(
            EnvHRLMix, kwargs_i, vm_agent, host_agent, manager_agent
        )

        logger.log(
            run_idx=idx,
            seed=sd,
            eval_vm_reward=float(out["eval_vm_reward"]),
            eval_host_reward=float(out["eval_host_reward"]),
            eval_mgr_reward=float(out["eval_mgr_reward"]),
            eval_energy=float(out["eval_energy"]),

            task_n_finished=int(out["task_n_finished"]),
            task_n_tardy=int(out["task_n_tardy"]),
            task_total_lateness=float(out["task_total_lateness"]),
            task_total_tardiness=float(out["task_total_tardiness"]),
            task_avg_lateness=float(out["task_avg_lateness"]),
            task_avg_tardiness=float(out["task_avg_tardiness"]),
            task_pct_tardy=float(out["task_pct_tardy"]),
            task_success_rate=float(out["task_success_rate"]),

            wf_n_finished=int(out["wf_n_finished"]),
            wf_n_tardy=int(out["wf_n_tardy"]),
            wf_total_lateness=float(out["wf_total_lateness"]),
            wf_total_tardiness=float(out["wf_total_tardiness"]),
            wf_avg_lateness=float(out["wf_avg_lateness"]),
            wf_avg_tardiness=float(out["wf_avg_tardiness"]),
            wf_pct_tardy=float(out["wf_pct_tardy"]),
            wf_success_rate=float(out["wf_success_rate"]),

            makespan=float(out["makespan"]),
        )

        vm_list.append(float(out["eval_vm_reward"]))
        host_list.append(float(out["eval_host_reward"]))
        mgr_list.append(float(out["eval_mgr_reward"]))
        energy_list.append(float(out["eval_energy"]))

        task_late_list.append(float(out["task_avg_lateness"]))
        task_tard_list.append(float(out["task_avg_tardiness"]))
        task_succ_list.append(float(out["task_success_rate"]))
        task_pct_list.append(float(out["task_pct_tardy"]))

        wf_late_list.append(float(out["wf_avg_lateness"]))
        wf_tard_list.append(float(out["wf_avg_tardiness"]))
        wf_succ_list.append(float(out["wf_success_rate"]))
        wf_pct_list.append(float(out["wf_pct_tardy"]))

        ms_list.append(float(out["makespan"]))

        print(
            f"[HRL_MIX] "
            f"vm={out['eval_vm_reward']:.6f} host={out['eval_host_reward']:.6f} mgr={out['eval_mgr_reward']:.6f} "
            f"energy={out['eval_energy']:.3f} J | "
            f"tasks={out['task_n_finished']} tardy={out['task_n_tardy']} | "
            f"task_avg_lateness={out['task_avg_lateness']:.6f} "
            f"task_avg_tardiness={out['task_avg_tardiness']:.6f} "
            f"task_success_rate={out['task_success_rate']:.6f} | "
            f"wf={out['wf_n_finished']} tardy={out['wf_n_tardy']} | "
            f"wf_avg_lateness={out['wf_avg_lateness']:.6f} "
            f"wf_avg_tardiness={out['wf_avg_tardiness']:.6f} "
            f"wf_success_rate={out['wf_success_rate']:.6f} | "
            f"makespan={out['makespan']:.3f}"
        )

        print(
            f"[Detail] "
            f"task_total_lateness={out['task_total_lateness']:.6f} "
            f"task_total_tardiness={out['task_total_tardiness']:.6f} | "
            f"wf_total_lateness={out['wf_total_lateness']:.6f} "
            f"wf_total_tardiness={out['wf_total_tardiness']:.6f}"
        )

    print("\n================= HRL_MIX（30 seeds）汇总 =================")
    print(f"avg_vm_reward        = {float(np.mean(vm_list)):.6f}")
    print(f"avg_host_reward      = {float(np.mean(host_list)):.6f}")
    print(f"avg_mgr_reward       = {float(np.mean(mgr_list)):.6f}")
    print(f"avg_energy           = {float(np.mean(energy_list)):.3f} J")

    print(f"task_avg_lateness    = {float(np.mean(task_late_list)):.6f}")
    print(f"task_avg_tardiness   = {float(np.mean(task_tard_list)):.6f}")
    print(f"task_avg_pct_tardy   = {float(np.mean(task_pct_list)):.6f}")
    print(f"task_avg_success     = {float(np.mean(task_succ_list)):.6f}")

    print(f"wf_avg_lateness      = {float(np.mean(wf_late_list)):.6f}")
    print(f"wf_avg_tardiness     = {float(np.mean(wf_tard_list)):.6f}")
    print(f"wf_avg_pct_tardy     = {float(np.mean(wf_pct_list)):.6f}")
    print(f"wf_avg_success       = {float(np.mean(wf_succ_list)):.6f}")

    print(f"avg_makespan         = {float(np.mean(ms_list)):.3f}")
    print("=========================================================")
    print(f"结果已写入 CSV: {csv_path}")


if __name__ == "__main__":
    main()
