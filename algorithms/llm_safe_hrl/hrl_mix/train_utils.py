# -*- coding: utf-8 -*-
"""
训练工具函数模块。

整体调用关系：
1. train.py 调用 train_runner.train()。
2. train_runner.py 在训练主循环中调用本文件的工具函数。
3. train_eval.py 在评估流程中复用 manager_apply_action() 和 sync_env_scales()。

文件职责：
- set_seed()：统一设置 Python、NumPy、PyTorch 的随机种子。
- manager_apply_action()：把 manager 选择的动作编号转换为环境可执行的 delta。
- apply_env_scales()：把 TrainConfig 中的 reward/归一化参数写入训练环境。
- sync_env_scales()：评估环境创建后，同步训练环境使用的尺度参数。
- compute_episode_task_lateness_metrics()：统计 episode 级任务迟延指标。
- warmup_ready()：判断 replay buffer 是否足够开始更新网络。
- print_device_info()：打印 CUDA 和三个 agent 的设备信息。

设计说明：
- 这里不放训练流程，只放小而稳定、可复用的工具函数。
- train_runner.py 和 train_eval.py 都可以安全复用这里的函数，避免重复实现。
"""
from __future__ import annotations

import random

import numpy as np
import torch

from base.d3qn_agent import D3QNAgent
from base.hrl_env import CloudWorkflowEnv_VMAgents, MANAGER_ACTION_TABLE


NUM_OPTIONS = int(MANAGER_ACTION_TABLE.shape[0])


def set_seed(seed: int) -> None:
    """设置所有常用随机源的种子，提升实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def manager_apply_action(env: CloudWorkflowEnv_VMAgents, act_idx: int) -> None:
    """按环境声明的 Manager 模式应用权重增量或启发式索引。"""
    if hasattr(env, "apply_manager_action"):
        env.apply_manager_action(int(act_idx))
        return
    # 保留旧环境替身兼容性。
    delta = MANAGER_ACTION_TABLE[int(act_idx)]
    env.apply_manager_delta(delta)


def manager_action_dim(env: CloudWorkflowEnv_VMAgents) -> int:
    """从实际 Manager mask 推断动作维度，避免硬编码 243。"""
    mask = np.asarray(
        env.get_manager_action_mask(),
        dtype=np.float32,
    ).reshape(-1)
    if mask.size <= 0 or not np.all(np.isfinite(mask)):
        raise ValueError("Manager action mask is empty or non-finite")
    return int(mask.size)


def layer_learning_action_mask(
    layer_state: dict,
    *,
    safe_rl_enabled: bool,
) -> np.ndarray:
    """返回实际供层级策略选择和 replay bootstrap 使用的 mask。

    safe 模式必须消费显式 ``final_action_mask``；旧模式继续使用历史
    ``mask`` 字段。返回副本，避免 agent 或调用方修改环境状态字典。
    """
    key = "final_action_mask" if safe_rl_enabled else "mask"
    if key not in layer_state:
        raise KeyError(f"layer state is missing {key}")
    mask = np.asarray(layer_state[key], dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(mask)):
        raise ValueError(f"{key} contains NaN or infinite values")
    return (mask > 0.5).astype(np.float32)


def select_layer_action(
    agent: D3QNAgent,
    layer_state: dict,
    *,
    safe_rl_enabled: bool,
    deterministic: bool,
    count_step: bool,
) -> tuple[int, bool, np.ndarray]:
    """从最终动作集合选动作；空集合时直接交给确定性回退。

    返回 ``(action, selected_by_agent, learning_mask)``。回退动作不是 RL
    提议，因此不会调用 ``agent.select_action``，也不应作为处于全零最终
    mask 内的策略动作写入 replay。
    """
    action, selected_by_agent, learning_mask, _ = (
        select_layer_action_with_info(
            agent,
            layer_state,
            safe_rl_enabled=safe_rl_enabled,
            deterministic=deterministic,
            count_step=count_step,
        )
    )
    return action, selected_by_agent, learning_mask


def select_layer_action_with_info(
    agent: D3QNAgent,
    layer_state: dict,
    *,
    safe_rl_enabled: bool,
    deterministic: bool,
    count_step: bool,
) -> tuple[int, bool, np.ndarray, dict]:
    """选取层级动作并显式区分安全探索、贪心和空集合回退。

    前三个返回值与 :func:`select_layer_action` 完全一致，第四个值只用于
    动作审计。安全模式的 RL 提议仅来自 ``final_action_mask``；该集合
    为空时不调用 agent，而是使用环境预先计算的确定性 fallback。
    """
    learning_mask = layer_learning_action_mask(
        layer_state,
        safe_rl_enabled=safe_rl_enabled,
    )
    if safe_rl_enabled and not np.any(learning_mask > 0.5):
        fallback_action = layer_state.get("fallback_action")
        if fallback_action is None:
            raise RuntimeError(
                "final_action_mask is empty but the environment did "
                "not provide a deterministic fallback_action"
            )
        fallback_action = int(fallback_action)
        decision = {
            "action": fallback_action,
            "proposed_action": None,
            "executed_action": fallback_action,
            "selected_by_agent": False,
            "selection_type": "fallback_action",
            "policy_selection_type": "fallback_action",
            "action_source": "fallback_action",
            "action_modified": False,
            "random_exploration": False,
            "safe_rl_enabled": True,
            "deterministic": True,
            "epsilon": 0.0,
            "valid_action_count": 0,
            "selection_mask": learning_mask.copy(),
        }
        return (
            fallback_action,
            False,
            learning_mask,
            decision,
        )

    if hasattr(agent, "select_action_with_info"):
        decision = dict(
            agent.select_action_with_info(
                layer_state["obs"],
                learning_mask,
                deterministic=deterministic,
                count_step=count_step,
            )
        )
        action = int(decision["action"])
    else:
        # 保留轻量测试替身和旧自定义 agent 的兼容性。
        action = int(
            agent.select_action(
                layer_state["obs"],
                learning_mask,
                deterministic=deterministic,
                count_step=count_step,
            )
        )
        decision = {
            "action": action,
            "proposed_action": action,
            "selected_by_agent": True,
            "selection_type": (
                "greedy_safe_action"
                if safe_rl_enabled
                else "greedy_action"
            ),
            "random_exploration": False,
            "safe_rl_enabled": bool(safe_rl_enabled),
            "deterministic": bool(deterministic),
            "epsilon": None,
            "valid_action_count": int(
                np.count_nonzero(learning_mask > 0.5)
            ),
            "selection_mask": learning_mask.copy(),
        }
    decision.setdefault(
        "policy_selection_type",
        decision["selection_type"],
    )
    decision.setdefault(
        "action_source",
        decision["selection_type"],
    )
    decision.setdefault("executed_action", action)
    decision.setdefault("action_modified", False)
    return action, True, learning_mask, decision


def finalize_action_audit(
    selection_info: dict,
    shield_decision: dict | None,
) -> dict:
    """合并策略提议与 shield 执行结果，形成单一动作审计记录。"""
    audit = dict(selection_info)
    shield = dict(shield_decision or {})
    selected_by_agent = bool(
        audit.get("selected_by_agent", True)
    )
    proposed_action = (
        audit.get("proposed_action")
        if selected_by_agent
        else None
    )
    executed_action = int(
        shield.get(
            "executed_action",
            audit.get("executed_action", audit["action"]),
        )
    )
    fallback = bool(
        shield.get("fallback_applied", False)
        or not selected_by_agent
    )
    modified = bool(
        shield.get("action_modified", False)
        or (
            proposed_action is not None
            and int(proposed_action) != executed_action
        )
    )
    intervened = bool(
        shield.get("shield_intervened", False)
    )
    policy_type = str(
        audit.get(
            "policy_selection_type",
            audit.get("selection_type", "unclassified_action"),
        )
    )
    if fallback:
        action_source = "fallback_action"
    elif modified or intervened:
        action_source = "shield_correction"
    else:
        action_source = policy_type

    audit.update(
        {
            "proposed_action": (
                int(proposed_action)
                if proposed_action is not None
                else None
            ),
            "executed_action": executed_action,
            "selected_by_agent": selected_by_agent,
            "policy_selection_type": policy_type,
            "action_source": action_source,
            "action_modified": modified,
            "shield_intervened": intervened,
            "fallback_triggered": fallback,
            "modification_reason": shield.get(
                "modification_reason",
                "none",
            ),
        }
    )
    return audit


def performance_reward_components(
    info: dict,
    *,
    total_performance_reward: float,
) -> dict:
    """提取五类性能分量；安全代价不在返回结构中。"""
    total = float(total_performance_reward)
    if all(
        key in info
        for key in (
            "energy_reward",
            "completion_reward",
            "waiting_reward",
            "utilization_reward",
            "communication_reward",
            "total_performance_reward",
        )
    ):
        return {
            "energy_reward": float(info["energy_reward"]),
            "completion_reward": float(
                info["completion_reward"]
            ),
            "waiting_reward": float(info["waiting_reward"]),
            "utilization_reward": float(
                info["utilization_reward"]
            ),
            "communication_reward": float(
                info["communication_reward"]
            ),
            "total_performance_reward": float(
                info["total_performance_reward"]
            ),
        }
    # 兼容测试替身或旧环境：整个性能 reward 视作 energy 分量。
    return {
        "energy_reward": total,
        "completion_reward": 0.0,
        "waiting_reward": 0.0,
        "utilization_reward": 0.0,
        "communication_reward": 0.0,
        "total_performance_reward": total,
    }


def safe_replay_metadata(
    layer_state: dict,
    action_audit: dict,
    safety_info: dict,
    *,
    shield_decision: dict | None = None,
) -> dict:
    """从层级状态、动作审计与环境 info 提取具名 replay 字段。"""
    decision = dict(shield_decision or {})
    final_mask = np.asarray(
        layer_state.get(
            "final_action_mask",
            layer_state.get("mask"),
        ),
        dtype=np.float32,
    ).reshape(-1)
    legal_mask = np.asarray(
        layer_state.get("legal_action_mask", final_mask),
        dtype=np.float32,
    ).reshape(-1)
    safety_mask = np.asarray(
        layer_state.get("safety_action_mask", final_mask),
        dtype=np.float32,
    ).reshape(-1)

    risk_finish = decision.get("predicted_risk")
    margin = decision.get("safety_margin")
    workflow_rows = list(
        safety_info.get("workflow_safety", []) or []
    )
    if risk_finish is None:
        risk_finish = max(
            (
                float(
                    row.get(
                        "fuzzy_finish_risk",
                        row.get("risk_finish", 0.0),
                    )
                )
                for row in workflow_rows
            ),
            default=0.0,
        )
    if margin is None:
        margin = safety_info.get(
            "minimum_fuzzy_safety_margin",
            safety_info.get("min_fuzzy_safety_margin", 0.0),
        )

    return {
        "legal_action_mask": legal_mask.copy(),
        "safety_action_mask": safety_mask.copy(),
        "final_action_mask": final_mask.copy(),
        "fallback_triggered": bool(
            action_audit.get("fallback_triggered", False)
            or decision.get("fallback_applied", False)
        ),
        "fuzzy_safety_margin": float(margin or 0.0),
        "predicted_risk_finish": float(risk_finish or 0.0),
        "violation_flag": bool(
            safety_info.get("violation_flag", False)
            or int(
                safety_info.get(
                    "deadline_violation_count",
                    0,
                )
            )
            > 0
        ),
        "manager_phase_id": int(
            safety_info.get("manager_phase_id", 0)
        ),
    }


def apply_env_scales(env: CloudWorkflowEnv_VMAgents, cfg) -> None:
    """把配置中的 reward 缩放和归一化参数写入训练环境。"""
    env.energy_reward_scale = cfg.energy_reward_scale
    env.task_baseline_norm = cfg.task_baseline_norm
    env.energy_norm_per_mi_ref = cfg.energy_norm_per_mi_ref
    env.alpha_delay_host = cfg.alpha_delay_host
    env.alpha_delay_vm = cfg.alpha_delay_vm


def sync_env_scales(eval_env: CloudWorkflowEnv_VMAgents, env_kwargs: dict) -> None:
    """把训练时使用的尺度参数同步到评估环境。"""
    scale_keys = [
        "energy_reward_scale",
        "task_baseline_norm",
        "energy_norm_per_mi_ref",
        "alpha_delay_host",
        "alpha_delay_vm",
    ]
    for key in scale_keys:
        if key in env_kwargs:
            setattr(eval_env, key, env_kwargs[key])


def compute_episode_task_lateness_metrics(env: CloudWorkflowEnv_VMAgents) -> tuple[float, float, float]:
    """计算当前 episode 的任务迟延和工作流平均迟延。

    返回值：
    - ep_total_late：所有已完成任务的 tardiness 总和。
    - ep_avg_late：已完成任务的平均 tardiness。
    - wf_avg_late：按 workflow 聚合后的平均 tardiness。
    """
    total_late = 0.0
    task_cnt = 0
    wf_late_sum = {}

    n_tasks = len(getattr(env, "task_state", []))
    for tid in range(n_tasks):
        if env.task_state[tid] != "Finished":
            continue
        if tid >= len(env.task_baseline_finish):
            continue

        finish_t = float(env.task_end_time[tid])
        deadline_t = float(env.task_baseline_finish[tid])
        late = max(0.0, finish_t - deadline_t)

        total_late += late
        task_cnt += 1

        wf_id, _local_id = env.task_meta[tid]
        wf_late_sum[wf_id] = wf_late_sum.get(wf_id, 0.0) + late

    ep_avg_late = total_late / max(task_cnt, 1)

    if getattr(env, "workflows_per_episode", None) is not None:
        n_wf = int(env.workflows_per_episode)
        wf_total = 0.0
        for wf_id in range(n_wf):
            wf_total += wf_late_sum.get(wf_id, 0.0)
        wf_avg_late = wf_total / max(n_wf, 1)
    else:
        wf_avg_late = (sum(wf_late_sum.values()) / max(len(wf_late_sum), 1)) if len(wf_late_sum) > 0 else 0.0

    return float(total_late), float(ep_avg_late), float(wf_avg_late)


def warmup_ready(agent: D3QNAgent, frac: float = 0.1) -> bool:
    """判断 replay buffer 是否达到 warmup 门槛，可以开始执行 update()。"""
    maxlen = agent.buffer.maxlen or 0
    need = int(maxlen * float(frac))
    need = max(need, int(agent.batch_size))
    return len(agent.buffer) >= need


def print_device_info(vm_agent: D3QNAgent, host_agent: D3QNAgent, manager_agent: D3QNAgent) -> None:
    """打印 CUDA 可用性以及三个 agent 当前所在设备。"""
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current_device:", torch.cuda.current_device())
        print("device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("vm_agent online device:", next(vm_agent.online.parameters()).device)
    print("host_agent online device:", next(host_agent.online.parameters()).device)
    print("manager_agent online device:", next(manager_agent.online.parameters()).device)
    if all(
        agent.safe_rl_enabled
        for agent in (vm_agent, host_agent, manager_agent)
    ):
        print(
            "vm_agent q_c device:",
            next(vm_agent.q_c_online.parameters()).device,
        )
        print(
            "host_agent q_c device:",
            next(host_agent.q_c_online.parameters()).device,
        )
        print(
            "manager_agent q_c device:",
            next(manager_agent.q_c_online.parameters()).device,
        )
