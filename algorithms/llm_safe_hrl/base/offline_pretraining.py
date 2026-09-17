"""用安全示范轨迹离线初始化三层 Q_r/Q_c 和可选行为倾向。

本模块不调用 :meth:`D3QNAgent.update`，不推进 epsilon、在线 update 计数或
PER 状态，也不改变 target 更新频率。离线 epoch 内 target 网络保持冻结；
可选的末尾单次同步仅用于把预训练参数作为在线训练初值，之后仍完全使用原
``D3QNAgent.update`` 的 Polyak/硬更新逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from base.d3qn_agent import D3QNAgent
from base.safe_demonstration import (
    DEMONSTRATION_LAYERS,
    load_demonstration_split,
)
from base.safe_replay import (
    SafeReplayTransition,
    stack_safe_replay_transitions,
)


@dataclass(frozen=True)
class OfflinePretrainingOptions:
    """安全示范离线预训练配置；默认关闭。"""

    enabled: bool = False
    dataset_manifest_path: str | None = None
    epochs: int = 5
    batch_size: int = 128
    performance_learning_rate: float = 1e-4
    safety_learning_rate: float = 1e-4
    train_q_r: bool = True
    train_q_c: bool = True
    behavior_cloning_enabled: bool = False
    behavior_cloning_weight: float = 0.1
    sync_targets_after_pretraining: bool = True
    random_seed: int = 17

    def __post_init__(self) -> None:
        if self.enabled and not self.dataset_manifest_path:
            raise ValueError(
                "enabled offline pretraining requires a dataset "
                "manifest"
            )
        if int(self.epochs) <= 0:
            raise ValueError("offline pretraining epochs must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError(
                "offline pretraining batch_size must be positive"
            )
        for name, value in (
            (
                "performance_learning_rate",
                self.performance_learning_rate,
            ),
            (
                "safety_learning_rate",
                self.safety_learning_rate,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(float(self.behavior_cloning_weight))
            or float(self.behavior_cloning_weight) < 0.0
        ):
            raise ValueError(
                "behavior_cloning_weight must be finite and "
                "non-negative"
            )
        if not (
            self.train_q_r
            or self.train_q_c
            or self.behavior_cloning_enabled
        ):
            raise ValueError(
                "offline pretraining must enable Q_r, Q_c or "
                "behavior cloning"
            )


def _agent_runtime_snapshot(agent: D3QNAgent) -> dict:
    return {
        "updates": int(agent._updates),
        "epsilon_steps": int(agent._eps_steps),
        "target_update_tau": float(agent.target_update_tau),
        "target_update_freq": int(agent.target_update_freq),
        "per_beta_count": int(agent.per_beta_count),
        "replay_size": len(agent.buffer),
    }


def _validate_agents_and_dataset(
    agents: Mapping[str, D3QNAgent],
    manifest: Mapping,
) -> None:
    if set(agents) != set(DEMONSTRATION_LAYERS):
        raise ValueError(
            "offline pretraining agents must contain manager, host "
            "and vm"
        )
    dimensions = manifest.get("layer_dimensions", {})
    schemas = manifest.get("observation_schema_versions", {})
    for layer in DEMONSTRATION_LAYERS:
        agent = agents[layer]
        if not isinstance(agent, D3QNAgent):
            raise TypeError(f"{layer} is not a D3QNAgent")
        if not agent.safe_rl_enabled:
            raise ValueError(
                f"{layer} agent must enable safe_rl for Q_c "
                "pretraining"
            )
        expected = dimensions.get(layer, {})
        if int(expected.get("input_dim", -1)) != agent.input_dim:
            raise ValueError(
                f"{layer} demonstration observation dimension mismatch"
            )
        if int(expected.get("action_dim", -1)) != agent.output_dim:
            raise ValueError(
                f"{layer} demonstration action dimension mismatch"
            )
        if str(schemas.get(layer, "")) != str(
            agent.observation_schema_version
        ):
            raise ValueError(
                f"{layer} demonstration observation schema mismatch"
            )


def _tensor_batch(
    agent: D3QNAgent,
    rows: Sequence[SafeReplayTransition],
) -> dict:
    batch = stack_safe_replay_transitions(
        rows,
        input_dim=agent.input_dim,
        action_dim=agent.output_dim,
    )
    device = agent.device
    return {
        "state": torch.as_tensor(
            batch["state"],
            dtype=torch.float32,
            device=device,
        ),
        "executed_action": torch.as_tensor(
            batch["executed_action"],
            dtype=torch.int64,
            device=device,
        ),
        "performance_reward": torch.as_tensor(
            batch["performance_reward"],
            dtype=torch.float32,
            device=device,
        ),
        "safety_cost": torch.as_tensor(
            batch["safety_cost"],
            dtype=torch.float32,
            device=device,
        ),
        "next_state": torch.as_tensor(
            batch["next_state"],
            dtype=torch.float32,
            device=device,
        ),
        "next_final_action_mask": torch.as_tensor(
            batch["next_final_action_mask"],
            dtype=torch.float32,
            device=device,
        ),
        "done": torch.as_tensor(
            batch["done"],
            dtype=torch.float32,
            device=device,
        ),
        "final_action_mask": torch.as_tensor(
            batch["final_action_mask"],
            dtype=torch.float32,
            device=device,
        ),
        "fallback_triggered": torch.as_tensor(
            batch["fallback_triggered"],
            dtype=torch.bool,
            device=device,
        ),
    }


def _batch_losses(
    agent: D3QNAgent,
    rows: Sequence[SafeReplayTransition],
    options: OfflinePretrainingOptions,
) -> dict:
    batch = _tensor_batch(agent, rows)
    state = batch["state"]
    action = batch["executed_action"]
    reward = batch["performance_reward"]
    cost = batch["safety_cost"]
    next_state = batch["next_state"]
    next_mask = batch["next_final_action_mask"]
    done = batch["done"]

    reward_target, safety_target, _ = (
        agent._compute_double_dqn_targets(
            next_state,
            next_mask,
            done,
            reward,
            cost,
        )
    )
    q_r_values = agent.q_r_online(state)
    q_r_action = q_r_values.gather(
        1,
        action.view(-1, 1),
    ).squeeze(1)
    q_r_loss = F.smooth_l1_loss(
        q_r_action,
        reward_target,
    )

    q_c_values = agent.q_c_online(state)
    q_c_action = q_c_values.gather(
        1,
        action.view(-1, 1),
    ).squeeze(1)
    q_c_loss = F.smooth_l1_loss(
        q_c_action,
        safety_target,
    )

    bc_loss = torch.zeros((), device=agent.device)
    bc_count = 0
    if options.behavior_cloning_enabled:
        # fallback 属于控制器动作，不是 Agent 策略动作，不能作为 BC 标签。
        eligible = ~batch["fallback_triggered"]
        if torch.any(eligible):
            final_mask = batch["final_action_mask"][eligible] > 0.5
            logits = q_r_values[eligible].masked_fill(
                ~final_mask,
                -1e9,
            )
            labels = action[eligible]
            if not torch.all(
                final_mask.gather(
                    1,
                    labels.view(-1, 1),
                ).squeeze(1)
            ):
                raise ValueError(
                    "behavior-cloning action is outside final mask"
                )
            # Q_c 继续只由 safety cost 学习；BC 仅初始化 Q_r 的动作倾向。
            bc_loss = F.cross_entropy(logits, labels)
            bc_count = int(torch.count_nonzero(eligible).item())
    return {
        "q_r_loss": q_r_loss,
        "q_c_loss": q_c_loss,
        "behavior_cloning_loss": bc_loss,
        "behavior_cloning_count": bc_count,
    }


def _run_layer_epoch(
    agent: D3QNAgent,
    transitions: Sequence[SafeReplayTransition],
    options: OfflinePretrainingOptions,
    *,
    rng: np.random.RandomState,
    training: bool,
    q_r_optimizer=None,
    q_c_optimizer=None,
) -> dict:
    rows = list(transitions)
    indices = np.arange(len(rows), dtype=np.int64)
    if training:
        rng.shuffle(indices)
    totals = {
        "q_r_loss": 0.0,
        "q_c_loss": 0.0,
        "behavior_cloning_loss": 0.0,
        "behavior_cloning_count": 0,
        "transition_count": 0,
        "batch_count": 0,
    }
    batch_size = int(options.batch_size)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch_rows = [rows[int(index)] for index in selected]
        if training:
            losses = _batch_losses(agent, batch_rows, options)
            q_r_objective = None
            if options.train_q_r:
                q_r_objective = losses["q_r_loss"]
            if options.behavior_cloning_enabled:
                bc_term = (
                    float(options.behavior_cloning_weight)
                    * losses["behavior_cloning_loss"]
                )
                q_r_objective = (
                    bc_term
                    if q_r_objective is None
                    else q_r_objective + bc_term
                )
            if q_r_objective is not None:
                q_r_optimizer.zero_grad()
                q_r_objective.backward()
                if agent.grad_clip is not None and agent.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        agent.q_r_online.parameters(),
                        agent.grad_clip,
                    )
                q_r_optimizer.step()
            if options.train_q_c:
                q_c_optimizer.zero_grad()
                losses["q_c_loss"].backward()
                if agent.grad_clip is not None and agent.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        agent.q_c_online.parameters(),
                        agent.grad_clip,
                    )
                q_c_optimizer.step()
        else:
            with torch.no_grad():
                losses = _batch_losses(
                    agent,
                    batch_rows,
                    options,
                )
        count = len(batch_rows)
        for name in (
            "q_r_loss",
            "q_c_loss",
            "behavior_cloning_loss",
        ):
            totals[name] += float(losses[name].item()) * count
        totals["behavior_cloning_count"] += int(
            losses["behavior_cloning_count"]
        )
        totals["transition_count"] += count
        totals["batch_count"] += 1
    denominator = max(totals["transition_count"], 1)
    for name in (
        "q_r_loss",
        "q_c_loss",
        "behavior_cloning_loss",
    ):
        totals[name] /= denominator
    return totals


def pretrain_agents_from_demonstrations(
    agents: Mapping[str, D3QNAgent],
    options: OfflinePretrainingOptions,
) -> dict:
    """在严格 train/validation 数据上离线初始化三层双价值网络。"""
    if not options.enabled:
        return {
            "enabled": False,
            "reason": "offline_pretraining_disabled",
        }
    train_data = load_demonstration_split(
        options.dataset_manifest_path,
        "train",
        require_safe=True,
    )
    validation_data = load_demonstration_split(
        options.dataset_manifest_path,
        "validation",
        require_safe=True,
    )
    if (
        train_data["manifest"]["manifest_content_sha256"]
        != validation_data["manifest"]["manifest_content_sha256"]
    ):
        raise ValueError(
            "train and validation demonstrations must use one "
            "immutable manifest"
        )
    _validate_agents_and_dataset(
        agents,
        train_data["manifest"],
    )

    before = {
        layer: _agent_runtime_snapshot(agents[layer])
        for layer in DEMONSTRATION_LAYERS
    }
    optimizers = {}
    for layer in DEMONSTRATION_LAYERS:
        agent = agents[layer]
        optimizers[layer] = {
            "q_r": (
                torch.optim.Adam(
                    agent.q_r_online.parameters(),
                    lr=float(
                        options.performance_learning_rate
                    ),
                )
                if (
                    options.train_q_r
                    or options.behavior_cloning_enabled
                )
                else None
            ),
            "q_c": (
                torch.optim.Adam(
                    agent.q_c_online.parameters(),
                    lr=float(options.safety_learning_rate),
                )
                if options.train_q_c
                else None
            ),
        }

    rng = np.random.RandomState(int(options.random_seed))
    history = []
    for epoch in range(int(options.epochs)):
        epoch_record = {"epoch": epoch + 1, "layers": {}}
        for layer in DEMONSTRATION_LAYERS:
            agent = agents[layer]
            train_metrics = _run_layer_epoch(
                agent,
                train_data["transitions"][layer],
                options,
                rng=rng,
                training=True,
                q_r_optimizer=optimizers[layer]["q_r"],
                q_c_optimizer=optimizers[layer]["q_c"],
            )
            validation_metrics = _run_layer_epoch(
                agent,
                validation_data["transitions"][layer],
                options,
                rng=rng,
                training=False,
            )
            epoch_record["layers"][layer] = {
                "train": train_metrics,
                "validation": validation_metrics,
            }
        history.append(epoch_record)

    if options.sync_targets_after_pretraining:
        for layer in DEMONSTRATION_LAYERS:
            agent = agents[layer]
            if (
                options.train_q_r
                or options.behavior_cloning_enabled
            ):
                agent.q_r_target.load_state_dict(
                    agent.q_r_online.state_dict()
                )
            if options.train_q_c:
                agent.q_c_target.load_state_dict(
                    agent.q_c_online.state_dict()
                )

    after = {
        layer: _agent_runtime_snapshot(agents[layer])
        for layer in DEMONSTRATION_LAYERS
    }
    for layer in DEMONSTRATION_LAYERS:
        if before[layer] != after[layer]:
            raise RuntimeError(
                "offline pretraining changed online replay, epsilon, "
                "update counter or target-update configuration"
            )

    return {
        "enabled": True,
        "dataset_manifest_path": str(
            options.dataset_manifest_path
        ),
        "dataset_manifest_sha256": train_data["manifest"][
            "manifest_content_sha256"
        ],
        "train_episode_count": len(train_data["episodes"]),
        "validation_episode_count": len(
            validation_data["episodes"]
        ),
        "train_transition_count": {
            layer: len(train_data["transitions"][layer])
            for layer in DEMONSTRATION_LAYERS
        },
        "validation_transition_count": {
            layer: len(validation_data["transitions"][layer])
            for layer in DEMONSTRATION_LAYERS
        },
        "train_q_r": bool(options.train_q_r),
        "train_q_c": bool(options.train_q_c),
        "behavior_cloning_enabled": bool(
            options.behavior_cloning_enabled
        ),
        "sync_targets_after_pretraining": bool(
            options.sync_targets_after_pretraining
        ),
        "online_target_update_logic": (
            "unchanged_D3QNAgent.update"
        ),
        "history": history,
    }


__all__ = [
    "OfflinePretrainingOptions",
    "pretrain_agents_from_demonstrations",
]
