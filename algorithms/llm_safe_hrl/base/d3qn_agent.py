# -*- coding: utf-8 -*-
"""Unified Dueling Double DQN agent with action masks.

This module replaces the previous variants:

- d3qn_agent.py: vanilla replay buffer and a shallow dueling head.
- d3qn_agent_EXDehid.py: vanilla replay buffer and a deeper dueling head.
- d3qn_agent_PER.py: prioritized experience replay.

The default network head matches the formerly active EXDehid variant.
Legacy checkpoints remain loadable in legacy observation mode when safe value
learning is disabled. Safe agents learn logically independent performance
``Q_r`` and safety-cost ``Q_c`` networks; schema, dimensions and safe/legacy
checkpoint modes are validated before state-dict loading.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.safe_replay import (
    SAFE_REPLAY_BUFFER_SCHEMA_VERSION,
    SAFE_REPLAY_TRANSITION_SCHEMA_VERSION,
    SafeReplayTransition,
    stack_safe_replay_transitions,
)


_UNSET = object()


def _as_tuple(values: Sequence[int] | int | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    if isinstance(values, int):
        return (int(values),)
    return tuple(int(v) for v in values)


class DuelingQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        head_hidden_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        dims = [int(input_dim)] + [int(v) for v in hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.feature = nn.Sequential(*layers)

        last_h = dims[-1]
        if head_hidden_dims is None:
            # EXDehid-compatible default: two hidden layers in each dueling head.
            head_hidden_dims = (last_h, last_h)

        self.adv = self._mlp(last_h, head_hidden_dims, int(output_dim))
        self.val = self._mlp(last_h, head_hidden_dims, 1)

    @staticmethod
    def _mlp(in_dim: int, hidden_dims: Iterable[int], out_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = int(in_dim)
        for h in hidden_dims:
            h = int(h)
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, int(out_dim)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        a = self.adv(h)
        v = self.val(h)
        return v + (a - a.mean(dim=1, keepdim=True))


def _polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_t, p_o in zip(target.parameters(), online.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


class D3QNAgent:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        buffer_size: int = 100000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 150000,
        target_update_freq: int = 2000,
        target_update_tau: float = 0.005,
        grad_clip: float = 10.0,
        hidden_dims: Sequence[int] = (512, 256),
        head_hidden_dims: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 150000,
        observation_schema_version: str = "legacy_observation",
        safe_rl_enabled: bool = False,
        safety_discount: float = 0.95,
        safety_learning_rate: float = 3e-4,
        safety_loss_weight: float = 1.0,
        initial_lagrange_multiplier: float = 1.0,
        safe_replay_near_boundary_margin: float = 0.0,
        safe_per_combined_priority: bool = False,
        safe_per_performance_td_weight: float = 1.0,
        safe_per_safety_td_weight: float = 1.0,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.observation_schema_version = str(
            observation_schema_version
        ).strip()
        if not self.observation_schema_version:
            raise ValueError(
                "observation_schema_version must be non-empty"
            )
        self.gamma = float(gamma)
        self.safe_rl_enabled = bool(safe_rl_enabled)
        self.safety_discount = float(safety_discount)
        self.safety_learning_rate = float(safety_learning_rate)
        self.safety_loss_weight = float(safety_loss_weight)
        self.lagrange_multiplier = float(
            initial_lagrange_multiplier
        )
        self.safe_replay_near_boundary_margin = float(
            safe_replay_near_boundary_margin
        )
        self.safe_per_combined_priority = bool(
            safe_per_combined_priority
        )
        self.safe_per_performance_td_weight = float(
            safe_per_performance_td_weight
        )
        self.safe_per_safety_td_weight = float(
            safe_per_safety_td_weight
        )
        # 动态拉格朗日控制器由训练编排层共享；Agent 只保留最近一次从
        # checkpoint 读取的控制器状态，避免三个 Agent 各自更新同一代价。
        self.lagrange_controller_state = None
        if (
            not np.isfinite(self.safety_discount)
            or not 0.0 <= self.safety_discount <= 1.0
        ):
            raise ValueError("safety_discount must be in [0, 1]")
        if (
            not np.isfinite(self.safety_learning_rate)
            or self.safety_learning_rate <= 0.0
        ):
            raise ValueError(
                "safety_learning_rate must be finite and positive"
            )
        if (
            not np.isfinite(self.safety_loss_weight)
            or self.safety_loss_weight < 0.0
        ):
            raise ValueError(
                "safety_loss_weight must be finite and non-negative"
            )
        if (
            not np.isfinite(self.lagrange_multiplier)
            or self.lagrange_multiplier < 0.0
        ):
            raise ValueError(
                "initial_lagrange_multiplier must be finite and "
                "non-negative"
            )
        if (
            not np.isfinite(
                self.safe_replay_near_boundary_margin
            )
            or self.safe_replay_near_boundary_margin < 0.0
        ):
            raise ValueError(
                "safe_replay_near_boundary_margin must be finite "
                "and non-negative"
            )
        for name, value in (
            (
                "safe_per_performance_td_weight",
                self.safe_per_performance_td_weight,
            ),
            (
                "safe_per_safety_td_weight",
                self.safe_per_safety_td_weight,
            ),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative"
                )
        if (
            self.safe_per_combined_priority
            and (
                self.safe_per_performance_td_weight
                + self.safe_per_safety_td_weight
            )
            <= 0.0
        ):
            raise ValueError(
                "combined safe PER requires a positive TD weight"
            )
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grad_clip = float(grad_clip) if grad_clip is not None else None

        self.hidden_dims = _as_tuple(hidden_dims) or (512, 256)
        self.head_hidden_dims = _as_tuple(head_hidden_dims)

        self.online = DuelingQNet(self.input_dim, self.output_dim, self.hidden_dims, self.head_hidden_dims).to(self.device)
        self.target = DuelingQNet(self.input_dim, self.output_dim, self.hidden_dims, self.head_hidden_dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        # 兼容旧属性名：online/target/optim 始终表示性能价值 Q_r。
        self.q_r_online = self.online
        self.q_r_target = self.target
        self.q_r_optim = self.optim

        if self.safe_rl_enabled:
            self.q_c_online = DuelingQNet(
                self.input_dim,
                self.output_dim,
                self.hidden_dims,
                self.head_hidden_dims,
            ).to(self.device)
            self.q_c_target = DuelingQNet(
                self.input_dim,
                self.output_dim,
                self.hidden_dims,
                self.head_hidden_dims,
            ).to(self.device)
            self.q_c_target.load_state_dict(
                self.q_c_online.state_dict()
            )
            self.q_c_optim = torch.optim.Adam(
                self.q_c_online.parameters(),
                lr=self.safety_learning_rate,
            )
        else:
            self.q_c_online = None
            self.q_c_target = None
            self.q_c_optim = None

        self.use_per = bool(use_per)
        if self.use_per:
            self.buffer: list[tuple] = []
            self.priorities: list[float] = []
        else:
            self.buffer = deque(maxlen=self.buffer_size)
            self.priorities = []

        self._eps_start = float(eps_start)
        self._eps_end = float(eps_end)
        self._eps_decay_steps = max(1, int(eps_decay_steps))
        self._eps_steps = 0

        self._updates = 0
        self.target_update_freq = max(1, int(target_update_freq))
        self.target_update_tau = float(target_update_tau)
        self._action_calls = 0

        self.per_alpha = float(per_alpha)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_steps = max(1, int(per_beta_steps))
        self.per_beta_count = 0
        self.last_update_info = None
        # 最近一次策略提议的只读审计快照。环境最终执行动作仍由
        # train_runner/shield 决定，不能用该字段替代 executed action。
        self.last_action_selection = None

    def epsilon(self) -> float:
        t = min(self._eps_steps, self._eps_decay_steps)
        frac = 1.0 - (t / self._eps_decay_steps)
        return float(self._eps_end + (self._eps_start - self._eps_end) * frac)

    def _per_beta(self) -> float:
        t = min(self.per_beta_count, self.per_beta_steps)
        frac = t / self.per_beta_steps
        return float(self.per_beta_start + (self.per_beta_end - self.per_beta_start) * frac)

    def _valid_indices_from_mask(self, mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask > 0.5)
        if idx.size == 0:
            if self.safe_rl_enabled:
                raise ValueError(
                    "safe value policy cannot select from an empty "
                    "final_action_mask; use the deterministic fallback "
                    "controller"
                )
            return np.array([self.output_dim - 1], dtype=np.int64)
        return idx

    def select_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
        count_step: bool = True,
    ) -> int:
        """保持历史整数返回接口，详细审计信息由新方法提供。"""
        return int(
            self.select_action_with_info(
                state,
                mask,
                deterministic=deterministic,
                count_step=count_step,
            )["action"]
        )

    def select_action_with_info(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
        count_step: bool = True,
    ) -> dict:
        """执行 epsilon-greedy，并返回策略提议的审计信息。

        safe 模式下调用方必须传入 ``final_action_mask``。随机探索和贪心
        利用都只在该 mask 内进行；全零 mask 继续明确报错，交由训练编排
        层调用确定性回退控制器。
        """
        if count_step:
            self._action_calls += 1

        selection_mask = np.asarray(mask, dtype=np.float32).reshape(-1)
        if selection_mask.size != self.output_dim:
            raise ValueError(
                "action mask dimension mismatch: "
                f"expected {self.output_dim}, got {selection_mask.size}"
            )
        if not np.all(np.isfinite(selection_mask)):
            raise ValueError("action mask contains NaN or infinite values")
        selection_mask = (selection_mask > 0.5).astype(np.float32)
        valid = self._valid_indices_from_mask(selection_mask)
        eps = 0.0 if deterministic else self.epsilon()
        random_branch = (
            (not deterministic)
            and (random.random() < eps)
        )
        if random_branch:
            action = int(np.random.choice(valid))
            selection_type = (
                "random_safe_exploration"
                if self.safe_rl_enabled
                else "random_exploration"
            )
        else:
            with torch.no_grad():
                s = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                q_r = self.online(s).squeeze(0)
                if self.safe_rl_enabled:
                    q_c = self.q_c_online(s).squeeze(0)
                    score = (
                        q_r
                        - self.lagrange_multiplier * q_c
                    ).clone()
                else:
                    score = q_r.clone()
                m = torch.tensor(
                    selection_mask,
                    dtype=torch.float32,
                    device=self.device,
                )
                score[m <= 0.5] = -torch.inf
                action = int(torch.argmax(score).item())
            selection_type = (
                "greedy_safe_action"
                if self.safe_rl_enabled
                else "greedy_action"
            )

        selection = {
            "action": int(action),
            "proposed_action": int(action),
            "selected_by_agent": True,
            "selection_type": selection_type,
            "random_exploration": bool(random_branch),
            "safe_rl_enabled": bool(self.safe_rl_enabled),
            "deterministic": bool(deterministic),
            "epsilon": float(eps),
            "valid_action_count": int(valid.size),
            "selection_mask": selection_mask.copy(),
        }
        self.last_action_selection = selection
        return selection

    def remember(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: float,
        cost: float | None = None,
        proposed_action=_UNSET,
        action_source: str | None = None,
        policy_selection_type: str | None = None,
        action_modified: bool = False,
        legal_action_mask: np.ndarray | None = None,
        safety_action_mask: np.ndarray | None = None,
        fallback_triggered: bool = False,
        fuzzy_safety_margin: float = 0.0,
        predicted_risk_finish: float = 0.0,
        violation_flag: bool = False,
        manager_phase_id: int = 0,
        performance_reward_components=None,
    ) -> None:
        if self.safe_rl_enabled:
            if cost is None or not np.isfinite(float(cost)):
                raise ValueError(
                    "safe value replay requires a finite safety cost"
                )
            final_mask = np.asarray(
                mask,
                dtype=np.float32,
            ).reshape(-1)
            legal_mask = (
                final_mask
                if legal_action_mask is None
                else legal_action_mask
            )
            safety_mask = (
                final_mask
                if safety_action_mask is None
                else safety_action_mask
            )
            proposal = (
                int(action)
                if proposed_action is _UNSET
                else proposed_action
            )
            transition = SafeReplayTransition(
                state=state,
                proposed_action=proposal,
                executed_action=int(action),
                performance_reward=float(reward),
                safety_cost=float(cost),
                next_state=next_state,
                done=float(done),
                legal_action_mask=legal_mask,
                safety_action_mask=safety_mask,
                final_action_mask=final_mask,
                next_final_action_mask=next_mask,
                shield_modified=bool(action_modified),
                fallback_triggered=bool(fallback_triggered),
                fuzzy_safety_margin=float(
                    fuzzy_safety_margin
                ),
                predicted_risk_finish=float(
                    predicted_risk_finish
                ),
                violation_flag=bool(violation_flag),
                manager_phase_id=int(manager_phase_id),
                near_boundary_margin=(
                    self.safe_replay_near_boundary_margin
                ),
                action_source=str(
                    action_source or "unspecified_safe_action"
                ),
                policy_selection_type=str(
                    policy_selection_type
                    or action_source
                    or "unspecified_safe_action"
                ),
                performance_reward_components=(
                    performance_reward_components or {}
                ),
            )
        else:
            # 保持旧 replay tuple 的 7 字段结构。
            transition = (
                np.array(state, copy=False),
                np.array(mask, copy=False),
                int(action),
                float(reward),
                np.array(next_state, copy=False),
                np.array(next_mask, copy=False),
                float(done),
            )

        if not self.use_per:
            self.buffer.append(transition)
            return

        max_prio = max(self.priorities) if self.priorities else 1.0
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(transition)
            self.priorities.append(max_prio)

    def _sample_batch(self):
        n = len(self.buffer)
        if self.use_per:
            prios = np.asarray(self.priorities, dtype=np.float32) + 1e-6
            probs = prios ** self.per_alpha
            probs /= probs.sum()
            indices = np.random.choice(n, self.batch_size, p=probs)

            beta = self._per_beta()
            self.per_beta_count += 1
            weights_np = (n * probs[indices]) ** (-beta)
            weights_np /= weights_np.max()
            weights = torch.tensor(weights_np, dtype=torch.float32, device=self.device)
        else:
            indices = np.array(random.sample(range(n), self.batch_size), dtype=np.int64)
            weights = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)

        batch = [self.buffer[int(i)] for i in indices]
        return indices, weights, batch

    def _compute_double_dqn_targets(
        self,
        next_states: torch.Tensor,
        next_masks: torch.Tensor,
        dones: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor | None = None,
    ):
        """按当前受约束策略计算 Q_r/Q_c Double-DQN 目标。

        safe 模式的下一动作由
        ``argmax(Q_r_online - lambda * Q_c_online)`` 在下一状态最终
        mask 内选择，再分别从两个 target 网络取值。无有效下一动作时两类
        bootstrap 均为 0。
        """
        with torch.no_grad():
            q_r_next_online = self.online(next_states)
            if self.safe_rl_enabled:
                if costs is None:
                    raise ValueError(
                        "safe Bellman target requires safety costs"
                    )
                q_c_next_online = self.q_c_online(next_states)
                next_scores = (
                    q_r_next_online
                    - self.lagrange_multiplier
                    * q_c_next_online
                )
            else:
                next_scores = q_r_next_online

            valid = next_masks > 0.5
            masked_scores = next_scores.masked_fill(
                ~valid,
                -torch.inf,
            )
            has_valid = valid.any(dim=1)
            # 全零 mask 的行稍后清零 bootstrap；这里的索引只作占位。
            next_actions = masked_scores.argmax(
                dim=1,
                keepdim=True,
            )

            q_r_next_target = self.target(next_states).masked_fill(
                ~valid,
                0.0,
            )
            q_r_next = q_r_next_target.gather(
                1,
                next_actions,
            ).squeeze(1)
            q_r_next = torch.where(
                has_valid,
                q_r_next,
                torch.zeros_like(q_r_next),
            )
            reward_targets = (
                rewards
                + (1.0 - dones) * self.gamma * q_r_next
            )

            safety_targets = None
            if self.safe_rl_enabled:
                q_c_next_target = self.q_c_target(
                    next_states
                ).masked_fill(~valid, 0.0)
                q_c_next = q_c_next_target.gather(
                    1,
                    next_actions,
                ).squeeze(1)
                q_c_next = torch.where(
                    has_valid,
                    q_c_next,
                    torch.zeros_like(q_c_next),
                )
                safety_targets = (
                    costs
                    + (1.0 - dones)
                    * self.safety_discount
                    * q_c_next
                )
        return reward_targets, safety_targets, next_actions.squeeze(1)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        indices, weights, batch = self._sample_batch()
        if self.safe_rl_enabled:
            safe_batch = stack_safe_replay_transitions(
                batch,
                input_dim=self.input_dim,
                action_dim=self.output_dim,
            )
            s = safe_batch["state"]
            m = safe_batch["final_action_mask"]
            a = safe_batch["executed_action"]
            r = safe_batch["performance_reward"]
            c = safe_batch["safety_cost"]
            s2 = safe_batch["next_state"]
            m2 = safe_batch["next_final_action_mask"]
            d = safe_batch["done"]
            # proposed/shield/fallback/category 只供审计与采样诊断；
            # Q_r/Q_c gather 始终使用上面的 executed_action。
            proposed_actions = safe_batch["proposed_action"]
            action_modified_flags = safe_batch[
                "shield_modified"
            ]
            fallback_flags = safe_batch["fallback_triggered"]
            risk_categories = safe_batch["risk_category"]
        else:
            s, m, a, r, s2, m2, d = map(
                np.array,
                zip(*batch),
            )
            c = None
            proposed_actions = None
            action_modified_flags = None
            fallback_flags = None
            risk_categories = None

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        m = torch.tensor(m, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        c_tensor = (
            torch.tensor(c, dtype=torch.float32, device=self.device)
            if c is not None
            else None
        )
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        m2 = torch.tensor(m2, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        q_all = self.online(s)
        q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)

        reward_target, safety_target, _next_actions = (
            self._compute_double_dqn_targets(
                s2,
                m2,
                d,
                r,
                c_tensor,
            )
        )

        reward_td_errors = reward_target - q_sa
        reward_loss_each = F.smooth_l1_loss(
            q_sa,
            reward_target,
            reduction="none",
        )
        reward_loss = (reward_loss_each * weights).mean()

        self.optim.zero_grad()
        reward_loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()

        safety_loss = None
        weighted_safety_loss = None
        safety_td_errors = None
        if self.safe_rl_enabled:
            q_c_all = self.q_c_online(s)
            q_c_sa = q_c_all.gather(
                1,
                a.view(-1, 1),
            ).squeeze(1)
            safety_td_errors = safety_target - q_c_sa
            safety_loss_each = F.smooth_l1_loss(
                q_c_sa,
                safety_target,
                reduction="none",
            )
            safety_loss = (safety_loss_each * weights).mean()
            weighted_safety_loss = (
                self.safety_loss_weight * safety_loss
            )
            self.q_c_optim.zero_grad()
            weighted_safety_loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.q_c_online.parameters(),
                    self.grad_clip,
                )
            self.q_c_optim.step()

        self._updates += 1
        self._eps_steps += 1

        if self.use_per:
            reward_abs = (
                reward_td_errors.detach().abs().cpu().numpy()
            )
            if (
                self.safe_rl_enabled
                and self.safe_per_combined_priority
            ):
                safety_abs = (
                    safety_td_errors.detach()
                    .abs()
                    .cpu()
                    .numpy()
                )
                weight_sum = (
                    self.safe_per_performance_td_weight
                    + self.safe_per_safety_td_weight
                )
                # 两类 TD error 显式加权并除以权重和，避免仅因权重
                # 总量改变 priority 绝对尺度。默认开关关闭，保留旧 Q_r PER。
                new_prios = (
                    self.safe_per_performance_td_weight
                    * reward_abs
                    + self.safe_per_safety_td_weight
                    * safety_abs
                ) / weight_sum
            else:
                new_prios = reward_abs
            new_prios = new_prios + 1e-6
            for idx, prio in zip(indices, new_prios):
                self.priorities[int(idx)] = float(prio)

        if self.target_update_tau > 0.0:
            _polyak_update(self.target, self.online, self.target_update_tau)
            if self.safe_rl_enabled:
                _polyak_update(
                    self.q_c_target,
                    self.q_c_online,
                    self.target_update_tau,
                )
        elif (self._updates % self.target_update_freq) == 0:
            self.target.load_state_dict(self.online.state_dict())
            if self.safe_rl_enabled:
                self.q_c_target.load_state_dict(
                    self.q_c_online.state_dict()
                )

        self.last_update_info = {
            "performance_loss": float(reward_loss.item()),
            "safety_loss": (
                float(safety_loss.item())
                if safety_loss is not None
                else None
            ),
            "weighted_safety_loss": (
                float(weighted_safety_loss.item())
                if weighted_safety_loss is not None
                else None
            ),
            "performance_target_mean": float(
                reward_target.mean().item()
            ),
            "safety_target_mean": (
                float(safety_target.mean().item())
                if safety_target is not None
                else None
            ),
            "performance_td_error_mean": float(
                reward_td_errors.mean().item()
            ),
            "safety_td_error_mean": (
                float(safety_td_errors.mean().item())
                if safety_td_errors is not None
                else None
            ),
            "executed_action_mean": float(
                a.detach().float().mean().item()
            ),
            "proposed_action_mean": (
                float(np.mean(proposed_actions))
                if proposed_actions is not None
                else None
            ),
            "action_modified_rate": (
                float(np.mean(action_modified_flags))
                if action_modified_flags is not None
                else None
            ),
            "fallback_rate": (
                float(np.mean(fallback_flags))
                if fallback_flags is not None
                else None
            ),
            "risk_category_counts": (
                {
                    str(category): int(
                        np.sum(risk_categories == category)
                    )
                    for category in np.unique(risk_categories)
                }
                if risk_categories is not None
                else None
            ),
            "per_priority_mode": (
                "combined_performance_safety_td"
                if (
                    self.use_per
                    and self.safe_rl_enabled
                    and self.safe_per_combined_priority
                )
                else (
                    "performance_td_only"
                    if self.use_per
                    else "disabled"
                )
            ),
        }
        # 保持旧 update() 返回性能 loss 浮点数。
        return float(reward_loss.item())

    def replay_state_dict(self) -> dict:
        """导出带版本号的 replay；不混入网络 checkpoint。"""
        if not self.safe_rl_enabled:
            raise RuntimeError(
                "versioned replay export is currently defined only "
                "for safe_rl agents"
            )
        transitions = []
        for transition in self.buffer:
            if not isinstance(
                transition,
                SafeReplayTransition,
            ):
                raise ValueError(
                    "legacy or unversioned safe replay transition "
                    "cannot be exported without explicit migration"
                )
            transitions.append(transition.to_dict())
        return {
            "replay_buffer_schema_version": (
                SAFE_REPLAY_BUFFER_SCHEMA_VERSION
            ),
            "transition_schema_version": (
                SAFE_REPLAY_TRANSITION_SCHEMA_VERSION
            ),
            "safe_rl_enabled": True,
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "use_per": bool(self.use_per),
            "safe_replay_near_boundary_margin": float(
                self.safe_replay_near_boundary_margin
            ),
            "safe_per_combined_priority": bool(
                self.safe_per_combined_priority
            ),
            "safe_per_performance_td_weight": float(
                self.safe_per_performance_td_weight
            ),
            "safe_per_safety_td_weight": float(
                self.safe_per_safety_td_weight
            ),
            "transitions": transitions,
            "priorities": (
                [float(value) for value in self.priorities]
                if self.use_per
                else []
            ),
        }

    def load_replay_state_dict(self, payload) -> None:
        """严格恢复当前 replay schema，拒绝旧 tuple 或维度错配。"""
        if not self.safe_rl_enabled:
            raise RuntimeError(
                "safe replay cannot be loaded into a legacy agent"
            )
        if not isinstance(payload, dict):
            raise ValueError(
                "safe replay checkpoint must be a versioned mapping"
            )
        version = payload.get("replay_buffer_schema_version")
        if version != SAFE_REPLAY_BUFFER_SCHEMA_VERSION:
            raise ValueError(
                "safe replay buffer schema mismatch: "
                f"checkpoint={version!r}, current="
                f"{SAFE_REPLAY_BUFFER_SCHEMA_VERSION}"
            )
        transition_version = payload.get(
            "transition_schema_version"
        )
        if (
            transition_version
            != SAFE_REPLAY_TRANSITION_SCHEMA_VERSION
        ):
            raise ValueError(
                "safe replay transition schema mismatch: "
                f"checkpoint={transition_version!r}, current="
                f"{SAFE_REPLAY_TRANSITION_SCHEMA_VERSION}"
            )
        if not bool(payload.get("safe_rl_enabled", False)):
            raise ValueError(
                "performance-only replay cannot initialize safe replay"
            )
        if int(payload.get("input_dim", -1)) != self.input_dim:
            raise ValueError(
                "safe replay observation dimension mismatch"
            )
        if int(payload.get("output_dim", -1)) != self.output_dim:
            raise ValueError(
                "safe replay action dimension mismatch"
            )
        if bool(payload.get("use_per", False)) != self.use_per:
            raise ValueError(
                "safe replay PER mode mismatch"
            )
        replay_config = (
            (
                "safe_replay_near_boundary_margin",
                self.safe_replay_near_boundary_margin,
            ),
            (
                "safe_per_performance_td_weight",
                self.safe_per_performance_td_weight,
            ),
            (
                "safe_per_safety_td_weight",
                self.safe_per_safety_td_weight,
            ),
        )
        for key, current_value in replay_config:
            checkpoint_value = payload.get(key)
            if checkpoint_value is None or not np.isclose(
                float(checkpoint_value),
                float(current_value),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"safe replay config mismatch for {key}: "
                    f"checkpoint={checkpoint_value!r}, "
                    f"current={current_value!r}"
                )
        checkpoint_combined_per = payload.get(
            "safe_per_combined_priority"
        )
        if (
            checkpoint_combined_per is None
            or bool(checkpoint_combined_per)
            != self.safe_per_combined_priority
        ):
            raise ValueError(
                "safe replay config mismatch for "
                "safe_per_combined_priority"
            )
        serialized = payload.get("transitions")
        if not isinstance(serialized, list):
            raise ValueError(
                "safe replay transitions must be a list"
            )
        if len(serialized) > self.buffer_size:
            raise ValueError(
                "safe replay checkpoint exceeds buffer_size"
            )
        transitions = [
            SafeReplayTransition.from_dict(item)
            for item in serialized
        ]
        # 这里同时验证 state/action mask 维度和旧 tuple 拒绝。
        if transitions:
            stack_safe_replay_transitions(
                transitions,
                input_dim=self.input_dim,
                action_dim=self.output_dim,
            )

        priorities = payload.get("priorities", [])
        if self.use_per:
            if (
                not isinstance(priorities, list)
                or len(priorities) != len(transitions)
            ):
                raise ValueError(
                    "PER replay priorities length mismatch"
                )
            priority_values = np.asarray(
                priorities,
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(priority_values))
                or np.any(priority_values <= 0.0)
            ):
                raise ValueError(
                    "PER replay priorities must be finite and positive"
                )
        elif priorities:
            raise ValueError(
                "non-PER replay checkpoint must not contain priorities"
            )

        if self.use_per:
            self.buffer = list(transitions)
            self.priorities = [
                float(value) for value in priorities
            ]
        else:
            self.buffer = deque(
                transitions,
                maxlen=self.buffer_size,
            )
            self.priorities = []

    def save(
        self,
        path: str,
        *,
        lagrange_controller_state=None,
    ) -> None:
        controller_state = (
            lagrange_controller_state
            if lagrange_controller_state is not None
            else self.lagrange_controller_state
        )
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optim": self.optim.state_dict(),
                "checkpoint_schema_version": 5,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "safe_rl_enabled": bool(self.safe_rl_enabled),
                "observation_schema_version": (
                    self.observation_schema_version
                ),
                "q_c_online": (
                    self.q_c_online.state_dict()
                    if self.safe_rl_enabled
                    else None
                ),
                "q_c_target": (
                    self.q_c_target.state_dict()
                    if self.safe_rl_enabled
                    else None
                ),
                "q_c_optim": (
                    self.q_c_optim.state_dict()
                    if self.safe_rl_enabled
                    else None
                ),
                "safety_discount": self.safety_discount,
                "safety_learning_rate": self.safety_learning_rate,
                "safety_loss_weight": self.safety_loss_weight,
                "lagrange_multiplier": self.lagrange_multiplier,
                "lagrange_controller_state": controller_state,
                "updates": self._updates,
                "eps_steps": self._eps_steps,
                "eps_start": self._eps_start,
                "eps_end": self._eps_end,
                "eps_decay_steps": self._eps_decay_steps,
                "tau": self.target_update_tau,
                "hard_freq": self.target_update_freq,
                "hidden_dims": self.hidden_dims,
                "head_hidden_dims": self.head_hidden_dims,
                "use_per": self.use_per,
                "per_alpha": self.per_alpha,
                "per_beta_start": self.per_beta_start,
                "per_beta_end": self.per_beta_end,
                "per_beta_steps": self.per_beta_steps,
                "per_beta_count": self.per_beta_count,
                "safe_replay_transition_schema_version": (
                    SAFE_REPLAY_TRANSITION_SCHEMA_VERSION
                    if self.safe_rl_enabled
                    else None
                ),
                "safe_per_combined_priority": (
                    self.safe_per_combined_priority
                ),
                "safe_per_performance_td_weight": (
                    self.safe_per_performance_td_weight
                ),
                "safe_per_safety_td_weight": (
                    self.safe_per_safety_td_weight
                ),
                "safe_replay_near_boundary_margin": (
                    self.safe_replay_near_boundary_margin
                ),
            },
            path,
        )

    @staticmethod
    def _checkpoint_network_dimensions(ckpt) -> tuple[int, int]:
        """读取新 checkpoint 元数据，或从旧参数形状显式推断维度。"""
        input_dim = ckpt.get("input_dim")
        output_dim = ckpt.get("output_dim")
        state_dict = ckpt.get("online")
        if not isinstance(state_dict, dict):
            raise ValueError(
                "checkpoint does not contain a valid online network"
            )

        if input_dim is None:
            first_weight = state_dict.get("feature.0.weight")
            if first_weight is None or first_weight.ndim != 2:
                raise ValueError(
                    "legacy checkpoint has no input_dim metadata and "
                    "its observation dimension cannot be inferred"
                )
            input_dim = int(first_weight.shape[1])

        if output_dim is None:
            advantage_weights = []
            for key, value in state_dict.items():
                if (
                    key.startswith("adv.")
                    and key.endswith(".weight")
                    and getattr(value, "ndim", 0) == 2
                ):
                    try:
                        layer_index = int(key.split(".")[1])
                    except (IndexError, ValueError):
                        continue
                    advantage_weights.append(
                        (layer_index, int(value.shape[0]))
                    )
            if not advantage_weights:
                raise ValueError(
                    "legacy checkpoint has no output_dim metadata and "
                    "its action dimension cannot be inferred"
                )
            output_dim = max(advantage_weights)[1]
        return int(input_dim), int(output_dim)

    def load(self, path: str, strict: bool = True) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)

        checkpoint_observation_schema = ckpt.get(
            "observation_schema_version"
        )
        if checkpoint_observation_schema is None:
            if self.observation_schema_version != "legacy_observation":
                raise ValueError(
                    "legacy checkpoint without observation schema "
                    "metadata can only be loaded in legacy_observation "
                    "mode"
                )
        elif (
            str(checkpoint_observation_schema)
            != self.observation_schema_version
        ):
            raise ValueError(
                "checkpoint observation schema mismatch: "
                f"checkpoint={checkpoint_observation_schema!r}, "
                f"current={self.observation_schema_version!r}"
            )

        checkpoint_safe_rl_enabled = bool(
            ckpt.get("safe_rl_enabled", False)
        )
        if checkpoint_safe_rl_enabled != self.safe_rl_enabled:
            if self.safe_rl_enabled:
                raise ValueError(
                    "legacy/performance-only checkpoint has no Q_c "
                    "state and cannot initialize safe value learning"
                )
            raise ValueError(
                "safe dual-value checkpoint cannot be loaded into a "
                "performance-only agent"
            )

        checkpoint_input_dim, checkpoint_output_dim = (
            self._checkpoint_network_dimensions(ckpt)
        )
        if checkpoint_input_dim != self.input_dim:
            raise ValueError(
                "checkpoint observation dimension mismatch: "
                f"checkpoint input_dim={checkpoint_input_dim}, "
                f"current input_dim={self.input_dim}. Load legacy "
                "checkpoints only with the legacy observation mode, or "
                "train a new checkpoint for the enabled safety-state "
                "schema."
            )
        if checkpoint_output_dim != self.output_dim:
            raise ValueError(
                "checkpoint action dimension mismatch: "
                f"checkpoint output_dim={checkpoint_output_dim}, "
                f"current output_dim={self.output_dim}."
            )
        self.online.load_state_dict(ckpt["online"], strict=strict)
        self.target.load_state_dict(ckpt["target"], strict=strict)
        self.optim.load_state_dict(ckpt["optim"])
        if self.safe_rl_enabled:
            for key in ("q_c_online", "q_c_target", "q_c_optim"):
                if ckpt.get(key) is None:
                    raise ValueError(
                        f"safe checkpoint is missing {key}"
                    )
            self.q_c_online.load_state_dict(
                ckpt["q_c_online"],
                strict=strict,
            )
            self.q_c_target.load_state_dict(
                ckpt["q_c_target"],
                strict=strict,
            )
            self.q_c_optim.load_state_dict(ckpt["q_c_optim"])
            self.safety_discount = float(
                ckpt.get(
                    "safety_discount",
                    self.safety_discount,
                )
            )
            self.safety_learning_rate = float(
                ckpt.get(
                    "safety_learning_rate",
                    self.safety_learning_rate,
                )
            )
            self.safety_loss_weight = float(
                ckpt.get(
                    "safety_loss_weight",
                    self.safety_loss_weight,
                )
            )
            self.lagrange_multiplier = float(
                ckpt.get(
                    "lagrange_multiplier",
                    self.lagrange_multiplier,
                )
            )
            if (
                not np.isfinite(self.lagrange_multiplier)
                or self.lagrange_multiplier < 0.0
            ):
                raise ValueError(
                    "checkpoint lagrange_multiplier must be finite "
                    "and non-negative"
                )
            self.lagrange_controller_state = ckpt.get(
                "lagrange_controller_state"
            )
        self._updates = ckpt.get("updates", 0)
        self._eps_steps = ckpt.get("eps_steps", 0)
        self._eps_start = ckpt.get("eps_start", self._eps_start)
        self._eps_end = ckpt.get("eps_end", self._eps_end)
        self._eps_decay_steps = ckpt.get("eps_decay_steps", self._eps_decay_steps)
        self.target_update_tau = ckpt.get("tau", self.target_update_tau)
        self.target_update_freq = ckpt.get("hard_freq", self.target_update_freq)
        self.per_alpha = ckpt.get("per_alpha", self.per_alpha)
        self.per_beta_start = ckpt.get("per_beta_start", self.per_beta_start)
        self.per_beta_end = ckpt.get("per_beta_end", self.per_beta_end)
        self.per_beta_steps = ckpt.get("per_beta_steps", self.per_beta_steps)
        self.per_beta_count = ckpt.get("per_beta_count", self.per_beta_count)


__all__ = ["D3QNAgent", "DuelingQNet"]
