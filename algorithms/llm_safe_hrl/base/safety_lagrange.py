# -*- coding: utf-8 -*-
"""Episode 级共享拉格朗日安全控制器。

Q_c 继续学习 dense ``safety_cost``；本控制器只消费 episode 实际工作流 DDL
违反率。Manager、Host、VM 的动作评分共享同一个 ``lambda_DDL``。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


class LagrangeSafetyController:
    """在 episode 边界用实际工作流 DDL 违反率更新共享 ``lambda_DDL``。

    ``cost_ema_factor`` 与旧状态字段仅为 checkpoint 兼容保留，不再参与更新。
    """

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        enabled: bool = False,
        lambda_init: float = 1.0,
        lambda_lr: float = 0.05,
        lambda_min: float = 0.0,
        lambda_max: float = 100.0,
        cost_budget: float = 0.01,
        update_interval: int = 1,
        cost_ema_factor: float = 0.9,
        warmup_steps: int = 0,
    ):
        self.enabled = bool(enabled)
        self.lambda_init = float(lambda_init)
        self.lambda_lr = float(lambda_lr)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.cost_budget = float(cost_budget)
        self.update_interval = int(update_interval)
        self.cost_ema_factor = float(cost_ema_factor)
        self.warmup_steps = int(warmup_steps)
        self._validate_config()

        self.current_lambda = float(
            min(
                max(self.lambda_init, self.lambda_min),
                self.lambda_max,
            )
        )
        self.mean_safety_cost = 0.0
        self.last_episode_safety_cost = 0.0
        self.constraint_gap = -self.cost_budget
        self.episode_violation_rate = 0.0
        self.lambda_before = float(self.current_lambda)
        self.lambda_after = float(self.current_lambda)
        self.lagrange_gap = -self.cost_budget
        self.lambda_update_count = 0
        self.observed_episode_count = 0
        self.invalid_sample_count = 0
        self._ema_initialized = False
        self.last_update_applied = False
        self.last_update_reason = "not_observed"

    def _validate_config(self) -> None:
        finite_fields = {
            "lambda_init": self.lambda_init,
            "lambda_lr": self.lambda_lr,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "cost_budget": self.cost_budget,
            "cost_ema_factor": self.cost_ema_factor,
        }
        for name, value in finite_fields.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.lambda_lr < 0.0:
            raise ValueError("lambda_lr must be non-negative")
        if self.lambda_min < 0.0:
            raise ValueError("lambda_min must be non-negative")
        if self.lambda_max < self.lambda_min:
            raise ValueError(
                "lambda_max must be greater than or equal to lambda_min"
            )
        if not self.lambda_min <= self.lambda_init <= self.lambda_max:
            raise ValueError(
                "lambda_init must be within [lambda_min, lambda_max]"
            )
        if self.cost_budget < 0.0:
            raise ValueError("cost_budget must be non-negative")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if not 0.0 <= self.cost_ema_factor < 1.0:
            raise ValueError("cost_ema_factor must be in [0, 1)")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

    @property
    def lagrange_multiplier(self) -> float:
        """兼容 Agent 命名的只读别名。"""
        return float(self.current_lambda)

    def observe_episode(
        self,
        *,
        episode_violation_rate: float,
    ) -> dict[str, Any]:
        """记录一个 episode，并按违反率相对预算的 gap 更新共享 lambda。"""
        self.last_update_applied = False
        self.lambda_before = float(self.current_lambda)
        self.lambda_after = float(self.current_lambda)
        if not self.enabled:
            self.last_update_reason = "disabled"
            return self.diagnostics()

        violation_rate = float(episode_violation_rate)
        if not math.isfinite(violation_rate) or violation_rate < 0.0:
            self.invalid_sample_count += 1
            self.last_update_reason = "invalid_violation_rate"
            return self.diagnostics()

        self.observed_episode_count += 1
        self.episode_violation_rate = violation_rate
        # 旧字段作为违反率别名保留，避免旧日志/checkpoint 读取器失效。
        self.last_episode_safety_cost = violation_rate
        self.mean_safety_cost = violation_rate
        self._ema_initialized = True
        self.lagrange_gap = float(violation_rate - self.cost_budget)
        self.constraint_gap = self.lagrange_gap

        if self.observed_episode_count <= self.warmup_steps:
            self.last_update_reason = "warmup"
            return self.diagnostics()

        eligible_index = (
            self.observed_episode_count - self.warmup_steps
        )
        if eligible_index % self.update_interval != 0:
            self.last_update_reason = "update_interval"
            return self.diagnostics()

        delta = self.lambda_lr * self.lagrange_gap
        candidate = self.current_lambda + delta
        if not math.isfinite(candidate):
            candidate = (
                self.lambda_max
                if self.lagrange_gap > 0.0
                else self.lambda_min
            )
        self.current_lambda = float(
            min(
                max(candidate, self.lambda_min),
                self.lambda_max,
            )
        )
        if not math.isfinite(self.current_lambda):
            # 配置已校验为有限值；此分支是最后一道防护。
            self.current_lambda = float(self.lambda_max)
        self.lambda_after = float(self.current_lambda)

        self.lambda_update_count += 1
        self.last_update_applied = True
        self.last_update_reason = "updated"
        return self.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        """返回日志需要的稳定字段。"""
        return {
            "current_lambda": float(self.current_lambda),
            "mean_safety_cost": float(self.mean_safety_cost),
            "episode_safety_cost": float(
                self.last_episode_safety_cost
            ),
            "cost_budget": float(self.cost_budget),
            "constraint_gap": float(self.constraint_gap),
            "episode_violation_rate": float(
                self.episode_violation_rate
            ),
            "lambda_before": float(self.lambda_before),
            "lambda_after": float(self.lambda_after),
            "lagrange_gap": float(self.lagrange_gap),
            "lambda_update_count": int(self.lambda_update_count),
            "observed_episode_count": int(
                self.observed_episode_count
            ),
            "invalid_sample_count": int(self.invalid_sample_count),
            "lambda_update_applied": bool(
                self.last_update_applied
            ),
            "lambda_update_reason": self.last_update_reason,
            "lagrange_enabled": bool(self.enabled),
            "lagrange_update_period": "episode_violation_rate",
        }

    def state_dict(self) -> dict[str, Any]:
        """返回仅含安全标量/整数的 checkpoint 状态。"""
        return {
            "state_version": self.STATE_VERSION,
            "constraint_signal": "episode_violation_rate",
            "enabled": bool(self.enabled),
            "lambda_init": float(self.lambda_init),
            "lambda_lr": float(self.lambda_lr),
            "lambda_min": float(self.lambda_min),
            "lambda_max": float(self.lambda_max),
            "cost_budget": float(self.cost_budget),
            "update_interval": int(self.update_interval),
            "cost_ema_factor": float(self.cost_ema_factor),
            "warmup_steps": int(self.warmup_steps),
            "current_lambda": float(self.current_lambda),
            "mean_safety_cost": float(self.mean_safety_cost),
            "last_episode_safety_cost": float(
                self.last_episode_safety_cost
            ),
            "constraint_gap": float(self.constraint_gap),
            "episode_violation_rate": float(
                self.episode_violation_rate
            ),
            "lambda_before": float(self.lambda_before),
            "lambda_after": float(self.lambda_after),
            "lagrange_gap": float(self.lagrange_gap),
            "lambda_update_count": int(self.lambda_update_count),
            "observed_episode_count": int(
                self.observed_episode_count
            ),
            "invalid_sample_count": int(self.invalid_sample_count),
            "ema_initialized": bool(self._ema_initialized),
            "last_update_applied": bool(
                self.last_update_applied
            ),
            "last_update_reason": str(self.last_update_reason),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        """恢复控制器状态；strict 模式拒绝训练超参数不一致。"""
        if not isinstance(state, Mapping):
            raise ValueError(
                "lagrange controller checkpoint must be a mapping"
            )
        if int(state.get("state_version", -1)) != self.STATE_VERSION:
            raise ValueError(
                "unsupported lagrange controller state version"
            )

        if strict:
            expected = {
                "enabled": self.enabled,
                "lambda_init": self.lambda_init,
                "lambda_lr": self.lambda_lr,
                "lambda_min": self.lambda_min,
                "lambda_max": self.lambda_max,
                "cost_budget": self.cost_budget,
                "update_interval": self.update_interval,
                "cost_ema_factor": self.cost_ema_factor,
                "warmup_steps": self.warmup_steps,
            }
            for key, expected_value in expected.items():
                if state.get(key) != expected_value:
                    raise ValueError(
                        "lagrange controller config mismatch for "
                        f"{key}: checkpoint={state.get(key)!r}, "
                        f"current={expected_value!r}"
                    )

        current_lambda = float(state["current_lambda"])
        mean_cost = float(state["mean_safety_cost"])
        last_cost = float(state["last_episode_safety_cost"])
        gap = float(state["constraint_gap"])
        if not all(
            math.isfinite(value)
            for value in (
                current_lambda,
                mean_cost,
                last_cost,
                gap,
            )
        ):
            raise ValueError(
                "lagrange controller checkpoint contains non-finite state"
            )
        if not self.lambda_min <= current_lambda <= self.lambda_max:
            raise ValueError(
                "checkpoint lambda is outside configured bounds"
            )

        self.current_lambda = current_lambda
        self.mean_safety_cost = mean_cost
        self.last_episode_safety_cost = last_cost
        has_violation_rate_state = (
            state.get("constraint_signal")
            == "episode_violation_rate"
            or "episode_violation_rate" in state
        )
        self.episode_violation_rate = float(
            state.get("episode_violation_rate", 0.0)
        )
        self.lambda_before = float(
            state.get("lambda_before", current_lambda)
        )
        self.lambda_after = float(
            state.get("lambda_after", current_lambda)
        )
        self.lagrange_gap = float(
            state.get(
                "lagrange_gap",
                (
                    gap
                    if has_violation_rate_state
                    else -self.cost_budget
                ),
            )
        )
        self.constraint_gap = self.lagrange_gap
        if not all(
            math.isfinite(value)
            for value in (
                self.episode_violation_rate,
                self.lambda_before,
                self.lambda_after,
                self.lagrange_gap,
            )
        ):
            raise ValueError(
                "lagrange controller checkpoint contains non-finite "
                "violation-rate state"
            )
        self.lambda_update_count = int(
            state["lambda_update_count"]
        )
        self.observed_episode_count = int(
            state["observed_episode_count"]
        )
        self.invalid_sample_count = int(
            state.get("invalid_sample_count", 0)
        )
        if min(
            self.lambda_update_count,
            self.observed_episode_count,
            self.invalid_sample_count,
        ) < 0:
            raise ValueError(
                "lagrange controller counters must be non-negative"
            )
        self._ema_initialized = bool(state["ema_initialized"])
        self.last_update_applied = bool(
            state["last_update_applied"]
        )
        self.last_update_reason = str(state["last_update_reason"])


def synchronize_lagrange_multiplier(
    controller: LagrangeSafetyController,
    agents: Iterable[Any],
) -> float:
    """把一个共享 lambda 同步到三层独立 Agent。"""
    value = float(controller.current_lambda)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "shared lagrange multiplier must be finite and non-negative"
        )
    for agent in agents:
        if not bool(getattr(agent, "safe_rl_enabled", False)):
            raise ValueError(
                "dynamic lagrange controller requires safe RL agents"
            )
        agent.lagrange_multiplier = value
    return value


__all__ = [
    "LagrangeSafetyController",
    "synchronize_lagrange_multiplier",
]
