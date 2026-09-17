"""Read-only reward estimates aligned with current fuzzy outcomes."""

from __future__ import annotations

import numpy as np

from .config import RewardConfig
from .metrics import comparison_key


def routing_reward(
    *,
    expected_risk_finish: float,
    selected_risk_finish: float,
    task_deadline: float,
    incremental_fuzzy_energy: float,
    risk_slack_improvement: float,
    config: RewardConfig,
) -> tuple[float, dict]:
    completion = float(expected_risk_finish - selected_risk_finish)
    deadline = float(
        min(
            float(config.alpha_deadline)
            * (float(task_deadline) - float(selected_risk_finish)),
            0.0,
        )
    )
    energy = float(
        -float(config.beta_energy)
        * max(0.0, float(incremental_fuzzy_energy))
        / max(float(config.energy_normalizer), 1e-12)
    )
    slack = float(
        float(config.slack_weight)
        * float(risk_slack_improvement)
        / max(float(config.time_normalizer), 1e-12)
    )
    if config.mode == "completion_only":
        total = completion
    elif config.mode == "deadline_energy":
        total = completion + deadline + energy
    elif config.mode == "deadline_energy_slack":
        total = completion + deadline + energy + slack
    else:
        raise ValueError(f"unsupported reward mode: {config.mode}")
    fields = {
        "completion_component": completion,
        "deadline_component": deadline,
        "energy_component": energy,
        "slack_component": slack,
        "routing_reward": float(total),
        "reward_mode": config.mode,
    }
    if not all(np.isfinite(value) for value in fields.values() if isinstance(value, float)):
        raise ValueError("routing reward contains NaN or Inf")
    return float(total), fields


def sequencing_reward(
    before_risk_slack: float,
    after_risk_slack: float,
    config: RewardConfig,
) -> tuple[float, dict]:
    normalized = (
        float(after_risk_slack) - float(before_risk_slack)
    ) / max(float(config.time_normalizer), 1e-12)
    reward = float(np.clip(normalized, -3.0, 3.0))
    return reward, {
        "risk_slack_before": float(before_risk_slack),
        "risk_slack_after": float(after_risk_slack),
        "normalized_risk_slack_improvement": float(normalized),
        "sequencing_reward": reward,
    }


def select_reward_mode(validation_metrics_by_mode: dict[str, dict]) -> str:
    """Select a formal reward configuration from validation metrics only."""

    if not validation_metrics_by_mode:
        raise ValueError("reward selection needs validation metrics")
    supported = {
        "completion_only",
        "deadline_energy",
        "deadline_energy_slack",
    }
    unknown = set(validation_metrics_by_mode) - supported
    if unknown:
        raise ValueError(f"unknown reward mode(s): {sorted(unknown)}")
    return min(
        validation_metrics_by_mode,
        key=lambda mode: comparison_key(
            validation_metrics_by_mode[mode]
        ),
    )
