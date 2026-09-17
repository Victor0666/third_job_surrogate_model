"""Trainable IRWS task-ranking and VM-assignment policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from algorithms.comparisons.fuzzy_common.agents import (
    MaskedPPOAgent,
    PPOConfig,
    StateNoveltyBonus,
    TrainableTaskRanker,
)
from algorithms.comparisons.fuzzy_common.environment import (
    FuzzyAssignmentResult,
    FuzzyBaselineEnv,
)


class IRWSPolicy:
    """IRWS with learned ready-task ranking and a masked PPO VM policy."""

    method_id = "fuzzy_irws"

    def __init__(
        self,
        env: FuzzyBaselineEnv,
        *,
        device: str = "cpu",
        intrinsic_reward_weight: float = 0.05,
        task_hidden_dim: int = 128,
        task_learning_rate: float = 3e-4,
        vm_learning_rate: float = 3e-4,
    ) -> None:
        self.task_ranker = TrainableTaskRanker(
            feature_dim=8,
            hidden_dim=int(task_hidden_dim),
            learning_rate=float(task_learning_rate),
            device=device,
        )
        self.vm_agent = MaskedPPOAgent(
            env.global_vm_obs_dim,
            env.num_vms,
            PPOConfig(
                learning_rate=float(vm_learning_rate),
                device=device,
            ),
        )
        self.intrinsic = StateNoveltyBonus(precision=2)
        self.intrinsic_reward_weight = float(intrinsic_reward_weight)
        if self.intrinsic_reward_weight < 0.0:
            raise ValueError("intrinsic_reward_weight must be non-negative")
        self._last_decision: dict[str, Any] | None = None

    def configuration(self) -> dict[str, Any]:
        return {
            "task_feature_dim": 8,
            "task_hidden_dim": int(
                self.task_ranker.network.network[0].out_features
            ),
            "task_learning_rate": float(
                self.task_ranker.optimizer.param_groups[0]["lr"]
            ),
            "vm_learning_rate": float(
                self.vm_agent.optimizer.param_groups[0]["lr"]
            ),
            "vm_policy": "masked_ppo",
            "intrinsic_reward": "deterministic_count_novelty",
            "intrinsic_reward_weight": float(self.intrinsic_reward_weight),
        }

    def _order_tasks(
        self,
        task_ids: Sequence[int],
        features: np.ndarray,
        training: bool,
    ) -> Sequence[int]:
        return self.task_ranker.order(
            task_ids,
            features,
            training=training,
        )

    def begin_episode(self, env: FuzzyBaselineEnv, *, training: bool) -> None:
        self.task_ranker.begin_episode()
        self.vm_agent.clear()
        self._last_decision = None
        env.set_task_orderer(self._order_tasks, training=training)

    def assign(
        self,
        env: FuzzyBaselineEnv,
        host_state: dict,
        *,
        training: bool,
    ) -> FuzzyAssignmentResult:
        del host_state
        observation, mask = env.global_vm_state()
        action, log_probability, value = self.vm_agent.select_action(
            observation,
            mask,
            deterministic=not training,
        )
        novelty = self.intrinsic.value(observation, update=training)
        self._last_decision = {
            "observation": observation,
            "mask": mask,
            "action": action,
            "log_probability": log_probability,
            "value": value,
            "novelty": novelty,
        }
        return env.assign_global_vm(action)

    def observe_assignment(
        self,
        env: FuzzyBaselineEnv,
        result: FuzzyAssignmentResult,
        *,
        training: bool,
    ) -> None:
        del env
        if not training:
            return
        if self._last_decision is None:
            raise RuntimeError("missing IRWS decision record")
        fused_reward = float(
            result.reward
            + self.intrinsic_reward_weight
            * float(self._last_decision["novelty"])
        )
        self.task_ranker.record_reward(result.task_id, fused_reward)
        self.vm_agent.remember(
            self._last_decision["observation"],
            self._last_decision["mask"],
            self._last_decision["action"],
            self._last_decision["log_probability"],
            self._last_decision["value"],
            fused_reward,
            0.0,
        )

    def end_phase(
        self,
        env: FuzzyBaselineEnv,
        reward: float,
        info: dict,
        *,
        training: bool,
    ) -> None:
        del env, reward, info, training

    def end_episode(self, env: FuzzyBaselineEnv, *, training: bool) -> None:
        del env
        if training:
            self.task_ranker.update()
            self.vm_agent.update()
        else:
            self.task_ranker.begin_episode()
            self.vm_agent.clear()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_version": 1,
                "method_id": self.method_id,
                "task_ranker": self.task_ranker.network.state_dict(),
                "task_optimizer": self.task_ranker.optimizer.state_dict(),
                "vm_agent": self.vm_agent.network.state_dict(),
                "vm_optimizer": self.vm_agent.optimizer.state_dict(),
                "intrinsic_counts": self.intrinsic.counts,
                "intrinsic_reward_weight": self.intrinsic_reward_weight,
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            payload = torch.load(path, map_location=self.vm_agent.device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=self.vm_agent.device)
        if payload.get("method_id") != self.method_id:
            raise ValueError("IRWS checkpoint method mismatch")
        self.task_ranker.network.load_state_dict(payload["task_ranker"])
        self.task_ranker.optimizer.load_state_dict(payload["task_optimizer"])
        self.vm_agent.network.load_state_dict(payload["vm_agent"])
        self.vm_agent.optimizer.load_state_dict(payload["vm_optimizer"])
        self.intrinsic.counts = dict(payload.get("intrinsic_counts", {}))


__all__ = ["IRWSPolicy"]
