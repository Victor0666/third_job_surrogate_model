"""FCFS task ordering with the project's existing PER Double-Dueling DQN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from algorithms.comparisons.fuzzy_common import LLM_SAFE_HRL_ROOT  # noqa: F401
from base.d3qn_agent import D3QNAgent
from algorithms.comparisons.fuzzy_common.environment import (
    FuzzyAssignmentResult,
    FuzzyBaselineEnv,
)


class PD3QNPolicy:
    """Performance-only PD3QN using a global VM action space and PER."""

    method_id = "fuzzy_pd3qn"

    def __init__(
        self,
        env: FuzzyBaselineEnv,
        *,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        replay_capacity: int = 100000,
    ) -> None:
        self.agent = D3QNAgent(
            input_dim=env.global_vm_obs_dim,
            output_dim=env.num_vms,
            lr=float(learning_rate),
            gamma=float(gamma),
            batch_size=int(batch_size),
            buffer_size=int(replay_capacity),
            hidden_dims=(512, 256),
            device=device,
            use_per=True,
            safe_rl_enabled=False,
            observation_schema_version="fuzzy_pd3qn_global_vm_v1",
        )
        self._pending: dict[str, Any] | None = None

    def configuration(self) -> dict[str, Any]:
        return {
            "learning_rate": float(self.agent.optim.param_groups[0]["lr"]),
            "gamma": float(self.agent.gamma),
            "batch_size": int(self.agent.batch_size),
            "replay_capacity": int(self.agent.buffer_size),
            "double_dqn": True,
            "dueling_network": True,
            "prioritized_replay": bool(self.agent.use_per),
            "task_order": "fcfs",
        }

    def begin_episode(self, env: FuzzyBaselineEnv, *, training: bool) -> None:
        self._pending = None
        env.set_task_orderer(None, training=training)

    def _finalize_pending(
        self,
        next_observation: np.ndarray,
        next_mask: np.ndarray,
        *,
        done: float,
    ) -> None:
        if self._pending is None:
            return
        self.agent.remember(
            self._pending["observation"],
            self._pending["mask"],
            self._pending["action"],
            self._pending["reward"],
            next_observation,
            next_mask,
            done,
        )
        self.agent.update()
        self._pending = None

    def assign(
        self,
        env: FuzzyBaselineEnv,
        host_state: dict,
        *,
        training: bool,
    ) -> FuzzyAssignmentResult:
        del host_state
        observation, mask = env.global_vm_state()
        if training:
            self._finalize_pending(observation, mask, done=0.0)
        action = self.agent.select_action(
            observation,
            mask,
            deterministic=not training,
            count_step=training,
        )
        result = env.assign_global_vm(action)
        if training:
            self._pending = {
                "observation": observation.copy(),
                "mask": mask.copy(),
                "action": int(action),
                "reward": float(result.reward),
            }
        return result

    def observe_assignment(
        self,
        env: FuzzyBaselineEnv,
        result: FuzzyAssignmentResult,
        *,
        training: bool,
    ) -> None:
        del env, result, training

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
        if training and self._pending is not None:
            self._finalize_pending(
                np.zeros(self.agent.input_dim, dtype=np.float32),
                np.zeros(self.agent.output_dim, dtype=np.float32),
                done=1.0,
            )
        self._pending = None

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path)

    def load(self, path: str) -> None:
        self.agent.load(path)


__all__ = ["PD3QNPolicy"]
