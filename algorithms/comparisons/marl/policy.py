"""Global Host A2C plus shared local VM actor with per-Host critics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from algorithms.comparisons.fuzzy_common.agents import (
    A2CConfig,
    LocalMultiCriticA2C,
    MaskedA2CAgent,
)
from algorithms.comparisons.fuzzy_common.environment import (
    FuzzyAssignmentResult,
    FuzzyBaselineEnv,
    _legal_mask,
)


class MARLPolicy:
    """FCFS task order with learned global Host and local VM policies."""

    method_id = "fuzzy_marl"

    def __init__(
        self,
        env: FuzzyBaselineEnv,
        *,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        gamma: float = 0.95,
    ) -> None:
        config = A2CConfig(
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            device=device,
        )
        self.host_agent = MaskedA2CAgent(
            env.host_obs_dim,
            env.host_act_dim,
            config,
        )
        self.vm_agent = LocalMultiCriticA2C(
            env.vm_obs_dim,
            env.vm_act_dim,
            env.num_hosts,
            config,
        )
        self._pending_host: dict[str, Any] | None = None
        self._pending_vm_by_host: dict[int, dict[str, Any]] = {}

    def configuration(self) -> dict[str, Any]:
        return {
            "learning_rate": float(
                self.host_agent.optimizer.param_groups[0]["lr"]
            ),
            "gamma": float(self.host_agent.config.gamma),
            "host_policy": "masked_a2c",
            "vm_policy": "shared_actor_host_specific_critic_a2c",
            "task_order": "fcfs",
        }

    def begin_episode(self, env: FuzzyBaselineEnv, *, training: bool) -> None:
        self._pending_host = None
        self._pending_vm_by_host = {}
        env.set_task_orderer(None, training=training)

    def _update_pending_host(
        self,
        next_observation: np.ndarray,
        *,
        done: float,
    ) -> None:
        if self._pending_host is None:
            return
        pending = self._pending_host
        self.host_agent.update_transition(
            pending["observation"],
            pending["mask"],
            pending["action"],
            pending["reward"],
            next_observation,
            done,
        )
        self._pending_host = None

    def _update_pending_vm(
        self,
        host_id: int,
        next_observation: np.ndarray,
        *,
        done: float,
    ) -> None:
        host_id = int(host_id)
        pending = self._pending_vm_by_host.pop(host_id, None)
        if pending is None:
            return
        self.vm_agent.update_transition(
            pending["observation"],
            pending["mask"],
            pending["action"],
            pending["reward"],
            next_observation,
            done,
            host_id,
        )

    def assign(
        self,
        env: FuzzyBaselineEnv,
        host_state: dict,
        *,
        training: bool,
    ) -> FuzzyAssignmentResult:
        host_observation = np.asarray(host_state["obs"], dtype=np.float32)
        host_mask = _legal_mask(host_state)
        if training:
            self._update_pending_host(host_observation, done=0.0)
        host_action = self.host_agent.select_action(
            host_observation,
            host_mask,
            deterministic=not training,
        )
        env.host_select(host_action)
        vm_state, available = env.get_vm_state_for_current_task()
        if not available:
            raise RuntimeError("MARL selected a Host without a VM decision")
        vm_observation = np.asarray(vm_state["obs"], dtype=np.float32)
        vm_mask = _legal_mask(vm_state)
        if training:
            # A Host-specific critic may only bootstrap from the same Host's
            # local VM state; other Hosts keep their transition pending.
            self._update_pending_vm(
                host_action,
                vm_observation,
                done=0.0,
            )
        vm_action = self.vm_agent.select_action(
            vm_observation,
            vm_mask,
            host_action,
            deterministic=not training,
        )
        result = env.assign_local_vm(vm_action)
        if training:
            self._pending_host = {
                "observation": host_observation.copy(),
                "mask": host_mask.copy(),
                "action": int(host_action),
                "reward": float(result.reward),
            }
            self._pending_vm_by_host[int(host_action)] = {
                "observation": vm_observation.copy(),
                "mask": vm_mask.copy(),
                "action": int(vm_action),
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
        if training:
            self._update_pending_host(
                np.zeros(self.host_agent.observation_dim, dtype=np.float32),
                done=1.0,
            )
            terminal_vm_observation = np.zeros(
                self.vm_agent.observation_dim,
                dtype=np.float32,
            )
            for host_id in sorted(tuple(self._pending_vm_by_host)):
                self._update_pending_vm(
                    host_id,
                    terminal_vm_observation,
                    done=1.0,
                )
        self._pending_host = None
        self._pending_vm_by_host = {}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_version": 1,
                "method_id": self.method_id,
                "host_network": self.host_agent.network.state_dict(),
                "host_optimizer": self.host_agent.optimizer.state_dict(),
                "vm_network": self.vm_agent.network.state_dict(),
                "vm_optimizer": self.vm_agent.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            payload = torch.load(
                path,
                map_location=self.host_agent.device,
                weights_only=True,
            )
        except TypeError:
            payload = torch.load(path, map_location=self.host_agent.device)
        if payload.get("method_id") != self.method_id:
            raise ValueError("MARL checkpoint method mismatch")
        self.host_agent.network.load_state_dict(payload["host_network"])
        self.host_agent.optimizer.load_state_dict(payload["host_optimizer"])
        self.vm_agent.network.load_state_dict(payload["vm_network"])
        self.vm_agent.optimizer.load_state_dict(payload["vm_optimizer"])


__all__ = ["MARLPolicy"]