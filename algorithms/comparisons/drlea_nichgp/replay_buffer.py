"""Replay memory with current and next legal-action masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_mask import validate_mask


@dataclass(frozen=True)
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    masks: np.ndarray
    next_masks: np.ndarray


class MaskedReplayBuffer:
    schema_version = 1

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        seed: int,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self.position = 0
        self.size = 0
        self.states = np.zeros(
            (self.capacity, self.state_dim), dtype=np.float32
        )
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.masks = np.zeros(
            (self.capacity, self.action_dim), dtype=np.float32
        )
        self.next_masks = np.zeros_like(self.masks)

    def add(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        mask,
        next_mask,
    ) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,) or next_state.shape != (
            self.state_dim,
        ):
            raise ValueError("replay state dimension mismatch")
        if not np.all(np.isfinite(state)) or not np.all(
            np.isfinite(next_state)
        ):
            raise ValueError("replay state contains NaN or Inf")
        mask = validate_mask(mask, self.action_dim)
        next_mask = validate_mask(
            next_mask,
            self.action_dim,
            allow_empty=bool(done),
        )
        action = int(action)
        if action < 0 or action >= self.action_dim or mask[action] <= 0.5:
            raise ValueError("stored action is not legal under current mask")
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = float(reward)
        self.next_states[index] = next_state
        self.dones[index] = bool(done)
        self.masks[index] = mask
        self.next_masks[index] = next_mask
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        batch_size = int(batch_size)
        if self.size < batch_size:
            raise ValueError("not enough replay samples")
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return ReplayBatch(
            states=self.states[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=self.next_states[indices].copy(),
            dones=self.dones[indices].copy(),
            masks=self.masks[indices].copy(),
            next_masks=self.next_masks[indices].copy(),
        )

    def metadata(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
        }
