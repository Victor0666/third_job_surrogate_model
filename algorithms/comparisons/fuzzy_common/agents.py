"""Small masked policy-gradient components used by comparison methods."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError("policy logits and action mask shapes differ")
    if torch.any(torch.sum(mask > 0.5, dim=-1) == 0):
        raise ValueError("policy received an empty action mask")
    return logits.masked_fill(mask <= 0.5, -1e9)


def _mlp(input_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
        previous = int(hidden)
    return nn.Sequential(*layers)


class ActorCriticNet(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.backbone = _mlp(observation_dim, hidden_dims)
        width = int(tuple(hidden_dims)[-1])
        self.policy = nn.Linear(width, int(action_dim))
        self.value = nn.Linear(width, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        return self.policy(features), self.value(features).squeeze(-1)


@dataclass(frozen=True)
class A2CConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.95
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gradient_clip: float = 10.0
    hidden_dims: tuple[int, ...] = (256, 128)
    device: str = "cpu"


class MaskedA2CAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: A2CConfig) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.device = torch.device(config.device)
        self.network = ActorCriticNet(
            self.observation_dim,
            self.action_dim,
            config.hidden_dims,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(config.learning_rate),
        )

    @torch.no_grad()
    def select_action(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        *,
        deterministic: bool,
    ) -> int:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.network(obs)
        logits = _mask_logits(logits, valid)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    def update_transition(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: float,
    ) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs = torch.as_tensor(
            next_observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        logits, value = self.network(obs)
        distribution = torch.distributions.Categorical(
            logits=_mask_logits(logits, valid)
        )
        action_tensor = torch.tensor([int(action)], device=self.device)
        reward_tensor = torch.tensor([float(reward)], device=self.device)
        done_tensor = torch.tensor([float(done)], device=self.device)
        with torch.no_grad():
            _, next_value = self.network(next_obs)
            target = reward_tensor + self.config.gamma * (1.0 - done_tensor) * next_value
        advantage = target - value
        loss = (
            -(distribution.log_prob(action_tensor) * advantage.detach()).mean()
            + self.config.value_coefficient * F.mse_loss(value, target)
            - self.config.entropy_coefficient * distribution.entropy().mean()
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        return float(loss.item())


class SharedActorMultiCriticNet(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        host_count: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.host_count = int(host_count)
        self.backbone = _mlp(observation_dim, hidden_dims)
        width = int(tuple(hidden_dims)[-1])
        self.policy = nn.Linear(width, int(action_dim))
        self.critics = nn.ModuleList(
            [nn.Linear(width, 1) for _ in range(self.host_count)]
        )

    def forward(
        self,
        observation: torch.Tensor,
        host_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        logits = self.policy(features)
        values = torch.cat(
            [
                self.critics[
                    max(0, min(self.host_count - 1, int(host_indices[index].item())))
                ](features[index : index + 1])
                for index in range(features.shape[0])
            ],
            dim=0,
        ).squeeze(-1)
        return logits, values


class LocalMultiCriticA2C:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        host_count: int,
        config: A2CConfig,
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.host_count = int(host_count)
        self.config = config
        self.device = torch.device(config.device)
        self.network = SharedActorMultiCriticNet(
            observation_dim,
            action_dim,
            host_count,
            config.hidden_dims,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(config.learning_rate),
        )

    @torch.no_grad()
    def select_action(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        host_index: int,
        *,
        deterministic: bool,
    ) -> int:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        hosts = torch.tensor([int(host_index)], device=self.device)
        logits, _ = self.network(obs, hosts)
        logits = _mask_logits(logits, valid)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    def update_transition(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: float,
        host_index: int,
    ) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        hosts = torch.tensor([int(host_index)], device=self.device)
        logits, value = self.network(obs, hosts)
        distribution = torch.distributions.Categorical(
            logits=_mask_logits(logits, valid)
        )
        action_tensor = torch.tensor([int(action)], device=self.device)
        reward_tensor = torch.tensor([float(reward)], device=self.device)
        done_tensor = torch.tensor([float(done)], device=self.device)
        with torch.no_grad():
            _, next_value = self.network(next_obs, hosts)
            target = reward_tensor + self.config.gamma * (1.0 - done_tensor) * next_value
        advantage = target - value
        loss = (
            -(distribution.log_prob(action_tensor) * advantage.detach()).mean()
            + self.config.value_coefficient * F.mse_loss(value, target)
            - self.config.entropy_coefficient * distribution.entropy().mean()
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        return float(loss.item())


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gradient_clip: float = 10.0
    update_epochs: int = 4
    minibatch_size: int = 128
    hidden_dims: tuple[int, ...] = (256, 256)
    device: str = "cpu"


class MaskedPPOAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: PPOConfig) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.device = torch.device(config.device)
        self.network = ActorCriticNet(
            observation_dim,
            action_dim,
            config.hidden_dims,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(config.learning_rate),
        )
        self.clear()

    def clear(self) -> None:
        self.buffer: list[dict[str, object]] = []

    @torch.no_grad()
    def select_action(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[int, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.network(obs)
        distribution = torch.distributions.Categorical(
            logits=_mask_logits(logits, valid)
        )
        action = (
            torch.argmax(logits.masked_fill(valid <= 0.5, -1e9), dim=-1)
            if deterministic
            else distribution.sample()
        )
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
        )

    def remember(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        action: int,
        log_probability: float,
        value: float,
        reward: float,
        done: float,
    ) -> None:
        self.buffer.append(
            {
                "observation": np.asarray(observation, dtype=np.float32).copy(),
                "mask": np.asarray(mask, dtype=np.float32).copy(),
                "action": int(action),
                "log_probability": float(log_probability),
                "value": float(value),
                "reward": float(reward),
                "done": float(done),
            }
        )

    def update(self) -> float | None:
        if len(self.buffer) < 2:
            self.clear()
            return None
        observations = torch.as_tensor(
            np.stack([item["observation"] for item in self.buffer]),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.as_tensor(
            np.stack([item["mask"] for item in self.buffer]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [item["action"] for item in self.buffer],
            dtype=torch.long,
            device=self.device,
        )
        old_logp = torch.tensor(
            [item["log_probability"] for item in self.buffer],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.tensor(
            [item["value"] for item in self.buffer],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            [item["reward"] for item in self.buffer],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [item["done"] for item in self.buffer],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.tensor(0.0, device=self.device)
        for index in reversed(range(len(self.buffer))):
            next_value = (
                torch.tensor(0.0, device=self.device)
                if index == len(self.buffer) - 1
                else old_values[index + 1]
            )
            non_terminal = 1.0 - dones[index]
            delta = rewards[index] + self.config.gamma * next_value * non_terminal - old_values[index]
            last_advantage = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * non_terminal
                * last_advantage
            )
            advantages[index] = last_advantage
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        indices = np.arange(len(self.buffer))
        losses: list[float] = []
        for _ in range(int(self.config.update_epochs)):
            np.random.shuffle(indices)
            for start in range(0, len(indices), int(self.config.minibatch_size)):
                batch = indices[start : start + int(self.config.minibatch_size)]
                logits, values = self.network(observations[batch])
                distribution = torch.distributions.Categorical(
                    logits=_mask_logits(logits, masks[batch])
                )
                logp = distribution.log_prob(actions[batch])
                ratio = torch.exp(logp - old_logp[batch])
                objective = torch.minimum(
                    ratio * advantages[batch],
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * advantages[batch],
                )
                loss = (
                    -objective.mean()
                    + self.config.value_coefficient
                    * F.mse_loss(values, returns[batch])
                    - self.config.entropy_coefficient
                    * distribution.entropy().mean()
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.gradient_clip,
                )
                self.optimizer.step()
                losses.append(float(loss.item()))
        self.clear()
        return float(np.mean(losses)) if losses else None


class TaskRankNetwork(nn.Module):
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class TrainableTaskRanker:
    """Plackett-Luce task permutation with an episode REINFORCE update."""

    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        entropy_coefficient: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.network = TaskRankNetwork(feature_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(learning_rate),
        )
        self.entropy_coefficient = float(entropy_coefficient)
        self.begin_episode()

    def begin_episode(self) -> None:
        self._decisions: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._rewards: dict[int, list[float]] = {}

    def order(
        self,
        task_ids: Sequence[int],
        features: np.ndarray,
        *,
        training: bool,
    ) -> list[int]:
        ids = [int(value) for value in task_ids]
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        logits = self.network(tensor)
        if not training:
            scores = logits.detach().cpu().numpy()
            return [
                ids[index]
                for index in sorted(
                    range(len(ids)),
                    key=lambda index: (-float(scores[index]), ids[index]),
                )
            ]
        remaining = list(range(len(ids)))
        ordering: list[int] = []
        while remaining:
            available_logits = logits[remaining]
            distribution = torch.distributions.Categorical(logits=available_logits)
            local_index = distribution.sample()
            selected_position = remaining.pop(int(local_index.item()))
            task_id = ids[selected_position]
            ordering.append(task_id)
            self._decisions.setdefault(task_id, []).append(
                (distribution.log_prob(local_index), distribution.entropy())
            )
        return ordering

    def record_reward(self, task_id: int, reward: float) -> None:
        self._rewards.setdefault(int(task_id), []).append(float(reward))

    def update(self) -> float | None:
        terms = []
        for task_id, decisions in self._decisions.items():
            rewards = self._rewards.get(task_id, [])
            if not rewards:
                continue
            signal = float(np.mean(rewards))
            for log_probability, entropy in decisions:
                terms.append(
                    -log_probability * signal
                    - self.entropy_coefficient * entropy
                )
        if not terms:
            self.begin_episode()
            return None
        loss = torch.stack(terms).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()
        value = float(loss.item())
        self.begin_episode()
        return value


class StateNoveltyBonus:
    """Deterministic count-based intrinsic reward used by IRWS."""

    def __init__(self, precision: int = 2) -> None:
        self.precision = int(precision)
        self.counts: dict[tuple[float, ...], int] = {}

    def value(self, observation: np.ndarray, *, update: bool) -> float:
        key = tuple(
            float(value)
            for value in np.round(
                np.asarray(observation, dtype=np.float64),
                self.precision,
            ).tolist()
        )
        count = int(self.counts.get(key, 0))
        if update:
            count += 1
            self.counts[key] = count
        return float(1.0 / math.sqrt(max(count, 1)))


__all__ = [
    "A2CConfig",
    "LocalMultiCriticA2C",
    "MaskedA2CAgent",
    "MaskedPPOAgent",
    "PPOConfig",
    "StateNoveltyBonus",
    "TrainableTaskRanker",
]
