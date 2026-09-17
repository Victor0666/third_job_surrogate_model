"""Masked Dueling Double DQN retained from the upstream DRL-EA identity."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .action_mask import (
    masked_argmax_numpy,
    masked_argmax_torch,
    sample_legal,
    validate_mask,
)
from . import METHOD_ID
from .config import AgentConfig
from .replay_buffer import MaskedReplayBuffer


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int],
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dims[0])),
            nn.ReLU(),
            nn.Linear(int(hidden_dims[0]), int(hidden_dims[1])),
            nn.ReLU(),
        )
        self.value = nn.Linear(int(hidden_dims[1]), 1)
        self.advantage = nn.Linear(int(hidden_dims[1]), int(action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.features(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class MaskedD3QN:
    checkpoint_schema_version = 2
    expected_role = None

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: AgentConfig,
        seed: int,
        device: str | None = None,
        role: str = "generic",
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.seed = int(seed)
        self.role = str(role)
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.device = torch.device(
            device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.online = DuelingQNetwork(
            self.state_dim,
            self.action_dim,
            config.hidden_dims,
        ).to(self.device)
        self.target = DuelingQNetwork(
            self.state_dim,
            self.action_dim,
            config.hidden_dims,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=float(config.learning_rate),
        )
        self.replay = MaskedReplayBuffer(
            self.state_dim,
            self.action_dim,
            config.replay_capacity,
            self.seed,
        )
        self.epsilon = float(config.epsilon_start)
        self.learn_steps = 0

    def choose_action(
        self,
        state,
        mask,
        *,
        training: bool,
    ) -> tuple[int, str]:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,) or not np.all(
            np.isfinite(state)
        ):
            raise ValueError("invalid D3QN state")
        mask = validate_mask(mask, self.action_dim)
        if training and self.rng.random() < self.epsilon:
            return sample_legal(mask, self.rng), "random_legal"
        with torch.no_grad():
            tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.online(tensor).squeeze(0).cpu().numpy()
        return masked_argmax_numpy(q_values, mask), "greedy_legal"

    def remember(self, *args, **kwargs) -> None:
        self.replay.add(*args, **kwargs)

    def soft_update(self) -> None:
        tau = float(self.config.tau)
        with torch.no_grad():
            for target, online in zip(
                self.target.parameters(), self.online.parameters()
            ):
                target.mul_(1.0 - tau).add_(online, alpha=tau)

    def compute_targets(self, batch) -> torch.Tensor:
        rewards = torch.as_tensor(
            batch.rewards, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch.dones, dtype=torch.bool, device=self.device
        )
        next_states = torch.as_tensor(
            batch.next_states, dtype=torch.float32, device=self.device
        )
        next_masks = torch.as_tensor(
            batch.next_masks, dtype=torch.float32, device=self.device
        )
        result = rewards.clone()
        active = ~dones
        if torch.any(active):
            active_states = next_states[active]
            active_masks = next_masks[active]
            with torch.no_grad():
                next_actions = masked_argmax_torch(
                    self.online(active_states), active_masks
                )
                target_values = self.target(active_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            result[active] += float(self.config.gamma) * target_values
        return result

    def learn(self) -> float | None:
        if self.replay.size < int(self.config.batch_size):
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(
            batch.states, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch.actions, dtype=torch.long, device=self.device
        )
        predicted = self.online(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        target = self.compute_targets(batch)
        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.soft_update()
        self.epsilon = max(
            float(self.config.epsilon_end),
            self.epsilon * float(self.config.epsilon_decay),
        )
        self.learn_steps += 1
        return float(loss.item())

    def checkpoint_payload(self, *, metrics: dict | None = None, protocol_identity: dict | None = None) -> dict:
        return {
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "method_id": METHOD_ID,
            "agent_role": self.role,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "agent_config": asdict(self.config),
            "seed": self.seed,
            "compute_device": str(self.device),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "replay_metadata": self.replay.metadata(),
            "selection_metrics": dict(metrics or {}),
            "protocol_identity": (
                dict(protocol_identity)
                if protocol_identity is not None
                else None
            ),
        }

    def save(self, path: str | Path, *, metrics: dict | None = None, protocol_identity: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(metrics=metrics, protocol_identity=protocol_identity), path)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        expected_protocol_identity: dict | None = None,
    ) -> "MaskedD3QN":
        try:
            payload = torch.load(
                Path(path), map_location="cpu", weights_only=False
            )
        except TypeError:
            payload = torch.load(Path(path), map_location="cpu")
        if (
            int(payload.get("checkpoint_schema_version", -1)) != 2
            or payload.get("method_id") != METHOD_ID
        ):
            raise ValueError("incompatible DRL-EA checkpoint schema")
        if expected_protocol_identity is not None:
            from .config import validate_protocol_artifact_identity
            validate_protocol_artifact_identity(
                expected_protocol_identity,
                payload.get("protocol_identity"),
                artifact_name="DRL-EA checkpoint",
            )
        expected_role = getattr(cls, "expected_role", None)
        if (
            expected_role is not None
            and payload.get("agent_role") != expected_role
        ):
            raise ValueError(
                f"checkpoint role {payload.get('agent_role')!r} is "
                f"incompatible with {expected_role!r}"
            )
        config = AgentConfig(**payload["agent_config"])
        agent = cls(
            int(payload["state_dim"]),
            int(payload["action_dim"]),
            config,
            int(payload["seed"]),
            device=device,
            role=str(payload["agent_role"]),
        )
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.epsilon = float(payload["epsilon"])
        agent.learn_steps = int(payload["learn_steps"])
        return agent
