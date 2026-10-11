"""Independent MADDPG with centralized critics and continuous VM preferences."""
from collections import deque
import copy

import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, observations, actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observations, 512), nn.ReLU(), nn.Linear(512, 256),
                                 nn.ReLU(), nn.Linear(256, actions), nn.Tanh())

    def forward(self, x):
        return (self.net(x) + 1) * 0.5


class Critic(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimensions, 512), nn.ReLU(), nn.Linear(512, 256),
                                 nn.ReLU(), nn.Linear(256, 1))

    def forward(self, observations, actions):
        return self.net(torch.cat((observations.flatten(1), actions.flatten(1)), dim=1)).squeeze(-1)


def choose_vm(preferences, legal):
    values = np.asarray(preferences, dtype=float)
    mask = np.asarray(legal, dtype=float)
    if values.ndim != 1 or values.shape != mask.shape or not np.isfinite(values).all():
        raise ValueError("preferences must be a finite vector matching VM count")
    if np.any(values < 0) or np.any(values > 1) or not np.any(mask > 0.5):
        raise ValueError("preferences outside [0,1] or empty legal set")
    return int(np.argmax(np.where(mask > 0.5, values, -np.inf)))


class MADDPG:
    def __init__(self, slots, observation_dim, action_dim, config, device="cpu"):
        self.slots, self.observation_dim, self.action_dim = slots, observation_dim, action_dim
        self.config = dict(config)
        self.device = torch.device(device)
        self.actors = nn.ModuleList([Actor(observation_dim, action_dim) for _ in range(slots)]).to(self.device)
        self.critics = nn.ModuleList([Critic(slots * (observation_dim + action_dim)) for _ in range(slots)]).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).requires_grad_(False)
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=config["lr_actor"]) for a in self.actors]
        self.critic_optimizers = [torch.optim.Adam(c.parameters(), lr=config["lr_critic"]) for c in self.critics]
        self.replay = deque(maxlen=config["buffer_size"])
        self.steps = 0
        self.actor_updates = 0
        self.critic_updates = 0

    def preferences(self, observations, owner, training):
        with torch.no_grad():
            value = self.actors[owner](torch.as_tensor(observations[owner], device=self.device)).cpu().numpy()
        if training:
            epsilon = max(0.05, self.config["epsilon"] - self.steps * 0.0000005)
            noise = max(0.05, self.config["noise_rate"] - self.steps * 0.0000005)
            value = np.random.uniform(0, 1, self.action_dim) if np.random.random() < epsilon else value + np.random.normal(0, noise, self.action_dim)
        return np.clip(value, 0, 1).astype(np.float32)

    def transition(self, observations, actions, rewards, next_observations, next_masks, done):
        self.replay.append(tuple(np.array(x, dtype=np.float32, copy=True) for x in
                                 (observations, actions, rewards, next_observations, next_masks, done)))
        self.steps += 1
        if len(self.replay) < self.config["batch_size"]:
            return None
        indices = np.random.choice(len(self.replay), self.config["batch_size"], replace=False)
        obs, actions, rewards, nxt, masks, done = [torch.as_tensor(np.stack(v), device=self.device)
                                                  for v in zip(*(self.replay[i] for i in indices))]
        decision = (masks.sum(dim=-1, keepdim=True) > 0).float()
        with torch.no_grad():
            next_actions = torch.stack([actor(nxt[:, i]) for i, actor in enumerate(self.target_actors)], dim=1) * decision
        losses = []
        for i in range(self.slots):
            # Train only on arrived, unfinished owners, never padded/future workflows.
            weight = obs[:, i, 0]
            if weight.sum() == 0:
                continue
            with torch.no_grad():
                target = rewards[:, i] + self.config["gamma"] * (1 - done[:, i]) * self.target_critics[i](nxt, next_actions)
            loss = (((self.critics[i](obs, actions) - target) ** 2) * weight).sum() / weight.sum()
            self.critic_optimizers[i].zero_grad()
            loss.backward()
            self.critic_optimizers[i].step()
            self.critic_updates += 1
            selecting = obs[:, i, 1]
            if selecting.sum() > 0:
                self.critics[i].requires_grad_(False)
                proposed = actions.clone()
                proposed[:, i] = self.actors[i](obs[:, i]) * selecting.unsqueeze(-1)
                actor_loss = -(self.critics[i](obs, proposed) * selecting).sum() / selecting.sum()
                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[i].step()
                self.actor_updates += 1
                self.critics[i].requires_grad_(True)
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            tau = self.config["tau"]
            for targets, sources in ((self.target_actors, self.actors), (self.target_critics, self.critics)):
                for target, source in zip(targets.parameters(), sources.parameters()):
                    target.lerp_(source, tau)
        return float(np.mean(losses)) if losses else None

    def save(self, path):
        torch.save(dict(schema=3, slots=self.slots, observation_dim=self.observation_dim,
                        action_dim=self.action_dim, config=self.config,
                        actors=self.actors.state_dict(), critics=self.critics.state_dict()), path)

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True)
        expected = (3, self.slots, self.observation_dim, self.action_dim, self.config)
        if (data["schema"], data["slots"], data["observation_dim"], data["action_dim"], data["config"]) != expected:
            raise ValueError("MADDPG checkpoint dimensions/config mismatch")
        self.actors.load_state_dict(data["actors"])
        self.critics.load_state_dict(data["critics"])
        self.target_actors.load_state_dict(data["actors"])
        self.target_critics.load_state_dict(data["critics"])
