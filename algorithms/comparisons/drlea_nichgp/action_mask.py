"""Mask utilities used by routing exploration and Double-DQN targets."""

from __future__ import annotations

import numpy as np
import torch


def validate_mask(mask, action_dim: int, *, allow_empty: bool = False):
    result = np.asarray(mask, dtype=np.float32).reshape(-1)
    if result.shape != (int(action_dim),):
        raise ValueError(
            f"mask shape {result.shape} != ({int(action_dim)},)"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("action mask contains NaN or Inf")
    result = (result > 0.5).astype(np.float32)
    if not allow_empty and not np.any(result):
        raise ValueError("action mask has no legal action")
    return result


def sample_legal(mask, rng: np.random.Generator) -> int:
    mask = validate_mask(mask, len(mask))
    return int(rng.choice(np.flatnonzero(mask > 0.5)))


def masked_argmax_numpy(values, mask) -> int:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = validate_mask(mask, values.size)
    masked = np.where(mask > 0.5, values, -np.inf)
    return int(np.argmax(masked))


def masked_argmax_torch(
    q_values: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    if q_values.ndim != 2 or masks.shape != q_values.shape:
        raise ValueError("Q values and masks must be equal 2-D tensors")
    legal = masks > 0.5
    if not torch.all(torch.any(legal, dim=1)):
        raise ValueError("non-terminal Double-DQN mask cannot be empty")
    return q_values.masked_fill(~legal, -torch.inf).argmax(dim=1)
