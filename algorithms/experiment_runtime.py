"""Shared runtime device selection and non-invasive training-budget audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


_COUNT_FIELDS = (
    "rl_training_episodes",
    "training_workflows",
    "environment_steps",
    "validation_runs",
    "llm_calls",
    "heuristic_evaluations",
    "cma_es_evaluations",
    "gp_evaluations",
    "offline_pretraining_epochs",
    "demonstration_environment_runs",
)


def resolve_torch_device(requested: str | None = "auto"):
    """Resolve ``auto`` to CUDA when available, otherwise CPU.

    An explicit device remains supported. Explicit CUDA requests fail closed
    when CUDA is unavailable instead of silently changing the experiment.
    """
    import torch

    value = "auto" if requested is None else str(requested).strip().lower()
    if value in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


@dataclass(frozen=True)
class TrainingBudgetAudit:
    """Machine-readable counters observed by one completed training runner."""

    method_id: str
    rl_training_episodes: int | None = 0
    training_workflows: int | None = 0
    environment_steps: int | None = 0
    validation_runs: int | None = 0
    llm_calls: int | None = 0
    heuristic_evaluations: int | None = 0
    cma_es_evaluations: int | None = 0
    gp_evaluations: int | None = 0
    offline_pretraining_epochs: int | None = 0
    demonstration_environment_runs: int | None = 0
    wall_clock_seconds: float = 0.0
    device: str = "cpu"
    train_seeds: tuple[int, ...] = ()
    counter_sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.method_id).strip():
            raise ValueError("training budget method_id must be non-empty")
        for name in _COUNT_FIELDS:
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"training budget {name} must be a non-negative int or null")
        wall = float(self.wall_clock_seconds)
        if not math.isfinite(wall) or wall < 0.0:
            raise ValueError("training budget wall_clock_seconds must be finite and non-negative")
        if not str(self.device).strip():
            raise ValueError("training budget device must be non-empty")
        seeds = tuple(int(seed) for seed in self.train_seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("training budget train_seeds must not contain duplicates")
        object.__setattr__(self, "train_seeds", seeds)
        object.__setattr__(self, "counter_sources", dict(self.counter_sources))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train_seeds"] = list(self.train_seeds)
        return payload


def write_training_budget(
    path: str | Path,
    audit: TrainingBudgetAudit | Mapping[str, Any],
) -> Path:
    """Validate and atomically persist one training-budget audit."""
    value = audit if isinstance(audit, TrainingBudgetAudit) else TrainingBudgetAudit(**dict(audit))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value.to_dict(),
            handle,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    temporary.replace(target)
    return target


def count_completed_workflows(record: Mapping[str, Any]) -> int:
    """Read an observed completed-workflow count without estimating it."""
    for name in ("completed_workflow_count", "workflow_count"):
        if name in record:
            value = int(record[name])
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            return value
    raise ValueError("episode metrics do not expose a completed-workflow count")


__all__ = [
    "TrainingBudgetAudit",
    "count_completed_workflows",
    "resolve_torch_device",
    "write_training_budget",
]
