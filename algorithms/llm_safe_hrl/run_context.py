"""Per-execution identity and paths for LLM-Safe-HRL experiments."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from algorithms.llm_safe_hrl.scenario_registry import (
    ExperimentProtocolContext,
)


@dataclass(frozen=True)
class DeadlineSetting:
    """Canonical deadline mixture used by both SeEvo and Safe-HRL."""

    code: str
    name: str
    alpha_small: float
    alpha_large: float
    alpha_small_probability: float

    def identity(self) -> dict[str, Any]:
        return {
            "ddl_code": self.code,
            "ddl_name": self.name,
            "deadline_alpha_small": self.alpha_small,
            "deadline_alpha_large": self.alpha_large,
            "deadline_alpha_small_prob": self.alpha_small_probability,
        }


DEADLINE_SETTING_REGISTRY: Mapping[str, DeadlineSetting] = MappingProxyType({
    "T": DeadlineSetting("T", "Tight", 2.0, 3.0, 0.8),
    "M": DeadlineSetting("M", "Medium", 2.0, 3.0, 0.5),
    "L": DeadlineSetting("L", "Loose", 2.0, 3.0, 0.2),
})
_DEADLINE_ALIASES = MappingProxyType({
    "T": "T",
    "TIGHT": "T",
    "M": "M",
    "MEDIUM": "M",
    "L": "L",
    "LOOSE": "L",
})
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve_deadline_setting(value: str) -> DeadlineSetting:
    """Normalize T/M/L and full names into one immutable setting."""
    key = str(value).strip().upper()
    try:
        return DEADLINE_SETTING_REGISTRY[_DEADLINE_ALIASES[key]]
    except KeyError as exc:
        raise ValueError(
            "ddl must be one of T, M, L, Tight, Medium, Loose"
        ) from exc


def validate_execution_identifier(value: str, label: str) -> str:
    """Accept one path-safe component and reject traversal/separators."""
    identifier = str(value).strip()
    if (
        not identifier
        or identifier in {".", ".."}
        or not _SAFE_IDENTIFIER.fullmatch(identifier)
    ):
        raise ValueError(
            f"{label} must contain only letters, digits, '.', '_', or '-' "
            "and must not be a path"
        )
    return identifier


@dataclass(frozen=True)
class ExperimentRunContext:
    """One collision-free LLM execution within a semantic experiment."""

    protocol_context: ExperimentProtocolContext
    deadline: DeadlineSetting
    execution_id: str
    runtime_output_root: Path
    run_name: str | None = None

    def __post_init__(self) -> None:
        validate_execution_identifier(self.execution_id, "execution_id")
        if self.run_name not in (None, ""):
            validate_execution_identifier(str(self.run_name), "run_name")
        object.__setattr__(
            self,
            "runtime_output_root",
            Path(self.runtime_output_root).resolve(),
        )

    @property
    def experiment_key(self) -> str:
        if self.protocol_context.protocol == "single":
            group = str(self.protocol_context.source_scenario)
        else:
            group = f"MULTI_{self.protocol_context.resource_scale}"
        return f"{group}_{self.deadline.code}"

    @property
    def artifact_output_root(self) -> Path:
        return (
            self.protocol_context.artifact_output_root
            / self.deadline.code
            / self.execution_id
        )

    @property
    def checkpoint_root(self) -> Path:
        return (
            self.protocol_context.checkpoint_root
            / self.deadline.code
            / self.execution_id
        )

    @property
    def library_path(self) -> Path:
        return (
            self.artifact_output_root
            / self.protocol_context.library_filename
        )

    def semantic_identity(self) -> dict[str, Any]:
        return {
            "experiment_protocol": self.protocol_context.identity(),
            "deadline_setting": self.deadline.identity(),
        }

    def execution_identity(self) -> dict[str, Any]:
        return {
            **self.semantic_identity(),
            "experiment_key": self.experiment_key,
            "execution_id": self.execution_id,
            "run_name": self.run_name,
            "runtime_output_root": str(self.runtime_output_root),
            "output_root": str(self.artifact_output_root.resolve()),
            "checkpoint_root": str(self.checkpoint_root.resolve()),
            "heuristic_library": str(self.library_path.resolve()),
        }


def deadline_identity_matches(
    expected: DeadlineSetting,
    actual: Mapping[str, Any],
) -> bool:
    """Compare a manifest deadline identity without accepting partial data."""
    return dict(actual or {}) == expected.identity()


__all__ = [
    "DEADLINE_SETTING_REGISTRY",
    "DeadlineSetting",
    "ExperimentRunContext",
    "deadline_identity_matches",
    "resolve_deadline_setting",
    "validate_execution_identifier",
]
