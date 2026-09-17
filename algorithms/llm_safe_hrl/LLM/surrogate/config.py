"""Validated configuration for offline surrogate screening."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class WarmupConfig:
    min_structures: int = 20
    min_exact_evaluations: int = 500


@dataclass(frozen=True)
class StructureGateConfig:
    initial_full_cma_fraction: float = 0.70
    mature_full_cma_fraction: float = 0.50
    min_true_structures: int = 2


@dataclass(frozen=True)
class ParameterGateConfig:
    min_exact_pairs: int = 12
    min_true_candidates_per_generation: int = 3


@dataclass(frozen=True)
class ActiveLearningConfig:
    uncertainty_fraction: float = 0.20
    random_fraction: float = 0.10


@dataclass(frozen=True)
class FailSafeConfig:
    min_feasible_recall: float = 0.98
    min_promising_recall: float = 0.90
    audit_window: int = 100
    min_audit_samples: int = 20
    audit_every_generations: int = 10
    stable_audit_every_generations: int = 30
    stable_audit_rounds: int = 3


@dataclass(frozen=True)
class SurrogateConfig:
    enabled: bool = False
    random_seed: int = 0
    retrain_every_exact_evaluations: int = 100
    conservative_sigma: float = 1.0
    schema_version: str = "llm_surrogate_v1"
    dataset_path: str = "surrogate/exact_labels.jsonl"
    checkpoint_path: str = "surrogate/model_checkpoint.pkl"
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    structure_gate: StructureGateConfig = field(default_factory=StructureGateConfig)
    parameter_gate: ParameterGateConfig = field(default_factory=ParameterGateConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    fail_safe: FailSafeConfig = field(default_factory=FailSafeConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SurrogateConfig":
        if not value:
            return cls()
        raw = dict(value)
        nested = {
            "warmup": WarmupConfig,
            "structure_gate": StructureGateConfig,
            "parameter_gate": ParameterGateConfig,
            "active_learning": ActiveLearningConfig,
            "fail_safe": FailSafeConfig,
        }
        known = set(cls.__dataclass_fields__)
        kwargs = {key: raw[key] for key in known if key in raw and key not in nested}
        for key, nested_type in nested.items():
            kwargs[key] = nested_type(**dict(raw.get(key, {})))
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed <= 2**32 - 1
        ):
            raise ValueError("surrogate.random_seed must be a uint32 integer")
        positive = {
            "retrain_every_exact_evaluations": self.retrain_every_exact_evaluations,
            "warmup.min_structures": self.warmup.min_structures,
            "warmup.min_exact_evaluations": self.warmup.min_exact_evaluations,
            "structure_gate.min_true_structures": self.structure_gate.min_true_structures,
            "parameter_gate.min_exact_pairs": self.parameter_gate.min_exact_pairs,
            "parameter_gate.min_true_candidates_per_generation": (
                self.parameter_gate.min_true_candidates_per_generation
            ),
            "fail_safe.audit_window": self.fail_safe.audit_window,
            "fail_safe.min_audit_samples": self.fail_safe.min_audit_samples,
            "fail_safe.audit_every_generations": self.fail_safe.audit_every_generations,
            "fail_safe.stable_audit_every_generations": (
                self.fail_safe.stable_audit_every_generations
            ),
            "fail_safe.stable_audit_rounds": self.fail_safe.stable_audit_rounds,
        }
        for name, number in positive.items():
            if int(number) < 1:
                raise ValueError(f"surrogate.{name} must be positive")
        fractions = {
            "structure_gate.initial_full_cma_fraction": (
                self.structure_gate.initial_full_cma_fraction
            ),
            "structure_gate.mature_full_cma_fraction": (
                self.structure_gate.mature_full_cma_fraction
            ),
            "active_learning.uncertainty_fraction": (
                self.active_learning.uncertainty_fraction
            ),
            "active_learning.random_fraction": self.active_learning.random_fraction,
            "fail_safe.min_feasible_recall": self.fail_safe.min_feasible_recall,
            "fail_safe.min_promising_recall": self.fail_safe.min_promising_recall,
        }
        for name, number in fractions.items():
            if not 0.0 <= float(number) <= 1.0:
                raise ValueError(f"surrogate.{name} must be in [0, 1]")
        if self.conservative_sigma < 0.0:
            raise ValueError("surrogate.conservative_sigma must be non-negative")
        if (
            self.fail_safe.stable_audit_every_generations
            < self.fail_safe.audit_every_generations
        ):
            raise ValueError(
                "surrogate stable audit interval must not be shorter than the initial interval"
            )
        if not self.schema_version.strip():
            raise ValueError("surrogate.schema_version must not be empty")
        if not str(self.dataset_path).strip() or not str(self.checkpoint_path).strip():
            raise ValueError("surrogate dataset/checkpoint paths must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
