"""Reproducible multi-seed training orchestration for fuzzy baselines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

from . import LLM_SAFE_HRL_ROOT  # noqa: F401
from hrl_mix.model_selection import FeasibilityFirstModelMetrics, is_better_model

from .evaluation import ComparisonPolicy, evaluate_policy, make_environment, run_episode
from .protocol import FuzzyComparisonProtocol


TRAINING_MANIFEST_VERSION = 1


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_identity(protocol: FuzzyComparisonProtocol, checkpoint: Path) -> dict[str, Any]:
    return {
        "artifact_protocol_status": (
            "legacy" if protocol.protocol_mode == "legacy" else "formal"
        ),
        "protocol_mode": protocol.protocol_mode,
        "source_scenario": protocol.source_scenario,
        "resource_scale": protocol.resource_scale,
        "training_scenarios": list(protocol.training_scenarios),
        "test_scenarios": list(protocol.test_scenarios),
        "train_seeds": list(protocol.train_seeds),
        "validation_seeds": list(protocol.validation_seeds),
        "test_seeds": list(protocol.test_seeds),
        "protocol_hash": protocol.protocol_hash,
        "checkpoint_sha256": _file_hash(checkpoint),
    }


def _validate_checkpoint_identity(
    protocol: FuzzyComparisonProtocol,
    checkpoint: Path,
    identity_path: Path,
) -> dict[str, Any]:
    if not identity_path.is_file():
        raise ValueError("baseline checkpoint is missing protocol identity")
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    expected = _checkpoint_identity(protocol, checkpoint)
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(
            "baseline checkpoint protocol identity mismatch: " + ", ".join(mismatches)
        )
    return payload

def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


@dataclass(frozen=True)
class TrainingResult:
    output_dir: str
    checkpoint_path: str
    best_validation: dict[str, Any]
    final_test: dict[str, Any]
    manifest_path: str


def train_baseline(
    protocol: FuzzyComparisonProtocol,
    policy: ComparisonPolicy,
    *,
    output_dir: str | Path,
    episodes: int,
    validation_interval: int,
    reward_config: Mapping[str, Any] | None = None,
    max_assignment_steps: int = 1_000_000,
    optimizer_seed: int = 0,
) -> TrainingResult:
    """Train and select a checkpoint without exposing final-test seeds."""
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    if int(validation_interval) <= 0:
        raise ValueError("validation_interval must be positive")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    train_log = root / "training_metrics.jsonl"
    validation_log = root / "validation_metrics.jsonl"
    checkpoint = root / "best_checkpoint.pt"
    checkpoint_identity_path = root / "checkpoint_identity.json"
    reward_values = dict(reward_config or {})
    seed_everything(int(optimizer_seed))
    incumbent: FeasibilityFirstModelMetrics | None = None
    best_episode = 0

    for episode in range(1, int(episodes) + 1):
        seed = int(protocol.train_seeds[(episode - 1) % len(protocol.train_seeds)])
        protocol.assert_not_test_seed(seed, "training")
        training_scenario = protocol.training_scenarios[
            (episode - 1) % len(protocol.training_scenarios)
        ]
        training_protocol = protocol.for_scenario(training_scenario)
        env = make_environment(training_protocol, seed, reward_config=reward_values)
        record = run_episode(
            env,
            policy,
            seed=seed,
            training=True,
            max_assignment_steps=max_assignment_steps,
        )
        _append_jsonl(train_log, {"episode": episode, **record})
        should_validate = (
            episode % int(validation_interval) == 0
            or episode == int(episodes)
        )
        if not should_validate:
            continue
        validation_results = [
            evaluate_policy(
                protocol.for_scenario(scenario),
                policy,
                split="validation",
                reward_config=reward_values,
                max_assignment_steps=max_assignment_steps,
            )
            for scenario in protocol.training_scenarios
        ]
        validation = max(
            validation_results,
            key=lambda item: item.model_selection.comparison_key,
        )
        validation_payload = {
            "episode": episode,
            "model_selection": validation.model_selection.to_dict(),
            "aggregate": validation.aggregate,
            "seed_records": list(validation.records),
        }
        _append_jsonl(validation_log, validation_payload)
        if is_better_model(validation.model_selection, incumbent):
            incumbent = validation.model_selection
            best_episode = int(episode)
            policy.save(str(checkpoint))
            _write_json(
                checkpoint_identity_path,
                _checkpoint_identity(protocol, checkpoint),
            )

    if incumbent is None or not checkpoint.is_file():
        raise RuntimeError("training produced no validated checkpoint")
    _validate_checkpoint_identity(
        protocol, checkpoint, checkpoint_identity_path
    )
    policy.load(str(checkpoint))
    final_tests = {
        scenario: evaluate_policy(
            protocol.for_scenario(scenario),
            policy,
            split="final_test",
            reward_config=reward_values,
            max_assignment_steps=max_assignment_steps,
        )
        for scenario in protocol.test_scenarios
    }
    final_test = final_tests[protocol.test_scenarios[0]]
    final_payload = {
        "split": "final_test",
        "used_for_model_selection": False,
        "model_selection_metrics_for_reporting_only": (
            final_test.model_selection.to_dict()
        ),
        "aggregate": final_test.aggregate,
        "seed_records": list(final_test.records),
        "scenario_results": {
            scenario: {
                "aggregate": result.aggregate,
                "seed_records": list(result.records),
            }
            for scenario, result in final_tests.items()
        },
        "frozen_checkpoint": checkpoint.name,
        "training_during_generalization_test": False,
        "checkpoint_reselection_during_generalization_test": False,
    }
    _write_json(root / "final_test_metrics.json", final_payload)
    manifest = {
        "training_manifest_version": TRAINING_MANIFEST_VERSION,
        "method_id": str(policy.method_id),
        "protocol": protocol.to_manifest(),
        "episodes": int(episodes),
        "validation_interval": int(validation_interval),
        "optimizer_seed": int(optimizer_seed),
        "deadline_cache_paths": dict(protocol.deadline_cache_paths),
        "best_episode": int(best_episode),
        "best_validation": incumbent.to_dict(),
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": _file_hash(checkpoint),
        "checkpoint_identity": checkpoint_identity_path.name,
        "training_log": train_log.name,
        "validation_log": validation_log.name,
        "final_test_report": "final_test_metrics.json",
        "final_test_used_for_selection": False,
        "reward_config": reward_values,
        "policy_config": (
            policy.configuration()
            if callable(getattr(policy, "configuration", None))
            else {}
        ),
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return TrainingResult(
        output_dir=str(root),
        checkpoint_path=str(checkpoint),
        best_validation=incumbent.to_dict(),
        final_test=final_payload,
        manifest_path=str(manifest_path),
    )


__all__ = ["TrainingResult", "seed_everything", "train_baseline"]
