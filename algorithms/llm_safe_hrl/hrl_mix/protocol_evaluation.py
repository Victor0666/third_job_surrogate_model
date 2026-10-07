# -*- coding: utf-8 -*-
"""Read-only Single/Multi protocol evaluation for frozen Safe-HRL agents.

This module is deliberately an orchestration layer. It reuses the production
environment, D3QN agents, checkpoint reader, heuristic-library loader, and
multi-seed evaluator. It never calls training, SeEvo, CMA-ES, counterfactual
analysis, replay insertion, or any parameter-update method.
"""

from __future__ import annotations

from base.energy_observation import ENERGY_DIAGNOSTIC_FIELDS
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algorithms.llm_safe_hrl.scenario_registry import (
    ExperimentProtocolContext,
    resolve_experiment_protocol,
)
from base.d3qn_agent import D3QNAgent
from base.hrl_env import CloudWorkflowEnv_VMAgents
from base.manager_heuristics import load_manager_heuristic_library
from hrl_mix.model_selection import read_best_checkpoint_manifest
from hrl_mix.train_config import (
    ROOT_DIR,
    environment_scenario_values,
    parse_deadline_cache_overrides,
    validate_single_deadline_cache_paths,
)
from hrl_mix.train_eval import evaluate_hrl_three_layer_multi_seed


DEFAULT_FINAL_TEST_SEEDS = tuple(range(201, 231))
FROZEN_EVALUATION_MANIFEST_VERSION = 1


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label} is missing {key}")
    return mapping[key]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            "evaluation result contains NaN or infinity"
        )

    return value


def _append_seed_result(
    output_dir: Path,
    scenario: str,
    seed: int,
    seed_result: Any,
) -> None:
    """Persist one completed test seed immediately.

    Two files are maintained:

    1. JSONL:
       Full per-seed result for lossless diagnostics.

    2. CSV:
       Core metrics for convenient real-time inspection.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = _json_value(seed_result)

    if not isinstance(result, list) or len(result) < 4:
        raise ValueError(
            "unexpected single-seed evaluation result"
        )

    avg_vm = result[0]
    avg_host = result[1]
    avg_manager = result[2]
    total_energy = result[3]

    safety = {}

    if len(result) >= 5 and isinstance(result[4], Mapping):
        safety = result[4]

    # ============================================================
    # JSONL
    # ============================================================
    jsonl_path = (
        output_dir
        / f"scenario_{scenario}_seed_results.jsonl"
    )

    json_record = {
        "scenario": str(scenario),
        "seed": int(seed),
        "avg_vm_reward": avg_vm,
        "avg_host_reward": avg_host,
        "avg_manager_reward": avg_manager,
        "total_energy": total_energy,
        "safety_metrics": safety,
    }

    with jsonl_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            _canonical_json(json_record) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())

    # ============================================================
    # CSV
    # ============================================================
    csv_path = (
        output_dir
        / f"scenario_{scenario}_seed_results.csv"
    )

    csv_fields = [
        "scenario",
        "seed",
        "avg_vm_reward",
        "avg_host_reward",
        "avg_manager_reward",
        "total_energy",
        "deadline_violation_rate",
        "max_fuzzy_lateness",
        "mean_fuzzy_lateness",
        "fuzzy_energy_mean",
        "fuzzy_energy_std",
        "fuzzy_energy_score",
        "fallback_rate",
        "shield_intervention_rate",
        "no_safe_action_rate",
        "safety_cost",
        "scheduling_time_seconds",
    ]

    csv_row = {
        "scenario": str(scenario),
        "seed": int(seed),
        "avg_vm_reward": avg_vm,
        "avg_host_reward": avg_host,
        "avg_manager_reward": avg_manager,
        "total_energy": total_energy,
        "deadline_violation_rate": safety.get(
            "deadline_violation_rate"
        ),
        "max_fuzzy_lateness": safety.get(
            "max_fuzzy_lateness"
        ),
        "mean_fuzzy_lateness": safety.get(
            "mean_fuzzy_lateness"
        ),
        "fuzzy_energy_mean": safety.get(
            "fuzzy_energy_mean"
        ),
        "fuzzy_energy_std": safety.get(
            "fuzzy_energy_std"
        ),
        "fuzzy_energy_score": safety.get(
            "fuzzy_energy_score"
        ),
        "fallback_rate": safety.get(
            "fallback_rate"
        ),
        "shield_intervention_rate": safety.get(
            "shield_intervention_rate"
        ),
        "no_safe_action_rate": safety.get(
            "no_safe_action_rate"
        ),
        "safety_cost": safety.get(
            "safety_cost"
        ),
        "scheduling_time_seconds": safety.get(
            "scheduling_time_seconds"
        ),
    }

    csv_fields += ["energy_decision_count", *ENERGY_DIAGNOSTIC_FIELDS]
    csv_row.update({field: safety.get(field) for field in ["energy_decision_count", *ENERGY_DIAGNOSTIC_FIELDS]})
    file_exists = csv_path.is_file()

    with csv_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(csv_row)

        # 立即刷新到磁盘。
        handle.flush()
        os.fsync(handle.fileno())

    print(
        "[frozen-eval] "
        f"scenario={scenario} "
        f"seed={seed} finished and saved -> "
        f"{csv_path}",
        flush=True,
    )


def _normalize_test_seeds(
    values: Sequence[int],
    context: ExperimentProtocolContext,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(
            "test_seeds must be a sequence of integers"
        )

    seeds = tuple(
        int(value)
        for value in values
    )

    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(seed < 0 for seed in seeds)
    ):
        raise ValueError(
            "test_seeds must be non-empty, non-negative, and unique"
        )

    forbidden = set(
        context.safe_hrl_train_seeds
    ).union(
        context.safe_hrl_validation_seeds
    )

    overlap = sorted(
        forbidden.intersection(seeds)
    )

    if overlap:
        raise ValueError(
            "frozen test seeds overlap training/validation seeds: "
            f"{overlap}"
        )

    formal_seeds = tuple(
        context.final_test_seeds
    )

    # ------------------------------------------------------------
    # Diagnostic-only partial-seed switch
    #
    # 默认：
    #   仍要求完整 201~230。
    #
    # 当：
    #   SAFE_HRL_DIAGNOSTIC_PARTIAL_TEST=1
    #
    # 时：
    #   可以测试正式种子集合的前 N 个，例如：
    #   201
    #   201 202 203
    #   201 ... 210
    #
    # 不允许任意挑 seed。
    # ------------------------------------------------------------
    diagnostic_partial = (
        os.environ.get(
            "SAFE_HRL_DIAGNOSTIC_PARTIAL_TEST",
            "0",
        )
        == "1"
    )

    if diagnostic_partial:
        expected_prefix = formal_seeds[
            : len(seeds)
        ]

        if seeds != expected_prefix:
            raise ValueError(
                "diagnostic partial test seeds must be the leading "
                "prefix of "
                f"{list(formal_seeds)}; "
                f"received {list(seeds)}"
            )

    elif seeds != formal_seeds:
        raise ValueError(
            "formal frozen evaluation seeds must exactly match "
            f"{list(formal_seeds)}"
        )

    return seeds


def _snapshot_config(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = _required(
        manifest,
        "config_snapshot",
        "checkpoint manifest",
    )

    if not isinstance(snapshot, Mapping):
        raise ValueError(
            "checkpoint config_snapshot must be a mapping"
        )

    config = _required(
        snapshot,
        "config",
        "checkpoint config_snapshot",
    )

    if not isinstance(config, Mapping):
        raise ValueError(
            "checkpoint config_snapshot config must be a mapping"
        )

    return dict(config)


def build_frozen_scenario_env_kwargs(
    config: Mapping[str, Any],
    context: ExperimentProtocolContext,
    scenario: str,
    library_path: str | os.PathLike[str],
    deadline_cache_overrides: Mapping[str, str] | None = None,
    required_manager_mode: str = "heuristic_selection_mode",
) -> dict[str, Any]:
    """Build one real registry-backed test environment from saved config.

    Only task/resource/cache fields vary by ``scenario``.

    Reward, fuzzy DDL, shield, state, and Manager settings are
    read from the frozen checkpoint's configuration snapshot.
    """
    scenario_id = (
        str(scenario)
        .strip()
        .upper()
    )

    if scenario_id not in context.test_scenarios:
        raise ValueError(
            f"scenario {scenario_id} is outside protocol "
            f"test_scenarios {list(context.test_scenarios)}"
        )

    values = environment_scenario_values(
        scenario_id
    )

    cache_path = parse_deadline_cache_overrides(
        deadline_cache_overrides
    ).get(
        scenario_id
    )

    if cache_path is not None:
        resolved_cache = Path(
            cache_path
        ).expanduser()

        if not resolved_cache.is_absolute():
            resolved_cache = (
                ROOT_DIR
                / resolved_cache
            )

        values[
            "deadline_cache_path"
        ] = str(
            resolved_cache.resolve()
        )

    safe = _required(
        config,
        "safe_rl",
        "checkpoint config",
    )

    if not isinstance(safe, Mapping):
        raise ValueError(
            "checkpoint safe_rl config must be a mapping"
        )

    shield = _required(
        safe,
        "shield",
        "checkpoint safe_rl config",
    )

    state = _required(
        safe,
        "state",
        "checkpoint safe_rl config",
    )

    manager = _required(
        safe,
        "manager_heuristics",
        "checkpoint safe_rl config",
    )

    if not all(
        isinstance(value, Mapping)
        for value in (
            shield,
            state,
            manager,
        )
    ):
        raise ValueError(
            "checkpoint nested Safe-HRL config is invalid"
        )

    if not bool(
        _required(
            safe,
            "enabled",
            "checkpoint safe_rl config",
        )
    ):
        raise ValueError(
            "frozen protocol evaluation requires a Safe-HRL checkpoint"
        )

    if str(
        _required(
            manager,
            "mode",
            "checkpoint manager config",
        )
    ) != required_manager_mode:
        raise ValueError(
            "frozen protocol evaluation requires "
            "heuristic_selection_mode"
        )

    return {
        "dax_paths": list(
            values["dax_list"]
        ),
        "horizon": float(
            _required(
                config,
                "horizon",
                "checkpoint config",
            )
        ),
        "arrival_lambda": float(
            _required(
                config,
                "arrival_lambda",
                "checkpoint config",
            )
        ),
        "random_seed": 0,
        "max_ready_tasks": _required(
            config,
            "max_ready_tasks",
            "checkpoint config",
        ),
        "normalize": bool(
            _required(
                config,
                "normalize_obs",
                "checkpoint config",
            )
        ),
        "workflows_per_episode": int(
            _required(
                config,
                "workflows_per_episode",
                "checkpoint config",
            )
        ),
        "num_cloud_hosts": int(
            values["num_cloud_hosts"]
        ),
        "num_edge_hosts": int(
            values["num_edge_hosts"]
        ),
        "cloud_vms_per_host": tuple(
            values["cloud_vms_per_host"]
        ),
        "edge_vms_per_host": tuple(
            values["edge_vms_per_host"]
        ),
        "cloud_pc_tiers": tuple(
            values["cloud_pc_tiers"]
        ),
        "edge_pc_tiers": tuple(
            values["edge_pc_tiers"]
        ),
        "cloud_bw_tiers": tuple(
            values["cloud_bw_tiers"]
        ),
        "edge_bw_tiers": tuple(
            values["edge_bw_tiers"]
        ),
        "fuzzy_delta1": 0.75,
        "fuzzy_delta2": 1.2,
        "deadline_mode": "cache_fcfs",
        "deadline_cache_path": str(
            values["deadline_cache_path"]
        ),
        "deadline_cache_strict": True,
        "deadline_alpha_small": float(
            _required(
                config,
                "deadline_alpha_small",
                "checkpoint config",
            )
        ),
        "deadline_alpha_large": float(
            _required(
                config,
                "deadline_alpha_large",
                "checkpoint config",
            )
        ),
        "deadline_alpha_small_prob": float(
            _required(
                config,
                "deadline_alpha_small_prob",
                "checkpoint config",
            )
        ),
        "manager_alpha_delay": float(
            _required(
                config,
                "manager_alpha_delay",
                "checkpoint config",
            )
        ),
        "manager_delay_mode": str(
            _required(
                config,
                "manager_delay_mode",
                "checkpoint config",
            )
        ),
        "safe_rl_enabled": True,
        "safe_rl_process_risk_aggregation": str(
            _required(
                safe,
                "process_risk_aggregation",
                "checkpoint safe_rl config",
            )
        ),
        "safe_rl_lateness_normalizer": float(
            safe.get(
                "lateness_normalizer",
                300.0,
            )
        ),
        "safe_rl_lateness_clip": float(
            safe.get(
                "lateness_clip",
                5.0,
            )
        ),
        "safe_rl_delta_risk_weight": float(
            safe.get(
                "delta_risk_weight",
                0.5,
            )
        ),
        "safe_rl_violation_weight": float(
            safe.get(
                "violation_weight",
                1.0,
            )
        ),
        "safe_rl_lateness_weight": float(
            safe.get(
                "lateness_weight",
                1.0,
            )
        ),
        "safe_rl_shield_enabled": bool(
            _required(
                shield,
                "enabled",
                "checkpoint shield config",
            )
        ),
        "safe_rl_fallback_controller": str(
            _required(
                shield,
                "fallback_controller",
                "checkpoint shield config",
            )
        ),
        "safe_rl_state_enabled": bool(
            _required(
                state,
                "enabled",
                "checkpoint state config",
            )
        ),
        "safe_rl_state_high_uncertainty_threshold": float(
            _required(
                state,
                "high_uncertainty_threshold",
                "checkpoint state config",
            )
        ),
        "safe_rl_state_recent_record_window": int(
            _required(
                state,
                "recent_record_window",
                "checkpoint state config",
            )
        ),
        "manager_mode": "heuristic_selection_mode",
        "manager_heuristic_library_path": str(
            Path(
                library_path
            ).resolve()
        ),
        "manager_heuristic_llm_only": bool(
            manager.get(
                "llm_only",
                False,
            )
        ),
        "experiment_protocol_identity": context.identity(),
        "manager_heuristic_recent_window": int(
            _required(
                manager,
                "recent_window",
                "checkpoint manager config",
            )
        ),
        "scenario_code": scenario_id,
        "task_code": values["task_code"],
        "resource_code": values[
            "resource_code"
        ],
        "workflow_families": tuple(
            values[
                "workflow_families"
            ]
        ),
        "fuzzy_enabled": True,
        "fuzzy_energy_uncertainty_weight": float(
            _required(
                safe,
                "fuzzy_energy_uncertainty_weight",
                "checkpoint safe_rl config",
            )
        ),
        "fuzzy_deadline_eta": float(
            _required(
                safe,
                "fuzzy_deadline_eta",
                "checkpoint safe_rl config",
            )
        ),
        "fuzzy_use_deadline_constraint": True,

        # train_eval consumes these after constructing
        # each seed environment.
        "energy_reward_scale": float(
            config.get("energy_reward_scale", 1e-3)  # historical reward scale
        ),
        "task_baseline_norm": float(
            _required(
                config,
                "task_baseline_norm",
                "checkpoint config",
            )
        ),
        "energy_norm_per_mi_ref": float(
            _required(
                config,
                "energy_norm_per_mi_ref",
                "checkpoint config",
            )
        ),
        "alpha_delay_host": float(
            _required(
                config,
                "alpha_delay_host",
                "checkpoint config",
            )
        ),
        "alpha_delay_vm": float(
            _required(
                config,
                "alpha_delay_vm",
                "checkpoint config",
            )
        ),
    }


def _load_checkpoint_payload(
    path: str | os.PathLike[str],
    device: str,
) -> dict:
    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location=device,
        )

    if not isinstance(
        payload,
        Mapping,
    ):
        raise ValueError(
            f"agent checkpoint is not a mapping: {path}"
        )

    return dict(payload)


def _load_frozen_agent(
    path: str | os.PathLike[str],
    *,
    replay_metadata: Mapping[str, Any],
    device: str,
) -> D3QNAgent:
    payload = _load_checkpoint_payload(
        path,
        device,
    )

    input_dim = int(
        _required(
            payload,
            "input_dim",
            "agent checkpoint",
        )
    )

    output_dim = int(
        _required(
            payload,
            "output_dim",
            "agent checkpoint",
        )
    )

    if input_dim != int(
        _required(
            replay_metadata,
            "input_dim",
            "replay metadata",
        )
    ):
        raise ValueError(
            "agent checkpoint input dimension differs from manifest"
        )

    if output_dim != int(
        _required(
            replay_metadata,
            "action_dim",
            "replay metadata",
        )
    ):
        raise ValueError(
            "agent checkpoint action dimension differs from manifest"
        )

    safe_enabled = bool(
        _required(
            payload,
            "safe_rl_enabled",
            "agent checkpoint",
        )
    )

    if not safe_enabled:
        raise ValueError(
            "protocol best checkpoint must contain Safe-HRL values"
        )

    agent = D3QNAgent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(
            payload.get(
                "hidden_dims"
            )
            or (
                512,
                256,
            )
        ),
        head_hidden_dims=(
            tuple(
                payload[
                    "head_hidden_dims"
                ]
            )
            if payload.get(
                "head_hidden_dims"
            )
            is not None
            else None
        ),
        device=device,
        use_per=bool(
            payload.get(
                "use_per",
                False,
            )
        ),
        buffer_size=max(
            1,
            int(
                replay_metadata.get(
                    "capacity",
                    1,
                )
            ),
        ),
        observation_schema_version=str(
            _required(
                payload,
                "observation_schema_version",
                "agent checkpoint",
            )
        ),
        safe_rl_enabled=True,
        safety_discount=float(
            payload.get(
                "safety_discount",
                0.95,
            )
        ),
        safety_learning_rate=float(
            payload.get(
                "safety_learning_rate",
                3e-4,
            )
        ),
        safety_loss_weight=float(
            payload.get(
                "safety_loss_weight",
                1.0,
            )
        ),
        initial_lagrange_multiplier=float(
            payload.get(
                "lagrange_multiplier",
                1.0,
            )
        ),
        safe_replay_near_boundary_margin=float(
            payload.get(
                "safe_replay_near_boundary_margin",
                0.0,
            )
        ),
        safe_per_combined_priority=bool(
            payload.get(
                "safe_per_combined_priority",
                False,
            )
        ),
        safe_per_performance_td_weight=float(
            payload.get(
                "safe_per_performance_td_weight",
                1.0,
            )
        ),
        safe_per_safety_td_weight=float(
            payload.get(
                "safe_per_safety_td_weight",
                1.0,
            )
        ),
    )

    agent.load(
        str(
            Path(
                path
            ).resolve()
        )
    )

    for network_name in (
        "online",
        "target",
        "q_c_online",
        "q_c_target",
    ):
        network = getattr(
            agent,
            network_name,
            None,
        )

        if network is not None:
            network.eval()

            for parameter in network.parameters():
                parameter.requires_grad_(
                    False
                )

    return agent


def _agent_read_only_fingerprint(
    agent: Any,
) -> tuple[Any, ...]:
    versions = []

    for network_name in (
        "online",
        "target",
        "q_c_online",
        "q_c_target",
    ):
        network = getattr(
            agent,
            network_name,
            None,
        )

        if network is not None:
            versions.extend(
                (
                    network_name,
                    name,
                    int(
                        getattr(
                            parameter,
                            "_version",
                            -1,
                        )
                    ),
                )
                for name, parameter
                in network.named_parameters()
            )

    return (
        int(
            getattr(
                agent,
                "_updates",
                0,
            )
        ),
        int(
            getattr(
                agent,
                "_eps_steps",
                0,
            )
        ),
        int(
            getattr(
                agent,
                "_action_calls",
                0,
            )
        ),
        len(
            getattr(
                agent,
                "buffer",
                (),
            )
        ),
        tuple(versions),
    )


def evaluate_frozen_protocol_scenarios(
    *,
    context: ExperimentProtocolContext,
    config: Mapping[str, Any],
    library_path: str | os.PathLike[str],
    agents: Mapping[str, Any],
    test_seeds: Sequence[int] = DEFAULT_FINAL_TEST_SEEDS,
    deadline_cache_overrides: Mapping[str, str] | None = None,
    progress_output_dir: str | os.PathLike[str] | None = None,
    env_cls=CloudWorkflowEnv_VMAgents,
    evaluator=evaluate_hrl_three_layer_multi_seed,
) -> dict[str, Any]:
    """Evaluate every protocol test scenario without mutating agents."""

    if set(agents) != {
        "manager",
        "host",
        "vm",
    }:
        raise ValueError(
            "frozen evaluation requires manager/host/vm agents"
        )

    seeds = _normalize_test_seeds(
        test_seeds,
        context,
    )

    before = {
        layer: _agent_read_only_fingerprint(
            agent
        )
        for layer, agent
        in agents.items()
    }

    scenario_results: dict[
        str,
        Any,
    ] = {}

    # ============================================================
    # Important:
    # All scenarios must complete before the function returns.
    # ============================================================
    for scenario in context.test_scenarios:
        scenario = str(
            scenario
        )

        env_kwargs = build_frozen_scenario_env_kwargs(
            config,
            context,
            scenario,
            library_path,
            deadline_cache_overrides,
        )

        seed_callback = None

        if progress_output_dir is not None:
            progress_dir = Path(
                progress_output_dir
            ).resolve()

            # ----------------------------------------------------
            # Clean old incremental output for this scenario.
            #
            # This prevents duplicate rows when the same evaluation
            # command is intentionally re-run.
            # ----------------------------------------------------
            jsonl_path = (
                progress_dir
                / (
                    f"scenario_{scenario}"
                    "_seed_results.jsonl"
                )
            )

            csv_path = (
                progress_dir
                / (
                    f"scenario_{scenario}"
                    "_seed_results.csv"
                )
            )

            if jsonl_path.exists():
                jsonl_path.unlink()

            if csv_path.exists():
                csv_path.unlink()

            def seed_callback(
                seed,
                seed_result,
                *,
                _scenario=scenario,
                _progress_dir=progress_dir,
            ):
                _append_seed_result(
                    _progress_dir,
                    _scenario,
                    seed,
                    seed_result,
                )

        evaluator_kwargs = {
            "return_safety_metrics": True,
        }

        if seed_callback is not None:
            evaluator_kwargs[
                "seed_result_callback"
            ] = seed_callback

        result = evaluator(
            env_cls,
            env_kwargs,
            agents["vm"],
            agents["host"],
            agents["manager"],
            seeds,
            **evaluator_kwargs,
        )

        scenario_results[
            scenario
        ] = _json_value(
            result
        )

    # ============================================================
    # IMPORTANT:
    # This block must be OUTSIDE the scenario loop.
    # ============================================================
    after = {
        layer: _agent_read_only_fingerprint(
            agent
        )
        for layer, agent
        in agents.items()
    }

    changed = sorted(
        layer
        for layer in before
        if before[layer] != after[layer]
    )

    if changed:
        raise RuntimeError(
            "frozen evaluation modified agent state: "
            + ", ".join(changed)
        )

    return {
        "test_scenarios": list(
            context.test_scenarios
        ),
        "test_seeds": list(
            seeds
        ),
        "scenario_results": scenario_results,
        "agents_remained_frozen": True,
    }


def _resolve_checkpoint_path(
    context: ExperimentProtocolContext,
    checkpoint_manifest: str | os.PathLike[str] | None,
) -> Path:
    root = (
        context
        .checkpoint_root
        .resolve()
    )

    if checkpoint_manifest is None:
        matches = sorted(
            root.glob(
                "*/best_checkpoint_manifest.json"
            )
        )

        if len(matches) != 1:
            raise FileNotFoundError(
                "specify --checkpoint-manifest; "
                "expected exactly one protocol "
                f"best manifest under {root}, "
                f"found {len(matches)}"
            )

        source = matches[
            0
        ].resolve()

    else:
        source = Path(
            checkpoint_manifest
        ).resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"best checkpoint manifest not found: {source}"
        )

    try:
        source.relative_to(
            root
        )

    except ValueError as exc:
        raise ValueError(
            "checkpoint manifest must be inside protocol "
            f"root {root}"
        ) from exc

    return source


def run_frozen_protocol_evaluation(
    *,
    protocol: str = "single",
    source_scenario: str | None = "SS",
    resource_scale: str | None = None,
    checkpoint_manifest: str | os.PathLike[str] | None = None,
    heuristic_library: str | os.PathLike[str] | None = None,
    test_seeds: Sequence[int] = DEFAULT_FINAL_TEST_SEEDS,
    device: str = "cpu",
    deadline_cache_overrides: Mapping[str, str] | None = None,
) -> str:
    """Load one frozen protocol bundle and run read-only evaluation."""

    if (
        deadline_cache_overrides
        and str(
            protocol
        ).strip().lower()
        != "single"
    ):
        raise ValueError(
            "deadline cache mapping is supported "
            "only for protocol=single"
        )

    context = resolve_experiment_protocol(
        protocol,
        source_scenario=source_scenario,
        resource_scale=resource_scale,
    )

    deadline_cache_overrides = (
        validate_single_deadline_cache_paths(
            context.protocol,
            deadline_cache_overrides,
            source_scenario=(
                context.source_scenario
            ),
            required_scenarios=(
                context.test_scenarios
            ),
        )
    )

    identity = context.identity()

    source = _resolve_checkpoint_path(
        context,
        checkpoint_manifest,
    )

    manifest = read_best_checkpoint_manifest(
        source,
        expected_protocol_identity=identity,
    )

    library_version = _required(
        manifest,
        "heuristic_library_version",
        "checkpoint manifest",
    )

    expected_library_hash = str(
        _required(
            library_version,
            "manifest_sha256",
            "checkpoint heuristic library",
        )
    )

    recorded_library = str(
        library_version.get(
            "manifest_path"
        )
        or ""
    ).strip()

    selected_library = (
        heuristic_library
        or recorded_library
        or context.library_path
    )

    library_path = Path(
        selected_library
    ).expanduser().resolve()

    if not library_path.is_file():
        raise FileNotFoundError(
            "protocol heuristic library not found: "
            f"{library_path}. "
            "Restore the training artifact or pass "
            "--heuristic-library with its current location."
        )

    actual_library_hash = _sha256_file(
        library_path
    )

    if (
        actual_library_hash
        != expected_library_hash
    ):
        raise ValueError(
            "protocol heuristic library hash differs "
            "from the frozen checkpoint: "
            f"expected={expected_library_hash}, "
            f"actual={actual_library_hash}, "
            f"path={library_path}"
        )

    # Verify LLM library protocol identity before evaluation.
    load_manager_heuristic_library(
        library_path,
        expected_protocol_identity=identity,
    )

    replay_metadata = _required(
        manifest,
        "replay_metadata",
        "checkpoint manifest",
    )

    checkpoints = _required(
        manifest,
        "resolved_agent_checkpoints",
        "checkpoint manifest",
    )

    if (
        not isinstance(
            replay_metadata,
            Mapping,
        )
        or not isinstance(
            checkpoints,
            Mapping,
        )
    ):
        raise ValueError(
            "checkpoint replay metadata or layer paths are invalid"
        )

    agents = {
        layer: _load_frozen_agent(
            checkpoints[layer],
            replay_metadata=(
                replay_metadata[layer]
            ),
            device=device,
        )
        for layer in (
            "manager",
            "host",
            "vm",
        )
    }

    # ------------------------------------------------------------
    # Create output directory BEFORE evaluation.
    #
    # This is required so each finished seed can immediately
    # persist its result.
    # ------------------------------------------------------------
    output_dir = (
        context.artifact_output_root
        / "test"
        / source.parent.name
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = evaluate_frozen_protocol_scenarios(
        context=context,
        config=_snapshot_config(
            manifest
        ),
        library_path=library_path,
        agents=agents,
        test_seeds=test_seeds,
        deadline_cache_overrides=(
            deadline_cache_overrides
        ),

        # IMPORTANT:
        # Without this line seed_result_callback is never enabled.
        progress_output_dir=output_dir,
    )

    # ------------------------------------------------------------
    # Whole-scenario aggregate results.
    # These are written only after all requested seeds for each
    # scenario have completed.
    # ------------------------------------------------------------
    for (
        scenario,
        scenario_result,
    ) in result[
        "scenario_results"
    ].items():
        target = (
            output_dir
            / f"scenario_{scenario}.json"
        )

        target.write_text(
            _canonical_json(
                {
                    "protocol_identity": identity,
                    "scenario": scenario,
                    "test_seeds": result[
                        "test_seeds"
                    ],
                    "result": scenario_result,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    output_manifest = {
        "manifest_version": (
            FROZEN_EVALUATION_MANIFEST_VERSION
        ),
        "experiment_protocol": identity,
        **identity,
        "checkpoint_manifest": str(
            source
        ),
        "checkpoint_manifest_sha256": (
            _sha256_file(
                source
            )
        ),
        "optimizer_seed": int(
            manifest.get(
                "optimizer_seed",
                _required(
                    _required(
                        manifest,
                        "config_snapshot",
                        "checkpoint manifest",
                    )["config"],
                    "optimizer_seed",
                    "checkpoint config snapshot",
                ),
            )
        ),
        "deadline_cache_paths": dict(
            deadline_cache_overrides
        ),
        "heuristic_library": str(
            library_path
        ),
        "heuristic_library_sha256": (
            actual_library_hash
        ),
        "test_scenarios": result[
            "test_scenarios"
        ],
        "test_seeds": result[
            "test_seeds"
        ],
        "agents_remained_frozen": result[
            "agents_remained_frozen"
        ],
        "training_or_parameter_updates_performed": False,
        "scenario_result_files": {
            scenario: (
                f"scenario_{scenario}.json"
            )
            for scenario in result[
                "test_scenarios"
            ]
        },
    }

    output_manifest[
        "manifest_sha256"
    ] = hashlib.sha256(
        _canonical_json(
            output_manifest
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    target = (
        output_dir
        / "frozen_test_manifest.json"
    )

    temporary = (
        output_dir
        / "frozen_test_manifest.json.tmp"
    )

    temporary.write_text(
        _canonical_json(
            output_manifest
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        target,
    )

    return str(
        target
    )


def main(
    argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only generalization test "
            "of a frozen Safe-HRL bundle."
        )
    )

    parser.add_argument(
        "--protocol",
        choices=(
            "single",
            "multi",
        ),
        default="single",
    )

    parser.add_argument(
        "--source-scenario",
        default=None,
    )

    parser.add_argument(
        "--resource-scale",
        choices=(
            "S",
            "M",
            "L",
        ),
        default=None,
    )

    parser.add_argument(
        "--checkpoint-manifest",
        default=None,
    )

    parser.add_argument(
        "--heuristic-library",
        default=None,
        help=(
            "Exact heuristic library used to train "
            "the checkpoint. Defaults to "
            "heuristic_library_version.manifest_path "
            "recorded in the checkpoint manifest."
        ),
    )

    parser.add_argument(
        "--deadline-cache",
        action="append",
        default=None,
        help=(
            "Scenario cache override as SCENARIO=PATH; "
            "repeat per test scenario."
        ),
    )

    parser.add_argument(
        "--test-seeds",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_FINAL_TEST_SEEDS
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args(
        argv
    )

    if args.protocol == "single":
        source = (
            args.source_scenario
            or "SS"
        )

    else:
        if args.source_scenario is not None:
            parser.error(
                "--source-scenario is only valid "
                "for protocol=single"
            )

        source = None

    if (
        args.deadline_cache
        and args.protocol != "single"
    ):
        parser.error(
            "--deadline-cache mapping is supported "
            "only for protocol=single"
        )

    deadline_cache_overrides = (
        parse_deadline_cache_overrides(
            args.deadline_cache,
            default_scenario=source,
        )
    )

    manifest = run_frozen_protocol_evaluation(
        protocol=args.protocol,
        source_scenario=source,
        resource_scale=args.resource_scale,
        checkpoint_manifest=(
            args.checkpoint_manifest
        ),
        heuristic_library=(
            args.heuristic_library
        ),
        test_seeds=args.test_seeds,
        device=args.device,
        deadline_cache_overrides=(
            deadline_cache_overrides
        ),
    )

    print(
        "Frozen protocol evaluation manifest: "
        f"{manifest}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_FINAL_TEST_SEEDS",
    "build_frozen_scenario_env_kwargs",
    "evaluate_frozen_protocol_scenarios",
    "run_frozen_protocol_evaluation",
]