"""Fast protocol tests for the read-only frozen Safe-HRL evaluator."""
from __future__ import annotations

from pathlib import Path

import pytest

from algorithms.llm_safe_hrl.scenario_registry import (
    resolve_experiment_protocol,
)
from hrl_mix.protocol_evaluation import (
    DEFAULT_FINAL_TEST_SEEDS,
    build_frozen_scenario_env_kwargs,
    evaluate_frozen_protocol_scenarios,
)
from hrl_mix import train as train_cli
from hrl_mix import train_runner


def _saved_config() -> dict:
    return {
        "horizon": 1e9,
        "arrival_lambda": 0.03,
        "max_ready_tasks": "auto",
        "normalize_obs": True,
        "workflows_per_episode": 2,
        "deadline_alpha_small": 2.0,
        "deadline_alpha_large": 3.0,
        "deadline_alpha_small_prob": 0.8,
        "manager_alpha_delay": 0.75,
        "manager_delay_mode": "tardiness",
        "energy_reward_scale": 1e-3,
        "task_baseline_norm": 300.0,
        "energy_norm_per_mi_ref": 20.0,
        "alpha_delay_host": 0.75,
        "alpha_delay_vm": 0.75,
        "safe_rl": {
            "enabled": True,
            "process_risk_aggregation": "mean",
            "fuzzy_energy_uncertainty_weight": 1.0,
            "fuzzy_deadline_eta": 0.95,
            "shield": {
                "enabled": True,
                "fallback_controller": "fixed_vm_rule",
            },
            "state": {
                "enabled": True,
                "high_uncertainty_threshold": 0.2,
                "recent_record_window": 100,
            },
            "manager_heuristics": {
                "mode": "heuristic_selection_mode",
                "recent_window": 20,
            },
        },
    }


class _FrozenAgent:
    def __init__(self):
        self._updates = 9
        self._eps_steps = 7
        self._action_calls = 3
        self.buffer = []

    def update(self):  # pragma: no cover - must never be called
        raise AssertionError("frozen protocol evaluation called update")


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("SS", ("SS", "MS", "LS")),
        ("SM", ("SM", "MM", "LM")),
        ("SL", ("SL", "ML", "LL")),
    ),
)
def test_single_frozen_evaluation_visits_only_generalization_group(
    source,
    expected,
):
    context = resolve_experiment_protocol(
        "single",
        source_scenario=source,
    )
    calls = []

    def evaluator(
        env_cls,
        env_kwargs,
        vm_agent,
        host_agent,
        manager_agent,
        seeds,
        *,
        return_safety_metrics,
    ):
        calls.append(
            (
                env_kwargs["scenario_code"],
                tuple(Path(path).name for path in env_kwargs["dax_paths"]),
                env_kwargs["num_cloud_hosts"],
                env_kwargs["deadline_cache_path"],
                tuple(seeds),
            )
        )
        assert return_safety_metrics is True
        return (0.0, 0.0, 0.0, 1.0, {"zero_violation_pass": True})

    result = evaluate_frozen_protocol_scenarios(
        context=context,
        config=_saved_config(),
        library_path="library.json",
        agents={
            "manager": _FrozenAgent(),
            "host": _FrozenAgent(),
            "vm": _FrozenAgent(),
        },
        evaluator=evaluator,
    )

    assert tuple(item[0] for item in calls) == expected
    assert tuple(result["scenario_results"]) == expected
    assert all(item[-1] == DEFAULT_FINAL_TEST_SEEDS for item in calls)
    assert result["agents_remained_frozen"] is True
    task_sets = tuple(set(item[1]) for item in calls)
    assert "CyberShake_30.xml" in task_sets[0]
    assert "CyberShake_50.xml" in task_sets[1]
    assert "CyberShake_100.xml" in task_sets[2]
    assert len({item[2] for item in calls}) == 1
    assert tuple(
        Path(item[3]).name for item in calls
    ) == tuple(
        f"fcfs_{size}Task_{source[1] == 'S' and 'small' or source[1] == 'M' and 'med' or 'large'}Res_exactmix_formal38.json"
        for size in ("small", "med", "large")
    )


def test_multi_frozen_evaluation_visits_all_training_scenarios():
    context = resolve_experiment_protocol(
        "multi",
        source_scenario=None,
        resource_scale="S",
    )
    seen = []

    def evaluator(*args, **kwargs):
        seen.append(args[1]["scenario_code"])
        return (0.0, 0.0, 0.0, 1.0, {})

    evaluate_frozen_protocol_scenarios(
        context=context,
        config=_saved_config(),
        library_path="library.json",
        agents={
            "manager": _FrozenAgent(),
            "host": _FrozenAgent(),
            "vm": _FrozenAgent(),
        },
        evaluator=evaluator,
    )

    assert seen == ["SS", "MS", "LS"]


def test_registry_inputs_change_task_resource_and_deadline():
    context = resolve_experiment_protocol(
        "single",
        source_scenario="SM",
    )
    small = build_frozen_scenario_env_kwargs(
        _saved_config(), context, "SM", "library.json"
    )
    large = build_frozen_scenario_env_kwargs(
        _saved_config(), context, "LM", "library.json"
    )

    assert small["dax_paths"] != large["dax_paths"]
    assert small["deadline_cache_path"] != large["deadline_cache_path"]
    assert small["num_cloud_hosts"] == large["num_cloud_hosts"] == 3
    assert small["cloud_vms_per_host"] == large["cloud_vms_per_host"]
    assert small["experiment_protocol_identity"] == context.identity()


def test_single_frozen_evaluation_uses_explicit_scenario_cache_mapping():
    context = resolve_experiment_protocol("single", source_scenario="SS")
    overrides = {
        scenario: f"cache_{scenario}.json"
        for scenario in context.test_scenarios
    }
    for scenario in context.test_scenarios:
        kwargs = build_frozen_scenario_env_kwargs(
            _saved_config(),
            context,
            scenario,
            "library.json",
            overrides,
        )
        assert Path(kwargs["deadline_cache_path"]).name == (
            f"cache_{scenario}.json"
        )


def test_safe_hrl_deadline_cache_cli_reaches_train_config(monkeypatch):
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(train_cli, "train", fake_train)
    train_cli.main(
        [
            "--deadline-cache", "SS=cache_SS.json",
            "--deadline-cache", "MS=cache_MS.json",
            "--deadline-cache", "LS=cache_LS.json",
        ]
    )
    assert Path(captured["deadline_cache_override"]).name == "cache_SS.json"
    assert set(captured["deadline_cache_paths"]) == {"SS", "MS", "LS"}

    class _StopBuild(Exception):
        pass

    def fake_build_train_config(**kwargs):
        captured.update(kwargs)
        raise _StopBuild

    monkeypatch.setattr(
        train_runner,
        "build_train_config",
        fake_build_train_config,
    )
    with pytest.raises(_StopBuild):
        train_runner.train(deadline_cache_override="custom_cache.json")
    assert captured["deadline_cache_override"] == "custom_cache.json"


def test_test_seed_overlap_is_rejected():
    context = resolve_experiment_protocol(
        "single",
        source_scenario="SS",
    )
    with pytest.raises(ValueError, match="overlap"):
        evaluate_frozen_protocol_scenarios(
            context=context,
            config=_saved_config(),
            library_path="library.json",
            agents={
                "manager": _FrozenAgent(),
                "host": _FrozenAgent(),
                "vm": _FrozenAgent(),
            },
            test_seeds=(1,),
            evaluator=lambda *args, **kwargs: None,
        )


def test_agent_training_state_mutation_is_rejected():
    context = resolve_experiment_protocol(
        "single",
        source_scenario="SS",
    )
    agents = {
        "manager": _FrozenAgent(),
        "host": _FrozenAgent(),
        "vm": _FrozenAgent(),
    }

    def mutating_evaluator(*args, **kwargs):
        agents["vm"]._updates += 1
        return (0.0, 0.0, 0.0, 1.0, {})

    with pytest.raises(RuntimeError, match="modified agent state"):
        evaluate_frozen_protocol_scenarios(
            context=context,
            config=_saved_config(),
            library_path="library.json",
            agents=agents,
            evaluator=mutating_evaluator,
        )
