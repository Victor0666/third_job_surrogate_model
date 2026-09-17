"""Protocol and scenario isolation tests for LLM-SAFE-DRL experiments."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from algorithms.llm_safe_hrl.scenario_registry import (
    SCENARIO_REGISTRY,
    apply_scenario_to_problem_config,
    resolve_experiment_protocol,
    validate_component_scenarios,
    validate_protocol_identity,
)
from hrl_mix.train_config import (
    build_train_config,
    environment_scenario_values,
)
import hrl_mix.train as safe_hrl_cli
from hrl_mix.train_runner import (
    _evaluate_training_scenarios,
    _scenario_env_kwargs,
    _seed_for_training_episode,
    _validate_pipeline_protocol_seed_split,
    scenario_for_training_episode,
)
from algorithms.llm_safe_hrl.LLM.protocol_config import (
    apply_seevo_scenario_config,
)
from algorithms.llm_safe_hrl.LLM.problems.cews_task_constructive import (
    eval as seevo_eval,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("source", "training", "testing"),
    (
        ("SS", ("SS",), ("SS", "MS", "LS")),
        ("SM", ("SM",), ("SM", "MM", "LM")),
        ("SL", ("SL",), ("SL", "ML", "LL")),
    ),
)
def test_single_protocol_mapping(source, training, testing):
    context = resolve_experiment_protocol(
        "single",
        source_scenario=source,
    )

    assert context.protocol == "single"
    assert context.source_scenario == source
    assert context.resource_scale == source[1]
    assert context.training_scenarios == training
    assert context.test_scenarios == testing
    assert context.train_seeds == (1, 2, 3)
    assert context.validation_seeds == (4, 5)


@pytest.mark.parametrize(
    ("scale", "scenarios"),
    (
        ("S", ("SS", "MS", "LS")),
        ("M", ("SM", "MM", "LM")),
        ("L", ("SL", "ML", "LL")),
    ),
)
def test_multi_protocol_mapping(scale, scenarios):
    context = resolve_experiment_protocol(
        "multi",
        source_scenario=None,
        resource_scale=scale,
    )

    assert context.protocol == "multi"
    assert context.source_scenario is None
    assert context.resource_scale == scale
    assert context.training_scenarios == scenarios
    assert context.test_scenarios == scenarios


@pytest.mark.parametrize(
    "component",
    ("seevo", "cma_es", "counterfactual", "admission", "safe_hrl"),
)
def test_single_protocol_rejects_cross_scenario_training(component):
    context = resolve_experiment_protocol(
        "single",
        source_scenario="SM",
    )

    assert validate_component_scenarios(
        context,
        component,
        ("SM",),
    ) == ("SM",)
    with pytest.raises(ValueError, match="do not match protocol"):
        validate_component_scenarios(
            context,
            component,
            ("SM", "MM", "LM"),
        )
    with pytest.raises(ValueError, match="do not match protocol"):
        validate_component_scenarios(context, component, ("MM",))


def test_registry_materializes_task_resource_and_deadline_inputs():
    assert set(SCENARIO_REGISTRY) == {
        "SS", "MS", "LS", "SM", "MM", "LM", "SL", "ML", "LL",
    }
    base = {
        "dataset": {
            "scenario": "SS",
            "dax_files": ["old.xml"],
            "deadline_cache_path": "old.json",
        },
        "resources": {"num_cloud_hosts": 99},
    }

    small = apply_scenario_to_problem_config(
        base,
        "SS",
        project_root=PROJECT_ROOT,
        require_files=False,
    )
    medium = apply_scenario_to_problem_config(
        base,
        "MM",
        project_root=PROJECT_ROOT,
        require_files=False,
    )
    large = apply_scenario_to_problem_config(
        base,
        "LL",
        project_root=PROJECT_ROOT,
        require_files=False,
    )

    assert small["dataset"]["dax_files"] != medium["dataset"]["dax_files"]
    assert medium["dataset"]["dax_files"] != large["dataset"]["dax_files"]
    assert small["resources"]["num_cloud_hosts"] == 2
    assert medium["resources"]["num_cloud_hosts"] == 3
    assert large["resources"]["num_cloud_hosts"] == 5
    assert "smallTask_smallRes" in small["dataset"]["deadline_cache_path"]
    assert "medTask_medRes" in medium["dataset"]["deadline_cache_path"]
    assert "largeTask_largeRes" in large["dataset"]["deadline_cache_path"]
    assert base["dataset"]["dax_files"] == ["old.xml"]
    assert base["resources"] == {"num_cloud_hosts": 99}


def test_safe_hrl_environment_values_are_derived_from_registry():
    values = environment_scenario_values("LM")
    spec = SCENARIO_REGISTRY["LM"]

    assert values["scenario"] == "LM"
    assert values["task_code"] == "L"
    assert values["resource_code"] == "M"
    assert tuple(Path(path).name for path in values["dax_list"]) == spec.dax_files
    assert values["num_cloud_hosts"] == spec.num_cloud_hosts
    assert values["cloud_vms_per_host"] == spec.cloud_vms_per_host
    assert Path(values["deadline_cache_path"]) == spec.deadline_cache_path(
        PROJECT_ROOT
    )


def test_single_and_multi_artifact_namespaces_do_not_overlap():
    single = resolve_experiment_protocol(
        "single",
        source_scenario="SS",
    )
    multi = resolve_experiment_protocol(
        "multi",
        source_scenario=None,
        resource_scale="S",
    )

    assert single.artifact_output_root == (
        PROJECT_ROOT / "out" / "main_single" / "SS"
    )
    assert multi.artifact_output_root == (
        PROJECT_ROOT / "out" / "enhancement_multi" / "S"
    )
    assert single.checkpoint_root == (
        PROJECT_ROOT / "checkpoints" / "main_single" / "SS"
    )
    assert multi.checkpoint_root == (
        PROJECT_ROOT / "checkpoints" / "enhancement_multi" / "S"
    )
    assert single.library_filename == "safe_heuristic_library_single_SS.json"
    assert multi.library_filename == "safe_heuristic_library_multi_S.json"
    assert single.artifact_output_root != multi.artifact_output_root
    assert single.checkpoint_root != multi.checkpoint_root
    assert single.library_path != multi.library_path


def test_protocol_identity_rejects_cross_protocol_and_cross_source_artifacts():
    single_ss = resolve_experiment_protocol(
        "single",
        source_scenario="SS",
    )
    single_sm = resolve_experiment_protocol(
        "single",
        source_scenario="SM",
    )
    multi_s = resolve_experiment_protocol(
        "multi",
        source_scenario=None,
        resource_scale="S",
    )

    assert validate_protocol_identity(
        single_ss,
        single_ss.identity(),
        artifact_name="checkpoint",
    ) == single_ss.identity()
    with pytest.raises(ValueError, match="checkpoint protocol identity mismatch"):
        validate_protocol_identity(
            single_ss,
            multi_s.identity(),
            artifact_name="checkpoint",
        )
    with pytest.raises(ValueError, match="library protocol identity mismatch"):
        validate_protocol_identity(
            single_ss,
            single_sm.identity(),
            artifact_name="library",
        )


def test_protocol_identity_rejects_seed_or_missing_identity_changes():
    context = resolve_experiment_protocol(
        "single",
        source_scenario="SL",
    )
    changed = context.identity()
    changed["llm_validation_seeds"] = [9]
    with pytest.raises(ValueError, match="llm_validation_seeds"):
        validate_protocol_identity(context, changed)

    incomplete = context.identity()
    incomplete.pop("training_scenarios")
    with pytest.raises(ValueError, match="protocol identity is missing"):
        validate_protocol_identity(context, incomplete)


@pytest.mark.parametrize(
    ("source", "expected_tests", "resource_hosts"),
    (
        ("SS", ("SS", "MS", "LS"), 2),
        ("SM", ("SM", "MM", "LM"), 3),
        ("SL", ("SL", "ML", "LL"), 5),
    ),
)
def test_safe_hrl_single_config_uses_isolated_identity_and_paths(
    source,
    expected_tests,
    resource_hosts,
):
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            protocol="single",
            source_scenario=source,
            require_deadline_cache=False,
        )

    assert config.protocol == "single"
    assert config.experiment_protocol == {
        "protocol": "single",
        "source_scenario": source,
        "training_scenarios": [source],
        "test_scenarios": list(expected_tests),
        "llm_train_seeds": [1, 2, 3],
        "llm_validation_seeds": [4, 5],
        "safe_hrl_train_seeds": [1, 2, 3, 4, 5],
        "safe_hrl_validation_seeds": [101, 102, 103],
        "final_test_seeds": list(range(201, 231)),
    }
    assert config.source_scenario == source
    assert config.training_scenarios == (source,)
    assert config.test_scenarios == expected_tests
    assert config.train_seeds == (1, 2, 3, 4, 5)
    assert config.validation_seeds == (101, 102, 103)
    assert config.num_cloud_hosts == resource_hosts
    assert tuple(Path(path).name for path in config.dax_list) == (
        SCENARIO_REGISTRY[source].dax_files
    )
    assert Path(config.save_dir).is_relative_to(
        PROJECT_ROOT / "checkpoints" / "main_single" / source
    )
    assert Path(config.log_path).is_relative_to(
        PROJECT_ROOT / "out" / "main_single" / source
    )


def test_safe_hrl_multi_config_records_all_group_scenarios():
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            protocol="multi",
            resource_scale="S",
            require_deadline_cache=False,
        )

    assert config.protocol == "multi"
    assert config.experiment_protocol == {
        "protocol": "multi",
        "source_scenario": None,
        "training_scenarios": ["SS", "MS", "LS"],
        "test_scenarios": ["SS", "MS", "LS"],
        "llm_train_seeds": [1, 2, 3],
        "llm_validation_seeds": [4, 5],
        "safe_hrl_train_seeds": [1, 2, 3, 4, 5],
        "safe_hrl_validation_seeds": [101, 102, 103],
        "final_test_seeds": list(range(201, 231)),
    }
    assert config.source_scenario is None
    assert config.scenario == "SS"
    assert config.training_scenarios == ("SS", "MS", "LS")
    assert config.test_scenarios == ("SS", "MS", "LS")
    assert Path(config.save_dir).is_relative_to(
        PROJECT_ROOT / "checkpoints" / "enhancement_multi" / "S"
    )
    assert Path(config.log_path).is_relative_to(
        PROJECT_ROOT / "out" / "enhancement_multi" / "S"
    )


def test_safe_hrl_protocol_seeds_drive_training_and_validation():
    with patch("hrl_mix.train_config.os.makedirs"):
        single = build_train_config(
            protocol="single",
            source_scenario="SM",
            require_deadline_cache=False,
        )
        multi = build_train_config(
            protocol="multi",
            resource_scale="S",
            require_deadline_cache=False,
        )

    for config in (single, multi):
        assert config.random_seed == config.train_seeds[0] == 1
        assert config.eval_seeds == config.validation_seeds == (101, 102, 103)


def test_safe_hrl_multi_episode_rotation_covers_scenario_seed_product():
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            protocol="multi",
            resource_scale="S",
            require_deadline_cache=False,
        )

    sequence = [
        (
            scenario_for_training_episode(config, index),
            _seed_for_training_episode(config, index),
        )
        for index in range(9)
    ]
    assert sequence == [
        ("SS", 1), ("MS", 1), ("LS", 1),
        ("SS", 2), ("MS", 2), ("LS", 2),
        ("SS", 3), ("MS", 3), ("LS", 3),
    ]


def test_safe_hrl_single_episode_rotation_never_leaves_source():
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            protocol="single",
            source_scenario="SL",
            require_deadline_cache=False,
        )

    assert [
        scenario_for_training_episode(config, index)
        for index in range(6)
    ] == ["SL"] * 6
    assert [
        _seed_for_training_episode(config, index)
        for index in range(6)
    ] == [1, 2, 3, 4, 5, 1]


def test_safe_hrl_episode_kwargs_switch_real_scenario_inputs():
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            protocol="multi",
            resource_scale="S",
            require_deadline_cache=False,
        )
    base = {"sentinel": object(), "random_seed": -1}

    ss = _scenario_env_kwargs(base, config, "SS", 1)
    ms = _scenario_env_kwargs(base, config, "MS", 2)
    ls = _scenario_env_kwargs(base, config, "LS", 3)

    assert base["random_seed"] == -1
    assert [value["scenario_code"] for value in (ss, ms, ls)] == [
        "SS", "MS", "LS"
    ]
    assert [value["task_code"] for value in (ss, ms, ls)] == [
        "S", "M", "L"
    ]
    assert [tuple(Path(path).name for path in value["dax_paths"]) for value in (ss, ms, ls)] == [
        SCENARIO_REGISTRY[scenario].dax_files
        for scenario in ("SS", "MS", "LS")
    ]
    assert len({value["deadline_cache_path"] for value in (ss, ms, ls)}) == 3
    assert [value["random_seed"] for value in (ss, ms, ls)] == [1, 2, 3]
    assert ss["num_cloud_hosts"] == ms["num_cloud_hosts"] == ls["num_cloud_hosts"]

    with pytest.raises(ValueError, match="outside training_scenarios"):
        _scenario_env_kwargs(base, config, "SM", 1)


def test_safe_hrl_episode_kwargs_preserve_explicit_deadline_cache():
    cache_path = (
        PROJECT_ROOT
        / "data"
        / "deadlines"
        / "fcfs"
        / "diagnostic"
        / "fcfs_fixed_SS_exactmix_formal38.json"
    ).resolve()
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            "SS",
            deadline_cache_paths={"SS": str(cache_path)},
            require_deadline_cache=False,
        )
    values = _scenario_env_kwargs({}, config, "SS", 1)
    assert Path(values["deadline_cache_path"]) == cache_path


def test_train_config_uses_scenario_cache_mapping_as_active_source():
    cache_root = (
        PROJECT_ROOT
        / "data"
        / "deadlines"
        / "fcfs"
        / "diagnostic"
    ).resolve()
    cache_paths = {
        scenario: (
            cache_root
            / f"fcfs_fixed_{scenario}_exactmix_formal38.json"
        )
        for scenario in ("SS", "MS", "LS")
    }
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            protocol="single",
            source_scenario="SS",
            deadline_cache_paths={
                scenario: str(path)
                for scenario, path in cache_paths.items()
            },
        )

    assert Path(config.deadline_cache_path) == cache_paths["SS"]
    assert {
        scenario: Path(path)
        for scenario, path in config.deadline_cache_paths.items()
    } == cache_paths


def test_train_config_reports_missing_explicit_cache_by_scenario():
    missing_cache = PROJECT_ROOT / ".missing_new_cache_SS.json"
    assert not missing_cache.exists()
    with pytest.raises(
        FileNotFoundError,
        match=r"explicit deadline cache file\(s\) not found: SS=",
    ):
        build_train_config(
            "SS",
            deadline_cache_paths={"SS": str(missing_cache)},
        )


def test_seevo_eval_uses_diagnostic_or_explicit_deadline_cache(monkeypatch):
    captured = {}

    class FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(seevo_eval, "HrlFcfsCacheEnv", FakeEnv)
    config = seevo_eval.load_problem_config()
    seevo_eval.build_environment(config, 1)
    assert Path(captured["deadline_cache_path"]).name == (
        "fcfs_fixed_SS_exactmix_formal38.json"
    )

    explicit = deepcopy(config)
    explicit["deadline_cache_paths"] = {"SS": "custom.json"}
    explicit = apply_seevo_scenario_config(
        explicit,
        "SS",
        explicit["deadline_cache_paths"],
        require_files=False,
    )
    captured.clear()
    seevo_eval.build_environment(explicit, 1)
    assert Path(captured["deadline_cache_path"]).name == "custom.json"


def test_safe_hrl_multi_validation_calls_every_training_scenario():
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            protocol="multi",
            resource_scale="S",
            require_deadline_cache=False,
        )
    calls = []

    def fake_evaluate(_env_cls, kwargs, *_agents, seeds, **_options):
        scenario = kwargs["scenario_code"]
        calls.append((scenario, tuple(seeds)))
        ordinal = {"SS": 1.0, "MS": 2.0, "LS": 3.0}[scenario]
        records = []
        for seed in seeds:
            records.append({
                "completed_workflow_count": 1,
                "expected_workflow_count": 1,
                "deadline_violation_count": 0,
                "fuzzy_ddl_violation_rate": 0.0,
                "feasible_workflow_count": 1,
                "feasible_workflow_ratio": 1.0,
                "fuzzy_lateness_sum": 0.0,
                "mean_fuzzy_lateness": 0.0,
                "max_fuzzy_lateness": 0.0,
                "minimum_fuzzy_safety_margin": 1.0,
                "evaluation_completed": True,
                "episode_feasible": True,
                "exact_fuzzy_timeline_reconstruction": True,
                "shield_record_count": 0,
                "shield_intervention_count": 0,
                "no_safe_action_count": 0,
                "fallback_count": 0,
                "proposed_executed_action_mismatch_count": 0,
                "heuristic_selection_count": 0,
                "selected_llm_heuristic_count": 0,
                "llm_associated_shield_record_count": 0,
                "llm_associated_shield_intervention_count": 0,
                "fuzzy_energy_mean": ordinal,
                "fuzzy_energy_std": 0.0,
                "fuzzy_energy_score": ordinal,
                "modal_energy": ordinal,
                "scheduling_time_seconds": 0.0,
                "safety_cost": 0.0,
                "seed": seed,
            })
        return (ordinal,) * 4 + ({
            "per_seed_metrics": records,
            "evaluation_violation_budget": 0.0,
        },)

    with patch(
        "hrl_mix.train_runner.evaluate_hrl_three_layer_multi_seed",
        side_effect=fake_evaluate,
    ):
        result = _evaluate_training_scenarios(
            object,
            base_env_kwargs={},
            active_env_kwargs={},
            cfg=config,
            controller=None,
            vm_agent=object(),
            host_agent=object(),
            manager_agent=object(),
            seeds=config.validation_seeds,
            return_safety_metrics=True,
        )

    assert calls == [
        ("SS", (101, 102, 103)), ("MS", (101, 102, 103)), ("LS", (101, 102, 103))
    ]
    assert result[:4] == (2.0, 2.0, 2.0, 2.0)
    assert result[4]["evaluation_scenarios"] == ["SS", "MS", "LS"]
    assert result[4]["evaluation_seed_count"] == 9
    assert result[4]["all_seed_feasible"] is True


def test_pipeline_seed_split_cannot_bypass_protocol_identity():
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            protocol="single",
            source_scenario="SS",
            require_deadline_cache=False,
        )

    class Split:
        training = (1, 2, 3, 4, 5)
        validation = (101, 102, 103)

    class Plan:
        seed_split = Split()

    _validate_pipeline_protocol_seed_split(config, Plan())
    Plan.seed_split.training = (1, 2)
    with pytest.raises(ValueError, match="training seeds do not match"):
        _validate_pipeline_protocol_seed_split(config, Plan())
    Plan.seed_split.training = (1, 2, 3, 4, 5)
    Plan.seed_split.validation = (4, 5)
    with pytest.raises(ValueError, match="validation seeds do not match"):
        _validate_pipeline_protocol_seed_split(config, Plan())


def test_legacy_python_train_config_call_remains_compatible():
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config("MM", require_deadline_cache=False)

    assert config.protocol == "legacy"
    assert config.experiment_protocol is None
    assert config.scenario == "MM"
    assert config.training_scenarios == ("MM",)
    assert "out\\ckpts" in str(config.save_dir).replace("/", "\\")


def test_safe_hrl_cli_forwards_single_and_multi_protocol_switches():
    cache_root = (
        PROJECT_ROOT / "data" / "deadlines" / "fcfs" / "diagnostic"
    )
    single_caches = {
        scenario: cache_root / f"fcfs_fixed_{scenario}_exactmix_formal38.json"
        for scenario in ("SM", "MM", "LM")
    }
    with patch.object(safe_hrl_cli, "train") as mocked_train:
        safe_hrl_cli.main(
            [
                "--protocol", "single", "--source-scenario", "SM",
                *[
                    value
                    for scenario, path in single_caches.items()
                    for value in (
                        "--deadline-cache", f"{scenario}={path}"
                    )
                ],
            ]
        )
    assert mocked_train.call_args.kwargs["protocol"] == "single"
    assert mocked_train.call_args.kwargs["source_scenario"] == "SM"
    assert mocked_train.call_args.kwargs["resource_scale"] is None
    assert mocked_train.call_args.kwargs["deadline_cache_paths"] == {
        scenario: str(path.resolve())
        for scenario, path in single_caches.items()
    }

    with patch.object(safe_hrl_cli, "train") as mocked_train:
        safe_hrl_cli.main(
            ["--protocol", "multi", "--resource-scale", "L"]
        )
    assert mocked_train.call_args.kwargs["protocol"] == "multi"
    assert mocked_train.call_args.kwargs["source_scenario"] is None
    assert mocked_train.call_args.kwargs["resource_scale"] == "L"


def test_safe_hrl_protocol_rejects_ambiguous_or_cross_domain_inputs():
    with pytest.raises(ValueError, match="conflicts"):
        build_train_config(
            scenario="SS",
            protocol="single",
            source_scenario="SM",
        )
    with pytest.raises(ValueError, match="does not accept"):
        build_train_config(
            scenario="SS",
            protocol="multi",
            resource_scale="S",
        )
    with pytest.raises(ValueError, match="require an explicit protocol"):
        build_train_config(source_scenario="SS")


def test_manager_online_modules_do_not_import_offline_evolution_engines():
    manager_files = (
        PROJECT_ROOT / "algorithms" / "llm_safe_hrl" / "base" / "manager_heuristics.py",
        PROJECT_ROOT / "algorithms" / "llm_safe_hrl" / "base" / "hrl_env.py",
    )
    # Frozen-rule parsing/validation is intentionally allowed. The online
    # Manager must not import any optimizer or offline analysis executor.
    forbidden = (
        "LLM.seevo",
        "cmaes_optimizer",
        "counterfactual_feedback",
        "critical_state_replay",
    )

    for path in manager_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any(
            token in module
            for module in imported
            for token in forbidden
        ), f"{path.name} imports an offline evolution engine: {imported}"
