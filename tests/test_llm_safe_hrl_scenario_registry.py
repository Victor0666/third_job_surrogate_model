from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from algorithms.llm_safe_hrl.scenario_registry import (
    SCENARIO_REGISTRY,
    apply_scenario_to_problem_config,
    resolve_experiment_protocol,
    validate_component_scenarios,
    validate_protocol_identity,
)
from algorithms.llm_safe_hrl.LLM.protocol_config import (
    _bind_component_scenarios,
    apply_seevo_scenario_config,
)
from hrl_mix.train_config import validate_single_deadline_cache_paths
from omegaconf import OmegaConf


def _base_config():
    return {
        "dataset": {
            "scenario": "SS",
            "dax_files": ["wrong.xml"],
            "deadline_cache_path": "wrong.json",
        },
        "resources": {"num_cloud_hosts": 99, "num_edge_hosts": 99},
        "fuzzy": {"enabled": True},
    }


def test_registry_contains_all_real_combinations():
    assert tuple(SCENARIO_REGISTRY) == (
        "SS", "MS", "LS", "SM", "MM", "LM", "SL", "ML", "LL"
    )
    assert SCENARIO_REGISTRY["SS"].dax_files[0] == "CyberShake_30.xml"
    assert "CyberShake_50.xml" in SCENARIO_REGISTRY["MS"].dax_files
    assert "CyberShake_100.xml" in SCENARIO_REGISTRY["LS"].dax_files
    assert [SCENARIO_REGISTRY[item].num_cloud_hosts for item in ("SS", "SM", "SL")] == [2, 3, 5]


@pytest.mark.parametrize(
    ("source", "training", "testing"),
    [
        ("SS", ("SS",), ("SS", "MS", "LS")),
        ("SM", ("SM",), ("SM", "MM", "LM")),
        ("SL", ("SL",), ("SL", "ML", "LL")),
    ],
)
def test_single_protocol_is_source_only(source, training, testing):
    context = resolve_experiment_protocol("single", source_scenario=source)
    assert context.training_scenarios == training
    assert context.test_scenarios == testing
    assert validate_component_scenarios(context, "CMA-ES", training) == training
    with pytest.raises(ValueError, match="do not match"):
        validate_component_scenarios(context, "SeEvo", testing)


@pytest.mark.parametrize(
    ("scale", "scenarios"),
    [
        ("S", ("SS", "MS", "LS")),
        ("M", ("SM", "MM", "LM")),
        ("L", ("SL", "ML", "LL")),
    ],
)
def test_multi_protocol_uses_one_resource_group(scale, scenarios):
    context = resolve_experiment_protocol(
        "multi", source_scenario=None, resource_scale=scale
    )
    assert context.training_scenarios == scenarios
    assert context.test_scenarios == scenarios


def test_apply_scenario_switches_all_environment_inputs_without_mutation():
    original = _base_config()
    snapshot = deepcopy(original)
    configured = apply_scenario_to_problem_config(original, "LM")
    assert original == snapshot
    assert configured["dataset"]["scenario"] == "LM"
    assert "CyberShake_100.xml" in configured["dataset"]["dax_files"]
    assert configured["dataset"]["deadline_cache_path"].endswith(
        "fcfs_largeTask_medRes_exactmix_formal38.json"
    )
    assert configured["resources"]["num_cloud_hosts"] == 3
    assert configured["resources"]["num_edge_hosts"] == 3
    assert configured["resources"]["cloud_vms_per_host"] == [9, 9, 8]
    assert configured["fuzzy"] == {"enabled": True}


def test_seevo_explicit_cache_survives_registry_scenario_switch():
    configured = apply_seevo_scenario_config(
        _base_config(),
        "MS",
        {"MS": "diagnostic_MS.json"},
        require_files=False,
    )
    assert configured["dataset"]["deadline_cache_path"] == "diagnostic_MS.json"


def test_formal_single_requires_complete_cache_mapping():
    with pytest.raises(ValueError, match="MS, LS"):
        validate_single_deadline_cache_paths(
            "single",
            {"SS": "cache_SS.json"},
            source_scenario="SS",
        )


def test_all_registered_dax_inputs_exist_and_cache_paths_are_versioned():
    root = Path(__file__).resolve().parents[1]
    for scenario_id in SCENARIO_REGISTRY:
        configured = apply_scenario_to_problem_config(
            _base_config(), scenario_id, project_root=root, require_files=False
        )
        assert configured["dataset"]["scenario"] == scenario_id
        assert "exact_mix_v1" in configured["dataset"]["deadline_cache_path"]
        assert all(
            (root / "data" / "dax" / path).is_file()
            for path in configured["dataset"]["dax_files"]
        )


def test_artifact_paths_and_identity_are_protocol_isolated():
    single = resolve_experiment_protocol("single", source_scenario="SM")
    multi = resolve_experiment_protocol(
        "multi", source_scenario=None, resource_scale="M"
    )
    assert single.artifact_output_root.parts[-3:] == ("out", "main_single", "SM")
    assert multi.artifact_output_root.parts[-3:] == ("out", "enhancement_multi", "M")
    assert single.checkpoint_root.parts[-3:] == ("checkpoints", "main_single", "SM")
    assert multi.checkpoint_root.parts[-3:] == ("checkpoints", "enhancement_multi", "M")
    assert single.library_filename == "safe_heuristic_library_single_SM.json"
    assert multi.library_filename == "safe_heuristic_library_multi_M.json"
    assert validate_protocol_identity(single, single.identity()) == single.identity()
    with pytest.raises(ValueError, match="protocol identity mismatch"):
        validate_protocol_identity(single, multi.identity(), artifact_name="checkpoint")


def test_identity_rejects_seed_mismatch():
    context = resolve_experiment_protocol("single", source_scenario="SS")
    changed = context.identity()
    changed["llm_validation_seeds"] = [40, 50]
    with pytest.raises(ValueError, match="llm_validation_seeds"):
        validate_protocol_identity(context, changed, artifact_name="library")


def test_invalid_protocol_inputs_fail_closed():
    with pytest.raises(ValueError, match="SS, SM, or SL"):
        resolve_experiment_protocol("single", source_scenario="MS")
    with pytest.raises(ValueError, match="resource_scale"):
        resolve_experiment_protocol("multi", source_scenario=None, resource_scale="X")
    with pytest.raises(ValueError, match="disjoint"):
        resolve_experiment_protocol(
            "single", source_scenario="SS", train_seeds=(1, 2), validation_seeds=(2, 3)
        )


def test_empty_component_scenarios_inherit_protocol_domain():
    context = resolve_experiment_protocol("single", source_scenario="SM")
    component = OmegaConf.create({"scenario_ids": []})
    _bind_component_scenarios(context, "CMA-ES", component)
    assert list(component.scenario_ids) == ["SM"]


def test_explicit_component_scenario_conflict_is_not_overwritten():
    context = resolve_experiment_protocol("single", source_scenario="SM")
    component = OmegaConf.create({"scenario_ids": ["SS"]})
    with pytest.raises(ValueError, match="do not match"):
        _bind_component_scenarios(context, "Counterfactual", component)
    assert list(component.scenario_ids) == ["SS"]
