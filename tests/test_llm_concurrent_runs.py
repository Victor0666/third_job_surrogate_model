from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

LLM_ROOT = (
    Path(__file__).resolve().parents[1]
    / "algorithms"
    / "llm_safe_hrl"
    / "LLM"
)
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from algorithms.llm_safe_hrl.LLM.main import (
    _experiment_key,
    _runtime_output_root,
    _translate_protocol_cli_args,
)
from algorithms.llm_safe_hrl.LLM.protocol_config import (
    configure_seevo_protocol,
    update_seevo_run_manifest,
)
from algorithms.llm_safe_hrl.run_context import (
    ExperimentRunContext,
    resolve_deadline_setting,
)
from algorithms.llm_safe_hrl.scenario_registry import (
    resolve_experiment_protocol,
)
from hrl_mix.train_config import (
    build_train_config,
    resolve_manager_heuristic_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("value", "code", "name", "probability"),
    (
        ("T", "T", "Tight", 0.8),
        ("Medium", "M", "Medium", 0.5),
        ("l", "L", "Loose", 0.2),
    ),
)
def test_deadline_setting_is_canonical(value, code, name, probability):
    setting = resolve_deadline_setting(value)
    assert setting.code == code
    assert setting.name == name
    assert setting.alpha_small_probability == probability


def test_run_context_isolates_ddl_and_execution_paths(tmp_path):
    protocol = resolve_experiment_protocol("single", source_scenario="SS")
    tight = ExperimentRunContext(
        protocol,
        resolve_deadline_setting("T"),
        "20260825_120000_000001_p1",
        tmp_path / "SS_T" / "20260825_120000_000001_p1",
    )
    loose = ExperimentRunContext(
        protocol,
        resolve_deadline_setting("L"),
        "20260825_120000_000002_p2",
        tmp_path / "SS_L" / "20260825_120000_000002_p2",
    )

    assert tight.experiment_key == "SS_T"
    assert loose.experiment_key == "SS_L"
    assert tight.artifact_output_root != loose.artifact_output_root
    assert tight.checkpoint_root != loose.checkpoint_root
    assert tight.library_path != loose.library_path
    assert tight.artifact_output_root.parts[-3:] == (
        "SS", "T", tight.execution_id
    )


def test_llm_cli_translates_condition_and_rejects_managed_paths():
    assert _translate_protocol_cli_args(
        [
            "--protocol", "single",
            "--source-scenario", "SS",
            "--ddl", "M",
            "--run-name", "paper",
        ]
    ) == [
        "protocol=single",
        "source_scenario=SS",
        "ddl=M",
        "run_name=paper",
    ]
    with pytest.raises(ValueError, match="managed automatically"):
        _translate_protocol_cli_args(["execution_id=a0"])
    with pytest.raises(ValueError, match="managed automatically"):
        _translate_protocol_cli_args(["hydra.run.dir=shared"])


def test_runtime_output_key_and_root_are_scenario_ddl_scoped():
    key = _experiment_key("single", "SS", None, "Loose")
    root = Path(_runtime_output_root(key, "20260825_120000_000003_p3"))
    assert key == "SS_L"
    assert root.parts[-3:-1] == ("formal", "SS_L")
    assert root.name == "20260825_120000_000003_p3"


@pytest.mark.parametrize(
    ("ddl", "probability"),
    (("T", 0.8), ("M", 0.5), ("L", 0.2)),
)
def test_safe_hrl_uses_same_deadline_mixture(ddl, probability):
    with patch("hrl_mix.train_config.os.makedirs"):
        config = build_train_config(
            "SS",
            ddl=ddl,
            require_deadline_cache=False,
        )
    assert config.deadline_alpha_small == 2.0
    assert config.deadline_alpha_large == 3.0
    assert config.deadline_alpha_small_prob == probability


def test_safe_hrl_resolves_only_completed_matching_llm_run(tmp_path):
    protocol = resolve_experiment_protocol("single", source_scenario="SS")
    library = tmp_path / "safe_heuristic_library_single_SS.json"
    library.write_text("{}\n", encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    payload = {
        "manifest_version": "llm_safe_hrl_run_v1",
        "status": "COMPLETED",
        "experiment_protocol": protocol.identity(),
        "deadline_setting": resolve_deadline_setting("M").identity(),
        "heuristic_library": str(library),
    }
    run_manifest.write_text(json.dumps(payload), encoding="utf-8")

    resolved = resolve_manager_heuristic_manifest(
        "S",
        protocol_context=protocol,
        llm_run_manifest=str(run_manifest),
        ddl="M",
    )
    assert Path(resolved) == library.resolve()

    with pytest.raises(ValueError, match="deadline setting"):
        resolve_manager_heuristic_manifest(
            "S",
            protocol_context=protocol,
            llm_run_manifest=str(run_manifest),
            ddl="T",
        )
    payload["status"] = "RUNNING"
    run_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="status=COMPLETED"):
        resolve_manager_heuristic_manifest(
            "S",
            protocol_context=protocol,
            llm_run_manifest=str(run_manifest),
            ddl="M",
        )


def test_protocol_configuration_routes_every_mutable_artifact_to_one_run(
    tmp_path,
):
    cfg = OmegaConf.load(LLM_ROOT / "cfg" / "config.yaml")
    cfg.problem = OmegaConf.load(
        LLM_ROOT / "cfg" / "problem" / "cews_task_constructive.yaml"
    )
    cfg.protocol = "single"
    cfg.source_scenario = "SS"
    cfg.resource_scale = None
    cfg.ddl = "M"
    cfg.run_name = None
    cfg.execution_id = "20260825_120000_000004_p4"
    cfg.experiment_key = "SS_M"
    cfg.runtime_output_root = str(tmp_path / "runtime" / cfg.execution_id)
    cfg.deadline_cache_paths = {
        scenario: str(
            PROJECT_ROOT
            / "data"
            / "deadlines"
            / "fcfs"
            / "exact_mix_v1"
            / file_name
        )
        for scenario, file_name in {
            "SS": "fcfs_smallTask_smallRes_exactmix_formal38.json",
            "MS": "fcfs_medTask_smallRes_exactmix_formal38.json",
            "LS": "fcfs_largeTask_smallRes_exactmix_formal38.json",
        }.items()
    }
    artifact_root = tmp_path / "out" / "main_single" / "SS" / "M" / cfg.execution_id
    checkpoint_root = (
        tmp_path / "checkpoints" / "main_single" / "SS" / "M" / cfg.execution_id
    )
    library_path = artifact_root / "safe_heuristic_library_single_SS.json"

    with (
        patch.object(
            ExperimentRunContext,
            "artifact_output_root",
            property(lambda _self: artifact_root),
        ),
        patch.object(
            ExperimentRunContext,
            "checkpoint_root",
            property(lambda _self: checkpoint_root),
        ),
        patch.object(
            ExperimentRunContext,
            "library_path",
            property(lambda _self: library_path),
        ),
    ):
        run = configure_seevo_protocol(cfg, llm_root=LLM_ROOT)
        assert run.deadline.name == "Medium"
        assert cfg.problem.dataset.deadline_alpha_small_prob == 0.5
        assert Path(cfg.parameter_optimization.cache_path).parent == artifact_root
        assert Path(cfg.counterfactual_feedback.cache.path).is_relative_to(
            artifact_root
        )
        assert Path(cfg.critical_state_replay.archive.archive_path).is_relative_to(
            artifact_root
        )
        assert Path(cfg.protocol_paths.library_path) == library_path
        assert Path(cfg.protocol_paths.checkpoint_root) == checkpoint_root
        runtime_manifest = Path(cfg.protocol_paths.run_manifest_path)
        assert json.loads(runtime_manifest.read_text(encoding="utf-8"))[
            "status"
        ] == "RUNNING"
        update_seevo_run_manifest(cfg, "COMPLETED")
        assert json.loads(runtime_manifest.read_text(encoding="utf-8"))[
            "status"
        ] == "COMPLETED"
