from __future__ import annotations

from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest


LLM_ROOT = (
    Path(__file__).resolve().parents[1]
    / "algorithms"
    / "llm_safe_hrl"
    / "LLM"
)
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from algorithms.llm_safe_hrl.scenario_registry import (  # noqa: E402
    resolve_experiment_protocol,
)
from algorithms.llm_safe_hrl.LLM.main import (  # noqa: E402
    _translate_protocol_cli_args,
)
from seevo import _resolve_seevo_protocol_context  # noqa: E402


def _cfg(source: str = "SM"):
    context = resolve_experiment_protocol("single", source_scenario=source)
    return OmegaConf.create(
        {
            "experiment_protocol": context.identity(),
            "problem": {"experiment_protocol": context.identity()},
        }
    ), context


def test_seevo_revalidates_protocol_and_problem_identity():
    cfg, expected = _cfg("SM")
    actual = _resolve_seevo_protocol_context(cfg)
    assert actual.identity() == expected.identity()


def test_seevo_rejects_problem_protocol_mismatch():
    cfg, _ = _cfg("SM")
    cfg.problem.experiment_protocol = resolve_experiment_protocol(
        "single", source_scenario="SS"
    ).identity()
    with pytest.raises(ValueError, match="protocol identity mismatch"):
        _resolve_seevo_protocol_context(cfg)


def test_seevo_requires_materialized_protocol_identity():
    with pytest.raises(ValueError, match="cfg.experiment_protocol is required"):
        _resolve_seevo_protocol_context(OmegaConf.create({"problem": {}}))


def test_seevo_cli_translates_repeated_deadline_cache_mappings():
    translated = _translate_protocol_cli_args(
        [
            "--deadline-cache", "SS=cache_SS.json",
            "--deadline-cache", "MS=cache_MS.json",
            "--deadline-cache=LS=cache_LS.json",
        ]
    )
    assert translated == [
        "+deadline_cache_paths.SS=cache_SS.json",
        "+deadline_cache_paths.MS=cache_MS.json",
        "+deadline_cache_paths.LS=cache_LS.json",
    ]
