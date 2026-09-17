"""Short, stable names for artifacts written below ``out/``.

Detailed experimental configuration belongs in config snapshots and
manifests, not in a filesystem component.  The helpers here keep names short
while preserving the method, scenario, deadline setting, seed, and a stable
configuration digest where needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


OUTPUT_NAMING_SCHEMA_VERSION = 1
MAX_OUTPUT_COMPONENT_LENGTH = 48

_SCENARIO_PATTERN = re.compile(r"^[SML]{2}$")
_EVAL_SCRIPT_PATTERN = re.compile(
    r"^eval_([SML]{2})_([TML])\.py$",
    re.IGNORECASE,
)
_DDL_CODES = {
    "T": "t",
    "TIGHT": "t",
    "M": "m",
    "MEDIUM": "m",
    "L": "l",
    "LOOSE": "l",
}
_SIZE_NAMES = {
    "S": "small",
    "M": "med",
    "L": "large",
}
_DDL_NAMES = {
    "t": "Tight",
    "m": "Medium",
    "l": "Loose",
}


@dataclass(frozen=True)
class TrainingOutputPaths:
    """Checkpoint directory and primary training log."""

    checkpoint_dir: Path
    log_path: Path


def _scenario_code(scenario: str) -> str:
    value = str(scenario).strip().upper()
    if _SCENARIO_PATTERN.fullmatch(value) is None:
        raise ValueError("scenario must be a two-letter S/M/L code")
    return value.lower()


def _ddl_code(ddl: str) -> str:
    value = str(ddl).strip().upper()
    if value not in _DDL_CODES:
        raise ValueError("ddl must be T/M/L or Tight/Medium/Loose")
    return _DDL_CODES[value]


def stable_digest(
    payload: Mapping[str, Any] | str,
    *,
    length: int = 10,
) -> str:
    """Return a deterministic lowercase hexadecimal configuration digest."""

    if not 8 <= int(length) <= 16:
        raise ValueError("digest length must be between 8 and 16")
    if isinstance(payload, str):
        serialized = payload
    else:
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def legacy_hrl_run_id(
    scenario: str,
    ddl: str,
    *,
    seed: int = 1,
) -> str:
    """Short ID for the original three-layer HRL configuration."""

    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    return (
        f"hrl-{_scenario_code(scenario)}-{_ddl_code(ddl)}-s{int(seed)}"
    )


def safe_hrl_run_id(
    scenario: str,
    ddl: str,
    config_payload: Mapping[str, Any],
) -> str:
    """Short ID for a safe-HRL configuration."""

    return (
        f"safe-{_scenario_code(scenario)}-{_ddl_code(ddl)}-"
        f"{stable_digest(config_payload)}"
    )


def pipeline_run_id(
    scenario: str,
    ddl: str,
    plan_hash: str,
) -> str:
    """Short ID for an explicitly configured training pipeline."""

    return (
        f"pipe-{_scenario_code(scenario)}-{_ddl_code(ddl)}-"
        f"{stable_digest(str(plan_hash))}"
    )


def training_output_paths(
    project_root: str | Path,
    run_id: str,
) -> TrainingOutputPaths:
    """Build the compact checkpoint and log paths for one run."""

    component = str(run_id).strip()
    if (
        not component
        or len(component) > MAX_OUTPUT_COMPONENT_LENGTH
        or re.fullmatch(r"[a-z0-9-]+", component) is None
    ):
        raise ValueError(
            "run_id must contain only lowercase letters, digits, hyphens "
            f"and be at most {MAX_OUTPUT_COMPONENT_LENGTH} characters"
        )
    root = Path(project_root).resolve()
    return TrainingOutputPaths(
        checkpoint_dir=root / "out" / "ckpts" / component,
        log_path=root / "out" / "logs" / component / "train.csv",
    )


def eval_identity_from_filename(
    script_path: str | Path,
) -> tuple[str, str]:
    """Return normalized scenario/DDL codes from ``eval_SS_T.py``."""

    match = _EVAL_SCRIPT_PATTERN.fullmatch(Path(script_path).name)
    if match is None:
        raise ValueError(
            "evaluation script must be named eval_<scenario>_<ddl>.py"
        )
    return match.group(1).lower(), match.group(2).lower()


def hrl_evaluation_csv_path(
    project_root: str | Path,
    scenario: str,
    ddl: str,
) -> Path:
    """Compact CSV location for a multi-seed HRL evaluation."""

    scenario_code = _scenario_code(scenario)
    ddl_code = _ddl_code(ddl)
    return (
        Path(project_root).resolve()
        / "out"
        / "eval"
        / "hrl"
        / f"{scenario_code}-{ddl_code}.csv"
    )


def _legacy_hrl_checkpoint_dir_name(
    scenario: str,
    ddl: str,
    *,
    seed: int,
) -> str:
    """Historical long directory name, used only for read compatibility."""

    scenario_code = _scenario_code(scenario).upper()
    ddl_code = _ddl_code(ddl)
    task_name = _SIZE_NAMES[scenario_code[0]]
    resource_name = _SIZE_NAMES[scenario_code[1]]
    return (
        "ckpts_hrl_3layer_routeA_mgc_ave_"
        f"{task_name}Task_{resource_name}Res_seed{int(seed)}_"
        "mgrDelayEnergy_mix_rAlpha075_dalphaMix_HVrn_"
        f"{_DDL_NAMES[ddl_code]}"
    )


def resolve_hrl_checkpoint_dir(
    project_root: str | Path,
    scenario: str,
    ddl: str,
    *,
    seed: int = 1,
) -> Path:
    """Resolve a new short checkpoint directory, then its exact legacy path.

    The function deliberately does not scan and select an arbitrary directory:
    a checkpoint from another scenario must never be loaded silently.
    """

    root = Path(project_root).resolve()
    checkpoint_root = root / "out" / "ckpts"
    candidates = (
        checkpoint_root
        / legacy_hrl_run_id(scenario, ddl, seed=seed),
        checkpoint_root
        / _legacy_hrl_checkpoint_dir_name(
            scenario,
            ddl,
            seed=seed,
        ),
    )
    required = ("best_vm.pth", "best_host.pth", "best_manager.pth")
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in required):
            return candidate
    formatted = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "No complete HRL checkpoint set found. Checked:\n"
        f"{formatted}"
    )

