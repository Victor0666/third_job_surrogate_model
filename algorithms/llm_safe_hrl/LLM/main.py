"""Hydra entry point for offline SeEvo rule evolution."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from protocol_config import (
    configure_seevo_protocol,
    parse_deadline_cache_overrides,
    update_seevo_run_manifest,
)
from algorithms.llm_safe_hrl.run_context import (
    resolve_deadline_setting,
    validate_execution_identifier,
)
from seevo import SeEvo
from utils.utils import init_client


_PROTOCOL_CLI_OPTIONS = {
    "--protocol": "protocol",
    "--source-scenario": "source_scenario",
    "--resource-scale": "resource_scale",
    "--ddl": "ddl",
    "--run-name": "run_name",
}
_LLM_ROOT = Path(__file__).resolve().parent
_INTERNAL_RUNTIME_KEYS = {
    "execution_id",
    "experiment_key",
    "runtime_output_root",
    "hydra.run.dir",
}


def _execution_id(run_name=None) -> str:
    label = None if run_name in (None, "", "null", "None") else str(run_name)
    if label is not None:
        validate_execution_identifier(label, "run_name")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    automatic = f"{timestamp}_p{os.getpid()}"
    return f"{label}_{automatic}" if label else automatic


def _experiment_key(protocol, source_scenario, resource_scale, ddl) -> str:
    deadline = resolve_deadline_setting(str(ddl))
    mode = str(protocol).strip().lower()
    if mode == "single":
        source = str(source_scenario).strip().upper()
        if source not in {"SS", "SM", "SL"}:
            raise ValueError("single source_scenario must be SS, SM, or SL")
        group = source
    elif mode == "multi":
        scale = str(resource_scale).strip().upper()
        if scale not in {"S", "M", "L"}:
            raise ValueError("multi resource_scale must be S, M, or L")
        group = f"MULTI_{scale}"
    else:
        raise ValueError("protocol must be 'single' or 'multi'")
    return f"{group}_{deadline.code}"


def _runtime_output_root(experiment_key, execution_id) -> str:
    experiment = validate_execution_identifier(
        str(experiment_key),
        "experiment_key",
    )
    execution = validate_execution_identifier(
        str(execution_id),
        "execution_id",
    )
    return str(
        (_LLM_ROOT / "outputs" / "formal" / experiment / execution).resolve()
    )


def _register_runtime_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "llm_execution_id",
        _execution_id,
        replace=True,
        use_cache=True,
    )
    OmegaConf.register_new_resolver(
        "llm_experiment_key",
        _experiment_key,
        replace=True,
        use_cache=True,
    )
    OmegaConf.register_new_resolver(
        "llm_runtime_output_root",
        _runtime_output_root,
        replace=True,
        use_cache=True,
    )


_register_runtime_resolvers()


def _case_numbers(value) -> list[int]:
    """Normalize Hydra list, scalar, or comma-separated seed values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [int(item) for item in value]
    if isinstance(value, str):
        return [
            int(item)
            for item in value.replace(",", " ").split()
            if item
        ]
    return [int(value)]


def _translate_protocol_cli_args(args: list[str]) -> list[str]:
    """Translate documented GNU-style flags into Hydra overrides."""
    translated: list[str] = []
    index = 0
    while index < len(args):
        argument = str(args[index])
        override_key = argument.split("=", 1)[0].lstrip("+")
        if override_key in _INTERNAL_RUNTIME_KEYS:
            raise ValueError(
                f"{override_key} is managed automatically; use --run-name "
                "only as an optional readable label"
            )
        if argument == "--deadline-cache" or argument.startswith(
            "--deadline-cache="
        ):
            if argument == "--deadline-cache":
                if index + 1 >= len(args):
                    raise ValueError(
                        "--deadline-cache requires SCENARIO=PATH"
                    )
                value = str(args[index + 1])
                index += 2
            else:
                value = argument.split("=", 1)[1]
                index += 1
            mapping = parse_deadline_cache_overrides([value])
            scenario, path = next(iter(mapping.items()))
            translated.append(
                f"+deadline_cache_paths.{scenario}={path}"
            )
            continue
        matched = False
        for option, hydra_key in _PROTOCOL_CLI_OPTIONS.items():
            if argument == option:
                if (
                    index + 1 >= len(args)
                    or str(args[index + 1]).startswith("--")
                ):
                    raise ValueError(f"{option} requires a value")
                translated.append(f"{hydra_key}={args[index + 1]}")
                index += 2
                matched = True
                break
            prefix = option + "="
            if argument.startswith(prefix):
                value = argument[len(prefix):]
                if not value:
                    raise ValueError(f"{option} requires a value")
                translated.append(f"{hydra_key}={value}")
                index += 1
                matched = True
                break
        if not matched:
            translated.append(argument)
            index += 1
    return translated


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    """Resolve the experiment protocol before any LLM or SeEvo work."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if str(cfg.algorithm).lower() != "seevo":
        raise ValueError(
            "This project snapshot provides the SeEvo implementation only; "
            f"got {cfg.algorithm!r}."
        )
    if str(cfg.mode).strip().lower() != "train":
        raise ValueError(
            "SeEvo is restricted to offline protocol training. "
            "Generalization testing must load frozen rules/checkpoints "
            "without LLM, CMA-ES, counterfactual feedback, or updates."
        )

    run_context = configure_seevo_protocol(
        cfg,
        llm_root=_LLM_ROOT,
    )
    protocol_context = run_context.protocol_context
    logging.info(
        "Experiment=%s execution_id=%s protocol=%s "
        "training_scenarios=%s test_scenarios=%s",
        run_context.experiment_key,
        run_context.execution_id,
        protocol_context.protocol,
        list(protocol_context.training_scenarios),
        list(protocol_context.test_scenarios),
    )
    logging.info(
        "Runtime output=%s artifact output=%s checkpoint output=%s",
        run_context.runtime_output_root,
        run_context.artifact_output_root,
        run_context.checkpoint_root,
    )

    try:
        # Configuration and protocol guards run before the first API call.
        init_client(cfg)
        algorithm = SeEvo(
            cfg,
            str(_LLM_ROOT),
            _case_numbers(cfg.case_num),
        )
        best_code, best_code_path = algorithm.evolve()
    except BaseException as exc:
        try:
            update_seevo_run_manifest(
                cfg,
                "FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            logging.exception("Could not update failed-run manifest")
        raise
    update_seevo_run_manifest(
        cfg,
        "COMPLETED",
        best_candidate_path=str(best_code_path),
    )
    logging.info("Best candidate path: %s", best_code_path)
    logging.info("Best candidate code:\n%s", best_code)


if __name__ == "__main__":
    sys.argv[1:] = _translate_protocol_cli_args(sys.argv[1:])
    main()
