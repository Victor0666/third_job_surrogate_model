"""Run a small, deterministic Safe-HRL runtime benchmark."""

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = PROJECT_ROOT / "algorithms" / "llm_safe_hrl"
for import_root in (str(PROJECT_ROOT), str(ALGORITHM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from hrl_mix.train_config import build_train_config
import hrl_mix.train_runner as train_runner


def _small_agent_config(config):
    return replace(
        config,
        batch_size=2,
        buffer_size=64,
        eps_start=0.2,
        eps_end=0.2,
        eps_decay_steps=10,
        hidden_dims=(32, 16),
    )


def _run_once(
    scenario: str,
    ddl: str,
    workflows: int,
    validation_workers: int,
) -> float:
    safe_options = {
        "safe_rl_enabled": True,
        "safe_rl_shield_enabled": True,
        "safe_rl_state_enabled": True,
        "safe_rl_dynamic_lambda_enabled": True,
        "safe_rl_heuristic_manager_enabled": True,
    }
    with patch("hrl_mix.train_config.os.makedirs"):
        base = build_train_config(
            scenario,
            ddl,
            1,
            **safe_options,
        )

    with tempfile.TemporaryDirectory() as temporary:
        output_root = Path(temporary)
        config = replace(
            base,
            task_code="",
            workflows_per_episode=int(workflows),
            max_episodes=1,
            save_interval=10**9,
            eval_seeds=(1,),
            warmup_frac=0.0,
            hard_max_steps=1_000_000,
            save_dir=str(output_root / "checkpoints"),
            log_path=str(output_root / "logs" / "train.csv"),
            vm_agent=_small_agent_config(base.vm_agent),
            host_agent=_small_agent_config(base.host_agent),
            manager_agent=_small_agent_config(base.manager_agent),
        )
        Path(config.save_dir).mkdir(parents=True)
        Path(config.log_path).parent.mkdir(parents=True)

        started = time.perf_counter()
        with (
            patch.object(
                train_runner,
                "build_train_config",
                return_value=config,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train_runner.train(
                scenario,
                ddl,
                1,
                validation_workers=int(validation_workers),
                **safe_options,
            )
        return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="SS")
    parser.add_argument("--ddl", default="T")
    parser.add_argument("--workflows", type=int, default=10)
    parser.add_argument("--validation-workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.workflows <= 0 or args.repeats <= 0:
        parser.error("--workflows and --repeats must be positive")

    elapsed = [
        _run_once(
            args.scenario,
            args.ddl,
            args.workflows,
            args.validation_workers,
        )
        for _ in range(args.repeats)
    ]
    print("seconds=" + ",".join(f"{value:.6f}" for value in elapsed))
    print(f"median_seconds={statistics.median(elapsed):.6f}")


if __name__ == "__main__":
    main()
