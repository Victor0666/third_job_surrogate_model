"""Prove the parallel validation path is bit-identical on this machine.

SAFE_HRL_VALIDATION_PARALLEL_AUDIT=1 makes every parallel batch re-run
serially in the parent and assert exact equality, so a green run here is the
end-to-end evidence that fan-out cannot move a best-checkpoint argmax.

Everything runs under the ``__main__`` guard on purpose: the pool uses the
spawn start method, so each worker re-imports this module, and module-level
training calls would make every worker start its own run.
"""
import os
import sys
import time


def main() -> None:
    started = time.perf_counter()
    train_runner.train(
        protocol="single",
        source_scenario="SS",
        ddl="T",
        max_episodes=2,
        optimizer_seed=99987,
        validation_workers=3,
    )
    print(
        "AUDITED PARALLEL VALIDATION OK in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )


os.environ["SAFE_HRL_VALIDATION_PARALLEL_AUDIT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for _root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from hrl_mix import train_runner

if __name__ == "__main__":
    main()
