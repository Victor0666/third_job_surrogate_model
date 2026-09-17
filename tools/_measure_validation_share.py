"""一次性测量脚本：统计 Safe-HRL 训练里验证评估占的墙钟比例。

不进入仓库常规流程，只用来决定"验证并行化"值不值得做。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "algorithms" / "llm_safe_hrl")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from hrl_mix import train_eval, train_runner

_original = train_eval.evaluate_hrl_three_layer_multi_seed
STATS = {"calls": 0, "seconds": 0.0, "seed_episodes": 0}


def _timed(env_cls, env_kwargs, vm_agent, host_agent, manager_agent, seeds, **kwargs):
    started = time.perf_counter()
    result = _original(
        env_cls, env_kwargs, vm_agent, host_agent, manager_agent, seeds, **kwargs
    )
    STATS["calls"] += 1
    STATS["seconds"] += time.perf_counter() - started
    STATS["seed_episodes"] += len(tuple(seeds))
    return result


train_eval.evaluate_hrl_three_layer_multi_seed = _timed
train_runner.evaluate_hrl_three_layer_multi_seed = _timed

episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 50
wall_started = time.perf_counter()
train_runner.train(
    protocol="single",
    source_scenario="SS",
    ddl="T",
    max_episodes=episodes,
    optimizer_seed=99991,
    safe_rl_enabled=True,
    safe_rl_shield_enabled=True,
    safe_rl_state_enabled=True,
    safe_rl_dynamic_lambda_enabled=True,
)
wall = time.perf_counter() - wall_started

print("=" * 60)
print(f"episodes            = {episodes}")
print(f"total wall          = {wall:.1f} s")
print(f"validation calls    = {STATS['calls']}")
print(f"validation episodes = {STATS['seed_episodes']}")
print(f"validation wall     = {STATS['seconds']:.1f} s")
print(f"validation share    = {100.0 * STATS['seconds'] / wall:.1f} %")
if STATS["seed_episodes"]:
    print(
        "per eval episode    = "
        f"{STATS['seconds'] / STATS['seed_episodes']:.2f} s"
    )
train_episodes_seconds = wall - STATS["seconds"]
print(f"per train episode   = {train_episodes_seconds / episodes:.2f} s")
