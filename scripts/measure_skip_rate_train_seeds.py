"""一次性测量：诊断扰动在 train seeds 上的"决策序列恒等"比例。

已落盘数据里的 37.1% 恒等率测于 confirm seeds [4,5]。T1 会把诊断扰动改到
train seeds [1,2]，本脚本验证该比例是否延续。

方法（不跑任何扰动的完整仿真）：
  1. 用 best 参数在一个 seed 上跑一次真实 episode，逐决策捕获 (ready_ids,
     features, selected_index) —— 这就是 T2 的基线轨迹。
  2. 对每个扰动的已冻结规则，在该轨迹上重算 scores 并比对 argmin ——
     这就是 T3 的重放门判定。
  3. 统计全程无分叉的比例。

成本：每结构每 seed 一次 episode + 约 23 次纯 numpy 重放。
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "algorithms" / "llm_safe_hrl" / "LLM"))

RUN = REPO / "out/main_single/SM/T/20260831_022241_926600_p19972/generated"
CONFIG = REPO / "algorithms/llm_safe_hrl/LLM/cfg/problem/cews_task_constructive.yaml"

FEATURES = (
    "min_exec_time",
    "min_comm_time",
    "min_incremental_energy",
    "slack",
    "upward_rank",
    "remaining_work",
    "ready_wait_time",
    "uncertainty",
)

_FLOAT = re.compile(r"-?\d+\.\d+(?:e[+-]?\d+)?")


def _load_eval_module():
    import importlib.util

    path = REPO / "algorithms/llm_safe_hrl/LLM/problems/cews_task_constructive/eval.py"
    spec = importlib.util.spec_from_file_location("cews_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literals(path: Path) -> set[str]:
    return set(_FLOAT.findall(path.read_text(encoding="utf-8")))


def _map_vector_to_file(files, values) -> Path | None:
    """扰动向量 -> 已冻结规则文件；要求唯一匹配，否则放弃该扰动。"""
    target = {repr(float(v)) for v in values}
    hits = [f for f in files if target <= _literals(f)]
    return hits[0] if len(hits) == 1 else None


def capture_trace(eval_module, config, seed, rule_path):
    """跑一次真实 episode，逐决策记录 ready 特征与被选中的下标。"""
    rule = eval_module.load_priority_function(rule_path)
    env = eval_module.build_environment(config, seed)
    env.reset()
    trace = []
    safety = config.get("safety", {})
    stalled = 0
    while not bool(env.done_flag):
        ready = env.get_ready_tasks()
        idle = env.idle_feasible_vm_ids(ready[0]) if ready else []
        if ready and idle:
            selected, details = env.select_task_with_priority_rule(
                ready, rule, return_details=True
            )
            trace.append(
                (
                    np.asarray(details["ready_task_ids"], dtype=np.int64),
                    {n: details["features"][n].astype(np.float64) for n in FEATURES},
                    int(details["selected_index"]),
                )
            )
            _host, vm, _d = env.select_host_then_vm_deterministic(
                selected, candidate_vm_ids=idle
            )
            env.assign_task(selected, vm)
            stalled = 0
        else:
            before = (float(env.current_time), int(env.completed_workflows))
            env.advance_to_next_event()
            stalled = 0 if (float(env.current_time), int(env.completed_workflows)) != before else stalled + 1
            if stalled > int(safety.get("max_stalled_steps", 10)):
                break
    return trace


def replay_identical(eval_module, trace, rule_path) -> bool:
    """T3 重放门判定：全部决策点 argmin 未变则恒等。"""
    rule = eval_module.load_priority_function(rule_path)
    for _ids, feats, selected in trace:
        scores = rule(*[feats[n] for n in FEATURES])
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(_ids),) or not np.all(np.isfinite(scores)):
            return False
        if int(np.argmin(scores)) != selected:
            return False
    return True


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seeds = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2"])]

    eval_module = _load_eval_module()
    config = eval_module.load_problem_config(str(CONFIG))
    config = eval_module._apply_eval_scenario_config(
        config, "SM", config.get("deadline_cache_paths", {})
    )

    searches = sorted(glob.glob(str(RUN / "optimization/*/*_search.json")))[:limit]
    total = identical = unmapped = 0

    for search_path in searches:
        data = json.loads(Path(search_path).read_text(encoding="utf-8"))
        perturbations = data.get("local_perturbations", [])
        best = data.get("best_parameters", {})
        if not perturbations or not best:
            continue
        structure = Path(search_path).parent.name
        files = sorted((RUN / "parameter_search" / structure).glob("params_*.py"))
        if not files:
            continue
        names = list(best)

        best_file = _map_vector_to_file(files, [best[n] for n in names])
        if best_file is None:
            continue

        mapped = []
        for item in perturbations:
            path = _map_vector_to_file(files, [item["parameters"][n] for n in names])
            if path is None:
                unmapped += 1
            else:
                mapped.append(path)
        if not mapped:
            continue

        for seed in seeds:
            trace = capture_trace(eval_module, config, seed, best_file)
            for path in mapped:
                total += 1
                identical += bool(replay_identical(eval_module, trace, path))
            print(
                f"  {structure} seed={seed} decisions={len(trace)} "
                f"perturbations={len(mapped)} running_identical={identical}/{total}",
                flush=True,
            )

    print()
    print(f"structures      : {len(searches)}")
    print(f"seeds           : {seeds}")
    print(f"unmapped vectors: {unmapped}")
    print(f"judgements      : {total}")
    if total:
        print(f"IDENTICAL RATE  : {identical}/{total} = {100*identical/total:.1f}%")
        print("(confirm seeds [4,5] 上的历史实测值为 37.1%)")


if __name__ == "__main__":
    main()
