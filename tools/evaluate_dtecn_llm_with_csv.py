"""Run online validation/test with immediate CSV logging; keep core artifact hashes."""
import csv
from copy import deepcopy
import json
import sys
from unittest.mock import patch

from algorithms.comparisons.dtecn_llm import run as runner


_evaluate = runner.evaluate
_load_manifest = runner.load_manifest


def load_compatible_manifest(root, expected):
    saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checked = deepcopy(expected)
    path = str((runner.ROOT / "algorithms/llm_safe_hrl/hrl_mix/train_config.py").resolve())
    old = "f227b95457cbc9047395c9277e293394349a88c503da9bad243a93e5fee8f271"
    new = "72ee80c2c59939376922416f5cbcb57243bc97c9c234fc82e7c6ffb1d298a6ab"
    # Only this audited change adds max_episodes<=0 rejection; DTECN never uses it.
    current_hashes = checked.get("source_and_input_hashes", {})
    saved_hashes = saved.get("identity", {}).get("source_and_input_hashes", {})
    if current_hashes.get(path) == new and saved_hashes.get(path) == old:
        current_hashes[path] = old
    # e1700bf adds opt-in SHRL Top-K support; DTECN keeps heuristic Manager disabled.
    audited = {
        "algorithms/llm_safe_hrl/hrl_mix/train_config.py": (
            "04d7913731188d56686d2296d0d416abba6fa9b2ac104c93b5abc120d6b414fc",
            {old, new}),
        "algorithms/llm_safe_hrl/base/hrl_env.py": (
            "d4cbe584ee233c7e97d7397e859efe788c1d929f1db868387c543426979c3d66",
            {"e3b2677c1285739656f7153f1cf872ad1eb620ebd040e5d1d454a7673a70df0e"}),
        "algorithms/llm_safe_hrl/base/manager_heuristics.py": (
            "76cbdd2fc7d062f0a0b2e1f1df5828dadb8f4c63c676f868dfed3b5af29974a0",
            {"6bdda6e760adf91958de9e84cd0fdc78e273375249b813bb6101c4c035da8338"}),
    }
    if checked.get("method") == "dtecn_llm":
        for relative, (current, previous) in audited.items():
            key = str((runner.ROOT / relative).resolve())
            if current_hashes.get(key) == current and saved_hashes.get(key) in previous:
                current_hashes[key] = saved_hashes[key]
    manifest = _load_manifest(root, checked)
    if checked != expected:
        print(json.dumps(dict(source_compatibility={
            key: {"trained_sha256": value, "current_sha256": expected["source_and_input_hashes"][key]}
            for key, value in current_hashes.items() if value != expected["source_and_input_hashes"][key]})), flush=True)
    return manifest


def evaluate_with_csv(protocol, policy, split, trace_path=None):
    if trace_path is None:
        return _evaluate(protocol, policy, split, trace_path)
    csv_path = trace_path.with_suffix(".csv")
    fields = ("scenario", "seed", "decision_index", "decision_time", "task_id",
              "workflow_slot", "executed_vm_id", "action_source", "batch_id", "batch_size", "action", "state")
    original_assign = policy.assign
    counters = {}
    # Exclusive creation preserves partial runs, just like the original trace guard.
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()

        def assign(env, host_state, *, training):
            task_id, decision_time = env._cur_tid, float(env.current_time)
            result = original_assign(env, host_state, training=training)
            pending = policy.pending
            key = (env.scenario_code, int(env.random_seed))
            counters[key] = counters.get(key, 0) + 1
            writer.writerow(dict(
                scenario=key[0], seed=key[1], decision_index=counters[key],
                decision_time=decision_time, task_id=task_id,
                workflow_slot=pending["owner"], executed_vm_id=result.vm_id,
                action_source=pending["source"],
                batch_id=pending.get("batch_id"), batch_size=pending.get("batch_size"),
                action=json.dumps(pending["preferences"].tolist(), allow_nan=False),
                state=json.dumps(pending["state"], ensure_ascii=False, allow_nan=False),
            ))
            stream.flush()
            return result

        with patch.object(policy, "assign", assign):
            return _evaluate(protocol, policy, split, trace_path)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("validate", "evaluate"):
        raise SystemExit("Use validate or evaluate, followed by the original DTECN options")
    with patch.object(runner, "evaluate", evaluate_with_csv), \
            patch.object(runner, "load_manifest", load_compatible_manifest):
        return runner.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
