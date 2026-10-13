"""Run online validation/test with immediate CSV logging; keep core artifact hashes."""
import csv
import json
import sys
from unittest.mock import patch

from algorithms.comparisons.dtecn_llm import run as runner


_evaluate = runner.evaluate


def evaluate_with_csv(protocol, policy, split, trace_path=None):
    if trace_path is None:
        return _evaluate(protocol, policy, split, trace_path)
    csv_path = trace_path.with_suffix(".csv")
    fields = ("scenario", "seed", "decision_index", "decision_time", "task_id",
              "workflow_slot", "executed_vm_id", "action_source", "action", "state")
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
    with patch.object(runner, "evaluate", evaluate_with_csv):
        return runner.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
