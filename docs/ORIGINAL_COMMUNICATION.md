# Restored original VM service model

Model identifier: `combined_input_compute_output_v1`.

The staged model is preserved on `codex/energy-only-llm-20261008` at `f83da5b`.
This branch, `codex/mtgp-original-model-20261010`, restores the original physical
timing and VM load convention for all algorithms:

```text
service_start = max(routing_time, VM_available_time)
input_arrival = service_start + input_time
completion = service_start + input_time + compute_time + output_time
VM load record = [service_start, completion]
```

Input transfer waits for VM availability. It cannot overlap the same VM's
previous computation. All three stages contribute to its load record, exactly
as in the historical combined-service model. Shared power integration, fuzzy
energy definitions, deadline risk and seed aggregation remain unchanged.
The original optimistic/modal/pessimistic task-start replay is restored too.

MTGP retains two jointly evolved trees: routing selects a VM; sequencing chooses
the next task in that VM's waiting queue. In this adaptation the logical queue
contains routed, ready tasks before their input transfer. An idle VM starts a
routed task directly; a busy VM selects a queued task after completion, then
serves its entire input/compute/output interval. Upload-before-queue admission
with overlapping transfers belongs to the preserved staged branch and is not
claimed for this restored-model adaptation.

Existing HRL, IRWS, MARL, PD3QN, FCFS, DRL-EA and exact LLM/SUR evaluations use the
same combined service and production evaluator. GP operators, fitness ordering,
population budgets, CPU acceleration, seed splits and frozen selection remain.

Artifacts are isolated under `combined_v1`; HRL run names end in `-cmb1`.
CEWS admission is v5. The source/model guards reject staged checkpoints, rules,
demonstrations and admission reports. Start fresh original-model training;
do not resume `transport_v1` or `-tx1` outputs. Historical raw weights are not
automatically admitted solely because their physical model was combined.

Fixed DDL caches remain shared inputs. Keep the same files across methods.
Physical-model restoration does not prove that previously reported tables used
identical seeds, cache contents, metric fields or evaluation scenario ranges.

The branch does not overwrite staged results. Desktop commands for the staged
branch remain separately saved; use the new original-model command file.
