# Shared staged communication model

Historical model preserved on `codex/energy-only-llm-20261008` at `f83da5b`.
The current branch restores combined service; see `ORIGINAL_COMMUNICATION.md`.

Model identifier: `input_transfer_then_compute_output_v1`.

The former model started a VM load record at routing and occupied it for input
transfer + computation + output transfer. All schedulers now share:

```text
input_arrival = routing_time + input_bits / bandwidth
execution_start = max(input_arrival, VM_available_time)
completion = execution_start + workload / processing_rate + output_bits / bandwidth
VM load record = [execution_start, completion]
```

Input transfers may overlap VM work, and concurrent transfers use fixed
bandwidth without sharing. Computation/output are non-preemptive and serial on
each VM. Successors become ready only at completion, including output transfer.
The existing SPECpower Host integration and triangular fuzzy mean/std/score
remain shared. There is no added network-energy coefficient. The integration
window/idle-power convention is the existing common energy function.

The production HRL simulator maintains input/start/finish events. Its new
`route_task_to_vm` interface performs input transfer before VM queue admission.
MTGP uses it with its per-VM sequencing tree. Existing learned and deterministic
policies retain their advance reservation behavior via `assign_task` or the
HRL assignment path; reservation fixes their order, but their actual load
starts after input arrival. They do not acquire GP queue sorting. Legacy scalar
FCFS/VM-only simulators use the same physical timing function and exclude input
transfer from VM load records. Their legacy interaction APIs remain intact.

Same-time HRL events use a stable heap order (finish, input/input_reserved,
start, then task/VM ID). Completion dispatches the existing arrived queue;
new successors are routed at the next policy interaction. An idle arrival
starts immediately, while an arrival at a busy VM waits. This tie convention
is frozen explicitly; the Java comparator is not reproduced byte-for-byte.

Optimistic/modal/pessimistic replay uses the same chosen mapping and VM order,
scenario-specific transfer/computation/output times, and preserved policy
deferral. It does not optimize a separate policy in each fuzzy scenario.
Task/VM risk predictions, remaining-workflow predictions and marginal energy
proxies use the staged model. Final metrics continue to use the production
Safe-HRL episode evaluator. DRL-EA now delegates to that evaluator too, including
completion checks and workflow-weighted aggregation. Mixed communication-model
records cannot be aggregated; incomplete evaluations cannot select a model.

## Fixed deadlines and old artifacts

Existing DDL caches are retained as **fixed shared external inputs**, so this
change does not simultaneously change deadline difficulty. Their numerical
deadlines are identical for all methods. They are historical FCFS references,
not newly recomputed FCFS results under the staged model. If deadlines are to
be redefined against the new FCFS model, generate a new complete cache set and
apply it to every method; do not replace only MTGP's cache.

New HRL run IDs end in `-tx1`. Fuzzy comparison outputs and DRL-EA outputs have
a `transport_v1` directory. Old training outputs are preserved. Formal HRL
evaluation and resume reject checkpoint snapshots without the current model
identifier. Fuzzy comparison checkpoint protocol hashes and DRL-EA protocol
identities also bind the model. Historical models need new training before
being included in staged-model main comparisons; historical metric tables
must not be merged with new results.

Use the protocol-aware formal evaluation runners. Historical ad-hoc scripts
that directly load raw network weights are not a substitute for these guards.
LLM rule/fitness caches already fingerprint the base/common sources and are
invalidated by this source change; existing raw rules are not assumed newly
admitted under the new model.

CEWS admission protocol is now v4, so historical v3 rule admission reports must
be re-evaluated. Demonstration training identities bind the communication model;
old offline trajectories cannot initialize a staged-model formal run.

## Verification

`tests/test_mtgp_transport.py` checks input overlap, no CPU load before arrival,
arrived-only sequencing, idle-arrival immediate execution, queue waiting-time
definition, whole-tree swapping, exact-zero division, identical schedule/energy
between reservation and queued execution, and exact evaluator parity with
DRL-EA. Existing fuzzy baseline, safety, model selection and frozen protocol
tests cover the other shared consumers. Full formal training is not a unit test.
