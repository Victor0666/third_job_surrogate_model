# MTGP comparison

This is the paper's two-tree genetic-programming mechanism adapted to the
project's fixed workflow data, resources, fuzzy uncertainty and DDL/energy
objective. It does not run RL, LLM, Niching-GP, a shield, fallback, safety-value
learning or demonstration pretraining.

The routing tree assigns each newly ready task to a feasible VM, including busy
VMs. Input transfer completes before the task can enter that VM's waiting queue.
An arrival at an idle VM starts immediately, without running the sequencing
tree. At completion, the sequencing tree selects the next already-arrived task
from that VM's queue. Mapping is irreversible and execution is non-preemptive.
Computation plus output transfer occupy the VM; input transfer does not. All
methods use the shared `common/scheduling_transport.py` timing formula and the
production HRL environment. Existing HRL policies retain their advance booking
of a selected VM; this is a policy reservation, not input-transfer CPU load.

The original ten terminals retain their meanings. UT/DT are named
IN_COMM/OUT_COMM, TWT starts at VM arrival, WIQ is queued computation time,
TTIQ is queued input/computation/output time, and MRT is an absolute availability
time. TIME_TO_DDL and DELTA_FUZZY_ENERGY extend the common terminal set for the
new objective. Features are not clipped or normalized. No future arrival or
realized outcome is available to a tree. Input/output data and bandwidth follow
the existing model; no mobile execution node, network contention, locality
shortcut, or additional network-energy coefficient is introduced.

An individual contains sequencing tree 0 and routing tree 1, evaluated jointly.
Initialization is ramped half-and-half with depths 2-6. Maximum depth is 8;
tournament size is 7; internal/terminal node selection is 90%/10%. Crossover
exchanges subtrees in one randomly selected tree and swaps the other whole
tree. Mutation uses a grow subtree of maximum depth 4. Division returns 1 only
for an exactly zero denominator; non-finite programs invalidate the candidate.
There is no queue-length cutoff. Formal defaults use paper Table VI: population
1000, 51 generations, 10 elites, and crossover/mutation/reproduction probabilities
0.80/0.15/0.05. The public Java parameter file instead contains population 100;
that discrepancy is not silently adopted.

Each generation rotates through the shared training scenario/seed pool. Every
individual, including elites, is scored on that instance; an identical tree
pair on an identical instance may reuse exact cached fitness. Each unique
generation winner is archived; after evolution, validation selects one whole
tree pair. Test scenarios/seeds are never used during evolution or selection.
Training and validation episode counts are reported separately.

Fitness, validation and final results use the exact production functions
`hrl_mix.safe_metrics.build_episode_metric_record`,
`aggregate_safe_metric_records` and
`hrl_mix.model_selection.aggregate_seed_feasibility_metrics`. They minimize
`(DDL violation rate, maximum fuzzy lateness, mean fuzzy lateness, fuzzy energy
score)` lexicographically. Incomplete or deadlocked episodes fail, rather than
being scored on a subset of workflows. Fuzzy energy std is the triangular-fuzzy
energy statistic, not the standard deviation across seeds.

From the repository root:

```powershell
python -m algorithms.comparisons.mtgp.run_pipeline --smoke
python -m algorithms.comparisons.mtgp.run_pipeline --scenario SS --ddl T --protocol single --algorithm-seed 0
python -m algorithms.comparisons.mtgp.run_pipeline --protocol multi --resource-scale S --ddl T --algorithm-seed 0
```

Formal Single cache overrides use the same complete source/target mapping as
the existing comparison runners, for example `--deadline-cache SS=...`
`--deadline-cache MS=... --deadline-cache LS=...`.

Outputs are isolated under `out/comparisons/mtgp/transport_v1/`. `rules.json`
contains the tree pair, protocol/model/input/source hashes, validation result
and GP budget. `history.json` records generation training fitness. `eval.json`
and `eval.csv` contain frozen final-test metrics. To evaluate again, provide
`--rules-file <rules.json>` with the identical protocol and cache arguments.
Loading mismatched model/source/input hashes fails. Existing trained rules are
not overwritten. Smoke uses 8 individuals, 2 generations and 2 workflows;
it demonstrates execution, not convergence or superiority.

## CPU acceleration and resume

```powershell
python -m algorithms.comparisons.mtgp.run_pipeline --smoke --workers 4 --threads-per-worker 1
python -m algorithms.comparisons.mtgp.run_pipeline --scenario SS --ddl T --protocol single --algorithm-seed 0 --workers 8 --threads-per-worker 1
python -m algorithms.comparisons.mtgp.run_pipeline --scenario SS --ddl T --protocol single --algorithm-seed 0 --workers 8 --threads-per-worker 1 --resume
```

Formal Single runs still require their complete `--deadline-cache` mappings.
Use the same protocol, seed, cache and output arguments when resuming. Worker
count may change on resume. `--workers 1` is the serial reference; process
evaluation uses Windows-safe spawn. Each worker owns a fresh environment and
never breeds. Results are collected in submission order, while population
generation and all GP RNG draws remain in the parent process. The pool lives
through training and validation; frozen final evaluation stays serial so its
reported scheduling time is not distorted by worker contention. Start with
4 workers and measure 2/4/8 against the available memory and actual throughput.

Within an episode, queue totals are computed once per decision in their
original summation order. Only terminals actually used by the tree are
extracted; duration-cache dictionaries are read without copying. NumPy float64
evaluates all candidates elementwise in the original expression order, with
exact-zero protected division, first-operand min/max ties and unchanged
task/VM tie-breaking. `task_terminals` and `evaluate_tree` remain the scalar
reference for exactness checks. No feature clipping or approximate energy
formula is introduced. MTGP skips intermediate energy-reward integration that
it does not consume; the final energy/DDL evaluator is unchanged.

`training_checkpoint.json` is atomically written after each completed generation
and includes the next population, RNG state, archive, history and fitness cache.
Cache keys contain both trees, scenario and seed; the checkpoint additionally
binds protocol, input hashes, source hashes and Python/NumPy versions. Validation
and final-test instances never enter the training fitness cache. Corrupt or
mismatched checkpoints fail closed; existing checkpoints are not overwritten
without `--resume`. An interrupted generation is recomputed from the previous
completed generation, and counts describe completed/checkpointed work only.
Validation can be repeated after interruption. If rules were already selected,
`--resume` only repeats frozen final evaluation. Neither test seeds nor test
results affect resumption, evolution or model selection.

Reported `training_episode_count` is the logical GP evaluation budget;
`simulated_training_episode_count` counts actual unique simulations, and
`training_cache_hits` reports saved simulations. Worker counts/threads are
recorded separately. They do not change the frozen training settings or RNG.

`tests/test_mtgp_acceleration.py` compares scalar/batch score bits, used terminal
values, full assignment/shadow timelines, serial/parallel selection and an
interrupted run resumed with a different worker count.

See `docs/STAGED_COMMUNICATION.md` for the shared physical-model change and
the treatment of historical checkpoints and fixed deadline inputs.

Upstream: https://github.com/MengBIT/Fog-Computing/tree/multiDeviceDebug
