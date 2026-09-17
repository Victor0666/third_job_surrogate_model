# DRL-EA / Niching-GP comparison

`method_id = drlea_nichgp` is an independent comparison algorithm over the
current project's `HrlFcfsCacheEnv` problem simulator. It does not invoke the
Manager, Host Agent, VM Agent, LLM, SeEvo, safety shield, fallback controller,
safety-value network, Lagrange controller, or their checkpoints. The shared
environment is instantiated with every safe-RL switch disabled.

## Architecture

Stage 1 uses FCFS over the global ready set and a flat Routing Agent (RA) over
all `global_vm_id` values. Stage 2 freezes the validation-selected RA, records
global-ready situations, and evolves four behaviorally distinct arithmetic GP
rules with clearing. Stage 3 freezes RA and the four rules, then trains a
Sequencing Agent (SA) whose four actions select one rule. The rule chooses a
global ready task; RA alone chooses its VM.

Both agents are Dueling Double DQN instances with independent online/target
networks, replay, soft target updates and epsilon-greedy. RA exploration,
greedy selection and Double-DQN bootstrap consume the legal-VM mask. With
`allow_busy_vm_queueing=false` (formal default), an all-busy state advances to
the next simulator event and creates no transition.

## Observations

RA dimension is `8 + 10 * num_vms`. Its task block is
`workload,input_size,output_size,fuzzy_slack,upward_rank,remaining_work,
ready_wait_time,uncertainty`; every VM block, ordered by `global_vm_id`, is
`legal,available_wait,modal_comm,modal_exec,modal_finish,risk_finish,
incremental_fuzzy_energy,host_utilization,host_type,fuzzy_uncertainty`.

SA is always 16-dimensional: normalized ready count; mean/std/max/min of
minimum execution; mean/std/max/min workflow completion ratio; mean/std fuzzy
slack; mean/std/max/min predicted fuzzy lateness; and at-risk workflow ratio.
Empty and singleton sets return finite vectors of the same size.

GP schema `drlea_gp14_v1` has the frozen order
`READY_COUNT,READY_WORK,MIN_INPUT_COMM,MIN_OUTPUT_COMM,MIN_COMP,MIN_ENERGY,
WF_AGE,TASK_WAIT,TIME_TO_DEADLINE,FUZZY_SLACK,REMAIN_TASKS,REMAIN_WORK,
TOTAL_WORK,UNCERTAINTY`. Smaller priority wins. Arithmetic, protected
division, `min`, and `max` are retained; non-finite programs are invalid.

Continuous values use fixed, training-independent `NormalizationConfig`
scales: workload 150000 MI, data `1e10` bit, time 300 s, upward rank 1000 s,
remaining work 3000000 MI, energy 1000 J and ready count 32. Positive values
use clipped `x/scale` in `[0,1]`; signed slack uses
`x/(|x|+scale)` in `[-1,1]`. Finish times are relative to current time.
`host_type` is a cloud/edge binary category, never an ID. No test statistics
or future outcomes are used.

## Fuzzy objective and selection

The adapter calls the current simulator's three shadow timelines and energy
replay, freezing `eta=0.95` and `lambda_E=1.0`:

```text
risk_finish = 0.05 * modal_finish + 0.95 * upper_finish
fuzzy_energy_score = fuzzy_energy_mean + fuzzy_energy_std
fuzzy_lateness = max(0, workflow_risk_finish - workflow_deadline)
```

RA checkpoints, GP fitness/archive, SA checkpoints and final models minimize
the ordinary lexicographic tuple `(violation_rate, max_lateness,
mean_lateness, fuzzy_energy_score)`.

## Commands and output

The four stage commands in the task specification are supported verbatim.
`python -m algorithms.comparisons.drlea_nichgp.run_pipeline` runs all stages;
add `--smoke` for the fixed 2/8/2/3 debug profile. Formal single-source
artifacts are written under
`out/comparisons/drlea_nichgp/main_single/<scenario>/<ddl>_a<seed>/`;
legacy smoke artifacts use
`out/comparisons/drlea_nichgp/<scenario>_<ddl>_a<seed>/`. Each directory
contains config/source hashes, `ra.pt`, `rules.json`, `sa.pt`, histories,
manifests, evaluation JSON/CSV, and workflow metrics.

Stage-2 fitness episodes can use a spawn-safe CPU process pool. The unit of
parallel work is one independent `(program, scenario, seed)` episode; program
generation, clearing, archive updates, validation selection and all SA
training remain in their original deterministic order. The default
`--workers 1` is the serial reference path. For example, a complete run with
eight GP workers is:

```powershell
python -m algorithms.comparisons.drlea_nichgp.run_pipeline --scenario SS --ddl T --algorithm-seed 0 --workers 8 --device cpu --threads-per-worker 1
```

When the validation-best RA already exists, the dedicated continuation entry
loads the exact persisted `config.json` and starts at decision-situation
collection, then runs GP, SA and frozen final evaluation:

```powershell
python -m algorithms.comparisons.drlea_nichgp.run_from_ra --ra-checkpoint out/comparisons/drlea_nichgp/main_single/SS/T_a0/ra.pt --workers 8 --device cpu --threads-per-worker 1
```

The formal nine-configuration continuation runner expects the existing
`main_single/{SS,SM,SL}/{T,M,L}_a0/config.json` and `ra.pt` artifacts. It
records per-configuration completion or failure in `gp_matrix_status.json`
and skips only outputs whose model/protocol/fingerprint checks still pass:

```powershell
python -m algorithms.comparisons.drlea_nichgp.run_gp_matrix --workers 8 --device cpu --threads-per-worker 1
```

GP fitness is checkpointed atomically after each generation in
`gp_fitness_cache.json`. A cache entry is reused only when the complete
configuration, RA online weights, DAX/deadline inputs, GP terminal schema,
relevant source hashes, runtime versions and evaluation device match. A stale
or corrupt cache is ignored and recomputed. Set `DRLEA_CACHE_AUDIT=1` to
recompute and assert the exactness of static environment-feature cache hits
during a diagnostic run.

Deadline-cache locations are persisted only as project-relative
`data/deadlines/...` paths. Loading a historical config automatically strips
an old machine root such as `/T20050013/.../` while retaining the same
`data/...` suffix, so RA continuation artifacts remain portable between
machines.

Formal train, validation and test seeds are strictly disjoint. Explicit test
seeds overlapping train or validation are rejected.

## Known limits

The checked-in smoke evidence uses three workflows and one seed per split; it
verifies executability, not convergence, statistical superiority or
cross-scenario robustness. Formal GP uses ready threshold 6, while smoke lowers
it to 2 only to keep the run short. Networks are VM-count specific. The
existing FCFS cache may emit a case-only `Ligo_30.xml`/`LIGO_30.xml` warning;
the Windows smoke run resolves the same cache entry, but case-sensitive
platforms should verify cache naming before formal experiments. Predicted
finish/slack features are simulator-model estimates, not guarantees of future
deadline feasibility.
