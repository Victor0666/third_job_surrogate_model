# FCFS baselines

Both formal FCFS baselines use `FuzzyBaselineEnv`, the shared comparison
protocol, and final-test seeds `201, 202, 203`.

- `fcfs_fcfs` (`FCFS-FCFS`) orders ready tasks by
  `(ready_time, workflow_arrival_time, workflow_id, task_id)` and selects the
  legal VM with minimum `(vm_available_at, vm_id)`.
- `fcfs_fixed` (`FCFS-Fixed`) uses the identical task order and delegates VM
  selection to the environment's shared `select_vm_deterministic()` rule.

Run the methods independently:

```powershell
python -m algorithms.comparisons.fcfs.run_fcfs_fcfs --scenario SS --ddl T
python -m algorithms.comparisons.fcfs.run_fcfs_fixed --scenario SS --ddl T
```

The historical `env_fcfs.py` is retained for source compatibility, but formal
evaluation does not import or instantiate it.
