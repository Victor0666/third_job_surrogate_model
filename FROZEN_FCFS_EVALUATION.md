# Frozen FCFS Manager evaluation (base: 78bd0bb)

This branch adds an evaluation-only comparison, not FCFS retraining. It retains
Host/VM Qr, Qc, checkpoint lambda, observations, resource masks, phase timing,
and safety behavior. The default `learned` policy is unchanged.
`fixed_fcfs` skips Manager inference at every phase, sorts ready tasks by
`(task_ready_time, task_id)`, and records FCFS identity rather than LLM history.

## New server

```bash
git clone --branch codex/frozen-fcfs-eval-20261008 --single-branch \
  https://github.com/Victor0666/third_job_surrogate_model.git \
  third_job_surrogate_model_fcfs_eval
cd third_job_surrogate_model_fcfs_eval
conda activate third_work_py311
export PYTHONPATH="$PWD:$PWD/algorithms/llm_safe_hrl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p out/console_logs
python -m hrl_mix.frozen_manager_ablation --help
```

Use the same Python dependencies as the 78bd0bb environment. Cloning does NOT
bring ignored/untracked trained weights or generated heuristic artifacts.
Transfer the complete checkpoint folder (best manifest plus all three weights),
and the complete original LLM execution artifact folder (Top-K JSON, generated
Python rules, reports and any referenced files). Preserve relative structure
and file contents; library SHA256 must match the training checkpoint.
The exact deadline caches for every test scenario must also be available.
Do not change library contents to bypass hashes. Use --heuristic-library to
supply its relocated path. Absolute references inside artifacts must remain
resolvable; preserve the original location when needed.

Use a completed, immutable checkpoint copy. Copying best weights while a
training process is overwriting them can produce an inconsistent bundle.
Checkpoint paths in the manifest are resolved relative to that manifest.
--allow-external-checkpoint allows a bundle outside this checkout, while
protocol identity, library hashes and network dimension checks remain active.

## SL_L example: same frozen bundle, same seeds, both policies

DDL is read from the checkpoint snapshot; there is no --ddl evaluation override.
Replace LIBRARY below with the actual training Top-K JSON location.

```bash
CKPT="$PWD/checkpoints/main_single/SL/safe-sl-l-c78d8a5ba2/best_checkpoint_manifest.json"
LIBRARY="/absolute/path/to/original/topk_heuristic_library_k10.json"
CACHE_ROOT="$PWD/data/deadlines/fcfs/exact_mix_v1"
EVAL_ARGS=(
  --protocol single --source-scenario SL
  --checkpoint-manifest "$CKPT" --allow-external-checkpoint
  --heuristic-library "$LIBRARY"
  --device cuda:0
  --deadline-cache "SL=$CACHE_ROOT/fcfs_smallTask_largeRes_exactmix_formal38.json"
  --deadline-cache "ML=$CACHE_ROOT/fcfs_medTask_largeRes_exactmix_formal38.json"
  --deadline-cache "LL=$CACHE_ROOT/fcfs_largeTask_largeRes_exactmix_formal38.json"
)
```

First run a one-seed check (evaluates SL/ML/LL):

```bash
python -u -m hrl_mix.frozen_manager_ablation "${EVAL_ARGS[@]}" \
  --manager-policy fixed_fcfs --test-seeds 201 \
  --output-dir "$PWD/out/frozen_manager_ablation/SL_L/smoke_$(date +%Y%m%d_%H%M%S)"
```

Then run the paired experiment, sequentially to limit resource contention.
Omitting --test-seeds uses the same held-out seeds 201..230 for both policies.

```bash
RUN_ROOT="$PWD/out/frozen_manager_ablation/SL_L/$(date +%Y%m%d_%H%M%S)"
for POLICY in learned fixed_fcfs; do
  python -u -m hrl_mix.frozen_manager_ablation "${EVAL_ARGS[@]}" \
    --manager-policy "$POLICY" --output-dir "$RUN_ROOT/$POLICY" \
    > "out/console_logs/SL_L_${POLICY}_$(date +%Y%m%d_%H%M%S).log" 2>&1 || break
done
```

Each explicit output directory must be empty. Results include per-seed CSV/JSONL,
scenario aggregate JSON and frozen_test_manifest.json recording policy,
checkpoint SHA256, test seeds and frozen-state verification. Checkpoint files
are checked again after evaluation; a change invalidates the run.
For other conditions change source, checkpoint and resource-matched cache set:
SS/MS/LS smallRes; SM/MM/LM medRes; SL/ML/LL largeRes.
Each T/M/L checkpoint retains its own DDL configuration.

Compare per-seed fuzzy_energy_score with paired seeds:
`FCFS saving (%) = 100 * (learned_energy - fcfs_energy) / learned_energy`.
Record lateness and violation too. This tests the frozen Manager replacement;
Host/VM were trained under the original Manager, so distribution shift prevents
attributing every difference exclusively to LLM rule quality.
No pretrained checkpoint is distributed with the branch, and local tests do not
replace a full evaluation of the user's transferred networks.
