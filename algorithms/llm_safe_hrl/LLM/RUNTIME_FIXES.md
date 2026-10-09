# Runtime fixes based on 4079077

Quick simulation, both surrogate gates, their warmup thresholds, CMA budget,
diagnostic perturbations and energy-only ranking are retained.

## Fixes

- Energy-only SyntaxError/ValueError becomes a candidate validation failure.
  Invalid rules use the existing repair/discard path instead of crashing a run.
- OpenAI SDK dependency is >=1.60,<3. GPT-6.1 startup checks reasoning_effort
  support before spending time evaluating the seed or making API requests.
- GPT/Qwen SDK retries are disabled; the shared caller owns bounded retries.
  Permanent request errors stop immediately. Exhaustion raises an exception,
  never exit(0). Empty content and wrong choice counts are checked.
- Candidate generation can retain successful replies when another request
  times out. Failed positions remain invalid candidates. A temporary repair
  API failure discards that candidate. An entirely failed generation batch
  stops the run with an error rather than reporting success.
- Shared API semaphore limits concurrent requests across generation and repair.
  `llm_api.max_parallel_requests=4`, `llm_api.max_attempts=3`,
  `llm_api.timeout=180` are defaults. This API timeout is separate from the
  existing 300-second simulation timeout. Retry-After seconds are respected
  up to 120 seconds. Permanent authentication/model/SDK errors are not retried.
- Failed/nonfinite/penalty energy labels cannot train either surrogate.
  Failed quick observations cannot become parameter model features.
- NumPy file I/O, aliases to those operations and private attributes are
  rejected, including helper and schema-less rules. These AST checks are not
  a full operating-system sandbox.
- Final evaluation launches no longer spin reading stdout until the first
  log line. The evaluator prints that line after import/config loading, so
  the old loop normally serialized startup, not the entire simulation.
  A silent child could previously wait forever before its timeout was checked.
  Final timeouts are now measured from launch; killed children are reaped.
  Final children also receive one-thread BLAS limits.

## Why this search can be slow

1. 20 initial candidates; each outer round generates 20 crossover candidates,
   20 self-evolution candidates and 10 mutations, before duplicates/failures.
   `max_fe=20` means 20 outer rounds, not 20 individual simulations. Iteration
   also increases at each population update, so iteration 17 is not round 17.
2. Each parameterized rule defaults to 6 vectors x 4 CMA generations. Before
   surrogate savings, the stage budget is 6 quick seed runs + 18 x 3 refine
   seed runs, followed by up to 2 elites x 2 validation seeds.
3. Parameter diagnostics are still enabled. With P parameters, 2 directions
   x P x 2 diagnostic seeds means up to 4P additional exact seed simulations.
   The energy-only diagnostic replay gate is disabled, so these perturbations
   are real simulations (cache hits can reduce them). At P=12 that is 48 runs.
   The final frozen rule receives another training-seed evaluation.
4. Surrogates need >=200 exact label records, >=20 structure records or >=12
   parameter records, and healthy validated models. `enabled=true` alone does
   not mean screening is active. Predictions skip only some refine/structure
   work, never all quick, diagnostics, elite supplementation or validation.
5. Every cache miss starts a fresh Python evaluator. Local Windows CLI cold
   startup measured about 0.43 seconds; that is not a Linux server benchmark.
6. In the user's deployment each experiment has its own 32-core server and
   28 simulation workers, so cross-experiment CPU competition is not assumed.
   Check per-server worker utilization and container CPU quota if simulations
   remain slow. Offline evaluation is CPU/NumPy work, not HRL GPU training.
7. Repair requests, API retries and ExtraTrees refits/fsync add overhead.
   A healthy Qwen service may be faster with higher API concurrency; the new
   default 4 favors stability and is not a promise of faster generation.

No search budget or diagnostic setting was reduced to obtain faster numbers.

## Server checks

GPT/Qwen now use streaming Chat Completions in the shared caller. The complete
text interface returned to SeEvo is unchanged. Reasoning deltas are excluded;
all choice indices must finish with `stop`. Partial, empty or truncated streams
are retried within the same limit and are never sent to rule evaluation.
The connection is closed after each attempt, and the API semaphore remains held
until stream consumption ends. Existing launch commands require no stream flag.
This does not guarantee a gateway will forward chunks promptly or fix billing.

Upgrade the SDK in the same environment used to launch experiments:

```bash
python -m pip install 'openai>=1.60.0,<3'
python -c 'import openai; print(openai.__version__)'
```

New logs separate API and simulation work:

```bash
grep -E '\[llm batch\]|\[evaluation batch\]|Parameter batch stage=|Parameter optimization|failed open|not_ready' /path/to/run.log | tail -n 80
ps -eo pid,ppid,etime,%cpu,%mem,args | grep '[p]ython'
cat /sys/fs/cgroup/cpu.max 2>/dev/null
cat /sys/fs/cgroup/cpu.stat 2>/dev/null
```

In cpu.max, `quota / period` is the CPU allowance when quota is numeric;
`max` means this cgroup has no quota (parents may still restrict it).
Compare cpu.stat at two times: increasing throttled_usec suggests quota pressure.

Inspect completed optimizer statistics (replace RUN with the exact execution):

```bash
export RUN="$PWD/out/main_energy_only/SS/L/EXECUTION_ID"
python - <<'PY'
import json, os
from pathlib import Path
paths = sorted(Path(os.environ['RUN']).rglob('*_search.json'), key=lambda p: p.stat().st_mtime)
for path in paths[-5:]:
    data = json.loads(path.read_text())
    stats = data.get('surrogate_stats', {})
    print(path.name)
    for key in ('gate_generations', 'predicted_refine_vectors', 'estimated_seed_calls_avoided', 'failure_reasons'):
        print(key, stats.get(key))
    manager = stats.get('manager', {})
    print('exact labels:', manager.get('exact_structure_labels'), manager.get('exact_parameter_labels'))
    print('gates:', manager.get('gates'))
PY
```

`[evaluation batch] wall_seconds` includes waiting for the shared worker pool,
simulation and cache writes; summing overlapping batches is not total wall time.
`estimated_seed_calls_avoided` is an estimate, not measured seconds saved.

The fixed generation defaults can be overridden without changing local
simulation concurrency, for example `llm_api.max_parallel_requests=8`.
Keep `timeout=300 surrogate.enabled=true` and quick stages unchanged.

Running processes do not load these edits automatically. Restart only when
ready; the normal startup creates a new execution. This patch adds no full
evolution resume, does not erase old artifacts and does not cleanse previously
polluted surrogate files. Use a fresh execution for the corrected run.
