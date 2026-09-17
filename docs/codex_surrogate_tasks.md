# Codex 实施指令：决策轨迹精确重放与 LLM 修复并行化

本文件是交给编码 agent 的任务书。T0–T4 不训练机器学习代理模型：T0 是
LLM 修复并行化，T1 是诊断数据口径修复，T2–T4 是确定性轨迹重放和真实仿真
剪枝。学习型代理只属于未来的 T5。

## 本轮范围

**只实现 T0 → T4，按顺序、每个任务独立提交，不要合并。** 每个提交必须能
独立运行对应测试；未达到正确性验收不得进入下一任务。

**T5 本轮不实现。** 不新增 scikit-learn、LightGBM、CatBoost 或其他模型依赖。
T4 只保留以后评估 T5 所需的真实数据，不实现任何拟合、预测或
`surrogate_inferred` 分支。

优先级：**T0 > T1 > T2 > T3 > T4**。T0 与轨迹重放解耦，可以先交付。
本文中的历史耗时只用于预算，不作为交付承诺；最终收益必须由完整成功运行实测。

## 术语与统计单位

- **context**：一个 `(scenario_id, seed)` 评价上下文。
- **perturbation vector**：一个完整的参数扰动向量。
- **context 恒等率**：单个扰动在单个 context 上与基线决策序列恒等的比例。
- **vector 跳过率**：一个扰动在全部 `scenario_ids × diagnostic_seeds` 上均恒等，
  因而可以整体跳过真实仿真的比例。T4 的收益只能使用该指标。
- **精确重放结果**：由完整轨迹证明与基线性能恒等的结果，不是模型预测。
- **规范化性能投影**：只包含约束、DDL、lateness、能耗、完成量以及逐 context
  性能行的结果视图，不包含候选路径、参数哈希、规则哈希和证据来源。

禁止把 context 恒等率直接当成 vector 跳过率。禁止把来源字段不同的两个结果做
“整个字典逐字段相等”验收。

## 关键调用链

```text
seevo.py  _prepare_individual_for_evaluation()
   └─ CMAESOptimizer.optimize(schema, evaluator, batch_evaluator=..., ...)
        └─ cmaes_optimizer.py  _safe_evaluate_many(..., "diagnostic", diagnostic_seeds)
             └─ seevo.py  _evaluate_parameter_maps(candidate, parameter_maps, stage, seeds)
                  ├─ freeze_rule_source(...) → params_{hash[:20]}.py
                  ├─ EvaluationCacheKey.create(...) → cache.get()
                  ├─ _evaluation_command(...) → subprocess: eval.py
                  └─ cache.put_many(cache_entries)
```

生产协议已经把训练场景绑定到 `parameter_optimization.scenario_ids`。启用重放门时
仍须在 `OptimizerConfig.from_mapping()` 和 `optimize()` 入口验证场景列表非空，
但不要另造一套场景发现逻辑。

## 全局约束

1. **不得修改** `algorithms/llm_safe_hrl/base/hrl_env.py` 的调度语义、能耗计算、
   任务选择、Host/VM 选择或事件推进逻辑。轨迹捕获只能复用现有接口。
2. **不得修改** 对比算法和 Safe-HRL 训练逻辑。
3. T0–T4 不新增第三方依赖。
4. 精确重放派生结果不得写入 `EvaluationCache`、进入 `elite_samples`、confirm
   阶段或最终性能报告。它只用于局部参数诊断。
5. 不得使用最终测试 seeds（201..230）做诊断、拟合或阈值标定。
6. confirm/validation seeds 可以用于最终候选选择，但不得把基于这些 seeds 的
   局部诊断重新送入反思提示词指导演化。
7. 不实现以下已知无效方案：
   - 用少量 CMA history 样本回归 Top-4 参数敏感度；
   - 只按参数名先验选择 Top-4 敏感参数。
8. 所有轨迹和临时文件必须在异常、取消和正常返回路径中通过 `finally` 清理。

---

## T0 — LLM 候选修复循环并行化

### 目标

并行处理 `_responses_to_validated_individuals()` 的外层候选；每个候选内部的修复
重试仍保持串行，因为下一次输入依赖上一次响应。

历史观测为 1,487 次修复调用、中位延迟约 53.8 秒、累计串行等待约 21.8 小时。
18.2 小时节省只能作为当前服务延迟和配额下的预算估算。

### 改动

- 使用独立 `ThreadPoolExecutor`，`thread_name_prefix="candidate-repair"`。
- 不复用 `SeEvo._shared_evaluation_executor`。
- 新增 `candidate_repair_workers` 配置，默认 20；实际并发为
  `min(len(responses), candidate_repair_workers)`。
- 每个 worker 返回 `(response_id, individual)`；主线程按 `response_id` 排序后
  组装结果，禁止依赖 future 完成顺序。
- `multi_chat_completion()`/`chat_completion()` 是直接依赖，必须保证并发失败
  可控：初始化响应变量，使用有限重试、请求超时和带 jitter 的退避；全部失败时
  抛出异常，不调用 `exit()`。不要重构其他 LLM 调用路径。

### 不变量

- `individuals` 的顺序与 `response_id` 完全一致。
- 文件名保持 `problem_iter{N}_response{id}_repair{attempt}.txt`。
- `candidate_generation_attempts` 和 `candidate_validation_history` 的语义不变。
- 相同确定性 mock 响应下，串行版和并发版的候选内容相同。

### 验收

- 新增 `tests/test_llm_candidate_repair_parallel.py`。
- 用 barrier/活动计数器断言最大同时活动 worker 大于 1 且不超过配置值；不要只用
  易波动的墙钟阈值证明并发。
- 确定性 mock 下比较完整 `individuals`。
- 覆盖单候选、部分候选无需修复、全部重试失败和输出顺序。
- `tests/test_llm_concurrent_runs.py` 继续通过。

---

## T1 — 诊断扰动改用训练 seeds，并闭合诊断基线口径

### 现状

当前代码使用：

```python
diagnostic_seeds = stage_seeds["confirm"] or stage_seeds["refine"]
```

默认会在 confirm/validation seeds `[4,5]` 上做局部扰动，随后把诊断写入反思
提示词。这属于验证集自适应使用风险，不是最终测试集泄漏。

### seed 选择

新增 `diagnostic_seed_count: int = 2`，使用：

```python
diagnostic_seeds = list(stage_seeds["refine"])[:config.diagnostic_seed_count]
```

`OptimizerConfig.from_mapping()` 必须验证：

- `diagnostic_seed_count > 0`；
- refine seed 数量不少于 `diagnostic_seed_count`；
- diagnostic seeds 唯一；
- diagnostic seeds 不与 validation/final-test seeds 重叠。

不再回退到只有一个 seed 的 quick 集，也不把全部三个 refine seeds 都用于诊断。

### 诊断基线

confirm 候选来自 refine history。构造 confirmed 条目时保留
`"refine_entry": elite`，从该条目的 `per_seed_metrics` 中按完整
`(scenario_id, seed)` 选择诊断 contexts，再使用现有
`aggregate_seed_evaluations()` 重新聚合，不新增仿真。

`OptimizationResult` 新增：

```text
diagnostic_baseline_metrics
diagnostic_baseline_seeds
diagnostic_baseline_contexts
diagnostic_baseline_source       # "refine_subset"
```

`best_metrics` 始终保留 confirm/final best 的语义。以下诊断必须改用
`diagnostic_baseline_metrics`：

- `_local_effects`
- `scenario_sensitivity`
- `fragility_analysis`

诊断产物记录 `baseline_contexts` 和 `perturbation_contexts`，二者必须完全相等。
只记录 seed 集合不足以支持多场景配置。

若当前在线优化结果缺少所需 context 行，**不得回退到 confirm 诊断**。应抛出
明确错误并阻止该诊断进入提示词。读取旧产物做离线展示时可以标记
`status="diagnostic_baseline_unavailable"`，但不得生成敏感度结论。

### 性能字段比较

新增一个共享的 `canonical_performance_projection(metrics)`，或等价的明确字段
列表，供 T1/T2/T3/T4 测试复用。子集重聚合与直接评价只比较该投影；候选哈希、
规则哈希、路径和证据来源单独验证。

### 验收

- 新增 `tests/test_diagnostic_seed_alignment.py`。
- 默认配置下 `diagnostic_seeds == [1,2]` 且数量为 2。
- `baseline_contexts == perturbation_contexts`。
- `[4,5]` 不出现在进入提示词的诊断证据里。
- 默认配置下诊断真实 episode 数与改动前相同。
- `[1,2]` 子集重聚合与直接真实评价的规范化性能投影相等。
- 缺行时不静默回退到 confirm。
- 更新 `tests/test_rule_optimization.py`。

---

## T2 — 在正式 evaluator 中捕获完整决策轨迹

### 原则

`hrl_env.py` 已有 `return_details=True`，不新增环境接口。轨迹捕获必须嵌入正式
`eval.py:run_instance()` 的现有事件循环，禁止复制另一套事件推进循环。

尤其要保留正式逻辑：

- 有 ready task 且有空闲 VM：选择并分配；
- 有 ready task 但无空闲 VM：`advance_to_next_resource_event()`；
- 无 ready task：`advance_to_next_event()`；
- 完成后验证 `completed_workflows == expected_workflows`。

### eval.py 侧

1. `parse_args()` 增加 `--capture-decision-trace DIR`，默认 None。
2. `run_instance()` 增加 `decision_trace_sink=None`。sink 非 None 时，在正式选择
   路径中记录：
   - `ready_task_ids`：int64；
   - 按 `trace_recorder.FEATURE_NAMES` 排列的 8 个 float64 特征数组；
   - `selected_index`：int32。
3. `evaluate_candidate()` 透传，每个 context 写入
   `{DIR}/{scenario_id}_seed{seed}.npz`。
4. 只有完整结束并通过 workflow 数量检查后才能把轨迹标记为 complete。

### 轨迹 schema

ragged 数据使用扁平数组和 offsets，不使用 object dtype。NPZ 至少包含：

```text
schema_version
feature_names
baseline_frozen_rule_hash
evaluation_config_sha256
scenario_id
seed
decision_count
expected_workflows
completed_workflows
complete
ready_ids_flat / ready_offsets
feature_*_flat / feature_offsets
selected_indices
```

注意：轨迹保存的是 `baseline_frozen_rule_hash`。被重放的扰动规则有自己的
`candidate_frozen_rule_hash`，二者不应相等，也不得互相校验为相等。

写盘要求：

- 临时文件写完、校验后原子替换目标文件；
- 加载时使用 `allow_pickle=False`；
- 校验 offsets、数组长度、有限值、决策数和 complete 标志；
- schema/config/context 不匹配时返回 trace mismatch 并回退真实仿真。

### seevo.py 侧

`_evaluate_parameter_maps()` 和 `_evaluation_command()` 增加：

```python
capture_trace_dir: str | None = None
bypass_cache_read: bool = False
```

只有基线轨迹捕获调用传入目录并设置 `bypass_cache_read=True`。缓存键不包含捕获
开关，因为捕获不改变性能；捕获调用仍可把真实结果写入缓存。

对最终 best 参数在 `diagnostic_seeds × scenario_ids` 上捕获一次基线轨迹。
捕获结果与 T1 的诊断基线只比较规范化性能投影；来源字段单独校验。

### 资源约束

- `max_trace_mb_per_structure` 必须在写入过程中执行，不只是事后检查。
- 增加全局 trace 预算或等价的并发结构上限，不能只写
  `per_structure × concurrency` 的事后公式。
- 删除操作放在结构级 `finally` 中。
- 分别测量不开捕获和开捕获的同 context 墙钟时间、峰值内存和文件大小。

### 验收

- 新增 `tests/test_decision_trace_capture.py`。
- 同一规则/context 开关捕获时，完整 RESULT_JSON 应相同；若增加了纯轨迹状态
  字段，则比较规范化性能投影并单独验证身份字段。
- 捕获轨迹必须 complete，决策数与 `selected_indices` 长度一致。
- 覆盖“ready 但无空闲 VM”路径，证明不会提前截断。
- `bypass_cache_read=True` 必须启动真实子进程，结果仍可写 cache。
- 全尺度捕获额外开销实测不超过 5%；未达到时先优化存储，不进入 T3。

---

## T3 — 精确重放门

### 先修正测量原型

现有 `scripts/measure_skip_rate_train_seeds.py` 只能作为待修正的测量工具，不能
作为生产事件循环的参考实现，原因包括：

- ready task 存在但无空闲 VM 时使用了错误的事件推进函数；
- stall 后可能返回不完整轨迹；
- 统计的是 context 恒等率，不是 T4 所需的 vector 跳过率；
- `structures` 输出的是搜索到的文件数，不是成功处理数。

旧输出 `513/1042 = 49.2%` 只能称为“旧原型的单 context 观测值”，不能作为
跳过率和收益输入。`1042 / 2 = 521` 个已映射向量，加上 21 个 unmapped 向量，
unmapped 比例约为 `21 / 542 = 3.9%`，不是 0.5%。

修正后的脚本必须复用 T2 的生产捕获路径，并输出：

```text
selected_structures
successfully_processed_structures
mapped_vectors
unmapped_vectors
context_judgement_count
context_identical_rate
vector_count
vector_identical_count
vector_skip_rate
skipped_episode_count
```

vector 恒等定义为该向量的全部 `scenario_ids × diagnostic_seeds` verdict 都满足
`status == "ok" and identical is True`。置信区间按结构做 cluster bootstrap，
不要把同一结构内的 contexts 当成独立二项样本。

### 新增实现

新增 `algorithms/llm_safe_hrl/LLM/surrogate/replay_gate.py`：

```python
@dataclass(frozen=True)
class ReplayVerdict:
    scenario_id: str
    seed: int
    identical: bool
    divergence_count_along_baseline: int
    first_divergence_index: int | None
    margin_stats: dict
    status: str  # "ok" | "trace_mismatch" | "rule_error"
    baseline_frozen_rule_hash: str
    candidate_frozen_rule_hash: str
    replay_seconds: float
```

加载完整轨迹和扰动规则，对每个决策点重算 scores，用现有
`validate_task_priority_scores()` 校验，并以与环境相同的 `np.argmin()` 取首个
最小值。一个规则对多个 contexts 重放时只加载一次。

复用 `critical_state_replay.load_frozen_priority_rule()`，不重新实现冻结、哈希、
AST 校验或加载。当前 loader 没有把模块登记到 `sys.modules`，因此不增加无效的
`sys.modules` 删除逻辑；用完释放普通对象引用即可。

### 语义约束

1. 只有完整轨迹上的 `identical=True` 才是可证明结论。
2. `identical=False` 不说明真实性能偏离大小，必须真实仿真。
3. 第一次分叉后的 baseline 特征不是真实候选状态；
   `divergence_count_along_baseline` 只能作为描述性特征。
4. NaN、inf、形状错误、加载失败和 trace mismatch 一律回退真实仿真。
5. 重放门不创建环境、不写 archive、不更新 EvaluationCache。

### 验收

- 新增 `tests/test_replay_gate_exactness.py`。
- 缩小配置取至少 30 个扰动，包含恒等和分叉样本。
- 对每个 `identical=True` 样本跑真实仿真，规范化性能投影必须与基线相等；
  来源和候选哈希必须属于扰动自身。
- 覆盖规则错误、轨迹不完整、schema/config/context 不匹配和 argmin 并列。
- 新增 `scripts/verify_replay_gate_full_scale.py`，在生产配置上验证至少 200 个
  真实扰动。恒等反例必须为 0；预计耗时只在实测后写入提交说明。
- 修正后的测量脚本必须报告 vector 跳过率；本任务书不预设 42%–56% 阈值。

---

## T4 — 把重放门接入 CMA 诊断

### 接口

`optimize()` 增加两个可选参数：

```text
replay_gate: Callable[[parameter_map, contexts], mapping[context, ReplayVerdict]] | None
trace_capture_evaluator: MetricEvaluator | None
```

使用批量 contexts 接口，避免每个 context 重复冻结和加载同一扰动规则。
`OptimizerConfig.scenario_ids` 是 contexts 的场景来源；启用时必须非空。

### 执行顺序

1. 使用 `trace_capture_evaluator` 对诊断基线捕获全部 contexts 的完整轨迹。
2. 生成全部扰动向量。
3. 通过一个共享小 helper 冻结参数规则并返回 source/hash/path；该 helper 同时供
   `_evaluate_parameter_maps()` 和重放门调用，禁止复制冻结逻辑。
4. 使用独立、有界的 replay executor 并发重放，规则每个向量只加载一次。
5. 一个向量的全部 contexts 均恒等时，派生 exact-identical 诊断结果；否则整个
   向量进入现有 `_safe_evaluate_many()` 做真实仿真。不做部分 context 混合复用。
6. 在 `finally` 中删除结构轨迹并释放规则引用。

### 派生结果与来源

不得“原样复制 baseline metrics”。exact-identical 条目必须：

- 复用基线的规范化性能字段和逐 context 性能行；
- 写入扰动自己的参数、参数哈希和候选规则哈希；
- 写入 `evidence_source="exact_identical"`；
- 写入 `baseline_evaluation_hash` 和 `trace_hashes`；
- 不进入 `_evaluate_parameter_maps()`，不写 `EvaluationCache`。

真实仿真条目标记 `evidence_source="real"`。若 exact-identical 条目出现在
`cache_entries`，立即抛出异常。

### 并发与统计

配置增加：

```yaml
parameter_optimization:
  diagnostic_seed_count: 2
  diagnostic_replay_gate:
    enabled: false                 # 通过正确性和正收益验收后改为 true
    replay_workers: 20
    max_trace_mb_per_structure: 128
    max_trace_mb_total: 1024
    delete_traces_after_use: true
```

`replay_workers` 必须配置化。29,162 个 context 若每次重放 2.1 秒，串行约
17.0 小时；20 路理想并发约 0.85 小时。必须实测，不能默认忽略回放开销。

`replay_gate_stats` 至少包含：

```text
context_judgement_count
context_identical_count
vector_count
skipped_vector_count
skipped_episode_count
real_vector_count
rule_error_context_count
trace_mismatch_context_count
replay_wall_seconds
replay_cpu_seconds
```

### 配置和默认开关

`from_mapping()` 验证：

- enabled 时 `diagnostic_perturbations=true`；
- `scenario_ids` 非空；
- worker 和内存上限为正；
- 总 trace 预算不小于单结构预算。

开发期间默认 false。只有同时满足以下条件，最终交付配置才改为 true：

1. 缩小配置和全尺度验证均为 0 个恒等反例；
2. 完整 SM/T 试运行的净墙钟收益为正；
3. 轨迹内存和磁盘峰值未超预算。

若净收益不为正，保留 `enabled: false` 并报告实测结果，不得为了开启功能而修改
恒等判定或放宽正确性要求。

### 验收

- disabled 时，与**完成 T1 后的关闭重放门基线**比较；原有字段和规范化性能
  投影相等。新增 stats 可以省略或明确标记 disabled。
- enabled 时必须有 `skipped_vector_count > 0`，但不设置未经验证的固定比例。
- `exact_identical` 的局部性能变化为 0，且身份字段属于扰动规则。
- 用唯一真实 `EvaluationCacheKey` 集合验证子进程数。缓存关闭且无重复上下文时，
  简化关系为：

  ```text
  baseline_capture_contexts
  + non_skipped_vectors × scenario_count × diagnostic_seed_count
  ```

  生产统计还要单列 cache hits、上下文去重、失败和重试。
- 新增 `tests/test_replay_gate_wiring.py`：enabled 时两个回调均非 None，disabled
  时均为 None。
- 新增端到端缩小配置测试，验证跳过项不写 cache、真实项正常写 cache。
- 输出真实的 trace capture、replay、避免仿真的耗时，并计算净收益。

---

## T5 — 未来的学习型代理（本轮不实现）

T4 只为每个真实仿真的扰动保存：

- 每个 `(scenario_id, seed)` 的 ReplayVerdict；
- 参数名、归一化值、边界距离和扰动方向；
- 完整真实约束指标、能耗指标和 `constraint_priority_key`；
- scenario、DDL、feature schema、配置和规则哈希。

写入 `{search_dir}/replay_verdicts.jsonl`，采用带 schema version 的 JSONL。写入
失败不得影响真实评价结果，但必须记录明确错误。

未来 T5 不能只预测 `|Δenergy|`：能耗近似不代表可行性、DDL 和 lateness 不变。
是否训练、使用何种模型、允许的错误上界和收益阈值都由独立任务评估。本轮不写
模型类、不新增依赖、不在诊断中加入 `surrogate_inferred` 处理逻辑。

---

## 诊断输出契约

T4 完成后只存在两类证据：

- `real`：真实仿真；
- `exact_identical`：完整决策轨迹证明与诊断基线恒等。

`parameter_diagnostics.py` 修改要求：

- `_local_effects` 每条 effect 增加 `evidence_source`；
- summary 分别统计 real 和 exact-identical；
- exact-identical 只能证明“在已检查的 contexts 和局部扰动范围内没有改变决策”，
  不得扩张为参数全局无效；
- boundary/correlation 逻辑不变；
- 所有局部比较使用 `diagnostic_baseline_metrics`。

---

## 耗时核算与交付报告

旧 SM/T 运行的 manifest 最终状态为 `FAILED`，因此 59.63 小时只能作为阶段预算，
不能称为完整成功条件的基准。旧的 49.2% 又是单 context 观测值，不能据此给出
“59.6 → 36.4 小时”或“九条件 22.4 → 13.7 天”的承诺。

统一使用以下公式：

```text
T_after = T_before
        - T0_measured_saving
        - vector_skip_rate × T_diagnostic_real
        + T_trace_capture
        + T_replay
        + T_replay_overhead_other
```

其中 `vector_skip_rate` 必须来自全部 contexts 的扰动级判定。报告 context 恒等率
仅用于诊断，不代入节省公式。

当前仅可保留以下待验证预算：

| 项目 | 预算值 | 状态 |
|---|---:|---|
| T0 修复并行化 | 最多约节省 18.2 h | 受 API 配额和批次结构影响，需实测 |
| T1 seed 对齐 | 约 0 h | 默认配置下 episode 数不变 |
| T2 基线轨迹捕获 | 约增加 0.55 h | 需实测 |
| T3/T4 避免真实诊断 | `vector_skip_rate × 13.0 h` | vector 跳过率尚未有效测出 |
| 重放计算 | 串行约 17.0 h；20 路理想约 0.85 h | 必须实测并发效率 |

每个性能数字必须注明：运行 ID、最终 manifest 状态、场景、seed、结构数、
context 数、worker 数、缓存命中数、墙钟时间和失败数。SL/LL 等场景分别测量，
不得假定 SM/T 的阶段占比保持不变。

最终交付说明必须包含：

1. 一个完整成功的 disabled 基线运行；
2. 同配置 enabled 运行；
3. vector 跳过率及按结构 bootstrap 的置信区间；
4. 0 个恒等反例；
5. trace capture、replay、真实仿真和总墙钟的分项对比；
6. 峰值内存和轨迹磁盘占用。

---

## 需要运行的测试

```bash
pytest tests/test_rule_optimization.py tests/test_cews_task_constructive.py tests/test_counterfactual_feedback.py tests/test_critical_state_replay.py tests/test_llm_concurrent_runs.py tests/test_evaluation_cache_journal.py tests/test_experiment_seed_protocols.py tests/test_llm_candidate_repair_parallel.py tests/test_diagnostic_seed_alignment.py tests/test_decision_trace_capture.py tests/test_replay_gate_exactness.py tests/test_replay_gate_wiring.py
```

测试完成后再运行修正后的测量脚本和全尺度验证脚本。不得用调低正确性要求、修改
argmin 语义或放宽恒等标准来满足时间目标。

## 明确不在本任务书范围内

- 删除 final admission；
- CMA refine 阶段晋级代理；
- 修改 `build_task_features` 或 `hrl_env.py` 性能路径；
- 学习型 surrogate 的训练、预测和上线；
- 对比算法和 Safe-HRL 训练逻辑的任何改动。
