# LLM 辅助安全分层强化学习改造进度

> 记录日期：2026-07-30  
> 当前阶段：阶段 18——最终代码审计与小规模验收  
> 本阶段代码边界：不新增核心安全算法；只审计真实调用链、修复明确缺陷、运行
> 原始/安全 HRL 与 CEWS 极小 smoke、验证 checkpoint 保存恢复并完善文档。
> 当前保存基线：`llm-safe-hrl-audited-baseline-20260730`，详见
> `docs/PROJECT_BASELINE_2026-07-30.md`。该名称是文档标识，不是 Git tag。

## 1. 当前分支与基线版本

- 当前分支：不可获取。项目根目录的 `.git/` 为空，`git status` 返回
  `not a git repository`，因此当前工作区没有可引用的 branch、commit 或 tag。
- 基线逻辑版本：`cews-pre-safe-rl-baseline-20260728`。
- 基线来源说明：现有快照文档记录源压缩包
  `aaf80c42-4639-4f97-98b9-bd70bab1e373.zip` 的 SHA-256 为
  `ea115f06aa716c5c1733da8546158dc289548188e09ef884d3b22b515df831ba`；
  该压缩包不在当前工作区，阶段 0 未能独立复算这个归档校验值。
- 关键代码清单指纹：对下列 14 个阶段 0 关键文件分别计算 SHA-256，按所列顺序拼接
  `"<file_sha256>  <relative_posix_path>"` 行并再次计算 SHA-256，得到
  `2b0cb419a4edd27b8b0062c27ea47cb26e325b80c7b3636fd89775f79d5e00f0`。
  该指纹用于发现后续关键代码漂移，不能代替完整 Git 提交或完整归档校验。

关键文件清单：

1. `main.py`
2. `LLM/main.py`
3. `LLM/seevo.py`
4. `LLM/cfg/problem/cews_task_constructive.yaml`
5. `LLM/problems/cews_task_constructive/eval.py`
6. `base/hrl_env.py`
7. `base/d3qn_agent.py`
8. `hrl_mix/train.py`
9. `hrl_mix/train_config.py`
10. `hrl_mix/train_runner.py`
11. `hrl_mix/train_eval.py`
12. `hrl_mix/train_utils.py`
13. `tests/test_cews_task_constructive.py`
14. `tests/test_fuzzy_energy.py`

## 2. 阶段 0 审计范围

已阅读和核对：

- 根入口 `main.py`；
- `LLM/main.py`、`LLM/seevo.py`；
- `LLM/problems/cews_task_constructive/eval.py`；
- `LLM/cfg/config.yaml` 与
  `LLM/cfg/problem/cews_task_constructive.yaml`；
- `base/hrl_env.py`、`base/d3qn_agent.py`；
- `hrl_mix/train.py`、`train_config.py`、`train_runner.py`、
  `train_eval.py`、`train_utils.py`；
- `run/hrl_mix/` 下 27 个场景/DDL 评估入口及其公共执行模式；
- `tests/` 下两个现有测试模块；
- `docs/CEWS_TASK_CONSTRUCTIVE_SNAPSHOT_2026-07-28_PRE_SAFE_RL.md`；
- `docs/LLM辅助安全分层强化学习改造步骤.docx`。任务中指定的同名
  `.md` 文件实际不存在，当前工作区只有 `.docx` 版本。

## 3. 当前架构确认

### 3.1 SeEvo / LLM 主线

真实调用链为：

```text
main.py
  -> LLM/main.py
  -> SeEvo.evolve()
  -> 为每个候选写入独立 candidate_*.py
  -> LLM/problems/cews_task_constructive/eval.py
  -> HrlFcfsCacheEnv
  -> RESULT_JSON
  -> SeEvo 可行性优先比较
```

确认结果：

- LLM 个体只实现现有 `get_task_priority_v2` 八参数接口。
- 候选函数只对当前 ready tasks 返回一维优先级分数，环境用 `np.argmin`
  选择任务。
- 候选函数不接收 Host、VM 或环境对象，不直接选择 Host 或 VM。
- 任务选定后，CEWS evaluator 调用环境的
  `select_vm_deterministic(task)` 完成固定 VM 选择，再调用
  `assign_task(task, vm)`。
- `get_task_priority_v2` 接口在阶段 0 没有修改。

### 3.2 原 HRL 主线

真实训练链为：

```text
python -m hrl_mix.train
  -> hrl_mix.train_runner.train()
  -> CloudWorkflowEnv_VMAgents（HrlFcfsCacheEnv 的兼容别名）
  -> Manager D3QN：phase 级调整 FCFS/SJF/MCF/HUR/EDF 五维组合权重
  -> Host D3QN：为当前任务选择 Host
  -> VM D3QN：在已选 Host 内选择空闲 VM 槽位
```

确认结果：

- 当前确实为 Manager–Host Agent–VM Agent 三层结构。
- 三层均实例化相互独立的 `D3QNAgent`。safe 模式下每层包含独立
  `Q_r/Q_c` online、target、optimizer；旧模式只构造原 `Q_r`。
- safe replay 包含独立 performance reward 与 safety cost；legacy replay
  继续使用 7 字段。可选 PER 当前仍只按性能 TD error 更新优先级。
- Manager 原 mask 仍只处理组合权重下界；Host/VM 原 mask 仍作为独立
  `legal_action_mask` 处理空闲性和槽位合法性。阶段 4 仅在显式开启时叠加
  `safety_action_mask`，Manager mask 和规则权重语义不变。
- 原 HRL 训练入口没有传入 fuzzy 参数，依赖
  `fuzzy_enabled=False` 的环境默认值，因此旧 HRL 仍按 modal 语义运行。

### 3.3 模糊目标、模糊 DDL 与可行性

确认结果：

- CEWS fuzzy 目标为
  `fuzzy_total_energy_mean + energy_uncertainty_weight * fuzzy_total_energy_std`。
- 当前 `energy_uncertainty_weight = 1.0`，即 `lambda_E = 1.0`。
- 模糊 DDL 风险测度实现为
  `upper - (1 - eta) * (upper - modal)`，等价于
  `(1 - eta) * modal + eta * upper`。
- 当前 `eta = 0.95`，因此为 `0.05 * modal + 0.95 * upper`。
- 工作流截止期 `D_i` 由到达时刻、FCFS cache 参考 makespan 和确定性 alpha
  生成，仍是确定标量，不是模糊截止期。
- SeEvo 候选比较采用可行性优先：
  可行个体先按能耗目标比较；不可行个体先比较违反率，再比较总延期，最后比较能耗。
- 需要严格区分：上述可行性优先属于 SeEvo/CEWS 候选评价；safe HRL 已将
  performance reward 与 DDL safety cost 分离，并可用动态拉格朗日权重评分，
  阶段 0 审计当时 best checkpoint 仍只按 `eval_energy` 保存；阶段 15 已将
  safe 模式改为可行性优先保存，legacy 模式仍保留原能耗保存语义。

### 3.4 固定 VM 规则

`base/hrl_env.py` 中的 `select_vm_deterministic(task)` 已存在并由 CEWS evaluator
实际调用：

- 存在按时 VM 时，优先最小边际能耗，再比较完成时刻和稳定 VM ID；
- 全部延期时，优先最小违反量，再比较能耗、完成时刻和稳定 VM ID；
- fuzzy 模式使用 eta 风险完成时刻和风险调整模糊边际能耗；
- modal 模式保留旧排序语义。

该规则具备确定性和稳定 tie-breaking。阶段 4 已增加可选候选 VM 子集参数；
阶段 5 又抽取其“给定指标字典序 + 稳定 VM ID”核心供安全回退复用。空安全集
回退使用动态 `D_safe` 违反量，不再直接用固定规则的静态任务子截止期；CEWS
原单参数调用、各分支字段顺序和选择语义不变。

## 4. 当前已实现功能

- SeEvo 生成和进化 ready-task 排序规则；
- `get_task_priority_v2` 八参数接口及严格输出校验；
- 候选文件隔离、子进程评价、`RESULT_JSON` 协议；
- CEWS 多 seed 评价与可行性优先候选比较；
- 三角模糊资源与 optimistic/modal/pessimistic 三场景重放；
- 风险调整模糊总能耗目标，`lambda_E = 1.0`；
- 模糊 DDL 风险测度，`eta = 0.95`，确定截止期；
- CEWS 固定、确定性的 VM 选择规则；
- 原 Manager–Host–VM 三层 D3QN、普通动作 mask、普通/PER replay；
- modal 旧 HRL 与 fuzzy CEWS 的开关式兼容；
- 默认关闭的 `safe_rl.enabled` 阶段配置和 `--safe-rl` 入口；
- 独立风险调整模糊能耗 `performance_reward`；
- 独立的完成违反、模糊超期和未完成过程风险 cost；
- 工作流完成事件去重及 phase/episode 安全诊断统计；
- 每个未完成工作流的 optimistic/modal/pessimistic/risk 完成时间预测；
- 动态模糊安全裕量、归一化裕量及全局风险聚合指标；
- 当前任务完成后的三场景剩余关键路径时间估计；
- 候选 VM 的模糊完成时刻、任务安全完成边界、裕量及预测违反量；
- 由 Host 内部当前可选 VM 预测聚合而成的 Host 风险诊断；
- 独立的 Host/VM 模糊 DDL safety shield；
- 分离记录的 legal/safety/final action mask；
- RL proposed/executed action、干预原因、预测风险和裕量记录；
- 空安全集合时按“动态模糊 DDL 违反量、模糊边际能耗、风险完成时间、稳定
  VM ID”在硬合法 VM 子集中确定性接管，并由入选 VM 唯一确定 Host；
- 显式版本化的三层安全 observation schema 和各字段归一化范围；
- Manager 11 项全局安全特征、每个 Host 6 项 Host 安全特征、每个候选 VM
  slot 12 项任务动作安全特征；
- 默认关闭的 `safe_rl.state.enabled` 子开关；关闭时三层 observation 维度和数值
  路径均保持旧语义；
- D3QN checkpoint 输入/输出维度元数据和旧 checkpoint 维度推断；维度不匹配时
  明确拒绝加载，包括 `strict=False`；
- Manager、Host、VM 各自独立的 `Q_r` online/target/optimizer 和
  `Q_c` online/target/optimizer，三层之间及同层两类价值之间均不共享参数；
- `Q_r` 使用 performance reward，`Q_c` 使用独立 safety cost 的 Double-DQN
  Bellman 更新和独立 loss；
- `score=Q_r-lambda_DDL*Q_c` 动作评分；动态开关关闭时继续使用阶段 7
  `initial_lagrange_multiplier=1.0` 固定权重；
- 默认关闭的共享 episode/EMA 拉格朗日控制器；显式开启时根据 episode
  平均环境安全转换 cost 更新一个全局 `lambda_DDL`，同步给三层独立 Agent；
- lambda warmup、更新间隔、EMA、上下界、非有限样本拒绝及诊断日志；
- 训练 `cost_budget` 与固定零违反评估合格线分离；
- safe 模式 Host/VM transition 链接到下一次同层决策以学习长期价值，Manager
  使用 phase transition；默认旧模式仍保留 Host/VM 单步终止 replay；
- safe 模式动作选择和两类 target 均使用 `final_action_mask`；空最终集合不调用
  Agent，由既有确定性控制器接管；
- safe checkpoint 分别保存、恢复两类 online/target/optimizer；
- 含 proposed/executed action、三类 mask、shield/fallback、裕量、风险、实际违反
  和 Manager phase ID 的版本化完整安全 replay，以及五类风险经验标签；
- FCFS/SJF/MCF/HUR/EDF 与已准入 SeEvo 规则组成的 heuristic-selection
  Manager，LLM 规则只排序 ready tasks；
- SeEvo 规则多 seed 零违反准入、来源/报告/配置 hash 和版本化 manifest；
- 与在线 replay 同 schema 的三层安全示范数据集、最终零违反安全标签、严格
  train/validation/final-test workflow/resource seed 隔离；
- 默认关闭的 Q_r/Q_c 离线预训练和可选非 fallback 行为克隆初始化；
- 可行性优先的多 seed best-checkpoint 排序及完整 bundle manifest；
- 分离 training/validation/final-test 的安全、控制、性能和 LLM 辅助指标
  schema 1，以及 per-seed/worst-seed 审计记录。

## 5. 当前未实现的安全强化学习功能

阶段 16 完成后，以下功能仍未实现或未完成正式实验验证：

- 对确定性回退控制器开展大规模场景与跨 seed 校准；
- 基于因果贡献或时间跨度的更细粒度三层 cost credit assignment；
- 动态 lambda、Q_c 与安全结果的完整训练收敛/灵敏度校准；
- 类别感知的边界/干预/实际违反经验采样比例控制；
- 真实 admitted SeEvo 规则的正式安全示范数据和跨 seed 离线预训练；
- 正式长训练、with/without LLM 严格配对消融和独立 final-test seed 评价。

## 6. 接口变化

- `get_task_priority_v2`：未修改。
- `HrlHeftEnv` 新增可选构造参数 `safe_rl_enabled=False` 和
  `safe_rl_process_risk_aggregation="mean"`，阶段 4 另新增
  `safe_rl_shield_enabled=False`；阶段 6 新增
  `safe_rl_state_enabled=False`、`safe_rl_state_high_uncertainty_threshold=0.2`
  和 `safe_rl_state_recent_record_window=100`。shield 和安全状态只能在
  safe/fuzzy 模式下显式开启；默认调用语义不变。
- `vm_assign()` 仍返回 `(r_host, r_vm, info)`；`finish_phase_and_advance()`
  仍返回 `(reward, info)`，未删除旧 reward。
- safe 模式的 `info` 新增 `performance_reward`，runner 显式选择该字段写入
  原 D3QN replay；`safe_rl.enabled=false` 时仍写入原返回 reward。
- `info` 新增稳定安全字段：
  `safety_cost`、`deadline_violation_cost`、`fuzzy_lateness_cost`、
  `process_risk_cost`、`deadline_violation_count`、
  `completed_workflow_count`，以及预测违反数、最小模糊安全裕量和累计统计。
- 新增只读接口 `get_dynamic_fuzzy_safety_margins()`，返回全部未完成工作流
  的统一安全裕量记录，以及 minimum/mean margin、风险工作流比例和预测违反率。
- 新增 `estimate_task_duration_components_scenario()`，在不改变原
  `estimate_task_duration_scenario()` 数值语义的前提下，分别返回当前任务的
  输入通信、执行、输出通信和总时长。
- 新增只读接口 `estimate_task_remaining_critical_path()`、
  `predict_task_vm_action_risk()`、`predict_task_host_action_risk()` 和
  `get_task_action_risk_predictions()`。最后一个接口仅在
  `safe_rl.enabled=true` 时生成当前任务的批量 Host/VM 预测；关闭时返回空预测。
- 所有任务级预测均返回 `prediction_only=true`、
  `action_mask_applied=false`；批量接口另返回
  `action_selection_changed=false`。这些仍是阶段 3 的只读预测接口。
- 新增独立类 `FuzzyDDLSafetyShield` 和环境接口
  `get_task_safety_action_masks()`、`get_safety_shield_records()`、
  `get_safety_shield_diagnostics()`。
- 新增独立类 `DeterministicFuzzyDDLFallbackController`；环境新增可选参数
  `safe_rl_fallback_controller="fixed_vm_rule"`，当前只支持这一兼容名称。
- Host/VM 状态字典新增 `legal_action_mask`、`safety_action_mask`、
  `final_action_mask`、`policy_action_mask` 和 `safety_fallback_required`；
  原 `mask` 字段保留。
- `vm_assign()` 返回元组不变，`info` 新增 proposed/executed action、
  `action_modified`、`shield_intervened`、`modification_reason`、
  `predicted_risk`、`safety_margin` 和 Host/VM 分层记录。
- 空安全集记录新增 `fallback_triggered`、`fallback_reason`、
  `candidate_count`、`minimum_violation`、`selected_host`、`selected_vm`
  和 `tie_break_stage`。
- `select_vm_deterministic(task)` 增加可选
  `candidate_vm_ids=None`；省略时 CEWS 原接口完全不变。
- safe 模式 phase CSV 新增 mean margin、risk workflow ratio 和 predicted
  violation rate；默认关闭时旧 CSV 表头仍完全不变。
- `hrl_mix.train` 新增 `--safe-rl`，默认不启用。启用时输出目录增加
  `_safeCostStage1_dualValueStage7_fixedLambda` 后缀，避免覆盖旧实验。
- `hrl_mix.train` 新增默认关闭的 `--safe-rl-shield`，且要求同时传入
  `--safe-rl`；启用后追加独立
  `_fuzzyDDLShieldStage4_dynamicFallbackStage5` 输出后缀。
- `hrl_mix.train` 新增默认关闭的 `--safe-rl-state`，且要求同时传入
  `--safe-rl`；启用后输出目录追加 `_safeStateStage6`。
- `hrl_mix.train` 新增默认关闭的 `--safe-rl-dynamic-lambda`，且要求同时
  传入 `--safe-rl`；启用后输出目录追加 `_dynamicLambdaStage8`。
- 新增 `HrlHeftEnv.get_observation_schema(layer=None)`，返回 schema 版本、
  legacy/safety/total 维度、重复次数、拼接布局、字段归一化范围和语义。
- legacy 维度保持为 Manager `15`、Host `22 + 5 * num_hosts`、VM
  `27 + 5 * max_vms_per_host`；安全状态开启后分别为 Manager `26`、Host
  `22 + 11 * num_hosts`、VM `27 + 17 * max_vms_per_host`。
- `SafeRLConfig` 新增 `safety_discount=0.95`、
  `safety_learning_rate=3e-4`、`safety_loss_weight=1.0` 和
  `initial_lagrange_multiplier=1.0`；阶段 8 新增嵌套
  `lagrangian.enabled/lambda_init/lambda_lr/lambda_min/lambda_max/`
  `cost_budget/update_interval/cost_ema_factor/warmup_steps`。
- `D3QNAgent` 新增同名安全参数及 `safe_rl_enabled=False`；旧
  `online/target/optim` 继续表示 Q_r，并新增
  `q_c_online/q_c_target/q_c_optim`。`update()` 返回值仍是 Q_r loss，
  两类 loss 和 target 诊断保存在 `last_update_info`。
- `D3QNAgent.remember()` 末尾新增可选 `cost=None`；旧模式保持原 7 字段 tuple，
  safe 模式要求有限 cost 并使用 8 字段 tuple。
- checkpoint schema 升级到 5，除 safe 模式、Q_c 三类 state 和安全超参数外，
  可嵌入共享拉格朗日控制器的 lambda、EMA、计数器及配置状态。
  performance-only 与 dual-value checkpoint 不能互相静默加载。
- Host/VM runner 在 safe 模式延迟写入 transition，取得下一次同层状态后以
  `done=0` bootstrap；episode 结束或训练截断才写零下一状态 `done=1`。
- `layer_learning_action_mask()` 和 `select_layer_action()` 保证 safe 模式只把
  `final_action_mask` 交给 Agent；空集合直接返回环境的确定性 fallback action，
  不把它伪装成 RL 动作写入 replay。
- 环境安全信息新增 `cumulative_safety_transition_count` 和
  `cumulative_mean_safety_cost`；原 safety cost 字段和 reward 返回值不变。
- `evaluate_hrl_three_layer_multi_seed(..., return_safety_metrics=False)` 新增可选
  安全评价返回；默认四元组接口不变，显式开启时以固定零违反率作为合格线，
  不继承非零训练 `cost_budget`。

## 7. 测试结果

### 7.1 权威现有测试运行

命令：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

环境：

- Windows 10.0.19045；
- Python 3.12.7；
- pytest 7.4.4；
- NumPy 1.26.4；
- PyYAML 6.0.1。

结果：

```text
82 passed, 15 skipped in 2.20s
```

- 成功：82；
- 测试断言失败：0；
- skipped：12；xfailed/xpassed：0；
- 原有 31 项 CEWS/fuzzy 回归全部通过；
- 新增 6 项阶段 1 测试，覆盖按时 cost、模糊超期 cost、未完成过程风险、
  完成事件去重、默认配置关闭和旧环境返回接口；
- 新增 6 项阶段 2 测试，覆盖宽松 DDL、边界 DDL、明显不可行 DDL、
  modal/upper 差异、无活动工作流和多工作流聚合；
- 新增 7 项阶段 3 测试，覆盖空剩余路径、单后继、多分支、零通信、跨 Host
  通信、VM 排队，以及安全边界/不安全动作和 Host 聚合；
- 新增 9 项阶段 4 测试，覆盖一个/多个安全 VM、Host 无安全 VM、全局空安全集、
  legal/safety 冲突、shield 关闭、安全提议不修改、不安全提议修正，以及真实环境
  固定 VM 回退接管与 Manager 权重不变。
- 新增 8 项阶段 5 测试，覆盖空安全集、违反量/能耗/风险完成时间平局、稳定
  VM ID、输入顺序确定性、多 Host 内部最佳 VM 聚合、安全动作存在和控制器关闭。
- 新增 6 项阶段 6 observation 测试，覆盖默认旧维度、配置依赖、三层新维度、
  数值有限性与归一化范围、schema 不含离散 ID，以及近期 shield/fallback 比率。
- 新增 8 项阶段 7 双价值测试，覆盖三层及 Q_r/Q_c 参数独立、两类 Bellman
  target、综合评分与 mask、Q_c cost 输入、双价值 checkpoint、旧模式兼容、
  final mask/回退边界和 Host/VM 下一同层状态 bootstrap。
- 新增 12 项阶段 8 测试，覆盖完整配置、预算上/下方更新、lambda 上下界、episode
  warmup/interval/EMA、非有限值防护、三层共享同步、checkpoint、旧模式关闭和
  固定零违反评估标准。
- 4 项 checkpoint schema/维度测试、8 项阶段 7 测试和 3 项阶段 8 集成测试在
  该 Anaconda 环境因缺少 PyTorch 被显式跳过；它们另在装有 PyTorch 的
  `D:\Python3.12\python.exe` 下以 `unittest` 运行并全部通过。

### 7.2 HRL 入口导入冒烟检查

使用 `D:\Python3.12\python.exe` 执行：

- `python -m hrl_mix.train --help`：通过，已显示默认关闭的 `--safe-rl`、
  `--safe-rl-shield`、`--safe-rl-state` 和
  `--safe-rl-dynamic-lambda`；
- `python -m unittest discover -s tests -v`：97 项全部通过；
- `python -m unittest tests.test_d3qn_checkpoint_dimensions
  tests.test_safe_dual_value_d3qn tests.test_safety_lagrange -v`：24 项全部
  通过；阶段 7 另完成真实
  25-task Montage 环境三层双价值 transition 冒烟，Manager/Host/VM replay
  均为 8 字段、Q_c loss 有限且三层 lambda 均保持 1.0；
- 导入 `base.hrl_env`、`base.d3qn_agent`、`hrl_mix.train_config`、
  `train_eval`、`train_runner`、`train_utils`：通过；
- 真实 25-task Montage safe 环境的当前任务动作风险查询：通过，得到 2 个 VM
  预测和 1 个 Host 聚合；调用前后 task state、ready 集、modal/shadow VM
  availability、事件堆、任务分配映射和当前时刻均保持不变。

该检查只证明入口、依赖和只读查询调用链可运行，不等同于完整训练 episode、
checkpoint 或模型评估通过。

### 7.3 缺失依赖与平台问题

- 项目 `.venv` 为 Python 3.12.7，虽有 pytest 8.3.5，但缺少 `numpy`；
  当前运行全量测试时八个测试模块均产生 collection error，0 项被收集。
- 该 `.venv` 同时未安装 `torch`、`gymnasium`、`hydra-core`、`openai` 和 `scipy`。
- 默认命令 `python` 指向 `D:\Python3.12\python.exe`，有 NumPy、PyTorch、
  Hydra、SciPy 和旧 `gym`，但没有 pytest。
- `D:\anaconda3\python.exe` 能完成 81 项测试，另有 15 项 D3QN/checkpoint
  测试因缺少 PyTorch 而跳过；它不能导入 `hrl_mix.train_runner`。
- 当前没有单一、已记录并验证的 Python 环境同时覆盖测试、HRL 训练和 SeEvo；
  `LLM/requirements.txt` 也未列出 HRL 必需的 PyTorch 与 gym 依赖。
- Windows 中文和空格路径当前由已有 UTF-8 子进程测试覆盖并通过；但
  `run/hrl_mix` 评估入口和完整训练未在本阶段做跨平台运行。

## 8. 发现的风险

### P0

1. 当前不是有效 Git 仓库，不能用 branch/commit/tag 真正冻结或追踪基线。
2. `LLM/main.py` 含硬编码 API 凭据。凭据内容不得进入文档、日志或后续提交；
   在任何共享或阶段 1 开发前应立即轮换，并改为仅从环境变量读取。

### P1

1. SeEvo/CEWS 默认使用 fuzzy 目标与 fuzzy DDL，原 HRL 默认使用 modal；
   两条主线尚未统一，也尚未集成。
2. legacy 模式仍保留原能耗/延期混合 reward；safe 模式已分离 Q_r/Q_c 并把
   cost 写入 replay，但 lambda 仍是固定值、尚无长期 cost 预算，best checkpoint
   仍只按 `eval_energy` 保存，因此尚不是完整拉格朗日安全强化学习。
3. 现有 73 项通过测试覆盖 CEWS、fuzzy 和阶段 1–6；另有 12 项
   D3QN/checkpoint 测试在 PyTorch 环境通过。但仍没有完整训练 episode、
   正式训练 checkpoint 或 27 个
   `run/hrl_mix` 入口的自动化回归测试。
4. FCFS deadline cache 的资源元数据与 CEWS 当前五 Host 云边资源规模不一致；
   候选间仍共享同一 DDL，但论文口径需要明确。
5. 阶段 5 已把 shield 回退违反量统一到阶段 3 动态 `D_safe`，并与 CEWS
   固定 VM 规则共享稳定字典序辅助函数；但完成时间和边际能耗仍是模型预测，
   需要通过大规模场景和跨 seed 实验继续校准。
6. `run/hrl_mix` 评估脚本存在大量重复代码；精确 checkpoint 不存在时会扫描
   其他目录回退，存在加载错误场景模型的风险。
7. 部分 `run/hrl_mix` 文件的注释/终端提示与实际
   `deadline_alpha_small_prob` 数值不一致，可能误导实验记录。
8. `common.workflow_opt.exec_time_components()` 已有同 VM/同 Host 父数据可免通信
   的可选近似，但当前 HRL 实际分配链仍对任务全部输入/输出按目标 VM 带宽计时。
   阶段 3 预测必须跟随实际运行时口径；若未来启用 locality，运行时与预测器必须
   同步修改，不能只改预测。
9. 任务级剩余关键路径对未来未提交任务不预占 VM，也不模拟并发工作流的后续竞争，
   可能低估真实剩余时长；输出只能作为模型风险预测。
10. 阶段 6 的安全 observation 复用上述模型预测；归一化限制了数值范围，但不把
    预测风险提升为真实执行后的 DDL 安全保证。旧 checkpoint 与新状态输入维度
    不兼容，必须在旧状态模式加载或为新状态重新训练。
11. 阶段 7 的 Host 与 VM 分别使用同一联合资源动作结果的 transition cost，
    Manager 使用 phase 内转换 cost 之和；这能避免跨层相加成单一样本，但尚未
    解决更细粒度的反事实 credit assignment。固定 lambda 也不保证 cost 预算满足。

### P2

1. 任务指定的改造步骤 `.md` 文件缺失，实际只有 `.docx`，自动化文档链路不统一。
2. 当前工作区未包含快照文档所述源压缩包，不能独立验证归档 SHA-256。
3. 生成候选、输出、缓存和 `__pycache__` 数量较多，当前无有效 `.gitignore`
   和版本控制边界。

## 9. 阶段 0 状态

- 代码审计：完成。
- 七项架构事实核对：完成，均与代码一致；其中“可行性优先”只适用于
  SeEvo/CEWS，“固定 VM 回退”目前只是可复用基础、尚未接入 HRL。
- 现有测试确认：完成，31/31 通过。
- 关键代码指纹：已记录。
- 业务逻辑修改：无。
- 安全强化学习实现：无。
- Git 级冻结：未完成，原因是 `.git/` 为空。

阶段 0 结论：审计和测试基线已确认；版本治理仍有 P0 前置项。只有在恢复原 Git
历史或经确认创建新仓库、保存当前快照提交，并轮换硬编码凭据后，才建议开始阶段 1
代码改造。不要直接在无法追溯的当前副本上实施后续功能。

## 10. 阶段 0 结束时记录的下一阶段目标（历史）

阶段 1 仅固定 CMDP 问题定义和兼容边界，不提前实现 reward/cost 分离、
safety shield、`Q_c`、拉格朗日乘子或安全 replay。开始前应先完成：

1. 建立可引用的基线 commit/tag 或恢复原 Git 历史；
2. 轮换并移除硬编码凭据；
3. 固化一套可同时运行测试和 HRL/SeEvo 的环境说明；
4. 明确 `safe_rl=false` 时必须保持的旧 HRL modal 行为；
5. 明确性能 reward、安全 cost、硬动作合法性和长期模糊 DDL 约束四者的定义边界。

## 11. 阶段 1：CMDP reward-cost 环境接口

### 11.1 数学定义

性能奖励仅在 `safe_rl.enabled=true` 时由 runner 消费，定义为一次调度动作引起的
风险调整模糊总能耗增量的负值：

```text
performance_reward_t
  = - energy_reward_scale
    * Delta(fuzzy_energy_mean + 1.0 * fuzzy_energy_std)
```

这没有修改 `lambda_E=1.0`。旧返回 reward 继续存在，用于默认关闭时保持历史训练
语义；DDL cost 不会再合并到 `performance_reward`。

对本次转换中首次结算的已完成工作流 `i`：

```text
R(T_i) = (1 - 0.95) * T_i_modal + 0.95 * T_i_upper
c_vio_i = 1[R(T_i) > D_i]
c_late_i = max(0, R(T_i) - D_i)
```

`D_i` 仍是确定值。对未完成工作流，先基于当前已提交任务时间线、VM 可用时刻、
DAG 依赖和三场景资源能力估计 `R(T_hat_i)`，再定义：

```text
c_risk_i = clip(
    (R(T_hat_i) - current_time) / (D_i - current_time),
    0,
    1
)
```

若 `current_time >= D_i` 且工作流未完成，则 `c_risk_i=1`。转换级
`process_risk_cost` 是所有未完成工作流 `c_risk_i` 的算术平均。

聚合规则固定且显式：

```text
deadline_violation_cost = sum(newly_completed c_vio_i)
fuzzy_lateness_cost = sum(newly_completed c_late_i)
process_risk_cost = mean(unfinished c_risk_i)
safety_cost = deadline_violation_cost
              + fuzzy_lateness_cost
              + process_risk_cost
```

三项分量分别保留，不将 `safety_cost` 加回 reward。这里 `fuzzy_lateness_cost`
单位为秒，其余分量无量纲；阶段 1 不引入隐藏归一化权重或拉格朗日乘子。

### 11.2 完成事件去重与诊断语义

- `_safety_accounted_workflow_ids` 在每个 episode reset 时清空；
- 一个工作流完成事件只在第一次安全诊断转换中结算 `c_vio/c_late`；
- 后续 VM 或 Manager 转换不会再次结算同一完成事件；
- 过程风险是当前状态的转换级快照，可以随状态变化在后续转换重新计算；
- `completed_workflow_count` 和 `deadline_violation_count` 是本次转换首次结算数；
  `cumulative_*` 字段保存 episode 内累计数；
- 三层 cost 的最终归属尚未实现，Host/VM/Manager 的 cost 不应在层之间直接相加。

过程完成时间预测目前采用不预占未来 VM 的最早可行三场景估计。它用于阶段 1
诊断和稠密反馈，不构成硬安全保证，也没有被用于动作屏蔽。

### 11.3 配置项

`hrl_mix.train_config.SafeRLConfig`：

```text
enabled = false
fuzzy_energy_uncertainty_weight = 1.0
fuzzy_deadline_eta = 0.95
process_risk_aggregation = "mean"
```

命令行使用 `--safe-rl` 显式开启。开启时 runner 同步启用模糊三场景跟踪；
未传该选项时，环境构造、旧 reward、D3QN replay tuple 和训练更新路径保持原样。

### 11.4 修改文件

1. `base/hrl_env.py`
2. `hrl_mix/train.py`
3. `hrl_mix/train_config.py`
4. `hrl_mix/train_runner.py`
5. `hrl_mix/train_eval.py`
6. `tests/test_safe_hrl_cost.py`
7. `docs/SAFE_HRL_PROGRESS.md`

`base/d3qn_agent.py`、LLM/SeEvo、CEWS evaluator、固定 VM 规则和
`get_task_priority_v2` 均未修改。

### 11.5 测试结果

运行命令：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`37 passed in 1.94s`，0 failed，0 skipped。另使用
`D:\Python3.12\python.exe` 完成：

- 五个修改 Python 文件的 `py_compile`；
- `python -m hrl_mix.train --help`；
- HRL 六模块导入冒烟；
- 一个 25-task Montage 工作流的 safe 模式完整环境冒烟：
  25 次分配、1 个工作流完成，`lambda_E=1.0`、`eta=0.95`。

项目 `.venv` 下同一命令未能收集测试：三个测试模块均因缺少 `numpy`
产生 `ModuleNotFoundError`。这是已有环境依赖问题，不是测试断言失败。

没有运行完整 D3QN 训练、checkpoint 评估或跨平台测试，不能声称这些项目通过。

### 11.6 阶段 1 状态与尚未实现功能

阶段 1 状态：完成并通过现有回归、新增单元测试及最小环境冒烟。

明确未实现：

- safety shield 和 Host/VM 安全动作集合；
- 固定 VM 规则作为 HRL 运行时回退控制器；
- `Q_c` 网络、双价值更新和 cost replay；
- 拉格朗日乘子或安全预算优化；
- 分层 cost 归因和安全状态扩展；
- LLM 规则接入 Manager；
- 可行性优先 checkpoint 与完整安全评价。

## 12. 阶段 1 结束时记录的下一阶段目标（历史）

阶段 2 只应建立动态模糊安全裕量的状态/统计边界和任务级安全边界定义，并补充
对应测试；不得提前实现 safety shield、`Q_c`、拉格朗日乘子或 LLM 接入。
开始前仍建议先解决无有效 Git 基线和硬编码凭据两个 P0 风险。

## 13. 阶段 2：工作流动态模糊安全裕量

### 13.1 统一定义

对每个已经到达但尚未完成的工作流 `i`，继续使用已有三场景资源和时间线预测：

```text
T_hat_i_lower = optimistic finish estimate
T_hat_i_modal = modal finish estimate
T_hat_i_upper = pessimistic finish estimate

R(T_hat_i)
  = (1 - 0.95) * T_hat_i_modal
    + 0.95 * T_hat_i_upper

S_i^F = D_i - R(T_hat_i)
```

没有修改 `eta=0.95`，`D_i` 仍为确定截止期。状态解释为：

```text
S_i^F > 0：仍有安全余量
S_i^F = 0：位于安全边界
S_i^F < 0：模型预测将违反 DDL
```

实现使用 `1e-9` 浮点容差识别边界。`is_predicted_at_risk` 包含边界和负裕量，
`is_predicted_violation` 只表示严格负裕量。

归一化口径为：

```text
normalized_fuzzy_safety_margin
  = clip(S_i^F / (D_i - arrival_i), -1, 1)
```

分母是工作流的确定性 DDL 总预算，不引入模糊截止期。

### 13.2 完成时间预测模型

预测严格复用 `base/hrl_env.py` 已有模型：

- 已 Running/Finished 任务使用实际已提交的 modal 完成时刻，以及同一映射下
  optimistic/pessimistic shadow 完成时刻；
- 未调度任务按现有 DAG 父子依赖做拓扑传播；
- 每个场景读取现有 `vm_available_at` 或对应
  `shadow_vm_available_at`；
- 任务时长调用现有 `estimate_task_duration_scenario()`，继续使用已有三角模糊
  pc/bw 分量；
- 每个未调度任务取当前 VM 状态下的最早可行预测完成时刻；
- 额外输出 finished/running/ready/remaining task 数、剩余 MI 和未完成任务
  maximum upward rank，供诊断当前工作流状态。

该预测不会为尚未提交的未来任务预占 VM，因而是基于当前状态的模型估计，
不是 safety shield，也不能描述成真实完成后的安全保证。

### 13.3 接口与输出字段

新增只读接口：

```python
env.get_dynamic_fuzzy_safety_margins()
```

每个未完成工作流至少输出：

```text
workflow_id
fuzzy_finish_estimate_lower
fuzzy_finish_estimate_modal
fuzzy_finish_estimate_upper
fuzzy_finish_risk
deadline
fuzzy_safety_margin
normalized_fuzzy_safety_margin
is_predicted_at_risk
```

同时输出 `is_at_safety_boundary`、`is_predicted_violation`、
`safety_status` 和任务进度诊断。

全局聚合：

```text
minimum_safety_margin
mean_safety_margin
risk_workflow_ratio
predicted_violation_rate
```

其中：

- `risk_workflow_ratio = count(S_i^F <= 0) / unfinished_count`；
- `predicted_violation_rate = count(S_i^F < 0) / unfinished_count`；
- 无活动工作流时，两种 margin 和两个 rate 都稳定返回 `0.0`。

`get_safety_diagnostics()` 复用该唯一接口，不再单独维护第二套未完成工作流预测。
接口仅在 `safe_rl.enabled=true` 时计算；关闭时返回空记录和零聚合，不改变旧训练。

### 13.4 修改文件

1. `base/hrl_env.py`
2. `hrl_mix/train_runner.py`
3. `tests/test_fuzzy_safety_margin.py`
4. `docs/SAFE_HRL_PROGRESS.md`

未修改：

- `base/d3qn_agent.py` 和 replay 结构；
- Host/VM/Manager 动作 mask 与动作选择；
- LLM/SeEvo、CEWS evaluator、固定 VM 规则；
- `get_task_priority_v2`；
- 模糊能耗和模糊 DDL 公式。

### 13.5 测试结果

最终全量命令：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`43 passed in 1.91s`，0 failed，0 skipped。

阶段 2 定向测试共 6 项，全部通过：

1. 宽松 DDL 的正裕量；
2. 风险完成时刻等于 DDL 的边界裕量；
3. 明显不可行 DDL 的负裕量；
4. modal 与 pessimistic 完成时刻保持不同；
5. 无活动工作流的零聚合；
6. 多工作流 minimum/mean/risk ratio/violation rate 聚合。

另完成：

- `base/hrl_env.py`、`hrl_mix/train_runner.py` 和新测试的 `py_compile`；
- 修改模块导入冒烟；
- 真实 25-task Montage 环境的只读查询冒烟：查询前后 task state、ready 集、
  modal/shadow VM availability 和 event heap 完全不变，且三场景预测值不同。

项目 `.venv` 的全量测试未能收集：4 个测试模块均因缺少 `numpy`
产生 `ModuleNotFoundError`。没有运行完整 D3QN 训练、checkpoint 评估或
跨平台测试，不能声称这些项目通过。

### 13.6 阶段 2 状态与风险

阶段 2 状态：完成并通过全量回归、指定单元测试和真实环境只读冒烟。

主要风险：

- 当前预测对未来未提交任务不预占 VM，可能低估后续资源竞争；
- 风险标签是预测标签，不等于工作流真实完成后的安全结论；
- margin 尚未加入 Agent 状态，也没有影响动作选择；
- 尚无任务级安全边界、动作屏蔽或安全回退接管。

## 14. 阶段 2 结束时记录的下一阶段目标（历史）

下一阶段只应基于本阶段统一工作流裕量定义任务级动态安全边界和剩余关键路径
风险时间，并增加对应测试；不得提前实现 safety shield、`Q_c`、拉格朗日乘子
或 LLM 接入。无有效 Git 基线和硬编码凭据两个 P0 风险仍建议优先处理。

## 15. 阶段 3：任务级动态安全完成边界与动作风险预测

### 15.1 数学定义

对当前任务 `v_ij`，先沿现有 DAG 后继关系估计当前任务完成后的三场景剩余
关键路径时间：

```text
T_rem_ij
  = (T_rem_lower, T_rem_modal, T_rem_upper)

R(T_rem_ij)
  = (1 - 0.95) * T_rem_modal
    + 0.95 * T_rem_upper

D_safe_ij = D_i - R(T_rem_ij)
```

`T_rem_ij` 明确不包含当前任务自身。对候选 VM 动作 `a`，当前任务自身的计算、
输入/输出通信和 VM 排队共同形成模糊完成时刻：

```text
F_hat_ij(a)
  = (F_lower(a), F_modal(a), F_upper(a))

R(F_hat_ij(a))
  = (1 - 0.95) * F_modal(a)
    + 0.95 * F_upper(a)

margin_ij(a)
  = D_safe_ij - R(F_hat_ij(a))

predicted_violation_amount_ij(a)
  = max(0, R(F_hat_ij(a)) - D_safe_ij)
```

动作预测安全条件为：

```text
R(F_hat_ij(a)) <= D_safe_ij
```

实现使用 `1e-9` 浮点容差，等号边界视为预测安全，同时单独输出
`is_at_safety_boundary=true`。`D_i` 仍为确定截止期，`eta` 仍为 `0.95`。

### 15.2 剩余关键路径估计

实现直接遍历环境当前 `task_children` DAG，不从文件名或离线模板重建第二套图：

- 从当前任务的直接后继开始递归，因此严格排除当前任务自身；
- 每个 optimistic/modal/pessimistic 场景继续读取现有三角模糊 VM
  `pc`/`bw` 对应分量；
- 每个未完成后继任务在该场景下取可行 VM 中最短的现有运行时总时长；
- 多分支取累计总时长最大的分支作为剩余关键路径；
- 已完成后继自身剩余时长记为 0，但继续检查其尚未完成的后继；
- 分别保留剩余执行时间、剩余通信时间、路径任务 ID 和诊断性 VM ID；
- `task_up_rank` 作为参考字段输出，但不以标量 rank 替代真实 DAG 路径遍历。

这个估计没有为未来任务预占 VM，故不包含未来工作流竞争或后继任务间的资源排队。
它是当前状态下的模糊关键路径模型预测，不是真实完工后的安全保证。

### 15.3 当前任务完成时间与通信口径

候选 VM 完成时刻复用现有 `estimate_task_finish_tfn()`：

- modal 场景读取 `current_time`、父任务完成时刻和 `vm_available_at`；
- optimistic/pessimistic 场景读取相同任务映射对应的两条
  `shadow_vm_available_at` 与 shadow parent finish；
- 当前任务执行与通信时长由新增的分量接口拆开输出；
- 忙 VM 可以被单独预测并显示三场景队列延迟，但不会被 Host 层误报为当前硬合法
  动作。

实际调用链存在一个需冻结的模型差异：

- `common.workflow_opt.exec_time_components()` 支持同 VM/同 Host 父数据传输为 0
  的可选 locality 近似；
- 当前 HRL 的 `estimate_comm_time()` 与 `_assign_task_to_specific_vm()` 并未接入
  该近似，而是对任务全部输入和输出按目标 VM 带宽计时；
- 阶段 3 预测严格跟随当前 HRL 运行时，避免产生比实际调度更乐观的安全预测；
- 同 VM、同 Host、跨 Host 和未知父位置的数据量仍作为诊断字段输出，但不会暗中
  改变运行时通信时长。

若后续决定启用 locality，必须同时修改调度运行时、影子时间线和风险预测，并增加
兼容开关；不能只修改安全预测器。

### 15.4 VM 与 Host 输出接口

新增只读接口：

```python
env.estimate_task_remaining_critical_path(task)
env.predict_task_vm_action_risk(task, vm)
env.predict_task_host_action_risk(task, host_id)
env.get_task_action_risk_predictions(task=None)
```

每个候选 VM 至少输出：

```text
optimistic_finish
modal_finish
pessimistic_finish
risk_finish
task_safe_deadline
safety_margin
predicted_violation_amount
```

另输出当前任务三场景执行/通信时间、三场景 VM 队列延迟、剩余关键路径各分量、
父任务位置诊断、`is_predicted_safe` 和 `is_at_safety_boundary`。

Host 层没有建立独立完成时间模型。它只聚合其内部当前可选 VM 的同一批预测，输出
候选/安全 VM 数量、安全比例、最小预测违反量、最大裕量和诊断性最佳 VM ID。
忙 VM 的预测保留在 `all_internal_vm_predictions`，但不计入当前 Host 动作候选。

批量接口由已有 `safe_rl.enabled` 控制：

- 关闭时返回空 Host/VM 预测，不改变旧 HRL 调用；
- 开启时默认读取当前 `_cur_tid`，也允许显式传入任务；
- 返回 `prediction_only=true`、`action_mask_applied=false`、
  `action_selection_changed=false`；
- 不执行动作，不修改 Host/VM mask，不调用固定 VM 回退规则。

### 15.5 修改文件

1. `base/hrl_env.py`
2. `tests/test_task_safety_boundary.py`
3. `docs/SAFE_HRL_PROGRESS.md`

未修改：

- `base/d3qn_agent.py`、D3QN 网络和 replay；
- `hrl_mix` 的 Manager/Host/VM 动作选择及现有 mask；
- LLM/SeEvo、CEWS evaluator、固定 VM 规则；
- `get_task_priority_v2`；
- 模糊能耗、模糊 DDL、reward 和 safety cost 数学定义。

### 15.6 测试结果

阶段 3 定向命令：

```powershell
D:\anaconda3\python.exe -m pytest tests\test_task_safety_boundary.py -q
```

结果：`7 passed in 0.29s`。覆盖：

1. 当前任务无后继时剩余关键路径为 0；
2. 单后继且当前任务执行时间与后继剩余时间严格分离；
3. 多分支 DAG 选择最长累计后继路径；
4. 输入/输出均为 0 时三场景通信时间为 0；
5. 已完成父任务位于另一 Host 时，跨 Host 数据量被识别，并按当前运行时口径计时；
6. VM 排队进入 optimistic/modal/pessimistic 完成时刻；
7. 安全边界等号、不安全动作预测违反量、Host 仅聚合 VM 结果，以及关闭
   `safe_rl` 时批量预测为空。

全量命令：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`50 passed in 1.97s`，0 failed，0 skipped。

另完成：

- `base/hrl_env.py` 与 `tests/test_task_safety_boundary.py` 的 `py_compile`；
- 修改模块导入冒烟；
- 真实 25-task Montage 环境的只读查询冒烟，查询前后状态与时间线不变；
- 项目 `.venv` 全量测试确认失败于收集阶段：5 个模块均缺少 `numpy`，
  不是测试断言失败。

没有运行完整 D3QN 训练、checkpoint 评估、27 个 `run/hrl_mix` 场景或跨平台
测试，不能声称这些项目通过。

### 15.7 阶段 3 状态与尚未实现功能

阶段 3 状态：任务级安全边界和候选动作风险预测已完成，并通过指定测试、全量回归
和真实环境只读冒烟。

仍未实现：

- 用预测结果正式构造或修改 Host/VM 安全动作集合；
- safety shield 或动作替换；
- 无安全动作时的固定 VM 回退接管；
- 任务级预测并入 Agent 状态；
- `Q_c`、cost replay、拉格朗日乘子或安全预算；
- 分层 cost 归因、LLM 接入与安全 checkpoint。

这些风险输出是模型预测，不是真实工作流完成后的安全保证。

## 16. 阶段 3 结束时记录的下一阶段目标（历史）

下一阶段可在不改变硬动作合法性定义的前提下，基于阶段 3 输出设计显式的 Host/VM
安全动作集合与诊断策略，并单独定义“存在安全动作”和“无安全动作”的行为；若要
正式启用 safety shield 或固定 VM 回退接管，必须通过新配置开关并继续保证
`safe_rl.enabled=false` 的旧训练语义。不得顺带实现 `Q_c`、拉格朗日乘子或
LLM 接入。

## 17. 阶段 4：Host/VM 模糊 DDL safety shield

### 17.1 安全动作集合与三类 mask

阶段 4 直接消费阶段 3 的候选 VM 预测：

```text
A_safe(s)
  = {a | R(F_hat(s,a)) <= D_safe}
```

每个决策层都明确区分：

```text
legal_action_mask   = 原有硬合法性约束
safety_action_mask  = 模糊 DDL 预测安全条件
final_action_mask   = legal_action_mask AND safety_action_mask
```

VM safety mask 按每个候选 VM 的 `is_predicted_safe` 构造，所有满足条件的 VM
都会保留，不只保留一个“最佳”动作。Host safety mask 不建立独立风险模型：
只有 Host 内至少存在一个同时满足 legal 和 safety 的 VM，该 Host 才是安全动作。
因此 Host 与 VM mask 使用同一批预测并保持一致。

shield 关闭时：

```text
final_action_mask = legal_action_mask
```

环境继续把原 legal mask 传给 Agent，原 Host/VM 执行路径和 Manager 语义不变。

### 17.2 独立屏蔽层

新增 `base/safety_shield.py`，其中 `FuzzyDDLSafetyShield` 独立负责：

- 规范化并组合 legal/safety/final mask；
- 从候选 VM 预测构造 VM safety mask；
- 从 Host 内安全合法 VM 数构造 Host safety mask；
- 接受原始安全动作；
- 把不安全或硬非法提议修正为确定性的安全动作；
- 安全集合为空时执行显式回退；
- 生成 proposed/executed action 与风险审计记录。

runner 不包含完成时间预测、mask 聚合或修正规则，只传入配置，并在当前单动作
replay 字段中使用实际 executed action。未扩展 replay tuple，也未加入 `Q_c`。

当存在多个可替换安全动作时，shield 使用稳定顺序：

```text
预测违反量最小
  -> 安全裕量最大
  -> 预测风险完成时间最小
  -> 动作编号最小
```

在安全集合内预测违反量均为 0，所以实际首先体现为裕量最大和风险完成时间最小。
该顺序只用于修正异常/外部不安全提议；正常 D3QN 已通过 final mask 在所有安全动作
中自行选择，不会被 shield 强制成单一动作。

### 17.3 空安全集合与固定 VM 回退

实际 D3QN 在全零 mask 下存在退化索引行为，因此不能把全零 mask 直接当作普通
RL 动作执行。阶段 4 采用以下显式协议：

1. `final_action_mask` 保持真实全零；
2. 状态中的兼容 `mask/policy_action_mask` 临时退化为 legal mask，只用于取得并
   记录 RL proposed action；
3. proposed action 不会直接执行；
4. 环境调用现有 `select_vm_deterministic()` 在当前硬合法、空闲 VM 子集中选择
   fallback VM；
5. Host 层执行该 VM 所属 Host，VM 层执行该 VM 槽位；
6. Host 和 VM 两次接管均记录 `fallback_applied=true` 与
   `modification_reason=no_safe_action_fallback`。

为复用固定规则，`select_vm_deterministic()` 新增可选
`candidate_vm_ids=None`。CEWS 继续使用原单参数调用，不改变 LLM 候选评价语义。

固定规则当前仍使用既有任务子截止期排序，而安全集合使用阶段 3 动态
`D_safe`。因此现阶段能保证确定性、硬合法和完整接管记录，但不能声称 fallback
一定最小化动态任务边界下的预测违反量；这是后续需要校准的模型口径风险。

### 17.4 动作记录与 Manager 边界

每个 Host/VM shield 决策至少记录：

```text
rl_proposed_action
executed_action
action_modified
shield_intervened
modification_reason
fallback_applied
predicted_risk
safety_margin
legal_action_mask
safety_action_mask
final_action_mask
```

其中：

- `action_modified` 严格表示 proposed 与 executed 数值是否不同；
- `shield_intervened` 表示提议是否被拒绝或由回退接管；
- 即使回退动作编号碰巧与提议相同，`action_modified=false`，
  `shield_intervened=true`，避免混淆“数值变化”和“控制权接管”。

`vm_assign()` 的 `info` 同时返回 Host/VM 分层记录；环境保存 episode 记录并在
phase 结束时向 Manager 信息提供 record/intervention/fallback 和分层修改计数。
Manager 的状态维度、动作 mask、五类规则权重和 `manager_apply_action()` 均未
修改，因此 Manager 本阶段只观察诊断结果，不接受硬安全屏蔽。

### 17.5 配置项

```text
safe_rl.enabled = false
safe_rl.shield.enabled = false
safe_rl.shield.fallback_controller = "fixed_vm_rule"
```

- shield 默认关闭；
- shield 开启要求 `safe_rl.enabled=true`，环境同时要求 fuzzy 三场景开启；
- CLI 使用 `--safe-rl --safe-rl-shield`；
- 只传 `--safe-rl` 仍保持阶段 1–3 的无屏蔽语义；
- shield 输出目录追加 `_fuzzyDDLShieldStage4`，避免覆盖旧实验。

### 17.6 修改文件

1. `base/safety_shield.py`
2. `base/hrl_env.py`
3. `hrl_mix/train_config.py`
4. `hrl_mix/train.py`
5. `hrl_mix/train_runner.py`
6. `tests/test_fuzzy_ddl_safety_shield.py`
7. `docs/SAFE_HRL_PROGRESS.md`

未修改：

- `base/d3qn_agent.py` 的网络结构、更新公式和 replay tuple；
- `hrl_mix/train_eval.py` 的策略选择逻辑；
- Manager 状态、动作 mask 和规则权重语义；
- LLM/SeEvo、CEWS evaluator 和 `get_task_priority_v2`；
- 模糊能耗、模糊 DDL、performance reward 和 safety cost 定义；
- `Q_c`、拉格朗日乘子和 LLM 接入。

### 17.7 测试结果

阶段 4 定向命令：

```powershell
D:\anaconda3\python.exe -m pytest tests\test_fuzzy_ddl_safety_shield.py -q
```

结果：`9 passed in 0.38s`，覆盖：

1. 只有一个安全 VM；
2. 多个安全 VM 全部保留；
3. Host 下没有安全 VM；
4. 所有 Host 均无安全 VM并触发回退；
5. legal mask 与 safety mask 冲突时按交集生成 final mask；
6. shield 关闭时 final mask 和执行动作保持 legacy 语义；
7. 原始动作安全时不修改；
8. 原始动作不安全时确定性修正到安全动作并记录风险与裕量；
9. 真实 25-task Montage 环境的全空安全集回退、记录、硬合法执行以及 Manager
   权重不变。

全量命令：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

当前结果：`59 passed in 1.96s`，0 failed，0 skipped。

另完成：

- 修改 Python 文件的 `py_compile`；
- 旧环境返回接口定向回归：`1 passed in 0.27s`；
- shield、环境和配置导入冒烟；
- 使用 `D:\Python3.12\python.exe -m hrl_mix.train --help` 导入训练 runner
  并确认两个安全开关均默认关闭；`D:\anaconda3\python.exe` 缺少 `torch`，
  因此同一入口命令在测试环境中不能运行；
- 项目 `.venv` 全量测试确认失败于收集阶段：6 个模块均缺少 `numpy`，
  不是测试断言失败。

没有运行完整 D3QN 训练、checkpoint 评估、27 个 `run/hrl_mix` 场景或跨平台
测试，不能声称这些项目通过。

### 17.8 阶段 4 状态与风险

阶段 4 状态：Host/VM 安全动作集合、动作修正、空安全集固定规则接管和分层记录已
实现，并通过指定测试、全量回归和真实环境集成测试。

主要风险：

- safety mask 依赖模型预测，不是真实完成后的安全保证；
- 剩余关键路径未预占未来 VM，仍可能低估资源竞争；
- 固定 VM fallback 的子截止期口径与动态 `D_safe` 尚未统一；
- 当前 replay 只有一个 action 字段，shield 开启时保存 executed action；
  proposed action 仅在环境记录中，尚未形成双动作安全 replay；
- `run/hrl_mix` 的 27 个独立脚本未新增 shield CLI，它们因环境默认关闭而维持旧
  行为；正式安全评估入口仍待后续统一。

## 18. 阶段 4 结束时记录的下一阶段目标（历史）

下一阶段应扩展三层安全状态，但继续保持 Manager 不做硬屏蔽：向 Manager 提供
全局 shield/裕量聚合，向 Host/VM 状态加入与当前动作候选对应的安全比例、风险与
裕量。不得顺带加入 `Q_c`、拉格朗日乘子或 LLM 接入。开始前还应决定是否先统一
固定 VM fallback 与动态 `D_safe` 的违反量排序口径。

## 19. 阶段 5：空安全动作集合的确定性回退控制器

### 19.1 回退定义与排序规则

回退只在以下条件同时满足时触发：

```text
safe_rl.enabled = true
safe_rl.shield.enabled = true
A_safe(s) = empty
```

对每个硬合法候选 VM `a`，复用阶段 3 的动态任务安全边界预测：

```text
violation(a) = max(0, R(F_hat(a)) - D_safe)

R(F_hat(a))
  = (1 - 0.95) * F_hat_modal(a)
    + 0.95 * F_hat_upper(a)
```

模糊边际能耗继续调用现有 `estimate_incremental_energy_score()`：

```text
fuzzy marginal energy
  = fuzzy energy mean + 1.0 * fuzzy energy std
```

最终严格按以下字典序选择：

```text
predicted_violation_amount
  -> fuzzy_marginal_energy
  -> risk_finish
  -> stable vm_id
```

没有修改 `eta=0.95`、`lambda_E=1.0` 或确定性工作流截止期。

### 19.2 与 CEWS 固定 VM 规则的复用关系

新增公共辅助函数 `select_vm_candidate_by_fixed_rule_order()`，统一实现：

1. 按调用方给定的指标优先级构造字典序；
2. 始终追加稳定 `vm_id` 作为最后平局键。

CEWS 的 `select_vm_deterministic()` 已机械替换为调用该函数，各按时/延期分支
原有字段顺序完全保留。安全回退控制器使用同一函数，但其违反量来自动态
`D_safe`，从而解决阶段 4 使用静态任务子截止期的口径冲突。LLM/SeEvo 调用、
`get_task_priority_v2` 和原固定 VM 规则的对外接口均未改变。

### 19.3 独立控制器与 Host 聚合

新增 `base/safety_fallback.py`：

```python
DeterministicFuzzyDDLFallbackController(enabled=False)
```

控制器只消费候选字典，不读取或修改环境状态，也不依赖 D3QN/runner。它先在每个
Host 内按完整字典序选出内部最佳 VM，再对各 Host 代表 VM 使用同一字典序选择
全局最佳项。因此：

```text
selected_vm = 全局最优的 Host 内部最佳 VM
selected_host = selected_vm 所属 Host
```

Host 不使用随机或独立于 VM 的回退估计。两级选择与直接取全部 VM 的全局最小值
数学等价，但显式保留 Host–VM 归属和审计信息。

### 19.4 记录字段

Host、VM 分层决策记录和 `vm_assign()` 的 `info` 新增：

```text
fallback_triggered
fallback_reason
candidate_count
minimum_violation
selected_host
selected_vm
tie_break_stage
```

`tie_break_stage` 取值为：

```text
single_candidate
minimum_violation
fuzzy_marginal_energy
risk_finish_time
stable_vm_id
not_triggered
```

完整嵌套记录另保留 `host_best_candidates`、`selected_candidate` 和排序规则说明。
正常存在安全动作时返回 `fallback_triggered=false` 和
`fallback_reason=safe_action_available`；控制器关闭时返回
`fallback_reason=fallback_controller_disabled`。

### 19.5 配置、兼容性与接口

现有配置继续使用：

```text
safe_rl.enabled = false
safe_rl.shield.enabled = false
safe_rl.shield.fallback_controller = "fixed_vm_rule"
```

- 默认两个安全开关均关闭，原 HRL 动作执行行为不变；
- shield 关闭时新控制器也关闭，不会覆盖旧动作；
- `safe_rl_fallback_controller="fixed_vm_rule"` 被传入环境，其他名称当前显式
  拒绝，避免静默采用未知回退语义；
- 正常安全动作集合非空时不计算回退用模糊边际能耗，也不替换 RL 动作；
- shield 实验输出目录后缀更新为
  `_fuzzyDDLShieldStage4_dynamicFallbackStage5`，避免覆盖阶段 4 结果；
- Manager 状态、规则权重、动作 mask 和 replay 均未修改；
- D3QN 网络结构、`Q_c` 和拉格朗日乘子均未实现。

### 19.6 修改文件

1. `base/safety_fallback.py`（新增）
2. `base/hrl_env.py`
3. `hrl_mix/train_config.py`
4. `hrl_mix/train_runner.py`
5. `tests/test_fuzzy_ddl_fallback_controller.py`（新增）
6. `tests/test_fuzzy_ddl_safety_shield.py`
7. `docs/SAFE_HRL_PROGRESS.md`

未修改：

- `base/d3qn_agent.py`；
- `base/safety_shield.py` 的动作 mask 和修正逻辑；
- `hrl_mix/train_eval.py`；
- LLM/SeEvo、CEWS evaluator 和 `get_task_priority_v2`；
- reward、cost、模糊能耗和模糊 DDL 数学定义。

### 19.7 测试结果

阶段 5 定向命令：

```powershell
D:\anaconda3\python.exe -m pytest tests\test_fuzzy_ddl_fallback_controller.py -q
```

结果：`8 passed in 0.01s`，覆盖：

1. 无安全 VM 时按最小违反量触发；
2. 违反量平局由模糊边际能耗决定；
3. 能耗平局由风险完成时间决定；
4. 风险完成时间平局由稳定 VM ID 决定；
5. 稳定 ID 结果与候选输入顺序无关；
6. 多 Host 先聚合各 Host 内部最佳 VM；
7. 存在安全动作时不触发；
8. 控制器关闭时不覆盖旧行为。

阶段 4 shield 与阶段 5 回退联合定向结果：`17 passed in 0.33s`。

共享固定规则、fuzzy 和回退相关回归结果：`48 passed in 1.98s`。

全量命令：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`67 passed in 1.99s`，0 failed，0 skipped。

另完成：

- 修改 Python 文件的 `py_compile`；
- 旧环境返回接口定向回归：`1 passed in 0.28s`；
- 使用 `D:\Python3.12\python.exe` 导入环境、D3QN、训练 runner、配置和新控制器；
- 项目 `.venv` 仍在收集阶段因缺少 `numpy` 产生 6 个模块错误。

没有运行完整 D3QN 训练、checkpoint 评估、27 个独立评估入口或跨平台测试。

### 19.8 阶段 5 状态与风险

阶段 5 状态：独立动态模糊 DDL 回退控制器、固定规则核心复用、Host/VM 两级选择
和完整记录已实现并通过指定测试与全量回归。

主要风险：

- 违反量、模糊边际能耗和风险完成时间均来自模型预测，不是真实完成后的安全保证；
- 剩余关键路径仍不预占未来 VM，可能低估后续资源竞争；
- 空安全集合意味着不存在预测安全动作，回退只能最小化预测违反量，不能保证
  实际执行满足 DDL；
- 当前 replay 只保存 executed action，完整回退记录尚未进入安全 replay；
- 当前没有单一 Python 环境同时覆盖 pytest、PyTorch 训练和 SeEvo。

## 20. 阶段 5 结束时记录的下一阶段目标（历史）

下一阶段可按改造步骤扩展三层安全状态，但继续保持 Manager 不做硬屏蔽；不得顺带
实现 `Q_c`、拉格朗日乘子或 LLM 接入。正式训练前仍建议统一 Python 环境，并用
多场景、多 seed 验证回退控制器的触发率和真实 DDL 结果。

## 21. 阶段 6：三层安全 observation 扩展

### 21.1 开关、职责和维度

阶段 6 新增独立子开关：

```text
safe_rl.enabled = false
safe_rl.state.enabled = false
safe_rl.state.high_uncertainty_threshold = 0.2
safe_rl.state.recent_record_window = 100
```

命令行入口为 `--safe-rl-state`，必须同时启用 `--safe-rl`。默认不开启，因而
`safe_rl.enabled=false` 或 `safe_rl.state.enabled=false` 时仍使用原 observation：

```text
Manager legacy dim = 15
Host legacy dim    = 22 + 5 * num_hosts
VM legacy dim      = 27 + 5 * max_vms_per_host
```

显式启用安全状态后的维度为：

```text
Manager safe dim = 15 + 11 = 26
Host safe dim    = 22 + (5 + 6) * num_hosts
VM safe dim      = 27 + (5 + 12) * max_vms_per_host
```

Manager 仍只选择任务调度规则，Host Agent 仍只选择 Host，VM Agent 仍只选择 VM。
本阶段没有改变任何动作数、D3QN 输出层、动作 mask、Manager 权重语义或环境调度行为。

### 21.2 Manager 安全状态

Manager 在原 15 维状态后追加以下 11 维：

| 字段 | 范围 | 归一化口径 |
|---|---:|---|
| `minimum_fuzzy_safety_margin_norm` | `[-1, 1]` | 所有未完成工作流归一化裕量的最小值 |
| `mean_fuzzy_safety_margin_norm` | `[-1, 1]` | 所有未完成工作流归一化裕量的均值 |
| `risk_workflow_ratio` | `[0, 1]` | `S_i^F <= 0` 的未完成工作流比例 |
| `predicted_ddl_violation_rate` | `[0, 1]` | `S_i^F < 0` 的未完成工作流比例 |
| `high_uncertainty_task_ratio` | `[0, 1]` | 最佳可行 VM 相对时长跨度超过阈值的 ready-task 比例 |
| `cloud_congestion` | `[0, 1]` | 云端 busy VM 数 / 云端 VM 数 |
| `edge_congestion` | `[0, 1]` | 边缘端 busy VM 数 / 边缘端 VM 数 |
| `safe_host_ratio` | `[0, 1]` | ready tasks 的硬合法 Host 中含安全 VM 的平均比例 |
| `mean_safe_vm_ratio` | `[0, 1]` | 合法 task-Host 对内安全 VM 比例的均值 |
| `recent_shield_intervention_rate` | `[0, 1]` | 最近配置窗口内 shield 干预记录比例 |
| `recent_fallback_trigger_rate` | `[0, 1]` | 最近配置窗口内回退触发记录比例 |

无活动对象或对应平台无 VM 时使用稳定的 `0.0`，不会产生 `NaN` 或无穷值。

### 21.3 Host Agent 安全状态

对每个 Host，在原每 Host 5 维后追加以下 6 维；Host 风险继续由该 Host 内候选 VM
的阶段 3 预测聚合，不建立与 VM 层无关的第二套动作风险模型：

| 字段 | 范围 | 归一化口径 |
|---|---:|---|
| `safe_vm_count_norm` | `[0, 1]` | 硬合法且预测安全 VM 数 / 每 Host 最大 VM slot 数 |
| `safe_vm_ratio` | `[0, 1]` | 安全 VM 数 / 当前可选 VM 数 |
| `host_queue_risk` | `[0, 1]` | 平均正 VM 可用延迟 / horizon 后裁剪 |
| `minimum_fuzzy_safety_margin_after_host_norm` | `[-1, 1]` | Host 内当前可选 VM 的最小任务裕量 / 工作流 DDL 总预算 |
| `cross_platform_communication_risk` | `[0, 1]` | cloud/edge 跨平台父数据量 / 总父输入数据量 |
| `future_critical_resource_occupation_risk` | `[0, 1]` | 预计 Host busy 比例 × 剩余关键路径压力 |

### 21.4 VM Agent 安全状态

对所选 Host 的每个 VM slot，在原每 slot 5 维后追加以下 12 维。finish、queue、
边际能耗、任务安全边界和剩余关键路径均复用现有三角模糊处理能力、带宽和三条
影子时间线：

| 字段 | 范围 | 归一化口径 |
|---|---:|---|
| `optimistic_finish_time_norm` | `[0, 1]` | `(optimistic finish - now) / DDL 总预算` 后裁剪 |
| `modal_finish_time_norm` | `[0, 1]` | `(modal finish - now) / DDL 总预算` 后裁剪 |
| `pessimistic_finish_time_norm` | `[0, 1]` | `(pessimistic finish - now) / DDL 总预算` 后裁剪 |
| `risk_finish_time_norm` | `[0, 1]` | `(R(finish) - now) / DDL 总预算` 后裁剪 |
| `task_safe_deadline_norm` | `[-1, 1]` | `(D_safe - now) / DDL 总预算` 后裁剪 |
| `fuzzy_safety_margin_horizon_norm` | `[-1, 1]` | 任务级模糊安全裕量 / horizon |
| `normalized_fuzzy_safety_margin` | `[-1, 1]` | 任务级模糊安全裕量 / 工作流 DDL 总预算 |
| `fuzzy_marginal_energy_norm` | `[0, 1]` | 风险调整边际能耗的有界单调变换 |
| `queue_time_norm` | `[0, 1]` | 模态 VM 排队时间 / horizon |
| `computation_uncertainty` | `[0, 1]` | `(pc_upper - pc_lower) / pc_modal` 后裁剪 |
| `bandwidth_uncertainty` | `[0, 1]` | `(bw_upper - bw_lower) / bw_modal` 后裁剪 |
| `remaining_critical_path_risk_time_norm` | `[0, 1]` | 剩余关键路径 `eta` 风险时间 / DDL 总预算 |

`R(finish)` 和剩余关键路径风险继续使用
`(1 - 0.95) * modal + 0.95 * upper`；截止期仍是确定值。VM padding slot
使用全零安全扩展，不把 task、Host 或 VM 的离散 ID 作为连续状态。

### 21.5 Schema 和 checkpoint 兼容

新增 `base/safety_observation.py`，schema 版本为 `safe_observation_v1`。每个字段
记录稳定名称、上下界、归一化公式和说明。环境接口：

```python
env.get_observation_schema()
env.get_observation_schema("manager")
env.get_observation_schema("host")
env.get_observation_schema("vm")
```

返回 legacy/safety/total 维度、重复次数、拼接布局和“不含离散 ID”说明。
环境在构造每层 observation 后校验长度、有限性和安全扩展范围。

新保存的 D3QN checkpoint 带 `input_dim`、`output_dim`、checkpoint schema 和
observation schema 版本。加载新旧 checkpoint 前都会核对 observation schema 和
网络维度；旧格式从 `feature.0.weight` 和 advantage 输出层推断维度，但因没有
observation schema 元数据，只允许在 legacy observation 模式加载。schema 或维度
不一致时明确抛出 `ValueError`，即使调用者传入 `strict=False` 也不会静默加载。
安全状态模式需要重新训练相应输入层。D3QN 网络结构和输出维度没有修改。

### 21.6 修改文件

1. `base/safety_observation.py`（新增）
2. `base/hrl_env.py`
3. `base/d3qn_agent.py`
4. `hrl_mix/train_config.py`
5. `hrl_mix/train.py`
6. `hrl_mix/train_runner.py`
7. `tests/test_safe_observation_state.py`（新增）
8. `tests/test_d3qn_checkpoint_dimensions.py`（新增）
9. `docs/SAFE_HRL_PROGRESS.md`

未修改 LLM/SeEvo、CEWS evaluator、`get_task_priority_v2`、固定 VM 规则、
safety shield、回退排序、replay tuple、D3QN 输出层或任何调度策略。

### 21.7 测试结果

全量测试：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`73 passed, 4 skipped in 2.09s`。4 项跳过均因该环境没有 PyTorch，分别是
同 schema/维度 checkpoint 加载、新格式维度不兼容拒绝、旧格式维度推断拒绝，
以及旧格式在安全 observation 模式下拒绝。

PyTorch 环境定向测试：

```powershell
D:\Python3.12\python.exe -m unittest tests.test_d3qn_checkpoint_dimensions -v
```

结果：`4 tests OK in 0.598s`。

阶段 6 observation 定向测试：

```powershell
D:\anaconda3\python.exe -m pytest tests/test_safe_observation_state.py -q
```

结果：`6 passed in 0.36s`。此外，CLI help、配置构造、修改文件语法编译和真实
25-task Montage 环境冒烟均通过；真实环境安全维度为 Manager `26`、Host `44`、
VM `61`（2 Host、每 Host 2 VM slot），全部状态值有限且安全扩展位于 schema
上下界内。

项目 `.venv` 的全量测试未能收集：7 个模块因缺少 `numpy` 报错。这是环境依赖
缺失，不计为断言失败，也不能声称该环境通过。没有运行完整 D3QN 训练、训练
checkpoint 评估、27 个独立评估入口或跨平台测试。

### 21.8 阶段 6 状态与风险

阶段 6 状态：完成。三层安全 observation、schema、归一化、默认旧维度兼容和
checkpoint 维度拒绝策略均已实现，并通过单元测试、全量回归和真实环境冒烟。

这些 finish、裕量、拥塞、通信和未来资源占用字段是基于当前调度状态与现有模糊
模型的预测，不是真实完成后的安全保证。尤其是剩余关键路径不预占未来 VM，也不
完整模拟并发工作流后续竞争，预测可能偏乐观。新状态会使旧 checkpoint 输入维度
不兼容；当前的明确拒绝是预期安全行为。

## 22. 阶段 6 结束时记录的下一阶段目标（历史）

下一阶段可只设计三层 transition 的 reward/cost 责任归属和安全 replay 数据边界，
继续保持 Manager–Host–VM 决策职责、模糊公式、SeEvo 接口、固定 VM 回退及默认
旧模式语义。不得把 cost 重新并入 reward，也不得在未单独批准的阶段提前实现
`Q_c`、拉格朗日乘子或 LLM 直接 Host/VM 选择。

## 23. 阶段 7：三层独立 Q_r/Q_c 安全价值学习

### 23.1 双价值结构

Manager、Host Agent 和 VM Agent 各自构造一个独立 `D3QNAgent`。safe 模式下，
每个 Agent 内包含：

```text
Q_r online network
Q_r target network
Q_r optimizer

Q_c online network
Q_c target network
Q_c optimizer
```

原 `online/target/optim` 属性继续表示 Q_r，以保持旧调用兼容；Q_c 使用
`q_c_online/q_c_target/q_c_optim`。Q_r 与 Q_c 不共享任何参数对象，三个 Agent
之间也不共享网络或 optimizer。

### 23.2 Bellman 目标与 loss

safe 模式的下一动作由当前固定受约束策略在下一状态最终动作集合内选择：

```text
a_next = argmax_{a in final_action_mask(s_next)}
         [Q_r_online(s_next, a)
          - lambda_DDL * Q_c_online(s_next, a)]

y_r = performance_reward
      + (1 - done) * gamma_r
        * Q_r_target(s_next, a_next)

y_c = safety_cost
      + (1 - done) * gamma_c
        * Q_c_target(s_next, a_next)
```

Q_r 和 Q_c 分别使用 Huber loss、独立 optimizer 和独立梯度裁剪。实际 Q_c
反向 loss 为：

```text
weighted_safety_loss = safety_loss_weight * safety_loss
```

`update()` 为兼容旧 runner 继续返回 Q_r loss 浮点数；完整
`performance_loss/safety_loss/target/TD error` 保存在 `last_update_info`。
当前 PER priority 继续只使用 Q_r TD error，风险经验重采样留到后续阶段。

### 23.3 固定 lambda 与动作 mask

本阶段只使用固定：

```text
lambda_DDL = initial_lagrange_multiplier = 1.0
score(s, a) = Q_r(s, a) - lambda_DDL * Q_c(s, a)
```

代码中没有 lambda 增减或预算反馈更新。mask 规则为：

- Host/VM 当前策略选择只接收 `final_action_mask`；
- epsilon 随机动作也只从该 mask 的有效动作中抽取；
- Double DQN online 下一动作选择使用同一 mask；
- Q_r/Q_c target 取值也应用同一 mask；
- 下一状态全零 final mask 时两类 bootstrap 都为 0；
- 当前 final mask 为空时不调用 Agent，由阶段 5 确定性回退控制器接管；
- 回退动作不被伪装成全零 final mask 下的 RL replay 动作；
- Manager 仍不做模糊 DDL 硬屏蔽，其现有合法动作 mask 是 Manager 的最终策略
  mask，不改变五种规则权重语义。

### 23.4 三层 reward/cost transition

```text
Host transition:
  reward = info_task.performance_reward_host
  cost   = info_task.safety_cost

VM transition:
  reward = info_task.performance_reward_vm
  cost   = info_task.safety_cost

Manager transition:
  reward = pinfo.performance_reward
  cost   = sum(safety_cost of transitions in the phase)
```

Host 和 VM 是联合资源动作的两层决策，因此分别在自己的 replay 中学习同一动作
结果的安全 cost；这些 replay 不跨 Agent 相加。Manager 每个 phase 只写一个聚合
cost。环境原有完成工作流去重集合继续保证最终违反和模糊超期完成事件不会在后续
转换重复结算。

旧 Host/VM runner 把每个任务 transition 固定为 `done=1`，无法学习长期价值。
阶段 7 只在 safe 模式改为保留 pending transition，在下一次同层决策出现时写入
真实 next state、next final mask 和 `done=0`；episode 完成或训练截断才以
`done=1` 结束。`safe_rl.enabled=false` 时仍使用原单步终止 transition。

### 23.5 Replay 与配置

legacy replay tuple 保持 7 字段：

```text
(state, mask, action, reward, next_state, next_mask, done)
```

safe replay 在本阶段只增加训练 Q_c 必需的 cost：

```text
(state, final_mask, executed_action, performance_reward,
 safety_cost, next_state, next_final_mask, done)
```

原始/执行动作双字段、shield 干预、裕量、风险、phase ID 和风险分层采样尚未提前
实现。

新增配置：

```text
safe_rl.enabled = false
safe_rl.safety_discount = 0.95
safe_rl.safety_learning_rate = 3e-4
safe_rl.safety_loss_weight = 1.0
safe_rl.initial_lagrange_multiplier = 1.0
```

`safe_rl.enabled=false` 时不构造 Q_c，动作评分、7 字段 replay 和 Q_r 更新保持旧
路径。safe 输出目录新增 `_dualValueStage7_fixedLambda` 标识。

### 23.6 Checkpoint

checkpoint schema 升级为版本 4。safe checkpoint 分别保存：

```text
online / target / optim
q_c_online / q_c_target / q_c_optim
safe_rl_enabled
safety_discount
safety_learning_rate
safety_loss_weight
lagrange_multiplier
```

加载时继续检查 observation schema、输入维度和动作维度，并新增 safe/legacy
模式检查。performance-only checkpoint 不能初始化 safe Agent；safe checkpoint
也不能静默丢弃 Q_c 后加载到 legacy Agent。`strict=False` 不能绕过这些检查。

### 23.7 修改文件

1. `base/d3qn_agent.py`
2. `base/hrl_env.py`
3. `hrl_mix/train_config.py`
4. `hrl_mix/train.py`
5. `hrl_mix/train_runner.py`
6. `hrl_mix/train_eval.py`
7. `hrl_mix/train_utils.py`
8. `tests/test_safe_dual_value_d3qn.py`（新增）
9. `tests/test_safe_observation_state.py`
10. `docs/SAFE_HRL_PROGRESS.md`

未修改 LLM/SeEvo、CEWS evaluator、`get_task_priority_v2`、模糊能耗与 DDL
公式、固定 VM 回退排序、D3QN 动作输出维度或 Manager–Host–VM 决策职责。

### 23.8 测试结果

全量测试：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`73 passed, 12 skipped in 2.08s`。12 项跳过均因该环境没有 PyTorch。

PyTorch 环境定向测试：

```powershell
D:\Python3.12\python.exe -m unittest `
  tests.test_d3qn_checkpoint_dimensions `
  tests.test_safe_dual_value_d3qn -v
```

结果：`12 tests OK in 0.727s`，其中阶段 7 新增 8 项，覆盖：

1. 三个 Agent 及同层 Q_r/Q_c 参数完全独立；
2. Q_r/Q_c Double-DQN target 和 final mask；
3. 当前动作的 `Q_r-lambda*Q_c` 评分及 mask；
4. Q_c target 使用 cost 而不是 performance reward；
5. 双价值 online/target/optimizer checkpoint 保存恢复；
6. safe 关闭时 Q_r 与 7 字段 replay 兼容；
7. final mask 传递与空安全集不调用 Agent；
8. Host/VM pending transition 使用真实下一同层状态 `done=0`。

另完成：

- 修改 Python 文件 `py_compile`：通过；
- `python -m hrl_mix.train --help`：通过；
- safe/legacy 配置构造及独立输出后缀冒烟：通过；
- 真实 25-task Montage 三层双价值 transition 冒烟：Manager/Host/VM replay
  均为 8 字段，两类 loss 有限，动作使用 final mask，三层固定 lambda 均为 1.0。

项目 `.venv` 因缺少 `numpy` 在收集阶段产生 8 个模块错误；没有运行完整训练
episode、正式训练 checkpoint 评估、27 个独立评估入口或跨平台测试，不能声称
这些项目已通过。

### 23.9 阶段 7 状态与风险

阶段 7 状态：完成。三层独立 Q_r/Q_c、cost Bellman 更新、固定 lambda 综合评分、
final mask、长期同层 transition 和双价值 checkpoint 已实现并通过定向测试、
全量回归及真实环境小规模冒烟。

主要风险：

- lambda 固定为 1.0，没有 cost 预算反馈，不能保证长期约束满足；
- Host/VM 使用联合动作结果的相同 cost，尚无更细的反事实安全归因；
- 过程风险 cost 是模型预测，不是真实完成后的安全保证；
- 空安全集的 fallback 经验当前不进入 RL replay，后续完整安全 replay 阶段需要
  显式记录并决定其训练用途；
- best checkpoint 仍按能耗保存，尚未改为可行性优先；
- 未运行完整训练，尚无双价值收敛性、Q_c 校准误差或跨 seed 安全结论。

## 24. 阶段 7 结束时记录的下一阶段目标（历史）

下一阶段可在不改变 Q_r/Q_c 网络职责的前提下，先定义长期 cost 预算，再实现动态
拉格朗日乘子更新及其稳定性测试。不得把 lambda 更新伪装成 reward 超期惩罚加权，
也不得顺带接入 LLM、修改模糊公式或让 LLM 选择 Host/VM。

## 25. 阶段 8：共享 episode/EMA 动态拉格朗日安全控制

### 25.1 统计周期与数学定义

本阶段选择一个清晰且可解释的共享控制方案：

```text
一个全局 lambda_DDL
  -> 同步给 Manager 的独立 Q_r/Q_c
  -> 同步给 Host Agent 的独立 Q_r/Q_c
  -> 同步给 VM Agent 的独立 Q_r/Q_c
```

三层 Q_c 网络及其参数仍完全独立，只共享约束价格 `lambda_DDL`。原因是三层
最终作用于同一个环境、同一组工作流 DDL；如果每层都用同一环境完成事件各更新
一次 lambda，会把同一个约束结果重复预算三次。当前尚无可靠的分层反事实 cost
归因，因此没有提前实现分层 lambda。

更新周期固定为 episode 边界，不逐 transition 更新。环境为每次实际安全转换
累计一个 cost 并增加一次计数：

```text
J_c_episode =
    episode cumulative safety_cost
    / episode safety transition count
```

这里的环境安全转换包括 `vm_assign()` 和 `finish_phase_and_advance()` 产生的
安全诊断；它不是 Manager/Host/VM replay cost 的跨层总和，因此不会因为三个
Agent 各训练一次而重复计入 lambda 预算。工作流完成违反/超期事件仍由环境已有
集合去重。

为降低 episode 间波动，更新使用 EMA：

```text
J_c^EMA(k) =
    beta * J_c^EMA(k-1)
    + (1 - beta) * J_c_episode(k)

lambda_DDL <- clip(
    lambda_DDL
    + lambda_lr * (J_c^EMA - cost_budget),
    lambda_min,
    lambda_max
)
```

首次有效 episode 直接初始化 EMA。`warmup_steps` 单位是“已观测完成的 episode
数”；warmup 结束后只在 `update_interval` 指定的 episode 间隔更新。默认：

```text
beta = 0.9
lambda_lr = 0.01
cost_budget = 0
update_interval = 1 episode
warmup_steps = 5 episodes
lambda bounds = [0, 100]
```

动作评分公式保持：

```text
Q_safe(s,a) = Q_r(s,a) - lambda_DDL * Q_c(s,a)
```

动态控制器不会修改 reward、cost、legal/safety/final mask、safety shield、
固定回退控制器或 Manager 规则权重语义。

### 25.2 训练预算与最终评价

`cost_budget` 只定义训练期间的拉格朗日预算，可以为训练稳定性显式配置为小的
非零值。它不能改变最终安全标准。

评估接口新增可选安全统计，最终合格线固定为：

```text
deadline_violation_rate <= 0.0
```

因此非零训练预算不会泄漏到评估合格判定。评估日志新增
`eval_deadline_violation_rate` 和 `eval_zero_violation_pass`。当前 best
checkpoint 仍按历史 `eval_energy` 保存；把保存顺序改为完整可行性优先属于后续
独立阶段，本阶段没有提前修改。

### 25.3 配置与兼容模式

新增嵌套配置：

```text
safe_rl.lagrangian.enabled = false
safe_rl.lagrangian.lambda_init = 1.0
safe_rl.lagrangian.lambda_lr = 0.01
safe_rl.lagrangian.lambda_min = 0.0
safe_rl.lagrangian.lambda_max = 100.0
safe_rl.lagrangian.cost_budget = 0.0
safe_rl.lagrangian.update_interval = 1
safe_rl.lagrangian.cost_ema_factor = 0.9
safe_rl.lagrangian.warmup_steps = 5
```

兼容矩阵：

```text
safe_rl.enabled = false:
    原 Q_r、旧 reward/replay/observation/动作行为；动态控制器不观测也不更新

safe_rl.enabled = true, lagrangian.enabled = false:
    保留阶段 7 固定 initial_lagrange_multiplier 行为

safe_rl.enabled = true, lagrangian.enabled = true:
    episode/EMA 共享动态 lambda
```

CLI 新增 `--safe-rl-dynamic-lambda`，必须与 `--safe-rl` 同时使用，默认关闭。
动态模式使用独立 `_dynamicLambdaStage8` 输出后缀，避免向固定 lambda 或 legacy
实验日志追加不同表头。

### 25.4 诊断、数值防护与 checkpoint

动态模式的 phase/episode CSV 记录：

```text
current_lambda
mean_safety_cost
cost_budget
constraint_gap
lambda_update_count
```

控制器还提供 episode cost、已观测 episode 数、无效样本数、是否更新和跳过原因。
所有配置先做有限性、非负性、上下界、EMA 范围和整数周期校验；非有限/负 episode
cost 不进入 EMA，候选 lambda 溢出时按 gap 方向夹到上下界，防止 NaN 或数值爆炸。

checkpoint schema 从 4 升级到 5。三个 Agent checkpoint 都嵌入同一个共享控制器
状态，包含当前 lambda、EMA、预算、更新计数、warmup/interval 及配置；加载 Agent
后可用 `lagrange_controller_state` 严格恢复控制器。严格恢复拒绝配置不一致、
非有限状态和越界 lambda。原版本 4 checkpoint 没有该字段时仍可按阶段 7 固定
lambda 语义加载。

### 25.5 修改文件

1. `base/safety_lagrange.py`（新增）
2. `base/hrl_env.py`
3. `base/d3qn_agent.py`
4. `hrl_mix/train_config.py`
5. `hrl_mix/train.py`
6. `hrl_mix/train_runner.py`
7. `hrl_mix/train_eval.py`
8. `tests/test_safety_lagrange.py`（新增）
9. `docs/SAFE_HRL_PROGRESS.md`

未修改 LLM/SeEvo、CEWS evaluator、`get_task_priority_v2`、模糊能耗公式、
`lambda_E=1.0`、模糊 DDL 公式、`eta=0.95`、确定截止期、固定 VM/回退排序、
Q_r/Q_c 网络结构、D3QN 输出维度或三层决策职责。

### 25.6 测试结果

全量 pytest：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`82 passed, 15 skipped in 2.20s`。15 项跳过均因该 Python 环境没有
PyTorch，包括 4 项 checkpoint 维度、8 项阶段 7 双价值和 3 项阶段 8
PyTorch 集成测试。

完整 PyTorch unittest：

```powershell
D:\Python3.12\python.exe -m unittest discover -s tests -v
```

结果：`97 tests OK in 2.138s`。

阶段 7/8 与 checkpoint 定向测试：

```powershell
D:\Python3.12\python.exe -m unittest `
  tests.test_d3qn_checkpoint_dimensions `
  tests.test_safe_dual_value_d3qn `
  tests.test_safety_lagrange -v
```

结果：`24 tests OK in 0.784s`。阶段 8 新增 12 项覆盖：

1. `J_c > d_c` 时 lambda 增大；
2. `J_c < d_c` 时 lambda 减小；
3. lambda 上下界；
4. episode warmup、update interval 和 EMA；
5. 动态配置字段、默认值及独立输出后缀；
6. 控制器 state checkpoint 恢复；
7. `safe_rl.enabled=false` 保持固定值和默认关闭配置；
8. NaN cost 拒绝且 lambda/EMA 仍有限；
9. 一个共享 lambda 同步到三层；
10. runner episode 边界使用环境 cost/转换数更新并同步；
11. Agent checkpoint 嵌入并恢复控制器；
12. 非零训练预算之外，评估仍使用零违反标准。

另执行：

- 修改 Python 文件 `py_compile`：通过；
- `D:\Python3.12\python.exe -m hrl_mix.train --help`：通过；
- 项目 `.venv` 全量 pytest：收集阶段 8 个模块失败，原因均为缺少 `numpy`；
  该环境结果不是测试通过。

本阶段没有运行完整训练 episode、正式长程训练、lambda 收敛曲线、跨 seed
训练或 27 个独立评估入口，不能声称这些项目已通过。

### 25.7 阶段状态、风险和尚未实现

阶段 8 状态：代码完成并通过纯控制器、三层同步、checkpoint、评估标准和全量
回归测试。

主要风险：

- `safety_cost` 同时包含违反指示、以秒计的模糊超期和归一化过程风险；默认
  `cost_budget=0` 时，只要 EMA 为正，lambda 就只会上升直至满足约束或触及上界。
  正式实验必须校准 cost 尺度、`lambda_lr`、EMA、上界和训练预算；
- 共享 lambda 可解释且避免重复预算，但不能表达三层不同的反事实安全责任；
- 过程风险和 Q_c 都是模型估计，不是真实完成后的安全保证；
- 本阶段只报告固定零违反评估是否通过，best checkpoint 仍按能耗保存；
- 尚无完整训练证明动态 lambda 能收敛，也无跨 seed 的零违反结论。

尚未实现：风险分层 replay、安全探索的进一步实验设计、LLM 安全启发式接入、
示范预训练、可行性优先 checkpoint、完整安全评价报告与消融实验。

## 26. 阶段 8 结束时记录的下一阶段目标（历史）

下一阶段可严格限定为安全探索/安全 replay 的信息完整化与采样策略，或先运行
小规模动态 lambda 收敛实验并校准 cost 尺度。进入任一路线前都应保留本阶段的
共享 lambda 口径、零违反评估线和默认关闭兼容开关，不得顺带修改模糊公式、
SeEvo 接口或让 LLM 选择 Host/VM。

## 27. 阶段 9：三层安全 epsilon-greedy 与动作审计

### 27.1 调用链审计与安全探索规则

审计确认，阶段 7 的 Host/VM 训练调用已经通过
`layer_learning_action_mask()` 把环境 `final_action_mask` 传给 D3QN，D3QN
的随机分支和贪心分支也都使用传入 mask；但此前没有统一记录探索类型，也没有
把 proposed/executed 关系写入 replay 审计字段。本阶段补齐这些接口，不改变
Host/VM/Manager 的职责。

安全模式的动作规则现在明确为：

```text
A_final(s) = {a | final_action_mask[a] = 1}

epsilon 分支:
    a ~ Uniform(A_final(s))

greedy 分支:
    a = argmax_{a in A_final(s)}
        [Q_r(s,a) - lambda_DDL * Q_c(s,a)]

A_final(s) 为空:
    不调用 D3QN，调用既有确定性模糊 DDL 回退控制器
```

Host/VM 的 `A_final` 是独立记录的 hard-legal mask 与 fuzzy-DDL safety mask
的交集。Manager 按前一阶段约束仍不做 DDL 硬屏蔽；其现有 Manager action
mask 是本层有效动作集合，安全模式的贪心评分仍使用独立 Q_r/Q_c 和共享
`lambda_DDL`。由于项目当前没有 Manager 动作级 DDL 预测器，Manager 集合中
不存在可供本阶段屏蔽的“已预测不安全 Manager 动作”；本阶段没有擅自发明一套
Manager 风险模型或修改规则权重语义。

### 27.2 动作分类与 proposed/executed 语义

新增统一的 `action_source` 分类：

```text
random_safe_exploration
greedy_safe_action
shield_correction
fallback_action
```

`policy_selection_type` 单独保留 shield 介入前的随机/贪心来源。例如随机提议
被 shield 修正时：

```text
policy_selection_type = random_safe_exploration
action_source = shield_correction
proposed_action = RL 原提议
executed_action = 环境实际执行动作
action_modified = true
```

正常安全探索直接从 `final_action_mask` 选择，因此不会主动产生不安全提议；
`shield_correction` 仍保留用于防御旧调用方、外部调用或状态/动作 mask 不同步。
空安全集时没有 RL 提议，记录 `proposed_action=None`、确定性
`executed_action` 和 `fallback_action`。

环境的 Host/VM shield 决策记录、任务 `info` 和 phase shield records 均保存
上述字段。诊断新增：

```text
random_safe_exploration_count
greedy_safe_action_count
shield_correction_count
fallback_action_count
```

phase/episode CSV 还记录 `manager_action_source`；Manager 的 proposed/executed
关系写入其安全 replay 审计字段。

### 27.3 Replay 与训练动作

安全 replay 从原 8 字段扩展为 12 字段：

```text
1  state
2  final_action_mask
3  executed_action
4  performance_reward
5  safety_cost
6  next_state
7  next_final_action_mask
8  done
9  proposed_action
10 action_source
11 policy_selection_type
12 action_modified
```

Q_r/Q_c 更新只解包并消费前 8 字段；两个网络的 `gather` 都使用第 3 字段
`executed_action`。第 9–12 字段只用于 shield/探索分析，不参与 Bellman target
或 loss。更新诊断额外报告 `executed_action_mean`、
`proposed_action_mean` 和 `action_modified_rate`，便于测试这一边界。

空 final mask 的回退动作不被伪装成该全零 mask 下的 RL 策略经验，因此仍不写入
Q replay；它完整保存在环境 shield/fallback 记录中。后续如需用回退示范训练，
必须另行定义 imitation/off-policy 语义，不能直接混入当前 Q transition。

### 27.4 接口与兼容性

新增且保持兼容的接口：

- `D3QNAgent.select_action_with_info(...) -> dict`：返回动作和探索审计；
- 原 `D3QNAgent.select_action(...) -> int` 保持不变；
- `select_layer_action_with_info(...)`：返回原三个结果外加审计字典；
- 原 `select_layer_action(...)` 三元组接口保持不变；
- `host_select(..., action_selection=None)` 和
  `vm_assign(..., action_selection=None)` 只增加可选参数；
- `D3QNAgent.remember(...)` 只增加可选审计参数。

本阶段没有新增核心依赖或单独配置。功能继续由已有
`safe_rl.enabled` 控制，该配置默认 `false`：

```text
safe_rl.enabled = false:
    原 epsilon-greedy、Q_r、7 字段 replay 和旧环境调用行为

safe_rl.enabled = true:
    三层安全评分及动作审计；Host/VM 始终消费显式 final mask

safe_rl.enabled = true, shield.enabled = false:
    final mask 按阶段 4 兼容语义等于 hard-legal mask，不启用 DDL 动作裁剪

safe_rl.enabled = true, shield.enabled = true:
    Host/VM final mask = hard-legal mask ∩ fuzzy-DDL safety mask；
    空 Host/VM 安全集走既有确定性回退
```

没有修改 D3QN 输出结构、Q_r/Q_c 网络参数、动态 lambda 更新、checkpoint
网络状态、Manager/Host/VM 决策职责或固定 VM 回退排序。

### 27.5 修改文件

1. `base/d3qn_agent.py`
2. `base/hrl_env.py`
3. `hrl_mix/train_utils.py`
4. `hrl_mix/train_runner.py`
5. `hrl_mix/train.py`（CLI 说明）
6. `tests/test_safe_exploration.py`（新增）
7. `tests/test_safe_dual_value_d3qn.py`
8. `tests/test_fuzzy_ddl_safety_shield.py`
9. `docs/SAFE_HRL_PROGRESS.md`

未修改 LLM/SeEvo、CEWS evaluator、`get_task_priority_v2`、模糊能耗公式、
`lambda_E=1.0`、模糊 DDL 公式、`eta=0.95`、确定截止期、固定 VM/回退
排序、D3QN 网络结构或动态拉格朗日公式。

### 27.6 测试结果

阶段 4/5/7/9 定向测试：

```powershell
D:\Python3.12\python.exe -m unittest `
  tests.test_safe_exploration `
  tests.test_safe_dual_value_d3qn `
  tests.test_fuzzy_ddl_safety_shield `
  tests.test_fuzzy_ddl_fallback_controller -v
```

结果：`31 tests OK in 0.738s`。阶段 9 新增 6 项覆盖：

1. `epsilon=1` 随机探索始终位于 final mask；
2. `epsilon=0` 只在 final mask 内最大化 Q_r-lambda*Q_c；
3. 空 final mask 不调用 Agent 并触发确定性 fallback；
4. shield correction 同时保留 proposed/executed 和原策略来源；
5. replay 第 3 字段训练 executed action，第 9 字段只记录 proposal；
6. `safe_rl=false` 保留旧 epsilon-greedy 和 7 字段 replay。

完整 PyTorch unittest：

```powershell
D:\Python3.12\python.exe -m unittest discover -s tests -v
```

结果：`103 tests OK in 2.116s`。

无 PyTorch 的 Anaconda pytest：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`82 passed, 21 skipped in 2.14s`。21 项跳过均明确因为该解释器未安装
PyTorch，其中包括 6 项阶段 9 测试；不是测试失败，也没有新增依赖缺失。

另对本阶段修改的 8 个 Python 文件执行 `py_compile`，通过，并执行
`D:\Python3.12\python.exe -m hrl_mix.train --help` 验证 CLI 导入与配置说明，
通过。没有运行完整训练
episode、正式长程训练、27 个评估入口、跨 seed 安全探索实验或 GPU/CUDA 训练，
不能声称这些项目已通过。

### 27.7 阶段状态、风险和尚未实现

阶段 9 状态：完成。三层 epsilon-greedy 的安全模式路径、空集合 fallback、
动作来源分类、proposed/executed 记录以及 executed-action replay 已实现，并通过
定向与全量回归。

主要风险：

- `final_action_mask` 来自模糊完成时间预测；“预测安全”不是工作流真实完成后的
  安全保证，模型误差仍可能导致实际 DDL 违反；
- Manager 当前没有动作级 fuzzy-DDL mask。它使用安全状态、Q_c 和共享 lambda
  评分，但随机探索集合仍是现有 Manager 合法动作集合；若要对 Manager 做动作级
  屏蔽，必须先独立设计可验证的 Manager 动作后果预测，不能直接复用 VM 风险；
- fallback 动作不进入 Q replay，避免把全零 mask 下的外部控制动作误当策略动作，
  但这也意味着 Q 网络不会直接从回退示范中学习；
- 本阶段没有长程训练证据，尚不能判断安全 mask 稀疏度、干预率、fallback 率、
  Q_c 校准和 lambda 收敛在多 seed 下是否稳定。

尚未实现：完整安全 replay/风险采样策略、fallback 示范预训练、LLM 安全启发式
接入、可行性优先 checkpoint、正式零违反训练评估与消融实验。

## 28. 阶段 9 结束时记录的下一阶段目标（历史）

下一阶段应先选择一个独立边界：补全安全 replay 的采样/统计语义，或运行小规模
多 seed 安全探索实验以校准 mask 稀疏度、干预率、fallback 率和 cost 尺度。
不得顺带修改模糊公式、固定 VM 回退规则、三层职责、SeEvo 接口或让 LLM 直接
选择 Host/VM。

## 29. 阶段 10：模糊能耗 performance reward 与版本化安全 replay

### 29.1 Reward 调用链审计

审计 `vm_assign()`、`_assign_task_to_specific_vm()`、
`get_fuzzy_energy_summary()`、`finish_phase_and_advance()` 和三层 runner 后确认：

- 每次实际任务分配会同时向 modal、optimistic、pessimistic 三条负载时间线
  追加相同 task-VM 映射；
- `get_fuzzy_energy_summary()` 对当前已提交的完整负载记录重放 Host 功率积分，
  生成 TFN 的 mean、std 和 score；
- Stage 1 已经用缓存 score 的负增量训练安全模式，但没有显式记录动作前后
  objective，也没有拆分 reward 分量；
- Manager 使用 phase 内任务增量之和，Host/VM 使用各自任务动作增量。三层网络
  彼此独立，因此同一物理结果分别用于三层信用分配，不是在同一个 return 中重复
  累计最终能耗。

### 29.2 Performance reward 数学定义与基准

对第 `t` 个实际调度动作，先在写入新任务记录之前取得：

```text
E_score_before =
    fuzzy_energy_mean_before
    + 1.0 * fuzzy_energy_std_before
```

把任务及其三条影子记录写入后，再取得：

```text
E_score_after =
    fuzzy_energy_mean_after
    + 1.0 * fuzzy_energy_std_after

Delta_E_score = E_score_after - E_score_before
raw_energy_reward = -Delta_E_score
energy_reward = raw_energy_reward * performance_reward_scale
```

当前 `performance_reward_scale` 沿用既有 `energy_reward_scale=1e-3`，只做正的
数值缩放，不改变最优策略对应的风险调整模糊能耗目标。未缩放的精确公式保存在
`raw_energy_reward`，动作前后基准保存在：

```text
fuzzy_energy_score_before
fuzzy_energy_score_after
performance_energy_delta
```

相邻动作使用连续基准，因此：

```text
sum_t Delta_E_score_t = final_fuzzy_energy_score - initial_score
```

episode 末尾不再额外添加一次完整最终能耗，避免重复累计。

本阶段选择不增加 shaping，避免悄悄改变既有安全实验语义，但显式输出全部分量：

```text
energy_reward
completion_reward       = 0
waiting_reward          = 0
utilization_reward      = 0
communication_reward    = 0
total_performance_reward
```

并强制：

```text
total_performance_reward =
    energy_reward
    + completion_reward
    + waiting_reward
    + utilization_reward
    + communication_reward
```

`safety_cost` 不在分量字典中，也不参与总性能奖励；
`performance_reward_safety_cost_included=false` 提供显式诊断。原
`performance_reward`、`performance_reward_host` 和
`performance_reward_vm` 字段保留为兼容别名，安全训练和评估优先读取
`total_performance_reward`。

### 29.3 版本化安全 transition

新增独立的 `SafeReplayTransition`，不再用位置易错的 12 元组表示安全经验。
schema version 当前为 1，字段包括：

```text
state
proposed_action
executed_action
performance_reward
safety_cost
next_state
done
legal_action_mask
safety_action_mask
final_action_mask
next_final_action_mask
shield_modified
fallback_triggered
fuzzy_safety_margin
predicted_risk_finish
violation_flag
manager_phase_id
near_boundary_margin
action_source
policy_selection_type
performance_reward_components
risk_category
schema_version
```

三类当前 mask 分开保存；Double DQN bootstrap 使用
`next_final_action_mask`。Q_r 和 Q_c 的当前动作 `gather` 都只使用
`executed_action`，`proposed_action` 仅用于 shield 干预分析。

环境新增 episode 内递增的 `manager_phase_id`：任务 info 和 phase info 使用
同一个 Manager 决策阶段编号，phase 完成后再加一。

### 29.4 Fallback 与风险分类

Stage 9 为避免歧义没有把空 final mask 的 fallback 写入 Q replay。本阶段需要
完整风险经验，因而调整为：

```text
正常 RL/shield transition:
    executed_action 必须属于 final_action_mask

fallback transition:
    final_action_mask 可以全零
    proposed_action = None
    executed_action 必须属于 legal_action_mask
    fallback_triggered = true
```

这样 fallback 不会被伪装成安全 RL 提议，但仍作为控制器经验保存，训练字段继续
明确使用实际执行动作。

风险类别采用互斥优先级：

```text
actual_violation
    > fallback
    > shield_intervention
    > near_boundary
    > normal_safe
```

当前 `near_boundary_margin=1.0` 秒，即未发生更高优先级事件且模糊安全裕量不大于
1 秒时分类为 `near_boundary`。该阈值只影响 replay 诊断/采样类别，不修改
shield 安全边界或 DDL 公式。

### 29.5 序列化、旧 replay 和 PER

transition 提供严格的 JSON-compatible `to_dict/from_dict` 和
`serialize/deserialize`。Agent 提供：

```text
replay_state_dict()
load_replay_state_dict()
```

buffer checkpoint 同时记录 buffer schema、transition schema、state/action
维度、PER 模式、组合 TD 权重、near-boundary 阈值、transitions 和
priorities。采样 batch 除训练字段外，也显式返回三类 mask、预测风险、
安全裕量、违反标志、phase ID、动作来源和 reward 分量，便于审计。以下情况
明确抛错：

- 缺少 schema version；
- 旧 tuple 或无版本 replay；
- schema 不匹配；
- observation/action mask 维度不匹配；
- PER 模式、组合权重、风险阈值或 priority 数量不匹配；
- 非有限字段、非法 mask、越界动作；
- 性能 reward 分量与 total 不一致。

历史 D3QN 网络 checkpoint 本来不保存 replay buffer，因此仍可按原网络/Q_c
规则加载；不会把旧 12 元组猜测成新字段。若未来确需迁移旧内存/外部 replay，
必须编写显式迁移器。

当前训练配置没有启用 PER，原行为不变。如果显式使用 PER，可选择：

```text
combined_per_priority = false:
    priority = abs(performance TD error) + epsilon

combined_per_priority = true:
    priority =
        [w_r * abs(delta_r) + w_c * abs(delta_c)]
        / (w_r + w_c)
        + epsilon
```

除以权重和避免仅因权重总量改变 priority 尺度。组合开关默认 `false`，不会
悄悄改变旧 PER 语义。

### 29.6 配置与兼容模式

新增：

```text
safe_rl.replay.transition_schema_version = 1
safe_rl.replay.near_boundary_margin = 1.0
safe_rl.replay.combined_per_priority = false
safe_rl.replay.performance_td_weight = 1.0
safe_rl.replay.safety_td_weight = 1.0
```

配置校验 schema、阈值、权重有限性和组合权重非零。总开关仍是
`safe_rl.enabled`，默认 `false`：

```text
safe_rl.enabled = false:
    原混合 reward、原 7 字段 replay、原 epsilon-greedy

safe_rl.enabled = true:
    增量模糊能耗 performance reward 分解
    + versioned SafeReplayTransition
```

safe 模式输出目录增加 `_performanceRewardReplayStage10` 后缀，防止向 Stage 9
旧 CSV 表头和实验目录追加不同 schema。没有升级核心依赖。

### 29.7 接口变化

- 新增 `base.safe_replay.SafeReplayTransition`；
- `D3QNAgent.remember()` 增加三类 mask、fallback、margin、risk finish、
  violation、phase 和 reward components 可选参数；原位置参数不变；
- `D3QNAgent.replay_state_dict()` / `load_replay_state_dict()` 新增；
- `vm_assign()`/phase info 新增 reward 分量、能耗前后基准和
  `manager_phase_id`；
- `safe_replay_metadata()` 和 `performance_reward_components()` 新增；
- safe replay buffer 元素从无版本 tuple 改为具名对象；legacy replay 保持
  原 7 元组。

未修改 D3QN 网络输入/输出层、Q_r/Q_c 结构、动态 lambda、shield 动作规则、
固定 VM 回退排序或三层决策职责。

### 29.8 修改文件

1. `base/safe_replay.py`（新增）
2. `base/d3qn_agent.py`
3. `base/hrl_env.py`
4. `hrl_mix/train_config.py`
5. `hrl_mix/train_utils.py`
6. `hrl_mix/train_runner.py`
7. `hrl_mix/train_eval.py`
8. `hrl_mix/train.py`
9. `tests/test_safe_performance_reward.py`（新增）
10. `tests/test_safe_replay.py`（新增）
11. `tests/test_safe_exploration.py`
12. `tests/test_safe_dual_value_d3qn.py`
13. `tests/test_fuzzy_ddl_safety_shield.py`
14. `tests/test_safe_hrl_cost.py`
15. `docs/SAFE_HRL_PROGRESS.md`

未修改 LLM/SeEvo、CEWS evaluator、`get_task_priority_v2`、模糊能耗定义、
`lambda_E=1.0`、模糊 DDL 公式、`eta=0.95`、确定截止期、固定 VM/回退
排序或 D3QN 网络结构。

### 29.9 测试结果

阶段 10 定向 reward/replay 及关联回归：

```powershell
D:\Python3.12\python.exe -m unittest `
  tests.test_safe_performance_reward `
  tests.test_safe_replay `
  tests.test_safe_exploration `
  tests.test_safe_dual_value_d3qn `
  tests.test_fuzzy_ddl_safety_shield `
  tests.test_safe_hrl_cost -v
```

最终结果：`45 tests OK in 0.734s`。其中阶段 10 新增测试覆盖：

- 动作前后模糊能耗基准及增量公式；
- 多动作增量望远镜求和、不重复添加最终能耗；
- 五类 shaping 独立且 safety cost 不进入总性能 reward；
- transition 必需字段、executed-action 训练语义和五类风险分类；
- fallback 的空 final mask/硬合法动作边界；
- JSON transition、buffer state 序列化与恢复；
- 无版本/旧 tuple/schema/维度/PER priority 明确拒绝；
- batch 形状、mask/action 边界和 reward 分量一致性；
- 组合性能/安全 TD error 的 PER priority。

完整 PyTorch unittest：

```powershell
D:\Python3.12\python.exe -m unittest discover -s tests -v
```

最终结果：`119 tests OK in 2.163s`。

无 PyTorch 的 Anaconda pytest：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

最终结果：`95 passed, 24 skipped in 2.13s`。24 项跳过均因该
解释器没有安装 PyTorch，不是失败。

另执行修改文件 `py_compile`，通过。没有运行完整训练 episode、长期训练、
GPU/CUDA、27 个独立评估入口或跨 seed replay 分布实验，不能声称这些项目通过。

### 29.10 阶段状态、风险和尚未实现

阶段 10 状态：代码与定向/全量回归完成。

主要风险：

- `performance_reward_scale=1e-3` 保留了既有学习数值尺度；虽然不改变能耗最优
  策略，Q 值绝对尺度仍需在正式训练中校准；
- `near_boundary_margin=1.0` 使用秒作为阈值，不同 DDL/工作流规模下可能需要
  改成归一化阈值，但本阶段没有修改安全边界；
- 性能和安全 TD error 单位不同。组合 PER 已提供显式权重且默认关闭，正式启用
  前必须校准，不能假设默认 1:1 合理；
- fallback 经验会训练其 executed action，即使当前 final mask 为空。策略仍不会
  在空集合中自行选择该动作，但共享函数逼近参数可能受到这些样本影响，需要在
  正式实验中做包含/排除 fallback replay 的消融；
- 风险完成时间和模糊安全裕量仍是模型预测，不是真实完成安全保证；
- 尚未运行完整训练，不能证明新版 replay 的类别比例、PER 稳定性、Q_c 校准或
  零违反结果。

尚未实现：类别感知的风险采样比例控制、fallback/LLM 示范预训练、LLM 安全
启发式准入、可行性优先 checkpoint、正式跨 seed 安全评价和消融实验。

## 30. 阶段 11：SeEvo 安全启发式接入安全 Manager

### 30.1 实际调用链与职责确认

阶段 11 沿实际环境调用链接入，而不是把 LLM 代码传给下层 Agent：

```text
Manager observation + current heuristic mask
    -> Manager D3QN selects a stable heuristic index
    -> environment applies the selected ready-task heuristic
    -> selected heuristic orders only current ready_task_ids
    -> Host Agent selects Host from final Host mask
    -> VM Agent selects VM from final VM mask
    -> fuzzy DDL safety shield corrects/falls back when required
```

传统规则 FCFS、SJF、MCF、HUR、EDF 直接复用 HRL 已有五维任务特征。SeEvo
规则仍严格调用原八数组接口：

```python
get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty,
)
```

返回分数按既有 CEWS 语义取较小值优先；相同分数使用稳定 task ID 打破平局。
LLM 函数看不到 Host、VM、Agent observation 或动作 mask，也不进入
`host_select()`/`vm_assign()`。Host/VM 动作空间、网络输出维度、固定 VM 回退
排序和 safety shield 均未修改。

### 30.2 Manager 两种显式模式

新增 `safe_rl.manager_heuristics`：

```text
mode
library_manifest_path
recent_window
```

- `legacy_rule_weight_mode`：默认模式。Manager 仍使用 243 个
  `{-0.1, 0, +0.1}^5` 权重增量动作；旧排序、状态和训练语义保持不变。
- `heuristic_selection_mode`：Manager 动作是候选启发式的稳定索引，不再解释为
  单个规则内部的连续权重。该模式要求 `safe_rl`、state extension 和 shield
  同时开启。

训练 CLI 新增 `--safe-rl-heuristic-manager`。未显式传入时仍是 legacy 模式；
不存在默认语义漂移。选择模式的 Manager 输出维度从环境实际 action mask
推导，不再硬编码 243。

### 30.3 安全准入、上下文和规则版本

`base/manager_heuristics.py` 提供独立规则库加载和验证。动作槽顺序固定为：

```text
FCFS, SJF, MCF, HUR, EDF, manifest 中按顺序声明的 SeEvo 规则
```

传统五规则始终可用。LLM 槽位只有同时满足下列条件才进入 Manager 可用集合：

1. manifest 明确 `admitted=true`；
2. `constraint_feasible=true`、`feasible_seed_rate=1`；
3. 模糊 DDL 违反率和总模糊延期均为 0；
4. `eta=0.95`、`lambda_E=1.0`；
5. 当前工作流随机种子包含在离线评价 seeds 中；
6. 当前 DAX 集、工作流数量、到达率、horizon、deadline 模式/系数、
   云边 Host/VM 拓扑、PC/BW 层级、模糊边界参数和资源种子与评价上下文一致；
7. candidate 文件位于 manifest 目录内且 SHA-256 精确匹配；
8. 函数名仍为 `get_task_priority_v2`，可接收八个数组；
9. N=1/N=3 探测返回有限的一维同长数组且不修改输入。

不合格规则保留稳定动作槽，但 mask 为 0；准入指标不合格时不会 import 其代码。
action schema 把规则 ID、source、version 和静态准入 mask 哈希进 Manager
observation/checkpoint schema，因此候选库变化不会被旧 checkpoint 静默接收。
工作流 seed 在 episode 间变化时还会动态重算 LLM mask。

### 30.4 当前随仓库规则的真实准入状态

新增离线配置
`LLM/cfg/problem/cews_task_constructive_hrl_ss_admission.yaml`，其资源拓扑与
HRL SS 配置一致：2 个 cloud Host、1 个 edge Host、cloud VM `[9, 8]`、
edge VM `[8]`、50 个工作流、seed 0、`eta=0.95`、`lambda_E=1.0`。

历史 `seevo_safe_iter4_ind0` 和 `seevo_safe_iter3_ind3` 曾在“5 工作流、
3 cloud + 2 edge”范围内通过，但这不构成 HRL SS 准入证据。同场景重评结果为：

| 规则 | 模糊 DDL 违反率 | 总模糊延期 | 风险调整模糊能耗 | 当前 mask |
|---|---:|---:|---:|---:|
| `seevo_safe_iter4_ind0` | 0.62 | 2274.540385 | 316847.439156 | 0 |
| `seevo_safe_iter3_ind3` | 0.56 | 2261.720519 | 316133.709686 | 0 |

另抽查 `iter2_ind0`、`iter1_ind3`、`iter0_ind0`、`iter1_ind0`，违反率分别为
0.56、0.54、0.62、0.60，同样未准入。因此当前 bundled manifest 的 Manager
mask 是 `[1,1,1,1,1,0,0,0]`：五个传统规则可用，三个历史 LLM 槽位均不可用。
这不是实现缺失，而是按零违反安全准入原则拒绝把不匹配或失败证据冒充为通过。
受控单元测试 manifest 验证了真正满足准入和上下文条件时 Manager 可选择 LLM
规则。

### 30.5 Manager 状态、记录字段和归一化

选择模式 Manager observation 为：

```text
10 个原全局系统特征
+ 11 个阶段 6 全局安全特征
+ 每个候选规则 4 个近期特征
```

每规则特征为：

| 字段 | 范围 | 归一化 |
|---|---:|---|
| `recent_energy_performance` | `[-1,1]` | `x/(1+abs(x))`，x 为近期增量模糊能耗 reward 均值 |
| `recent_safety_performance` | `[0,1]` | `1/(1+mean safety_cost)`；无记录时为 0 |
| `recent_shield_intervention_rate` | `[0,1]` | 近期 shield 干预率 |
| `heuristic_available` | `{0,1}` | 当前 Manager action mask |

默认 manifest 共 8 个动作槽，因此选择模式 Manager 维度为
`10 + 11 + 8*4 = 53`。规则统计使用显式 `recent_window` 有界窗口，未执行规则
的性能字段为 0，不伪装成满分。状态中没有把 task/Host/VM ID 当连续特征。

环境 phase info 和训练 CSV 记录：

```text
selected_heuristic_id
heuristic_source
llm_rule_version
ready_task_ordering
subsequent_safety_interventions
heuristic_shield_intervention_count
heuristic_shield_intervention_rate
```

phase 完成后把所选规则的增量模糊能耗 reward、独立 safety cost 和 shield
干预率写入近期表现窗口。安全代价仍未合并进 performance reward。

### 30.6 接口变化与修改文件

新增接口：

- `load_manager_heuristic_library(..., runtime_context=...)`
- `heuristic_availability_mask(...)`
- `heuristic_action_schema_version(...)`
- `HrlHeftEnv.apply_manager_heuristic(index)`
- `HrlHeftEnv.apply_manager_action(index)`
- `HrlHeftEnv.get_manager_action_semantics()`
- `manager_action_dim(env)`

`get_task_priority_v2` 未改签名。原 `apply_manager_delta()` 保留，并在 selection
模式明确拒绝以防动作语义混用。Host/VM 网络和动作接口未改变。

本阶段修改文件：

1. `base/manager_heuristics.py`（新增）
2. `base/hrl_env.py`
3. `LLM/problems/cews_task_constructive/safe_heuristic_library.json`（新增）
4. `LLM/cfg/problem/cews_task_constructive_hrl_ss_admission.yaml`（新增）
5. `hrl_mix/train_config.py`
6. `hrl_mix/train_utils.py`
7. `hrl_mix/train_runner.py`
8. `hrl_mix/train.py`
9. `tests/test_safe_manager_heuristics.py`（新增）
10. `docs/SAFE_HRL_PROGRESS.md`

未修改 `LLM/seevo.py`、CEWS evaluator 评价公式、D3QN 网络结构、Q_r/Q_c
Bellman 结构、动态 lambda、Host/VM 选择职责、固定 VM 回退规则、模糊能耗公式、
模糊 DDL 公式或确定截止期。

### 30.7 测试结果

阶段 11 定向测试：

```powershell
D:\Python3.12\python.exe -m unittest `
  tests.test_safe_manager_heuristics -v
```

结果：`13 tests OK in 0.150s`。覆盖传统规则选择、受控已准入 LLM 选择、
无效规则不导入、评价上下文不匹配屏蔽、未评价 seed 动态屏蔽、LLM 不参与
Host/VM、legacy 243 动作/15 维状态兼容、确定性排序、状态维度/有限性、
ordering/版本/shield 干预记录和真实 phase 路径。

阶段 11 与 shield/Q/replay 关联定向回归：

```powershell
D:\Python3.12\python.exe -m unittest `
  tests.test_safe_manager_heuristics `
  tests.test_safe_exploration `
  tests.test_safe_dual_value_d3qn `
  tests.test_fuzzy_ddl_safety_shield `
  tests.test_safe_performance_reward `
  tests.test_safe_replay -v
```

结果：`52 tests OK in 0.891s`。

完整 PyTorch unittest：

```powershell
D:\Python3.12\python.exe -m unittest discover -s tests -v
```

结果：`132 tests OK in 2.361s`。

无 PyTorch 的 Anaconda pytest：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`108 passed, 24 skipped in 2.35s`；24 项均因该解释器未安装 PyTorch
而跳过，不是失败。另执行修改文件 `py_compile` 和
`python -m hrl_mix.train --help`，均成功。

离线准入复核实际运行了 6 条候选、HRL SS seed 0、每条 50 个工作流；结果均
为非零违反，已如实记录为未准入。未运行完整训练 episode、GPU/CUDA、跨 seed
正式训练评估或消融实验，不声称这些项目通过。

### 30.8 阶段状态与风险

阶段 11 状态：接入、审计、定向和全量回归完成；默认 LLM 规则没有一条满足
当前 HRL SS 零违反准入，因此当前可安全运行的是五传统规则候选集合。要实际
训练“Manager 在传统与 LLM 规则间选择”，必须先离线产生并在目标场景、目标
seed 集上通过准入的 SeEvo 规则，不能放宽为惩罚系数或复用 5-workflow 证据。

主要风险：

- 当前无有效 Git branch/commit，仍只能依赖文档、文件哈希和 manifest 版本，
  无法提供标准版本回滚点；
- candidate 是受控 Python 代码。当前有路径、哈希、接口和不变性探测，但不是
  OS 级沙箱；生产化前仍应把离线准入与训练进程隔离；
- 单 seed 通过只允许该 seed，不能解释为跨 seed 安全保证；
- Manager 的规则近期表现是在线有限窗口统计，冷启动规则均为 0，需要正式训练
  校准探索覆盖和窗口长度；
- shield 干预率和预测风险来自当前模糊完成时间模型，属于模型预测诊断，不是
  真实完成后的安全保证；
- 当前候选在 HRL SS 上高违反，可能说明 deadline cache、资源拓扑和固定评价
  控制器组合下本身很紧，也可能说明现有 SeEvo 搜索样本不足；需要先做可行性
  基线审计，再决定扩大 SeEvo 搜索，而不能把失败规则直接开放。

## 31. 阶段 12：SeEvo 启发式安全准入与版本管理

### 31.1 调用链与准入边界

阶段 12 的实际数据流为：

```text
SeEvo candidate_iter{iteration}_ind{individual}.py
    -> 原 CEWS evaluator 在配置 seed 集上完成调度评价
    -> evaluator protocol v2 RESULT_JSON（绑定源码和配置哈希）
    -> 离线注册器只读取源码字节、报告和 YAML，不导入候选
    -> 生成 admitted/rejected 版本记录并追加 manifest
    -> Manager 先校验记录、策略、场景、seed、报告和源码哈希
    -> 仅无拒绝原因的 admitted 规则才导入并探测接口
    -> Manager availability mask
```

注册器不会自动运行候选，也不会把未验证文件交给 Python。候选必须位于 manifest
目录下的 `generated/` 受信根中，报告必须位于 `admission_reports/` 受信根中。
来源未知、越界、未准入、记录/源码/报告哈希不一致的规则均不导入。已经通过静态
证据检查但运行时接口探测失败的规则会保留稳定动作槽并明确置 mask 为 0；manifest
结构或顶层准入策略无效时则明确抛错，不做静默降级。

### 31.2 数学定义与准入策略

没有修改 CEWS 目标或可行性优先比较。目标仍为：

```text
fuzzy_energy_score
  = fuzzy_energy_mean + 1.0 * fuzzy_energy_std
```

模糊 DDL 风险时间仍为：

```text
R(T_i) = 0.05 * T_i_modal + 0.95 * T_i_upper
```

专用 HRL SS 准入配置 `hrl_ss_50wf_3seed_v1` 要求：

1. 函数名和报告接口均为 `get_task_priority_v2`；
2. seed 集严格等于 `[0, 1, 2]`，三个 seed 均正常完成，且每个 seed 都完成
   50 个工作流；
3. 每个 seed 及聚合最大模糊 DDL 违反率均为 0；
4. 每个 seed 及聚合 `max_fuzzy_lateness` 均为 0；
5. `feasible_seed_rate = 1` 且聚合约束可行；
6. `fuzzy_energy_score <= 330000`；
7. 跨 seed 能耗目标变异系数
   `std(objective_seed) / max(abs(mean(objective_seed)), 1e-12) <= 0.05`；
8. DAX 集、工作流数量/到达/DDL、Host/VM 拓扑、PC/BW 层级、模糊参数和资源
   seed 模式与运行环境精确匹配，当前工作流 seed 必须属于评价 seed 集。

阈值是离线准入门槛，不进入 CEWS objective、HRL reward 或 safety cost。Manager
还会独立检查每条记录的策略与 manifest 顶层策略完全一致，记录不能自行放宽能耗
或稳定性阈值。

### 31.3 Manifest 与版本字段

manifest 升级为 schema 2，记录 schema 为 1。每条记录至少保存：

```text
heuristic_id
display_name
source_file
source_hash
seevo_iteration
seevo_individual
version
evaluation_report_file
evaluation_report_hash
evaluation_result_sha256
evaluation_config_sha256
evaluation_seeds
fuzzy_energy_mean
fuzzy_energy_std
fuzzy_energy_score
deadline_violation_rate
max_fuzzy_lateness
feasible_seed_rate
objective_cv_across_seeds
admitted
admission_status
rejection_reason / rejection_reasons
record_sha256
```

规则版本由 SeEvo iteration/individual、源码哈希、评价结果哈希和准入策略版本共同
组成。注册采用原子替换写入，拒绝覆盖重复 `heuristic_id` 或重复版本，并递增
`manifest_revision`。当前随附的三个历史 LLM 记录均保持 `rejected`：它们只有旧
协议或单 seed 证据，且其中两个在 HRL SS 复核中有非零违反；因此默认 Manager
mask 仍为 `[1,1,1,1,1,0,0,0]`，没有把历史结果伪装成新准入。

### 31.4 CEWS evaluator 协议扩展

`RESULT_JSON` 协议版本升为 2，新增准入证据：

```text
evaluator_protocol_version
function_name
interface_valid
candidate_source_file
candidate_sha256
evaluation_config_sha256
evaluation_seed_count
completed_seed_count
all_evaluation_seeds_completed
per_seed_metrics
max_deadline_violation_rate_across_seeds
max_fuzzy_lateness
objective_cv_across_seeds
```

`per_seed_metrics` 明确保存 seed、完成工作流数、可行状态、违反率、最大模糊延迟
及模糊能耗 mean/std/score。新增字段仅供准入和审计；`_add_objective_and_constraints`
及 SeEvo 的可行性优先排序未改变。

### 31.5 配置、工具与接口变化

离线准入 YAML 新增 `admission` 段：

```text
policy_version
required_evaluation_seeds
minimum_evaluation_seed_count
deadline_violation_rate_max
max_fuzzy_lateness_max
feasible_seed_rate_min
fuzzy_energy_score_max
objective_cv_max
```

新增命令行工具：

```powershell
python tools/manage_safe_heuristic_manifest.py `
  --manifest <manifest.json> `
  --candidate <generated/candidate_iterN_indM.py> `
  --evaluation-report <admission_reports/report.json> `
  --config <admission.yaml> `
  --heuristic-id <stable-id> `
  --seevo-iteration N `
  --seevo-individual M
```

退出码 0 表示 admitted，2 表示记录成功但被拒绝；解析、路径或 manifest 错误明确
失败。`get_task_priority_v2` 的八参数接口没有变化。Host Agent、VM Agent、
D3QN、replay、shield 和固定 VM 控制器接口均未改变。默认
`legacy_rule_weight_mode` 不加载该候选库，因此 `safe_rl.enabled=false` 的旧
训练语义保持不变。

### 31.6 修改文件

1. `base/heuristic_admission.py`（新增）
2. `tools/manage_safe_heuristic_manifest.py`（新增）
3. `LLM/problems/cews_task_constructive/eval.py`
4. `base/manager_heuristics.py`
5. `base/hrl_env.py`
6. `LLM/cfg/problem/cews_task_constructive_hrl_ss_admission.yaml`
7. `LLM/problems/cews_task_constructive/safe_heuristic_library.json`
8. `LLM/problems/cews_task_constructive/admission_reports/README.md`（新增）
9. `tests/test_heuristic_admission.py`（新增）
10. `tests/test_safe_manager_heuristics.py`
11. `tests/test_fuzzy_energy.py`
12. `docs/SAFE_HRL_PROGRESS.md`

本阶段未修改 `LLM/seevo.py`、D3QN 网络、Q_r/Q_c、拉格朗日更新、reward/cost、
Host/VM 动作选择、shield、固定 VM 回退、模糊能耗公式、模糊 DDL 公式或确定
截止期。

### 31.7 测试结果

定向测试：

```powershell
python -m unittest `
  tests.test_heuristic_admission `
  tests.test_safe_manager_heuristics `
  tests.test_fuzzy_energy -v
```

结果：`34 tests OK in 1.758s`。覆盖正常准入、各类拒绝、所有 seed 完成、零
违反、零最大模糊延迟、能耗/CV 阈值、重复版本、受信目录、源码与报告哈希篡改、
记录不得放宽 manifest 策略、失效接口明确屏蔽、拒绝规则不执行、Manager 只选择
可用规则、LLM 不参与 Host/VM、legacy 模式及模糊公式/可行性优先回归。

完整测试：

```powershell
python -m unittest discover -s tests -v
```

结果：`141 tests OK in 2.497s`，0 失败、0 跳过。修改文件 `py_compile` 和
`python tools/manage_safe_heuristic_manifest.py --help` 也成功。本次使用的
Python 环境无缺失依赖，未观察到 Windows 平台相关问题。

没有运行新的真实 HRL SS 三 seed 离线准入、完整训练 episode、GPU/CUDA、正式
训练评估或消融实验；不得把合成单元测试解释为某条实际 LLM 规则已经获得准入。

### 31.8 阶段状态、尚未实现项与风险

阶段 12 状态：实现和回归完成。只有完整通过 schema、准入策略、场景、seed、
记录哈希、报告哈希、源码哈希和运行时接口探测的规则才进入 Manager 可用动作
集合；拒绝记录只保留为版本化审计证据。

尚未实现：自动触发 SeEvo 搜索、自动运行来源未知候选、实际三 seed 准入批处理、
规则签名/公钥验证、OS 级沙箱和跨机器 artifact 仓库。本阶段也未改变后续 HRL
训练或资源动作。

主要风险：

- 当前 `.git/` 为空，无法记录 branch/commit/tag；manifest 和 SHA-256 能发现
  artifact 变化，但不是 Git 提交历史或数字签名；
- admitted Python 最终仍需在 Manager 进程中运行接口探测和 ready-task 排序。
  当前先做受信路径、评价证据和多重哈希校验，但这不是 OS 级沙箱；
- 当前真实历史 LLM 规则均未满足新三 seed 零违反准入，不能据此开始包含 LLM
  动作的正式训练；
- 能耗阈值 `330000` 和 CV 阈值 `0.05` 是当前 HRL SS 配置值，迁移工作流规模
  或资源场景时必须创建新策略版本，不能复用旧准入结论；
- 离线预测/评价只覆盖配置场景和 seed，不构成对未评价场景的真实安全保证。

## 32. 阶段 12 结束时记录的下一阶段目标（历史）

先用 evaluator protocol v2 在目标 HRL SS 场景真实运行 `[0,1,2]` 三 seed
评价，把每条候选的 stdout/RESULT_JSON 固化到 `admission_reports/`，再通过
注册工具生成不可覆盖的 admitted/rejected 记录。至少获得一条真实 admitted
LLM 规则后，才运行小规模 `heuristic_selection_mode` 训练并检查规则覆盖率、
能耗、safety cost、shield 干预率和 fallback 率。不得让 LLM 选择 Host/VM，
不得修改模糊能耗、模糊 DDL、确定截止期、固定 VM 回退或三层职责。

## 33. 阶段 13：安全示范轨迹与离线预训练

### 33.1 实际调用链与策略边界

示范生成使用真实 `HrlFcfsCacheEnv`，不是脱离环境的合成调度器：

```text
版本化 Manager 启发式库
  -> 每个 phase 固定选择一个 available/admitted heuristic index
  -> 传统或 SeEvo get_task_priority_v2 只对 ready tasks 排序
  -> 对当前任务读取全局 legal/safety/final VM mask
  -> final VM 集非空：在该集合内调用原 select_vm_deterministic
  -> 由选中 VM 所属 Host 唯一确定 Host 动作
  -> final VM 集为空：消费环境已生成的确定性 safety fallback
  -> env.host_select / env.vm_assign
  -> 与在线 safe replay 同 schema 的 Manager/Host/VM transition
```

因此，LLM 规则没有获得 Host、VM、资源对象或动作接口。Host 示范动作由固定 VM
规则选出的 VM 所属 Host 决定；空安全集合时，Host 同样由 fallback 最优 VM
唯一聚合得到。正常安全集合中的资源排序继续复用 CEWS 固定 VM 规则；空集合继续
复用阶段 5 的“最小预测模糊 DDL 违反量、最小模糊边际能耗、最小风险完成时间、
稳定 VM ID”顺序。

当前随附真实 SeEvo 历史规则仍全部为 rejected，因而本阶段真实环境 smoke test
使用 `traditional_edf`。代码只接受 Manager 库中 `available=true` 的规则；
未来只有通过阶段 12 准入且当前 seed 可用的 SeEvo 规则才能生成 LLM 示范，
不能把 rejected 规则伪装成安全示范。

### 33.2 Transition 与安全标签

每层示范轨迹直接保存 `SafeReplayTransition` schema 1，核心字段与在线 replay
一致：

```text
state
proposed_action
executed_action
performance_reward
safety_cost
next_state
done
legal_action_mask
safety_action_mask
final_action_mask
next_final_action_mask
shield_modified
fallback_triggered
fuzzy_safety_margin
predicted_risk_finish
violation_flag
manager_phase_id
action_source
policy_selection_type
performance_reward_components
risk_category
```

训练字段固定使用 `executed_action`；`proposed_action` 只供 fixed-rule/shield 审计。
Host/VM 与在线 runner 一样延迟到下一次同层决策后再写入 next state，episode
结束时使用零 next state/mask 和 `done=1`。Manager 仍使用 phase transition。
性能 reward 仍是动作增量的风险调整模糊能耗 reward，safety cost 没有合回
performance reward。

`safe_demonstration` 不是由生成策略名称直接声明，而是用 episode 最终实际结果
重新计算。当前配置标准 `zero_fuzzy_ddl_violation_v1` 要求：

```text
deadline_violation_rate = 0
max_fuzzy_lateness = 0
constraint_feasible = true
all_workflows_completed = true
```

最终完成工作流的风险完成时刻仍为：

```text
R(T_i) = 0.05 * T_i_modal + 0.95 * T_i_upper
```

能耗记录仍为：

```text
fuzzy_energy_score
  = fuzzy_energy_mean + 1.0 * fuzzy_energy_std
```

不满足标准的 episode 可以作为审计记录写入数据集，但不会被标记为 safe，
`require_safe=true` 的离线训练加载器会排除它。

### 33.3 数据集 manifest、版本与严格划分

新增数据集 schema 1 和 episode schema 1。数据集 manifest 至少记录：

```text
dataset_id / dataset_version / manifest_revision
generator_policy
safe replay schema version
strict seed split and split version
safety standard and standard version
observation schema versions
layer input/action dimensions
episode count / safe episode count
trajectory count by layer and total
manifest content SHA-256
```

每个 episode 索引记录至少包含：

```text
generator_policy
heuristic_id / heuristic_source / heuristic_version
resource_seed / workflow_seed
DDL setting
fuzzy parameters
feasibility and rejection reasons
fuzzy energy score
trajectory count
episode file SHA-256
```

episode JSON 与 manifest 分离并带文件 SHA-256；加载时还会核对 generator
policy、seed、heuristic、DDL、模糊参数、观测 schema、维度、能耗和轨迹计数，
不接受无版本、哈希不一致或 manifest/episode 字段不一致的数据。

`StrictSeedSplit` 同时划分 workflow seed 和 fuzzy resource seed。为防止同一
随机源换角色泄漏，同一个整数 seed 不允许在 train、validation、final_test
任意两组之间出现；一个 episode 的 workflow/resource seed 必须共同属于唯一
split。预训练加载器只接受 `train` 或 `validation`，明确拒绝加载
`final_test`。这保证最终测试 seed 不会被预训练读取，但不等价于对未见场景的
安全泛化保证。

### 33.4 离线 Q_r/Q_c 与可选行为克隆

新增默认关闭的 `safe_rl.offline_pretraining`：

```text
enabled
dataset_manifest_path
epochs
batch_size
performance_learning_rate
safety_learning_rate
train_q_r
train_q_c
behavior_cloning_enabled
behavior_cloning_weight
sync_targets_after_pretraining
random_seed
```

离线 Q_r 使用 `performance_reward` 的 Double-DQN Bellman target；离线 Q_c
使用独立 `safety_cost` target。两者使用各自 online 网络和独立临时 optimizer。
可选行为克隆只对非 fallback、且 executed action 位于 `final_action_mask` 的
样本初始化 Q_r 动作偏好；Q_c 仍只由 safety cost 学习。fallback 是控制器动作，
不会冒充 Agent 行为克隆标签。

预训练发生在三个 Agent 构造完成后、在线训练循环之前。实现不调用
`D3QNAgent.update()`，不写在线 replay，不推进 PER beta、epsilon step 或在线
update 计数，也不改 target tau/frequency。每个离线 epoch 内 target 网络冻结；
可配置在预训练边界做一次 online-to-target 初始化，之后仍完全由原
`D3QNAgent.update()` 的 Polyak/硬更新逻辑控制。训练和 validation loss 及数据
manifest hash 写入 `offline_pretraining_report.json`。

命令行新增：

```text
--offline-pretrain-manifest
--offline-pretrain-epochs
--offline-pretrain-behavior-cloning
--offline-pretrain-no-q-r
--offline-pretrain-no-q-c
```

只有显式提供 manifest 才启用预训练，并要求同时启用 safe RL、shield、安全
状态和 heuristic Manager。所有开关默认值都保持旧实验不预训练。

示范生成入口：

```powershell
python -m tools.generate_safe_demonstrations `
  --manifest <dataset/manifest.json> `
  --heuristic traditional_edf `
  --generate-split train `
  --scenario SS --ddl T `
  --train-workflow-seeds 1 --train-resource-seeds 1 `
  --validation-workflow-seeds 2 --validation-resource-seeds 2 `
  --final-test-workflow-seeds 3 --final-test-resource-seeds 3
```

生产场景使用实际 FCFS deadline cache；示例 seed 必须存在于对应 cache，且
SeEvo heuristic 还必须覆盖该评价 seed。

### 33.5 修改文件

1. `base/safe_demonstration.py`（新增）
2. `base/offline_pretraining.py`（新增）
3. `hrl_mix/safe_demonstrations.py`（新增）
4. `tools/generate_safe_demonstrations.py`（新增）
5. `hrl_mix/train_config.py`
6. `hrl_mix/train_runner.py`
7. `hrl_mix/train.py`
8. `tests/test_safe_demonstration_pretraining.py`（新增）
9. `docs/SAFE_HRL_PROGRESS.md`

没有修改 `get_task_priority_v2`、CEWS evaluator 可行性优先原则、模糊能耗/
DDL 公式、确定截止期、固定 VM 排序、fallback 排序、safety shield、D3QN
网络结构、在线 Bellman 更新或三层动作职责。

### 33.6 测试结果

阶段 13 定向测试：

```powershell
python -m unittest tests.test_safe_demonstration_pretraining -v
```

结果：`9 tests OK`。覆盖 manifest 必需字段、安全/不安全 episode 标签、
episode 哈希篡改、严格 train/validation/final-test seed 隔离、final-test
拒绝训练、默认关闭配置、Q_r/Q_c 离线更新、在线计数/target 周期保持以及一个
真实 `Montage_25` 环境端到端示范 smoke test。

完整 PyTorch unittest：

```powershell
python -m unittest discover -s tests -v
```

结果：`150 tests OK`，0 失败、0 跳过。

另一个 Anaconda/pytest 环境没有安装 PyTorch：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`124 passed, 26 skipped`；跳过项明确标记为 `PyTorch is not installed`，
其中包含 2 个阶段 13 的网络预训练/真实生成测试。系统 Python 有 NumPy/Torch
但没有 pytest；项目 `.venv` 有 pytest 但缺少 NumPy。Windows 文件路径、原子
JSON 写入和真实环境 smoke test 未观察到平台错误。没有安装或升级核心依赖。

未运行：正式 SS 50-workflow 数据集生成、真实 admitted SeEvo LLM 示范、完整
离线预训练、在线训练、GPU/CUDA、跨 seed 能耗/安全评估和消融实验；不得声称
这些项目已经通过。

### 33.7 阶段状态与风险

阶段 13 代码与小规模测试完成。示范完成后的零违反标签是对已执行 episode 的
最终评价；动作前的 fuzzy safety margin 和 risk finish 仍是模型预测，不是未来
真实完成安全保证。主要风险：

- 当前真实历史 LLM 规则没有 admitted 记录，因此目前只能生成传统启发式示范；
- safe demonstration 过滤可能使训练/验证集为空，加载器会明确报错，不会静默
  回退到不安全数据；
- 行为克隆把 Dueling Q_r 输出当作动作偏好 logits，只适合可选初始化，不替代
  Q-learning，也不用于 Q_c；
- 数据集 SHA-256 可检测篡改但不是数字签名，且当前 `.git/` 为空，没有可追踪
  branch/commit/tag；
- 严格 seed 隔离防止显式数据泄漏，但不能证明跨工作流规模、资源拓扑或 DDL
  场景的安全泛化；
- 正式数据生成与预训练可能消耗较长 CPU/GPU 时间，本阶段只运行单工作流 smoke。

## 34. 下一阶段目标

先用阶段 12 协议获得至少一条目标场景真实 admitted SeEvo 规则，并冻结完整
train/validation/final-test seed 清单；再分别用传统 EDF 和 admitted LLM 规则
生成小规模安全示范，审核 unsafe episode 排除率、fallback 比例、shield 干预率、
轨迹维度和能耗分布。审核通过后才运行 Q_r/Q_c/可选 BC 离线初始化与在线微调，
并在从未参与训练或验证的 final-test seeds 上按零模糊 DDL 违反标准评价。

## 35. 阶段 14：安全 HRL 阶段化训练流程

### 35.1 五阶段职责与实际调用链

新增版本化训练计划 `hrl_mix/config/safe_hrl_training_pipeline.json`，显式定义：

1. `demonstration_generation`：继续调用阶段 13 的真实环境示范生成入口；该步骤
   是外部数据准备步骤，runner 启动前会 fail-closed 检查数据集 manifest；
2. `offline_pretraining`：runner 在构造三层 Agent 后、在线循环前执行已有
   Q_r/Q_c/可选 BC 预训练；从 pipeline checkpoint 恢复时明确跳过，避免覆盖
   已有在线参数；
3. `shield_online_training`：启用既有 Host/VM fuzzy DDL shield 的在线训练；
4. `curriculum_training`：按显式 DDL、到达强度、模糊不确定性、资源容量和
   workflow 数量 profile 切换；
5. `cross_seed_robust_training`：必须使用 `training_seed_mode=round_robin`，
   只轮换 training seeds。

阶段 14 只增加训练编排，不修改 Manager–Host–VM 三层职责、动作语义、D3QN
网络结构、reward/cost 公式、shield、fallback 或 SeEvo 接口。`safe_rl.enabled`
或 pipeline 开关关闭时，原训练入口、原 `cfg.eval_seeds` 和原 episode reset
路径保持不变。

### 35.2 课程配置与维度边界

每个在线阶段都显式给出：

```text
deadline:
  level = loose | medium | tight
  alpha_small / alpha_large / alpha_small_probability
arrival:
  level = low | medium | high
  poisson_lambda
uncertainty:
  level = low | medium | high
  fuzzy_delta1 / fuzzy_delta2
resource:
  scale / capacity_scale / topology=inherit
workflow:
  scale / workflows_per_episode
training_seed_mode = primary | round_robin
```

课程切换会重建训练环境，使到达率、DDL alpha、模糊资源参数和资源容量真正进入
下一 episode；PC 与带宽 tier 使用同一个 `capacity_scale`，Host 数、VM 数及其
映射保持不变。每次重建后重新探测 Manager/Host/VM observation/action 维度，
与初始维度不一致时明确拒绝。

这里存在一个与当前 D3QN 调用链有关的设计边界：改变 Host/VM 数量会改变
Host/VM 输出动作维度，不能在同一组网络参数中安全热切换。因此当前同一训练
计划支持“固定拓扑下的资源容量规模”和 workflow 数量课程；真正的 S/M/L
Host/VM 拓扑变化必须使用独立训练计划、独立 Agent 和独立 checkpoint，不能
静默加载旧 checkpoint。

课程不会修改以下冻结定义：

```text
fuzzy energy score = fuzzy energy mean + 1.0 * fuzzy energy std
R(T_i) = 0.05 * T_i_modal + 0.95 * T_i_upper
D_i = deterministic deadline
```

### 35.3 阶段切换与 seed 隔离

支持两类切换：

- `fixed_episodes`：达到配置 episode 数后切换；
- `validation_threshold`：至少训练指定 episode 数，并连续指定次数通过配置的
  validation 阈值后切换。

可配置阈值包括 fuzzy energy score、safety cost、violation rate、shield
intervention rate、fallback rate 和 Q_c prediction error。控制器只接受
`source=validation` 的指标；传入 `final_test` 会明确抛错，不能用最终测试结果
调课程。训练计划强制 `training`、`validation`、`final_test` 三组 seed
非空、组内无重复且任意两组无交集：

- 环境训练只使用 training seeds；
- 阶段评估和阈值切换只使用 validation seeds；
- final-test seeds 仅写入审计信息，训练 runner 不消费。

`primary` 模式固定使用第一个 training seed；`round_robin` 模式在当前阶段内
循环全部 training seeds。`cross_seed_robust_training` 配置为其他模式会被
拒绝。

### 35.4 阶段指标

`SafeStageMetricsLogger` 以版本化 JSON Lines 逐次记录 validation 指标，并用
`stage_id`、`stage_type`、plan hash、validation seeds 和 transition reason
标识归属。每阶段累计均值写入阶段结束 summary。字段包括：

```text
fuzzy_energy_score
safety_cost
violation_rate
shield_intervention_rate
fallback_rate
lambda
q_c_prediction_error
q_c_prediction_error_sample_count
```

其中 fuzzy energy score 是各 validation seed 环境
`get_fuzzy_energy_summary()["fuzzy_total_energy_score"]` 的均值；safety cost
是独立 episode safety cost 的跨 seed 均值，不合回 performance reward；
violation rate 继续按已完成工作流的实际 fuzzy DDL 违反统计。Q_c prediction
error 当前明确定义为“最近一次在线更新中，已发生 Q_c 更新的三层 Agent 的
Safety Bellman Huber loss 均值”，并额外记录实际提供该值的层数；它不是最终
测试误差，也不是安全保证。

### 35.5 Checkpoint 与恢复

pipeline checkpoint schema 1 包含：

```text
pipeline_id / plan_hash
current stage index / stage ID
stage episode count / total episode count
consecutive validation pass count
stage metric accumulators and completed summaries
Manager/Host/VM checkpoint paths
shared Lagrange controller state
global step / next episode / best validation energy
```

恢复时先校验 pipeline ID、完整 plan hash、阶段 ID、计数和 metric schema，再由
三个 Agent 原有 `load()` 校验 observation/action 维度、Q_r/Q_c 模式和
checkpoint schema。恢复后不会重新执行离线预训练。已经完成全部在线阶段的
checkpoint 会被拒绝继续训练，应转入最终评价。

当前 Agent checkpoint 不包含 replay buffer，因此恢复能准确恢复课程阶段、
Q_r/Q_c 网络、optimizer、epsilon/update 计数和共享 lambda，但不是包含在线
replay 与所有随机数状态的 bit-exact 恢复；这是正式长训练前需要继续评估的
可复现性风险。

### 35.6 接口变化与默认兼容

新增配置：

```text
safe_rl.training_pipeline.enabled
safe_rl.training_pipeline.plan_path
safe_rl.training_pipeline.resume_checkpoint_path
```

新增命令行参数：

```text
--safe-training-pipeline-config
--safe-training-resume
```

`build_train_config()` 和 `train()` 增加同名可选参数。显式启用 pipeline 时要求
safe RL、shield、安全状态、动态 lambda 和 heuristic Manager 均已启用；阶段 2
离线预训练参数由计划文件读取。未提供计划文件时，pipeline 默认关闭，不改变旧
实验语义。

`evaluate_hrl_three_layer_multi_seed(..., return_safety_metrics=True)` 的返回
tuple 形状不变，但安全字典新增 fuzzy energy score、safety cost、shield/
fallback 计数和比例。`return_safety_metrics=False` 时不执行这些额外统计，
保持旧评估路径。

Windows 下原安全 run name 拼接全部历史阶段后缀会使单个目录名超过平台限制。
仅对显式启用 pipeline 的新实验改用 `safeHRL14 + 场景 + pipeline ID + plan
hash` 的短目录名；旧实验目录命名不变。

### 35.7 修改文件

1. `hrl_mix/safe_training_pipeline.py`（新增）
2. `hrl_mix/config/safe_hrl_training_pipeline.json`（新增）
3. `hrl_mix/train_config.py`
4. `hrl_mix/train_eval.py`
5. `hrl_mix/train_runner.py`
6. `hrl_mix/train.py`
7. `tests/test_safe_training_pipeline.py`（新增）
8. `docs/SAFE_HRL_PROGRESS.md`

没有修改 `base/d3qn_agent.py`、`base/hrl_env.py`、`get_task_priority_v2`、CEWS
evaluator、固定 VM 规则或神经网络结构。

### 35.8 测试结果

阶段 14 定向与相邻回归：

```powershell
python -m unittest tests.test_safe_training_pipeline `
  tests.test_safety_lagrange `
  tests.test_safe_demonstration_pretraining -v
```

结果：`33 tests OK`，0 失败、0 跳过。覆盖五阶段配置、严格 seed 隔离、
validation-only 切换、连续达标、资源/工作流 profile、跨 training seed
round-robin、阶段指标、Q_c error、默认关闭、Windows 短路径、checkpoint
阶段与 Agent/lambda 恢复，以及五阶段短流程 smoke。

完整系统 Python unittest：

```powershell
python -m unittest discover -s tests -v
```

结果：`162 tests OK`，0 失败、0 跳过。

Anaconda/pytest 环境：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`136 passed, 26 skipped`；26 项均明确因为该环境未安装 PyTorch而跳过。
没有安装或升级任何核心依赖。`python -m py_compile` 覆盖本阶段全部 Python
修改文件并成功；`python -m hrl_mix.train --help` 成功显示两个新增入口。

未运行：正式五阶段长训练、正式安全示范数据生成、真实 admitted SeEvo 规则
数据、GPU/CUDA、最终测试 seed 评估、跨 Host/VM 拓扑训练和消融实验。不得声称
这些项目已经通过。

### 35.9 阶段状态与风险

阶段 14 编排代码、配置 schema、恢复协议和短流程测试完成，可以在准备好真实
阶段 1 数据 manifest 后进入小规模受控运行。主要风险：

- 配置模板引用的 `out/safe_demonstrations/manifest.json` 目前不是随库提供的
  正式数据；缺失时会明确停止，不能直接声称五阶段训练可产出结果；
- 当前真实历史 SeEvo LLM 规则仍没有 admitted 记录；
- validation 连续达标仅是所配置 seeds 上的经验标准，不是未见场景安全保证；
- fuzzy safety margin、risk finish 和 shield 风险仍是模型预测，不是真实完成后
  的安全保证；
- replay 和完整 RNG 状态未进入 pipeline checkpoint，恢复不是 bit-exact；
- 不同 Host/VM 拓扑必须独立训练，不能把同一 checkpoint 跨动作维度复用；
- 当前 `.git` 不是有效 repository，无法记录实际 branch/commit；pipeline
  以 plan hash 和各 Agent checkpoint schema 追踪版本，但不替代源码版本控制。

## 36. 下一阶段目标

先生成并审核真实阶段 1 数据 manifest，确认 train/validation/final-test seed
与 FCFS deadline cache 覆盖一致；然后用很小的固定 episode 计划实际串行运行
阶段 2、阶段 3 和至少一次课程切换，检查 JSONL 指标、环境 profile 重建、
checkpoint 中断恢复和显存/内存占用。该 smoke 审核通过后再扩大 episode 数，
最后只在冻结训练与课程后使用从未被 runner 消费的 final-test seeds 做零违反
标准评价。

## 37. 阶段 15：最优模型可行性优先保存

### 37.1 阶段范围与冻结定义

本阶段只修改 validation 指标聚合、best-model 比较、checkpoint 审计元数据和测试，
不修改 Manager–Host–VM 三层职责、D3QN 网络结构、动作选择、reward/cost、shield、
fallback、SeEvo evaluator 或 `get_task_priority_v2`。以下定义继续冻结：

```text
fuzzy energy score
  = fuzzy energy mean + 1.0 * fuzzy energy std

R(T_i)
  = 0.05 * T_i_modal + 0.95 * T_i_upper

D_i = deterministic deadline
```

安全模式下不再以平均 `eval_energy` 单独决定 best checkpoint。比较器使用严格字典序：

```text
(
    deadline_violation_rate,
    max_fuzzy_lateness,
    mean_fuzzy_lateness,
    fuzzy_energy_score,
)
```

因此零违反模型总是优于非零违反模型；都违反时依次比较违反率、最大模糊延迟和平均
模糊延迟；前三项相同时才比较风险调整模糊能耗。完全相同的 key 不覆盖已有 best，
避免相同结果反复写盘。`safe_rl.enabled=false` 时仍沿用历史
`eval_energy < best_eval_energy` 路径，旧训练语义不变。

### 37.2 多 seed 聚合定义

`evaluate_hrl_three_layer_multi_seed(..., return_safety_metrics=True)` 对每个 validation
seed 从现有 `wf_finish_time`、`_workflow_finish_tfn()` 和
`fuzzy_deadline_measure()` 重建已完成工作流的模糊延迟：

```text
fuzzy_lateness_i = max(0, R(T_i) - D_i)
```

没有引入第二套模糊模型。新增/明确字段如下：

- `all_seed_feasible`：每个 seed 都完成预期工作流，并且违反数与最大模糊延迟均为 0；
- `all_seed_evaluation_completed`：每个 seed 都完成其预期工作流；若为 false，runner
  fail-closed 停止本次 best-model 比较，避免把不完整评估误当成零违反；
- `feasible_seed_rate`：可行 seed 数 / validation seed 数；
- `worst_seed_violation`：各 seed 工作流违反率的最大值，字段单位是“率”而不是计数；
- `worst_seed_lateness`：各 seed 最大模糊延迟的最大值；
- `max_fuzzy_lateness`：全部 seed 已完成工作流的最大模糊延迟；
- `mean_fuzzy_lateness`：全部已完成工作流的模糊延迟总和 / 完成工作流数，包含未超期的
  零值；
- `per_seed_safety_metrics`：逐 seed 完成数、违反数、违反率、延迟和可行性审计记录。

配置 `safe_rl.model_selection.require_all_validation_seeds_feasible=true` 默认要求正式
validation 的所有 seed 可行。训练过程中尚无可行模型时仍会保留当前字典序最优的
非可行 checkpoint，供诊断和继续训练；该 checkpoint 的
`all_seed_feasible=false` 明确表明它不满足最终零违反准入，不得把“当前最好”描述成
“安全合格”。

### 37.3 Checkpoint 接口与版本

安全 best checkpoint 现在是一个版本化 bundle：

```text
best_manager.pth
best_host.pth
best_vm.pth
best_checkpoint_manifest.json
```

三个 Agent 文件继续由 `D3QNAgent.save()` 保存，分别包含：

```text
Q_r online / target / optimizer  -> online / target / optim
Q_c online / target / optimizer  -> q_c_online / q_c_target / q_c_optim
agent lagrange_multiplier
```

bundle manifest 同时绑定：

- 可行性优先指标、字段顺序和实际 comparison key；
- 共享 Lagrange controller 状态；
- 当前 curriculum stage/controller 状态；
- 三层 replay schema、transition 数、容量、PER 和风险类别计数等元数据；
- 启发式库 manifest ID/version/revision/hash 与 admitted heuristic IDs；
- 完整训练配置快照及其 SHA-256；
- 三个 Agent checkpoint 的相对路径与实际 checkpoint key 映射。

这里按需求保存的是 replay metadata，不包含 replay transition 本体，manifest 中
`replay_transitions_embedded=false` 明确标记这一边界。因此中断恢复仍不是 replay/RNG
层面的 bit-exact 恢复。

阶段化 pipeline checkpoint schema 从 1 升级为 2，并加入同样的 best metrics、
replay metadata、启发式库版本和配置快照。schema 1 会被明确拒绝，不能被静默当成
新版完整 checkpoint 恢复。兼容别名 `best_validation_energy` 仍写入 manifest，但它
只等于 best 的 `fuzzy_energy_score`，不再用于安全模式的模型排序。

### 37.4 配置与接口变化

新增：

```text
safe_rl.model_selection.enabled = true
safe_rl.model_selection.require_all_validation_seeds_feasible = true
safe_rl.model_selection.comparison_key =
  deadline_violation_rate,
  max_fuzzy_lateness,
  mean_fuzzy_lateness,
  fuzzy_energy_score
```

`return_safety_metrics=false` 的四元组评估返回值保持不变；开启安全统计时只扩展最后一个
字典。安全训练 CSV 增加 max/mean fuzzy lateness、全 seed 可行率和 worst-seed 字段。
未新增命令行参数，也未修改旧 checkpoint 的 Agent 网络加载协议。

### 37.5 修改文件

1. `hrl_mix/model_selection.py`（新增）
2. `hrl_mix/train_config.py`
3. `hrl_mix/train_eval.py`
4. `hrl_mix/train_runner.py`
5. `hrl_mix/safe_training_pipeline.py`
6. `tests/test_feasibility_first_model_selection.py`（新增）
7. `tests/test_safe_training_pipeline.py`
8. `docs/SAFE_HRL_PROGRESS.md`

未修改 `base/d3qn_agent.py`、`base/hrl_env.py`、SeEvo、CEWS evaluator、固定 VM
选择规则或任一神经网络结构。

### 37.6 测试结果

阶段 15 定向测试：

```powershell
python -m unittest tests.test_feasibility_first_model_selection `
  tests.test_safe_training_pipeline `
  tests.test_safety_lagrange `
  tests.test_safe_hrl_cost -v
```

结果：`42 tests OK`，0 失败、0 跳过。

完整系统 unittest：

```powershell
python -m unittest discover -s tests -v
```

结果：`174 tests OK`，0 失败、0 跳过。

Anaconda/pytest 环境：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`147 passed, 27 skipped`。27 项均因该环境未安装 PyTorch 而跳过，其中阶段 15
的纯排序、聚合和 checkpoint manifest 测试仍运行；只有需要通过 `train_eval` 导入
PyTorch 训练工具的模糊时间线集成测试跳过。未安装或升级任何核心依赖。

### 37.7 风险与下一阶段

- validation seed 上的零违反仍是有限样本的经验结果，不是真实部署安全保证；
- `worst_seed_violation` 是最大 seed 违反率，分析脚本不得将其误读为违反工作流数；
- schema 1 pipeline checkpoint 缺少本阶段要求的完整审计字段，必须重新保存为
  schema 2，不能直接恢复；
- periodic step/final Agent 文件仍是网络级 checkpoint；正式 best 与 pipeline 恢复应
  使用其 bundle manifest 进行整体审计；
- replay transition 与完整 RNG 状态未嵌入 checkpoint，恢复不保证 bit-exact；
- 当前目录的 `.git` 不是有效 Git repository，无法记录真实 branch/commit；
- 要求阅读的 `docs/LLM辅助安全分层强化学习改造步骤.md` 在工作区不存在；本阶段审计
  使用同名 `.docx` 的完整内容，后续应补回受版本控制的 Markdown 副本。

阶段 15 代码和回归测试已完成。可以进入下一阶段的小规模真实 validation-seed
训练/恢复 smoke，但只有当 best manifest 中 `all_seed_feasible=true` 且最终独立
test seeds 也满足零违反时，才可以把模型称为相应评估集上的可行模型。

## 38. 阶段 16：安全 HRL 训练与测试评价指标

### 38.1 阶段范围与不变项

本阶段新增统一的只读指标层，不修改环境转移、reward、safety cost、动作 mask、
safety shield、确定性 fallback、Manager/Host/VM 职责、D3QN 网络或 SeEvo
候选评价。以下定义保持冻结：

```text
fuzzy_energy_score
  = fuzzy_energy_mean + 1.0 * fuzzy_energy_std

R(T_i)
  = 0.05 * T_i_modal + 0.95 * T_i_upper

D_i = deterministic deadline
```

指标层通过环境已有的 `wf_finish_time`、`_workflow_finish_tfn()`、
`fuzzy_deadline_measure()`、`get_fuzzy_energy_summary()`、
`get_safety_shield_records()` 和 phase 启发式审计字段取数，没有建立第二套模糊
时间或能耗模型。`safe_rl.enabled=false` 时不创建新指标存储器，原训练和四元组
评估接口保持不变。

### 38.2 稳定 schema 与分源保存

新增 `safe_metrics` schema 1。显式启用 safe RL 且
`safe_rl.metrics.enabled=true` 时，在 checkpoint 输出目录的
`safe_metrics/` 子目录分别写入：

```text
training_metrics.csv
training_metrics.jsonl
validation_metrics.csv
validation_metrics.jsonl
final_test_metrics.csv
final_test_metrics.jsonl
final_test_metrics.json
llm_comparison_metrics.json      # 仅显式执行同 seed 严格 DDL 配对消融时生成
```

训练 episode 与 validation 不再混在同一个新指标文件中。CSV 三类来源共享冻结的
字段顺序 `SAFE_METRIC_CSV_FIELDS`；JSONL 额外保留 `per_seed_metrics`，防止
CSV 聚合值隐藏单个 seed。向已有 CSV 追加前会核对完整表头；schema 不一致时明确
拒绝，不会静默混写。`metrics_schema_version`、`metric_source`、`global_step`
和 `episode` 可追踪每条记录。最终测试只能通过
`evaluate_and_save_safe_hrl_final_test()` 显式执行；该入口要求 training、
validation、final-test 三组 seed 两两无交集，在线 trainer 不调用它，也不消费
final-test seeds。

新增配置：

```text
safe_rl.metrics.enabled = true
safe_rl.metrics.schema_version = 1
safe_rl.metrics.output_subdir = "safe_metrics"
safe_rl.metrics.convergence_window = 5
```

外层 `safe_rl.enabled` 仍默认为 `false`。`output_subdir` 必须是非空相对子路径，
不得用绝对路径或 `..` 跳出 run 目录。

### 38.3 安全结果指标定义

对已真实完成的工作流使用
`L_i^F = max(0, R(T_i) - D_i)` 和
`S_i^F = D_i - R(T_i)`：

| 稳定字段 | 定义 |
|---|---|
| `fuzzy_ddl_violation_rate` | 全部已完成工作流中 `R(T_i) > D_i` 的比例 |
| `feasible_workflow_ratio` | 全部已完成工作流中 `R(T_i) <= D_i` 的比例 |
| `feasible_episode_ratio` | 完整结束且零违反 episode 数 / 本次聚合 episode 数；多 seed validation 中每个 seed 对应一个 episode |
| `all_seed_feasible` | 每个 seed 都完整结束且该 seed 的完成工作流全部零违反 |
| `feasible_seed_rate` | 可行 seed 数 / evaluation seed 数 |
| `mean_fuzzy_lateness` | `sum_i L_i^F / completed_workflow_count`，未超期工作流以 0 计入 |
| `max_fuzzy_lateness` | 全部 seed、全部已完成工作流的最大 `L_i^F` |
| `minimum_fuzzy_safety_margin` | 全部已完成工作流的最小 `S_i^F`；负值表示真实完成后违反 |

同时稳定输出 `worst_seed_violation_rate`、
`worst_seed_fuzzy_lateness` 和
`worst_seed_minimum_fuzzy_safety_margin`。这些字段直接取最坏 seed，不对
worst case 求 episode 平均；`all_seed_evaluation_completed` 与
`completed_evaluation_seed_rate` 单独标记是否所有 seed 完整执行，避免把未完成
评估误报为安全。

### 38.4 安全控制指标定义

控制指标的基本分母 `shield_record_count` 是环境记录的 Host/VM shield 决策数：

| 稳定字段 | 定义 |
|---|---|
| `shield_intervention_count/rate` | `shield_intervened=true` 的记录数及其占全部 shield 决策的比例 |
| `no_safe_action_count/rate` | `fallback_reason=empty_safe_action_set` 或等价空安全集原因的记录数及比例 |
| `fallback_count/rate` | `fallback_applied=true` 的 Host/VM 执行动作数及比例 |
| `proposed_executed_action_mismatch_count/rate` | proposed 与 executed action 不相同的记录数及比例 |
| `q_c_prediction_error` | 最近一次已有 Q_c 更新的三层 Agent 的 Safety Bellman Huber loss 均值 |
| `q_c_prediction_error_sample_count` | 实际提供上述误差的 Agent 层数；不是 transition 数 |
| `lambda_current` | episode 结束并执行本次 episode/EMA 更新后的共享 `lambda_DDL` |
| `lambda_trajectory` | 按 training、validation、final-test 来源分别追加的 `lambda_current` 序列 |

`Q_c prediction error` 是训练诊断，不是校准后的违反概率误差，也不是安全保证。
空安全集、fallback 与 proposed/executed 不一致分别统计：一次 fallback 可以同时
落入多个诊断类别，但每个字段内部对一条 shield 记录只计一次。

### 38.5 性能指标定义

| 稳定字段 | 定义 |
|---|---|
| `fuzzy_energy_mean` | 各 seed 环境模糊总能耗 mean 的算术平均 |
| `fuzzy_energy_std` | 各 seed 环境模糊总能耗内生 std 的算术平均 |
| `fuzzy_energy_score` | 各 seed 的 `mean + 1.0 * std` 的算术平均 |
| `fuzzy_energy_score_across_seed_std` | 各 seed fuzzy energy score 的总体标准差；与内生 `fuzzy_energy_std` 分开 |
| `modal_energy` | 各 seed modal 总能耗的算术平均 |
| `scheduling_time_seconds` | 本次聚合中各 seed 调度墙钟时间总和 |
| `mean_seed_scheduling_time_seconds` | 单 seed 调度墙钟时间均值 |
| `worst_seed_scheduling_time_seconds` | 最慢 seed 调度墙钟时间 |
| `convergence_speed` | 首次出现连续 `convergence_window` 次 validation 均 `all_seed_feasible=true` 时，取 `1 / evaluations_to_convergence`；未达到时为 0 |

训练 scheduling time 从 episode 开始计至 episode 完成，包含在线动作推理、
transition 写入和该 episode 内的网络更新，但不包含 episode 结束后的 lambda
更新、指标聚合或 validation；validation/final-test 时间只覆盖确定性推理和环境
调度。该墙钟指标受平台、负载和硬件影响，不应跨异构机器直接比较。当前
convergence 定义衡量达到连续安全可行窗口的速度，不代表能耗目标已经全局收敛。

### 38.6 LLM 辅助指标定义

| 稳定字段 | 定义 |
|---|---|
| `selected_llm_heuristic_frequency` | `heuristic_source=seevo_llm` 的 Manager phase 数 / 全部启发式选择 phase 数 |
| `llm_associated_shield_rate` | LLM 启发式被选中 phase 内的 shield 干预数 / 同类 phase 的 shield 决策数 |
| `energy_improvement_from_llm` | 同 seed/场景配对时 `(score_without_llm - score_with_llm) / score_without_llm` |
| `convergence_acceleration` | 同配置配对时 `(evals_without_llm - evals_with_llm) / evals_without_llm` |
| `strict_ddl_feasibility_with_llm` | 严格 DDL、有 LLM 候选运行的 `feasible_episode_ratio` |
| `strict_ddl_feasibility_without_llm` | 严格 DDL、无 LLM 候选配对运行的 `feasible_episode_ratio` |
| `strict_ddl_feasibility_improvement` | 上述两种严格 DDL 可行 episode 比例之差 |

能耗改善、收敛加速和严格 DDL 对比只能由
`compute_llm_assistance_metrics(..., strict_ddl=true)` 或
`SafeMetricStore.save_llm_comparison()` 对同一 evaluation seed 集的有/无 LLM
报告配对生成。seed 数或 seed ID 不一致、或调用者未明确声明 strict DDL 时拒绝
计算。单次训练/评估报告中的这些比较字段保持 `null`，不得用 0 伪装成已完成消融
实验。该指标只描述 Manager 选择 ready-task 启发式带来的关联表现；LLM 仍不选择
Host 或 VM。

### 38.7 接口变化

- 新增 `hrl_mix.safe_metrics.build_episode_metric_record()`：从一个已结束环境构造
  seed/episode 明细。
- 新增 `aggregate_safe_metric_records()`：按工作流加权聚合比例和 lateness，同时
  保留 per-seed 与 worst-seed 证据。
- 新增 `SafeMetricStore`：分源保存 CSV/JSONL、恢复 lambda/validation 序列并计算
  可追踪 convergence。
- `evaluate_hrl_three_layer_multi_seed(..., return_safety_metrics=true)` 的 tuple
  形状不变，最后一个安全字典扩展为 schema 1 全量指标；`false` 时原四元组不变。
- 新增 `evaluate_and_save_safe_hrl_final_test()`：显式校验 seed 隔离并保存最终测试
  报告。
- `train_runner` 仅在 safe RL 和 metrics 子开关同时启用时分别记录 training 与
  validation；不自动运行 final test。
- `get_task_priority_v2`、环境 step/phase 返回元组、replay、checkpoint 和三层
  Agent 接口均未改变。

### 38.8 修改文件

1. `hrl_mix/safe_metrics.py`（新增）
2. `hrl_mix/train_config.py`
3. `hrl_mix/train_eval.py`
4. `hrl_mix/train_runner.py`
5. `tests/test_safe_metrics.py`（新增）
6. `docs/SAFE_HRL_PROGRESS.md`

未修改 `base/hrl_env.py`、`base/d3qn_agent.py`、SeEvo、CEWS evaluator、
固定 VM 规则、神经网络结构或动作选择策略。

### 38.9 测试结果

阶段 16 聚合、持久化及相邻回归测试：

```powershell
python -m unittest tests.test_safe_metrics `
  tests.test_feasibility_first_model_selection `
  tests.test_safe_training_pipeline `
  tests.test_safety_lagrange -v
```

结果：`49 tests OK`，0 失败，0 跳过。覆盖安全/控制/性能/LLM 四类聚合、
worst-seed 保留、冻结模糊风险公式、分源 CSV/JSON、稳定表头、旧 schema 拒绝、
lambda trajectory、convergence、同 seed LLM 配对、final-test seed 隔离以及
`safe_rl=false` 默认兼容。

完整系统 unittest：

```powershell
python -m unittest discover -s tests -v
```

结果：`187 tests OK`，0 失败，0 跳过。

Anaconda/pytest 环境：

```powershell
D:\anaconda3\python.exe -m pytest tests -q -ra
```

结果：`158 passed, 29 skipped`，0 失败。29 项均明确因该环境未安装 PyTorch 而
跳过，其中阶段 16 的两个 `train_eval` 最终测试入口集成测试跳过；纯 Python 的
指标聚合、CSV/JSON 持久化、worst-seed 和 LLM 配对测试均已执行通过。没有安装或
升级任何核心依赖。

### 38.10 风险与下一阶段

- 当前只运行了单元/集成测试，没有运行正式长训练、GPU/CUDA、跨平台计时基准、
  admitted SeEvo 真实规则消融或 withheld final-test seed 实验；不得声称这些实验
  已通过。
- `scheduling_time_seconds` 是墙钟时间，受操作系统、CPU/GPU、磁盘和并发负载影响。
- `convergence_speed` 是“连续 validation 全 seed 可行”的操作性定义，不代表
  fuzzy energy 已达到全局最优。
- 有/无 LLM 指标必须来自冻结配置和相同 seed 的严格配对；否则辅助效果不可归因。
- validation/final-test 上的零违反是有限样本经验结果，不是真实部署安全保证。
- fuzzy finish、动态 safety margin 与动作风险仍是模型预测；真实完成后的
  violation/lateness 指标才是结果统计。
- 当前 `.git` 不是有效 repository，仍无法记录真实 branch/commit。
- 要求阅读的 `docs/LLM辅助安全分层强化学习改造步骤.md` 仍不存在；本阶段使用
  同名 `.docx` 的完整内容完成审计。

阶段 16 的指标实现、分源持久化和回归测试已完成，可以进入下一阶段的小规模真实
训练/validation 指标 smoke。进入正式实验前，应先冻结 training/validation/
final-test seeds，并以显式最终测试入口验证 withheld seeds；只有最终测试报告也
满足零违反，才能把模型称为该测试集上的可行模型。

## 39. 阶段 17：可复现对比与安全机制消融实验配置

### 39.1 阶段范围与冻结语义

本阶段新增独立的实验规划层，不修改环境、训练 runner、D3QN、SeEvo evaluator、
Manager/Host/VM 动作语义或固定 VM 规则。规划层只读取已有配置和数据，验证实验
公平性，物化随机夹具并输出确定性 JSON manifest；它不会加载或执行任意 LLM
Python 源文件，也不会启动训练。

以下数学语义继续冻结：

```text
fuzzy_energy_score
  = fuzzy_energy_mean + 1.0 * fuzzy_energy_std

R(T_i)
  = 0.05 * T_i_modal + 0.95 * T_i_upper

D_i = deterministic deadline
```

### 39.2 共享实验协议

`hrl_mix/config/safe_hrl_experiment_matrix.json` 冻结一个 SS/Tight 参考矩阵。全部
方法和消融共享：

- 5 个 small DAX 文件及各自 SHA-256；
- FCFS deadline cache 及 SHA-256；
- 50 个工作流、`arrival_lambda=0.03`、相同 DDL alpha 采样规则；
- 2 cloud Host + 1 edge Host、相同 VM 数量、计算能力与带宽档位；
- `delta1=0.75`、`delta2=1.2`、`lambda_E=1.0`、`eta=0.95`；
- 相同的 training、validation、final-test case 集；
- safe metrics schema 1 的同一稳定指标字段集合。

当前环境的 `random_seed` 同时驱动 DAX 选择、工作负载随机化、到达过程和 deadline
cache 索引，尚不能把 workflow seed 与 arrival seed 独立注入。因此矩阵显式要求
每个 case 的 `episode_seed == workflow_seed == arrival_seed`，并把这种耦合写入
manifest；`resource_seed` 仍单独记录。training 使用 seed 1--5，validation 使用
101--103，final-test 使用 201--203，三组任意 seed 角色均不得交叉。

生成器按照当前环境实际调用链物化并记录：

- 精确 arrival time 序列；
- 精确 DAX 名称序列；
- 每个工作流的 payload seed；
- 每个工作流的 DDL alpha 序列；
- fuzzy resource seed；
- 每个 case fixture 的 SHA-256。

除 `single_seed_training` 仅有意缩减 training subset 外，任何方法或消融均不能用
`shared_overrides` 修改工作流、到达、资源、DDL、模糊参数或指标。所有运行都绑定
同一个 `shared_protocol_sha256`，并使用完全相同的 validation/final-test case ID。

### 39.3 七种对比方法

| 方法 ID | 冻结语义 | 当前执行状态 |
|---|---|---|
| `original_hrl` | 原规则权重 Manager + Host/VM D3QN；全部 safe/LLM 开关关闭 | 现有公开 runner 可达 |
| `original_hrl_plus_llm` | 旧 observation/reward 不变，Manager 选择传统或已准入 ready-task 规则 | 已配置；缺少 legacy heuristic-selection adapter |
| `safe_hrl_without_llm` | 完整安全 HRL，Manager 只选 5 个传统启发式 | 已配置；现有 loader 不能在未来存在 admitted LLM 时显式排除其动作槽 |
| `llm_augmented_safe_hrl` | 完整安全 HRL，Manager 可选传统和 admitted SeEvo 规则 | 已配置；当前无 admitted LLM 且示范 manifest 缺失 |
| `seevo_best_heuristic_only` | 固定最佳 admitted SeEvo ready-task 规则 + 现有固定 VM 资源规则 | 已配置；当前无 admitted LLM 且缺少统一 baseline adapter |
| `edf_baseline` | 固定 EDF ready-task 排序 + 现有固定 VM 资源规则 | 已配置；缺少统一 baseline adapter |
| `fcfs_fcfs` | 固定 FCFS ready-task 排序 + 最早可用合法 VM | 已配置；独立共享模糊环境 runner 可达 |
| `fcfs_fixed` | 固定 FCFS ready-task 排序 + 现有确定性 fixed VM 规则 | 已配置；独立共享模糊环境 runner 可达 |

这里“已配置”不等于“已训练”。manifest 对未实现 adapter、无 admitted LLM 或缺失
示范数据逐项写入 `blocking_reasons` 并设置 `execution_ready=false`，不会默默切换
模式或执行来源不明的规则。当前只有 `original_hrl` 可直接按配置入口执行；本阶段
没有运行它的正式训练。

### 39.4 七种安全机制消融

每个消融都以 `llm_augmented_safe_hrl` 为基准，并以结构化
`disabled_components` 记录 component、原值、目标值和影响：

| 消融 ID | 明确关闭/替换内容 |
|---|---|
| `no_safety_shield` | 关闭 DDL shield；因空安全集合不存在，同时关闭 safety fallback，动作只受 legal mask |
| `no_safety_value_network` | 关闭三层 Q_c 与 Q_c 拉格朗日评分；cost、task boundary、shield、fallback 保留 |
| `fixed_lambda` | Q_c 保留；关闭动态更新，固定 `lambda_DDL=1.0` |
| `no_task_level_safety_deadline` | 关闭 `D_safe_ij` 任务级动态边界；工作流级 safety cost/诊断保留 |
| `no_fallback_controller` | shield 保留；空安全集合时显式终止不可行决策，不偷偷选择动作 |
| `no_llm_demonstration_pretraining` | 关闭全部示范预训练；在线 Manager 的 admitted LLM 候选语义保留 |
| `single_seed_training` | 只用 training split 第一个 seed；validation/final-test 完全不变 |

校验器要求文档声明与实际 component override 完全一致，字段遗漏或 from/to 不一致
会立即拒绝配置。

### 39.5 信息访问合同

每个方法绑定一个可哈希的 `information_contract_id`。合同分别列出 Manager、Host、
VM 和 LLM rule 的输入/输出：

- LLM rule 非空输入必须严格等于现有 `get_task_priority_v2` 的 8 个 ready-task
  数组，唯一输出是 `ready_task_priority_scores`；
- Host/VM 输入字段中禁止出现 LLM 信息；
- manifest 中所有运行均固定 `llm_host_vm_access=false`；
- legacy HRL 禁止打开 safety cost、Q_c、shield、task boundary、fallback 或
  lambda；
- 固定启发式 baseline 不训练，也不读取 Manager/Host/VM observation；
- 方法和消融不能覆盖共享数据协议。

因此 LLM 仍只影响 ready-task 排序，不选择 Host 或 VM；资源动作仍由对应 Agent、
shield 或已有固定 VM 规则负责。

### 39.6 新增接口与 manifest

新增只读接口：

- `load_experiment_matrix(path)`：严格验证 schema、公平性、seed 隔离、模糊常数、
  信息合同、方法及消融语义；
- `build_experiment_manifest(path, smoke=False)`：物化确定性 case fixtures 和
  运行就绪性，不执行调度或训练；
- `write_experiment_manifest(manifest, output_path)`：原子写入稳定 JSON；
- CLI：`python -m hrl_mix.experiment_matrix --config ... --output ...
  [--smoke]`。

没有修改任何已有训练、环境、Agent、replay、checkpoint 或
`get_task_priority_v2` 接口。输出：

```text
out/experiment_manifests/safe_hrl_stage17_smoke_manifest.json
  manifest_sha256 = 5e4b922aaa2055eca777c7a7bafd839038db06391b32ebfe9adc31be103c132f
  file_sha256 = c2ff48c4611c7192abb09122cd6606a13e1646bf4398b13d3c532133d123e1a6

out/experiment_manifests/safe_hrl_stage17_full_plan_manifest.json
  manifest_sha256 = eb04a912382a1fcb7011a3cbb562d9e5380769b15624539ed520df6de0254245
  file_sha256 = a5869353a6cfb15b96a416716ce6d45fb00e9672cbbd8e8311a4a2c167ebfae3
```

第一个只包含每个 split 1 个 case、每个 episode 2 个工作流的配置 smoke；第二个
是完整实验计划。二者均明确记录 `training_executed=false` 和
`formal_experiment_executed=false`，没有产生性能或安全结论。

### 39.7 修改文件

1. `hrl_mix/experiment_matrix.py`（新增）
2. `hrl_mix/config/safe_hrl_experiment_matrix.json`（新增）
3. `tests/test_experiment_matrix.py`（新增）
4. `out/experiment_manifests/safe_hrl_stage17_smoke_manifest.json`（新增）
5. `out/experiment_manifests/safe_hrl_stage17_full_plan_manifest.json`（新增）
6. `docs/SAFE_HRL_PROGRESS.md`

未修改 `base/hrl_env.py`、`base/d3qn_agent.py`、`hrl_mix/train_runner.py`、
SeEvo、CEWS evaluator、固定 VM 规则、神经网络或调度策略。

### 39.8 测试结果

极小配置 smoke：

```powershell
python -m hrl_mix.experiment_matrix `
  --config hrl_mix/config/safe_hrl_experiment_matrix.json `
  --output out/experiment_manifests/safe_hrl_stage17_smoke_manifest.json `
  --smoke
```

结果：成功解析 7 个方法和 7 个消融，生成 14 个 run；1 个 run 当前
`execution_ready=true`，13 个按已记录原因 fail-closed；未执行训练。

阶段 17 与相邻回归：

```powershell
python -m unittest tests.test_experiment_matrix `
  tests.test_safe_metrics `
  tests.test_safe_training_pipeline `
  tests.test_safe_manager_heuristics `
  tests.test_heuristic_admission `
  tests.test_feasibility_first_model_selection -v
```

结果：`72 tests OK`，0 失败，0 跳过。

完整系统 unittest：

```powershell
python -m unittest discover -s tests -v
```

结果：`200 tests OK`，0 失败，0 跳过。运行环境为 Python 3.12.10、
Windows 10 10.0.19045；本阶段未发现新的平台相关失败，也未安装或升级依赖。

### 39.9 风险与下一步

- 现有 `heuristic_selection_mode` 被实际调用链约束为同时开启 safe RL、shield 和
  safe state，不能直接实现 `original_hrl_plus_llm`；建议下一阶段增加窄范围的
  legacy Manager heuristic-selection adapter，而不是为旧基线偷偷启用安全栈。
- `load_manager_heuristic_library()` 总会建立传统规则和 manifest 中所有稳定 LLM
  动作槽。为保证 `safe_hrl_without_llm` 在未来出现 admitted LLM 后仍严格无 LLM，
  需要增加显式 candidate-source filter；本阶段没有改写 loader。
- 目前启发式 manifest 中没有 admitted LLM 规则，示范数据 manifest 也不存在，
  所以所有依赖 LLM 或示范预训练的配置必须保持不可执行。
- EDF、FCFS 和 best-SeEvo-only 尚缺统一的固定启发式评估 adapter；不能把单独的
  CEWS 候选评价调用伪装成已完成的 HRL 对比实验。
- 当前环境耦合 episode/workflow/arrival seed。manifest 已如实冻结这种耦合，但若
  研究需要独立控制到达与工作流随机性，应在后续专门阶段扩展环境接口并保持默认
  兼容，而不是在配置层伪造独立 seed。
- 配置 smoke 只证明矩阵、输入哈希、随机夹具、公平性与信息合同可解析，不证明训练
  收敛、零违反、能耗改进、LLM 有效性或跨平台运行性能。
- validation/final-test 的未来零违反仍只会是有限 seed 的经验结果，不是真实部署
  安全保证；预测 fuzzy finish 和 safety margin 也不是事后安全保证。
- `.git` 仍不是有效 repository，无法给这些 manifest 绑定真实 branch/commit。
- 要求阅读的 `docs/LLM辅助安全分层强化学习改造步骤.md` 仍不存在；本阶段继续使用
  同名 `.docx` 的完整内容审计。

阶段 17 的可复现实验矩阵、fail-closed 校验、manifest 和极小配置 smoke 已完成。
可以进入下一阶段的 adapter 补齐与逐方法短运行，但在 13 个阻塞运行逐一变为
`execution_ready=true`、admitted LLM 与示范数据就绪前，不可以开始或声称完成
完整对比/消融实验。

## 40. 阶段 18：最终代码审计与小规模验收

### 40.1 审计边界与版本状态

- 审计日期：2026-07-28。
- 审计标识：`safe-hrl-final-small-acceptance-20260728`。
- 当前分支/commit：仍不可获取；根目录 `.git/` 为空，`git status` 返回
  `not a git repository`。因此本次只能用文件 hash、manifest 和测试记录追踪，
  不能声称已经绑定 Git branch、commit 或 tag。
- 已完整阅读
  `docs/CEWS_TASK_CONSTRUCTIVE_SNAPSHOT_2026-07-28_PRE_SAFE_RL.md`。
- 要求的 `docs/LLM辅助安全分层强化学习改造步骤.md` 实际不存在；已完整读取同名
  `.docx`。这是交付文档缺口，不能把 DOCX 描述成仓库中存在的 Markdown。
- 本阶段没有修改模糊能耗、模糊 DDL、reward/cost 定义、shield/fallback 算法、
  D3QN 网络结构或 `get_task_priority_v2`。

### 40.2 最终架构确认

| 审计项 | 结论 | 验证级别 |
|---|---|---|
| 1. 原始 HRL 可运行 | 通过；`safe_rl=false` 的真实 1-workflow/1-episode runner 完整结束并保存、恢复三层 checkpoint | 已测试 |
| 2. safe RL 正确启用 | 通过；safe、shield、safe state、dynamic lambda、heuristic Manager 同时开启的真实短运行完整结束 | 已测试 |
| 3. SeEvo 只生成 ready-task 排序 | 通过；候选仅实现八参数 `get_task_priority_v2` 并返回 ready-task 分数 | 代码审计 + 测试 |
| 4. Manager/Host/VM 职责 | 通过；heuristic 模式 Manager 动作为 `heuristic_index`，Host/VM 仍由各自 Agent 决策，LLM 无 Host/VM 输入 | 代码审计 + 测试 |
| 5. 模糊能耗公式 | 仍为 `mean + 1.0 * std` | 代码审计 + 实际 CEWS smoke |
| 6. 模糊 DDL 公式 | 仍为 `0.05 * modal + 0.95 * upper`，`D_i` 仍为确定值 | 代码审计 + 单元测试 |
| 7. reward/cost 分离 | performance reward 与 safety cost 独立存储、独立 Bellman target，cost 未回加 reward | 单元测试 + runner 日志字段 |
| 8. 三类 mask 分离 | `legal_action_mask`、`safety_action_mask`、`final_action_mask` 独立保留 | 单元/环境集成测试 |
| 9. 空安全集合 fallback | 调用确定性 fuzzy-DDL fallback；Host 来自其内部最佳 VM | 单元/环境集成测试 |
| 10. `Q_r/Q_c` 独立 | 每层均有独立 online、target、optimizer，三层也不共享参数 | 单元测试 + checkpoint 检查 |
| 11. lambda 保存恢复 | 三层 checkpoint 均含 multiplier 和 controller state，可恢复 | 单元测试 + 真实 checkpoint |
| 12. 安全探索 | 探索/利用均限制在 final mask，空集合直接 fallback | 单元测试 |
| 13. replay 动作语义 | transition 用 executed action 训练；proposed action 只用于审计 | 单元测试 + 调用链审计 |
| 14. 模型保存 | safe 模式按违反率、最大/平均模糊延迟、模糊能耗字典序保存 | 单元测试 + 真实 manifest |
| 15. 多 seed worst seed | 保存 `all_seed_feasible`、`feasible_seed_rate`、`worst_seed_violation`、`worst_seed_lateness` | 单元测试 + 真实 manifest |
| 16. LLM 准入 | 只加载 admitted 且 hash、接口、上下文有效的规则；失败关闭 | 单元测试；真实 admitted 规则尚未具备 |
| 17. safe 关闭兼容 | 保留旧 observation、replay、epsilon-greedy、Q_r 和输出表头/命名 | 单元测试 + 真实短运行 |
| 18. NaN/空 mask/维度/循环 | 有有限性、mask、checkpoint 维度、最大步数与 stalled-step 防护；本次短运行无 NaN、空 mask 崩溃或死循环 | 已测试到短流程；未做压力证明 |
| 19. 关键分支覆盖 | 核心算法分支均有单元测试；真实 admitted LLM、长课程恢复、跨平台和正式多 seed 仍是缺口 | 部分已测试 |
| 20. 文档状态区分 | 本节明确区分已实现、已测试、尚未验证和尚未实现/不具备工件 | 已更新 |

预测 fuzzy finish、动态 safety margin 和动作风险始终是模型预测，不是实际完成后的
安全保证。实际 `violation/lateness` 才是事后结果指标；有限 seed 的零违反也不是
部署级安全证明。

### 40.3 本阶段确认并修复的缺陷

1. `LLM/main.py` 曾硬编码并覆盖 `QWEN_API_KEY`。已删除该赋值；现在只允许
   `LLM/utils/utils.py` 从环境变量读取。原密钥曾进入源码，必须在供应商侧轮换/吊销，
   仅从工作区删除不能证明旧密钥已失效。
2. safe 配置把全部阶段标签拼入目录名，在当前 Windows 工作区触发
   `WinError 123`。safe 非 pipeline 运行现使用短、确定且带 12 位配置摘要的名称；
   legacy 名称保持逐字不变，不同 safe 配置不会静默共用目录。
3. Manager safe replay 把 `final_action_mask` 同时作为历史位置参数和命名参数传给
   `D3QNAgent.remember()`，真实运行报 `unexpected keyword argument`。现由位置参数
   继续表达学习用 final mask，写入前从具名附加 metadata 中移除重复项；其余
   legal/safety mask、proposed/executed 等字段不变。
4. runner 正常结束时未显式关闭 CSV logger。现正常结束前关闭；logger 的关闭操作
   也改为幂等，并支持 context/finalizer 的尽力释放。异常中止后的全部资源清理仍未
   经过故障注入验收。
5. 阶段 17 实验矩阵把 `get_task_priority_v2` 输入写成与真实接口不一致的泛化名称。
   已改为实际八参数，并增加与 reference 函数签名逐项一致的测试；两个阶段 17
   manifest 已重新生成。

### 40.4 配置与接口说明

- `safe_rl.enabled=false`：保持 legacy observation、单 `Q_r`、旧 replay、
  原 epsilon-greedy、legacy Manager 权重语义和旧 run name。
- 完整短验收 safe 配置显式开启：
  `safe_rl`、`shield`、`state`、`dynamic lambda` 和
  `heuristic_selection_mode`。这些子功能不会因默认值被偷偷开启。
- `lambda_E=1.0`、`eta=0.95`、确定截止期、最终评估零违反预算均未改变。
- safe run name 是输出路径兼容性修复，不改变训练语义；摘要绑定关键 safe/offline
  配置。pipeline 仍沿用其独立的短名称。
- `get_task_priority_v2` 公开接口未改变，真实输入依次为：
  `min_exec_time`、`min_comm_time`、`min_incremental_energy`、`slack`、
  `upward_rank`、`remaining_work`、`ready_wait_time`、`uncertainty`。
- `train()`、环境 step/phase 返回元组、D3QN 输出维度均未改变。

重新生成的阶段 17 manifest：

```text
out/experiment_manifests/safe_hrl_stage17_smoke_manifest.json
  manifest_sha256 = 5e4b922aaa2055eca777c7a7bafd839038db06391b32ebfe9adc31be103c132f
  file_sha256 = c2ff48c4611c7192abb09122cd6606a13e1646bf4398b13d3c532133d123e1a6

out/experiment_manifests/safe_hrl_stage17_full_plan_manifest.json
  manifest_sha256 = eb04a912382a1fcb7011a3cbb562d9e5380769b15624539ed520df6de0254245
  file_sha256 = a5869353a6cfb15b96a416716ce6d45fb00e9672cbbd8e8311a4a2c167ebfae3
```

两者仍是配置/manifest 生成，不是正式实验结果。

### 40.5 修改文件

1. `LLM/main.py`
2. `common/metrics_logger.py`
3. `hrl_mix/train_config.py`
4. `hrl_mix/train_runner.py`
5. `hrl_mix/experiment_matrix.py`
6. `hrl_mix/config/safe_hrl_experiment_matrix.json`
7. `tests/test_experiment_matrix.py`
8. `tests/test_safety_lagrange.py`
9. `tests/test_final_safe_hrl_acceptance.py`（新增）
10. `out/experiment_manifests/safe_hrl_stage17_smoke_manifest.json`
11. `out/experiment_manifests/safe_hrl_stage17_full_plan_manifest.json`
12. `docs/SAFE_HRL_PROGRESS.md`

### 40.6 测试与环境结果

环境：Windows 10 10.0.19045、Python 3.12.10、NumPy 2.1.2、
PyTorch 2.5.1+cu121、PyYAML 6.0.1；`torch.cuda.is_available()` 为 true。

1. 静态语法检查：

   ```powershell
   python -m compileall -q .
   ```

   结果：通过，退出码 0。

2. 完整 unittest（包含全部原测试和新增最终验收）：

   ```powershell
   python -m unittest discover -s tests -p "test_*.py" -v
   ```

   结果：`205 tests OK`，0 失败，0 跳过。

   另按 shield、fallback、双价值、exploration、replay、模型选择、准入、
   heuristic Manager、指标和 lambda 显式列出安全模块运行：
   `106 tests OK`，0 失败，0 跳过。

3. 新增最终验收包含：
   - 原始 HRL：真实 1 workflow、1 episode runner，三层 final checkpoint 保存并
     用新 Agent 实际恢复；
   - safe HRL：真实 1 workflow、1 episode，完整 safe 开关和 heuristic Manager，
     三层 `Q_r/Q_c`、target、optimizer、lambda/controller state 保存并恢复；
   - best manifest：验证可行性优先策略、worst-seed、replay metadata、
     heuristic library version 和 config snapshot；
   - CEWS：reference `get_task_priority_v2`、1 seed、1 workflow 的实际 evaluator
     smoke，验证接口、完成状态及 `objective=mean+std`；
   - 源码凭据回归：验证 `LLM/main.py` 不再嵌入 Qwen 密钥。

4. 阶段 17 配置 smoke 重新运行：
   - 14 个 run 均成功解析；
   - 当前 1 个 `execution_ready=true`、13 个 fail-closed；
   - 未运行训练，未产生性能或安全结论。

5. 在线 SeEvo 极小生成命令已尝试，但在 LLM 客户端初始化阶段按预期明确失败：
   `AssertionError: Please set the environment variable QWEN_API_KEY`。因此：
   - SeEvo/LLM 在线规则生成 smoke：**未完成，缺少凭据**；
   - CEWS evaluator 离线 smoke：**已完成并通过**；
   - 没有发出 LLM API 请求，也没有把该在线失败记为通过。

6. `python -m pytest ...` 已尝试，当前解释器返回
   `No module named pytest`。缺失依赖为 pytest；没有为本次审计安装或升级依赖。
   可用的标准库 unittest 全量入口已完成上述 205 项。

7. `python -m hrl_mix.train --help` 与 `python main.py --help` 均退出码 0，确认
   HRL CLI 和根 SeEvo/Hydra 入口可解析；这不等同于在线训练/生成已经执行。

### 40.7 已知限制、尚未验证和尚未实现

已实现但尚未正式验证：

- 跨多个训练/验证 seed 的长时间收敛、动态 lambda 稳定性和 Q_c 校准；
- 课程各阶段的真实长运行切换/恢复以及大 replay 的持久化压力；
- 大规模 no-safe-action/fallback 触发分布和 safety prediction 校准；
- 不同 Windows/Linux、不同 GPU/CPU 和多进程并发下的性能与确定性；
- 异常中止、磁盘满、部分 checkpoint 写入失败后的完整恢复。

当前不具备或尚未实现的实验工件/adapter：

- 默认安全启发式库没有与当前上下文匹配的 admitted LLM 规则；测试中的 admitted
  规则是隔离 fixture，不能当作真实准入结果；
- 缺少正式安全示范数据 manifest，离线预训练未做真实数据长运行；
- `original_hrl_plus_llm` 的 legacy heuristic-selection adapter、
  `safe_hrl_without_llm` 的显式来源过滤器，以及 EDF/FCFS/best-SeEvo-only 的统一
  实验 adapter 仍未补齐；
- 在线 SeEvo smoke 因缺少 `QWEN_API_KEY` 未运行；
- `pytest` 未安装；
- 指定的改造步骤 Markdown 和有效 Git 元数据缺失。

### 40.8 尚不能声称的结论

- 不能声称系统已在正式规模、全部场景或 withheld final-test seeds 上零违反；
- 不能声称安全预测构成真实安全保证或 safety shield 永远存在安全动作；
- 不能声称 safe HRL 已收敛、优于 original HRL 或提高能耗表现；
- 不能声称 LLM 带来能耗改进、收敛加速或严格 DDL 可行性提升；
- 不能声称真实 admitted SeEvo 规则、正式示范预训练或完整消融实验已完成；
- 不能声称跨平台可复现，也不能给当前工作区绑定 Git commit；
- 不能声称在线 SeEvo 或 pytest 测试通过。

### 40.9 正式大规模实验前检查清单

- [ ] 在密钥供应商侧轮换/吊销曾写入源码的 Qwen 密钥，并只通过安全环境变量注入；
- [ ] 恢复有效 Git 仓库，冻结 branch/commit/tag、依赖锁和代码 hash；
- [ ] 补齐要求的 Markdown，核对其与 DOCX、实际配置和本进度文档一致；
- [ ] 安装测试依赖并在目标 Python 环境再次执行 pytest 与 unittest；
- [ ] 用受控凭据完成极小在线 SeEvo 生成 smoke，禁止打印密钥；
- [ ] 生成真实多 seed CEWS 报告，只准入零违反、零最大模糊延迟且满足能耗/稳定性
  阈值的规则，复核 source/report/config hash；
- [ ] 生成并冻结严格 train/validation/final-test 分离的安全示范数据；
- [ ] 补齐 13 个 fail-closed 对比/消融 adapter 或明确从正式方案移除；
- [ ] 冻结工作流、到达、资源、DDL、模糊参数和所有 seed manifest；
- [ ] 分别在 CPU 和目标 GPU 上运行更长的 legacy/safe smoke，检查 NaN、空 mask、
  hard step limit、内存/replay 增长与异常恢复；
- [ ] 对 no-safe-action、fallback、shield 干预和实际违反做覆盖率/频率核验；
- [ ] 完成多 seed checkpoint 恢复一致性、课程阶段恢复和旧 checkpoint 拒绝测试；
- [ ] 预注册可行性优先模型选择、零最终预算和 worst-seed 报告规则；
- [ ] 保持 final-test seeds 隔离，禁止用最终测试结果调课程、阈值或超参数；
- [ ] 先完成小规模完整矩阵，再启动正式训练；保留所有 config snapshot、manifest、
  日志、checkpoint 和失败记录。

阶段 18 的代码审计、离线 CEWS、原始 HRL、安全 HRL、checkpoint 保存恢复和全量
unittest 小规模验收已完成。当前可以进入“正式实验前工件与 adapter 补齐”，但在
上述检查清单完成前不应启动或声称完成正式大规模对比实验。

## 41. 2026-07-30 项目保存说明更新

### 41.1 本次范围

本次只修改项目说明，不修改业务代码、配置、测试、数学公式、Agent、环境、SeEvo、
shield、fallback、replay 或训练行为。

新增文档基线标识：

```text
llm-safe-hrl-audited-baseline-20260730
```

当前 `.git/HEAD` 仍不存在，因此该标识仅用于识别用户准备保存的项目副本，不是
branch、commit 或 tag。实际压缩或复制完成后，应对最终归档计算 SHA-256；在归档
生成前不记录虚假的预期 hash。

### 41.2 文档变化

1. 重写根目录 `README.md`：
   - 从 2026-07-23 的 CEWS 目录说明更新为当前两条代码主线；
   - 增加冻结数学语义、legacy/safe 运行方式、测试边界和文档导航；
   - 明确当前没有真实 admitted LLM 规则；
   - 明确后续修改不得沿用旧测试结果。
2. 新增 `docs/PROJECT_BASELINE_2026-07-30.md`：
   - 记录保存基线身份；
   - 说明保存副本必须包含和可清理的内容；
   - 区分已实现、已测试、尚未验证和不具备工件；
   - 规定后续修改、测试、进度追加和新基线命名方式。
3. 在本进度文档顶部加入当前保存基线入口，并追加本节，不覆盖阶段 0–18 的历史。

### 41.3 当前基线结论

当前保存点仍以阶段 18 的实际测试记录为准：静态语法检查通过、205 项 unittest
通过、原始 HRL/safe HRL/离线 CEWS/checkpoint 小规模验收通过。由于本次没有修改
代码，这些是保存点的历史验收记录；后续代码一旦变化，必须重新运行测试后才能形成
新的结论。

本次说明更新不表示正式实验、在线 SeEvo、真实 LLM 准入、多 seed 长训练、跨平台
复现或部署安全已经完成。

### 41.4 检查结果

- README 和新基线文档均可按 UTF-8 完整读取；
- 两份文档中的本地 Markdown 链接全部存在；
- `python -m unittest -v tests.test_final_safe_hrl_acceptance`：
  `4 tests OK`，0 失败，0 跳过；
- 本次没有重新运行完整 205 项测试，不能把 4 项回归描述为新的全量测试；
- 未修改业务代码、配置或依赖。

## 42. 2026-07-30 算法目录集中与对比算法隔离

### 42.1 本次范围

为后续引入对比算法，本阶段只重组算法代码目录、修正物理路径、保留公共入口并完善
说明。没有增加安全 HRL 核心功能，也没有修改 reward/cost、shield、Q_r/Q_c、
lambda、Manager–Host–VM 职责或 SeEvo 规则接口。

目录布局修订标识：

```text
llm-safe-hrl-algorithm-layout-20260730
```

### 42.2 目录变化

迁移映射：

```text
base/           -> algorithms/llm_safe_hrl/base/
hrl_mix/        -> algorithms/llm_safe_hrl/hrl_mix/
LLM/            -> algorithms/llm_safe_hrl/LLM/
baseline_fcfs/  -> algorithms/comparisons/fcfs/
```

根目录 `base/`、`hrl_mix/`、`LLM/` 和 `baseline_fcfs/` 现在只保留兼容
`__init__.py`。它们扩展包搜索路径到新实现目录，所以历史 import 和
`python -m hrl_mix.train` 仍可使用。根目录 `main.py` 仍是 SeEvo 的兼容入口。

`common/`、`data/`、`tests/`、`docs/`、`out/`、`run/` 和 `tools/` 保持项目级
目录；它们分别属于跨算法共享模型/数据、测试文档、产物或公共入口，不混入任一
算法实现。

### 42.3 新增和调整的接口

新增只读路径模块：

- `project_paths.py`：所有算法可共享的 `PROJECT_ROOT`、`ALGORITHMS_ROOT`；
- `algorithms/llm_safe_hrl/paths.py`：当前方法的 `ALGORITHM_ROOT`、`BASE_ROOT`、
  `HRL_ROOT`、`LLM_ROOT`、`COMPARISON_ROOT`。

已有 Python import 接口未变：

```text
base.*
hrl_mix.*
LLM.*
baseline_fcfs.*
```

物理配置路径变为
`algorithms/llm_safe_hrl/hrl_mix/config/`；配置内相对 `data/`、`out/` 和 LLM
启发式库路径已按新层级修正。未来新生成的 experiment manifest 会记录新的
`source_config_path` 和代码/config hash；旧 manifest 不应伪装成新布局产物。

### 42.4 修改内容

主要变更：

- 移动 `base/`、`hrl_mix/`、`LLM/` 和 `baseline_fcfs/` 的实现文件；
- 新增 `algorithms/` 包、当前方法/对比方法 README 和四个根目录兼容包；
- 根 `main.py` 改为定位新 SeEvo 物理入口；
- `train_config.py`、`experiment_matrix.py`、CEWS `eval.py` 使用统一项目根；
- 修正两份 HRL JSON 配置中的项目级相对路径；
- 更新依赖提示、准入报告命令和受影响测试中的物理路径；
- 新增 `tests/test_algorithm_layout.py`，防止业务代码再次散落到根兼容包；
- 将历史 FCFS 数据/输出路径改为项目根，并统一 DAX 文件名 `Ligo_30.xml`；
- 更新 `README.md`、`docs/rename_manifest.csv` 和
  `docs/PROJECT_LAYOUT_2026-07-30.md`。

### 42.5 不变的算法语义

- 模糊能耗仍为 `mean + 1.0 * std`；
- 模糊 DDL 风险时间仍为 `0.05 * modal + 0.95 * upper`；
- 截止期仍为确定值；
- LLM 仍只产生 `get_task_priority_v2` ready-task 排序；
- Manager 选择规则，Host/VM Agent 选择资源；
- `safe_rl=false` 仍走旧模式；
- 未新增或修改 reward/cost、mask、shield、fallback、网络或训练策略。

### 42.6 测试结果

最终代码状态执行：

```powershell
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -q
```

结果：

- 静态语法检查通过；
- `209 tests OK`，0 失败，0 跳过；
- 其中新增 4 项目录布局/兼容解析测试；
- 原始 HRL 1 episode、safe HRL 1 episode、三层 checkpoint 恢复和离线 CEWS
  小规模验收通过。

额外入口检查：

- `python -m hrl_mix.train --help`：通过；
- `python main.py --help`：通过；
- 新物理路径下 CEWS `eval.py --help`：通过；
- FCFS 项目根、5 个 DAX 和项目级 `out/` 解析：通过。

### 42.7 风险和未验证范围

- `run/hrl_mix/` 是历史场景启动脚本，仍留在公共 `run/`；它不是新的业务实现位置，
  后续可在单独阶段压缩为薄入口，不能在本次无测试依据地批量重写；
- 历史 FCFS 入口没有 argparse/`--help`，直接执行会启动硬编码 50 episode
  仿真；本次只验证路径解析，没有运行完整 FCFS；
- 在线 SeEvo 仍未运行，且没有把 `--help` 视为真实规则生成；
- 未运行正式训练、多 seed 长实验、跨平台或 GPU 测试；
- 移动会改变源文件/配置 hash，正式实验前必须重新生成并冻结相关 manifest；
- 当前工作区仍无有效 Git `HEAD`，无法把布局修订绑定到 commit。

本阶段可以作为后续添加对比算法的目录起点。新增算法应进入
`algorithms/comparisons/<method_id>/`，并继续遵守共享输入、公平信息访问和独立
实验 manifest 规则。

## 43. 2026-07-30 `out/` 输出短名称改造

### 43.1 问题与范围

审计发现 `out/ckpts/` 和 `out/logs/` 的历史训练目录把算法、场景、全部阶段开关和
配置描述全部展开，最长单个目录名达到 208 个字符。历史评估脚本还会生成超过
100 个字符的 CSV 文件名。

本阶段只改变新产物的命名和路径解析，不修改训练目标、环境、Agent、reward/cost、
mask、shield、fallback、Q_r/Q_c、lambda 或 SeEvo。

### 43.2 新命名规则

新增 `output_naming_schema_version=1`，示例：

```text
out/ckpts/hrl-ss-t-s1/
out/logs/hrl-ss-t-s1/train.csv
out/ckpts/safe-ss-t-e42f608daf/
out/logs/safe-ss-t-e42f608daf/train.csv
out/ckpts/pipe-ss-t-0123456789/
out/logs/fcfs-ss/train.csv
out/eval/hrl/ss-t.csv
```

普通 HRL 保留方法、场景、DDL 和 seed。safe HRL 使用方法、场景、DDL 和 10 位
规范化配置摘要；pipeline 使用 plan hash 摘要。完整配置继续写入 config snapshot、
checkpoint metadata 和 manifest，不依赖文件名恢复实验语义。

新 run ID 最长 48 个字符，只允许小写字母、数字和连字符。

### 43.3 兼容行为

- 新训练不再写 `ckpts_<long-run-name>` 和 `logs_<long-run-name>`；
- 训练主日志统一为 `train.csv`；
- 27 个 `run/hrl_mix/eval_*.py` 从自身短文件名推导 scenario/DDL；
- 评估优先读取新短 checkpoint 路径；
- 新路径不存在时，只读取 scenario、DDL、seed 精确匹配的旧 legacy checkpoint；
- 删除了“遍历并选择第一个包含 best checkpoint 的目录”的不安全回退；
- 现有 `out/` 历史目录和文件没有自动移动、删除或覆盖。

旧 safe/pipeline checkpoint 若要继续训练，仍应通过显式 resume 路径加载；本阶段
不猜测旧长名称与新配置摘要之间的映射。

### 43.4 修改文件

新增：

- `output_naming.py`；
- `tests/test_output_naming.py`；
- `docs/OUTPUT_NAMING.md`。

修改：

- `algorithms/llm_safe_hrl/hrl_mix/train_config.py`；
- `algorithms/comparisons/fcfs/train_fcfs.py`；
- `run/hrl_mix/eval_*.py`，共 27 个；
- `tests/test_safety_lagrange.py`；
- `README.md`；
- `algorithms/llm_safe_hrl/README.md`；
- `docs/SAFE_HRL_PROGRESS.md`。

### 43.5 接口变化

公共训练命令和 Python import 不变。`TrainConfig` 字段仍为 `run_name`、`save_dir`
和 `log_path`，但字段值改为短格式：

```text
run_name: hrl-ss-t-s1 / safe-ss-t-<digest> / pipe-ss-t-<digest>
save_dir: out/ckpts/<run_name>
log_path: out/logs/<run_name>/train.csv
```

新增公共路径函数：

- `legacy_hrl_run_id()`；
- `safe_hrl_run_id()`；
- `pipeline_run_id()`；
- `training_output_paths()`；
- `hrl_evaluation_csv_path()`；
- `resolve_hrl_checkpoint_dir()`。

### 43.6 测试结果

最终代码状态执行：

```powershell
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -q
```

结果：

- 静态语法检查通过；
- `215 tests OK`，0 失败，0 跳过；
- 6 项输出命名测试覆盖确定性、长度、路径、旧 checkpoint 和全部评估入口；
- 原始 HRL、安全 HRL、checkpoint 恢复、训练 pipeline 和离线 CEWS 回归仍通过。

### 43.7 未验证范围

- 没有运行完整 30-seed 历史评估；
- 没有启动正式训练验证长期产物增长；
- 没有自动迁移现有长名称目录；
- 配置摘要用于命名和防覆盖，不替代完整 config snapshot 或 manifest；
- 当前仍无有效 Git `HEAD`，无法将命名 schema 绑定到 commit。

## 44. 2026-07-31 LLM 规则资源任务域准入

### 44.1 语义变更

`required_evaluation_seeds` 现在只用于离线多 seed 准入证据。Manager 在运行时不再
要求当前 workflow seed 出现在这组评价 seed 中；`manager_heuristics.py` 的
`workflow_seed_not_in_evaluation_seeds` 拒绝路径以及 `hrl_env.py` 的逐 episode
seed 动态 mask 均已删除。

精确 `dax_files` 绑定改为显式 `admission_scope`：

```yaml
mode: resource_task_domain
resource_code: S
allowed_scenarios: [SS, MS, LS]
workflow_families:
  [CyberShake, Epigenomics, Ligo, Montage, Sipht]
```

资源域映射固定为：

- `S -> [SS, MS, LS]`；
- `M -> [SM, MM, LM]`；
- `L -> [SL, ML, LL]`。

scope 只放宽运行 seed 和同资源域内的任务规模；运行时仍逐字段严格比较
`workflows_per_instance`、arrival、horizon、DDL 配置、Host/VM 拓扑、PC/BW
档位及 fuzzy 参数。候选源码、评价报告、记录和评价结果 hash，以及
`get_task_priority_v2` 八参数接口仍按原准入链校验。运行时 workflow family
集合必须与 scope 的五类集合一致。

### 44.2 Schema 与配置

- safe heuristic manifest schema：`2 -> 3`；
- admission record schema：`1 -> 2`；
- Manager observation/action schema：
  `safe_manager_heuristic_selection_v3 -> v4`；
- manifest 顶层和每条 record 都必须持久化同一个规范化
  `admission_scope`；
- schema 2 的旧 manifest 会明确抛出
  `safe heuristic manifest schema mismatch`，不会静默准入；
- 新增 `--manager-heuristic-manifest`，仅在
  `--safe-rl-heuristic-manager` 下有效；
- 未指定 manifest 时按场景第二位资源码自动选择
  `safe_heuristic_library_resS.json`、`resM.json` 或 `resL.json`；
- 三个新资源域 manifest 当前只含五个内建传统规则动作槽，没有伪造或迁移历史
  LLM 准入记录。通过新 schema 离线评价的记录应注册到对应资源 manifest。

`TrainConfig` 新增 `workflow_families`；`train_runner.py` 将
`scenario_code/task_code/resource_code/workflow_families` 传入环境。
`HrlHeftEnv` 的四个新构造参数均为可选参数，旧 positional/keyword 构造不变。
如果旧调用在 heuristic selection mode 下未显式传 families，环境会从已有
DAX 路径提取；legacy Manager 不受影响。

### 44.3 修改文件

核心代码：

- `algorithms/llm_safe_hrl/base/heuristic_admission.py`；
- `algorithms/llm_safe_hrl/base/manager_heuristics.py`；
- `algorithms/llm_safe_hrl/base/hrl_env.py`；
- `algorithms/llm_safe_hrl/hrl_mix/train.py`；
- `algorithms/llm_safe_hrl/hrl_mix/train_runner.py`；
- `algorithms/llm_safe_hrl/hrl_mix/train_config.py`。

配置和准入资产：

- `cews_task_constructive_hrl_ss_admission.yaml`；
- `safe_heuristic_library_resS.json`；
- `safe_heuristic_library_resM.json`；
- `safe_heuristic_library_resL.json`；
- `safe_hrl_experiment_matrix.json`；
- `admission_reports/README.md`。

测试：

- `tests/test_heuristic_admission.py`；
- `tests/test_safe_manager_heuristics.py`；
- `tests/test_safe_demonstration_pretraining.py`（默认测试 manifest 跟随
  schema 3 资源域文件）。

没有修改 SeEvo generated 候选、历史评价报告、历史实验输出、reward、Q_r/Q_c、
shield、fallback、安全状态特征、传统规则顺序或 Host/VM 动作逻辑。

### 44.4 测试结果

最终代码状态执行：

```powershell
D:\anaconda3\python.exe -m compileall -q .
D:\anaconda3\python.exe -m pytest `
  tests\test_heuristic_admission.py `
  tests\test_safe_manager_heuristics.py -q
```

结果：

- compileall：通过；
- 定向 pytest：`28 passed in 0.98s`，0 failed，0 skipped；
- 覆盖未见运行 seed 同域准入、SS 记录在 MS/LS 可用、资源域/workflow
  family/DDL/fuzzy/hash 不匹配禁用、同资源域 train/test availability mask
  与 action schema 一致、旧 schema 明确拒绝，以及 S/M/L 默认 manifest 解析；
- 三个默认资源 manifest 均可加载，当前各返回 5 个传统规则动作槽；
- CLI help 已真实解析并显示 `--manager-heuristic-manifest`。

直接使用系统默认 `D:\Python3.12\python.exe -m pytest` 会因缺少 `pytest`
失败；本次没有安装或升级依赖，而是使用项目既有的 Anaconda pytest 环境。

### 44.5 尚未验证

- 没有重跑离线 SeEvo 多 seed 评价，因此三个新 manifest 尚无 admitted LLM
  record；
- 没有运行正式 HRL 训练或跨资源域长实验；
- M/L 资源域在注册首条 LLM 记录前，需要使用各自明确配置和能耗阈值完成离线
  准入，不能复用 S 域评价结果；
- 当前工作区无有效 Git `HEAD`，本次 schema 和 manifest 版本尚不能绑定到 commit。
