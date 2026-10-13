# 纯 GP-HRL：无 CMA-ES 的独立版本

本版本位于 `algorithms/comparisons/pure_gp_hrl/`；原 `gp_hrl/` 仍为 GP+CMA-ES。
主算法及旧版文件不修改。新版本的方法标识为 `pure_gp_hrl`，生成器为 `pure_gp`，
规则、训练日志、checkpoint 和测试结果均写入独立的 `ablations/pure_gp_hrl` 目录。

```text
离线：随机表达式树与常数 → 真实仿真适应度 → 选择/交叉/变异 → 冻结 Top-K
在线：原 Manager 选择 GP 规则 → 规则选择 ready task → 原 Host/VM 分配资源
```

## GP 设置

- 表达式叶节点是原 ready-task 特征或随机常数。常数从 `Uniform(-4,4)` 采样，
  生成叶节点时有 25% 概率选常数，否则均匀选可用特征。
- 原八项特征：min_exec_time、min_comm_time、min_incremental_energy、slack、
  upward_rank、remaining_work、ready_wait_time、uncertainty。
  `energy_only` 模式禁止读取 slack/uncertainty，保持相同八参数函数接口。
- 算子为 add/sub/mul/pdiv/min/max。除法在分母绝对值不超过 `1e-12` 时返回 1；
  非有限或绝对值超过 `1e12` 的运算中间值变为 NaN，由原评价器拒绝。
- 表达式直接使用特征及常数，例如 `sub(mul(0.6,min_incremental_energy),upward_rank)`。
  没有外部的每特征系数、PARAMS、参数 schema、CMA-ES 或局部数值搜索。
  常数可以通过子树交叉继承，通过子树变异重新采样；没有额外的 Gaussian 微调。
- 默认种群 50、10 代（包含初始代）、最大深度 8；初始 grow 树的深度上限轮换 2..6。
  每代保留 1 个精英，锦标赛大小 3，交叉/变异/复制概率为 .80/.15/.05。
  变异的新子树深度上限为 2，超深子代退回父代。
- 至少使用一个任务特征。常数-only 或数值异常的候选被拒绝并写入 history，
  不使用修复后的伪分数，不把常数-only 个体当作有效调度规则。
- 每个新完整个体（结构和常数值）只在训练 seeds 1/2/3 上直接评价一次。
  相同个体复用已存适应度，不重复仿真。候选评价使用原 CEWS 确定性资源策略，
  不为每个候选训练 HRL 网络。
- 整个候选档案先按原适应度排序，再复用原 Top-K 选择。original 模式依次比较
  可行性、跨 seeds 最大 DDL 违反率、最大模糊迟到、模糊能耗得分、跨 seeds CV。
  energy_only 模式仅按模糊能耗得分排序。两种模式均有确定性哈希并列处理。
- original 模式优先选不同表达式结构，结构哈希不包含随机常数的具体值；
  不足 K 个结构时可选同结构的不同常数个体，仍须具有不同冻结源码。
  不足 K 个有效个体时失败，不复制规则凑数。没有额外的行为多样性清除。

## 独立实验配置

生成入口不再接受 `--reference-library`，不读取 LLM 规则库或历史 LLM 配置。
默认读取本目录受 Git 管理的 `experiment.yaml`：K=10、original 可行性优先目标、
50 工作流、到达率 0.03 及模糊评价参数。用 `--config <YAML>` 可以指定独立配置；
`--top-k N`、`--objective original|energy_only` 覆盖其中的规则数和目标。
场景和 DDL 必须通过 `--scenario SS|SM|SL|... --ddl T|M|L` 指定；DAX、资源拓扑、
训练/验证/测试 seed 划分复用共享注册表，deadline cache 由显式命令参数提供。
运行时保存 `effective_experiment_config.yaml`，其路径和哈希写入规则库、训练记录
及最终报告。GP 只从终端集、常数和算子随机生成种群，不需要任何 LLM 实验产物。

规则训练 seeds 是 1..3。4/5 仍为协议保留的规则验证 seeds，但本版没有参数 confirm，
因此生成流程不消耗这两个 seeds，`validation_episode_count=0`；不声称已做验证选择。
Top-K 按训练评价选择。HRL 使用训练 1..5、验证 101/102/103，
最终测试 201..230，在 SL/ML/LL 三个任务规模下冻结评测。

没有代理模型、反事实反馈、关键状态回放、CMA 诊断或额外的安全/能耗准入门槛。
这用于比较完整 LLM 规则生成方法与纯 GP；Full 可以保留其完整生成流程，但要对齐
任务信息、规则数、真实仿真预算和 HRL 训练设置，不直接比较相同代数或候选数量。
`--episode-budget N` 是仿真请求上限，不保证较小种群/代数会用满 N；需要相应增加
搜索规模才能比较同样的实际预算。

每个候选没有额外参数试验，成功评价通常消耗 3 个 episode。预算统计包含失败批次
的预留尝试数，成功训练数单独报告；缓存命中不计真实仿真。预算耗尽后先收回已派发
候选的结果，再从已完整评价的档案导出；不足 K 则失败。缓存仅在本次运行内有效。

## CPU 并行评价

生成命令支持 `--workers N`，默认 1（串行）；例如 `--workers 15` 使用最多 15 个
CPU 候选评价进程。并行进程使用 spawn 启动，每个进程的数值库线程限制为 1；每个
候选内部的训练 seeds 仍串行评价，不需要 GPU。进化、随机采样、缓存和预算由主进程
管理，按原候选顺序收集结果；重复候选不重复成功仿真，失败候选仍按原顺序重试。
进程数不写入规则源码的搜索配置哈希，改变并行度不改变规则和 Top-K 的排序逻辑。
实际进程数记录在导出库的 `search_budget.evaluation_workers`。

32 核服务器可先用单组 `--workers 15`；九组同时运行建议从每组 3 个进程起步。
并行度应根据可用 CPU 和内存调整，超过待评价候选数不会增加有效并行度。
`train --validation-workers` 仅控制 HRL 验证，不控制 GP 搜索；尚不支持进化断点恢复。

## SL-L 正式命令（PowerShell，仓库根目录）

```powershell
$PureRuleDir = 'out/ablations/pure_gp_hrl/rules/SL/L/pure_a0'
$PureLibrary = "$PureRuleDir/topk_heuristic_library_k10.json"
$CacheArgs = @(
  '--deadline-cache', 'SL=data/deadlines/fcfs/exact_mix_v1/fcfs_smallTask_largeRes_exactmix_formal38.json',
  '--deadline-cache', 'ML=data/deadlines/fcfs/exact_mix_v1/fcfs_medTask_largeRes_exactmix_formal38.json',
  '--deadline-cache', 'LL=data/deadlines/fcfs/exact_mix_v1/fcfs_largeTask_largeRes_exactmix_formal38.json'
)

# 1. 纯 GP 生成 Top-10；拒绝覆盖已有 output-dir。
python -m algorithms.comparisons.pure_gp_hrl.generate `
  --scenario SL --ddl L --top-k 10 --objective original --output-dir $PureRuleDir `
  --algorithm-seed 0 --workers 15 --population-size 50 --generations 10 --max-depth 8 @CacheArgs

# 2. 原三层从头训练。以下是纯在线示例，安全参数须与 Full 训练命令一致。
python -m algorithms.comparisons.pure_gp_hrl.train `
  --gp-library $PureLibrary --episodes 600 --optimizer-seed 0 `
  --safe-rl-energy-reward-scale 0.001 --safe-rl-cost-budget 0.01 `
  --safe-rl-lambda-init 1.0 --safe-rl-lambda-lr 0.05 @CacheArgs

# 3. 明确选择本次纯 GP 训练的 best checkpoint，不混用 GP+CMA 或 Full。
Get-ChildItem checkpoints/ablations/pure_gp_hrl/SL/Loose -Recurse -Filter best_checkpoint_manifest.json
$PureCheckpoint = Read-Host '输入本次纯 GP-HRL best_checkpoint_manifest.json 路径'
python -m algorithms.comparisons.pure_gp_hrl.evaluate `
  --checkpoint-manifest $PureCheckpoint --gp-library $PureLibrary --device cpu @CacheArgs
```

`--algorithm-seed` 控制 GP 结构与常数；训练入口的 `--optimizer-seed` 是原 RL 的网络/
训练随机种子，这个参数名不表示使用了 CMA-ES。训练入口支持原离线预训练 manifest、
pipeline plan/resume 及 curriculum，需使用纯 GP 规则 ID 对应的独立演示并与 Full 对齐。
Full 和 GP 的目标、K、仿真预算、reward scale、lambda、预训练步数与 curriculum
必须显式对齐；独立配置不会自动继承 Full 的训练命令。

## 兼容标记及输出

库、各规则、best checkpoint 和最终报告明确记录 `generator=pure_gp`、
`cmaes_enabled=false`、`parameter_optimization_enabled=false`。
原 Manager 的固定协议仍要求 `llm_rules`、`source_kind=seevo_generated` 和若干
optimization metadata 字段；这些仅为接口兼容。元数据中的 best_parameters 保存树内
随机常数，optimizer_config_hash 对应 GP 搜索设置，诊断字段明确对应 not_run；
没有运行参数优化，也没有调用 LLM API。主算法的校验逻辑不修改。

- 规则：`out/ablations/pure_gp_hrl/rules/<source>/<DDL>/<run>/`。
- 训练：`out/ablations/pure_gp_hrl/<source>/<ddl_name>/<run>/train.csv`。
- 网络：`checkpoints/ablations/pure_gp_hrl/<source>/<ddl_name>/<run>/`。
- 测试：`out/ablations/pure_gp_hrl/test/<source>/<DDL>/<run_timestamp>/`。

训练和评测入口拒绝 GP+CMA/Full 规则库与 checkpoint；历史纯 GP 规则库仍可加载。
小规模规则标记 smoke_only，不能
交给正式训练/评测入口。未提供进化阶段 resume，不改写旧实验或 deadline cache。

## 检查

```powershell
# 4个体、2代、每实例2工作流、Top-1，仅用于生成/加载冒烟检查。
python -m algorithms.comparisons.pure_gp_hrl.generate --scenario SL --ddl L --workers 2 --smoke @CacheArgs

python -m unittest tests.test_pure_gp_hrl tests.test_gp_hrl_ablation tests.test_topk_heuristic_selection -q
```

已检查随机常数与标量/批量分数一致、空/单任务输入、固定种子的可复现性、深度上限、
非法常数与 energy_only 限制、仿真预算、测试 seed 隔离、Top-K 排序和生产 loader。
测试显式将 CMA optimize 调用设为报错，纯 GP 直接评价仍通过。
并行测试覆盖候选顺序、重复候选、失败重试、预算耗尽时的结果回收和线程环境恢复。
小规模实际运行已完成规则生成、原三层训练、best checkpoint 保存和 SL 冻结推理，
并比较三个 Agent 的推理前后指纹。没有运行正式 K=10/600轮实验。
缩小工作流数会改变 M/L 精确类别配额，不能拿 2 工作流直接匹配正式 50 工作流 cache。
