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

## 与参考库对齐

已提供的参考库：

```text
out/main_single/SL/L/20260919_120714_908169_p3865/topk_heuristic_library_k10.json
```

从它及同目录 `effective_admission_config.yaml` 读取：Single、SL、Loose、K=10、
original 可行性优先目标、工作流数、资源和模糊评价设置。显式 cache 参数替换历史
Linux 路径，并核对当前场景注册表与参考 evaluation context 一致。
本版不读取 `.hydra/config.yaml` 中的 CMA 设置，也不初始化或调用 CMA optimizer。

规则训练 seeds 是 1..3。4/5 仍为协议保留的规则验证 seeds，但本版没有参数 confirm，
因此生成流程不消耗这两个 seeds，`validation_episode_count=0`；不声称已做验证选择。
Top-K 与参考库一样按训练评价选择。HRL 使用训练 1..5、验证 101/102/103，
最终测试 201..230，在 SL/ML/LL 三个任务规模下冻结评测。

没有代理模型、反事实反馈、关键状态回放、CMA 诊断或额外的安全/能耗准入门槛。
这用于比较完整 LLM 规则生成方法与纯 GP；Full 可以保留其完整生成流程，但要对齐
任务信息、规则数、真实仿真预算和 HRL 训练设置，不直接比较相同代数或候选数量。
`--episode-budget N` 是仿真请求上限，不保证较小种群/代数会用满 N；需要相应增加
搜索规模才能比较同样的实际预算。

每个候选没有额外参数试验，成功评价通常消耗 3 个 episode。预算统计包含失败批次
的预留尝试数，成功训练数单独报告；缓存命中不计真实仿真。预算耗尽后仅从已完整
评价的档案导出，不足 K 则失败。当前串行运行，缓存仅在本次运行内有效。

## SL-L 正式命令（PowerShell，仓库根目录）

```powershell
$ReferenceLibrary = 'out/main_single/SL/L/20260919_120714_908169_p3865/topk_heuristic_library_k10.json'
$PureRuleDir = 'out/ablations/pure_gp_hrl/rules/SL/L/pure_a0'
$PureLibrary = "$PureRuleDir/topk_heuristic_library_k10.json"
$CacheArgs = @(
  '--deadline-cache', 'SL=data/deadlines/fcfs/exact_mix_v1/fcfs_smallTask_largeRes_exactmix_formal38.json',
  '--deadline-cache', 'ML=data/deadlines/fcfs/exact_mix_v1/fcfs_medTask_largeRes_exactmix_formal38.json',
  '--deadline-cache', 'LL=data/deadlines/fcfs/exact_mix_v1/fcfs_largeTask_largeRes_exactmix_formal38.json'
)

# 1. 纯 GP 生成 Top-10；拒绝覆盖已有 output-dir。
python -m algorithms.comparisons.pure_gp_hrl.generate `
  --reference-library $ReferenceLibrary --output-dir $PureRuleDir `
  --algorithm-seed 0 --population-size 50 --generations 10 --max-depth 8 @CacheArgs

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
参考规则库不能说明 Full 实际采用的 reward scale、lambda、预训练步数或 curriculum。

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

纯 GP 入口拒绝 GP+CMA/Full 规则库与 checkpoint。小规模规则标记 smoke_only，不能
交给正式训练/评测入口。未提供进化阶段 resume，不改写旧实验或 deadline cache。

## 检查

```powershell
# 4个体、2代、每实例2工作流、Top-1，仅用于生成/加载冒烟检查。
python -m algorithms.comparisons.pure_gp_hrl.generate --reference-library $ReferenceLibrary --smoke @CacheArgs

python -m unittest tests.test_pure_gp_hrl tests.test_gp_hrl_ablation tests.test_topk_heuristic_selection -q
```

已检查随机常数与标量/批量分数一致、空/单任务输入、固定种子的可复现性、深度上限、
非法常数与 energy_only 限制、仿真预算、测试 seed 隔离、Top-K 排序和生产 loader。
测试显式将 CMA optimize 调用设为报错，纯 GP 直接评价仍通过。
小规模实际运行已完成规则生成、原三层训练、best checkpoint 保存和 SL 冻结推理，
并比较三个 Agent 的推理前后指纹。没有运行正式 K=10/600轮实验。
缩小工作流数会改变 M/L 精确类别配额，不能拿 2 工作流直接匹配正式 50 工作流 cache。
