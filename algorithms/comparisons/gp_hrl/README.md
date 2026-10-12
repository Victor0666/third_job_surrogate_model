# GP-HRL 替换消融

对照设置：**GP 生成 Top-K → 原 Manager → 原 Host Agent → 原 VM Agent**。
只新增本目录与独立测试，不修改主算法、原 GP 比较算法或已有实验产物。
训练直接调用 `hrl_mix.train_runner.train` 的现有扩展接口；环境、三层网络、
replay、reward/cost、shield、fallback、动态 lambda 和验证模型选择都复用原实现。
三个网络重新训练，不载入 Full 的网络来代替正式消融。

## 已对齐的参考库

```text
out/main_single/SL/L/20260919_120714_908169_p3865/topk_heuristic_library_k10.json
```

该库为 Single、SL、Loose、K=10、`original` 可行性优先目标。
规则搜索使用 seeds 1/2/3，CMA confirm 使用 4/5；HRL 使用训练 1..5、
验证 101/102/103、最终测试 201..230。测试域为 SL/ML/LL。
生成入口从参考库与同目录 `effective_admission_config.yaml` 读取 K、目标、
模糊参数、DDL 和评价口径，并检查当前场景注册表与参考 evaluation context 一致。
历史 Linux cache 路径通过显式 `--deadline-cache` 替换，不改原配置文件。
有对应 `.hydra/config.yaml` 时读取其中 CMA 设置，否则使用当前默认 CMA 配置；
实际生效设置完整保存到 GP manifest，不能把后一种情况声称为历史超参数完全一致。

此参考运行最终安全准入失败后导出了 Top-K，但 Top-K 本身的排名策略仍可作为参考。
GP 同样使用 Top-K 排名，记录 `admitted=false`，不虚报安全准入。
目前没有找到引用这个库的 Full 训练 checkpoint，因而无法从规则库推断 Full 的
reward scale、lambda、预训练、curriculum 或实际训练轮数；这些必须和 Full 训练命令对齐。

## GP 搜索与边界

- 使用原八参数 `get_task_priority_v2` 接口、原 ready-task 特征和原并列排序逻辑。
  `original` 可用八个特征；`energy_only` 保持八参数 API，但树只使用六个非安全特征。
- 复用已有 GP 的表达式、交叉、变异；算子为 add/sub/mul/pdiv/min/max。
  近零保护除法为 `|denominator| <= 1e-12` 时返回 1；非有限中间值或绝对值超过
  `1e12` 的中间值返回 NaN，由正式 evaluator/loader 拒绝，不做分数修复。
- 每个使用到的特征有一个共享系数，初值 1、范围 [-4,4]；复用原 CMA-ES 和诊断逻辑。
  GP 随机种子为 `--algorithm-seed`，CMA 使用 `algorithm_seed+1`，避免 pycma 将 0
  解释为按时间随机。实际种子写入规则元数据。
- 默认种群 50、10 代、最大深度 8，初始 grow 树使用 2..6 深度上限，交替允许根终端；
  保留 1 个精英、锦标赛大小 3、交叉/变异/复制概率 .80/.15/.05。
  所有成功评价的唯一树进入候选档案，再复用 Full 的 `select_topk` 和排序键。
  不足 K 条有效规则则失败，不复制规则凑数。
- 每棵树先优化参数，再按完整训练 seeds 精确评价冻结规则；不访问最终测试 seeds。
  每个参数源文件路径绑定其内容，防止 Python 字节码缓存错用另一个参数向量。
  进化是串行执行，缓存仅在本次运行有效。
- `--episode-budget` 限制离线搜索训练与验证合计请求的真实仿真预算。
  CMA、诊断和冻结重评都计入。预算耗尽后仅从已完整评价的规则导出；不足 K 则失败。
  manifest 分别记录成功的训练/验证 episode、尝试预算和缓存命中。发生失败时尝试数
  是已预留的上界，成功数不包含失败批次。默认搜索规模是起点，不表示已与 Full 等预算。
- 不使用代理模型、LLM 反事实反馈、关键状态回放或硬能耗准入阈值。
  所以这是“完整 LLM 规则生成流程替换为 GP”的对照。若要严格隔离结构生成器，
  Full 对照也应关闭这些专属机制并匹配仿真预算；算术树与 Python 规则的表达空间仍不同。

为了原 Manager 无需修改即可加载，manifest 保留生产接口必需的 `llm_rules`、
`source_kind=seevo_generated`、`seevo_iteration/individual` 等兼容字段；这些是接口标签。
真实来源在库、每条规则、checkpoint 与最终报告中明确标记 `generator=gp`、
`llm_enabled=false`、`ablation=gp_hrl`。GP 入口不初始化 LLM 客户端、不调用 LLM API。

## SL-L 完整运行命令（PowerShell，仓库根目录）

```powershell
$ReferenceLibrary = 'out/main_single/SL/L/20260919_120714_908169_p3865/topk_heuristic_library_k10.json'
$GpRuleDir = 'out/ablations/gp_hrl/rules/SL/L/gp_a0'
$GpLibrary = "$GpRuleDir/topk_heuristic_library_k10.json"
$CacheArgs = @(
  '--deadline-cache', 'SL=data/deadlines/fcfs/exact_mix_v1/fcfs_smallTask_largeRes_exactmix_formal38.json',
  '--deadline-cache', 'ML=data/deadlines/fcfs/exact_mix_v1/fcfs_medTask_largeRes_exactmix_formal38.json',
  '--deadline-cache', 'LL=data/deadlines/fcfs/exact_mix_v1/fcfs_largeTask_largeRes_exactmix_formal38.json'
)

# 1. 正式生成 GP Top-10；拒绝覆盖已存在的 output-dir。
python -m algorithms.comparisons.gp_hrl.generate `
  --reference-library $ReferenceLibrary --output-dir $GpRuleDir `
  --algorithm-seed 0 --population-size 50 --generations 10 --max-depth 8 @CacheArgs

# 可选：在生成命令加入 --episode-budget N，N 取与 Full 可比的仿真预算。
# 不建议直接比较两种算法的代数或候选数。

# 2. 三层从头训练，保持与 Full 相同的安全参数。
# 以下为纯在线训练示例；数值须和实际 Full 训练命令一致。
python -m algorithms.comparisons.gp_hrl.train `
  --gp-library $GpLibrary --episodes 600 --optimizer-seed 0 `
  --safe-rl-energy-reward-scale 0.001 --safe-rl-cost-budget 0.01 `
  --safe-rl-lambda-init 1.0 --safe-rl-lambda-lr 0.05 @CacheArgs

# 3. 选择本次训练生成的 best_checkpoint_manifest.json。
# 不自动选“最新”checkpoint，避免多种子/多次实验混淆。
Get-ChildItem checkpoints/ablations/gp_hrl/SL/Loose -Recurse -Filter best_checkpoint_manifest.json
$GpCheckpoint = Read-Host '输入本次 GP-HRL best_checkpoint_manifest.json 路径'
python -m algorithms.comparisons.gp_hrl.evaluate `
  --checkpoint-manifest $GpCheckpoint --gp-library $GpLibrary --device cpu @CacheArgs
```

需要更多搜索/RL seeds 时分别调整 `--algorithm-seed`、`--optimizer-seed`，并为每个
搜索 seed 使用不同 `--output-dir`。其他场景/DDL使用对应 Full 参考库和三个 cache 映射；
生成及训练入口自动从 GP 库读取源场景、DDL 和 K，不需要手动再次填写。

如果 Full 使用离线预训练，可将 GP 库传给现有
`python -m tools.generate_safe_demonstrations --manager-heuristic-manifest <GP库> ...`，
生成独立的 GP demonstration（用其 `--manifest` 指向 GP 独立目录下的演示 manifest）；
然后传入 `train --safe-rl-offline-pretrain-manifest <GP演示manifest>`。
训练入口也支持现有 pipeline plan/resume 和 `--safe-rl-curriculum`。
样本量、预训练步数、curriculum 必须与 Full 对齐；不要复用含 LLM 规则 ID 的演示。

## 输出与验证

- 规则：`out/ablations/gp_hrl/rules/<source>/<DDL>/<run>/`，包含规则、真实报告、
  CMA 记录、参数候选、有效配置、搜索 history 和 Top-K manifest。
- 训练日志：`out/ablations/gp_hrl/<source>/<ddl_name>/<run>/train.csv`。
- 网络：`checkpoints/ablations/gp_hrl/<source>/<ddl_name>/<run>/`；
  `gp_ablation.json` 和 best manifest 记录来源，三个网络仍按原格式保存。
- 测试：`out/ablations/gp_hrl/test/<source>/<DDL>/<run_timestamp>/`；
  逐 seed CSV/JSONL 与 `evaluation.json`。原 evaluator 检查三个 Agent 均保持冻结。
  不允许把普通 Full checkpoint 或 `--smoke` 库交给正式 GP 训练/评测入口。

```powershell
# 小规模生成链路：4个体、2代、每实例2工作流、K=1、缩小 CMA。
# 自动生成独立目录；产物标记 smoke_only，不能用于正式训练。
python -m algorithms.comparisons.gp_hrl.generate --reference-library $ReferenceLibrary --smoke @CacheArgs

python -m unittest tests.test_gp_hrl_ablation tests.test_topk_heuristic_selection -q
```

小规模运行只能说明实现可执行，不能说明收敛或优于 LLM。
已验证 GP 生成/生产加载、小规模三层训练及 SL 的 30 seeds 冻结推理。
将每实例工作流数缩为 2 会改变 M/L 任务域的精确类别配额和打乱顺序，不能直接复用
正式 50 工作流 deadline cache 做跨任务规模冒烟评测。ML 的正式 50 工作流序列已核对
与现有 cache 一致；冒烟规模的 mismatch 不表示正式 cache 错误。
参考库来自 2026-09-19，而项目之后调整过通信/安全实现；正式比较应在同一当前版本下
重新评价冻结 LLM 规则并训练 Full 与 GP-HRL，不直接把历史能耗作为今天的对照。
