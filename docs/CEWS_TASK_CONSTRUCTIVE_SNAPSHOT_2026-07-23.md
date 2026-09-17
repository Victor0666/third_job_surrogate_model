# CEWS Task Constructive 当前版本快照

> 快照日期：2026-07-23  
> 项目目录：`sec_job`  
> 当前主线：使用 SeEvo + 大语言模型生成“云边工作流 ready task 优先级规则”  
> 优化目标：在模糊 DDL 约束下最小化风险调整三角模糊总能耗  
> 说明：本文记录当前工作区的代码与实验状态，便于归档后快速恢复上下文；它本身不是 Git 提交。

## 1. 当前版本一句话说明

当前版本把 SeEvo 中由 LLM 生成的启发式限定为“对当前所有 ready tasks 打分”，LLM 不选择 Host 或 VM；环境先选择最低分任务，再使用统一的确定性 VM 规则完成资源分配。所有候选规则在相同数据、相同资源、相同 VM 选择逻辑和相同事件推进逻辑下评价。

本项目仍保留原有 HRL、FCFS、工作流、资源、能耗和数据加载代码。`cews_task_constructive` 是 SeEvo 构造式优化问题，不是强化学习。

## 2. 当前完成情况

已经完成的核心功能：

- 已建立 `cews_task_constructive` 问题配置、Prompt、种子规则和评价器。
- LLM 个体仍是一个 Python 代码字符串，只实现任务优先级函数。
- 已实现八类 ready task 特征及严格的候选函数输出验证。
- 已实现统一、确定性的 VM 选择规则，LLM 不参与 VM 选择。
- fuzzy 模式评价目标为三场景总能耗 TFN 的 `mean + weight × std`；DDL 仍作为
  独立硬约束进行可行性优先比较，不与能耗加权求和。
- 已采用 `RESULT_JSON=` 结构化评价结果，不再依赖 stdout 固定行号。
- 并发评价时，每个个体使用独立候选 Python 文件，避免共享 `gpt.py` 的竞态。
- 已支持多随机种子评价并汇总指标。
- 已添加中文注释和 CEWS 专用单元测试。
- 旧的 `cews_constructive` 目录当前已经不存在，不影响 `cews_task_constructive`。

需要注意：

- 当前 `LLM/cfg/problem/`、`LLM/prompts/` 和 `LLM/problems/` 中只看到了 `cews_task_constructive` 对应内容，没有看到原 JSP 问题文件。因此只能确认 CEWS 当前链路可用，不能仅凭本工作区证明 JSP 配置仍完整。
- 当前训练已有一次完整运行结果，但还不能认定已经统计收敛或具有充分泛化能力。

## 3. 项目目录导览

```text
sec_job/
├─ main.py                         # 根入口，转交给 LLM/main.py
├─ README.md                       # 项目总览与本文入口
├─ common/                         # 工作流、Task、Host、VM、能耗、XML 等公共模型
├─ base/                           # 公共环境与 HRL 环境；CEWS 复用 hrl_env.py
├─ baseline_fcfs/                  # FCFS 基线
├─ hrl_mix/                        # 原有混合分层强化学习训练代码
├─ run/hrl_mix/                    # HRL 评价入口
├─ data/dax/                       # 工作流 DAG/DAX XML 数据
├─ data/deadlines/                 # FCFS/HEFT 截止期缓存
├─ LLM/
│  ├─ main.py                      # Hydra + SeEvo 实际入口
│  ├─ seevo.py                     # 种群生成、评价、反思、交叉、自进化和变异
│  ├─ cfg/config.yaml              # SeEvo 全局配置
│  ├─ cfg/problem/
│  │  └─ cews_task_constructive.yaml
│  ├─ prompts/common/              # SeEvo 公共 Prompt
│  ├─ prompts/cews_task_constructive/
│  │  ├─ seed_func.txt
│  │  ├─ func_signature.txt
│  │  ├─ func_desc.txt
│  │  └─ external_knowledge.txt
│  ├─ problems/cews_task_constructive/
│  │  ├─ eval.py                   # CEWS 候选规则评价器
│  │  └─ generated/                # 每个候选的独立 Python 模块
│  └─ utils/utils.py               # LLM 客户端和通用工具
├─ tests/
│  ├─ test_cews_task_constructive.py
│  └─ test_fuzzy_energy.py
├─ tools/
│  └─ plot_seevo_convergence.py    # 从日志绘制收敛曲线
├─ outputs/                        # Hydra/SeEvo 运行日志和评价 stdout
└─ plots/                          # 收敛曲线及 CSV
```

## 4. 程序从哪里启动

在项目根目录 `sec_job` 执行：

```powershell
python main.py problem=cews_task_constructive ...
```

这里的根目录 `main.py` 是一个薄包装器：它把 `LLM` 加入 Python 搜索路径，然后执行 `LLM/main.py`。真正创建 Hydra 配置和 `SeEvo` 对象的是 `LLM/main.py`。

主要调用链如下：

```text
根 main.py
  → LLM/main.py
  → LLM/seevo.py
  → 为每个个体写入独立 candidate_*.py
  → LLM/problems/cews_task_constructive/eval.py
  → base/hrl_env.py 中的云边工作流环境
  → RESULT_JSON
  → SeEvo 解析 objective、约束和完整 metrics
```

## 5. LLM 候选规则接口

每个个体只能生成一个任务优先级函数：

```python
import numpy as np

def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty,
):
    ...
```

接口约束：

- 八个输入都是长度为 `N` 的一维 NumPy 数组。
- `N` 是当前 ready task 数量，同一下标始终对应同一个任务。
- 返回值必须是形状为 `(N,)` 的一维有限浮点数组。
- 分数越小，任务越先执行。
- 输入数组不得原地修改。
- 候选代码不得执行文件、网络或子进程操作，也不得修改全局状态。
- 候选函数不选择 VM，不创建任务，也不删除任务。
- 单个 ready task 时必须正常返回长度为 1 的数组。
- 返回长度错误、NaN、正负无穷或候选执行异常时，该个体按无效处理，`obj=inf`。

八类特征含义：

| 特征 | 含义 |
|---|---|
| `min_exec_time` | 任务在所有可行 VM 上的最小预计执行时间 |
| `min_comm_time` | 任务在所有可行 VM 上的最小前驱数据通信时间 |
| `min_incremental_energy` | 任务在所有可行 VM 上的最小预计增量能耗 |
| `slack` | 子截止期减去预计完成时间后的剩余松弛量，负值表示延期风险 |
| `upward_rank` | HEFT upward rank/关键路径重要性 |
| `remaining_work` | 当前任务到出口任务的预计剩余计算工作量 |
| `ready_wait_time` | 任务进入 ready 集合后的等待时间 |
| `uncertainty` | 模糊处理能力和带宽引起的执行、传输时间风险 |

这些特征由 `base/hrl_env.py` 的 `build_task_features()` 构造。对与 VM 相关的特征，环境先遍历该任务的所有可行 VM，再形成任务级特征；当前 `uncertainty` 统一取所有可行 VM 风险中的最小值。

## 6. 一次调度决策如何执行

评价器反复执行以下过程，直到所有工作流完成或触发安全终止条件：

1. 调用 `get_ready_tasks()` 取得当前 ready tasks。
2. 对每个 ready task 调用 `get_feasible_vms(task)`。
3. 使用 `build_task_features(ready_tasks)` 构造八类数组。
4. 调用候选 `get_task_priority_v2(...)`。
5. 严格检查返回值的类型、长度和有限性。
6. 用 `np.argmin(scores)` 选择任务。
7. 环境使用固定规则 `select_vm_deterministic(task)` 选择 VM。
8. 调用 `assign_task(task, vm)` 分配任务。
9. 如果当前没有可分配任务，则调用 `advance_to_next_event()` 推进仿真时间。

因此，不同 LLM 启发式之间会产生不同的任务顺序，但不会因为各自采用了不同的 VM 算法而失去评价可比性。

## 7. 固定 VM 选择规则

候选 LLM 选定任务后，环境只在该任务的可行 VM 中比较：

- `exec_time`
- `comm_time`
- `queue_time`
- `predicted_finish_time`
- `incremental_energy`
- `deadline_violation`

确定性排序规则：

1. 如果存在 `predicted_finish_time <= task.sub_deadline` 的 VM：
   - 先选预计增量能耗最小者；
   - 能耗相同时选预计完成时间最小者；
   - 最后按 `vm_id` 稳定打破平局。
2. 如果所有 VM 都会延期：
   - 先选截止期违反量最小者；
   - 再选预计增量能耗最小者；
   - 再选预计完成时间最小者；
   - 最后按 `vm_id` 稳定打破平局。

当 `fuzzy.enabled=true` 时，“按时”改为 eta 风险完成时刻满足子截止期，能耗键
改为风险调整模糊边际能耗；当该开关为 false 时完整使用上述历史 modal 排序。

这里所说的“公平”是候选启发式之间的实验公平：所有候选共享同一 VM 规则、资源实例、数据集和 tie-breaking。它不是在 VM 之间平均分配任务。

## 8. 优化目标与 DDL 约束

YAML 中仍保留 `objective.metric: total_energy`，表示目标类别始终是总能耗而非
货币成本、延迟或 weighted sum。实际 objective 由配置开关决定：

```text
fuzzy.enabled = false:
    objective = total_energy

fuzzy.enabled = true:
    objective = fuzzy_total_energy_mean
                + energy_uncertainty_weight × fuzzy_total_energy_std
```

两种模式的单位均为焦耳或风险调整焦耳。延期和违反率从不与能耗加权求和。
fuzzy 模式使用 eta 风险完成时刻统计违反率和迟延，modal 指标另行保留诊断。

DDL 采用可行性优先比较。SeEvo 对个体的排序键等价于：

```text
可行个体：  (0, 0, 0, objective)
不可行个体：(1, violation_rate, total_lateness, objective)
无效个体：  (2, inf, inf, inf)
```

这意味着：

- 任何满足 DDL 约束的有效个体，都优先于违反 DDL 的个体。
- 在全部可行的个体之间，选择总能耗更小者。
- 在都不可行时，先降低违反率，再降低总延期，最后比较能耗。

当前配置中：

```yaml
objective:
  metric: total_energy

constraints:
  deadline_violation_rate_max: 0.0
  comparison: feasibility_first

fuzzy:
  enabled: true
  energy_uncertainty_weight: 1.0
  deadline_eta: 0.95
  use_fuzzy_deadline_constraint: true
```

所以默认要求工作流截止期违反率为 0。

## 9. 评价输出

`LLM/problems/cews_task_constructive/eval.py` 最后输出一行：

```text
RESULT_JSON={...}
```

`LLM/seevo.py` 从 stdout 后向前寻找最后一条该记录，使用 `json.loads()` 解析：

- `individual["obj"]` 保存当前模式的总能耗目标；默认 fuzzy 模式为风险调整值；
- `individual["metrics"]` 保存完整指标；
- 缺少结果行、JSON 错误或 objective 非有限时，个体无效。

当前完整指标包括：

- `total_energy`：历史 modal 能耗
- `energy`：fuzzy 模式下为风险调整能耗，modal 模式下等于 `total_energy`
- `fuzzy_total_energy_lower/modal/upper/mean/std/score`
- `energy_optimistic` / `energy_pessimistic`
- `makespan`
- `average_workflow_completion_time`
- `total_lateness`
- `average_lateness`
- `deadline_violation_rate` / `violation_rate`：默认基于 fuzzy finish risk
- `modal_deadline_violation_rate`
- `modal_total_lateness` / `modal_average_lateness`
- `edge_task_ratio`
- `cloud_task_ratio`
- `assigned_tasks`
- `completed_workflows`
- `constraint_feasible`
- 各随机种子的分项结果

## 10. 多随机种子评价

`"case_num=[0]"` 表示只评价随机种子 0，不是运行多个种子再求平均。

例如：

```powershell
"case_num=[0,1,2,3,4]"
```

才会评价五个实例。当前聚合逻辑会：

- 对各 seed 的总能耗取平均作为聚合 objective；
- 汇总/平均其他评价指标；
- 只有全部 seed 都满足 DDL 时，聚合个体才标记为可行。

在论文实验中，建议训练使用多个固定训练种子，最终测试使用互不重叠的多个测试种子。

## 11. 工作流与测试案例如何生成

当前 `cews_task_constructive.yaml` 默认设置：

- 场景：`SS`
- 训练 seed：`[0]`
- 测试 seed：`[100]`
- 每个实例 50 个工作流
- 工作流到达率参数：`arrival_lambda=0.03`
- DAX 模板：
  - `CyberShake_30.xml`
  - `Epigenomics_24.xml`
  - `Ligo_30.xml`
  - `Montage_25.xml`
  - `Sipht_29.xml`
- DDL 来源：`cache_fcfs`
- 较小 DDL 系数：2.0，概率 0.8
- 较大 DDL 系数：3.0，概率 0.2

主要过程：

1. 按固定 seed 生成泊松到达过程。
2. 从 DAX 文件读取真实 DAG 结构。
3. 为任务生成工作量和输入/输出数据规模。
4. 从 FCFS deadline cache 取得参考完成时间。
5. 按 alpha 系数构造工作流 DDL。
6. 从工作流 DDL 向后计算任务子截止期。

同一个 seed 和同一套配置会复现同一个评价实例，因此不同候选之间具有可比性。

## 12. 当前云边资源配置

资源配置位于：

```text
LLM/cfg/problem/cews_task_constructive.yaml
```

当前默认值：

```yaml
resources:
  num_cloud_hosts: 3
  num_edge_hosts: 2
  cloud_vms_per_host: [9, 8, 6]
  edge_vms_per_host: [8, 6]
  cloud_pc_tiers: [1.0, 2.0, 4.0, 6.0, 8.0]
  edge_pc_tiers: [1.0, 2.0, 4.0, 6.0, 8.0]
  cloud_bw_tiers: [1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
  edge_bw_tiers: [1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
```

修改 Host 数量时，必须同步保证：

```text
len(cloud_vms_per_host) == num_cloud_hosts
len(edge_vms_per_host) == num_edge_hosts
```

Host 和 VM 的实际构造仍复用 `common/resource_opt.py` 中的资源类与模型。

## 13. 模糊资源模型

`common/resource_opt.py` 中使用 `TriangularFuzzyNumber(lower, modal, upper)` 表示处理能力和带宽的不确定性，并提供均值、标准差等运算。

当前 CEWS 在资源创建阶段一次性生成三角模糊 pc/bw，并对同一任务顺序和 VM
映射重放 optimistic、modal、pessimistic 三个场景。三个场景分别构造 Host
并发负载时间线并通过同一 SPECpower 曲线积分；三者的 min、原始 modal、max
构成模糊总能耗。最终 fuzzy objective 为其均值加不确定性权重乘标准差。

候选规则的 `uncertainty` 仍表示任务模糊总时长标准差；`min_incremental_energy`
改为风险调整模糊边际能耗；`slack` 默认基于 eta=0.95 的风险完成时刻。

相同 episode seed 生成完全相同的三角模糊资源。关闭 `fuzzy.enabled` 后，三场景
目标和约束不参与选择，系统回退为旧 modal 能耗、截止期和 VM 排序。

## 14. SPEC 能耗模型位置

能耗链路没有删除，主要位于：

- `common/resource_opt.py`
  - `_power_model_factory()`：根据 Host/服务器类型建立 SPEC 功率曲线。
  - `Host.power`：保存 Host 的功率模型。
- `common/workflow_opt.py`
  - `energy_from_records()`：依据任务负载时间线积分 `power × duration`。
- `base/hrl_env.py`
  - 任务分配时记录 `LoadRecord`。
  - 环境按当前时间累计云端、边缘端和总能耗。
  - `_estimate_marginal_energy_proxy()` 为固定 VM 规则提供预计增量能耗代理。

需要区分：

- VM 选择阶段用的是增量能耗代理，用来对候选 VM 排序。
- 最终 `total_energy` 来自实际调度记录经过 Host 功率曲线积分的结果。

## 15. SeEvo 训练做什么

SeEvo 框架仍保留：

- 种子个体评价
- 初始种群生成
- 个体评价
- 父代选择
- 短期反思
- 交叉
- 自进化
- 长期反思
- 变异
- 精英保留

训练不是调整神经网络参数。训练过程是反复调用 Qwen，让模型生成或改写 Python 启发式代码，再用完整云边工作流仿真评价每段代码，最终保留满足 DDL 且能耗较低的规则。

候选代码写入：

```text
LLM/problems/cews_task_constructive/generated/
candidate_iter{内部迭代编号}_ind{个体编号}.py
```

`iter` 是 SeEvo 的内部评价编号，不应直接当作论文中的训练代数。

## 16. 模糊改造前的一次完整训练结果

以下结果产生于本次三场景模糊 objective 改造之前，只能作为 modal 历史基线；
现有候选八参数接口保持兼容，但必须使用当前评价器重新评价后才能与新训练比较。

当前已核对的正式训练目录：

```text
outputs/2026-07-22/20-09-01/
```

训练参数：

```text
problem=cews_task_constructive
mode=train
model=qwen-plus
max_fe=10
pop_size=20
init_pop_size=20
mutation_rate=0.25
timeout=600
case_num=[0]
problem.dataset.workflows_per_instance=50
```

运行时间约 35 分 52 秒。该次运行最终记录的最佳候选为：

```text
LLM/problems/cews_task_constructive/generated/candidate_iter26_ind10.py
```

该候选在此次训练评价中的结果：

| 指标 | 数值 |
|---|---:|
| 工作流数 | 50 |
| 已分配任务数 | 1371 |
| objective / total_energy | 311882.320527 J |
| 平均工作流完成时间 | 804.933160 s |
| makespan | 2829.224067 s |
| total_lateness | 0 |
| average_lateness | 0 |
| deadline_violation_rate | 0 |
| edge_task_ratio | 0.323122 |
| cloud_task_ratio | 0.676878 |
| constraint_feasible | true |

最佳值随外层代数的大致变化：

```text
第 1 代：312748.532343
第 2 代：312472.444775
第 3–6 代：312472.444775
第 7 代：312359.661182
第 8 代：311932.574344
第 9–10 代：311882.320527
```

当前已有：

```text
plots/seevo_convergence.csv
plots/seevo_convergence.png
```

该运行已经正常完成，但只能说明在 seed 0 的训练案例上获得了更低能耗且满足 DDL。由于末期仍出现改进、训练 seed 只有一个，因此暂时不能认定充分收敛或具有稳定泛化能力。

## 17. 常用运行命令

所有命令默认在项目根目录 `sec_job` 下执行。

### 17.1 配置 Qwen 密钥

不要把真实密钥写进代码或提交到版本库。PowerShell 当前终端临时设置：

```powershell
$env:QWEN_API_KEY = "<YOUR_QWEN_API_KEY>"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

`LLM/utils/utils.py` 会从环境变量 `QWEN_API_KEY` 读取密钥。

### 17.2 最小训练/冒烟测试

```powershell
python main.py problem=cews_task_constructive `
  mode=train `
  model=qwen-plus `
  max_fe=1 `
  pop_size=4 `
  init_pop_size=4 `
  mutation_rate=0.25 `
  timeout=300 `
  "case_num=[0]" `
  problem.dataset.workflows_per_instance=5
```

### 17.3 当前正式训练形式

```powershell
python main.py problem=cews_task_constructive `
  mode=train `
  model=qwen-plus `
  max_fe=10 `
  pop_size=20 `
  init_pop_size=20 `
  mutation_rate=0.25 `
  timeout=600 `
  "case_num=[0]" `
  problem.dataset.workflows_per_instance=50
```

### 17.4 多 seed 训练示例

```powershell
python main.py problem=cews_task_constructive `
  mode=train `
  model=qwen-plus `
  max_fe=30 `
  pop_size=20 `
  init_pop_size=20 `
  mutation_rate=0.25 `
  timeout=1200 `
  "case_num=[0,1,2,3,4]" `
  problem.dataset.workflows_per_instance=50
```

多 seed 会显著增加一次个体评价时间和总训练成本，正式运行前应先做小规模冒烟测试。

### 17.5 纯评价，不调用 Qwen

```powershell
python .\LLM\problems\cews_task_constructive\eval.py `
  --candidate .\LLM\problems\cews_task_constructive\generated\candidate_iter26_ind10.py `
  --config .\LLM\cfg\problem\cews_task_constructive.yaml `
  --dataset-mode test `
  --cases 100
```

纯评价只加载已有候选规则，不需要 Qwen API 密钥。

### 17.6 运行测试

```powershell
python -m pytest tests/test_cews_task_constructive.py -q
python -m pytest tests/test_fuzzy_energy.py -q
```

### 17.7 绘制收敛曲线

```powershell
python .\tools\plot_seevo_convergence.py `
  ".\outputs\2026-07-22\20-09-01\main.log" `
  --output ".\plots\seevo_convergence.png"
```

注意应从项目根目录执行。若在 `tools` 目录执行，日志相对路径会错误地从 `tools` 开始解析。

## 18. 当前测试状态

已执行：

结果：原 CEWS 18 个测试与新增 fuzzy 12 个测试全部通过，共 30 个，0 个失败。

覆盖范围包括：

- 单个和多个 ready task
- 无 ready task 时推进事件
- 候选返回错误长度
- 候选返回 NaN
- 所有 VM 不可行
- 存在按时完成的 VM
- 所有 VM 都会延期
- 云端和边缘端均存在 VM
- 确定性 VM tie-breaking
- 两个候选模块并发加载隔离
- Windows 非 UTF-8 子进程输出处理
- `RESULT_JSON` 正常与异常解析
- 错误候选 `obj=inf`
- modal 纯能耗与 fuzzy 风险调整总能耗目标
- DDL 可行性优先比较
- TFN 均值、标准差、显式运算和退化场景
- 三场景 shadow finish 顺序与 Host 能耗积分
- 相同/不同 seed 的模糊资源复现性
- fuzzy VM 可行性和能耗字典序
- fuzzy RESULT_JSON 完整字段

尚缺少更高层级的自动化测试，例如多 seed 的完整端到端回归、正式 50 工作流性能回归，以及同一候选重复运行的跨进程确定性检查。

## 19. 当前已知问题与风险

### P0：代码中存在硬编码 API 密钥

当前 `LLM/main.py` 中发现了直接设置 `QWEN_API_KEY` 的代码。保存、压缩、共享或纳入版本控制之前必须：

1. 删除代码中的真实密钥赋值；
2. 到服务商控制台立即轮换/作废已经暴露的密钥；
3. 改用 PowerShell 环境变量或不提交的本地 `.env`；
4. 确保日志、IDE 配置和历史副本中没有密钥。

本文没有记录密钥内容。

### P0：当前不是有效的 Git 仓库

项目根目录存在 `.git` 文件夹，但该文件夹为空；`git status` 会报告“not a git repository”。因此当前版本没有可引用的 commit、branch 或 tag。

在保存版本前应二选一：

- 如果原仓库的 `.git` 目录有备份，恢复原版本历史；
- 如果没有历史且准备从当前状态开始，确认后再执行 `git init`，建立 `.gitignore` 并创建首个快照提交。

不要在没有额外备份的情况下仅依赖当前空 `.git` 目录。

### P1：DDL cache 与当前资源规模不一致

当前 FCFS deadline cache 元数据记录的是：

```text
3 个 Host，VM 数量 [9, 8, 8]
```

而 CEWS 当前资源是：

```text
3 个 cloud Host + 2 个 edge Host
cloud VM [9, 8, 6]，edge VM [8, 6]
```

所有候选仍使用同一批 DDL，所以候选间比较是公平的；但 DDL 基准并不是由当前 5-Host 云边资源重新生成的。正式论文实验前应明确采用哪一种口径，并按当前资源重新生成基线 DDL cache，或在实验设计中解释旧缓存的来源。

### P1：DAX 名称大小写警告

当前日志中出现过 `Ligo_30` 与缓存中的 `LIGO_30` 大小写不一致警告。现在只是警告，不会直接终止评价，但建议统一命名，避免错误匹配或跨平台差异。

### P1：能耗边界口径需要确认

当前 `energy_from_records()` 主要按存在任务记录的 Host 活动时间线积分。需要进一步确认论文采用：

- 全部基础设施从仿真开始到 makespan 的总能耗；
- 扣除 idle 基线后的动态能耗；
- 或当前 Host 活动区间能耗。

从未使用 Host 的 idle 能耗、首次任务前和末次任务后的 idle 能耗是否计入，会影响绝对能耗和不同资源规模之间的比较。

另外，VM 选择使用的是边际能耗代理，而最终目标使用完整功率曲线积分，两者口径并不完全相同。该设计可以运行，但论文中应说明，后续也可进一步校准代理。

### P1：尚未证明收敛与泛化

当前正式训练只有 10 个外层代数、一个训练 seed。建议：

- 增加到 30–50 代观察长期平台；
- 使用多个独立训练 seed；
- 使用不同的 LLM 随机运行做独立重复实验；
- 在训练未见过的 test seeds 上统一纯评价；
- 同时报告均值、标准差、最优/中位数和 DDL 可行率。

### P2：生成候选和输出文件较多

当前 `generated/` 中已有数百个候选模块，`outputs/` 中也保存了大量 stdout。正式版本控制前建议区分：

- 必须提交的源代码和配置；
- 需要长期保存的最佳候选与实验摘要；
- 可以压缩归档但不进入 Git 的完整运行日志；
- 可以忽略的缓存和 `__pycache__`。

不要直接删除现有实验产物；应先建立可验证的归档副本。

### P2：配置描述文字仍可能引起歧义

YAML 中的结构化字段已经明确 objective 为 `total_energy`、DDL 为硬约束，但英文 `description` 仍有“controlling workflow lateness and deadline violations”的旧式表述。后续可把描述同步改成“minimize total energy subject to workflow deadline constraints”，避免被误解为加权多目标。

## 20. 建议的版本保存方式

由于当前 Git 元数据不可用，建议先完成一个不会破坏现状的文件级归档：

1. 关闭仍在运行的训练和评价进程。
2. 复制整个项目目录到带日期的只读备份位置。
3. 额外保存当前最佳候选、YAML、Prompt、`main.log`、最佳 stdout、收敛 CSV/PNG 和本文。
4. 记录 Python 版本及 `pip freeze`。
5. 删除并轮换硬编码 API 密钥后，再建立新的 Git 仓库或恢复旧仓库。
6. 创建清晰标签，例如 `cews-task-constructive-energy-ddl-snapshot-20260723`。

建议至少稳定保存以下实验文件：

```text
LLM/problems/cews_task_constructive/generated/candidate_iter26_ind10.py
LLM/cfg/problem/cews_task_constructive.yaml
LLM/prompts/cews_task_constructive/
outputs/2026-07-22/20-09-01/main.log
outputs/2026-07-22/20-09-01/problem_iter26_stdout10.txt
plots/seevo_convergence.csv
plots/seevo_convergence.png
docs/CEWS_TASK_CONSTRUCTIVE_SNAPSHOT_2026-07-23.md
```

## 21. 后续重新接手时的五分钟检查清单

1. 先阅读本文，确认目标仍是“风险调整模糊总能耗 + 模糊 DDL 硬约束”。
2. 检查 `LLM/cfg/problem/cews_task_constructive.yaml` 的数据、Host/VM 和 seed。
3. 检查 `LLM/main.py` 中是否已移除硬编码密钥。
4. 运行原 CEWS 18 个测试和 fuzzy 12 个测试。
5. 对保存的最佳候选运行一次 test seed 纯评价。
6. 检查输出末尾是否有合法 `RESULT_JSON=`。
7. 再决定是继续增加训练代数、扩展 seed，还是先修正 DDL/能耗口径。

## 22. 最重要的代码定位

| 文件 | 当前职责 |
|---|---|
| `main.py` | 项目根训练入口包装器 |
| `LLM/main.py` | Hydra 配置与 SeEvo 实际入口 |
| `LLM/seevo.py` | LLM 种群进化、独立候选文件、并发评价、结果解析和约束比较 |
| `LLM/cfg/config.yaml` | SeEvo 全局模型、种群、迭代和超时配置 |
| `LLM/cfg/problem/cews_task_constructive.yaml` | CEWS 数据、资源、目标、约束和安全配置 |
| `LLM/prompts/cews_task_constructive/` | 函数接口、种子、说明和调度知识 |
| `LLM/problems/cews_task_constructive/eval.py` | 候选动态导入、仿真循环、指标汇总和 RESULT_JSON |
| `base/hrl_env.py` | ready task、八类特征、固定 VM 规则、分配和事件推进 |
| `common/resource_opt.py` | Host、VM、三角模糊资源和 SPEC 功率曲线 |
| `common/workflow_opt.py` | Workflow/Task 辅助逻辑和基于记录的能耗积分 |
| `LLM/utils/utils.py` | Qwen/OpenAI 兼容客户端初始化 |
| `tests/test_cews_task_constructive.py` | CEWS 接口、VM 规则、并发与结果协议回归测试 |
| `tests/test_fuzzy_energy.py` | TFN、三场景能耗、模糊 DDL、资源 seed 和兼容性测试 |
| `tools/plot_seevo_convergence.py` | 从训练日志提取并绘制收敛曲线 |

---

后续若对调度逻辑、目标函数、资源规模、DDL 或模糊模型做实质修改，应同步更新本文顶部日期、当前完成情况、正式运行结果和已知问题，避免文档与代码再次分离。
