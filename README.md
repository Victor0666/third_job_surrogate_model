# LLM 辅助安全分层强化学习云边工作流调度项目

> 当前保存基线：`llm-safe-hrl-audited-baseline-20260730`  
> 基线性质：阶段 18 最终代码审计与小规模验收后的保存副本起点  
> 注意：当前目录没有有效 Git `HEAD`，上述名称是文档标识，不是 Git tag 或 commit。
>
> 当前目录布局修订：`llm-safe-hrl-algorithm-layout-20260730`  
> 当前方法已集中到 `algorithms/llm_safe_hrl/`，对比算法统一放入
> `algorithms/comparisons/`。

本项目研究云边动态工作流调度，当前同时保留 SeEvo/LLM 启发式规则生成主线和
Manager–Host Agent–VM Agent 三层 HRL 主线。在此保存基线之上继续修改时，应保留
现有模糊能耗、模糊 DDL、LLM 职责边界以及 `safe_rl=false` 的旧模式兼容性。

保存本项目副本或开始后续开发前，请先阅读
[当前保存基线说明](docs/PROJECT_BASELINE_2026-07-30.md)；完整分阶段实现、测试结果
和尚未验证事项见
[安全 HRL 改造进度](docs/SAFE_HRL_PROGRESS.md)。

## 当前架构

### SeEvo / LLM 主线

```text
main.py
  -> algorithms/llm_safe_hrl/LLM/main.py
  -> SeEvo 生成 get_task_priority_v2
  -> CEWS evaluator 评价候选
  -> 环境固定 VM 规则完成资源分配
```

- LLM 规则只对当前 ready tasks 排序。
- `get_task_priority_v2` 返回 ready-task 优先级分数，不选择 Host 或 VM。
- CEWS evaluator 使用模糊总能耗目标和模糊 DDL 可行性优先规则。
- 只有通过准入、接口、来源和 hash 校验的 LLM 规则才能进入安全 Manager 候选集。
- 当前随项目保存的三个历史 LLM 规则均为 `rejected`，不能当作真实 admitted 规则。

### HRL 主线

```text
Manager
  -> 选择旧规则权重，或在安全 heuristic 模式下选择 ready-task 启发式
Host Agent
  -> 选择 Host
VM Agent
  -> 在所选 Host 内选择 VM
safety shield
  -> 检查并修正 Host/VM 资源动作；空安全集合时调用确定性 fallback
```

- 三层 Agent 基于 D3QN。
- safe 模式下每层分别学习性能价值 `Q_r` 和安全代价价值 `Q_c`。
- Manager 选择启发式时，LLM 仍只影响 ready-task 顺序；Host/VM 动作仍由对应 Agent、
  shield 或固定 VM 回退规则负责。

## 冻结数学语义

后续修改不得在没有新阶段设计和回归验证的情况下改变以下定义：

```text
fuzzy_energy_score
  = fuzzy_energy_mean + 1.0 * fuzzy_energy_std

R(T_i)
  = 0.05 * T_i_modal + 0.95 * T_i_upper

D_i
  = deterministic deadline
```

同时必须继续区分：

- performance reward；
- safety cost；
- hard legal-action constraint；
- fuzzy DDL safety constraint。

## 运行模式

### 原始 HRL

```powershell
python -m hrl_mix.train --scenario SS --ddl T
```

`safe_rl` 默认关闭。关闭时保留原 observation、单 `Q_r`、旧 replay、
legacy epsilon-greedy 和 Manager 规则权重语义。

可使用 `--episodes N` 进行短运行：

```powershell
python -m hrl_mix.train --scenario SS --ddl T --episodes 1
```

### 安全 HRL

完整安全短运行的显式开关示例：

```powershell
python -m hrl_mix.train `
  --scenario SS `
  --ddl T `
  --episodes 1 `
  --safe-rl `
  --safe-rl-shield `
  --safe-rl-state `
  --safe-rl-dynamic-lambda `
  --safe-rl-heuristic-manager
```

各安全子功能默认关闭，不会因运行旧命令而自动启用。

### SeEvo / CEWS

```powershell
python main.py problem=cews_task_constructive
```

默认 Qwen 客户端需要从环境变量读取 `QWEN_API_KEY`。项目中不得保存明文 API
密钥。依赖说明位于 `algorithms/llm_safe_hrl/LLM/requirements.txt`；CEWS 离线
evaluator 可以独立评价已有、受信任的候选规则。

## 输出命名

新实验不再把完整算法描述和全部安全阶段拼入 `out/` 目录名。示例：

```text
out/ckpts/hrl-ss-t-s1/
out/logs/hrl-ss-t-s1/train.csv
out/ckpts/safe-ss-t-e42f608daf/
out/eval/hrl/ss-t.csv
```

详细配置保存在 checkpoint config snapshot、manifest 和日志字段中。历史长目录不
自动移动或覆盖；评估入口仍可按场景、DDL 和 seed 精确读取旧 checkpoint。完整规则
见 [输出短名称规范](docs/OUTPUT_NAMING.md)。

## 测试

当前保存基线在 2026-07-28 最终审计时运行：

```powershell
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -v
```

结果为：

- 静态语法检查通过；
- `205 tests OK`，0 失败，0 跳过；
- 原始 HRL、完整 safe HRL、三层 checkpoint 保存恢复和离线 CEWS 小规模验收通过。

2026-07-30 修改项目说明后，重新运行了上述最终验收模块：
`4 tests OK`，0 失败，0 跳过；README 与保存基线文档的 UTF-8 读取和本地链接检查
也通过。本次没有重新运行全部 205 项，因此全量数字仍明确引用 2026-07-28 的记录。

2026-07-30 完成算法目录整理后，再次运行：

```powershell
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -q
```

最终结果为：静态语法检查通过，`209 tests OK`，0 失败，0 跳过。新增 4 项目录测试
验证当前方法、FCFS 对比实现和根目录兼容包的实际解析位置；原始 HRL、安全 HRL、
checkpoint 恢复和离线 CEWS 小规模验收仍包含在这 209 项中。

2026-07-30 完成 `out/` 短名称改造后，最终代码状态重新通过静态语法检查和
`215 tests OK`，0 失败，0 跳过。该数字包含新增的短名称确定性、长度上限、
旧 checkpoint 精确兼容和 27 个评估入口一致性测试。

该结果只代表当时的 Windows/Python 小规模环境，不代表正式大规模实验、跨平台复现、
训练收敛、全部 seed 零违反或 LLM 性能提升。当前解释器当时没有安装 `pytest`，
在线 SeEvo 也因未提供 `QWEN_API_KEY` 未完成。

后续再次修改后应重新运行测试，并把新的结果追加到进度文档，不得沿用旧测试结论。

## 主要目录

- `algorithms/llm_safe_hrl/`：本文当前方法的全部核心算法代码。
  - `base/`：HRL 环境、D3QN、shield、fallback、replay、lambda 和准入；
  - `hrl_mix/`：三层训练、评估、安全指标、训练流水线和实验矩阵；
  - `LLM/`：SeEvo、提示词、CEWS evaluator、候选规则和启发式 manifest。
- `algorithms/comparisons/`：对比算法；已有历史 FCFS 位于其 `fcfs/` 子目录。
- `common/`：共享工作流、资源、XML、模糊数和指标工具。
- 根目录 `base/`、`hrl_mix/`、`LLM/`、`baseline_fcfs/`：仅用于兼容历史 import，
  不再放置业务实现。
- `run/hrl_mix/`：历史场景/DDL 评估入口。
- `data/dax/`：工作流 DAX XML。
- `data/deadlines/`：FCFS/HEFT 截止期参考缓存。
- `tests/`：CEWS、模糊模型、safe HRL 与最终验收测试。
- `docs/`：基线快照、改造步骤、进度和保存说明。
- `out/`：checkpoint、日志、实验 manifest 和运行输出。
- `plots/`：绘图脚本、论文图表和汇总文件。
- `tools/`：维护、数据准备和 manifest 管理工具。

## 场景命名

场景代码由任务规模和资源规模组成：

- `S`：small；
- `M`：medium；
- `L`：large。

例如 `SS` 表示 small task + small resource。DDL 使用：

- `T`：Tight；
- `M`：Medium；
- `L`：Loose。

## 文档导航

- [2026-07-30 算法目录整理说明](docs/PROJECT_LAYOUT_2026-07-30.md)
- [`out/` 输出短名称规范](docs/OUTPUT_NAMING.md)
- [2026-07-30 当前保存基线](docs/PROJECT_BASELINE_2026-07-30.md)
- [安全 HRL 分阶段改造进度](docs/SAFE_HRL_PROGRESS.md)
- [安全改造前 CEWS 快照](docs/CEWS_TASK_CONSTRUCTIVE_SNAPSHOT_2026-07-28_PRE_SAFE_RL.md)
- [早期 CEWS 快照](docs/CEWS_TASK_CONSTRUCTIVE_SNAPSHOT_2026-07-23.md)
- `docs/LLM辅助安全分层强化学习改造步骤.docx`

## 后续修改约定

1. 不回写或改名历史快照；用新日期追加文档。
2. 每次修改记录变更文件、接口、配置、测试和未验证范围。
3. 新功能使用显式配置开关，默认不得改变旧实验语义。
4. 不得让 LLM 直接选择 Host 或 VM。
5. 不得删除固定 VM 规则；它仍是确定性安全回退基础。
6. 不得把 safety cost 重新合并进 performance reward。
7. checkpoint、replay、启发式和实验 manifest 的 schema 变化必须显式拒绝旧版本，
   不得静默错误读取。
8. 正式实验前冻结训练、验证和最终测试 seed，最终测试不得用于调参。
9. 当前方法的新业务代码必须放入 `algorithms/llm_safe_hrl/`；新增对比算法必须放入
   `algorithms/comparisons/<method_id>/`，不得重新把实现散落到根目录兼容包。

## 新增 DRL-EA / Niching-GP 对比方法

独立对比方法 `method_id=drlea_nichgp` 位于
[`algorithms/comparisons/drlea_nichgp/`](algorithms/comparisons/drlea_nichgp/README.md)。
它按“RA 训练 → 固定 RA 的 Niching GP 四规则进化 → 固定 RA/规则的 SA 训练”
运行，共享当前项目的 DAX、模糊资源、三时间线、能耗积分、截止期和最终指标，
但不接入主算法的 Manager–Host–VM 策略、LLM、SeEvo 或安全控制器。调试输出
使用短路径 `out/comparisons/drlea_nichgp/<scenario>_<ddl>_s<seed>/`。
