# 项目保存基线说明（2026-07-30）

## 1. 基线身份

- 文档基线标识：`llm-safe-hrl-audited-baseline-20260730`
- 保存日期：2026-07-30
- 代码状态来源：阶段 18 最终代码审计与小规模验收完成后的工作区
- 后续用途：作为后续功能修改、正式实验准备和消融适配器补齐的共同起点
- Git 状态：当前根目录没有有效 `.git/HEAD`，无法记录 branch、commit 或 tag

因此，该标识只表示“这份保存副本的项目状态”，不是 Git 版本。复制或压缩项目后，
建议对最终归档文件另外计算 SHA-256，并把归档文件名和 hash 补充到本节：

```text
archive_file = 待保存后填写
archive_sha256 = 待保存后填写
```

不要预先填写一个尚未实际生成的归档 hash。

## 2. 保存时必须包含

建议完整保存当前工作区。至少必须包含：

- `main.py`、`README.md`；
- `LLM/`、`base/`、`common/`、`hrl_mix/`；
- `baseline_fcfs/`、`run/`、`tools/`；
- `data/dax/` 和 `data/deadlines/`；
- `tests/`、`docs/`；
- `hrl_mix/config/` 下的训练流水线和实验矩阵配置；
- `LLM/problems/cews_task_constructive/safe_heuristic_library.json`；
- `out/experiment_manifests/` 下已经记录在进度文档中的阶段 17 manifest。

视是否需要复现实验结果决定是否保存：

- `out/ckpts/`；
- `out/logs/`；
- 其他 `out/` 运行产物；
- `plots/` 中的结果表和图。

可以不保存或在副本中清理的可再生内容：

- `__pycache__/`、`*.pyc`；
- `.pytest_cache/`；
- 临时日志、临时 Hydra 输出和未完成运行目录；
- 可重新创建且不包含实验结果的本地虚拟环境。

删除任何运行产物前应先确认它不是论文结果、正式 checkpoint、准入证据或唯一日志。

## 3. 当前两条主线

### 3.1 SeEvo / LLM

- LLM 生成 `get_task_priority_v2`。
- 该函数只负责 ready-task 排序。
- LLM 不直接选择 Host 或 VM。
- CEWS evaluator 使用固定 VM 规则完成资源选择。
- 候选比较采用模糊 DDL 可行性优先，再比较风险调整模糊能耗。
- LLM 规则进入安全 Manager 前必须经过版本化准入。
- 当前保存的三个历史 LLM 规则全部为 rejected；尚无真实 admitted LLM 工件。

### 3.2 HRL

- 保留 Manager–Host Agent–VM Agent 三层职责。
- Manager 使用 legacy 规则权重模式，或显式使用 heuristic-selection 模式。
- Host Agent 选择 Host，VM Agent 选择 VM。
- safe 模式提供独立 performance reward、安全代价、三类 mask、safety shield、
  确定性 fallback、双价值网络、共享动态 lambda、安全 replay 和可行性优先保存。
- `safe_rl=false` 时继续使用原 HRL 路径。

## 4. 不得静默改变的语义

```text
fuzzy_energy_score
  = fuzzy_energy_mean + lambda_E * fuzzy_energy_std
lambda_E
  = 1.0

R(T_i)
  = (1 - eta) * T_i_modal + eta * T_i_upper
eta
  = 0.95

D_i
  = deterministic deadline
```

还必须保持：

1. `get_task_priority_v2` 接口兼容；
2. LLM 只参与 ready-task 排序；
3. 固定 VM 规则不删除；
4. performance reward、safety cost、硬动作合法性和长期模糊 DDL 约束分离；
5. 空安全动作集合使用确定性 fallback；
6. replay 用 executed action 训练，proposed action 只用于分析；
7. 所有新功能具有显式配置开关；
8. 关闭 safe RL 时不悄悄改变旧实验语义。

## 5. 当前已实现

- 模糊能耗与模糊 DDL 三影子时间线；
- CMDP reward-cost 环境信息；
- 工作流和任务级安全裕量及动作风险预测；
- Host/VM legal、safety、final mask；
- safety shield 与确定性 fallback；
- Manager、Host、VM 安全 observation；
- 三层独立 `Q_r/Q_c`；
- episode/EMA 共享动态拉格朗日控制；
- final-mask 安全探索；
- 版本化安全 replay；
- Manager 启发式选择模式；
- SeEvo 规则准入和版本管理；
- 安全示范数据及离线预训练接口；
- 五阶段训练流水线和课程配置；
- 可行性优先 best checkpoint；
- training/validation/final-test 安全指标；
- 对比与消融实验 manifest。

“已实现”只说明代码和测试入口存在，不等于已经完成正式实验。

## 6. 当前已测试

2026-07-28 的最终验收记录：

- `python -m compileall -q .`：通过；
- `python -m unittest discover -s tests -p "test_*.py" -v`：
  `205 tests OK`，0 失败，0 跳过；
- 原始 HRL：真实 1 workflow、1 episode 短运行通过；
- safe HRL：完整安全开关、真实 1 workflow、1 episode 短运行通过；
- 三层 legacy/safe checkpoint 保存和恢复通过；
- CEWS reference 规则的 1 seed、1 workflow 离线 evaluator smoke 通过。

测试环境为 Windows 10、Python 3.12.10、NumPy 2.1.2、
PyTorch 2.5.1+cu121、PyYAML 6.0.1。

该测试记录属于保存基线。后续代码修改后必须重新测试，不能继续引用该结果证明新代码
已经通过。

2026-07-30 文档更新后又运行：

```powershell
python -m unittest -v tests.test_final_safe_hrl_acceptance
```

结果：`4 tests OK`，0 失败，0 跳过。另行检查 README 和本文件可以按 UTF-8
读取，且其中全部本地 Markdown 链接均存在。本次没有重新运行全量 205 项。

## 7. 当前尚未验证或不具备

- 未完成正式多 seed 长训练和 withheld final-test；
- 未证明训练收敛、全场景零违反、跨平台复现或部署安全；
- 未校准 Q_c、动态 lambda、risk prediction 和 fallback 触发分布；
- 没有真实 admitted SeEvo LLM 规则；
- 没有正式安全示范数据 manifest；
- 在线 SeEvo smoke 未完成，当前审计环境未提供 `QWEN_API_KEY`；
- 当前 Python 环境未安装 `pytest`；
- 13 个对比/消融运行仍 fail-closed，只有 1 个当前可执行；
- `original_hrl_plus_llm`、严格 no-LLM 来源过滤及固定 EDF/FCFS/SeEvo-only 统一
  adapter 尚未补齐；
- 指定的 `docs/LLM辅助安全分层强化学习改造步骤.md` 不存在，只有同名 DOCX；
- 没有有效 Git 元数据。

预测 fuzzy finish、safety margin 和动作风险不是实际完成后的安全保证；有限 seed
的零违反也不是部署级安全证明。

## 8. 后续修改流程

每次在此基线上继续开发时：

1. 先阅读本文件、`README.md` 和 `docs/SAFE_HRL_PROGRESS.md`；
2. 明确本次阶段边界，不提前实现后续阶段；
3. 修改前核对真实调用链；
4. 保持冻结公式和职责边界；
5. 新增或修改配置时默认关闭新语义；
6. 为新行为增加单元或 smoke 测试；
7. 运行相关测试并记录实际结果；
8. 在 `SAFE_HRL_PROGRESS.md` 追加新日期和阶段，不覆盖本基线记录；
9. 列出修改文件、接口变化、风险和未验证结论；
10. 形成重要里程碑时保存新副本并创建新的日期基线文档。

建议后续副本命名：

```text
sec_job_YYYYMMDD_<stage-or-purpose>
```

例如：

```text
sec_job_20260730_safe_hrl_audited_baseline
```

## 9. 文档关系

- `README.md`：当前项目入口和快速说明；
- 本文件：2026-07-30 保存副本的冻结说明；
- `SAFE_HRL_PROGRESS.md`：从阶段 0 到当前阶段的完整修改、测试和风险记录；
- `CEWS_TASK_CONSTRUCTIVE_SNAPSHOT_2026-07-28_PRE_SAFE_RL.md`：
  安全改造前 CEWS 基线；
- `LLM辅助安全分层强化学习改造步骤.docx`：总体设计步骤。

本文件记录的是当前保存点，不应在后续修改中改写为新的代码状态。后续状态应新建日期
文档并在进度文件中追加。
