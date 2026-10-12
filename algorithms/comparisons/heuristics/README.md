# 固定启发式对比算法

这四个方法使用同一 `FuzzyBaselineEnv`，只改变 ready-task 排序。
策略位于 `policies.py`；FCFS 和 VM 选择直接复用已有 `fcfs/policies.py`，
执行与评测复用 `fcfs/train_fcfs.py` 和 `fuzzy_common/evaluation.py`。

| method_id | 名称 | 排序定义（小分数优先） |
|---|---|---|
| fcfs_fixed | FCFS-Fixed | `(ready_time, workflow_arrival_time, workflow_id, task_id)` |
| spt_fixed | SPT-Fixed | 最小可行 VM 模态纯计算时长，不含通信和排队 |
| edd_fixed | EDD-Fixed | 工作流绝对截止期，不是 slack 或任务子截止期 |
| mwkr_fixed | MWKR-Fixed | 当前任务及所有后继的去重 MI 总和取负值 |

SPT、EDD、MWKR 分数相同时使用上述 FCFS 顺序，不依赖 ready 集输入顺序。
MWKR 是 DAG 后继子图适配版，不是整个工作流未完成工作量；汇合节点不会重复计数。
SPT 排序使用结构可行 VM（可含忙碌 VM）的最短预计计算时长，与现有任务特征一致；
实际分配只能在当前合法空闲 VM 中选择。每个阶段排序一次，复用共享环境的事件推进。

## VM 选择与公平性

四个方法全部调用 `select_vm_fixed`，它将当前合法空闲 VM 集合传给
`select_vm_deterministic`。若存在按时 VM，按风险调整增量能耗、完成时刻、
VM 编号选择；否则按子截止期违反量、能耗、完成时刻、VM 编号选择。
合法动作约束始终保留，但禁用安全 shield、fallback、Q_c、Lagrange、
LLM、示范预训练和策略学习。

统一使用主算法相同的 DAG、到达过程、资源规模、通信模型和指标实现。
默认每实例 50 个工作流，模糊参数 delta1=0.75、delta2=1.2、eta=0.95、
energy_lambda=1.0；模糊资源种子等于 episode seed。
T/M/L 的小 deadline 系数概率分别为 0.8/0.5/0.2，系数与主算法共享。
正式测试固定为 201-230，共 30 个种子，无训练或测试集调参。

**必须传入与 LLM_SHRL 评测相同的截止期缓存**，不能为每种启发式重新生成缓存。
缺失文件或缓存数据不匹配会报错，不回退到另一套截止期。
如果主算法使用自定义工作负载或模糊参数，必须同步修改共享 `--config`。
完整 LLM_SHRL 与这些基线的差异包含任务排序、资源决策和安全机制，
不能把全部结果差异解释为任务排序的单独贡献。

## 执行

在项目根目录运行；以下 SS 示例使用项目实验矩阵现有的缓存路径。
若主算法使用另一份缓存，替换为其实际评测路径。

```powershell
$cache = 'data/deadlines/fcfs/diagnostic/fcfs_fixed_SS_exactmix_formal38.json'
foreach ($method in @('fcfs_fixed', 'spt_fixed', 'edd_fixed', 'mwkr_fixed')) {
    python -m algorithms.comparisons.heuristics.run_heuristic --method $method --scenario SS --ddl T --deadline-cache "SS=$cache"
}
```

将 `--ddl T` 改为 M/L 可评测其他紧迫度；`--scenario` 支持全部九个场景，
每个场景使用自己的主算法缓存。结果默认保存至
`out/comparisons/<method_id>/<scenario>_<T|M|L>/final_test_metrics.json`，
包含规则定义、无训练标记、环境协议与参数、缓存路径、逐种子记录及汇总指标。
不同设置请使用 `--output` 指定独立文件，重复同一设置会覆盖其结果。

Single 泛化评测使用 `--single-source SS`，并提供 SS、MS、LS 三个
`--deadline-cache SCENARIO=PATH`；SM/SL 同理。固定规则无须训练，
结果记录每个目标场景的完整环境协议。

实验矩阵也已注册四个 method_id。验证命令：

```powershell
python -m pytest tests/test_fixed_heuristics.py tests/test_fcfs_variants.py tests/test_experiment_matrix.py -q
```
