# Energy-only LLM mode

将离线 LLM 的 task-ordering 能耗搜索与在线 Safe-HRL 的 DDL 安全学习分开。
原模式默认仍为 `original`，原 YAML 和 Prompt 未修改。
新增 `problem=cews_task_constructive_energy_only`，只支持正式 `protocol single`。

LLM 的八参数接口不变。允许 `min_exec_time`、`min_comm_time`、
`min_incremental_energy`、`upward_rank`、`remaining_work`、`ready_wait_time`。
`upward_rank` 是环境预计算的 HEFT DAG 执行/通信结构特征，不融合 deadline。
`slack` 和 `uncertainty` 可出现在签名中，但禁止读取，包括改名后的第 4/8 参数。
AST 检查同时禁止 deadline、predicted_violation、safety_margin、violation_risk、
risk、Qc、q_c、lambda 及动态反射访问；原有候选代码合法性检查继续执行。

唯一 fitness 为各评价 seed 的 `fuzzy_total_energy_score` 算术均值，单位 J。
沿用正式每个 seed 的 fuzzy energy score 定义，不把它替换成 modal energy。
SeEvo 按该值升序；CMA-ES `tell()` 接收原始能耗标量，阶段筛选/confirm/最终参数
选择也按能耗。不使用违反率、lateness、feasibility、跨 seed 方差或 tolerance bucket。
这些安全指标仍计算并保存在评价报告中。

Top-K 严格选能耗最低的 10 个不同 frozen-source/heuristic ID，能耗相同时按
source SHA256 排序。不使用原模式的 unique-structure-first，避免挤掉更低能耗候选。
Prompt 鼓励不同节能机制；不保证 LLM 一定产出 10 种不同语义的规则。
程序必须能 import/execute、返回正确 shape 的 finite priorities，并完成全部指定
seed 的仿真，得到 finite energy。导出/加载仍检查文件、报告、配置、规则元数据、
seed 和 protocol 的来源/一致性。非零违反率或 lateness 不拒绝候选。

本模式不执行最终 safety admission，也关闭 counterfactual feedback、critical-state
replay 和 CMA 诊断 replay gate。保留 surrogate 加速，默认启用；可显式设置
`surrogate.enabled=false` 关闭。Energy-only 的结构/参数代理只训练能耗回归器，
quick 特征中的 feasibility/violation/lateness/跨 seed 方差置零。筛选、验证召回率
和审计只按能耗判断，探索只使用能耗预测不确定性；仍保留随机探索、异常时全量
真实评估回退、best/elite 和最终验证的真实复核。代理数据/模型与原模式隔离。
参数诊断仍可保存，但
LLM 的性能反馈只包含 mean fuzzy energy，避免安全指标或建议重新进入反思 Prompt。
不新增 LLM 请求；generation/population/repair budget 与原配置一致。
默认单次评价子进程超时为 300 秒（`timeout=300`）。代理需要先积累真实标签并
通过验证：默认至少 300 个真实标签，结构至少 20 个、参数至少 12 个；冷启动
阶段不会立即加速。代理不改变 LLM 请求预算，实际节省时间取决于模型质量。

## 生成与导出：SS_T 示例

在 Linux 仓库根目录、已安装依赖的环境执行。API 凭据沿用原 LLM 设置。

```bash
export PYTHONPATH="$PWD:$PWD/algorithms/llm_safe_hrl:$PWD/algorithms/llm_safe_hrl/LLM${PYTHONPATH:+:$PYTHONPATH}"

CACHE_ARGS=(
  --deadline-cache "SS=$PWD/data/deadlines/fcfs/exact_mix_v1/fcfs_smallTask_smallRes_exactmix_formal38.json"
  --deadline-cache "MS=$PWD/data/deadlines/fcfs/exact_mix_v1/fcfs_medTask_smallRes_exactmix_formal38.json"
  --deadline-cache "LS=$PWD/data/deadlines/fcfs/exact_mix_v1/fcfs_largeTask_smallRes_exactmix_formal38.json"
)

python algorithms/llm_safe_hrl/LLM/main.py \
  problem=cews_task_constructive_energy_only \
  --protocol single --source-scenario SS --ddl T "${CACHE_ARGS[@]}"

# 使用上一次生成日志打印的 execution_id；不要混入原模式的 execution。
read -r -p 'energy-only execution_id: ' EXECUTION_ID
python -m algorithms.llm_safe_hrl.LLM.export_topk \
  --llm-objective energy_only --source-scenario SS --ddl T \
  --execution-id "$EXECUTION_ID" --top-k 10

export TOPK="$PWD/out/main_energy_only/SS/T/$EXECUTION_ID/topk_heuristic_library_k10.json"
```

原模式继续使用默认 problem、默认 export objective `original`。
生成/Top-K 输出在 `out/main_energy_only/{source}/{T|M|L}/{execution_id}/`，
不会覆盖 `out/main_single`。原 runtime 目录仍按唯一 execution_id 隔离。
移机后可给 exporter 传实际 `--run-dir` 和 `--runtime-dir`。

## 重新生成 Safe-HRL demonstrations（需要离线预训练时）

新的 library 必须重新生成 demo。旧 demo 的 heuristic IDs/order/manifest SHA256
校验原样保留；objective identity 字段也包含在 manifest 文件 hash 中。
改 objective 或换库，即使 IDs 不变，旧 demo 也会因 hash 不匹配 fail-fast。
demo 的 Qr/Qc、legal mask、monitor-only shield、energy observation 和安全筛选保持原定义。

```bash
export DEMO="$PWD/out/safe_demonstrations_energy_only/SS_T/$EXECUTION_ID/manifest.json"
HEURISTIC=$(python -c 'import json,os; print(json.load(open(os.environ["TOPK"]))["llm_rules"][0]["heuristic_id"])')

for SPLIT in train validation; do
  python -m tools.generate_safe_demonstrations \
    --manifest "$DEMO" --manager-heuristic-manifest "$TOPK" \
    --heuristic "$HEURISTIC" --generate-split "$SPLIT" --scenario SS --ddl T \
    --train-workflow-seeds 1,2,3,4,5 --train-resource-seeds 1,2,3,4,5 \
    --validation-workflow-seeds 101,102,103 --validation-resource-seeds 101,102,103 \
    --final-test-workflow-seeds 201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230 \
    --final-test-resource-seeds 201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230 \
    --safe-rl-energy-reward-scale 0.002 "${CACHE_ARGS[@]}"
done
```

上面只生成 train/validation；final-test seed 参数声明保留集，不生成测试 demo。
也可按各 ID 分别导出 demo，再使用原离线预训练流程；不要复用旧库的 demo。

## 在线 Safe-HRL 训练

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m hrl_mix.train \
  --protocol single --source-scenario SS --ddl T --episodes 600 --optimizer-seed 0 \
  --safe-rl --safe-rl-shield --safe-rl-state --safe-rl-dynamic-lambda \
  --safe-rl-heuristic-manager --llm-only-heuristics \
  --manager-heuristic-manifest "$TOPK" --without-curriculum \
  --safe-rl-energy-reward-scale 0.002 \
  --safe-rl-cost-budget 0.02 --safe-rl-lambda-init 0.5 --safe-rl-lambda-lr 0.02 \
  --validation-workers 1 "${CACHE_ARGS[@]}"
```

这是纯在线训练，不消费刚生成的 demo；需要预训练时使用现有 offline 参数/pipeline。
训练从库自动读取 objective、验证 identity，并将 byte SHA256 与 objective 写入
run ID/`heuristic_library_version`；energy-only checkpoint 顶层也记录两者。
训练输出自动隔离到 `out/main_energy_only`、`checkpoints/main_energy_only`。
export CLI 显式 expected objective 不匹配会报错；training 自动读取，不新增 objective CLI。

Manager/Host/VM 仍为 `Qr - lambda * Qc`，Qc 仍只有 actual DDL violation。
网络、能耗 reward/scale、观测、PER、lambda、shield/fallback、validation/final-test seeds、
模型选择和评测指标不变。原固定 Host→VM 规则评价器也未改动，所以本消融只改变
LLM 的 objective/候选特征/筛选，不代表移除了仿真资源控制器自身的 DDL 偏好。

比较 original LLM / energy-only LLM / Fixed-FCFS 时保持在线 seed、600 episodes、
Safe-RL 参数、reward scale、资源/DDL/caches 和预训练/curriculum设置一致。
Full 使用了预训练而 FCFS 纯在线时，不可把差异全部归因于排序规则。
