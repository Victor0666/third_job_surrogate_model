# w/o LLM Manager: Fixed-FCFS

研究 task ordering 是否影响最终能耗和 Host/VM energy regret。
Full 为 LLM Top-K → Manager DQN → Host DQN → VM DQN；本消融为固定 FCFS → Host DQN → VM DQN。
正式环境、Top-K、LLM admission、模型选择规则不变。

## Canonical FCFS

直接调用 `algorithms.comparisons.fcfs.policies.fcfs_task_order`。

```text
canonical FCFS ordering key =
(task_ready_time[tid], workflow.arrival_time, workflow_id, global_task_id)
```

仅对当前 ready tasks 排序，不读取未来任务或最终能耗。
`FCFSManagerAblationEnv` 继承正式 `CloudWorkflowEnv_VMAgents`，只替换阶段排序、
忽略兼容 Manager action，并提供 FCFS 审计身份。没有修改正式 `hrl_env.py`。
构造时强制关闭 heuristic library，不读取 Top-K 文件，也不调用 priority_rule。

## 模式与兼容性

* `retrained`：Host/VM 从头在 FCFS 下训练，正式论文消融。
* `diagnostic`：Full 的冻结 Host/VM 改用 FCFS；不是重训练消融，存在排序分布变化。

共享 checkpoint schema 要求三层，因此训练保留序列化用 ManagerPlaceholder。
它从不选动作、不执行 epsilon 探索、不记 replay、不更新 target；调用学习方法会报错。
manifest 标记 `manager_trainable=false`、`manager_placeholder`，replay/update 都为 0。
评测不加载 Manager checkpoint，只使用不含网络的兼容对象。
旧版无 energy observation 的 Host/VM 会因维度不匹配拒绝评测。

## SL_T 完整示例

以下命令在仓库根目录执行；先激活已安装项目依赖的 Python 环境。
不需要 Top-K 文件。

```bash
CACHE_ARGS=(
  --deadline-cache "SL=$PWD/data/deadlines/fcfs/exact_mix_v1/fcfs_smallTask_largeRes_exactmix_formal38.json"
  --deadline-cache "ML=$PWD/data/deadlines/fcfs/exact_mix_v1/fcfs_medTask_largeRes_exactmix_formal38.json"
  --deadline-cache "LL=$PWD/data/deadlines/fcfs/exact_mix_v1/fcfs_largeTask_largeRes_exactmix_formal38.json"
)

# 可选：生成独立 FCFS Host/VM demonstration（训练 seeds 1..5）。
python -m tools.generate_fcfs_ablation_demonstrations \
  --source-scenario SL --ddl T --split train \
  --safe-rl-energy-reward-scale 0.002 "${CACHE_ARGS[@]}"

# 如需验证 demonstration，将 --split train 改为 --split validation。
# 生成结果不是 Full 三层 demonstration，不自动进入下面的在线训练。

# 正式从头在线训练；600 episodes；每 25 episodes 验证并选择 best。
CUDA_VISIBLE_DEVICES=0 python -u -m hrl_mix.ablations.fcfs_manager.train_fcfs \
  --source-scenario SL --ddl T --episodes 600 --optimizer-seed 0 \
  --safe-rl-energy-reward-scale 0.002 \
  --safe-rl-cost-budget 0.02 --safe-rl-lambda-init 0.5 --safe-rl-lambda-lr 0.02 \
  "${CACHE_ARGS[@]}"

# 从训练输出中确认要评测的 manifest；存在多个 run 时明确选择本次 run。
find checkpoints/ablations/fcfs_manager/SL -name best_checkpoint_manifest.json
read -r -p '本次 FCFS best_checkpoint_manifest.json 路径: ' CKPT
CUDA_VISIBLE_DEVICES=0 python -u -m hrl_mix.ablations.fcfs_manager.evaluate_fcfs \
  --mode retrained --source-scenario SL --checkpoint-manifest "$CKPT" \
  --device cuda:0 "${CACHE_ARGS[@]}"

# 快速诊断：使用 Full 新观测版本的 manifest，不加载其中的 Manager。
read -r -p 'Full best_checkpoint_manifest.json 路径: ' FULL_CKPT
CUDA_VISIBLE_DEVICES=0 python -u -m hrl_mix.ablations.fcfs_manager.evaluate_fcfs \
  --mode diagnostic --source-scenario SL --checkpoint-manifest "$FULL_CKPT" \
  --device cuda:0 "${CACHE_ARGS[@]}"
```

## 输出与协议

* Checkpoints：`checkpoints/ablations/fcfs_manager/{source}/{DDL}/{run}/`
* 训练 CSV：`out/ablations/fcfs_manager/{source}/{DDL}/{run}/train.csv`
* training/validation diagnostics：checkpoint 目录下 `safe_metrics/*.csv`、`*.jsonl`。
* 评测：`out/ablations/fcfs_manager/{diagnostic|retrained}/{source}/{checkpoint_dir_name}/`，
  包括 `evaluation.json` 和每个目标场景的 seed CSV/JSONL。
* Demonstration：`out/safe_demonstrations_fcfs_ablation/SL_T/{run}/{split}/`。
  独立 `fcfs_host_vm_demonstrations_v1` manifest，不包含 Top-K IDs、顺序或 SHA256；
  只包含 Host/VM trajectories，复用正式固定资源分配、pending-transition、Qr/Qc。
  记录真实违反和原有 constraint_feasible 定义，不把不安全轨迹标成安全。
  不允许交给 Full 的三层 offline-pretraining loader。

每个 DDL 独立运行。训练 seeds 1..5，validation 101/102/103，测试 201..230。
SS→SS/MS/LS，SM→SM/MM/LM，SL→SL/ML/LL。
评测冻结网络与探索计数，不训练、不重选模型。验证仍使用原 feasibility-first 字典序。
七项 energy diagnostics 保留，按 placement 数加权，出现在 seed CSV 中，不参与 reward/selection。

## 公平性与边界

Host/VM 网络、超参数、PER、actual-violation Qc、monitor-only shield、fallback、
动态 lambda、energy observation/reward/scale、种子、缓存及模型选择完全复用正式实现。
与此前 Full 命令一致，本入口是无 curriculum、无 offline pretraining 的在线训练。
默认 scale=.002、budget=.02、lambda_init=.5、lambda_lr=.02；对比时参数必须相同。

验证使用单进程而非 Full 常用的三个并行 worker，仍在同一设备按相同种子、相同函数聚合；
这影响耗时而不改变验证定义。删除 Manager 探索会改变共享 RNG 后续消耗，
相同 optimizer seed 不保证逐步 Host/VM 采样与 Full 相同；论文建议报告多 optimizer seeds。
固定排序本身也会改变资源状态分布，这是该消融要测量的影响。
Manager placeholder 仅为现有 checkpoint 格式保留，其存储/初始化开销不能计为有效 Manager 学习。

不提供 Full 的三层 pipeline/resume 或 FCFS offline-pretraining 入口；不要把这套纯在线对照
直接与使用了 demonstrations/curriculum 的 Full 比较并归因于 ordering。
