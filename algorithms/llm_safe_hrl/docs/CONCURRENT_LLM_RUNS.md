# LLM_Safe_HRL 并发运行说明

## LLM/SeEvo 训练命令

在项目根目录运行。`--source-scenario` 指定源场景，`--ddl` 指定
Tight/Medium/Loose；执行 ID 由程序自动生成，不使用 DRL_EA 的 `a0`。

下面三个命令可以分别放在三个终端并发运行：

```bash
python algorithms/llm_safe_hrl/LLM/main.py --protocol single --source-scenario SS --ddl T --deadline-cache SS=data/deadlines/fcfs/exact_mix_v1/fcfs_smallTask_smallRes_exactmix_formal38.json --deadline-cache MS=data/deadlines/fcfs/exact_mix_v1/fcfs_medTask_smallRes_exactmix_formal38.json --deadline-cache LS=data/deadlines/fcfs/exact_mix_v1/fcfs_largeTask_smallRes_exactmix_formal38.json
python algorithms/llm_safe_hrl/LLM/main.py --protocol single --source-scenario SS --ddl M --deadline-cache SS=data/deadlines/fcfs/exact_mix_v1/fcfs_smallTask_smallRes_exactmix_formal38.json --deadline-cache MS=data/deadlines/fcfs/exact_mix_v1/fcfs_medTask_smallRes_exactmix_formal38.json --deadline-cache LS=data/deadlines/fcfs/exact_mix_v1/fcfs_largeTask_smallRes_exactmix_formal38.json
python algorithms/llm_safe_hrl/LLM/main.py --protocol single --source-scenario SS --ddl L --deadline-cache SS=data/deadlines/fcfs/exact_mix_v1/fcfs_smallTask_smallRes_exactmix_formal38.json --deadline-cache MS=data/deadlines/fcfs/exact_mix_v1/fcfs_medTask_smallRes_exactmix_formal38.json --deadline-cache LS=data/deadlines/fcfs/exact_mix_v1/fcfs_largeTask_smallRes_exactmix_formal38.json
```

可选的 `--run-name paper` 只增加可读标签，仍会保留自动时间戳、微秒和
进程号。例如 `paper_20260825_120000_123456_p31842`。不要设置
`hydra.run.dir`、`execution_id`、`experiment_key` 或
`runtime_output_root`；这些字段由程序管理并会拒绝命令行覆盖。

## 输出目录

以 `SS/T/<execution_id>` 为例：

```text
algorithms/llm_safe_hrl/LLM/outputs/formal/SS_T/<execution_id>/
  .hydra/
  responses/
  candidate_stdout/
  reflections/
  run_manifest.json

out/main_single/SS/T/<execution_id>/
  generated/
  counterfactual_feedback/
  critical_state_replay/
  parameter_evaluation_cache.json
  effective_admission_config.yaml
  safe_heuristic_library_single_SS.json
  experiment_protocol_manifest.json
  run_manifest.json

checkpoints/main_single/SS/T/<execution_id>/
```

M、L 分别使用 `SS_M`/`SS_L` 和 `SS/M`/`SS/L`。即使同一时刻启动
相同场景和相同 DDL，自动执行 ID 也会把可写目录继续分开。

DAX 和 deadline cache 是输入，只读共享，不需要为每个进程复制或设置独立
cache 路径。上面三个进程可以读取同一组 FCFS deadline cache。T/M/L 的区别
来自 deadline 混合参数：T=0.8、M=0.5、L=0.2；alpha 仍为 2.0/3.0。

目录和文件冲突已经隔离，但 CPU、内存和 LLM API 并发额度仍由多个进程共享。
每个 SeEvo 进程内部还会并行执行候选评估；同时启动三个任务前应根据机器核心数、
内存和 API 限流额度设置总并发量。这只影响运行资源与速度，不改变规则、适应度、
随机种子或模型选择逻辑。

## 让 Safe-HRL 使用指定的 LLM 结果

LLM 训练成功后，两处 `run_manifest.json` 的 `status` 都会变成
`COMPLETED`。后续 Safe-HRL 使用 `--llm-run-manifest` 显式选择结果，例如：

```bash
python -m hrl_mix.train --protocol single --source-scenario SS --ddl T --safe-rl --safe-rl-shield --safe-rl-state --safe-rl-heuristic-manager --llm-run-manifest out/main_single/SS/T/<execution_id>/run_manifest.json --deadline-cache SS=data/deadlines/fcfs/exact_mix_v1/fcfs_smallTask_smallRes_exactmix_formal38.json --deadline-cache MS=data/deadlines/fcfs/exact_mix_v1/fcfs_medTask_smallRes_exactmix_formal38.json --deadline-cache LS=data/deadlines/fcfs/exact_mix_v1/fcfs_largeTask_smallRes_exactmix_formal38.json
```

程序会在训练前检查 LLM 运行已经完成、Single/Multi 协议与源场景一致、DDL
一致，并检查规则库文件存在。不会自动选择“最新目录”，因此并发运行时不会把
SS_T、SS_M、SS_L 或两次相同实验的结果串用。
