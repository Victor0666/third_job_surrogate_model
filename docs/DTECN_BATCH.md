# DTECN-LLM 批量在线调度

MADDPG 网络、训练奖励、训练预算和更新规则保持不变。批量模式使用单独入口，旧逐任务工件不覆盖、不冒充批量结果。LLM API 请求参数与原入口相同。

每次取当前模拟时刻的 FCFS 就绪任务最长可行前缀。可行性检查采用二分匹配，仅检查存在无冲突合法分配，不优化能耗或延迟，不跳过前面的任务。每个空闲 VM 批内只接收一个任务；忙 VM 不排队。

非唯一批次只调用一次 LLM。输出 `assignments`，每个任务明确对应 VM；整批校验顺序、任务身份、合法性及冲突后，按 FCFS 顺序提交，提交过程中不推进时间。联合映射唯一时直接执行。临时失败/非法输出使用确定性合法联合回退，无修复对话和未来 rollout；认证和配置错误直接停止。

教师案例仍由已训练 MADDPG 逐任务生成。选择动作时，仅保留能够让批次后续任务合法分配的候选，再按 Actor 偏好选择；这是联合合法性约束，不新增优化目标。收集新的批量案例，不使用旧逐任务案例拼凑。批次反馈从提交前到下一批次开始（或 episode 完成）按实际已观察区间计算一次，不将总成本按任务数重复累加。每个种子重新初始化在线案例。

## 旧模型复用与完整命令

从项目根目录执行；已有 API 密钥和依赖应保持可用。教师目录必须对应同一个场景、协议、DDL、种子、模型参数、工作负载和训练实现。先严格验证原教师 manifest、模型及案例哈希；仅沿用已审核的 train_config.py 非正训练轮数校验兼容。

SS 示例（使用此前已有模型）：

```bash
TEACHER='out/comparisons/dtecn_llm/combined_v1/artifacts_v3/single/SS/tight/s1/02c8907b15'
for stage in collect evaluate; do
    python -u -m tools.run_dtecn_llm_batch "$stage" \
        --teacher-run "$TEACHER" \
        --protocol single --scenario SS --ddl T --optimizer-seed 1 \
        --model qwen3.8-max --reasoning-effort medium --max-tokens 32768 --device cuda || break
done
```

`collect` 不调用 LLM、不训练网络。可在 collect 后执行同参数的 `validate`，它是可选在线检查。已有模型不用重训；新源场景先通过原 MADDPG `run train` 生成教师模型，再执行批量 collect/evaluate。

SS 教师评测 SS/MS/LS，各使用 201–230。SM 教师评测 SM/MM/LM，SL 教师评测 SL/ML/LL。使用各自已训练的教师目录，不能拿 SS 模型或案例冒充 SM/SL 模型。

## 输出

结果在 `TEACHER/batch_v1/`，包含独立 checkpoint 副本、manifest、新批量 cases.jsonl，以及：

- `icl_final_test_decisions.csv`：每成功提交一个任务立即写一行，包含 batch_id/batch_size。同批 5 个任务形成 5 行，但最多一次请求。
- `icl_final_test_decisions.jsonl`：每个种子完成后保存逐任务记录。
- `icl_final_test_batches.jsonl`：每批一条已观察反馈，包含 seed/scenario、联合分配、队列目标及请求次数、耗时、已报告 token 和缺失计数。
- `icl_final_test.json` / `.csv`：所有场景和种子完成后的汇总；requests/fallbacks 按批统计，assignment_steps 按任务统计。

验证文件以 `icl_validation` 开头。CSV/批次反馈及时 flush；不承诺断电持久化。已有正式或部分结果会停止以防覆盖，不自动续跑。不修改共享模拟器、其他算法或旧原始 manifest。

联合批次及案例是 DAG/VM 问题适配，不是原论文车辆连续卸载/CPU份额的严格复现；API 成本、批大小与回退率需随成绩一起报告。批量效果仍需正式训练模型和真实 API 评测验证。
