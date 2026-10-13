# DTECN-LLM 实时决策 CSV

在仓库根目录，将在线命令入口改为 `python -u -m tools.evaluate_dtecn_llm_with_csv`，其余参数保持原值。例如：

```bash
python -u -m tools.evaluate_dtecn_llm_with_csv evaluate \
  --protocol single --scenario SS --ddl T --optimizer-seed 1 \
  --model qwen3.8-max --reasoning-effort medium --max-tokens 32768 --device cuda
```

也支持 `validate`。训练和案例收集继续使用原入口。

每次成功执行资源分配后立即写入并 flush 一行 CSV，包含场景、种子、种子内决策序号、决策时刻、任务 ID、工作流槽位、VM、动作来源、偏好向量和状态。LLM、单一合法 VM 和回退分配都记录。

新增文件与原 JSONL 同目录，分别为 `icl_validation_decisions.csv` 和 `icl_final_test_decisions.csv`。原逐种子 JSONL、最终 JSON/CSV 的行为保持不变。CSV 不含尚未发生的奖励或区间反馈；这些仍在原 JSONL 中记录。

该入口不改核心源码身份，因此已有 MADDPG 检查点和初始案例无需重训。中断会保留已写行；已有同名 CSV 时停止，不覆盖也不自动续跑。flush 允许实时查看，但不承诺断电时的磁盘持久化。

尚在运行的进程不会自动加载新入口，需在新启动的评测中使用。
