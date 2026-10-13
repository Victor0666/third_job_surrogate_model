## 在 v5 SHRL 中保留 v3 固定 Top-K

训练命令增加 `--allow-v3-energy-only-topk`。默认关闭，只允许 v5 加载器接收
v3 energy-only Top-K；v4、原目标模式以及安全准入记录不享受例外。规则库无需
改写或重评，原规则顺序、源码、报告版本和哈希保持原样。文件/报告哈希、接口、
场景、seed 隔离等校验仍执行。训练记录为“v3 筛选规则库 + v5 SHRL 训练”，
不能称为 v5 筛选的 Top-K，也不能用旧分数替代 v5 的评测结果。

开关保存在 `safe_rl.manager_heuristics.allow_v3_energy_only_topk` 的 checkpoint
配置快照中；并行验证继承训练环境设置，独立评测自动恢复，不需要重复传参。
该开关不修改 Top-K 导出器的版本要求，也不允许旧网络绕过观测/安全语义校验。
