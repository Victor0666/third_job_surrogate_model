# `out/` 短名称规范

## 1. 目的

从本规范开始，输出文件名只保留检索所需的最小身份信息。完整超参数、安全开关、
课程阶段和启发式版本写入 checkpoint config snapshot、JSON manifest 或日志字段，
不再全部拼进目录名。

命名 schema：

```text
output_naming_schema_version = 1
```

## 2. 新路径

| 产物 | 路径示例 |
|---|---|
| 原始 HRL checkpoint | `out/ckpts/hrl-ss-t-s1/` |
| 原始 HRL 日志 | `out/logs/hrl-ss-t-s1/train.csv` |
| safe HRL checkpoint | `out/ckpts/safe-ss-t-e42f608daf/` |
| safe HRL 日志 | `out/logs/safe-ss-t-e42f608daf/train.csv` |
| pipeline checkpoint | `out/ckpts/pipe-ss-t-0123456789/` |
| FCFS 日志 | `out/logs/fcfs-ss/train.csv` |
| HRL 多 seed 评估 | `out/eval/hrl/ss-t.csv` |

其中：

- `ss`：任务规模和资源规模；
- `t/m/l`：Tight/Medium/Loose；
- `s1`：随机 seed；
- safe/pipeline 末尾 10 位十六进制是规范化配置或 plan hash 的稳定摘要。

单个新输出组件最多 48 个字符，只允许小写字母、数字和连字符。

## 3. 兼容规则

- 新训练只写短目录，不再生成 `ckpts_hrl_3layer_...`、`logs_hrl_3layer_...`
  或展开全部 stage 名称的目录。
- 27 个历史 `run/hrl_mix/eval_*.py` 入口优先读取新短 checkpoint 目录。
- 如果新目录不存在，评估入口只尝试场景、DDL 和 seed 精确匹配的旧目录。
- 不再遍历并选择“第一个包含 best checkpoint”的目录，避免加载错误场景模型。
- 现有 `out/` 历史产物不会自动移动或覆盖；需要保留时仍可被精确兼容读取。

## 4. 统一实现

相关代码统一使用根目录 `output_naming.py`：

```python
from output_naming import (
    hrl_evaluation_csv_path,
    legacy_hrl_run_id,
    pipeline_run_id,
    safe_hrl_run_id,
    training_output_paths,
)
```

后续对比算法也应复用 `training_output_paths()`，使用简短、稳定且唯一的
`method_id`，不得重新把全部配置写入文件名。

## 5. 验证结果

2026-07-30 执行：

```powershell
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -q
```

结果为静态语法检查通过、`215 tests OK`，0 失败，0 跳过。测试覆盖：

- legacy/safe/pipeline 名称的稳定性和长度上限；
- checkpoint/log 路径结构；
- 短 checkpoint 优先和旧 checkpoint 精确回退；
- 27 个历史 HRL 评估入口统一使用短路径；
- 原始 HRL、安全 HRL、checkpoint 恢复和离线 CEWS 既有回归。

本次没有运行完整 30-seed 历史评估或正式训练，也没有自动重命名现有历史产物。
