# 1当前方法：LLM 辅助安全分层强化学习

本目录集中保存论文当前方法的算法代码，目录标识为：

```text
llm_safe_hrl
```

其内部职责保持分离：

- `base/`：环境、D3QN、模糊安全估计、shield、fallback、replay、lambda、
  启发式准入和离线预训练；
- `hrl_mix/`：Manager–Host–VM 三层训练、评估、配置、指标、模型选择和实验矩阵；
- `LLM/`：SeEvo、`get_task_priority_v2` 规则、CEWS evaluator、提示词及启发式
  manifest；
- `paths.py`：当前算法目录与项目级数据、输出目录的统一绝对路径定义。

`LLM/` 只生成或加载 ready-task 排序规则，不选择 Host 或 VM。Manager 选择规则，
Host Agent 选择 Host，VM Agent 选择 VM；资源动作仍受 legal/safety/final mask、
safety shield 和确定性 fallback 控制。

## 公共启动方式

为保持已有脚本和实验兼容，以下命令不变：

```powershell
python -m hrl_mix.train --scenario SS --ddl T
python main.py problem=cews_task_constructive
```

根目录 `base/`、`hrl_mix/` 和 `LLM/` 只包含兼容 `__init__.py`，通过扩展包搜索
路径转发到本目录。新业务代码应直接添加到本目录相应子目录，不应重新放回根目录
兼容包。

## 项目级共享资源

以下内容有意不搬入算法目录：

- `common/`：各方法公平共享的工作流、资源、模糊数和 XML 基础模型；
- `data/`：固定输入数据和 DDL 缓存；
- `tests/`：跨方法测试；
- `out/`：日志、checkpoint 和 manifest；
- `run/`、`tools/`：兼容启动和维护工具。

新增对比算法应放入 `algorithms/comparisons/<method_id>/`，不得把实现混入本目录。

## 输出名称

训练和评估统一通过项目根目录 `output_naming.py` 生成短路径。不要在业务代码中
重新拼接包含全部超参数的目录名；详细配置应进入 checkpoint snapshot 或 manifest。
具体格式见 `docs/OUTPUT_NAMING.md`。
