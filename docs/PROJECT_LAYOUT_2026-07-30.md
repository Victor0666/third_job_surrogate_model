# 算法代码目录整理说明（2026-07-30）

## 1. 目的

为避免后续引入对比算法时混淆“本文方法”和“基线方法”，本次只调整代码物理布局、
路径解析和说明，不改变调度逻辑、数学公式、模型结构或配置默认值。

当前布局修订标识：

```text
llm-safe-hrl-algorithm-layout-20260730
```

该标识不是 Git tag 或 commit；当前工作区仍没有有效 Git `HEAD`。

## 2. 新布局

```text
algorithms/
├── llm_safe_hrl/             # 本文当前方法
│   ├── base/                 # 环境和安全 HRL 核心
│   ├── hrl_mix/              # 三层训练、评估和配置
│   ├── LLM/                  # SeEvo 与 CEWS ready-task 规则
│   ├── paths.py              # 统一项目/算法路径
│   └── README.md
└── comparisons/              # 对比算法
    ├── fcfs/                 # 已有历史 FCFS
    └── README.md
```

以下目录仍位于项目根目录，因为它们是跨算法共享资源或公共入口：

- `common/`、`data/`、`tests/`、`docs/`、`out/`；
- `run/` 和 `tools/`；
- 根目录 `main.py`；
- 根目录 `base/`、`hrl_mix/`、`LLM/`、`baseline_fcfs/` 兼容包。

## 3. 迁移映射

| 旧路径 | 新路径 | 说明 |
|---|---|---|
| `base/` | `algorithms/llm_safe_hrl/base/` | 当前方法核心 |
| `hrl_mix/` | `algorithms/llm_safe_hrl/hrl_mix/` | 当前方法训练/评估 |
| `LLM/` | `algorithms/llm_safe_hrl/LLM/` | SeEvo/CEWS |
| `baseline_fcfs/` | `algorithms/comparisons/fcfs/` | 历史对比算法 |

根目录旧包名仅通过 `__path__` 转发，不复制实现。因此旧 import 和下列命令保持
兼容：

```powershell
python -m hrl_mix.train --scenario SS --ddl T
python main.py problem=cews_task_constructive
```

## 4. 路径规则

算法代码不得再通过 `Path(__file__).parents[n]` 猜测项目根目录。所有方法可从
`project_paths.py` 读取共享项目根；当前方法内部统一使用：

```python
from algorithms.llm_safe_hrl.paths import (
    ALGORITHM_ROOT,
    BASE_ROOT,
    HRL_ROOT,
    LLM_ROOT,
    PROJECT_ROOT,
)
```

配置文件中的相对路径仍相对配置文件本身解析。本次已按新目录深度修正项目级
`data/` 和 `out/` 引用；启发式库仍位于当前方法自己的 `LLM/` 子目录。

## 5. 后续对比算法约定

1. 新算法使用 `algorithms/comparisons/<method_id>/`，不得把文件放入
   `llm_safe_hrl/`。
2. 各方法可以共享 `common/` 和冻结输入数据，但不得通过导入当前方法私有实现而
   获得额外状态或安全信息，除非实验 manifest 明确把它定义为所有方法共享组件。
3. 入口、配置、checkpoint、日志和 manifest 必须带稳定 `method_id`。
4. 对比实验继续共享工作流、到达序列、资源 seed、DDL、模糊参数和评价指标。
5. 根目录兼容包不得添加新业务模块。

## 6. 不变项

本次没有改变：

- `fuzzy_energy_mean + 1.0 * fuzzy_energy_std`；
- `R(T)=0.05*T_modal+0.95*T_upper`；
- 确定性截止期；
- `get_task_priority_v2` 接口和 LLM 职责；
- Manager–Host–VM 三层职责；
- reward/cost、legal/safety/final mask、shield、fallback、Q_r/Q_c 或 lambda；
- `safe_rl=false` 默认行为。

## 7. 本次验证

在 2026-07-30 的 Windows/Python 环境中执行：

```powershell
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -q
python -m hrl_mix.train --help
python main.py --help
python .\algorithms\llm_safe_hrl\LLM\problems\cews_task_constructive\eval.py --help
```

结果：

- 静态语法检查通过；
- `209 tests OK`，0 失败，0 跳过；
- HRL、SeEvo 和 CEWS 三个兼容/物理入口的帮助命令均正常；
- FCFS 的项目根、5 个 DAX 输入和项目级输出目录解析检查通过。

全量测试包含原始 HRL、安全 HRL、checkpoint 恢复和离线 CEWS 小规模验收，但未运行
在线 SeEvo，也未运行硬编码为 50 episode 的完整 FCFS 脚本或正式大规模实验。
