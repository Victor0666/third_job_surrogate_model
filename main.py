"""项目根目录兼容启动入口。

当前算法已集中到 ``algorithms/llm_safe_hrl``。本入口仍允许用户在项目根目录直接
运行 ``python main.py problem=cews_task_constructive``。
"""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    # 使用文件绝对位置定位实现目录，不依赖当前工作目录；run_name="__main__"
    # 保持与直接运行算法目录中的 LLM/main.py 相同的 Hydra 解析行为。
    llm_root = (
        Path(__file__).resolve().parent
        / "algorithms"
        / "llm_safe_hrl"
        / "LLM"
    )
    sys.path.insert(0, str(llm_root))
    runpy.run_path(str(llm_root / "main.py"), run_name="__main__")
