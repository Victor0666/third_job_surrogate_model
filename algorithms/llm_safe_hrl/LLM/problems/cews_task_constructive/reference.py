"""用于纯评价与冒烟测试的可执行 v2 参考任务优先级规则。

该文件不是 SeEvo 的共享候选文件。进化期间每个个体仍会写入 generated 下的
独立模块；这里仅提供一个不依赖外部 LLM API 的稳定基准，便于验证完整评价链路。
"""

import numpy as np


def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty,
):
    """计算 ready-task 分数，返回形状为 (N,) 的有限数组，较小者优先。

    fuzzy 模式不会改变八参数接口：环境已把 min_incremental_energy 转换为风险
    调整模糊边际能耗，把 slack 转换为 eta 风险完成时刻的余量；本规则只排序。
    """
    # epsilon 只用于数值保护，不承担调度偏好。
    eps = 1e-8
    # 批量创建 float 数组/视图，不对环境传入的八组特征做原地修改。
    # 解包后相同下标仍严格对应同一个 ready task。
    arrays = [
        np.asarray(value, dtype=float)
        for value in (
            min_exec_time,
            min_comm_time,
            min_incremental_energy,
            slack,
            upward_rank,
            remaining_work,
            ready_wait_time,
            uncertainty,
        )
    ]
    (
        min_exec_time,
        min_comm_time,
        min_incremental_energy,
        slack,
        upward_rank,
        remaining_work,
        ready_wait_time,
        uncertainty,
    ) = arrays

    def normalize(value):
        """用平均绝对值统一不同物理量的数量级，并用 epsilon 防止除零。"""
        return value / (np.mean(np.abs(value)) + eps)

    # 负 slack 的绝对值表示预计超出子截止期的风险程度。
    deadline_risk = np.maximum(-slack, 0.0)

    # slack 越小，urgency 越接近 1；slack 充足时 urgency 自动下降。
    slack_scale = np.mean(np.abs(slack)) + eps

    urgency = 1.0 / (
        1.0
        + np.maximum(slack, 0.0) / slack_scale
    )

    # 只对紧迫的高不确定任务给予提前量，
    # 避免高不确定但 slack 充足的任务无条件抢占资源。
    uncertainty_priority = (
        normalize(uncertainty) * urgency
    )
    
    # 组合规则与 seed v1 保持一致，作为可重复的评价基准。能耗项现在可接收
    # mean(fuzzy marginal energy)+weight*std，仍需和秒/MI 特征分别归一化。
    # deadline_risk、upward_rank 和 ready_wait_time 的负系数会令紧急、关键或
    # 久候任务分数更小，从任务排序层面优先保障 DDL 可行性。
    score = (
        0.25 * normalize(min_exec_time)
        + 0.15 * normalize(min_comm_time)
        + 0.25 * normalize(min_incremental_energy)
        - 2.00 * normalize(deadline_risk)
        + 0.10 * normalize(remaining_work)
        - 0.10 * uncertainty_priority
        - 0.15 * normalize(upward_rank)
        - 0.05 * normalize(ready_wait_time)
    )
    # 保证参考规则返回有限数值；环境之后还会再次严格验证形状与有限性。
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
