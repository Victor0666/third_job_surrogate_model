# -*- coding: utf-8 -*-
"""
训练评估模块。

整体调用关系：
1. train.py 调用 train_runner.train()。
2. train_runner.py 在每个 episode 结束时调用 evaluate_hrl_three_layer_multi_seed()。
3. evaluate_hrl_three_layer_multi_seed() 使用当前训练好的 VM、Host、Manager agent，
   在指定 eval_seeds 上执行确定性评估。
4. 评估结果返回给 train_runner.py，由 train_runner.py 写入日志并判断是否保存 best checkpoint。

文件职责：
- 只负责评估，不负责训练参数解析、环境创建配置、模型保存或日志字段定义。
- 评估时使用 deterministic=True，尽量反映当前策略本身的效果，而不是探索噪声。
- 多 seed 结果取平均，返回 VM reward、Host reward、Manager reward 和总能耗。
"""
from __future__ import annotations

import time

import numpy as np

from hrl_mix.safe_metrics import (
    SafeMetricStore,
    aggregate_safe_metric_records,
    build_episode_metric_record,
)
from hrl_mix.train_utils import (
    manager_apply_action,
    select_layer_action,
    sync_env_scales,
)

# 训练 cost_budget 可以为稳定性临时设为非零，但评估合格标准不可继承该
# 容忍度。这里固定为零完成工作流 DDL 违反。
FINAL_EVALUATION_VIOLATION_BUDGET = 0.0


def _completed_fuzzy_lateness_summary(eval_env):
    """Reconstruct exact fuzzy lateness from the environment timelines.

    Production environments expose ``wf_finish_time``,
    ``_workflow_finish_tfn`` and ``fuzzy_deadline_measure``. The fallback is
    retained only for lightweight legacy test doubles that expose cumulative
    safety counters but not fuzzy timelines.
    """
    finish_ids = sorted(
        int(workflow_id)
        for workflow_id in getattr(
            eval_env,
            "wf_finish_time",
            {},
        )
    )
    finish_getter = getattr(
        eval_env,
        "_workflow_finish_tfn",
        None,
    )
    risk_measure = getattr(
        eval_env,
        "fuzzy_deadline_measure",
        None,
    )
    workflows = getattr(eval_env, "workflows", None)
    if (
        finish_ids
        and callable(finish_getter)
        and callable(risk_measure)
        and workflows is not None
    ):
        values = []
        for workflow_id in finish_ids:
            finish_tfn = finish_getter(workflow_id)
            risk_finish = float(risk_measure(finish_tfn))
            deadline = float(workflows[workflow_id].deadline)
            values.append(max(0.0, risk_finish - deadline))
        return {
            "fuzzy_lateness_sum": float(sum(values)),
            "max_fuzzy_lateness": float(
                max(values, default=0.0)
            ),
            "fuzzy_lateness_values": values,
            "exact_fuzzy_timeline_reconstruction": True,
        }

    cumulative = max(
        0.0,
        float(
            getattr(
                eval_env,
                "_safety_cumulative_fuzzy_lateness_cost",
                0.0,
            )
        ),
    )
    return {
        "fuzzy_lateness_sum": cumulative,
        # With no per-workflow timeline the cumulative value is the only
        # conservative upper bound available to a compatibility test double.
        "max_fuzzy_lateness": cumulative,
        "fuzzy_lateness_values": [],
        "exact_fuzzy_timeline_reconstruction": False,
    }


def evaluation_ctor_kwargs(env_kwargs):
    """去掉需要在环境建好后手动同步的尺度参数，得到构造函数入参。

    这些键不是环境构造函数的形参，必须由 :func:`sync_env_scales` 事后写入。
    并行 worker 也要按同一口径拆分，因此单独提出来共用。
    """
    ctor_block = {
        "energy_reward_scale",
        "task_baseline_norm",
        "energy_norm_per_mi_ref",
        "alpha_delay_host",
        "alpha_delay_vm",
    }
    return {k: v for k, v in env_kwargs.items() if k not in ctor_block}


def evaluate_one_seed(
    env_cls,
    env_kwargs,
    vm_agent,
    host_agent,
    manager_agent,
    seed,
    *,
    return_safety_metrics=False,
):
    """在单个随机种子上跑完一个确定性评估 episode。

    这是多 seed 评估的最小工作单元，也是进程级并行的作业体。返回
    ``(avg_vm, avg_host, avg_mgr, total_energy, record)``，其中 ``record``
    只在 ``return_safety_metrics`` 为真时非空。

    该函数对三个 agent 只做只读推理：``deterministic=True`` 关掉了 epsilon
    分支，``count_step=False`` 关掉了步数计数，因此它不修改任何 agent 状态，
    多个 seed 之间也没有共享可变状态——并行执行与串行执行逐位一致。
    """
    base_ctor_kwargs = evaluation_ctor_kwargs(env_kwargs)
    sd = seed
    seed_started_at = time.perf_counter()
    phase_metric_records = []
    # 每个 seed 创建一个独立评估环境，避免评估过程互相污染状态。
    ctor_kwargs = dict(base_ctor_kwargs)
    ctor_kwargs["random_seed"] = int(sd)

    eval_env = env_cls(**ctor_kwargs)
    sync_env_scales(eval_env, env_kwargs)

    eval_env.reset()

    # 评估时 manager 先选择一个阶段动作，后续进入任务分配循环。
    sH = eval_env.get_manager_state()
    m_mask = eval_env.get_manager_action_mask()
    m_act = manager_agent.select_action(sH, m_mask, deterministic=True, count_step=False)
    manager_apply_action(eval_env, m_act)

    done = bool(getattr(eval_env, "done_flag", False))
    phases = 0
    ret_mgr = 0.0
    ret_vm_phase_mean = 0.0
    ret_host_phase_mean = 0.0

    while not done:
        vm_rewards = []
        host_rewards = []

        # 一个 phase 内不断让 HostAgent 选 host，再让 VMAgent 选 VM 槽位。
        while True:
            st_host, has_next = eval_env.get_host_state_for_next_assignment()
            if not has_next:
                break

            a_host, _, _ = select_layer_action(
                host_agent,
                st_host,
                safe_rl_enabled=bool(
                    getattr(
                        eval_env,
                        "safe_rl_enabled",
                        False,
                    )
                ),
                deterministic=True,
                count_step=False,
            )
            eval_env.host_select(int(a_host))

            st_vm, ok_vm = eval_env.get_vm_state_for_current_task()
            if not ok_vm:
                break

            a_vm, _, _ = select_layer_action(
                vm_agent,
                st_vm,
                safe_rl_enabled=bool(
                    getattr(
                        eval_env,
                        "safe_rl_enabled",
                        False,
                    )
                ),
                deterministic=True,
                count_step=False,
            )
            r_host, r_vm, info_task = eval_env.vm_assign(int(a_vm))

            if getattr(eval_env, "safe_rl_enabled", False):
                host_rewards.append(
                    float(
                        info_task.get(
                            "total_performance_reward",
                            info_task.get(
                                "performance_reward_host",
                                r_host,
                            ),
                        )
                    )
                )
                vm_rewards.append(
                    float(
                        info_task.get(
                            "total_performance_reward",
                            info_task.get(
                                "performance_reward_vm",
                                r_vm,
                            ),
                        )
                    )
                )
            else:
                host_rewards.append(float(r_host))
                vm_rewards.append(float(r_vm))

        # phase 结束后由环境推进时间，并得到 manager 层 reward。
        r_manager_raw, pinfo = eval_env.finish_phase_and_advance()
        if return_safety_metrics:
            phase_metric_records.append(dict(pinfo))
        if getattr(eval_env, "safe_rl_enabled", False):
            ret_mgr += float(
                pinfo.get(
                    "total_performance_reward",
                    pinfo.get(
                        "performance_reward",
                        r_manager_raw,
                    ),
                )
            )
        else:
            ret_mgr += float(r_manager_raw)
        ret_vm_phase_mean += float(np.mean(vm_rewards)) if len(vm_rewards) > 0 else 0.0
        ret_host_phase_mean += float(np.mean(host_rewards)) if len(host_rewards) > 0 else 0.0
        phases += 1

        done = bool(getattr(eval_env, "done_flag", False))
        if done:
            break

        # 如果 episode 未结束，manager 为下一 phase 选择新动作。
        sH = eval_env.get_manager_state()
        m_mask = eval_env.get_manager_action_mask()
        m_act = manager_agent.select_action(sH, m_mask, deterministic=True, count_step=False)
        manager_apply_action(eval_env, m_act)

    avg_vm = ret_vm_phase_mean / max(phases, 1)
    avg_host = ret_host_phase_mean / max(phases, 1)
    avg_mgr = ret_mgr / max(phases, 1)
    total_energy = float(eval_env.total_energy)

    record = None
    if return_safety_metrics:
        record = build_episode_metric_record(
            eval_env,
            seed=int(sd),
            scheduling_time_seconds=(
                time.perf_counter() - seed_started_at
            ),
            phase_records=phase_metric_records,
        )
    return (avg_vm, avg_host, avg_mgr, total_energy, record)


def aggregate_seed_results(seed_results, *, return_safety_metrics):
    """把按 seed 顺序排好的单 seed 结果汇总成对外返回值。

    串行与并行两条路径都只经过这里，聚合顺序完全由入参顺序决定，所以两者
    逐位一致——浮点加法不满足结合律，``np.mean`` 的入参顺序不能变。
    """
    vm_list = [row[0] for row in seed_results]
    host_list = [row[1] for row in seed_results]
    mgr_list = [row[2] for row in seed_results]
    energy_list = [row[3] for row in seed_results]

    base_result = (
        float(np.mean(vm_list)),
        float(np.mean(host_list)),
        float(np.mean(mgr_list)),
        float(np.mean(energy_list)),
    )
    if not return_safety_metrics:
        return base_result

    safety_metrics = aggregate_safe_metric_records(
        [row[4] for row in seed_results]
    )
    safety_metrics.update(
        {
            "evaluation_violation_budget": (
                FINAL_EVALUATION_VIOLATION_BUDGET
            ),
            "zero_violation_pass": bool(
                safety_metrics["fuzzy_ddl_violation_rate"]
                <= FINAL_EVALUATION_VIOLATION_BUDGET
            ),
        }
    )
    return (*base_result, safety_metrics)


def evaluate_hrl_three_layer_multi_seed(
    env_cls,
    env_kwargs,
    vm_agent,
    host_agent,
    manager_agent,
    seeds,
    *,
    return_safety_metrics=False,
    evaluation_pool=None,
):
    """在多个随机种子上评估当前三层 HRL 策略。

    参数：
    - env_cls：环境类，一般为 CloudWorkflowEnv_VMAgents。
    - env_kwargs：创建环境所需参数，同时包含部分需要手动同步的尺度参数。
    - vm_agent、host_agent、manager_agent：当前训练中的三个智能体。
    - seeds：评估使用的随机种子列表。
    - evaluation_pool：可选的
      :class:`hrl_mix.validation_parallel.ValidationEvaluationPool`。给了就把
      各 seed 的 episode 分发到常驻工作进程，返回值仍按 seed 顺序聚合。

    返回：
    - eval_vm：各评估 seed 上 VM 层平均 reward 的均值。
    - eval_host：各评估 seed 上 Host 层平均 reward 的均值。
    - eval_mgr：各评估 seed 上 Manager 层平均 reward 的均值。
    - eval_energy：各评估 seed 上总能耗的均值。
    - return_safety_metrics=True 时额外返回安全字典；其中合格线固定为
      deadline violation rate == 0，不继承训练 cost_budget。
    """
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError(
            "multi-seed evaluation requires at least one seed"
        )

    if evaluation_pool is None:
        seed_results = [
            evaluate_one_seed(
                env_cls,
                env_kwargs,
                vm_agent,
                host_agent,
                manager_agent,
                sd,
                return_safety_metrics=return_safety_metrics,
            )
            for sd in seeds
        ]
    else:
        # 逐位审计由池内部负责，两个调用点共用同一份实现。
        seed_results = evaluation_pool.evaluate_seeds(
            env_kwargs,
            seeds,
            return_safety_metrics=return_safety_metrics,
        )

    return aggregate_seed_results(
        seed_results,
        return_safety_metrics=return_safety_metrics,
    )


#: 墙钟类字段在串行/并行之间本就不可比，审计时按名字剔除。
_WALL_CLOCK_METRIC_FIELDS = ("scheduling_time_seconds",)


def assert_seed_results_identical(parallel_results, serial_results):
    """逐位比对并行与串行的单 seed 结果，供审计开关使用。"""
    if len(parallel_results) != len(serial_results):
        raise AssertionError(
            "validation parallel audit: seed count mismatch "
            f"{len(parallel_results)} != {len(serial_results)}"
        )
    for index, (got, want) in enumerate(
        zip(parallel_results, serial_results)
    ):
        for field, left, right in zip(
            ("avg_vm", "avg_host", "avg_mgr", "total_energy"),
            got[:4],
            want[:4],
        ):
            if float(left) != float(right):
                raise AssertionError(
                    "validation parallel audit: seed index "
                    f"{index} field {field} differs: {left!r} != {right!r}"
                )
        got_record, want_record = got[4], want[4]
        if (got_record is None) != (want_record is None):
            raise AssertionError(
                "validation parallel audit: seed index "
                f"{index} record presence differs"
            )
        if got_record is None:
            continue
        keys = set(got_record) | set(want_record)
        for key in sorted(keys - set(_WALL_CLOCK_METRIC_FIELDS)):
            if got_record.get(key) != want_record.get(key):
                raise AssertionError(
                    "validation parallel audit: seed index "
                    f"{index} metric {key} differs: "
                    f"{got_record.get(key)!r} != {want_record.get(key)!r}"
                )


def evaluate_and_save_safe_hrl_final_test(
    env_cls,
    env_kwargs,
    vm_agent,
    host_agent,
    manager_agent,
    *,
    training_seeds,
    validation_seeds,
    final_test_seeds,
    metrics_output_directory,
    convergence_window=5,
    q_c_prediction_error=0.0,
    q_c_prediction_error_sample_count=0,
    lambda_current=0.0,
):
    """Run an explicit withheld-seed final test and persist its report.

    The online trainer intentionally never calls this function. Final-test
    seeds must be non-empty and disjoint from both training and validation.
    """

    split = {
        "training": tuple(int(seed) for seed in training_seeds),
        "validation": tuple(int(seed) for seed in validation_seeds),
        "final_test": tuple(int(seed) for seed in final_test_seeds),
    }
    if not split["final_test"]:
        raise ValueError("final_test_seeds must be non-empty")
    for first, second in (
        ("training", "validation"),
        ("training", "final_test"),
        ("validation", "final_test"),
    ):
        overlap = sorted(
            set(split[first]).intersection(split[second])
        )
        if overlap:
            raise ValueError(
                f"{first} and {second} seeds overlap: {overlap}"
            )

    result = evaluate_hrl_three_layer_multi_seed(
        env_cls,
        env_kwargs,
        vm_agent,
        host_agent,
        manager_agent,
        split["final_test"],
        return_safety_metrics=True,
    )
    report = aggregate_safe_metric_records(
        result[-1]["per_seed_metrics"],
        q_c_prediction_error=q_c_prediction_error,
        q_c_prediction_error_sample_count=(
            q_c_prediction_error_sample_count
        ),
        lambda_current=lambda_current,
    )
    report.update(
        {
            "evaluation_violation_budget": (
                FINAL_EVALUATION_VIOLATION_BUDGET
            ),
            "zero_violation_pass": bool(
                report["fuzzy_ddl_violation_rate"]
                <= FINAL_EVALUATION_VIOLATION_BUDGET
            ),
            "seed_split": {
                key: list(value)
                for key, value in split.items()
            },
        }
    )
    store = SafeMetricStore(
        metrics_output_directory,
        convergence_window=convergence_window,
    )
    persisted = store.append(
        "final_test",
        report,
        global_step=None,
        episode=None,
    )
    return (*result[:-1], persisted)
