# -*- coding: utf-8 -*-
"""
训练评估模块。

整体调用关系：

1. train.py 调用 train_runner.train()。
2. train_runner.py 在每个 episode 结束时调用
   evaluate_hrl_three_layer_multi_seed()。
3. evaluate_hrl_three_layer_multi_seed() 使用当前训练好的
   VM、Host、Manager agent，在指定 eval_seeds 上执行确定性评估。
4. 评估结果返回给 train_runner.py，由 train_runner.py
   写入日志并判断是否保存 best checkpoint。

文件职责：

- 只负责评估，不负责训练参数解析、环境创建配置、模型保存或日志字段定义。
- 评估时使用 deterministic=True，尽量反映当前策略本身的效果，
  而不是探索噪声。
- 多 seed 结果取平均，返回 VM reward、Host reward、
  Manager reward 和总能耗。
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


# 训练 cost_budget 可以为稳定性临时设为非零，
# 但最终评估合格标准不继承训练容忍度。
FINAL_EVALUATION_VIOLATION_BUDGET = 0.0


def _completed_fuzzy_lateness_summary(
    eval_env,
):
    """Reconstruct exact fuzzy lateness from environment timelines.

    Production environments expose:
    - wf_finish_time
    - _workflow_finish_tfn
    - fuzzy_deadline_measure

    Fallback is retained only for lightweight legacy test doubles
    that expose cumulative safety counters but not fuzzy timelines.
    """
    finish_ids = sorted(
        int(workflow_id)
        for workflow_id
        in getattr(
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

    workflows = getattr(
        eval_env,
        "workflows",
        None,
    )

    if (
        finish_ids
        and callable(
            finish_getter
        )
        and callable(
            risk_measure
        )
        and workflows is not None
    ):
        values = []

        for workflow_id in finish_ids:
            finish_tfn = finish_getter(
                workflow_id
            )

            risk_finish = float(
                risk_measure(
                    finish_tfn
                )
            )

            deadline = float(
                workflows[
                    workflow_id
                ].deadline
            )

            values.append(
                max(
                    0.0,
                    risk_finish
                    - deadline,
                )
            )

        return {
            "fuzzy_lateness_sum": float(
                sum(values)
            ),
            "max_fuzzy_lateness": float(
                max(
                    values,
                    default=0.0,
                )
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

        # With no per-workflow timeline,
        # cumulative value is the only conservative
        # upper bound available.
        "max_fuzzy_lateness": cumulative,

        "fuzzy_lateness_values": [],
        "exact_fuzzy_timeline_reconstruction": False,
    }


def evaluation_ctor_kwargs(
    env_kwargs,
):
    """Remove parameters that must be synchronized after construction."""
    ctor_block = {
        "energy_reward_scale",
        "task_baseline_norm",
        "energy_norm_per_mi_ref",
        "alpha_delay_host",
        "alpha_delay_vm",
    }

    return {
        key: value
        for key, value
        in env_kwargs.items()
        if key not in ctor_block
    }


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
    """Run one deterministic evaluation episode for one random seed.

    Returns:

    (
        avg_vm,
        avg_host,
        avg_mgr,
        total_energy,
        record,
    )

    Agent inference is read-only:
    - deterministic=True disables epsilon exploration.
    - count_step=False prevents action-step counters from changing.
    """
    base_ctor_kwargs = (
        evaluation_ctor_kwargs(
            env_kwargs
        )
    )

    sd = seed

    seed_started_at = (
        time.perf_counter()
    )

    phase_metric_records = []

    # 每个 seed 创建独立环境。
    ctor_kwargs = dict(
        base_ctor_kwargs
    )

    ctor_kwargs[
        "random_seed"
    ] = int(sd)

    eval_env = env_cls(
        **ctor_kwargs
    )

    sync_env_scales(
        eval_env,
        env_kwargs,
    )

    eval_env.reset()

    # Reject old observation dimensions before taking any evaluation action.
    for layer, agent in (("host", host_agent), ("vm", vm_agent), ("manager", manager_agent)):
        expected = getattr(eval_env, f"{layer}_obs_dim", None)
        actual = getattr(agent, "input_dim", None)
        if expected is not None and actual is not None and int(actual) != int(expected):
            raise ValueError(
                f"{layer} checkpoint observation dimension mismatch: "
                f"checkpoint={actual}, environment={expected}"
            )

    # ------------------------------------------------------------
    # Initial Manager decision
    # ------------------------------------------------------------
    sH = eval_env.get_manager_state()

    m_mask = (
        eval_env
        .get_manager_action_mask()
    )

    m_act = manager_agent.select_action(
        sH,
        m_mask,
        deterministic=True,
        count_step=False,
    )

    manager_apply_action(
        eval_env,
        m_act,
    )

    done = bool(
        getattr(
            eval_env,
            "done_flag",
            False,
        )
    )

    phases = 0
    ret_mgr = 0.0
    ret_vm_phase_mean = 0.0
    ret_host_phase_mean = 0.0

    # ============================================================
    # Episode loop
    # ============================================================
    while not done:
        vm_rewards = []
        host_rewards = []

        # --------------------------------------------------------
        # One Manager phase
        # --------------------------------------------------------
        while True:
            (
                st_host,
                has_next,
            ) = (
                eval_env
                .get_host_state_for_next_assignment()
            )

            if not has_next:
                break

            (
                a_host,
                _,
                _,
            ) = select_layer_action(
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

            eval_env.host_select(
                int(
                    a_host
                )
            )

            (
                st_vm,
                ok_vm,
            ) = (
                eval_env
                .get_vm_state_for_current_task()
            )

            if not ok_vm:
                break

            (
                a_vm,
                _,
                _,
            ) = select_layer_action(
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

            (
                r_host,
                r_vm,
                info_task,
            ) = eval_env.vm_assign(
                int(
                    a_vm
                )
            )

            if getattr(
                eval_env,
                "safe_rl_enabled",
                False,
            ):
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
                host_rewards.append(
                    float(
                        r_host
                    )
                )

                vm_rewards.append(
                    float(
                        r_vm
                    )
                )

        # --------------------------------------------------------
        # Finish current Manager phase
        # --------------------------------------------------------
        (
            r_manager_raw,
            pinfo,
        ) = (
            eval_env
            .finish_phase_and_advance()
        )

        if return_safety_metrics:
            phase_metric_records.append(
                dict(
                    pinfo
                )
            )

        if getattr(
            eval_env,
            "safe_rl_enabled",
            False,
        ):
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
            ret_mgr += float(
                r_manager_raw
            )

        ret_vm_phase_mean += (
            float(
                np.mean(
                    vm_rewards
                )
            )
            if len(
                vm_rewards
            )
            > 0
            else 0.0
        )

        ret_host_phase_mean += (
            float(
                np.mean(
                    host_rewards
                )
            )
            if len(
                host_rewards
            )
            > 0
            else 0.0
        )

        phases += 1

        done = bool(
            getattr(
                eval_env,
                "done_flag",
                False,
            )
        )

        if done:
            break

        # --------------------------------------------------------
        # Manager decision for next phase
        # --------------------------------------------------------
        sH = (
            eval_env
            .get_manager_state()
        )

        m_mask = (
            eval_env
            .get_manager_action_mask()
        )

        m_act = manager_agent.select_action(
            sH,
            m_mask,
            deterministic=True,
            count_step=False,
        )

        manager_apply_action(
            eval_env,
            m_act,
        )

    # ============================================================
    # Single-seed result
    # ============================================================
    avg_vm = (
        ret_vm_phase_mean
        / max(
            phases,
            1,
        )
    )

    avg_host = (
        ret_host_phase_mean
        / max(
            phases,
            1,
        )
    )

    avg_mgr = (
        ret_mgr
        / max(
            phases,
            1,
        )
    )

    total_energy = float(
        eval_env.total_energy
    )

    record = None

    if return_safety_metrics:
        record = (
            build_episode_metric_record(
                eval_env,
                seed=int(
                    sd
                ),
                scheduling_time_seconds=(
                    time.perf_counter()
                    - seed_started_at
                ),
                phase_records=(
                    phase_metric_records
                ),
            )
        )

    return (
        avg_vm,
        avg_host,
        avg_mgr,
        total_energy,
        record,
    )


def aggregate_seed_results(
    seed_results,
    *,
    return_safety_metrics,
):
    """Aggregate ordered per-seed results.

    Serial and parallel paths both go through this function.
    Keeping seed order unchanged avoids floating-point reduction
    differences caused by non-associative arithmetic.
    """
    vm_list = [
        row[0]
        for row in seed_results
    ]

    host_list = [
        row[1]
        for row in seed_results
    ]

    mgr_list = [
        row[2]
        for row in seed_results
    ]

    energy_list = [
        row[3]
        for row in seed_results
    ]

    base_result = (
        float(
            np.mean(
                vm_list
            )
        ),
        float(
            np.mean(
                host_list
            )
        ),
        float(
            np.mean(
                mgr_list
            )
        ),
        float(
            np.mean(
                energy_list
            )
        ),
    )

    if not return_safety_metrics:
        return base_result

    safety_metrics = (
        aggregate_safe_metric_records(
            [
                row[4]
                for row
                in seed_results
            ]
        )
    )

    safety_metrics.update(
        {
            "evaluation_violation_budget": (
                FINAL_EVALUATION_VIOLATION_BUDGET
            ),
            "zero_violation_pass": bool(
                safety_metrics[
                    "fuzzy_ddl_violation_rate"
                ]
                <= (
                    FINAL_EVALUATION_VIOLATION_BUDGET
                )
            ),
        }
    )

    return (
        *base_result,
        safety_metrics,
    )


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
    seed_result_callback=None,
):
    """Evaluate the current three-layer HRL policy on multiple seeds.

    Parameters
    ----------
    seed_result_callback:
        Optional callback called after one seed result becomes available.

        Serial evaluation:
            Callback is invoked immediately after each seed finishes.

        Parallel evaluation:
            Current ValidationEvaluationPool API returns all seed results
            together, so callback is invoked after the pool returns.
    """
    seeds = tuple(
        seeds
    )

    if not seeds:
        raise ValueError(
            "multi-seed evaluation requires at least one seed"
        )

    # ============================================================
    # Serial evaluation
    #
    # protocol_evaluation.py currently uses this path.
    # Each seed is persisted immediately through the callback.
    # ============================================================
    if evaluation_pool is None:
        seed_results = []

        for sd in seeds:
            seed_result = evaluate_one_seed(
                env_cls,
                env_kwargs,
                vm_agent,
                host_agent,
                manager_agent,
                sd,
                return_safety_metrics=(
                    return_safety_metrics
                ),
            )

            seed_results.append(
                seed_result
            )

            # ----------------------------------------------------
            # Critical modification:
            # Persist immediately after this seed has completed.
            # ----------------------------------------------------
            if (
                seed_result_callback
                is not None
            ):
                seed_result_callback(
                    int(
                        sd
                    ),
                    seed_result,
                )

    # ============================================================
    # Parallel evaluation
    # ============================================================
    else:
        seed_results = (
            evaluation_pool.evaluate_seeds(
                env_kwargs,
                seeds,
                return_safety_metrics=(
                    return_safety_metrics
                ),
            )
        )

        # Current evaluation_pool returns the complete list.
        # Therefore this callback is not truly real-time in the
        # parallel branch.
        if (
            seed_result_callback
            is not None
        ):
            for (
                sd,
                seed_result,
            ) in zip(
                seeds,
                seed_results,
            ):
                seed_result_callback(
                    int(
                        sd
                    ),
                    seed_result,
                )

    return aggregate_seed_results(
        seed_results,
        return_safety_metrics=(
            return_safety_metrics
        ),
    )


# 墙钟时间在串行和并行模式下不可直接比较。
_WALL_CLOCK_METRIC_FIELDS = (
    "scheduling_time_seconds",
)


def assert_seed_results_identical(
    parallel_results,
    serial_results,
):
    """Compare parallel and serial per-seed results."""
    if (
        len(
            parallel_results
        )
        != len(
            serial_results
        )
    ):
        raise AssertionError(
            "validation parallel audit: seed count mismatch "
            f"{len(parallel_results)} "
            "!= "
            f"{len(serial_results)}"
        )

    for (
        index,
        (
            got,
            want,
        ),
    ) in enumerate(
        zip(
            parallel_results,
            serial_results,
        )
    ):
        for (
            field,
            left,
            right,
        ) in zip(
            (
                "avg_vm",
                "avg_host",
                "avg_mgr",
                "total_energy",
            ),
            got[:4],
            want[:4],
        ):
            if float(
                left
            ) != float(
                right
            ):
                raise AssertionError(
                    "validation parallel audit: seed index "
                    f"{index} field {field} differs: "
                    f"{left!r} != {right!r}"
                )

        (
            got_record,
            want_record,
        ) = (
            got[4],
            want[4],
        )

        if (
            got_record is None
        ) != (
            want_record is None
        ):
            raise AssertionError(
                "validation parallel audit: seed index "
                f"{index} record presence differs"
            )

        if got_record is None:
            continue

        keys = (
            set(
                got_record
            )
            | set(
                want_record
            )
        )

        for key in sorted(
            keys
            - set(
                _WALL_CLOCK_METRIC_FIELDS
            )
        ):
            if (
                got_record.get(
                    key
                )
                != want_record.get(
                    key
                )
            ):
                raise AssertionError(
                    "validation parallel audit: seed index "
                    f"{index} metric {key} differs: "
                    f"{got_record.get(key)!r} "
                    "!= "
                    f"{want_record.get(key)!r}"
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

    The online trainer intentionally never calls this function.

    Final-test seeds must:
    - be non-empty;
    - not overlap training seeds;
    - not overlap validation seeds.
    """
    split = {
        "training": tuple(
            int(
                seed
            )
            for seed
            in training_seeds
        ),
        "validation": tuple(
            int(
                seed
            )
            for seed
            in validation_seeds
        ),
        "final_test": tuple(
            int(
                seed
            )
            for seed
            in final_test_seeds
        ),
    }

    if not split[
        "final_test"
    ]:
        raise ValueError(
            "final_test_seeds must be non-empty"
        )

    for (
        first,
        second,
    ) in (
        (
            "training",
            "validation",
        ),
        (
            "training",
            "final_test",
        ),
        (
            "validation",
            "final_test",
        ),
    ):
        overlap = sorted(
            set(
                split[
                    first
                ]
            ).intersection(
                split[
                    second
                ]
            )
        )

        if overlap:
            raise ValueError(
                f"{first} and {second} seeds overlap: "
                f"{overlap}"
            )

    result = (
        evaluate_hrl_three_layer_multi_seed(
            env_cls,
            env_kwargs,
            vm_agent,
            host_agent,
            manager_agent,
            split[
                "final_test"
            ],
            return_safety_metrics=True,
        )
    )

    report = (
        aggregate_safe_metric_records(
            result[-1][
                "per_seed_metrics"
            ],
            q_c_prediction_error=(
                q_c_prediction_error
            ),
            q_c_prediction_error_sample_count=(
                q_c_prediction_error_sample_count
            ),
            lambda_current=(
                lambda_current
            ),
        )
    )

    report.update(
        {
            "evaluation_violation_budget": (
                FINAL_EVALUATION_VIOLATION_BUDGET
            ),
            "zero_violation_pass": bool(
                report[
                    "fuzzy_ddl_violation_rate"
                ]
                <= (
                    FINAL_EVALUATION_VIOLATION_BUDGET
                )
            ),
            "seed_split": {
                key: list(
                    value
                )
                for key, value
                in split.items()
            },
        }
    )

    store = SafeMetricStore(
        metrics_output_directory,
        convergence_window=(
            convergence_window
        ),
    )

    persisted = store.append(
        "final_test",
        report,
        global_step=None,
        episode=None,
    )

    return (
        *result[:-1],
        persisted,
    )
