# -*- coding: utf-8 -*-
"""
train_runner.py 训练主程序模块。

整体调用关系：
1. train.py 解析命令行参数。
2. train.py 调用本文件的 train(scenario, ddl, max_episodes)。
3. train() 先调用 train_config.build_train_config() 生成完整配置。
4. train() 使用 train_utils.py 中的工具函数完成种子设置、环境尺度同步、动作应用和指标计算。
5. train() 在每个 episode 结束时调用 train_eval.evaluate_hrl_three_layer_multi_seed() 进行评估。
6. safe 模式按模糊 DDL 可行性优先键保存 best checkpoint；旧模式仍按评估
   能耗保存，并在训练结束时保存 final checkpoint。

文件职责：
- 创建训练环境 CloudWorkflowEnv_VMAgents。
- 推断 HostAgent、VMAgent、ManagerAgent 的状态维度和动作维度。
- 创建三层 D3QNAgent。
- 执行三层 HRL 训练循环：
  Manager 选择 phase 级动作，HostAgent 选择 host，VMAgent 选择 VM 槽位。
- 写入 phase 和 episode 两类训练日志。
- 保存 best / step / final 模型。

注意：
- 本文件是训练流程的“编排层”，不再直接存放大段配置常量。
- 超参数和路径由 train_config.py 生成，基础工具函数由 train_utils.py 提供。
"""

import json
import os
import time

import numpy as np
import torch

from base.d3qn_agent import D3QNAgent
from base.hrl_env import CloudWorkflowEnv_VMAgents
from base.offline_pretraining import (
    pretrain_agents_from_demonstrations,
)
from base.safety_lagrange import (
    LagrangeSafetyController,
    synchronize_lagrange_multiplier,
)
from common.metrics_logger import CSVLogger
from hrl_mix.model_selection import (
    FeasibilityFirstModelMetrics,
    build_config_snapshot,
    build_heuristic_library_version,
    build_replay_metadata,
    is_better_model,
    save_best_checkpoint_bundle,
)
from hrl_mix.safe_metrics import (
    SafeMetricStore,
    aggregate_safe_metric_records,
    build_episode_metric_record,
)
from hrl_mix.train_config import (
    build_train_config,
    environment_scenario_values,
)
from hrl_mix.train_eval import (
    aggregate_seed_results,
    evaluate_hrl_three_layer_multi_seed,
)
from hrl_mix.validation_parallel import (
    ValidationEvaluationPool,
    resolve_worker_count as resolve_validation_workers,
)
from hrl_mix.safe_training_pipeline import (
    SafeStageMetricsLogger,
    SafeTrainingController,
    StageMetrics,
    apply_curriculum_to_env_kwargs,
    load_safe_training_plan,
    q_c_prediction_error_from_agents,
    read_pipeline_checkpoint,
    restore_pipeline_agents,
    save_pipeline_checkpoint,
    validate_preparation_artifacts,
)
from hrl_mix.train_utils import (
    apply_env_scales,
    compute_episode_task_lateness_metrics,
    finalize_action_audit,
    manager_apply_action,
    manager_action_dim,
    performance_reward_components,
    print_device_info,
    layer_learning_action_mask,
    safe_replay_metadata,
    select_layer_action_with_info,
    set_seed,
    warmup_ready,
)


def _commit_safe_pending_transition(
    agent,
    pending,
    *,
    next_state,
    next_mask,
    done,
    warmup_frac,
):
    """写入一个已经获得同层下一决策状态的安全 transition。"""
    if pending is None:
        return
    agent.remember(
        pending["state"],
        pending["mask"],
        pending["action"],
        pending["reward"],
        next_state,
        next_mask,
        float(done),
        cost=pending["cost"],
        proposed_action=(
            pending["proposed_action"]
            if "proposed_action" in pending
            else pending["action"]
        ),
        action_source=pending.get("action_source"),
        policy_selection_type=pending.get(
            "policy_selection_type"
        ),
        action_modified=bool(
            pending.get("action_modified", False)
        ),
        legal_action_mask=pending.get(
            "legal_action_mask"
        ),
        safety_action_mask=pending.get(
            "safety_action_mask"
        ),
        fallback_triggered=bool(
            pending.get("fallback_triggered", False)
        ),
        fuzzy_safety_margin=float(
            pending.get("fuzzy_safety_margin", 0.0)
        ),
        predicted_risk_finish=float(
            pending.get("predicted_risk_finish", 0.0)
        ),
        violation_flag=bool(
            pending.get("violation_flag", False)
        ),
        manager_phase_id=int(
            pending.get("manager_phase_id", 0)
        ),
        performance_reward_components=pending.get(
            "performance_reward_components"
        ),
    )
    if warmup_ready(agent, warmup_frac):
        agent.update()


def _select_manager_training_action(
    agent,
    state,
    mask,
    *,
    safe_rl_enabled,
):
    """Manager 保持原职责，仅在安全模式记录受约束 epsilon-greedy 类型。"""
    if safe_rl_enabled:
        selection = agent.select_action_with_info(
            state,
            mask,
            deterministic=False,
            count_step=True,
        )
        action = int(selection["action"])
        return action, finalize_action_audit(selection, None)
    action = agent.select_action(
        state,
        mask,
        deterministic=False,
        count_step=True,
    )
    return int(action), None


def _save_agent_checkpoint(
    agent,
    path,
    lagrange_controller=None,
):
    """保存 Agent；动态模式同时嵌入共享控制器状态。"""
    controller_state = (
        lagrange_controller.state_dict()
        if (
            lagrange_controller is not None
            and lagrange_controller.enabled
        )
        else None
    )
    agent.save(
        path,
        lagrange_controller_state=controller_state,
    )


def _checkpoint_curriculum_state(controller):
    """Return explicit curriculum state even without a staged pipeline."""
    if controller is None:
        return {
            "enabled": False,
            "stage_index": None,
            "stage_id": "single_stage_online_training",
            "stage_type": "shield_online_training",
        }
    return {
        "enabled": True,
        "stage_index": int(controller.current_stage_index),
        "stage_id": controller.current_stage.stage_id,
        "stage_type": controller.current_stage.stage_type,
        "controller_state": controller.state_dict(),
    }


def _skipped_validation_result(skipped_energy, *, return_safety_metrics):
    """构造非验证 episode 的评估占位值。

    元数必须与 :func:`_evaluate_training_scenarios` 在同一 ``return_safety_metrics``
    下的返回元数一致：训练循环用同一个布尔量决定请求、占位与解包三处，任何
    一处分叉都会在某些开关组合下抛 ``too many values to unpack``。
    """
    skipped_energy = float(skipped_energy)
    base = (0.0, 0.0, 0.0, skipped_energy)
    if not return_safety_metrics:
        return base
    return (
        *base,
        {
            "deadline_violation_rate": 0.0,
            "zero_violation_pass": False,
            "fuzzy_energy_score": skipped_energy,
            "safety_cost": 0.0,
            "shield_intervention_rate": 0.0,
            "fallback_rate": 0.0,
            "max_fuzzy_lateness": 0.0,
            "mean_fuzzy_lateness": 0.0,
            "all_seed_feasible": False,
            "all_seed_evaluation_completed": False,
            "completed_evaluation_seed_rate": 0.0,
            "feasible_seed_rate": 0.0,
            "worst_seed_violation": 0.0,
            "worst_seed_lateness": 0.0,
            "validation_seed_count": 0,
        },
    )


def _checkpoint_runtime_metadata(
    *,
    cfg,
    env,
    agents,
    training_controller,
):
    """Build the non-network state bound to a checkpoint manifest."""
    return {
        "curriculum_state": _checkpoint_curriculum_state(
            training_controller
        ),
        "replay_metadata": {
            layer: build_replay_metadata(agent)
            for layer, agent in agents.items()
        },
        "heuristic_library_version": (
            build_heuristic_library_version(
                env,
                manifest_path=(
                    cfg.safe_rl.manager_heuristics
                    .library_manifest_path
                ),
            )
        ),
        "experiment_protocol": (
            None
            if cfg.experiment_protocol is None
            else dict(cfg.experiment_protocol)
        ),
        "config_snapshot": build_config_snapshot(cfg),
    }


def _update_shared_lagrange_at_episode_end(
    controller,
    env,
    agents,
):
    """用环境 episode cost 均值更新一次共享 lambda 并同步三层。"""
    diagnostics = controller.observe_episode(
        episode_safety_cost=float(
            getattr(env, "_safety_cumulative_cost", 0.0)
        ),
        safety_transition_count=int(
            getattr(
                env,
                "_safety_cumulative_transition_count",
                0,
            )
        ),
    )
    synchronize_lagrange_multiplier(controller, agents)
    return diagnostics


_ENV_SCALE_KEYS = {
    "energy_reward_scale",
    "task_baseline_norm",
    "energy_norm_per_mi_ref",
    "alpha_delay_host",
    "alpha_delay_vm",
}


def _environment_ctor_kwargs(env_kwargs):
    """Remove values synchronized after construction."""
    return {
        key: value
        for key, value in env_kwargs.items()
        if key not in _ENV_SCALE_KEYS
    }


def _sync_env_kwargs_scales(env_kwargs, env):
    """Copy post-construction reward scales into evaluation kwargs."""
    env_kwargs.update(
        {
            "energy_reward_scale": env.energy_reward_scale,
            "task_baseline_norm": env.task_baseline_norm,
            "energy_norm_per_mi_ref": env.energy_norm_per_mi_ref,
            "alpha_delay_host": env.alpha_delay_host,
            "alpha_delay_vm": env.alpha_delay_vm,
        }
    )


def scenario_for_training_episode(cfg, episode_index):
    '''Return the deterministic protocol scenario for one episode.'''
    index = int(episode_index)
    if index < 0:
        raise ValueError('episode_index must be non-negative')
    scenarios = tuple(str(value).upper() for value in cfg.training_scenarios)
    if not scenarios:
        raise ValueError('training_scenarios must be non-empty')
    protocol = str(cfg.protocol).strip().lower()
    if protocol == 'single':
        source = str(cfg.source_scenario).strip().upper()
        if scenarios != (source,):
            raise ValueError(
                'Single Safe-HRL training requires exactly the source '
                'scenario'
            )
        return source
    if protocol == 'multi':
        return scenarios[index % len(scenarios)]
    if protocol == 'legacy' and len(scenarios) == 1:
        return scenarios[0]
    raise ValueError(f'unsupported Safe-HRL protocol: {cfg.protocol!r}')


def _seed_for_training_episode(cfg, episode_index):
    '''Cover the scenario/seed Cartesian product deterministically.'''
    index = int(episode_index)
    seeds = tuple(int(value) for value in cfg.train_seeds)
    scenario_count = len(tuple(cfg.training_scenarios))
    if index < 0 or not seeds or scenario_count < 1:
        raise ValueError(
            'episode_index must be non-negative and train_seeds non-empty'
        )
    return seeds[(index // scenario_count) % len(seeds)]


def _scenario_env_kwargs(base, cfg, scenario, seed):
    '''Build real task/resource/deadline inputs for one episode.'''
    scenario_id = str(scenario).strip().upper()
    allowed = tuple(
        str(value).strip().upper()
        for value in cfg.training_scenarios
    )
    if scenario_id not in allowed:
        raise ValueError(
            f'scenario {scenario_id} is outside training_scenarios {allowed}'
        )
    values = environment_scenario_values(scenario_id)
    cache_paths = getattr(cfg, 'deadline_cache_paths', {}) or {}
    deadline_cache_path = cache_paths.get(scenario_id)
    if deadline_cache_path is None:
        source_scenario = getattr(cfg, 'source_scenario', None)
        if source_scenario is None:
            source_scenario = getattr(cfg, 'scenario', None)
        if (
            source_scenario is not None
            and str(source_scenario).strip().upper() == scenario_id
        ):
            deadline_cache_path = getattr(cfg, 'deadline_cache_path', None)
    if deadline_cache_path is None:
        deadline_cache_path = values['deadline_cache_path']
    result = dict(base)
    result.update({
        'dax_paths': list(values['dax_list']),
        'num_cloud_hosts': values['num_cloud_hosts'],
        'num_edge_hosts': values['num_edge_hosts'],
        'cloud_vms_per_host': values['cloud_vms_per_host'],
        'edge_vms_per_host': values['edge_vms_per_host'],
        'cloud_pc_tiers': values['cloud_pc_tiers'],
        'edge_pc_tiers': values['edge_pc_tiers'],
        'cloud_bw_tiers': values['cloud_bw_tiers'],
        'edge_bw_tiers': values['edge_bw_tiers'],
        'deadline_cache_path': deadline_cache_path,
        'scenario_code': scenario_id,
        'task_code': values['task_code'],
        'resource_code': values['resource_code'],
        'workflow_families': values['workflow_families'],
        'random_seed': int(seed),
    })
    return result


def _evaluate_training_scenarios(
    env_cls,
    *,
    base_env_kwargs,
    active_env_kwargs,
    cfg,
    controller,
    vm_agent,
    host_agent,
    manager_agent,
    seeds,
    return_safety_metrics,
    scenarios=None,
    evaluation_pool=None,
):
    '''Evaluate an explicit scenario set using the shared validation metrics.'''
    scenarios = tuple(cfg.training_scenarios if scenarios is None else scenarios)
    if not scenarios:
        raise ValueError('validation scenarios must be non-empty')
    if len(scenarios) == 1:
        return evaluate_hrl_three_layer_multi_seed(
            env_cls,
            dict(active_env_kwargs),
            vm_agent,
            host_agent,
            manager_agent,
            seeds=seeds,
            return_safety_metrics=return_safety_metrics,
            evaluation_pool=evaluation_pool,
        )

    scenario_kwargs = []
    for scenario in scenarios:
        kwargs = _scenario_env_kwargs(
            base_env_kwargs,
            cfg,
            scenario,
            seed=cfg.validation_seeds[0],
        )
        if controller is not None and cfg.curriculum_enabled:
            kwargs = apply_curriculum_to_env_kwargs(
                kwargs,
                controller.current_stage.curriculum,
                training_seed=cfg.validation_seeds[0],
            )
        for key in _ENV_SCALE_KEYS:
            if key in active_env_kwargs:
                kwargs[key] = active_env_kwargs[key]
        scenario_kwargs.append((str(scenario), kwargs))

    seeds = tuple(seeds)
    if evaluation_pool is None:
        results = [
            (
                scenario,
                evaluate_hrl_three_layer_multi_seed(
                    env_cls,
                    kwargs,
                    vm_agent,
                    host_agent,
                    manager_agent,
                    seeds=seeds,
                    return_safety_metrics=return_safety_metrics,
                ),
            )
            for scenario, kwargs in scenario_kwargs
        ]
    else:
        # 场景与 seed 一起摊平成一张作业表再分发。只按 seed 并行的话，场景
        # 之间仍是串行的，课程学习下就浪费掉大半 worker。
        jobs = [
            (kwargs, seed)
            for _, kwargs in scenario_kwargs
            for seed in seeds
        ]
        flat = evaluation_pool.evaluate_jobs(
            jobs,
            return_safety_metrics=return_safety_metrics,
        )
        results = []
        for position, (scenario, _) in enumerate(scenario_kwargs):
            offset = position * len(seeds)
            results.append(
                (
                    scenario,
                    aggregate_seed_results(
                        flat[offset:offset + len(seeds)],
                        return_safety_metrics=return_safety_metrics,
                    ),
                )
            )

    means = tuple(
        float(np.mean([result[index] for _, result in results]))
        for index in range(4)
    )
    if not return_safety_metrics:
        return means

    records = []
    for scenario, result in results:
        for record in result[4]['per_seed_metrics']:
            row = dict(record)
            row['scenario_id'] = scenario
            records.append(row)
    safety = aggregate_safe_metric_records(records)
    violation_budget = float(
        results[0][1][4].get('evaluation_violation_budget', 0.0)
    )
    safety.update({
        'evaluation_violation_budget': violation_budget,
        'zero_violation_pass': bool(
            safety['fuzzy_ddl_violation_rate'] <= violation_budget
        ),
        'evaluation_scenarios': list(scenarios),
    })
    return (*means, safety)


def _probe_environment_dimensions(env, cfg):
    """Probe all three layers, then restore a clean episode."""
    st_host, ok = env.get_host_state_for_next_assignment()
    while (not ok) and (not env.done_flag):
        _r_manager, _ = env.finish_phase_and_advance()
        st_host, ok = env.get_host_state_for_next_assignment()
    if not ok:
        raise RuntimeError(
            "environment dimension probe found no schedulable task"
        )
    host_state_dim = int(st_host["obs"].shape[0])
    host_act_dim = int(st_host["mask"].shape[0])
    a_host0 = (
        int(np.argmax(st_host["mask"]))
        if np.sum(st_host["mask"]) > 0
        else 0
    )
    env.host_select(a_host0)
    st_vm, ok_vm = env.get_vm_state_for_current_task()
    if not ok_vm:
        raise RuntimeError(
            "environment dimension probe found no VM state"
        )
    dimensions = {
        "host_state_dim": host_state_dim,
        "host_action_dim": host_act_dim,
        "vm_state_dim": int(st_vm["obs"].shape[0]),
        "vm_action_dim": int(st_vm["mask"].shape[0]),
        "manager_state_dim": int(
            env.get_manager_state().shape[0]
        ),
        "manager_action_dim": int(manager_action_dim(env)),
    }
    env.reset()
    apply_env_scales(env, cfg)
    return dimensions


def _construct_pipeline_environment(
    env_cls,
    *,
    base_env_kwargs,
    controller,
    cfg,
):
    episode_index = controller.total_episode_count
    training_seed = (
        controller.training_seed_for_next_episode()
        if cfg.experiment_protocol is None
        else _seed_for_training_episode(cfg, episode_index)
    )
    scenario_kwargs = _scenario_env_kwargs(
        base_env_kwargs,
        cfg,
        scenario_for_training_episode(cfg, episode_index),
        seed=training_seed,
    )
    active_kwargs = (
        apply_curriculum_to_env_kwargs(
            scenario_kwargs,
            controller.current_stage.curriculum,
            training_seed=training_seed,
        )
        if cfg.curriculum_enabled
        else scenario_kwargs
    )
    env = env_cls(**_environment_ctor_kwargs(active_kwargs))
    env.reset()
    apply_env_scales(env, cfg)
    _sync_env_kwargs_scales(active_kwargs, env)
    return env, active_kwargs


def _validate_pipeline_protocol_seed_split(cfg, training_plan):
    '''Reject pipeline plans that bypass the experiment protocol seeds.'''
    if cfg.experiment_protocol is None:
        return
    actual_training = tuple(training_plan.seed_split.training)
    actual_validation = tuple(training_plan.seed_split.validation)
    if actual_training != tuple(cfg.train_seeds):
        raise ValueError(
            'safe training plan training seeds do not match protocol '
            f'train_seeds: {actual_training} != {tuple(cfg.train_seeds)}'
        )
    if actual_validation != tuple(cfg.validation_seeds):
        raise ValueError(
            'safe training plan validation seeds do not match protocol '
            f'validation_seeds: {actual_validation} != '
            f'{tuple(cfg.validation_seeds)}'
        )


def train(
    scenario: str | None = None,
    ddl: str = "T",
    max_episodes: int | None = None,
    safe_rl_enabled: bool = False,
    safe_rl_shield_enabled: bool = False,
    safe_rl_state_enabled: bool = False,
    safe_rl_dynamic_lambda_enabled: bool = False,
    safe_rl_heuristic_manager_enabled: bool = False,
    manager_heuristic_llm_only: bool = False,
    manager_heuristic_manifest: str | None = None,
    llm_run_manifest: str | None = None,
    safe_rl_offline_pretrain_manifest: str | None = None,
    safe_rl_offline_pretrain_epochs: int = 5,
    safe_rl_offline_pretrain_behavior_cloning: bool = False,
    safe_rl_offline_pretrain_q_r: bool = True,
    safe_rl_offline_pretrain_q_c: bool = True,
    safe_rl_training_pipeline_plan: str | None = None,
    safe_rl_training_resume_checkpoint: str | None = None,
    safe_rl_curriculum_enabled: bool = True,
    optimizer_seed: int = 0,
    deadline_cache_override: str | None = None,
    deadline_cache_paths: dict[str, str] | None = None,
    protocol: str | None = None,
    source_scenario: str | None = None,
    resource_scale: str | None = None,
    validation_workers: int = 1,
):
    """执行一次完整训练

    参数：
    - scenario：两位场景代码，例如 SS 表示 small task + small resource
    - ddl：deadline 条件，例如 T/M/L 或 Tight/Medium/Loose
    - max_episodes：可选训练轮数；为空时使用配置默认值
    - validation_workers：验证评估的工作进程数。``1``（默认）完全不建进程
      池，与改动前逐字相同；``0`` 表示自动取 ``cpu_count - 2``。worker 与父
      进程同设备，返回值按 seed 顺序聚合，因此与串行逐位一致。
    """
    # 统一从配置模块生成所有训练参数，避免主循环中散落大量局部超参数
    cfg = build_train_config(
        scenario=scenario,
        ddl=ddl,
        max_episodes=max_episodes,
        safe_rl_enabled=safe_rl_enabled,
        safe_rl_shield_enabled=safe_rl_shield_enabled,
        safe_rl_state_enabled=safe_rl_state_enabled,
        safe_rl_dynamic_lambda_enabled=(
            safe_rl_dynamic_lambda_enabled
        ),
        safe_rl_heuristic_manager_enabled=(
            safe_rl_heuristic_manager_enabled
        ),
        manager_heuristic_llm_only=(
            manager_heuristic_llm_only
        ),
        manager_heuristic_manifest=(
            manager_heuristic_manifest
        ),
        llm_run_manifest=llm_run_manifest,
        safe_rl_offline_pretrain_manifest=(
            safe_rl_offline_pretrain_manifest
        ),
        safe_rl_offline_pretrain_epochs=(
            safe_rl_offline_pretrain_epochs
        ),
        safe_rl_offline_pretrain_behavior_cloning=(
            safe_rl_offline_pretrain_behavior_cloning
        ),
        safe_rl_offline_pretrain_q_r=(
            safe_rl_offline_pretrain_q_r
        ),
        safe_rl_offline_pretrain_q_c=(
            safe_rl_offline_pretrain_q_c
        ),
        safe_rl_training_pipeline_plan=(
            safe_rl_training_pipeline_plan
        ),
        safe_rl_training_resume_checkpoint=(
            safe_rl_training_resume_checkpoint
        ),
        safe_rl_curriculum_enabled=safe_rl_curriculum_enabled,
        optimizer_seed=optimizer_seed,
        deadline_cache_override=deadline_cache_override,
        deadline_cache_paths=deadline_cache_paths,
        protocol=protocol,
        source_scenario=source_scenario,
        resource_scale=resource_scale,
    )

    print(
        f"[deadline cache] active scenario={cfg.scenario} "
        f"path={cfg.deadline_cache_path}"
    )
    if cfg.deadline_cache_paths:
        print(
            "[deadline cache] scenario mappings: "
            + ", ".join(
                f"{scenario_id}={path}"
                for scenario_id, path in sorted(
                    cfg.deadline_cache_paths.items()
                )
            )
        )

    training_plan = None
    training_controller = None
    training_resume_payload = None
    stage_metrics_logger = None
    if cfg.safe_rl.training_pipeline.enabled:
        training_plan = load_safe_training_plan(
            cfg.safe_rl.training_pipeline.plan_path
        )
        _validate_pipeline_protocol_seed_split(cfg, training_plan)
        training_controller = SafeTrainingController(
            training_plan
        )
        resume_path = (
            cfg.safe_rl.training_pipeline.resume_checkpoint_path
        )
        if resume_path:
            training_resume_payload = read_pipeline_checkpoint(
                resume_path,
                controller=training_controller,
                expected_protocol_identity=cfg.experiment_protocol,
            )
            if training_controller.completed:
                raise ValueError(
                    "safe training checkpoint already completed the "
                    "configured online pipeline; use it for evaluation "
                    "instead of resuming training"
                )
        else:
            validate_preparation_artifacts(training_plan)
        stage_metrics_logger = SafeStageMetricsLogger(
            training_plan.metrics_path
        )

    lagrange_cfg = cfg.safe_rl.lagrangian
    lagrange_controller = LagrangeSafetyController(
        enabled=bool(
            cfg.safe_rl.enabled and lagrange_cfg.enabled
        ),
        lambda_init=lagrange_cfg.lambda_init,
        lambda_lr=lagrange_cfg.lambda_lr,
        lambda_min=lagrange_cfg.lambda_min,
        lambda_max=lagrange_cfg.lambda_max,
        cost_budget=lagrange_cfg.cost_budget,
        update_interval=lagrange_cfg.update_interval,
        cost_ema_factor=lagrange_cfg.cost_ema_factor,
        warmup_steps=lagrange_cfg.warmup_steps,
    )
    initial_agent_lambda = (
        lagrange_controller.current_lambda
        if lagrange_controller.enabled
        else cfg.safe_rl.initial_lagrange_multiplier
    )

    # 训练前固定随机种子，便于复现实验
    set_seed(cfg.optimizer_seed)

    # 构造环境入参。部分 reward 尺度参数不是构造函数参数，
    # 会在环境创建和 reset 之后通过 apply_env_scales() 手动写入
    EnvCls = CloudWorkflowEnv_VMAgents # 这里只是给环境类起一个局部别名
    env_kwargs = dict(
        dax_paths=cfg.dax_list,
        horizon=cfg.horizon,
        arrival_lambda=cfg.arrival_lambda,
        random_seed=cfg.random_seed,
        max_ready_tasks=cfg.max_ready_tasks,
        normalize=cfg.normalize_obs,
        workflows_per_episode=cfg.workflows_per_episode,

        num_cloud_hosts=cfg.num_cloud_hosts,
        num_edge_hosts=cfg.num_edge_hosts,
        cloud_vms_per_host=cfg.cloud_vms_per_host,
        edge_vms_per_host=cfg.edge_vms_per_host,
        cloud_pc_tiers=cfg.cloud_pc_tiers,
        edge_pc_tiers=cfg.edge_pc_tiers,
        cloud_bw_tiers=cfg.cloud_bw_tiers,
        edge_bw_tiers=cfg.edge_bw_tiers,
        fuzzy_delta1=0.75,
        fuzzy_delta2=1.2,

        deadline_mode="cache_fcfs",
        deadline_cache_path=cfg.deadline_cache_path,
        deadline_cache_strict=True,
        deadline_alpha_small=cfg.deadline_alpha_small,
        deadline_alpha_large=cfg.deadline_alpha_large,
        deadline_alpha_small_prob=cfg.deadline_alpha_small_prob,

        manager_alpha_delay=cfg.manager_alpha_delay,
        manager_delay_mode=cfg.manager_delay_mode,

        # 阶段 1 显式开启时同步启用三场景跟踪；默认 false 保持旧 modal 行为。
        safe_rl_enabled=cfg.safe_rl.enabled,
        safe_rl_process_risk_aggregation=(
            cfg.safe_rl.process_risk_aggregation
        ),
        safe_rl_shield_enabled=cfg.safe_rl.shield.enabled,
        safe_rl_fallback_controller=(
            cfg.safe_rl.shield.fallback_controller
        ),
        safe_rl_state_enabled=cfg.safe_rl.state.enabled,
        safe_rl_state_high_uncertainty_threshold=(
            cfg.safe_rl.state.high_uncertainty_threshold
        ),
        safe_rl_state_recent_record_window=(
            cfg.safe_rl.state.recent_record_window
        ),
        manager_mode=cfg.safe_rl.manager_heuristics.mode,
        manager_heuristic_library_path=(
            cfg.safe_rl.manager_heuristics.library_manifest_path
        ),
        manager_heuristic_llm_only=(
            cfg.safe_rl.manager_heuristics.llm_only
        ),
        experiment_protocol_identity=cfg.experiment_protocol,
        manager_heuristic_recent_window=(
            cfg.safe_rl.manager_heuristics.recent_window
        ),
        scenario_code=cfg.scenario,
        task_code=cfg.task_code,
        resource_code=cfg.res_code,
        workflow_families=cfg.workflow_families,
        fuzzy_enabled=cfg.safe_rl.enabled,
        fuzzy_energy_uncertainty_weight=(
            cfg.safe_rl.fuzzy_energy_uncertainty_weight
        ),
        fuzzy_deadline_eta=cfg.safe_rl.fuzzy_deadline_eta,
        fuzzy_use_deadline_constraint=True,
    )
    base_env_kwargs = dict(env_kwargs)
    if training_controller is not None:
        env, env_kwargs = _construct_pipeline_environment(
            EnvCls,
            base_env_kwargs=base_env_kwargs,
            controller=training_controller,
            cfg=cfg,
        )
    else:
        if cfg.experiment_protocol is not None:
            env_kwargs = _scenario_env_kwargs(
                base_env_kwargs,
                cfg,
                scenario_for_training_episode(cfg, 0),
                seed=_seed_for_training_episode(cfg, 0),
            )
        env = EnvCls(**_environment_ctor_kwargs(env_kwargs)) # 创建环境
        env.reset() # 初始化环境， 函数为自己创建

    # 将配置中的 reward/归一化尺度写入训练环境（手动化写入参数）
    apply_env_scales(env, cfg)

    # 评估环境也需要复用这些尺度参数，因此把它们补充进 env_kwargs，保证后续评估环境与训练环境使用相同尺度
    _sync_env_kwargs_scales(env_kwargs, env)

    # 探测三层 observation/action 维度。课程切换后的新环境会与该基线比较，
    # 不允许在同一 checkpoint 中静默改变 Host/VM 动作维度。
    environment_dimensions = _probe_environment_dimensions(
        env,
        cfg,
    )
    host_state_dim = environment_dimensions["host_state_dim"]
    host_act_dim = environment_dimensions["host_action_dim"]
    vm_state_dim = environment_dimensions["vm_state_dim"]
    vm_act_dim = environment_dimensions["vm_action_dim"]

    print(f"[env] host_obs_dim={host_state_dim} host_act_dim={host_act_dim} | vm_obs_dim={vm_state_dim} vm_act_dim={vm_act_dim}")
    manager_act_dim = environment_dimensions[
        "manager_action_dim"
    ]
    manager_semantics = env.get_manager_action_semantics()
    print(
        f"[manager] mode={manager_semantics['manager_mode']} "
        f"action_type={manager_semantics['action_type']} "
        f"action_dim={manager_act_dim} "
        f"manager_state_dim={env.get_manager_state().shape[0]}"
    )
    print(
        f"[manager reward cfg] alpha_delay={cfg.manager_alpha_delay:.3f} "
        f"alpha_energy={1.0-cfg.manager_alpha_delay:.3f} delay_mode={cfg.manager_delay_mode}"
    )
    print(
        f"[ddl alpha cfg] P({cfg.deadline_alpha_small})={cfg.deadline_alpha_small_prob:.2f}, "
        f"P({cfg.deadline_alpha_large})={1.0-cfg.deadline_alpha_small_prob:.2f}"
    )
    print(
        f"[safe rl stage1] enabled={cfg.safe_rl.enabled} "
        f"lambda_E={cfg.safe_rl.fuzzy_energy_uncertainty_weight:.2f} "
        f"eta={cfg.safe_rl.fuzzy_deadline_eta:.2f} "
        f"process_risk_aggregation={cfg.safe_rl.process_risk_aggregation}"
    )
    print(
        f"[safe rl shield stage4] enabled="
        f"{cfg.safe_rl.shield.enabled} fallback="
        f"{cfg.safe_rl.shield.fallback_controller}"
    )
    print(
        f"[safe rl dual value stage7] enabled="
        f"{cfg.safe_rl.enabled} gamma_c="
        f"{cfg.safe_rl.safety_discount:.3f} lr_c="
        f"{cfg.safe_rl.safety_learning_rate:.6g} "
        f"loss_weight_c={cfg.safe_rl.safety_loss_weight:.3f} "
        f"lambda_initial={initial_agent_lambda:.3f}"
    )
    print(
        f"[safe rl lagrange stage8] enabled="
        f"{lagrange_controller.enabled} sharing=global_three_layer "
        f"period=episode_ema lr={lagrange_cfg.lambda_lr:.6g} "
        f"bounds=[{lagrange_cfg.lambda_min:.3f},"
        f"{lagrange_cfg.lambda_max:.3f}] "
        f"cost_budget={lagrange_cfg.cost_budget:.6g} "
        f"interval={lagrange_cfg.update_interval}episodes "
        f"ema_factor={lagrange_cfg.cost_ema_factor:.3f} "
        f"warmup={lagrange_cfg.warmup_steps}episodes"
    )
    print(
        f"[safe replay stage10] enabled={cfg.safe_rl.enabled} "
        f"schema={cfg.safe_rl.replay.transition_schema_version} "
        f"near_boundary_margin="
        f"{cfg.safe_rl.replay.near_boundary_margin:.6g}s "
        f"combined_per_priority="
        f"{cfg.safe_rl.replay.combined_per_priority} "
        f"td_weights=("
        f"{cfg.safe_rl.replay.performance_td_weight:.3f},"
        f"{cfg.safe_rl.replay.safety_td_weight:.3f})"
    )
    print(
        f"[safe Manager stage11] mode="
        f"{cfg.safe_rl.manager_heuristics.mode} "
        f"manifest="
        f"{cfg.safe_rl.manager_heuristics.library_manifest_path} "
        f"recent_window="
        f"{cfg.safe_rl.manager_heuristics.recent_window}"
    )
    if training_controller is not None:
        print(
            "[safe training pipeline stage14] "
            f"id={training_plan.pipeline_id} "
            f"plan_hash={training_plan.plan_hash} "
            f"current_stage="
            f"{training_controller.current_stage.stage_id} "
            f"train_seeds={training_plan.seed_split.training} "
            f"validation_seeds="
            f"{training_plan.seed_split.validation} "
            f"final_test_seeds_withheld="
            f"{training_plan.seed_split.final_test}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager_observation_schema_version = (
        env.get_observation_schema("manager")["schema_version"]
        if cfg.safe_rl.state.enabled
        else "legacy_observation"
    )
    host_observation_schema_version = (
        env.get_observation_schema("host")["schema_version"]
        if cfg.safe_rl.state.enabled
        else "legacy_observation"
    )
    vm_observation_schema_version = (
        env.get_observation_schema("vm")["schema_version"]
        if cfg.safe_rl.state.enabled
        else "legacy_observation"
    )

    # VM 层智能体：在 HostAgent 已选定 host 后，为当前任务选择具体 VM 槽位。
    vm_agent = D3QNAgent(
        input_dim=vm_state_dim,
        output_dim=vm_act_dim,
        lr=cfg.vm_agent.lr,
        gamma=cfg.vm_agent.gamma,
        batch_size=cfg.vm_agent.batch_size,
        buffer_size=cfg.vm_agent.buffer_size,
        eps_start=cfg.vm_agent.eps_start,
        eps_end=cfg.vm_agent.eps_end,
        eps_decay_steps=cfg.vm_agent.eps_decay_steps,
        target_update_tau=cfg.vm_agent.target_update_tau,
        grad_clip=cfg.vm_agent.grad_clip,
        hidden_dims=cfg.vm_agent.hidden_dims,
        device=device,
        observation_schema_version=vm_observation_schema_version,
        safe_rl_enabled=cfg.safe_rl.enabled,
        safety_discount=cfg.safe_rl.safety_discount,
        safety_learning_rate=cfg.safe_rl.safety_learning_rate,
        safety_loss_weight=cfg.safe_rl.safety_loss_weight,
        initial_lagrange_multiplier=(
            initial_agent_lambda
        ),
        safe_replay_near_boundary_margin=(
            cfg.safe_rl.replay.near_boundary_margin
        ),
        safe_per_combined_priority=(
            cfg.safe_rl.replay.combined_per_priority
        ),
        safe_per_performance_td_weight=(
            cfg.safe_rl.replay.performance_td_weight
        ),
        safe_per_safety_td_weight=(
            cfg.safe_rl.replay.safety_td_weight
        ),
    )

    # Host 层智能体：为当前待调度任务选择 host。
    host_agent = D3QNAgent(
        input_dim=host_state_dim,
        output_dim=host_act_dim,
        lr=cfg.host_agent.lr,
        gamma=cfg.host_agent.gamma,
        batch_size=cfg.host_agent.batch_size,
        buffer_size=cfg.host_agent.buffer_size,
        eps_start=cfg.host_agent.eps_start,
        eps_end=cfg.host_agent.eps_end,
        eps_decay_steps=cfg.host_agent.eps_decay_steps,
        target_update_tau=cfg.host_agent.target_update_tau,
        grad_clip=cfg.host_agent.grad_clip,
        hidden_dims=cfg.host_agent.hidden_dims,
        device=device,
        observation_schema_version=host_observation_schema_version,
        safe_rl_enabled=cfg.safe_rl.enabled,
        safety_discount=cfg.safe_rl.safety_discount,
        safety_learning_rate=cfg.safe_rl.safety_learning_rate,
        safety_loss_weight=cfg.safe_rl.safety_loss_weight,
        initial_lagrange_multiplier=(
            initial_agent_lambda
        ),
        safe_replay_near_boundary_margin=(
            cfg.safe_rl.replay.near_boundary_margin
        ),
        safe_per_combined_priority=(
            cfg.safe_rl.replay.combined_per_priority
        ),
        safe_per_performance_td_weight=(
            cfg.safe_rl.replay.performance_td_weight
        ),
        safe_per_safety_td_weight=(
            cfg.safe_rl.replay.safety_td_weight
        ),
    )

    # Manager 层智能体：在 phase 级别选择调度策略参数 delta。
    sH0 = env.get_manager_state()
    state_dim_mgr = int(sH0.shape[0])
    manager_agent = D3QNAgent(
        input_dim=state_dim_mgr,
        output_dim=manager_act_dim,
        lr=cfg.manager_agent.lr,
        gamma=cfg.manager_agent.gamma,
        batch_size=cfg.manager_agent.batch_size,
        buffer_size=cfg.manager_agent.buffer_size,
        eps_start=cfg.manager_agent.eps_start,
        eps_end=cfg.manager_agent.eps_end,
        eps_decay_steps=cfg.manager_agent.eps_decay_steps,
        target_update_tau=cfg.manager_agent.target_update_tau,
        grad_clip=cfg.manager_agent.grad_clip,
        hidden_dims=cfg.manager_agent.hidden_dims,
        device=device,
        observation_schema_version=(
            manager_observation_schema_version
        ),
        safe_rl_enabled=cfg.safe_rl.enabled,
        safety_discount=cfg.safe_rl.safety_discount,
        safety_learning_rate=cfg.safe_rl.safety_learning_rate,
        safety_loss_weight=cfg.safe_rl.safety_loss_weight,
        initial_lagrange_multiplier=(
            initial_agent_lambda
        ),
        safe_replay_near_boundary_margin=(
            cfg.safe_rl.replay.near_boundary_margin
        ),
        safe_per_combined_priority=(
            cfg.safe_rl.replay.combined_per_priority
        ),
        safe_per_performance_td_weight=(
            cfg.safe_rl.replay.performance_td_weight
        ),
        safe_per_safety_td_weight=(
            cfg.safe_rl.replay.safety_td_weight
        ),
    )
    safe_agents = (manager_agent, host_agent, vm_agent)
    agents_by_layer = {
        "manager": manager_agent,
        "host": host_agent,
        "vm": vm_agent,
    }
    if lagrange_controller.enabled:
        synchronize_lagrange_multiplier(
            lagrange_controller,
            safe_agents,
        )
    if training_resume_payload is not None:
        restore_pipeline_agents(
            training_resume_payload,
            agents=agents_by_layer,
            lagrange_controller=lagrange_controller,
        )

    # 离线初始化发生在在线循环之前。该入口不调用 D3QNAgent.update，
    # 因而不推进 replay/PER、epsilon、online update 计数或 target 周期。
    if training_resume_payload is None:
        pretraining_report = pretrain_agents_from_demonstrations(
            agents_by_layer,
            cfg.safe_rl.offline_pretraining,
        )
    else:
        # 恢复阶段化 checkpoint 时不得再次执行阶段 2，否则会覆盖已在线
        # 学到的 Q_r/Q_c。
        pretraining_report = {
            "enabled": False,
            "resume_skipped": True,
        }
    if pretraining_report["enabled"]:
        pretraining_report_path = os.path.join(
            cfg.save_dir,
            "offline_pretraining_report.json",
        )
        with open(
            pretraining_report_path,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                pretraining_report,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
        print(
            "[safe demonstration pretraining] "
            f"train_episodes="
            f"{pretraining_report['train_episode_count']} "
            f"validation_episodes="
            f"{pretraining_report['validation_episode_count']} "
            f"report={pretraining_report_path}"
        )

    # CSVLogger 同时记录 phase 粒度和 episode 粒度指标。
    # phase 行关注即时 reward 和阶段成本，episode 行关注评估结果和迟延统计。
    logger_fields = [
        "step", "episode", "type",
        "ep_length", "env_time",
        "episode_energy",
        "wf_completed", "wf_target",
        "eps_vm", "eps_host", "eps_mgr",
        "assign_cnt", "phase_size_sum_mi",
        "r_vm_phase_mean", "r_host_phase_mean",
        "r_manager_raw",
        "phase_energy", "phase_delay",
        "phase_energy_cost", "phase_delay_cost",
        "manager_reward_old", "manager_reward_new",
        "ep_total_lateness", "ep_avg_lateness", "wf_avg_lateness",
        "ep_wf_lateness_sum",
        "eval_vm", "eval_host", "eval_mgr", "eval_energy",
    ]
    if cfg.safe_rl.enabled:
        # safe 模式使用独立输出目录，可以安全增加阶段 1 字段；关闭时保持旧
        # CSV 表头完全不变，避免向历史日志追加不同列数的行。
        safety_fields = [
            "performance_reward",
            "energy_reward",
            "completion_reward",
            "waiting_reward",
            "utilization_reward",
            "communication_reward",
            "total_performance_reward",
            "safety_cost",
            "deadline_violation_cost",
            "fuzzy_lateness_cost",
            "process_risk_cost",
            "deadline_violation_count",
            "completed_workflow_count",
            "predicted_deadline_violation_count",
            "min_fuzzy_safety_margin",
            "mean_fuzzy_safety_margin",
            "risk_workflow_ratio",
            "predicted_violation_rate",
            "manager_action_source",
            "random_safe_exploration_count",
            "greedy_safe_action_count",
            "shield_correction_count",
            "fallback_action_count",
            "eval_max_fuzzy_lateness",
            "eval_mean_fuzzy_lateness",
            "eval_all_seed_feasible",
            "eval_feasible_seed_rate",
            "eval_worst_seed_violation",
            "eval_worst_seed_lateness",
        ]
        insert_at = logger_fields.index("ep_total_lateness")
        logger_fields[insert_at:insert_at] = safety_fields
    if cfg.safe_rl.shield.enabled:
        shield_fields = [
            "shield_record_count",
            "shield_intervention_count",
            "shield_fallback_count",
            "host_action_modified_count",
            "vm_action_modified_count",
        ]
        insert_at = logger_fields.index("ep_total_lateness")
        logger_fields[insert_at:insert_at] = shield_fields
    if lagrange_controller.enabled:
        lagrange_fields = [
            "current_lambda",
            "mean_safety_cost",
            "cost_budget",
            "constraint_gap",
            "lambda_update_count",
            "eval_deadline_violation_rate",
            "eval_zero_violation_pass",
        ]
        insert_at = logger_fields.index("ep_total_lateness")
        logger_fields[insert_at:insert_at] = lagrange_fields
    if (
        cfg.safe_rl.manager_heuristics.mode
        == "heuristic_selection_mode"
    ):
        heuristic_fields = [
            "selected_heuristic_id",
            "heuristic_source",
            "llm_rule_version",
            "ready_task_ordering",
            "heuristic_shield_intervention_count",
            "heuristic_shield_intervention_rate",
        ]
        insert_at = logger_fields.index("ep_total_lateness")
        logger_fields[insert_at:insert_at] = heuristic_fields
    logger = CSVLogger(
        filepath=cfg.log_path,
        fieldnames=logger_fields,
    )
    metrics_cfg = cfg.safe_rl.metrics
    metric_store = (
        SafeMetricStore(
            os.path.join(
                cfg.save_dir,
                metrics_cfg.output_subdir,
            ),
            convergence_window=metrics_cfg.convergence_window,
        )
        if cfg.safe_rl.enabled and metrics_cfg.enabled
        else None
    )
    if metric_store is not None:
        print(
            "[safe metrics stage16] "
            f"schema={metrics_cfg.schema_version} "
            f"convergence_window="
            f"{metrics_cfg.convergence_window} "
            f"output={metric_store.output_directory}"
        )

    feasibility_first_selection = bool(
        cfg.safe_rl.enabled
        and cfg.safe_rl.model_selection.enabled
    )
    # 评估返回值是 4 元组还是 5 元组，必须只有这一处判据：请求安全指标的
    # 条件、跳过验证时构造的占位元组、以及解包的分支三者一旦分叉，就会在
    # 某些开关组合下抛 "too many values to unpack"。这四个子条件在整个训练
    # 循环内都不变（都在循环之前一次性确定），所以在这里求一次即可。
    validation_returns_safety_metrics = bool(
        feasibility_first_selection
        or lagrange_controller.enabled
        or training_controller is not None
        or metric_store is not None
    )
    best_model_metrics = None
    if (
        feasibility_first_selection
        and training_resume_payload is not None
        and training_resume_payload.get("best_model_metrics")
        is not None
    ):
        best_model_metrics = (
            FeasibilityFirstModelMetrics.from_mapping(
                training_resume_payload["best_model_metrics"]
            )
        )
    # Legacy mode retains the historical energy-only checkpoint rule. Safe
    # mode uses the feasibility-first key and keeps this scalar as a
    # compatibility/reporting alias only.
    best_eval_energy = (
        float(best_model_metrics.fuzzy_energy_score)
        if best_model_metrics is not None
        else (
            float(
                training_resume_payload[
                    "best_validation_energy"
                ]
            )
            if (
                not feasibility_first_selection
                and training_resume_payload is not None
                and training_resume_payload.get(
                    "best_validation_energy"
                )
                is not None
            )
            else float("inf")
        )
    )
    best_ckpt_vm = os.path.join(cfg.save_dir, "best_vm.pth")
    best_ckpt_host = os.path.join(cfg.save_dir, "best_host.pth")
    best_ckpt_mgr = os.path.join(cfg.save_dir, "best_manager.pth")

    # 验证评估的进程级并行。worker 与父进程同设备：验证结果直接决定 best
    # checkpoint，masked argmax 在 cpu/cuda 之间的最后一两个 ulp 差异有极小
    # 概率翻转选择，所以这里绝不降级到 cpu。
    validation_pool = None
    # 课程学习开启时一次验证是"场景 × seed"摊平后的作业表，比单场景的
    # seed 数大；按最大那一批定容量，另一批多出来的 worker 只是闲置。
    validation_job_count = len(cfg.validation_seeds) * (
        len(tuple(cfg.training_scenarios))
        if training_controller is not None
        else 1
    )
    resolved_validation_workers = resolve_validation_workers(
        validation_workers,
        job_count=validation_job_count,
    )
    if resolved_validation_workers > 1:
        validation_pool = ValidationEvaluationPool(
            EnvCls,
            vm_agent,
            host_agent,
            manager_agent,
            workers=resolved_validation_workers,
            device=str(device),
            weights_path=os.path.join(
                cfg.save_dir, "validation_weights.pt"
            ),
        )
    print(
        "[validation parallel] "
        f"requested={validation_workers} "
        f"workers={resolved_validation_workers} "
        f"device={device} "
        f"seeds={len(cfg.validation_seeds)} "
        f"jobs_per_validation={validation_job_count} "
        f"audit={'on' if validation_pool is not None and validation_pool.audit_enabled else 'off'}"
    )
    # 并行只改变执行方式、不改变结果，因此不进 config_snapshot（否则会动到
    # checkpoint 身份哈希）。单独落一份出处文件供复现时查证。
    validation_parallel_manifest_path = os.path.join(
        cfg.save_dir, "validation_parallel.json"
    )
    with open(
        validation_parallel_manifest_path, "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "requested_workers": int(validation_workers),
                "worker_count": int(resolved_validation_workers),
                "parallel_backend": (
                    "process_pool_spawn"
                    if validation_pool is not None
                    else "serial"
                ),
                "worker_device": str(device),
                "validation_seed_count": len(cfg.validation_seeds),
                "jobs_per_validation": int(validation_job_count),
                "bitwise_audit_enabled": bool(
                    validation_pool is not None
                    and validation_pool.audit_enabled
                ),
            },
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")

    global_step = (
        int(training_resume_payload["global_step"])
        if training_resume_payload is not None
        else 0
    )
    episode_idx = (
        int(training_resume_payload["next_episode"])
        if training_resume_payload is not None
        else 0
    )
    episode_steps = 0
    ep_energy0 = float(env.total_energy)
    episode_started_at = time.perf_counter()
    episode_phase_metric_records = []

    # 每个 episode 开始时，Manager 先选择一个 phase 动作并应用到环境
    sH = env.get_manager_state()
    m_mask = env.get_manager_action_mask()
    m_act, m_action_audit = _select_manager_training_action(
        manager_agent,
        sH,
        m_mask,
        safe_rl_enabled=cfg.safe_rl.enabled,
    )
    manager_apply_action(env, m_act)
    pending_host_transition = None
    pending_vm_transition = None

    print_device_info(vm_agent, host_agent, manager_agent)

    while (episode_idx < cfg.max_episodes) and (global_step < cfg.hard_max_steps):
        # episode 结束后：执行评估、记录 episode 日志、保存 checkpoint，然后 reset 环境。
        if env.done_flag:
            episode_scheduling_seconds = (
                time.perf_counter() - episode_started_at
            )
            if lagrange_controller.enabled:
                lagrange_diagnostics = (
                    _update_shared_lagrange_at_episode_end(
                        lagrange_controller,
                        env,
                        safe_agents,
                    )
                )
            else:
                lagrange_diagnostics = (
                    lagrange_controller.diagnostics()
                )
            if cfg.safe_rl.enabled:
                q_c_error, q_c_error_count = (
                    q_c_prediction_error_from_agents(
                        safe_agents
                    )
                )
            else:
                q_c_error, q_c_error_count = 0.0, 0
            if metric_store is not None:
                training_seed_record = (
                    build_episode_metric_record(
                        env,
                        seed=int(
                            getattr(
                                env,
                                "random_seed",
                                cfg.random_seed,
                            )
                        ),
                        scheduling_time_seconds=(
                            episode_scheduling_seconds
                        ),
                        phase_records=(
                            episode_phase_metric_records
                        ),
                    )
                )
                training_metric_report = (
                    aggregate_safe_metric_records(
                        [training_seed_record],
                        q_c_prediction_error=q_c_error,
                        q_c_prediction_error_sample_count=(
                            q_c_error_count
                        ),
                        lambda_current=float(
                            lagrange_diagnostics[
                                "current_lambda"
                            ]
                        ),
                    )
                )
                metric_store.append(
                    "training",
                    training_metric_report,
                    global_step=global_step,
                    episode=episode_idx,
                )
            validation_due = (
                (episode_idx + 1) % cfg.validation_interval == 0
                or episode_idx + 1 == cfg.max_episodes
            )
            if validation_due:
                if validation_pool is not None:
                    # 本轮验证的全部场景共用同一份权重，只广播一次。
                    validation_pool.publish_weights()
                curriculum_eval_result = None
                if training_controller is not None:
                    curriculum_eval_result = _evaluate_training_scenarios(
                        EnvCls,
                        base_env_kwargs=base_env_kwargs,
                        active_env_kwargs=env_kwargs,
                        cfg=cfg,
                        controller=training_controller,
                        vm_agent=vm_agent,
                        host_agent=host_agent,
                        manager_agent=manager_agent,
                        seeds=cfg.validation_seeds,
                        return_safety_metrics=True,
                        scenarios=tuple(cfg.training_scenarios),
                        evaluation_pool=validation_pool,
                    )
                # Only fixed source-scenario validation may select checkpoint.
                eval_result = _evaluate_training_scenarios(
                    EnvCls,
                    base_env_kwargs=base_env_kwargs,
                    active_env_kwargs=base_env_kwargs,
                    cfg=cfg,
                    controller=None,
                    vm_agent=vm_agent,
                    host_agent=host_agent,
                    manager_agent=manager_agent,
                    seeds=cfg.validation_seeds,
                    scenarios=(str(cfg.source_scenario or cfg.training_scenarios[0]).upper(),),
                    return_safety_metrics=(
                        validation_returns_safety_metrics
                    ),
                    evaluation_pool=validation_pool,
                )
            else:
                eval_result = _skipped_validation_result(
                    float(env.total_energy - ep_energy0),
                    return_safety_metrics=(
                        validation_returns_safety_metrics
                    ),
                )
            if validation_returns_safety_metrics:
                (
                    eval_vm,
                    eval_host,
                    eval_mgr,
                    eval_energy,
                    eval_safety,
                ) = eval_result
            else:
                eval_vm, eval_host, eval_mgr, eval_energy = (
                    eval_result
                )
                eval_safety = {
                    "deadline_violation_rate": 0.0,
                    "zero_violation_pass": True,
                    "fuzzy_energy_score": float(eval_energy),
                    "safety_cost": 0.0,
                    "shield_intervention_rate": 0.0,
                    "fallback_rate": 0.0,
                    "max_fuzzy_lateness": 0.0,
                    "mean_fuzzy_lateness": 0.0,
                    "all_seed_feasible": True,
                    "all_seed_evaluation_completed": True,
                    "completed_evaluation_seed_rate": 1.0,
                    "feasible_seed_rate": 1.0,
                    "worst_seed_violation": 0.0,
                    "worst_seed_lateness": 0.0,
                    "validation_seed_count": len(
                        cfg.eval_seeds
                    ),
                }
            if (
                feasibility_first_selection
                and validation_due
                and not bool(
                    eval_safety[
                        "all_seed_evaluation_completed"
                    ]
                )
            ):
                raise RuntimeError(
                    "feasibility-first model selection requires every "
                    "validation seed to complete its expected workflows"
                )
            candidate_model_metrics = (
                FeasibilityFirstModelMetrics.from_mapping(
                    eval_safety
                )
                if feasibility_first_selection and validation_due
                else None
            )
            if metric_store is not None and validation_due:
                eval_safety["q_c_prediction_error"] = float(
                    q_c_error
                )
                eval_safety[
                    "q_c_prediction_error_sample_count"
                ] = int(q_c_error_count)
                eval_safety["lambda_current"] = float(
                    lagrange_diagnostics["current_lambda"]
                )
                eval_safety = metric_store.append(
                    "validation",
                    eval_safety,
                    global_step=global_step,
                    episode=episode_idx,
                )

            ep_energy = float(env.total_energy - ep_energy0)
            ep_total_late, ep_avg_late, wf_avg_late = compute_episode_task_lateness_metrics(env)
            ep_wf_late_sum = float(getattr(env, "ep_wf_lateness_sum", 0.0))
            episode_shield = env.get_safety_shield_diagnostics()
            pipeline_transition_event = {
                "transitioned": False,
                "pipeline_completed": False,
            }
            pipeline_stage_metrics = None
            if (
                training_controller is not None
                and validation_due
                and not training_controller.completed
            ):
                curriculum_safety = curriculum_eval_result[4]
                pipeline_stage_metrics = StageMetrics(
                    fuzzy_energy_score=float(
                        curriculum_safety["fuzzy_energy_score"]
                    ),
                    safety_cost=float(curriculum_safety["safety_cost"]),
                    violation_rate=float(
                        curriculum_safety["deadline_violation_rate"]
                    ),
                    shield_intervention_rate=float(
                        curriculum_safety["shield_intervention_rate"]
                    ),
                    fallback_rate=float(curriculum_safety["fallback_rate"]),
                    lagrange_multiplier=float(
                        lagrange_diagnostics["current_lambda"]
                    ),
                    q_c_prediction_error=q_c_error,
                    q_c_prediction_error_sample_count=q_c_error_count,
                )
                curriculum_observation = (
                    training_controller.observe_validation(
                        pipeline_stage_metrics,
                        source="curriculum_validation",
                        count_episode=False,
                    )
                )
            else:
                curriculum_observation = None
            if training_controller is not None:
                pipeline_transition_event = training_controller.record_episode()
                pipeline_transition_event["training_seed_used"] = (
                    _seed_for_training_episode(cfg, episode_idx)
                )
                pipeline_transition_event["training_scenario_used"] = (
                    scenario_for_training_episode(cfg, episode_idx)
                )
                if curriculum_observation is not None:
                    pipeline_transition_event[
                        "curriculum_validation_observation"
                    ] = curriculum_observation
                    stage_metrics_logger.log(
                        controller=training_controller,
                        metrics=pipeline_stage_metrics,
                        transition_event=pipeline_transition_event,
                        global_step=global_step,
                        episode=episode_idx,
                    )
            logger.log(
                step=global_step, episode=episode_idx, type="episode",
                ep_length=episode_steps, env_time=getattr(env, "current_time", 0.0),
                episode_energy=ep_energy,
                wf_completed=getattr(env, "completed_workflows", 0),
                wf_target=(getattr(env, "workflows_per_episode", None) or ""),
                eps_vm=vm_agent.epsilon(),
                eps_host=host_agent.epsilon(),
                eps_mgr=manager_agent.epsilon(),
                assign_cnt="",
                phase_size_sum_mi="",
                r_vm_phase_mean="",
                r_host_phase_mean="",
                r_manager_raw="",
                phase_energy="",
                phase_delay="",
                phase_energy_cost="",
                phase_delay_cost="",
                manager_reward_old="",
                manager_reward_new="",
                performance_reward="",
                energy_reward="",
                completion_reward="",
                waiting_reward="",
                utilization_reward="",
                communication_reward="",
                total_performance_reward="",
                safety_cost=float(
                    getattr(env, "_safety_cumulative_cost", 0.0)
                ),
                deadline_violation_cost=float(
                    getattr(
                        env,
                        "_safety_cumulative_deadline_violation_count",
                        0,
                    )
                ),
                fuzzy_lateness_cost=float(
                    getattr(
                        env,
                        "_safety_cumulative_fuzzy_lateness_cost",
                        0.0,
                    )
                ),
                process_risk_cost=float(
                    getattr(
                        env,
                        "_safety_cumulative_process_risk_cost",
                        0.0,
                    )
                ),
                deadline_violation_count=int(
                    getattr(
                        env,
                        "_safety_cumulative_deadline_violation_count",
                        0,
                    )
                ),
                completed_workflow_count=int(
                    getattr(
                        env,
                        "_safety_cumulative_completed_workflow_count",
                        0,
                    )
                ),
            predicted_deadline_violation_count="",
            min_fuzzy_safety_margin="",
            mean_fuzzy_safety_margin="",
            risk_workflow_ratio="",
                predicted_violation_rate="",
                selected_heuristic_id="",
                heuristic_source="",
                llm_rule_version="",
                ready_task_ordering="",
                heuristic_shield_intervention_count="",
                heuristic_shield_intervention_rate="",
                manager_action_source=(
                    m_action_audit["action_source"]
                    if m_action_audit is not None
                    else ""
                ),
                random_safe_exploration_count=int(
                    episode_shield[
                        "random_safe_exploration_count"
                    ]
                ),
                greedy_safe_action_count=int(
                    episode_shield[
                        "greedy_safe_action_count"
                    ]
                ),
                shield_correction_count=int(
                    episode_shield[
                        "shield_correction_count"
                    ]
                ),
                fallback_action_count=int(
                    episode_shield[
                        "fallback_action_count"
                    ]
                ),
                shield_record_count=int(
                    episode_shield["shield_record_count"]
                ),
                shield_intervention_count=int(
                    episode_shield[
                        "shield_intervention_count"
                    ]
                ),
                shield_fallback_count=int(
                    episode_shield["shield_fallback_count"]
                ),
                host_action_modified_count=int(
                    episode_shield[
                        "host_action_modified_count"
                    ]
                ),
                vm_action_modified_count=int(
                    episode_shield[
                        "vm_action_modified_count"
                    ]
                ),
                current_lambda=float(
                    lagrange_diagnostics["current_lambda"]
                ),
                mean_safety_cost=float(
                    lagrange_diagnostics["mean_safety_cost"]
                ),
                cost_budget=float(
                    lagrange_diagnostics["cost_budget"]
                ),
                constraint_gap=float(
                    lagrange_diagnostics["constraint_gap"]
                ),
                lambda_update_count=int(
                    lagrange_diagnostics[
                        "lambda_update_count"
                    ]
                ),
                eval_deadline_violation_rate=float(
                    eval_safety["deadline_violation_rate"]
                ),
                eval_zero_violation_pass=bool(
                    eval_safety["zero_violation_pass"]
                ),
                eval_max_fuzzy_lateness=float(
                    eval_safety["max_fuzzy_lateness"]
                ),
                eval_mean_fuzzy_lateness=float(
                    eval_safety["mean_fuzzy_lateness"]
                ),
                eval_all_seed_feasible=bool(
                    eval_safety["all_seed_feasible"]
                ),
                eval_feasible_seed_rate=float(
                    eval_safety["feasible_seed_rate"]
                ),
                eval_worst_seed_violation=float(
                    eval_safety["worst_seed_violation"]
                ),
                eval_worst_seed_lateness=float(
                    eval_safety["worst_seed_lateness"]
                ),
                ep_total_lateness=ep_total_late,
                ep_avg_lateness=ep_avg_late,
                wf_avg_lateness=wf_avg_late,
                ep_wf_lateness_sum=ep_wf_late_sum,
                eval_vm=eval_vm, eval_host=eval_host, eval_mgr=eval_mgr, eval_energy=eval_energy,
            )

            print(
                f"[episode={episode_idx}] len={episode_steps} energy={ep_energy:.3f}J | "
                f"task_late_sum={ep_total_late:.3f}s task_late_avg={ep_avg_late:.6f}s | "
                f"wf_late_sum={ep_wf_late_sum:.3f}s | "
                f"eval_vm={eval_vm:.6f} eval_host={eval_host:.6f} eval_mgr={eval_mgr:.6f} eval_energy={eval_energy:.3f}J "
                f"(mean over validation seeds="
                f"{training_plan.seed_split.validation if training_plan is not None else cfg.eval_seeds}) | "
                f"wf={getattr(env,'completed_workflows',0)}/{getattr(env,'workflows_per_episode','unknown')} | "
                f"lambda={lagrange_diagnostics['current_lambda']:.6f} "
                f"Jc_ema={lagrange_diagnostics['mean_safety_cost']:.6f} "
                f"gap={lagrange_diagnostics['constraint_gap']:.6f} "
                f"lambda_updates={lagrange_diagnostics['lambda_update_count']} | "
                f"eval_vio_rate={eval_safety['deadline_violation_rate']:.6f} "
                f"max_fuzzy_late={eval_safety['max_fuzzy_lateness']:.6f} "
                f"mean_fuzzy_late={eval_safety['mean_fuzzy_lateness']:.6f} "
                f"all_seed_feasible={eval_safety['all_seed_feasible']} "
                f"feasible_seed_rate={eval_safety['feasible_seed_rate']:.6f} "
                f"worst_seed_vio={eval_safety['worst_seed_violation']:.6f} "
                f"worst_seed_late={eval_safety['worst_seed_lateness']:.6f}"
            )

            best_improved = validation_due and (
                is_better_model(
                    candidate_model_metrics,
                    best_model_metrics,
                )
                if feasibility_first_selection
                else eval_energy < best_eval_energy
            )
            if best_improved:
                if feasibility_first_selection:
                    best_model_metrics = candidate_model_metrics
                    best_eval_energy = float(
                        candidate_model_metrics.fuzzy_energy_score
                    )
                    checkpoint_metadata = (
                        _checkpoint_runtime_metadata(
                            cfg=cfg,
                            env=env,
                            agents=agents_by_layer,
                            training_controller=(
                                training_controller
                            ),
                        )
                    )
                    best_manifest = (
                        save_best_checkpoint_bundle(
                            cfg.save_dir,
                            agents=agents_by_layer,
                            lagrange_controller=(
                                lagrange_controller
                            ),
                            model_metrics=best_model_metrics,
                            **checkpoint_metadata,
                        )
                    )
                    validation_requirement_met = bool(
                        best_model_metrics.all_seed_feasible
                        if (
                            cfg.safe_rl.model_selection
                            .require_all_validation_seeds_feasible
                        )
                        else (
                            best_model_metrics
                            .deadline_violation_rate
                            == 0.0
                        )
                    )
                    print(
                        "[best feasibility-first] key="
                        f"{best_model_metrics.comparison_key} "
                        f"all_seed_required="
                        f"{cfg.safe_rl.model_selection.require_all_validation_seeds_feasible} "
                        f"requirement_met={validation_requirement_met} "
                        f"manifest={best_manifest}"
                    )
                else:
                    best_eval_energy = eval_energy
                    _save_agent_checkpoint(
                        vm_agent,
                        best_ckpt_vm,
                        lagrange_controller,
                    )
                    _save_agent_checkpoint(
                        host_agent,
                        best_ckpt_host,
                        lagrange_controller,
                    )
                    _save_agent_checkpoint(
                        manager_agent,
                        best_ckpt_mgr,
                        lagrange_controller,
                    )
                    print(
                        f"[best legacy] eval_energy="
                        f"{best_eval_energy:.3f}J -> saved "
                        f"{best_ckpt_vm} / {best_ckpt_host} / "
                        f"{best_ckpt_mgr}"
                    )

            if (global_step % cfg.save_interval) == 0:
                _save_agent_checkpoint(
                    vm_agent,
                    os.path.join(
                        cfg.save_dir,
                        f"vm_step{global_step}.pth",
                    ),
                    lagrange_controller,
                )
                _save_agent_checkpoint(
                    host_agent,
                    os.path.join(
                        cfg.save_dir,
                        f"host_step{global_step}.pth",
                    ),
                    lagrange_controller,
                )
                _save_agent_checkpoint(
                    manager_agent,
                    os.path.join(
                        cfg.save_dir,
                        f"mgr_step{global_step}.pth",
                    ),
                    lagrange_controller,
                )

            # The completed episode is already logged and checkpointed. Do
            # not construct/reset an unused environment after the configured
            # final episode; doing so can consume the next environment seed.
            if episode_idx + 1 >= cfg.max_episodes:
                episode_idx += 1
                break

            if training_controller is not None:
                checkpoint_due = (
                    training_controller.total_episode_count
                    % training_plan.checkpoint_interval_episodes
                    == 0
                )
                if (
                    checkpoint_due
                    or pipeline_transition_event["transitioned"]
                    or pipeline_transition_event[
                        "pipeline_completed"
                    ]
                ):
                    pipeline_checkpoint_metadata = (
                        _checkpoint_runtime_metadata(
                            cfg=cfg,
                            env=env,
                            agents=agents_by_layer,
                            training_controller=(
                                training_controller
                            ),
                        )
                    )
                    pipeline_checkpoint_path = (
                        save_pipeline_checkpoint(
                            os.path.join(
                                cfg.save_dir,
                                "pipeline_latest",
                            ),
                            controller=training_controller,
                            agents=agents_by_layer,
                            lagrange_controller=(
                                lagrange_controller
                            ),
                            global_step=global_step,
                            next_episode=episode_idx + 1,
                            best_model_metrics=(
                                best_model_metrics.to_dict()
                                if best_model_metrics is not None
                                else None
                            ),
                            replay_metadata=(
                                pipeline_checkpoint_metadata[
                                    "replay_metadata"
                                ]
                            ),
                            heuristic_library_version=(
                                pipeline_checkpoint_metadata[
                                    "heuristic_library_version"
                                ]
                            ),
                            config_snapshot=(
                                pipeline_checkpoint_metadata[
                                    "config_snapshot"
                                ]
                            ),
                        )
                    )
                    print(
                        "[safe training pipeline checkpoint] "
                        f"{pipeline_checkpoint_path}"
                    )
                if pipeline_transition_event[
                    "pipeline_completed"
                ]:
                    print(
                        "[safe training pipeline] curriculum completed after "
                        f"{training_controller.total_episode_count} "
                        "online episodes; continuing frozen final stage to "
                        f"the configured {cfg.max_episodes} episodes"
                    )
                env, next_env_kwargs = (
                    _construct_pipeline_environment(
                        EnvCls,
                        base_env_kwargs=base_env_kwargs,
                        controller=training_controller,
                        cfg=cfg,
                    )
                )
                next_dimensions = _probe_environment_dimensions(
                    env,
                    cfg,
                )
                if next_dimensions != environment_dimensions:
                    raise ValueError(
                        "curriculum profile changed observation/action "
                        "dimensions. Use a separate training plan and "
                        "checkpoint for different Host/VM topology or "
                        "observation schema."
                    )
                env_kwargs = next_env_kwargs
            else:
                if cfg.experiment_protocol is None:
                    env.random_seed = (
                        getattr(env, "random_seed", 0) + 1
                    )
                    env.reset()
                    apply_env_scales(env, cfg)
                else:
                    next_episode_index = episode_idx + 1
                    next_env_kwargs = _scenario_env_kwargs(
                        base_env_kwargs,
                        cfg,
                        scenario_for_training_episode(
                            cfg,
                            next_episode_index,
                        ),
                        seed=_seed_for_training_episode(
                            cfg,
                            next_episode_index,
                        ),
                    )
                    env = EnvCls(
                        **_environment_ctor_kwargs(next_env_kwargs)
                    )
                    env.reset()
                    apply_env_scales(env, cfg)
                    _sync_env_kwargs_scales(next_env_kwargs, env)
                    env_kwargs = next_env_kwargs
            pending_host_transition = None
            pending_vm_transition = None

            ep_energy0 = float(env.total_energy)
            episode_steps = 0
            episode_idx += 1
            episode_started_at = time.perf_counter()
            episode_phase_metric_records = []

            sH = env.get_manager_state()
            m_mask = env.get_manager_action_mask()
            m_act, m_action_audit = (
                _select_manager_training_action(
                    manager_agent,
                    sH,
                    m_mask,
                    safe_rl_enabled=cfg.safe_rl.enabled,
                )
            )
            manager_apply_action(env, m_act)
            continue

        # 一个 phase 内可能包含多个任务分配。
        # 对每个任务，先由 HostAgent 选 host，再由 VMAgent 选 VM 槽位。
        phase_vm_rewards = []
        phase_host_rewards = []
        phase_safety = {
            "safety_cost": 0.0,
            "deadline_violation_cost": 0.0,
            "fuzzy_lateness_cost": 0.0,
            "process_risk_cost": 0.0,
            "deadline_violation_count": 0,
            "completed_workflow_count": 0,
            "predicted_deadline_violation_count": 0,
            "min_fuzzy_safety_margin": None,
        }

        while True:
            st_host, has_next = env.get_host_state_for_next_assignment()
            if not has_next:
                break

            if cfg.safe_rl.enabled and pending_host_transition is not None:
                next_host_mask = layer_learning_action_mask(
                    st_host,
                    safe_rl_enabled=True,
                )
                _commit_safe_pending_transition(
                    host_agent,
                    pending_host_transition,
                    next_state=st_host["obs"],
                    next_mask=next_host_mask,
                    done=0.0,
                    warmup_frac=cfg.warmup_frac,
                )
                pending_host_transition = None

            (
                a_host,
                host_action_selected_by_agent,
                host_learning_mask,
                host_selection,
            ) = select_layer_action_with_info(
                host_agent,
                st_host,
                safe_rl_enabled=cfg.safe_rl.enabled,
                deterministic=False,
                count_step=True,
            )
            env.host_select(
                int(a_host),
                action_selection=(
                    host_selection
                    if cfg.safe_rl.enabled
                    else None
                ),
            )

            st_vm, ok_vm = env.get_vm_state_for_current_task()
            if not ok_vm:
                break

            if cfg.safe_rl.enabled and pending_vm_transition is not None:
                next_vm_mask = layer_learning_action_mask(
                    st_vm,
                    safe_rl_enabled=True,
                )
                _commit_safe_pending_transition(
                    vm_agent,
                    pending_vm_transition,
                    next_state=st_vm["obs"],
                    next_mask=next_vm_mask,
                    done=0.0,
                    warmup_frac=cfg.warmup_frac,
                )
                pending_vm_transition = None

            (
                a_vm,
                vm_action_selected_by_agent,
                vm_learning_mask,
                vm_selection,
            ) = select_layer_action_with_info(
                vm_agent,
                st_vm,
                safe_rl_enabled=cfg.safe_rl.enabled,
                deterministic=False,
                count_step=True,
            )
            r_host, r_vm, info_task = env.vm_assign(
                int(a_vm),
                action_selection=(
                    vm_selection
                    if cfg.safe_rl.enabled
                    else None
                ),
            )

            if cfg.safe_rl.enabled:
                learning_r_host = float(
                    info_task.get(
                        "total_performance_reward",
                        info_task.get(
                            "performance_reward_host",
                            r_host,
                        ),
                    )
                )
                learning_r_vm = float(
                    info_task.get(
                        "total_performance_reward",
                        info_task.get(
                            "performance_reward_vm",
                            r_vm,
                        ),
                    )
                )
                learning_c_host = float(
                    info_task.get("safety_cost", 0.0)
                )
                learning_c_vm = float(
                    info_task.get("safety_cost", 0.0)
                )
            else:
                learning_r_host = float(r_host)
                learning_r_vm = float(r_vm)

            phase_host_rewards.append(learning_r_host)
            phase_vm_rewards.append(learning_r_vm)
            if cfg.safe_rl.enabled:
                for key in (
                    "safety_cost",
                    "deadline_violation_cost",
                    "fuzzy_lateness_cost",
                    "process_risk_cost",
                    "deadline_violation_count",
                    "completed_workflow_count",
                ):
                    phase_safety[key] += info_task.get(key, 0)
                phase_safety["predicted_deadline_violation_count"] = max(
                    phase_safety["predicted_deadline_violation_count"],
                    int(
                        info_task.get(
                            "predicted_deadline_violation_count",
                            0,
                        )
                    ),
                )
                margin = float(
                    info_task.get("min_fuzzy_safety_margin", 0.0)
                )
                if phase_safety["min_fuzzy_safety_margin"] is None:
                    phase_safety["min_fuzzy_safety_margin"] = margin
                else:
                    phase_safety["min_fuzzy_safety_margin"] = min(
                        phase_safety["min_fuzzy_safety_margin"],
                        margin,
                    )

            # replay 第 3 字段严格使用环境实际执行动作；proposed action
            # 与探索类型只追加为审计字段。安全模式延迟到下一次同层决策再写入，
            # 使 Host/VM 的 Q_r/Q_c 能学习长期 bootstrap；旧模式保持历史
            # 单步终止 transition。
            host_action_audit = finalize_action_audit(
                host_selection,
                info_task.get("host_shield_decision", {}),
            )
            vm_action_audit = finalize_action_audit(
                vm_selection,
                info_task.get("vm_shield_decision", {}),
            )
            host_replay_metadata = safe_replay_metadata(
                st_host,
                host_action_audit,
                info_task,
                shield_decision=info_task.get(
                    "host_shield_decision",
                    {},
                ),
            )
            vm_replay_metadata = safe_replay_metadata(
                st_vm,
                vm_action_audit,
                info_task,
                shield_decision=info_task.get(
                    "vm_shield_decision",
                    {},
                ),
            )
            task_reward_components_host = (
                performance_reward_components(
                    info_task,
                    total_performance_reward=learning_r_host,
                )
            )
            task_reward_components_vm = (
                performance_reward_components(
                    info_task,
                    total_performance_reward=learning_r_vm,
                )
            )
            host_replay_action = int(
                host_action_audit["executed_action"]
            )
            vm_replay_action = int(
                vm_action_audit["executed_action"]
            )
            if cfg.safe_rl.enabled:
                # Stage 10 起 fallback 也作为明确分类的控制器经验保存；
                # executed action 必须 hard-legal，但在空 final mask 时不伪装
                # 成 RL 提议。Q_r/Q_c 仍统一使用 executed action。
                pending_host_transition = {
                    "state": np.array(
                        st_host["obs"],
                        copy=True,
                    ),
                    "mask": np.array(
                        host_learning_mask,
                        copy=True,
                    ),
                    "action": host_replay_action,
                    "reward": learning_r_host,
                    "cost": learning_c_host,
                    "proposed_action": host_action_audit[
                        "proposed_action"
                    ],
                    "action_source": host_action_audit[
                        "action_source"
                    ],
                    "policy_selection_type": (
                        host_action_audit[
                            "policy_selection_type"
                        ]
                    ),
                    "action_modified": host_action_audit[
                        "action_modified"
                    ],
                    "performance_reward_components": (
                        task_reward_components_host
                    ),
                    **host_replay_metadata,
                }
                pending_vm_transition = {
                    "state": np.array(
                        st_vm["obs"],
                        copy=True,
                    ),
                    "mask": np.array(
                        vm_learning_mask,
                        copy=True,
                    ),
                    "action": vm_replay_action,
                    "reward": learning_r_vm,
                    "cost": learning_c_vm,
                    "proposed_action": vm_action_audit[
                        "proposed_action"
                    ],
                    "action_source": vm_action_audit[
                        "action_source"
                    ],
                    "policy_selection_type": (
                        vm_action_audit[
                            "policy_selection_type"
                        ]
                    ),
                    "action_modified": vm_action_audit[
                        "action_modified"
                    ],
                    "performance_reward_components": (
                        task_reward_components_vm
                    ),
                    **vm_replay_metadata,
                }
            else:
                z_host_s = np.zeros_like(st_host["obs"])
                z_host_m = np.zeros_like(st_host["mask"])
                host_agent.remember(
                    st_host["obs"],
                    st_host["mask"],
                    host_replay_action,
                    learning_r_host,
                    z_host_s,
                    z_host_m,
                    1.0,
                )
                if warmup_ready(host_agent, cfg.warmup_frac):
                    host_agent.update()

                z_vm_s = np.zeros_like(st_vm["obs"])
                z_vm_m = np.zeros_like(st_vm["mask"])
                vm_agent.remember(
                    st_vm["obs"],
                    st_vm["mask"],
                    vm_replay_action,
                    learning_r_vm,
                    z_vm_s,
                    z_vm_m,
                    1.0,
                )
                if warmup_ready(vm_agent, cfg.warmup_frac):
                    vm_agent.update()

        # phase 内任务分配结束，环境推进到下一 phase，并返回 manager reward 与阶段信息。
        r_manager_raw, pinfo = env.finish_phase_and_advance()
        if metric_store is not None:
            episode_phase_metric_records.append(dict(pinfo))
        if cfg.safe_rl.enabled:
            for key in (
                "safety_cost",
                "deadline_violation_cost",
                "fuzzy_lateness_cost",
                "process_risk_cost",
                "deadline_violation_count",
                "completed_workflow_count",
            ):
                phase_safety[key] += pinfo.get(key, 0)
            phase_safety["predicted_deadline_violation_count"] = max(
                phase_safety["predicted_deadline_violation_count"],
                int(
                    pinfo.get(
                        "predicted_deadline_violation_count",
                        0,
                    )
                ),
            )
            margin = float(pinfo.get("min_fuzzy_safety_margin", 0.0))
            if phase_safety["min_fuzzy_safety_margin"] is None:
                phase_safety["min_fuzzy_safety_margin"] = margin
            else:
                phase_safety["min_fuzzy_safety_margin"] = min(
                    phase_safety["min_fuzzy_safety_margin"],
                    margin,
                )
        assign_cnt = int(pinfo.get("assign_cnt", 0))
        phase_size_sum_mi = float(pinfo.get("phase_size_sum_mi", 0.0))

        if cfg.safe_rl.enabled and env.done_flag:
            if pending_host_transition is not None:
                _commit_safe_pending_transition(
                    host_agent,
                    pending_host_transition,
                    next_state=np.zeros_like(
                        pending_host_transition["state"]
                    ),
                    next_mask=np.zeros_like(
                        pending_host_transition["mask"]
                    ),
                    done=1.0,
                    warmup_frac=cfg.warmup_frac,
                )
                pending_host_transition = None
            if pending_vm_transition is not None:
                _commit_safe_pending_transition(
                    vm_agent,
                    pending_vm_transition,
                    next_state=np.zeros_like(
                        pending_vm_transition["state"]
                    ),
                    next_mask=np.zeros_like(
                        pending_vm_transition["mask"]
                    ),
                    done=1.0,
                    warmup_frac=cfg.warmup_frac,
                )
                pending_vm_transition = None

        r_vm_phase_mean = float(np.mean(phase_vm_rewards)) if len(phase_vm_rewards) > 0 else 0.0
        r_host_phase_mean = float(np.mean(phase_host_rewards)) if len(phase_host_rewards) > 0 else 0.0

        # 记录 phase 级日志；episode 级字段留空。
        logger.log(
            step=global_step, episode=episode_idx, type="phase",
            ep_length="", env_time=pinfo.get("current_time", 0.0),
            episode_energy="",
            wf_completed="", wf_target="",
            eps_vm=vm_agent.epsilon(),
            eps_host=host_agent.epsilon(),
            eps_mgr=manager_agent.epsilon(),
            assign_cnt=assign_cnt,
            phase_size_sum_mi=phase_size_sum_mi,
            r_vm_phase_mean=r_vm_phase_mean,
            r_host_phase_mean=r_host_phase_mean,
            r_manager_raw=float(r_manager_raw),
            phase_energy=float(pinfo.get("phase_energy", 0.0)),
            phase_delay=float(pinfo.get("phase_delay", 0.0)),
            phase_energy_cost=float(pinfo.get("phase_energy_cost", 0.0)),
            phase_delay_cost=float(pinfo.get("phase_delay_cost", 0.0)),
            manager_reward_old=float(pinfo.get("manager_reward_old", 0.0)),
            manager_reward_new=float(pinfo.get("manager_reward_new", 0.0)),
            performance_reward=float(
                pinfo.get("performance_reward", r_manager_raw)
            ),
            energy_reward=float(
                pinfo.get("energy_reward", 0.0)
            ),
            completion_reward=float(
                pinfo.get("completion_reward", 0.0)
            ),
            waiting_reward=float(
                pinfo.get("waiting_reward", 0.0)
            ),
            utilization_reward=float(
                pinfo.get("utilization_reward", 0.0)
            ),
            communication_reward=float(
                pinfo.get("communication_reward", 0.0)
            ),
            total_performance_reward=float(
                pinfo.get(
                    "total_performance_reward",
                    pinfo.get("performance_reward", r_manager_raw),
                )
            ),
            safety_cost=float(phase_safety["safety_cost"]),
            deadline_violation_cost=float(
                phase_safety["deadline_violation_cost"]
            ),
            fuzzy_lateness_cost=float(
                phase_safety["fuzzy_lateness_cost"]
            ),
            process_risk_cost=float(
                phase_safety["process_risk_cost"]
            ),
            deadline_violation_count=int(
                phase_safety["deadline_violation_count"]
            ),
            completed_workflow_count=int(
                phase_safety["completed_workflow_count"]
            ),
            predicted_deadline_violation_count=int(
                phase_safety["predicted_deadline_violation_count"]
            ),
            min_fuzzy_safety_margin=float(
                pinfo.get("minimum_fuzzy_safety_margin", 0.0)
            ),
            mean_fuzzy_safety_margin=float(
                pinfo.get("mean_fuzzy_safety_margin", 0.0)
            ),
            risk_workflow_ratio=float(
                pinfo.get("risk_workflow_ratio", 0.0)
            ),
            predicted_violation_rate=float(
                pinfo.get("predicted_violation_rate", 0.0)
            ),
            selected_heuristic_id=pinfo.get(
                "selected_heuristic_id",
                "",
            ),
            heuristic_source=pinfo.get(
                "heuristic_source",
                "",
            ),
            llm_rule_version=pinfo.get(
                "llm_rule_version",
                "",
            ),
            ready_task_ordering=pinfo.get(
                "ready_task_ordering",
                [],
            ),
            heuristic_shield_intervention_count=int(
                pinfo.get(
                    "heuristic_shield_intervention_count",
                    0,
                )
            ),
            heuristic_shield_intervention_rate=float(
                pinfo.get(
                    "heuristic_shield_intervention_rate",
                    0.0,
                )
            ),
            manager_action_source=(
                m_action_audit["action_source"]
                if m_action_audit is not None
                else ""
            ),
            random_safe_exploration_count=int(
                pinfo.get(
                    "random_safe_exploration_count",
                    0,
                )
            ),
            greedy_safe_action_count=int(
                pinfo.get("greedy_safe_action_count", 0)
            ),
            shield_correction_count=int(
                pinfo.get("shield_correction_count", 0)
            ),
            fallback_action_count=int(
                pinfo.get("fallback_action_count", 0)
            ),
            shield_record_count=int(
                pinfo.get("shield_record_count", 0)
            ),
            shield_intervention_count=int(
                pinfo.get("shield_intervention_count", 0)
            ),
            shield_fallback_count=int(
                pinfo.get("shield_fallback_count", 0)
            ),
            host_action_modified_count=int(
                pinfo.get("host_action_modified_count", 0)
            ),
            vm_action_modified_count=int(
                pinfo.get("vm_action_modified_count", 0)
            ),
            current_lambda=float(
                lagrange_controller.current_lambda
            ),
            mean_safety_cost=float(
                lagrange_controller.mean_safety_cost
            ),
            cost_budget=float(
                lagrange_controller.cost_budget
            ),
            constraint_gap=float(
                lagrange_controller.constraint_gap
            ),
            lambda_update_count=int(
                lagrange_controller.lambda_update_count
            ),
            eval_deadline_violation_rate="",
            eval_zero_violation_pass="",
            ep_total_lateness="",
            ep_avg_lateness="",
            wf_avg_lateness="",
            ep_wf_lateness_sum="",
            eval_vm="", eval_host="", eval_mgr="", eval_energy="",
        )

        sH_next = env.get_manager_state()
        m_mask_next = env.get_manager_action_mask()

        # 如果某个 phase 没有实际分配任务，可以选择跳过 manager 更新，避免无效样本进入 replay buffer。
        if not (cfg.skip_manager_update_if_zero_assign and assign_cnt == 0):
            learning_r_manager = (
                float(
                    pinfo.get(
                        "total_performance_reward",
                        pinfo.get(
                            "performance_reward",
                            r_manager_raw,
                        ),
                    )
                )
                if cfg.safe_rl.enabled
                else float(r_manager_raw)
            )
            if cfg.safe_rl.enabled:
                manager_replay_info = {
                    **dict(pinfo),
                    "deadline_violation_count": int(
                        phase_safety[
                            "deadline_violation_count"
                        ]
                    ),
                }
                manager_replay_metadata = (
                    safe_replay_metadata(
                        {
                            "mask": m_mask,
                            "legal_action_mask": m_mask,
                            "safety_action_mask": m_mask,
                            "final_action_mask": m_mask,
                        },
                        m_action_audit,
                        manager_replay_info,
                    )
                )
                # ``mask`` 位置参数已经是本 transition 的 final mask。
                # metadata 中的同名字段只用于审计，不能作为 remember()
                # 不支持的重复关键字再次传入。
                manager_replay_metadata.pop(
                    "final_action_mask",
                    None,
                )
                manager_agent.remember(
                    sH,
                    m_mask,
                    int(m_act),
                    learning_r_manager,
                    sH_next,
                    m_mask_next,
                    float(env.done_flag),
                    cost=float(phase_safety["safety_cost"]),
                    proposed_action=m_action_audit[
                        "proposed_action"
                    ],
                    action_source=m_action_audit[
                        "action_source"
                    ],
                    policy_selection_type=m_action_audit[
                        "policy_selection_type"
                    ],
                    action_modified=bool(
                        m_action_audit["action_modified"]
                    ),
                    performance_reward_components=(
                        performance_reward_components(
                            pinfo,
                            total_performance_reward=(
                                learning_r_manager
                            ),
                        )
                    ),
                    **manager_replay_metadata,
                )
            else:
                manager_agent.remember(
                    sH,
                    m_mask,
                    int(m_act),
                    learning_r_manager,
                    sH_next,
                    m_mask_next,
                    float(env.done_flag),
                )
            if warmup_ready(manager_agent, cfg.warmup_frac):
                manager_agent.update()

        sH = sH_next
        m_mask = m_mask_next

        episode_steps += 1
        global_step += 1

        # 如果 episode 未结束，manager 为下一 phase 选择动作。
        if not env.done_flag:
            m_act, m_action_audit = (
                _select_manager_training_action(
                    manager_agent,
                    sH,
                    m_mask,
                    safe_rl_enabled=cfg.safe_rl.enabled,
                )
            )
            manager_apply_action(env, m_act)

    # hard_max_steps 截断时把尚未获得同层下一状态的安全样本作为截断终止
    # transition 写入，避免静默丢失最后一次 Host/VM 决策。
    if cfg.safe_rl.enabled:
        if pending_host_transition is not None:
            _commit_safe_pending_transition(
                host_agent,
                pending_host_transition,
                next_state=np.zeros_like(
                    pending_host_transition["state"]
                ),
                next_mask=np.zeros_like(
                    pending_host_transition["mask"]
                ),
                done=1.0,
                warmup_frac=cfg.warmup_frac,
            )
        if pending_vm_transition is not None:
            _commit_safe_pending_transition(
                vm_agent,
                pending_vm_transition,
                next_state=np.zeros_like(
                    pending_vm_transition["state"]
                ),
                next_mask=np.zeros_like(
                    pending_vm_transition["mask"]
                ),
                done=1.0,
                warmup_frac=cfg.warmup_frac,
            )

    # 训练达到 episode 上限或 step 上限后，保存最终模型。
    vm_final = os.path.join(cfg.save_dir, "vm_final.pth")
    host_final = os.path.join(cfg.save_dir, "host_final.pth")
    mgr_final = os.path.join(cfg.save_dir, "manager_final.pth")
    _save_agent_checkpoint(
        vm_agent,
        vm_final,
        lagrange_controller,
    )
    _save_agent_checkpoint(
        host_agent,
        host_final,
        lagrange_controller,
    )
    _save_agent_checkpoint(
        manager_agent,
        mgr_final,
        lagrange_controller,
    )
    final_pipeline_checkpoint = None
    if training_controller is not None:
        final_checkpoint_metadata = (
            _checkpoint_runtime_metadata(
                cfg=cfg,
                env=env,
                agents=agents_by_layer,
                training_controller=training_controller,
            )
        )
        final_pipeline_checkpoint = save_pipeline_checkpoint(
            os.path.join(cfg.save_dir, "pipeline_final"),
            controller=training_controller,
            agents=agents_by_layer,
            lagrange_controller=lagrange_controller,
            global_step=global_step,
            next_episode=(
                training_controller.total_episode_count
            ),
            best_model_metrics=(
                best_model_metrics.to_dict()
                if best_model_metrics is not None
                else None
            ),
            replay_metadata=final_checkpoint_metadata[
                "replay_metadata"
            ],
            heuristic_library_version=(
                final_checkpoint_metadata[
                    "heuristic_library_version"
                ]
            ),
            config_snapshot=final_checkpoint_metadata[
                "config_snapshot"
            ],
        )

    logger.close()
    if validation_pool is not None:
        validation_pool.close()
    print(
        "Training finished.\n"
        f"VM final model: {vm_final}\n"
        f"Host final model: {host_final}\n"
        f"Manager final model: {mgr_final}\n"
        f"Best eval models: {best_ckpt_vm} / {best_ckpt_host} / {best_ckpt_mgr}\n"
        f"Safe training pipeline checkpoint: "
        f"{final_pipeline_checkpoint or 'disabled'}\n"
        f"Log: {cfg.log_path}"
    )
