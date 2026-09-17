# -*- coding: utf-8 -*-
"""
train.py 训练入口模块。

整体调用关系：
1. 用户在命令行运行：
   python -m hrl_mix.train --scenario SS --ddl T --episodes 1
2. Python 进入本文件，执行 main()。
3. main() 只负责解析命令行参数，不包含训练细节。
4. main() 将参数传给 hrl_mix.train_runner.train()。
5. train_runner.py 再调用：
   - train_config.py：构造场景、DDL、超参数、路径配置。
   - train_utils.py：设置随机种子、执行 manager 动作、计算指标等工具函数。
   - train_eval.py：训练过程中做多 seed 评估。

文件职责：
- train.py = 入口，只做参数解析和调用转发。
- train_runner.py = 训练主程序，包含环境、Agent、训练循环、日志与保存。
- train_config.py = 配置中心，统一管理场景、DDL、资源规模、超参数和输出路径。
- train_utils.py = 工具函数，放置可复用的小函数。
- train_eval.py = 评估模块，封装训练中的评估流程。
"""
from __future__ import annotations

import argparse  # Python 标准库中的命令行参数解析模块

from hrl_mix.train_runner import train
from hrl_mix.train_config import (
    parse_deadline_cache_overrides,
    validate_single_deadline_cache_paths,
)


def main(argv=None):
    """解析命令行参数，并把参数转交给训练主函数"""
    parser = argparse.ArgumentParser(description="Train the HRL Mix model for a scenario/deadline setting.")
    parser.add_argument(
        "--scenario",
        default=None,
        help=(
            "Legacy scenario alias. New experiments should use "
            "--protocol with --source-scenario or --resource-scale."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("single", "multi"),
        default="single",
        help="Isolated LLM-SAFE-DRL experiment protocol.",
    )
    parser.add_argument(
        "--source-scenario",
        default=None,
        help="Single protocol source scenario: SS, SM, or SL.",
    )
    parser.add_argument(
        "--resource-scale",
        choices=("S", "M", "L"),
        default=None,
        help="Multi protocol resource group: S, M, or L.",
    )
    parser.add_argument(
        "--ddl",
        default="T",
        help="Deadline condition: T/M/L or Tight/Medium/Loose.",
    )
    parser.add_argument(
        "--optimizer-seed",
        type=int,
        default=0,
        help="Algorithm/network seed; environment seeds remain protocol-controlled.",
    )
    parser.add_argument(
        "--deadline-cache",
        action="append",
        default=None,
        help="Single cache mapping as SCENARIO=PATH; repeat per test scenario.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the default episode count; useful for smoke tests.",
    )
    parser.add_argument(
        "--safe-rl",
        action="store_true",
        help=(
            "Enable CMDP reward-cost output and stage-7 dual Q_r/Q_c "
            "value learning, stage-9 safe-action audit, and stage-10 "
            "fuzzy-energy reward/versioned safe replay. Host/VM "
            "fuzzy-DDL pruning requires --safe-rl-shield. Lambda "
            "remains fixed unless "
            "--safe-rl-dynamic-lambda is also enabled. Default is off, "
            "so legacy HRL reward/replay semantics remain unchanged."
        ),
    )
    parser.add_argument(
        "--safe-rl-shield",
        action="store_true",
        help=(
            "Enable the stage-4 fuzzy DDL Host/VM safety shield. "
            "Requires --safe-rl; default is off."
        ),
    )
    parser.add_argument(
        "--safe-rl-state",
        action="store_true",
        help=(
            "Enable the stage-6 Manager/Host/VM safety observation "
            "extension. Requires --safe-rl; default is off."
        ),
    )
    parser.add_argument(
        "--safe-rl-dynamic-lambda",
        action="store_true",
        help=(
            "Enable the stage-8 shared episode/EMA dynamic Lagrange "
            "controller. Requires --safe-rl; default is off."
        ),
    )
    parser.add_argument(
        "--safe-rl-heuristic-manager",
        action="store_true",
        help=(
            "Enable stage-11 heuristic_selection_mode: Manager "
            "chooses an admitted FCFS/SJF/MCF/HUR/EDF/SeEvo "
            "ready-task rule index. Requires --safe-rl, "
            "--safe-rl-shield, and --safe-rl-state. Default is off; "
            "the legacy five-rule weight-delta Manager is preserved."
        ),
    )
    parser.add_argument(
        "--llm-only-heuristics",
        action="store_true",
        help=(
            "Remove the five built-in FCFS/SJF/MCF/HUR/EDF action "
            "slots so the Manager can only select available SeEvo LLM "
            "rules from safe-admission or Top-K manifests. Requires "
            "--safe-rl-heuristic-manager. Default is "
            "off. The Manager action dimension and heuristic action "
            "schema version both change, so runs with and without "
            "this flag get separate output directories and are not "
            "checkpoint-compatible."
        ),
    )
    parser.add_argument(
        "--manager-heuristic-manifest",
        default=None,
        help=(
            "Optional versioned safe-heuristic manifest. Requires "
            "--safe-rl-heuristic-manager. If omitted, the manifest is "
            "selected from the isolated Single/Multi protocol artifact "
            "namespace. Legacy Python callers retain the old resource-code "
            "fallback, but formal CLI runs never use it."
        ),
    )
    parser.add_argument(
        "--llm-run-manifest",
        default=None,
        help=(
            "run_manifest.json from one completed, matching SeEvo run. "
            "The protocol identity and DDL setting are validated before "
            "its heuristic library is used. Mutually exclusive with "
            "--manager-heuristic-manifest."
        ),
    )
    parser.add_argument(
        "--offline-pretrain-manifest",
        default=None,
        help=(
            "Versioned safe-demonstration dataset manifest. Providing "
            "this enables offline Q_r/Q_c initialization before the "
            "online loop and requires all safe-RL, shield, state, and "
            "heuristic-Manager switches."
        ),
    )
    parser.add_argument(
        "--offline-pretrain-epochs",
        type=int,
        default=5,
        help="Offline demonstration pretraining epochs (default: 5).",
    )
    parser.add_argument(
        "--offline-pretrain-behavior-cloning",
        action="store_true",
        help=(
            "Additionally initialize the Q_r action preference from "
            "non-fallback executed demonstration actions."
        ),
    )
    parser.add_argument(
        "--offline-pretrain-no-q-r",
        action="store_true",
        help="Disable the offline Q_r Bellman loss.",
    )
    parser.add_argument(
        "--offline-pretrain-no-q-c",
        action="store_true",
        help="Disable the offline Q_c Bellman loss.",
    )
    parser.add_argument(
        "--safe-training-pipeline-config",
        default=None,
        help=(
            "Versioned JSON plan for the five-stage safe-HRL "
            "training pipeline. The plan explicitly defines "
            "demonstration generation, offline pretraining, "
            "shielded online training, curriculum, cross-seed "
            "training, seed splits, transitions, and metrics. "
            "Requires all safe-RL/shield/state/dynamic-lambda/"
            "heuristic-Manager switches."
        ),
    )
    parser.add_argument(
        "--without-curriculum",
        action="store_true",
        help=(
            "Ablation: keep all Safe-HRL components and 600 total episodes, "
            "but always train on the formal target scenario configuration."
        ),
    )
    parser.add_argument(
        "--safe-training-resume",
        default=None,
        help=(
            "safe_training_checkpoint.json to resume the current "
            "online curriculum stage. Requires "
            "--safe-training-pipeline-config."
        ),
    )
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=1,
        help=(
            "Worker processes for periodic validation episodes. 1 "
            "(default) keeps the serial path unchanged; 0 auto-selects "
            "cpu_count-2. Workers run on the same device as the parent, "
            "so validation results and best-checkpoint selection are "
            "bit-identical to the serial path. Set "
            "SAFE_HRL_VALIDATION_PARALLEL_AUDIT=1 to re-run every batch "
            "serially and assert that."
        ),
    )
    args = parser.parse_args(argv)
    if args.validation_workers < 0:
        parser.error("--validation-workers must be non-negative")
    if args.llm_run_manifest and args.manager_heuristic_manifest:
        parser.error(
            "--llm-run-manifest and --manager-heuristic-manifest are "
            "mutually exclusive"
        )
    if args.episodes not in (None, 600):
        parser.error("formal Safe-HRL training requires exactly 600 total episodes")
    source_scenario = args.source_scenario or args.scenario or "SS"
    deadline_cache_paths = parse_deadline_cache_overrides(
        args.deadline_cache,
        default_scenario=source_scenario,
    )
    deadline_cache_paths = validate_single_deadline_cache_paths(
        args.protocol,
        deadline_cache_paths,
        source_scenario=(source_scenario if args.protocol == "single" else None),
    )
    train(
        scenario=args.scenario,
        ddl=args.ddl,
        max_episodes=args.episodes,
        safe_rl_enabled=args.safe_rl,
        safe_rl_shield_enabled=args.safe_rl_shield,
        safe_rl_state_enabled=args.safe_rl_state,
        safe_rl_dynamic_lambda_enabled=(
            args.safe_rl_dynamic_lambda
        ),
        safe_rl_heuristic_manager_enabled=(
            args.safe_rl_heuristic_manager
        ),
        manager_heuristic_llm_only=args.llm_only_heuristics,
        manager_heuristic_manifest=(
            args.manager_heuristic_manifest
        ),
        llm_run_manifest=args.llm_run_manifest,
        safe_rl_offline_pretrain_manifest=(
            args.offline_pretrain_manifest
        ),
        safe_rl_offline_pretrain_epochs=(
            args.offline_pretrain_epochs
        ),
        safe_rl_offline_pretrain_behavior_cloning=(
            args.offline_pretrain_behavior_cloning
        ),
        safe_rl_offline_pretrain_q_r=(
            not args.offline_pretrain_no_q_r
        ),
        safe_rl_offline_pretrain_q_c=(
            not args.offline_pretrain_no_q_c
        ),
        safe_rl_training_pipeline_plan=(
            args.safe_training_pipeline_config
        ),
        safe_rl_training_resume_checkpoint=(
            args.safe_training_resume
        ),
        safe_rl_curriculum_enabled=(
            not args.without_curriculum
        ),
        optimizer_seed=args.optimizer_seed,
        deadline_cache_override=deadline_cache_paths.get(source_scenario),
        deadline_cache_paths=deadline_cache_paths,
        protocol=args.protocol,
        source_scenario=args.source_scenario,
        resource_scale=args.resource_scale,
        validation_workers=args.validation_workers,
    )



if __name__ == "__main__":
    main()
