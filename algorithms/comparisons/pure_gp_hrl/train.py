"""Train the unchanged three-layer Safe-HRL with a frozen GP-only library."""
import argparse
import json
from pathlib import Path

from algorithms.llm_safe_hrl.base.heuristic_admission import file_sha256
from algorithms.llm_safe_hrl.base.manager_heuristics import load_manager_heuristic_library
from algorithms.llm_safe_hrl.run_context import resolve_deadline_setting
from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
from hrl_mix.model_selection import save_best_checkpoint_bundle
from hrl_mix.train_config import build_train_config, validate_single_deadline_cache_paths
from hrl_mix.train_runner import train
from .generate import write_json


def identity():
    return dict(experiment_type='ablation', ablation='pure_gp_hrl', generator='pure_gp',
                llm_enabled=False, cmaes_enabled=False, parameter_optimization_enabled=False,
                manager_trainable=True,
                manager_policy='unchanged_heuristic_selection',
                task_ordering='manager_selected_pure_gp_rule', retrained_under_gp=True)


def validate_gp_library(path, *, source_scenario=None, ddl=None):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('method_id') != 'pure_gp_hrl' or payload.get('generator') != 'pure_gp':
        raise ValueError('GP-HRL requires a GP-generated library')
    if payload.get('cmaes_enabled') is not False or payload.get('parameter_optimization_enabled') is not False:
        raise ValueError('pure GP library must disable external parameter optimization')
    if payload.get('smoke_only'):
        raise ValueError('smoke library cannot be used for formal HRL training/evaluation')
    if payload.get('llm_enabled') is not False:
        raise ValueError('GP library must declare llm_enabled=false')
    context = resolve_experiment_protocol('single', source_scenario=(
        source_scenario or payload['experiment_protocol']['source_scenario']))
    if context.identity() != payload['experiment_protocol']:
        raise ValueError('GP library source/protocol mismatch')
    if ddl is not None and resolve_deadline_setting(ddl).identity() != payload['deadline_setting']:
        raise ValueError('GP library DDL mismatch')
    if any(rule.get('generator') != 'pure_gp' for rule in payload['llm_rules']):
        raise ValueError('GP-HRL may not mix LLM or traditional rules into its K actions')
    library = load_manager_heuristic_library(path, include_traditional=False,
                                              expected_protocol_identity=context.identity())
    if any(not rule.available for rule in library):
        raise ValueError('GP library contains unavailable rules')
    return payload, context


def build_gp_config(**kwargs):
    payload, _ = validate_gp_library(kwargs['manager_heuristic_manifest'],
                                     source_scenario=kwargs['source_scenario'], ddl=kwargs['ddl'])
    cfg = build_train_config(**kwargs, output_namespace='ablations/pure_gp_hrl')
    write_json(Path(cfg.save_dir) / 'gp_ablation.json',
               dict(**identity(), rule_count=len(payload['llm_rules']),
                    gp_library=str(Path(kwargs['manager_heuristic_manifest']).resolve()),
                    gp_library_sha256=file_sha256(kwargs['manager_heuristic_manifest']),
                    reference_library=payload['reference_library'],
                    reference_library_sha256=payload['reference_library_sha256']))
    return cfg


def save_gp_checkpoint(*args, **kwargs):
    result = save_best_checkpoint_bundle(*args, **kwargs)
    directory = Path(args[0] if args else kwargs['directory'])
    path = directory / 'best_checkpoint_manifest.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload.update(json.loads((directory / 'gp_ablation.json').read_text(encoding='utf-8')))
    write_json(path, payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gp-library', type=Path, required=True)
    parser.add_argument('--deadline-cache', action='append', required=True)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--optimizer-seed', type=int, default=0)
    parser.add_argument('--validation-workers', type=int, default=3)
    parser.add_argument('--safe-rl-energy-reward-scale', type=float, default=.001)
    parser.add_argument('--safe-rl-cost-budget', type=float, default=.01)
    parser.add_argument('--safe-rl-lambda-init', type=float, default=1.)
    parser.add_argument('--safe-rl-lambda-lr', type=float, default=.05)
    parser.add_argument('--safe-rl-offline-pretrain-manifest')
    parser.add_argument('--safe-rl-offline-pretrain-epochs', type=int, default=5)
    parser.add_argument('--safe-rl-offline-pretrain-behavior-cloning', action='store_true')
    parser.add_argument('--safe-rl-training-pipeline-plan')
    parser.add_argument('--safe-rl-training-resume-checkpoint')
    parser.add_argument('--safe-rl-curriculum', action='store_true')
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error('episodes must be positive')
    payload, context = validate_gp_library(args.gp_library)
    caches = validate_single_deadline_cache_paths('single', args.deadline_cache,
                                                 source_scenario=context.source_scenario)
    return train(protocol='single', source_scenario=context.source_scenario,
                 ddl=payload['deadline_setting']['ddl_code'], max_episodes=args.episodes,
                 optimizer_seed=args.optimizer_seed, deadline_cache_paths=caches,
                 safe_rl_enabled=True, safe_rl_shield_enabled=True, safe_rl_state_enabled=True,
                 safe_rl_dynamic_lambda_enabled=True, safe_rl_heuristic_manager_enabled=True,
                 manager_heuristic_llm_only=True, manager_heuristic_manifest=str(args.gp_library.resolve()),
                 safe_rl_energy_reward_scale=args.safe_rl_energy_reward_scale,
                 safe_rl_cost_budget=args.safe_rl_cost_budget,
                 safe_rl_lambda_init=args.safe_rl_lambda_init, safe_rl_lambda_lr=args.safe_rl_lambda_lr,
                 safe_rl_offline_pretrain_manifest=args.safe_rl_offline_pretrain_manifest,
                 safe_rl_offline_pretrain_epochs=args.safe_rl_offline_pretrain_epochs,
                 safe_rl_offline_pretrain_behavior_cloning=args.safe_rl_offline_pretrain_behavior_cloning,
                 safe_rl_training_pipeline_plan=args.safe_rl_training_pipeline_plan,
                 safe_rl_training_resume_checkpoint=args.safe_rl_training_resume_checkpoint,
                 safe_rl_curriculum_enabled=args.safe_rl_curriculum,
                 config_builder=build_gp_config, checkpoint_writer=save_gp_checkpoint,
                 validation_workers=args.validation_workers)


if __name__ == '__main__':
    main()
