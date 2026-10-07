"""Independent FCFS entry point; production Host/VM training loop is reused."""
import argparse
from hrl_mix.train_config import build_train_config, validate_single_deadline_cache_paths
from hrl_mix.train_runner import train
from .fcfs_env import FCFSManagerAblationEnv
from .runtime import ManagerPlaceholder, save_fcfs_checkpoint


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-scenario', choices=['SS', 'SM', 'SL'], required=True)
    p.add_argument('--ddl', choices=['T', 'M', 'L'], required=True)
    p.add_argument('--episodes', type=int, choices=[600], default=600)
    p.add_argument('--optimizer-seed', type=int, default=0)
    p.add_argument('--deadline-cache', action='append', required=True)
    p.add_argument('--safe-rl-energy-reward-scale', type=float, default=.002)
    p.add_argument('--safe-rl-cost-budget', type=float, default=.02)
    p.add_argument('--safe-rl-lambda-init', type=float, default=.5)
    p.add_argument('--safe-rl-lambda-lr', type=float, default=.02)
    return p


def training_kwargs(args):
    caches = validate_single_deadline_cache_paths('single', args.deadline_cache,
                                                source_scenario=args.source_scenario)
    return dict(protocol='single', source_scenario=args.source_scenario, ddl=args.ddl,
                max_episodes=args.episodes, optimizer_seed=args.optimizer_seed,
                deadline_cache_paths=caches, safe_rl_enabled=True,
                safe_rl_shield_enabled=True, safe_rl_state_enabled=True,
                safe_rl_dynamic_lambda_enabled=True, safe_rl_heuristic_manager_enabled=False,
                manager_heuristic_llm_only=False, safe_rl_curriculum_enabled=False,
                safe_rl_energy_reward_scale=args.safe_rl_energy_reward_scale,
                safe_rl_cost_budget=args.safe_rl_cost_budget,
                safe_rl_lambda_init=args.safe_rl_lambda_init,
                safe_rl_lambda_lr=args.safe_rl_lambda_lr)


def build_fcfs_config(**kwargs):
    return build_train_config(**kwargs, output_namespace='ablations/fcfs_manager')


def main(argv=None):
    args = parser().parse_args(argv)
    return train(**training_kwargs(args), environment_class=FCFSManagerAblationEnv,
                 config_builder=build_fcfs_config, manager_factory=ManagerPlaceholder,
                 checkpoint_writer=save_fcfs_checkpoint, validation_workers=1)


if __name__ == '__main__':
    main()
