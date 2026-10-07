"""Generate independent FCFS Host/VM demonstrations without Top-K artifacts."""
import json
from pathlib import Path
from hrl_mix.ablations.fcfs_manager.train_fcfs import parser, training_kwargs
from hrl_mix.ablations.fcfs_manager.fcfs_env import FCFSManagerAblationEnv
from hrl_mix.ablations.fcfs_manager.demonstrations import generate_episode
from hrl_mix.ablations.fcfs_manager.runtime import identity
from hrl_mix.train_config import ROOT_DIR, build_train_config, safety_learning_identity
from hrl_mix.model_selection import build_config_snapshot
from hrl_mix.protocol_evaluation import _json_value
from tools.generate_safe_demonstrations import _environment_kwargs


def main(argv=None):
    p = parser()
    p.add_argument('--split', choices=['train', 'validation'], default='train')
    args = p.parse_args(argv)
    cfg = build_train_config(**training_kwargs(args), output_namespace='ablations/fcfs_manager')
    seeds = cfg.train_seeds if args.split == 'train' else cfg.validation_seeds
    output = ROOT_DIR / 'out' / 'safe_demonstrations_fcfs_ablation' / f'{args.source_scenario}_{args.ddl}' / cfg.run_name / args.split
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for seed in seeds:
        env = FCFSManagerAblationEnv(**_environment_kwargs(cfg, seed, seed))
        record = generate_episode(env)
        filename = f'seed_{seed}.json'
        (output / filename).write_text(json.dumps(_json_value(record), ensure_ascii=False), encoding='utf-8')
        episodes.append({'workflow_seed': seed, 'resource_seed': seed, 'file': filename})
    manifest = {**identity(), 'schema': 'fcfs_host_vm_demonstrations_v1',
                **safety_learning_identity(cfg.safe_rl, cfg.energy_reward_scale),
                'config_snapshot': build_config_snapshot(cfg), 'split': args.split,
                'observation_schema': {layer: env.get_observation_schema(layer) for layer in ('host', 'vm')},
                'episodes': episodes}
    (output / 'manifest.json').write_text(json.dumps(_json_value(manifest), ensure_ascii=False, indent=2), encoding='utf-8')
    print(output / 'manifest.json')


if __name__ == '__main__':
    main()
