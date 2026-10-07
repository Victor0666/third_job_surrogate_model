"""Frozen Host/VM evaluation with FCFS, retrained or diagnostic-only."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
from hrl_mix.model_selection import read_best_checkpoint_manifest
from hrl_mix.protocol_evaluation import (
    _load_frozen_agent, _agent_read_only_fingerprint, _append_seed_result,
    build_frozen_scenario_env_kwargs, _json_value,
)
from hrl_mix.train_eval import evaluate_hrl_three_layer_multi_seed
from hrl_mix.train_config import ROOT_DIR, validate_single_deadline_cache_paths
from .fcfs_env import FCFSManagerAblationEnv
from .runtime import identity


def validate_mode(manifest, mode):
    retrained = manifest.get('ablation') == 'fcfs_manager' and manifest.get('retrained_under_fcfs') is True
    if (mode == 'retrained') != retrained:
        raise ValueError('Retrained mode requires an FCFS checkpoint; diagnostic mode requires a Full checkpoint')
    return identity(diagnostic=mode == 'diagnostic')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['diagnostic', 'retrained'], required=True)
    p.add_argument('--checkpoint-manifest', type=Path, required=True)
    p.add_argument('--source-scenario', choices=['SS', 'SM', 'SL'], required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--deadline-cache', action='append', required=True)
    args = p.parse_args(argv)
    context = resolve_experiment_protocol('single', source_scenario=args.source_scenario)
    manifest = read_best_checkpoint_manifest(args.checkpoint_manifest,
                                             expected_protocol_identity=context.identity())
    metadata = validate_mode(manifest, args.mode)
    cfg = manifest['config_snapshot']['config']
    caches = validate_single_deadline_cache_paths('single', args.deadline_cache,
                                                 source_scenario=args.source_scenario)
    agents = {layer: _load_frozen_agent(manifest['resolved_agent_checkpoints'][layer],
                                      replay_metadata=manifest['replay_metadata'][layer], device=args.device)
              for layer in ('host', 'vm')}
    before = {key: _agent_read_only_fingerprint(value) for key, value in agents.items()}
    # No Manager checkpoint is loaded or policy instantiated, even in diagnostic mode.
    placeholder = SimpleNamespace(trainable=False)
    output = ROOT_DIR / 'out' / 'ablations' / 'fcfs_manager' / args.mode / args.source_scenario / args.checkpoint_manifest.resolve().parent.name
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for scenario in context.test_scenarios:
        kwargs = build_frozen_scenario_env_kwargs(
            cfg, context, scenario, '', caches,
            required_manager_mode=cfg['safe_rl']['manager_heuristics']['mode'])
        for suffix in ('csv', 'jsonl'):
            (output / f'scenario_{scenario}_seed_results.{suffix}').unlink(missing_ok=True)
        results[scenario] = evaluate_hrl_three_layer_multi_seed(
            FCFSManagerAblationEnv, kwargs, agents['vm'], agents['host'], placeholder,
            tuple(range(201, 231)), return_safety_metrics=True,
            seed_result_callback=lambda seed, result, scenario=scenario: _append_seed_result(output, scenario, seed, result))
    assert before == {key: _agent_read_only_fingerprint(value) for key, value in agents.items()}
    report = {**metadata, 'checkpoint_manifest': str(args.checkpoint_manifest.resolve()),
              'experiment_protocol': context.identity(), 'test_seeds': list(range(201, 231)),
              'config_snapshot': manifest['config_snapshot'], 'scenario_results': _json_value(results)}
    (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output / 'evaluation.json')


if __name__ == '__main__':
    main()
