"""Frozen GP-HRL evaluation using the production evaluator and three agents."""
import argparse
from datetime import datetime
from pathlib import Path

from algorithms.llm_safe_hrl.base.heuristic_admission import file_sha256
from hrl_mix.model_selection import read_best_checkpoint_manifest
from hrl_mix.protocol_evaluation import (
    _load_frozen_agent, _snapshot_config, _json_value, evaluate_frozen_protocol_scenarios,
)
from hrl_mix.train_config import ROOT_DIR, validate_single_deadline_cache_paths
from .generate import write_json
from .train import validate_gp_library, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-manifest', type=Path, required=True)
    parser.add_argument('--gp-library', type=Path)
    parser.add_argument('--deadline-cache', action='append', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    manifest = read_best_checkpoint_manifest(args.checkpoint_manifest)
    if manifest.get('ablation') != 'pure_gp_hrl' or manifest.get('retrained_under_gp') is not True:
        raise ValueError('evaluation requires a retrained GP-HRL best checkpoint')
    library = (args.gp_library or Path(manifest['heuristic_library_version']['manifest_path'])).resolve()
    payload, context = validate_gp_library(library,
        source_scenario=manifest['experiment_protocol']['source_scenario'],
        ddl=manifest['config_snapshot']['config']['ddl_name'])
    if context.identity() != manifest['experiment_protocol']:
        raise ValueError('checkpoint/library protocol mismatch')
    if file_sha256(library) != manifest['heuristic_library_version']['manifest_sha256']:
        raise ValueError('GP library differs from frozen training checkpoint')
    caches = validate_single_deadline_cache_paths('single', args.deadline_cache,
                                                 source_scenario=context.source_scenario)
    agents = {layer: _load_frozen_agent(manifest['resolved_agent_checkpoints'][layer],
                                      replay_metadata=manifest['replay_metadata'][layer],
                                      device=args.device) for layer in ('manager', 'host', 'vm')}
    directory = (ROOT_DIR / 'out/ablations/pure_gp_hrl/test' / context.source_scenario
                 / payload['deadline_setting']['ddl_code']
                 / f"{args.checkpoint_manifest.parent.name}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    directory.mkdir(parents=True, exist_ok=False)
    result = evaluate_frozen_protocol_scenarios(
        context=context, config=_snapshot_config(manifest), library_path=library, agents=agents,
        deadline_cache_overrides=caches, progress_output_dir=directory)
    path = directory / 'evaluation.json'
    write_json(path, dict(**identity(), checkpoint_manifest=str(args.checkpoint_manifest.resolve()),
                         gp_library=str(library), gp_library_sha256=file_sha256(library),
                         **{key: payload[key] for key in ('experiment_config', 'experiment_config_sha256',
                             'reference_library', 'reference_library_sha256') if key in payload},
                         experiment_protocol=context.identity(),
                         config_snapshot=manifest['config_snapshot'], **_json_value(result)))
    print(path)
    return path


if __name__ == '__main__':
    main()
