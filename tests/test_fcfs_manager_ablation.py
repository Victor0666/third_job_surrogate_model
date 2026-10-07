import json
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from hrl_mix.ablations.fcfs_manager.fcfs_env import FCFSManagerAblationEnv
from hrl_mix.ablations.fcfs_manager.fcfs_policy import order_ready_tasks_fcfs
from hrl_mix.ablations.fcfs_manager.runtime import ManagerPlaceholder, save_fcfs_checkpoint
from hrl_mix.ablations.fcfs_manager.evaluate_fcfs import validate_mode
from hrl_mix.ablations.fcfs_manager.demonstrations import generate_episode
from hrl_mix.train_config import build_train_config
from hrl_mix import train_runner
from base.d3qn_agent import D3QNAgent


def make_env(**kwargs):
    kwargs.update(dax_paths=['data/dax/Montage_25.xml'], deadline_mode='none',
                  workflows_per_episode=1, horizon=1e6, arrival_lambda=.03,
                  num_cloud_hosts=1, num_edge_hosts=1, cloud_vms_per_host=(2,), edge_vms_per_host=(2,),
                  fuzzy_enabled=True, safe_rl_enabled=True, safe_rl_shield_enabled=True,
                  safe_rl_state_enabled=True)
    return FCFSManagerAblationEnv(**kwargs)


def test_canonical_fcfs_ties():
    env = SimpleNamespace(task_meta=[(1, 0), (0, 1), (0, 2), (2, 0), (1, 1)],
                          task_ready_time=[5, 5, 5, 4, 5],
                          workflows=[SimpleNamespace(arrival_time=1), SimpleNamespace(arrival_time=2), SimpleNamespace(arrival_time=3)])
    assert order_ready_tasks_fcfs(env, [4, 2, 0, 3, 1]) == [3, 1, 2, 0, 4]
    env.workflows[1].arrival_time = 1
    assert order_ready_tasks_fcfs(env, [4, 2, 0, 3, 1]) == [3, 1, 2, 0, 4]


def forbid_library(monkeypatch):
    import base.hrl_env as module
    def forbidden(*args, **kwargs):
        raise AssertionError('No Top-K access permitted')
    monkeypatch.setattr(module, 'load_manager_heuristic_library', forbidden)


def test_fcfs_demo_no_library_and_energy_semantics(monkeypatch):
    forbid_library(monkeypatch)
    env = make_env(random_seed=1, energy_reward_scale=.002,
                   manager_mode='heuristic_selection_mode', manager_heuristic_library_path='must-not-read.json')
    result = generate_episode(env)
    assert set(result['trajectories']) == {'host', 'vm'}
    assert result['diagnostics']['energy_decision_count'] > 0
    assert result['diagnostics']['mean_selected_vm_energy_rank'] >= 1
    for layer in ('host', 'vm'):
        assert env.get_observation_schema(layer)['energy_features']
        assert result['trajectories'][layer]
        for row in result['trajectories'][layer]:
            assert row['safety_cost'] in (0, 1)
            assert not row['shield_modified']


def test_mode_identity():
    from hrl_mix.ablations.fcfs_manager.runtime import identity
    assert validate_mode({}, 'diagnostic')['diagnostic_only']
    assert validate_mode(identity(), 'retrained')['retrained_under_fcfs']
    with pytest.raises(ValueError):
        validate_mode({}, 'retrained')
    with pytest.raises(ValueError):
        validate_mode(identity(), 'diagnostic')


def test_real_training_host_vm_update_manager_inert(tmp_path, monkeypatch):
    forbid_library(monkeypatch)
    agents = []
    def factory(**kwargs):
        agent = D3QNAgent(**kwargs)
        agents.append(agent)
        return agent
    placeholders = []
    def manager_factory(**kwargs):
        agent = ManagerPlaceholder(**kwargs)
        placeholders.append(agent)
        return agent
    monkeypatch.setattr(train_runner, 'D3QNAgent', factory)
    def config_builder(**kwargs):
        kwargs['require_deadline_cache'] = False
        cfg = build_train_config(**kwargs, output_namespace='ablations/fcfs_manager/test')
        tiny = dict(batch_size=1, buffer_size=32, hidden_dims=(8,))
        return replace(cfg, max_episodes=1, validation_interval=1, workflows_per_episode=1,
                       validation_seeds=(101,), warmup_frac=0, hard_max_steps=200,
                       save_dir=str(tmp_path / 'checkpoints'), log_path=str(tmp_path / 'train.csv'),
                       host_agent=replace(cfg.host_agent, **tiny), vm_agent=replace(cfg.vm_agent, **tiny),
                       manager_agent=replace(cfg.manager_agent, **tiny))
    (tmp_path / 'checkpoints').mkdir()
    train_runner.train(protocol='single', source_scenario='SS', ddl='T', safe_rl_enabled=True,
                       safe_rl_shield_enabled=True, safe_rl_state_enabled=True,
                       safe_rl_dynamic_lambda_enabled=True, safe_rl_heuristic_manager_enabled=False,
                       manager_heuristic_llm_only=False, validation_workers=1,
                       environment_class=make_env, config_builder=config_builder,
                       manager_factory=manager_factory, checkpoint_writer=save_fcfs_checkpoint)
    assert len(agents) == 2
    assert all(agent._updates > 0 and len(agent.buffer) > 0 for agent in agents)
    assert placeholders[0]._updates == 0 and len(placeholders[0].buffer) == 0
    manifest = json.loads((tmp_path / 'checkpoints/best_checkpoint_manifest.json').read_text(encoding='utf-8'))
    assert manifest['retrained_under_fcfs'] is True
    assert manifest['manager_trainable'] is False
    assert manifest['manager_placeholder']['replay_size'] == 0
    # Evaluate real saved Host/VM weights through both entry modes, without ever
    # loading the Manager artifact. Keep this smoke test to one seed per target.
    from hrl_mix.ablations.fcfs_manager import evaluate_fcfs as evaluation
    from hrl_mix.train_eval import evaluate_hrl_three_layer_multi_seed
    monkeypatch.setattr(evaluation, 'ROOT_DIR', tmp_path)
    monkeypatch.setattr(evaluation, 'validate_single_deadline_cache_paths', lambda *a, **k: {})
    monkeypatch.setattr(evaluation, 'build_frozen_scenario_env_kwargs', lambda *a, **k: {})
    monkeypatch.setattr(evaluation, 'FCFSManagerAblationEnv', make_env)
    def short_eval(env_cls, kwargs, vm, host, manager, seeds, **options):
        assert tuple(seeds) == tuple(range(201, 231))
        assert manager.trainable is False
        return evaluate_hrl_three_layer_multi_seed(env_cls, kwargs, vm, host, manager, [201], **options)
    monkeypatch.setattr(evaluation, 'evaluate_hrl_three_layer_multi_seed', short_eval)
    loaded = []
    real_loader = evaluation._load_frozen_agent
    def loader(path, **kwargs):
        assert 'manager' not in str(path)
        loaded.append(path)
        return real_loader(path, **kwargs)
    monkeypatch.setattr(evaluation, '_load_frozen_agent', loader)
    manifest_path = tmp_path / 'checkpoints/best_checkpoint_manifest.json'
    for mode in ['retrained', 'diagnostic']:
        if mode == 'diagnostic':
            for key in ('ablation', 'retrained_under_fcfs'):
                manifest.pop(key)
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        evaluation.main(['--mode', mode, '--checkpoint-manifest', str(manifest_path),
                         '--source-scenario', 'SS', '--deadline-cache', 'SS=unused'])
        report = json.loads((tmp_path / 'out/ablations/fcfs_manager' / mode / 'SS/checkpoints/evaluation.json').read_text(encoding='utf-8'))
        assert report['retrained_under_fcfs'] == (mode == 'retrained')
        assert set(report['scenario_results']) == {'SS', 'MS', 'LS'}
        assert report['scenario_results']['SS'][4]['energy_decision_count'] > 0
    assert len(loaded) == 4


def test_fcfs_masks_allow_predicted_unsafe_and_observation_matches_full():
    from tests.test_safe_observation_state import _make_environment
    full = _make_environment(safe_rl_enabled=True, safe_rl_state_enabled=True)
    fcfs = make_env(random_seed=0)
    fcfs.reset()
    for layer in ('host', 'vm'):
        assert fcfs.get_observation_schema(layer) == full.get_observation_schema(layer)
    state, ready = fcfs.get_host_state_for_next_assignment()
    assert ready
    fcfs.host_select(int(np.flatnonzero(state['mask'])[0]))
    vm_state, ready = fcfs.get_vm_state_for_current_task()
    assert ready
    context = fcfs._current_safety_shield_context
    context['current_vm_masks']['safety_action_mask'][:] = 0
    slot = int(np.flatnonzero(vm_state['legal_action_mask'])[0])
    _, _, info = fcfs.vm_assign(slot)
    assert not info['vm_shield_decision']['action_modified']


def test_fcfs_defaults_and_isolated_output(tmp_path, monkeypatch):
    from hrl_mix import train_config
    from hrl_mix.ablations.fcfs_manager.train_fcfs import build_fcfs_config
    monkeypatch.setattr(train_config, 'ROOT_DIR', tmp_path)
    cfg = build_fcfs_config(protocol='single', source_scenario='SL', ddl='T',
                            safe_rl_enabled=True, require_deadline_cache=False)
    assert cfg.max_episodes == 600
    assert cfg.train_seeds == (1, 2, 3, 4, 5)
    assert cfg.validation_seeds == (101, 102, 103)
    assert cfg.final_test_seeds == tuple(range(201, 231))
    assert cfg.energy_reward_scale == .002
    assert 'ablations' in Path(cfg.save_dir).parts
    assert not (tmp_path / 'checkpoints/main_single').exists()


def test_demo_cli_manifest_has_no_topk_dependency(tmp_path, monkeypatch):
    import tools.generate_fcfs_ablation_demonstrations as tool
    from hrl_mix import train_config
    forbid_library(monkeypatch)
    monkeypatch.setattr(tool, 'ROOT_DIR', tmp_path)
    monkeypatch.setattr(train_config, 'ROOT_DIR', tmp_path)
    monkeypatch.setattr(tool, 'training_kwargs', lambda args: dict(
        protocol='single', source_scenario='SL', ddl='T', require_deadline_cache=False,
        safe_rl_enabled=True, safe_rl_state_enabled=True, safe_rl_shield_enabled=True))
    monkeypatch.setattr(tool, 'FCFSManagerAblationEnv', make_env)
    tool.main(['--source-scenario', 'SL', '--ddl', 'T', '--deadline-cache', 'SL=unused'])
    manifests = list((tmp_path / 'out/safe_demonstrations_fcfs_ablation').rglob('manifest.json'))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
    assert manifest['energy_reward_scale'] == .002
    assert manifest['safety_cost_definition'] == 'actual_deadline_violation'
    assert manifest['shield_semantics'] == 'monitor_only'
    assert [item['workflow_seed'] for item in manifest['episodes']] == [1, 2, 3, 4, 5]
    assert not any(key in manifest for key in ('heuristic_ids', 'heuristic_library_sha256', 'manager_heuristic_ids'))
