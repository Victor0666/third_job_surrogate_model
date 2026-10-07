import numpy as np
import pytest

from base.energy_observation import relative_energy, placement_energy_diagnostics
from tests.test_safe_observation_state import _make_environment
from tests.test_safe_performance_reward import _reward_stub, _summary
from hrl_mix.train_config import build_train_config, safety_learning_identity


@pytest.mark.parametrize('values,legal,expected', [
    ([100, 200, 300], [1, 1, 1], [0, .5, 1]),
    ([100, 100, 100], [1, 1, 1], [0, 0, 0]),
    ([50, 100, 150], [1, 1, 1], [0, .5, 1]),
    ([50, 0, 0], [1, 0, 0], [0, 1, 1]),
    ([0, 0], [0, 0], [1, 1]),
])
def test_relative_energy(values, legal, expected):
    result = relative_energy(values, legal)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_host_and_vm_features_use_legal_candidates_and_one_estimator():
    env = _make_environment(safe_rl_enabled=True, safe_rl_state_enabled=True)
    state, ready = env.get_host_state_for_next_assignment()
    assert ready
    context = env._current_safety_shield_context
    legal = context['vm_masks_global']['legal_action_mask'] > .5
    energies = {i: 100 + 30 * i for i in range(len(env.vm_ids))}
    calls = []
    def estimate(task, vm):
        calls.append(vm)
        return energies[env._vm_index_by_id[vm]]
    env.estimate_incremental_energy_score = estimate
    context.pop('legal_candidate_energy', None)
    context['vm_masks_global']['safety_action_mask'][:] = 0
    host_features = env._host_energy_observation(env._cur_tid, context)
    groups = [[energies[i] for i in env.host_to_vm_indices[h] if legal[i]] for h in env.host_ids]
    expected = np.column_stack((relative_energy([min(g) for g in groups], [1, 1]),
                                relative_energy([np.mean(g) for g in groups], [1, 1]))).ravel()
    np.testing.assert_allclose(host_features, expected)
    env.host_select(0)
    features = env._vm_energy_observation(env._cur_tid, context)
    np.testing.assert_allclose(features, [0, 1], atol=1e-6)
    assert len(calls) == int(legal.sum())  # cached until the placement mutates state
    vm_state, ready = env.get_vm_state_for_current_task()
    assert ready and vm_state['mask'][0] == 1
    _, _, info = env.vm_assign(0)
    assert info['selected_best_energy_vm'] == 1
    assert env.get_energy_decision_diagnostics()['energy_decision_count'] == 1
    env.reset()
    assert env.get_energy_decision_diagnostics()['energy_decision_count'] == 0


def test_regret_and_tie_ranks():
    d = placement_energy_diagnostics({0: 100, 1: 130, 2: 160}, [1, 2], 2)
    assert d['host_energy_regret'] == d['vm_energy_regret'] == 30
    assert d['selected_vm_energy_rank'] == 2
    assert d['selected_best_energy_vm'] == 0
    assert d['host_energy_regret_norm'] + d['vm_energy_regret_norm'] == pytest.approx(1)
    tied = placement_energy_diagnostics({0: 100, 1: 100}, [0, 1], 1)
    assert tied['selected_vm_energy_rank'] == 1
    assert tied['selected_global_best_energy_vm'] == 1


def test_reward_scale_and_identity():
    rewards = []
    for scale in (.001, .002):
        env = _reward_stub([_summary(130)])
        env.energy_reward_scale = scale
        rewards.append(env._safe_performance_reward_breakdown(_summary(100))['total_performance_reward'])
    assert rewards[1] == 2 * rewards[0] < 0
    a = build_train_config(safe_rl_enabled=True, require_deadline_cache=False)
    b = build_train_config(safe_rl_enabled=True, require_deadline_cache=False, safe_rl_energy_reward_scale=.001)
    assert a.energy_reward_scale == .002 and b.energy_reward_scale == .001
    assert a.run_name != b.run_name
    assert safety_learning_identity(a.safe_rl, a.energy_reward_scale) != safety_learning_identity(b.safe_rl, b.energy_reward_scale)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_scale(scale):
    with pytest.raises(ValueError, match='finite and positive'):
        build_train_config(require_deadline_cache=False, safe_rl_energy_reward_scale=scale)


def test_empty_host_single_vm_and_predicted_unsafe_candidate():
    env = _make_environment(safe_rl_enabled=True, safe_rl_state_enabled=True)
    _, ready = env.get_host_state_for_next_assignment()
    assert ready
    context = env._current_safety_shield_context
    context.pop('legal_candidate_energy', None)
    mask = context['vm_masks_global']['legal_action_mask']
    mask[:] = 0
    mask[0] = 1
    context['vm_masks_global']['safety_action_mask'][:] = 0
    env.estimate_incremental_energy_score = lambda task, vm: 100
    np.testing.assert_allclose(env._host_energy_observation(env._cur_tid, context), [0, 0, 1, 1])
    env._cur_host_id = env.host_ids[0]
    np.testing.assert_allclose(env._vm_energy_observation(env._cur_tid, context), [0, 1])
    assert context['legal_candidate_energy'] == {0: 100}


def test_diagnostics_weighted_aggregation_and_seed_csv(tmp_path):
    import csv
    from tests.test_safe_metrics import _safe_report
    from hrl_mix.safe_metrics import aggregate_safe_metric_records
    from hrl_mix.protocol_evaluation import _append_seed_result
    from base.energy_observation import ENERGY_DIAGNOSTIC_FIELDS
    rows = []
    for seed, count, value in [(1, 1, 0), (2, 3, 1)]:
        row = _safe_report(seed=seed)['per_seed_safety_metrics'][0]
        row.update(energy_decision_count=count)
        row.update({field: value for field in ENERGY_DIAGNOSTIC_FIELDS})
        rows.append(row)
    report = aggregate_safe_metric_records(rows)
    assert report['energy_decision_count'] == 4
    for field in ENERGY_DIAGNOSTIC_FIELDS:
        assert report[field] == .75
    _append_seed_result(tmp_path, 'SS', 1, (0, 0, 0, 10, report))
    with (tmp_path / 'scenario_SS_seed_results.csv').open(encoding='utf-8') as stream:
        saved = next(csv.DictReader(stream))
    assert float(saved['mean_host_energy_regret']) == .75


def test_old_checkpoint_dimensions_fail_fast(tmp_path):
    from base.d3qn_agent import D3QNAgent
    for old_dim, new_dim in [(55, 61), (180, 189)]:
        kwargs = dict(output_dim=3, hidden_dims=(8,), device='cpu', safe_rl_enabled=True)
        old = D3QNAgent(input_dim=old_dim, **kwargs)
        path = tmp_path / f'{old_dim}.pth'
        old.save(str(path))
        new = D3QNAgent(input_dim=new_dim, **kwargs)
        with pytest.raises(ValueError, match='observation dimension mismatch'):
            new.load(str(path))


def test_cli_scale_reaches_training(monkeypatch):
    from hrl_mix import train as cli
    captured = {}
    monkeypatch.setattr(cli, 'train', lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(cli, 'validate_single_deadline_cache_paths', lambda *args, **kwargs: {})
    cli.main(['--scenario', 'SS', '--safe-rl', '--safe-rl-energy-reward-scale', '0.005'])
    assert captured['safe_rl_energy_reward_scale'] == .005


def test_demonstration_reward_identity_rejects_old_or_different_scale():
    from base.safe_demonstration import _validate_safety_definition
    cfg = build_train_config(safe_rl_enabled=True, require_deadline_cache=False)
    identity = safety_learning_identity(cfg.safe_rl, .002)
    manifest = dict(safety_cost_definition='actual_deadline_violation',
                    shield_semantics='monitor_only', safe_rl_identity=identity)
    _validate_safety_definition(manifest, identity)
    with pytest.raises(ValueError, match='identity mismatch'):
        _validate_safety_definition(manifest, safety_learning_identity(cfg.safe_rl, .001))
    old = {k: v for k, v in identity.items() if k not in ('energy_reward_scale', 'energy_observation')}
    with pytest.raises(ValueError, match='identity mismatch'):
        _validate_safety_definition({**manifest, 'safe_rl_identity': old}, identity)


def test_evaluation_rejects_old_dimensions_before_any_action():
    from types import SimpleNamespace
    from hrl_mix.train_eval import evaluate_one_seed
    env = _make_environment(safe_rl_enabled=True, safe_rl_state_enabled=True)
    old_host = SimpleNamespace(input_dim=env.host_obs_dim - 2 * env.num_hosts)
    with pytest.raises(ValueError, match='host checkpoint observation dimension mismatch'):
        evaluate_one_seed(lambda **kwargs: env, {}, None, old_host, None, 1)
