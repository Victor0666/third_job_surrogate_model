import numpy as np
import pytest

from tests.test_safe_observation_state import _make_environment
from tests.test_safe_performance_reward import _reward_stub, _summary
from hrl_mix.train_config import build_train_config, safety_learning_identity


def test_reward_scale_and_identity():
    rewards = []
    for scale in (.001, .002):
        env = _reward_stub([_summary(130)])
        env.energy_reward_scale = scale
        rewards.append(env._safe_performance_reward_breakdown(_summary(100))['total_performance_reward'])
    assert rewards[1] == 2 * rewards[0] < 0
    a = build_train_config(safe_rl_enabled=True, require_deadline_cache=False)
    b = build_train_config(safe_rl_enabled=True, require_deadline_cache=False, safe_rl_energy_reward_scale=.002)
    assert a.energy_reward_scale == .001 and b.energy_reward_scale == .002
    assert a.run_name != b.run_name
    assert safety_learning_identity(a.safe_rl, a.energy_reward_scale) != safety_learning_identity(b.safe_rl, b.energy_reward_scale)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_scale(scale):
    with pytest.raises(ValueError, match='finite and positive'):
        build_train_config(require_deadline_cache=False, safe_rl_energy_reward_scale=scale)


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
    manifest = dict(safety_cost_definition='dense_fuzzy_ddl_cost_v1',
                    shield_semantics='active_fuzzy_ddl_shield_v1', safe_rl_identity=identity)
    _validate_safety_definition(manifest, identity)
    with pytest.raises(ValueError, match='safety-cost definition mismatch'):
        _validate_safety_definition({**manifest, 'safety_cost_definition': 'actual_deadline_violation', 'shield_semantics': 'monitor_only'}, identity)
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


def test_restored_observation_has_no_relative_energy_extension():
    env = _make_environment(safe_rl_enabled=True, safe_rl_state_enabled=True)
    assert not hasattr(env, '_host_energy_observation')
    assert not hasattr(env, '_vm_energy_observation')
    assert not hasattr(env, 'get_energy_decision_diagnostics')
    cfg = build_train_config(safe_rl_enabled=True, require_deadline_cache=False)
    assert safety_learning_identity(cfg.safe_rl, cfg.energy_reward_scale)['energy_observation'] == 'no_relative_energy_v1'
    assert cfg.safe_rl.lagrangian.lambda_init == 1.0
    assert cfg.safe_rl.lagrangian.lambda_lr == .05
    assert cfg.safe_rl.lagrangian.cost_budget == .01
