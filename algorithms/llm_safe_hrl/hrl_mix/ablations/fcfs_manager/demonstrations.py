"""FCFS trajectories using production fixed placement and replay semantics."""
import numpy as np
import time
from hrl_mix.safe_demonstrations import (
    _resolve_fixed_resource_actions, _demonstration_selection, _pending_transition,
    _commit_pending, _copy_mask, _final_episode_metrics,
)
from hrl_mix.train_utils import finalize_action_audit, performance_reward_components
from hrl_mix.safe_metrics import build_episode_metric_record


def generate_episode(env, *, max_phases=100000):
    started = time.perf_counter()
    env.reset()
    trajectories = {'host': [], 'vm': []}
    pending = {'host': None, 'vm': None}
    phases = 0
    while not env.done_flag:
        phases += 1
        if phases > max_phases:
            raise RuntimeError('FCFS demonstration exceeded max phases')
        while True:
            host_state, ready = env.get_host_state_for_next_assignment()
            if not ready:
                break
            if pending['host'] is not None:
                _commit_pending(trajectories['host'], pending['host'], next_state=host_state['obs'],
                                next_mask=_copy_mask(host_state, 'final_action_mask'), done=False, near_boundary_margin=0)
            ha, va, fallback = _resolve_fixed_resource_actions(env)
            selections = {'host': _demonstration_selection(ha, fallback=fallback),
                          'vm': _demonstration_selection(va, fallback=fallback)}
            env.host_select(ha, action_selection=selections['host'])
            vm_state, ready = env.get_vm_state_for_current_task()
            if not ready:
                raise RuntimeError('Fixed placement returned unavailable VM')
            if pending['vm'] is not None:
                _commit_pending(trajectories['vm'], pending['vm'], next_state=vm_state['obs'],
                                next_mask=_copy_mask(vm_state, 'final_action_mask'), done=False, near_boundary_margin=0)
            _, _, info = env.vm_assign(va, action_selection=selections['vm'])
            for layer, state in [('host', host_state), ('vm', vm_state)]:
                reward = float(info['total_performance_reward'])
                shield = info.get(f'{layer}_shield_decision', {})
                pending[layer] = _pending_transition(
                    state, finalize_action_audit(selections[layer], shield), info,
                    reward=reward, cost=float(info['safety_cost']),
                    reward_components=performance_reward_components(info, total_performance_reward=reward),
                    shield_decision=shield)
        env.finish_phase_and_advance()
    for layer in pending:
        if pending[layer] is not None:
            _commit_pending(trajectories[layer], pending[layer], next_state=np.zeros_like(pending[layer]['state']),
                            next_mask=np.zeros_like(pending[layer]['final_action_mask']), done=True, near_boundary_margin=0)
    return {'trajectories': {key: [row.to_dict() for row in value] for key, value in trajectories.items()},
            'episode_metrics': _final_episode_metrics(env),
            'diagnostics': build_episode_metric_record(env, seed=env.random_seed, scheduling_time_seconds=time.perf_counter() - started)}
