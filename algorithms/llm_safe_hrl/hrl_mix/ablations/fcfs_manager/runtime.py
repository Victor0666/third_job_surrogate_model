"""Compatibility metadata for the shared three-layer checkpoint writer."""
import json
from pathlib import Path

from base.d3qn_agent import D3QNAgent
from hrl_mix.model_selection import save_best_checkpoint_bundle


def identity(*, diagnostic=False):
    return dict(experiment_type="ablation", ablation="fcfs_manager",
                manager_policy="fixed_fcfs", manager_trainable=False,
                llm_enabled=False, task_ordering="canonical_fcfs",
                retrained_under_fcfs=not diagnostic, diagnostic_only=diagnostic)


class ManagerPlaceholder(D3QNAgent):
    """Serialization-only shell; no action or learning method may run."""
    trainable = False

    def epsilon(self):
        return 0.0

    def select_action(self, *args, **kwargs):
        raise AssertionError("Fixed FCFS must not invoke a Manager policy")

    select_action_with_info = select_action
    remember = select_action
    update = select_action

    def save(self, path, **kwargs):
        assert self._updates == 0 and len(self.buffer) == 0
        super().save(path, **kwargs)


def save_fcfs_checkpoint(*args, **kwargs):
    result = save_best_checkpoint_bundle(*args, **kwargs)
    # Preserve the production schema and selection key; label the inert shell.
    directory = Path(args[0] if args else kwargs['directory'])
    path = directory / "best_checkpoint_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(identity())
    payload['manager_placeholder'] = dict(update_count=0, replay_size=0, evaluation_used=False)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return result
