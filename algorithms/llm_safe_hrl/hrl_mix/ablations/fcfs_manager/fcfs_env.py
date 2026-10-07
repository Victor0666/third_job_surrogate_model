"""Only phase ordering differs from the production environment."""
from base.hrl_env import CloudWorkflowEnv_VMAgents
from .fcfs_policy import order_ready_tasks_fcfs


class FCFSManagerAblationEnv(CloudWorkflowEnv_VMAgents):
    def __init__(self, *args, **kwargs):
        kwargs.update(manager_mode="legacy_rule_weight_mode",
                      manager_heuristic_library_path=None,
                      manager_heuristic_llm_only=False)
        super().__init__(*args, **kwargs)

    def _phase_prepare_tasks(self):
        self._add_workflow_if_arrived()
        ordering = order_ready_tasks_fcfs(self, self.get_ready_tasks())
        self._phase_tasks = list(ordering)
        self._phase_ready_task_ordering = list(ordering)

    def apply_manager_action(self, action):
        # Compatibility phase call: no policy output influences ordering.
        pass

    def _manager_heuristic_identity(self):
        return dict(selected_heuristic_id="fixed_fcfs", heuristic_source="fixed_fcfs",
                    llm_rule_version="", heuristic_rule_version="canonical_fcfs",
                    manager_mode="fixed_fcfs", selected_heuristic_index=None)
