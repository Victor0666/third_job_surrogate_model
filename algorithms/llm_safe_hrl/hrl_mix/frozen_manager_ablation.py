"""Evaluation-only FCFS replacement; retains frozen Host/VM observation schemas."""
from base.hrl_env import CloudWorkflowEnv_VMAgents


class FixedFCFSEvaluationEnv(CloudWorkflowEnv_VMAgents):
    fixed_fcfs_manager = True

    def apply_manager_action(self, action_index):
        # The evaluation driver passes a placeholder, not an LLM action index.
        self._phase_started = False
        self._phase_tasks = []
        self._phase_ready_task_ordering = []

    def _phase_prepare_tasks(self):
        self._add_workflow_if_arrived()
        ready_ids, _ = self._compute_task_heuristics_for_ready()
        ordering = sorted(ready_ids, key=lambda tid: (self.task_ready_time[tid], tid))
        self._phase_tasks = list(ordering)
        self._phase_ready_task_ordering = list(ordering)

    def _manager_heuristic_identity(self):
        return {
            "selected_heuristic_id": "fixed_fcfs",
            "heuristic_source": "traditional",
            "llm_rule_version": "",
            "heuristic_rule_version": "ready_time_task_id_v1",
            "manager_mode": self.manager_mode,
            "selected_heuristic_index": None,
        }

    def _record_selected_heuristic_phase(self, **kwargs):
        # FCFS results must not be attributed to an LLM candidate.
        return None


if __name__ == "__main__":
    from hrl_mix.protocol_evaluation import main
    main()
