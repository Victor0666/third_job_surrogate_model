"""FCFS frozen-policy smoke checks, including an actual LLM-only episode."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from hrl_mix.frozen_manager_ablation import FixedFCFSEvaluationEnv
from hrl_mix.train_eval import evaluate_one_seed
from hrl_mix.protocol_evaluation import _resolve_checkpoint_path, _agent_read_only_fingerprint
from algorithms.llm_safe_hrl.scenario_registry import resolve_experiment_protocol
import runpy
_fixture = runpy.run_path(str(Path(__file__).with_name("test_topk_heuristic_selection.py")))
_write_topk_library = _fixture["_write_topk_library"]
WORKFLOW_FAMILIES = _fixture["WORKFLOW_FAMILIES"]


def test_ready_order_ties_empty_and_phase_reset():
    env = object.__new__(FixedFCFSEvaluationEnv)
    env.manager_mode = "heuristic_selection_mode"
    env.task_ready_time = [5., 1., 1., 8.]
    env._add_workflow_if_arrived = lambda: None
    env._compute_task_heuristics_for_ready = lambda: ([3, 2, 0, 1], None)
    env._phase_prepare_tasks()
    assert env._phase_tasks == [1, 2, 0, 3]
    assert env._phase_ready_task_ordering == [1, 2, 0, 3]
    env.apply_manager_action(999)
    assert env._phase_tasks == [] and not env._phase_started
    env._compute_task_heuristics_for_ready = lambda: ([], None)
    env._phase_prepare_tasks()
    assert env._phase_tasks == []
    assert env._manager_heuristic_identity()["selected_heuristic_id"] == "fixed_fcfs"


def test_actual_llm_only_episode_never_calls_manager_or_llm(tmp_path):
    library = _write_topk_library(tmp_path)
    kwargs = dict(
        dax_paths=[str(Path(__file__).resolve().parents[1] / "data/dax/CyberShake_30.xml")],
        deadline_mode="none", workflows_per_episode=1, horizon=1e6,
        arrival_lambda=.03, random_seed=201, max_ready_tasks=32,
        num_cloud_hosts=1, num_edge_hosts=1, cloud_vms_per_host=[2], edge_vms_per_host=[2],
        cloud_pc_tiers=[1., 2.], edge_pc_tiers=[1., 2.],
        cloud_bw_tiers=[1000., 2000.], edge_bw_tiers=[1000., 2000.],
        deadline_alpha_small_prob=.2,
        fuzzy_enabled=True, safe_rl_enabled=True, safe_rl_shield_enabled=True, safe_rl_state_enabled=True,
        manager_mode="heuristic_selection_mode", manager_heuristic_llm_only=True,
        manager_heuristic_library_path=str(library),
        scenario_code="SS", task_code="S", resource_code="S", workflow_families=WORKFLOW_FAMILIES,
    )
    class FirstValidAgent:
        def select_action(self, state, mask, **options):
            assert options == {"deterministic": True, "count_step": False}
            return int(np.flatnonzero(mask > .5)[0])
    class ForbiddenManager:
        def select_action(self, *args, **kwargs):
            raise AssertionError("Manager network was called")
    with patch.object(FixedFCFSEvaluationEnv, "_selected_manager_heuristic",
                      side_effect=AssertionError("LLM rule was called")):
        result = evaluate_one_seed(FixedFCFSEvaluationEnv, kwargs,
                                   FirstValidAgent(), FirstValidAgent(), ForbiddenManager(),
                                   201, return_safety_metrics=True)
    assert np.isfinite(result[3]) and result[3] > 0


def test_external_checkpoint_requires_opt_in_and_lambda_fingerprint(tmp_path):
    path = tmp_path / "best_checkpoint_manifest.json"
    path.write_text("{}")
    context = resolve_experiment_protocol("single", source_scenario="SS")
    with pytest.raises(ValueError, match="inside protocol"):
        _resolve_checkpoint_path(context, path)
    assert _resolve_checkpoint_path(context, path, allow_external=True) == path
    class Agent:
        lagrange_multiplier = .5
    agent = Agent()
    before = _agent_read_only_fingerprint(agent)
    agent.lagrange_multiplier = .6
    assert before != _agent_read_only_fingerprint(agent)


def test_incremental_seed_output(tmp_path):
    from hrl_mix.protocol_evaluation import _append_seed_result
    _append_seed_result(tmp_path, "SS", 201, (1., 2., 3., 100., {}))
    assert (tmp_path / "scenario_SS_seed_results.csv").exists()
    assert (tmp_path / "scenario_SS_seed_results.jsonl").exists()
