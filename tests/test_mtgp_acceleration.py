"""Acceleration must preserve scores, trajectories, selection and resume state."""

from dataclasses import replace
import json

import numpy as np
import pytest

from algorithms.comparisons.mtgp import core
from algorithms.comparisons.mtgp import checkpoint
from algorithms.comparisons.mtgp.run_pipeline import input_identity
from algorithms.comparisons.fuzzy_common.protocol import FuzzyComparisonProtocol
from tests.test_mtgp_transport import environment, tree


def protocol():
    return FuzzyComparisonProtocol(scenario="SS", ddl_setting="T", train_seeds=(1, 2),
                                   validation_seeds=(101,), test_seeds=(201,), workflows_per_episode=1)


def metrics(result):
    return tuple(result[name] for name in (
        "deadline_violation_rate", "max_fuzzy_lateness", "mean_fuzzy_lateness",
        "fuzzy_energy_mean", "fuzzy_energy_std", "fuzzy_energy_score",
    ))


def test_numpy_scores_match_scalar_bits_including_ties_and_protected_division():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(50, len(core.TERMINALS)))
    values[0] = 0.0
    values[1] = -0.0
    values[2, core.TERMINALS.index("TWT")] = 1e-13
    trees = [tree(operator, "PT", "TWT") for operator in core.OPERATORS]
    trees.extend(core.random_tree(rng, 4, index % 2 == 0) for index in range(30))
    for program in trees:
        expected = np.array([core.evaluate_tree(program, row) for row in values], dtype=np.float64)
        actual = core.evaluate_tree_batch(program, values)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    overflow = values.copy()
    overflow[:, core.TERMINALS.index("PT")] = 1e308
    with pytest.raises(ValueError, match="non-finite"):
        core.evaluate_tree_batch(tree("mul", "PT", "PT"), overflow)


def test_used_terminal_values_match_reference_and_unused_energy_is_not_computed(monkeypatch):
    env = environment((2, 4), (1, 8), (0, 0))
    env.vm_available_at[0] = 20.0
    for task in (0, 1):
        env.route_task_to_vm(task, 0)
    env.current_time = 5.0
    env._process_finish_events_at_current_time()
    program = tree("add", "WIQ", "TTIQ")
    expected = np.array([core.task_terminals(env, task, 0) for task in (0, 1)])
    calls = []
    getter = env._task_duration_components_ref
    def counted(*args):
        calls.append(args)
        return getter(*args)
    monkeypatch.setattr(env, "_task_duration_components_ref", counted)
    monkeypatch.setattr(env, "estimate_incremental_energy_score", lambda *args: pytest.fail("unused energy was computed"))
    actual = core.terminal_matrix(env, (0, 1), (0, 0), program)
    assert len(calls) == 2  # The queue is scanned once, not once per candidate.
    for name in ("WIQ", "TTIQ"):
        index = core.TERMINALS.index(name)
        np.testing.assert_array_equal(actual[:, index].view(np.uint64), expected[:, index].view(np.uint64))


def test_optimized_episode_matches_reference_assignment_and_shadow_timelines(monkeypatch):
    original_factory = core.make_environment
    original_matrix, original_evaluator = core.terminal_matrix, core.evaluate_tree_batch
    captured = []
    reference = False
    def factory(*args, **kwargs):
        env = original_factory(*args, **kwargs)
        captured.append(env)
        if reference:
            advance = env.advance_to_next_resource_event
            env.advance_to_next_resource_event = lambda **kwargs: advance()
        return env
    monkeypatch.setattr(core, "make_environment", factory)
    rng = np.random.default_rng(4)
    pairs = [core.RulePair(tree("sub", "MRT", "WIQ"), tree("add", "TIME_TO_DDL", "DELTA_FUZZY_ENERGY"))]
    pairs += [core.RulePair(core.random_tree(rng, 3, True), core.random_tree(rng, 3, True)) for _ in range(2)]
    for pair in pairs:
        reference = False
        monkeypatch.setattr(core, "terminal_matrix", original_matrix)
        monkeypatch.setattr(core, "evaluate_tree_batch", original_evaluator)
        fast = core.run_episode(replace(protocol(), workflows_per_episode=3), pair, 1)
        reference = True
        monkeypatch.setattr(core, "terminal_matrix", lambda env, tasks, vms, program: np.array([
            core.task_terminals(env, task, vm) for task, vm in zip(tasks, vms)
        ]))
        monkeypatch.setattr(core, "evaluate_tree_batch", lambda program, values: np.array([
            core.evaluate_tree(program, row) for row in values
        ]))
        slow = core.run_episode(replace(protocol(), workflows_per_episode=3), pair, 1)
        assert metrics(fast) == metrics(slow)
        assert captured[-2].assignment_history == captured[-1].assignment_history
        assert captured[-2].task_end_time == captured[-1].task_end_time
        assert captured[-2].shadow_task_end_time == captured[-1].shadow_task_end_time


def test_serial_parallel_and_interrupted_resume_choose_identical_trees(tmp_path, monkeypatch):
    options = dict(algorithm_seed=7, population_size=8, generations=3, elite_size=2)
    serial = core.train(protocol(), workers=1, **options)
    parallel = core.train(protocol(), workers=2, **options)
    assert serial[0] == parallel[0]
    assert metrics(serial[1]) == metrics(parallel[1])
    assert serial[2] == parallel[2]
    assert serial[3]["simulated_training_episode_count"] == parallel[3]["simulated_training_episode_count"]
    path = tmp_path / "checkpoint.json"
    writer = checkpoint.write_checkpoint
    def interrupt_after_generation(path, state):
        writer(path, state)
        if state["next_generation"] == 2:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(checkpoint, "write_checkpoint", interrupt_after_generation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        core.train(protocol(), workers=1, checkpoint_path=path, **options)
    monkeypatch.setattr(checkpoint, "write_checkpoint", writer)
    resumed = core.train(protocol(), workers=2, checkpoint_path=path, resume=True, **options)
    assert resumed[0] == serial[0]
    assert metrics(resumed[1]) == metrics(serial[1])
    assert resumed[2] == serial[2]
    assert resumed[3] == parallel[3]


def test_cache_is_per_instance_and_resume_rejects_stale_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "random_tree", lambda *args: tree("PT"))
    path = tmp_path / "checkpoint.json"
    options = dict(population_size=4, generations=3, elite_size=1, checkpoint_path=path)
    result = core.train(protocol(), **options)
    assert result[3]["training_episode_count"] == 12
    assert result[3]["simulated_training_episode_count"] == 2
    assert result[3]["training_cache_hits"] == 10
    state = json.loads(path.read_text(encoding="utf-8"))
    assert {row["seed"] for row in state["fitness_cache"]} == {1, 2}
    state["identity"]["protocol_hash"] = "stale"
    checkpoint.write_checkpoint(path, state)
    with pytest.raises(ValueError, match="identity mismatch"):
        core.train(protocol(), resume=True, **options)


def test_no_checkpoint_overwrite_and_no_final_seed_cache(tmp_path):
    path = tmp_path / "checkpoint.json"
    options = dict(population_size=4, generations=1, elite_size=1, checkpoint_path=path)
    core.train(protocol(), **options)
    with pytest.raises(FileExistsError):
        core.train(protocol(), **options)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["fitness_cache"][0]["seed"] = 201
    checkpoint.write_checkpoint(path, state)
    with pytest.raises(ValueError, match="non-training"):
        core.train(protocol(), resume=True, **options)


def test_pool_start_failure_restores_thread_environment(monkeypatch):
    import os
    from algorithms.comparisons.mtgp import parallel
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    monkeypatch.setenv(names[0], "3")
    for name in names[1:]:
        monkeypatch.delenv(name, raising=False)
    before = {name: os.environ.get(name) for name in names}
    def fail(**kwargs):
        raise RuntimeError("pool start failed")
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", fail)
    with pytest.raises(RuntimeError, match="pool start failed"):
        with parallel.EvaluationPool(workers=2):
            pass
    assert {name: os.environ.get(name) for name in names} == before
