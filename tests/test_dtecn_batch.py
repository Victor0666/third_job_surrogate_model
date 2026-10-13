import json
from types import SimpleNamespace

import numpy as np
import pytest

from algorithms.comparisons.dtecn_llm.run import CONFIG, environment, episode
from algorithms.comparisons.fuzzy_common.protocol import FuzzyComparisonProtocol
from tools.dtecn_batch_policy import BatchPolicy, matching, stable_assignment, parse_assignments


def test_matching_and_stable_fallback_preserve_constrained_tail():
    assert matching([[1], [1]]) is None
    assert stable_assignment([[1, 2], [1]]) == [2, 1]
    assert stable_assignment([]) == []


@pytest.mark.parametrize("rows", [
    [{"task_id": 1, "vm_id": 1}, {"task_id": 2, "vm_id": 1}],
    [{"task_id": 2, "vm_id": 2}, {"task_id": 1, "vm_id": 1}],
    [{"task_id": 1, "vm_id": 99}, {"task_id": 2, "vm_id": 1}],
    [{"task_id": True, "vm_id": 2}, {"task_id": 2, "vm_id": 1}],
])
def test_joint_response_rejects_conflicts_reordering_and_invalid_ids(rows):
    with pytest.raises(ValueError):
        parse_assignments(json.dumps({"assignments": rows}), [1, 2], [[1, 2], [1]])


def fake_env(tasks=5, vms=7):
    env = SimpleNamespace(_cur_tid=0, _phase_tasks=list(range(1, tasks)), current_time=0.,
        vm_ids=list(range(vms)), vm_available_at=np.zeros(vms), num_vms=vms,
        _vm_index_by_id={i: i for i in range(vms)},
        vms={i: SimpleNamespace(pc=100., bw=10.) for i in range(vms)},
        workflows=[SimpleNamespace()], random_seed=201, scenario_code="SS")
    env.set_task_orderer = lambda *args, **kwargs: None
    env.snapshot = lambda: dict(time=0., queue=np.zeros(5), totals=np.zeros(5), service=np.zeros(5),
        qos=np.ones(1), remaining=np.ones(1), active=np.ones(1), unused_capacity=1.)
    env.observed_energy = lambda t: 0.
    env.get_feasible_vms = lambda tid: list(range(vms))
    env.observations = lambda snap: (np.zeros((1, 18 + 3*vms)),
                                     (env.vm_available_at <= 0).astype(float)[None, :], 0)
    def assign(index):
        assert env.vm_available_at[index] == 0
        env.vm_available_at[index] = 10.
        return SimpleNamespace(vm_id=index, task_id=env._cur_tid)
    env.assign_global_vm = assign
    return env


def test_five_ready_tasks_make_one_request_and_five_csv_rows(tmp_path):
    from tools import evaluate_dtecn_llm_with_csv as logger
    from unittest.mock import patch
    config = json.loads(CONFIG.read_text())
    requests = []
    def client(prompt):
        state = json.loads(prompt.split("\nCurrent state: ", 1)[1])
        requests.append(state)
        return json.dumps({"assignments": [{"task_id": t["task_id"], "vm_id": i}
                                           for i, t in enumerate(state["tasks"])]}), {"total_tokens": 50}
    policy = BatchPolicy(config, client=client)
    env = fake_env()
    policy.begin_episode(env, training=False)
    feedback = []
    policy.batch_callback = feedback.append
    def evaluate(protocol, policy, split, path):
        for tid in range(5):
            env._cur_tid = tid
            env._phase_tasks = list(range(tid+1, 5))
            policy.assign(env, None, training=False)
        policy.end_episode(env, training=False)
    path = tmp_path / "decisions.jsonl"
    with patch.object(logger, "_evaluate", evaluate):
        logger.evaluate_with_csv(None, policy, "final_test", path)
    import csv
    with path.with_suffix(".csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(requests) == 1 and len(requests[0]["tasks"]) == 5
    assert len(rows) == 5 and {r["batch_id"] for r in rows} == {"1"}
    assert {r["batch_size"] for r in rows} == {"5"}
    assert len(feedback) == 1 and len(feedback[0]["assignments"]) == 5
    assert policy.stats["requests"] == policy.stats["feedback_count"] == 1


def test_real_episode_batch_collection_and_mock_icl():
    config = json.loads(CONFIG.read_text())
    protocol = FuzzyComparisonProtocol(scenario="SS", ddl_setting="T", train_seeds=(1,),
        validation_seeds=(101,), test_seeds=(201,), workflows_per_episode=1)
    env = environment(protocol, "SS", 1)
    agent = SimpleNamespace(preferences=lambda obs, owner, training: np.ones(env.num_vms))
    teacher = BatchPolicy(config, agent=agent, collect=True)
    record = episode(protocol, "SS", 1, teacher)
    assert sum(len(c["assignments"]) for c in teacher.collected) == record["assignment_steps"]
    assert any(len(c["assignments"]) > 1 for c in teacher.collected)
    def client(prompt):
        state = json.loads(prompt.split("\nCurrent state: ", 1)[1])
        candidates = [t["legal_vm_ids"] for t in state["tasks"]]
        return json.dumps({"assignments": [{"task_id": t["task_id"], "vm_id": vm}
            for t, vm in zip(state["tasks"], stable_assignment(candidates))]}), {"total_tokens": 1}
    online = BatchPolicy(config, client=client, initial_cases=teacher.collected[:1])
    result = episode(protocol, "SS", 201, online)
    assert len(online.decision_records) == result["assignment_steps"]
    assert result["requests"] <= result["batches"] < result["assignment_steps"]
    assert result["feedback_count"] == result["batches"]


def test_prefix_resource_limit_and_invalid_joint_output_fallback():
    config = json.loads(CONFIG.read_text())
    env = fake_env(tasks=5, vms=2)
    policy = BatchPolicy(config, client=lambda prompt: ('{"assignments": []}', None))
    policy.begin_episode(env, training=False)
    first = policy.assign(env, None, training=False)
    assert policy.pending["batch_size"] == 2
    assert env.vm_available_at.tolist().count(10.) == 1
    env._cur_tid = 1
    second = policy.assign(env, None, training=False)
    assert first.vm_id != second.vm_id
    assert policy.stats["requests"] == policy.stats["fallbacks"] == 1
    assert policy.pending["source"] == "legal_batch_fallback"


def test_collect_and_evaluate_entry_reuses_teacher_without_overwriting(tmp_path, monkeypatch):
    from tools import run_dtecn_llm_batch as entry
    from algorithms.comparisons.dtecn_llm import run as runner
    from algorithms.comparisons.dtecn_llm.maddpg import MADDPG
    config = json.loads(CONFIG.read_text())
    protocol = FuzzyComparisonProtocol(scenario="SS", ddl_setting="T", train_seeds=(1,),
        validation_seeds=(101,), test_seeds=(201,), workflows_per_episode=1)
    env = environment(protocol, "SS", 1)
    agent = MADDPG(1, env.observation_dim, env.num_vms, config["maddpg"])
    agent.save(tmp_path / "maddpg.pt")
    metadata = dict(identity=runner.identity(protocol, config, 0, False),
        artifacts={"maddpg.pt": runner.file_hash(tmp_path / "maddpg.pt")})
    for name in ("training_steps", "training_episodes", "training_seconds", "actor_updates", "critic_updates",
                 "validation_count", "validation_assignment_steps", "trainable_actor_parameters", "trainable_critic_parameters"):
        metadata[name] = 1
    runner.write_json(tmp_path / "manifest.json", metadata)
    original = (tmp_path / "manifest.json").read_bytes()
    monkeypatch.setattr(runner, "protocol_from_config", lambda *args, **kwargs: protocol)
    args = ["--teacher-run", str(tmp_path), "--device", "cpu"]
    entry.main(["collect"] + args)
    with pytest.raises(FileExistsError):
        entry.main(["collect"] + args)
    class Client:
        service_identity = {"mock": True}
        def __init__(self, config):
            pass
        def __call__(self, prompt):
            state = json.loads(prompt.split("\nCurrent state: ", 1)[1])
            plan = stable_assignment([t["legal_vm_ids"] for t in state["tasks"]])
            return json.dumps({"assignments": [{"task_id": t["task_id"], "vm_id": vm}
                    for t, vm in zip(state["tasks"], plan)]}), {"total_tokens": 1}
    monkeypatch.setattr(entry, "LLMClient", Client)
    entry.main(["evaluate"] + args)
    roots = list((tmp_path / "batch_v1/evaluations").iterdir())
    assert len(roots) == 1
    root = roots[0]
    assert (root / "icl_final_test.csv").exists()
    assert (root / "icl_final_test_decisions.csv").exists()
    assert (root / "icl_final_test_batches.jsonl").exists()
    assert (tmp_path / "manifest.json").read_bytes() == original
    assert runner.file_hash(tmp_path / "maddpg.pt") == runner.file_hash(tmp_path / "batch_v1/maddpg.pt")
    with pytest.raises(FileExistsError):
        entry.main(["evaluate"] + args)
    cases_hash = runner.file_hash(tmp_path / "batch_v1/cases.jsonl")
    entry.main(["evaluate"] + args + ["--model", "qwen-plus", "--thinking", "off", "--max-tokens", "128"])
    assert len(list((tmp_path / "batch_v1/evaluations").iterdir())) == 2
    assert runner.file_hash(tmp_path / "batch_v1/cases.jsonl") == cases_hash
    assert (tmp_path / "manifest.json").read_bytes() == original


def test_legacy_batch_case_migration_keeps_all_other_fields_checked(tmp_path):
    from copy import deepcopy
    from pathlib import Path
    from tools import run_dtecn_llm_batch as entry
    from algorithms.comparisons.dtecn_llm import run as runner
    import hashlib
    from tools.dtecn_batch_policy import BATCH_PROMPT
    teacher = dict(schema=3, method="dtecn_llm", config={"llm": {"model": "qwen3.8-max"}},
                   source_and_input_hashes={"workload": "unchanged"})
    expected = dict(teacher_checkpoint_sha256="same", batch_policy_sha256="policy")
    legacy = deepcopy(teacher)
    legacy.update(decision_mode="fcfs_joint_batch_v1", teacher_identity=teacher,
        teacher_checkpoint_sha256="same", prompt_sha256=hashlib.sha256(BATCH_PROMPT.encode()).hexdigest(),
        batch_source_hashes={str(Path(entry.__file__).resolve()):
            "68a54ac46bfce29b989eaefb6242543ae85db6b9626b2550dafb899f935722b7",
            str(Path(entry.__file__).with_name("dtecn_batch_policy.py").resolve()): "policy",
            str(Path(entry.__file__).with_name("evaluate_dtecn_llm_with_csv.py").resolve()):
                "5926b798c248d83ddd2a5f66fdef18327c0e2d943be49751874a15fbb4fa6f65"})
    manifest = dict(identity=legacy, artifacts={})
    runner.write_json(tmp_path / "manifest.json", manifest)
    original = (tmp_path / "manifest.json").read_bytes()
    assert entry.load_batch_manifest(tmp_path, expected, teacher, teacher) == manifest
    assert (tmp_path / "manifest.json").read_bytes() == original
    bad = deepcopy(teacher)
    bad["source_and_input_hashes"]["workload"] = "changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        entry.load_batch_manifest(tmp_path, expected, bad, teacher)
