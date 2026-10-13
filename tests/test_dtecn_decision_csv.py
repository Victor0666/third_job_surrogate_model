import csv
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import evaluate_dtecn_llm_with_csv as logging


def test_decision_csv_is_flushed_before_next_decision_and_survives_failure(tmp_path, monkeypatch):
    path = tmp_path / "icl_final_test_decisions.jsonl"
    env = SimpleNamespace(_cur_tid=7, current_time=12., scenario_code="SS", random_seed=201)
    policy = SimpleNamespace(pending=None)
    result = SimpleNamespace(vm_id=3)

    def original_assign(env, state, *, training):
        policy.pending = dict(owner=0, source="single_legal",
                              preferences=np.array([0., 1.]), state={"legal_mask": [0, 1]})
        return result

    policy.assign = original_assign

    def evaluate(protocol, policy, split, trace_path):
        assert policy.assign(env, None, training=False) is result
        with path.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["task_id"] == "7"
        assert rows[0]["seed"] == "201"
        assert json.loads(rows[0]["action"]) == [0., 1.]
        env._cur_tid = 8
        policy.assign(env, None, training=False)
        raise RuntimeError("next request failed")

    monkeypatch.setattr(logging, "_evaluate", evaluate)
    with pytest.raises(RuntimeError, match="next request failed"):
        logging.evaluate_with_csv(None, policy, "final_test", path)
    assert policy.assign is original_assign
    with path.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["decision_index"] for row in rows] == ["1", "2"]
    with pytest.raises(FileExistsError):
        logging.evaluate_with_csv(None, policy, "final_test", path)


def test_real_episode_logs_every_assignment_without_changing_results(tmp_path):
    from algorithms.comparisons.dtecn_llm.policy import DtecnPolicy
    from algorithms.comparisons.fuzzy_common.protocol import FuzzyComparisonProtocol

    config = json.loads(logging.runner.CONFIG.read_text(encoding="utf-8"))
    protocol = FuzzyComparisonProtocol(scenario="SS", ddl_setting="T", train_seeds=(1,),
        validation_seeds=(101,), test_seeds=(201,), workflows_per_episode=1)

    def client(prompt):
        state = json.loads(prompt.split("\nCurrent state: ", 1)[1])
        return json.dumps({"resource_preferences": [.5] * len(state["vm_ids"])}), None

    path = tmp_path / "icl_final_test_decisions.jsonl"
    result = logging.evaluate_with_csv(protocol, DtecnPolicy(config, client=client), "final_test", path)
    with path.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    record = result["scenario_results"]["SS"]["seed_records"][0]
    assert len(rows) == record["assignment_steps"]
    trace = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(trace)
    assert [int(row["executed_vm_id"]) for row in rows] == [row["executed_vm_id"] for row in trace]


def test_only_audited_config_hash_change_is_compatible(tmp_path):
    path = str((logging.runner.ROOT / "algorithms/llm_safe_hrl/hrl_mix/train_config.py").resolve())
    old = "f227b95457cbc9047395c9277e293394349a88c503da9bad243a93e5fee8f271"
    new = "72ee80c2c59939376922416f5cbcb57243bc97c9c234fc82e7c6ffb1d298a6ab"
    saved = dict(schema=3, source_and_input_hashes={path: old, "environment.py": "unchanged"},
                 config={"gamma": .95})
    manifest = dict(identity=saved, artifacts={"maddpg.pt": logging.runner.file_hash(logging.runner.CONFIG)})
    (tmp_path / "maddpg.pt").write_bytes(logging.runner.CONFIG.read_bytes())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    expected = deepcopy(saved)
    expected["source_and_input_hashes"][path] = new
    assert logging.load_compatible_manifest(tmp_path, expected) == manifest
    assert expected["source_and_input_hashes"][path] == new
    for changed in ("other_source", "algorithm_config", "unknown_hash"):
        bad = deepcopy(expected)
        if changed == "other_source":
            bad["source_and_input_hashes"]["environment.py"] = "changed"
        elif changed == "algorithm_config":
            bad["config"]["gamma"] = .9
        else:
            bad["source_and_input_hashes"][path] = "unreviewed"
        with pytest.raises(ValueError, match="identity mismatch"):
            logging.load_compatible_manifest(tmp_path, bad)
    (tmp_path / "maddpg.pt").write_text("corrupt")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        logging.load_compatible_manifest(tmp_path, expected)
