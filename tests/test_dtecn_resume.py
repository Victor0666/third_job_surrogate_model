import json
from copy import deepcopy

import pytest

from algorithms.comparisons.dtecn_llm.run import CONFIG, episode
from algorithms.comparisons.fuzzy_common.protocol import FuzzyComparisonProtocol
from tools.dtecn_batch_policy import BatchPolicy, stable_assignment
from tools.resume_dtecn_llm_batch import ReplayPolicy, json_rows, recover, summarize, continue_evaluation


def test_replay_recovers_metrics_usage_and_decisions_without_api(tmp_path):
    config = json.loads(CONFIG.read_text())
    protocol = FuzzyComparisonProtocol(scenario='SS', ddl_setting='L', train_seeds=(1,),
        validation_seeds=(101,), test_seeds=(201,), workflows_per_episode=1)
    batches = []
    def client(prompt):
        state = json.loads(prompt.split('\nCurrent state: ', 1)[1])
        plan = stable_assignment([t['legal_vm_ids'] for t in state['tasks']])
        return json.dumps({'assignments': [dict(task_id=t['task_id'], vm_id=v)
                         for t, v in zip(state['tasks'], plan)]}), {'total_tokens': 17}
    original = BatchPolicy(config, client=client, batch_callback=batches.append)
    result = episode(protocol, 'SS', 201, original)
    replay = ReplayPolicy(config, batches)
    restored = episode(protocol, 'SS', 201, replay)
    for field, value in result.items():
        if field != 'scheduling_time_seconds':
            assert restored[field] == value, field
    assert replay.decision_records == original.decision_records
    with (tmp_path / 'icl_final_test_decisions.jsonl').open('w') as stream:
        for row in original.decision_records:
            stream.write(json.dumps(dict(seed=201, scenario='SS', **row)) + '\n')
    with (tmp_path / 'icl_final_test_batches.jsonl').open('w') as stream:
        for row in batches:
            stream.write(json.dumps(row) + '\n')
        stream.write('{"seed":')
    output = tmp_path / 'recovery_v1'
    output.mkdir()
    recover(tmp_path, output, protocol, config, [('SS', 201)])
    assert json.loads((output / 'SS_201.json').read_text())['recovered_from_actions']
    recover(tmp_path, output, protocol, config, [('SS', 201)])
    summarize(output, protocol, dict(cases_sha256='cases',
        case_identity={'teacher_checkpoint_sha256': 'checkpoint'}), {})
    summary = json.loads((output / 'icl_final_test.json').read_text())
    assert summary['scenario_results']['SS']['aggregate']['reported_tokens'] == result['reported_tokens']
    assert summary['scenario_results']['SS']['aggregate']['scheduling_time_seconds'] is None
    assert summary['scenario_results']['SS']['seed_records'][0]['scheduling_time_seconds'] is None
    bad = deepcopy(batches)
    bad[0]['feedback']['objective'] += 1
    with pytest.raises(AssertionError):
        episode(protocol, 'SS', 201, ReplayPolicy(config, bad))


def test_only_truncated_tail_is_accepted(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text('{"ok":1}\n{"bad":')
    assert list(json_rows(path)) == [{'ok': 1}]
    path.write_text('{"bad":\n{"ok":1}\n')
    with pytest.raises(ValueError, match='non-final'):
        list(json_rows(path))


def test_continue_keeps_completed_seeds_and_retries_only_failed_seed(tmp_path, monkeypatch):
    from tools import resume_dtecn_llm_batch as module
    from types import SimpleNamespace
    config = json.loads(CONFIG.read_text())
    protocol = FuzzyComparisonProtocol(scenario='SS', ddl_setting='L', train_seeds=(1,),
        validation_seeds=(101,), test_seeds=(201, 202), workflows_per_episode=1)
    (tmp_path / 'recovery_config.json').write_text(json.dumps(
        {'evaluation': {'service_settings': {'mock': True}}}))
    monkeypatch.setattr(module.runner, 'load_cases', lambda *a: [{}])
    monkeypatch.setattr(module, 'EvaluationLLMClient', lambda *a: SimpleNamespace(service_identity={'mock': True}))
    calls = []
    def evaluate(p, *a):
        seed = p.test_seeds[0]
        calls.append(seed)
        if seed == 202 and calls.count(202) == 1:
            raise OSError('disk full')
        return {'scenario_results': {'SS': {'seed_records': [{'scheduling_time_seconds': 1}]}}}
    monkeypatch.setattr(module, 'evaluate_with_csv', evaluate)
    remaining = [('SS', 201), ('SS', 202)]
    with pytest.raises(OSError):
        continue_evaluation(tmp_path, protocol, config, tmp_path, remaining, 1)
    original = (tmp_path / 'SS_201.json').read_bytes()
    continue_evaluation(tmp_path, protocol, config, tmp_path, remaining, 1)
    assert calls == [201, 202, 202]
    assert (tmp_path / 'SS_201.json').read_bytes() == original
    assert len(list((tmp_path / 'SS_202_attempts').iterdir())) == 2
