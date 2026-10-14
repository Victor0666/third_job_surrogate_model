"""Recover completed episodes offline, then resume at an episode boundary.

Original evaluation files are read-only. New records live in recovery_v1.
"""
import argparse
import csv
from copy import deepcopy
from dataclasses import fields, replace
import itertools
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from algorithms.comparisons.dtecn_llm import run as runner
from algorithms.comparisons.fuzzy_common.protocol import FuzzyComparisonProtocol
from tools.dtecn_batch_policy import BatchPolicy, parse_assignments
from tools.dtecn_evaluation_llm import EvaluationLLMClient
from tools.evaluate_dtecn_llm_with_csv import evaluate_with_csv, load_compatible_manifest


REQUEST_FIELDS = ('requests', 'failures', 'fallbacks', 'llm_seconds',
                  'reported_tokens', 'usage_missing')


def json_rows(path):
    """Only a malformed final line can be a disk-full tail."""
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if stream.read(1):
                    raise ValueError(f'corrupt non-final line: {path}:{number}')
                print(f'Ignoring truncated final line: {path}:{number}', flush=True)


def key(row):
    return row['scenario'], int(row['seed'])


def assert_same(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('replay dictionary fields differ')
        for name in expected:
            assert_same(actual[name], expected[name])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('replay list length differs')
        for a, b in zip(actual, expected):
            assert_same(a, b)
    elif isinstance(expected, (int, float)):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-7)
    elif actual != expected:
        raise ValueError(f'replay value differs: {actual!r} != {expected!r}')


class ReplayPolicy(BatchPolicy):
    def __init__(self, config, batches):
        # A sentinel preserves online_icl identity, but is never called.
        super().__init__(config, client=lambda _: (_ for _ in ()).throw(
            RuntimeError('offline replay must never call an API')))
        self.saved_batches = batches
        self.cursor = 0
        self.batch_callback = self.verify_feedback

    def verify_feedback(self, case):
        saved = self.saved_batches[self.cursor - 1]
        for name in ('batch_id', 'state', 'assignments', 'rewards', 'feedback',
                     'action_source', 'seed', 'scenario', 'request'):
            assert_same(case[name], saved[name])

    def _start_batch(self, env):
        self._finish_batch(env)
        if self.cursor >= len(self.saved_batches):
            raise ValueError('completed seed has missing batch records')
        saved = self.saved_batches[self.cursor]
        self.cursor += 1
        tasks = saved['state']['tasks']
        self.batch_id += 1
        if saved['batch_id'] != self.batch_id or key(saved) != (env.scenario_code, int(env.random_seed)):
            raise ValueError('batch seed/order mismatch')
        assert_same(float(env.current_time), saved['state']['time'])
        snapshot = env.snapshot()
        assert_same(snapshot['queue'].tolist(), saved['state']['queue_mi'])
        candidates = [task['legal_vm_ids'] for task in tasks]
        plan = parse_assignments(json.dumps({'assignments': saved['assignments']}),
                                 [t['task_id'] for t in tasks], candidates)
        self.stats['batches'] += 1
        self.stats['unique_batch_skips'] += int(saved['action_source'] == 'unique_batch')
        for name in REQUEST_FIELDS:
            self.stats[name] += saved['request'][name]
        self.batch = dict(id=self.batch_id, before=snapshot, state=saved['state'],
            tasks=tasks, candidates=candidates, owners=[t['workflow_slot'] for t in tasks],
            plan=plan, source=saved['action_source'], assignments=[], next=0, used=set(),
            request=saved['request'])

    def end_episode(self, env, *, training):
        super().end_episode(env, training=training)
        if self.cursor != len(self.saved_batches):
            raise ValueError('unused batch records after replay')


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    runner.write_json(temporary, data)
    temporary.replace(path)


def load_context(root):
    saved = json.loads((root / 'evaluation_config.json').read_text(encoding='utf-8'))
    for path, digest in saved['evaluation_source_hashes'].items():
        if runner.file_hash(Path(path)) != digest:
            raise ValueError(f'evaluation source changed: {path}')
    batch_root = root.parent.parent
    identity = saved['case_identity']
    if runner.file_hash(Path(__file__).with_name('dtecn_batch_policy.py')) != identity['batch_policy_sha256']:
        raise ValueError('batch policy source changed')
    teacher = identity['teacher_identity']
    config = deepcopy(teacher['config'])
    protocol_data = teacher['protocol']
    protocol = FuzzyComparisonProtocol(**{
        f.name: protocol_data[f.name] for f in fields(FuzzyComparisonProtocol)
    })
    if json.loads(json.dumps(protocol.to_manifest())) != protocol_data:
        raise ValueError('protocol reconstruction mismatch')
    expected = runner.identity(protocol, config, teacher['optimizer_seed'], False)
    load_compatible_manifest(batch_root.parent, expected)
    manifest = json.loads((batch_root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['identity'] != identity:
        raise ValueError('batch manifest identity changed')
    for name, digest in manifest['artifacts'].items():
        if Path(name).name != name or runner.file_hash(batch_root / name) != digest:
            raise ValueError(f'batch artifact changed: {name}')
    if runner.file_hash(batch_root / 'cases.jsonl') != saved['cases_sha256']:
        raise ValueError('case library changed')
    config['llm'] = deepcopy(saved['llm_config'])
    return saved, protocol, config, batch_root, manifest


def recover(root, output, protocol, config, completed):
    decisions = itertools.groupby(json_rows(root / 'icl_final_test_decisions.jsonl'), key)
    batches = itertools.groupby(json_rows(root / 'icl_final_test_batches.jsonl'), key)
    for expected in completed:
        dk, dr = next(decisions, (None, ()))
        bk, br = next(batches, (None, ()))
        if dk != expected or bk != expected:
            raise ValueError(f'missing completed seed {expected}: decisions={dk}, batches={bk}')
        dr, br = list(dr), list(br)
        path = output / f'{expected[0]}_{expected[1]}.json'
        if path.exists():
            continue
        policy = ReplayPolicy(config, br)
        record = runner.episode(protocol, *expected, policy)
        if policy.decision_records != [
            {k: value for k, value in row.items() if k not in ('scenario', 'seed')} for row in dr
        ]:
            raise ValueError(f'replayed decisions differ: {expected}')
        record.update(recovered_from_actions=True, original_wall_time_available=False,
                      replay_seconds=record['scheduling_time_seconds'])
        atomic_json(path, record)
        print(f'Recovered without API: {expected[0]}/{expected[1]}', flush=True)
    extra, _ = next(decisions, (None, ()))
    if extra is not None:
        raise ValueError(f'completed decision log has unexpected seed: {extra}')


def continue_evaluation(output, protocol, config, batch_root, remaining, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cases = runner.load_cases(batch_root, protocol)
    indices = np.linspace(0, len(cases) - 1,
                          min(config['llm']['initial_example_count'], len(cases)), dtype=int)
    client = EvaluationLLMClient(config['llm'])
    saved_service = json.loads((output / 'recovery_config.json').read_text())['evaluation']['service_settings']
    actual = deepcopy(client.service_identity)
    for name in ('sdk_version', 'returned_models', 'system_fingerprints'):
        actual.pop(name, None)
    if actual != saved_service:
        raise ValueError('evaluation service settings changed')
    for scenario, episode_seed in remaining:
        path = output / f'{scenario}_{episode_seed}.json'
        if path.exists():
            continue
        # Failed attempts remain separate; a fresh attempt restarts this seed only.
        attempts = output / f'{scenario}_{episode_seed}_attempts'
        attempts.mkdir(exist_ok=True)
        attempt = attempts / str(len(list(attempts.iterdir())) + 1)
        attempt.mkdir()
        with (attempt / 'icl_final_test_batches.jsonl').open('x', encoding='utf-8') as stream:
            def callback(case):
                stream.write(json.dumps(case, allow_nan=False) + '\n')
                stream.flush()
            policy = BatchPolicy(config, client=client, initial_cases=[cases[i] for i in indices],
                                 batch_callback=callback)
            subset = replace(protocol, test_scenarios=(scenario,), test_seeds=(episode_seed,))
            print(f'Evaluating: {scenario}/{episode_seed}', flush=True)
            result = evaluate_with_csv(subset, policy, 'final_test',
                                      attempt / 'icl_final_test_decisions.jsonl')
        record = result['scenario_results'][scenario]['seed_records'][0]
        record.update(recovered_from_actions=False, original_wall_time_available=True)
        atomic_json(path, record)
        print(f'Saved completed seed: {scenario}/{episode_seed}', flush=True)


def summarize(output, protocol, saved, manifest):
    records = {(s, seed): json.loads((output / f'{s}_{seed}.json').read_text())
               for s in protocol.test_scenarios for seed in protocol.test_seeds}
    stats = dict.fromkeys((*REQUEST_FIELDS, 'single_legal_skips', 'lyapunov_objective_sum',
                          'feedback_count', 'batches', 'unique_batch_skips'), 0)
    with patch.object(runner, 'episode', lambda p, s, seed, policy: records[(s, seed)]):
        result = runner.evaluate(protocol, SimpleNamespace(stats=stats), 'final_test')
    # Replay wall time is not original scheduling overhead; never publish it as such.
    for value in result['scenario_results'].values():
        if any(r['recovered_from_actions'] for r in value['seed_records']):
            for field in ('scheduling_time_seconds', 'mean_seed_scheduling_time_seconds',
                          'worst_seed_scheduling_time_seconds'):
                value['aggregate'][field] = None
        for record in value['seed_records']:
            if record['recovered_from_actions']:
                record['scheduling_time_seconds'] = None
    result.update(identity=saved, recovery_mode='completed_seed_action_replay',
                  original_wall_time_available=False,
                  cases_sha256=saved['cases_sha256'],
                  checkpoint_sha256=saved['case_identity']['teacher_checkpoint_sha256'])
    result['preparation_cost'] = {k: v for k, v in manifest.items()
                                 if k not in ('identity', 'artifacts')}
    atomic_json(output / 'icl_final_test.json', result)
    rows = [dict(scenario=s, **v['aggregate']) for s, v in result['scenario_results'].items()]
    with (output / 'icl_final_test.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('recover', 'continue'))
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--restart-scenario', required=True)
    parser.add_argument('--restart-seed', type=int, required=True)
    args = parser.parse_args()
    root = args.evaluation_dir.resolve()
    saved, protocol, config, batch_root, manifest = load_context(root)
    torch.set_num_threads(1)
    ordered = [(s, seed) for s in protocol.test_scenarios for seed in protocol.test_seeds]
    boundary = ordered.index((args.restart_scenario, args.restart_seed))
    output = root / 'recovery_v1'
    output.mkdir(exist_ok=True)
    recovery_identity = dict(evaluation=saved, restart=[args.restart_scenario, args.restart_seed],
                             recovery_source_sha256=runner.file_hash(Path(__file__)))
    settings = output / 'recovery_config.json'
    if settings.exists():
        if json.loads(settings.read_text()) != recovery_identity:
            raise ValueError('recovery identity changed')
    else:
        atomic_json(settings, recovery_identity)
    if args.stage == 'recover':
        recover(root, output, protocol, config, ordered[:boundary])
    else:
        if any(not (output / f'{s}_{seed}.json').exists() for s, seed in ordered[:boundary]):
            raise ValueError('run recover first')
        continue_evaluation(output, protocol, config, batch_root, ordered[boundary:],
                            saved['case_identity']['teacher_identity']['optimizer_seed'])
        summarize(output, protocol, saved, manifest)
    print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
