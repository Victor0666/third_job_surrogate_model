import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from algorithms.comparisons.dtecn_llm.environment import completed_work, DtecnEnv, interval_feedback
from algorithms.comparisons.dtecn_llm.maddpg import MADDPG, choose_vm
from algorithms.comparisons.dtecn_llm.policy import DtecnPolicy, parse_preferences, request_parameters, LLMClient
from algorithms.comparisons.dtecn_llm.run import CONFIG, identity, load_manifest, load_cases, write_json, file_hash
from algorithms.comparisons.fuzzy_common.protocol import FuzzyComparisonProtocol
from algorithms.comparisons.fuzzy_common.evaluation import run_episode, make_environment


class DtecnTests(unittest.TestCase):
    def test_hidden_width_and_old_checkpoint_rejection(self):
        config = json.loads(CONFIG.read_text())
        agent = MADDPG(1, 5, 25, config["maddpg"])
        for network in (agent.actors[0], agent.critics[0]):
            self.assertEqual(network.net[0].out_features, 512)
            self.assertEqual(network.net[2].in_features, 512)
            self.assertEqual(network.net[2].out_features, 256)
            self.assertEqual(network.net[4].in_features, 256)
        self.assertEqual(agent.actors[0].net[4].out_features, 25)
        self.assertEqual(agent.critics[0].net[4].out_features, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            agent.save(path)
            data = torch.load(path, weights_only=True)
            for old_schema in (1, 2):
                data["schema"] = old_schema
                torch.save(data, path)
                with self.assertRaises(ValueError):
                    agent.load(path)

    def test_feedback_has_no_cross_state_improvement_flag(self):
        config = json.loads(CONFIG.read_text())
        protocol = FuzzyComparisonProtocol(scenario="SS", ddl_setting="T", train_seeds=(1,),
            validation_seeds=(101,), test_seeds=(201,), workflows_per_episode=1)
        env = DtecnEnv(**protocol.environment_kwargs(1))
        agent = MADDPG(1, env.observation_dim, env.num_vms, config["maddpg"])
        policy = DtecnPolicy(config, agent=agent, collect=True)
        run_episode(env, policy, seed=1, training=False)
        self.assertTrue(policy.collected)
        self.assertTrue(all("improved_objective" not in case for case in policy.collected))

    def test_service_identity_records_actual_model(self):
        client = LLMClient.__new__(LLMClient)
        client.service_identity = dict(returned_models=[], system_fingerprints=[])
        response = SimpleNamespace(model="gpt-6.1-sol-snapshot", system_fingerprint="backend1")
        client._record_identity(response)
        client._record_identity(response)
        self.assertEqual(client.service_identity["returned_models"], ["gpt-6.1-sol-snapshot"])
        self.assertEqual(client.service_identity["system_fingerprints"], ["backend1"])
    def test_model_specific_requests_and_streams(self):
        config = json.loads(CONFIG.read_text())["llm"]
        for model in ("gpt-6.1-sol", "qwen3.8-max", "meta/llama-3.1-8b-instruct"):
            config["model"] = model
            args = request_parameters(config, "state")
            if model.startswith("gpt"):
                self.assertNotIn("temperature", args)
                self.assertNotIn("top_p", args)
                self.assertNotIn("max_tokens", args)
                self.assertEqual(args["reasoning_effort"], "medium")
            elif model.startswith("qwen"):
                self.assertTrue(args["extra_body"]["enable_thinking"])
                self.assertFalse(args["extra_body"]["preserve_thinking"])
            else:
                self.assertFalse(args["stream"])
                self.assertEqual(args["temperature"], 0.2)
                continue
            self.assertTrue(args["stream_options"]["include_usage"])
            usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 23})
            chunk = lambda text, finish=None: SimpleNamespace(usage=None, choices=[SimpleNamespace(
                index=0, delta=SimpleNamespace(content=text, reasoning_content="not JSON"), finish_reason=finish)])
            class Stream:
                closed = False
                def __iter__(self):
                    yield chunk(None)
                    yield chunk('{"resource_preferences":')
                    yield chunk('[0.4]}', 'stop')
                    yield SimpleNamespace(usage=usage, choices=[])
                def close(self):
                    self.closed = True
            stream = Stream()
            captured = {}
            def create(**kwargs):
                captured.update(kwargs)
                return stream
            client = LLMClient.__new__(LLMClient)
            client.service_identity = dict(returned_models=[], system_fingerprints=[])
            client.config = dict(config)
            client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            content, tokens = client("state")
            self.assertEqual(parse_preferences(content, 1).tolist(), [0.4])
            self.assertEqual(tokens["total_tokens"], 23)
            self.assertTrue(stream.closed)
            self.assertEqual(captured, args)

    def test_truncated_reasoning_response_is_rejected(self):
        config = json.loads(CONFIG.read_text())["llm"]
        config["model"] = "gpt-6.1-sol"
        class Stream:
            closed = False
            def __iter__(self):
                yield SimpleNamespace(usage=None, choices=[SimpleNamespace(index=0,
                    delta=SimpleNamespace(content="{}"), finish_reason="length")])
            def close(self):
                self.closed = True
        stream = Stream()
        client = LLMClient.__new__(LLMClient)
        client.config = config
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: stream)))
        with self.assertRaises(ValueError):
            client("state")
        self.assertTrue(stream.closed)

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        np.random.seed(7)
        self.config = json.loads(CONFIG.read_text())
        self.config["maddpg"]["batch_size"] = 2
        self.protocol = FuzzyComparisonProtocol(scenario="SS", ddl_setting="T", train_seeds=(1,),
                                               validation_seeds=(101,), test_seeds=(201,), workflows_per_episode=1)

    def env(self, seed=1):
        return DtecnEnv(**self.protocol.environment_kwargs(seed))

    def test_cpu_work_excludes_transfers(self):
        self.assertEqual(completed_work(100, 10, 20, 5), 0)
        self.assertEqual(completed_work(100, 10, 20, 15), 50)
        self.assertEqual(completed_work(100, 10, 20, 25), 100)
        with self.assertRaises(ValueError):
            completed_work(100, 10, 10, 20)

    def test_queue_conservation_and_distinct_reward(self):
        before = dict(time=0, totals=np.array([100.]), service=np.array([0.]), queue=np.array([100.]))
        after = dict(time=1, totals=np.array([120.]), service=np.array([50.]), queue=np.array([70.]),
                     qos=np.array([0.8]), unused_capacity=0.25)
        env = SimpleNamespace(workflows=[1], observed_energy=lambda t: t * 100)
        reward, feedback = interval_feedback(env, before, after, self.config["objective"])
        self.assertAlmostEqual(reward[0], 0.55, places=6)
        self.assertAlmostEqual(feedback["drift_term"], -3000)
        self.assertAlmostEqual(feedback["objective"], -2900.8)
        self.assertNotAlmostEqual(reward[0], -feedback["objective"])
        after["queue"] = np.array([80.])
        with self.assertRaises(ValueError):
            interval_feedback(env, before, after, self.config["objective"])

    def test_json_and_legal_projection(self):
        self.assertEqual(choose_vm([1, 0.5, 0.5], [0, 1, 1]), 1)
        for bad in ('{"resource_preferences":[true]}', '{"resource_preferences":[NaN]}',
                    '{"resource_preferences":[1.1]}', '{"resource_preferences":[0.5],"host":1}', '[]'):
            with self.assertRaises(ValueError):
                parse_preferences(bad, 1)

    def test_joint_updates_and_target_soft_update(self):
        agent = MADDPG(2, 3, 2, self.config["maddpg"])
        obs = np.array([[1, 1, 0.2], [0, 0, 0]], dtype=np.float32)
        nxt = np.array([[1, 0, 0.3], [1, 1, 0.4]], dtype=np.float32)
        action = np.array([[0.4, 0.8], [0, 0]], dtype=np.float32)
        masks = np.array([[0, 0], [1, 1]], dtype=np.float32)
        old = [p.detach().clone() for p in agent.actors[0].parameters()]
        padded = [p.detach().clone() for p in agent.actors[1].parameters()]
        for _ in range(3):
            loss = agent.transition(obs, action, [1, 0], nxt, masks, [0, 0])
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old, agent.actors[0].parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(padded, agent.actors[1].parameters())))
        self.assertTrue(all(p.grad is None for p in agent.target_actors.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            agent.save(path)
            restored = MADDPG(2, 3, 2, self.config["maddpg"])
            restored.load(path)
            np.testing.assert_array_equal(agent.preferences(obs, 0, False), restored.preferences(obs, 0, False))

    def test_real_episode_training_fcfs_and_terminal_queue(self):
        env = self.env()
        agent = MADDPG(1, env.observation_dim, env.num_vms, self.config["maddpg"])
        policy = DtecnPolicy(self.config, agent=agent, collect=True)
        record = run_episode(env, policy, seed=1, training=True)
        self.assertGreater(record["assignment_steps"], 0)
        self.assertEqual(len(policy.collected), record["assignment_steps"])
        np.testing.assert_allclose(env.snapshot()["queue"], 0, atol=1e-6)
        self.assertTrue(np.all(agent.replay[-1][-1] == 1))
        self.assertFalse(env.safe_rl_enabled)
        self.assertFalse(env.safe_rl_shield_enabled)
        env.reset(seed=1)
        policy.begin_episode(env, training=False)
        env.get_host_state_for_next_assignment()
        self.assertEqual(env._phase_ready_task_ordering, sorted(env._phase_ready_task_ordering,
                         key=lambda t: (env.task_ready_time[t], t)))
        obs, _, _ = env.observations(env.snapshot())
        self.assertEqual(obs.shape, (1, env.observation_dim))

    def test_online_updates_reset_fallback_and_simulator_parity(self):
        def mock(prompt):
            state = json.loads(prompt.split("\nCurrent state: ")[1])
            return json.dumps({"resource_preferences": [0.5] * len(state["vm_ids"])}), None
        env = self.env()
        policy = DtecnPolicy(self.config, client=mock)
        first = run_episode(env, policy, seed=1, training=False)
        history = copy.deepcopy(env.assignment_history)
        cases = copy.deepcopy(policy.online_cases)
        second = run_episode(env, policy, seed=1, training=False)
        self.assertEqual(cases, policy.online_cases)
        for key in ("fuzzy_energy_score", "deadline_violation_rate", "max_fuzzy_lateness"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(len(policy.online_cases), first["assignment_steps"])
        base = make_environment(self.protocol, 1)
        # Replay identical global VM choices through the unchanged shared environment.
        replay = DtecnPolicy(self.config, client=mock)
        replay.begin_episode(base, training=False)
        base.reset(seed=1)
        index = 0
        while not base.done_flag:
            while True:
                _, available = base.get_host_state_for_next_assignment()
                if not available:
                    break
                expected = history[index]
                self.assertEqual(base._cur_tid, expected["task_id"])
                base.assign_global_vm(base.vm_ids.index(expected["vm_id"]))
                index += 1
            base.finish_phase_and_advance()
        self.assertEqual(history, base.assignment_history)
        self.assertEqual(env.get_fuzzy_energy_summary(), base.get_fuzzy_energy_summary())
        def invalid(prompt):
            return "not json", None
        failed = DtecnPolicy(self.config, client=invalid)
        run_episode(self.env(), failed, seed=1, training=False)
        self.assertGreater(failed.stats["fallbacks"], 0)
        self.assertEqual(failed.stats["fallbacks"], failed.stats["requests"])

    def test_unarrived_slots_are_zero(self):
        env = DtecnEnv(**dict(self.protocol.environment_kwargs(1), workflows_per_episode=3))
        env.reset(seed=1)
        env.get_host_state_for_next_assignment()
        obs, masks, _ = env.observations(env.snapshot())
        self.assertTrue(np.all(obs[len(env.workflows):] == 0))
        self.assertTrue(np.all(masks[len(env.workflows):] == 0))

    def test_multi_workflow_joint_training_and_online_causality(self):
        protocol = FuzzyComparisonProtocol(scenario="SS", ddl_setting="T", train_seeds=(1,),
                                          validation_seeds=(101,), test_seeds=(201,), workflows_per_episode=3)
        env = DtecnEnv(**protocol.environment_kwargs(1))
        agent = MADDPG(3, env.observation_dim, env.num_vms, self.config["maddpg"])
        policy = DtecnPolicy(self.config, agent=agent, collect=True)
        result = run_episode(env, policy, seed=1, training=True)
        self.assertEqual(result["completed_workflow_count"], 3)
        self.assertEqual(len(policy.collected), result["assignment_steps"])
        for case in policy.collected:
            self.assertGreaterEqual(case["feedback"]["elapsed"], 0)
            self.assertGreaterEqual(min(case["feedback"]["service_mi"]), -1e-7)
        np.testing.assert_allclose(env.snapshot()["queue"], 0, atol=1e-6)

    def test_prompt_retains_case_feedback_and_budget_is_checked(self):
        policy = DtecnPolicy(self.config, client=lambda p: ("", None),
                            initial_cases=[dict(state={}, action=[0.5], reward=0.2, feedback={"objective": 1})])
        policy.online_cases = []
        prompt = policy.prompt(dict(observation=[0.1], legal_mask=[1], vm_ids=[0]))
        self.assertIn('"objective":1', prompt)
        self.assertIn("MADDPG reward", prompt)
        policy.config = copy.deepcopy(self.config)
        policy.config["llm"]["prompt_char_budget"] = 10
        with self.assertRaises(ValueError):
            policy.prompt(dict(observation=[0.1], legal_mask=[1], vm_ids=[0]))

    def test_full_global_context_fits_and_invalid_response_counts_usage(self):
        state = dict(workflow_slot=49, observation=[0.123456789] * 93, legal_mask=[1] * 25,
                     vm_ids=list(range(25)), queue_mi=[1000] * 5,
                     arrived_workflows=[dict(slot=i, local_observation=[0.123456789] * 13) for i in range(50)])
        case = dict(state=state, action=[0.5] * 25, reward=0.2, feedback={"objective": 1})
        policy = DtecnPolicy(self.config, client=lambda p: ("", None), initial_cases=[case] * 8)
        policy.online_cases = [case] * 8
        prompt = policy.prompt(state)
        self.assertLessEqual(len(prompt), self.config["llm"]["prompt_char_budget"])
        self.assertIn('"slot":49', prompt)
        def invalid(prompt):
            return "invalid", {"total_tokens": 7}
        policy = DtecnPolicy(self.config, client=invalid)
        run_episode(self.env(), policy, seed=1, training=False)
        self.assertEqual(policy.stats["reported_tokens"], 7 * policy.stats["requests"])
        self.assertEqual(policy.stats["usage_missing"], 0)

    def test_identity_and_case_leak_guards(self):
        expected = identity(self.protocol, self.config, 0, True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.pt").write_bytes(b"checkpoint")
            manifest = dict(identity=expected, artifacts={"model.pt": file_hash(root / "model.pt")})
            write_json(root / "manifest.json", manifest)
            load_manifest(root, expected)
            (root / "model.pt").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                load_manifest(root, expected)
            (root / "cases.jsonl").write_text(json.dumps(dict(seed=201, scenario="SS")))
            with self.assertRaises(ValueError):
                load_cases(root, self.protocol)


if __name__ == "__main__":
    unittest.main()
