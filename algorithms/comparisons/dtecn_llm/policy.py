"""FCFS, one joint resource decision, observed feedback and online case updates."""
import json
import time
import os
from urllib.parse import urlsplit, urlunsplit
from importlib.metadata import version

import numpy as np

from .environment import FAMILIES, interval_feedback
from .maddpg import choose_vm

PROMPT = """You allocate resources to the current FCFS task in a DAG workflow.
Do not choose or reorder tasks. Output only JSON {\"resource_preferences\": [numbers]}.
Each number is in [0,1]; length equals the number of VMs. The highest preference
among legal VMs selects the complete task's VM; Host is implicit. Partial
offloading and CPU frequency control are unavailable in this shared simulator.
Minimize J = beta*sum(Q_k*(A_k-S_k)) + observed_energy_J
- system_QoS. Q is arrived unfinished CPU work by workflow family, A is new
arrived work, S is executed CPU work. Transmission does not serve CPU queues.
QoS = 1 - elapsed_workflow_time/deadline_budget. The examples' MADDPG reward
is QoS - eta*unused_capacity_fraction (Eq.27 adaptation), separate from J.
Feedback contains only already observed intervals, never future rollouts.
Observation order: active, current_FCFS_owner, log1p(remaining_MI), elapsed/budget,
five family indicators (CyberShake,Epigenomics,Ligo,Montage,Sipht),
log1p(task_MI), log1p(input_bits), log1p(output_bits), task_deadline_remaining/budget,
five log1p(queue_MI), then for each VM log1p(pc), log1p(bandwidth), log1p(busy_seconds).
arrived_workflows contains the first 13 local fields for every already arrived
workflow, as global observed context; missing future slots have no information.
Learn from the state/action/reward examples and their queue objective feedback.
"""


def parse_preferences(content, count):
    data = json.loads(content)
    if not isinstance(data, dict) or set(data) != {"resource_preferences"}:
        raise ValueError("expected resource_preferences JSON object")
    values = data["resource_preferences"]
    if not isinstance(values, list) or len(values) != count or any(
        isinstance(v, bool) or not isinstance(v, (float, int)) for v in values
    ):
        raise ValueError("preference vector type/length mismatch")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("preferences must be finite and in [0,1]")
    return values


class LLMClient:
    def __init__(self, config):
        from openai import OpenAI
        model = config["model"]
        if model.startswith("gpt"):
            key, endpoint = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        elif model.startswith("qwen"):
            key, endpoint = os.getenv("QWEN_API_KEY"), os.getenv("QWEN_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        else:
            key, endpoint = os.getenv("CUSTOM_API_KEY", "EMPTY"), os.getenv("CUSTOM_API_BASE_URL", "http://localhost:8000/v1/")
        if not key:
            raise ValueError("Missing API credentials for selected model")
        self.client = OpenAI(api_key=key, base_url=endpoint, timeout=config["timeout"], max_retries=0)
        self.config = config
        parsed = urlsplit(endpoint)
        self.service_identity = dict(requested_model=model, endpoint=urlunsplit((parsed.scheme,
            parsed.netloc.split("@")[-1], parsed.path, "", "")), sdk_version=version("openai"),
            returned_models=[], system_fingerprints=[])

    def _record_identity(self, response):
        for attribute, field in (("model", "returned_models"), ("system_fingerprint", "system_fingerprints")):
            value = getattr(response, attribute, None)
            if value and value not in self.service_identity[field]:
                self.service_identity[field].append(value)

    def __call__(self, prompt):
        kwargs = request_parameters(self.config, prompt)
        response = self.client.chat.completions.create(**kwargs)
        if kwargs["stream"]:
            parts, usage, finished = [], None, False
            started = time.perf_counter()
            try:
                for chunk in response:
                    self._record_identity(chunk)
                    if time.perf_counter() - started > self.config["timeout"]:
                        raise TimeoutError("LLM stream exceeded total request timeout")
                    if chunk.usage is not None:
                        usage = chunk.usage.model_dump()
                    for choice in chunk.choices:
                        if choice.index != 0:
                            raise ValueError("unexpected response index")
                        # Reasoning is not a resource action; parse final content only.
                        if choice.delta.content is not None:
                            parts.append(choice.delta.content)
                        if choice.finish_reason is not None:
                            if choice.finish_reason != "stop":
                                raise ValueError("incomplete LLM response")
                            finished = True
                content = "".join(parts)
                if not finished or not content.strip():
                    raise ValueError("empty or interrupted LLM stream")
                return content, usage
            finally:
                response.close()
        if len(response.choices) != 1 or response.choices[0].finish_reason != "stop":
            raise ValueError("incomplete LLM response")
        usage = response.usage.model_dump() if response.usage is not None else None
        self._record_identity(response)
        return response.choices[0].message.content, usage


def request_parameters(config, prompt):
    model = config["model"]
    gpt61 = model.startswith("gpt-6.1")
    qwen = model.startswith("qwen")
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], stream=gpt61 or qwen)
    if gpt61:
        kwargs.update(reasoning_effort=config["reasoning_effort"], max_completion_tokens=config["max_tokens"], n=1)
    else:
        kwargs.update(temperature=config["temperature"], top_p=config["top_p"], max_tokens=config["max_tokens"])
    if model.startswith("qwen3.8-max"):
        kwargs.update(reasoning_effort=config["reasoning_effort"], extra_body={
            "enable_thinking": config["enable_thinking"], "preserve_thinking": False})
    if kwargs["stream"]:
        kwargs["stream_options"] = {"include_usage": True}
    return kwargs


class DtecnPolicy:
    method_id = "dtecn_llm"

    def __init__(self, config, agent=None, client=None, initial_cases=(), collect=False):
        self.config, self.agent, self.client = config, agent, client
        self.initial_cases = tuple(initial_cases)
        self.collect = collect
        self.collected = []

    def begin_episode(self, env, *, training):
        env.set_task_orderer(None, training=training)
        self.pending = None
        self.online_cases = []
        self.decision_records = []
        self.stats = dict(requests=0, failures=0, fallbacks=0, single_legal_skips=0,
                          llm_seconds=0.0, reported_tokens=0, usage_missing=0,
                          lyapunov_objective_sum=0.0, feedback_count=0)

    def prompt(self, state):
        budget = self.config["llm"]["prompt_char_budget"]
        initial = list(self.initial_cases[:self.config["llm"]["initial_example_count"]])
        recent = self.online_cases[-self.config["llm"]["online_example_count"]:]
        prefix = PROMPT + "\nConstants: " + json.dumps(self.config["objective"], sort_keys=True)
        while True:
            value = prefix + "\nMARL examples: " + json.dumps(initial, separators=(",", ":"))
            value += "\nObserved online examples: " + json.dumps(recent, separators=(",", ":"))
            value += "\nCurrent state: " + json.dumps(state, separators=(",", ":"))
            if len(value) <= budget:
                return value
            if recent:
                recent.pop(0)
            elif len(initial) > 1:
                initial.pop()
            else:
                raise ValueError("prompt budget cannot hold current state and one initial example")

    def _finalize(self, env, after, observations, masks, terminal, training):
        if self.pending is None:
            return
        p = self.pending
        rewards, feedback = interval_feedback(env, p["snapshot"], after, self.config["objective"])
        done = np.maximum(1 - after["active"], float(terminal)).astype(np.float32)
        if training:
            self.agent.transition(p["observations"], p["actions"], rewards, observations, masks, done)
        case = dict(state=p["state"], action=p["preferences"].tolist(),
                    reward=float(rewards[p["owner"]]), feedback=feedback,
                    executed_vm_id=p["vm_id"], action_source=p["source"])
        self.stats["lyapunov_objective_sum"] += feedback["objective"]
        self.stats["feedback_count"] += 1
        self.decision_records.append(case)
        if self.collect:
            self.collected.append(dict(case, seed=int(env.random_seed), scenario=env.scenario_code))
        if self.client is not None:
            self.online_cases.append(case)
        self.pending = None

    def assign(self, env, host_state, *, training):
        del host_state
        snapshot = env.snapshot()
        observations, masks, owner = env.observations(snapshot)
        self._finalize(env, snapshot, observations, masks, False, training)
        state = dict(workflow_slot=owner, observation=observations[owner].tolist(),
                     legal_mask=masks[owner].tolist(), queue_mi=snapshot["queue"].tolist(),
                     family_order=list(FAMILIES), vm_ids=list(map(int, env.vm_ids)),
                     arrived_workflows=[dict(slot=i, local_observation=observations[i, :13].tolist())
                                        for i in range(len(env.workflows))])
        source = "maddpg"
        if self.client is None:
            preferences = self.agent.preferences(observations, owner, training)
        else:
            if training:
                raise ValueError("LLM policy does not train MADDPG on online/test feedback")
            legal = np.flatnonzero(masks[owner] > 0.5)
            preferences = np.zeros(env.num_vms)
            preferences[legal[0]] = 1
            source = "single_legal"
            if len(legal) == 1:
                self.stats["single_legal_skips"] += 1
            else:
                prompt = self.prompt(state)
                self.stats["requests"] += 1
                started = time.perf_counter()
                usage = None
                try:
                    content, usage = self.client(prompt)
                    preferences = parse_preferences(content, env.num_vms)
                    source = "llm"
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    if status in (400, 401, 403, 404):
                        raise
                    if not isinstance(exc, (ValueError, TimeoutError, ConnectionError)):
                        from openai import APIConnectionError, APIStatusError
                        if not isinstance(exc, (APIConnectionError, APIStatusError)):
                            raise
                    self.stats["failures"] += 1
                    self.stats["fallbacks"] += 1
                    source = "legal_fallback"
                    state["request_error"] = type(exc).__name__
                finally:
                    self.stats["llm_seconds"] += time.perf_counter() - started
                    if usage is None or usage.get("total_tokens") is None:
                        self.stats["usage_missing"] += 1
                    else:
                        self.stats["reported_tokens"] += int(usage["total_tokens"])
        index = choose_vm(preferences, masks[owner])
        actions = np.zeros((env.workflows_per_episode, env.num_vms), dtype=np.float32)
        actions[owner] = preferences
        result = env.assign_global_vm(index)
        self.pending = dict(snapshot=snapshot, observations=observations, actions=actions,
                            owner=owner, state=state, preferences=np.asarray(preferences),
                            vm_id=result.vm_id, source=source)
        return result

    def observe_assignment(self, env, result, *, training):
        pass

    def end_phase(self, env, reward, info, *, training):
        pass

    def end_episode(self, env, *, training):
        env._cur_tid = None
        snapshot = env.snapshot()
        observations, masks, _ = env.observations(snapshot)
        self._finalize(env, snapshot, observations, masks, True, training)

    def save(self, path):
        self.agent.save(path)

    def load(self, path):
        self.agent.load(path)
