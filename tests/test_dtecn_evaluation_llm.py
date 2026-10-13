import json
from types import SimpleNamespace

import pytest

from algorithms.comparisons.dtecn_llm.run import CONFIG
from tools.dtecn_evaluation_llm import EvaluationLLMClient, evaluation_llm_config, request_parameters


def test_qwen_plus_does_not_inherit_teacher_reasoning():
    teacher = json.loads(CONFIG.read_text())["llm"]
    teacher.update(model="qwen3.8-max", reasoning_effort="medium")
    runtime = evaluation_llm_config(teacher, model="qwen-plus", tokens=32768)
    kwargs = request_parameters(runtime, "state")
    assert kwargs["model"] == "qwen-plus"
    assert "reasoning_effort" not in kwargs and "extra_body" not in kwargs
    assert runtime["reasoning_effort"] is None
    assert teacher["model"] == "qwen3.8-max"
    assert request_parameters(evaluation_llm_config(teacher, model="qwen-plus", thinking="off"), "state")["extra_body"] == {"enable_thinking": False}
    with pytest.raises(ValueError, match="reasoning-effort"):
        evaluation_llm_config(teacher, model="qwen-plus", effort="medium")


def test_qwen38_gpt61_and_unsupported_options():
    teacher = json.loads(CONFIG.read_text())["llm"]
    runtime = evaluation_llm_config(teacher, model="qwen3.8-max", effort="high", thinking="on")
    kwargs = request_parameters(runtime, "state")
    assert kwargs["reasoning_effort"] == "high" and kwargs["extra_body"]["enable_thinking"]
    kwargs = request_parameters(evaluation_llm_config(teacher, model="qwen3.8-max", thinking="off"), "state")
    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"]["enable_thinking"] is False
    assert request_parameters(evaluation_llm_config(teacher, model="gpt-6.1-sol"), "state")["reasoning_effort"] == "medium"
    with pytest.raises(ValueError):
        evaluation_llm_config(teacher, model="unknown", thinking="on")
    with pytest.raises(ValueError):
        evaluation_llm_config(teacher, tokens=0)
    with pytest.raises(ValueError):
        evaluation_llm_config(teacher, model="qwen3.8-max", effort="high", thinking="off")


def test_client_sends_resolved_parameters_and_restores_api_function():
    from algorithms.comparisons.dtecn_llm import policy as api
    teacher = json.loads(CONFIG.read_text())["llm"]
    config = evaluation_llm_config(teacher, model="qwen-plus", thinking="off")
    client = EvaluationLLMClient.__new__(EvaluationLLMClient)
    client.config = config
    client.service_identity = dict(returned_models=[], system_fingerprints=[])
    class Stream:
        closed = False
        def __iter__(self):
            yield SimpleNamespace(model="qwen-plus-snapshot", system_fingerprint=None, usage=None,
                choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content='{"assignments":[]}'), finish_reason="stop")])
        def close(self):
            self.closed = True
    stream, sent = Stream(), {}
    def create(**kwargs):
        sent.update(kwargs)
        return stream
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    original = api.request_parameters
    content, usage = client("state")
    assert content == '{"assignments":[]}' and usage is None and stream.closed
    assert sent["extra_body"] == {"enable_thinking": False} and "reasoning_effort" not in sent
    assert api.request_parameters is original
