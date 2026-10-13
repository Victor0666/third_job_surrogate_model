"""Evaluation-only request settings, independent of the MADDPG teacher."""
from copy import deepcopy
from unittest.mock import patch

from algorithms.comparisons.dtecn_llm import policy as api


def evaluation_llm_config(teacher, model=None, effort=None, tokens=None, thinking="auto"):
    config = deepcopy(teacher)
    config["model"] = model or teacher["model"]
    changed = config["model"] != teacher["model"]
    config["reasoning_effort"] = effort or (teacher["reasoning_effort"] if not changed else None)
    config["max_tokens"] = tokens if tokens is not None else teacher["max_tokens"]
    config["thinking"] = thinking
    if type(config["max_tokens"]) is not int or config["max_tokens"] <= 0:
        raise ValueError("max_tokens must be a positive integer")
    qwen38 = config["model"].startswith("qwen3.8-max")
    gpt61 = config["model"].startswith("gpt-6.1")
    if effort is not None and not (qwen38 or gpt61):
        raise ValueError("explicit reasoning-effort is not supported by this model adapter")
    if thinking != "auto" and not config["model"].startswith(("qwen-plus", "qwen-flash", "qwen3.8-max")):
        raise ValueError("explicit thinking is not supported by this model adapter; use auto")
    if thinking == "off" and effort is not None:
        raise ValueError("reasoning-effort conflicts with thinking off")
    if not (qwen38 or gpt61):
        config["reasoning_effort"] = None
    request_parameters(config, "check")
    return config


def request_parameters(config, prompt):
    kwargs = api.request_parameters(config, prompt)
    mode = config["thinking"]
    if config["model"].startswith("qwen3.8-max"):
        enabled = config["enable_thinking"] if mode == "auto" else mode == "on"
        kwargs["extra_body"]["enable_thinking"] = enabled
        if enabled:
            kwargs["reasoning_effort"] = config["reasoning_effort"] or "medium"
        else:
            kwargs.pop("reasoning_effort", None)
    elif config["model"].startswith("gpt-6.1"):
        kwargs["reasoning_effort"] = config["reasoning_effort"] or "medium"
    elif config["model"].startswith("qwen") and mode != "auto":
        kwargs["extra_body"] = {"enable_thinking": mode == "on"}
    return kwargs


class EvaluationLLMClient(api.LLMClient):
    def __init__(self, config):
        super().__init__(config)
        self.service_identity["request_parameters"] = {
            k: v for k, v in request_parameters(config, "").items() if k != "messages"}

    def __call__(self, prompt):
        # Reuse stream/error handling; one request at a time in this evaluator.
        prepared = request_parameters(self.config, prompt)
        with patch.object(api, "request_parameters", lambda config, prompt: prepared):
            return super().__call__(prompt)
