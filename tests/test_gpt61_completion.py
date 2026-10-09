"""Verify GPT-6.1 request compatibility without sending API requests."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from algorithms.llm_safe_hrl.LLM.utils import utils


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_gpt61_multi_preserves_candidate_count_and_order():
    create = Mock(side_effect=lambda **kw: response(kw["messages"][0]["content"]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(utils, "client", client, create=True):
        assert utils.multi_chat_completion(
            [[{"role": "user", "content": "first"}], [{"role": "user", "content": "second"}]],
            1, "gpt-6.1-sol", 0.0,
        ) == ["first", "second"]
        assert utils.multi_chat_completion(
            [{"role": "user", "content": "candidate"}], 3, "gpt-6.1-sol", 1.0,
        ) == ["candidate"] * 3
    assert create.call_count == 5
    for call in create.call_args_list:
        assert "temperature" not in call.kwargs
        assert call.kwargs["reasoning_effort"] == "medium"
        assert call.kwargs["n"] == 1


def test_direct_gpt61_call_preserves_n():
    create = Mock(side_effect=[response("one"), response("two")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(utils, "client", client, create=True):
        choices = utils.chat_completion(2, [], "gpt-6.1-sol", 0.0)
    assert [choice.message.content for choice in choices] == ["one", "two"]


def test_existing_gpt_and_qwen_request_parameters():
    create = Mock(side_effect=lambda **kw: SimpleNamespace(
        choices=response("ok").choices * kw.get("n", 1)))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(utils, "client", client, create=True):
        utils.chat_completion(3, [], "gpt-4.1", 2.0)
        assert create.call_args.kwargs == dict(model="gpt-4.1", messages=[], temperature=1.0, n=3)
        assert utils.multi_chat_completion(
            [{"role": "user", "content": "test"}], 2, "qwen-plus", 0.0,
        ) == ["ok", "ok"]
        assert create.call_args.kwargs == dict(
            model="qwen-plus", messages=[{"role": "user", "content": "test"}], temperature=0.0,
        )


def test_qwen38_max_and_snapshot_use_consistent_thinking_settings():
    create = Mock(return_value=response("code"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(utils, "client", client, create=True):
        for model in ("qwen3.8-max", "qwen3.8-max-0902"):
            assert utils.multi_chat_completion(
                [{"role": "user", "content": "generate"}], 2, model, 0.0,
            ) == ["code", "code"]
            assert create.call_args.kwargs == dict(
                model=model, messages=[{"role": "user", "content": "generate"}],
                temperature=0.0, reasoning_effort="medium",
                extra_body={"enable_thinking": True, "preserve_thinking": False},
            )
