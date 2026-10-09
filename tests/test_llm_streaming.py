"""Streaming transport tests without API access or model charges."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from algorithms.llm_safe_hrl.LLM.utils import utils


def chunk(index=0, text=None, finish=None, reasoning=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        index=index, delta=SimpleNamespace(content=text, reasoning_content=reasoning),
        finish_reason=finish,
    )])


class Stream:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error
        self.closed = False

    def __iter__(self):
        yield from self.events
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


def test_chunks_reassemble_in_choice_order_without_thinking_or_usage():
    stream = Stream([
        chunk(1, reasoning="hidden thought"),
        chunk(0, "```py"), chunk(1, "second"),
        chunk(0, "thon\n# 中文\n"), chunk(0, "return x\n```", "stop"),
        chunk(1, " reply", "stop"), SimpleNamespace(choices=[]),
    ])
    choices = utils._collect_stream_choices(stream, 2, utils.time.perf_counter())
    assert [c.message.content for c in choices] == ["```python\n# 中文\nreturn x\n```", "second reply"]
    assert stream.closed


@pytest.mark.parametrize("events", [
    [], [chunk(text="partial")], [chunk(text="partial", finish="length")],
    [chunk(finish="stop")], [chunk(index=9, text="bad", finish="stop")],
    [chunk(text="blocked", finish="content_filter")],
])
def test_incomplete_empty_or_truncated_stream_cannot_be_used_as_code(events):
    stream = Stream(events)
    with pytest.raises(utils.LLMRequestError):
        utils._collect_stream_choices(stream, 1, utils.time.perf_counter())
    assert stream.closed


def test_stream_timeout_closes_connection():
    stream = Stream([chunk(text="late", finish="stop")])
    with patch.object(utils, "_api_timeout", 3), patch.object(utils.time, "perf_counter", return_value=4):
        with pytest.raises(utils.LLMRequestError, match="timeout"):
            utils._collect_stream_choices(stream, 1, 0)
    assert stream.closed


def test_midstream_failure_retries_without_concatenating_old_partial_text():
    broken = Stream([chunk(text="corrupted prefix")], OSError("connection reset"))
    good = Stream([chunk(text="complete code", finish="stop")])
    create = Mock(side_effect=[broken, good])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(utils, "client", client, create=True), patch.object(utils.time, "sleep"):
        choices = utils.chat_completion(1, [], "gpt-6.1-sol", 1)
    assert choices[0].message.content == "complete code"
    assert create.call_count == 2
    assert broken.closed and good.closed
    assert all(call.kwargs["stream"] for call in create.call_args_list)


def test_other_provider_keeps_existing_nonstream_interface():
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(utils, "client", client, create=True):
        assert utils.chat_completion(1, [], "GLM-4", .5)[0].message.content == "ok"
    assert "stream" not in create.call_args.kwargs
