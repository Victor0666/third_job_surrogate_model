"""No-network regression checks for long-running offline rule generation."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading
import time

import pytest

from algorithms.llm_safe_hrl.LLM.utils import utils
from algorithms.llm_safe_hrl.LLM.surrogate.manager import _label
from algorithms.llm_safe_hrl.paths import LLM_ROOT
from rule_optimization import OptimizerConfig, RuleValidationError, parse_rule_candidate
from seevo import SeEvo


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_permanent_error_never_retries_or_exits_successfully():
    for error in (TypeError("bad SDK argument"), SimpleAPIError(401)):
        create = Mock(side_effect=error)
        with patch.object(utils, "client", client(create), create=True), patch.object(utils.time, "sleep") as sleep:
            with pytest.raises(RuntimeError, match="Permanent"):
                utils.chat_completion(1, [], "qwen-plus", 1)
        assert create.call_count == 1
        sleep.assert_not_called()


class SimpleAPIError(Exception):
    def __init__(self, status):
        self.status_code = status
        super().__init__(str(status))


def test_temporary_and_empty_responses_have_finite_retry_limit():
    for result in (SimpleAPIError(503), response(None), response("")):
        create = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
        with patch.object(utils, "client", client(create), create=True), patch.object(utils.time, "sleep"):
            with pytest.raises(utils.LLMRequestError):
                utils.chat_completion(1, [], "qwen-plus", 1)
        assert create.call_count == 3


def test_partial_batch_preserves_successful_candidate_positions():
    def complete(n, messages, model, temperature):
        if messages[0]["content"] == "bad":
            raise utils.LLMRequestError("timeout")
        return response(messages[0]["content"]).choices
    messages = [[{"role": "user", "content": value}] for value in ("first", "bad", "last")]
    with patch.object(utils, "chat_completion", side_effect=complete):
        assert utils.multi_chat_completion(messages, 1, "qwen-plus", 1, allow_partial=True) == ["first", "", "last"]
        with pytest.raises(utils.LLMRequestError):
            utils.multi_chat_completion(messages[1:2], 1, "qwen-plus", 1, allow_partial=True)


def test_api_concurrency_is_bounded_across_independent_batches():
    lock = threading.Lock()
    active = peak = 0
    def create(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.01)
        with lock:
            active -= 1
        return response("ok")
    with patch.object(utils, "client", client(create), create=True), \
         patch.object(utils, "_api_slots", threading.BoundedSemaphore(2)):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(utils.multi_chat_completion, [{"role": "user", "content": "code"}], 6, "qwen-plus", 1) for _ in range(2)]
            assert all(job.result() == ["ok"] * 6 for job in jobs)
    assert peak <= 2


@pytest.mark.parametrize("source", [
    "import numpy as same as np\ndef get_task_priority_v2(a,b,c,d,e,f,g,h):\n return a",
    "def get_task_priority_v2(a,b,c,d,e,f,g,h):\n return d",
])
def test_bad_energy_source_is_candidate_failure_instead_of_run_failure(tmp_path, source):
    algo = object.__new__(SeEvo)
    algo.problem = "cews_task_constructive"
    algo.llm_objective = "energy_only"
    algo.iteration = 1
    algo.parameter_optimizer_config = OptimizerConfig()
    algo._runtime_artifact_path = lambda category, name: str(tmp_path / name)
    result = algo.response_to_individual("```python\n" + source + "\n```", 0)
    assert result["candidate_validation_error"]
    empty = algo.response_to_individual(None, 1)
    assert empty["code"] is None


def test_syntax_failure_can_be_repaired_and_repair_timeout_only_discards_candidate(tmp_path, monkeypatch):
    import seevo
    monkeypatch.chdir(tmp_path)
    algo = object.__new__(SeEvo)
    algo.problem = "cews_task_constructive"
    algo.llm_objective = "energy_only"
    algo.iteration = 1
    algo.parameter_optimizer_config = OptimizerConfig()
    algo._runtime_artifact_path = lambda category, name: str(tmp_path / name)
    algo.cfg = SimpleNamespace(candidate_generation=dict(validation_retries=1, candidate_repair_workers=1), model="qwen-plus")
    algo.candidate_repair_prompt = "repair {validation_error}"
    invalid = "```python\nimport numpy as same as np\ndef get_task_priority_v2(a,b,c,d,e,f,g,h):\n return a\n```"
    valid = (LLM_ROOT / "prompts/cews_task_constructive_energy_only/parameterized_seed_func.txt").read_text(encoding="utf-8").replace("get_task_priority_v1", "get_task_priority_v2")
    messages = [[{"role": "user", "content": "generate"}]]
    with patch("seevo.multi_chat_completion", return_value=[valid]):
        repaired = algo._responses_to_validated_individuals([invalid], messages)[0]
    assert "candidate_validation_error" not in repaired
    assert repaired["candidate_generation_attempts"] == 2
    with patch("seevo.multi_chat_completion", side_effect=seevo.LLMRequestError("timeout")):
        failed = algo._responses_to_validated_individuals([invalid], messages)[0]
    assert "repair API failed" in failed["candidate_validation_error"]


@pytest.mark.parametrize("failed", [
    {"fuzzy_total_energy_score": 1e300, "evaluation_error": "timeout"},
    {"fuzzy_total_energy_score": 1e300},
    {"fuzzy_total_energy_score": float("nan")},
])
def test_failed_energy_cannot_become_a_surrogate_label(failed):
    with pytest.raises(ValueError):
        _label(failed, "energy_only")


@pytest.mark.parametrize("extra", [
    '    np.save("output.npy", min_incremental_energy)\n',
    '    write = np.save\n',
    '    hidden = min_incremental_energy.__class__\n',
])
def test_numpy_side_effects_and_aliases_are_rejected(extra):
    source = utils.extract_code_from_generator((
        LLM_ROOT / "prompts/cews_task_constructive_energy_only/parameterized_seed_func.txt"
    ).read_text(encoding="utf-8"))
    source = source.replace("get_task_priority_v1", "get_task_priority_v2")
    source = source.replace("    energy =", extra + "    energy =", 1)
    with pytest.raises(RuleValidationError):
        parse_rule_candidate(source)


def test_legacy_and_helper_rules_cannot_bypass_file_operation_checks():
    source = '''import numpy as np
def helper(value):
    np.save("output.npy", value)
    return value
def get_task_priority_v2(min_exec_time,min_comm_time,min_incremental_energy,slack,upward_rank,remaining_work,ready_wait_time,uncertainty):
    return helper(min_incremental_energy)
'''
    with pytest.raises(RuleValidationError, match="save"):
        parse_rule_candidate(source)


def test_silent_final_evaluator_is_launched_without_waiting_for_stdout(tmp_path):
    algo = object.__new__(SeEvo)
    algo.generated_dir = str(tmp_path)
    algo.iteration = 1
    algo._evaluation_command = lambda *args: ["unused"]
    item = {"code": "pass", "stdout_filepath": str(tmp_path / "stdout.txt")}
    process = SimpleNamespace()
    with patch("seevo.subprocess.Popen", return_value=process), \
         patch("seevo.block_until_running", side_effect=AssertionError("would wait forever")):
        assert algo._run_code(item, 0, [1]) is process
    assert process._seevo_started > 0
