from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from algorithms.comparisons import evaluate_fuzzy_baseline as evaluator


def _cache_paths(tmp_path: Path) -> dict[str, str]:
    return {
        scenario: str(tmp_path / f"cache_{scenario}.json")
        for scenario in ("SM", "MM", "LM")
    }


def test_single_evaluation_requires_complete_cache_mapping(tmp_path: Path):
    checkpoint = tmp_path / "best_checkpoint.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="MM, LM"):
        evaluator.evaluate_checkpoint(
            method="marl",
            checkpoint=checkpoint,
            source_scenario="SM",
            protocol_name="single",
            ddl="M",
            device="cpu",
            deadline_cache_paths={"SM": "cache_SM.json"},
        )


@pytest.mark.parametrize("method", ("irws", "marl", "pd3qn"))
def test_frozen_single_evaluation_writes_json_and_csv(
    tmp_path: Path,
    monkeypatch,
    method: str,
):
    checkpoint = tmp_path / "best_checkpoint.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"optimizer_seed": 17}),
        encoding="utf-8",
    )
    cache_paths = _cache_paths(tmp_path)
    calls = {
        "builder": [],
        "loaded": [],
        "evaluated": [],
    }

    class FakePolicy:
        def load(self, path: str) -> None:
            calls["loaded"].append(path)

    def fake_builder(method_id, env, device, method_config):
        calls["builder"].append(
            (method_id, env, device, dict(method_config))
        )
        return FakePolicy()

    def fake_evaluate(protocol, policy, **kwargs):
        del policy
        scenario = protocol.scenario
        calls["evaluated"].append(
            (
                scenario,
                Path(protocol.deadline_cache_path).name,
                kwargs["split"],
            )
        )
        records = tuple(
            {
                "seed": seed,
                "deadline_violation_rate": 0.0,
                "fuzzy_energy_score": float(seed),
                "nested_seed_metric": {"scenario": scenario},
            }
            for seed in protocol.test_seeds
        )
        return SimpleNamespace(
            aggregate={
                "deadline_violation_rate": 0.0,
                "max_fuzzy_lateness": 0.0,
                "mean_fuzzy_lateness": 0.0,
                "fuzzy_energy_mean": 1.0,
                "fuzzy_energy_std": 0.1,
                "fuzzy_energy_score": 1.2,
                "feasible_seed_rate": 1.0,
                "mean_seed_scheduling_time_seconds": 0.5,
                "nested_aggregate_metric": {"scenario": scenario},
            },
            records=records,
        )

    monkeypatch.setattr(evaluator, "make_environment", lambda *a, **k: "env")
    monkeypatch.setattr(evaluator, "build_policy", fake_builder)
    monkeypatch.setattr(evaluator, "evaluate_policy", fake_evaluate)

    payload = evaluator.evaluate_checkpoint(
        method=method,
        checkpoint=checkpoint,
        source_scenario="SM",
        protocol_name="single",
        ddl="M",
        device="cpu",
        deadline_cache_paths=cache_paths,
    )

    assert calls["builder"][0][0] == method
    assert calls["loaded"] == [str(checkpoint.resolve())]
    assert calls["evaluated"] == [
        (scenario, f"cache_{scenario}.json", "final_test")
        for scenario in ("SM", "MM", "LM")
    ]
    assert payload["training_during_test"] is False
    assert payload["checkpoint_reselection"] is False
    assert payload["test_seeds"] == list(range(201, 231))
    assert payload["optimizer_seed"] == 17
    assert set(payload["deadline_cache_paths"]) == {"SM", "MM", "LM"}

    json_path = tmp_path / "final_test_metrics_reval.json"
    summary_path = tmp_path / "final_test_metrics_reval_summary.csv"
    seed_path = tmp_path / "final_test_metrics_reval_seed_records.csv"
    assert json_path.is_file()
    assert summary_path.is_file()
    assert seed_path.is_file()

    restored = json.loads(json_path.read_text(encoding="utf-8"))
    assert restored["checkpoint"] == str(checkpoint.resolve())
    assert restored["test_seeds"] == list(range(201, 231))
    assert all(
        len(result["seed_records"]) == 30
        for result in restored["scenario_results"].values()
    )

    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert [row["test_scenario"] for row in summary_rows] == [
        "SM", "MM", "LM"
    ]
    assert {
        "deadline_violation_rate",
        "max_fuzzy_lateness",
        "mean_fuzzy_lateness",
        "fuzzy_energy_mean",
        "fuzzy_energy_std",
        "fuzzy_energy_score",
    }.issubset(summary_rows[0])
    assert json.loads(summary_rows[0]["nested_aggregate_metric"])[
        "scenario"
    ] == "SM"

    with seed_path.open(encoding="utf-8-sig", newline="") as handle:
        seed_rows = list(csv.DictReader(handle))
    assert len(seed_rows) == 90
    assert json.loads(seed_rows[0]["nested_seed_metric"])["scenario"] == "SM"


def test_evaluator_does_not_reference_training_entrypoint():
    source = inspect.getsource(evaluator)
    assert "train_baseline" not in source
    assert "optimizer.step" not in source
