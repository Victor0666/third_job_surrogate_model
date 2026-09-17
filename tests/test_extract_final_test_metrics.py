import csv
from pathlib import Path

import pytest

from tools.extract_final_test_metrics import extract_metrics


def _write_summary(path: Path, **overrides: object) -> None:
    row = {
        "method": "marl",
        "source_scenario": "SL",
        "test_scenario": "SL",
        "ddl": "L",
        "evaluation_seed_count": 30,
        "completed_workflow_count": 1500,
        "deadline_violation_count": 59,
        "fuzzy_ddl_violation_rate": 0.039333,
        "feasible_episode_count": 4,
        "feasible_seed_rate": 0.133333,
        "mean_fuzzy_lateness": 4.827039,
        "worst_seed_fuzzy_lateness": 583.671038,
        "fuzzy_energy_mean": 369347.108207,
        "fuzzy_energy_score": 380486.957994,
        "fuzzy_energy_score_across_seed_std": 52595.566819,
        "mean_seed_scheduling_time_seconds": 111.939332,
        "zero_violation_pass": "False",
    }
    row.update(overrides)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def test_extract_metrics_writes_auditable_table(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    output = tmp_path / "core_metrics.csv"
    _write_summary(source)

    assert extract_metrics([source], output) == 1

    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["算法"] == "marl"
    assert row["DDL违反率展示"] == "59/1500 = 3.933%"
    assert row["零违反种子率展示"] == "4/30 = 13.33%"
    assert float(row["跨种子能耗标准差(J)"]) == pytest.approx(52595.566819)
    assert row["严格零违反通过"] == "False"


def test_extract_metrics_rejects_inconsistent_rate(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    _write_summary(source, fuzzy_ddl_violation_rate=0.5)

    with pytest.raises(ValueError, match="fuzzy_ddl_violation_rate is inconsistent"):
        extract_metrics([source], tmp_path / "core_metrics.csv")


def test_extract_metrics_requires_overwrite_flag(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    output = tmp_path / "core_metrics.csv"
    _write_summary(source)
    output.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        extract_metrics([source], output)
    assert output.read_text(encoding="utf-8") == "keep me"
