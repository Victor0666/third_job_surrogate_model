import csv
import json

import pytest

from tools.export_heuristic_summary import (
    COLUMNS, DDL_SETTINGS, PAPER_METRICS, SCENARIOS, export_summaries,
)


def populate(root, version="combined_v1"):
    for scenario in SCENARIOS:
        for ddl in DDL_SETTINGS:
            path = root / "fcfs_fixed" / f"{scenario}_{ddl}" / "final_test_metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "method_id": "fcfs_fixed", "display_name": "FCFS-Fixed",
                "protocol": {"scenario": scenario, "ddl_setting": ddl,
                             "source_scenario": scenario, "protocol_mode": "legacy"},
                "seed_records": [{"seed": seed, "communication_model_version": version}
                                 for seed in range(201, 231)],
                "aggregate": {**dict.fromkeys(PAPER_METRICS, 12.5),
                              "deadline_violation_rate": 0.25,
                              "max_fuzzy_lateness": 100,
                              "mean_seed_scheduling_time_seconds": 3.5},
            }), encoding="utf-8")


def test_exact_schema_and_27_cases(tmp_path):
    populate(tmp_path / "input")
    out = tmp_path / "output"
    export_summaries(tmp_path / "input", out, ["fcfs_fixed"])
    with (out / "all_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(COLUMNS)
        rows = list(reader)
    assert len(COLUMNS) == 78
    assert len(rows) == 27
    assert rows[0]["test_scenario"] == "SS"
    assert rows[0]["ddl"] == "T"
    assert rows[0]["checkpoint"] == rows[0]["optimizer_seed"] == ""
    assert rows[0]["evaluation_seed_count"] == "30"
    assert rows[0]["deadline_violation_rate"] == "0.25"
    assert rows[0]["max_fuzzy_lateness"] == "100"
    assert (out / "all_summary.csv").read_bytes() == (
        out / "fcfs_fixed/all_summary.csv"
    ).read_bytes()


@pytest.mark.parametrize("failure", ["missing_case", "duplicate_seed", "mixed_version"])
def test_invalid_inputs_do_not_create_outputs(tmp_path, failure):
    root = tmp_path / "input"
    populate(root)
    path = root / "fcfs_fixed/SS_T/final_test_metrics.json"
    if failure == "missing_case":
        path.unlink()
    else:
        data = json.loads(path.read_text())
        if failure == "duplicate_seed":
            data["seed_records"][0]["seed"] = 202
        else:
            for record in data["seed_records"]:
                record["communication_model_version"] = "staged_v1"
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        export_summaries(root, tmp_path / "output", ["fcfs_fixed"])
    assert not (tmp_path / "output").exists()
