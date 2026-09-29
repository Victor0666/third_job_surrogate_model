from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# /root/lyz/third_job_surrogate_model/out/main_single/SL/test/diag_step398000_n3
# /root/lyz/third_job_surrogate_model/out/main_single/SS/test/pipe-ss-l-0d87a7d3a6
# /root/lyz/third_job_surrogate_model/out/main_single/SL/test/pipe-sl-t-0d87a7d3a6
INPUT_DIR = Path(
    "out/main_single/SL/test/pipe-sl-t-0d87a7d3a6"
)


def csv_value(value: Any) -> Any:
    """将嵌套对象转换为可存入单个 CSV 单元格的 JSON。"""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return value


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    preferred_fields: list[str],
) -> None:
    all_fields = {
        key
        for row in rows
        for key in row
    }
    fieldnames = [
        field
        for field in preferred_fields
        if field in all_fields
    ]
    fieldnames.extend(
        sorted(all_fields.difference(fieldnames))
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: csv_value(row.get(key))
                    for key in fieldnames
                }
            )


def main() -> None:
    scenario_files = sorted(
        INPUT_DIR.glob("scenario_*.json")
    )
    if not scenario_files:
        raise FileNotFoundError(
            f"No scenario JSON files found in {INPUT_DIR}"
        )

    summary_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    expected_seed_rows = 0

    for scenario_path in scenario_files:
        payload = json.loads(
            scenario_path.read_text(encoding="utf-8")
        )

        scenario = str(payload["scenario"])
        test_seeds = [int(v) for v in payload["test_seeds"]]
        result = payload["result"]

        if len(result) != 5:
            raise ValueError(
                f"{scenario_path} has an unexpected result structure"
            )

        eval_vm = float(result[0])
        eval_host = float(result[1])
        eval_manager = float(result[2])
        eval_energy = float(result[3])
        safety = dict(result[4])

        per_seed = list(safety["per_seed_metrics"])

        if len(per_seed) != len(test_seeds):
            raise ValueError(
                f"{scenario}: test seed count "
                f"{len(test_seeds)} != per-seed record count "
                f"{len(per_seed)}"
            )

        actual_seeds = {
            int(record["seed"])
            for record in per_seed
        }
        if actual_seeds != set(test_seeds):
            raise ValueError(
                f"{scenario}: per-seed records do not match "
                "the declared test seeds"
            )

        expected_seed_rows += len(test_seeds)

        for record in per_seed:
            seed_rows.append(
                {
                    "scenario": scenario,
                    **dict(record),
                }
            )

        summary = {
            "scenario": scenario,
            "test_seed_count": len(test_seeds),
            "first_test_seed": min(test_seeds),
            "last_test_seed": max(test_seeds),
            "eval_vm_reward": eval_vm,
            "eval_host_reward": eval_host,
            "eval_manager_reward": eval_manager,
            "eval_energy_mean": eval_energy,
        }

        # 两个逐种子数组单独写入明细 CSV，
        # 不放入汇总 CSV 的单个单元格。
        for key, value in safety.items():
            if key in {
                "per_seed_metrics",
                "per_seed_safety_metrics",
            }:
                continue
            summary[key] = value

        summary_rows.append(summary)

    if len(seed_rows) != expected_seed_rows:
        raise RuntimeError(
            f"Expected {expected_seed_rows} seed rows, "
            f"got {len(seed_rows)}"
        )

    unique_keys = {
        (row["scenario"], int(row["seed"]))
        for row in seed_rows
    }
    if len(unique_keys) != len(seed_rows):
        raise RuntimeError(
            "Duplicate scenario/seed records detected"
        )

    summary_path = INPUT_DIR / "final_test_summary.csv"
    seed_path = INPUT_DIR / "final_test_seed_records.csv"

    write_csv(
        summary_path,
        summary_rows,
        preferred_fields=[
            "scenario",
            "test_seed_count",
            "first_test_seed",
            "last_test_seed",
            "eval_vm_reward",
            "eval_host_reward",
            "eval_manager_reward",
            "eval_energy_mean",
            "deadline_violation_count",
            "deadline_violation_rate",
            "feasible_seed_rate",
            "all_seed_feasible",
            "fuzzy_energy_score",
            "max_fuzzy_lateness",
            "mean_fuzzy_lateness",
            "shield_intervention_rate",
            "fallback_rate",
        ],
    )

    write_csv(
        seed_path,
        seed_rows,
        preferred_fields=[
            "scenario",
            "seed",
            "evaluation_completed",
            "seed_feasible",
            "expected_workflow_count",
            "completed_workflow_count",
            "deadline_violation_count",
            "deadline_violation_rate",
            "feasible_workflow_ratio",
            "modal_energy",
            "fuzzy_energy_mean",
            "fuzzy_energy_std",
            "fuzzy_energy_score",
            "fuzzy_lateness_sum",
            "max_fuzzy_lateness",
            "mean_fuzzy_lateness",
            "minimum_fuzzy_safety_margin",
            "safety_cost",
            "shield_intervention_count",
            "shield_intervention_rate",
            "fallback_count",
            "fallback_rate",
            "selected_llm_heuristic_count",
            "selected_llm_heuristic_frequency",
            "scheduling_time_seconds",
        ],
    )

    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Per-seed CSV: {seed_path.resolve()}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Per-seed rows: {len(seed_rows)}")


if __name__ == "__main__":
    main()