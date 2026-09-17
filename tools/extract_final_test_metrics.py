"""Extract a compact comparison table from final-test summary CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FIELDS = (
    "来源文件", "算法", "源场景", "测试场景", "DDL",
    "DDL违反数", "工作流总数", "DDL违反率", "DDL违反率展示",
    "零违反种子数", "测试种子数", "零违反种子率", "零违反种子率展示",
    "平均模糊迟延(s)", "最坏模糊迟延(s)", "模糊能耗均值(J)",
    "模糊能耗得分(J)", "跨种子能耗标准差(J)", "平均调度时间(s)",
    "严格零违反通过",
)

REQUIRED = (
    "method", "source_scenario", "test_scenario", "ddl",
    "evaluation_seed_count", "completed_workflow_count",
    "deadline_violation_count", "fuzzy_ddl_violation_rate",
    "feasible_episode_count", "feasible_seed_rate",
    "mean_fuzzy_lateness", "fuzzy_energy_mean", "fuzzy_energy_score",
    "fuzzy_energy_score_across_seed_std",
    "mean_seed_scheduling_time_seconds",
)


def _number(row: Mapping[str, str], name: str) -> float:
    raw = row.get(name, "").strip()
    if not raw:
        raise ValueError(f"field {name!r} is empty")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"field {name!r} must be finite")
    return value


def _count(row: Mapping[str, str], name: str) -> int:
    value = _number(row, name)
    if value < 0 or not value.is_integer():
        raise ValueError(f"field {name!r} must be a non-negative integer")
    return int(value)


def _check_rate(name: str, reported: float, calculated: float) -> None:
    if not math.isclose(reported, calculated, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"{name} is inconsistent: reported={reported}, "
            f"calculated={calculated}"
        )


def _boolean(raw: str, default: bool) -> bool:
    value = raw.strip().lower()
    if not value:
        return default
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {raw!r}")


def _worst_lateness(row: Mapping[str, str]) -> float:
    for name in ("worst_seed_fuzzy_lateness", "max_fuzzy_lateness"):
        if row.get(name, "").strip():
            return _number(row, name)
    raise ValueError(
        "one of worst_seed_fuzzy_lateness or max_fuzzy_lateness is required"
    )


def compact_row(row: Mapping[str, str], source: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED if name not in row]
    if missing:
        raise ValueError(f"{source}: missing columns: {', '.join(missing)}")

    workflows = _count(row, "completed_workflow_count")
    violations = _count(row, "deadline_violation_count")
    seeds = _count(row, "evaluation_seed_count")
    feasible_seeds = _count(row, "feasible_episode_count")
    if workflows == 0 or seeds == 0:
        raise ValueError(f"{source}: workflow and seed totals must be positive")
    if violations > workflows or feasible_seeds > seeds:
        raise ValueError(f"{source}: count exceeds its total")

    violation_rate = violations / workflows
    feasible_rate = feasible_seeds / seeds
    _check_rate(
        "fuzzy_ddl_violation_rate",
        _number(row, "fuzzy_ddl_violation_rate"),
        violation_rate,
    )
    _check_rate(
        "feasible_seed_rate",
        _number(row, "feasible_seed_rate"),
        feasible_rate,
    )
    strict_default = violations == 0 and feasible_seeds == seeds

    return {
        "来源文件": source.name,
        "算法": row["method"],
        "源场景": row["source_scenario"],
        "测试场景": row["test_scenario"],
        "DDL": row["ddl"],
        "DDL违反数": violations,
        "工作流总数": workflows,
        "DDL违反率": violation_rate,
        "DDL违反率展示": f"{violations}/{workflows} = {violation_rate:.3%}",
        "零违反种子数": feasible_seeds,
        "测试种子数": seeds,
        "零违反种子率": feasible_rate,
        "零违反种子率展示": f"{feasible_seeds}/{seeds} = {feasible_rate:.2%}",
        "平均模糊迟延(s)": _number(row, "mean_fuzzy_lateness"),
        "最坏模糊迟延(s)": _worst_lateness(row),
        "模糊能耗均值(J)": _number(row, "fuzzy_energy_mean"),
        "模糊能耗得分(J)": _number(row, "fuzzy_energy_score"),
        "跨种子能耗标准差(J)": _number(
            row, "fuzzy_energy_score_across_seed_std"
        ),
        "平均调度时间(s)": _number(
            row, "mean_seed_scheduling_time_seconds"
        ),
        "严格零违反通过": _boolean(
            row.get("zero_violation_pass", ""), strict_default
        ),
    }


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [compact_row(row, path) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def extract_metrics(
    inputs: Iterable[Path], output: Path, *, overwrite: bool = False
) -> int:
    sources = [path.resolve() for path in inputs]
    target = output.resolve()
    if not sources:
        raise ValueError("at least one input CSV is required")
    if target in sources:
        raise ValueError("output must not overwrite an input CSV")
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"output exists: {target}; pass --overwrite to replace it"
        )

    rows: list[dict[str, object]] = []
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.extend(read_rows(source))

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Extract core metrics from final-test summary CSV files."
    )
    result.add_argument("inputs", nargs="+", help="input summary CSV path(s)")
    result.add_argument("--output", required=True, help="output compact CSV")
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output)
    count = extract_metrics(
        [Path(item) for item in args.inputs],
        output,
        overwrite=args.overwrite,
    )
    print(f"Wrote {count} rows to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
