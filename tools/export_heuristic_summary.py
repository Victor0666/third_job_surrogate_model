"""Export fixed-heuristic JSON summaries using the reference workbook's 78 columns.

Standard library only; run on the evaluation server without Excel dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METHODS = ("fcfs_fixed", "spt_fixed", "edd_fixed", "mwkr_fixed")
SCENARIOS = ("SS", "MS", "LS", "SM", "MM", "LM", "SL", "ML", "LL")
DDL_SETTINGS = ("T", "M", "L")
COLUMNS = (
    'algorithm_folder',
    'source_file',
    'method',
    'source_scenario',
    'test_scenario',
    'protocol',
    'ddl',
    'checkpoint',
    'optimizer_seed',
    'evaluation_seed_count',
    'deadline_violation_rate',
    'max_fuzzy_lateness',
    'mean_fuzzy_lateness',
    'fuzzy_energy_mean',
    'fuzzy_energy_std',
    'fuzzy_energy_score',
    'feasible_seed_rate',
    'mean_seed_scheduling_time_seconds',
    'metrics_schema_version',
    'record_kind',
    'completed_workflow_count',
    'deadline_violation_count',
    'fuzzy_ddl_violation_rate',
    'feasible_workflow_ratio',
    'feasible_episode_count',
    'feasible_episode_ratio',
    'all_seed_feasible',
    'all_seed_evaluation_completed',
    'completed_evaluation_seed_rate',
    'minimum_fuzzy_safety_margin',
    'worst_seed_violation_rate',
    'worst_seed_fuzzy_lateness',
    'worst_seed_minimum_fuzzy_safety_margin',
    'shield_record_count',
    'shield_intervention_count',
    'shield_intervention_rate',
    'no_safe_action_count',
    'no_safe_action_rate',
    'fallback_count',
    'fallback_rate',
    'proposed_executed_action_mismatch_count',
    'proposed_executed_action_mismatch_rate',
    'q_c_prediction_error',
    'q_c_prediction_error_sample_count',
    'lambda_current',
    'fuzzy_energy_score_across_seed_std',
    'modal_energy',
    'scheduling_time_seconds',
    'worst_seed_scheduling_time_seconds',
    'worst_seed_fuzzy_energy_score',
    'convergence_speed',
    'evaluations_to_convergence',
    'convergence_episode',
    'convergence_global_step',
    'heuristic_selection_count',
    'selected_llm_heuristic_count',
    'selected_llm_heuristic_frequency',
    'llm_associated_shield_record_count',
    'llm_associated_shield_intervention_count',
    'llm_associated_shield_rate',
    'energy_improvement_from_llm',
    'convergence_acceleration',
    'strict_ddl_feasibility_with_llm',
    'strict_ddl_feasibility_without_llm',
    'strict_ddl_feasibility_improvement',
    'safety_cost',
    'energy_decision_count',
    'mean_host_energy_regret',
    'mean_host_energy_regret_norm',
    'mean_vm_energy_regret',
    'mean_vm_energy_regret_norm',
    'mean_selected_vm_energy_rank',
    'best_energy_vm_selection_rate',
    'global_best_energy_vm_selection_rate',
    'worst_seed_violation',
    'worst_seed_lateness',
    'validation_seed_count',
    'zero_violation_pass',
)
PAPER_METRICS = (
    "deadline_violation_rate", "max_fuzzy_lateness", "mean_fuzzy_lateness",
    "fuzzy_energy_mean", "fuzzy_energy_std", "fuzzy_energy_score",
)
NOT_APPLICABLE = (
    "checkpoint", "optimizer_seed", "convergence_speed", "evaluations_to_convergence",
    "convergence_episode", "convergence_global_step",
)


def summary_row(path, method, scenario, ddl):
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    records = data["seed_records"]
    if data["method_id"] != method or protocol["scenario"] != scenario:
        raise ValueError(f"Method/scenario mismatch: {path}")
    if protocol["ddl_setting"][0].upper() != ddl:
        raise ValueError(f"Deadline setting mismatch: {path}")
    if len(records) != 30 or {r["seed"] for r in records} != set(range(201, 231)):
        raise ValueError(f"Expected exactly 30 seeds (201-230): {path}")
    versions = {r.get("communication_model_version", "unknown") for r in records}
    if len(versions) != 1 or "unknown" in versions:
        raise ValueError(f"Missing/mixed communication model versions: {path}")
    aggregate = data["aggregate"]
    missing = set(PAPER_METRICS).difference(aggregate)
    if missing:
        raise ValueError(f"Missing metrics {sorted(missing)}: {path}")
    # Use the evaluator's aggregates; rates and worst-case values are not re-averaged.
    row = {column: aggregate.get(column) for column in COLUMNS}
    row.update(
        algorithm_folder=data.get("display_name", method),
        source_file=str(path), method=method,
        source_scenario=protocol.get("source_scenario", scenario),
        test_scenario=scenario, protocol=protocol.get("protocol_mode", "legacy"),
        ddl=ddl, evaluation_seed_count=len(records),
    )
    for column in NOT_APPLICABLE:
        row[column] = None
    return row, versions.pop()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def export_summaries(input_root, output_dir, methods=METHODS):
    batches = {}
    versions = set()
    # Validate every requested case before writing any output.
    for method in methods:
        rows = []
        for scenario in SCENARIOS:
            for ddl in DDL_SETTINGS:
                path = Path(input_root) / method / f"{scenario}_{ddl}" / "final_test_metrics.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Required case is missing: {path}")
                row, version = summary_row(path, method, scenario, ddl)
                rows.append(row)
                versions.add(version)
        batches[method] = rows
    if len(versions) != 1:
        raise ValueError(f"Cannot merge different communication models: {sorted(versions)}")
    all_rows = []
    output_dir = Path(output_dir)
    for method, rows in batches.items():
        write_csv(output_dir / method / "all_summary.csv", rows)
        all_rows.extend(rows)
    destination = output_dir / "all_summary.csv"
    write_csv(destination, all_rows)
    print(f"Communication model: {versions.pop()}")
    print(f"Columns: {len(COLUMNS)}; merged rows: {len(all_rows)}")
    print(f"Combined summary: {destination.resolve()}")
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("out/comparisons"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("out/comparisons/heuristics_formatted"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must not contain duplicates")
    export_summaries(args.input_root, args.output_dir, args.methods)


if __name__ == "__main__":
    main()
