"""Constraint-aware aggregation of decision-level diagnostic reports."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Iterable

from .schemas import (
    AggregatedFeedback,
    CounterfactualComparison,
    CounterfactualConfig,
    DecisionTrace,
    DiagnosticReport,
    canonical_hash,
)


class FeedbackAggregator:
    """Aggregate without ever summing DDL regret and energy in one score."""

    def __init__(self, config: CounterfactualConfig):
        self.config = config

    def aggregate(
        self,
        reports: Iterable[DiagnosticReport],
        traces: Iterable[DecisionTrace],
        comparisons: Iterable[CounterfactualComparison],
        *,
        structure_hash: str,
        frozen_rule_hash: str,
    ) -> AggregatedFeedback:
        report_rows = list(reports)
        trace_rows = list(traces)
        comparison_rows = list(comparisons)
        substantive = [row for row in report_rows if row.diagnosis_code != "insufficient_evidence"]
        by_agent = defaultdict(list)
        by_code = defaultdict(list)
        for row in substantive:
            by_agent[row.agent_name].append(row)
            by_code[row.diagnosis_code].append(row)

        patterns = []
        for code, rows in sorted(by_code.items()):
            seeds = sorted({row.seed for row in rows})
            scenarios = sorted({row.scenario_id for row in rows})
            repeated = len(rows) >= self.config.min_repeated_evidence
            high = repeated and (
                len(seeds) >= self.config.high_confidence_seed_count
                or len(scenarios) >= self.config.high_confidence_scenario_count
            )
            patterns.append(
                {
                    "diagnosis_code": code,
                    "agent_name": rows[0].agent_name,
                    "occurrences": len(rows),
                    "decision_ids": [row.decision_id for row in rows[:8]],
                    "seeds": seeds,
                    "scenarios": scenarios,
                    "confidence": "high" if high else ("medium" if repeated else "low"),
                    "suggested_structural_actions": sorted(
                        {action for row in rows for action in row.suggested_structural_actions}
                    ),
                }
            )

        load_condition_rows = list(by_agent.get("energy_diagnostic_agent", []))
        if len(load_condition_rows) >= self.config.min_repeated_evidence:
            observed_loads = [
                float(row.state_conditions.get("host_load_maximum", 0.0))
                for row in load_condition_rows
            ]
            load_midpoint = (min(observed_loads) + max(observed_loads)) / 2.0
            low_load_codes = {
                row.diagnosis_code
                for row in load_condition_rows
                if float(row.state_conditions.get("host_load_maximum", 0.0)) <= load_midpoint
            }
            high_load_codes = {
                row.diagnosis_code
                for row in load_condition_rows
                if float(row.state_conditions.get("host_load_maximum", 0.0)) > load_midpoint
            }
            preference_reversal = bool(
                low_load_codes
                and high_load_codes
                and low_load_codes - high_load_codes
                and high_load_codes - low_load_codes
            )
            if max(observed_loads) - min(observed_loads) >= 0.25 and preference_reversal:
                seeds = sorted({row.seed for row in load_condition_rows})
                scenarios = sorted({row.scenario_id for row in load_condition_rows})
                high = (
                    len(seeds) >= self.config.high_confidence_seed_count
                    or len(scenarios) >= self.config.high_confidence_scenario_count
                )
                patterns.append(
                    {
                        "diagnosis_code": "load_conditioned_preference_reversal",
                        "agent_name": "energy_diagnostic_agent",
                        "occurrences": len(load_condition_rows),
                        "decision_ids": [row.decision_id for row in load_condition_rows[:8]],
                        "seeds": seeds,
                        "scenarios": scenarios,
                        "confidence": "high" if high else "medium",
                        "evidence": {
                            "minimum_host_load": min(observed_loads),
                            "maximum_host_load": max(observed_loads),
                            "low_load_diagnosis_codes": sorted(low_load_codes),
                            "high_load_diagnosis_codes": sorted(high_load_codes),
                        },
                        "suggested_structural_actions": ["add_host_load_conditional_gate"],
                    }
                )

        action_rows = defaultdict(list)
        for row in substantive:
            for action in row.suggested_structural_actions:
                action_rows[action].append(row)
        if any(
            row["diagnosis_code"] == "load_conditioned_preference_reversal"
            for row in patterns
        ):
            action_rows["add_host_load_conditional_gate"].extend(
                load_condition_rows
            )
        high_actions = []
        medium_actions = []
        for action, rows in sorted(action_rows.items()):
            seeds = {row.seed for row in rows}
            scenarios = {row.scenario_id for row in rows}
            repeated = len(rows) >= self.config.min_repeated_evidence
            high = repeated and (
                len(seeds) >= self.config.high_confidence_seed_count
                or len(scenarios) >= self.config.high_confidence_scenario_count
            )
            item = {
                "action": action,
                "occurrences": len(rows),
                "agents": sorted({row.agent_name for row in rows}),
                "seed_count": len(seeds),
                "scenario_count": len(scenarios),
                "evidence_decision_ids": [row.decision_id for row in rows[:6]],
            }
            if high:
                high_actions.append(item)
            elif repeated:
                medium_actions.append(item)

        preferences = defaultdict(dict)
        for row in substantive:
            preferences[row.decision_id][row.agent_name] = row.preferred_task
        conflicts = []
        consensus = []
        for decision_id, values in sorted(preferences.items()):
            chosen = {value for value in values.values() if value is not None}
            if len(chosen) > 1:
                conflicts.append({"decision_id": decision_id, "agent_preferences": values})
            elif len(values) >= 2 and len(chosen) == 1:
                consensus.append(
                    {
                        "decision_id": decision_id,
                        "preferred_task": next(iter(chosen)),
                        "agents": sorted(values),
                    }
                )

        trace_by_id = {row.decision_id: row for row in trace_rows}
        comparison_by_decision = defaultdict(list)
        for row in comparison_rows:
            comparison_by_decision[row.decision_id].append(row)
        severity_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        representatives = []
        used = set()
        for report in sorted(substantive, key=lambda row: (severity_order.get(row.severity, 4), row.decision_id)):
            if report.decision_id in used:
                continue
            trace = trace_by_id.get(report.decision_id)
            if trace is None:
                continue
            compared = comparison_by_decision.get(report.decision_id, [])
            representatives.append(
                {
                    "decision_id": report.decision_id,
                    "scenario_id": trace.scenario_id,
                    "seed": trace.seed,
                    "selected_task_id": trace.selected_task_id,
                    "critical_reasons": trace.critical_reasons,
                    "diagnosis_code": report.diagnosis_code,
                    "quantitative_deltas": report.quantitative_deltas,
                    "alternatives": [row.alternative_task_id for row in compared],
                    "estimator_type": "local_one_step",
                }
            )
            used.add(report.decision_id)
            if len(representatives) >= self.config.max_representative_cases:
                break

        insufficient = [
            {
                "agent_name": row.agent_name,
                "decision_id": row.decision_id,
                "limitations": row.limitations,
            }
            for row in report_rows
            if row.diagnosis_code == "insufficient_evidence"
        ][:20]
        run_ids = {row.run_id for row in trace_rows}
        core = {
            "summary_version": "counterfactual_feedback_v1",
            "structure_hash": structure_hash,
            "frozen_rule_hash": frozen_rule_hash,
            "analyzed_runs": len(run_ids),
            "analyzed_decisions": sum(row.is_critical for row in trace_rows),
            "compared_alternatives": len(comparison_rows),
            "repeated_failure_patterns": [row for row in patterns if row["occurrences"] >= self.config.min_repeated_evidence],
            "ddl_patterns": [row for row in patterns if row["agent_name"] == "ddl_diagnostic_agent"],
            "energy_patterns": [row for row in patterns if row["agent_name"] == "energy_diagnostic_agent"],
            "uncertainty_patterns": [row for row in patterns if row["agent_name"] == "uncertainty_diagnostic_agent"],
            "cross_agent_consensus": consensus,
            "cross_agent_conflicts": conflicts,
            "high_confidence_structural_actions": high_actions,
            "medium_confidence_actions": medium_actions,
            "insufficient_evidence_items": insufficient,
            "representative_cases": representatives,
            "limitations": [
                "local_one_step comparisons are mechanism evidence, not strict causal proof",
                "bounded replay is disabled because the environment has no safe snapshot API",
                "energy is considered only after DDL evidence; no weighted DDL-energy sum is used",
            ],
            "config_hash": self.config.config_hash,
        }
        diagnostics_hash = canonical_hash(core)
        return AggregatedFeedback(diagnostics_hash=diagnostics_hash, **core)


def compact_feedback_for_prompt(feedback: dict, max_chars: int) -> dict:
    """Apply evidence-priority truncation before one existing LLM call."""
    keys = (
        "summary_version",
        "structure_hash",
        "analyzed_runs",
        "analyzed_decisions",
        "compared_alternatives",
        "repeated_failure_patterns",
        "ddl_patterns",
        "uncertainty_patterns",
        "energy_patterns",
        "cross_agent_consensus",
        "cross_agent_conflicts",
        "high_confidence_structural_actions",
        "representative_cases",
        "medium_confidence_actions",
        "limitations",
    )
    result = {key: feedback.get(key) for key in keys if key in feedback}
    priority_drop = (
        "energy_patterns",
        "cross_agent_consensus",
        "limitations",
        "uncertainty_patterns",
        "ddl_patterns",
        "repeated_failure_patterns",
        "cross_agent_conflicts",
        "medium_confidence_actions",
    )
    while len(json.dumps(result, ensure_ascii=True, sort_keys=True)) > max_chars:
        changed = False
        for key in priority_drop:
            values = result.get(key)
            if isinstance(values, list) and values:
                values.pop()
                changed = True
                break
        representatives = result.get("representative_cases")
        if not changed and isinstance(representatives, list) and representatives:
            representatives.pop()
            changed = True
        if not changed:
            result = {
                "summary_version": feedback.get("summary_version"),
                "structure_hash": feedback.get("structure_hash"),
                "high_confidence_structural_actions": feedback.get("high_confidence_structural_actions", [])[:1],
                "limitations": ["feedback truncated to configured character budget"],
            }
            break
    result["prompt_payload_chars"] = len(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return result
