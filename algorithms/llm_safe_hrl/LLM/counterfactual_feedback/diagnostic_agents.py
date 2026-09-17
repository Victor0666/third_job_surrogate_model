"""Three deterministic diagnostic agents with distinct evidence rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .schemas import CounterfactualComparison, DecisionTrace, DiagnosticReport


ALLOWED_STRUCTURAL_ACTIONS = {
    "add_conditional_ddl_protection_gate",
    "add_upward_rank_remaining_work_interaction",
    "add_successor_release_interaction",
    "add_small_energy_gap_high_ddl_risk_gate",
    "add_uncertainty_ddl_interaction",
    "add_pessimistic_risk_gate",
    "add_host_load_conditional_gate",
    "add_load_successor_release_interaction",
    "replace_linear_risk_with_smooth_nonlinear_form",
    "normalize_slack_by_workflow_deadline_budget",
    "remove_redundant_local_energy_term",
}


def _snapshots(trace: DecisionTrace) -> dict[int, dict]:
    return {item.task_id: item.to_dict() for item in trace.candidate_tasks}


def _insufficient(agent: str, trace: DecisionTrace, limitation: str) -> DiagnosticReport:
    return DiagnosticReport(
        agent_name=agent,
        decision_id=trace.decision_id,
        scenario_id=trace.scenario_id,
        seed=trace.seed,
        evidence={"comparison_count": 0},
        diagnosis_code="insufficient_evidence",
        severity="none",
        confidence="low",
        state_conditions={
            "minimum_slack": trace.minimum_slack,
            "maximum_uncertainty": trace.maximum_uncertainty,
            "queue_size": trace.queue_size,
        },
        preferred_task=None,
        rejected_task=None,
        quantitative_deltas={},
        suggested_structural_actions=[],
        limitations=[limitation],
    )


class DiagnosticAgent(ABC):
    name: str

    @abstractmethod
    def analyze(
        self,
        trace: DecisionTrace,
        comparisons: Iterable[CounterfactualComparison],
        context: dict | None = None,
    ) -> DiagnosticReport:
        raise NotImplementedError


class DDLDiagnosticAgent(DiagnosticAgent):
    """Diagnose slack, violation, critical-path, and successor-release errors."""

    name = "ddl_diagnostic_agent"

    def analyze(self, trace, comparisons, context=None) -> DiagnosticReport:
        rows = list(comparisons)
        if not rows:
            return _insufficient(self.name, trace, "no local alternatives were evaluated")
        snapshots = _snapshots(trace)

        def priority(row):
            alternative = snapshots.get(row.alternative_task_id, {})
            alt_slack = float(alternative.get("features", {}).get("slack", float("inf")))
            low_slack_regret = 1.0 if alt_slack <= 0.0 else 0.0
            return (
                float(row.predicted_violation_delta),
                float(row.safety_margin_delta),
                low_slack_regret,
                float(row.released_critical_successor_delta),
            )

        row = max(rows, key=priority)
        selected_snapshot = snapshots.get(row.selected_task_id, {})
        alternative_snapshot = snapshots.get(row.alternative_task_id, {})
        selected_slack = float(selected_snapshot.get("features", {}).get("slack", float("inf")))
        alternative_slack = float(alternative_snapshot.get("features", {}).get("slack", float("inf")))
        codes = []
        actions = []
        if alternative_slack <= 0.0 and selected_slack > alternative_slack:
            codes.append("low_slack_task_deferred")
            actions.extend(["add_conditional_ddl_protection_gate", "normalize_slack_by_workflow_deadline_budget"])
        if row.released_critical_successor_delta > 0 or row.remaining_critical_path_delta > 0.0:
            codes.append("critical_path_or_successor_blocked")
            actions.extend(["add_upward_rank_remaining_work_interaction", "add_successor_release_interaction"])
        if row.predicted_violation_delta > 1e-9 or row.safety_margin_delta > 1e-9:
            codes.append("higher_ddl_risk_choice")
            actions.append("add_conditional_ddl_protection_gate")
        if not codes:
            return _insufficient(self.name, trace, "no alternative has a material DDL advantage")
        alternative_advantage = (
            row.predicted_violation_delta > 1e-9
            or row.safety_margin_delta > 1e-9
            or row.released_critical_successor_delta > 0
            or row.remaining_critical_path_delta > 1e-9
        )
        severity = "high" if row.predicted_violation_delta > 1e-9 else "medium"
        confidence = "high" if row.evidence_quality == "high" and severity == "high" else "medium"
        actions = list(dict.fromkeys(action for action in actions if action in ALLOWED_STRUCTURAL_ACTIONS))
        return DiagnosticReport(
            agent_name=self.name,
            decision_id=trace.decision_id,
            scenario_id=trace.scenario_id,
            seed=trace.seed,
            evidence={
                "selected_slack": selected_slack,
                "alternative_slack": alternative_slack,
                "selected_upward_rank": selected_snapshot.get("features", {}).get("upward_rank"),
                "alternative_upward_rank": alternative_snapshot.get("features", {}).get("upward_rank"),
                "selected_safety_margin": row.selected_outcome.safety_margin,
                "alternative_safety_margin": row.alternative_outcome.safety_margin,
                "selected_pessimistic_finish": row.selected_outcome.pessimistic_finish,
                "alternative_pessimistic_finish": row.alternative_outcome.pessimistic_finish,
            },
            diagnosis_code="+".join(codes),
            severity=severity,
            confidence=confidence,
            state_conditions={
                "negative_slack_present": trace.minimum_slack <= 0.0,
                "critical_reasons": trace.critical_reasons,
                "queue_size": trace.queue_size,
            },
            preferred_task=row.alternative_task_id if alternative_advantage else None,
            rejected_task=row.selected_task_id if alternative_advantage else None,
            quantitative_deltas={
                "predicted_violation_regret": row.predicted_violation_delta,
                "safety_margin_regret": row.safety_margin_delta,
                "critical_successor_release_regret": float(row.released_critical_successor_delta),
                "remaining_critical_path_regret": row.remaining_critical_path_delta,
            },
            suggested_structural_actions=actions,
            limitations=(
                ["local one-step evidence is not a long-horizon causal proof"]
                + ([] if alternative_advantage else [
                    "the structural symptom does not prove that this alternative is better"
                ])
            ),
        )


class EnergyDiagnosticAgent(DiagnosticAgent):
    """Diagnose energy/load/communication choices while preserving DDL priority."""

    name = "energy_diagnostic_agent"

    def analyze(self, trace, comparisons, context=None) -> DiagnosticReport:
        rows = list(comparisons)
        if not rows:
            return _insufficient(self.name, trace, "no local alternatives were evaluated")
        ddl_safe = [
            row for row in rows
            if row.predicted_violation_delta >= -1e-9 and row.safety_margin_delta >= -1e-9
        ]
        energy_waste = max(ddl_safe, key=lambda row: row.marginal_fuzzy_energy_delta, default=None)
        ddl_tradeoff = max(rows, key=lambda row: row.predicted_violation_delta)
        if (
            ddl_tradeoff.predicted_violation_delta > 1e-9
            and ddl_tradeoff.marginal_fuzzy_energy_delta < 0.0
        ):
            row = ddl_tradeoff
            energy_saving = -row.marginal_fuzzy_energy_delta
            relative_saving = energy_saving / max(abs(row.alternative_outcome.marginal_fuzzy_energy), 1e-12)
            code = "small_energy_saving_large_ddl_regret"
            severity = "high"
            preferred = row.alternative_task_id
            rejected = row.selected_task_id
            actions = ["add_small_energy_gap_high_ddl_risk_gate"]
        elif energy_waste is not None and energy_waste.marginal_fuzzy_energy_delta > 1e-9:
            row = energy_waste
            relative_saving = row.marginal_fuzzy_energy_delta / max(
                abs(row.selected_outcome.marginal_fuzzy_energy), 1e-12
            )
            code = "avoidable_marginal_energy_or_load_cost"
            severity = "medium"
            preferred = row.alternative_task_id
            rejected = row.selected_task_id
            actions = ["add_host_load_conditional_gate"]
            if row.communication_delta > 0.0:
                actions.append("add_load_successor_release_interaction")
        else:
            return _insufficient(
                self.name,
                trace,
                "energy differences are immaterial or justified by a selected-task DDL advantage",
            )
        if (
            row.selected_outcome.server_active_time_extension
            > row.alternative_outcome.server_active_time_extension + 1e-9
        ):
            actions.append("add_load_successor_release_interaction")
        return DiagnosticReport(
            agent_name=self.name,
            decision_id=trace.decision_id,
            scenario_id=trace.scenario_id,
            seed=trace.seed,
            evidence={
                "selected_energy": row.selected_outcome.marginal_fuzzy_energy,
                "alternative_energy": row.alternative_outcome.marginal_fuzzy_energy,
                "relative_energy_difference": relative_saving,
                "selected_host_load_after": row.selected_outcome.host_load_after,
                "alternative_host_load_after": row.alternative_outcome.host_load_after,
                "selected_active_time_extension": row.selected_outcome.server_active_time_extension,
                "alternative_active_time_extension": row.alternative_outcome.server_active_time_extension,
                "ddl_priority_applied": True,
            },
            diagnosis_code=code,
            severity=severity,
            confidence="medium" if row.evidence_quality != "low" else "low",
            state_conditions={
                "host_load_maximum": trace.host_load_summary.get("maximum", 0.0),
                "queue_size": trace.queue_size,
            },
            preferred_task=preferred,
            rejected_task=rejected,
            quantitative_deltas={
                "marginal_fuzzy_energy_regret": row.marginal_fuzzy_energy_delta,
                "communication_regret": row.communication_delta,
                "predicted_violation_regret": row.predicted_violation_delta,
                "host_load_delta_difference": (
                    row.selected_outcome.host_load_delta - row.alternative_outcome.host_load_delta
                ),
            },
            suggested_structural_actions=list(dict.fromkeys(actions)),
            limitations=["marginal energy and active-time extension are local proxies"],
        )


class UncertaintyDiagnosticAgent(DiagnosticAgent):
    """Diagnose modal/pessimistic disagreement and uncertainty-risk coupling."""

    name = "uncertainty_diagnostic_agent"

    def analyze(self, trace, comparisons, context=None) -> DiagnosticReport:
        rows = list(comparisons)
        if not rows:
            return _insufficient(self.name, trace, "no local alternatives were evaluated")
        candidates = []
        for row in rows:
            selected = row.selected_outcome
            modal_safe = selected.modal_finish <= selected.task_safe_deadline + 1e-9
            pessimistic_unsafe = selected.pessimistic_finish > selected.task_safe_deadline + 1e-9
            if (modal_safe and pessimistic_unsafe) or row.uncertainty_risk_delta > 1e-9:
                candidates.append(row)
        if not candidates:
            return _insufficient(self.name, trace, "modal and pessimistic evidence do not disagree materially")
        row = max(
            candidates,
            key=lambda item: (
                item.uncertainty_risk_delta,
                item.pessimistic_finish_delta,
                item.safety_margin_delta,
            ),
        )
        selected = row.selected_outcome
        modal_safe = selected.modal_finish <= selected.task_safe_deadline + 1e-9
        pessimistic_unsafe = selected.pessimistic_finish > selected.task_safe_deadline + 1e-9
        joint_condition = trace.minimum_slack <= 0.0 and trace.maximum_uncertainty > 0.0
        alternative_advantage = (
            row.uncertainty_risk_delta > 1e-9
            or row.pessimistic_finish_delta > 1e-9
            or row.safety_margin_delta > 1e-9
        )
        code = (
            "modal_safe_pessimistic_violation"
            if modal_safe and pessimistic_unsafe
            else "high_uncertainty_low_slack_choice"
        )
        severity = "high" if modal_safe and pessimistic_unsafe and joint_condition else "medium"
        return DiagnosticReport(
            agent_name=self.name,
            decision_id=trace.decision_id,
            scenario_id=trace.scenario_id,
            seed=trace.seed,
            evidence={
                "selected_optimistic_finish": selected.optimistic_finish,
                "selected_modal_finish": selected.modal_finish,
                "selected_pessimistic_finish": selected.pessimistic_finish,
                "selected_task_safe_deadline": selected.task_safe_deadline,
                "selected_uncertainty_width": selected.uncertainty_width,
                "alternative_uncertainty_width": row.alternative_outcome.uncertainty_width,
            },
            diagnosis_code=code,
            severity=severity,
            confidence="high" if severity == "high" and row.evidence_quality == "high" else "medium",
            state_conditions={
                "low_slack_high_uncertainty_joint": joint_condition,
                "minimum_slack": trace.minimum_slack,
                "maximum_uncertainty": trace.maximum_uncertainty,
            },
            preferred_task=row.alternative_task_id if alternative_advantage else None,
            rejected_task=row.selected_task_id if alternative_advantage else None,
            quantitative_deltas={
                "pessimistic_finish_regret": row.pessimistic_finish_delta,
                "uncertainty_low_slack_regret": row.uncertainty_risk_delta,
                "safety_margin_regret": row.safety_margin_delta,
            },
            suggested_structural_actions=[
                "add_uncertainty_ddl_interaction",
                "add_pessimistic_risk_gate",
                "replace_linear_risk_with_smooth_nonlinear_form",
            ],
            limitations=(
                ["fuzzy timelines express model uncertainty, not strict causal effects"]
                + ([] if alternative_advantage else [
                    "timeline disagreement alone does not show that this alternative is better"
                ])
            ),
        )


DEFAULT_AGENTS = (
    DDLDiagnosticAgent(),
    EnergyDiagnosticAgent(),
    UncertaintyDiagnosticAgent(),
)
