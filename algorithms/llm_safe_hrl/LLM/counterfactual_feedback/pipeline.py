"""Offline trace-to-diagnostics pipeline and artifact writer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .cache import CounterfactualCache, LOCAL_ESTIMATOR_VERSION
from .counterfactual_estimator import LocalCounterfactualEstimator
from .diagnostic_agents import DEFAULT_AGENTS
from .schemas import (
    CounterfactualComparison,
    CounterfactualConfig,
    DecisionTrace,
    DiagnosticReport,
    canonical_hash,
    reject_test_seeds,
)
from .trace_recorder import TraceRecorder


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n")


class CounterfactualRunSession:
    """Collect one frozen-rule scenario/seed run before actual assignment."""

    def __init__(
        self,
        config: CounterfactualConfig,
        metadata: dict[str, Any],
        output_root: str | Path,
        resource_config_hash: str,
        *,
        final_test_seeds: Iterable[int],
        planning_only: bool = False,
        analysis_decision_indices: Iterable[int] | None = None,
    ):
        self.config = config
        self.metadata = dict(metadata)
        self.output_root = Path(output_root).resolve()
        self.resource_config_hash = str(resource_config_hash)
        self.final_test_seeds = tuple(int(seed) for seed in final_test_seeds)
        reject_test_seeds([int(self.metadata["seed"])], list(self.final_test_seeds))
        self.used_test_seed = int(self.metadata["seed"]) in set(self.final_test_seeds)
        self.planning_only = bool(planning_only)
        self.analysis_decision_indices = (
            None
            if analysis_decision_indices is None
            else {int(index) for index in analysis_decision_indices}
        )
        self._critical_count = 0
        cache_path = Path(config.cache_path)
        if not cache_path.is_absolute():
            cache_path = self.output_root / "cache.json"
        self.cache = CounterfactualCache(enabled=config.cache_enabled, path=cache_path)
        self.recorder = TraceRecorder(config)
        self.estimator = LocalCounterfactualEstimator(config, self.cache, resource_config_hash)
        self.traces: list[DecisionTrace] = []
        self.comparisons: list[CounterfactualComparison] = []
        self.reports: list[DiagnosticReport] = []

    def observe_decision(
        self,
        environment,
        ready_tasks,
        selection_details: dict[str, Any],
        decision_index: int,
    ) -> None:
        recorded = self.recorder.record(
            environment,
            ready_tasks,
            selection_details,
            self.metadata,
            decision_index,
            store_full_ready_features=(
                bool(self.metadata.get("archive_ready_features", False))
                and not self.planning_only
                and (
                    self.analysis_decision_indices is None
                    or decision_index in self.analysis_decision_indices
                )
            ),
        )
        if recorded is None:
            return
        trace = recorded.trace
        if self.planning_only:
            self.traces.append(trace)
            return
        if self.analysis_decision_indices is None:
            selected_for_analysis = (
                trace.is_critical
                and self._critical_count < self.config.max_critical_decisions_per_run
            )
        else:
            selected_for_analysis = (
                trace.is_critical
                and decision_index in self.analysis_decision_indices
            )
        if not selected_for_analysis:
            trace = replace(trace, is_critical=False, critical_reasons=[])
        else:
            self._critical_count += 1
        self.traces.append(trace)
        if not trace.is_critical:
            return
        comparisons = [
            self.estimator.compare(environment, trace, task_id)
            for task_id in recorded.alternative_task_ids
        ]
        self.comparisons.extend(comparisons)
        context = {"estimator_type": "local_one_step", "causal_claim": False}
        self.reports.extend(
            agent.analyze(trace, comparisons, context)
            for agent in DEFAULT_AGENTS
        )

    def selected_critical_decision_indices(self) -> list[int]:
        """Return deterministic Top-N critical decisions from the traced window."""
        candidates = [trace for trace in self.traces if trace.is_critical]
        ranked = sorted(
            candidates,
            key=lambda trace: (-trace.criticality_score, trace.decision_index),
        )[: self.config.max_critical_decisions_per_run]
        return sorted(trace.decision_index for trace in ranked)

    def finalize(self) -> dict[str, Any]:
        if self.planning_only:
            raise RuntimeError("planning-only counterfactual sessions do not write artifacts")
        run_suffix = canonical_hash(str(self.metadata["run_id"]))[:10]
        stem = (
            f"{str(self.metadata['structure_hash'])[:16]}_"
            f"{self.metadata['scenario_id']}_{int(self.metadata['seed'])}_{run_suffix}"
        )
        trace_path = self.output_root / "traces" / f"{stem}.jsonl"
        comparison_path = self.output_root / "comparisons" / f"{stem}.jsonl"
        diagnostics_path = self.output_root / "diagnostics" / f"{stem}_agent_reports.json"
        manifest_path = self.output_root / "manifests" / f"{stem}_manifest.json"
        trace_payload = [row.to_dict() for row in self.traces]
        comparison_payload = [row.to_dict() for row in self.comparisons]
        report_payload = [row.to_dict() for row in self.reports]
        _write_jsonl(trace_path, trace_payload)
        _write_jsonl(comparison_path, comparison_payload)
        _write_json(diagnostics_path, report_payload)
        manifest = {
            "manifest_version": "counterfactual_manifest_v1",
            "run_id": self.metadata["run_id"],
            "structure_hash": self.metadata["structure_hash"],
            "frozen_rule_hash": self.metadata["frozen_rule_hash"],
            "parameter_hash": self.metadata.get("parameter_hash", ""),
            "scenario_id": self.metadata["scenario_id"],
            "seed": int(self.metadata["seed"]),
            "analyzed_seeds": [int(self.metadata["seed"])],
            "analyzed_scenarios": [self.metadata["scenario_id"]],
            "counterfactual_config": self.config.to_dict(),
            "counterfactual_config_hash": self.config.config_hash,
            "resource_config_hash": self.resource_config_hash,
            "trace_hash": canonical_hash(trace_payload),
            "comparison_hash": canonical_hash(comparison_payload),
            "diagnostics_hash": canonical_hash(report_payload),
            "trace_path": str(trace_path),
            "comparison_path": str(comparison_path),
            "diagnostics_path": str(diagnostics_path),
            "traced_decisions": len(trace_payload),
            "critical_decisions": sum(row.is_critical for row in self.traces),
            "compared_alternatives": len(comparison_payload),
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "cache_hit_rate": self.cache.hit_rate,
            "estimator_type": "local_one_step",
            "estimator_version": LOCAL_ESTIMATOR_VERSION,
            "bounded_replay_capable": False,
            "used_test_seed": bool(self.used_test_seed),
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        _write_json(manifest_path, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
