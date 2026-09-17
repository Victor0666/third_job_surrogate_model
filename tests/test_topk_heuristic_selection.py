"""Parameterized Top-K export and Manager-loading tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from algorithms.llm_safe_hrl.LLM.export_topk import (
    _build_record,
    resolve_topk_paths,
    select_topk,
    topk_ranking_key,
)
from algorithms.llm_safe_hrl.base.heuristic_admission import (
    file_sha256,
)
from algorithms.llm_safe_hrl.base.manager_heuristics import (
    load_manager_heuristic_library,
)
from algorithms.llm_safe_hrl.base.topk_schema import (
    TOPK_MANIFEST_SCHEMA_VERSION,
    TOPK_SELECTION_MODE,
)
from algorithms.llm_safe_hrl.scenario_registry import (
    resolve_experiment_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_FAMILIES = [
    "CyberShake",
    "Epigenomics",
    "Ligo",
    "Montage",
    "Sipht",
]


def _context():
    return {
        "workflow_families": list(WORKFLOW_FAMILIES),
        "workflows_per_instance": 1,
        "arrival_lambda": 0.03,
        "horizon": 1e6,
        "deadline_mode": "none",
        "deadline_alpha_small": 2.0,
        "deadline_alpha_large": 3.0,
        "deadline_alpha_small_prob": 0.2,
        "num_cloud_hosts": 1,
        "num_edge_hosts": 1,
        "cloud_vms_per_host": [2],
        "edge_vms_per_host": [2],
        "cloud_pc_tiers": [1.0, 2.0],
        "edge_pc_tiers": [1.0, 2.0],
        "cloud_bw_tiers": [1000.0, 2000.0],
        "edge_bw_tiers": [1000.0, 2000.0],
        "fuzzy_delta1": 0.75,
        "fuzzy_delta2": 1.2,
        "fuzzy_deadline_eta": 0.95,
        "fuzzy_energy_lambda": 1.0,
        "fuzzy_resource_seed_mode": "episode_seed",
        "fuzzy_resource_seed_offset": 0,
    }


def _scope():
    return {
        "mode": "resource_task_domain",
        "resource_code": "S",
        "allowed_scenarios": ["SS", "MS", "LS"],
        "workflow_families": list(WORKFLOW_FAMILIES),
    }


def _metrics(source: Path, metadata: dict) -> dict:
    source_hash = file_sha256(source)
    return {
        "evaluator_protocol_version": 3,
        "function_name": "get_task_priority_v2",
        "interface_valid": True,
        "candidate_sha256": source_hash,
        "candidate_source_file": source.name,
        "frozen_rule_hash": source_hash,
        "scenario_id": "SS",
        "seeds": [1, 2, 3],
        "evaluation_seed_count": 3,
        "completed_seed_count": 3,
        "all_evaluation_seeds_completed": True,
        "constraint_feasible": True,
        "deadline_violation_rate": 0.0,
        "max_deadline_violation_rate_across_seeds": 0.0,
        "total_lateness": 0.0,
        "max_fuzzy_lateness": 0.0,
        "fuzzy_total_energy_mean": 390000.0,
        "fuzzy_total_energy_std": 1000.0,
        "fuzzy_total_energy_score": 391000.0,
        "objective": 391000.0,
        "objective_cv_across_seeds": 0.01,
        "feasible_seed_rate": 1.0,
        "training_seeds": [1, 2, 3],
        "validation_seeds": [4, 5],
        "optimizer_seed": 0,
        **metadata,
    }


def _write_topk_library(directory: Path) -> Path:
    generated = directory / "generated"
    reports = directory / "topk_reports_k1"
    generated.mkdir()
    reports.mkdir()
    metadata = {
        "structure_hash": "a" * 64,
        "parameter_schema_hash": "b" * 64,
        "best_parameter_hash": "c" * 64,
        "optimizer_config_hash": "d" * 64,
        "parameter_diagnostics_hash": "e" * 64,
    }
    source = generated / "candidate_iter1_ind0.py"
    source.write_text(
        "import numpy as np\n"
        f"RULE_METADATA = {metadata | {'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}}\n"
        "def get_task_priority_v2(min_exec_time, min_comm_time, "
        "min_incremental_energy, slack, upward_rank, remaining_work, "
        "ready_wait_time, uncertainty):\n"
        "    return np.asarray(min_exec_time, dtype=float).reshape(-1)\n",
        encoding="utf-8",
    )
    metrics = _metrics(source, metadata)
    report = reports / "candidate_iter1_ind0.json"
    report.write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    protocol = resolve_experiment_protocol(
        "single",
        source_scenario="SS",
    ).identity()
    candidate = {
        "metrics": metrics,
        "source_path": source,
        "source_hash": file_sha256(source),
        "iteration": 1,
        "individual": 0,
        "ranking_key": topk_ranking_key(metrics),
    }
    record = _build_record(
        candidate,
        rank=1,
        run_dir=directory,
        report_file=report,
        evaluation_context=_context(),
        admission_scope=_scope(),
        experiment_protocol=protocol,
    )
    manifest = directory / "topk_heuristic_library_k1.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": TOPK_MANIFEST_SCHEMA_VERSION,
                "manifest_id": "test_topk_SS_T_run1",
                "manifest_version": "test.topk.v1",
                "selection_mode": TOPK_SELECTION_MODE,
                "selection_policy": {
                    "policy_version": "feasibility_first_topk_v1",
                    "requested_k": 1,
                    "selected_count": 1,
                    "unique_structure_first": True,
                    "hard_energy_threshold": None,
                    "ranking_fields": [
                        "constraint_feasible",
                        "max_deadline_violation_rate_across_seeds",
                        "max_fuzzy_lateness",
                        "fuzzy_total_energy_score",
                        "objective_cv_across_seeds",
                        "candidate_sha256",
                    ],
                },
                "trusted_source_root": "generated",
                "trusted_report_root": "topk_reports_k1",
                "admission_scope": _scope(),
                "experiment_protocol": protocol,
                "llm_rules": [record],
            }
        ),
        encoding="utf-8",
    )
    return manifest


class ParameterizedTopKTests(unittest.TestCase):
    def test_all_nine_single_conditions_are_derived_from_arguments(self):
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as raw_directory:
            root = Path(raw_directory)
            for source in ("SS", "SM", "SL"):
                for ddl in ("T", "M", "L"):
                    with self.subTest(source=source, ddl=ddl):
                        artifact, runtime = resolve_topk_paths(
                            project_root=root,
                            source_scenario=source,
                            ddl=ddl,
                            execution_id="run_1",
                        )
                        self.assertEqual(
                            artifact,
                            (
                                root
                                / "out"
                                / "main_single"
                                / source
                                / ddl
                                / "run_1"
                            ).resolve(),
                        )
                        self.assertEqual(
                            runtime,
                            (
                                root
                                / "algorithms"
                                / "llm_safe_hrl"
                                / "LLM"
                                / "outputs"
                                / "formal"
                                / f"{source}_{ddl}"
                                / "run_1"
                            ).resolve(),
                        )

    def test_feasible_rule_precedes_lower_energy_infeasible_rule(self):
        feasible = {
            "constraint_feasible": True,
            "max_deadline_violation_rate_across_seeds": 0.0,
            "max_fuzzy_lateness": 0.0,
            "fuzzy_total_energy_score": 400000.0,
            "objective_cv_across_seeds": 0.02,
            "candidate_sha256": "a" * 64,
        }
        infeasible = {
            "constraint_feasible": False,
            "max_deadline_violation_rate_across_seeds": 0.01,
            "max_fuzzy_lateness": 1.0,
            "fuzzy_total_energy_score": 300000.0,
            "objective_cv_across_seeds": 0.01,
            "candidate_sha256": "b" * 64,
        }
        self.assertLess(
            topk_ranking_key(feasible),
            topk_ranking_key(infeasible),
        )

    def test_structure_diversity_is_applied_before_fill(self):
        def candidate(name, structure, energy):
            metrics = {
                "constraint_feasible": True,
                "max_deadline_violation_rate_across_seeds": 0.0,
                "max_fuzzy_lateness": 0.0,
                "fuzzy_total_energy_score": energy,
                "objective_cv_across_seeds": 0.01,
                "candidate_sha256": name * 64,
                "structure_hash": structure * 64,
            }
            return {
                "metrics": metrics,
                "ranking_key": topk_ranking_key(metrics),
            }

        first = candidate("a", "1", 100.0)
        same_structure = candidate("b", "1", 101.0)
        different_structure = candidate("c", "2", 102.0)
        selected = select_topk(
            [first, same_structure, different_structure],
            2,
        )
        self.assertEqual(
            selected,
            [first, different_structure],
        )

    def test_topk_rule_is_available_without_claiming_admission(self):
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as raw_directory:
            manifest = _write_topk_library(
                Path(raw_directory)
            )
            heuristics = load_manager_heuristic_library(
                manifest,
                include_traditional=False,
            )
        self.assertEqual(len(heuristics), 1)
        self.assertTrue(heuristics[0].available)
        self.assertFalse(heuristics[0].admitted)
        self.assertTrue(
            heuristics[0].selected_for_manager
        )
        self.assertEqual(
            heuristics[0].availability_reason,
            "top_k_selected",
        )


if __name__ == "__main__":
    unittest.main()
