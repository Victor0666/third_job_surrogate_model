"""阶段 12 SeEvo 安全启发式准入与版本管理测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from base.heuristic_admission import (
    ADMISSION_MANIFEST_SCHEMA_VERSION,
    ADMISSION_RECORD_SCHEMA_VERSION,
    CEWS_EVALUATOR_PROTOCOL_VERSION,
    append_admission_record,
    build_admission_record,
    canonical_json_sha256,
    evaluate_admission_result,
    file_sha256,
    record_sha256,
)
from base.manager_heuristics import (
    heuristic_availability_mask,
    load_manager_heuristic_library,
)
from LLM.rule_optimization import (
    freeze_rule_source,
    parse_rule_candidate,
)
from algorithms.llm_safe_hrl.scenario_registry import (
    resolve_experiment_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config():
    return {
        "problem_size": 2,
        "dataset": {
            "scenario": "SS",
            "dax_files": [
                "CyberShake_30.xml",
                "Epigenomics_24.xml",
                "Ligo_30.xml",
                "Montage_25.xml",
                "Sipht_29.xml",
            ],
            "workflows_per_instance": 2,
            "arrival_lambda": 0.03,
            "horizon": 1000.0,
            "deadline_mode": "none",
            "deadline_alpha_small": 2.0,
            "deadline_alpha_large": 3.0,
            "deadline_alpha_small_prob": 0.2,
        },
        "resources": {
            "num_cloud_hosts": 1,
            "num_edge_hosts": 1,
            "cloud_vms_per_host": [2],
            "edge_vms_per_host": [2],
            "cloud_pc_tiers": [1.0, 2.0],
            "edge_pc_tiers": [1.0, 2.0],
            "cloud_bw_tiers": [1000.0, 2000.0],
            "edge_bw_tiers": [1000.0, 2000.0],
        },
        "fuzzy": {
            "delta1": 0.75,
            "delta2": 1.2,
            "deadline_eta": 0.95,
            "energy_uncertainty_weight": 1.0,
            "resource_seed_mode": "episode_seed",
            "resource_seed_offset": 0,
        },
        "admission_scope": {
            "mode": "resource_task_domain",
            "resource_code": "S",
            "allowed_scenarios": ["SS", "MS", "LS"],
            "workflow_families": [
                "CyberShake",
                "Epigenomics",
                "Ligo",
                "Montage",
                "Sipht",
            ],
        },
        "admission": {
            "policy_version": "test_2seed_v1",
            "required_evaluation_seeds": [3, 7],
            "minimum_evaluation_seed_count": 2,
            "deadline_violation_rate_max": 0.0,
            "max_fuzzy_lateness_max": 0.0,
            "feasible_seed_rate_min": 1.0,
            "fuzzy_energy_score_max": 120.0,
            "objective_cv_max": 0.05,
        },
    }


def _result(source: Path, config, *, score=100.0):
    return {
        "evaluator_protocol_version": (
            CEWS_EVALUATOR_PROTOCOL_VERSION
        ),
        "function_name": "get_task_priority_v2",
        "interface_valid": True,
        "candidate_source_file": source.name,
        "candidate_sha256": file_sha256(source),
        "evaluation_config_sha256": canonical_json_sha256(config),
        "seeds": [3, 7],
        "evaluation_seed_count": 2,
        "completed_seed_count": 2,
        "all_evaluation_seeds_completed": True,
        "constraint_feasible": True,
        "feasible_seed_rate": 1.0,
        "deadline_violation_rate": 0.0,
        "max_deadline_violation_rate_across_seeds": 0.0,
        "total_lateness": 0.0,
        "max_fuzzy_lateness": 0.0,
        "fuzzy_total_energy_mean": score - 5.0,
        "fuzzy_total_energy_std": 5.0,
        "fuzzy_total_energy_score": score,
        "objective_cv_across_seeds": 0.01,
        "per_seed_metrics": [
            {
                "seed": seed,
                "completed_workflows": 2,
                "constraint_feasible": True,
                "deadline_violation_rate": 0.0,
                "max_fuzzy_lateness": 0.0,
                "fuzzy_total_energy_mean": score - 5.0,
                "fuzzy_total_energy_std": 5.0,
                "fuzzy_total_energy_score": score,
            }
            for seed in (3, 7)
        ],
    }


class AdmissionFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        )
        self.root = Path(self.temporary.name)
        self.generated = self.root / "generated"
        self.reports = self.root / "admission_reports"
        self.generated.mkdir()
        self.reports.mkdir()
        self.manifest = self.root / "manifest.json"
        self.config = _config()

    def tearDown(self):
        self.temporary.cleanup()

    def make_source(self, iteration=4, individual=2, body=None):
        source = (
            self.generated
            / f"candidate_iter{iteration}_ind{individual}.py"
        )
        source.write_text(
            body
            or (
                "import numpy as np\n"
                "def get_task_priority_v2(a,b,c,d,e,f,g,h):\n"
                "    return np.asarray(a, dtype=float).reshape(-1)\n"
            ),
            encoding="utf-8",
        )
        return source

    def make_record(
        self,
        source,
        result=None,
        *,
        heuristic_id="safe_rule",
        iteration=4,
        individual=2,
    ):
        payload = result or _result(source, self.config)
        report = self.reports / f"{heuristic_id}.json"
        report.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return build_admission_record(
            heuristic_id=heuristic_id,
            source_path=source,
            evaluation_report_path=report,
            evaluation_config=self.config,
            manifest_path=self.manifest,
            seevo_iteration=iteration,
            seevo_individual=individual,
        )


class AdmissionRecordTests(AdmissionFixture):
    def test_admitted_record_contains_required_versioned_fields(self):
        source = self.make_source()
        record = self.make_record(source)
        self.assertTrue(record["admitted"])
        self.assertEqual(record["admission_status"], "admitted")
        self.assertEqual(
            record["record_schema_version"],
            ADMISSION_RECORD_SCHEMA_VERSION,
        )
        self.assertEqual(
            record["admission_scope"],
            self.config["admission_scope"],
        )
        self.assertEqual(record["rejection_reasons"], [])
        self.assertEqual(record["source_hash"], file_sha256(source))
        self.assertEqual(record["seevo_iteration"], 4)
        self.assertEqual(record["evaluation_seeds"], [3, 7])
        self.assertEqual(record["fuzzy_energy_mean"], 95.0)
        self.assertEqual(record["fuzzy_energy_std"], 5.0)
        self.assertEqual(record["fuzzy_energy_score"], 100.0)
        self.assertEqual(record["deadline_violation_rate"], 0.0)
        self.assertEqual(record["max_fuzzy_lateness"], 0.0)
        self.assertEqual(record["feasible_seed_rate"], 1.0)
        self.assertEqual(
            record["record_sha256"],
            record_sha256(record),
        )
        self.assertIn("src" + file_sha256(source)[:12], record["version"])

        append_admission_record(self.manifest, record)
        manifest_payload = json.loads(
            self.manifest.read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest_payload["schema_version"],
            ADMISSION_MANIFEST_SCHEMA_VERSION,
        )
        self.assertEqual(
            manifest_payload["admission_scope"],
            self.config["admission_scope"],
        )
        heuristics = load_manager_heuristic_library(self.manifest)
        self.assertEqual(
            heuristic_availability_mask(heuristics).tolist(),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )

    def test_frozen_parameter_hashes_are_admitted_and_tamper_invalidates_report(self):
        template = '''import numpy as np
PARAMETER_SCHEMA = {"parameters": [
    {"name": "weight", "initial_value": 1.0, "lower_bound": 0.0,
     "upper_bound": 2.0, "semantic_description": "slack weight"},
    {"name": "bias", "initial_value": 0.0, "lower_bound": -1.0,
     "upper_bound": 1.0, "semantic_description": "priority bias"}
]}
def get_task_priority_v2(
    min_exec_time, min_comm_time, min_incremental_energy, slack,
    upward_rank, remaining_work, ready_wait_time, uncertainty
):
    return PARAMS["weight"] * np.asarray(slack, dtype=float) + PARAMS["bias"]
'''
        candidate = parse_rule_candidate(template)
        metadata = {
            "structure_hash": candidate.structure_hash,
            "parameter_schema_hash": candidate.parameter_schema.schema_hash,
            "best_parameters": {"weight": 1.25, "bias": 0.1},
            "optimizer_config_hash": "b" * 64,
            "parameter_diagnostics_hash": "c" * 64,
            "optimizer_seed": 5,
            "training_seeds": [3, 7],
            "validation_seeds": [],
        }
        metadata["best_parameter_hash"] = canonical_json_sha256(
            metadata["best_parameters"]
        )
        source = self.make_source(
            body=freeze_rule_source(
                template,
                candidate.parameter_schema,
                {"weight": 1.25, "bias": 0.1},
                metadata=metadata,
            )
        )
        result = _result(source, self.config)
        result.update(metadata)
        result["frozen_rule_hash"] = file_sha256(source)
        record = self.make_record(source, result)
        self.assertTrue(record["admitted"])
        self.assertTrue(record["optimization_metadata_present"])
        self.assertEqual(record["structure_hash"], candidate.structure_hash)
        self.assertEqual(record["frozen_rule_hash"], file_sha256(source))
        append_admission_record(self.manifest, record)
        source.write_text(
            source.read_text(encoding="utf-8").replace("1.25", "1.5"),
            encoding="utf-8",
        )
        heuristics = load_manager_heuristic_library(self.manifest)
        self.assertFalse(heuristics[-1].admitted)
        self.assertIn(
            "candidate_source_hash_mismatch",
            heuristics[-1].availability_reason,
        )

    def test_counterfactual_metadata_rejects_test_seed_and_records_hashes(self):
        source = self.make_source()
        result = _result(source, self.config)
        result.update(
            {
                "counterfactual_feedback_hash": "a" * 64,
                "counterfactual_config_hash": "b" * 64,
                "counterfactual_analyzed_seeds": [3, 7],
                "counterfactual_analyzed_scenarios": ["SS", "MS"],
                "counterfactual_used_test_seed": False,
            }
        )
        record = self.make_record(source, result)
        self.assertTrue(record["admitted"])
        self.assertTrue(record["counterfactual_metadata_present"])
        self.assertEqual(record["counterfactual_feedback_hash"], "a" * 64)

        isolated_config = copy.deepcopy(self.config)
        isolated_config["dataset"]["test_seeds"] = [100]
        contaminated = copy.deepcopy(result)
        contaminated["counterfactual_analyzed_seeds"] = [3, 100]
        report = self.reports / "counterfactual_test_seed.json"
        report.write_text(json.dumps(contaminated), encoding="utf-8")
        rejected = build_admission_record(
            heuristic_id="counterfactual_test_seed",
            source_path=source,
            evaluation_report_path=report,
            evaluation_config=isolated_config,
            manifest_path=self.manifest,
            seevo_iteration=4,
            seevo_individual=2,
        )
        self.assertFalse(rejected["admitted"])
        self.assertIn(
            "counterfactual_feedback_uses_final_test_seed",
            rejected["rejection_reasons"],
        )

    def test_critical_state_replay_metadata_is_recorded_and_fails_closed(self):
        source = self.make_source()
        result = _result(source, self.config)
        result.update(
            {
                "critical_state_archive_hash": "a" * 64,
                "critical_state_replay_hash": "b" * 64,
                "critical_state_config_hash": "c" * 64,
                "critical_state_replayed_state_count": 4,
                "critical_state_verified_failure_count": 1,
                "critical_state_used_test_seed": False,
                "critical_state_archive_version": "critical_state_archive_v1",
            }
        )
        record = self.make_record(source, result, heuristic_id="critical_state_safe")
        self.assertTrue(record["admitted"])
        self.assertTrue(record["critical_state_metadata_present"])
        self.assertEqual(record["critical_state_replayed_state_count"], 4)

        contaminated = copy.deepcopy(result)
        contaminated["critical_state_used_test_seed"] = True
        rejected = self.make_record(
            source,
            contaminated,
            heuristic_id="critical_state_contaminated",
        )
        self.assertFalse(rejected["admitted"])
        self.assertIn(
            "critical_state_replay_uses_final_test_seed",
            rejected["rejection_reasons"],
        )

    def test_policy_rejects_incomplete_unsafe_expensive_or_unstable(self):
        source = self.make_source()
        base = _result(source, self.config)
        cases = {}

        incomplete = copy.deepcopy(base)
        incomplete["all_evaluation_seeds_completed"] = False
        incomplete["completed_seed_count"] = 1
        cases["incomplete"] = (
            incomplete,
            "not_all_evaluation_seeds_completed",
        )

        violation = copy.deepcopy(base)
        violation["constraint_feasible"] = False
        violation["feasible_seed_rate"] = 0.5
        violation[
            "max_deadline_violation_rate_across_seeds"
        ] = 0.1
        violation["per_seed_metrics"][1][
            "deadline_violation_rate"
        ] = 0.1
        violation["per_seed_metrics"][1][
            "constraint_feasible"
        ] = False
        cases["violation"] = (
            violation,
            "seed_nonzero_deadline_violation:7",
        )

        late = copy.deepcopy(base)
        late["max_fuzzy_lateness"] = 0.5
        late["per_seed_metrics"][0][
            "max_fuzzy_lateness"
        ] = 0.5
        cases["lateness"] = (
            late,
            "nonzero_max_fuzzy_lateness",
        )

        expensive = copy.deepcopy(base)
        expensive["fuzzy_total_energy_score"] = 121.0
        cases["energy"] = (
            expensive,
            "fuzzy_energy_score_above_threshold",
        )

        unstable = copy.deepcopy(base)
        unstable["objective_cv_across_seeds"] = 0.051
        cases["stability"] = (
            unstable,
            "objective_stability_below_requirement",
        )

        policy = self.config["admission"]
        for name, (payload, expected_reason) in cases.items():
            with self.subTest(name=name):
                reasons = evaluate_admission_result(
                    payload,
                    policy,
                    expected_workflows_per_seed=2,
                )
                self.assertIn(expected_reason, reasons)

    def test_rejected_rule_is_registered_but_never_imported(self):
        source = self.make_source(
            body=(
                "raise RuntimeError('must never import rejected source')\n"
                "def get_task_priority_v2(a,b,c,d,e,f,g,h):\n"
                "    return a\n"
            )
        )
        payload = _result(source, self.config)
        payload["constraint_feasible"] = False
        payload["feasible_seed_rate"] = 0.5
        payload[
            "max_deadline_violation_rate_across_seeds"
        ] = 0.1
        record = self.make_record(source, payload)
        self.assertFalse(record["admitted"])
        append_admission_record(self.manifest, record)
        heuristic = load_manager_heuristic_library(
            self.manifest
        )[-1]
        self.assertFalse(heuristic.available)
        self.assertIsNone(heuristic.priority_rule)
        self.assertIn(
            "manifest_not_admitted",
            heuristic.availability_reason,
        )

    def test_duplicate_id_or_version_cannot_overwrite_history(self):
        source = self.make_source()
        record = self.make_record(source)
        append_admission_record(self.manifest, record)
        with self.assertRaisesRegex(ValueError, "duplicate heuristic_id"):
            append_admission_record(self.manifest, record)


class AdmissionLoaderIntegrityTests(AdmissionFixture):
    def test_legacy_manifest_schema_is_rejected_explicitly(self):
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "manifest_id": "legacy",
                    "manifest_version": "legacy",
                    "llm_rules": [],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ValueError,
            "manifest schema mismatch",
        ):
            load_manager_heuristic_library(self.manifest)

    def test_source_hash_mismatch_masks_rule_without_executing_tamper(self):
        source = self.make_source()
        record = self.make_record(source)
        append_admission_record(self.manifest, record)
        source.write_text(
            "raise RuntimeError('tampered source executed')\n",
            encoding="utf-8",
        )
        heuristic = load_manager_heuristic_library(
            self.manifest
        )[-1]
        self.assertFalse(heuristic.available)
        self.assertIsNone(heuristic.priority_rule)
        self.assertIn(
            "candidate_source_hash_mismatch",
            heuristic.availability_reason,
        )

    def test_invalid_interface_is_explicitly_masked(self):
        source = self.make_source(
            body=(
                "def get_task_priority_v2(a,b,c,d,e,f,g,h):\n"
                "    return 1.0\n"
            )
        )
        record = self.make_record(source)
        append_admission_record(self.manifest, record)
        heuristic = load_manager_heuristic_library(
            self.manifest
        )[-1]
        self.assertFalse(heuristic.available)
        self.assertIn(
            "candidate_validation_failed:ValueError",
            heuristic.availability_reason,
        )

    def test_evaluation_report_hash_mismatch_masks_rule(self):
        source = self.make_source()
        record = self.make_record(source)
        append_admission_record(self.manifest, record)
        report = self.root / record["evaluation_report_file"]
        report.write_text("{}", encoding="utf-8")
        heuristic = load_manager_heuristic_library(
            self.manifest
        )[-1]
        self.assertFalse(heuristic.available)
        self.assertIn(
            "evaluation_report_hash_mismatch",
            heuristic.availability_reason,
        )

    def test_record_cannot_relax_manifest_admission_policy(self):
        source = self.make_source()
        record = self.make_record(source)
        append_admission_record(self.manifest, record)
        payload = json.loads(
            self.manifest.read_text(encoding="utf-8")
        )
        stored = payload["llm_rules"][0]
        stored["admission_policy"][
            "fuzzy_energy_score_max"
        ] = 1_000_000.0
        stored["record_sha256"] = record_sha256(stored)
        self.manifest.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        heuristic = load_manager_heuristic_library(
            self.manifest
        )[-1]
        self.assertFalse(heuristic.available)
        self.assertIsNone(heuristic.priority_rule)
        self.assertIn(
            "manifest_admission_policy_mismatch",
            heuristic.availability_reason,
        )

    def test_source_outside_trusted_generated_root_is_rejected(self):
        source = self.root / "unknown.py"
        source.write_text(
            "raise RuntimeError('unknown source executed')\n",
            encoding="utf-8",
        )
        report = self.reports / "unknown.json"
        report.write_text(
            json.dumps(_result(source, self.config)),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ValueError,
            "candidate source must remain under",
        ):
            build_admission_record(
                heuristic_id="unknown",
                source_path=source,
                evaluation_report_path=report,
                evaluation_config=self.config,
                manifest_path=self.manifest,
                seevo_iteration=0,
                seevo_individual=0,
            )


class ManifestCliProtocolTests(AdmissionFixture):
    """验证登记工具把实验协议身份带进记录并对照 manifest 校验。"""

    def setUp(self):
        super().setUp()
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        self.addCleanup(
            lambda: sys.path.remove(str(PROJECT_ROOT / "tools"))
        )
        import manage_safe_heuristic_manifest

        self.cli = manage_safe_heuristic_manifest
        self.protocol = resolve_experiment_protocol(
            "single", source_scenario="SS"
        ).identity()

    def _write(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _argv(self, *extra):
        source = self.make_source()
        report = self.reports / "cli.json"
        report.write_text(
            json.dumps(_result(source, self.config)),
            encoding="utf-8",
        )
        return [
            "--manifest", str(self.manifest),
            "--candidate", str(source),
            "--evaluation-report", str(report),
            "--config", self._write("config.json", self.config),
            "--heuristic-id", "cli_rule",
            "--seevo-iteration", "4",
            "--seevo-individual", "2",
            *extra,
        ]

    def test_protocol_identity_is_stamped_into_record_and_manifest(self):
        argv = self._argv(
            "--experiment-protocol",
            self._write("protocol.json", self.protocol),
        )
        self.assertEqual(self.cli.main(argv), 0)

        payload = json.loads(
            self.manifest.read_text(encoding="utf-8")
        )
        self.assertEqual(
            payload["experiment_protocol"], self.protocol
        )
        self.assertEqual(
            payload["llm_rules"][0]["experiment_protocol"],
            self.protocol,
        )
        # 只有带上协议身份，加载器才能强制执行协议绑定。
        load_manager_heuristic_library(
            self.manifest,
            expected_protocol_identity=self.protocol,
        )

    def test_protocol_less_registration_is_refused_by_default(self):
        # 评价配置里没有协议字段，也没有 --experiment-protocol：
        # 若放行就会建出一条事后无法再绑定协议的 manifest 谱系。
        with self.assertRaises(SystemExit):
            self.cli.main(self._argv())
        self.assertFalse(self.manifest.exists())

        # 明确声明后仍可用于旧的无协议 manifest。
        self.assertEqual(
            self.cli.main(
                self._argv("--allow-missing-protocol")
            ),
            0,
        )
        payload = json.loads(
            self.manifest.read_text(encoding="utf-8")
        )
        self.assertNotIn("experiment_protocol", payload)

    def test_manifest_from_another_protocol_is_rejected(self):
        self.assertEqual(
            self.cli.main(
                self._argv(
                    "--experiment-protocol",
                    self._write("protocol.json", self.protocol),
                )
            ),
            0,
        )
        other = resolve_experiment_protocol(
            "multi", source_scenario=None, resource_scale="S"
        ).identity()
        argv = [
            *self._argv(
                "--experiment-protocol",
                self._write("other.json", other),
            ),
        ]
        argv[argv.index("--heuristic-id") + 1] = "cli_rule_2"
        # 拦截来自 append_admission_record 已有的“记录 vs manifest”校验，
        # 而不是本次新增的 expected_protocol_identity——后者在本工具里是
        # 冗余的断言，因为记录本身就是用同一份身份构造的。这里锁定的是
        # 跨协议登记必须失败这一行为，与由哪一道检查拦下无关。
        with self.assertRaises(ValueError):
            self.cli.main(argv)


if __name__ == "__main__":
    unittest.main()
