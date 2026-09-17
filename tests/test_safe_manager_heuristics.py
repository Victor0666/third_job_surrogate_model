"""阶段 11 SeEvo 安全启发式 Manager 接入测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT
from base.heuristic_admission import (
    CEWS_EVALUATOR_PROTOCOL_VERSION,
    append_admission_record,
    build_admission_record,
    canonical_json_sha256,
    file_sha256,
)
from base.hrl_env import HrlFcfsCacheEnv
from base.manager_heuristics import (
    HEURISTIC_SELECTION_MODE,
    LEGACY_RULE_WEIGHT_MODE,
    heuristic_availability_mask,
    load_manager_heuristic_library,
)
from hrl_mix.train_config import (
    SafeManagerHeuristicConfig,
    SafeRLConfig,
    resolve_manager_heuristic_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = (
    LLM_ROOT
    / "problems"
    / "cews_task_constructive"
    / "safe_heuristic_library_resS.json"
)
LEGACY_LIBRARY = (
    LLM_ROOT
    / "problems"
    / "cews_task_constructive"
    / "safe_heuristic_library.json"
)
WORKFLOW_FAMILIES = [
    "CyberShake",
    "Epigenomics",
    "Ligo",
    "Montage",
    "Sipht",
]
TASK_DAX_FILES = {
    "S": [
        "CyberShake_30.xml",
        "Epigenomics_24.xml",
        "Ligo_30.xml",
        "Montage_25.xml",
        "Sipht_29.xml",
    ],
    "M": [
        "CyberShake_50.xml",
        "Epigenomics_47.xml",
        "Ligo_50.xml",
        "Montage_50.xml",
        "Sipht_58.xml",
    ],
    "L": [
        "CyberShake_100.xml",
        "Epigenomics_100.xml",
        "Ligo_100.xml",
        "Montage_100.xml",
        "Sipht_97.xml",
    ],
}


def _test_evaluation_context(scenario="SS"):
    scenario = str(scenario).upper()
    return {
        "scenario_code": scenario,
        "task_code": scenario[0],
        "resource_code": scenario[1],
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
        "cloud_pc_tiers": [1.0, 2.0, 4.0, 6.0, 8.0],
        "edge_pc_tiers": [1.0, 2.0, 4.0, 6.0, 8.0],
        "cloud_bw_tiers": [
            1000.0,
            2000.0,
            4000.0,
            6000.0,
            8000.0,
        ],
        "edge_bw_tiers": [
            1000.0,
            2000.0,
            4000.0,
            6000.0,
            8000.0,
        ],
        "fuzzy_delta1": 0.75,
        "fuzzy_delta2": 1.2,
        "fuzzy_deadline_eta": 0.95,
        "fuzzy_energy_lambda": 1.0,
        "fuzzy_resource_seed_mode": "episode_seed",
        "fuzzy_resource_seed_offset": 0,
    }


def _test_admission_config():
    return {
        "problem_size": 1,
        "dataset": {
            "scenario": "SS",
            "dax_files": list(TASK_DAX_FILES["S"]),
            "workflows_per_instance": 1,
            "arrival_lambda": 0.03,
            "horizon": 1e6,
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
            "cloud_pc_tiers": [1.0, 2.0, 4.0, 6.0, 8.0],
            "edge_pc_tiers": [1.0, 2.0, 4.0, 6.0, 8.0],
            "cloud_bw_tiers": [
                1000.0,
                2000.0,
                4000.0,
                6000.0,
                8000.0,
            ],
            "edge_bw_tiers": [
                1000.0,
                2000.0,
                4000.0,
                6000.0,
                8000.0,
            ],
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
            "workflow_families": list(WORKFLOW_FAMILIES),
        },
        "admission": {
            "policy_version": "test_v1",
            "required_evaluation_seeds": [0, 1],
            "minimum_evaluation_seed_count": 2,
            "deadline_violation_rate_max": 0.0,
            "max_fuzzy_lateness_max": 0.0,
            "feasible_seed_rate_min": 1.0,
            "fuzzy_energy_score_max": 200.0,
            "objective_cv_max": 0.05,
        },
    }


def _test_evaluation_result(source: Path, config, violation=0.0):
    lateness = 0.0 if violation == 0.0 else 1.0
    feasible = violation == 0.0
    return {
        "evaluator_protocol_version": (
            CEWS_EVALUATOR_PROTOCOL_VERSION
        ),
        "function_name": "get_task_priority_v2",
        "interface_valid": True,
        "candidate_sha256": file_sha256(source),
        "evaluation_config_sha256": canonical_json_sha256(config),
        "seeds": [0, 1],
        "evaluation_seed_count": 2,
        "completed_seed_count": 2,
        "all_evaluation_seeds_completed": True,
        "constraint_feasible": feasible,
        "feasible_seed_rate": 1.0 if feasible else 0.0,
        "deadline_violation_rate": float(violation),
        "max_deadline_violation_rate_across_seeds": float(
            violation
        ),
        "total_lateness": lateness,
        "max_fuzzy_lateness": lateness,
        "fuzzy_total_energy_mean": 90.0,
        "fuzzy_total_energy_std": 10.0,
        "fuzzy_total_energy_score": 100.0,
        "objective_cv_across_seeds": 0.01,
        "per_seed_metrics": [
            {
                "seed": seed,
                "completed_workflows": 1,
                "constraint_feasible": feasible,
                "deadline_violation_rate": float(violation),
                "max_fuzzy_lateness": lateness,
                "fuzzy_total_energy_mean": 90.0,
                "fuzzy_total_energy_std": 10.0,
                "fuzzy_total_energy_score": 100.0,
            }
            for seed in (0, 1)
        ],
    }


def _write_test_library(directory: Path) -> Path:
    generated = directory / "generated"
    generated.mkdir()
    reports = directory / "admission_reports"
    reports.mkdir()
    good = generated / "candidate_iter1_ind0.py"
    good.write_text(
        (
            "import numpy as np\n"
            "def get_task_priority_v2(a,b,c,d,e,f,g,h):\n"
            "    return np.asarray(a, dtype=float).reshape(-1)\n"
        ),
        encoding="utf-8",
    )
    # 因准入指标不合格，该文件绝不能被 import。
    rejected = generated / "candidate_iter2_ind0.py"
    rejected.write_text(
        (
            "raise RuntimeError('rejected rule was imported')\n"
            "def get_task_priority_v2(a,b,c,d,e,f,g,h):\n"
            "    return a\n"
        ),
        encoding="utf-8",
    )
    config = _test_admission_config()
    good_report = reports / "good.json"
    good_report.write_text(
        json.dumps(_test_evaluation_result(good, config)),
        encoding="utf-8",
    )
    rejected_report = reports / "rejected.json"
    rejected_report.write_text(
        json.dumps(
            _test_evaluation_result(
                rejected,
                config,
                violation=0.1,
            )
        ),
        encoding="utf-8",
    )
    manifest = directory / "safe_library.json"
    good_record = build_admission_record(
        heuristic_id="test_llm_safe",
        source_path=good,
        evaluation_report_path=good_report,
        evaluation_config=config,
        manifest_path=manifest,
        seevo_iteration=1,
        seevo_individual=0,
    )
    rejected_record = build_admission_record(
        heuristic_id="test_llm_rejected",
        source_path=rejected,
        evaluation_report_path=rejected_report,
        evaluation_config=config,
        manifest_path=manifest,
        seevo_iteration=2,
        seevo_individual=0,
    )
    append_admission_record(manifest, good_record)
    append_admission_record(manifest, rejected_record)
    return manifest


def _make_environment(
    *,
    manager_mode=LEGACY_RULE_WEIGHT_MODE,
    manifest_path=None,
    scenario="SS",
    random_seed=0,
    llm_only=False,
):
    scenario = str(scenario).upper()
    task_code = scenario[0]
    resource_code = scenario[1]
    selection = manager_mode == HEURISTIC_SELECTION_MODE
    environment = HrlFcfsCacheEnv(
        dax_paths=[
            str(
                PROJECT_ROOT
                / "data"
                / "dax"
                / dax_name
            )
            for dax_name in TASK_DAX_FILES[task_code]
        ],
        deadline_mode="none",
        workflows_per_episode=1,
        horizon=1e6,
        arrival_lambda=0.03,
        random_seed=random_seed,
        max_ready_tasks=32,
        num_cloud_hosts=1,
        num_edge_hosts=1,
        cloud_vms_per_host=(2,),
        edge_vms_per_host=(2,),
        fuzzy_enabled=selection,
        safe_rl_enabled=selection,
        safe_rl_shield_enabled=selection,
        safe_rl_state_enabled=selection,
        manager_mode=manager_mode,
        manager_heuristic_library_path=manifest_path,
        manager_heuristic_llm_only=llm_only,
        manager_heuristic_recent_window=4,
        scenario_code=scenario,
        task_code=task_code,
        resource_code=resource_code,
        workflow_families=WORKFLOW_FAMILIES,
    )
    environment.reset()
    return environment


class SafeHeuristicLibraryTests(unittest.TestCase):
    def test_legacy_manifest_is_not_silently_loaded(self):
        with self.assertRaisesRegex(
            ValueError,
            "manifest schema mismatch",
        ):
            load_manager_heuristic_library(LEGACY_LIBRARY)

    def test_resource_default_library_keeps_traditional_order(self):
        heuristics = load_manager_heuristic_library(
            DEFAULT_LIBRARY
        )
        self.assertEqual(
            [row.heuristic_id for row in heuristics[:5]],
            [
                "traditional_fcfs",
                "traditional_sjf",
                "traditional_mcf",
                "traditional_hur",
                "traditional_edf",
            ],
        )
        mask = heuristic_availability_mask(heuristics)
        np.testing.assert_array_equal(
            mask,
            [1, 1, 1, 1, 1, 0],
        )

    def test_invalid_llm_rule_is_masked_without_importing_it(self):
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as raw_directory:
            manifest = _write_test_library(
                Path(raw_directory)
            )
            heuristics = load_manager_heuristic_library(
                manifest
            )
            mask = heuristic_availability_mask(heuristics)
            np.testing.assert_array_equal(
                mask,
                [1, 1, 1, 1, 1, 1, 0],
            )
            self.assertIn(
                "manifest_not_admitted",
                heuristics[-1].availability_reason,
            )

    def test_context_mismatch_masks_otherwise_admitted_llm_rule(self):
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as raw_directory:
            manifest = _write_test_library(
                Path(raw_directory)
            )
            runtime_context = _test_evaluation_context()
            runtime_context["num_cloud_hosts"] = 2
            heuristics = load_manager_heuristic_library(
                manifest,
                runtime_context=runtime_context,
            )
            self.assertFalse(heuristics[5].available)
            self.assertIn(
                "evaluation_context_mismatch:num_cloud_hosts",
                heuristics[5].availability_reason,
            )

    def test_unseen_runtime_seed_does_not_mask_same_domain_rule(self):
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as raw_directory:
            manifest = _write_test_library(
                Path(raw_directory)
            )
            train_context = _test_evaluation_context("SS")
            test_context = _test_evaluation_context("SS")
            train_context["workflow_random_seed"] = 0
            test_context["workflow_random_seed"] = 999
            train_rules = load_manager_heuristic_library(
                manifest,
                runtime_context=train_context,
            )
            test_rules = load_manager_heuristic_library(
                manifest,
                runtime_context=test_context,
            )
            np.testing.assert_array_equal(
                heuristic_availability_mask(train_rules),
                heuristic_availability_mask(test_rules),
            )
            self.assertTrue(test_rules[5].available)

    def test_ss_admission_is_available_in_ms_and_ls(self):
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as raw_directory:
            manifest = _write_test_library(
                Path(raw_directory)
            )
            for scenario in ("MS", "LS"):
                with self.subTest(scenario=scenario):
                    heuristics = load_manager_heuristic_library(
                        manifest,
                        runtime_context=(
                            _test_evaluation_context(scenario)
                        ),
                    )
                    self.assertTrue(heuristics[5].available)

    def test_resource_domain_family_ddl_and_fuzzy_mismatch_mask_rule(
        self,
    ):
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as raw_directory:
            manifest = _write_test_library(
                Path(raw_directory)
            )
            cases = {}
            resource = _test_evaluation_context("SM")
            cases["resource"] = (
                resource,
                "admission_scope_mismatch:resource_code",
            )
            workflow_family = _test_evaluation_context("SS")
            workflow_family["workflow_families"] = list(
                WORKFLOW_FAMILIES[:-1]
            )
            cases["workflow_family"] = (
                workflow_family,
                "admission_scope_mismatch:workflow_families",
            )
            ddl = _test_evaluation_context("SS")
            ddl["deadline_alpha_small_prob"] = 0.8
            cases["ddl"] = (
                ddl,
                "evaluation_context_mismatch:"
                "deadline_alpha_small_prob",
            )
            fuzzy = _test_evaluation_context("SS")
            fuzzy["fuzzy_delta1"] = 0.8
            cases["fuzzy"] = (
                fuzzy,
                "evaluation_context_mismatch:fuzzy_delta1",
            )
            for name, (context, reason) in cases.items():
                with self.subTest(name=name):
                    heuristic = load_manager_heuristic_library(
                        manifest,
                        runtime_context=context,
                    )[5]
                    self.assertFalse(heuristic.available)
                    self.assertIn(
                        reason,
                        heuristic.availability_reason,
                    )


class HeuristicSelectionManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        )
        self.manifest = _write_test_library(
            Path(self.temporary_directory.name)
        )
        self.environment = _make_environment(
            manager_mode=HEURISTIC_SELECTION_MODE,
            manifest_path=self.manifest,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_manager_selects_traditional_heuristic_rule(self):
        environment = self.environment
        # SJF 是动作 1，排序应直接使用历史五维特征的第 1 列。
        environment.apply_manager_heuristic(1)
        ids, features = (
            environment._compute_task_heuristics_for_ready()
        )
        expected = [
            int(ids[index])
            for index in np.lexsort(
                (
                    np.asarray(ids, dtype=np.int64),
                    -features[:, 1],
                )
            ).tolist()
        ]
        environment._phase_prepare_tasks()
        self.assertEqual(environment._phase_tasks, expected)
        self.assertEqual(
            environment._manager_heuristic_identity()[
                "heuristic_source"
            ],
            "traditional",
        )

    def test_manager_selects_llm_rule_for_ready_task_ordering(self):
        environment = self.environment
        llm_index = 5
        environment.apply_manager_heuristic(llm_index)
        ready_ids = list(environment.ready_task_ids)
        task_features = environment.build_task_features(
            ready_ids
        )
        expected = [
            int(ready_ids[index])
            for index in np.lexsort(
                (
                    np.asarray(ready_ids, dtype=np.int64),
                    task_features["min_exec_time"],
                )
            ).tolist()
        ]
        environment._phase_prepare_tasks()
        self.assertEqual(environment._phase_tasks, expected)
        identity = environment._manager_heuristic_identity()
        self.assertEqual(
            identity["selected_heuristic_id"],
            "test_llm_safe",
        )
        self.assertEqual(
            identity["heuristic_source"],
            "seevo_llm",
        )
        self.assertTrue(
            identity["llm_rule_version"].startswith(
                "seevo.iter1.ind0."
            )
        )

    def test_llm_rule_does_not_participate_in_host_or_vm_selection(self):
        environment = self.environment
        legacy_environment = _make_environment()
        self.assertEqual(
            environment.host_act_dim,
            legacy_environment.host_act_dim,
        )
        self.assertEqual(
            environment.vm_act_dim,
            legacy_environment.vm_act_dim,
        )
        environment.host_select = mock.Mock()
        environment.get_vm_state_for_current_task = mock.Mock()
        environment.apply_manager_heuristic(5)
        environment._phase_prepare_tasks()
        environment.host_select.assert_not_called()
        environment.get_vm_state_for_current_task.assert_not_called()
        self.assertEqual(
            environment.manager_heuristics[5]
            .priority_rule.__name__,
            "get_task_priority_v2",
        )

    def test_heuristic_selection_is_reproducible(self):
        environment = self.environment
        environment.apply_manager_heuristic(5)
        environment._phase_prepare_tasks()
        first = list(environment._phase_tasks)
        environment._phase_prepare_tasks()
        second = list(environment._phase_tasks)
        self.assertEqual(first, second)

    def test_train_and_unseen_test_seed_keep_mask_and_action_schema(self):
        environment = self.environment
        self.assertEqual(
            environment.get_manager_action_mask()[5],
            1.0,
        )
        test_environment = _make_environment(
            manager_mode=HEURISTIC_SELECTION_MODE,
            manifest_path=self.manifest,
            scenario="LS",
            random_seed=999,
        )
        np.testing.assert_array_equal(
            environment.get_manager_action_mask(),
            test_environment.get_manager_action_mask(),
        )
        self.assertEqual(
            environment.manager_heuristic_schema_version,
            test_environment.manager_heuristic_schema_version,
        )
        test_environment.apply_manager_heuristic(5)

    def test_manager_state_contains_risk_rule_metrics_and_mask(self):
        environment = self.environment
        environment.apply_manager_heuristic(5)
        environment._record_selected_heuristic_phase(
            performance_reward=-0.25,
            safety_cost=0.0,
            shield_intervention_rate=0.5,
        )
        state = environment.get_manager_state()
        schema = environment.get_observation_schema("manager")
        self.assertEqual(state.shape, (environment.manager_obs_dim,))
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertEqual(schema["action_type"], "heuristic_index")
        self.assertEqual(schema["heuristic_count"], 7)
        self.assertEqual(
            [
                feature["name"]
                for feature in schema["heuristic_features"]
            ],
            [
                "recent_energy_performance",
                "recent_safety_performance",
                "recent_shield_intervention_rate",
                "heuristic_available",
            ],
        )
        rule_blocks = state[
            environment.manager_system_obs_dim
            + environment.manager_safety_feature_dim :
        ].reshape(7, 4)
        np.testing.assert_array_equal(
            rule_blocks[:, 3],
            environment.get_manager_action_mask(),
        )
        self.assertAlmostEqual(rule_blocks[5, 1], 1.0)
        self.assertAlmostEqual(rule_blocks[5, 2], 0.5)

    def test_audit_records_ordering_and_subsequent_intervention(self):
        environment = self.environment
        environment.apply_manager_heuristic(5)
        environment._phase_prepare_tasks()
        audit = environment._manager_heuristic_audit_info(
            [
                {
                    "shield_intervened": True,
                    "action_modified": True,
                    "fallback_triggered": False,
                    "layer": "vm",
                }
            ]
        )
        self.assertEqual(
            audit["selected_heuristic_id"],
            "test_llm_safe",
        )
        self.assertEqual(
            audit["ready_task_ordering"],
            environment._phase_tasks,
        )
        self.assertEqual(
            audit["heuristic_shield_intervention_count"],
            1,
        )
        self.assertEqual(
            len(audit["subsequent_safety_interventions"]),
            1,
        )

    def test_real_phase_records_selected_rule_and_shield_outcomes(self):
        environment = self.environment
        environment.apply_manager_heuristic(5)
        while True:
            host_state, has_host = (
                environment.get_host_state_for_next_assignment()
            )
            if not has_host:
                break
            host_mask = np.asarray(
                host_state["final_action_mask"]
            )
            host_action = (
                int(np.flatnonzero(host_mask > 0.5)[0])
                if np.any(host_mask > 0.5)
                else int(host_state["fallback_action"])
            )
            environment.host_select(host_action)
            vm_state, has_vm = (
                environment.get_vm_state_for_current_task()
            )
            self.assertTrue(has_vm)
            vm_mask = np.asarray(
                vm_state["final_action_mask"]
            )
            vm_action = (
                int(np.flatnonzero(vm_mask > 0.5)[0])
                if np.any(vm_mask > 0.5)
                else int(vm_state["fallback_action"])
            )
            environment.vm_assign(vm_action)

        _, phase_info = environment.finish_phase_and_advance()
        self.assertEqual(
            phase_info["selected_heuristic_id"],
            "test_llm_safe",
        )
        self.assertEqual(
            phase_info["heuristic_source"],
            "seevo_llm",
        )
        self.assertTrue(phase_info["ready_task_ordering"])
        self.assertIn(
            "subsequent_safety_interventions",
            phase_info,
        )
        self.assertGreaterEqual(
            phase_info["heuristic_phase_safety_cost"],
            0.0,
        )
        history = environment._heuristic_recent_metrics[
            "test_llm_safe"
        ]
        self.assertEqual(len(history["performance_reward"]), 1)
        self.assertEqual(len(history["safety_cost"]), 1)


class LegacyManagerModeTests(unittest.TestCase):
    def test_resource_code_selects_default_manifest_and_override(self):
        for resource_code in ("S", "M", "L"):
            selected = Path(
                resolve_manager_heuristic_manifest(resource_code)
            )
            self.assertEqual(
                selected.name,
                "safe_heuristic_library_"
                f"res{resource_code}.json",
            )
        self.assertEqual(
            Path(
                resolve_manager_heuristic_manifest(
                    "S",
                    str(DEFAULT_LIBRARY),
                )
            ),
            DEFAULT_LIBRARY.resolve(),
        )

    def test_legacy_mode_keeps_weight_delta_action_semantics(self):
        self.assertEqual(
            SafeRLConfig().manager_heuristics.mode,
            LEGACY_RULE_WEIGHT_MODE,
        )
        config = SafeManagerHeuristicConfig()
        self.assertEqual(config.mode, LEGACY_RULE_WEIGHT_MODE)
        environment = _make_environment()
        before = environment.manager_raw_w.copy()
        mask = environment.get_manager_action_mask()
        self.assertEqual(mask.shape, (243,))
        environment.apply_manager_action(121)
        # 动作 121 是全零 delta，精确保留当前五维权重。
        np.testing.assert_array_equal(
            environment.manager_raw_w,
            before,
        )
        self.assertEqual(
            environment.get_manager_state().shape,
            (15,),
        )

    def test_heuristic_mode_requires_full_safe_stack(self):
        with self.assertRaisesRegex(
            ValueError,
            "safe_rl_enabled",
        ):
            HrlFcfsCacheEnv(
                dax_paths=[
                    str(
                        PROJECT_ROOT
                        / "data"
                        / "dax"
                        / "Montage_25.xml"
                    )
                ],
                deadline_mode="none",
                workflows_per_episode=1,
                manager_mode=HEURISTIC_SELECTION_MODE,
                manager_heuristic_library_path=DEFAULT_LIBRARY,
            )


class LlmOnlyHeuristicTests(unittest.TestCase):
    """验证 --llm-only-heuristics 真正移除传统规则动作槽。"""

    def test_traditional_slots_are_removed_not_just_masked(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = _write_test_library(Path(td))
            full = load_manager_heuristic_library(manifest)
            llm_only = load_manager_heuristic_library(
                manifest, include_traditional=False
            )

        # 完整库含五个传统槽；LLM-only 库里一个都不剩。
        self.assertEqual(len(full) - len(llm_only), 5)
        self.assertTrue(
            all(item.is_llm_rule for item in llm_only)
        )
        self.assertTrue(
            any(item.source == "traditional" for item in full)
        )
        # 不是“留槽 + mask”：传统 ID 必须彻底消失，
        # 否则任何绕过可用性 mask 的路径仍能选到传统规则。
        self.assertEqual(
            {item.heuristic_id for item in llm_only}
            & {"traditional_fcfs", "traditional_edf"},
            set(),
        )

    def test_action_dim_and_schema_version_follow_the_flag(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = _write_test_library(Path(td))
            full_env = _make_environment(
                manager_mode=HEURISTIC_SELECTION_MODE,
                manifest_path=str(manifest),
            )
            llm_env = _make_environment(
                manager_mode=HEURISTIC_SELECTION_MODE,
                manifest_path=str(manifest),
                llm_only=True,
            )

        self.assertEqual(
            full_env.manager_action_dim - llm_env.manager_action_dim,
            5,
        )
        self.assertEqual(
            llm_env.manager_action_dim,
            len(llm_env.manager_heuristics),
        )
        # 动作集合变了，schema 版本必须跟着变，
        # 这样与旧 checkpoint 的不兼容是显式失败而不是静默错位。
        self.assertNotEqual(
            full_env.manager_heuristic_schema_version,
            llm_env.manager_heuristic_schema_version,
        )
        # 起始动作必须落在某个被准入的 LLM 规则上。
        self.assertTrue(
            llm_env.manager_heuristics[
                llm_env.selected_heuristic_index
            ].is_llm_rule
        )

    def test_library_without_any_llm_rule_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = _write_test_library(Path(td))
            payload = json.loads(
                manifest.read_text(encoding="utf-8")
            )
            payload["llm_rules"] = []
            manifest.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            # 传统规则不再兜底，空动作空间必须当场报错。
            with self.assertRaisesRegex(
                ValueError,
                "LLM-only Manager heuristic library is empty",
            ):
                load_manager_heuristic_library(
                    manifest, include_traditional=False
                )
            # 同一个 manifest 在默认模式下仍然可用，
            # 说明报错来自开关本身而不是 manifest 损坏。
            self.assertEqual(
                len(load_manager_heuristic_library(manifest)), 5
            )

    def test_llm_only_requires_heuristic_selection_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            "llm_only requires heuristic_selection_mode",
        ):
            SafeManagerHeuristicConfig(
                mode="legacy_rule_weight_mode",
                llm_only=True,
            )

    def test_run_name_separates_the_two_variants(self):
        from hrl_mix.train_config import build_train_config

        def _run_name(llm_only):
            with mock.patch("hrl_mix.train_config.os.makedirs"):
                return build_train_config(
                    scenario="SS",
                    ddl="T",
                    max_episodes=1,
                    safe_rl_enabled=True,
                    safe_rl_shield_enabled=True,
                    safe_rl_state_enabled=True,
                    safe_rl_heuristic_manager_enabled=True,
                    manager_heuristic_llm_only=llm_only,
                ).run_name

        # 两种配置必须写到不同目录，否则会互相覆盖且事后无法分辨。
        self.assertNotEqual(_run_name(False), _run_name(True))
        # 关闭时的 run_name 不受本次改动影响：payload 只在开启时才新增该键，
        # 因此既有 safe_rl 运行目录保持稳定。
        self.assertEqual(_run_name(False), _run_name(False))

    def test_llm_only_requires_the_heuristic_manager(self):
        from hrl_mix.train_config import build_train_config

        with self.assertRaisesRegex(
            ValueError,
            "manager_heuristic_llm_only=True requires",
        ):
            build_train_config(
                scenario="SS",
                ddl="T",
                max_episodes=1,
                safe_rl_enabled=True,
                manager_heuristic_llm_only=True,
            )


if __name__ == "__main__":
    unittest.main()
