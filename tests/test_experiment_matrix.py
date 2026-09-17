# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from algorithms.llm_safe_hrl.paths import HRL_ROOT
from algorithms.llm_safe_hrl.scenario_registry import SCENARIO_REGISTRY
from hrl_mix.experiment_matrix import (
    GET_TASK_PRIORITY_V2_INPUTS,
    REQUIRED_ABLATION_IDS,
    REQUIRED_METHOD_IDS,
    build_experiment_manifest,
    load_experiment_matrix,
    write_experiment_manifest,
)
from LLM.problems.cews_task_constructive.reference import (
    get_task_priority_v2,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    HRL_ROOT
    / "config"
    / "safe_hrl_experiment_matrix.json"
)


class ExperimentMatrixTests(unittest.TestCase):
    def _source_payload(self):
        with CONFIG.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _temporary_config(self, payload):
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=".test_experiment_matrix_",
            dir=CONFIG.parent,
            delete=False,
        )
        try:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
        finally:
            handle.close()
        self.addCleanup(
            lambda: Path(handle.name).unlink(missing_ok=True)
        )
        return Path(handle.name)

    def test_required_methods_ablations_and_fuzzy_constants(self):
        matrix = load_experiment_matrix(CONFIG)
        self.assertEqual(
            set(matrix["methods"]),
            set(REQUIRED_METHOD_IDS),
        )
        self.assertEqual(
            set(matrix["ablations"]),
            set(REQUIRED_ABLATION_IDS),
        )
        self.assertEqual(
            matrix["methods"]["fcfs_fcfs"]["display_name"],
            "FCFS-FCFS",
        )
        self.assertEqual(
            matrix["methods"]["fcfs_fixed"]["display_name"],
            "FCFS-Fixed",
        )
        fuzzy = matrix["shared_protocol"]["fuzzy"]
        self.assertEqual(fuzzy["energy_lambda"], 1.0)
        self.assertEqual(fuzzy["deadline_eta"], 0.95)
        self.assertIs(
            fuzzy["deadline_is_deterministic"],
            True,
        )
        self.assertEqual(
            matrix["shared_protocol"]["resource_topology"],
            SCENARIO_REGISTRY["SS"].resource_mapping(),
        )

    def test_all_runs_share_fixtures_evaluation_and_metrics(self):
        manifest = build_experiment_manifest(CONFIG)
        self.assertEqual(len(manifest["runs"]), 20)
        self.assertEqual(
            manifest["generator"]["module"],
            "hrl_mix.experiment_matrix",
        )
        self.assertEqual(
            len(manifest["generator"]["code_sha256"]),
            64,
        )
        protocol_hashes = {
            run["shared_protocol_sha256"]
            for run in manifest["runs"]
        }
        self.assertEqual(
            protocol_hashes,
            {manifest["shared_protocol_sha256"]},
        )
        evaluation_ids = {
            tuple(run["evaluation_case_ids"])
            for run in manifest["runs"]
        }
        self.assertEqual(len(evaluation_ids), 1)
        self.assertTrue(
            manifest["smoke_validation"][
                "all_runs_share_evaluation_cases"
            ]
        )
        fields = manifest["shared_protocol"][
            "evaluation_metrics"
        ]["fields"]
        self.assertEqual(len(fields), len(set(fields)))
        for artifact in manifest["artifact_integrity"]:
            self.assertEqual(len(artifact["sha256"]), 64)

    def test_case_fixtures_are_deterministic_and_splits_disjoint(self):
        first = build_experiment_manifest(CONFIG)
        second = build_experiment_manifest(CONFIG)
        self.assertEqual(
            first["manifest_sha256"],
            second["manifest_sha256"],
        )
        self.assertEqual(
            first["case_fixtures"],
            second["case_fixtures"],
        )
        split_seeds = {}
        for split, fixtures in first["case_fixtures"].items():
            split_seeds[split] = {
                value
                for fixture in fixtures
                for value in (
                    fixture["episode_seed"],
                    fixture["workflow_seed"],
                    fixture["arrival_seed"],
                    fixture["resource_seed"],
                )
            }
            for fixture in fixtures:
                self.assertEqual(
                    len(fixture["arrival_times"]),
                    50,
                )
                self.assertEqual(
                    len(fixture["dax_sequence"]),
                    50,
                )
                self.assertEqual(
                    len(fixture["fixture_sha256"]),
                    64,
                )
        self.assertFalse(
            split_seeds["training"]
            & split_seeds["validation"]
        )
        self.assertFalse(
            split_seeds["training"]
            & split_seeds["final_test"]
        )
        self.assertFalse(
            split_seeds["validation"]
            & split_seeds["final_test"]
        )

    def test_llm_contract_is_task_ordering_only(self):
        matrix = load_experiment_matrix(CONFIG)
        self.assertEqual(
            tuple(
                inspect.signature(
                    get_task_priority_v2
                ).parameters
            ),
            GET_TASK_PRIORITY_V2_INPUTS,
        )
        for contract in matrix["information_contracts"].values():
            llm_inputs = tuple(contract["llm_rule_inputs"])
            if llm_inputs:
                self.assertEqual(
                    llm_inputs,
                    GET_TASK_PRIORITY_V2_INPUTS,
                )
                self.assertEqual(
                    contract["llm_rule_outputs"],
                    ["ready_task_priority_scores"],
                )
            for value in (
                contract["host_inputs"]
                + contract["vm_inputs"]
            ):
                self.assertNotIn("llm", str(value).lower())
        for run in build_experiment_manifest(CONFIG)["runs"]:
            self.assertIs(run["llm_host_vm_access"], False)
            self.assertIs(
                run[
                    "uses_information_outside_declared_contract"
                ],
                False,
            )

    def test_unavailable_artifacts_and_adapters_fail_closed(self):
        manifest = build_experiment_manifest(CONFIG, smoke=True)
        runs = {
            run["run_id"]: run for run in manifest["runs"]
        }
        self.assertEqual(
            manifest["admitted_llm_heuristic_ids"],
            [],
        )
        self.assertEqual(
            load_experiment_matrix(CONFIG)["heuristic_library_status"],
            "missing_formal",
        )
        self.assertIn(
            "out/main_single/SS/safe_heuristic_library_single_SS.json",
            load_experiment_matrix(CONFIG)["expected_heuristic_library_path"],
        )
        self.assertTrue(runs["original_hrl"]["execution_ready"])
        self.assertTrue(runs["fuzzy_irws"]["execution_ready"])
        self.assertTrue(runs["fuzzy_marl"]["execution_ready"])
        self.assertTrue(runs["fuzzy_pd3qn"]["execution_ready"])
        self.assertTrue(runs["fcfs_fcfs"]["execution_ready"])
        self.assertTrue(runs["fcfs_fixed"]["execution_ready"])
        self.assertFalse(
            runs["safe_hrl_without_llm"]["execution_ready"]
        )
        self.assertIn(
            "execution_adapter_not_implemented:"
            "safe_hrl_without_llm_adapter",
            runs["safe_hrl_without_llm"][
                "blocking_reasons"
            ],
        )
        self.assertFalse(
            runs["original_hrl_plus_llm"]["execution_ready"]
        )
        self.assertIn(
            "execution_adapter_not_implemented:"
            "legacy_heuristic_manager_adapter",
            runs["original_hrl_plus_llm"][
                "blocking_reasons"
            ],
        )
        self.assertIn(
            "no_admitted_llm_heuristic",
            runs["llm_augmented_safe_hrl"][
                "blocking_reasons"
            ],
        )
        self.assertFalse(
            runs["edf_baseline"]["execution_ready"]
        )

    def test_each_ablation_documents_its_effective_change(self):
        matrix = load_experiment_matrix(CONFIG)
        for ablation_id in REQUIRED_ABLATION_IDS:
            ablation = matrix["ablations"][ablation_id]
            base = matrix["methods"][
                ablation["base_method_id"]
            ]["components"]
            effective = ablation["effective_components"]
            for change in ablation["disabled_components"]:
                component = change["component"]
                self.assertEqual(
                    base[component],
                    change["from"],
                )
                self.assertEqual(
                    effective[component],
                    change["to"],
                )
                self.assertTrue(change["effect"])

        no_shield = matrix["ablations"][
            "no_safety_shield"
        ]["effective_components"]
        self.assertFalse(no_shield["safety_shield"])
        self.assertFalse(no_shield["fallback_controller"])
        no_qc = matrix["ablations"][
            "no_safety_value_network"
        ]["effective_components"]
        self.assertFalse(no_qc["safety_value_network"])
        self.assertEqual(
            no_qc["lagrange_mode"],
            "performance_only",
        )
        fixed = matrix["ablations"]["fixed_lambda"][
            "effective_components"
        ]
        self.assertEqual(fixed["lagrange_mode"], "fixed")
        self.assertEqual(fixed["lambda_value"], 1.0)

    def test_single_seed_changes_training_only(self):
        manifest = build_experiment_manifest(CONFIG)
        runs = {
            run["run_id"]: run for run in manifest["runs"]
        }
        baseline = runs["llm_augmented_safe_hrl"]
        single = runs["single_seed_training"]
        self.assertEqual(len(baseline["training_case_ids"]), 5)
        self.assertEqual(len(single["training_case_ids"]), 1)
        self.assertEqual(
            single["training_case_ids"],
            baseline["training_case_ids"][:1],
        )
        self.assertEqual(
            single["evaluation_case_ids"],
            baseline["evaluation_case_ids"],
        )
        self.assertEqual(
            single["intentional_protocol_deviation"],
            "training_seed_subset_only",
        )

    def test_tiny_smoke_resolves_without_training(self):
        manifest = build_experiment_manifest(
            CONFIG,
            smoke=True,
        )
        self.assertEqual(manifest["profile_kind"], "smoke")
        self.assertEqual(
            manifest["active_workflows_per_episode"],
            2,
        )
        for fixtures in manifest["case_fixtures"].values():
            self.assertEqual(len(fixtures), 1)
            self.assertEqual(
                len(fixtures[0]["arrival_times"]),
                2,
            )
        smoke = manifest["smoke_validation"]
        self.assertTrue(
            smoke["configuration_resolution_passed"]
        )
        self.assertFalse(smoke["training_executed"])
        self.assertFalse(smoke["formal_experiment_executed"])

    def test_manifest_round_trip(self):
        manifest = build_experiment_manifest(
            CONFIG,
            smoke=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            returned = write_experiment_manifest(
                manifest,
                output,
            )
            self.assertEqual(Path(returned), output.resolve())
            with output.open("r", encoding="utf-8") as handle:
                restored = json.load(handle)
        self.assertEqual(restored, manifest)

    def test_rejects_changed_fuzzy_semantics(self):
        payload = self._source_payload()
        payload["shared_protocol"]["fuzzy"][
            "deadline_eta"
        ] = 0.9
        with self.assertRaisesRegex(ValueError, "eta"):
            load_experiment_matrix(
                self._temporary_config(payload)
            )

    def test_rejects_seed_coupling_or_split_leakage(self):
        payload = self._source_payload()
        payload["shared_protocol"]["case_splits"][
            "training"
        ][0]["arrival_seed"] = 999
        with self.assertRaisesRegex(
            ValueError,
            "episode_seed == workflow_seed == arrival_seed",
        ):
            load_experiment_matrix(
                self._temporary_config(payload)
            )

        payload = self._source_payload()
        payload["shared_protocol"]["case_splits"][
            "validation"
        ][0] = copy.deepcopy(
            payload["shared_protocol"]["case_splits"][
                "training"
            ][0]
        )
        payload["shared_protocol"]["case_splits"][
            "validation"
        ][0]["case_id"] = "validation_leak"
        with self.assertRaisesRegex(ValueError, "seed leakage"):
            load_experiment_matrix(
                self._temporary_config(payload)
            )

    def test_rejects_shared_overrides_and_resource_llm_leakage(self):
        payload = self._source_payload()
        payload["methods"][0]["shared_overrides"] = {
            "ddl_setting": "Loose"
        }
        with self.assertRaisesRegex(
            ValueError,
            "may not override shared protocol",
        ):
            load_experiment_matrix(
                self._temporary_config(payload)
            )

        payload = self._source_payload()
        payload["information_contracts"][
            "legacy_hrl_with_task_rule"
        ]["vm_inputs"].append("llm_source_code")
        with self.assertRaisesRegex(
            ValueError,
            "leaks LLM information",
        ):
            load_experiment_matrix(
                self._temporary_config(payload)
            )

    def test_rejects_mismatched_ablation_documentation(self):
        payload = self._source_payload()
        change = payload["ablations"][0][
            "disabled_components"
        ][0]
        change["to"] = True
        with self.assertRaisesRegex(
            ValueError,
            "does not match effective override",
        ):
            load_experiment_matrix(
                self._temporary_config(payload)
            )


if __name__ == "__main__":
    unittest.main()
