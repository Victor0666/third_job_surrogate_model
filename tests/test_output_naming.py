"""Regression tests for compact names below the project ``out`` tree."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hrl_mix.train_config import build_train_config
from output_naming import (
    MAX_OUTPUT_COMPONENT_LENGTH,
    eval_identity_from_filename,
    hrl_evaluation_csv_path,
    legacy_hrl_run_id,
    pipeline_run_id,
    resolve_hrl_checkpoint_dir,
    safe_hrl_run_id,
    training_output_paths,
)
from project_paths import PROJECT_ROOT


class OutputNamingTests(unittest.TestCase):
    def test_training_ids_are_short_and_deterministic(self):
        self.assertEqual(
            legacy_hrl_run_id("SS", "Tight", seed=1),
            "hrl-ss-t-s1",
        )
        first = safe_hrl_run_id(
            "SM",
            "M",
            {"shield": True, "lambda": "dynamic"},
        )
        second = safe_hrl_run_id(
            "SM",
            "Medium",
            {"lambda": "dynamic", "shield": True},
        )
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("safe-sm-m-"))
        self.assertLessEqual(len(first), MAX_OUTPUT_COMPONENT_LENGTH)
        self.assertTrue(
            pipeline_run_id("LL", "L", "abc").startswith(
                "pipe-ll-l-"
            )
        )

    def test_training_paths_do_not_repeat_long_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = training_output_paths(
                directory,
                "safe-ss-t-0123456789",
            )
            self.assertEqual(
                paths.checkpoint_dir.name,
                "safe-ss-t-0123456789",
            )
            self.assertEqual(paths.log_path.name, "train.csv")
            self.assertEqual(
                paths.log_path.parent.name,
                paths.checkpoint_dir.name,
            )

    def test_eval_paths_are_derived_from_short_script_identity(self):
        scenario, ddl = eval_identity_from_filename(
            "eval_ML_T.py"
        )
        self.assertEqual((scenario, ddl), ("ml", "t"))
        with tempfile.TemporaryDirectory() as directory:
            path = hrl_evaluation_csv_path(
                directory,
                scenario,
                ddl,
            )
            self.assertEqual(path.name, "ml-t.csv")
            self.assertEqual(path.parent.name, "hrl")

    def test_checkpoint_resolution_prefers_short_and_reads_exact_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = (
                Path(directory)
                / "out"
                / "ckpts"
                / "hrl-ss-t-s1"
            )
            checkpoint_dir.mkdir(parents=True)
            for filename in (
                "best_vm.pth",
                "best_host.pth",
                "best_manager.pth",
            ):
                (checkpoint_dir / filename).touch()
            self.assertEqual(
                resolve_hrl_checkpoint_dir(
                    directory,
                    "SS",
                    "T",
                ),
                checkpoint_dir,
            )
            for path in checkpoint_dir.iterdir():
                path.unlink()
            checkpoint_dir.rmdir()

            legacy_dir = (
                Path(directory)
                / "out"
                / "ckpts"
                / (
                    "ckpts_hrl_3layer_routeA_mgc_ave_"
                    "smallTask_smallRes_seed1_mgrDelayEnergy_mix_"
                    "rAlpha075_dalphaMix_HVrn_Tight"
                )
            )
            legacy_dir.mkdir(parents=True)
            for filename in (
                "best_vm.pth",
                "best_host.pth",
                "best_manager.pth",
            ):
                (legacy_dir / filename).touch()
            self.assertEqual(
                resolve_hrl_checkpoint_dir(
                    directory,
                    "SS",
                    "T",
                ),
                legacy_dir,
            )

    def test_train_config_uses_short_components_for_all_modes(self):
        with patch("hrl_mix.train_config.os.makedirs"):
            legacy = build_train_config(safe_rl_enabled=False)
            safe = build_train_config(safe_rl_enabled=True)
        for config in (legacy, safe):
            with self.subTest(run_name=config.run_name):
                self.assertLessEqual(
                    len(config.run_name),
                    MAX_OUTPUT_COMPONENT_LENGTH,
                )
                self.assertEqual(
                    Path(config.save_dir).name,
                    config.run_name,
                )
                self.assertEqual(
                    Path(config.log_path).name,
                    "train.csv",
                )

    def test_all_historical_eval_entries_use_shared_short_paths(self):
        scripts = sorted(
            (PROJECT_ROOT / "run" / "hrl_mix").glob("eval_*.py")
        )
        self.assertEqual(len(scripts), 27)
        forbidden = (
            "ckpts_hrl_3layer",
            "hrl_mix_wfInfo",
        )
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            with self.subTest(script=script.name):
                self.assertIn(
                    "resolve_hrl_checkpoint_dir(",
                    source,
                )
                self.assertIn(
                    "hrl_evaluation_csv_path(",
                    source,
                )
                for token in forbidden:
                    self.assertNotIn(token, source)
