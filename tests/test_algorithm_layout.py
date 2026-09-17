"""Regression tests for the algorithm/comparison directory split."""

from __future__ import annotations

from pathlib import Path
import unittest

import base.hrl_env
import baseline_fcfs.env_fcfs
import hrl_mix.train_config
import LLM.problems.cews_task_constructive.eval

from algorithms.comparisons.fcfs import train_fcfs
from algorithms.llm_safe_hrl.paths import (
    ALGORITHM_ROOT,
    BASE_ROOT,
    COMPARISON_ROOT,
    HRL_ROOT,
    LLM_ROOT,
    PROJECT_ROOT,
)


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


class AlgorithmLayoutTests(unittest.TestCase):
    def test_current_method_modules_resolve_inside_algorithm_directory(self):
        modules_and_roots = (
            (base.hrl_env, BASE_ROOT),
            (hrl_mix.train_config, HRL_ROOT),
            (
                LLM.problems.cews_task_constructive.eval,
                LLM_ROOT,
            ),
        )
        for module, expected_root in modules_and_roots:
            with self.subTest(module=module.__name__):
                self.assertTrue(
                    _is_below(Path(module.__file__), expected_root)
                )

    def test_fcfs_resolves_inside_comparison_directory(self):
        self.assertTrue(
            _is_below(
                Path(baseline_fcfs.env_fcfs.__file__),
                COMPARISON_ROOT / "fcfs",
            )
        )
        self.assertEqual(Path(train_fcfs.ROOT_DIR), PROJECT_ROOT)

    def test_root_compatibility_packages_contain_no_business_modules(self):
        for package_name in ("base", "hrl_mix", "LLM", "baseline_fcfs"):
            package_root = PROJECT_ROOT / package_name
            python_files = sorted(
                path.name for path in package_root.glob("*.py")
            )
            with self.subTest(package=package_name):
                self.assertEqual(python_files, ["__init__.py"])

    def test_shared_data_and_outputs_remain_project_level(self):
        self.assertTrue((PROJECT_ROOT / "data").is_dir())
        self.assertTrue((PROJECT_ROOT / "common").is_dir())
        self.assertEqual(
            hrl_mix.train_config.ROOT_DIR,
            PROJECT_ROOT,
        )
