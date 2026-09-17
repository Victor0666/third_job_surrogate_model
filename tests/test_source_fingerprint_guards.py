"""源码指纹守卫：缓存复用必须挡住"环境代码已改"这件事。

两个缓存的历史口径都太窄，改环境代码后旧结果会被静默复用：

* DRL-EA 的 ``source_hash`` 只覆盖 ``drlea_nichgp/`` 自己，改 ``base/hrl_env.py``
  或 ``common/`` 都看不见；
* SeEvo 的 ``EvaluationCacheKey`` 根本没有代码指纹，只有*配置*哈希。

两边的处置刻意不同，本测试把这个区别也锁住：

* DRL-EA 的 RA manifest 是**只告警**——那份 checkpoint 花了约 120000 s 算力，
  硬失败等于作废它；
* SeEvo 的评估缓存是**失败即丢弃**——缓存条目重算得起，而复用一条由不同环境
  代码算出来的结果会污染全部下游适应度比较。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from algorithms.comparisons.drlea_nichgp import checkpointing
from rule_optimization import evaluation_cache as evaluation_cache_module
from rule_optimization.evaluation_cache import EvaluationCache, EvaluationCacheKey


def _key(index: int) -> EvaluationCacheKey:
    return EvaluationCacheKey.create(
        structure_hash="a" * 64,
        parameter_vector=[0.1 * index, 0.2],
        seed=1 + index,
        scenario_id="SS",
        evaluation_config_hash="b" * 64,
        resource_config_hash="c" * 64,
    )


_METRICS = {"objective": 1.0, "constraint_feasible": True}


class SimulatorFingerprintTests(unittest.TestCase):
    """``simulator_fingerprint`` 本身的取值语义。"""

    def _isolated_root(self, directory: str) -> Path:
        """搭一棵最小的假项目树，避免测试依赖真实仓库内容。"""
        root = Path(directory)
        for parts in evaluation_cache_module._SIMULATOR_TREES:
            tree = root.joinpath(*parts)
            tree.mkdir(parents=True, exist_ok=True)
            (tree / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        for parts in evaluation_cache_module._SIMULATOR_FILES:
            path = root.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("REGISTRY = {}\n", encoding="utf-8")
        self._use_root(root)
        return root

    def _use_root(self, root: Path) -> None:
        original = evaluation_cache_module._PROJECT_ROOT
        evaluation_cache_module._PROJECT_ROOT = root
        evaluation_cache_module.simulator_fingerprint.cache_clear()

        def restore():
            evaluation_cache_module._PROJECT_ROOT = original
            evaluation_cache_module.simulator_fingerprint.cache_clear()

        self.addCleanup(restore)

    def test_real_repository_fingerprint_covers_expected_trees(self):
        # 不断言具体摘要值（改代码就会变），只断言覆盖面。
        identity_roots = [
            "algorithms/llm_safe_hrl/base/",
            "algorithms/llm_safe_hrl/LLM/problems/cews_task_constructive/",
            "common/",
        ]
        root = evaluation_cache_module._PROJECT_ROOT
        for prefix in identity_roots:
            self.assertTrue(
                (root / prefix).is_dir(),
                f"指纹覆盖的目录 {prefix} 不存在，说明布局已变",
            )
        self.assertTrue(
            (root / "algorithms/llm_safe_hrl/scenario_registry.py").is_file()
        )
        self.assertEqual(len(evaluation_cache_module.simulator_fingerprint()), 64)

    def test_editing_a_fingerprinted_file_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._isolated_root(directory)
            before = evaluation_cache_module.simulator_fingerprint()
            target = root.joinpath(
                *evaluation_cache_module._SIMULATOR_TREES[0], "module.py"
            )
            target.write_text("VALUE = 2\n", encoding="utf-8")
            evaluation_cache_module.simulator_fingerprint.cache_clear()
            self.assertNotEqual(
                before, evaluation_cache_module.simulator_fingerprint()
            )

    def test_editing_the_individually_listed_file_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._isolated_root(directory)
            before = evaluation_cache_module.simulator_fingerprint()
            root.joinpath(*evaluation_cache_module._SIMULATOR_FILES[0]).write_text(
                "REGISTRY = {'SS': 1}\n", encoding="utf-8"
            )
            evaluation_cache_module.simulator_fingerprint.cache_clear()
            self.assertNotEqual(
                before, evaluation_cache_module.simulator_fingerprint()
            )

    def test_empty_layout_raises_instead_of_hashing_nothing(self):
        # 空指纹在任意两个代码库之间都相等，正是这个守卫要防的情况。
        with tempfile.TemporaryDirectory() as directory:
            self._use_root(Path(directory))
            with self.assertRaises(RuntimeError):
                evaluation_cache_module.simulator_fingerprint()


class EvaluationCacheFingerprintTests(unittest.TestCase):
    """SeEvo 评估缓存：指纹不符即丢弃（fail-closed）。"""

    def _seeded_cache(self, directory: str, *, flush: bool) -> Path:
        path = Path(directory) / "cache.json"
        cache = EvaluationCache(path=path)
        cache.put(_key(0), _METRICS)
        if flush:
            cache.flush()
        return path

    def test_unchanged_fingerprint_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._seeded_cache(directory, flush=True)
            reopened = EvaluationCache(path=path)
            self.assertIsNotNone(reopened.get(_key(0)))
            self.assertIsNone(reopened.discarded_reason)
            self.assertFalse(path.with_name(path.name + ".stale").exists())

    def test_snapshot_records_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._seeded_cache(directory, flush=True)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(
                payload["simulator_fingerprint"],
                evaluation_cache_module.simulator_fingerprint(),
            )
            self.assertEqual(
                payload["simulator_fingerprint_scheme"],
                evaluation_cache_module.SIMULATOR_FINGERPRINT_SCHEME,
            )

    def test_snapshot_with_foreign_fingerprint_is_set_aside(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._seeded_cache(directory, flush=True)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["simulator_fingerprint"] = "f" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")

            reopened = EvaluationCache(path=path)
            self.assertIsNone(reopened.get(_key(0)))
            self.assertIsNotNone(reopened.discarded_reason)
            # 丢弃是改名而非删除，事后可查。
            self.assertTrue(path.with_name(path.name + ".stale").is_file())
            self.assertFalse(path.exists())

    def test_snapshot_without_fingerprint_is_set_aside(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._seeded_cache(directory, flush=True)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.pop("simulator_fingerprint")
            path.write_text(json.dumps(payload), encoding="utf-8")

            reopened = EvaluationCache(path=path)
            self.assertIsNone(reopened.get(_key(0)))
            self.assertIn("unknown", reopened.discarded_reason)

    def test_journal_carries_its_own_fingerprint_header(self):
        # 日志在第一次压实前是独立存在的，必须自己可校验。
        with tempfile.TemporaryDirectory() as directory:
            path = self._seeded_cache(directory, flush=False)
            journal = path.with_name(path.name + ".journal")
            self.assertFalse(path.exists(), "单次 put 不应重写快照")
            header = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(
                header["header"]["simulator_fingerprint"],
                evaluation_cache_module.simulator_fingerprint(),
            )

    def test_journal_with_foreign_fingerprint_is_set_aside(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._seeded_cache(directory, flush=False)
            journal = path.with_name(path.name + ".journal")
            lines = journal.read_text(encoding="utf-8").splitlines()
            lines[0] = json.dumps(
                {"header": {"simulator_fingerprint": "f" * 64}}
            )
            journal.write_text("\n".join(lines) + "\n", encoding="utf-8")

            reopened = EvaluationCache(path=path)
            self.assertIsNone(reopened.get(_key(0)))
            self.assertTrue(journal.with_name(journal.name + ".stale").is_file())

    def test_headerless_journal_is_set_aside(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._seeded_cache(directory, flush=False)
            journal = path.with_name(path.name + ".journal")
            lines = journal.read_text(encoding="utf-8").splitlines()
            journal.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")

            reopened = EvaluationCache(path=path)
            self.assertIsNone(reopened.get(_key(0)))
            self.assertIsNotNone(reopened.discarded_reason)

    def test_stats_expose_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            stats = EvaluationCache(path=path).stats()
            self.assertEqual(
                stats["simulator_fingerprint"],
                evaluation_cache_module.simulator_fingerprint(),
            )
            self.assertIsNone(stats["discarded_reason"])


class DrleaSourceHashTests(unittest.TestCase):
    """DRL-EA 的 manifest 指纹：覆盖共享代码，但只告警不失败。"""

    def test_fingerprint_covers_shared_trees_not_only_the_package(self):
        identity = checkpointing.source_fingerprint()
        prefixes = {
            "algorithms/comparisons/drlea_nichgp/": False,
            "algorithms/llm_safe_hrl/base/": False,
            "common/": False,
        }
        for path in identity:
            for prefix in prefixes:
                if path.startswith(prefix):
                    prefixes[prefix] = True
        for prefix, seen in prefixes.items():
            self.assertTrue(seen, f"指纹未覆盖 {prefix}")
        self.assertIn(
            "algorithms/llm_safe_hrl/scenario_registry.py", identity
        )

    def test_hash_is_bound_to_the_scheme_version(self):
        current = checkpointing.source_hash()
        original = checkpointing.SOURCE_HASH_SCHEME_VERSION
        checkpointing.SOURCE_HASH_SCHEME_VERSION = original + 1
        self.addCleanup(
            setattr, checkpointing, "SOURCE_HASH_SCHEME_VERSION", original
        )
        self.assertNotEqual(current, checkpointing.source_hash())

    def test_absent_hash_is_reported_as_absent(self):
        status, _ = checkpointing.compare_recorded_source_hash({})
        self.assertEqual(status, "absent")
        status, _ = checkpointing.compare_recorded_source_hash(None)
        self.assertEqual(status, "absent")

    def test_old_scheme_is_not_reported_as_a_mismatch(self):
        # 已训练好的 stage-1 RA 记的就是这种没有 scheme 字段的旧口径。
        status, message = checkpointing.compare_recorded_source_hash(
            {"source_hash": "0" * 64}
        )
        self.assertEqual(status, "legacy_scheme")
        self.assertIn("not comparable", message)

        status, _ = checkpointing.compare_recorded_source_hash(
            {"source_hash": "0" * 64, "source_hash_scheme": 1}
        )
        self.assertEqual(status, "legacy_scheme")

    def test_same_scheme_different_hash_is_a_mismatch(self):
        status, _ = checkpointing.compare_recorded_source_hash(
            {
                "source_hash": "0" * 64,
                "source_hash_scheme": (
                    checkpointing.SOURCE_HASH_SCHEME_VERSION
                ),
            }
        )
        self.assertEqual(status, "mismatch")

    def test_current_code_matches_itself(self):
        status, _ = checkpointing.compare_recorded_source_hash(
            {
                "source_hash": checkpointing.source_hash(),
                "source_hash_scheme": (
                    checkpointing.SOURCE_HASH_SCHEME_VERSION
                ),
            }
        )
        self.assertEqual(status, "match")

    def test_warn_helper_never_raises(self):
        """整个守卫的要害：任何一种坏输入都只能降级为一个状态字符串。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "absent.json"
            self.assertEqual(
                checkpointing.warn_if_source_changed(missing, label="t"),
                "absent",
            )

            unreadable = root / "broken.json"
            unreadable.write_text("{not json", encoding="utf-8")
            self.assertEqual(
                checkpointing.warn_if_source_changed(unreadable, label="t"),
                "absent",
            )

            changed = root / "changed.json"
            checkpointing.write_json(
                changed,
                {
                    "source_hash": "0" * 64,
                    "source_hash_scheme": (
                        checkpointing.SOURCE_HASH_SCHEME_VERSION
                    ),
                },
            )
            self.assertEqual(
                checkpointing.warn_if_source_changed(changed, label="t"),
                "mismatch",
            )

            same = root / "same.json"
            checkpointing.write_json(
                same,
                {
                    "source_hash": checkpointing.source_hash(),
                    "source_hash_scheme": (
                        checkpointing.SOURCE_HASH_SCHEME_VERSION
                    ),
                },
            )
            self.assertEqual(
                checkpointing.warn_if_source_changed(same, label="t"),
                "match",
            )


if __name__ == "__main__":
    unittest.main()
