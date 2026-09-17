"""参数评估缓存的追加式日志与压实语义回归测试。

`EvaluationCache` 以前每次 `put`/`put_many` 都原子重写整份快照，条目数一多就
变成 O(n^2)：正式 SeEvo 运行里快照会涨到几十 MB，单次 `json.dumps` 就要半秒。
改成"快照 + 追加日志"后，本测试保证三件事：

1. 落盘语义不变——写进去的条目重新打开必须还在（无论是否已压实）；
2. 磁盘格式不变——压实后的 `<path>` 仍是完整的 `schema_version == 1` 文档，
   且日志文件被删除；
3. 崩溃安全——日志末尾被截断的半行只丢它自己，之前的条目照常恢复。
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


def _metrics(objective: float) -> dict:
    return {"objective": float(objective), "constraint_feasible": True}


class EvaluationCacheJournalTests(unittest.TestCase):
    def _paths(self, directory: str) -> tuple[Path, Path]:
        path = Path(directory) / "cache.json"
        return path, path.with_name(path.name + ".journal")

    def test_journaled_entry_survives_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path, journal = self._paths(directory)
            cache = EvaluationCache(path=path)
            cache.put(_key(0), _metrics(1.0))
            # 单次写入只追加日志，不重写快照。
            self.assertTrue(journal.is_file())
            self.assertFalse(path.is_file())

            reopened = EvaluationCache(path=path)
            self.assertEqual(reopened.get(_key(0))["objective"], 1.0)

    def test_flush_writes_complete_snapshot_and_drops_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            path, journal = self._paths(directory)
            cache = EvaluationCache(path=path)
            cache.put_many([(_key(i), _metrics(i)) for i in range(5)])
            cache.flush()

            self.assertFalse(journal.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["entries"]), 5)
            for entry in payload["entries"].values():
                self.assertEqual(entry["status"], "ok")

            reopened = EvaluationCache(path=path)
            for i in range(5):
                self.assertEqual(reopened.get(_key(i))["objective"], float(i))

    def test_flush_is_idempotent_and_cheap_when_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            path, journal = self._paths(directory)
            cache = EvaluationCache(path=path)
            cache.put(_key(0), _metrics(1.0))
            cache.flush()
            first = path.read_text(encoding="utf-8")
            cache.flush()
            self.assertEqual(path.read_text(encoding="utf-8"), first)
            self.assertFalse(journal.exists())

    def test_compaction_triggers_and_keeps_every_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path, journal = self._paths(directory)
            cache = EvaluationCache(path=path)
            count = EvaluationCache._MIN_COMPACTION_ENTRIES + 10
            for i in range(count):
                cache.put(_key(i), _metrics(i))
            # 超过阈值后至少压实过一次，快照已经存在。
            self.assertTrue(path.is_file())

            cache.flush()
            self.assertFalse(journal.exists())
            reopened = EvaluationCache(path=path)
            self.assertEqual(reopened.stats()["entries"], count)
            for i in range(count):
                self.assertEqual(reopened.get(_key(i))["objective"], float(i))

    def test_truncated_journal_tail_only_loses_its_own_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path, journal = self._paths(directory)
            cache = EvaluationCache(path=path)
            cache.put(_key(0), _metrics(1.0))
            cache.put(_key(1), _metrics(2.0))
            self.assertFalse(path.exists())

            # 模拟写日志途中掉电：最后一行只写了一半。
            text = journal.read_text(encoding="utf-8")
            journal.write_text(text[: -len(text.splitlines()[-1]) // 2], encoding="utf-8")

            reopened = EvaluationCache(path=path)
            self.assertEqual(reopened.get(_key(0))["objective"], 1.0)
            self.assertIsNone(reopened.get(_key(1)))

    def test_snapshot_and_journal_merge_on_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path, journal = self._paths(directory)
            cache = EvaluationCache(path=path)
            cache.put(_key(0), _metrics(1.0))
            cache.flush()
            cache.put(_key(1), _metrics(2.0))
            self.assertTrue(path.is_file())
            self.assertTrue(journal.is_file())

            reopened = EvaluationCache(path=path)
            self.assertEqual(reopened.get(_key(0))["objective"], 1.0)
            self.assertEqual(reopened.get(_key(1))["objective"], 2.0)

    def test_failed_and_disabled_stores_touch_no_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path, journal = self._paths(directory)
            cache = EvaluationCache(path=path)
            cache.put(_key(0), {"error": "boom"}, successful=False)
            self.assertFalse(path.exists())
            self.assertFalse(journal.exists())

            disabled = EvaluationCache(enabled=False, path=path)
            disabled.put(_key(0), _metrics(1.0))
            disabled.flush()
            self.assertFalse(path.exists())
            self.assertFalse(journal.exists())


if __name__ == "__main__":
    unittest.main()
