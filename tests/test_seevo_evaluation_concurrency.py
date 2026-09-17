"""SeEvo 结构级并发准备与共享评价池的语义回归测试。

一代里每个个体各跑一次 CMA-ES，而单批 CMA-ES 只有
`population_size × scenarios × stage_seeds` 个上下文，填不满
`max_parallel_evaluations` 个槽位。因此结构准备改成并发，评价子进程改投同一个
共享池统一限流。本测试锁住三条不能被并发破坏的性质：

1. **重复结构判定仍按顺序**：同代出现相同 structure_hash 时，必须是先出现的
   保留、后出现的作废。这一步留在串行阶段，并发化会翻转谁被判无效。
2. **失败隔离**：单个候选准备失败只影响它自己，其余个体照常进入评价。
3. **共享池是全进程上限**：所有批次投同一个池，并发的批次加起来也不会超过
   `max_parallel_evaluations` 个同时运行的评价。
"""

from __future__ import annotations

import sys
import threading
import time
from types import MethodType
import unittest

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from rule_optimization import OptimizerConfig
from seevo import SeEvo


class _StubCandidate:
    def __init__(self, structure_hash: str) -> None:
        self.structure_hash = structure_hash


def _algorithm(max_parallel_evaluations: int = 4) -> SeEvo:
    algorithm = object.__new__(SeEvo)
    algorithm.parameter_optimizer_config = OptimizerConfig(
        enabled=True,
        max_parallel_evaluations=max_parallel_evaluations,
    )
    algorithm.iteration = 0
    return algorithm


class DuplicateStructureOrderTests(unittest.TestCase):
    """并发准备之前的串行门必须保持"先到先得"。"""

    def test_later_duplicate_is_the_one_marked_invalid(self):
        algorithm = _algorithm()
        # mode 不是 train 时 evaluate_population 结尾不触发反事实分析。
        algorithm.mode = "eval"
        handed_to_prepare = []

        def fake_parse(self, individual):
            del self
            return _StubCandidate(individual["structure_hash"])

        def fake_prepare_all(self, population, response_ids):
            del self, population
            handed_to_prepare.extend(response_ids)

        algorithm._parse_rule_candidate = MethodType(fake_parse, algorithm)
        # 并发准备整体打桩：本用例只验证它之前那道串行门的判定。
        # 没有个体被标记 rule_prepared，第三阶段因此不会启动任何子进程。
        algorithm._prepare_individuals_concurrently = MethodType(
            fake_prepare_all, algorithm
        )

        population = [
            {"code": "x", "structure_hash": "a" * 64},
            {"code": "x", "structure_hash": "b" * 64},
            {"code": "x", "structure_hash": "a" * 64},
            {"code": None, "structure_hash": "d" * 64},
        ]
        algorithm.evaluate_population(population, [1])

        # 先出现的 0 号保留，重复的 2 号作废；无代码的 3 号也不进入准备。
        self.assertEqual(handed_to_prepare, [0, 1])
        self.assertEqual(population[2]["duplicate_structure_hash"], "a" * 64)
        self.assertEqual(population[2]["obj"], float("inf"))
        self.assertFalse(population[2]["exec_success"])
        self.assertNotIn("duplicate_structure_hash", population[0])


class ConcurrentPrepareTests(unittest.TestCase):
    def test_one_failure_does_not_stop_the_other_structures(self):
        algorithm = _algorithm()
        population = [{"tag": index} for index in range(4)]

        def fake_prepare(self, individual):
            del self
            if individual["tag"] == 2:
                raise ValueError("boom")
            individual["rule_prepared"] = True
            return individual

        algorithm._prepare_individual_for_evaluation = MethodType(
            fake_prepare, algorithm
        )
        algorithm._prepare_individuals_concurrently(population, [0, 1, 2, 3])

        for index in (0, 1, 3):
            self.assertTrue(population[index]["rule_prepared"])
        self.assertFalse(population[2].get("rule_prepared", False))
        self.assertIn("Rule preparation failed", population[2]["traceback_msg"])
        self.assertEqual(population[2]["obj"], float("inf"))

    def test_prepared_individuals_keep_their_own_slot(self):
        algorithm = _algorithm()
        population = [{"tag": index} for index in range(6)]

        def fake_prepare(self, individual):
            del self
            # 打乱完成顺序，验证结果按下标写回而不是按完成顺序。
            time.sleep(0.01 * ((7 - individual["tag"]) % 4))
            individual["rule_prepared"] = True
            individual["seen_tag"] = individual["tag"]
            return individual

        algorithm._prepare_individual_for_evaluation = MethodType(
            fake_prepare, algorithm
        )
        algorithm._prepare_individuals_concurrently(
            population, list(range(6))
        )
        self.assertEqual(
            [individual["seen_tag"] for individual in population],
            list(range(6)),
        )


class SharedEvaluationPoolTests(unittest.TestCase):
    def test_pool_is_reused_and_caps_process_wide_concurrency(self):
        algorithm = _algorithm(max_parallel_evaluations=3)
        try:
            first = algorithm._shared_evaluation_executor()
            self.assertIs(first, algorithm._shared_evaluation_executor())

            running = 0
            peak = 0
            guard = threading.Lock()

            def task():
                nonlocal running, peak
                with guard:
                    running += 1
                    peak = max(peak, running)
                time.sleep(0.05)
                with guard:
                    running -= 1

            futures = [first.submit(task) for _ in range(12)]
            for future in futures:
                future.result()
            # 12 个任务投给同一个 3 槽的池，无论来自几个批次都不会超订。
            self.assertLessEqual(peak, 3)
        finally:
            algorithm._shutdown_evaluation_executor()
        self.assertIsNone(algorithm._evaluation_executor)


if __name__ == "__main__":
    unittest.main()
