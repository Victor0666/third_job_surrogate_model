"""锁住"评估返回值元数"这一条不变量。

``_evaluate_training_scenarios`` 在 ``return_safety_metrics`` 为真时返回 5 元组、
为假时返回 4 元组。训练循环里有三处必须跟着同一个判据走：

1. 验证轮请求指标时传的 ``return_safety_metrics``；
2. 非验证轮构造的占位元组；
3. 解包 ``eval_result`` 的分支。

历史上第 1 处比第 3 处多了 ``or metric_store is not None``，而第 2 处无条件构造
5 元组，于是纯 legacy 配置（安全开关全关、无课程控制器）下每个非验证 episode
都会抛 ``ValueError: too many values to unpack (expected 4)``。本测试从 AST 上
确认三处引用的是同一个布尔名字，并单测占位构造函数的元数。
"""

from __future__ import annotations

import ast
import inspect
import sys
import unittest

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from hrl_mix import train_runner

_UNPACK_NAMES = [
    "eval_vm",
    "eval_host",
    "eval_mgr",
    "eval_energy",
    "eval_safety",
]


def _train_ast():
    module = ast.parse(inspect.getsource(train_runner))
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            return node
    raise AssertionError("train() not found in train_runner")


def _safety_unpack_guard(train_node):
    """返回守卫 5 元组解包的那个 ``if`` 判据节点。"""
    for node in ast.walk(train_node):
        if not isinstance(node, ast.If) or not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Assign):
            continue
        target = first.targets[0]
        if not isinstance(target, ast.Tuple):
            continue
        if not isinstance(first.value, ast.Name):
            continue
        if first.value.id != "eval_result":
            continue
        names = [
            element.id
            for element in target.elts
            if isinstance(element, ast.Name)
        ]
        if names == _UNPACK_NAMES:
            return node.test
    raise AssertionError("the 5-tuple eval_result unpack was not found")


def _keyword_value(train_node, callee, keyword):
    """收集对 ``callee`` 的调用里 ``keyword`` 的实参节点。"""
    found = []
    for node in ast.walk(train_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else getattr(func, "attr", None)
        )
        if name != callee:
            continue
        for kw in node.keywords:
            if kw.arg == keyword:
                found.append(kw.value)
    return found


class SafetyMetricArityGuardTests(unittest.TestCase):
    def setUp(self):
        self.train_node = _train_ast()

    def test_unpack_guard_is_a_single_boolean_name(self):
        guard = _safety_unpack_guard(self.train_node)
        # 内联的 or 链是当年分叉的根源，判据必须是一个具名布尔量。
        self.assertIsInstance(
            guard,
            ast.Name,
            "eval_result 解包必须由单个具名布尔量守卫，不能内联 or 链",
        )

    def test_all_three_sites_reference_the_same_boolean(self):
        guard = _safety_unpack_guard(self.train_node)
        self.assertIsInstance(guard, ast.Name)

        requested = _keyword_value(
            self.train_node,
            "_evaluate_training_scenarios",
            "return_safety_metrics",
        )
        # 课程评估恒为 True，只有 source 场景那次跟着判据走。
        variable_requests = [
            node for node in requested if isinstance(node, ast.Name)
        ]
        self.assertTrue(
            variable_requests,
            "至少要有一次验证评估按判据请求安全指标",
        )
        for node in variable_requests:
            self.assertEqual(node.id, guard.id)

        placeholder = _keyword_value(
            self.train_node,
            "_skipped_validation_result",
            "return_safety_metrics",
        )
        self.assertEqual(
            len(placeholder),
            1,
            "非验证轮的占位值只应有一处构造",
        )
        self.assertIsInstance(placeholder[0], ast.Name)
        self.assertEqual(placeholder[0].id, guard.id)


class SkippedValidationResultTests(unittest.TestCase):
    def test_legacy_mode_returns_four_values(self):
        result = train_runner._skipped_validation_result(
            123.5, return_safety_metrics=False
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(result, (0.0, 0.0, 0.0, 123.5))

    def test_safe_mode_returns_five_values(self):
        result = train_runner._skipped_validation_result(
            123.5, return_safety_metrics=True
        )
        self.assertEqual(len(result), 5)
        self.assertEqual(result[:4], (0.0, 0.0, 0.0, 123.5))
        self.assertEqual(result[4]["fuzzy_energy_score"], 123.5)
        # 跳过的 episode 没有真正评估过，不得被当成"已通过"。
        self.assertFalse(result[4]["zero_violation_pass"])
        self.assertFalse(result[4]["all_seed_evaluation_completed"])
        self.assertEqual(result[4]["validation_seed_count"], 0)


if __name__ == "__main__":
    unittest.main()
