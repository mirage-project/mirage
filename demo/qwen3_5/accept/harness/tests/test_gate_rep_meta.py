#!/usr/bin/env python3
"""Regression tests for two defects that made harness/gate_ac3_stable.py unable
to produce a single rep at HEAD.  No GPU: both are reachable without CUDA.

1. ``cmd_rep``'s meta dict referred to a bare ``cap`` after 348a601a moved cap
   resolution into ``admission_policy.py`` -> ``NameError`` on EVERY rep.  The
   name is gone, so the test asserts the function body no longer references it
   and that the replacement expression evaluates.
2. ``_assertion`` formatted a ``None`` divergence rate with ``%``-precision ->
   ``TypeError`` while trying to report the failure that caused it.
"""
import ast
import inspect
import sys
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS.parent))

import admission_policy                                  # noqa: E402
import gate_ac3_stable as g                              # noqa: E402


class CmdRepMetaTest(unittest.TestCase):
    def test_no_undefined_cap_reference(self):
        src = inspect.getsource(g.cmd_rep)
        tree = ast.parse(src.lstrip())
        assigned = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        self.assertNotIn("cap", used - assigned,
                         "cmd_rep reads a name it never binds -- this is the "
                         "NameError that killed every rep")

    def test_the_replacement_expression_works(self):
        self.assertIsNone(admission_policy.resolve_int("policy", g.AC3_MBT, 1))
        self.assertEqual(admission_policy.resolve_int("policy", g.AC3_MBT, 4), 4)
        self.assertIn("cap_mode", admission_policy.summary())


class AssertionFormattingTest(unittest.TestCase):
    report = {"per_bs": {"1": {}, "2": {}}}

    def test_none_rate_does_not_raise(self):
        for rc in (0, 1, 2):
            s = g._assertion(self.report, 3, None, rc)
            self.assertIsInstance(s, str)
            self.assertTrue(s)

    def test_real_rate_still_renders_as_a_percentage(self):
        self.assertIn("4.2%", g._assertion(self.report, 3, 0.0417, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
