"""M3-I11 protocol amendment: `run_ac3.py --engine-dump-dir` must refuse an
ambiguous tree instead of silently scoring one dump out of many.

M3-I9 pointed the flag at a parent of twenty run directories; a single
anomalous `bs4.json` out of eighty dumps was then reported as a policy effect,
and M3-I9b had to root-cause it back to run-to-run nondeterminism. The rule the
harness now enforces: one run directory, `bs<N>.json` at its top level, every
other shape refused with the full candidate list.

Run directly: `python3 tests/test_dump_tree.py`.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_ac3 import resolve_dump_tree  # noqa: E402

BATCH_SIZES = [1, 2, 4, 8, 16]


class ResolveDumpTreeTest(unittest.TestCase):
    def test_clean_single_run_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for bs in BATCH_SIZES:
                (d / f"bs{bs}.json").write_text("{}")
            self.assertEqual(resolve_dump_tree(d, BATCH_SIZES), "")

    def test_nested_duplicate_is_refused_and_lists_both(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for bs in BATCH_SIZES:
                (d / f"bs{bs}.json").write_text("{}")
            (d / "s7_rep1").mkdir()
            (d / "s7_rep1" / "bs4.json").write_text("{}")
            err = resolve_dump_tree(d, BATCH_SIZES)
            self.assertIn("exit 2", err)
            self.assertIn("bs=4: 2 candidates", err)
            self.assertIn(str(d / "bs4.json"), err)
            self.assertIn(str(d / "s7_rep1" / "bs4.json"), err)

    def test_parent_of_run_directories_is_refused(self):
        # The exact M3-I9 shape: no dump at the top level, several underneath.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for rep in ("s7_rep1", "s7_rep2", "s7_rep3"):
                (d / rep).mkdir()
                (d / rep / "bs4.json").write_text("{}")
            err = resolve_dump_tree(d, [4])
            self.assertIn("bs=4: 3 candidates", err)

    def test_single_nested_candidate_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "s0_base").mkdir()
            (d / "s0_base" / "bs16.json").write_text("{}")
            err = resolve_dump_tree(d, [16])
            self.assertIn("nested, not at the top level", err)

    def test_absent_dump_is_not_an_ambiguity_error(self):
        # Missing files stay the existing NOT-APPLICABLE (exit 3) path; this
        # check only rejects trees that offer a CHOICE.
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(resolve_dump_tree(Path(td), BATCH_SIZES), "")


if __name__ == "__main__":
    unittest.main()
