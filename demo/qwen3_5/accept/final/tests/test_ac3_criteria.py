#!/usr/bin/env python3
"""Unit tests for the AC-3 scoring primitives.  No GPU, no torch, stdlib only.

Every threshold these tests pin is quoted from ``.pm/goal.md`` AC-3 or from a
pinned protocol document -- see ``ac3_criteria`` for the citations.  The point of
the tests is that the criteria cannot be silently loosened: change a constant and
one of these fails.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ac3_criteria import (  # noqa: E402
    AGREEMENT_FLOOR, CLASS_MECHANISM, CLASS_NEAR_TIE_ENGINE, CLASS_NEAR_TIE_REF,
    CLASS_UNEXPLAINED, agreement_verdict, bf16_ulp, classify_difference,
    non_language_counts, nonlanguage_verdict, perplexity_verdict,
    repetition_verdict, worst_ngram_repetition,
)


class Bf16UlpTest(unittest.TestCase):
    def test_spacing_is_2_pow_exp_minus_7(self):
        self.assertEqual(bf16_ulp(1.0), 2 ** -7)
        self.assertEqual(bf16_ulp(16.0), 0.125)
        self.assertEqual(bf16_ulp(31.9), 0.125)
        self.assertEqual(bf16_ulp(32.0), 0.25)
        self.assertEqual(bf16_ulp(-20.0), 0.125)

    def test_reproduces_the_documented_p10_logic_calibration(self):
        """bench-protocol.md ("Determinism protocol", rule b) records the
        p10-logic flip as 0.625 reference-side but "0.375 = 3 bf16 ULPs"
        engine-side.  Logits there are ~20, so 3 ULP must be exactly 0.375."""
        self.assertAlmostEqual(3 * bf16_ulp(20.0), 0.375, places=12)

    def test_zero_is_inside_every_budget(self):
        self.assertGreater(bf16_ulp(0.0), 0.0)


class RepetitionTest(unittest.TestCase):
    def test_no_repetition(self):
        n, gram, cnt = worst_ngram_repetition(list(range(64)))
        self.assertLessEqual(cnt, 1)
        self.assertTrue(repetition_verdict(list(range(64)))["pass"])

    def test_three_repeats_pass_four_fail(self):
        # AC-3(a): "repeated >3x" fails, so exactly 3 must pass.
        three = [1, 2, 3, 4] * 3 + list(range(100, 152))
        four = [1, 2, 3, 4] * 4 + list(range(100, 148))
        self.assertTrue(repetition_verdict(three)["pass"])
        v = repetition_verdict(four)
        self.assertFalse(v["pass"])
        self.assertGreaterEqual(v["worst_count"], 4)

    def test_catches_a_long_degenerate_tail(self):
        ids = list(range(20)) + [7, 8, 9, 10, 11] * 8
        v = repetition_verdict(ids)
        self.assertFalse(v["pass"])
        self.assertGreaterEqual(v["worst_n"], 4)

    def test_reference_derived_bar(self):
        """The bar is max(3, the reference continuation's own worst count): the
        engine may never be MORE repetitive than the pinned reference."""
        eng4 = [1, 2, 3, 4] * 4 + list(range(100, 148))
        eng5 = [1, 2, 3, 4] * 5 + list(range(100, 144))
        ref4 = [9, 8, 7, 6] * 4 + list(range(200, 248))
        ref1 = list(range(64))
        self.assertTrue(repetition_verdict(eng4, ref4)["pass"])
        self.assertFalse(repetition_verdict(eng5, ref4)["pass"])
        self.assertFalse(repetition_verdict(eng4, ref1)["pass"])
        self.assertEqual(repetition_verdict(eng4, ref1)["max_allowed_count"], 3)

    def test_the_committed_reference_would_fail_the_literal_bound(self):
        """Empirical basis for the reference term: p07-format's own reference
        continuation repeats the markdown list-item 4-gram (198,262,348,256) four
        times, so the goal's literal ">3x fails" bound rejects the HF reference
        itself.  If this ever stops being true the reference term can be dropped --
        the test is here so that decision is data-driven."""
        import json
        ref = json.loads((Path(__file__).resolve().parents[2] / "reference"
                          / "reference_outputs.json").read_text())["results"]
        ids = ref["p07-format"]["output_ids"]
        self.assertFalse(repetition_verdict(ids)["pass"])
        self.assertTrue(repetition_verdict(ids, ids)["pass"])

    def test_trigram_repetition_is_not_a_violation(self):
        # n >= 4 only: a repeated 3-gram is ordinary language ("the the the" is
        # not what AC-3(a) targets), so it must not fail the gate.
        ids = [5, 6, 7] * 8 + list(range(40, 80))
        v = repetition_verdict(ids)
        self.assertGreaterEqual(v["worst_count"], 4)  # the 4-gram 5,6,7,5 repeats
        # sanity: a sequence with a repeated 3-gram but no repeated 4-gram passes
        ids2 = [1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3, 7, 1, 2, 3, 6] + list(range(50, 90))
        self.assertTrue(repetition_verdict(ids2)["pass"])


class NonLanguageTest(unittest.TestCase):
    def test_clean_text_has_no_nonlanguage(self):
        c = non_language_counts("Hello, world!\nSecond line\t— 中文 ok.")
        self.assertEqual(c["total_nonlanguage"], 0)

    def test_replacement_and_control_chars_counted(self):
        c = non_language_counts("ab�c\x00\x07d")
        self.assertEqual(c["replacement_chars"], 1)
        self.assertEqual(c["control_chars"], 2)

    def test_bar_is_the_reference_continuations_own_count(self):
        self.assertTrue(nonlanguage_verdict("clean text", "clean reference")["pass"])
        self.assertFalse(nonlanguage_verdict("so�up", "clean reference")["pass"])
        # a reference that itself carries one control char tolerates one
        self.assertTrue(nonlanguage_verdict("a\x00b", "ref\x00")["pass"])
        self.assertFalse(nonlanguage_verdict("a\x00b\x01", "ref\x00")["pass"])


class PerplexityTest(unittest.TestCase):
    def test_ratio_bound_is_1_5(self):
        self.assertTrue(perplexity_verdict(1.5, 1.0)["pass"])
        self.assertFalse(perplexity_verdict(1.5001, 1.0)["pass"])

    def test_missing_numbers_are_not_a_pass(self):
        v = perplexity_verdict(None, 1.0)
        self.assertFalse(v["pass"])
        self.assertFalse(v["available"])


class AgreementTest(unittest.TestCase):
    def test_floor_is_90_percent_inclusive(self):
        self.assertEqual(AGREEMENT_FLOOR, 0.90)
        self.assertTrue(agreement_verdict(64, 58)["pass"])       # 0.90625
        self.assertFalse(agreement_verdict(64, 57)["pass"])      # 0.890625
        self.assertTrue(agreement_verdict(10, 9)["pass"])        # exactly 0.90

    def test_empty_is_not_a_pass(self):
        self.assertFalse(agreement_verdict(0, 0)["pass"])


class ClassifyDifferenceTest(unittest.TestCase):
    base = dict(prompt_id="p06-poem", batch_size=4, position=60,
                ref_top1_id=31000, ref_top2_id=81316, engine_id=40581)

    def test_reference_near_tie(self):
        r = classify_difference(**self.base, ref_margin=0.0)
        self.assertEqual(r["classification"], CLASS_NEAR_TIE_REF)
        self.assertTrue(r["accounted"])

    def test_wide_reference_margin_without_evidence_is_unexplained(self):
        r = classify_difference(**self.base, ref_margin=9.0)
        self.assertEqual(r["classification"], CLASS_UNEXPLAINED)
        self.assertFalse(r["accounted"])

    def test_engine_side_three_ulps(self):
        r = classify_difference(**self.base, ref_margin=9.0,
                                engine_margin=0.375, engine_margin_ref_logit=20.0)
        self.assertEqual(r["classification"], CLASS_NEAR_TIE_ENGINE)
        r2 = classify_difference(**self.base, ref_margin=9.0,
                                 engine_margin=0.5, engine_margin_ref_logit=20.0)
        self.assertEqual(r2["classification"], CLASS_UNEXPLAINED)

    def test_mechanism_entry_must_match_exactly(self):
        m = [{"id": "M1", "prompt_id": "p06-poem", "position": 60,
              "batch_sizes": [4], "ref_top1_id": 31000, "engine_id": 40581,
              "mechanism": "documented reordering", "evidence": "opt/x"}]
        r = classify_difference(**self.base, ref_margin=9.0, mechanisms=m)
        self.assertEqual(r["classification"], CLASS_MECHANISM)
        for bad in ({"position": 61}, {"batch_sizes": [8]}, {"engine_id": 1},
                    {"ref_top1_id": 2}, {"prompt_id": "p01-history"},
                    {"mechanism": "   "}):
            mm = [dict(m[0], **bad)]
            r = classify_difference(**self.base, ref_margin=9.0, mechanisms=mm)
            self.assertEqual(r["classification"], CLASS_UNEXPLAINED,
                             f"a mechanism entry differing in {bad} must not match")

    def test_engine_contradiction_is_flagged_but_still_accounted(self):
        r = classify_difference(**self.base, ref_margin=0.25,
                                engine_margin=5.0, engine_margin_ref_logit=20.0)
        self.assertEqual(r["classification"], CLASS_NEAR_TIE_REF)
        self.assertTrue(r["engine_contradicts_reference_near_tie"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
