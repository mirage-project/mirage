"""Unit tests for tie_classifier.classify_position — required by M2-I1's acceptance
("unit test for the argmax-tie classifier"). Pure function, no I/O: every case is
constructed by hand so each branch in mpk-gaps.md §6.5's evidence rule is independently
verified. Run directly: `python3 tests/test_tie_classifier.py`.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac3_types import TieVerdict  # noqa: E402
from tie_classifier import classify_position  # noqa: E402


class ClassifyPositionTest(unittest.TestCase):
    def test_match(self):
        v = classify_position(ref_top1_id=10, ref_top2_id=11, margin=4.0, engine_argmax_id=10)
        self.assertEqual(v, TieVerdict.MATCH)

    def test_match_even_without_any_ref_top2_or_margin(self):
        # A position can MATCH even when the reference has no top-2/margin data at all - the
        # match test only needs the ref's top-1 id, which is always present.
        v = classify_position(ref_top1_id=10, ref_top2_id=None, margin=None, engine_argmax_id=10)
        self.assertEqual(v, TieVerdict.MATCH)

    def test_engine_too_short(self):
        v = classify_position(ref_top1_id=10, ref_top2_id=11, margin=4.0, engine_argmax_id=None)
        self.assertEqual(v, TieVerdict.ENGINE_TOO_SHORT)

    def test_insufficient_evidence_no_top2_id(self):
        # Mirrors the real committed reference_outputs.json: no top-2 id/logit at all.
        v = classify_position(ref_top1_id=10, ref_top2_id=None, margin=None, engine_argmax_id=999)
        self.assertEqual(v, TieVerdict.INSUFFICIENT_EVIDENCE)

    def test_insufficient_evidence_no_margin_even_with_top2_id(self):
        v = classify_position(ref_top1_id=10, ref_top2_id=11, margin=None, engine_argmax_id=999)
        self.assertEqual(v, TieVerdict.INSUFFICIENT_EVIDENCE)

    def test_implementation_bug_wrong_id_not_top2(self):
        # Mismatch, and the engine's pick isn't even the reference's runner-up.
        v = classify_position(ref_top1_id=10, ref_top2_id=11, margin=0.05, engine_argmax_id=999)
        self.assertEqual(v, TieVerdict.IMPLEMENTATION_BUG)

    def test_implementation_bug_top2_but_wide_margin(self):
        # Engine picked exactly the reference's runner-up, but the reference's own margin over
        # it was wide - not plausibly a numeric tie.
        v = classify_position(ref_top1_id=10, ref_top2_id=11, margin=4.0, engine_argmax_id=11)
        self.assertEqual(v, TieVerdict.IMPLEMENTATION_BUG)

    def test_candidate_tie_flip_narrow_margin_top2(self):
        v = classify_position(ref_top1_id=10, ref_top2_id=11, margin=0.05, engine_argmax_id=11)
        self.assertEqual(v, TieVerdict.CANDIDATE_TIE_FLIP)

    def test_margin_exactly_at_threshold_is_still_a_tie_candidate(self):
        # Boundary: margin == threshold should NOT be treated as "too wide" (only margin >
        # threshold is). Uses the default threshold (0.5) explicitly for clarity.
        v = classify_position(
            ref_top1_id=10, ref_top2_id=11, margin=0.5, engine_argmax_id=11,
            tie_margin_threshold=0.5,
        )
        self.assertEqual(v, TieVerdict.CANDIDATE_TIE_FLIP)

    def test_margin_just_above_threshold_is_a_bug(self):
        v = classify_position(
            ref_top1_id=10, ref_top2_id=11, margin=0.5000001, engine_argmax_id=11,
            tie_margin_threshold=0.5,
        )
        self.assertEqual(v, TieVerdict.IMPLEMENTATION_BUG)

    def test_custom_threshold_is_honored(self):
        # Same margin, tighter custom threshold flips the call.
        v = classify_position(
            ref_top1_id=10, ref_top2_id=11, margin=0.2, engine_argmax_id=11,
            tie_margin_threshold=0.1,
        )
        self.assertEqual(v, TieVerdict.IMPLEMENTATION_BUG)

    def test_candidate_tie_flip_confirmed_by_engine_logits(self):
        # Engine also reports its own logits at both reference ids, and they genuinely
        # invert (its logit at the ref's top-2 id outranks its logit at the ref's top-1 id) -
        # the stronger form of evidence.
        v = classify_position(
            ref_top1_id=10, ref_top2_id=11, margin=0.05, engine_argmax_id=11,
            engine_logit_at_ref_top1=2.0, engine_logit_at_ref_top2=2.05,
        )
        self.assertEqual(v, TieVerdict.CANDIDATE_TIE_FLIP)

    def test_engine_logits_that_contradict_inversion_force_implementation_bug(self):
        # Argmax landed on the ref's top-2 id with a narrow ref margin, but the engine's own
        # reported logits say its top-1-id logit is actually still higher than its top-2-id
        # logit - that contradicts a genuine ranking inversion, so this is NOT waved through
        # as a tie candidate on argmax equality alone.
        v = classify_position(
            ref_top1_id=10, ref_top2_id=11, margin=0.05, engine_argmax_id=11,
            engine_logit_at_ref_top1=3.0, engine_logit_at_ref_top2=1.0,
        )
        self.assertEqual(v, TieVerdict.IMPLEMENTATION_BUG)

    def test_engine_logits_tie_exactly_is_not_a_confirmed_inversion(self):
        # Equal logits are not a strict inversion (">"), so this should not be treated as
        # confirmed - falls back to IMPLEMENTATION_BUG rather than assuming the best case.
        v = classify_position(
            ref_top1_id=10, ref_top2_id=11, margin=0.05, engine_argmax_id=11,
            engine_logit_at_ref_top1=2.0, engine_logit_at_ref_top2=2.0,
        )
        self.assertEqual(v, TieVerdict.IMPLEMENTATION_BUG)


if __name__ == "__main__":
    unittest.main()
