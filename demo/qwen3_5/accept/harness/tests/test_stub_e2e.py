"""Stub/integration tests for the AC-3 harness — required by M2-I1's acceptance: "stub-tested
against the vLLM smoke outputs before MPK exists" + "synthetic mismatch/tie fixtures". MPK does
not exist yet, so these are the only correctness evidence available for this issue; every case
here is either (a) real committed data (reference_outputs.json, vllm_smoke_result.json) or (b)
an explicit synthetic fixture under fixtures/, never fabricated inline. Run directly:
`python3 tests/test_stub_e2e.py`.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac3_runner import run_ac3  # noqa: E402
from ac3_types import EngineSequence, PositionRecord, TieVerdict, WaiverRequest  # noqa: E402
from engine_adapter import JSONDumpAdapter, StaticMappingAdapter, load_vllm_smoke  # noqa: E402
from reference_loader import load_reference  # noqa: E402

HARNESS_DIR = Path(__file__).resolve().parent.parent
ACCEPT_DIR = HARNESS_DIR.parent
REAL_REFERENCE = ACCEPT_DIR / "reference" / "reference_outputs.json"
REAL_VLLM_SMOKE = ACCEPT_DIR / "reference" / "vllm_smoke" / "vllm_smoke_result.json"
SYNTHETIC_REFERENCE = HARNESS_DIR / "fixtures" / "synthetic_reference.json"
SYNTHETIC_ENGINE_DUMP = HARNESS_DIR / "fixtures" / "synthetic_engine_dump.json"


class RealVllmSmokeStubTest(unittest.TestCase):
    """The exact required stub test: replay the committed vLLM smoke artifact (real engine,
    real reference, one prompt, bs=1 only) through the harness."""

    @classmethod
    def setUpClass(cls):
        cls.references = load_reference(REAL_REFERENCE)
        cls.engine_map = load_vllm_smoke(REAL_VLLM_SMOKE)

    def test_vllm_smoke_matches_reference_in_partial_mode(self):
        self.assertEqual(list(self.engine_map.keys()), ["p01-history"])
        adapter = StaticMappingAdapter({1: self.engine_map})

        report = run_ac3(
            adapter=adapter,
            references=self.references,
            batch_sizes=[1],
            prompt_ids=["p01-history"],
            allow_partial=True,
        )

        self.assertEqual(report.status, "partial_smoke_only")
        self.assertTrue(report.overall_pass)
        self.assertEqual(len(report.prompt_results), 1)
        result = report.prompt_results[0]
        self.assertEqual(result.prompt_id, "p01-history")
        self.assertTrue(result.passed)
        self.assertEqual(len(result.positions), 64)
        self.assertTrue(all(p.match for p in result.positions))
        self.assertEqual(result.waiver_request, None)

        # Confirms the documented gap: the real committed reference has no top-2/margin data,
        # so this is honestly reported as unavailable rather than a fabricated value.
        self.assertFalse(report.margin_evidence["available"])

    def test_non_partial_mode_hard_fails_on_the_9_missing_prompts(self):
        # Without --allow-partial, the vLLM smoke artifact's single prompt is not a valid AC-3
        # run: the other 9 pinned prompts are simply absent, and that must fail loudly, not be
        # silently treated as a pass on the one prompt that happens to be present.
        adapter = StaticMappingAdapter({1: self.engine_map})
        report = run_ac3(
            adapter=adapter,
            references=self.references,
            batch_sizes=[1],
            prompt_ids=list(self.references.keys()),
            allow_partial=False,
        )
        self.assertEqual(report.status, "fail")
        self.assertFalse(report.overall_pass)
        missing_prompt_results = [r for r in report.prompt_results if r.prompt_id != "p01-history"]
        self.assertTrue(all(not r.passed for r in missing_prompt_results))
        self.assertTrue(
            all(p.verdict == TieVerdict.ENGINE_TOO_SHORT.value for r in missing_prompt_results for p in r.positions)
        )


class FullSweepPerfectEngineTest(unittest.TestCase):
    """Proves the orchestration itself (10 prompts x {1,2,4,8,16}) is correct by echoing the
    reference's own output_ids back as a 'perfect' engine - not a claim about any real engine,
    just a test that the harness's full sweep machinery produces a clean 0/0 mismatch result
    when given a trivially-correct one."""

    def test_perfect_engine_passes_the_full_sweep(self):
        references = load_reference(REAL_REFERENCE)
        batch_sizes = [1, 2, 4, 8, 16]
        mapping = {
            bs: {pid: EngineSequence(token_ids=list(r.output_ids)) for pid, r in references.items()}
            for bs in batch_sizes
        }
        adapter = StaticMappingAdapter(mapping)

        report = run_ac3(adapter=adapter, references=references, batch_sizes=batch_sizes)

        self.assertEqual(report.status, "pass")
        self.assertTrue(report.overall_pass)
        self.assertEqual(len(report.prompt_results), 10 * 5)
        self.assertEqual(sum(len(r.positions) for r in report.prompt_results), 10 * 5 * 64)
        self.assertEqual(report.waiver_requests, [])

    def test_bs_dependent_regression_is_caught_against_the_single_reference(self):
        # Covers "padding/order/state-reset" per the issue contract: the same fixed reference
        # is checked at every batch size, so a batching-induced corruption at one bs (but not
        # another) for the SAME prompt surfaces directly as a mismatch, with no separate
        # padding/state machinery needed in the harness itself.
        references = load_reference(REAL_REFERENCE)
        pref = references["p01-history"]
        correct = EngineSequence(token_ids=list(pref.output_ids))
        corrupted_ids = list(pref.output_ids)
        corrupted_ids[10] = 424242  # simulate cross-request state leakage at bs=2
        corrupted = EngineSequence(token_ids=corrupted_ids)

        adapter = StaticMappingAdapter({1: {"p01-history": correct}, 2: {"p01-history": corrupted}})
        report = run_ac3(
            adapter=adapter, references=references, batch_sizes=[1, 2], prompt_ids=["p01-history"]
        )

        self.assertFalse(report.overall_pass)
        by_bs = {r.batch_size: r for r in report.prompt_results}
        self.assertTrue(by_bs[1].passed)
        self.assertFalse(by_bs[2].passed)
        self.assertEqual(by_bs[2].first_divergent_position, 10)


class SyntheticFixtureTest(unittest.TestCase):
    """Exercises every tie_classifier branch through the real reference-loading +
    orchestration path (not just the pure unit test), using the hand-authored fixtures under
    fixtures/ so each scenario is independently inspectable as data."""

    @classmethod
    def setUpClass(cls):
        cls.references = load_reference(SYNTHETIC_REFERENCE)
        cls.adapter = JSONDumpAdapter({1: SYNTHETIC_ENGINE_DUMP})

    def _run_one(self, prompt_id):
        report = run_ac3(
            adapter=self.adapter, references=self.references, batch_sizes=[1], prompt_ids=[prompt_id]
        )
        self.assertEqual(len(report.prompt_results), 1)
        return report, report.prompt_results[0]

    def test_widebug_wrong_id_not_top2(self):
        report, result = self._run_one("syn-widebug")
        self.assertFalse(result.passed)
        self.assertEqual(result.first_divergent_position, 1)
        self.assertEqual(result.waiver_request.classifier_verdict, TieVerdict.IMPLEMENTATION_BUG.value)
        self.assertFalse(report.overall_pass)

    def test_topbut_wide_margin_is_a_bug(self):
        _, result = self._run_one("syn-topbut-wide")
        self.assertFalse(result.passed)
        self.assertEqual(result.waiver_request.classifier_verdict, TieVerdict.IMPLEMENTATION_BUG.value)

    def test_tie_candidate_never_auto_waives_the_gate(self):
        report, result = self._run_one("syn-tie")
        self.assertFalse(result.passed)  # a candidate tie-flip is still a FAILING gate
        self.assertFalse(report.overall_pass)
        w = result.waiver_request
        self.assertIsNotNone(w)
        self.assertEqual(w.classifier_verdict, TieVerdict.CANDIDATE_TIE_FLIP.value)
        self.assertFalse(w.auto_waived)
        self.assertTrue(w.needs_human_adjudication)

    def test_tie_confirmed_by_engine_logits(self):
        _, result = self._run_one("syn-tie-confirmed")
        w = result.waiver_request
        self.assertEqual(w.classifier_verdict, TieVerdict.CANDIDATE_TIE_FLIP.value)
        # The topk_logits plumbing actually round-tripped through JSON (string keys -> int).
        self.assertEqual(w.evidence.engine_logit_at_ref_top1, 2.0)
        self.assertEqual(w.evidence.engine_logit_at_ref_top2, 2.05)

    def test_tie_refuted_by_engine_logits(self):
        _, result = self._run_one("syn-tie-refuted")
        w = result.waiver_request
        self.assertEqual(w.classifier_verdict, TieVerdict.IMPLEMENTATION_BUG.value)

    def test_insufficient_evidence_mirrors_the_real_reference_gap(self):
        _, result = self._run_one("syn-insufficient")
        w = result.waiver_request
        self.assertEqual(w.classifier_verdict, TieVerdict.INSUFFICIENT_EVIDENCE.value)

    def test_post_divergence_positions_are_not_independent_evidence(self):
        _, result = self._run_one("syn-postdiv")
        self.assertEqual(result.first_divergent_position, 1)
        # Exactly one waiver request for the whole prompt, at the FIRST divergence.
        self.assertIsNotNone(result.waiver_request)
        self.assertEqual(result.waiver_request.first_divergent_position, 1)
        # Position 2 would look like its own candidate tie-flip in isolation (engine picked
        # exactly the ref's top-2 id at a narrow margin) but must be downgraded because it
        # follows position 1's divergence.
        pos2 = result.positions[2]
        self.assertEqual(pos2.verdict, TieVerdict.POST_DIVERGENCE.value)
        # The raw factual comparison is still preserved, just not classified as independent
        # evidence.
        self.assertFalse(pos2.match)


class MissingPromptSemanticsTest(unittest.TestCase):
    def setUp(self):
        self.references = load_reference(REAL_REFERENCE)
        self.some_prompt = "p02-math"
        # Adapter that simply never returns anything for `some_prompt`.
        good_echo = {
            pid: EngineSequence(token_ids=list(r.output_ids))
            for pid, r in self.references.items()
            if pid != self.some_prompt
        }
        self.adapter = StaticMappingAdapter({1: good_echo})

    def test_missing_prompt_hard_fails_by_default(self):
        report = run_ac3(
            adapter=self.adapter,
            references=self.references,
            batch_sizes=[1],
            prompt_ids=list(self.references.keys()),
            allow_partial=False,
        )
        self.assertFalse(report.overall_pass)
        missing = [r for r in report.prompt_results if r.prompt_id == self.some_prompt]
        self.assertEqual(len(missing), 1)
        self.assertFalse(missing[0].passed)
        self.assertEqual(missing[0].positions[0].verdict, TieVerdict.ENGINE_TOO_SHORT.value)

    def test_missing_prompt_skipped_in_partial_mode(self):
        report = run_ac3(
            adapter=self.adapter,
            references=self.references,
            batch_sizes=[1],
            prompt_ids=list(self.references.keys()),
            allow_partial=True,
        )
        scored_ids = {r.prompt_id for r in report.prompt_results}
        self.assertNotIn(self.some_prompt, scored_ids)
        self.assertEqual(report.status, "partial_smoke_only")


class WaiverRequestNeverAutoWaivesTest(unittest.TestCase):
    def test_constructing_an_auto_waived_request_is_rejected(self):
        dummy = PositionRecord(
            prompt_id="x", batch_size=1, position=0, ref_top1_id=1, ref_top1_logit=1.0,
            ref_top2_id=2, ref_top2_logit=0.9, margin=0.1, engine_argmax_id=2,
            engine_logit_at_ref_top1=None, engine_logit_at_ref_top2=None, match=False,
            verdict=TieVerdict.CANDIDATE_TIE_FLIP.value, is_first_divergence=True,
        )
        with self.assertRaises(ValueError):
            WaiverRequest(
                prompt_id="x", batch_size=1, first_divergent_position=0, evidence=dummy,
                classifier_verdict=TieVerdict.CANDIDATE_TIE_FLIP.value, auto_waived=True,
            )


if __name__ == "__main__":
    unittest.main()
