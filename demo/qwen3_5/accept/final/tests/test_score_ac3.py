#!/usr/bin/env python3
"""End-to-end tests for ``final/score_ac3.py``.  No GPU.

The inputs are the COMMITTED artifacts wherever real data exists:
  * ``reference/reference_outputs.json``  -- the real reference, real top-k margins,
    real decoded text (used as the reference continuation text);
  * ``results/dumps_final/bs<N>.json``    -- the real committed engine tokens, laid
    out as a cold-rep tree exactly the way ``harness/gate_ac3_stable.sh`` writes it;
  * ``opt/m4i0/results/gateA.json``       -- a real fingerprint gate report, fed in
    as the stability-diagnostics input.
Only the perplexity NUMBERS are synthetic (they need the HF model on a GPU); the
tests state which value they inject and why.

The scenario list deliberately includes must-FAIL and must-NOT-EVALUATE cases:
a scorer that only ever passes is not a gate.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1]
ACC = FINAL.parent
SCORER = FINAL / "score_ac3.py"
REF_PATH = ACC / "reference" / "reference_outputs.json"
DUMPS = ACC / "results" / "dumps_final"
GATE_REPORT = ACC / "opt" / "m4i0" / "results" / "gateA.json"
REF = json.loads(REF_PATH.read_text())["results"]
BSS = (1, 2, 4, 8, 16)


def sha_ids(ids):
    import hashlib
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def build_tree(root: Path, *, batch_sizes=BSS, reps=3, mutate=None, errors=()):
    """A rep tree in gate_ac3_stable.sh's on-disk shape, tokens from dumps_final."""
    reps_root = root / "sweep" / "reps"
    for bs in batch_sizes:
        base = json.loads((DUMPS / f"bs{bs}.json").read_text())
        for r in range(1, reps + 1):
            tag = f"bs{bs}_r{r}"
            d = reps_root / tag
            d.mkdir(parents=True, exist_ok=True)
            if (bs, r) in errors:
                (d / f"meta_{tag}.json").write_text(json.dumps(
                    {"tag": tag, "status": "error", "rc": 134,
                     "note": "rep process exited rc=134"}))
                continue
            dump = {pid: {"token_ids": list(v["token_ids"])}
                    for pid, v in base.items()}
            if mutate:
                mutate(bs, r, dump)
            (d / f"bs{bs}.json").write_text(json.dumps(dump, indent=2))
            (d / f"meta_{tag}.json").write_text(json.dumps(
                {"tag": tag, "status": "ok", "bs": bs, "rep": r, "secs": 300.0,
                 "n_waves": 2, "dump_md5": "fixture",
                 "device": {"phys_index": 6, "uuid": "GPU-fixture"},
                 "gpu_before": {"memory_used_mib": 5, "utilization_pct": 0}}))
    return reps_root


def build_coherence(root: Path, reps_root: Path, *, ppl_ratio=1.0,
                    text_override=None, ppl_override=None):
    """coherence_inputs.json in hf_score.py's schema.

    Reference text is the committed ``decoded_output`` (real).  Reference ppl is
    fixed at 10.0 and engine ppl at ``10.0 * ppl_ratio``: AC-3(a) is a RATIO test,
    so the absolute value is irrelevant and only the ratio is under test here.
    """
    doc = {"schema": "final/coherence_inputs/v1", "model_id": "fixture",
           "revision": "fixture", "tokenizer_vocab_len": 248044,
           "reference": {}, "engine": []}
    for pid, e in REF.items():
        doc["reference"][pid] = {"n": 64, "nll_sum": 0.0, "ppl": 10.0,
                                 "token_ids_sha256": sha_ids(e["output_ids"]),
                                 "text": e["decoded_output"]}
    seen = set()
    for d in sorted(reps_root.glob("bs*_r*")):
        bs = int(d.name.split("_r", 1)[0][2:])
        f = d / f"bs{bs}.json"
        if not f.exists():
            continue
        for pid, e in json.loads(f.read_text()).items():
            key = (bs, pid, sha_ids(e["token_ids"]))
            if key in seen:
                continue
            seen.add(key)
            text = REF[pid]["decoded_output"]
            ppl = 10.0 * ppl_ratio
            if text_override and (bs, pid) in text_override:
                text = text_override[(bs, pid)]
            if ppl_override and (bs, pid) in ppl_override:
                ppl = ppl_override[(bs, pid)]
            doc["engine"].append({"batch_size": bs, "prompt_id": pid,
                                  "token_ids_sha256": key[2], "reps": [d.name],
                                  "n": len(e["token_ids"]), "nll_sum": 0.0,
                                  "ppl": ppl, "invalid_token_ids": [],
                                  "text": text})
    p = root / "coherence_inputs.json"
    p.write_text(json.dumps(doc, indent=2))
    return p


def run_scorer(root: Path, reps_root: Path, coherence, *, batch_sizes=BSS,
               reps_required=3, mechanisms=None, gate_report=True):
    out = root / "ac3_score.json"
    cmd = [sys.executable, str(SCORER), "--reference", str(REF_PATH),
           "--reps-root", str(reps_root), "--baseline", str(DUMPS),
           "--batch-sizes", ",".join(str(b) for b in batch_sizes),
           "--reps-required", str(reps_required), "--output-json", str(out),
           "--mechanisms", str(mechanisms or (FINAL / "mechanisms.json"))]
    if coherence:
        cmd += ["--coherence", str(coherence)]
    if gate_report and GATE_REPORT.exists():
        cmd += ["--gate-report", str(GATE_REPORT)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p, (json.loads(out.read_text()) if out.exists() else None)


class ScoreAc3Test(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="final_ac3_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------------------------------------------------------------- PASS --
    def test_committed_state_passes_the_repinned_ac3(self):
        """The committed dumps_final state: p06-poem differs at 60 (an exact
        reference tie, margin 0.0) and cascades through 61-63.  Under the
        re-pinned AC-3 that is a PASS: agreement 60/64 = 0.9375 >= 0.90, the first
        divergence is a near-tie, the rest is the accounted cascade."""
        reps_root = build_tree(self.tmp)
        coh = build_coherence(self.tmp, reps_root)
        p, rep = run_scorer(self.tmp, reps_root, coh)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(rep["verdict"], "PASS")
        case = next(c for c in rep["per_bs"]["4"]["reps"][0]["cases"]
                    if c["prompt_id"] == "p06-poem")
        self.assertAlmostEqual(case["checks"]["agreement_floor"]["agreement"],
                               60 / 64, places=6)
        classes = case["checks"]["all_differences_accounted"]["classes"]
        self.assertEqual(classes["near_tie_reference_margin"], 1)
        self.assertEqual(classes["post_divergence_cascade"], 3)
        self.assertEqual(classes["unexplained_divergence"], 0)
        # AC-3(c): every case byte-identical to dumps_final -> zero degradation
        ex = rep["diagnostics"]["exactness"]
        self.assertEqual(ex["cases_degraded"], 0)
        self.assertEqual(ex["cases_compared"], 5 * 3 * 10)
        # the real m4i0 gate report is carried through as the stability diagnostic
        self.assertIn("fingerprint_divergence_rate",
                      rep["diagnostics"]["stability"]["totals"])

    # ------------------------------------------------------ NOT EVALUABLE --
    def test_missing_coherence_inputs_is_not_a_pass(self):
        reps_root = build_tree(self.tmp)
        p, rep = run_scorer(self.tmp, reps_root, None)
        self.assertEqual(p.returncode, 3)
        self.assertEqual(rep["verdict"], "NOT_EVALUABLE")
        self.assertTrue(any("coherence_inputs_missing" in x
                            for x in rep["not_evaluable"]))

    def test_under_quorum_is_not_a_pass(self):
        reps_root = build_tree(self.tmp, batch_sizes=(1,), reps=1)
        coh = build_coherence(self.tmp, reps_root)
        p, rep = run_scorer(self.tmp, reps_root, coh, batch_sizes=(1,))
        self.assertEqual(p.returncode, 3)
        self.assertTrue(any("scored rep" in x for x in rep["not_evaluable"]))

    def test_run_error_rep_is_counted_not_dropped(self):
        reps_root = build_tree(self.tmp, batch_sizes=(1,), reps=3,
                               errors={(1, 2)})
        coh = build_coherence(self.tmp, reps_root)
        p, rep = run_scorer(self.tmp, reps_root, coh, batch_sizes=(1,))
        self.assertEqual(rep["per_bs"]["1"]["reps_error"], 1)
        self.assertEqual(rep["per_bs"]["1"]["reps_scored"], 2)
        self.assertEqual(p.returncode, 3)          # 2 < 3 required

    # ---------------------------------------------------------------- FAIL --
    def test_unexplained_first_divergence_fails_and_blocks_the_cascade(self):
        def mut(bs, r, dump):
            if bs == 4 and r == 1:
                # position 30 of p01-history: a wide-margin reference position
                dump["p01-history"]["token_ids"][30] = 999
                dump["p01-history"]["token_ids"][31] = 998
        reps_root = build_tree(self.tmp, batch_sizes=(4,), mutate=mut)
        coh = build_coherence(self.tmp, reps_root)
        p, rep = run_scorer(self.tmp, reps_root, coh, batch_sizes=(4,))
        self.assertEqual(p.returncode, 1)
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertTrue(any("unexplained_differing_positions" in f
                            for f in rep["failures"]))
        case = next(c for c in rep["per_bs"]["4"]["reps"][0]["cases"]
                    if c["prompt_id"] == "p01-history")
        acc = case["checks"]["all_differences_accounted"]
        self.assertFalse(acc["first_divergence_accounted"])
        self.assertEqual(acc["n_unexplained"], 2)   # cascade NOT granted

    def test_below_agreement_floor_fails(self):
        def mut(bs, r, dump):
            for i in range(10, 20):
                dump["p02-math"]["token_ids"][i] = 500 + i
        reps_root = build_tree(self.tmp, batch_sizes=(1,), mutate=mut)
        coh = build_coherence(self.tmp, reps_root)
        p, rep = run_scorer(self.tmp, reps_root, coh, batch_sizes=(1,))
        self.assertEqual(p.returncode, 1)
        self.assertTrue(any("agreement_below_floor" in f for f in rep["failures"]))

    def test_degenerate_repetition_fails(self):
        def mut(bs, r, dump):
            dump["p03-python"]["token_ids"] = [11, 12, 13, 14] * 16
        reps_root = build_tree(self.tmp, batch_sizes=(1,), mutate=mut)
        coh = build_coherence(self.tmp, reps_root)
        p, rep = run_scorer(self.tmp, reps_root, coh, batch_sizes=(1,))
        self.assertEqual(p.returncode, 1)
        self.assertTrue(any("degenerate_repetition" in f for f in rep["failures"]))

    def test_perplexity_above_1_5x_fails(self):
        reps_root = build_tree(self.tmp, batch_sizes=(1,))
        coh = build_coherence(self.tmp, reps_root,
                              ppl_override={(1, "p05-cuda"): 15.01})
        p, rep = run_scorer(self.tmp, reps_root, coh, batch_sizes=(1,))
        self.assertEqual(p.returncode, 1)
        self.assertTrue(any("perplexity_above_1.5x" in f for f in rep["failures"]))

    def test_byte_soup_fails(self):
        reps_root = build_tree(self.tmp, batch_sizes=(1,))
        coh = build_coherence(self.tmp, reps_root,
                              text_override={(1, "p09-translate"): "��\x00 soup"})
        p, rep = run_scorer(self.tmp, reps_root, coh, batch_sizes=(1,))
        self.assertEqual(p.returncode, 1)
        self.assertTrue(any("non_language_byte_soup" in f for f in rep["failures"]))

    def test_wrong_generated_length_fails(self):
        def mut(bs, r, dump):
            dump["p10-logic"]["token_ids"] = dump["p10-logic"]["token_ids"][:63]
        reps_root = build_tree(self.tmp, batch_sizes=(1,), mutate=mut)
        coh = build_coherence(self.tmp, reps_root)
        p, rep = run_scorer(self.tmp, reps_root, coh, batch_sizes=(1,))
        self.assertEqual(p.returncode, 1)
        self.assertTrue(any("wrong_generated_length" in f for f in rep["failures"]))

    # --------------------------------------------------------- AC-3(c) ------
    def test_a_passing_but_degraded_case_is_reported_not_absorbed(self):
        """A near-tie flip at a margin-0 reference position: still PASSES the
        re-pinned AC-3, but it is NO LONGER byte-identical to dumps_final, so
        AC-3(c) must list it as a degradation and mark it unexplained."""
        pid, pos = self._find_zero_margin_position()
        def mut(bs, r, dump):
            dump[pid]["token_ids"][pos] = REF[pid]["top2_id_per_step"][pos]
        reps_root = build_tree(self.tmp, batch_sizes=(1,), mutate=mut)
        coh = build_coherence(self.tmp, reps_root)
        p, rep = run_scorer(self.tmp, reps_root, coh, batch_sizes=(1,))
        ex = rep["diagnostics"]["exactness"]
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertEqual(rep["verdict"], "PASS")
        self.assertEqual(ex["cases_degraded"], 3)          # one per rep
        self.assertEqual(ex["degradations_unexplained"], 3)
        self.assertTrue(all(d["still_passes_repinned_ac3"]
                            for d in ex["degradations"]))

    @staticmethod
    def _find_zero_margin_position():
        for pid, e in sorted(REF.items()):
            for i in range(64):
                m = e["top1_logit_per_step"][i] - e["top2_logit_per_step"][i]
                if m == 0.0 and e["output_ids"][i] == e["topk_ids_per_step"][i][0]:
                    return pid, i
        raise unittest.SkipTest("no exact-tie position in the reference")


if __name__ == "__main__":
    unittest.main(verbosity=2)
