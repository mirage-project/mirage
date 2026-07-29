#!/usr/bin/env python3
"""End-to-end tests for ``final/score_perf.py``.  No GPU, no vLLM.

The main fixture is REAL committed evidence: M3-I7's per-rep timings
(``opt/m3i7/raw_meta/perf/...``) assembled into the layout ``collect_perf.sh``
produces, at the CURRENT pinned admission-cap policy (uncapped bs1/bs2, capped
bs4/bs8/bs16), against the committed pinned vLLM baseline table.  That fixture
must reproduce M3-I7's own published decode numbers and must FAIL AC-4 -- which is
the true current state of the project.

The other scenarios are synthesized to prove the scorer cannot be talked into a
PASS: a losing candidate fails, a drifting comparator fails, a winning candidate
with an invalid measurement is NOT_EVALUABLE rather than PASS.
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
SCORER = FINAL / "score_perf.py"
BENCH = ACC / "bench_vllm.py"
PINNED = ACC / "baselines" / "vllm-0.25.1-20260725"
I7 = ACC / "opt" / "m3i7" / "raw_meta" / "perf"

# (bs) -> (full arm dir, pre arm dir) at the CURRENT pinned cap policy:
# bs1/bs2 uncapped, bs4/bs8/bs16 capped ("auto"), bench-protocol.md
# "Admission-cap policy, amended (M3-I7 milestone gate, 2026-07-29)".
ARMS = {1: ("perf_M2_full", "perf_M2_pre"),
        2: ("perf_M_full", "perf_M_pre"),
        4: ("capsweep_Mfull_bs4", "capsweep_Mpre_bs4"),
        8: ("capsweep_Mfull_bs8", "capsweep_Mpre_bs8"),
        16: ("cap16_Mfull_cap", "cap16_Mpre_cap")}
sys.path.insert(0, str(ACC))
import admission_policy as POLICY                    # noqa: E402  THE authority


def build_mpk_root(root: Path, batch_sizes, *, scale=1.0, dirty=(),
                   cap_override=None, break_wave=()):
    """Assemble a collect_perf.sh-shaped MPK root from the committed I7 timings."""
    (root / "full").mkdir(parents=True, exist_ok=True)
    (root / "pre").mkdir(parents=True, exist_ok=True)
    cells = {}
    for bs in batch_sizes:
        fdir, pdir = ARMS[bs]
        for arm, src in (("full", fdir), ("pre", pdir)):
            for rep in range(3):
                srcp = I7 / src / f"timings_bs{bs}_rep{rep}.json"
                if not srcp.exists():
                    continue
                d = json.loads(srcp.read_text())
                # The committed I7 artifacts predate the policy module, so the
                # run-recorded block is synthesized here from the SAME authority
                # the scorer resolves against -- the arms themselves were run at
                # exactly the policy's cap (uncapped bs1/bs2, capped bs4/8/16),
                # which is why these dirs were chosen.
                cap = POLICY.resolve_int(cap_override or "policy", 16, bs)
                d["admission_policy"] = dict(POLICY.summary(),
                                             requested=cap_override or "policy",
                                             compiled_cap=cap)
                if arm == "full" and scale != 1.0:
                    for w in d["waves"]:
                        w["wall_ms"] = w["wall_ms"] * scale
                if (bs, arm, rep) in break_wave:
                    d["waves"] = d["waves"] + [dict(d["waves"][0], wave=1)]
                tag = f"{arm}_bs{bs}_rep{rep}"
                (root / arm / f"timings_bs{bs}_rep{rep}.json").write_text(
                    json.dumps(d))
                cells[tag] = {"tag": tag, "arm": arm, "bs": bs, "rep": rep,
                              "status": "ok", "rc": 0, "gpu_index": 6,
                              "gpu_before_mib": (40000 if (bs, arm, rep) in dirty
                                                 else 5),
                              "gpu_after_mib": 37000,
                              "kernel_dir": f"/k/{arm}_bs{bs}", "cap_define": "",
                              "cap_flag": ""}
    (root / "audit.json").write_text(json.dumps({
        "schema": "final/perf_audit/v1",
        "device": {"phys_index": 6, "uuid": "GPU-fixture"},
        "foreign_floor_mib": 5,
        "admission_policy": POLICY.summary(), "cells": cells}, indent=2))
    return root


def build_vllm_fresh(root: Path, batch_sizes, *, decode_scale=1.0, e2e_scale=1.0,
                     source="full"):
    """A fresh-boot vLLM directory.  By default it IS the pinned capture's own
    `full` boot re-labelled, i.e. a fresh run that agrees perfectly -- the
    stand-in is stated in the test names that use it."""
    root.mkdir(parents=True, exist_ok=True)
    for bs in batch_sizes:
        d = json.loads((PINNED / source / f"bs{bs}.json").read_text())
        for r in d["reps"]:
            r["decode_tokens_per_second"] *= decode_scale
            r["e2e_wall_seconds"] *= e2e_scale
        if d.get("summary"):
            d["summary"]["dispersion_ok"] = True
        (root / f"bs{bs}.json").write_text(json.dumps(d))
    return root


def run(root: Path, mpk, fresh, batch_sizes, **kw):
    out = root / "perf_score.json"
    cmd = [sys.executable, str(SCORER), "--mpk-root", str(mpk),
           "--vllm-fresh", str(fresh), "--vllm-pinned", str(PINNED),
           "--bench-vllm", str(BENCH), "--output-json", str(out),
           "--batch-sizes", ",".join(str(b) for b in batch_sizes)]
    for k, v in kw.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p, (json.loads(out.read_text()) if out.exists() else None)


class ScorePerfTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="final_perf_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------------------------------------------------- the real fixture --
    def test_committed_i7_evidence_reproduces_the_published_numbers_and_fails_ac4(self):
        bss = (1, 2, 4, 8, 16)
        mpk = build_mpk_root(self.tmp / "mpk", bss)
        fresh = build_vllm_fresh(self.tmp / "fresh", bss)
        p, rep = run(self.tmp, mpk, fresh, bss)
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertEqual(rep["ac4"]["verdict"], "FAIL")
        # M3-I7's own published table (README 2b + cap tables), recomputed here
        # from the same per-rep artifacts through this scorer.
        expect = {1: 102.2, 2: 200.4, 16: 1342.2}
        for bs, want in expect.items():
            got = rep["per_bs"][str(bs)]["mpk"]["decode_tokens_per_second"]
            self.assertAlmostEqual(got, want, delta=1.5,
                                   msg=f"bs{bs}: {got} vs published {want}")
        # AC-4 fails at every batch size, and the gap is the known ~2.2-2.8x
        for bs in bss:
            a4 = rep["per_bs"][str(bs)]["ac4"]
            self.assertEqual(a4["verdict"], "FAIL")
            self.assertLess(a4["ratio_mpk_over_vllm"], 1.0)
        # AC-5 also fails today (e2e 2.6-2.9x vs the 1.25x bound)
        self.assertEqual(rep["ac5"]["verdict"], "FAIL")
        # the drift rule is satisfied by construction here (same data)
        for bs in bss:
            self.assertTrue(rep["per_bs"][str(bs)]["vllm"]["drift"]["decode"]["valid"])

    # ------------------------------------------------------------- PASS ----
    def test_a_real_win_passes(self):
        bss = (8,)
        # scale the FULL arm's wall down so the prefill-subtracted slope beats
        # vLLM; nothing else about the fixture changes.
        mpk = build_mpk_root(self.tmp / "mpk", bss, scale=0.30)
        fresh = build_vllm_fresh(self.tmp / "fresh", bss)
        p, rep = run(self.tmp, mpk, fresh, bss)
        a4 = rep["per_bs"]["8"]["ac4"]
        self.assertGreater(a4["mpk_decode_tok_s"], a4["vllm_decode_tok_s"])
        self.assertEqual(a4["verdict"], "PASS")
        self.assertEqual(rep["per_bs"]["8"]["ac5"]["verdict"], "PASS")
        self.assertEqual(rep["verdict"], "PASS")
        self.assertEqual(p.returncode, 0)

    # ------------------------------------------------------------- DRIFT ---
    def test_comparator_drift_fails_loudly_and_does_not_pick_a_number(self):
        bss = (8,)
        mpk = build_mpk_root(self.tmp / "mpk", bss, scale=0.30)
        # a fresh boot 30% off the pinned capture: beyond the protocol's own
        # two-boot agreement statistic
        fresh = build_vllm_fresh(self.tmp / "fresh", bss, decode_scale=0.70)
        p, rep = run(self.tmp, mpk, fresh, bss)
        self.assertEqual(p.returncode, 1)
        self.assertFalse(rep["per_bs"]["8"]["vllm"]["drift"]["decode"]["valid"])
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertTrue(any("COMPARATOR DRIFT" in f for f in rep["failures"]))

    def test_identity_drift_fails(self):
        bss = (8,)
        mpk = build_mpk_root(self.tmp / "mpk", bss, scale=0.30)
        fresh = build_vllm_fresh(self.tmp / "fresh", bss)
        d = json.loads((fresh / "bs8.json").read_text())
        d["shared_meta"]["versions"]["vllm"] = "0.99.0"
        (fresh / "bs8.json").write_text(json.dumps(d))
        p, rep = run(self.tmp, mpk, fresh, bss)
        self.assertEqual(p.returncode, 1)
        self.assertTrue(any("BASELINE IDENTITY DRIFT" in f
                            for f in rep["failures"]))

    # ------------------------------------------------- validity asymmetry --
    def test_winning_but_invalid_measurement_is_not_a_pass(self):
        bss = (8,)
        mpk = build_mpk_root(self.tmp / "mpk", bss, scale=0.30,
                             dirty={(8, "pre", 0), (8, "pre", 1)})
        fresh = build_vllm_fresh(self.tmp / "fresh", bss)
        p, rep = run(self.tmp, mpk, fresh, bss)
        self.assertEqual(rep["per_bs"]["8"]["ac4"]["verdict"], "NOT_EVALUABLE")
        self.assertEqual(p.returncode, 3)
        self.assertTrue(any("required reps" in x for x in rep["not_evaluable"]))

    def test_losing_candidate_fails_even_with_an_invalid_measurement(self):
        bss = (8,)
        mpk = build_mpk_root(self.tmp / "mpk", bss, dirty={(8, "pre", 0)})
        fresh = build_vllm_fresh(self.tmp / "fresh", bss)
        p, rep = run(self.tmp, mpk, fresh, bss)
        self.assertEqual(rep["per_bs"]["8"]["ac4"]["verdict"], "FAIL")
        self.assertEqual(p.returncode, 1)

    def test_multi_wave_run_is_excluded_not_averaged(self):
        bss = (8,)
        mpk = build_mpk_root(self.tmp / "mpk", bss, break_wave={(8, "full", 0)})
        fresh = build_vllm_fresh(self.tmp / "fresh", bss)
        p, rep = run(self.tmp, mpk, fresh, bss)
        full = rep["per_bs"]["8"]["mpk"]["full"]
        self.assertEqual(full["n"], 2)
        self.assertTrue(any("waves" in e.get("excluded_why", "")
                            for e in full["excluded"]))

    def test_a_run_compiled_off_policy_is_excluded_and_reported(self):
        """A cell whose COMPILED cap is not the shipped policy's value cannot be
        used: the cap is a compile-time define, so that cell measured a different
        binary.  The scorer reads the value out of the run's own artifact."""
        bss = (16,)
        mpk = build_mpk_root(self.tmp / "mpk", bss, cap_override="none")
        fresh = build_vllm_fresh(self.tmp / "fresh", bss)
        p, rep = run(self.tmp, mpk, fresh, bss)
        probs = rep["per_bs"]["16"]["mpk"]["problems"]
        self.assertTrue(any("compiled admission cap" in x for x in probs), probs)
        self.assertEqual(rep["per_bs"]["16"]["mpk"]["full"]["n"], 0)
        self.assertNotEqual(p.returncode, 0)

    def test_batch_sizes_outside_the_pinned_set_are_refused(self):
        mpk = build_mpk_root(self.tmp / "mpk", (1,))
        fresh = build_vllm_fresh(self.tmp / "fresh", (1,))
        p, _ = run(self.tmp, mpk, fresh, (3,))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("not in the pinned set", p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
