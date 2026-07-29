#!/usr/bin/env python3
"""Synthetic exercise of gate_ac3_stable.py score -- no GPU needed.

Run: python3 test_gate_scorer.py   (writes under $SCORERTEST_DIR or a temp dir)

Builds fake rep trees covering: all-clean, one fingerprint-divergent rep with
clean tokens (the campaign-2 sub-argmax case), a token-mismatched rep, a
run-error rep, and an under-quorum bs. Checks the verdicts + exit codes.
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

GATE = Path(__file__).resolve().parents[3] / "harness" / "gate_ac3_stable.py"
ROOT = Path(os.environ.get("SCORERTEST_DIR") or (tempfile.gettempdir() + "/gate_ac3_scorertest"))
PROMPTS = [f"p{i:02d}-x" for i in range(1, 11)]


def baseline_dir(root, bss):
    d = root / "baseline"; d.mkdir(parents=True, exist_ok=True)
    for bs in bss:
        (d / f"bs{bs}.json").write_text(json.dumps(
            {p: {"token_ids": [bs * 100 + i for i in range(64)]} for p in PROMPTS}, indent=2))
    return d


def mkrep(root, bs, r, *, fp_seed=0, tokens_bad=False, error=False, waves=2):
    d = root / "reps" / f"bs{bs}_r{r}"; d.mkdir(parents=True, exist_ok=True)
    tag = f"bs{bs}_r{r}"
    if error:
        (d / f"meta_{tag}.json").write_text(json.dumps(
            {"tag": tag, "status": "error", "rc": 134, "note": "rep process exited rc=134"}))
        return
    toks = {p: {"token_ids": [bs * 100 + i for i in range(64)]} for p in PROMPTS}
    if tokens_bad:
        toks[PROMPTS[3]]["token_ids"][17] = 999999
    (d / f"bs{bs}.json").write_text(json.dumps(toks, indent=2))
    fps = {}
    rng = np.random.default_rng(1234)
    for w in range(waves):
        for k in ("k", "v", "conv", "rec"):
            base = rng.integers(0, 1 << 60, size=(4, 8), dtype=np.int64)
            if fp_seed and w == 1 and k in ("k", "v"):
                base[0, 0] += fp_seed
            fps[f"w{w}_{k}"] = base
    for p in PROMPTS:
        fps[f"tok_{p}"] = np.asarray(toks[p]["token_ids"], dtype=np.int64)
    np.savez_compressed(d / f"fp_{tag}.npz", **fps)
    (d / f"meta_{tag}.json").write_text(json.dumps(
        {"tag": tag, "status": "ok", "bs": bs, "rep": r, "secs": 120.0,
         "n_waves": waves, "dump_md5": "deadbeef",
         "device": {"phys_index": 3, "uuid": "GPU-abc", "cuda_visible_devices": "3"},
         "gpu_before": {"memory_used_mib": 4, "utilization_pct": 0}}))


def run(case, bss, baseline, reps=3, expect_rc=None):
    out = ROOT / case / "report.json"
    p = subprocess.run([sys.executable, str(GATE), "score",
                        "--reps-root", str(ROOT / case / "reps"),
                        "--baseline", str(baseline),
                        "--batch-sizes", ",".join(str(b) for b in bss),
                        "--reps", str(reps),
                        "--output-json", str(out)],
                       capture_output=True, text=True)
    print(f"\n######## {case}: rc={p.returncode} (expected {expect_rc})")
    print(p.stdout[-3000:])
    if p.returncode == 3:
        print("STDERR:", p.stderr[-2000:])
    assert p.returncode == expect_rc, f"{case}: rc {p.returncode} != {expect_rc}"
    return json.loads(out.read_text())


shutil.rmtree(ROOT, ignore_errors=True)
ROOT.mkdir(parents=True)
BSS = [1, 2]
BASE = baseline_dir(ROOT, BSS)

# case 1: all clean -> STABLE, rc 0
c = ROOT / "clean"
for bs in BSS:
    for r in (1, 2, 3):
        mkrep(c, bs, r)
rep = run("clean", BSS, BASE, expect_rc=0)
assert rep["verdict"] == "STABLE"
assert rep["totals"]["fingerprint_divergence_rate"] == 0.0
assert rep["per_bs"]["1"]["reps_needed_to_reach_verdict"] == 3

# case 2: bs1 rep2 fingerprint-divergent, tokens clean -> quarantine + 4th rep
c = ROOT / "quar"
for bs in BSS:
    for r in (1, 2, 3):
        mkrep(c, bs, r, fp_seed=(7 if (bs == 1 and r == 2) else 0))
mkrep(c, 1, 4)
rep = run("quar", BSS, BASE, expect_rc=0)
assert rep["verdict"] == "STABLE", rep["verdict"]
b1 = rep["per_bs"]["1"]
assert b1["accepted"] == 3 and b1["quarantined"] == 1, b1
assert b1["reps_needed_to_reach_verdict"] == 4, b1["reps_needed_to_reach_verdict"]
assert abs(b1["divergence_rate"] - 0.25) < 1e-9
q = [r for r in b1["reps"] if r["classification"] == "quarantined"][0]
assert q["fingerprint_delta_vs_consensus"]["waves_touched"] == ["w1"], q
assert q["tokens"]["all_identical"] is True
assert rep["totals"]["token_divergence_rate"] == 0.0

# case 3: a token mismatch anywhere -> FAIL, rc 1
c = ROOT / "tokfail"
for bs in BSS:
    for r in (1, 2, 3):
        mkrep(c, bs, r, tokens_bad=(bs == 2 and r == 3))
rep = run("tokfail", BSS, BASE, expect_rc=1)
assert rep["verdict"] == "FAIL"
assert rep["per_bs"]["2"]["verdict"] == "FAIL"
bad = [r for r in rep["per_bs"]["2"]["reps"] if r.get("token_mismatch")][0]
assert bad["tokens"]["mismatched"][0]["first_divergent_position"] == 17

# case 4: under quorum (2 clean + 2 divergent, need 3) -> UNSTABLE, rc 2
c = ROOT / "unstable"
mkrep(c, 1, 1); mkrep(c, 1, 2, fp_seed=5); mkrep(c, 1, 3); mkrep(c, 1, 4, fp_seed=9)
for r in (1, 2, 3):
    mkrep(c, 2, r)
rep = run("unstable", [1, 2], BASE, expect_rc=2)
assert rep["verdict"] == "UNSTABLE"
assert rep["per_bs"]["1"]["accepted"] == 2 and rep["per_bs"]["1"]["quarantined"] == 2
assert rep["per_bs"]["1"]["reps_needed_to_reach_verdict"] is None

# case 5: a run error is recorded, not counted as divergence
c = ROOT / "err"
mkrep(c, 1, 1); mkrep(c, 1, 2, error=True); mkrep(c, 1, 3); mkrep(c, 1, 4)
for r in (1, 2, 3):
    mkrep(c, 2, r)
rep = run("err", [1, 2], BASE, expect_rc=0)
assert rep["per_bs"]["1"]["errors"] == 1
assert rep["per_bs"]["1"]["accepted"] == 3
assert rep["per_bs"]["1"]["divergence_rate"] == 0.0
assert rep["totals"]["run_errors"] == 1

print("\nALL SCORER CASES PASS")

# case 6: a LOST rep (ledger says launched, no dir) is reported as an error
c = ROOT / "lost"
for bs in (1, 2):
    for r in (1, 2, 3):
        mkrep(c, bs, r)
(c / "launched.txt").write_text("bs1_r1\nbs1_r2\nbs1_r3\nbs1_r4\nbs2_r1\nbs2_r2\nbs2_r3\n")
rep = run("lost", [1, 2], BASE, expect_rc=0)
b1 = rep["per_bs"]["1"]
assert b1["reps_launched"] == 4 and b1["errors"] == 1, b1
lost = [r for r in b1["reps"] if r["tag"] == "bs1_r4"][0]
assert lost["classification"] == "run_error" and "LOST" in lost["error"], lost
assert b1["accepted"] == 3 and b1["verdict"] == "STABLE"
print("\nLOST-REP CASE PASSES")

# case 7: truncated / zero-byte artifacts are run errors, never a crash
c = ROOT / "corrupt"
for bs in (1, 2):
    for r in (1, 2, 3):
        mkrep(c, bs, r)
mkrep(c, 1, 4); (c / "reps/bs1_r4/meta_bs1_r4.json").write_text("")          # ENOSPC meta
mkrep(c, 1, 5); (c / "reps/bs1_r5/fp_bs1_r5.npz").write_bytes(b"PK\x03\x04") # truncated npz
mkrep(c, 1, 6); (c / "reps/bs1_r6/bs1.json").write_text("{trunc")            # truncated dump
rep = run("corrupt", [1, 2], BASE, expect_rc=0)
b1 = rep["per_bs"]["1"]
assert b1["errors"] == 3, b1["errors"]
assert b1["accepted"] == 3 and b1["verdict"] == "STABLE", b1
notes = [r["error"] for r in b1["reps"] if r["classification"] == "run_error"]
assert all("unreadable" in (n or "") for n in notes), notes
print("\nCORRUPT-ARTIFACT CASE PASSES")
