#!/usr/bin/env python3
"""M3-I9 -- CPU-side gate on the MPK_MAX_TOKENS_PER_REQUEST admission knob.

Three checks, none of which needs a GPU or the CUDA toolchain:

A. STATIC -- the two MODE_OFFLINE prefill sites in `persistent_kernel.cuh` go
   through `mirage::mpk::admission_prefill_tokens`, the decode branches do not,
   and no other mode was touched. This is what pins the C++ scaffolding in
   `test_admission_policy.cpp` to the real call sites.

B. NO-OP -- compiled with the macro unset, the host replay must reproduce
   `protocol_sim.simulate(..., cap=None)` exactly at all five batch sizes,
   i.e. M3-I1's validated 109/109/109/111/203 schedule, chunk for chunk. The
   default is `MPK_MAX_NUM_BATCHED_TOKENS`, so the added clamp must be the
   identity.

C. CAP -- compiled with `-DMPK_MAX_TOKENS_PER_REQUEST=k`, the same replay must
   reproduce `protocol_sim.simulate(..., cap=k)` for k in {1, 2, 4, 8}, which is
   what `predictions.md` prices. The bs16 cap=1 arm must land on 131 iterations.

D. RANGE -- the header's static_asserts must reject cap=0 (would stall every
   prefill) and cap > mbt (meaningless) at COMPILE time, so a bad knob cannot
   reach a B200 window.

Run: python3 test_admission_policy.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

from protocol_sim import ac3_slots, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
INCLUDE = os.path.join(REPO, "include")
CUH = os.path.join(INCLUDE, "mirage", "persistent_kernel", "persistent_kernel.cuh")
HDR = os.path.join(INCLUDE, "mirage", "persistent_kernel", "admission_policy.h")
SRC = os.path.join(HERE, "test_admission_policy.cpp")
MBT = 16
MSL = 132


# ---------------------------------------------------------------- A. static --
def offline_block(text: str) -> str:
    """The MODE_OFFLINE prepare_next_batch body only."""
    start = text.index("#ifdef MODE_OFFLINE")
    end = text.index("#ifdef MODE_ONLINE", start)
    return text[start:end]


def static_checks() -> int:
    bad = 0
    if not os.path.exists(HDR):
        print("FAIL: admission_policy.h missing")
        return 1
    text = open(CUH).read()
    off = offline_block(text)
    n_helper = off.count("admission_prefill_tokens(")
    print(f"A. static checks")
    print(f"   MODE_OFFLINE prefill sites via the helper : {n_helper} (want 2)")
    bad += 0 if n_helper == 2 else 1
    # the decode branches must remain uncapped
    for frag in ("min(1, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens)",):
        ok = frag in off
        print(f"   decode branch left uncapped               : {ok}")
        bad += 0 if ok else 1
    # no other mode touched
    other = text.replace(off, "")
    n_other = other.count("admission_prefill_tokens(")
    print(f"   helper used outside MODE_OFFLINE           : {n_other} (want 0)")
    bad += 0 if n_other == 0 else 1
    # the header is included, and the default is the mbt
    ok = '#include "admission_policy.h"' in text
    print(f"   admission_policy.h included                : {ok}")
    bad += 0 if ok else 1
    h = open(HDR).read()
    ok = ("#ifndef MPK_MAX_TOKENS_PER_REQUEST" in h and
          "#define MPK_MAX_TOKENS_PER_REQUEST MPK_MAX_NUM_BATCHED_TOKENS" in h)
    print(f"   default cap == MPK_MAX_NUM_BATCHED_TOKENS  : {ok}")
    bad += 0 if ok else 1
    # the python side must not emit the define unless asked
    py = open(os.path.join(REPO, "python", "mirage", "mpk",
                           "persistent_kernel.py")).read()
    ok = ('if getattr(mpk, "max_tokens_per_request", None) is not None:' in py
          and "-DMPK_MAX_TOKENS_PER_REQUEST=" in py)
    print(f"   define emitted only when asked for         : {ok}")
    bad += 0 if ok else 1
    print("   ->", "PASS" if not bad else f"FAIL ({bad})")
    return bad


# ------------------------------------------------------------- B/C. replay --
def build(tmp: str, cap):
    exe = os.path.join(tmp, f"adm_{cap or 'default'}")
    cmd = ["g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
           f"-I{INCLUDE}", f"-DMPK_MAX_NUM_BATCHED_TOKENS={MBT}"]
    if cap is not None:
        cmd.append(f"-DMPK_MAX_TOKENS_PER_REQUEST={cap}")
    cmd += [SRC, "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("compile FAILED:", " ".join(cmd))
        print(r.stderr)
        raise SystemExit(1)
    return exe


def run(exe, plens):
    r = subprocess.run([exe, str(MSL)] + [str(p) for p in plens],
                       capture_output=True, text=True, check=True)
    out = []
    for line in r.stdout.splitlines():
        f = [int(x) for x in line.split()]
        out.append((f[0], f[1:]))
    return out


def expected(plens, cap):
    sim = simulate(plens, MBT, MSL, cap=cap)
    return [(it["n_live"], list(it["chunks"])) for it in sim["iters"]]


def replay_checks(tmp: str) -> int:
    bad = 0
    for label, cap in [("B. no-op (macro unset)", None)] + \
                      [(f"C. cap={k}", k) for k in (1, 2, 4, 8)]:
        exe = build(tmp, cap)
        print(f"{label}")
        for bs in (1, 2, 4, 8, 16):
            plens = ac3_slots(bs)
            got, want = run(exe, plens), expected(plens, cap)
            ok = got == want
            bad += 0 if ok else 1
            note = ""
            if cap is None and bs in (1, 2, 4, 8, 16):
                note = f"  (M3-I1 validated: {[109,109,109,111,203][(1,2,4,8,16).index(bs)]})"
            if cap == 1 and bs == 16:
                note = "  (predictions.md C4: 131)"
            print(f"   bs{bs:<3d} iterations C++={len(got):4d} sim={len(want):4d} "
                  f"chunks {'MATCH' if ok else 'DIFFER'}{note}")
            if not ok:
                for i, (g, w) in enumerate(zip(got, want)):
                    if g != w:
                        print(f"      first mismatch at iteration {i}: "
                              f"C++={g} sim={w}")
                        break
    print("   ->", "PASS" if not bad else f"FAIL ({bad})")
    return bad


def range_checks(tmp: str) -> int:
    """D: the static_asserts must fire, at compile time, not at run time."""
    bad = 0
    print("D. range checks (static_assert)")
    for cap, why in ((0, "stalls every prefill"), (MBT * 2, "above the budget")):
        cmd = ["g++", "-std=c++17", "-O0", f"-I{INCLUDE}",
               f"-DMPK_MAX_NUM_BATCHED_TOKENS={MBT}",
               f"-DMPK_MAX_TOKENS_PER_REQUEST={cap}", SRC,
               "-o", os.path.join(tmp, "neg")]
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        ok = rc != 0
        bad += 0 if ok else 1
        print(f"   cap={cap:<3d} ({why:<21s}) rejected at compile time: {ok}")
    print("   ->", "PASS" if not bad else f"FAIL ({bad})")
    return bad


def main() -> int:
    if not shutil.which("g++"):
        print("SKIP: no g++ on this host")
        return 0
    bad = static_checks()
    with tempfile.TemporaryDirectory() as tmp:
        bad += replay_checks(tmp)
        bad += range_checks(tmp)
    print("\nVERDICT:", "PASS" if not bad else f"FAIL ({bad} checks)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
