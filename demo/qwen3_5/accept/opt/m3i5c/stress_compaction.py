#!/usr/bin/env python3
"""M3-I5c -- race-targeted stress for the SM100 router's Phase-7 compaction.

PRE-REGISTERED for the GPU window; written on CPU, never executed. Needs a
B200 and the sm100_moe_block_qwen35 extension:

    cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe_block_qwen35
    $PY setup.py build_ext --inplace
    $PY $MIRAGE/demo/qwen3_5/accept/opt/m3i5c/stress_compaction.py \
        --iters 2000 --rows 16 --out stress.json

The pre-fix compaction was a data race, so a single pass proves nothing: the
schedule has to be sampled. Four checks per iteration, each with its own
counter so a failure says WHICH invariant broke:

  C1 count        mask[NUM_EXPERTS] == number of distinct ids  (the existing
                  detector; fires on the phantom-expert inflation the race
                  produced)
  C2 set          reconstruct a 0/1 mask from ids[:n] and compare against the
                  oracle mask built from torch.topk -- catches both phantoms
                  and (hypothetical) dropped experts
  C3 ascending    ids[:n] is strictly increasing. NEW, and only meaningful
                  after this fix: the pre-fix order was atomicAdd arrival
                  order. This is the determinism claim's direct test.
  C4 replay       running the SAME logits again returns byte-identical ids,
                  weights and routing indices. Distinguishes "the kernel is
                  deterministic" from "the kernel happens to agree with an
                  oracle".

`--rows` should be the largest row count the window intends to defend. The
issue's prerequisite for an mbt>16 default is this stress AT that mbt, so run
it again at 32/64/128 before any default flip.
"""
import argparse
import json
import sys
import time

import torch

import runtime_kernel_blackwell_moe_block_qwen35 as mk

NUM_EXPERTS = 256
TOPK = 8


def alloc(rows):
    return (
        torch.empty((rows, TOPK), device="cuda", dtype=torch.float32),
        torch.empty((NUM_EXPERTS, rows), device="cuda", dtype=torch.int32),
        torch.empty((NUM_EXPERTS + 1,), device="cuda", dtype=torch.int32),
    )


def run(logits, vpt, round_weights=False):
    rows = logits.size(0)
    w, r, a = alloc(rows)
    mk.topk_softmax_sm100(logits, w, r, a, vpt, round_weights)
    torch.cuda.synchronize()
    return w, r, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--vpt", type=int, default=0, help="0 = shipped default")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    fail = {"C1_count": 0, "C2_set": 0, "C3_ascending": 0, "C4_replay": 0}
    first = []
    n_active_hist = []
    t0 = time.time()

    for it in range(args.iters):
        # Sweep the activation density: a nearly-saturated mask (many marks,
        # many compacted writes) is the pre-fix race's worst case, a sparse one
        # its easiest. `scale` widens/narrows the logit spread, which moves how
        # many DISTINCT experts `rows x TOPK` picks.
        scale = (0.05, 0.5, 4.0, 40.0)[it % 4]
        logits = (torch.randn(args.rows, NUM_EXPERTS, device=dev) * scale).to(
            torch.bfloat16)

        w, r, a = run(logits, args.vpt)
        n = int(a[NUM_EXPERTS].item())
        ids = a[:n].to(torch.long)
        n_active_hist.append(n)

        # ---- C1: the existing detector -----------------------------------
        if n != int(torch.unique(ids).numel()):
            fail["C1_count"] += 1
            first.append(("C1", it, n, ids[:16].tolist()))

        # ---- C2: the set matches a torch oracle --------------------------
        ref = torch.topk(logits.float(), TOPK, dim=1).indices
        want = torch.zeros(NUM_EXPERTS, device=dev, dtype=torch.int32)
        want[ref.reshape(-1)] = 1
        got = torch.zeros(NUM_EXPERTS, device=dev, dtype=torch.int32)
        if n > 0:
            got.index_fill_(0, ids, 1)
        if not torch.equal(got, want):
            fail["C2_set"] += 1
            if len(first) < 8:
                first.append(("C2", it, n,
                              torch.nonzero(got != want).flatten()[:8].tolist()))

        # ---- C3: strictly ascending (only true after M3-I5c) -------------
        if n > 1 and not bool((ids[1:] > ids[:-1]).all()):
            fail["C3_ascending"] += 1
            if len(first) < 8:
                first.append(("C3", it, n, ids[:16].tolist()))

        # ---- C4: same input -> byte-identical output ---------------------
        w2, r2, a2 = run(logits, args.vpt)
        if not (torch.equal(a, a2) and torch.equal(w, w2)
                and torch.equal(r, r2)):
            fail["C4_replay"] += 1
            if len(first) < 8:
                first.append(("C4", it, n, None))

    dt = time.time() - t0
    total = sum(fail.values())
    rep = {
        "iters": args.iters,
        "rows": args.rows,
        "vpt": args.vpt if args.vpt else mk.topk_softmax_default_vpt(),
        "rows_per_pass": mk.topk_softmax_rows_per_task(args.vpt),
        "seconds": round(dt, 1),
        "n_active_min": min(n_active_hist),
        "n_active_max": max(n_active_hist),
        "n_active_mean": round(sum(n_active_hist) / len(n_active_hist), 1),
        "failures": fail,
        "first_failures": first[:8],
        "verdict": "PASS" if total == 0 else "FAIL",
    }
    print(json.dumps(rep, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
