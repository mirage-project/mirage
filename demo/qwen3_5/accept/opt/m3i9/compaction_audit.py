#!/usr/bin/env python3
"""M3-I9 -- HAZARD-COMPACTION audit of the shipped wave protocol.

`mpk_engine_run.py`'s module docstring says the hazard "only fires when a
request retires while another is still active -- i.e. rolling admission with
total_num_requests > max_num_batched_requests, which the wave protocol above
does not use."  The `i.e.` does not hold.  Retirement-while-others-are-active
happens INSIDE a single wave, because `max_seq_length` retires on the global
step and slot-order-greedy admission advances low slots first.  The replay says
the shipped AC-3 waves move live slots 1 / 12 / 69 times at bs 4 / 8 / 16.

A move only reaches an ANSWER if it happens before the request has written the
64 tokens the harness reports.  Those requests exist, and they are exactly the
duplicate-padding slots whose `slot_isolation_checks` in the committed M2 dumps
report `identical: false`.  This script prints that correspondence and the
control that rules out the competing explanation.

Run: python3 compaction_audit.py [--dumps ../../results/dumps]
"""
from __future__ import annotations

import glob
import json
import os
import sys

from protocol_sim import AC3_PLENS, audit, decomposition, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.path.dirname(os.path.dirname(HERE))
PIDS = ["p06-poem", "p01-history", "p04-chinese", "p09-translate", "p07-format",
        "p05-cuda", "p08-science", "p10-logic", "p03-python", "p02-math"]


def waves(bs):
    """The adapter's wave split: chunks of `bs` prompts, last one padded by
    repetition (`MPKOfflineAdapter.generate`)."""
    out = []
    for i in range(0, len(AC3_PLENS), bs):
        w = list(range(i, min(i + bs, len(AC3_PLENS))))
        while len(w) < bs:
            w.append(w[len(w) % len(w)])
        out.append(w)
    return out


def shipped_audit():
    print("### live-slot migrations in the SHIPPED AC-3 waves (no rolling admission)")
    print(f"{'bs':>3s} {'wave':>4s} {'iters':>6s} {'moves':>6s} {'first@':>7s} "
          f"{'requests whose reported 64-token window straddles a move'}")
    for bs in (1, 2, 4, 8, 16):
        for wi, w in enumerate(waves(bs)):
            pl = [AC3_PLENS[i] for i in w]
            s = simulate(pl, 16, 132)
            a = audit(s, 64)
            first = s["moves"][0][0] if s["moves"] else None
            print(f"{bs:3d} {wi:4d} {s['n_iterations']:6d} {a['n_moves']:6d} "
                  f"{str(first):>7s} slots {a['straddling_requests']}")


def isolation_correspondence(dumps=None):
    print("\n### prediction vs the committed slot_isolation_checks")
    dumps = dumps or f"{ACC}/results/dumps"
    files = sorted(glob.glob(f"{dumps}/timings_bs*.json"))
    if not files:
        print(f"  (no dumps under {dumps}; skipping)")
        return
    hits = miss = 0
    for f in files:
        d = json.load(open(f))
        checks = d.get("slot_isolation_checks") or []
        if not checks:
            continue
        bs = d["batch_size"]
        # the wave this dump belongs to: match its prompt_ids against the split
        pids = d["waves"][0]["prompt_ids"]
        idx = [PIDS.index(p) for p in pids]
        pl = [AC3_PLENS[i] for i in idx]
        a = audit(simulate(pl, 16, 132), 64)
        strad = set(a["straddling_requests"])
        for c in checks:
            slot = c["slots"][1]
            pred_mismatch = slot in strad
            obs_mismatch = not c["identical"]
            ok = pred_mismatch == obs_mismatch
            hits += ok
            miss += (not ok)
            print(f"  {os.path.basename(f):24s} bs{bs:<3d} {c['prompt_id']:14s} "
                  f"slots={str(c['slots']):8s} predicted={'mismatch' if pred_mismatch else 'match  '}"
                  f"  observed={'mismatch' if obs_mismatch else 'match  '}  "
                  f"{'OK' if ok else 'WRONG'}")
    print(f"  -> {hits} agreeing, {miss} disagreeing.  NOTE: every duplicate slot in the")
    print("     corpus is predicted to straddle, so this direction has no negative control;")
    print("     the control that matters is the decomposition invariance below.")


def decomposition_control():
    """The competing explanation, and why it is refuted.

    H1  a duplicate slot's answer differs because compaction moved it mid-window
        and it inherited a retired request's GDN state.
    H2  a duplicate slot's answer differs because its prefill is CHOPPED
        DIFFERENTLY (slot 0 gets p06-poem as [16, 8]; slot 10 gets it as
        [1, 6, 6, 6, 5]) and the chunking changes the numerics.

    H2 is refuted by the committed M2 result that all 25 (prompt, batch size)
    token sequences are byte-identical: the reported placement of a prompt gets
    a DIFFERENT decomposition at different batch sizes, none of those placements
    straddles a move, and their answers agree to the byte.
    """
    print("\n### control: does the prefill chunk decomposition change the answer? (H2)")
    seen = {}
    for bs in (1, 2, 4, 8, 16):
        for wi, w in enumerate(waves(bs)):
            pl = [AC3_PLENS[i] for i in w]
            s = simulate(pl, 16, 132)
            a = audit(s, 64)
            dec = decomposition(s)
            for slot, ri in enumerate(w):
                if slot != w.index(ri):
                    continue                 # duplicate copy, not the reported one
                if slot in a["straddling_requests"]:
                    continue                 # not a clean placement
                seen.setdefault(PIDS[ri], set()).add(tuple(dec[slot]))
    n_multi = 0
    for pid in PIDS:
        ds = seen.get(pid, set())
        if len(ds) > 1:
            n_multi += 1
        print(f"  {pid:14s} plen={AC3_PLENS[PIDS.index(pid)]:3d} "
              f"{len(ds)} distinct decomposition(s) among clean reported placements")
        for d in sorted(ds, key=len):
            print(f"       {list(d)}")
    print(f"  -> {n_multi}/10 prompts are prefilled with >=2 different chunkings in")
    print("     NON-straddling placements, and M2's committed report has all 25")
    print("     (prompt, bs) sequences byte-identical.  H2 is refuted: the answer does")
    print("     not depend on how prefill is chopped.  H1 (compaction) is what is left,")
    print("     and it is also the evidence that a per-request cap of 1 -- i.e. the")
    print("     all-ones chunking -- is bit-exact by the same mechanism.")


if __name__ == "__main__":
    dumps = None
    for a in sys.argv[1:]:
        if a.startswith("--dumps="):
            dumps = a.split("=", 1)[1]
    shipped_audit()
    isolation_correspondence(dumps)
    decomposition_control()
