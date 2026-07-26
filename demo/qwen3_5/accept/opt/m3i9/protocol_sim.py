#!/usr/bin/env python3
"""M3-I9 -- admission/scheduling protocol design tool for the AC-3 wave.

Extends `opt/schedule_sim.py` (M3-I1's validated `prepare_next_batch` replay)
in three ways it needs to price a *policy* rather than describe the status quo:

1. **Slot compaction is modelled.**  M3-I1's replay kept every slot in place,
   which is exact for iteration counts but silently drops the one thing an
   admission protocol has to respect: `prepare_next_batch` step 3 rewrites
   `request_ids[num_reqs] = request_id` with `num_reqs <= i`, so a survivor
   *moves down* when a lower slot retires.  The GDN conv/recurrent pools are
   `[max_num_batched_requests, ...]` indexed by that slot
   (`builder._alloc_state_pools`), so a move is HAZARD-COMPACTION firing.
2. **The rolling-admission loop is modelled** (`total > mbr`), so staggered /
   refill policies can be priced instead of guessed at.
3. **Per-slot chunk vectors are kept**, because the measured cost of an
   iteration is set by the *largest* per-slot chunk, not by the token total
   (see `cost_model.py`).

Self-check (`--self-check`) reproduces M3-I1's validated iteration counts
109/109/109/111/203 at bs 1/2/4/8/16 -- the same falsifiable test that
validated the original replay against the profiler's BEGIN_TASK_GRAPH count.

Policies are expressed as (slot order, mbt, per-request token cap).  The cap is
the one knob MODE_OFFLINE does not have today; see README.md for its spec.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional


def simulate(plens: List[int], mbt: int, msl: int, *, mbr: Optional[int] = None,
             total: Optional[int] = None, order: Optional[List[int]] = None,
             cap: Optional[int] = None, hold_decode: bool = False) -> dict:
    """Replay MODE_OFFLINE `prepare_next_batch` (persistent_kernel.cuh:225-400).

    plens  -- prompt length per REQUEST id.
    mbr    -- max_num_batched_requests (slots); defaults to len(plens).
    total  -- total_num_requests; defaults to len(plens).  total > mbr is
              rolling admission, which is what `assert_no_rolling_admission`
              refuses today.
    order  -- admission order (indices into plens); defaults to id order.
    cap    -- HYPOTHETICAL per-request per-iteration token cap.  None = today's
              runtime (a prefilling request takes the whole remaining budget).
    hold_decode -- HYPOTHETICAL: a request that finished prefill emits nothing
              while any request is still prefilling or unadmitted
              ("admit all, prefill all, then pure decode").
    """
    n_req = total if total is not None else len(plens)
    mbr = mbr if mbr is not None else n_req
    order = order if order is not None else list(range(n_req))
    step = [0] * n_req
    slot_of = [-1] * mbr          # config.request_ids[]
    next_ptr = 0                  # *config.next_request_id
    iters: List[dict] = []
    moves: List[tuple] = []       # (iteration, request, from_slot, to_slot)
    guard = 0

    while guard < 500000:
        guard += 1
        pending = (next_ptr < n_req) or any(
            slot_of[i] != -1 and step[slot_of[i]] < plens[slot_of[i]]
            for i in range(mbr))
        # ---- step 3: compact survivors toward slot 0, then fill the budget --
        num_reqs = num_tokens = 0
        new: List[int] = []
        slot_req: List[int] = []
        for i in range(mbr):
            r = slot_of[i]
            if r == -1:
                continue
            if num_reqs != i:
                moves.append((len(iters), r, i, num_reqs))
            rem = plens[r] - step[r]
            if rem > 0:
                k = min(rem, mbt - num_tokens)
                if cap is not None:
                    k = min(k, cap)
            elif hold_decode and pending:
                k = 0
            else:
                k = min(1, mbt - num_tokens)
            slot_req.append(r)
            new.append(k)
            num_tokens += k
            num_reqs += 1
        # ---- admit new prefill requests until capacity ----------------------
        while num_reqs < mbr and num_tokens < mbt and next_ptr < n_req:
            r = order[next_ptr]
            next_ptr += 1
            k = min(plens[r], mbt - num_tokens)
            if cap is not None:
                k = min(k, cap)
            slot_req.append(r)
            new.append(k)
            num_tokens += k
            num_reqs += 1
        slot_of = slot_req + [-1] * (mbr - num_reqs)
        if num_tokens == 0:
            break
        n_prefill = sum(1 for j in range(num_reqs)
                        if plens[slot_req[j]] - step[slot_req[j]] > 0)
        iters.append(dict(
            iteration=len(iters), tokens=num_tokens, n_live=num_reqs,
            n_prefill=n_prefill, n_decode=num_reqs - n_prefill,
            n_active=sum(1 for k in new if k > 0),
            max_chunk=max(new) if new else 0,
            chunks=list(new), slots=list(slot_req),
            steps=[step[r] for r in slot_req]))
        # ---- step 1 of the next prepare: advance, then retire in place ------
        for j in range(num_reqs):
            r = slot_req[j]
            step[r] += new[j]
            if step[r] + 1 >= msl:
                slot_of[j] = -1
    return dict(n_iterations=len(iters), iters=iters, moves=moves, plens=plens,
                mbt=mbt, msl=msl, mbr=mbr, total=n_req, cap=cap,
                hold_decode=hold_decode)


def audit(sim: dict, report_window: int = 64) -> dict:
    """Compaction exposure.

    `moves` counts live-slot migrations.  A migration only corrupts a *reported*
    answer if it happens before the request has written the `report_window`
    tokens the harness slices out of `tokens[plen : plen + window]`; the GDN
    state a survivor inherits after a move belongs to the retired request.
    """
    plens = sim["plens"]
    done: Dict[int, int] = {}
    for it in sim["iters"]:
        for j, r in enumerate(it["slots"]):
            end = it["steps"][j] + it["chunks"][j]
            # tokens land at index `step` for step in (plen .. plen+window-1)
            if r not in done and end >= plens[r] + report_window - 1:
                done[r] = it["iteration"]
    first_move: Dict[int, int] = {}
    for (i, r, _f, _t) in sim["moves"]:
        first_move.setdefault(r, i)
    straddle = sorted(r for r in first_move
                      if done.get(r, 1 << 30) >= first_move[r])
    return dict(n_moves=len(sim["moves"]), moved_requests=sorted(first_move),
                straddling_requests=straddle,
                window_done=done, first_move=first_move)


def decomposition(sim: dict) -> Dict[int, List[int]]:
    """Per-request prefill chunk decomposition, e.g. {0: [16, 8]}."""
    plens = sim["plens"]
    out: Dict[int, List[int]] = {r: [] for r in range(sim["total"])}
    for it in sim["iters"]:
        for j, r in enumerate(it["slots"]):
            if it["steps"][j] < plens[r] and it["chunks"][j] > 0:
                out[r].append(it["chunks"][j])
    return out


def regimes(sim: dict) -> dict:
    """Iteration counts by regime, in M3-I1's vocabulary."""
    n = sim["mbr"]
    pf = mixed = full = drain = 0
    for r in sim["iters"]:
        if r["n_prefill"] > 0:
            if r["n_decode"] > 0:
                mixed += 1
            else:
                pf += 1
        elif r["n_live"] == n:
            full += 1
        else:
            drain += 1
    return dict(prefill=pf, mixed=mixed, decode_full=full, decode_draining=drain,
                total=sim["n_iterations"])


AC3_PLENS = [24, 30, 32, 32, 33, 36, 40, 43, 44, 68]
I1_ITERATIONS = {1: 109, 2: 109, 4: 109, 8: 111, 16: 203}


def ac3_slots(bs: int) -> List[int]:
    """The adapter's slot fill: the wave's prompts, padded by repetition
    (`MPKOfflineAdapter.generate`)."""
    w = AC3_PLENS[:bs]
    return (w + [w[i % len(w)] for i in range(len(w), bs)])[:bs]


def _self_check() -> int:
    bad = 0
    for bs, want in I1_ITERATIONS.items():
        s = simulate(ac3_slots(bs), 16, 132)
        got = s["n_iterations"]
        ok = got == want
        bad += 0 if ok else 1
        print(f"bs{bs:<3d} iterations={got:4d} expected={want:4d} "
              f"{'OK' if ok else 'MISMATCH'}   live-slot moves={len(s['moves'])}")
    print("self-check:", "PASS" if not bad else f"FAIL ({bad})")
    return bad


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        raise SystemExit(_self_check())
    bs = 16
    for a in sys.argv[1:]:
        if a.startswith("--bs="):
            bs = int(a.split("=")[1])
    s = simulate(ac3_slots(bs), 16, 132)
    print(json.dumps(dict(batch_size=bs, slot_plens=ac3_slots(bs),
                          regimes=regimes(s), audit={
                              k: v for k, v in audit(s).items()
                              if k in ("n_moves", "moved_requests",
                                       "straddling_requests")}), indent=2))
