#!/usr/bin/env python3
"""Exact replay of MODE_OFFLINE `prepare_next_batch` (persistent_kernel.cuh
:225-400) so every profiled iteration can be labelled prefill / decode /
retired without guessing.

Why this is needed.  The obvious model -- "prefill for ceil(sum(plens)/mbt)
iterations, then a clean decode steady state" -- is WRONG for this engine.
Step 3 fills the `max_num_batched_tokens` budget **greedily in slot order**:

    remaining = prompt_length - step
    num_new_tokens = min(remaining, MBT - num_tokens)      # still prefilling
                   = min(1,         MBT - num_tokens)      # already decoding

With mbt=16 and slots sorted by ascending prompt length, slot 0 consumes the
whole budget on iteration 0 and every later slot gets nothing.  The result is a
long MIXED phase where the low slots are already decoding while the high slots
have not started, and the first request retires (step+1 >= max_seq_length)
while others are still in prefill.  At bs16 that mixed phase is ~96 of the 203
iterations, which is why the wave-level tok/s figure is not a decode number.

The replay is deterministic given (prompt lengths, mbt, max_seq_length), and
its predicted iteration count is checked against the profiler's own
BEGIN_TASK_GRAPH count -- an independent falsifiable test of the model.
"""
from __future__ import annotations

import json
from typing import List


def simulate(plens: List[int], mbt: int, max_seq_length: int) -> dict:
    """Returns per-iteration state.  Slots are the adapter's wave slots in the
    order it fills them (ascending prompt length, padded by repetition)."""
    n = len(plens)
    step = [0] * n
    live = [True] * n
    iters = []
    guard = 0
    while any(live) and guard < 100000:
        guard += 1
        # --- Step 3: build this iteration's batch (slot-order greedy) ---
        num_tokens = 0
        new = [0] * n
        phase = ["retired"] * n
        for i in range(n):
            if not live[i]:
                continue
            rem = plens[i] - step[i]
            if rem > 0:
                k = min(rem, mbt - num_tokens)
                phase[i] = "prefill"
            else:
                k = min(1, mbt - num_tokens)
                phase[i] = "decode"
            new[i] = k
            num_tokens += k
        if num_tokens == 0:
            break
        iters.append(dict(
            iteration=len(iters), tokens=num_tokens,
            n_live=sum(live),
            n_prefill=sum(1 for i in range(n) if phase[i] == "prefill"),
            n_decode=sum(1 for i in range(n) if phase[i] == "decode"),
            n_decode_active=sum(1 for i in range(n)
                                if phase[i] == "decode" and new[i] > 0),
            n_starved=sum(1 for i in range(n) if live[i] and new[i] == 0),
            steps=list(step),
        ))
        # --- Step 1 of the NEXT prepare: advance and retire ---
        for i in range(n):
            if not live[i]:
                continue
            step[i] += new[i]
            if step[i] + 1 >= max_seq_length:
                live[i] = False
        # slot compaction keeps relative order, which the loop above already
        # respects, so no explicit compaction is needed for this model.
    return dict(n_iterations=len(iters), iters=iters, plens=plens, mbt=mbt,
                max_seq_length=max_seq_length)


def label(sim: dict) -> List[str]:
    """Coarse per-iteration label used by the attribution tables."""
    out = []
    n = len(sim["plens"])
    for r in sim["iters"]:
        if r["n_prefill"] > 0:
            out.append("mixed" if r["n_decode"] > 0 else "prefill")
        elif r["n_live"] == n:
            out.append("decode_full")
        else:
            out.append("decode_draining")
    return out


def regime_key(r: dict) -> tuple:
    return (r["n_live"], r["n_prefill"], r["n_decode_active"], r["tokens"])


def steady_window(sim: dict, min_len: int = 5):
    """The maximal single-regime run that best represents "a step at this
    batch size": PREFILL-FREE first (see the tie-break note below), then most
    live requests, then most tokens in the step, then most of those tokens
    coming from decode, then longest.

    TIE-BREAK FIX (found by the M3-I10 matched-geometry re-measure, validated
    there in `remeasure/opt_fixed/schedule_sim.py`, landed here by M3-I7): the
    original key was (n_live, tokens, n_decode_active, run_length) -- "most
    tokens in the step" ranked ABOVE "most of those tokens being decode". That
    is silently correct only while decode's tokens-per-step (<= batch_size, 1
    per live request) stays below a prefill iteration's mbt-bounded
    tokens-per-step. It holds for the AC-3 geometry (bs 1-8: mbt=16 usually
    exceeds a small bs) but INVERTS for uniform 256-token prompts, where
    prefill iterations are ALSO mbt=16-token-bounded while decode is still
    bs-per-step: at bs1/bs8 (1 and 8 tokens/step) the all-prefill regime (16
    tokens/step, n_live constant, tying the first key) won the old tie-break
    outright, so this returned the PREFILL window instead of decode (bs1 arm A
    returned iterations [8,16), entirely inside the 16-iteration prefill
    phase). Prepending "is this regime prefill-free" fixes the ordering
    (True > False in a tuple compare) and changes nothing else: ties among
    prefill-free regimes, and the case where no prefill-free regime of length
    >= min_len exists (bs16's documented exception), fall through to the exact
    original ranking.

    At bs 1-8 this picks the genuine `decode_full` phase.  At bs16 no such
    phase exists at all -- the first request retires (iteration 101) while
    another is still prefilling -- so it picks (live=16, prefill=1, decode=15,
    tokens=16), an honest 16-token step whose regime tuple is reported next to
    every number derived from it rather than being quietly called "bs16
    decode"."""
    best, best_key = (0, 0), None
    i, n = 0, len(sim["iters"])
    while i < n:
        k = regime_key(sim["iters"][i])
        j = i
        while j < n and regime_key(sim["iters"][j]) == k:
            j += 1
        if j - i >= min_len:
            # (k[1] == 0) FIRST: an n_prefill==0 regime always outranks one
            # with prefill activity, regardless of token count. Original key
            # (k[0], k[3], k[2], j-i) preserved after it for every other tie.
            key = (k[1] == 0, k[0], k[3], k[2], j - i)
            if best_key is None or key > best_key:
                best_key, best = key, (i, j)
        i = j
    if best_key is None:  # nothing long enough: fall back to the longest run
        i, longest = 0, (0, 0, 0)
        while i < n:
            k = regime_key(sim["iters"][i])
            j = i
            while j < n and regime_key(sim["iters"][j]) == k:
                j += 1
            if j - i > longest[0]:
                longest = (j - i, i, j)
            i = j
        best = (longest[1], longest[2])
    return best


if __name__ == "__main__":
    import sys
    meta = json.load(open(sys.argv[1]))
    plens = meta["prompt_lens"]
    bs = meta["batch_size"]
    slots = plens + [plens[i % len(plens)] for i in range(len(plens), bs)]
    sim = simulate(slots, meta["mbt"], meta["max_seq_length"])
    lab = label(sim)
    lo, hi = steady_window(sim)
    print(json.dumps(dict(
        batch_size=bs, slot_plens=slots, n_iterations=sim["n_iterations"],
        n_prefill_or_mixed=sum(1 for x in lab if x in ("prefill", "mixed")),
        n_decode_full=sum(1 for x in lab if x == "decode_full"),
        n_decode_draining=sum(1 for x in lab if x == "decode_draining"),
        steady_window=[lo, hi],
        first_retirement=next((r["iteration"] for r in sim["iters"]
                               if r["n_live"] < bs), None),
    ), indent=2))
