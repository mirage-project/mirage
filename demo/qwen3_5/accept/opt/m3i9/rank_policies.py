#!/usr/bin/env python3
"""M3-I9 -- rank admission/scheduling policies, and re-derive backlog #4.

Every number is a prediction from `protocol_sim` (schedule) x `cost_model`
(measured per-iteration cost).  Nothing here is measured; `plan_m3i9.sh` is what
settles it.  Run with no arguments for the full report.
"""
from __future__ import annotations

from cost_model import (COEF, DELTA_BAND, MEAS_PROFILED, MEAS_UNPROFILED,
                        PROF_OVERHEAD, wave_ms)
from protocol_sim import ac3_slots, audit, simulate

BS = 16
AC3 = dict(name="AC-3 (24-68 in, msl=132)", plens=ac3_slots(16), msl=132, window=64)
BENCH = dict(name="pinned 256/1024", plens=[256] * 16, msl=1280, window=1024)


def price(geo, mbt=16, cap=None, order=None, hold=False, delta=0.35, bs=BS):
    pl = geo["plens"]
    if order is not None:
        pl = [pl[i] for i in order]
    sim = simulate(pl, mbt, geo["msl"], cap=cap, hold_decode=hold)
    ms = wave_ms(bs, sim, mbt=mbt, delta=delta)
    a = audit(sim, geo["window"])
    return sim, ms, a


def table(geo, rows, bs=BS):
    print(f"\n### {geo['name']}, bs{bs}   (predicted; profiled clock)")
    print(f"{'#':>2s} {'policy':38s} {'lane':11s} {'iters':>6s} {'ms':>7s} "
          f"{'vs base':>8s} {'moves':>6s} {'strad':>6s} {'mxchk':>6s}")
    base = None
    out = []
    for tag, lane, kw in rows:
        sim, ms, a = price(geo, bs=bs, **kw)
        if base is None:
            base = ms
        out.append((tag, lane, sim, ms, a, base / ms))
    for i, (tag, lane, sim, ms, a, gain) in enumerate(
            sorted(out, key=lambda r: -r[5])):
        mx = max(it["max_chunk"] for it in sim["iters"])
        print(f"{i+1:2d} {tag:38s} {lane:11s} {sim['n_iterations']:6d} {ms:7.0f} "
              f"{gain:7.3f}x {a['n_moves']:6d} {len(a['straddling_requests']):6d} {mx:6d}")
    return out


ROWS = [
    ("P0 today (wave order, mbt16)",        "shipped",    dict()),
    ("P1 slot order = ascending plen",      "py/adapter", dict(order=sorted(range(16), key=lambda i: ac3_slots(16)[i]))),
    ("P2 slot order = descending plen",     "py/adapter", dict(order=sorted(range(16), key=lambda i: -ac3_slots(16)[i]))),
    ("P3 per-request cap = 1",              "runtime",    dict(cap=1)),
    ("P4 per-request cap = 2",              "runtime",    dict(cap=2)),
    ("P5 per-request cap = 4",              "runtime",    dict(cap=4)),
    ("P6 hold-decode until all prefilled",  "runtime",    dict(hold=True)),
    ("P7 mbt=32 (needs I5b)",               "runtime",    dict(mbt=32)),
    ("P8 mbt=32 + cap=2",                   "runtime",    dict(mbt=32, cap=2)),
    ("P9 mbt=64 + cap=4",                   "runtime",    dict(mbt=64, cap=4)),
    ("P10 mbt=64 (needs I5b)",              "runtime",    dict(mbt=64)),
]


def rederive_44():
    """Backlog #4 said +44% wave-level at bs16.  Show the arithmetic."""
    print("\n### backlog #4 (+44% wave-level at bs16) -- re-derivation")
    sim0, ms0, a0 = price(AC3)
    print(f"  measured wave, profiled          4695.2 ms   model {ms0:.1f} ms  "
          f"({(ms0-MEAS_PROFILED[16])/MEAS_PROFILED[16]:+.2%})")
    print(f"  measured wave, unprofiled        4566.5 ms")
    print("  backlog delta_basis: 36 prefill iterations @ 25.5 ms + 107 decode @ 22.0 ms")
    bl = 36 * 25.5 + 107 * 22.0
    print(f"    36*25.5 + 107*22.0            = {bl:.0f} ms   "
          f"-> 4695.2/{bl:.0f} = {4695.2/bl:.3f}x = {4695.2/bl-1:+.1%}   (backlog: +44%)")
    print(f"    quoted tok/s 234 -> 327        = {327/234-1:+.1%}, i.e. 1070 tok / {bl/1000:.3f} s")
    print( "    -> INCONSISTENT: +43.6% is 4695.2 (PROFILED) / 3272; 234.2 tok/s is the")
    print( "       UNPROFILED 4566.5 ms wave.  One basis or the other, not both.")
    # correction A: price the backlog's own schedule with the measured cost law
    a, b, c = COEF[16]
    pf_iter = a + b * 16 + c * 16     # 16 tokens delivered as one 16-token chunk
    dec_iter = a + b * 1 + c * 16     # 16 tokens delivered one per slot
    A = (36 * pf_iter + 107 * dec_iter) / 1000.0
    print(f"\n  correction A -- price the SAME 143-iteration schedule with the fitted law")
    print(f"    a packed prefill iteration (16 tokens in one slot) = {pf_iter/1000:.1f} ms, not 25.5 ms")
    print(f"      (25.5 ms was the MEAN of the 108 measured mixed iterations, 95% of which")
    print(f"       are starved 1-token-per-slot steps -- the cheap kind, not the packed kind)")
    print(f"    a 16-slot 1-token step = {dec_iter/1000:.1f} ms (measured 22.003 ms, n=18, spread 0.49%)")
    print(f"    36*{pf_iter/1000:.1f} + 107*{dec_iter/1000:.1f} = {A:.0f} ms -> {ms0/A:.3f}x = {ms0/A-1:+.1%}")
    # correction B: the reachable optimum
    simB, msB, aB = price(AC3, cap=1)
    floor = 16 * 131 // 16
    print(f"\n  correction B -- 143 iterations is not the floor, and it is not reachable")
    print(f"    floor at mbt=16,bs=16: every request must walk step 0->131, <=16 tokens/iter")
    print(f"    total token-slots = 16*131 = 2096, /16 = {floor} iterations")
    print(f"    per-request cap=1 attains it exactly: {simB['n_iterations']} iterations, every one a")
    print(f"      16-slot 1-token step, {aB['n_moves']} live-slot moves, max_chunk=1")
    print(f"    {simB['n_iterations']} * {dec_iter/1000:.3f} ms = {msB:.0f} ms -> {ms0/msB:.3f}x = {ms0/msB-1:+.1%}")
    unpro = msB / PROF_OVERHEAD[16]
    print(f"    on the unprofiled clock: {MEAS_UNPROFILED[16]:.1f} -> {unpro:.0f} ms = "
          f"{MEAS_UNPROFILED[16]/unpro-1:+.1%}")
    print(f"\n  VERDICT: +44% is not reproducible as stated.  The same schedule, correctly")
    print(f"  priced, is {ms0/A-1:+.1%}; the reachable optimum is {ms0/msB-1:+.1%}.  Backlog #4's")
    print(f"  MECHANISM was also wrong: it is not 'mbt=16 is too small' (raising mbt needs")
    print(f"  I5b and costs step time); it is 'one slot may take the whole budget'.")
    print(f"\n  metric caveat: wave tok/s = len(wave)*max_decode_steps/wall")
    print(f"  (mpk_engine_run.py:385) = 10*107/4.5686 = 234.2 at bs16 -- 10 DISTINCT prompts,")
    print(f"  not 16 slots, and 107 steps, not the reported 64.  It is a wall-clock proxy, not")
    print(f"  a throughput comparable with the vLLM baseline's 16 real requests.  Policy")
    print(f"  ratios are unaffected; the absolute 234 -> 3018 gap is not a like-for-like ratio.")


def per_bs():
    print("\n### winning policy per batch size: cap = max(1, mbt // bs)")
    print(f"{'bs':>3s} {'cap':>4s} {'base_iters':>11s} {'new_iters':>10s} {'base_ms':>8s} "
          f"{'new_ms':>8s} {'gain':>8s} {'moves':>12s}")
    for bs in (1, 2, 4, 8, 16):
        pl = ac3_slots(bs)
        s0 = simulate(pl, 16, 132)
        cap = max(1, 16 // bs)
        s1 = simulate(pl, 16, 132, cap=cap)
        m0, m1 = wave_ms(bs, s0), wave_ms(bs, s1)
        a0, a1 = audit(s0, 64), audit(s1, 64)
        print(f"{bs:3d} {cap:4d} {s0['n_iterations']:11d} {s1['n_iterations']:10d} "
              f"{m0:8.0f} {m1:8.0f} {m0/m1-1:+7.1%} {a0['n_moves']:5d} -> {a1['n_moves']:<4d}")
    print("  cap only pays where the budget is contended (bs16).  At bs<=8 it throttles a")
    print("  prefill that was not starving anyone, so it must be bound to bs, not global.")


def sensitivity():
    print("\n### mbt sensitivity (mbt!=16 is EXTRAPOLATED -- no measured iteration exists)")
    print(f"{'policy':28s} " + " ".join(f"delta={d:<5.2f}" for d in DELTA_BAND))
    for tag, kw in [("mbt=32", dict(mbt=32)), ("mbt=32 + cap=2", dict(mbt=32, cap=2)),
                    ("mbt=64 + cap=4", dict(mbt=64, cap=4)),
                    ("mbt=16 + cap=1 (no extrap.)", dict(cap=1))]:
        cells = []
        _, base, _ = price(AC3)
        for d in DELTA_BAND:
            _, ms, _ = price(AC3, delta=d, **kw)
            cells.append(f"{base/ms:6.3f}x   ")
        print(f"{tag:28s} " + " ".join(cells))
    print("  delta = share of the step that scales with mbt (source audit: 0.35 central).")
    print("  cap=1 at mbt=16 is the only candidate that needs no extrapolation at all.")


def out_of_scope():
    print("\n### policies ruled OUT on the compaction hazard")
    for tag, kw in [("staggered admission (mbr=8, total=16)", dict(mbr=8, total=16)),
                    ("drain refill (rolling admission)", dict(mbr=16, total=24))]:
        pl = (AC3["plens"] * 2)[:kw.get("total", 16)]
        s = simulate(pl, 16, 132, mbr=kw["mbr"], total=kw["total"])
        a = audit(s, 64)
        print(f"  {tag:42s} iters={s['n_iterations']:4d} live-slot moves={a['n_moves']:4d} "
              f"straddling={len(a['straddling_requests']):3d}  -> OUT")
    print("  Both move live slots by construction and are exactly what")
    print("  `assert_no_rolling_admission` refuses.  They stay OUT until GDN conv/recurrent")
    print("  state migrates with the slot (M2-I9 open item).  cap=1 needs none of that: it")
    print("  REMOVES every live-slot move instead of adding more.")


if __name__ == "__main__":
    table(AC3, ROWS)
    table(BENCH, ROWS)
    rederive_44()
    per_bs()
    sensitivity()
    out_of_scope()
