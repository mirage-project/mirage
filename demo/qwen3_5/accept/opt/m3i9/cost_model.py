#!/usr/bin/env python3
"""M3-I9 -- per-iteration cost model, fit on M3-I1's measured iteration timeline.

M3-I1 published one number per batch size ("the steady decode step").  To price
a *schedule* you need the cost of every iteration it contains, so this fits
`tables/bs<N>_iters.csv` (638 measured iterations, three profiler reps each,
event counts bit-identical across reps) against the schedule replay.

The model, per batch size:

    iter_us = a + b * max_chunk + c * n_live

`max_chunk` is the largest number of tokens any single slot contributes to the
iteration.  It beats the token total decisively at bs16 (R2 0.960 vs 0.510),
which is the mechanism this issue turns on: an iteration's critical path is one
slot's *sequential* work -- the GDN recurrent scan is `grid_dim=(v_heads, mbr)`,
one block per slot, and that block walks its slot's tokens in order -- so 16
tokens in one slot cost ~1.66x the same 16 tokens spread one per slot.

`--check` is the closure gate: the model must reproduce all five measured
profiled wave wall times.  It does, to <=0.04%.

Fits are on PROFILED iterations, so predictions are on the profiled clock;
M3-I1 measured profiling overhead at 2.85-3.59% (`PROF_OVERHEAD`).  Ratios
between policies cancel it; absolute unprofiled predictions divide by it.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict

from protocol_sim import ac3_slots, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)

# fitted by `--fit` from tables/bs<N>_iters.csv (values pinned so the tool runs
# without the 100 MB trace tables; `--fit` re-derives and must reproduce them)
COEF: Dict[int, tuple] = {
    1:  (7447.0, 390.9, 7446.9),
    2:  (15765.0, 370.8, -207.6),
    4:  (16050.0, 348.8, -172.7),
    8:  (18040.0, 449.6, 36.2),
    16: (18917.0, 982.4, 142.7),
}
# measured wave wall (ms), M3-I1 attribution.csv
MEAS_PROFILED = {1: 1674.998, 2: 1730.556, 4: 1746.802, 8: 2182.658, 16: 4695.247}
MEAS_UNPROFILED = {1: 1616.9, 2: 1670.2, 4: 1686.5, 8: 2116.7, 16: 4566.5}
PROF_OVERHEAD = {1: 1.0358, 2: 1.0359, 4: 1.0358, 8: 1.0320, 16: 1.0285}

# --- mbt extrapolation -------------------------------------------------------
# Every measured iteration is at mbt=16 (the MoE router 16-row cap, M2-I9), so
# any mbt!=16 number is an EXTRAPOLATION.  Source-level audit of the builder
# call sites splits the step into three classes:
#   grid ~ mbt  (quantize, moe_silu_mul, moe combine, argmax/embed)  -> tasks
#       scale, per-task work constant; quantize runs at concurrency 33/128 so
#       there is width headroom: wall ~flat.
#   grid fixed, per-task row loop over mbt (LINEAR_FP8_BLOCKSCALE 2973 us,
#       LINEAR_SM100 921 us, the router 632 us)                      -> wall ~x2
#       per doubling.
#   grid ~ mbr (GDN conv/recurrent, attention)                       -> flat.
# MoE w13/w2 (5009+2641 us) sit between: grid_x = min(256, mbt*topk) saturates
# at mbt=32, and worker depth is ceil(live/128) (M3-I8), so ~x1.5 per doubling
# until it saturates.
# => increment per mbt doubling ~ (2973+921+632) + 0.5*(5009+2641) = 8351 us on
#    a 23942 us wall-span basis = 34.9% of the step.  DELTA is that fraction;
#    the band is reported, never a point estimate.
DELTA_BAND = (0.15, 0.35, 0.55)


def cost_us(bs: int, it: dict, mbt: int = 16, delta: float = 0.35,
            step16: float = None) -> float:
    a, b, c = COEF[bs]
    if mbt != 16:
        base = step16 if step16 is not None else MEAS_PROFILED[bs] * 0  # unused
        a = a + delta * (mbt / 16.0 - 1.0) * _step16(bs)
    return a + b * it["max_chunk"] + c * it["n_live"]


def _step16(bs: int) -> float:
    """The measured 16-slot steady step (us) used as the mbt-scaling basis."""
    return {1: 15264.0, 2: 15647.6, 4: 15645.1, 8: 18618.2, 16: 22005.2}[bs]


def wave_ms(bs: int, sim: dict, mbt: int = 16, delta: float = 0.35,
            profiled: bool = True) -> float:
    total = sum(cost_us(bs, it, mbt, delta) for it in sim["iters"]) / 1000.0
    return total if profiled else total / PROF_OVERHEAD[bs]


def fit(bs: int):
    """Re-derive (a, b, c) from the trace tables.  Requires opt/tables/."""
    import numpy as np
    meta = json.load(open(f"{OPT}/meta/meta_bs{bs}_rep0.json"))
    sim = simulate(ac3_slots(bs), meta["mbt"], meta["max_seq_length"])
    meas = list(csv.DictReader(open(f"{OPT}/tables/bs{bs}_iters.csv")))
    X, y = [], []
    for it, row in zip(sim["iters"], meas):
        X.append([1.0, it["max_chunk"], it["n_live"]])
        y.append(float(row["iter_us"]))
    X, y = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - float((res ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return beta, r2, float(abs(res).max()), float(np.median(abs(res))), len(y)


def check() -> int:
    """Closure gate: reproduce every measured profiled wave wall time."""
    bad = 0
    print(f"{'bs':>3s} {'iters':>6s} {'model_ms':>9s} {'measured_ms':>12s} {'err':>8s}")
    for bs in (1, 2, 4, 8, 16):
        sim = simulate(ac3_slots(bs), 16, 132)
        ms = wave_ms(bs, sim)
        err = (ms - MEAS_PROFILED[bs]) / MEAS_PROFILED[bs]
        ok = abs(err) < 0.005
        bad += 0 if ok else 1
        print(f"{bs:3d} {sim['n_iterations']:6d} {ms:9.1f} {MEAS_PROFILED[bs]:12.1f} "
              f"{err:+7.2%} {'OK' if ok else 'FAIL'}")
    print("closure:", "PASS (<0.5% at every batch size)" if not bad else f"FAIL ({bad})")
    return bad


if __name__ == "__main__":
    import sys
    if "--fit" in sys.argv:
        for bs in (1, 2, 4, 8, 16):
            beta, r2, mx, md, n = fit(bs)
            pin = COEF[bs]
            drift = max(abs(beta[i] - pin[i]) for i in range(3))
            print(f"bs{bs:<3d} n={n:4d} a={beta[0]:9.1f} b={beta[1]:7.1f} c={beta[2]:8.1f} "
                  f"R2={r2:.4f} med|res|={md:6.0f}us max|res|={mx:7.0f}us  drift_vs_pinned={drift:.1f}")
        raise SystemExit(0)
    raise SystemExit(check())
