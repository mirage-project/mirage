#!/usr/bin/env python3
"""M3-I9 C7 closure -- refit / re-anchor the bs16 per-iteration cost law.

The pre-registered C7 falsifier fired: `iter_us = a + b*max_chunk + c*n_live`
(R2 0.960 on M3-I1's 202 measured bs16 iterations) predicted 4050-4350 ms
profiled for `--slot-order sorted-padded`, and the window measured 3437 ms
profiled (3341 ms unprofiled, 3 reps, spread 0.41%).

This script separates the two things that miss can mean:

  SHAPE   -- the functional form does not transfer to a new iteration mix.
  SCALE   -- the form transfers but the coefficients were fit on a BINARY that
             no longer exists (M3-I1 predates 624e8e1, the quantize
             row-partition fix, which is default-ON for qwen3.5).

The separator is that the SAME window also measured the shipped-order schedule
(stage 0, 3689.27 ms) and a longer-context shipped-order schedule (stage 1,
msl=212, 5282.53 ms).  A scale error cancels in a RATIO between two runs of the
same window; a shape error does not.  Every candidate model is therefore scored
on held-out RATIOS, not on absolute milliseconds.

Fit set   : opt/tables/bs16_iters.csv  (M3-I1, 202 profiled iterations)
Held out  : m3i9/results/out/s2_sorted_rep{1,2,3}  (ratio vs stage 0)
            m3i9/results/out/s1_msl212             (ratio vs stage 0)
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M3I9 = os.path.dirname(HERE)
OPT = os.path.dirname(M3I9)
sys.path.insert(0, M3I9)

from protocol_sim import ac3_slots, audit, simulate  # noqa: E402

# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------
# M3-I1 window (opt/attribution.csv): the fit set's own window.
I1_WAVE_PROFILED_MS = 4696.8
I1_WAVE_UNPROFILED_MS = 4566.5
PROF_OVERHEAD_16 = 1.0285

# M3-I9 window (results/out/*/timings_bs16.json), all unprofiled wall_ms.
S0_SHIPPED_MS = 3689.2685546875            # n=1
S1_MSL212_MS = 5282.529296875              # n=1
S2_SORTED_MS = [3340.61083984375, 3341.490478515625, 3354.408447265625]

AC3 = ac3_slots(16)
SORTED_ORDER = sorted(range(16), key=lambda i: AC3[i])


def sim_for(order=None, cap=None, msl=132, mbt=16, hold=False):
    pl = [AC3[i] for i in order] if order is not None else list(AC3)
    return simulate(pl, mbt, msl, cap=cap, hold_decode=hold)


POLICIES = {
    "shipped(msl132)": dict(),
    "sorted-padded(msl132)": dict(order=SORTED_ORDER),
    "shipped(msl212)": dict(msl=212),
    "cap1": dict(cap=1),
    "cap2": dict(cap=2),
    "cap4": dict(cap=4),
    "cap8": dict(cap=8),
    "hold-decode": dict(hold=True),
}
SIMS = {k: sim_for(**v) for k, v in POLICIES.items()}


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def feats(it, msl=132):
    mc = float(it["max_chunk"])
    nl = float(it["n_live"])
    tk = float(it["tokens"])
    npf = float(it["n_prefill"])
    return dict(one=1.0, max_chunk=mc, n_live=nl, tokens=tk, n_prefill=npf,
                is_prefill=1.0 if npf > 0 else 0.0,
                log_chunk=float(np.log2(mc)) if mc > 0 else 0.0,
                sqrt_chunk=float(np.sqrt(mc)),
                chunk_m1=mc - 1.0,
                excess_tokens=tk - mc,          # tokens NOT on the critical slot
                ceil2=float(np.ceil(mc / 2.0)),
                ceil4=float(np.ceil(mc / 4.0)),
                nl_x_mc=nl * mc)


MODELS = {
    "M0 const":                     ["one"],
    "M1 a+b*chunk":                 ["one", "max_chunk"],
    "M2 a+c*live":                  ["one", "n_live"],
    "M3 a+d*tokens":                ["one", "tokens"],
    "M4 SHIPPED a+b*chunk+c*live":  ["one", "max_chunk", "n_live"],
    "M5 +tokens":                   ["one", "max_chunk", "n_live", "tokens"],
    "M6 +n_prefill":                ["one", "max_chunk", "n_live", "n_prefill"],
    "M7 +prefill dummy":            ["one", "max_chunk", "n_live", "is_prefill"],
    "M8 log2(chunk)":               ["one", "log_chunk", "n_live"],
    "M9 sqrt(chunk)":               ["one", "sqrt_chunk", "n_live"],
    "M10 ceil(chunk/2)":            ["one", "ceil2", "n_live"],
    "M11 ceil(chunk/4)":            ["one", "ceil4", "n_live"],
    "M12 chunk + excess_tokens":    ["one", "max_chunk", "n_live", "excess_tokens"],
    "M13 + live*chunk":             ["one", "max_chunk", "n_live", "nl_x_mc"],
}


def design(iters, cols, msl=132):
    return np.array([[feats(it, msl)[c] for c in cols] for it in iters])


def fit_model(cols):
    sim = SIMS["shipped(msl132)"]
    meas = list(csv.DictReader(open(f"{OPT}/tables/bs16_iters.csv")))
    its = sim["iters"][:len(meas)]
    X = design(its, cols)
    y = np.array([float(r["iter_us"]) for r in meas])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - float((res ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return beta, r2, float(np.median(abs(res))), float(abs(res).max()), len(y)


def total_ms(cols, beta, key):
    sim = SIMS[key]
    msl = 212 if "212" in key else 132
    X = design(sim["iters"], cols, msl)
    return float((X @ beta).sum() / 1000.0)


# --------------------------------------------------------------------------
def main():
    med_sorted = float(np.median(S2_SORTED_MS))
    r_sorted_meas = med_sorted / S0_SHIPPED_MS
    r_msl212_meas = S1_MSL212_MS / S0_SHIPPED_MS
    print("=" * 100)
    print("SAME-WINDOW MEASUREMENTS (M3-I9, unprofiled wall_ms)")
    print(f"  stage 0  shipped   msl=132  {S0_SHIPPED_MS:9.2f} ms  (n=1)")
    print(f"  stage 2  sorted    msl=132  {med_sorted:9.2f} ms  (median of 3, "
          f"spread {100*(max(S2_SORTED_MS)-min(S2_SORTED_MS))/med_sorted:.2f}%)")
    print(f"  stage 1  shipped   msl=212  {S1_MSL212_MS:9.2f} ms  (n=1)")
    print(f"  measured ratio sorted/shipped  = {r_sorted_meas:.5f}")
    print(f"  measured ratio msl212/shipped  = {r_msl212_meas:.5f}")
    print()
    print("CROSS-WINDOW BASELINE DRIFT (same 203-iteration schedule, same prompts, msl=132)")
    drift = S0_SHIPPED_MS / I1_WAVE_UNPROFILED_MS - 1
    print(f"  M3-I1 unprofiled {I1_WAVE_UNPROFILED_MS:.1f} ms -> M3-I9 {S0_SHIPPED_MS:.1f} ms"
          f"   = {drift:+.2%}")
    print(f"  => any ABSOLUTE ms predicted from M3-I1-fitted coefficients is high by "
          f"{-drift/(1+drift):+.1%} before any shape question is asked.")
    print()

    print("=" * 100)
    print("ITERATION-SHAPE LEVERAGE (how far each policy sits outside the fit set)")
    hdr = f"{'policy':24s} {'iters':>6s} {'mean_mc':>8s} {'max_mc':>7s} {'mean_live':>10s} {'%mc=1':>7s} {'%mc>=8':>7s}"
    print(hdr)
    for k, s in SIMS.items():
        mc = np.array([it["max_chunk"] for it in s["iters"]], float)
        nl = np.array([it["n_live"] for it in s["iters"]], float)
        print(f"{k:24s} {len(mc):6d} {mc.mean():8.3f} {mc.max():7.0f} {nl.mean():10.3f} "
              f"{100*(mc == 1).mean():6.1f}% {100*(mc >= 8).mean():6.1f}%")
    print()

    print("=" * 100)
    print("REFIT TABLE -- fit on M3-I1 shipped-order ONLY, scored on same-window HELD-OUT ratios")
    print(f"{'model':30s} {'R2':>7s} {'med|r|':>7s} {'max|r|':>7s} "
          f"{'r_sorted':>9s} {'err%':>7s} {'r_212':>7s} {'err%':>7s} "
          f"{'absMS_sorted':>12s} {'absErr%':>8s}")
    rows = []
    for name, cols in MODELS.items():
        beta, r2, med, mx, n = fit_model(cols)
        t0 = total_ms(cols, beta, "shipped(msl132)")
        ts = total_ms(cols, beta, "sorted-padded(msl132)")
        t2 = total_ms(cols, beta, "shipped(msl212)")
        r_s, r_2 = ts / t0, t2 / t0
        e_s = r_s / r_sorted_meas - 1
        e_2 = r_2 / r_msl212_meas - 1
        # what the window actually did: absolute profiled prediction vs measured
        abs_meas_prof = med_sorted * PROF_OVERHEAD_16
        e_abs = ts / abs_meas_prof - 1
        rows.append(dict(name=name, cols=cols, beta=beta, r2=r2, med=med, mx=mx,
                         t0=t0, ts=ts, t2=t2, r_s=r_s, r_2=r_2, e_s=e_s,
                         e_2=e_2, e_abs=e_abs))
        print(f"{name:30s} {r2:7.4f} {med:7.0f} {mx:7.0f} "
              f"{r_s:9.5f} {100*e_s:+6.2f}% {r_2:7.4f} {100*e_2:+6.2f}% "
              f"{ts:12.0f} {100*e_abs:+7.1f}%")
    print()
    print("  r_sorted  = predicted total(sorted-padded) / predicted total(shipped)")
    print(f"  err%      = that ratio vs the MEASURED same-window ratio {r_sorted_meas:.5f}")
    print("  absMS     = the M3-I1-anchored absolute profiled prediction (what predictions.md quoted)")
    print(f"  absErr%   = vs measured {med_sorted*PROF_OVERHEAD_16:.0f} ms profiled -- "
          f"this column is the reported 'miss'")
    print()

    # ---------------------------------------------------------------- re-anchor
    print("=" * 100)
    print("RE-ANCHORED CAP PREDICTIONS (unprofiled ms, M3-I9-window binary)")
    print("  prediction = 3689.27 ms (same-window shipped control) x model ratio")
    keys = ["cap1", "cap2", "cap4", "cap8", "hold-decode", "sorted-padded(msl132)"]
    print(f"{'model':30s} " + " ".join(f"{k:>14s}" for k in keys))
    band = {k: [] for k in keys}
    for r in rows:
        if r["r2"] < 0.5:
            continue
        cells = []
        for k in keys:
            t = total_ms(r["cols"], r["beta"], k)
            ms = S0_SHIPPED_MS * t / r["t0"]
            band[k].append(ms)
            cells.append(f"{ms:14.0f}")
        print(f"{r['name']:30s} " + " ".join(cells))
    print()
    print(f"{'ACROSS-MODEL SPREAD':30s} " +
          " ".join(f"{min(band[k]):6.0f}-{max(band[k]):<7.0f}" for k in keys))
    print(f"{'speedup vs shipped':30s} " +
          " ".join(f"{S0_SHIPPED_MS/max(band[k]):5.3f}-{S0_SHIPPED_MS/min(band[k]):<8.3f}"
                   for k in keys))
    print()

    # ------------------------------------------- 2-parameter re-anchor identify
    print("=" * 100)
    print("IS THE DRIFT SHAPE-NEUTRAL?  2-parameter re-anchor, exactly identified")
    print("  cost' = s*(a + c*n_live) + t*b*max_chunk ; solve (s,t) from the TWO")
    print("  same-window msl=132 totals (shipped, sorted).  t != 1 would mean the")
    print("  new binary changed the chunk term specifically.")
    base = [r for r in rows if r["name"].startswith("M4")][0]
    a, b, c = base["beta"]
    def parts(key):
        s = SIMS[key]
        X = design(s["iters"], ["one", "max_chunk", "n_live"],
                   212 if "212" in key else 132)
        flat = float((X[:, 0] * a + X[:, 2] * c).sum() / 1000.0)
        chunk = float((X[:, 1] * b).sum() / 1000.0)
        return flat, chunk
    f0, k0 = parts("shipped(msl132)")
    f1, k1 = parts("sorted-padded(msl132)")
    A = np.array([[f0, k0], [f1, k1]])
    yv = np.array([S0_SHIPPED_MS * PROF_OVERHEAD_16, med_sorted * PROF_OVERHEAD_16])
    st = np.linalg.solve(A, yv)
    print(f"  s (flat+live scale) = {st[0]:.4f}   t (chunk-term scale) = {st[1]:.4f}")
    print(f"  1-parameter (shape-neutral) scale from stage 0 alone: "
          f"{S0_SHIPPED_MS*PROF_OVERHEAD_16/base['t0']:.4f}")
    print(f"{'policy':24s} {'1-param ms':>11s} {'2-param ms':>11s} {'spread':>8s}")
    s1 = S0_SHIPPED_MS / base["t0"]
    for k in ["cap1", "cap2", "cap4", "cap8", "shipped(msl212)"]:
        f, kk = parts(k)
        p1 = s1 * (f + kk)
        p2 = (st[0] * f + st[1] * kk) / PROF_OVERHEAD_16
        print(f"{k:24s} {p1:11.0f} {p2:11.0f} {100*abs(p2-p1)/p1:7.1f}%")
    print()

    # ------------------------------------------------------------- migrations
    print("=" * 100)
    print("CORRECTNESS SIDE (unchanged by any pricing question)")
    print(f"{'policy':24s} {'iters':>6s} {'migrations':>11s} {'straddling':>11s}")
    for k in ["shipped(msl132)", "sorted-padded(msl132)", "cap1", "cap2",
              "cap4", "cap8"]:
        s = SIMS[k]
        au = audit(s, 64)
        print(f"{k:24s} {s['n_iterations']:6d} {au['n_moves']:11d} "
              f"{len(au['straddling_requests']):11d}")

    # ------------------------------------------------- affine re-anchor + test
    print()
    print("=" * 100)
    print("AFFINE RE-ANCHOR  new_us = alpha + beta * old_us")
    print("  M3-I8's window measured the bs16 steady step on the CURRENT binary:")
    print("  22002 us (M3-I1, n=19, spread 0.49%) -> 18736 us (M3-I8 base arm,")
    print("  I8's own change proven inert at bs16).  That shape -- (max_chunk=1,")
    print("  n_live=16) -- is EXACTLY every one of cap=1's 131 iterations.")
    meas = list(csv.DictReader(open(f"{OPT}/tables/bs16_iters.csv")))
    y = np.array([float(r["iter_us"]) for r in meas])
    sim0 = SIMS["shipped(msl132)"]
    mc = np.array([it["max_chunk"] for it in sim0["iters"]][:len(y)])
    nl = np.array([it["n_live"] for it in sim0["iters"]][:len(y)])
    sel = (mc == 1) & (nl == 16)
    old_step, new_step = float(y[sel].mean()), 18736.0
    # constraint 2: the affine map must reproduce stage 0 on the profiled clock
    tgt0 = S0_SHIPPED_MS * PROF_OVERHEAD_16 * 1000.0            # us, 203 iters
    # extend the 202 measured to 203 with the fitted law for the missing tail it
    y_full = np.concatenate([y, [y[-1]]])
    A = np.array([[1.0, old_step], [len(y_full), y_full.sum()]])
    beta_af = np.linalg.solve(A, np.array([new_step, tgt0]))
    al, be = float(beta_af[0]), float(beta_af[1])
    print(f"  alpha = {al:+.1f} us   beta = {be:.4f}   "
          f"(pure-multiplicative would be alpha=0)")
    # held-out: predict sorted-padded with the affine map applied to the LAW
    base = [r for r in rows if r["name"].startswith("M4")][0]
    def law_us(key):
        s = SIMS[key]
        X = design(s["iters"], ["one", "max_chunk", "n_live"],
                   212 if "212" in key else 132)
        return X @ base["beta"]
    for key, meas_ms in (("sorted-padded(msl132)", med_sorted),
                         ("shipped(msl212)", S1_MSL212_MS)):
        p = float((al + be * law_us(key)).sum() / 1000.0) / PROF_OVERHEAD_16
        print(f"  held-out {key:24s} affine {p:8.0f} ms  measured {meas_ms:8.0f} ms"
              f"   {p/meas_ms-1:+.2%}")
    p_uni = S0_SHIPPED_MS * base["ts"] / base["t0"]
    print(f"  (uniform-scale, same held-out: sorted {p_uni:.0f} ms "
          f"{p_uni/med_sorted-1:+.2%})")
    print()
    print("FOUR INDEPENDENT PRICES FOR cap=1 AT bs16 (unprofiled ms, current binary)")
    cap1_n = SIMS["cap1"]["n_iterations"]
    prices = {
        "A direct measured-regime (131 x I8's 18736 us)":
            cap1_n * new_step / 1000.0 / PROF_OVERHEAD_16,
        "B shape-free null (131/203 x stage 0)":
            S0_SHIPPED_MS * cap1_n / sim0["n_iterations"],
        "C affine re-anchor (alpha,beta above)":
            float((al + be * law_us("cap1")).sum() / 1000.0) / PROF_OVERHEAD_16,
        "D uniform rescale of the M3-I1 law":
            S0_SHIPPED_MS * total_ms(base["cols"], base["beta"], "cap1") / base["t0"],
    }
    for k, v in prices.items():
        print(f"  {k:52s} {v:7.0f} ms   {S0_SHIPPED_MS/v:5.3f}x  "
              f"{S0_SHIPPED_MS/v-1:+.1%}")
    lo, hi = min(prices.values()), max(prices.values())
    print(f"  {'BAND':52s} {lo:.0f}-{hi:.0f} ms   "
          f"{S0_SHIPPED_MS/hi:.3f}-{S0_SHIPPED_MS/lo:.3f}x")
    print()

    out = dict(
        affine=dict(alpha=al, beta=be, old_step=old_step, new_step=new_step,
                    cap1_prices=prices),
        measured=dict(s0=S0_SHIPPED_MS, s1_msl212=S1_MSL212_MS,
                      s2_sorted=S2_SORTED_MS, s2_median=med_sorted,
                      r_sorted=r_sorted_meas, r_msl212=r_msl212_meas),
        i1_baseline=dict(profiled=I1_WAVE_PROFILED_MS,
                         unprofiled=I1_WAVE_UNPROFILED_MS, drift=drift),
        models=[{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in r.items() if k != "cols"} for r in rows],
        reanchored={k: dict(min=min(band[k]), max=max(band[k]),
                            n_models=len(band[k])) for k in keys},
        two_param=dict(s=float(st[0]), t=float(st[1])),
    )
    with open(os.path.join(HERE, "costlaw_refit.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
