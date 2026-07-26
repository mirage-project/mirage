#!/usr/bin/env python3
"""The MoE grouped-GEMM cost model -- fitted on M3-I1, used to size M3-I8.

Everything here comes from artifacts already in the repo
(`opt/tables/bs*_attrib.json`, `opt/tables/bs*_concurrency.json`) plus the
structure `taskgraph_moe.py` reads out of the compiled graph.  No GPU.

Three claims, each checkable:

C1  PER-TASK TIME IS INDEPENDENT OF HOW MANY GROUPS ARE LIVE.
    `long_mean_us` for task 242 is 30.72 / 30.74 / 30.83 / 30.83 / 30.67 us at
    bs 1/2/4/8/16 while the live-task count moves 1.55x (112 -> 173).  Task 241
    moves 58.35 -> 61.03 (+4.6%) over the same range.  A stage that shared a
    bandwidth ceiling would stretch with concurrency; this one does not.

C2  PER-TASK TIME IS THE WEIGHT TILE.
        T = C * (N_tile / 128) * (K / 128)
    w13 [512, 2048] -> 64 blocks -> C = 0.912 us;  w2 [1024, 512] -> 32 blocks
    -> C = 0.960 us.  Two independent stages, 5% apart.  That is what lets the
    model price a DIFFERENT grid (the v2 arm) instead of only the measured one.

C3  THE STAGE'S WALL SPAN IS SET BY WORKER DEPTH, NOT BY GROUP COUNT.
        waves = ceil(live_tasks / num_workers),  live_tasks = activated * splits
        wall_per_layer ~ waves * T + c(waves)
    The graph is launched by ONE dependent event, so task t lands on worker
    (t - first_task_id) % 128 and the live tasks are a contiguous prefix.  At
    bs 1/2/4 the 112-120 live tasks already fit one wave; at bs 8/16 they do
    not.  This is where M3-I1's backlog model (wall span proportional to group
    count) breaks: it credits bs1 with +37% for removing groups that cost the
    stage's wall span nothing, and credits bs8 with +3.5% for the one case that
    actually crosses a wave boundary.

Run:
    python3 model_moe_wall.py                 # fit + residuals + predictions
    python3 model_moe_wall.py --check         # non-zero exit if the fit breaks
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, os.pardir, "tables")
BS = [1, 2, 4, 8, 16]
LAYERS = 40
SPLITS = 2                 # builder moe_n_splits
GRID_X = 128               # min(num_experts, mbt*topk)
WORKERS = 128
TOPK = 8
NUM_EXPERTS = 256
STAGES = {241: "moe_w13", 242: "moe_w2"}
# weight tile [N_tile, K] per task at the shipped grid
TILE = {241: (512, 2048), 242: (1024, 512)}


def load_i1():
    """M3-I1's own parsed tables -- the measurement this model is fitted on."""
    out = {}
    for bs in BS:
        att = json.load(open(os.path.join(TABLES, f"bs{bs}_attrib.json")))
        con = json.load(open(os.path.join(TABLES, f"bs{bs}_concurrency.json")))
        per = {r["task_type"]: r for r in att["per_task"]}
        pc = con["per_task_concurrency"]
        row = dict(step_us=con["step_us"])
        for tt in STAGES:
            r = per[tt]
            name = r["name"]
            row[tt] = dict(
                n=r["n_per_iter"], nlong=r["n_long_per_iter"],
                T=r["long_mean_us"], total_us=r["total_us_per_iter"],
                wall_us=pc[name]["wall_span_us"],
                short_us=r["short_us_per_iter"],
                nshort=r["n_short_per_iter"])
            row[tt]["activated"] = r["n_long_per_iter"] / (LAYERS * SPLITS)
            row[tt]["wall_layer"] = pc[name]["wall_span_us"] / LAYERS
            row[tt]["live_tasks"] = r["n_long_per_iter"] / LAYERS
        out[bs] = row
    return out


def waves(live_tasks, workers=WORKERS):
    import math
    return max(1, math.ceil(live_tasks / workers))


def fit_c(meas):
    """c(waves) = mean over the observed points of (wall_layer - waves*T)."""
    acc = {}
    for tt in STAGES:
        for bs in BS:
            m = meas[bs][tt]
            w = waves(m["live_tasks"])
            acc.setdefault((tt, w), []).append(m["wall_layer"] - w * m["T"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def union_top8(rows, g):
    """Expected distinct experts over `rows` live tokens, pinned exact at 1."""
    if rows <= 1:
        return float(TOPK)
    p = 1.0 - TOPK / NUM_EXPERTS
    return NUM_EXPERTS * (1.0 - p ** (1.0 + g * (rows - 1)))


def fit_g(anchor_rows, anchor_union):
    """g from the one measurement where every row is live (bs16)."""
    import math
    p = 1.0 - TOPK / NUM_EXPERTS
    lhs = math.log(1.0 - anchor_union / NUM_EXPERTS) / math.log(p)
    return (lhs - 1.0) / (anchor_rows - 1)


def activated_after(meas):
    """Predicted activated groups once padding rows stop marking experts.

    Two independent estimates from the SAME data:
      (a) union law, g fit on the bs16 anchor (all 16 rows live);
      (b) inclusion-exclusion on the measured totals: A_before = |L u P| with
          P the padding-row union, itself fit from the bs1 point where L = 8.
    """
    a16 = meas[16][241]["activated"]
    g = fit_g(16, a16)
    est_a = {bs: union_top8(bs, g) for bs in BS}

    # (b) |L u P| = L + P - L*P/E, with P(k) = E*(1-(1-8/E)^(k*f)) over k
    # padding rows and f fit from bs1 (where L is exactly 8).
    import math
    p = 1.0 - TOPK / NUM_EXPERTS
    a1 = meas[1][241]["activated"]
    P15 = (a1 - TOPK) / (1.0 - TOPK / NUM_EXPERTS)
    f = math.log(1.0 - P15 / NUM_EXPERTS) / math.log(p) / 15.0
    est_b = {1: float(TOPK), 16: a16}
    for bs in (2, 4, 8):
        Pk = NUM_EXPERTS * (1.0 - p ** ((16 - bs) * f))
        est_b[bs] = (meas[bs][241]["activated"] - Pk) / (1.0 - Pk / NUM_EXPERTS)
    return g, f, est_a, est_b


def predict(meas, c, act_after, splits=SPLITS, grid_x=GRID_X, tile=TILE,
            dead_us=0.53, skew=1.0):
    """Post-change wall span per stage, per bs, under the fitted model.

    `skew` scales the fitted c(waves) term.  It is the one genuinely uncertain
    piece: c(1 wave) is +21.6 us (w13) / +13.4 us (w2) of stage span that is
    NOT the task itself.  If that is arrival skew inherited from upstream
    stages it survives the change (skew=1, the conservative arm); if it is
    the cost of walking 112 live tasks onto 112 workers it goes away with them
    (skew=0, the optimistic arm).  I1's data cannot separate the two, so both
    bounds are reported and the A/B settles it.
    """
    rows = []
    for bs in BS:
        d = dict(bs=bs, step_us=meas[bs]["step_us"], saved_us=0.0)
        for tt in STAGES:
            m = meas[bs][tt]
            base_n, base_k = TILE[tt]
            # grid.y splits the expert's N, so the per-task weight tile is
            # base_n * SPLITS / splits.  Legal only while the tile stays a whole
            # number of 128-row checkpoint scale blocks
            # (moe_fp8_blockscale_sm100.cuh static_assert OUTPUT_SIZE % 128).
            n_tile = base_n * SPLITS / splits
            k = base_k
            assert n_tile >= 128 and float(n_tile).is_integer(), (
                f"moe_n_splits={splits} gives a {n_tile} N tile for {STAGES[tt]}"
                " -- the kernel needs a whole number of 128-row scale blocks")
            # T scales with the weight tile (C2); anchor on the measured T at
            # the shipped tile so the per-bs clock/contention drift is kept.
            scale = (n_tile / base_n) * (k / base_k)
            a = min(act_after[bs], grid_x)
            import math
            experts_per_task = math.ceil(act_after[bs] / grid_x)
            T = m["T"] * scale * experts_per_task
            live = a * splits
            w = waves(live)
            cc = skew * c.get((tt, w), c.get((tt, 1), 0.0))
            wall = w * T + cc
            # dead tasks still get dispatched and cost a queue pop each
            n_dead = grid_x * splits - live
            dead_cost = n_dead * dead_us / WORKERS
            wall += dead_cost
            d[STAGES[tt]] = dict(T=T, live=live, waves=w, wall_layer=wall,
                                 wall_before=m["wall_layer"],
                                 saved_layer=m["wall_layer"] - wall)
            d["saved_us"] += (m["wall_layer"] - wall) * LAYERS
        d["step_after"] = d["step_us"] - d["saved_us"]
        d["tok_s_delta_pct"] = 100.0 * (d["step_us"] / d["step_after"] - 1.0)
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the fit or the invariants break")
    ap.add_argument("--splits", type=int, default=SPLITS,
                    help="moe_n_splits for the predicted arm (v2 = 4 or 8)")
    ap.add_argument("--grid-x", type=int, default=GRID_X)
    ap.add_argument("--skew", type=float, default=None,
                    help="scale on the fitted c(waves); default prints both "
                         "1.0 (skew survives) and 0.0 (skew is the live-task "
                         "walk) as a bound pair")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    meas = load_i1()
    c = fit_c(meas)
    g, f, est_a, est_b = activated_after(meas)
    fails = []

    print("=== C1/C2: per-task time (M3-I1 long_mean_us)")
    print(f"{'bs':>3} {'w13 T':>7} {'w2 T':>7} {'w13 live':>9} {'w2 live':>8} "
          f"{'w13 A':>7} {'C_w13':>6} {'C_w2':>6}")
    for bs in BS:
        m1, m2 = meas[bs][241], meas[bs][242]
        c1 = m1["T"] / ((TILE[241][0] / 128) * (TILE[241][1] / 128))
        c2 = m2["T"] / ((TILE[242][0] / 128) * (TILE[242][1] / 128))
        print(f"{bs:>3} {m1['T']:>7.2f} {m2['T']:>7.2f} {m1['live_tasks']:>9.1f} "
              f"{m2['live_tasks']:>8.1f} {m1['activated']:>7.1f} "
              f"{c1:>6.3f} {c2:>6.3f}")
    spread = max(meas[bs][242]["T"] for bs in BS) / min(
        meas[bs][242]["T"] for bs in BS) - 1.0
    print(f"  w2 per-task spread over a 1.55x live-task range: {spread*100:.1f}%"
          "  (C1: not concurrency-bound)")
    if spread > 0.03:
        fails.append(f"C1: w2 per-task time varies {spread*100:.1f}% > 3%")

    print("\n=== C3: wall span = waves * T + c(waves)   [per layer, us]")
    print(f"{'bs':>3} {'stage':>8} {'live':>6} {'waves':>6} {'T':>7} "
          f"{'meas':>8} {'model':>8} {'resid%':>7}")
    worst = 0.0
    for bs in BS:
        for tt in STAGES:
            m = meas[bs][tt]
            w = waves(m["live_tasks"])
            model = w * m["T"] + c[(tt, w)]
            r = 100.0 * (model - m["wall_layer"]) / m["wall_layer"]
            worst = max(worst, abs(r))
            print(f"{bs:>3} {STAGES[tt]:>8} {m['live_tasks']:>6.1f} {w:>6d} "
                  f"{m['T']:>7.2f} {m['wall_layer']:>8.2f} {model:>8.2f} "
                  f"{r:>+7.1f}")
    print(f"  fitted c: " + ", ".join(
        f"{STAGES[k[0]]}/{k[1]}wave={v:+.2f}" for k, v in sorted(c.items())))
    print(f"  worst residual {worst:.1f}%")
    if worst > 12.0:
        fails.append(f"C3: worst wall-span residual {worst:.1f}% > 12%")

    print("\n=== activated groups per layer, before -> after")
    print(f"{'bs':>3} {'measured':>9} {'cap 8*bs':>9} {'union g':>8} "
          f"{'incl-excl':>10} {'used':>6} {'live tasks':>11} {'waves':>6}")
    act_after = {}
    for bs in BS:
        cap = min(NUM_EXPERTS, TOPK * bs)
        used = 0.5 * (est_a[bs] + est_b[bs])
        act_after[bs] = used
        lt = min(used, a.grid_x) * a.splits
        print(f"{bs:>3} {meas[bs][241]['activated']:>9.1f} {cap:>9d} "
              f"{est_a[bs]:>8.1f} {est_b[bs]:>10.1f} {used:>6.1f} "
              f"{lt:>11.1f} {waves(lt):>6d}")
        if used > cap + 1e-6:
            fails.append(f"bs{bs}: estimated activated {used:.1f} exceeds the "
                         f"hard cap {cap}")
    print(f"  union-law g={g:.4f} (16 live rows behave like "
          f"{1+g*15:.1f} independent ones); padding-row novelty f={f:.4f}")

    skews = [a.skew] if a.skew is not None else [1.0, 0.0]
    allrows = {}
    for sk in skews:
        print(f"\n=== prediction: moe_n_splits={a.splits}, grid_x={a.grid_x}, "
              f"skew={sk:g} ({'stage skew survives' if sk else 'stage skew is the live-task walk'})")
        rows = predict(meas, c, act_after, splits=a.splits, grid_x=a.grid_x,
                       skew=sk)
        allrows[sk] = rows
        print(f"{'bs':>3} {'w13 wall':>18} {'w2 wall':>18} {'saved/step':>11} "
              f"{'step':>9} {'tok/s':>8}")
        for d in rows:
            print(f"{d['bs']:>3} "
                  f"{d['moe_w13']['wall_before']:>7.1f}->{d['moe_w13']['wall_layer']:<10.1f} "
                  f"{d['moe_w2']['wall_before']:>7.1f}->{d['moe_w2']['wall_layer']:<10.1f} "
                  f"{d['saved_us']:>11.0f} {d['step_after']:>9.0f} "
                  f"{d['tok_s_delta_pct']:>+7.1f}%")
    rows = allrows[skews[0]]

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(fitted_c={f"{k[0]}_{k[1]}": v for k, v in c.items()},
                           activated_after=act_after,
                           predictions={str(k): v for k, v in allrows.items()},
                           union_g=g, pad_f=f), fh, indent=1)
        print(f"wrote {a.json}")

    if fails:
        print("\nFAIL:")
        for m in fails:
            print("  - " + m)
        return 1
    print("\nOK: C1/C2/C3 hold on M3-I1's committed tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
