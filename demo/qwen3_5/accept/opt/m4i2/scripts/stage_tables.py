#!/usr/bin/env python3
"""M4-I2: the dense-fp8 stage wallspan before/after, and the WIDTH RESIDUAL.

Reads the per-arm concurrency.py outputs and reports, for one steady-window
decode iteration:

  step_us                the whole iteration
  dense_proj total_us    the stage's WORK (sum of its 160 tasks' durations)
  dense_proj wall_span   the stage's WALLSPAN -- what the step actually pays
  mean_conc_during       worker CTAs inside a task while the stage runs (of 128)

and derives:

  work speedup    A.total_us / B.total_us
      the kernel-level effect, seen through MPK's own dispatch.
  wallspan gain   A.wall_span - B.wall_span
      what the step actually got back.
  width residual  wall_span - total_us / NW
      the stage's cost BEYOND a perfectly-wide execution of the same work. This
      is the number M4-I5 owns: driving the kernel to zero cannot remove it.
  residual share  width_residual / wall_span
      how much of the stage's remaining wallspan is width, not kernel.

WHY THE ARITHMETIC IS STATED THIS WAY. M3-I7 put dense fp8 at 2.07x slower than
vLLM as a stage, and the ferret kernel now measures at parity standalone
(min_ratio 1.011). If the integrated stage does NOT improve by ~2x, the gap did
not vanish -- it moved, and `width_residual` is where it went. A stage whose
wall_span is dominated by its residual is width-bound, and no further kernel work
on it pays.
"""
import argparse
import glob
import json
import os
import re
import sys

NW = 128           # MPK worker CTAs (concurrency.py's own NW)
STAGE = "dense_proj"   # task 279, TASK_LINEAR_FP8_BLOCKSCALE_SM100 (trace_lib.py)
ARM_LABEL = {"A": "base(slice128+golden)", "B": "new(ferret v011)"}


def load(stage_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(stage_dir, "conc_*.json"))):
        m = re.match(r"conc_([AB])_bs(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        out[(m.group(1), int(m.group(2)))] = json.load(open(f))
    return out


def stage_row(d):
    per = d.get("per_task_concurrency", {})
    s = per.get(STAGE)
    if s is None:
        return None
    total = s["total_us"]
    span = s["wall_span_us"]
    return {
        "n_tasks": s["n"],
        "total_us": total,
        "wall_span_us": span,
        "mean_conc_during": s["mean_concurrency_during"],
        "ideal_span_us": total / NW,
        "width_residual_us": span - total / NW,
        "residual_share": (span - total / NW) / span if span else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="/var/tmp/m4i2_prof/stage")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.stage_dir)
    if not d:
        print(f"no conc_*.json under {a.stage_dir}", file=sys.stderr)
        return 2
    bss = sorted({k[1] for k in d})

    L = []
    P = L.append
    P("M4-I2 dense-fp8 stage wallspan (task 279 / dense_proj), ONE steady-window")
    P("decode iteration, profiler ON -- DIAGNOSTIC attribution. The performance")
    P("claim is the --no-profiler A/B in tables/m4i2_tables.txt.")
    P("")
    P(f"arm A = {ARM_LABEL['A']}      arm B = {ARM_LABEL['B']}      NW = {NW}")
    P("")
    P("== the whole step, and the dense stage inside it ==")
    P(f"{'bs':>3} {'arm':>3} {'step_us':>10} {'tasks':>6} {'work_us':>10} "
      f"{'span_us':>10} {'span/step':>9} {'meanconc':>9}")
    for bs in bss:
        for arm in ("A", "B"):
            if (arm, bs) not in d:
                continue
            row = stage_row(d[(arm, bs)])
            st = d[(arm, bs)]["step_us"]
            if row is None:
                P(f"{bs:>3} {arm:>3} {st:10.1f}   {STAGE} absent")
                continue
            P(f"{bs:>3} {arm:>3} {st:10.1f} {row['n_tasks']:>6} "
              f"{row['total_us']:10.1f} {row['wall_span_us']:10.1f} "
              f"{row['wall_span_us']/st:9.3f} {row['mean_conc_during']:9.1f}")
    P("")
    P("== before -> after, and the width residual M4-I5 owns ==")
    rows = []
    for bs in bss:
        if ("A", bs) not in d or ("B", bs) not in d:
            continue
        A, B = stage_row(d[("A", bs)]), stage_row(d[("B", bs)])
        if A is None or B is None:
            continue
        sA, sB = d[("A", bs)]["step_us"], d[("B", bs)]["step_us"]
        r = {
            "bs": bs,
            "step_us_A": sA, "step_us_B": sB,
            "work_us_A": A["total_us"], "work_us_B": B["total_us"],
            "work_speedup": A["total_us"] / B["total_us"],
            "span_us_A": A["wall_span_us"], "span_us_B": B["wall_span_us"],
            "span_speedup": A["wall_span_us"] / B["wall_span_us"],
            "span_gain_us": A["wall_span_us"] - B["wall_span_us"],
            "step_gain_us": sA - sB,
            "ideal_span_us_B": B["ideal_span_us"],
            "width_residual_us_B": B["width_residual_us"],
            "residual_share_B": B["residual_share"],
            "mean_conc_A": A["mean_conc_during"],
            "mean_conc_B": B["mean_conc_during"],
        }
        rows.append(r)
        P(f"bs{bs}:")
        P(f"   stage WORK      {A['total_us']:9.1f} -> {B['total_us']:9.1f} us "
          f" ({r['work_speedup']:.3f}x)")
        P(f"   stage WALLSPAN  {A['wall_span_us']:9.1f} -> {B['wall_span_us']:9.1f} us "
          f" ({r['span_speedup']:.3f}x, -{r['span_gain_us']:.1f} us)")
        P(f"   step            {sA:9.1f} -> {sB:9.1f} us  (-{r['step_gain_us']:.1f} us)")
        P(f"   mean concurrency during the stage  {A['mean_conc_during']:.1f} "
          f"-> {B['mean_conc_during']:.1f}  of {NW}")
        P(f"   WIDTH RESIDUAL (after) {B['width_residual_us']:.1f} us = "
          f"{100*B['residual_share']:.1f}% of the stage's remaining wallspan")
        P(f"     (ideal span at full width = work/{NW} = "
          f"{B['ideal_span_us']:.1f} us)")
        P("")
    P("READING IT: 'work' is what the kernel change bought inside MPK's own")
    P("dispatch. 'wallspan' is what the step got. The WIDTH RESIDUAL is the part")
    P("of the stage's remaining cost that a faster kernel cannot remove, because")
    P("the stage is not running at the machine's width -- that is M4-I5's lever,")
    P("not another round of kernel work on task 279.")

    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "stage_wallspan.txt"), "w").write(txt)
    json.dump({"NW": NW, "stage": STAGE, "arms": ARM_LABEL, "rows": rows},
              open(os.path.join(a.out, "stage_wallspan.json"), "w"), indent=1)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
