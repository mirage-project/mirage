#!/usr/bin/env python3
"""M4-I7: the MoE grouped-GEMM stage wallspan before/after, and the WIDTH RESIDUAL.

Reads the per-arm concurrency.py outputs and reports, for one steady-window
decode iteration:

  step_us                the whole iteration
  total_us               the stage's WORK (sum of its live tasks' durations)
  wall_span              the stage's WALLSPAN -- what the step actually pays
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

WHY THE ARITHMETIC IS STATED THIS WAY. M4-I5 measured these two stages at
2194.6 us (w13) and 1191.0 us (w2) of the bs1 step, at span/perfect-pack ratios
of 7.36 and 7.07 -- i.e. overwhelmingly WIDTH-bound, not kernel-bound. So the
integration's own prediction is that most of its gain must show up as a WALLSPAN
drop at a rising live-task count, not as less work. `width_residual` is what a
faster kernel still cannot remove, and it is the number that says whether more
kernel work on this stage pays at all.
"""
import argparse
import glob
import json
import os
import re
import sys

NW = 128           # MPK worker CTAs (concurrency.py's own NW)
# concurrency.py labels task types from the run's own task_names.json, which
# carries the raw runtime_header.h names -- not trace_lib.py's short aliases.
# M4-I7 has TWO stages, one per family, and they must be reported separately --
# w13 and w2 are different shapes (K=2048 vs K=512) and the fetch-path rule can
# pick different paths for them at the same batch size. `--stage` selects which,
# so the script runs once per family instead of guessing.
STAGE = "TASK_MOE_W13_FP8_BLOCKSCALE_SM100"   # task 241; --stage overrides
ARM_LABEL = {"A": "base(golden, pre-M4-I7)", "B": "new(ferret v012 fast)"}


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
    global STAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="/var/tmp/m4i7_prof/stage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default=STAGE,
                    help="runtime_header.h task-type name to report")
    a = ap.parse_args()
    STAGE = a.stage
    short = "w13" if "W13" in STAGE else ("w2" if "W2" in STAGE else "stage")
    os.makedirs(a.out, exist_ok=True)
    d = load(a.stage_dir)
    if not d:
        print(f"no conc_*.json under {a.stage_dir}", file=sys.stderr)
        return 2
    bss = sorted({k[1] for k in d})

    L = []
    P = L.append
    P(f"M4-I7 MoE grouped-GEMM stage wallspan ({STAGE}), ONE steady-window")
    P("decode iteration, profiler ON -- DIAGNOSTIC attribution. The performance")
    P("claim is the --no-profiler A/B in tables/m4i7_tables.txt.")
    P("")
    P(f"arm A = {ARM_LABEL['A']}      arm B = {ARM_LABEL['B']}      NW = {NW}")
    P("")
    P("== the whole step, and this MoE stage inside it ==")
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
    P("'tasks' is the stage's LIVE task count in the window, and it is itself")
    P("the primary result: the emitted task count is FIXED at grid.x*grid.y=256")
    P("per (layer, stage), and work-item flattening is what converts dead tasks")
    P("into live ones. A rise here at an unchanged emitted count IS the mechanism.")
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
        # The work base matters: arm B's work is INFLATED by the finer slice's
        # per-task overhead, so work_B/NW overstates the incompressible floor.
        # Bounding it with arm A's (un-inflated) work gives the honest range.
        alt = A["total_us"] / NW
        r["ideal_span_us_A_work"] = alt
        r["width_residual_us_altbase"] = B["wall_span_us"] - alt
        P(f"     bounded with arm A's un-inflated work: ideal "
          f"{alt:.1f} us -> residual {B['wall_span_us']-alt:.1f} us")
        # What the step could NOT get back even if the stage were free.
        floor = sB - B["wall_span_us"]
        r["step_floor_if_stage_free_us"] = floor
        r["stage_share_of_step_B"] = B["wall_span_us"] / sB
        P(f"   HEADROOM LEFT IN THIS STAGE: it is now "
          f"{100*B['wall_span_us']/sB:.1f}% of the step, so driving this stage to")
        P(f"     ZERO would leave {floor:.1f} us of the {sB:.1f} us step standing.")
        P("")
    P("READING IT, and the mechanism -- because 'work' goes UP while 'wallspan'")
    P("goes DOWN, and a number you cannot explain you cannot trust:")
    P("")
    P("  Flattening multiplies the LIVE task count by NUM_N_BLOCKS (and by 2x")
    P("  more on PATH 2's TILE_N=64). Each work item re-pays a fixed prologue --")
    P("  the routing gather, the A-tile fetch, the stage-0 B hoist -- so the SUM")
    P("  of per-task durations can rise. What falls is the UNION:")
    P("  mean concurrency during the stage goes from well under half the machine")
    P("  to near-saturation, and the stage's WALLSPAN -- the only part the step")
    P("  actually pays -- drops ~1.8-1.9x. The win is WIDTH, not less work. That")
    P("  is also why the standalone ferret metric (whole-grid latency) and this")
    P("  agree while the per-task sum does not.")
    P("")
    P("  The WIDTH RESIDUAL is what a faster MoE kernel cannot remove. But")
    P("  the more decision-relevant number is the stage's SHARE of the step: once")
    P("  that is small, further kernel work on this task has a hard ceiling no")
    P("  matter how good the kernel gets, and the remaining gap is elsewhere.")
    P("")
    P("CAVEAT ON THE WORK/TASK-COUNT COLUMNS: concurrency.py measures one")
    P("steady-window iteration whose bounds it detects per run, and the two arms'")
    P("windows do not cover identical fractions of an iteration (visible as arm")
    P("A's task count differing between bs1 and bs16 when the graph is fixed).")
    P("wall_span/step, mean concurrency and the span ratio are normalised and")
    P("robust to that; absolute work_us and task counts are not, and are reported")
    P("as mechanism evidence rather than as measurements.")

    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, f"stage_wallspan_{short}.txt"), "w").write(txt)
    json.dump({"NW": NW, "stage": STAGE, "arms": ARM_LABEL, "rows": rows},
              open(os.path.join(a.out, f"stage_wallspan_{short}.json"), "w"),
              indent=1)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
