#!/usr/bin/env python3
"""M4-I6: the MoE-router stage before/after -- per-task time, wallspan, and how
much of M4-I5's 842.1 us critical-path contribution the integration recovers.

Reads the per-arm concurrency.py outputs and reports, for one steady-window
decode iteration:

  step_us               the whole iteration
  router total_us       the stage's WORK (sum of its per-layer tasks' durations)
  router wall_span_us   the stage's WALLSPAN -- what the step actually pays
  mean_conc_during      worker CTAs inside a task while the stage runs (of 128)

WHY THE ROUTER'S ARITHMETIC IS SIMPLER THAN M4-I2's, AND WHY THAT CUTS BOTH WAYS.
M4-I2's dense-fp8 stage got FASTER while its per-task work went UP 1.24x: the
finer slice multiplied the task count, each task re-paid a fixed prologue, and the
win came from concurrency rising toward saturation -- width, not less work. That
mechanism CANNOT apply here. Task 260 launches as exactly ONE task per layer per
step, grid=(1,1,1), and M4-I5 measured its live/lvl at 1.0 -- there is no width to
recover and nothing to overlap it against, which is precisely why M4-I5 called it
the most serialized stage in the graph. So wallspan can only follow per-task
latency, the standalone ferret gain should translate almost directly, and a
per-task regression would be unrecoverable by any packing.

THE NUMBER M4-I5 ASKED FOR. Its bs1 decomposition: 40 path tasks, 842.1 us on the
7957.5 us critical path (10.58%), T = 21.053 us/task, and "must reach 3.697
us/task at bs1" for the five-stage parity scenario. The recovery this script
reports is (T_A - T_B) * path_tasks, with T taken from THIS pair of runs so the
two arms share a profiling basis, and with M4-I5's own recorded basis shown
alongside so a change in the basis itself is visible rather than absorbed.
"""
import argparse
import glob
import json
import os
import re
import sys

NW = 128           # MPK worker CTAs (concurrency.py's own NW)
STAGE = "TASK_MOE_TOPK_SOFTMAX_SM100"    # task 260
ARM_LABEL = {"A": "base(M3-I5b/I5c/I8 router)", "B": "new(ferret v013 router)"}

# M4-I5's recorded critical-path basis (opt/m4i5/README.md sections (a) and (c)),
# per batch size: path tasks, us on the path, measured T us/task, the vLLM
# per-call floor this stage is measured against, and the base critical path.
M4I5 = {
    1:  {"path_tasks": 40, "us_on_path": 842.1, "T_us": 21.053,
         "vllm_us": 3.697, "cp_us": 7957.5},
    8:  {"path_tasks": 40, "us_on_path": 903.3, "T_us": 22.582,
         "vllm_us": 4.602, "cp_us": 8240.9},
    16: {"path_tasks": 40, "us_on_path": 982.1, "T_us": 24.552,
         "vllm_us": 5.955, "cp_us": 8642.0},
}


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
    total, span, n = s["total_us"], s["wall_span_us"], s["n"]
    return {
        "n_tasks": n,
        "total_us": total,
        "wall_span_us": span,
        "per_task_us": total / n if n else None,
        "mean_conc_during": s["mean_concurrency_during"],
        "span_per_task_us": span / n if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="/var/tmp/m4i6_prof/stage")
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
    P("M4-I6 MoE-router stage (task 260 / TASK_MOE_TOPK_SOFTMAX_SM100), ONE")
    P("steady-window decode iteration, profiler ON -- DIAGNOSTIC attribution. The")
    P("performance claim is the --no-profiler A/B in tables/m4i6_tables.txt.")
    P("")
    P(f"arm A = {ARM_LABEL['A']}      arm B = {ARM_LABEL['B']}      NW = {NW}")
    P("")
    P("== the whole step, and the router stage inside it ==")
    P(f"{'bs':>3} {'arm':>3} {'step_us':>10} {'tasks':>6} {'work_us':>10} "
      f"{'us/task':>9} {'span_us':>10} {'span/step':>9} {'meanconc':>9}")
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
              f"{row['total_us']:10.1f} {row['per_task_us']:9.3f} "
              f"{row['wall_span_us']:10.1f} {row['wall_span_us']/st:9.3f} "
              f"{row['mean_conc_during']:9.2f}")
    P("")
    P("mean concurrency near 1.0 is the point: one task per layer per step means")
    P("the stage IS its own critical path, so work_us ~ wall_span_us and per-task")
    P("latency is the only lever. Contrast M4-I2's dense stage, whose concurrency")
    P("rose from well under half the machine to near-saturation.")
    P("")
    P("== before -> after, and the share of M4-I5's path contribution recovered ==")
    rows = []
    for bs in bss:
        if ("A", bs) not in d or ("B", bs) not in d:
            continue
        A, B = stage_row(d[("A", bs)]), stage_row(d[("B", bs)])
        if A is None or B is None:
            continue
        sA, sB = d[("A", bs)]["step_us"], d[("B", bs)]["step_us"]
        ref = M4I5.get(bs)
        r = {
            "bs": bs,
            "step_us_A": sA, "step_us_B": sB, "step_gain_us": sA - sB,
            "n_tasks_A": A["n_tasks"], "n_tasks_B": B["n_tasks"],
            "work_us_A": A["total_us"], "work_us_B": B["total_us"],
            "work_speedup": A["total_us"] / B["total_us"],
            "per_task_us_A": A["per_task_us"], "per_task_us_B": B["per_task_us"],
            "per_task_speedup": A["per_task_us"] / B["per_task_us"],
            "span_us_A": A["wall_span_us"], "span_us_B": B["wall_span_us"],
            "span_speedup": A["wall_span_us"] / B["wall_span_us"],
            "span_gain_us": A["wall_span_us"] - B["wall_span_us"],
            "mean_conc_A": A["mean_conc_during"],
            "mean_conc_B": B["mean_conc_during"],
            "stage_share_of_step_B": B["wall_span_us"] / sB,
        }
        P(f"bs{bs}:")
        P(f"   per-task LATENCY  {A['per_task_us']:9.3f} -> {B['per_task_us']:9.3f} us "
          f" ({r['per_task_speedup']:.3f}x)")
        P(f"   stage WORK        {A['total_us']:9.1f} -> {B['total_us']:9.1f} us "
          f" ({r['work_speedup']:.3f}x)")
        P(f"   stage WALLSPAN    {A['wall_span_us']:9.1f} -> {B['wall_span_us']:9.1f} us "
          f" ({r['span_speedup']:.3f}x, -{r['span_gain_us']:.1f} us)")
        P(f"   step              {sA:9.1f} -> {sB:9.1f} us  (-{r['step_gain_us']:.1f} us)")
        P(f"   mean concurrency during the stage  {A['mean_conc_during']:.2f} "
          f"-> {B['mean_conc_during']:.2f}  of {NW}")
        if ref:
            # Recovery on the critical path, measured on THIS pair's basis.
            rec_us = (A["per_task_us"] - B["per_task_us"]) * ref["path_tasks"]
            frac_of_m4i5 = rec_us / ref["us_on_path"]
            # Where the stage's path contribution lands, on this pair's basis
            # and on M4-I5's recorded basis.
            path_B_own = B["per_task_us"] * ref["path_tasks"]
            scale = ref["T_us"] / A["per_task_us"] if A["per_task_us"] else None
            path_B_m4i5 = (B["per_task_us"] * scale * ref["path_tasks"]
                           if scale else None)
            r.update({
                "m4i5_T_us": ref["T_us"], "m4i5_us_on_path": ref["us_on_path"],
                "m4i5_cp_us": ref["cp_us"], "vllm_us_per_call": ref["vllm_us"],
                "path_tasks": ref["path_tasks"],
                "recovered_us_own_basis": rec_us,
                "recovered_frac_of_m4i5_842": frac_of_m4i5,
                "path_contrib_B_own_basis_us": path_B_own,
                "profiler_basis_ratio_A_over_m4i5": (
                    A["per_task_us"] / ref["T_us"] if ref["T_us"] else None),
                "path_contrib_B_rescaled_to_m4i5_us": path_B_m4i5,
                "vllm_floor_ratio_B": (B["per_task_us"] / ref["vllm_us"]
                                       if ref["vllm_us"] else None),
                "m4i5_target_met": B["per_task_us"] <= ref["vllm_us"],
            })
            P(f"   -- against M4-I5's critical-path basis --")
            P(f"   M4-I5 recorded T = {ref['T_us']:.3f} us/task, "
              f"{ref['us_on_path']:.1f} us on the {ref['cp_us']:.1f} us path "
              f"({ref['path_tasks']} tasks)")
            P(f"   this run's arm A  T = {A['per_task_us']:.3f} us/task "
              f"(profiler basis ratio A/M4-I5 = "
              f"{A['per_task_us']/ref['T_us']:.3f}x)")
            P(f"   RECOVERED on the path (own basis): "
              f"({A['per_task_us']:.3f} - {B['per_task_us']:.3f}) x "
              f"{ref['path_tasks']} = {rec_us:.1f} us")
            P(f"     = {100*frac_of_m4i5:.1f}% of M4-I5's {ref['us_on_path']:.1f} us "
              f"contribution")
            P(f"   path contribution after: {path_B_own:.1f} us on this basis"
              + (f", {path_B_m4i5:.1f} us rescaled to M4-I5's basis"
                 if path_B_m4i5 else ""))
            P(f"   vLLM per-call floor {ref['vllm_us']:.3f} us -> arm B is "
              f"{B['per_task_us']/ref['vllm_us']:.2f}x it "
              f"({'MET' if B['per_task_us'] <= ref['vllm_us'] else 'NOT met'})")
        P(f"   HEADROOM LEFT IN THIS STAGE: it is now "
          f"{100*B['wall_span_us']/sB:.1f}% of the step, so driving task 260 to")
        P(f"     ZERO would leave {sB - B['wall_span_us']:.1f} us of the "
          f"{sB:.1f} us step standing.")
        P("")
        rows.append(r)
    P("READING IT:")
    P("")
    P("  The profiler inflates per-task time, so arm A's T will not equal M4-I5's")
    P("  recorded T exactly; the 'profiler basis ratio' column is that offset,")
    P("  stated rather than hidden. The RECOVERY is computed from the A->B pair")
    P("  measured in the same session, which is the comparison that is sound; the")
    P("  rescaled path contribution is the same delta expressed in M4-I5's units")
    P("  and is a projection, not a measurement.")
    P("")
    P("  A stage at concurrency ~1.0 has NO width residual to hide behind: what")
    P("  the kernel saves, the step collects, minus only whatever the step's")
    P("  other stages were already overlapping. That is why the honest test of")
    P("  this integration is the e2e A/B, and why the e2e delta should land close")
    P("  to the wallspan delta rather than far below it.")

    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "stage_wallspan.txt"), "w").write(txt)
    json.dump({"NW": NW, "stage": STAGE, "arms": ARM_LABEL,
               "m4i5_basis": M4I5, "rows": rows},
              open(os.path.join(a.out, "stage_wallspan.json"), "w"), indent=1)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
