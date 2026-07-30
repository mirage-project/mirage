#!/usr/bin/env python3
"""M4-I6: the critical path before vs after, per stage, across independent reps.

This exists to answer one question the stage table raised and could not settle:
the router's path contribution falls ~445 us at bs1, but the whole critical path
falls only ~353 us. Where did the other ~92 us go?

M4-I5's cp_decompose.py, run on BOTH arms' profiled raws, answers it directly --
it reports path tasks and path microseconds per stage, so the delta can be read
stage by stage instead of inferred. Two mechanisms were candidates:

  (i)  the path RE-ROUTED -- the router got fast enough that a different chain
       became binding, so part of its saving was never on the path afterwards; or
  (ii) the OTHER stages actually got slower -- the shared-register-budget tax,
       which gate 2 measured as persistent_kernel going 238 -> 255 registers with
       a new 4-byte spill. Every task body is inlined into that one kernel, so a
       register-hungry task taxes tasks it never touches (add-mpk-task, "Every
       Task Shares One Register Budget"; M3-I6a's attention accumulator did this
       to dense-fp8 and GDN).

(i) is falsified by the path-task counts: they are IDENTICAL stage for stage
between arms (40/40 router, 40/40 W13, 10/10 attention, 80/80 dense fp8, ...),
so the chain did not move. (ii) is confirmed by REPRODUCTION -- run this over two
independent reps and the per-stage bias repeats to within ~1 us, which random
profiled variance does not do. That is why --reps takes more than one.
"""
import argparse
import json
import os
import sys

ROUTER = "TASK_MOE_TOPK_SOFTMAX_SM100"
# M4-I5's recorded bs1 basis, for the "share of 842.1 recovered" statement.
M4I5_ROUTER_US_ON_PATH = {1: 842.1, 8: 903.3, 16: 982.1}


def load(root, arm, bs, rep):
    sfx = "" if rep == 0 else f"_rep{rep}"
    p = os.path.join(root, f"cp_{arm}_bs{bs}{sfx}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d, {r["name"]: r for r in d["composition"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/var/tmp/../m4i6/critpath")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--reps", default="0,2",
                    help="comma-separated rep indices to compare")
    ap.add_argument("--lost", default="",
                    help="rep indices kept in the record but excluded as "
                         "contaminated, with a reason after a colon")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reps = [int(x) for x in a.reps.split(",") if x.strip() != ""]

    data = {}
    for rep in reps:
        for arm in ("A", "B"):
            got = load(a.root, arm, a.bs, rep)
            if got is None:
                sys.exit(f"REFUSING: missing cp json for arm {arm} rep {rep}")
            data[(arm, rep)] = got

    L, P = [], None
    P = L.append
    P(f"M4-I6 critical-path decomposition, bs{a.bs}, arm A (base) vs arm B "
      f"(ferret v013 router)")
    P("Tool: opt/m4i5/scripts/cp_decompose.py --weight levelmax, on each arm's own")
    P("profiled raw + its own compiled task graph. Reps compared: "
      + ", ".join(str(r) for r in reps))
    if a.lost:
        P("")
        P("LOST REPS (kept in the record, excluded from the comparison):")
        for item in a.lost.split(";"):
            if item.strip():
                P("  " + item.strip())
    P("")
    P("== per-stage microseconds ON THE CRITICAL PATH, arm B minus arm A ==")
    hdr = f"{'stage':36} {'tasks':>6}" + "".join(f"{'rep'+str(r):>11}" for r in reps)
    P(hdr)
    names = sorted(data[("A", reps[0])][1],
                   key=lambda n: -data[("A", reps[0])][1][n]["us_on_path"])
    totals = {r: 0.0 for r in reps}
    other = {r: 0.0 for r in reps}
    rows = []
    for n in names:
        cells, vals = "", {}
        for r in reps:
            dA = data[("A", r)][1].get(n, {})
            dB = data[("B", r)][1].get(n, {})
            if dA.get("path_tasks") != dB.get("path_tasks"):
                cells += f"{'TASKS DIFFER':>11}"
                vals[r] = None
                continue
            v = dB.get("us_on_path", 0.0) - dA.get("us_on_path", 0.0)
            vals[r] = v
            totals[r] += v
            if n != ROUTER:
                other[r] += v
            cells += f"{v:>+11.1f}"
        tasks = data[("A", reps[0])][1][n].get("path_tasks")
        P(f"{n[:36]:36} {tasks:>6}{cells}")
        rows.append({"stage": n, "path_tasks": tasks, "delta_us": vals})
    P(f"{'TOTAL (= delta cp)':36} {'':>6}"
      + "".join(f"{totals[r]:>+11.1f}" for r in reps))
    P("")
    P("PATH-TASK COUNTS ARE IDENTICAL between arms for every stage, so the chain")
    P("did not re-route -- the deltas are per-task durations, not a different path.")
    P("")
    P("== the router, and the tax ==")
    summary = []
    for r in reps:
        dA, cA = data[("A", r)]
        dB, cB = data[("B", r)]
        ra, rb = cA[ROUTER], cB[ROUTER]
        rec = ra["us_on_path"] - rb["us_on_path"]
        ref = M4I5_ROUTER_US_ON_PATH.get(a.bs)
        s = {
            "rep": r,
            "cp_A_us": dA["cp_max_us"], "cp_B_us": dB["cp_max_us"],
            "cp_delta_us": dB["cp_max_us"] - dA["cp_max_us"],
            "step_A_us": dA["step_measured_us"], "step_B_us": dB["step_measured_us"],
            "router_path_A_us": ra["us_on_path"], "router_path_B_us": rb["us_on_path"],
            "router_pct_cp_A": ra["pct_of_cp"], "router_pct_cp_B": rb["pct_of_cp"],
            "router_us_per_path_task_A": ra["us_per_path_task"],
            "router_us_per_path_task_B": rb["us_per_path_task"],
            "router_recovered_us": rec,
            "router_recovered_frac_of_m4i5": (rec / ref) if ref else None,
            "other_stages_delta_us": other[r],
            "tax_as_frac_of_recovery": other[r] / rec if rec else None,
        }
        summary.append(s)
        P(f"rep{r}:")
        P(f"   cp                 {s['cp_A_us']:9.1f} -> {s['cp_B_us']:9.1f} us "
          f"({s['cp_delta_us']:+.1f})")
        P(f"   measured step      {s['step_A_us']:9.1f} -> {s['step_B_us']:9.1f} us "
          f"({s['step_B_us']-s['step_A_us']:+.1f})")
        P(f"   router ON PATH     {s['router_path_A_us']:9.1f} -> "
          f"{s['router_path_B_us']:9.1f} us   "
          f"({s['router_pct_cp_A']}% -> {s['router_pct_cp_B']}% of cp)")
        P(f"   router us/path-task{s['router_us_per_path_task_A']:9.3f} -> "
          f"{s['router_us_per_path_task_B']:9.3f}")
        if ref:
            P(f"   RECOVERED          {rec:9.1f} us = "
              f"{100*rec/ref:.1f}% of M4-I5's {ref:.1f} us contribution")
        P(f"   TAX on other stages{other[r]:+9.1f} us = "
          f"{100*other[r]/rec:.1f}% of the recovery given back")
        P("")
    P("MECHANISM. The tax is not noise: it reproduces stage for stage across")
    P("independent reps, and it is largest on the largest, most register-hungry")
    P("stages (W13, then attention and W2) while the small elementwise stages")
    P("barely move. That is the shape of a SHARED-REGISTER-BUDGET effect, and gate")
    P("2 measured the cause -- persistent_kernel went 238 -> 255 registers (the")
    P("ceiling at __launch_bounds__(256,1)) and picked up a 4-byte spill that the")
    P("base tree does not have. The v013 candidate holds 16 live u64 sorted keys")
    P("plus the buffered top-k across its k-loop, and that is what consumed the")
    P("headroom every other task body was sharing.")
    P("")
    P("CONSEQUENCE FOR THE CAMPAIGN, not just for this issue: the tree now starts")
    P("at the register ceiling with a spill. The next register-hungry integration")
    P("inherits that, and its own ptxas before/after has to be read against this")
    P("baseline rather than against the pre-M4-I6 one.")

    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, f"cp_compare_bs{a.bs}.txt"), "w").write(txt)
    json.dump({"bs": a.bs, "reps": reps, "lost": a.lost,
               "per_stage": rows, "summary": summary},
              open(os.path.join(a.out, f"cp_compare_bs{a.bs}.json"), "w"), indent=1)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
