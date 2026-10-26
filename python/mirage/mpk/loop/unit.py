"""The kernel sub-loop's unit gate (docs/megakernel_dev_loop.md section 10, step 4): a new template variant of one node, measured
inside the megakernel at the production grid, alternated with its parent.

For `rounds` rounds: the parent candidate, then the parent with the node's variant changed, each a run.py timing build (--record
into a scratch cost file, so the loop's cost file only gets records through a round). Per run, from its stamps
(task_times.npy, timing.npy): the node's task durations by cost_records' rule (D1: end - max(previous end on the SM, latest
input end)), per launch the slowest task (the slowest CTA decides when the node's readers can go on) and the median task, then the
median over launches. Both runs must pass the part's check (run.py exits 1 otherwise).

Verdict: KEEP when the variant's slowest task is faster by more than `noise_us` in every round; LOSS when slower by more than
noise in every round; else NULL. The result goes to <out>/unit.json (the driver writes it to the ledger).

python -m mirage.mpk.loop.unit --part demo/kimi_k3_moe --candidate CAND.json --node 26 --variant stages=3 [--rounds 2] --out DIR
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List

from . import candidate as cand_mod


def node_durations(run_dir: str, node: int) -> List[Dict[str, float]]:
    """Per timed launch in the run's stamps: {"slowest", "median"} of the node's task durations (us) over all GPUs."""
    import numpy as np
    from ..cost_records import launch_durations
    schedules = []
    for g in range(64):
        path = os.path.join(run_dir, "schedules", f"schedule_gpu{g}.json")
        if not os.path.exists(path):
            break
        with open(path) as f:
            schedules.append(json.load(f))
    times = np.load(os.path.join(run_dir, "task_times.npy")).tolist()
    timing = np.load(os.path.join(run_dir, "timing.npy")).tolist()
    out = []
    for t, tm in zip(times, timing):
        placed, _, _ = launch_durations(schedules, t, [b for b, _ in tm], None, set())
        d = [us for g, s in enumerate(schedules) for tid, us in placed[g].items() if s["all_tasks"][tid]["node"] == node]
        if d:
            out.append({"slowest": max(d), "median": statistics.median(d)})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", required=True)
    p.add_argument("--candidate", required=True, help="the parent candidate.json")
    p.add_argument("--node", type=int, required=True, help="graph_idx of the node whose variant changes")
    p.add_argument("--variant", action="append", required=True, metavar="NAME=VALUE")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--noise-us", type=float, default=0.3)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    parent = cand_mod.load(a.candidate)
    plan = cand_mod.plan_from_doc(parent["plan"])
    variants = dict(plan.get("variants") or {})
    variants[a.node] = dict(variants.get(a.node, {}), **{k: int(v) for k, v in (kv.split("=", 1) for kv in a.variant)})
    child = cand_mod.derive(parent, f"unit gate node {a.node} variant {variants[a.node]}", plan=dict(plan, variants=variants))
    os.makedirs(a.out, exist_ok=True)
    paths = {}
    for name, c in (("parent", parent), ("variant", child)):
        paths[name] = os.path.join(a.out, name, "candidate.json")
        cand_mod.save(c, paths[name])
    rounds = []
    for r in range(a.rounds):
        row = {}
        for name in ("parent", "variant"):
            out = os.path.join(a.out, f"r{r}_{name}")
            cmd = [sys.executable, "-m", "mirage.mpk.loop.run", "--part", a.part, "--candidate", paths[name], "--reps", str(a.reps),
                   "--record", os.path.join(a.out, "scratch_costs.json"), "--out", out]
            e = subprocess.run(cmd, capture_output=True, text=True)
            with open(out + ".txt", "w") as f:
                f.write(e.stdout + e.stderr)
            if e.returncode != 0:
                result = {"verdict": "FAIL-GATE", "round": r, "run": name, "exit": e.returncode}
                print(json.dumps(result))
                json.dump(result, open(os.path.join(a.out, "unit.json"), "w"), indent=1)
                return 1
            per = node_durations(out, a.node)
            row[name] = {k: statistics.median(x[k] for x in per) for k in ("slowest", "median")}
        row["delta_slowest"] = round(row["variant"]["slowest"] - row["parent"]["slowest"], 3)
        rounds.append(row)
        print(f"round {r}: node {a.node} slowest task parent {row['parent']['slowest']:.2f} variant {row['variant']['slowest']:.2f} us "
              f"(median task {row['parent']['median']:.2f} / {row['variant']['median']:.2f})", flush=True)
    d = [r["delta_slowest"] for r in rounds]
    verdict = "KEEP" if all(x < -a.noise_us for x in d) else "LOSS" if all(x > a.noise_us for x in d) else "NULL"
    result = {"verdict": verdict, "node": a.node, "variant": variants[a.node], "parent": parent["id"], "child": child["id"],
              "rounds": rounds}
    with open(os.path.join(a.out, "unit.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"UNIT {verdict}: deltas of the slowest task {d} us (noise {a.noise_us})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
