#!/usr/bin/env python3
"""Audit the MoE grouped-GEMM dispatch in a COMPILED MPK task graph -- no GPU.

M3-I2b's `taskgraph_quantize.py` asked "do several tasks of one call site write
the same output?".  The MoE grouped GEMMs answer "no" and are excused there as
metadata-addressed: every task takes its identity from
`task_desc->task_metadata.expert_offset`.  That is exactly why they need their
own audit -- their waste is not redundancy, it is DISPATCH WIDTH against a
runtime mask.

    moe_fp8_blockscale_sm100.cuh:196
        for (int ae = expert_offset; ae < num_activated; ae += expert_stride)

`num_activated` is `mask[NUM_EXPERTS]`, written by the router at runtime.  A
task whose `expert_offset >= num_activated` executes ZERO loop iterations: the
early exit is already in the kernel, so right-sizing is a ROUTER problem, not a
grouped-GEMM one.  What the compiled graph pins, statically, is everything
else -- how many tasks exist, which expert_offset each carries, in what order
they sit in the graph, and therefore which WORKER each one lands on:

  * MPK launches the whole iteration with one EVENT_LAUNCH_DEPENDENT_TASKS
    covering [first_task_id, last_task_id) = the entire graph, and
    persistent_kernel.cuh:1319-1340 assigns
        position_index = first_task_id + i * num_workers + j
    to the scheduler that owns worker j, round-robin inside that scheduler's
    worker range.  Net effect: task t goes to worker (t - first_task_id) %
    num_workers, deterministically, and each worker executes its queue IN
    ORDER.
  * The grid is (grid_x, moe_n_splits, 1) and runtime.cc:325-327 walks bid
    x-outer / y-inner, so a site's tasks are (e=0,split0), (e=0,split1),
    (e=1,split0), ...  The live tasks -- those with expert_offset <
    num_activated -- are therefore always a CONTIGUOUS PREFIX of the site.

Put together, the stage's cost is set by the deepest worker load
`ceil(live_tasks / num_workers)`, not by the number of activated groups.  That
is the whole difference between M3-I1's backlog model (wall span proportional to
group count, +37% at bs1) and what the profile actually shows.

Usage:
    python3 taskgraph_moe.py <task_graph_rank0.json> [...]
    python3 taskgraph_moe.py --activated 56.4 <task_graph_rank0.json>
    python3 taskgraph_moe.py --workers 128 --json out.json <graph>
"""
import argparse
import json
from collections import Counter, defaultdict

# The two Qwen3.5 grouped GEMMs (preserved fp32 block scales, M2-I12/I13).
MOE_GROUPED = {
    241: "TASK_MOE_W13_FP8_BLOCKSCALE_SM100",
    242: "TASK_MOE_W2_FP8_BLOCKSCALE_SM100",
}
# The UE8M0 grouped siblings share the mask/expert_offset contract, so audit
# them too when a graph happens to contain them.
MOE_GROUPED_ALSO = {
    227: "TASK_MOE_W13_FP8_SM100",
    228: "TASK_MOE_W2_FP8_SM100",
    231: "TASK_MOE_W13_LINEAR_SM100",
    232: "TASK_MOE_W2_LINEAR_SM100",
}


def sites_of(tasks):
    """A call site = a maximal run of same-type tasks sharing a trigger event."""
    sites, cur, cur_ev = [], [], None
    for t in tasks:
        ev = t["trigger_event"]
        if ev != cur_ev and cur:
            sites.append(cur)
            cur = []
        cur_ev = ev
        cur.append(t)
    if cur:
        sites.append(cur)
    return sites


def describe_site(site):
    """Static shape of one grouped-GEMM call site."""
    t0 = site[0]
    w = t0["inputs"][2]            # weight_fp8 [num_experts, N_tile, K]
    out = t0["outputs"][0]         # [batch, topk, N_tile]
    num_experts, n_tile, k = w["dims"]
    batch, topk, _ = out["dims"]
    orig_n = out["strides"][1]     # the expert's FULL N (stride between slots)
    offs = [t["expert_offset"] for t in site]
    grid_x = len(set(offs))
    splits = len(site) // grid_x if grid_x else 0
    # y-inner ordering means offsets repeat in blocks of `splits`
    interleaved = offs == [i // splits for i in range(len(site))]
    return dict(num_experts=num_experts, n_tile=n_tile, k=k, batch=batch,
                topk=topk, orig_n=orig_n, n_tasks=len(site), grid_x=grid_x,
                splits=splits, interleaved=interleaved,
                first_index=site[0]["_idx"], layer=t0["inputs"][2]["base_ptr"])


def worker_of(task_index, first_task_id, num_workers):
    """persistent_kernel.cuh:1329 -- position_index = first + i*W + j -> worker j."""
    return (task_index - first_task_id) % num_workers


def wave_profile(site_desc, activated, first_task_id, num_workers):
    """Worker load for this site at a given runtime `num_activated`.

    Live tasks are the contiguous prefix with expert_offset < activated; each
    such task loops over ceil((activated - offset) / grid_x) experts, which is
    1 whenever activated <= grid_x (always true for the Qwen3.5 build, where
    grid_x = min(num_experts, mbt*topk) = 128 and activated <= 128).
    """
    grid_x, splits = site_desc["grid_x"], site_desc["splits"]
    base = site_desc["first_index"]
    live_offsets = min(int(activated), grid_x)
    loads = Counter()
    experts_per_task = {}
    for x in range(live_offsets):
        n_exp = (int(activated) - x + grid_x - 1) // grid_x
        for y in range(splits):
            idx = base + x * splits + y
            loads[worker_of(idx, first_task_id, num_workers)] += n_exp
            experts_per_task[idx] = n_exp
    live_tasks = live_offsets * splits
    return dict(activated=activated, live_tasks=live_tasks,
                dead_tasks=site_desc["n_tasks"] - live_tasks,
                workers_used=len(loads),
                max_expert_tiles=max(loads.values()) if loads else 0,
                waves=max(loads.values()) if loads else 0)


def audit(path, activated_by_type, workers, verbose):
    d = json.load(open(path))
    tasks = d["all_tasks"]
    for i, t in enumerate(tasks):
        t["_idx"] = i
    # The one dependent event that launches the whole iteration.
    launch = [e for e in d["all_events"] if e["event_type"] == 903]
    first_task_id = launch[0]["first_task_id"] if launch else 0
    by_type = defaultdict(list)
    for t in tasks:
        by_type[t["task_type"]].append(t)

    known = dict(MOE_GROUPED)
    known.update(MOE_GROUPED_ALSO)
    out = dict(graph=path, n_tasks=len(tasks), first_task_id=first_task_id,
               num_workers=workers, stages=[])
    print(f"\n=== {path}")
    print(f"    {len(tasks)} tasks; iteration launched by one "
          f"EVENT_LAUNCH_DEPENDENT_TASKS [{first_task_id}, "
          f"{launch[0]['last_task_id'] if launch else '?'}); "
          f"task t -> worker (t-{first_task_id}) % {workers}")
    for tt, name in sorted(known.items()):
        got = by_type.get(tt)
        if not got:
            continue
        sites = [s for s in sites_of(got)]
        descs = [describe_site(s) for s in sites]
        shapes = Counter((d_["n_tasks"], d_["grid_x"], d_["splits"],
                          d_["n_tile"], d_["k"], d_["batch"], d_["topk"],
                          d_["orig_n"], d_["interleaved"]) for d_ in descs)
        print(f"\n  {name}: {len(got)} tasks in {len(sites)} call sites")
        for (n, gx, sp, n_tile, k, batch, topk, orig_n, inter), cnt in shapes.items():
            print(f"    x{cnt:<4d} tasks/site {n:<5d} grid ({gx},{sp},1)  "
                  f"weight tile [E,{n_tile},{k}] of N={orig_n}  "
                  f"act [{batch},{topk}]  y-inner order: {inter}")
            if not inter:
                print("      !! expert_offset is NOT y-inner: the live prefix "
                      "assumption below does not hold for this graph")
        # worker/wave model over the runtime activated counts
        act_list = activated_by_type.get(tt) or activated_by_type.get(None) or []
        if act_list and descs:
            d0 = descs[0]
            print(f"    {'activated':>10} {'live tasks':>11} {'dead':>6} "
                  f"{'workers':>8} {'waves':>6}   (per call site)")
            for a in act_list:
                wp = wave_profile(d0, a, first_task_id, workers)
                print(f"    {a:>10.1f} {wp['live_tasks']:>11d} "
                      f"{wp['dead_tasks']:>6d} {wp['workers_used']:>8d} "
                      f"{wp['waves']:>6d}")
                out["stages"].append(dict(task_type=tt, name=name, **d0, **wp))
        if verbose:
            for d_ in descs[:2]:
                print(f"      site @task {d_['first_index']} "
                      f"({d_['layer']}) -> workers "
                      f"{[worker_of(d_['first_index'] + i, first_task_id, workers) for i in range(min(6, d_['n_tasks']))]}...")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("--activated", type=float, action="append", default=None,
                    help="runtime num_activated to model (repeatable)")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--json", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    acts = {None: a.activated or [8, 16, 32, 64, 128]}
    res = [audit(p, acts, a.workers, a.verbose) for p in a.graphs]
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
