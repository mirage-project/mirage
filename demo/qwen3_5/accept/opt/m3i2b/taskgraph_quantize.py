#!/usr/bin/env python3
"""Audit a COMPILED MPK task graph for redundant task instances.

An MPK stage is redundant when several task instances of one call site are handed
the SAME tile — same base pointer, same dims — because the layer registered its
tensors with `input_map = (-1,-1,-1)` while launching a grid > 1. Every task is
then honestly busy computing the identical answer, so a per-task profile shows
nothing wrong; the only symptom is a stage costing far more worker time than its
arithmetic can justify.

`task_graph_rank0.json` settles it without a GPU: for each call site, count the
DISTINCT byte offsets across its task instances.

**The discriminator is the OUTPUT offsets, not the input offsets.** Shared inputs
are normal and correct — every task of a GEMM reads the whole activation row and
writes its own slice of the output columns (`TASK_LINEAR_SM100` at 32 tasks/site:
1 distinct input offset, 32 distinct output offsets). Duplicated WORK is what
shows up as several tasks writing the SAME output address:

    distinct OUT offsets == 1 and tasks/site > 1  -> N-fold redundant, UNLESS the
                                                     kernel derives its identity
                                                     from `task_metadata`
    distinct OUT offsets > 1                      -> partitioned by output; fine

The metadata case is why this is an AUDITOR and not a checker: `gdn_conv1d`,
`gdn_recurrent`, `paged_attention` and the MoE grouped GEMMs all share one tile
and pick their slice from `task_desc->task_metadata.*` inside the generated code,
which is legitimate. Cross-check any flagged stage against its `register_*`
function in `src/kernel/task_register.cc`: if the emitted code reads
`task_desc->task_metadata`, the identity is per-task and the sharing is fine.

Found by M3-I2b: `TASK_QUANTIZE_FP8_SM100` was 16x redundant at all 240 call
sites (124800 wasted row-quantizations per decode step, 84 ms of worker time at
bs1 for 5.3 ms of useful work).

Usage:
    python3 taskgraph_quantize.py <task_graph_rank0.json> [...]
    python3 taskgraph_quantize.py --task-type 275 <task_graph_rank0.json>
    python3 taskgraph_quantize.py --all <task_graph_rank0.json>
"""
import argparse
import json
from collections import Counter, defaultdict

QUANTIZE = 275
# Task types whose kernel takes its identity from task_desc->task_metadata, so a
# shared tile is by design rather than a bug. Verified against task_register.cc.
METADATA_ADDRESSED = {
    234: "gdn_conv1d (request_id + kv_idx -> channel block)",
    237: "gdn_recurrent (request_id + head)",
    241: "moe_w13 grouped GEMM (expert_offset)",
    242: "moe_w2 grouped GEMM (expert_offset)",
    257: "paged_attention (request_id + kv_idx + merge_task_offset)",
}


def sites_of(tasks):
    """A call site is a maximal run of same-type tasks sharing a trigger event."""
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


def audit(path, task_types, names):
    d = json.load(open(path))
    by_type = defaultdict(list)
    for t in d["all_tasks"]:
        by_type[t["task_type"]].append(t)
    print(f"\n=== {path}  ({len(d['all_tasks'])} tasks)")
    total_redundant = 0
    for tt in sorted(task_types):
        tasks = by_type.get(tt)
        if not tasks:
            continue
        label = names.get(str(tt), f"task_type {tt}")
        if not tasks[0].get("inputs"):
            continue  # control tasks (begin/terminate/scheduler) carry no tensors
        sites = sites_of(tasks)
        shapes = Counter()
        redundant_rows = 0
        for s in sites:
            dims = tuple(s[0]["inputs"][0]["dims"])
            n_in = len({t["inputs"][0]["offset"] for t in s})
            n_out = len({t["outputs"][0]["offset"] for t in s}) if s[0]["outputs"] else 0
            shapes[(dims, len(s), n_in, n_out)] += 1
            rows = dims[0] * (dims[1] if len(dims) == 3 else 1)
            if n_out == 1 and len(s) > 1 and tt not in METADATA_ADDRESSED:
                redundant_rows += rows * (len(s) - 1)
        note = METADATA_ADDRESSED.get(tt)
        print(f"  {label}: {len(tasks)} tasks in {len(sites)} call sites"
              + (f"   [metadata-addressed: {note}]" if note else ""))
        for (dims, n, n_in, n_out), k in sorted(shapes.items(),
                                                key=lambda kv: -kv[1]):
            if n_out == 1 and n > 1:
                verdict = (f"shared tile x{n} (by design)" if note
                           else f"SHARED OUTPUT x{n}  <-- REDUNDANT")
            elif n_out == n:
                verdict = "partitioned by output"
            else:
                verdict = f"partitioned by output ({n_out} of {n})"
            print(f"    x{k:4d}  tile {list(dims)!s:<18} tasks/site {n:4d}  "
                  f"distinct in-off {n_in:4d}  out-off {n_out:4d}   {verdict}")
        if not note:
            total_redundant += redundant_rows
    if total_redundant:
        print(f"  redundant row-computations per step (non-metadata stages): "
              f"{total_redundant}")
    return total_redundant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("--task-type", type=int, action="append", default=None)
    ap.add_argument("--all", action="store_true",
                    help="audit every task type present, not just quantize")
    ap.add_argument("--names", default=None,
                    help="task_names.json from the capture, for readable labels")
    a = ap.parse_args()
    names = json.load(open(a.names)) if a.names else {}
    for p in a.graphs:
        if a.all:
            tts = {t["task_type"] for t in json.load(open(p))["all_tasks"]}
        else:
            tts = set(a.task_type or [QUANTIZE])
        audit(p, tts, names)


if __name__ == "__main__":
    main()
