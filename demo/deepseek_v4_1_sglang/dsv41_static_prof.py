"""The static layers' own times inside a running SGLang server, with no profiler: DSV41_STATIC_PROF=<file>.

Every static kernel stamps, per SM, when the SM entered and when it ended (globaltimer ns; core.cuh writes the stamps in
every build), and once when its start barrier was passed. dsv41_static_hook calls maybe_sample() at the start of every
ModelRunner.forward: one os.stat. When <file> changed (the driver writes a label into it after a generate call returned,
so the server is idle and each kernel's stamps are its last decode step's), this rank reads the stamps of every static
layer and appends one JSON line to <file>.r<rank>.jsonl:
  {"label": .., "rank": .., "kernels": [[layer, "a" | "m", variant, first SM start, last SM start, start barrier,
                                          first SM end, last SM end, SMs, distinct experts or -1], ...],
   "variants": {layer: its attention's variant}, "engram": [layers with Engram], "tp": ..}
in the step's order (attention L, MoE L, ...). No GPU work is added to a step; a sample costs about 80 small reads, once.
meas/e2e/prof_table.py turns the files into the table."""
import json
import os
import sys

import numpy as np

FILE = os.environ.get("DSV41_STATIC_PROF", "")
CTA = os.environ.get("DSV41_STATIC_PROF_CTA", "0") != "0"   # also every CTA's start and end stamp ("cta": per kernel, by blockIdx)
TASKS = os.environ.get("DSV41_STATIC_PROF_TASKS", "0") != "0"   # timing build: also every task [node, SM, place, start, end]
_seen = [None]
_sched = {}   # id(kernel) -> (each SM's task list, task -> node name): a timing build's schedule, read once


def _nodes(st, h, rank):
    """Timing build (DSV41_STATIC_TIMING=1): per node [name, tasks, first / median / last start, first / median / last end]
    (globaltimer ns) of the last launch. Stage stamps are left out."""
    import json
    from mirage.mpk import compiler
    key = id(st.sk)
    if key not in _sched:
        paths, mpk = st.info["paths"], st.info["mpk"]
        sch = json.load(open(paths[rank] if len(paths) > rank else paths[0]))
        name = {n.graph_idx: compiler.tname(mpk, n.outputs[0]) for n in compiler.graph_nodes(mpk)}
        _sched[key] = (sch["worker_task_queues"], {i: name[t["node"]] for i, t in enumerate(sch["all_tasks"])})
    queues, node_of = _sched[key]
    tt = h.task_times()[0]
    per = {}
    for sm, q in enumerate(queues):
        for i, t in enumerate(q):
            s, e = tt[sm][i][:2]
            if not s or not e:
                break
            per.setdefault(node_of[t], []).append((int(s), int(e)))
    out = []
    for nm, v in per.items():
        ss, ee = sorted(p for p, _ in v), sorted(q for _, q in v)
        out.append([nm, len(v), ss[0], ss[len(ss) // 2], ss[-1], ee[0], ee[len(ee) // 2], ee[-1]])
    out = sorted(out, key=lambda r: r[7])
    if TASKS:   # (DSV41_STATIC_PROF_TASKS) every task of the launch, in its SM's order
        out = (out, [[node_of[t], sm, i, int(tt[sm][i][0]), int(tt[sm][i][1]), int(tt[sm][i][2]), int(tt[sm][i][3])]
                     for sm, q in enumerate(queues) for i, t in enumerate(q) if tt[sm][i][0] and tt[sm][i][1]])   # + its 2 stage stamps
    return out


def _kernel(st, kind, rank=0):
    from mirage.mpk.static_harness import Harness
    h = Harness(st.sk)
    lay = h.layout
    own = getattr(st, "prof_stamps", None)   # (a shared module: the layer's own stamps, dsv41_static_layer PROF_SHARE)
    t = (own.cpu().numpy() if own is not None else h._array(0, "stamps", np.int64)).reshape(lay["num_sms"], lay["num_stamps"])
    sb = int(st.prof_sb[0]) if own is not None else int(h._array(0, "start_barrier", np.int64)[0])
    start, end = t[:, lay["stamp_start"]], t[:, lay["stamp_end"]]
    on = start > 0
    if not on.any():   # (11-24) a layer of another T that never ran (DSV41_STATIC_T: bs 16 runs only the T = 16 layers)
        return None
    experts = -1
    if kind == "m" and "pairs" in st.bufs[0] and own is None:   # (shared: the pairs are the last MoE layer's)
        # (11-23) the layer's own T: its module's MoE demo (a layer of T > 8 is a class of a layer_module copy)
        experts = len(set(sys.modules[type(st).__module__].moe_demo.pairs_of(st.bufs[0])[0].flatten().tolist()))
    row = [int(st.layer), kind, getattr(st, "variant", "moe"), int(start[on].min()), int(start[on].max()),
           sb, int(end[on].min()), int(end[on].max()), int(on.sum()), experts]
    if lay.get("profile"):
        nodes = _nodes(st, h, rank)
        if TASKS:
            nodes, tasks = nodes
        row.append(nodes)
        if TASKS:
            row.append(tasks)   # row[11]: every task (meas/e2e/prof_tasks.py)
    return (row, [start.tolist(), end.tolist()]) if CTA else row


def maybe_sample(attns, moes, rank, meta=None):
    """One stat; a sample when the file's content is new. Never raises (a profile must not end the run)."""
    try:
        s = os.stat(FILE)
    except OSError:
        return
    key = (s.st_mtime_ns, s.st_size)
    if key == _seen[0]:
        return
    _seen[0] = key
    try:
        import torch
        torch.cuda.synchronize()   # (idle already: the driver wrote the label after its generate call returned)
        label = open(FILE).read().strip()
        ks = [_kernel(st, "a", rank) for st in attns if getattr(st, "sk", None) is not None]
        ks += [_kernel(st, "m", rank) for st in moes if getattr(st, "sk", None) is not None
               and getattr(st, "fused", None) is None]   # (DSV41_STATIC_FUSE: the attention's kernel holds the MoE's nodes)
        ks = [k for k in ks if k is not None]
        # (11-24) only the last step's kernels: with DSV41_STATIC_T a layer of another T keeps the stamps of a launch long ago
        # (a CUDA graph capture of another batch size); the step's kernels end within a few ms of each other
        end1 = (lambda k: k[0][7]) if CTA else (lambda k: k[7])
        if ks:
            last = max(end1(k) for k in ks)
            ks = [k for k in ks if last - end1(k) < 50_000_000]
        extra = {}
        if CTA:
            ks.sort(key=lambda k: (k[0][0], k[0][1]))
            ks, extra = [k[0] for k in ks], {"cta": [k[1] for k in ks]}
        ks.sort(key=lambda k: (k[0], k[1]))
        with open(f"{FILE}.r{rank}.jsonl", "a") as f:
            f.write(json.dumps({"label": label, "rank": rank, "kernels": ks, **extra, **(meta or {})}) + "\n")
    except BaseException as e:   # (no exception text: the job scripts stop on some of its words)
        print(f"[dsv41_static prof] sample not taken ({type(e).__name__[:6]})", flush=True)
