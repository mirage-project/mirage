#!/usr/bin/env python3
"""SCHEDULE-AWARE perfetto-trace analyzer for the MPK megakernel.

Built to answer the trace-review questions that per_position_grid.py could not:

  V1  OVER-DISPATCH AUDIT   every call whose CTA count > num_workers (136) runs
                            in >1 wave -> the SAME logical task executes twice
                            on the timeline (unnecessary). Flags wave count +
                            wasted wall. (the MLA-reduce 256-CTA finding.)

  V2  WITHIN-CALL IMBALANCE for each (task,occ) the min/median/max CTA BODY +
                            slow/fast ratio. A position whose AVERAGE looks fast
                            but has one slow CTA (load imbalance / different
                            shape under one name) is exposed here. (the group-
                            GEMM "some big some small" finding + Hypothesis-1.)

  V3  LAYER TIMELINE        one steady-state decode layer, positions sorted by
                            begin. Shows [t0,t1] offsets, whether each position
                            OVERLAPS the previous (parallel) or follows a GAP
                            (idle bubble = the whole GPU waited = pure schedule
                            overhead). Sums busy-union vs idle-gap. (the
                            Linear || Fine-n overlap question + bubble budget.)

Robust id->name: parses runtime_header.h so a CSV written with a stale
event_name_list (UNKNOWN_322/323, the 320-mislabel) is RE-LABELED correctly
here -- never trust the CSV's task_type_name column.

Usage: trace_schedule_analyzer.py <perfetto.csv> [num_workers=136]
"""
import sys, csv, os, re, statistics
from collections import defaultdict

CSV = sys.argv[1]
NUM_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 136
U32 = 1 << 32

def dur(b, e):
    x = (e - b) % U32
    return x if x < (1 << 31) else 0

# --- robust id->name from the C++ enum (do not trust the CSV's stale names) ---
def load_names():
    # Walk up from this file to find include/mirage/persistent_kernel/runtime_header.h
    # so the tool works from any location (package dir, scripts/, repo root).
    here = os.path.dirname(os.path.abspath(__file__))
    rel = os.path.join("include", "mirage", "persistent_kernel", "runtime_header.h")
    d = here
    for _ in range(8):
        p = os.path.join(d, rel)
        if os.path.exists(p):
            return {int(i): n for n, i in re.findall(
                r"\b(TASK_[A-Z0-9_]+)\s*=\s*(\d+)", open(p).read())}
        d = os.path.dirname(d)
    return {}
ID2NAME = load_names()

SKIP = {"TASK_SCHD_EVENTS", "TASK_SCHD_PREPARE_BATCH", "TASK_SCHD_TASKS",
        "TASK_GET_EVENT", "TASK_GET_NEXT_TASK", "TASK_BEGIN_TASK_GRAPH",
        "TASK_SM100_TASK_END", "TASK_NVSHMEM_GLOBAL_ARGMAX"}

# --- load events; re-label by numeric id ---
evs = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        try:
            tid = int(r["task_type_id"]); b = int(r["begin_ts"]); e = int(r["end_ts"])
            bi = int(r["block_idx"])
        except (KeyError, ValueError):
            continue
        if int(r.get("duration_ns", 1)) == 0:
            continue
        nm = ID2NAME.get(tid, r.get("task_type_name") or f"UNKNOWN_{tid}")
        evs.append((b, e, bi, nm, tid))
evs.sort()

# --- segment into decode layers on TOPK_SIGMOID boundaries (steady-state slice) ---
topk = sorted(b for b, _, _, nm, _ in evs if nm == "TASK_MOE_TOPK_SIGMOID_SM100")
segs = list(zip(topk, topk[1:]))
use = segs[2:9] if len(segs) > 9 else segs[1:]
print(f"# {os.path.basename(CSV)}  | num_workers={NUM_WORKERS} | "
      f"{len(topk)} topk markers -> {len(use)} steady-state layers analyzed")

def cluster(es):
    """Group same-type events into CALLS: within one call each block_idx
    appears once; a repeated block_idx (= a 2nd wave) starts a new call."""
    es = sorted(es); out = []; cur = [es[0]]; seen = {es[0][2]}
    for x in es[1:]:
        if x[2] in seen or x[0] - cur[-1][0] > 50000:
            out.append(cur); cur = [x]; seen = {x[2]}
        else:
            cur.append(x); seen.add(x[2])
    out.append(cur); return out

# Per (name,occ) collect calls across the analyzed layers (one call per layer).
pos_calls = defaultdict(list)   # (name,occ) -> list of call (each a list of evs)
# Logical-dispatch CTA counts (waves MERGED): (name,docc) -> list of total CTAs
disp_ctas = defaultdict(list)
# Per-layer ordered position list for V3 (use the FIRST analyzed layer).
layer_for_v3 = None
for li, (s, e0) in enumerate(use):
    layer = [(b, e, bi, nm, tid) for b, e, bi, nm, tid in evs
             if s <= b < e0 and nm not in SKIP]
    bytype = defaultdict(list)
    for x in layer:
        bytype[x[3]].append(x)
    insts = []
    for nm, xs in bytype.items():
        # When a call spans >1 wave, block_idx repeats -> cluster() splits it
        # into back-to-back clusters (the @1/@2 of MLA reduce). For V1 we MERGE
        # adjacent same-type clusters separated by < 2us (= waves of ONE
        # dispatch); a real 2nd call has a downstream-dep gap >> that.
        cls = cluster(xs)
        for w in cls:
            insts.append((min(x[0] for x in w), nm, w))
    insts.sort()
    occ = defaultdict(int)
    ordered = []
    for _, nm, w in insts:
        occ[nm] += 1
        pos_calls[(nm, occ[nm])].append(w)
        ordered.append((nm, occ[nm], w))
    # merge adjacent same-name positions (gap<2us) into logical dispatches
    docc = defaultdict(int)
    i = 0
    while i < len(ordered):
        nm, _, w = ordered[i]
        total = list(w)
        j = i + 1
        while j < len(ordered) and ordered[j][0] == nm and \
                (min(x[0] for x in ordered[j][2]) - max(x[1] for x in total)) % U32 < 2000:
            total += ordered[j][2]
            j += 1
        docc[nm] += 1
        disp_ctas[(nm, docc[nm])].append(total)   # merged events across waves
        i = j
    if li == 0:
        layer_for_v3 = (s, e0, ordered)

# ============================ V1: OVER-DISPATCH ============================
print("\n" + "=" * 90)
print("V1  OVER-DISPATCH AUDIT  (CTA count > num_workers -> runs in >1 wave; "
      "the SAME task twice)")
print("=" * 90)
print(f"{'position (logical dispatch)':<44}{'grid':>6}{'waves':>6}{'1wave_us':>9}{'allwave_us':>11}{'waste_us':>9}")
print("-" * 90)
v1rows = []
for (nm, oc), calls in disp_ctas.items():
    # worst layer by total instances (= logical grid size)
    best = max(calls, key=len)
    grid = len(best)                      # total task instances dispatched
    waves = (grid + NUM_WORKERS - 1) // NUM_WORKERS
    if waves <= 1:
        continue
    es = sorted(best)
    allwall = (max(x[1] for x in es) - min(x[0] for x in es)) % U32 / 1000.0
    # per-wave wall: instances split into waves by begin order
    perwave = []
    for wv in range(waves):
        chunk = es[wv*NUM_WORKERS:(wv+1)*NUM_WORKERS]
        if chunk:
            perwave.append((max(x[1] for x in chunk) - min(x[0] for x in chunk)) % U32 / 1000.0)
    onewave = max(perwave) if perwave else 0.0
    waste = allwall - onewave
    short = nm.replace("TASK_", "").replace("_SM100", "")
    v1rows.append((waste, f"{short} @{oc}", grid, waves, onewave, allwall, waste))
if v1rows:
    for _, p, ctas, waves, ow, aw, waste in sorted(v1rows, reverse=True):
        print(f"{p:<44}{ctas:>6}{waves:>6}{ow:>9.2f}{aw:>11.2f}{waste:>9.2f}")
    print(f"\n  -> total wasted wall from over-dispatch (per layer): "
          f"{sum(r[0] for r in v1rows):.2f} us  x{len(use)} layers analyzed")
else:
    print("  (none — every call fits in <=1 wave)")

# ====================== V2: WITHIN-CALL CTA IMBALANCE ======================
print("\n" + "=" * 90)
print("V2  WITHIN-CALL CTA IMBALANCE  (one call's CTA bodies: a low mean can "
      "hide a slow CTA / mixed shape)")
print("=" * 90)
print(f"{'position':<40}{'CTAs':>6}{'minCTA':>8}{'medCTA':>8}{'maxCTA':>8}{'max/med':>8}  flag")
print("-" * 90)
v2rows = []
for (nm, oc), calls in pos_calls.items():
    # pick the layer-call with the slowest max-CTA (worst case)
    best = max(calls, key=lambda w: max(dur(x[0], x[1]) for x in w))
    bodies = sorted(dur(x[0], x[1]) / 1000.0 for x in best)
    if not bodies:
        continue
    mn, mx = bodies[0], bodies[-1]
    med = statistics.median(bodies)
    ratio = mx / med if med > 1e-6 else 0.0
    ctas = len(set(x[2] for x in best))
    short = nm.replace("TASK_", "").replace("_SM100", "")
    v2rows.append((mx, short + f" @{oc}", ctas, mn, med, mx, ratio))
for _, p, ctas, mn, med, mx, ratio in sorted(v2rows, reverse=True)[:24]:
    flag = "IMBALANCED" if ratio >= 1.5 and ctas > 1 else ""
    print(f"{p:<40}{ctas:>6}{mn:>8.2f}{med:>8.2f}{mx:>8.2f}{ratio:>8.2f}  {flag}")

# ===================== V3: LAYER TIMELINE (serial/parallel) =================
print("\n" + "=" * 90)
print("V3  LAYER EXECUTION TIMELINE  (one steady-state layer; GAP=whole-GPU "
      "idle bubble; ‖=overlaps prev)")
print("=" * 90)
s, e0, ordered = layer_for_v3
layer_wall = (e0 - s) % U32 / 1000.0
# Build (t0,t1,name,occ,ctas) sorted by t0
spans = []
for nm, oc, w in ordered:
    t0 = (min(x[0] for x in w) - s) % U32 / 1000.0
    t1 = (max(x[1] for x in w) - s) % U32 / 1000.0
    ctas = len(set(x[2] for x in w))
    spans.append((t0, t1, nm, oc, ctas))
spans.sort()
print(f"layer wall = {layer_wall:.2f} us   ({len(spans)} positions)")
print(f"{'t0_us':>8}{'t1_us':>8}{'dur':>7}{'gapPrev':>8}{'CTAs':>6}  position")
print("-" * 90)
prev_t1 = 0.0
busy_union = []  # (t0,t1) for union-busy computation
gap_total = 0.0
for t0, t1, nm, oc, ctas in spans:
    gap = t0 - prev_t1
    marker = "‖" if t0 < prev_t1 - 0.05 else (f"GAP{gap:5.1f}" if gap > 0.3 else "")
    if gap > 0.3:
        gap_total += gap
    short = nm.replace("TASK_", "").replace("_SM100", "")
    print(f"{t0:>8.2f}{t1:>8.2f}{t1-t0:>7.2f}{gap:>8.2f}{ctas:>6}  {short} @{oc}  {marker}")
    busy_union.append((t0, t1))
    prev_t1 = max(prev_t1, t1)
# union-busy
busy_union.sort()
uni = 0.0; cs = ce = None
for a, b in busy_union:
    if cs is None:
        cs, ce = a, b
    elif a <= ce:
        ce = max(ce, b)
    else:
        uni += ce - cs; cs, ce = a, b
if cs is not None:
    uni += ce - cs
print("-" * 90)
print(f"  busy-union (>=1 CTA active) = {uni:.2f} us | whole-GPU-idle gaps = "
      f"{gap_total:.2f} us | layer wall = {layer_wall:.2f} us")
print(f"  => {100*gap_total/layer_wall:.0f}% of the layer is whole-GPU-idle "
      f"bubble (pure schedule/handoff between serial tasks)")
