#!/usr/bin/env python3
"""M4-I8 -- EXACT decomposition of (realized step - critical path).

WHY THIS TOOL EXISTS.  M4-I5 established the two bounds (critical path, work
bound); M4-status showed the step now sits 1.41-1.88x above the path.  Neither
tool can say WHAT the multiplier is made of: both are models over the static
graph plus per-TYPE mean durations.  This one reconstructs the REALIZED schedule
-- which worker ran which graph task, when -- and partitions the measured step
into named additive terms with no free parameters.

THE STRUCTURAL FACT THAT MAKES RECONSTRUCTION POSSIBLE (runtime.cc:972-993):

    // Prelaunch all tasks at the begining of an iteration
    all_events[1].first_task_id = 2;
    all_events[1].last_task_id  = all_tasks.size();
    for (e = 2; e < all_events.size(); e++)
      if (LAUNCH_TASKS || LAUNCH_MASSIVE_TASKS) {
        all_events[e].event_type = EVENT_EMPTY;
        for (t in [first,last)) all_tasks[t].dependent_event = e;
      }

MPK does not schedule dynamically.  Event 1 (LAUNCH_DEPENDENT_TASKS, triggered
by TASK_BEGIN_TASK_GRAPH) pushes EVERY task of the iteration into a worker queue
at iteration start; every other event is rewritten to EVENT_EMPTY and becomes a
pure counter.  A worker then drains its queue STRICTLY IN ORDER, blocking on
each task's `dependent_event` before it may start (persistent_kernel.cuh
:981-1009).  The "schedule" is a static round-robin of the graph's task order
over 128 in-order, blocking queues -- fixed at compile time.

The assignment is therefore derivable.  For EVENT_LAUNCH_DEPENDENT_TASKS the
scheduler owning workers [f,l) walks `position = first + i*num_workers + j`,
j in [f,l), pushing each to `next_worker` round-robin inside [f,l)
(persistent_kernel.cuh:1328-1376).  With num_workers == 128 that is exactly

    worker(p) = (p - 2) mod 128,   order within a worker = ascending p

up to one degeneracy: `get_first_last_ids(128, 80, s)` gives schedulers 0..47 a
PAIR of workers (2s, 2s+1) and schedulers 48..79 a single worker, and a
2-worker scheduler's persistent `next_worker` can be left one step out of phase
by an odd tail push or by the END_OF_TASK_GRAPH handler -- which swaps that
pair's two predicted sequences.  So the reconstruction is FITTED per pair (two
candidates) and then VERIFIED: the profiler records a task TYPE per event, so
the predicted per-worker type sequence must equal the observed one element for
element, on all 128 workers.  `assign_qc` reports that and FAIL is fatal.

WHAT COMES OUT.  Every profiled record on a worker track is a node.  Walking
backwards from the last record to finish, each node's start is explained by
exactly one binding predecessor:

  * DATA edge     -- its `dependent_event` had not yet fired; the predecessor is
                     the last producer of that event to finish.
  * RESOURCE edge -- the event had already fired but the worker was still
                     occupied; the predecessor is the previous record in that
                     worker's queue.

Because the predecessor of a RESOURCE edge is literally the previous record, the
sum telescopes and the partition of the measured step is EXACT (identity checked
to the nanosecond):

  step = head + SUM(node durations) + SUM(data gaps) + SUM(resource gaps) + tail

Node durations split three ways by what the node is: PATH work (a task entered
by a data edge -- the realized dependency chain), QUEUE work (a task the path
had to wait behind in its worker's queue), and TRIGGER work (TASK_SCHD_EVENTS
bookkeeping records, the per-event cost paid on the producer's worker).
QUEUE work + resource gaps IS the packing cost, mechanistically attributed
rather than inferred from a ratio.

Resource stalls are then re-examined machine-wide: at each stall, was some task
LATER in that same worker's queue already ready?  If yes the stall is
head-of-line blocking, recoverable by an out-of-order pop, and the recoverable
total is reported as a function of the reorder window W -- which the hardware
caps at TASK_DESCS_BUFFER_LENGTH (8 on sm100 with MPK_ENABLE_TMA).  If no, the
worker was genuinely starved and only a different task->worker mapping helps.

Finally the same reconstruction drives a discrete-event simulator.  Its
in-order (W=1) policy must reproduce the measured step; that is the validation.
Window-W, and a 128-server list schedule over the same durations, are then
predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from heapq import heappush, heappop
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import trace_lib as tl  # noqa: E402

EVENT_INVALID_ID = 0x7FFFFFFFFFFFFFFE
DEAD_NS = 1000.0   # per_task_table's "dispatched but idle" threshold
SIM_WINDOWS = (1, 2, 4, 8, 16, 32)


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------
def load_graph(path):
    g = json.load(open(path))
    tasks, events = g["all_tasks"], g["all_events"]
    n_t, n_e = len(tasks), len(events)
    ttype = np.zeros(n_t, dtype=np.int32)
    dep = np.full(n_t, -1, dtype=np.int64)
    trg = np.full(n_t, -1, dtype=np.int64)
    for p, t in enumerate(tasks):
        ttype[p] = t["task_type"]
        d, r = t["dependent_event"], t["trigger_event"]
        if d != EVENT_INVALID_ID:
            dep[p] = d & 0xFFFFFFFF
        if r != EVENT_INVALID_ID:
            trg[p] = r & 0xFFFFFFFF
    ntrig = np.array([e["num_triggers"] for e in events], dtype=np.int64)
    etype = np.array([e["event_type"] for e in events], dtype=np.int64)
    prod = defaultdict(list)
    for p in range(n_t):
        if trg[p] >= 0:
            prod[int(trg[p])].append(p)
    n_wp = sum(1 for e in range(n_e) if len(prod[e]) > 0)
    ff = sum(1 for e in range(n_e)
             if len(prod[e]) > 0 and ntrig[e] == len(prod[e]))
    return dict(n_tasks=n_t, n_events=n_e, ttype=ttype, dep=dep, trg=trg,
                ntrig=ntrig, etype=etype, prod=prod,
                full_fanin=ff, n_events_with_producers=n_wp,
                prelaunch_range=[events[1]["first_task_id"],
                                 events[1]["last_task_id"]],
                prelaunch_type=int(events[1]["event_type"]),
                n_empty=int((etype == 900).sum()))


# ---------------------------------------------------------------------------
# per-iteration worker record stream
# ---------------------------------------------------------------------------
def iteration_records(pairs, bounds, it, n_workers):
    """Records of one iteration, per worker, in queue order.  pair_events()
    sorted by (block, slot) and slot order is chronological inside a track, so a
    stable per-block slice is queue order."""
    lo, hi = int(bounds[it]), int(bounds[it + 1])
    b = pairs["begin"]
    idx = np.flatnonzero((pairs["block"] < n_workers) & (b >= lo) & (b < hi))
    blk = pairs["block"][idx]
    order = np.argsort(blk, kind="stable")
    idx, blk = idx[order], blk[order]
    starts = np.searchsorted(blk, np.arange(n_workers), side="left")
    stops = np.searchsorted(blk, np.arange(n_workers), side="right")
    per = {}
    for w in range(n_workers):
        ii = idx[starts[w]:stops[w]]
        per[w] = dict(tt=pairs["task_type"][ii], begin=pairs["begin"][ii],
                      end=pairs["end"][ii])
    return per, lo, hi


def predicted_order(gr, n_workers):
    first, last = gr["prelaunch_range"]
    pos = np.arange(first, last, dtype=np.int64)
    w = (pos - first) % n_workers
    return {k: pos[w == k] for k in range(n_workers)}


def fit_assignment(per, pred, gr, n_workers):
    tt = gr["ttype"]
    obs = {}
    for w in range(n_workers):
        m = (per[w]["tt"] != tl.TASK_SCHD_EVENTS) & \
            (per[w]["tt"] != tl.TASK_BEGIN_TASK_GRAPH)
        obs[w] = per[w]["tt"][m]

    def match(w, positions):
        o = obs[w]
        return len(o) == len(positions) and bool(np.array_equal(o,
                                                               tt[positions]))

    assign, n_direct, n_swapped, bad, done = {}, 0, 0, [], set()
    for w in range(n_workers):
        if w in done:
            continue
        if match(w, pred[w]):
            assign[w] = pred[w]; done.add(w); n_direct += 1
            continue
        partner = w + 1 if w % 2 == 0 else w - 1
        if partner < n_workers and partner not in done \
                and match(w, pred[partner]) and match(partner, pred[w]):
            assign[w], assign[partner] = pred[partner], pred[w]
            done.add(w); done.add(partner); n_swapped += 2
            continue
        bad.append(w); assign[w] = pred[w]; done.add(w)
    return assign, dict(n_workers=n_workers, n_direct=n_direct,
                        n_swapped=n_swapped, n_mismatch=len(bad),
                        mismatched_workers=bad[:8],
                        verdict="PASS" if not bad else "FAIL")


# ---------------------------------------------------------------------------
# the exact decomposition
# ---------------------------------------------------------------------------
def decompose(per, assign, gr, lo, hi, n_workers, names):
    n_t = gr["n_tasks"]
    start = np.full(n_t, -1, dtype=np.int64)
    end = np.full(n_t, -1, dtype=np.int64)
    wof = np.full(n_t, -1, dtype=np.int32)
    rix = np.full(n_t, -1, dtype=np.int32)
    rec, sev_us = {}, []
    for w in range(n_workers):
        tt, b, e = per[w]["tt"], per[w]["begin"], per[w]["end"]
        pos = np.full(len(tt), -1, dtype=np.int64)
        seq, k = assign[w], 0
        for i in range(len(tt)):
            if tt[i] == tl.TASK_SCHD_EVENTS:
                sev_us.append((e[i] - b[i]) / 1e3)
                continue
            if tt[i] == tl.TASK_BEGIN_TASK_GRAPH:
                continue
            p = int(seq[k]); k += 1
            pos[i] = p
            start[p], end[p], wof[p], rix[p] = b[i], e[i], w, i
        rec[w] = dict(tt=tt, begin=b, end=e, pos=pos, n=len(tt))

    prod = gr["prod"]
    fire = np.full(gr["n_events"], -1, dtype=np.int64)
    fire_arg = np.full(gr["n_events"], -1, dtype=np.int64)
    for e, ps in prod.items():
        if not ps:                 # load_graph's QC pass materialises empty keys
            continue
        es = [end[p] for p in ps]
        if min(es) < 0:
            continue
        j = int(np.argmax(es))
        fire[e] = es[j]; fire_arg[e] = ps[j]

    dep = gr["dep"]
    ready = np.full(n_t, -1, dtype=np.int64)
    have = np.flatnonzero(start >= 0)
    for p in have:
        d = dep[p]
        ready[p] = fire[d] if d >= 0 else lo

    # terminal = the last worker record to finish anywhere in the iteration
    tw = max(range(n_workers),
             key=lambda w: int(rec[w]["end"][-1]) if rec[w]["n"] else -1)
    cur = (tw, rec[tw]["n"] - 1)
    term_end = int(rec[tw]["end"][-1])

    chain, guard = [], 0
    while guard < 4_000_000:
        guard += 1
        w, i = cur
        r = rec[w]
        p = int(r["pos"][i])
        prev_end = int(r["end"][i - 1]) if i > 0 else lo
        rdy = int(ready[p]) if p >= 0 else -1
        if p >= 0 and rdy > prev_end:
            binding, gap = "data", int(r["begin"][i]) - rdy
            q = int(fire_arg[dep[p]]) if dep[p] >= 0 else -1
            nxt = (int(wof[q]), int(rix[q])) if q >= 0 and start[q] >= 0 else None
        else:
            binding, gap = "resource", int(r["begin"][i]) - prev_end
            nxt = (w, i - 1) if i > 0 else None
        kind = ("sev" if r["tt"][i] == tl.TASK_SCHD_EVENTS else
                "btg" if r["tt"][i] == tl.TASK_BEGIN_TASK_GRAPH else "task")
        chain.append(dict(pos=p, tt=int(r["tt"][i]), kind=kind, worker=w,
                          rix=i, dur_ns=int(r["end"][i] - r["begin"][i]),
                          binding=binding, gap_ns=int(gap),
                          start=int(r["begin"][i]), ready_ns=rdy,
                          prev_end=prev_end))
        if nxt is None:
            break
        cur = nxt
    chain.reverse()

    head_ns = chain[0]["start"] - chain[0]["gap_ns"] - lo
    tail_ns = hi - term_end
    sum_dur = sum(c["dur_ns"] for c in chain)
    sum_gap = sum(c["gap_ns"] for c in chain)
    ident = head_ns + sum_dur + sum_gap + tail_ns - (hi - lo)

    def us(x):
        return x / 1e3

    # a node's duration is PATH work if its SUCCESSOR reached it by a data edge
    path_w = q_w = trig_w = 0
    for k, c in enumerate(chain):
        nxt_bind = chain[k + 1]["binding"] if k + 1 < len(chain) else "data"
        if c["kind"] != "task":
            trig_w += c["dur_ns"]
        elif nxt_bind == "resource":
            q_w += c["dur_ns"]
        else:
            path_w += c["dur_ns"]
    data_gap = sum(c["gap_ns"] for c in chain if c["binding"] == "data")
    res_gap = sum(c["gap_ns"] for c in chain if c["binding"] == "resource")

    by_type = defaultdict(lambda: dict(n=0, dur_ns=0, role=Counter()))
    for k, c in enumerate(chain):
        nxt_bind = chain[k + 1]["binding"] if k + 1 < len(chain) else "data"
        r = by_type[c["tt"]]
        r["n"] += 1
        r["dur_ns"] += c["dur_ns"]
        r["role"]["trigger" if c["kind"] != "task"
                  else "queue" if nxt_bind == "resource" else "path"] += 1

    # ---- machine-wide stall attribution ----------------------------------
    # Two different questions, both asked.  (1) FULLY recoverable: some task
    # already READY when the worker went free sits within W of the head, so the
    # whole stall is avoidable.  (2) PARTIALLY recoverable: no such task, but
    # some task within W becomes ready EARLIER than the head does, so part of
    # the stall can be filled.  The first-order pass over the realized trace
    # only measures (1); (2) is what a reordering simulator actually exploits,
    # so reporting only (1) would refute the lever for the wrong reason.
    W_LIST = list(SIM_WINDOWS) + [64]
    tot = poll = starved = 0
    hol_full = {W: 0 for W in W_LIST}
    hol_part = {W: 0 for W in W_LIST}
    n_stall = 0
    for w in range(n_workers):
        r = rec[w]
        seq = assign[w]
        kk, qpos = 0, {}
        for i in range(r["n"]):
            if r["pos"][i] >= 0:
                qpos[i] = kk; kk += 1
        for i in range(r["n"]):
            p = int(r["pos"][i])
            if p < 0:
                continue
            prev_end = int(r["end"][i - 1]) if i > 0 else lo
            idle = int(r["begin"][i]) - prev_end
            if idle <= 0:
                continue
            tot += idle; n_stall += 1
            rdy_p = int(ready[p])
            if rdy_p <= prev_end:
                poll += idle
                continue
            k0 = qpos[i]
            span = min(len(seq) - k0, max(W_LIST))
            best_full, best_rdy = None, None
            for off in range(1, span):
                q = int(seq[k0 + off])
                if start[q] < 0 or ready[q] < 0:
                    continue
                rq = int(ready[q])
                if best_rdy is None or rq < best_rdy[0]:
                    best_rdy = (rq, off)
                if rq <= prev_end and best_full is None:
                    best_full = off
            if best_full is None and (best_rdy is None
                                      or best_rdy[0] >= rdy_p):
                starved += idle
                continue
            for W in W_LIST:
                if best_full is not None and best_full < W:
                    hol_full[W] += idle
                # partially recoverable: the earliest-ready task within W
                cand = None
                for off in range(1, min(span, W)):
                    q = int(seq[k0 + off])
                    if start[q] < 0 or ready[q] < 0:
                        continue
                    rq = int(ready[q])
                    if cand is None or rq < cand:
                        cand = rq
                if cand is not None and cand < rdy_p:
                    hol_part[W] += max(0, min(rdy_p, int(r["begin"][i]))
                                       - max(prev_end, cand))

    return dict(
        window_ns=hi - lo, step_us=us(hi - lo), identity_error_ns=int(ident),
        chain_len=len(chain),
        chain_n_task=sum(1 for c in chain if c["kind"] == "task"),
        chain_n_sev=sum(1 for c in chain if c["kind"] == "sev"),
        head_us=us(head_ns), tail_us=us(tail_ns),
        path_work_us=us(path_w), queue_work_us=us(q_w),
        trigger_work_us=us(trig_w),
        data_gap_us=us(data_gap), resource_gap_us=us(res_gap),
        n_data_edges=sum(1 for c in chain if c["binding"] == "data"),
        n_resource_edges=sum(1 for c in chain if c["binding"] == "resource"),
        chain_dead_tasks=sum(1 for c in chain
                             if c["kind"] == "task" and c["dur_ns"] < DEAD_NS),
        chain_dead_us=us(sum(c["dur_ns"] for c in chain
                             if c["kind"] == "task" and c["dur_ns"] < DEAD_NS)),
        sev_n=len(sev_us), sev_mean_us=float(np.mean(sev_us)) if sev_us else 0.,
        sev_total_us=float(np.sum(sev_us)) if sev_us else 0.,
        data_gap_median_ns=float(np.median([c["gap_ns"] for c in chain
                                            if c["binding"] == "data"]) or 0)
        if any(c["binding"] == "data" for c in chain) else 0.,
        res_gap_median_ns=float(np.median([c["gap_ns"] for c in chain
                                           if c["binding"] == "resource"]))
        if any(c["binding"] == "resource" for c in chain) else 0.,
        machine_idle_us=us(tot), machine_idle_poll_us=us(poll),
        machine_idle_starved_us=us(starved), n_stalls=n_stall,
        machine_idle_hol_full_us_by_window={str(W): us(v)
                                            for W, v in sorted(
                                                hol_full.items())},
        machine_idle_hol_partial_us_by_window={str(W): us(v)
                                               for W, v in sorted(
                                                   hol_part.items())},
        chain_by_type=[dict(task_type=t, name=names.get(str(t), f"T{t}"),
                            n=v["n"], dur_us=us(v["dur_ns"]),
                            as_path=v["role"]["path"],
                            as_queue=v["role"]["queue"],
                            as_trigger=v["role"]["trigger"])
                       for t, v in sorted(by_type.items(),
                                          key=lambda kv: -kv[1]["dur_ns"])],
    ), dict(chain=chain, start=start, end=end, ready=ready, fire=fire)


# ---------------------------------------------------------------------------
# simulators
# ---------------------------------------------------------------------------
def simulate_window(gr, assign, dur, n_workers, window, l_poll, l_trig, l_sev,
                    batch=False):
    """MPK's own policy for window=1: static per-worker FIFO, blocking pop.

    window=W lets a worker pop the first READY task within W of its head.
    batch=False is a SLIDING window -- always W un-run candidates, which in the
    kernel needs the smem task-desc buffer compacted and topped up.
    batch=True is the BATCHED buffer MPK actually has: load W descs, drain them
    out of order, and only refill once the buffer is EMPTY.  The real
    TASK_DESCS_BUFFER_LENGTH is 8 on sm100 (WORKER_RESERVED_STATIC_SHARED_MEMORY
    _SIZE 3 KiB, sizeof(TaskDesc) 352 B with MPK_ENABLE_TMA), so batch=True,
    W=8 is what a minimal kernel change can deliver and slide is the ceiling."""
    dep, trg = gr["dep"], gr["trg"]
    n_prod = np.zeros(gr["n_events"], dtype=np.int64)
    for e, ps in gr["prod"].items():
        n_prod[e] = len(ps)
    ctr = np.zeros(gr["n_events"], dtype=np.int64)
    # fire_t[e] = the finish time of the LAST producer of e to finish.  It is
    # resolved only once every producer has been scheduled, and it is the MAX of
    # their finishes -- not the finish of whichever producer happened to be
    # scheduled last.  Getting this wrong let consumers start before their
    # producers had finished (6077 violations at bs1, worst 62.2 us); the
    # dep_qc field below is the standing check that it stays right.
    fire_t = np.full(gr["n_events"], -1, dtype=np.int64)
    maxfin = np.zeros(gr["n_events"], dtype=np.int64)
    q = {w: list(assign[w]) for w in range(n_workers)}
    nxt = {w: 0 for w in range(n_workers)}       # next unloaded queue index
    buf = {w: [] for w in range(n_workers)}      # loaded-but-unrun descs
    waiters = defaultdict(set)
    ev, done, makespan = [], 0, 0
    total = sum(len(q[w]) for w in range(n_workers))
    # A worker can sit in several waiters[] sets at once, so a later fire can
    # wake it at a timestamp EARLIER than the task it is currently running
    # finishes.  free_at is what stops a busy worker being double-booked into
    # the past -- without it the zero-latency variants schedule below the
    # critical path, which is how this bug was caught.
    free_at = np.zeros(n_workers, dtype=np.int64)
    sim_start, sim_fin = {}, {}
    for w in range(n_workers):
        heappush(ev, (0, w))
    guard = 0
    while done < total and ev and guard < 60 * total + 4_000_000:
        guard += 1
        t, w = heappop(ev)
        if free_at[w] > t:
            heappush(ev, (int(free_at[w]), w))
            continue
        if batch:
            if not buf[w]:                       # refill only when EMPTY
                buf[w] = q[w][nxt[w]:nxt[w] + window]
                nxt[w] += len(buf[w])
            cand = buf[w]
        else:
            cand = q[w][nxt[w]:nxt[w] + window]  # sliding
        if not cand:
            continue
        pick, off, future = -1, -1, None
        for o, p in enumerate(cand):
            d = dep[p]
            if d < 0:
                pick, off = p, o
                break
            ft = fire_t[d]
            if ft >= 0 and ft + l_trig <= t:
                pick, off = p, o
                break
            if ft >= 0:
                ready_at = int(ft) + l_trig
                future = ready_at if future is None else min(future, ready_at)
        if pick < 0:
            for p in cand:
                d = dep[p]
                if d >= 0 and fire_t[d] < 0:
                    waiters[int(d)].add(w)
            if future is not None:
                heappush(ev, (future, w))
            continue
        if batch:
            del buf[w][off]
        else:
            del q[w][nxt[w] + off]
        st = t + l_poll
        fin = st + int(dur[pick])
        after = fin
        sim_start[int(pick)], sim_fin[int(pick)] = st, fin
        e = int(trg[pick])
        if e >= 0:
            ctr[e] += 1
            if fin > maxfin[e]:
                maxfin[e] = fin
            if ctr[e] == n_prod[e]:
                fire_t[e] = maxfin[e]
                after = fin + l_sev
                for ww in waiters.pop(e, ()):
                    heappush(ev, (int(maxfin[e]) + l_trig, ww))
        makespan = max(makespan, after)
        free_at[w] = after
        done += 1
        heappush(ev, (int(after), w))
    return dict(makespan_us=makespan / 1e3, done=done, total=total,
                complete=bool(done == total),
                dep_qc=verify_schedule(gr, sim_start, sim_fin))


def verify_schedule(gr, sim_start, sim_fin):
    """Hard invariant: no task may start before EVERY producer of its
    dependent_event has finished.  A simulated makespan that has not passed
    this is not a bound, it is a bug."""
    dep, prod = gr["dep"], gr["prod"]
    worst, n_bad = 0, 0
    fire = {}
    for e, ps in prod.items():
        if not ps:
            continue
        fs = [sim_fin.get(int(p)) for p in ps]
        if any(f is None for f in fs):
            continue
        fire[e] = max(fs)
    for p, st in sim_start.items():
        d = int(dep[p])
        if d < 0 or d not in fire:
            continue
        v = fire[d] - st
        if v > 0:
            n_bad += 1
            worst = max(worst, v)
    return dict(verdict="PASS" if n_bad == 0 else "FAIL", n_violations=n_bad,
                worst_violation_ns=int(worst))


def cp_priority(gr, dur, positions):
    """Longest remaining weighted path to a sink, per task -- the standard
    critical-path list-scheduling priority (HLFET).  Computed on the event DAG:
    prio(task) = dur(task) + max over events it triggers of max prio(consumer)."""
    dep, trg = gr["dep"], gr["trg"]
    cons = defaultdict(list)
    for p in positions:
        d = dep[p]
        if d >= 0:
            cons[int(d)].append(int(p))
    prio = {}
    order = sorted((int(p) for p in positions), reverse=True)
    ev_best = defaultdict(int)
    for p in order:                     # positions are topologically ordered
        e = int(trg[p])
        nxt = ev_best[e] if e >= 0 else 0
        prio[p] = int(dur[p]) + nxt
        d = dep[p]
        if d >= 0 and prio[p] > ev_best[int(d)]:
            ev_best[int(d)] = prio[p]
    return prio


def list_schedule(gr, dur, positions, n_workers, l_poll, l_trig, l_sev,
                  priority=None):
    """128-server list schedule over the SAME dependency graph and the SAME
    measured durations, with NO worker affinity: what a dynamic scheduler could
    do with these task times.  `priority` (default: longest remaining path)
    breaks ties among simultaneously-ready tasks; FIFO-by-ready-time is a poor
    priority and can lose to an affinity policy, which is why HLFET is used."""
    dep, trg = gr["dep"], gr["trg"]
    n_prod = np.zeros(gr["n_events"], dtype=np.int64)
    for e, ps in gr["prod"].items():
        n_prod[e] = len(ps)
    ctr = np.zeros(gr["n_events"], dtype=np.int64)
    maxfin = np.zeros(gr["n_events"], dtype=np.int64)
    consumers = defaultdict(list)
    for p in positions:
        d = dep[p]
        if d >= 0:
            consumers[int(d)].append(int(p))
    if priority is None:
        priority = cp_priority(gr, dur, positions)
    n = len(positions)
    # (available_time, -priority, pos)
    avail = []
    for p in positions:
        if dep[p] < 0:
            heappush(avail, (0, -priority[int(p)], int(p)))
    free = [(0, w) for w in range(n_workers)]
    import heapq
    heapq.heapify(free)
    ready_now, t_now = [], 0
    sim_start, sim_fin = {}, {}
    done, makespan, guard = 0, 0, 0
    while done < n and guard < 30 * n + 1_000_000:
        guard += 1
        t_w, w = heappop(free)
        t_now = max(t_now, t_w)
        # move everything available by now into the priority pool
        while avail and avail[0][0] <= t_now:
            ta, negp, p = heappop(avail)
            heappush(ready_now, (negp, p))
        if not ready_now:
            if not avail:
                heappush(free, (t_w, w))
                break
            t_now = max(t_now, avail[0][0])
            heappush(free, (t_now, w))
            continue
        negp, p = heappop(ready_now)
        st = t_now + l_poll
        fin = st + int(dur[p])
        after = fin
        e = int(trg[p])
        if e >= 0:
            ctr[e] += 1
            if fin > maxfin[e]:
                maxfin[e] = fin
            if ctr[e] == n_prod[e]:
                after = fin + l_sev
                rel = int(maxfin[e]) + l_trig
                for c in consumers.get(e, ()):
                    heappush(avail, (rel, -priority[c], c))
        heappush(free, (after, w))
        sim_start[int(p)], sim_fin[int(p)] = st, fin
        makespan = max(makespan, after)
        done += 1
    return dict(makespan_us=makespan / 1e3, done=done, total=n,
                complete=bool(done == n),
                dep_qc=verify_schedule(gr, sim_start, sim_fin))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("meta"); ap.add_argument("names")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--window", required=True, help="lo,hi iteration window")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    names = tl.load_names(Path(a.names))
    gr = load_graph(a.graph)
    z = np.load(a.raw)                      # sparse dump: idx / val / header
    idx, val = z["idx"], z["val"]
    buf = np.zeros(int(idx.max()) + 1, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    buf[:1] = z["header"].view(np.uint64)
    del idx, val, z
    ev = tl.decode_events(buf)
    del buf
    pairs = tl.pair_events(ev)
    bounds = tl.iteration_bounds(pairs)
    lo_it, hi_it = (int(x) for x in a.window.split(","))

    out = dict(raw=a.raw, graph=a.graph, batch_size=meta.get("batch_size"),
               window=[lo_it, hi_it], n_iterations=len(bounds) - 1,
               nblocks=int(ev["nblocks"]), ngroups=int(ev["ngroups"]),
               n_tasks=gr["n_tasks"], n_events=gr["n_events"],
               graph_qc=dict(prelaunch_event_type=gr["prelaunch_type"],
                             prelaunch_range=gr["prelaunch_range"],
                             n_empty_events=gr["n_empty"],
                             full_fanin=gr["full_fanin"],
                             n_events_with_producers=gr[
                                 "n_events_with_producers"]),
               dropped_begin=pairs["dropped_begin"],
               dropped_end=pairs["dropped_end"], iterations=[])
    pred = predicted_order(gr, a.workers)
    its = list(range(lo_it, min(hi_it, len(bounds) - 1)))[:a.iters]
    keep = None
    for it in its:
        per, lo, hi = iteration_records(pairs, bounds, it, a.workers)
        assign, qc = fit_assignment(per, pred, gr, a.workers)
        r = dict(iteration=it, assign_qc=qc)
        if qc["verdict"] != "PASS":
            out["iterations"].append(r); continue
        d, aux = decompose(per, assign, gr, lo, hi, a.workers, names)
        r.update(d)
        r["chain_head"] = aux["chain"][:5]
        r["chain_tail"] = aux["chain"][-5:]
        out["iterations"].append(r)
        if keep is None:
            dur = (aux["end"] - aux["start"]).astype(np.int64)
            dur[aux["start"] < 0] = 0
            keep = (assign, dur, d)

    if a.sim and keep is not None:
        assign, dur, d = keep
        L = dict(l_poll=int(d["res_gap_median_ns"]),
                 l_trig=int(d["data_gap_median_ns"]),
                 l_sev=int(d["sev_mean_us"] * 1000))
        Z = dict(l_poll=0, l_trig=0, l_sev=0)
        sims = {}
        for W in SIM_WINDOWS:
            sims[f"slide_{W}"] = simulate_window(
                gr, assign, dur, a.workers, W, **L)
            if W > 1:
                sims[f"batch_{W}"] = simulate_window(
                    gr, assign, dur, a.workers, W, batch=True, **L)
        # zero-latency counterfactuals: how much of each policy's makespan is
        # the spin-wait/poll latency rather than work or dependency structure
        for W in (1, 8):
            sims[f"slide_{W}_nolat"] = simulate_window(
                gr, assign, dur, a.workers, W, **Z)
        allpos = np.concatenate([assign[w] for w in range(a.workers)])
        sims["list_schedule"] = list_schedule(gr, dur, allpos, a.workers, **L)
        sims["list_schedule_nolat"] = list_schedule(gr, dur, allpos, a.workers,
                                                    **Z)
        out["sim_latencies_ns"] = L
        out["sim"] = sims
        # EXACT floors for this iteration, from the same per-task durations the
        # sims use.  cp_decompose.py's cp is a MODEL (static longest chain,
        # per-TYPE mean T_live under levelmax), so it can sit either side of the
        # realized longest path; this is the realized one.
        prio = cp_priority(gr, dur, allpos)
        tot = int(sum(int(dur[p]) for p in allpos))
        cp0 = max(prio.values()) / 1e3
        out["floors"] = dict(
            cp_exact_us=cp0,
            work_bound_us=tot / a.workers / 1e3,
            total_task_us=tot / 1e3,
            n_dead_tasks=int(sum(1 for p in allpos if 0 < dur[p] < DEAD_NS)),
            dead_task_us=float(sum(int(dur[p]) for p in allpos
                                   if 0 < dur[p] < DEAD_NS)) / 1e3)
        # ---- COUNTERFACTUALS on the exact floor ---------------------------
        # Both floors are what remain after every scheduling and latency fix, so
        # the only way past them is to change the GRAPH: make a stage cheaper, or
        # take it off the chain entirely.  Zeroing one task type's duration
        # prices "this stage becomes free / gets fused into its neighbour" for
        # both floors at once, which is the honest way to rank the next levers.
        ttype = gr["ttype"]
        chain_types = [r["task_type"] for r in d["chain_by_type"]][:14]
        cf = []
        for t in chain_types:
            d2 = dur.copy()
            m = (ttype == t)
            removed = int(d2[m].sum())
            d2[m] = 0
            p2 = cp_priority(gr, d2, allpos)
            cf.append(dict(task_type=t, name=names.get(str(t), f"T{t}"),
                           cp_exact_us=max(p2.values()) / 1e3,
                           cp_delta_us=cp0 - max(p2.values()) / 1e3,
                           work_bound_us=(tot - removed) / a.workers / 1e3,
                           removed_task_us=removed / 1e3))
        cf.sort(key=lambda r: -r["cp_delta_us"])
        out["floor_counterfactuals"] = cf

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"=== bs{out['batch_size']} win {a.window} tasks={gr['n_tasks']} "
          f"events={gr['n_events']} full_fanin={gr['full_fanin']}/"
          f"{gr['n_events_with_producers']} empty={gr['n_empty']} "
          f"prelaunch={gr['prelaunch_range']} (type {gr['prelaunch_type']})")
    for r in out["iterations"]:
        if "step_us" not in r:
            print(f"  it{r['iteration']}: assign_qc FAIL {r['assign_qc']}")
            continue
        q = r["assign_qc"]
        print(f"  it{r['iteration']}: step={r['step_us']:.1f}us "
              f"assign={q['verdict']}(direct={q['n_direct']},swap="
              f"{q['n_swapped']}) ident_err={r['identity_error_ns']}ns "
              f"chain={r['chain_len']}({r['chain_n_task']}task+"
              f"{r['chain_n_sev']}sev)")
        print(f"     PATHwork={r['path_work_us']:.1f} "
              f"QUEUEwork={r['queue_work_us']:.1f} "
              f"TRIGwork={r['trigger_work_us']:.1f} | "
              f"data_gap={r['data_gap_us']:.1f} res_gap={r['resource_gap_us']:.1f}"
              f" | head={r['head_us']:.1f} tail={r['tail_us']:.1f}")
        print(f"     edges data={r['n_data_edges']} res={r['n_resource_edges']} "
              f"| gap med data={r['data_gap_median_ns']:.0f}ns "
              f"res={r['res_gap_median_ns']:.0f}ns "
              f"| sev {r['sev_n']}x{r['sev_mean_us']:.3f}us"
              f" | chain dead {r['chain_dead_tasks']}="
              f"{r['chain_dead_us']:.1f}us")
        print(f"     machine idle={r['machine_idle_us']:.0f}us poll="
              f"{r['machine_idle_poll_us']:.0f} starved="
              f"{r['machine_idle_starved_us']:.0f}")
        print("     hol_full="
              f"{ {k: round(v) for k, v in r['machine_idle_hol_full_us_by_window'].items()} }")
        print("     hol_part="
              f"{ {k: round(v) for k, v in r['machine_idle_hol_partial_us_by_window'].items()} }")
    if "floors" in out:
        print("  floors:", {k: (round(v, 1) if isinstance(v, float) else v)
                            for k, v in out["floors"].items()})
    if "sim" in out:
        print("  sim_lat_ns:", out["sim_latencies_ns"])
        print("  sim:", {k: (round(v['makespan_us'], 1), v['complete'],
                             v['dep_qc']['verdict'])
                         for k, v in out["sim"].items()})
        bad = {k: v['dep_qc'] for k, v in out["sim"].items()
               if v['dep_qc']['verdict'] != 'PASS'}
        if bad:
            print("  SIM DEP QC FAILURES:", bad)
    if a.out:
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
