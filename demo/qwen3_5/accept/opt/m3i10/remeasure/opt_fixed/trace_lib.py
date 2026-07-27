#!/usr/bin/env python3
"""Vectorised reader/attributor for the MPK persistent-kernel profiler buffer.

Why this exists instead of ``mirage.mpk.profiler_persistent``: the stock
exporters walk the device buffer one element at a time in Python
(``for i in range(1, len(profiler_buffer_host))``) and then hand *every* event
to ``tg4perfetto``.  A full-length Qwen3.5 decode wave emits 10-15 M events
(41 048 tasks/iteration at bs1, 59 348 at bs16, x2 events, x~110-145
iterations), which makes both exporters unusable -- minutes of Python plus a
multi-GB trace nobody can open.  Everything here is numpy over the same buffer
bytes, and the Perfetto export is windowed to a handful of iterations.

Buffer layout (include/mirage/persistent_kernel/profiler.h):

    slot 0                 : header, {u32 nblocks, u32 ngroups}
    slot 1 + b*G + g       : first event of track (block b, group g)
    stride                 : nblocks * ngroups

so a track's slots are strictly increasing in *slot index* and therefore also
in time.  Tag layout (32 bits): ``[31:19] event_no | [18:11] block*G+g |
[10:2] task-type id | [1:0] begin/end/instant``.  The paired u32 is
``%globaltimer_lo`` -- nanoseconds, but only the low 32 bits, so it wraps every
4.295 s.  Waves here run 1.5-4.6 s, so unwrapping is mandatory, not optional;
``decode_events`` does it per track.

Two facts about *placement* drive the whole attribution, both from
``persistent_kernel.cuh``:

* ``PROFILER_EVENT_START`` is emitted **after** the worker's dependency-wait
  loop (``while (actual_counts < needed_counts)``), so a worker blocked on an
  event contributes a *gap* on its track, never a long task.
* ``TASK_SCHD_PREPARE_BATCH`` is emitted by the scheduler warp that consumes
  ``EVENT_END_OF_TASK_GRAPH``; ``TASK_BEGIN_TASK_GRAPH`` (task type 10) is the
  first worker task of the next iteration.  The latter is the iteration
  delimiter used here -- exactly one per iteration, on a rotating worker.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EVENT_BEGIN, EVENT_END, EVENT_INSTANT = 0, 1, 2

TASK_TERMINATE = 0
TASK_BEGIN_TASK_GRAPH = 10
TASK_SCHD_TASKS = 200
TASK_SCHD_EVENTS = 201
TASK_GET_EVENT = 202
TASK_GET_NEXT_TASK = 203
TASK_SCHD_PREPARE_BATCH = 204
SCHED_TASK_TYPES = (TASK_SCHD_TASKS, TASK_SCHD_EVENTS, TASK_GET_EVENT,
                    TASK_GET_NEXT_TASK, TASK_SCHD_PREPARE_BATCH)

# task-type id -> coarse layer bucket for the per-layer-type table.  Ids are
# `enum TaskType` (runtime_header.h); the qwen3.5 graph only uses the subset
# marked (*), the rest are here so a future graph change shows up under a real
# name instead of "other".
LAYER_BUCKET = {
    101: "embedding",             # *
    102: "norm",
    118: "moe_silu_mul",          # * SiLU-mul between w13 and w2 (MoE act)
    119: "norm",
    154: "norm",                  # * RMS_NORM_HOPPER
    232: "copy",
    233: "concat",
    234: "gdn_conv",              # *
    237: "gdn_recurrent",         # *
    238: "gdn_gate",              # * sigmoid gate mul add
    241: "moe_w13",               # * fp8 blockscale
    242: "moe_w2",                # * fp8 blockscale
    248: "moe_w13",
    249: "moe_w2",
    251: "dense_proj",
    252: "dense_proj",
    253: "dense_proj",            # * LINEAR_SM100
    254: "moe_w13",
    255: "moe_w2",
    257: "attention",             # * ATTN_SM100
    258: "argmax",                # *
    259: "argmax",                # *
    260: "moe_router",            # * topk softmax
    261: "moe_combine",           # * mul sum add
    262: "tensor_init",
    263: "attention",
    264: "attention",
    275: "quantize_fp8",          # *
    276: "dense_proj",
    277: "dense_proj",
    279: "dense_proj",            # * LINEAR_FP8_BLOCKSCALE
    280: "moe_router",
    281: "elementwise_add",
    296: "attention",
    297: "attention_norm_rope",
    298: "kv_write",
    200: "sched", 201: "sched_events", 202: "sched", 203: "sched",
    204: "sched_prepare_batch",
    10: "begin_task_graph", 0: "terminate",
}


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------
def decode_events(buf: np.ndarray) -> dict:
    """u64 profiler buffer -> per-event arrays, timestamps unwrapped to ns."""
    buf = np.asarray(buf)
    assert buf.dtype == np.uint64, buf.dtype
    w = buf.view(np.uint32).reshape(-1, 2)
    nblocks, ngroups = int(w[0, 0]), int(w[0, 1])
    if ngroups < 1:
        ngroups = 1

    slot = np.flatnonzero(buf)
    slot = slot[slot > 0]
    tag = w[slot, 0].astype(np.uint32)
    ts32 = w[slot, 1].astype(np.uint64)

    event_no = (tag >> 19).astype(np.int32)
    bg = ((tag >> 11) & 0xFF).astype(np.int32)
    task_type = ((tag >> 2) & 0x1FF).astype(np.int32)
    etype = (tag & 0x3).astype(np.int8)
    block = bg // ngroups
    group = bg % ngroups

    # unwrap the 32-bit globaltimer per (block,group) track.  slot index is
    # ascending == chronological within a track, and np.flatnonzero already
    # returns ascending indices, so a stable sort by track preserves that.
    track = bg
    order = np.argsort(track, kind="stable")
    ts = np.empty(len(slot), dtype=np.int64)
    t_sorted = ts32[order].astype(np.int64)
    tr_sorted = track[order]
    new_track = np.empty(len(slot), dtype=bool)
    new_track[0] = True
    new_track[1:] = tr_sorted[1:] != tr_sorted[:-1]
    d = np.zeros(len(slot), dtype=np.int64)
    d[1:] = (t_sorted[1:] < t_sorted[:-1]) & (~new_track[1:])
    # cumulative wrap count, reset at every track boundary
    wraps = np.cumsum(d)
    base = np.maximum.accumulate(np.where(new_track, wraps, 0))
    ts[order] = t_sorted + ((wraps - base) << 32)

    return dict(nblocks=nblocks, ngroups=ngroups, slot=slot, block=block,
                group=group, task_type=task_type, etype=etype,
                event_no=event_no, ts=ts, n_events=len(slot))


def pair_events(ev: dict) -> dict:
    """BEGIN/END -> intervals.  Per track events are strictly
    BEGIN,END,BEGIN,END (no nesting: the worker's SCHD_EVENTS pair is emitted
    after the task's END), so pairing is positional.  A trailing unmatched
    BEGIN means the buffer filled up mid-run and is dropped, with a count."""
    keep = ev["etype"] != EVENT_INSTANT
    block = ev["block"][keep]
    tt = ev["task_type"][keep]
    et = ev["etype"][keep]
    ts = ev["ts"][keep]
    slot = ev["slot"][keep]

    order = np.lexsort((slot, block))
    block, tt, et, ts = block[order], tt[order], et[order], ts[order]

    is_begin = et == EVENT_BEGIN
    is_end = ~is_begin
    # a valid pair: begin at i, end at i+1, same block, same task type
    ok = np.zeros(len(block), dtype=bool)
    ok[:-1] = (is_begin[:-1] & is_end[1:] & (block[:-1] == block[1:])
               & (tt[:-1] == tt[1:]))
    bi = np.flatnonzero(ok)
    dropped_begin = int(is_begin.sum() - len(bi))
    dropped_end = int(is_end.sum() - len(bi))
    return dict(block=block[bi], task_type=tt[bi], begin=ts[bi],
                end=ts[bi + 1], dur=ts[bi + 1] - ts[bi],
                dropped_begin=dropped_begin, dropped_end=dropped_end)


# ---------------------------------------------------------------------------
# iteration segmentation + attribution
# ---------------------------------------------------------------------------
def iteration_bounds(pairs: dict) -> np.ndarray:
    """Ascending BEGIN_TASK_GRAPH start timestamps = iteration boundaries."""
    m = pairs["task_type"] == TASK_BEGIN_TASK_GRAPH
    t = np.sort(pairs["begin"][m])
    return t


def union_length(begin: np.ndarray, end: np.ndarray) -> int:
    """Total length covered by the union of [begin,end) intervals."""
    if len(begin) == 0:
        return 0
    o = np.argsort(begin, kind="stable")
    b, e = begin[o], end[o]
    emax = np.maximum.accumulate(e)
    new = np.empty(len(b), dtype=bool)
    new[0] = True
    new[1:] = b[1:] > emax[:-1]
    idx = np.flatnonzero(new)
    seg_end = np.maximum.reduceat(e, idx)
    return int((seg_end - b[idx]).sum())


def attribute(pairs: dict, bounds: np.ndarray, n_workers: int) -> dict:
    """Per-iteration attribution.  Returns arrays of length len(bounds)-1."""
    is_worker = pairs["block"] < n_workers
    tt = pairs["task_type"]
    is_task = is_worker & (tt != TASK_SCHD_EVENTS) & (tt != TASK_BEGIN_TASK_GRAPH)
    is_sev = is_worker & (tt == TASK_SCHD_EVENTS)
    is_prep = tt == TASK_SCHD_PREPARE_BATCH

    b, e, d = pairs["begin"], pairs["end"], pairs["dur"]
    it = np.searchsorted(bounds, b, side="right") - 1
    n_it = len(bounds) - 1

    in_win = (it >= 0) & (it < n_it)

    def bin_sum(mask):
        sel = mask & in_win
        return np.bincount(it[sel], weights=d[sel].astype(np.float64),
                           minlength=n_it)[:n_it]

    def bin_count(mask):
        sel = mask & in_win
        return np.bincount(it[sel], minlength=n_it)[:n_it].astype(np.int64)

    dur_it = np.diff(bounds)
    task_ns = bin_sum(is_task)
    sev_ns = bin_sum(is_sev)
    prep_ns = bin_sum(is_prep)
    n_task = bin_count(is_task)
    n_sev = bin_count(is_sev)

    busy_any = np.zeros(n_it, dtype=np.int64)
    occ_sel = is_worker & (it >= 0) & (it < n_it)
    o_it, o_b, o_e = it[occ_sel], b[occ_sel], e[occ_sel]
    ordr = np.argsort(o_it, kind="stable")
    o_it, o_b, o_e = o_it[ordr], o_b[ordr], o_e[ordr]
    starts = np.searchsorted(o_it, np.arange(n_it), side="left")
    stops = np.searchsorted(o_it, np.arange(n_it), side="right")
    for i in range(n_it):
        s, t = starts[i], stops[i]
        if t > s:
            bb = np.clip(o_b[s:t], bounds[i], bounds[i + 1])
            ee = np.clip(o_e[s:t], bounds[i], bounds[i + 1])
            busy_any[i] = union_length(bb, ee)

    dead = dur_it - busy_any
    perfect_pack = (task_ns + sev_ns) / float(n_workers)
    worker_idle = dur_it - perfect_pack - dead
    return dict(iter_ns=dur_it, task_ns=task_ns, sched_events_ns=sev_ns,
                prepare_batch_ns=prep_ns, busy_any_ns=busy_any, dead_ns=dead,
                perfect_pack_ns=perfect_pack, worker_idle_ns=worker_idle,
                n_task=n_task, n_sched_events=n_sev,
                occupancy=(task_ns + sev_ns) / np.maximum(
                    dur_it * float(n_workers), 1))


def per_task_table(pairs: dict, bounds: np.ndarray, lo: int, hi: int,
                   n_workers: int, names: dict) -> list:
    """Per task-type stats over iterations [lo,hi) -- per-iteration averages."""
    b = pairs["begin"]
    it = np.searchsorted(bounds, b, side="right") - 1
    sel = (it >= lo) & (it < hi)
    tt = pairs["task_type"][sel]
    d = pairs["dur"][sel]
    blk = pairs["block"][sel]
    n_it = float(hi - lo)
    out = []
    for t in np.unique(tt):
        m = tt == t
        dd = d[m].astype(np.float64)
        # A task that finds no rows to process still costs a queue pop, a
        # dependency check and two profiler stores; empirically those land
        # under 1 us while any real tile is >= 4 us.  Splitting on 1 us
        # separates "dispatched but idle" from "did work" without needing the
        # task's own arguments, which the profiler does not record.
        short = dd < 1000.0
        out.append(dict(
            task_type=int(t),
            name=names.get(str(int(t)), names.get(int(t), f"UNKNOWN_{int(t)}")),
            bucket=LAYER_BUCKET.get(int(t), "other"),
            n_per_iter=float(m.sum()) / n_it,
            total_us_per_iter=float(dd.sum()) / 1e3 / n_it,
            per_worker_us_per_iter=float(dd.sum()) / 1e3 / n_it / n_workers,
            mean_us=float(dd.mean()) / 1e3,
            p50_us=float(np.percentile(dd, 50)) / 1e3,
            p95_us=float(np.percentile(dd, 95)) / 1e3,
            max_us=float(dd.max()) / 1e3,
            n_short_per_iter=float(short.sum()) / n_it,
            short_us_per_iter=float(dd[short].sum()) / 1e3 / n_it,
            n_long_per_iter=float((~short).sum()) / n_it,
            long_mean_us=(float(dd[~short].mean()) / 1e3
                          if (~short).any() else 0.0),
            n_blocks=int(len(np.unique(blk))),
        ))
    out.sort(key=lambda r: -r["total_us_per_iter"])
    return out


# ---------------------------------------------------------------------------
# windowed perfetto export
# ---------------------------------------------------------------------------
def export_window_perfetto(ev: dict, t0: int, t1: int, path: str,
                           names: dict) -> int:
    """Perfetto trace for events inside [t0,t1) only.  Full-run traces are
    10-15 M events; a 3-iteration window is ~200-400 k and actually opens."""
    from tg4perfetto import TraceGenerator

    m = (ev["ts"] >= t0) & (ev["ts"] < t1)
    order = np.argsort(ev["slot"][m], kind="stable")
    blk = ev["block"][m][order]
    grp = ev["group"][m][order]
    tt = ev["task_type"][m][order]
    et = ev["etype"][m][order]
    ts = ev["ts"][m][order]

    tgen = TraceGenerator(path)
    tid_map, track_map, open_tracks = {}, {}, set()
    for i in range(len(blk)):
        key = (int(blk[i]), int(grp[i]))
        if key not in tid_map:
            pid = tgen.create_group(f"block_{key[0]}")
            tid_map[key] = pid.create_group(f"group_{key[1]}")
        tkey = (key[0], key[1], int(tt[i]))
        if tkey not in track_map:
            track_map[tkey] = tid_map[key].create_track()
        track = track_map[tkey]
        nm = names.get(str(int(tt[i])), f"UNKNOWN_{int(tt[i])}")
        if et[i] == EVENT_BEGIN:
            track.open(int(ts[i]), nm)
            open_tracks.add(tkey)
        elif et[i] == EVENT_END:
            if tkey in open_tracks:
                track.close(int(ts[i]))
                open_tracks.discard(tkey)
        else:
            track.instant(int(ts[i]), nm)
    for tkey in list(open_tracks):
        track_map[tkey].close(int(t1))
    tgen.flush()
    return int(m.sum())


def load_names(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
