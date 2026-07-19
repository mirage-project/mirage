#!/usr/bin/env python3
"""Export an MPK Runtime-V2 profiler buffer to a Chrome Trace Event JSON.

The v2 runtime profiles into a v2-specific buffer that only ``mirage.mpk.prof``
(the text tool: ``cmd_summary``/``cmd_check``) understands.  The v1 exporter
``profiler_persistent.export_to_perfetto_trace`` parses that buffer as the
legacy FlashInfer format and produces garbage / an empty trace.  This script
is the missing v2 exporter: it reuses ``mirage.mpk.prof.Dump`` VERBATIM to
parse the buffer, then emits a Chrome Trace Event JSON that ui.perfetto.dev
loads directly.

Usage::

    python scripts/v2_perfetto_export.py <buffer.npy> <out.json>
        [--last-steps K] [--sm N] [--full]

``<buffer.npy>`` is the raw v2 profiler buffer saved with ``np.save`` (see the
``use_v2_runtime``-gated block in ``persistent_kernel.py::__call__``).  The
input may also be any ``.npy`` holding the raw buffer array.

A full-window trace is millions of slices / hundreds of MB and OOMs
ui.perfetto.dev.  Cut it down:
  * ``--last-steps K``  keep only the last K decode steps (segmented by the
    controller ITER_SYNC barriers). One step ≈ one decoded token.
  * ``--sm N``          keep only SM N's tracks (≈ 1/nSM the size).
  * ``--full``          force the complete trace (overrides the above).
Default (no flag): the WHOLE buffer — use a flag for a loadable trace.

Output is ``{"traceEvents": [...], "displayTimeUnit": "ns"}``.  Each parsed
window ``(start, end, ev)`` on track ``(sm, group)`` becomes one complete
("ph":"X") slice with ``pid = sm`` (process = "SM <n>") and ``tid = group``
(thread = the warp-role name).  Timestamps are normalized so the earliest
event sits at t=0.
"""
from __future__ import annotations

import json
import sys

import numpy as np

# Reuse the v2 buffer parser + name tables verbatim — do NOT re-derive the
# buffer format here (it must track runtime_v2.cuh, and Dump already does).
try:
    from mirage.mpk.prof import (
        BUCKET_NAMES,
        EVENT_NAMES,
        GROUP_NAMES,
        V2_PROF_TAIL,
        Dump,
        dur_us,
    )
except Exception:  # pragma: no cover - allow running from a source checkout
    import os

    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, os.pardir, "python"))
    from mirage.mpk.prof import (  # noqa: E402
        BUCKET_NAMES,
        EVENT_NAMES,
        GROUP_NAMES,
        V2_PROF_TAIL,
        Dump,
        dur_us,
    )

# --------------------------------------------------------------------------
# task-type-id -> name.  On the ROLE tracks (consumer/loader/launcher/storer)
# a slice's "ev" field is the TASK-TYPE id (the megakernel task the SM ran),
# NOT a phase/wait event — so EVENT_NAMES (which only has the phase/wait
# events) misses them and they render as cryptic ev<id> (e.g. ev353).  This
# table is the authoritative TaskType enum from
# include/mirage/persistent_kernel/runtime_header.h (cross-checked against the
# task_type_to_name[...] assignments in src/kernel/runtime.cc).  Names are
# stripped of the leading "TASK_" for compactness.  Regenerate with:
#   python - <<'PY'  (see the parser used when this table was authored)
#   import re; txt=open("include/.../runtime_header.h").read()
#   body=re.search(r'enum\s+TaskType\s*\{(.*?)\};',txt,re.S).group(1)
#   for n,v in re.findall(r'\b(TASK_[A-Z0-9_]+)\s*=\s*(\d+)',body): ...
#   PY
TASK_TYPE_NAMES = {
    0: "TERMINATE", 10: "BEGIN_TASK_GRAPH",
    101: "EMBEDDING", 102: "RMS_NORM_LINEAR", 103: "ATTENTION_1",
    104: "ATTENTION_2", 105: "SILU_MUL_LINEAR_WITH_RESIDUAL",
    106: "ALLREDUCE", 107: "REDUCE", 108: "LINEAR_WITH_RESIDUAL",
    109: "ARGMAX", 110: "ARGMAX_PARTIAL", 111: "ARGMAX_REDUCE",
    112: "FIND_NGRAM_PARTIAL", 113: "FIND_NGRAM_GLOBAL",
    114: "TARGET_VERIFY_GREEDY", 115: "SINGLE_BATCH_EXTEND_ATTENTION",
    116: "PAGED_ATTENTION_1", 117: "PAGED_ATTENTION_2", 118: "SILU_MUL",
    119: "RMS_NORM", 120: "LINEAR", 121: "IDENTITY",
    150: "HOPPER_TASK_BEGIN", 151: "LINEAR_WITH_RESIDUAL_HOPPER",
    152: "LINEAR_HOPPER", 153: "PAGED_ATTENTION_HOPPER",
    154: "RMS_NORM_HOPPER", 155: "LINEAR_SWAPAB_HOPPER",
    156: "LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER", 157: "LINEAR_CUTLASS_HOPPER",
    158: "LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER", 159: "SILU_MUL_HOPPER",
    160: "EMBEDDING_HOPPER", 161: "MOE_W13_LINEAR_SM90",
    162: "MOE_W2_LINEAR_SM90", 163: "SPLITK_LINEAR_SWAPAB_HOPPER",
    164: "PAGED_ATTENTION_SPLIT_KV_HOPPER", 198: "HOPPER_TASK_END",
    200: "SCHD_TASKS", 201: "SCHD_EVENTS", 202: "GET_EVENT",
    203: "GET_NEXT_TASK", 204: "SCHD_PREPARE_BATCH",
    230: "SM100_TASK_BEGIN", 231: "SM100_TMA_START_TASK", 232: "COPY",
    233: "CONCAT", 235: "EAGLE3_D2T_REMAP", 236: "EAGLE3_COMMIT",
    242: "LINEAR_SM100_V2", 243: "LINEAR_WITH_RESIDUAL_SM100_V2",
    244: "LINEAR_SM100_V3", 245: "LINEAR_WITH_RESIDUAL_SM100_V3",
    246: "SPLITK_LINEAR_FP8_SWAPAB_SM100", 247: "LINEAR_FP8_SWAPAB_SM100",
    248: "MOE_W13_FP8_SM100", 249: "MOE_W2_FP8_SM100",
    250: "LINEAR_FP8_SWAPAB_WITH_RESIDUAL_SM100", 251: "SPLITK_LINEAR_SM100",
    252: "LINEAR_WITH_RESIDUAL_SM100", 253: "LINEAR_SM100",
    254: "MOE_W13_LINEAR_SM100", 255: "MOE_W2_LINEAR_SM100",
    256: "SM100_TMA_END_TASK", 257: "ATTN_SM100",
    258: "ARGMAX_REDUCE_SM100", 259: "ARGMAX_PARTIAL_SM100",
    260: "MOE_TOPK_SOFTMAX_SM100", 261: "MOE_MUL_SUM_ADD_SM100",
    262: "TENSOR_INIT", 263: "PAGED_ATTENTION_SPLIT_KV_SM100",
    264: "PAGED_ATTENTION_SPLIT_KV_MERGE_SM100", 265: "SAMPLING_SM100",
    266: "MLA_DECODE_SM100", 267: "MLA_REDUCE_SM100",
    268: "MLA_PREFILL_SM100", 269: "MLA_MTP_DECODE_SM100",
    270: "MLA_MTP_REDUCE_SM100", 271: "MTP_VERIFY_STRICT",
    272: "MTP_ACCEPT_COMMIT", 273: "MTP_TOKEN_SCATTER",
    274: "MTP_PREPARE_VERIFY", 275: "QUANTIZE_FP8_SM100",
    276: "LINEAR_FP8_SM100", 277: "LINEAR_FP8_WITH_RESIDUAL_SM100",
    278: "MLA_KV_GATHER_SM100", 279: "LINEAR_FP8_BMM_SM100",
    280: "MOE_TOPK_SIGMOID_SM100", 281: "ELEMENTWISE_ADD_SM100",
    282: "SOFTMAX_GATHER_SM100", 283: "MTP_VERIFY_PROBABILISTIC",
    284: "PROB_SCATTER_SM100", 285: "MTP_FLOAT_SCATTER",
    286: "PROB_EXTRACT_SM100", 287: "MLA_MTP_DECODE_TP2_SM100",
    288: "MLA_MTP_DECODE_TP_REDUCE_SM100", 289: "MLA_MTP_DECODE_TP4_SM100",
    291: "MLA_MTP_DECODE_TP8_SM100", 293: "MLA_KV_GATHER_SPLIT_SM100",
    294: "MTP_BUILD_EMBED_INPUT", 295: "MLA_PREFILL_TP8_SM100",
    296: "MLA_UNIFIED_SM100", 297: "MLA_KV_GATHER_UNIFIED_SM100",
    298: "MLA_PREFILL_TP8_CHUNKED_SM100",
    299: "MLA_PREFILL_TP8_CHUNKED_SPLITK_SM100",
    300: "MULTIGPU_TASK_BEGIN", 301: "NVSHMEM_ALLGATHER_STRIDED_PUT",
    302: "NVSHMEM_TILE_ALLREDUCE", 303: "NVSHMEM_GLOBAL_ARGMAX",
    304: "DEEPSEEK_MLA_ROPE_SM100", 305: "MLA_PREFILL_TP8_CHUNKED_REDUCE_SM100",
    306: "FP8_GEMM_DENSE_SM100", 307: "DSV3_DENSE_MLP_FUSED_SM100",
    308: "FP8_GEMM_DENSE_FINEN_SM100", 309: "FUSED_RMSNORM_QUANTIZE_FP8_SM100",
    310: "MOE_TOPK_COMPACT_SM100", 311: "FP8_GROUP_GEMM_SMALLM_SM100",
    312: "FP8_GROUP_GEMM_LARGEM_SM100", 313: "MOE_PERMUTE_SM100",
    314: "MOE_UNPERMUTE_SM100", 315: "TRANSPOSE_SCALE_SM100",
    316: "ASSEMBLE_Q_DECODE_SM100", 317: "FP8_GROUP_GEMM_LARGEM_COMPACT_SM100",
    318: "DSV3_ROUTER_GATE_GEMV_SM100", 319: "ATTN_BLOCK_MEGAKERNEL_SM100",
    320: "MOE_TOPK_MARKER_INIT_SM100",
    321: "FP8_GROUP_GEMM_LARGEM_COMPACT_FUSED_SM100",
    322: "LINEAR_FP8_BMM_DENSE_SM100", 323: "MLA_KV_APPEND_SM100",
    325: "FFN_FULL_MEGAKERNEL_SM100", 326: "RMS_NORM_HOPPER_V2",
    327: "SILU_MUL_V2", 328: "EMBEDDING_V2", 329: "ATTN_SM100_V2",
    330: "ARGMAX_PARTIAL_SM100_V2", 331: "ARGMAX_REDUCE_SM100_V2",
    332: "DSV3_FFN_ROUTER_QUANT_V2", 333: "DSV3_FFN_TOPK_SIGMOID_V2",
    334: "DSV3_FFN_W13_GEMV_V2", 335: "DSV3_FFN_SILU_QUANT_V2",
    336: "DSV3_FFN_W2_GEMV_V2", 337: "DSV3_FFN_ROUTER_QUANT_RMS_V2",
    338: "DSV3_FFN_W13_TOPK_V2", 339: "DSV3_FFN_W2_SILU_V2",
    340: "DSV3_ATTN_P0_QKVA_V2", 341: "DSV3_ATTN_QB_ROPE_KV_V2",
    342: "DSV3_ATTN_MLA_PARTIAL_V2", 343: "DSV3_ATTN_MLA_MERGE_V2",
    344: "DSV3_ATTN_WUV_V2", 345: "DSV3_ATTN_OPROJ_V2",
    346: "DSV3_ATTN_MLA_FUSED_V2", 347: "DSV3_FFN_W13_RQR_TOPK_V2",
    348: "DSV3_FFN_MEGA_V2", 349: "DSV3_FFN_MEGA_FG_V2",
    350: "NVSHMEM_TILE_ALLREDUCE_V2",
    351: "NVSHMEM_TILE_ALLREDUCE_WITH_RESIDUAL_V2", 352: "TENSOR_INIT_V2",
    353: "ATTN_BLOCK_MEGAKERNEL_V2", 354: "DSV3_DENSE_MLP_FUSED_V2",
    355: "DSV3_LMHEAD_GEMV_V2", 356: "DSV3_FFN_W13_PIPE_V2",
    357: "DSV3_FFN_W2_PIPE_V2", 358: "SM100_TASK_END",
}

# ev_id -> perf bucket (mirrors prof.cmd_summary's b_of, extended to the v2
# task-type ids so task slices are attributed instead of all falling in
# "other").  {0:linear, 1:attn, 2:rmsnorm, 3:silu, 4:argmax, 5:embed, 6:other}.
_EV_TO_BUCKET = {
    # phase/wait event ids (from EVENT_NAMES) — leave to "other".
    # v2 linear GEMM task types
    244: 0, 245: 0, 242: 0, 243: 0, 253: 0, 252: 0, 251: 0,
    334: 0, 336: 0, 348: 0, 349: 0, 354: 0, 355: 0,  # FFN/dense/lmhead GEMV
    356: 0, 357: 0,  # FFN W13/W2 per-tile pipes
    # attention task types
    329: 1, 319: 1, 353: 1,
    340: 1, 341: 1, 342: 1, 343: 1, 344: 1, 345: 1, 346: 1,  # DSv3 attn stages
    # rmsnorm
    326: 2, 337: 2,
    # silu
    327: 3, 335: 3, 339: 3,
    # argmax
    330: 4, 331: 4,
    # embed
    328: 5,
    # NOTE: allreduce (350/351), tensor_init (352), router/topk (332/333/338)
    # intentionally map to "other" (bucket 6) — they're comm/init/routing, not
    # one of the 6 compute buckets. The per-task-type breakdown names them.
}


def _ev_name(ev: int) -> str:
    """Human-readable name for a slice's event/task id.

    Resolution order (per the coordinator's spec):
      1. EVENT_NAMES  — phase/wait events (W_TMA_WAIT, PAGE_WAIT, ...)
      2. TASK_TYPE_NAMES — the megakernel task-type ids on role tracks
      3. bucket name  — coarse fallback
      4. ev<id>       — last resort
    """
    name = EVENT_NAMES.get(ev)
    if name is not None:
        return name
    name = TASK_TYPE_NAMES.get(ev)
    if name is not None:
        return name
    b = _EV_TO_BUCKET.get(ev)
    if b is not None:
        return BUCKET_NAMES.get(b, "other")
    return f"ev{ev}"


def _group_name(group: int) -> str:
    if 0 <= group < len(GROUP_NAMES):
        return GROUP_NAMES[group]
    return f"group_{group}"


# Compiled full buffer size from runtime_v2.cuh: V2_PROF_BUF_ENTRIES =
# 120000*128.  The device writes the tail accumulators (spin/suffix/dropped/
# cursors/trigger-ring) at ABSOLUTE indices measured back from this size
# (base = V2_PROF_BUF_ENTRIES - offset), so those regions only land inside the
# allocation when the buffer is EXACTLY this size.  prof.py does not export
# this constant (only V2_PROF_TAIL), so we mirror it here — keep in sync with
# runtime_v2.cuh if it ever changes.
V2_PROF_BUF_ENTRIES = 120000 * 128  # = 15,360,000


def is_undersized(d: Dump) -> bool:
    """True if the buffer is smaller than the compiled v2 layout.

    Dump parses the event body as ``buf[1 : len(buf) - V2_PROF_TAIL]`` where
    ``V2_PROF_TAIL`` is hard-coded to match ``V2_PROF_BUF_ENTRIES``
    (=120000*128).  If the buffer was allocated SMALLER than that (e.g. the
    DSv3 demo's default 6000*128 = 768000), two things break:
      * ``len - V2_PROF_TAIL`` goes negative/tiny, so numpy's negative-stop
        slice silently DROPS a large fraction of the real events;
      * the device's tail accumulators (written at absolute indices near
        V2_PROF_BUF_ENTRIES) are OUT OF BOUNDS of the allocation — they were
        never written, so prof.py's spin/suffix/dropped reads (at
        ``len(buf) - offset``) land back inside the event body and return
        garbage.
    Any buffer smaller than the full compiled size hits the OOB-accumulator
    problem, so we treat < V2_PROF_BUF_ENTRIES as under-sized and re-parse the
    full event region (recovering all events; the accumulators are simply
    absent and must come from a correctly-sized re-run).
    """
    return len(d.buf) < V2_PROF_BUF_ENTRIES


def reparse_full(d: Dump) -> Dump:
    """Rebuild ``d.windows`` over the FULL event region ``buf[1:]``.

    Only used for an under-sized buffer (see ``is_undersized``): there is no
    valid accumulator tail to protect, so every entry after the header is a
    real begin/end event.  Uses the SAME begin/end pairing logic as
    ``prof.Dump`` (do not re-derive the tag format) — just over the whole
    array so no events are dropped.  Mutates and returns ``d``.
    """
    import numpy as np
    from collections import defaultdict

    buf = d.buf
    d.windows = defaultdict(list)
    d.stray = 0
    ents = buf[1:]  # no tail trim — the whole post-header region is events
    nz = np.nonzero(ents)[0]
    d.n_entries = len(nz)
    ntracks = d.ntracks
    st = {}
    for i in nz:
        v = int(ents[i])
        tag, ts = v & 0xFFFFFFFF, v >> 32
        ev = (tag >> 2) & 0x1FF
        et = tag & 3
        tr = int(i) % ntracks
        if et == 0:
            if tr in st:
                d.stray += 1
            st[tr] = (ts, ev)
        elif et == 1:
            if tr not in st:
                d.stray += 1
                continue
            s, ev0 = st.pop(tr)
            d.windows[(tr // d.ngroups, tr % d.ngroups)].append((s, ts, ev0))
    return d


# --------------------------------------------------------------------------
# Step segmentation + filtering, so a huge full-window trace (millions of
# slices → hundreds of MB) can be cut to something ui.perfetto.dev loads.
V2_PROF_CONTROLLER_GROUP = 4
V2_PROF_ITER_SYNC = 205  # controller end-of-iter barrier — marks step edges


def step_boundaries(d: Dump):
    """Global decode-step boundaries as raw ITER_SYNC end-timestamps.

    The controller (group 4) emits one ITER_SYNC per decode step, and it's a
    grid-wide barrier (verified ~0.2us spread across all 136 SMs), so any SM's
    ITER_SYNC series is a valid global segmentation.  Returns the ascending
    list of raw ITER_SYNC end-timestamps, or [] if none.

    NOTE: the whole captured window (last V2_PROF_WINDOW_ITERS decode steps) is
    only ~250 ms = ~2.5e8 ns, well under 2^32, and does NOT wrap within itself
    (verified: monotonic, max inter-event gap ~0.1 ms). So raw timestamps are
    directly comparable here — no 32-bit unwrap needed for step segmentation.
    (dur_us still masks each individual slice against a per-slice 32-bit wrap.)
    """
    best = []
    for sm in range(d.nblocks):
        ends = sorted(int(e) for s, e, ev in d.windows.get(
            (sm, V2_PROF_CONTROLLER_GROUP), []) if ev == V2_PROF_ITER_SYNC)
        if len(ends) > len(best):
            best = ends
    return best


def filter_windows(d: Dump, last_steps=None, sm=None):
    """Return a NEW ``(sm, group) -> [(start,end,ev),...]`` dict, cut to the
    last ``last_steps`` decode steps and/or a single ``sm``.

    ``last_steps`` uses the ITER_SYNC step boundaries (see ``step_boundaries``):
    a slice is kept if its start falls at/after the boundary ``last_steps``
    steps before the end.  ``sm`` keeps only that SM's tracks.  Returns
    ``(filtered_windows, info_dict)``.
    """
    from collections import defaultdict

    info = {"last_steps": last_steps, "sm": sm}
    lo_ts = None
    if last_steps is not None:
        bounds = step_boundaries(d)
        info["n_steps_total"] = len(bounds)
        if not bounds:
            info["step_fallback"] = "no ITER_SYNC events — kept all steps"
        elif len(bounds) > last_steps:
            # bounds[i] = END of step i (ascending, non-wrapping window).
            # The boundary that OPENS the last `last_steps` steps is
            # bounds[-1 - last_steps]; keep every slice starting at/after it.
            lo_ts = bounds[len(bounds) - 1 - last_steps]
            info["kept_steps"] = last_steps
        else:
            info["step_fallback"] = (
                f"only {len(bounds)} steps <= last_steps={last_steps}; "
                f"kept all")

    out = defaultdict(list)
    for (bsm, group), wins in d.windows.items():
        if sm is not None and bsm != sm:
            continue
        if lo_ts is None:
            out[(bsm, group)] = list(wins)
            continue
        kept = [(s, e, ev) for (s, e, ev) in wins if int(s) >= lo_ts]
        if kept:
            out[(bsm, group)] = kept
    return out, info


def build_trace(d: Dump, windows=None) -> dict:
    """Turn a parsed ``Dump`` (or a filtered ``windows`` dict) into a Chrome
    Trace Event dict.

    ``windows`` (optional) overrides ``d.windows`` — pass the output of
    ``filter_windows`` to emit a subset (last-K-steps / single-SM) so the JSON
    is small enough for ui.perfetto.dev.  Returns ``{"traceEvents": [...],
    "displayTimeUnit": "ns"}``.  Raises ``ValueError`` if there are no usable
    (non-degenerate) windows.
    """
    if windows is None:
        windows = d.windows
    # Global min start, for normalizing t=0. Timestamps are 32-bit-wrapping
    # %globaltimer ns; within a single windowed capture (<=25 decode steps, a
    # few ms) they do not wrap across the whole trace, so a plain min is a
    # sound origin. dur_us() still masks per-slice against a 32-bit end-start
    # wrap, so an individual slice that straddles the 2^32 boundary is fine.
    starts = [s for w in windows.values() for (s, _e, _ev) in w]
    if not starts:
        raise ValueError("no windows parsed from buffer")
    t0 = min(starts)

    events = []
    seen_pids = set()
    seen_tids = set()  # (pid, tid)
    n_slices = 0
    n_skipped = 0
    tracks = set()
    per_bucket_us = {}
    per_type_us = {}   # resolved-name -> [total_us, count]
    min_ts_us = None
    max_ts_us = None

    # Deterministic ordering: by SM, then group.
    for (sm, group) in sorted(windows.keys()):
        wins = windows[(sm, group)]
        tracks.add((sm, group))
        pid = int(sm)
        tid = int(group)

        # Process/thread metadata (one per pid / per (pid,tid)).
        if pid not in seen_pids:
            seen_pids.add(pid)
            events.append({
                "ph": "M", "name": "process_name", "pid": pid,
                "args": {"name": f"SM {pid}"},
            })
        if (pid, tid) not in seen_tids:
            seen_tids.add((pid, tid))
            events.append({
                "ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                "args": {"name": _group_name(group)},
            })

        for (start, end, ev) in wins:
            dur = dur_us(start, end)  # masks the 32-bit end-start wrap
            if dur <= 0.0:
                n_skipped += 1  # skip zero-duration / degenerate slices
                continue
            ts_us = ((start - t0) & 0xFFFFFFFF) / 1e3
            nm = _ev_name(ev)
            events.append({
                "name": nm,
                "cat": "v2",
                "ph": "X",
                "ts": ts_us,
                "dur": dur,
                "pid": pid,
                "tid": tid,
            })
            n_slices += 1
            min_ts_us = ts_us if min_ts_us is None else min(min_ts_us, ts_us)
            end_us = ts_us + dur
            max_ts_us = end_us if max_ts_us is None else max(max_ts_us, end_us)
            b = _EV_TO_BUCKET.get(ev, 6)
            per_bucket_us[b] = per_bucket_us.get(b, 0.0) + dur
            slot = per_type_us.setdefault(nm, [0.0, 0])
            slot[0] += dur
            slot[1] += 1

    if n_slices == 0:
        raise ValueError("no non-degenerate slices to export")

    trace = {"traceEvents": events, "displayTimeUnit": "ns"}
    # attach a small summary the CLI prints (not part of the perfetto schema
    # proper, but Chrome/perfetto ignore unknown top-level keys)
    trace["_summary"] = {
        "n_slices": n_slices,
        "n_tracks": len(tracks),
        "n_skipped": n_skipped,
        "span_us": (0.0 if min_ts_us is None else (max_ts_us - min_ts_us)),
        "per_bucket_us": {BUCKET_NAMES.get(b, f"b{b}"): round(v, 3)
                          for b, v in sorted(per_bucket_us.items())},
        # per-task-type busy-us, descending by total (name -> [total_us, n])
        "per_type_us": {nm: [round(v[0], 3), v[1]]
                        for nm, v in sorted(per_type_us.items(),
                                            key=lambda kv: -kv[1][0])},
    }
    return trace


def _buffer_is_empty(d: Dump) -> bool:
    """True if the parsed buffer has no events at all (empty / all-zero)."""
    return d.n_entries == 0 or not d.windows


def main(argv) -> int:
    import argparse
    import os

    p = argparse.ArgumentParser(
        prog="v2_perfetto_export.py",
        description="Export an MPK Runtime-V2 profiler buffer to a Chrome "
                    "Trace Event JSON (loads directly in ui.perfetto.dev). "
                    "A full-window trace is millions of slices / hundreds of "
                    "MB — use --last-steps / --sm to cut it to a loadable "
                    "size (target < ~15 MB).")
    p.add_argument("buffer", help="raw v2 profiler buffer .npy")
    p.add_argument("out", help="output Chrome-Trace .json")
    p.add_argument("--last-steps", type=int, default=None, metavar="K",
                   help="keep only the last K decode steps (segmented by the "
                        "controller ITER_SYNC barriers). Default: keep all.")
    p.add_argument("--sm", type=int, default=None, metavar="N",
                   help="keep only SM N's tracks (~1/nSM the size).")
    p.add_argument("--full", action="store_true",
                   help="force the complete trace (all steps, all SMs); "
                        "overrides --last-steps/--sm. WARNING: can be >100 MB "
                        "and OOM the browser.")
    args = p.parse_args(argv[1:])

    in_path, out_path = args.buffer, args.out
    last_steps = None if args.full else args.last_steps
    only_sm = None if args.full else args.sm

    try:
        d = Dump(in_path)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to load/parse buffer {in_path!r}: {e}",
              file=sys.stderr)
        return 1

    if d.ngroups < 5:
        print(f"ERROR: buffer header reports ngroups={d.ngroups} (<5) — this "
              "looks like a v1 profiler buffer, not a v2 one. Use "
              "profiler_persistent.export_to_perfetto_trace for v1.",
              file=sys.stderr)
        return 1

    if is_undersized(d):
        n_before = d.n_entries
        reparse_full(d)
        print(f"WARNING: buffer len={len(d.buf)} < compiled v2 size "
              f"V2_PROF_BUF_ENTRIES={V2_PROF_BUF_ENTRIES}; Dump's tail-trim "
              f"(V2_PROF_TAIL={V2_PROF_TAIL}) would silently drop events. "
              f"Re-parsed the FULL region: {n_before} -> {d.n_entries} events. "
              f"NOTE: the tail spin/wait ACCUMULATORS were never written (OOB "
              f"of this alloc) — the busy-vs-spin/wait per-role view is ABSENT; "
              f"re-run with MPK_PROFILER_BUFFER_ENTRIES={V2_PROF_BUF_ENTRIES} "
              f"for it.", file=sys.stderr)

    if _buffer_is_empty(d):
        print(f"ERROR: empty buffer — no v2 events in {in_path!r} "
              f"(nblocks={d.nblocks} ngroups={d.ngroups} entries="
              f"{d.n_entries}). Nothing to export.", file=sys.stderr)
        return 1

    # ---- optional filtering (step window / single SM) --------------------
    windows = None
    finfo = {}
    if last_steps is not None or only_sm is not None:
        windows, finfo = filter_windows(d, last_steps=last_steps, sm=only_sm)
        if not windows:
            print(f"ERROR: filter (last_steps={last_steps}, sm={only_sm}) "
                  f"left no slices.", file=sys.stderr)
            return 1

    try:
        trace = build_trace(d, windows=windows)
    except ValueError as e:
        print(f"ERROR: {e} (nblocks={d.nblocks} ngroups={d.ngroups} "
              f"entries={d.n_entries}).", file=sys.stderr)
        return 1

    summary = trace.pop("_summary")
    with open(out_path, "w") as f:
        json.dump(trace, f)
    out_mb = os.path.getsize(out_path) / (1024 * 1024)

    # ---- stdout summary --------------------------------------------------
    print(f"wrote {out_path}  ({out_mb:.1f} MB)")
    print(f"  header:   nblocks={d.nblocks} ngroups={d.ngroups} "
          f"entries={d.n_entries} stray={d.stray}")
    if last_steps is not None or only_sm is not None:
        print(f"  filter:   last_steps={last_steps} sm={only_sm}  {finfo}")
    elif args.full:
        print("  filter:   --full (all steps, all SMs)")
    print(f"  slices:   {summary['n_slices']} "
          f"(skipped {summary['n_skipped']} zero-duration)")
    print(f"  tracks:   {summary['n_tracks']} (SM x role)")
    print(f"  span:     {summary['span_us']:.3f} us")
    if out_mb > 60:
        print(f"  NOTE: {out_mb:.1f} MB may be too big for ui.perfetto.dev "
              f"(it OOMs around a few hundred MB) — try --last-steps 1 "
              f"and/or --sm N.", file=sys.stderr)
    print("  per-bucket busy-us (sum over all role tracks):")
    for name, us in summary["per_bucket_us"].items():
        print(f"    {name:8s} {us:14.3f} us")
    print("  per-task-type busy-us (desc; sum over all role tracks):")
    print(f"    {'task/event':40s} {'total_us':>14s} {'n':>9s} "
          f"{'us/slice':>10s}")
    for nm, (tot, n) in summary["per_type_us"].items():
        per = tot / n if n else 0.0
        print(f"    {nm:40s} {tot:14.3f} {n:9d} {per:10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
