import argparse
import csv
import json
from collections import namedtuple
from enum import Enum
from typing import List

import torch
from tg4perfetto import TraceGenerator

event_name_list = {
    10: "TASK_BEGIN_TASK_GRAPH",
    101: "TASK_EMBEDDING",
    102: "TASK_RMS_NORM_LINEAR",
    103: "TASK_ATTENTION_1",
    104: "TASK_ATTENTION_2",
    105: "TASK_SILU_MUL_LINEAR",
    106: "TASK_ALLREDUCE",
    107: "TASK_REDUCE",
    108: "TASK_LINEAR_WITH_RESIDUAL",
    109: "TASK_ARGMAX",
    110: "TASK_ARGMAX_PARTIAL",
    111: "TASK_ARGMAX_REDUCE",
    112: "TASK_FIND_NGRAM_PARTIAL",
    113: "TASK_FIND_NGRAM_GLOBAL",
    114: "TASK_TARGET_VERIFY_GREEDY",
    115: "TASK_SINGLE_BATCH_EXTEND_ATTENTION",
    116: "TASK_PAGED_ATTENTION_1",
    117: "TASK_PAGED_ATTENTION_2",
    118: "TASK_SILU_MUL",
    119: "TASK_RMS_NORM",
    120: "TASK_LINEAR",
    121: "TASK_IDENTITY",
    150: "TASK_HOPPER_TASK_BEGIN",
    151: "TASK_LINEAR_WITH_RESIDUAL_HOPPER",
    152: "TASK_LINEAR_HOPPER",
    153: "TASK_PAGED_ATTENTION_HOPPER",
    154: "TASK_RMS_NORM_HOPPER",
    155: "TASK_LINEAR_SWAPAB_HOPPER",
    156: "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER",
    157: "TASK_LINEAR_CUTLASS_HOPPER",
    158: "TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER",
    159: "TASK_SILU_MUL_HOPPER",
    160: "TASK_EMBEDDING_HOPPER",
    161: "TASK_MOE_W13_LINEAR_SM90",
    162: "TASK_MOE_W2_LINEAR_SM90",
    163: "TASK_SPLITK_LINEAR_SWAPAB_HOPPER",
    198: "TASK_HOPPER_TASK_END",
    200: "TASK_SCHD_TASKS",
    201: "TASK_SCHD_EVENTS",
    202: "TASK_GET_EVENT",
    203: "TASK_GET_NEXT_TASK",
    230: "TASK_SM100_TASK_BEGIN",
    251: "TASK_SPLITK_LINEAR_SM100",
    252: "TASK_LINEAR_WITH_RESIDUAL_SM100",
    253: "TASK_LINEAR_SM100",
    254: "TASK_MOE_W13_LINEAR_SM100",
    255: "TASK_MOE_W2_LINEAR_SM100",
    257: "TASK_ATTN_SM100",
    258: "TASK_ARGMAX_REDUCE_SM100",
    259: "TASK_ARGMAX_PARTIAL_SM100",
    260: "TASK_MOE_TOPK_SOFTMAX_SM100",
    261: "TASK_MOE_MUL_SUM_ADD_SM100",
    262: "TASK_TENSOR_INIT",
    298: "TASK_SM100_TASK_END",
    301: "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT",
    302: "TASK_NVSHMEM_TILE_ALLREDUCE",
}


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    event_no = tag >> 19
    block_group_tag = (tag >> 11) & 0xFF
    event_idx = (tag >> 2) & 0x1FF
    event_type = tag & 0x3
    return (
        event_no,
        block_group_tag // num_groups,
        block_group_tag % num_groups,
        event_idx,
        event_type,
    )


def analyze_profiler_data(profiler_buffer: torch.Tensor) -> None:
    """Print analysis of profiling data: per-task-type timing, per-worker
    load balance, and gap (sync/idle) time between tasks."""
    from collections import defaultdict

    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    # Collect all events: (block_idx, group_idx, event_no, event_idx, event_type, timestamp)
    events = []
    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        event_no, block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )
        events.append((block_idx, group_idx, event_no, event_idx, event_type, timestamp))

    if not events:
        print("[profiler-analysis] No events found.")
        return

    # ── 1. Match BEGIN/END pairs per (block_idx, group_idx, event_idx, event_no) ──
    # key -> list of (begin_ts, end_ts)
    open_events = {}  # key -> begin_ts
    matched_pairs = []  # (block_idx, event_idx, event_no, begin_ts, end_ts)

    for block_idx, group_idx, event_no, event_idx, event_type, timestamp in events:
        key = (block_idx, group_idx, event_idx, event_no)
        if event_type == EventType.kBegin.value:
            open_events[key] = timestamp
        elif event_type == EventType.kEnd.value:
            if key in open_events:
                begin_ts = open_events.pop(key)
                duration = timestamp - begin_ts  # in ns (globaltimer)
                if duration < 0:
                    duration += (1 << 32)  # handle 32-bit wrap
                matched_pairs.append((block_idx, event_idx, event_no, begin_ts, timestamp, duration))

    if not matched_pairs:
        print("[profiler-analysis] No matched BEGIN/END pairs found.")
        return

    # ── 2. Per-task-type aggregate ──
    # event_idx -> list of durations
    type_durations = defaultdict(list)
    for block_idx, event_idx, event_no, begin_ts, end_ts, duration in matched_pairs:
        type_durations[event_idx].append(duration)

    print("\n" + "=" * 80)
    print("  PER-TASK-TYPE TIMING SUMMARY (across all workers, first iteration)")
    print("=" * 80)
    print(f"  {'Task Type':<45s} {'Count':>6s} {'Total (ms)':>11s} {'Avg (us)':>10s} {'Max (us)':>10s} {'Min (us)':>10s}")
    print("-" * 80)
    sorted_types = sorted(type_durations.keys(), key=lambda k: sum(type_durations[k]), reverse=True)
    grand_total = 0
    for eidx in sorted_types:
        durs = type_durations[eidx]
        name = event_name_list.get(eidx, f"UNKNOWN_{eidx}")
        total_ns = sum(durs)
        grand_total += total_ns
        avg_ns = total_ns / len(durs)
        max_ns = max(durs)
        min_ns = min(durs)
        print(f"  {name:<45s} {len(durs):>6d} {total_ns/1e6:>11.3f} {avg_ns/1e3:>10.1f} {max_ns/1e3:>10.1f} {min_ns/1e3:>10.1f}")
    print("-" * 80)
    print(f"  {'TOTAL':<45s} {len(matched_pairs):>6d} {grand_total/1e6:>11.3f}")
    print()

    # ── 3. Per-worker load balance ──
    # block_idx -> list of (begin_ts, end_ts, duration, event_idx)
    worker_tasks = defaultdict(list)
    for block_idx, event_idx, event_no, begin_ts, end_ts, duration in matched_pairs:
        worker_tasks[block_idx].append((begin_ts, end_ts, duration, event_idx))

    # Sort each worker's tasks by begin_ts
    for wid in worker_tasks:
        worker_tasks[wid].sort(key=lambda x: x[0])

    print("=" * 80)
    print("  PER-WORKER LOAD BALANCE")
    print("=" * 80)
    print(f"  {'Worker':<10s} {'#Tasks':>6s} {'Busy (ms)':>11s} {'Gap (ms)':>11s} {'Span (ms)':>11s} {'Busy%':>7s}")
    print("-" * 80)

    worker_stats = {}
    for wid in sorted(worker_tasks.keys()):
        tasks = worker_tasks[wid]
        busy_ns = sum(d for _, _, d, _ in tasks)
        first_begin = tasks[0][0]
        last_end = tasks[-1][1]
        span_ns = last_end - first_begin
        if span_ns < 0:
            span_ns += (1 << 32)
        gap_ns = span_ns - busy_ns
        if gap_ns < 0:
            gap_ns = 0
        busy_pct = 100.0 * busy_ns / span_ns if span_ns > 0 else 0.0
        worker_stats[wid] = (len(tasks), busy_ns, gap_ns, span_ns, busy_pct)

    # Print a subset: first, 1/4, 1/2, 3/4, last, and the slowest/fastest
    all_wids = sorted(worker_stats.keys())
    n = len(all_wids)
    sample_wids = set()
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(frac * (n - 1)), n - 1)
        sample_wids.add(all_wids[idx])
    # Also add the worker with most gap and least gap
    if n > 0:
        max_gap_wid = max(all_wids, key=lambda w: worker_stats[w][2])
        min_gap_wid = min(all_wids, key=lambda w: worker_stats[w][2])
        max_span_wid = max(all_wids, key=lambda w: worker_stats[w][3])
        sample_wids.update([max_gap_wid, min_gap_wid, max_span_wid])

    for wid in sorted(sample_wids):
        ntasks, busy_ns, gap_ns, span_ns, busy_pct = worker_stats[wid]
        label = f"  W{wid:<7d}"
        print(f"{label} {ntasks:>6d} {busy_ns/1e6:>11.3f} {gap_ns/1e6:>11.3f} {span_ns/1e6:>11.3f} {busy_pct:>6.1f}%")

    # Aggregate
    all_busy = [worker_stats[w][1] for w in all_wids]
    all_gap = [worker_stats[w][2] for w in all_wids]
    all_span = [worker_stats[w][3] for w in all_wids]
    print("-" * 80)
    print(f"  {'Avg':<10s} {'':>6s} {sum(all_busy)/n/1e6:>11.3f} {sum(all_gap)/n/1e6:>11.3f} {sum(all_span)/n/1e6:>11.3f} {100.0*sum(all_busy)/sum(all_span) if sum(all_span)>0 else 0:>6.1f}%")
    print(f"  {'Max':<10s} {'':>6s} {max(all_busy)/1e6:>11.3f} {max(all_gap)/1e6:>11.3f} {max(all_span)/1e6:>11.3f}")
    print(f"  {'Min':<10s} {'':>6s} {min(all_busy)/1e6:>11.3f} {min(all_gap)/1e6:>11.3f} {min(all_span)/1e6:>11.3f}")
    print()

    # ── 4. Per-worker gap breakdown: show largest gaps per worker 0 ──
    if 0 in worker_tasks:
        tasks = worker_tasks[0]
        gaps = []
        for j in range(1, len(tasks)):
            prev_end = tasks[j - 1][1]
            cur_begin = tasks[j][0]
            gap = cur_begin - prev_end
            if gap < 0:
                gap += (1 << 32)
            prev_task_name = event_name_list.get(tasks[j - 1][3], f"UNK_{tasks[j-1][3]}")
            cur_task_name = event_name_list.get(tasks[j][3], f"UNK_{tasks[j][3]}")
            gaps.append((gap, j, prev_task_name, cur_task_name))

        gaps.sort(reverse=True)
        print("=" * 80)
        print("  TOP 20 GAPS (sync/idle) ON WORKER 0  (between consecutive tasks)")
        print("=" * 80)
        print(f"  {'#':>3s} {'Gap (us)':>10s} {'After Task':<35s} {'Before Task':<35s}")
        print("-" * 80)
        for rank_i, (gap, idx, prev_name, cur_name) in enumerate(gaps[:20]):
            print(f"  {rank_i+1:>3d} {gap/1e3:>10.1f} {prev_name:<35s} {cur_name:<35s}")
        print()

    # ── 5. ASCII Gantt Chart (all workers) ──
    # Short task-type labels for compact display
    short_labels = {
        10: "BG", 101: "EM", 102: "NL", 103: "A1", 104: "A2",
        105: "SL", 106: "AR", 107: "RD", 108: "LR", 109: "AX",
        110: "AP", 111: "XR", 112: "FN", 113: "FG", 114: "TV",
        115: "EA", 116: "P1", 117: "P2", 118: "SM", 119: "RN",
        120: "LN", 121: "ID",
        151: "LR", 152: "LH", 153: "PA", 154: "RN", 155: "LS",
        156: "RS", 157: "LC", 158: "RC", 159: "SM", 160: "EM",
        161: "M1", 162: "M2", 163: "SK",
        200: "ST", 201: "SE", 202: "GE", 203: "GN",
        251: "SK", 252: "LR", 253: "LN", 254: "M1", 255: "M2",
        257: "AT", 258: "XR", 259: "AP", 260: "TK", 261: "MS",
        262: "TI", 263: "PK", 264: "PM", 265: "SA",
        301: "AG", 302: "TA",
    }

    if not worker_tasks:
        return

    # Find global time range
    global_min = min(t[0] for tasks in worker_tasks.values() for t in tasks)
    global_max = max(t[1] for tasks in worker_tasks.values() for t in tasks)
    total_span = global_max - global_min
    if total_span <= 0:
        total_span += (1 << 32)

    CHART_WIDTH = 160  # characters wide
    us_per_col = (total_span / 1e3) / CHART_WIDTH  # microseconds per column

    print("=" * (CHART_WIDTH + 12))
    print(f"  ASCII GANTT CHART  (each col ≈ {us_per_col:.1f} us, total span ≈ {total_span/1e3:.0f} us)")
    print(f"  Legend: LN=LINEAR  LR=LIN+RES  AT=ATTN  RN=RMSNORM  SM=SILU_MUL  AP=ARGMAX_P  EM=EMBED  SE=SCHED  ·=idle")
    print("=" * (CHART_WIDTH + 12))

    # Print sampled workers (every 8th + first + last)
    all_wids = sorted(worker_tasks.keys())
    sample_step = max(1, len(all_wids) // 16)
    sample_wids = list(range(0, len(all_wids), sample_step))
    if (len(all_wids) - 1) not in sample_wids:
        sample_wids.append(len(all_wids) - 1)

    for widx in sample_wids:
        wid = all_wids[widx]
        tasks = worker_tasks[wid]
        # Build the row: for each column, find what task (if any) is running
        row = ['·'] * CHART_WIDTH  # idle by default
        label_row = [' '] * CHART_WIDTH

        for begin_ts, end_ts, dur, eidx in tasks:
            col_start = int((begin_ts - global_min) / 1e3 / us_per_col)
            col_end = int((end_ts - global_min) / 1e3 / us_per_col)
            if col_start < 0:
                col_start = 0
            if col_end >= CHART_WIDTH:
                col_end = CHART_WIDTH - 1
            lbl = short_labels.get(eidx, "??")
            for c in range(col_start, col_end + 1):
                row[c] = '█'
            # Place label at midpoint if there's room
            mid = (col_start + col_end) // 2
            if col_end - col_start >= 2 and mid + 1 < CHART_WIDTH:
                label_row[mid] = lbl[0]
                if mid + 1 < CHART_WIDTH:
                    label_row[mid + 1] = lbl[1] if len(lbl) > 1 else ' '

        print(f"  W{wid:<4d} | {''.join(row)}")

    # Time axis
    print(f"  {'':6s} +{''.join(['-'] * CHART_WIDTH)}")
    axis = [' '] * CHART_WIDTH
    for tick_us in range(0, int(total_span / 1e3) + 1, max(1, int(total_span / 1e3) // 10)):
        col = int(tick_us / us_per_col)
        if col < CHART_WIDTH:
            label = f"{tick_us}"
            for ci, ch in enumerate(label):
                if col + ci < CHART_WIDTH:
                    axis[col + ci] = ch
    print(f"  {'us':6s} {''.join(axis)}")
    print()

    # ── 6. Per-worker task sequence (compact) ──
    print("=" * 120)
    print("  PER-WORKER TASK SEQUENCE (compact, first 8 workers)")
    print("  Format: TaskLabel(duration_us)")
    print("=" * 120)
    for wid in all_wids[:8]:
        tasks = worker_tasks[wid]
        parts = []
        for begin_ts, end_ts, dur, eidx in tasks:
            lbl = short_labels.get(eidx, "??")
            parts.append(f"{lbl}({dur/1e3:.0f})")
        seq = " ".join(parts)
        # Wrap at 110 chars
        lines = []
        while len(seq) > 110:
            cut = seq.rfind(" ", 0, 110)
            if cut <= 0:
                cut = 110
            lines.append(seq[:cut])
            seq = seq[cut:].lstrip()
        lines.append(seq)
        print(f"  W{wid:<4d}: {lines[0]}")
        for extra in lines[1:]:
            print(f"         {extra}")
    print()


def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    file_name: str,
) -> None:

    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)
    print(f"[profiler] header: num_blocks={num_blocks}, num_groups={num_groups}, raw={int(profiler_buffer_host[0]):#018x}")

    # Run analysis first
    analyze_profiler_data(profiler_buffer)

    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            tid = pid.create_group(f"group_{group_idx}")
            tid_map[(block_idx, group_idx)] = tid

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue

        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        event_no, block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )

        event = event_name_list.get(event_idx, f"UNKNOWN_{event_idx}") + f"_{event_no}"
        tid = tid_map[(block_idx, group_idx)]

        if (block_idx, group_idx, event_idx) in track_map:
            track = track_map[(block_idx, group_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(block_idx, group_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()
