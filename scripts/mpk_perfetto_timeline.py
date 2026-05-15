#!/usr/bin/env python3
"""Summarize Mirage MPK Perfetto traces by time order.

The MPK profiler records one slice per worker block. This helper reads the
Perfetto protobuf produced by python/mirage/mpk/profiler_persistent.py and
prints either raw slices or collapsed per-task waves. A wave is a contiguous
time range where slices of the same task overlap or touch.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tg4perfetto import perfetto_trace_pb2 as perfetto_pb2
except ImportError as exc:
    raise SystemExit(
        "tg4perfetto is required to parse MPK Perfetto traces. "
        "Run this with the Mirage Python environment."
    ) from exc


TYPE_SLICE_BEGIN = 1
TYPE_SLICE_END = 2
TYPE_INSTANT = 3


@dataclass(frozen=True)
class Slice:
    rank: int
    trace: str
    track_uuid: int
    block: Optional[int]
    group: Optional[int]
    task: str
    event_no: Optional[int]
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class Wave:
    rank: int
    task: str
    event_no: Optional[int]
    start_ns: int
    end_ns: int
    slices: int
    blocks: int
    total_task_ns: int
    max_parallel: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def _infer_rank(path: str, fallback: int) -> int:
    match = re.search(r"(?:^|[_/-])rank(\d+)(?:\.|_|$)", path)
    if match:
        return int(match.group(1))
    return fallback


def _split_task_name(name: str) -> Tuple[str, Optional[int]]:
    match = re.match(r"^(.*)_(\d+)$", name)
    if not match:
        return name, None
    return match.group(1), int(match.group(2))


def _ancestor_names(
    uuid: int,
    track_names: Dict[int, str],
    parents: Dict[int, int],
) -> List[str]:
    names = []
    seen = set()
    cur = uuid
    while cur and cur not in seen:
        seen.add(cur)
        name = track_names.get(cur)
        if name:
            names.append(name)
        cur = parents.get(cur, 0)
    return names


def _track_block_group(
    uuid: int,
    track_names: Dict[int, str],
    parents: Dict[int, int],
) -> Tuple[Optional[int], Optional[int]]:
    block = None
    group = None
    for name in _ancestor_names(uuid, track_names, parents):
        if block is None:
            match = re.match(r"block_(\d+)$", name)
            if match:
                block = int(match.group(1))
        if group is None:
            match = re.match(r"group_(\d+)$", name)
            if match:
                group = int(match.group(1))
    return block, group


def parse_trace(path: str, rank: int) -> List[Slice]:
    trace = perfetto_pb2.Trace()
    with open(path, "rb") as f:
        trace.ParseFromString(f.read())

    event_names: Dict[int, str] = {}
    track_names: Dict[int, str] = {}
    parents: Dict[int, int] = {}
    stacks: Dict[int, List[Tuple[str, Optional[int], int]]] = defaultdict(list)
    slices: List[Slice] = []

    for packet in trace.packet:
        for event_name in packet.interned_data.event_names:
            event_names[int(event_name.iid)] = event_name.name

        if packet.HasField("track_descriptor"):
            desc = packet.track_descriptor
            track_names[int(desc.uuid)] = desc.name
            if desc.parent_uuid:
                parents[int(desc.uuid)] = int(desc.parent_uuid)

        if not packet.HasField("track_event"):
            continue
        event = packet.track_event
        track_uuid = int(event.track_uuid)
        timestamp = int(packet.timestamp)

        if event.type == TYPE_SLICE_BEGIN:
            if event.name:
                raw_name = event.name
            else:
                raw_name = event_names.get(int(event.name_iid), f"event_{event.name_iid}")
            task, event_no = _split_task_name(raw_name)
            stacks[track_uuid].append((task, event_no, timestamp))
        elif event.type == TYPE_SLICE_END:
            if not stacks[track_uuid]:
                continue
            task, event_no, start_ns = stacks[track_uuid].pop()
            block, group = _track_block_group(track_uuid, track_names, parents)
            slices.append(
                Slice(
                    rank=rank,
                    trace=path,
                    track_uuid=track_uuid,
                    block=block,
                    group=group,
                    task=task,
                    event_no=event_no,
                    start_ns=start_ns,
                    end_ns=timestamp,
                )
            )

    return [s for s in slices if s.duration_ns >= 0]


def _max_parallel(intervals: Sequence[Tuple[int, int]]) -> int:
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    cur = 0
    best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best


def collapse_waves(
    slices: Sequence[Slice],
    gap_ns: int,
    group_by_event: bool = False,
) -> List[Wave]:
    by_rank_task: Dict[Tuple[int, str, Optional[int]], List[Slice]] = defaultdict(list)
    for slc in slices:
        event_no = slc.event_no if group_by_event else None
        by_rank_task[(slc.rank, slc.task, event_no)].append(slc)

    waves: List[Wave] = []
    for (rank, task, event_no), task_slices in by_rank_task.items():
        task_slices = sorted(task_slices, key=lambda s: (s.start_ns, s.end_ns))
        cur: List[Slice] = []
        cur_end = -1
        for slc in task_slices:
            if not cur or slc.start_ns <= cur_end + gap_ns:
                cur.append(slc)
                cur_end = max(cur_end, slc.end_ns)
            else:
                waves.append(_make_wave(rank, task, event_no, cur))
                cur = [slc]
                cur_end = slc.end_ns
        if cur:
            waves.append(_make_wave(rank, task, event_no, cur))

    return sorted(
        waves,
        key=lambda w: (
            w.start_ns,
            w.end_ns,
            w.rank,
            w.task,
            w.event_no if w.event_no is not None else -1,
        ),
    )


def _make_wave(
    rank: int,
    task: str,
    event_no: Optional[int],
    slices: Sequence[Slice],
) -> Wave:
    start = min(s.start_ns for s in slices)
    end = max(s.end_ns for s in slices)
    blocks = {s.block for s in slices if s.block is not None}
    return Wave(
        rank=rank,
        task=task,
        event_no=event_no,
        start_ns=start,
        end_ns=end,
        slices=len(slices),
        blocks=len(blocks),
        total_task_ns=sum(s.duration_ns for s in slices),
        max_parallel=_max_parallel([(s.start_ns, s.end_ns) for s in slices]),
    )


def _fmt_us(ns: int) -> str:
    return f"{ns / 1000.0:.3f}"


def _parse_window_us(window: Optional[str]) -> Optional[Tuple[float, float]]:
    if not window:
        return None
    for sep in (":", ",", "-"):
        if sep in window:
            left, right = window.split(sep, 1)
            start_us = float(left)
            end_us = float(right)
            if end_us < start_us:
                raise ValueError("--window-us end must be >= start")
            return start_us, end_us
    raise ValueError("--window-us must look like START:END, in microseconds")


def _overlaps_window(
    start_ns: int,
    end_ns: int,
    origin_ns: int,
    window_us: Optional[Tuple[float, float]],
) -> bool:
    if window_us is None:
        return True
    start_us = (start_ns - origin_ns) / 1000.0
    end_us = (end_ns - origin_ns) / 1000.0
    return end_us >= window_us[0] and start_us <= window_us[1]


def _filter_slices(
    slices: Iterable[Slice],
    task_filter: Optional[str],
    min_duration_ns: int,
) -> List[Slice]:
    result = []
    for slc in slices:
        if task_filter and task_filter not in slc.task:
            continue
        if slc.duration_ns < min_duration_ns:
            continue
        result.append(slc)
    return result


def _filter_slices_window(
    slices: Iterable[Slice],
    origin_ns: int,
    window_us: Optional[Tuple[float, float]],
) -> List[Slice]:
    return [
        slc
        for slc in slices
        if _overlaps_window(slc.start_ns, slc.end_ns, origin_ns, window_us)
    ]


def _filter_waves_window(
    waves: Iterable[Wave],
    origin_ns: int,
    window_us: Optional[Tuple[float, float]],
) -> List[Wave]:
    return [
        wave
        for wave in waves
        if _overlaps_window(wave.start_ns, wave.end_ns, origin_ns, window_us)
    ]


def print_waves(waves: Sequence[Wave], limit: Optional[int], origin_ns: int) -> None:
    print(
        "idx rank event start_us dur_us end_us task slices blocks max_parallel total_task_us"
    )
    rows = waves if limit is None else waves[:limit]
    for idx, wave in enumerate(rows):
        event = "-" if wave.event_no is None else str(wave.event_no)
        print(
            f"{idx:4d} {wave.rank:4d} {event:>5s} "
            f"{_fmt_us(wave.start_ns - origin_ns):>10} "
            f"{_fmt_us(wave.duration_ns):>9} "
            f"{_fmt_us(wave.end_ns - origin_ns):>10} "
            f"{wave.task:<42} "
            f"{wave.slices:6d} {wave.blocks:6d} {wave.max_parallel:12d} "
            f"{_fmt_us(wave.total_task_ns):>13}"
        )


def print_slices(slices: Sequence[Slice], limit: Optional[int], origin_ns: int) -> None:
    print("idx rank start_us dur_us end_us task event_no block group track")
    rows = slices if limit is None else slices[:limit]
    for idx, slc in enumerate(rows):
        print(
            f"{idx:4d} {slc.rank:4d} "
            f"{_fmt_us(slc.start_ns - origin_ns):>10} "
            f"{_fmt_us(slc.duration_ns):>9} "
            f"{_fmt_us(slc.end_ns - origin_ns):>10} "
            f"{slc.task:<42} "
            f"{str(slc.event_no):>8} {str(slc.block):>5} "
            f"{str(slc.group):>5} {slc.track_uuid}"
        )


def print_bins(
    slices: Sequence[Slice],
    origin_ns: int,
    window_us: Tuple[float, float],
    bin_us: float,
    limit: Optional[int],
) -> None:
    if bin_us <= 0:
        raise ValueError("--bin-us must be > 0")
    rows = []
    start_us, end_us = window_us
    idx = 0
    cur = start_us
    while cur < end_us:
        nxt = min(cur + bin_us, end_us)
        mid_ns = origin_ns + int(((cur + nxt) * 0.5) * 1000)
        counts = Counter(
            slc.task for slc in slices if slc.start_ns <= mid_ns < slc.end_ns
        )
        if counts:
            rows.append((idx, cur, nxt, counts))
        idx += 1
        cur = nxt

    if limit is not None:
        rows = rows[:limit]

    print("idx start_us end_us active top_tasks")
    for idx, start, end, counts in rows:
        top = ", ".join(
            f"{task}:{count}" for task, count in counts.most_common(5)
        )
        print(
            f"{idx:4d} {start:9.3f} {end:9.3f} "
            f"{sum(counts.values()):6d} {top}"
        )


def print_summary(waves: Sequence[Wave], slices: Sequence[Slice], top: int) -> None:
    by_task: Dict[str, List[Wave]] = defaultdict(list)
    slice_durations: Dict[str, List[int]] = defaultdict(list)
    slice_counts = Counter()
    block_sets: Dict[str, set] = defaultdict(set)
    for wave in waves:
        by_task[wave.task].append(wave)
    for slc in slices:
        slice_counts[slc.task] += 1
        slice_durations[slc.task].append(slc.duration_ns)
        if slc.block is not None:
            block_sets[slc.task].add((slc.rank, slc.block))

    rows = []
    for task, task_waves in by_task.items():
        wave_span_ns = sum(w.duration_ns for w in task_waves)
        total_task_ns = sum(w.total_task_ns for w in task_waves)
        durations = slice_durations[task]
        rows.append(
            (
                wave_span_ns,
                task,
                len(task_waves),
                slice_counts[task],
                len(block_sets[task]),
                max(w.max_parallel for w in task_waves),
                total_task_ns,
                statistics.mean(durations) if durations else 0,
                max(durations) if durations else 0,
            )
        )
    rows.sort(reverse=True)

    print(
        "task_summary sorted_by_wave_span_us: "
        "task waves slices blocks max_parallel wave_span_us total_task_us "
        "avg_slice_us max_slice_us"
    )
    for row in rows[:top]:
        (
            wave_span_ns,
            task,
            num_waves,
            num_slices,
            num_blocks,
            max_parallel,
            total_task_ns,
            avg_slice_ns,
            max_slice_ns,
        ) = row
        print(
            f"{task:<42} {num_waves:6d} {num_slices:7d} {num_blocks:6d} "
            f"{max_parallel:12d} {_fmt_us(wave_span_ns):>12} "
            f"{_fmt_us(total_task_ns):>13} {_fmt_us(int(avg_slice_ns)):>12} "
            f"{_fmt_us(max_slice_ns):>12}"
        )


def write_csv(path: str, waves: Sequence[Wave], origin_ns: int) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "event_no",
                "start_us",
                "duration_us",
                "end_us",
                "task",
                "slices",
                "blocks",
                "max_parallel",
                "total_task_us",
            ]
        )
        for wave in waves:
            writer.writerow(
                [
                    wave.rank,
                    wave.event_no,
                    (wave.start_ns - origin_ns) / 1000.0,
                    wave.duration_ns / 1000.0,
                    (wave.end_ns - origin_ns) / 1000.0,
                    wave.task,
                    wave.slices,
                    wave.blocks,
                    wave.max_parallel,
                    wave.total_task_ns / 1000.0,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print time-ordered MPK task slices or collapsed waves from Perfetto traces."
    )
    parser.add_argument("traces", nargs="+", help="Perfetto trace file(s)")
    parser.add_argument("--mode", choices=["waves", "slices", "bins"], default="waves")
    parser.add_argument("--limit", type=int, default=120, help="Rows to print; <=0 means all")
    parser.add_argument("--top-summary", type=int, default=30)
    parser.add_argument("--min-us", type=float, default=0.0)
    parser.add_argument("--gap-us", type=float, default=0.0, help="Wave merge gap")
    parser.add_argument("--task-filter", default=None, help="Substring filter on task name")
    parser.add_argument(
        "--group-by-event",
        action="store_true",
        help="Do not merge same-name task waves with different profiler event numbers",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV output path for waves")
    parser.add_argument(
        "--window-us",
        default=None,
        help="Only print entries overlapping START:END microseconds from trace origin",
    )
    parser.add_argument(
        "--bin-us",
        type=float,
        default=1.0,
        help="Bin width for --mode bins, in microseconds",
    )
    args = parser.parse_args()

    all_slices: List[Slice] = []
    for fallback_rank, trace_path in enumerate(args.traces):
        rank = _infer_rank(os.path.basename(trace_path), fallback_rank)
        all_slices.extend(parse_trace(trace_path, rank))

    min_duration_ns = int(args.min_us * 1000)
    filtered_slices = _filter_slices(all_slices, args.task_filter, min_duration_ns)
    filtered_slices.sort(key=lambda s: (s.start_ns, s.end_ns, s.rank, s.block or -1))
    if not filtered_slices:
        print("No slices matched.")
        return

    origin_ns = min(s.start_ns for s in all_slices)
    window_us = _parse_window_us(args.window_us)
    slices = _filter_slices_window(filtered_slices, origin_ns, window_us)
    if not slices:
        print("No slices matched.")
        return

    limit = None if args.limit <= 0 else args.limit
    waves = collapse_waves(
        slices,
        int(args.gap_us * 1000),
        group_by_event=args.group_by_event,
    )
    waves = _filter_waves_window(waves, origin_ns, window_us)

    print_summary(waves, slices, args.top_summary)
    print()
    if args.mode == "waves":
        print_waves(waves, limit, origin_ns)
        if args.csv:
            write_csv(args.csv, waves, origin_ns)
    elif args.mode == "slices":
        print_slices(slices, limit, origin_ns)
    else:
        if window_us is None:
            first_us = min(s.start_ns for s in slices) - origin_ns
            last_us = max(s.end_ns for s in slices) - origin_ns
            window_us = (first_us / 1000.0, last_us / 1000.0)
        print_bins(slices, origin_ns, window_us, args.bin_us, limit)


if __name__ == "__main__":
    main()
