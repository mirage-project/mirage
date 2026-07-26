"""Tolerant decode of a raw MPK profiler buffer saved by p9_profile_capture.py.

mirage.mpk.profiler_persistent.export_to_csv raises RuntimeError on the
first inconsistent (block,group,event_idx) BEGIN/END pairing and discards
everything decoded so far. For this run's config (num_workers=128,
num_schedulers=80 -- workers and schedulers are separate kernel launches,
each computing profiler_write_ptr/stride from its OWN local blockIdx/gridDim,
so worker-block-b and scheduler-block-b alias the same shared-buffer offset
on every write where their write-cycle counts coincide) that first mismatch
comes almost immediately (observed: block=4 group=0 event_no=0), so the
strict exporter yields ZERO usable rows.

This script reuses the framework's own tag-decoding
(_decode_events -- pure bit math, not implicated in the bug) but replaces
export_to_csv's pairing loop with a tolerant one: on a mismatch, drop just
that one dangling entry (with a logged reason) and keep going, instead of
raising and losing the whole trace. Output is the SAME CSV schema
export_to_csv would have produced, so p9_attribution_analysis.py is unchanged.
"""
import argparse
import csv
import sys

import torch

sys.path.insert(0, "/home/muhengl/mpk-qwen35/mirage/python")
from mirage.mpk.profiler_persistent import _decode_events, event_name_list  # noqa: E402


def tolerant_pairs(profiler_buffer):
    events = _decode_events(profiler_buffer)
    _, num_blocks, num_groups = next(events)

    pending = {}
    rows = []
    dropped = {"dangling_begin_overwritten": 0, "end_without_begin": 0, "trailing_dangling_begin": 0}

    for block_idx, group_idx, event_idx, event_no, event_type, timestamp in events:
        key = (block_idx, group_idx, event_idx)
        name = event_name_list.get(event_idx, f"UNKNOWN_{event_idx}")
        if event_type == 0:  # BEGIN
            if key in pending:
                # a BEGIN clobbered a still-open BEGIN for this key (aliasing
                # collision) -- drop the stale one, keep the newer BEGIN.
                dropped["dangling_begin_overwritten"] += 1
            pending[key] = (event_no, timestamp)
        elif event_type == 1:  # END
            if key not in pending:
                dropped["end_without_begin"] += 1
                continue
            begin_no, begin_ts = pending.pop(key)
            duration = (timestamp - begin_ts) & 0xFFFFFFFF
            rows.append((event_idx, name, block_idx, group_idx, begin_no, begin_ts, timestamp, duration))
        elif event_type == 2:  # INSTANT
            rows.append((event_idx, name, block_idx, group_idx, event_no, timestamp, timestamp, 0))

    dropped["trailing_dangling_begin"] = len(pending)
    return rows, dropped, num_blocks, num_groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-path", required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    buf = torch.load(args.raw_path)
    rows, dropped, num_blocks, num_groups = tolerant_pairs(buf)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_type_id", "task_type_name", "block_idx", "group_idx",
                    "event_no", "begin_ts", "end_ts", "duration_ns"])
        w.writerows(rows)

    total_dropped = sum(dropped.values())
    print(f"[decode_tolerant] {args.raw_path}: num_blocks={num_blocks} num_groups={num_groups} "
          f"clean_rows={len(rows)} dropped={dropped} "
          f"(dropped {100*total_dropped/max(1,len(rows)+total_dropped):.1f}% of would-be entries)")
    print(f"[decode_tolerant] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
