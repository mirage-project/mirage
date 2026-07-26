"""Regression guards for the two MPK profiler export bugs M2-I11 reproduced.

Both are consequences of workers and schedulers being two SEPARATE kernel
launches under `split_worker_scheduler`, each numbering its blocks from 0
(`demo/qwen3_5/accept/probes/runtime/p9_methodology.md`, step 2):

  bug 1  `export_to_perfetto_trace` raised `KeyError: (80, 0)` - `tid_map` is
         pre-populated for `range(header.nblocks)` only, and the header was
         written by whichever launch's block 0 got there first (the 80-block
         scheduler launch), while worker blocks index up to 127. Because
         `PersistentKernel.__call__` ran the Perfetto export before the CSV
         export with no guard, this also destroyed the otherwise-fine CSV.

  bug 2  slot aliasing: worker block b and scheduler block b both wrote their
         first event at buffer offset `1 + b` with the same stride, so the two
         launches shared slots and one tag namespace.

The C++ fix (`PROFILER_INIT_GLOBAL` in
include/mirage/persistent_kernel/profiler.h) gives both launches ONE shared
block-index space: workers [0, num_workers), schedulers after them, stride =
the total. These tests rebuild both layouts on a synthetic buffer and check
that the fixed one round-trips every event with distinct block ids while the
pre-fix one provably cannot, and that neither exporter raises on inputs it
used to reject.

Pure CPU: no GPU, no model, no build artifacts.

Run standalone:  python tests/runtime_python/test_profiler_export.py
Run under pytest: pytest tests/runtime_python/test_profiler_export.py
"""

import os
import sys
import tempfile

import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
)

from mirage.mpk.profiler_persistent import (  # noqa: E402
    event_name_list,
    export_to_csv,
    export_to_perfetto_trace,
)

# Mirrors tb::encode_tag / make_event_tag_* and the ProfilerEntry union in
# include/mirage/persistent_kernel/profiler.h (little-endian:
# raw = low32 | high32 << 32).
EVENT_IDX_SHIFT = 2
BLOCK_GROUP_IDX_SHIFT = 11
EVENT_NO_SHIFT = 19
EVENT_BEGIN, EVENT_END = 0, 1

NUM_WORKERS = 128
NUM_SCHEDULERS = 80
TOTAL_BLOCKS = NUM_WORKERS + NUM_SCHEDULERS
EVENTS_PER_BLOCK = 3
CAPACITY = 40000

_TMPDIR = None


def tmpdir():
    global _TMPDIR
    if _TMPDIR is None:
        _TMPDIR = tempfile.mkdtemp(prefix="mpk_profiler_export_")
    return _TMPDIR


def make_tag(block_group, event_idx, event_no, event_type):
    return (
        (block_group << BLOCK_GROUP_IDX_SHIFT)
        | (event_idx << EVENT_IDX_SHIFT)
        | (event_no << EVENT_NO_SHIFT)
        | event_type
    )


def pack(low32, high32):
    return (high32 << 32) | low32


def build_buffer(blocks, header_nblocks, stride):
    """Write EVENTS_PER_BLOCK BEGIN/END pairs per (block_id, task_id) entry.

    Slot formula matches PROFILER_INIT_GLOBAL with num_groups == 1:
    first write at `1 + block`, then `+= stride`.
    """
    buf = torch.zeros(CAPACITY, dtype=torch.uint64)
    buf[0] = pack(header_nblocks, 1)  # {nblocks, ngroups}
    ts = 1000
    for block, task_id in blocks:
        ptr = 1 + block
        for no in range(EVENTS_PER_BLOCK):
            buf[ptr] = pack(make_tag(block, task_id, no, EVENT_BEGIN), ts)
            ptr += stride
            buf[ptr] = pack(make_tag(block, task_id, no, EVENT_END), ts + 10)
            ptr += stride
            ts += 100
    return buf


def read_csv(path):
    with open(path) as f:
        return [line.rstrip("\n").split(",") for line in f][1:]


# --------------------------------------------------------------------------


def test_csv_decodes_the_fixed_disjoint_layout():
    """Every worker AND scheduler event survives, with distinct block ids and
    the right symbolic names."""
    blocks = [(b, 253) for b in range(NUM_WORKERS)]           # TASK_LINEAR_SM100
    blocks += [(NUM_WORKERS + b, 204) for b in range(NUM_SCHEDULERS)]  # PREPARE_BATCH
    buf = build_buffer(blocks, TOTAL_BLOCKS, TOTAL_BLOCKS)

    path = os.path.join(tmpdir(), "fixed.csv")
    export_to_csv(buf, path)
    rows = read_csv(path)

    assert len(rows) == TOTAL_BLOCKS * EVENTS_PER_BLOCK, (
        f"expected {TOTAL_BLOCKS * EVENTS_PER_BLOCK} paired events, "
        f"got {len(rows)}"
    )
    names = {r[1] for r in rows}
    assert names == {"TASK_LINEAR_SM100", "TASK_SCHD_PREPARE_BATCH"}, names
    assert {int(r[2]) for r in rows} == set(range(TOTAL_BLOCKS)), (
        "worker and scheduler blocks must occupy distinct indices"
    )
    assert all(int(r[7]) == 10 for r in rows), "durations must be 10 ns"
    print(f"  fixed layout: {len(rows)} events over {TOTAL_BLOCKS} distinct "
          f"blocks, names {sorted(names)}")


def test_pre_fix_layout_cannot_represent_both_launches():
    """With per-launch block indices and strides the scheduler launch reuses
    the workers' slots and tags: events are lost and the two launches are
    indistinguishable. Pins the mechanism the C++ fix removes."""
    blocks = [(b, 253) for b in range(NUM_WORKERS)]
    blocks += [(b, 204) for b in range(NUM_SCHEDULERS)]  # SAME ids as workers
    buf = build_buffer(blocks, NUM_WORKERS, NUM_WORKERS)

    path = os.path.join(tmpdir(), "aliased.csv")
    try:
        export_to_csv(buf, path)
        rows = read_csv(path)
    except RuntimeError as e:
        print(f"  pre-fix layout rejected outright: {type(e).__name__}")
        return

    assert max(int(r[2]) for r in rows) < NUM_WORKERS, (
        "aliased layout somehow produced scheduler-range block ids"
    )
    assert len(rows) < TOTAL_BLOCKS * EVENTS_PER_BLOCK, (
        "aliased layout lost no events - the reproduction no longer models "
        "the bug"
    )
    lost = TOTAL_BLOCKS * EVENTS_PER_BLOCK - len(rows)
    print(f"  pre-fix layout: {len(rows)} events, {lost} lost, block ids "
          f"capped at {max(int(r[2]) for r in rows)} (schedulers invisible)")


def test_unmapped_task_id_does_not_raise():
    """An id absent from event_name_list must degrade to UNKNOWN_<id> in BOTH
    exporters instead of raising - the Perfetto path used to index the dict
    directly."""
    unmapped = 400
    assert unmapped not in event_name_list
    buf = build_buffer([(0, unmapped), (1, 253)], 2, 2)

    path = os.path.join(tmpdir(), "unmapped.csv")
    export_to_csv(buf, path)
    with open(path) as f:
        assert f"UNKNOWN_{unmapped}" in f.read()

    if not _have_tg4perfetto():
        print("  tg4perfetto missing - Perfetto half skipped")
        return
    export_to_perfetto_trace(buf, os.path.join(tmpdir(), "unmapped.perfetto-trace"))
    print(f"  UNKNOWN_{unmapped} handled by both exporters")


def test_perfetto_survives_a_short_header():
    """The exact p9 failure: header says 80 blocks, the trace contains block
    100. Must not raise."""
    if not _have_tg4perfetto():
        print("  tg4perfetto missing - skipped")
        return
    buf = build_buffer([(0, 204), (100, 253)], NUM_SCHEDULERS, NUM_WORKERS)
    export_to_perfetto_trace(
        buf, os.path.join(tmpdir(), "short_header.perfetto-trace")
    )
    print("  short header tolerated (no KeyError)")


def _have_tg4perfetto():
    try:
        import tg4perfetto  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    for fn in (
        test_csv_decodes_the_fixed_disjoint_layout,
        test_pre_fix_layout_cannot_represent_both_launches,
        test_unmapped_task_id_does_not_raise,
        test_perfetto_survives_a_short_header,
    ):
        print(fn.__name__)
        fn()
    print("PROFILER EXPORT REGRESSION GUARDS PASSED")
