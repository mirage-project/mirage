"""Single source for the v2 profiler buffer layout and track labels.

Must match the V2_PROF_* constants in runtime_v2.cuh. Imported by both
prof.py (dump analysis) and profiler_persistent.py (Perfetto export) so the
two decoders can't drift out of sync with each other.
"""

# Reserved buffer tail (raw accumulators + emitter cursors), in entries.
# Layout low->high: trig_cursor(1) + trig_ring(1<<20) + misc(256) +
# cursors(1024) + spin(NUM_BUCKETS*256) + suffix(256).
V2_PROF_NUM_BUCKETS = 7
V2_PROF_TAIL_ENTRIES = (1048576 + 1) + 256 + 1024 + V2_PROF_NUM_BUCKETS * 256 + 256

# Offset of V2_PROF_MISC_BASE from the buffer end (misc sits below cursors,
# spin and suffix): the per-SM emitter drop counters live here.
V2_PROF_MISC_FROM_END = 256 + 1024 + V2_PROF_NUM_BUCKETS * 256 + 256

# Only the last WINDOW_ITERS decode steps are recorded (V2_PROF_WINDOW_ITERS).
WINDOW_ITERS = 25

# One Perfetto track per (SM, group): five warp-role tracks (0-4) plus three
# stall tracks (5-7, sub-slices inside a role's task window).
ROLE_NAMES = ["compute", "loader", "mma", "storer", "dispatcher",
              "compute-stall", "loader-stall", "mma-stall"]
