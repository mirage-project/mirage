# M3-I9 predictions addendum — registered 2026-07-27, BEFORE any cap arm has run

Basis: `results/costlaw_refit.md` (offline root-cause of the stage-2 C7 miss). The original
C5/C7 bands were priced against M3-I1's bs16 anchor (4566.5 ms), which went stale when
`624e8e1` (quantize row-partition, default ON) landed between windows: the SAME 203-iteration
shipped schedule measured 3689.27 ms in this window (-19.2%). The linear cost law's shape is
NOT at fault (held-out ratio error -0.90% sorted-padded / -0.92% msl-212 control; every
candidate form including a constant-cost null shows the same ~+21% absolute anchor error).

## Re-registered predictions (supersede C5/C7's absolute bands; ratios vs the SAME-WINDOW
## shipped-order control, which every stage now carries at >=3 reps)

- **C5':** cap=1 bs16 wave time = **1.50-1.75x** the same-window shipped control
  (anchor-independent form). In absolute terms against this window's 3689 ms control:
  2.16-2.39 s. The original C5 band (2.75-3.05 s) is WITHDRAWN — a working cap=1 would
  falsify it as written purely through the stale anchor.
- **Iteration/migration counts (unchanged, simulator-exact):** cap=1 -> 131 iterations,
  0 migrations, 0 straddling.
- **Sensitivity (registered):** cap=2 -> 1.08-1.10x, cap=4 -> 1.04-1.05x, cap=8 ->
  1.01-1.02x, with 65-73 migrations remaining — the optimum is cap=1; if measurement shows
  cap=2/4 beating cap=1, the law's max_chunk coefficient is wrong in a NEW way (fresh
  falsifier, not a re-fit license).

## Plan changes bound into the next window (from the refit's three required changes)

1. Every stage runs a same-window shipped-order CONTROL at >=3 reps; all headline claims are
   ratios to it (stage 0's n=1 control was the largest residual uncertainty).
2. One PROFILED bs16 shipped-order rep runs BEFORE stage 4, and (a,b,c) are re-fit on the
   binary that will actually run; the C5' band above does not move on that re-fit (it is
   ratio-registered), but the absolute expectation sharpens.
3. Kernel-dir naming carries every compile-time -D knob (the as-run plan's `_cap<n>` suffix
   discipline; upstream `_save_kernel_metadata` fix queued separately).

## Also riding the next window (registered here for one-place accounting)

- F1 oracle extension to bs4/8/16 with the corrected 4 us separator (~2 GPU-min;
  `--save-raw` then offline bucketing). Falsifier: activated > 8*n_live in any steady-decode
  iteration.
- M3-I5b `test_gate_topk.py` per-bs coverage completion (9/16/17/33) via per-bs invocations;
  the VPT=8 bs8 boundary-tie case is a KNOWN pre-existing artifact (never-shipped path) and
  is recorded, not a gate.
- Analysis-tooling convention adopted going forward: task-241/242 long-task classification
  threshold = 4 us (was 1 us). Historical tables (M3-I1/M3-I8 "activated groups" columns)
  stand as published with this note; their conclusions turn on real-tile counts, which do
  not move.
