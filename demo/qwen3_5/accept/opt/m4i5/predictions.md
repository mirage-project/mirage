# M4-I5 pre-registered predictions — `MPK_MOE_N_SPLITS` ∈ {2, 4, 8}

Written and committed **before** the A/B ran. The point of pre-registering is
that k=4 was already measured and rejected by M3-I8, so a re-test only counts if
it says in advance what is different and what would falsify it.

## What is different from M3-I8's rejection

M3-I8 staged this knob as two arms and reported:

| arm | bs1 | bs8 | bs16 |
|---|---|---|---|
| v1+v2a (`moe_n_splits = 4`) | +24.6 % | **+19.6 %, worse than v1 alone (+25.1 %)** | not run |
| v1+v2b (`moe_n_splits = 8`) | +24.7 % | **not run** | not run |

The disposition (`opt/backlog.json` lever 2) is *rejected-with-evidence*, and the
evidence is **v2a at bs8**. v2b was measured at bs1 only. The mechanism M3-I8
itself gave for the bs8 regression — "splitting returns bs8 to two waves" — is a
worker-depth argument, and worked through it predicts that k=4 is the *worst*
choice available and k=8 is not the same bet:

MPK hands the tasks of one event to workers in launch order (task at position j
goes to worker `j % 128`, each worker drains its queue in order), and the routed
tasks are a contiguous prefix of a call site. So a level costs
`max_w [ live_w·T + dead_w·T_dead ]`, with `live = g·k`, `emitted = 128·k` and
`T ∝ N_tile ∝ 1/k` (M3-I8's own fit `T = 0.93 µs · N_tile/128 · K/128`, no
intercept, validated on both stages across a 1.55× live-count range). Doubling
k doubles the depth and halves the task — neutral — until the extra tasks stop
crossing a wave boundary. `g` is the activated expert-group count, measured off
the M3-I7 traces at 8 (bs1), 32.9 (bs8), 60.6 (bs16 at the window's 12 live
slots).

## The numbers (opt/m4i5/tables/ceiling.json, `ceiling.py`)

Predicted wall span per step of each stage, calibrated per stage on the measured
span at k=2 (calibration factor in the table; 1.06–1.36):

| bs | stage | k=2 (measured) | k=4 | k=8 |
|---|---|---:|---:|---:|
| 1 | w13 (241) | 2346.0 | 1260.7 | **780.6** |
| 1 | w2 (242) | 1316.9 | 734.6 | **497.8** |
| 8 | w13 | 2496.2 | **2571.8 (worse)** | 2054.9 |
| 8 | w2 | 1394.9 | **1466.7 (worse)** | 1219.7 |
| 16 (12 live) | w13 | 3147.5 | 3232.5 | **3345.9 (worse)** |
| 16 (12 live) | w2 | 1677.9 | 1768.2 | **1888.6 (worse)** |

**P1 (the discriminating prediction).** At bs8, k=4 is a regression and k=8 is an
improvement. This is the non-monotonicity the wave-depth model implies and the
reason M3-I8's rejection does not carry to k=8. Falsified if k=8 ≤ k=4 at bs8.

**P2.** At bs1, k=8 is the best arm, by a wide margin: predicted step
9822.8 → 7438…7544 µs, i.e. **×1.30…×1.32** on decode. (Lower bound uses each
stage's measured sole-occupancy fraction, upper bound the full span delta.)
Falsified if bs1 k=8 is not the best of the three arms.

**P3.** At bs8, k=8 predicts ×1.057…×1.061.

**P4 (the one that can go against the lever).** At bs16 the routed GEMMs are
already nearly packed — 121.2 live tasks per level of 128 — so k=8 predicts a
**3 % regression** in the 12-live regime. At a genuinely 16-live bs16 decode step
(g ≈ 87, M3-I8's measured activation) the same model predicts k=8 *helps* by
≈ 24 %, because 174 live tasks are already two waves at k=2. The pinned 256/1024
bs16 arm runs the admission cap and spends most of its decode at 16 live, so the
measured bs16 result discriminates between those two regimes. Either outcome is
informative; a regression at bs16 means the knob is batch-conditional and does
not ship as an unconditional default.

**P5 (bit-exactness).** `grid.y` partitions the output columns: each task owns a
disjoint output range, the whole K reduction stays inside one task, each
n-block's accumulator was already independent, and no cross-task reduction
exists. So tokens must be **byte-identical** across arms at fixed batch size and
seed. Falsified by any token difference — and a difference would mean the split
is not what it claims to be, which voids the perf result regardless of its sign.

**P6 (register pressure).** MPK inlines every task into one `persistent_kernel`,
so one register budget is shared (M3-I6a). `moe_n_splits` shrinks the per-task
`NUM_N_BLOCKS`, hence the accumulator, so the megakernel's register count and
spill must not increase. Falsified by any rise in spill stores/loads on
`-Xptxas -v` of the generated TU; a rise would tax unrelated stages and is the
mechanism by which M3-I8's grid-widen could have lost at bs8 for a reason other
than waves.

## Voiding rules

* Any rep whose pinned device was not below 500 MiB at start is discarded, from
  that rep's OWN `meta.gpu_before` and `meta.cuda_visible_devices`, never from
  the candidate list the guard was handed (M3-I7's audit-join bug).
* Arms share nothing: a kernel dir per (geometry, arm, batch size), because
  `moe_n_splits` is a compile-time template argument. Sharing one would report
  the arms identical while the binary never changed (M3-I7 defect 3).
* An arm is reported only at n = 3 clean reps.
