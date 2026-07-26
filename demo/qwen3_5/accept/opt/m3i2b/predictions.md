# M3-I2b predictions — written BEFORE any measurement

Baseline: M3-I1 profiled steady decode step at the AC-3 geometry (msl=132, mbt=16,
page 256, 64 new tokens): 15264 / 15648 / 15645 / 18618 / 22005 us at bs 1/2/4/8/16.

## The mechanism I found in the source (not yet confirmed on hardware)

`per_token_group_quantize_fp8.cuh:87-92` loops over **all** `BATCH_SIZE` rows of the
tile it is handed, deliberately: under the persistent runtime `blockIdx.x` is the
physical worker id, not the task's row, so the kernel cannot index a row from it and
instead makes every task instance produce the whole (identical) output.
`persistent_kernel.py:quantize_fp8_layer` then registers all three tensors with
`input_map = (-1,-1,-1)`, so a task's tile *is* the whole tensor. With
`grid_dim=(mbt,1,1)` the qwen3.5 builder therefore runs the complete quantize
**mbt = 16 times** at each of its 6 call sites per layer.

That is exactly what I1 measured without naming the cause: 3840 tasks/step at every
batch size, 21.9 us each, 84.1 ms of worker time at bs1 for what should be ~5.3 ms,
and a 4540 us wall span (29.7% of the bs1 step) at a self-concurrency of only
84002/4540 = 18.5.

## The change (v1)

`row_partition=(0,-1,-1)` on the 4 qwen3.5 quantize call sites: grid.x now splits
tensor dim 0 (the token axis), so `BATCH_SIZE` becomes 1 (2-D sites) or `topk`
(the 3-D MoE-activation site) and each task quantizes only its own rows. Task
**count is unchanged** (16 per call site), so graph width is unchanged; only the
redundancy goes away.

Bit-exactness argument: a 128-element group's fp8 bytes and its fp32 block scale are
functions of that group's own 128 elements alone, and the kernel's row loop carries no
state across rows. Redistributing rows across CTAs cannot move a bit. Both qwen3.5
scale layouts are `scale_ue8m0=False`, i.e. row-major with the input's row axes.

## Predictions (falsifiable)

P1. **Oracle**: the row-partitioned quantize is byte-identical to the whole-tensor
    grid, 2-D and 3-D, and both match the PyTorch reference primitive. If P1 fails
    the whole lever is void.

P2. **AC-3**: 50/50 (prompt, bs) token sequences byte-identical to the committed M2
    dumps. Confidence high; a failure means my bit-exactness argument is wrong and I
    root-cause the cast position rather than tune a tolerance.

P3. **Per-task**: `TASK_QUANTIZE_FP8_SM100` total_us at bs1 falls 84002 -> 5000-9000
    (mean per task 21.9 -> 1.4-2.5 us). n stays 3840.

P4. **Wall span**: quantize wall span at bs1 falls 4540 -> 300-700 us.

P5. **Step time at bs1**: central estimate **12.0-13.0 ms (+17% to +27% tok/s)**;
    full plausible range 11.2-13.5 ms. The optimistic bound (15264 - 4040 = 11.2 ms,
    +36%) assumes the whole recovered wall span is on the critical path; it is not
    quite, because mean total concurrency during quantize is 34.6, i.e. ~16 workers
    are doing other stages' work concurrently.

P6. **Occupancy**: perfect-pack bound at bs1 falls 5195 -> ~4600 us; occupancy
    (busy/step) RISES even though total work falls, because the step falls faster
    than the work does. Predicted bs1 occupancy 0.34 -> 0.38-0.42.

P7. **Batch-size shape**: the win is roughly batch-independent in absolute us (the
    quantize wall span is 4540 at bs1 and 4244 at bs16), so the relative win is
    LARGEST at bs1 and smallest at bs16: predict +17-27% at bs1/2/4, +14-22% at bs8,
    +12-19% at bs16.

## The falsifier that would redirect me

If P3 holds (work removed) but P4 fails — quantize wall span stays high because each
stage now costs launch/dispatch latency rather than work — then the stage is
LATENCY-bound, not work-bound, and widening cannot fix it. The next lever would then
be fusion (quantize in the norm epilogue / GEMM prologue), which is real kernel work
under the P10/P2 numerics classes, not a wiring change. The discriminator is v1's
measured mean_us per quantize task: <= ~2.5 us means work-bound (P4 should hold);
>= ~8 us with n unchanged means per-task overhead dominates.

## v2 (second lever: widen the remaining narrow stages) — sized before measuring

From I1's `per_task_concurrency` at bs1, the stages still running at concurrency <= 16
after v1, with their wall spans: router 565 us (conc 9.0), attention 513 (2.0),
sigmoid_gate 428 (15.3), moe_combine 222 (16.0), gdn_conv 193 (8.0), rms_norm 162
(15.5) = 2083 us total. Only two are widenable without touching a kernel:

  - `moe_mul_sum_add`: the layer API already exposes a grid.y split of the hidden
    axis (`input_map` (0,2,-1)/(0,1,-1)); the builder passes grid (mbt,1,1).
    Going to (mbt,8,1) is bit-exact (disjoint output columns, the sum over topk
    stays inside one task). 222 -> ~30 us.
  - `gdn_conv1d`: `gdn_conv_channel_blocks` is already a builder knob at 8.
    8 -> 32 is bit-exact (disjoint channel blocks). 193 -> ~50 us.

Predicted v2 gain: ~335 us, i.e. **+2.2% at the I1 bs1 step and ~+3% against the
post-v1 step**. I am recording this size UP FRONT so the result is judged against it:
this lever is small at this geometry, and the backlog's "+33%" figure for rank 3 was
"half the <=16-concurrency wall recovered", which credits stage tails and ramps that
no builder-side widening can reach. attention (conc 2.0) and the router (conc 9.0)
are the two big narrow stages and both need kernel work owned by other issues
(M3-I6a, and the MOE_ROUTER_MAX_ROWS_PER_TASK row loop).

## Pre-measurement refinement of P4/P5 (still before any GPU run)

Two artifacts sharpened the estimate before measuring, so I am recording the
refinement rather than quietly grading myself against the looser band.

**1. The redundancy is exact, and it is in the compiled artifact.**
`taskgraph_quantize.py` over M3-I1's own `kernel_bs1_prof/task_graph_rank0.json`:
240 quantize call sites, 16 tasks each, and at every site all 16 tasks carry ONE
distinct input offset and ONE distinct output offset with full-tensor tile dims
([16,2048] x120, [16,4096] x40, [16,512] x40, [16,8,512] x40). Useful rows per
step = 8320; row-quantizations actually executed = 133120; **124800 of them
(93.75%) are redundant.**

**2. Per-stage fixed overhead is ~0.6 us, not ~5 us.**
From I1's bs1 tables, wall span / number of stages vs mean task time:
rms_norm 162/81 = 2.0 us per stage vs 1.36 us mean task; moe_combine 222/40 =
5.6 vs 4.94; gdn_conv 193/30 = 6.4 vs 5.78. The gap is 0.6-0.65 us in all three,
so a stage costs its task plus ~0.6 us of dispatch, not more.

Applying that to quantize: 4540/240 = 18.9 us per stage today, of which ~18.3 is
work. After v1 each task does 1/16 of the rows, so a stage should cost
~1.2 + 0.6 = ~1.8 us and the wall span ~430-500 us.

Refined **P4: quantize bs1 wall span 4540 -> 400-700 us** (was 300-700).
Refined **P5: bs1 step 11.2-12.0 ms, i.e. +27% to +36% decode tok/s** (was
12.0-13.0 ms / +17-27%). Derivation: the per-type wall spans sum to 112% of the
step at bs1, i.e. the 40 layers really are close to serial, so removing ~4070 us
of wall removes (17120-4070)/1.12 = 11.65 ms of step under the overlap-preserving
model and 15264-4070 = 11.19 ms under the fully-serial model.

If the measured step lands ABOVE 13 ms with P3 satisfied, the serial-stage model
is wrong and I owe a mechanism for where the recovered wall went.
