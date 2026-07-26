# M3-I2b — quantize fuse/widen + widening the narrow task stages

M3-I1 ranked `quantize_fp8` the #1 measured lever (+11-42% decode) and "widen the narrow
task stages" #3 (+18-35%), on the evidence that the megakernel is width-bound: occupancy
0.34-0.53 and 6.7-9.1 ms of every step running at <= 16 of 128 workers. This issue owns
both.

## What the quantize lever actually was

Not a kernel-efficiency problem and not a fusion problem. **93.75% of the work was
redundant**, and it is visible in the compiled artifact.

`per_token_group_quantize_fp8.cuh:87-92` loops over every row of the tile it is handed,
deliberately:

> Under the MPK persistent runtime, blockIdx.x is the PHYSICAL worker id, not the task's
> batch row: a task dispatched to worker W quantizes row W regardless of which row its
> task pointers describe [...] Fix: each task processes ALL rows (writes are identical
> across the redundant per-row task instances).

That fix is correct, but it only pays off if the grid partitions the tensor so that a
task's "all rows" is a small slice. `persistent_kernel.py:quantize_fp8_layer` registered
all three tensors with `input_map = (-1,-1,-1)` — no partition at all — so a task's tile
was the *whole* tensor. The qwen3.5 builder launched it at `grid_dim=(mbt,1,1)`, so every
one of the 16 tasks recomputed the entire quantize and wrote the same bytes to the same
addresses.

From M3-I1's own compiled task graph (`taskgraph_quantize.py`, no GPU needed):

```
=== m3i1/kernel_bs1_prof/task_graph_rank0.json
    3840 quantize tasks of 41048 total
    240 call sites
      x 120  tile dims [16, 2048]     tasks/site 16  distinct in-off 1  out-off 1  REDUNDANT x16
      x  40  tile dims [16, 4096]     tasks/site 16  distinct in-off 1  out-off 1  REDUNDANT x16
      x  40  tile dims [16, 512]      tasks/site 16  distinct in-off 1  out-off 1  REDUNDANT x16
      x  40  tile dims [16, 8, 512]   tasks/site 16  distinct in-off 1  out-off 1  REDUNDANT x16
    redundant row-quantizations per step: 124800
```

`--all` audits every task type in the megakernel, and quantize is the **only** redundant
one. The discriminator is the OUTPUT offsets, not the input offsets: shared inputs are
normal (every task of a GEMM reads the whole activation row and writes its own output
columns — `TASK_LINEAR_SM100` has 1 distinct input offset and 32 distinct output offsets),
whereas several tasks writing the SAME output address is duplicated work. `gdn_conv1d`,
`gdn_recurrent`, `paged_attention` and the two MoE grouped GEMMs also share one tile but
pick their slice from `task_desc->task_metadata`, which is legitimate and is excused by
name. Everything else — rmsnorm, sigmoid_gate, moe_mul_sum_add, silu_mul, argmax_partial,
embed, the dense GEMMs — is properly partitioned.

8320 rows of activation need quantizing per decode step; 133120 row-quantizations were
executed. That is exactly I1's unexplained measurement: 3840 tasks at *every* batch size,
21.9 us each, 84.1 ms of worker time at bs1 for ~5.3 ms of useful work, and a 4540 us wall
span — 29.7% of the bs1 step and the largest single task type at bs <= 4.

## The change

`quantize_fp8_layer` gains `row_partition`, defaulting to today's `(-1,-1,-1)` so no other
model moves. The four qwen3.5 call sites pass `(0,-1,-1)`: grid.x now splits tensor dim 0
(the token axis), the registered `BATCH_SIZE` becomes `dim0/grid.x`, and each task
quantizes only its own rows. **Task count is unchanged**, so graph width is unchanged —
only the redundancy goes away.

Bit-exact by construction: a 128-element group's fp8 bytes and its fp32 block scale are
computed from that group's own 128 elements, and the kernel's row loop carries no state
across rows, so redistributing rows over CTAs cannot move a byte. Only row axes may be
split; the API refuses to partition the UE8M0 path, whose scale is column-major
`[packed_k, aligned_batch]` and whose dim 0 is therefore the group axis, not a row axis.

## Files

| file | what |
|---|---|
| `predictions.md` | predictions written before any GPU run, with the refinement and its derivation |
| `taskgraph_quantize.py` | redundancy auditor over `task_graph_rank0.json`; `--all` sweeps every task type |
| `plan_m3i2b.sh` | the staged driver: oracle -> bs1 A/B -> AC-3 -> full sweep -> v2 |
| `run_m3i2b.sh` | per-arm capture, reusing M3-I1's `profile_wave.py`/`parse_profile.py` verbatim |
| `ac3_m3i2b.sh` | focused oracle + full AC-3 sweep + per-case byte diff + Qwen3-8B CI |
| `gpu_guard_m3i2b.sh` | 3-sample exclusive-GPU guard + `.gpu-locks/M3-I2b.lock` |
| `analyze_m3i2b.py` | the A/B tables |
| `v2-widen-narrow-stages.patch` | lever 2, staged on the box as arm `v2` but NOT applied to the tree until it is measured |

## Lever 2 — widening the remaining narrow stages, sized honestly up front

The backlog credited "widen the narrow task stages" with +18-35%, from "half of the
<=16-concurrency wall recovered" (7570 us at bs1). That aggregate over-credits the lever:
of those 7570 us, only 2083 us belongs to stages whose OWN concurrency is <= 16 (router
565 at 9.0, attention 513 at 2.0, sigmoid_gate 428 at 15.3, moe_combine 222 at 16.0,
gdn_conv 193 at 8.0, rms_norm 162 at 15.5); the rest is stage tails and ramps that no grid
change reaches. Size a width lever from `per_task_concurrency`, not from the concurrency
histogram.

Of that 2083 us, exactly two stages widen without touching a kernel, and both are
bit-exact (disjoint outputs, no cross-task reduction):

- `moe_mul_sum_add` — the layer API already exposes a grid.y split of the hidden axis
  (`input_map` (0,2,-1)/(0,1,-1)); the builder passed grid (mbt,1,1). (mbt,8,1) makes each
  task own 256 of the 2048 output columns, with the sum over topk staying inside one task.
- `gdn_conv1d` — `gdn_conv_channel_blocks` is already a builder knob at 8; 32 is legal
  (conv_dim 8192) and the kernel takes its channel block from `task_metadata.kv_idx`.

Predicted together: ~335 us at bs1, i.e. ~+2% of the I1 step and ~+3% of the post-v1 step.
Recorded before measuring so the result is judged against the right bar. The two large
narrow stages, attention at concurrency 2.0 and the router at 9.0, need kernel work owned
by M3-I6a and by the `MOE_ROUTER_MAX_ROWS_PER_TASK` row loop respectively.

## If the oracle ever disagrees

The decisive check is the generated CUDA itself, not the profiler. After any compile,

```bash
grep -A3 per_token_group_quantize_fp8_task_impl <kernel-dir>/test_rank0.cu | sort -u
```

prints the template arguments `<BATCH_SIZE, HIDDEN_SIZE, GROUP_SIZE, GLOBAL_STRIDE, ...>`
the registration derived. `BATCH_SIZE` comes from the PARTITIONED tile
(`output_tensors[0].dim`) while `GLOBAL_STRIDE` comes from the FULL tensor
(`dtensor.dim`), which is exactly what makes a row split legal — the tile shrinks, the
row stride does not. The `base` column below is not a guess — it is what M3-I1's own
`kernel_bs1_prof/test_rank0.cu` actually contains:

| site | base | v1 |
|---|---|---|
| `[mbt, 2048]`      | `<16, 2048, 128, 2048>` | `<1, 2048, 128, 2048>` |
| `[mbt, 4096]`      | `<16, 4096, 128, 4096>` | `<1, 4096, 128, 4096>` |
| `[mbt, 512]`       | `<16, 512, 128, 512>`   | `<1, 512, 128, 512>`   |
| `[mbt, topk, 512]` | `<128, 512, 128, 512>`  | `<8, 512, 128, 512>`   |

and `taskgraph_quantize.py` on the new `task_graph_rank0.json` must report
`distinct in-off 16` at every site instead of 1.

## Collecting the run

The capture is self-firing on the B200 box: `plan_m3i2b.sh` polls for an exclusive GPU
(10 h budget), then runs oracle -> bs1 A/B -> AC-3 -> 5-bs sweep -> v2, in that order so
a short window still yields a gate-complete result.

```bash
ssh catalyst-B200 'tail -40 ~/mpk-qwen35/m3i2b/logs/plan.log'        # stage progress
ssh catalyst-B200 'grep -E "^#####|identical|counts|CHANGED" ~/mpk-qwen35/m3i2b/logs/stage3.log'
ssh catalyst-B200 'cd ~/mpk-qwen35/m3i2b && ~/mpk-qwen35/venv-mpk/bin/python analyze_m3i2b.py . base v1 v2'
```

Outputs land as `~/mpk-qwen35/m3i2b/{ab.md,ab_step.csv,ab_pertask.csv,ab_tokens.json}`,
`bytediff_<arm>.json` (per-case AC-3 vs the committed `results/dumps_final`) and
`run_report_<arm>.json`. `tables_i1base/` holds M3-I1's own parsed tables as a labelled
fallback baseline, so a partial window can still be compared against a known reference —
it is a DIFFERENT session's capture and must be labelled as such in any report.
