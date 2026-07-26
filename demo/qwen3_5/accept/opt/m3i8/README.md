# M3-I8 — right-sizing MoE expert activation

M3-I1 ranked this the #2 measured lever: 56.4 expert groups run per layer at
bs1 where top-8 on one token needs 8, credited at +37/+33/+17/+3.5/0% decode
throughput at bs 1/2/4/8/16. This issue owns it.

Prepared in **prep mode**: the analysis, the in-tree change and the capture are
done; no GPU was used beyond read-only artifact pulls. All eight B200s are
contended and M3-I2b owns the next window, so `plan_m3i8.sh` is written and
**not armed** — it refuses to start without `M3I8_ARMED=1`, and refuses while
I2b's GPU lock still exists.

## What the waste actually is

`mbt = max_num_batched_tokens = 16` is a compile-time batch dimension.
`prepare_next_batch` fills only `qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]`
rows; the rest are never refreshed, because attention and both GDN tasks write
per REQUEST slot and no request owns them. They keep the previous iteration's
residue, and the residue is *different in every row* — which is why 15 padding
rows produce ~48 extra distinct experts rather than one repeated set.

The router routes them anyway: `register_moe_topk_softmax_sm100_task` passes
`num_rows = batch_size = 16` and `finished = nullptr`, so every row's top-8
marks its experts in `moe_mask`, and the grouped GEMM's loop bound
`num_activated = mask[256]` sizes the entire stage.

Two things the issue brief guessed at, settled:

- **No earlier fix zeroed padding-row routing.** M2's "router fix" (b1e1e16)
  went the other way — it raised the kernel's VPT so a task covers 16 rows
  instead of silently dropping rows 8-15. That fix is what put the padding rows
  into the mask.
- **The task-level early exit already exists.** `for (ae = expert_offset; ae <
  num_activated; ae += expert_stride)` means a task with `expert_offset >=
  num_activated` runs zero iterations — M3-I1 measured 143 of 256 such tasks
  per layer at bs1 at ~0.53 µs each. So nothing in the grouped GEMM needs an
  early exit or a weight-gated skip; the fix belongs entirely in the router,
  and the static graph never has to change.

## The correction to the backlog's sizing

The backlog priced the lever as "MoE wall span × (1 − cap/measured)", i.e. wall
span proportional to group count. The compiled graph says otherwise:

- MPK launches the whole iteration with ONE `EVENT_LAUNCH_DEPENDENT_TASKS`, so
  task *t* runs on worker `(t − first_task_id) % 128` and each worker drains
  its queue in order (`persistent_kernel.cuh:1319-1340`).
- The grid `(128, moe_n_splits, 1)` is walked x-outer/y-inner, so live tasks
  are a **contiguous prefix** of the call site — `taskgraph_moe.py` confirms
  `expert_offset` = 0,0,1,1,2,2,… at all 80 sites in all three graphs.
- Per-task time is **flat**: task 242's `long_mean_us` is 30.72/30.74/30.83/
  30.83/30.67 µs across a 1.55× range in live-task count, and follows the
  weight tile exactly (`T ≈ 0.93 µs × N_tile/128 × K/128` fits both stages to
  5%). The stage is per-CTA latency-bound, not bandwidth-shared.

So stage cost is **worker depth**, `ceil(live_tasks / 128)`, and
`wall/layer ≈ waves·T + c(waves)` reproduces all ten of M3-I1's measured MoE
wall spans within 6.9%. At bs 1/2/4 the 112-120 live tasks already fit one
wave, so removing groups there removes work but not wall span:

| bs | live tasks now | waves | live tasks gated | waves |
|---:|---:|---:|---:|---:|
| 1  | 112.7 | 1 | 16.0 | 1 |
| 2  | 118.8 | 1 | 29.5 | 1 |
| 4  | 120.4 | 1 | 49.3 | 1 |
| 8  | 140.2 | **2** | 95.7 | **1** |
| 16 | 173.3 | 2 | 173.3 | 2 |

bs8 is the only case that crosses a boundary, and it crosses provably: 8 live
tokens × top-8 ≤ 64 groups × 2 splits = 128 tasks, one wave, whatever the
collision rate. Recorded prediction: **+3/+5/+3/+17/+1%**, not +37/+33/+17/
+3.5/0. Full derivation and bounds in `predictions.md`.

The +37% is still reachable — as **v1 × v2**. Right-sizing turns the stage from
expensive into *narrow* (16 of 128 workers at bs1), and only then does splitting
each expert's N further pay: `moe_n_splits = 4` predicts +10 to +23% at bs1 on
top of v1. That is staged as an arm, not applied to the tree.

## The change

| file | what |
|---|---|
| `tasks/blackwell/topk_softmax_sm100.cuh` | `num_active_rows` parameter (default −1 = no gating) gating `row_is_active` |
| `src/kernel/task_register.cc` | `params[1]` emits `runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]`; off ⇒ byte-identical call |
| `python/mirage/mpk/persistent_kernel.py` | `gate_padding_rows: bool = False` on the layer API |
| `models/qwen3_5/builder.py` | `MOE_GATE_PADDING_ROWS = True` + the router call site |

Only the two MARKING writes are gated. The row read, the input-buffer zeroing
(without which a split-k gate linear would accumulate padding-row logits across
iterations) and the top-k weight write are deliberately left alone. Bit-exact
for every live row; the argument and its failure modes are adjudicated in
`predictions.md`, anchored on the fact M2 already measured — every prompt's 64
output token ids are byte-identical at bs 1/2/4/8/16, i.e. a live row's output
is already known not to depend on what the other 15 rows contain.

**This issue changes C++**, unlike M3-I2b. The box's clone needs one rebuild
(`plan_m3i8.sh` stage 0); after it, arms are pure Python again, because `base`
is the same binary with `MOE_GATE_PADDING_ROWS = False` and stage 0 proves that
regenerates the pre-M3-I8 router call byte-for-byte.

## Files

| file | what | GPU? |
|---|---|---|
| `predictions.md` | pre-registered deltas, bit-exactness adjudication, falsifiers | no |
| `taskgraph_moe.py` | MoE dispatch auditor over a compiled `task_graph_rank0.json` | no |
| `model_moe_wall.py` | the cost model: fit on M3-I1, `--check` gates it, `--splits N` prices v2 | no |
| `static_checks.py` | the CPU-side gate (source, consumers, graph, model) | no |
| `mask_probe.py` | reads `layer_i_moe_mask[256]` out of a live run — the primary falsifier | yes |
| `stage_arms.sh` | materialises base / v1 / v2a / v2b on the box | no |
| `plan_m3i8.sh` | the staged capture — **refuses without `M3I8_ARMED=1`** | yes |
| `run_m3i8.sh` | per-arm capture, reusing M3-I1's instrument verbatim | yes |
| `ac3_m3i8.sh` | mechanism oracle → inert-at-bs16 → AC-3 sweep → byte diff → CI | yes |
| `analyze_m3i8.py` | the A/B tables, graded against `predictions.md` | no |
| `v2-moe-grid-widen.patch` | lever 2, staged as an arm, NOT applied | no |

## Running the static gate now

```bash
python3 static_checks.py                       # source + consumers + model
python3 static_checks.py --graph <task_graph_rank0.json> ...   # + the graph
python3 model_moe_wall.py --splits 4           # price the v2a arm
```

The compiled graphs are on the B200 at
`~/mpk-qwen35/m3i1/kernel_bs{1,2,4,8,16}_prof/task_graph_rank0.json`; they are
read-only inputs, 66-100 MB each.

## What is queued for the window

Stage order is value order, so a short window still settles the issue:

0. rebuild + codegen identity (`base` must emit the pre-M3-I8 router call)
1. **mechanism**: `mask_probe.py` base vs v1 at all five bs — hard cap check
2. AC-3 on v1: bs16 inert control first, then the sweep + per-case byte diff
3. perf **bs8 first**, then bs1 — the two cases the prediction lives or dies on
4. perf bs 2/4/16
5. v2a (`moe_n_splits = 4`) perf + AC-3
6. v2b (`moe_n_splits = 8`) perf
7. analysis

Arm it with `M3I8_ARMED=1` once M3-I2b's window has released its GPU lock.
