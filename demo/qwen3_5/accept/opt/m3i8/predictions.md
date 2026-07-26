# M3-I8 predictions — written BEFORE any measurement

Baseline: M3-I1's profiled steady decode step at the AC-3 geometry (msl=132,
mbt=16, page 256, 64 new tokens): **15264 / 15648 / 15645 / 18618 / 22005 µs**
at bs 1/2/4/8/16. Every number below is derived from artifacts already in the
repo (`opt/tables/bs*_{attrib,concurrency}.json`) plus the compiled task graph;
no GPU was used. `model_moe_wall.py --check` reproduces all of it.

## The mechanism, exactly

`mbt = max_num_batched_tokens = 16` is a **compile-time** batch dimension.
`prepare_next_batch` packs only `qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]`
live tokens into rows `[0, live)` (persistent_kernel.cuh:317/372/404). Rows
above that are never refreshed — attention and GDN write per REQUEST slot, and
no request owns them — so they hold the previous iteration's residue, and the
residue differs from row to row.

`register_moe_topk_softmax_sm100_task` hard-codes `num_rows = batch_size = 16`
and passes `finished = nullptr` (task_register.cc:2616/2614), so
`topk_softmax_task_impl` routes **all 16 rows**, and every row's top-8 marks its
experts in `mpk_active_expert_ids`. The grouped GEMM then loops

```
for (int ae = expert_offset; ae < num_activated; ae += expert_stride)
```

so `num_activated = mask[256]` — the count of experts marked by ANY of the 16
rows — is what sizes the whole stage. **The premise in the issue brief is
wrong in one respect and it matters:** no earlier fix zeroed padding-row
routing weights. The M2 "router fix" (b1e1e16) went the *other* way — it raised
the kernel's VPT so the task covers 16 rows instead of silently dropping rows
8–15, which is what put the padding rows into the mask in the first place.

And the "do the GEMM tasks still dispatch?" question has a clean answer: **the
task-level early exit already exists**. A task whose `expert_offset >=
num_activated` runs zero loop iterations (M3-I1 measured 143 of 256 such tasks
per layer at bs1, ~0.53 µs each). So the right-sizing is a **router** change,
not a grouped-GEMM change — nothing in-kernel needs an early exit or a
weight-gated skip.

Measured `num_activated` per layer (M3-I1 `nlong / (40 × moe_n_splits)`):

| bs | activated (measured) | live tokens | cap = min(256, 8·bs) | excess |
|---:|---:|---:|---:|---:|
| 1  | 56.4 | 1  | 8   | 7.0× |
| 2  | 59.4 | 2  | 16  | 3.7× |
| 4  | 60.2 | 4  | 32  | 1.9× |
| 8  | 70.1 | 8  | 64  | 1.1× |
| 16 | 86.7 | 16 | 128 | 1.0× (nothing to remove) |

## The change

`topk_softmax_task_impl` takes `num_active_rows` (default −1 = no gating) and
gates only `row_is_active`, which guards only the two MARKING writes
(`mpk_routing_indices`, `mpk_active_expert_ids`). The row read, the
input-buffer zeroing that lets a split-k gate linear accumulate, and the top-k
weight write are all untouched. The registration emits
`runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]` — the same
scalar `argmax_reduce` and `reduction` already consume — and only when
`params[1] == 1`, so with the flag off the generated call is byte-identical to
the pre-M3-I8 one. The qwen3.5 builder turns it on via `MOE_GATE_PADDING_ROWS`.

## Bit-exactness

Claim: **every live row's output bytes are unchanged.** Three legs.

1. *Per-row independence of the router.* A row's top-k is a reduction over its
   own 256 logits inside one warp sub-group (`warp_mask` restricted to the
   sub-group, topk_softmax_sm100.cuh:155-163). Nothing a padding row does can
   reach a live row's `routing` or `topk_w` entry.
2. *Per-(token,slot) ownership in the GEMM.* Each (token, slot) pair is routed
   to exactly one expert, so no output element changes owner. The gather loop
   walks `t` in ascending order and padding rows have `t >= live`, so removing
   them cannot shift a live row's slot in the A tile — and even if it did, the
   `mma_m16n8k32` accumulates each row independently. The mask list's ORDER is
   already nondeterministic (an `atomicAdd` compaction) and already produces
   byte-identical AC-3 output, so a shorter list cannot break what a reordered
   one does not.
3. *Measured cross-row independence.* M2's AC-3 emitted **byte-identical token
   ids for every prompt at bs 1/2/4/8/16**. The harness fills a wave with
   `bs` DIFFERENT prompts, so row 0's neighbours are 15 stale residue rows at
   bs1 and 15 real, different live prompts at bs16 — and row 0's 64 output
   tokens do not move. That is a measurement, not an argument: live-row output
   is already known to be independent of what the other rows contain, and this
   change only alters what the other rows contain.

**Where it could break, adjudicated.**

- *A cross-row reduction anywhere in the 40-layer stack* would void leg 3. None
  exists in the source: rmsnorm / linear / silu / combine / argmax are per row,
  quantize is per 128-element group within a row, attention and both GDN tasks
  take their identity from `task_metadata.request_id` and skip slots with no
  new tokens. Leg 3 already tests this empirically at five batch sizes.
- *Padding rows going stale rather than being recomputed.* After the change a
  padding row's `mid`/`down`/`act` keep the previous iteration's values instead
  of fresh garbage. If that residue ever became Inf/NaN it would stay confined
  to that row by the same per-row scoping. Flagged, not dismissed: the AC-3
  sweep is what confirms it, and the bs16 arm (where no row is padding) is the
  control.
- *The router's input-buffer zeroing.* Deliberately NOT gated. Gating it would
  leave padding rows' logits accumulating across iterations under a split-k
  gate linear. This is the one clause where the "obvious" version of the change
  would be wrong.
- *Test mode / any caller with no live-row count.* `num_active_rows <= 0` or a
  value above `num_rows` means "no gating", so a single-layer harness that
  never populates `qo_indptr` keeps today's behaviour instead of routing zero
  rows.
- *DeepSeek-V3's `moe_topk_sigmoid_routing_layer`* has the identical latent
  behaviour and is deliberately NOT changed here; it needs its own validation.

## Predictions (falsifiable)

**P1 — codegen identity.** With `MOE_GATE_PADDING_ROWS = False` the emitted
`topk_softmax_task_impl(...)` call is byte-identical to M3-I1's compiled
`test_rank0.cu`. If it is not, `base` is not a baseline and the whole A/B is
void. (plan stage 0)

**P2 — the mechanism, per bs.** `mask[256]` per layer, gated arm:

| bs | base (measured) | predicted gated | HARD cap |
|---:|---:|---:|---:|
| 1  | 56.4 | **8.0** (exact — one row's top-8 is 8 distinct experts) | 8 |
| 2  | 59.4 | 14.7 ± 1 | 16 |
| 4  | 60.2 | 24.6 ± 2 | 32 |
| 8  | 70.1 | 47.9 ± 2 | 64 |
| 16 | 86.7 | 86.7 (unchanged — every row is live) | 128 |

The ± band comes from two independent estimates that agree (a union law fit on
the bs16 anchor, and an inclusion–exclusion decomposition of the measured
`|live ∪ padding|` totals; `model_moe_wall.py` prints both). The **cap** is not
a fit — exceeding it means the runtime scalar never reached the kernel.
Equivalently, `routing[:, r]` must be all-zero for `r >= live` and have exactly
8 non-zeros for `r < live`.

**P3 — AC-3.** 50/50 (prompt, bs) token sequences byte-identical to the
committed M2 dumps. bs16 first, as the inert control. A failure means leg 1 or
2 of the bit-exactness argument is wrong and I root-cause it rather than tune a
tolerance.

**P4 — per-task time must NOT move.** `long_mean_us` for tasks 241/242 stays
58.4/30.7 µs ±5%. It is flat to 0.5% across a 1.55× range in live-task count
today, which is the evidence that this stage is per-CTA latency-bound, not
bandwidth-shared. If it drops when the group count drops, the stage *was*
contended and every step prediction below is too pessimistic.

**P5 — step time. This is where I disagree with the backlog.**

M3-I1's backlog rank 2 predicted **+37/+33/+17/+3.5/0%** from "MoE wall span
scaled by (1 − cap/measured)", i.e. wall span proportional to group count. The
compiled graph and the profile say that model is wrong:

- MPK launches the whole iteration with ONE `EVENT_LAUNCH_DEPENDENT_TASKS`, so
  task *t* runs on worker `(t − first_task_id) % 128` (persistent_kernel.cuh:
  1319-1340) and each worker drains its queue in order.
- The grid is `(128, moe_n_splits, 1)` walked x-outer/y-inner, so the live
  tasks are always a **contiguous prefix** of the call site
  (`taskgraph_moe.py` confirms `expert_offset` = 0,0,1,1,2,2,… at all 80 sites).
- Therefore stage cost is set by **worker depth**, `ceil(live_tasks / 128)`,
  and `wall/layer ≈ waves·T + c(waves)` reproduces all ten measured wall spans
  within 6.9% (worst case is the bs2 point, whose sampled iteration was itself
  2.4% slow).
- At bs 1/2/4 the 112–120 live tasks **already fit one wave**. Removing groups
  there removes work, not wall span.

| bs | live tasks now | waves now | live tasks gated | waves gated |
|---:|---:|---:|---:|---:|
| 1  | 112.7 | 1 | 16.0  | 1 |
| 2  | 118.8 | 1 | 29.5  | 1 |
| 4  | 120.4 | 1 | 49.3  | 1 |
| 8  | 140.2 | **2** | 95.7 | **1** |
| 16 | 173.3 | 2 | 173.3 | 2 |

bs8 is the only case that crosses a boundary — and it crosses it *provably*:
`8 live tokens × top-8 ≤ 64` groups × 2 splits `= 128` tasks, exactly one wave,
whatever the collision rate.

Predicted decode-throughput delta for v1, as a bound pair. The uncertain term
is `c(1 wave)` = +21.6 µs (w13) / +13.4 µs (w2) of stage span that is not the
task itself: arrival skew inherited from upstream (survives, conservative) or
the cost of walking 112 tasks onto 112 workers (goes away, optimistic).

| bs | backlog said | conservative | optimistic | **recorded central** |
|---:|---:|---:|---:|---:|
| 1  | +37% | −1.7% | +8.1%  | **+3% (0 to +9)** |
| 2  | +33% | +1.8% | +11.8% | **+5% (0 to +12)** |
| 4  | +17% | −1.6% | +8.1%  | **+3% (0 to +9)** |
| 8  | +3.5%| +12.3%| +22.4% | **+17% (+12 to +24)** |
| 16 | 0%   | +0.8% | +1.3%  | **+1% (0 to +2)** |

So the shape is close to **inverted** from the backlog's: bs8 is the win, bs1
is not. Recording it here so the result is graded against the right bar.

Two effects the model does NOT price, both of which can only help, and both of
which I would attribute a bs1/2/4 gain to before claiming the wave model was
wrong: (a) the change removes ~6 GB of expert-weight traffic per step at bs1
(112.7 × 1.05 MB + 112 × 0.52 MB per layer × 40), which is ~460 GB/s of HBM
returned to whatever else is running; (b) MoE worker time drops from 3178 to
~450 µs per worker per step. Neither shortens the critical path under I1's own
finding that occupancy is 0.34 and the per-task-type wall spans already sum to
109-114% of the step — the step is latency-bound, not throughput-bound — but
both are real and unmodelled.

**P6 — v2 (staged, not in the tree).** Right-sizing turns the MoE stage from
expensive into NARROW (16 of 128 workers at bs1). Splitting each expert's N
further then pays, and only then. With `moe_n_splits = 4`, charging the extra
dead tasks at the measured 0.53 µs:

| bs | v1 central | v1+v2a conservative | v1+v2a optimistic |
|---:|---:|---:|---:|
| 1  | +3%  | +10.4% | +22.8% |
| 2  | +5%  | +14.3% | +27.0% |
| 4  | +3%  | +10.7% | +23.1% |
| 8  | +17% | +21.2% | +22.0% |
| 16 | +1%  | +3.2%  | +10.5% |

`moe_n_splits = 8` is comparable at bs1/2 and worse at bs8 (3 waves), so v2a is
the arm to beat. **The backlog's +37% is reachable — but as v1×v2, not v1.**

## The falsifier that would redirect me

**F1.** If the gated arm's `mask[256]` exceeds `min(256, 8·live)` at any layer,
the runtime scalar never reached the kernel. Everything else is void until
that is explained; `mask_probe.py` exits non-zero on it.

**F2.** If P2 holds (groups drop to the cap) but the bs8 step does **not**
improve by at least 10%, the wave model in C3 is wrong — the stage's cost is
not worker depth — and I owe a different mechanism before proposing v2, because
v2 is priced entirely by that same model. The discriminator is the measured
`wall_span/layer` for task 241 at bs8: the model says 117.4 → 61–83 µs.

**F3.** If P4 fails downward (per-task time drops with the group count), the
stage *was* contended, the conservative bound was the wrong one, and both v1
and v2 are worth more than recorded here — I would then re-derive the roofline
before claiming the extra.
