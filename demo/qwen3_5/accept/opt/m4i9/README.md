# M4-I9 — can fusion move the AC-4 floor?

**Answer: it moves it, and it does not reach 1.0 — and the second half of this
issue found the reason the first half's projection was too optimistic.**

Three admissible fusions were implemented, each behind its own default-off flag,
and all three plus their stack were measured end to end and re-derived on the
floors:

| arm | flags | bs1 floor / vLLM | bs8 | bs16 | e2e bs1 / bs2 / bs4 / bs8 / bs16 |
|---|---|---|---|---|---|
| A | HEAD | 4130.7 = 1.179x | 5275.7 = 1.116x | 5976.0 = 1.127x | — |
| **F** | `MPK_FUSE_SILU_QUANT` | **3750.1 = 1.071x** | 5032.1 = 1.065x | **5890.6 = 1.111x** | 1.032 / 1.021 / 0.999 / 1.003 / 1.006 |
| N | `MPK_FUSE_NORM_QUANT` | 4074.5 = 1.163x | 5317.6 = 1.125x | 6038.8 = 1.139x | 1.016 / — / — / 0.997 / 0.986 |
| G | `MPK_FUSE_RECUR_QUANT` | 4003.2 = 1.143x | 5233.0 = 1.107x | 6083.1 = 1.148x | 1.018 / — / — / 1.002 / 0.999 |
| **S** | all three | 3797.6 = 1.084x | **4936.7 = 1.044x** | 5994.4 = 1.131x | **1.026 / 1.034 / 1.038 / 1.021 / 1.004** |

**The best measured floor is 1.071 / 1.044 / 1.111 of vLLM's whole step. Nothing
reaches 1.0 anywhere.** The stack is the best end-to-end arm at bs 2/4/8 (up to
1.038x, 15/15 paired reps favour it at every batch size) and flag F alone is
marginally better at bs1 and bs16.

**Why the first half's projection (1.017 / 1.024 / 1.105) was wrong, and this is
the durable finding.** That projection modelled a fusion as `dur[victim] <- inc`
with `inc = 0`, i.e. the host absorbs the victim for free. Measured, **`inc` is
not a constant — it is ~0 only where the machine has slack.** At bs1 the work
bound is 2039.7 us = 0.58x of vLLM's step, the machine is starved, and every
flag's measured cp_exact matches the model to within 5%. At bs8/bs16 the work
bound is at or above cp, the machine is saturated, and the host's extra work
lands on a busy worker, lengthens the fused task, and — because the fused task is
*on the chain* — **lengthens cp**. Flags N and G go from a modelled −63/−68 and
−130/−107 us to a measured **+42/+64** and **−43/+25**. A fusion's sign inverts
with batch size.

Flag F is the exception, for a structural reason worth carrying forward: it is
the only one of the three that also **widens** its victim, moving the quantize
from 16 tasks per layer onto the silu grid's 128. Removing a chain record while
spreading its work wider is why F's work bound falls at every batch size while
N's and G's rise.

**Fusion cannot get under 1.0, now for two independent reasons.** First, the two
largest tiny stages left on the chain are structurally unfusable: the MoE router
pair (`topk_softmax` + the router `linear`, worth 602.6 us of cp_exact jointly at
bs1) and the MoE combine (230.8 us) are cross-task reductions. Second — the new
one — for the stages that *are* fusable, the benefit collapses and reverses once
the machine is work-bound, so the remaining candidates are worth less than the
inc=0 model said and at bs16 are worth less than nothing. At bs16 the work bound
binds, it is a *sum* and therefore exactly additive, and making every non-GEMM
stage in the graph free still leaves 1.030x.

So the disposition is: **ship the stack (or F alone) as a real 2-4% decode win,
and stop treating fusion as an AC-4 lever.** bs16 needs faster GEMMs; bs1/bs8
need the attention and w13 loops.

---

## 1. The instrument, and why it is the same one

M4-I8 established that the schedule is **static** (`runtime.cc:972-993` prelaunches every task
of the iteration and rewrites every other event to `EVENT_EMPTY`, so `worker(p) = (p-2) mod
128` and each worker drains its queue in order), which is what makes the realized schedule
*reconstructible* rather than modelled. This issue reuses that reconstruction unchanged:
`scripts/fuse_model.py` imports M4-I8's `sched_gap.py` and calls its `load_graph`,
`predicted_order`, `fit_assignment`, `decompose` and `cp_priority`. Every cell reports
`assign_qc: PASS` and `identity_error_ns: 0`, and the arm-A floors reproduce M4-I8's published
4130.7 / 5275.7 / 5638.7 us **exactly** — the same-basis control that says the tool is the tool.

Two things were added, because a fusion decision needs them and M4-I8 did not:

1. **Per-CALL-SITE resolution.** M4-I8 aggregated the chain by task *type*. `quantize_fp8`
   appears at 5 sites per layer with completely different producers, and only some are fusable
   at all, so the site is read off the task's own tensor names in `task_graph_rank0.json`
   (`outputs[0]` with the `layer_<i>_` prefix stripped) — the handle M4-I5's `cp_decompose.py`
   already uses for per-layer structure. The per-task output *tile* comes from the same
   descriptor, which is what decides grid compatibility.

2. **Fusion counterfactuals on both floors.** Fusing victim V into host H replaces
   `(dur[H], dur[V])` with one task of duration `dur[H] + inc`. On the longest-weighted-path
   computation that is exactly `dur[V] <- inc`, and on the work bound exactly
   `total -= (dur[V] - inc)`. So both floors are exact functions of `inc`, and `inc = 0` is the
   optimistic bound. Sets are evaluated **jointly on the DAG**, never summed — one-at-a-time
   deltas are sub-additive and summing them overstates.

The basis is M4-I8's own profiled buffers. That is HEAD's data: 5756c789 -> 8ff7be39 changes
only default-off JIT flags and `persistent_kernel.cuh`, whose default path M4-I8's gate 1c
proved differs by `__LINE__` immediates alone, and the task graph is built entirely by the
python builder plus `core.so`, neither of which changed. The graph JSONs confirm it —
identical task and event counts.

---

## 2. The chain, enumerated per site

bs1, iteration 288, window `[288,384)`. `n` is records on the *realized* critical chain,
`gap_us` their event-visibility + queue-pop cost, `tasks/step` the site's whole-graph task
count, `tile` the per-task output tile. Full tables for all three batch sizes:
`tables/fusion_floors.txt`.

| chain site | n | dur us | us/rec | gap us | tasks/step | tile | fusable? |
|---|---|---|---|---|---|---|---|
| `MOE_W13_FP8_BLOCKSCALE|moe_mid` | 80 | 581.3 | 7.27 | 148.7 | 10240 | [16,8,512] | GEMM |
| `ATTN|attn_out` | 10 | 588.4 | 58.84 | 15.3 | 20 | [16,2048] | (parity loop owns it) |
| `MOE_TOPK_SOFTMAX|topk_w` | 40 | 368.6 | 9.22 | 48.3 | 40 | [16,8]+[256,16]+[257] | **NO** — needs all 256 logits, produced by 32 tasks |
| `LINEAR|router_logits` | 40 | 350.0 | 8.75 | 46.9 | 1280 | [16,8] | **NO** — 32 tasks/layer, each needs the whole input row |
| `LINEAR_FP8_BLOCKSCALE|attn_resid` | 40 | 329.9 | 8.25 | 58.7 | 5120 | [16,16] | GEMM |
| `MOE_W2_FP8_BLOCKSCALE|moe_down` | 80 | 249.8 | 3.12 | 123.7 | 10240 | [16,8,1024] | GEMM |
| `LINEAR|gdn_ba` | 25 | 287.5 | 11.50 | 27.9 | 30 | [16,64] | off exact path (cpΔ 0) |
| `MOE_MUL_SUM_ADD|moe_out` | 40 | 243.5 | 6.09 | 46.1 | 640 | [1,2048] | **NO into producer** — sums over topk experts, each a different w2 task |
| `SIGMOID_GATE_MUL_ADD|r_prime` | 39 | 175.0 | 4.49 | 77.9 | 640 | [1,2048] | off exact path (cpΔ 0) |
| `GDN_RECURRENT|gdn_out` | 30 | 197.9 | 6.60 | 34.8 | 3840 | — | (its own loop) |
| **`QUANTIZE|moe_actq`** | **39** | **160.1** | **4.10** | **49.7** | **640** | **[1,8,512]** | **YES — into `silu_mul|moe_act` (SHIPPED)** |
| `GDN_CONV1D|gdn_qkv_c` | 29 | 170.3 | 5.87 | 38.2 | 240 | [16,8192] | real dependency |
| `LINEAR_FP8_BLOCKSCALE|qkvg` | 14 | 117.3 | 8.38 | 17.7 | 1440 | [16,64] | GEMM |
| **`QUANTIZE|gdn_out_proj_xq`** | **30** | **94.5** | **3.15** | **37.3** | **480** | **[1,4096]** | **yes, with care** — producer `gdn_recurrent` owns exactly one 128-group per v-head, but grid.z>1 has an arrival-counter epilogue |
| `RMS_NORM|post_norm` | 40 | 58.6 | 1.47 | 71.6 | 640 | [1,2048] | **NO** — its chain successor is the 32-task router |
| `SILU_MUL|moe_act` | 40 | 64.1 | 1.60 | 40.3 | 5120 | [1,1,512] | host of the shipped fusion |
| **`RMS_NORM|pre_norm`** | **40** | **48.6** | **1.22** | **48.0** | **640** | **[1,2048]** | **yes** — into `moe_mul_sum_add` (both grid `(mbt,1,1)`, row-local) |
| `SILU_MUL|shared_act` | 26 | 22.3 | 0.86 | 41.4 | 160 | [16,128] | off-chain |
| `EMBEDDING|embed_out` | 1 | 54.6 | 54.62 | 6.6 | 1 | [16,2048] | once per step |
| **`QUANTIZE|o_proj_xq`** | **10** | **32.2** | **3.22** | **12.2** | **160** | **[1,4096]** | **yes, with care** — attention's kv-head tile is 2048 cols = 16 whole groups |
| **`QUANTIZE|qkvg_proj_xq`** | **10** | **14.8** | **1.48** | **12.3** | **160** | **[1,2048]** | **YES** — tile-identical to its producer `rms_norm|pre_norm` |
| **`QUANTIZE|gdn_xq`** | **5** | **7.9** | **1.57** | **5.8** | **480** | **[1,2048]** | **YES** — same |

The per-layer chain is 14 records on attention layers and 15-16 on GDN layers (M4-I5, exact
from tensor names), so **3 of them are removable by fusion at GDN layers and 4 at attention
layers** — a fifth of the chain's record count. Each removal takes the record's duration *and*
its ~1.15 us of event visibility or ~1.55 us of queue-pop latency *and* its barrier pair.

### The two verdicts that decide the issue

**The MoE router pair is structurally unfusable, and it is what blocks 1.0.** Jointly it is
602.6 / 928.4 / 924.1 us of cp_exact at bs 1/8/16 on top of every admissible fusion, which is
the entire remaining distance. `router_logits` is dispatched as
`min(grid_for_rmsnorm_linear_layer(256), 256//8) = 32` tasks per layer, each computing 8 of the
256 expert logits for **all** 16 token rows (tile `[16,8]`), and `topk_softmax` runs as ONE
task that must reduce over all 256. Fusing them means collapsing the router to a single task,
i.e. streaming its whole 1 MiB bf16 weight through one CTA instead of 32 in parallel — and the
pair only costs 18.0 us per layer in total at bs1, so there is no headroom to pay for a 32x
serialization of the weight read. Fusing the router's *producer* (`rms_norm|post_norm`, a
row-local task) into the router is worse: all 32 router tasks would redo the whole-row norm.
Fusing `topk` into its consumer `w13` is worse still: 256 w13 tasks per layer would each redo
the top-k.

**The MoE combine cannot fuse into its producer.** `moe_mul_sum_add` weights and sums `moe_down`
over the topk experts, and each expert's slab is written by a different `w2` task. That is a
cross-task reduction, which under a persistent work-queue scheduler needs a barrier or
atomics — the thing M3-I3's rule forbids, because the scheduler cannot guarantee co-residency.
It *can* fuse into its **consumer** (the next layer's `rms_norm|pre_norm`, tile-identical), and
that is counted as admissible below; it removes the norm's record, not the combine's.

---

## 3. The floor model

Optimistic (`inc = 0`, the victim's work becomes free) unless stated. Joint sets computed on
the DAG. Full tables including an `inc = 500 ns` variant: `tables/fusion_floors.txt`.

| set | records removed | bs1 floor | x vLLM | bs8 floor | x vLLM | bs16 floor | x vLLM |
|---|---|---|---|---|---|---|---|
| HEAD | — | 4130.7 | 1.179 | 5275.7 | 1.116 | 5976.0 | 1.127 |
| F1 `rms_pre + quant` | 15 | 4068.7 | 1.162 | 5212.4 | 1.103 | 5968.7 | 1.126 |
| **F2 `silu + quant` (shipped)** | **39** | **3953.4** | **1.129** | **5086.7** | **1.076** | **5956.4** | **1.124** |
| F3 `combine + rms_pre` | 40 | 4083.2 | 1.166 | 5226.0 | 1.106 | 5967.3 | 1.126 |
| F4 `recurrent/attn + quant` | 40 | 4000.7 | 1.142 | 5142.1 | 1.088 | 5964.6 | 1.125 |
| F2+F4 | 79 | 3823.4 | 1.091 | 4953.1 | 1.048 | 5945.0 | 1.121 |
| **every admissible fusion** | **130** | **3761.7** | **1.074** | **4890.2** | **1.035** | **5921.8** | **1.117** |
| same, `inc = 500 ns` | 130 | 3826.7 | 1.092 | 4955.2 | 1.048 | 5934.9 | 1.120 |
| *+ router pair free* (inadmissible) | 210 | 3042.2 | **0.868** | 4122.8 | **0.872** | 5821.9 | 1.098 |
| *all tiny stages free* (inadmissible) | 361 | 2947.7 | **0.841** | 4077.7 | **0.863** | 5781.6 | 1.091 |
| *all non-GEMM free* (absurd) | 433 | 2444.4 | 0.698 | 3862.6 | 0.817 | 5459.5 | 1.030 |

**Which fusions would be needed to get under 1.0, and are they admissible?** At bs1 and bs8 the
required set is "every admissible fusion **plus** the router pair" — and the router pair is
exactly the part that is not admissible. At bs16 no set of fusions reaches 1.0 at all: the work
bound binds there, it is a sum over ~70000 tasks and therefore *exactly* additive, and the
whole tiny-stage work is only 24885 us = 194 us of work bound out of 5976. That is arithmetic,
not estimation.

---

## 4. What was implemented

`MPK_FUSE_SILU_QUANT=1`, **default off**, arm F. It fuses `moe_silu_mul` into its only
consumer, the fp32-block-scale activation quantize — the largest **admissible** fusion at every
batch size. New task type `TASK_MOE_SILU_MUL_QUANTIZE_FP8_SM100 = 243`, new device impl
(`tasks/blackwell/moe_silu_mul_quantize_fp8_sm100.cuh`), registered at the **silu** grid
`(mbt, topk, 1)` — one task per `(token, expert-slot)`, finer than the standalone quantize's
`(mbt, 1, 1)`. The bf16 `layer_i_moe_act` is never materialised.

**Bit-exact by construction — no cast position moves.** The fused body evaluates HEAD's own
expressions on HEAD's own operand groups:

* the activation is `T(input_val / (1.0f + expf(-input_val))) * mul_val` in type `T`, i.e.
  verbatim what `silu_mul_task_impl` stores into `moe_act` — fp32 sigmoid, **rounded to bf16**,
  then multiplied by the bf16 `up` half through `bfloat16_t::operator*` and kept in bf16. The
  only change is that the bf16 value stays in a register instead of visiting global memory and
  being read straight back. The CAST-POSITION RULE therefore does not bite: the rounding
  happens where HEAD rounds it.
* the amax is over the same 128-element group with the same `lane + e*WARP_SIZE` map and the
  same `eps` seed, reduced with the same `group_reduce_max<WARP_SIZE>`. `fmaxf` is exact and
  order-independent, so even the reduction order could not move a bit.
* `y_scale = group_max / max_8bit`, the fp32 scale store, and
  `fp8(clamp(orig / y_scale, min_8bit, max_8bit))` are copied verbatim from
  `per_token_group_quantize_fp8_task_impl`'s `SCALE_UE8M0 = false` branch.

**No extra barrier.** Warp `w` produces exactly the group it consumes, so the activation never
crosses a warp boundary and no `__syncthreads()` and no shared-memory staging is needed. That
is deliberate: M4-I8's arm O measured ~470 ns of makespan per extra scoped load + barrier pair
per chain record, and a fusion that paid for a block-wide barrier would give back much of its
own win.

**Soundness under the persistent scheduler** (the M3-I3 test): each task owns a disjoint output
range, the SwiGLU is elementwise and the amax reduction stays inside one warp, so there is no
cross-task reduction, no arrival counter and no co-residency requirement.

The unfused pair is kept for `expose_intermediates`, because the single-layer test-mode gates
and M2-I9's divergence bisection read `layer_i_moe_act` as a probe point.

---

## 5. Gates

### Gate 1 — registers. The gate that could have voided the issue.

MPK compiles ONE `__global__` with every task body inlined, so ptxas allocates a single
register budget for all of them, and HEAD already sits at 255. Fusion raises per-task pressure
by construction. Unlike M4-I8's arms this is not a `-D`: it changes the generated TU, so the
two arms are two different TUs compiled with identical flags at identical geometry (bs1,
msl=353). The probe therefore *cannot* precede the implementation; it is the first gate run
after it, before any e2e or AC-3 time was spent.

| arm | registers | barriers | stack | smem | spill st/ld | SASS lines | TU: silu / quantize / fused |
|---|---|---|---|---|---|---|---|
| A | **255** | 16 | 96 B | 5856 B | 0 / 0 | 149760 | 2 / 4 / 0 |
| F | **255** | 16 | 96 B | 5856 B | 0 / 0 | 150880 | 1 / 3 / 1 |

**Identical.** The fusion is register-neutral, so its e2e result is not confounded by spill —
the M3-I6a / M4-I6 failure mode does not apply. The TU census is also the flag-landed proof:
one `silu` call and one `quantize` call disappeared and one fused call appeared.
(`gates/ptxas/gate1_registers.txt`.)

### Gate 1c — the shipped default is unchanged, in the strong form

The fusion is gated on an env var read in the *builder*, not on an `#ifdef`, and it also adds a
task-type enum value, a `task_type_to_name` entry, a `tma.cuh` case and a `task_header.cuh`
include — all on every path. So the arm-A generated TU is compared against a pristine reference
built at the same geometry by a tree with none of this code (M4-I8's clone at 5756c789, whose
`kernel_A_bs1` came from the same harness at msl=353).

**BYTE-IDENTICAL** — `sha256 30fba2212b9a973425ab9803520c502e` on both, 6267 lines, 353568
bytes; same task/event counts (55528 / 2278) and same per-type census. Getting the strong form
required one fix: hoisting the shared `actq`/`acts` tensors above the `if/else` moved 240
`cudaMalloc`s in the default TU, so the `else` branch now declares them in the original order.
Arm F's TU is byte-identical before and after that fix, so both A/B passes measure the same
arm F. (`gates/ptxas/gate1c_default_unchanged.txt`.)

### Gate 2 — unit/oracle, both nvcc lanes

The reference is the **shipped unfused pair** — silu into a bf16 buffer, then the
fp32-block-scale quantize over it — not a torch ideal; a fp32 reference would disagree with
both arms and prove nothing about the fusion. Both arms live in the same TU. 8 shapes (the real
site `inter=512` at rows 1/2/8/16/128, plus 256 and 1024) x 4 value scales spanning the E4M3
clamp, the `1e-10` amax floor and the denormals.

| lane | fused vs pair | standalone quantize test |
|---|---|---|
| no `-use_fast_math` | **PASS — byte-identical, 32/32 cases** (fp8 bytes *and* fp32 scales) | PASS |
| `-use_fast_math` (what the megakernel ships) | **PASS — byte-identical, 32/32 cases** | fails (see below) |

Both lanes matter specifically here: `-use_fast_math` rewrites the SwiGLU's `expf` and both
divisions. The pre-existing standalone quantize test asserts its fp32 scale equals torch's
`absmax/448`, and that assert fires under fast math (delta 3.6e-12; its fp8 *values* are still
0-ULP). That is a property of the flag, not of M4-I9, and it is shown with a control rather
than asserted: the same test in the same lane on a **pristine** tree fails identically.
(`gates/unit/`.)

### Gate 3 — AC-3, arm F, all five batch sizes

`harness/gate_ac3_stable.sh` with `MPK_FUSE_SILU_QUANT=1` exported: 10 pinned prompts, msl 132,
64 new tokens, a **cold kernel compile per rep**, then the re-pinned three-part report.

* Stage 1 stability: **verdict STABLE**, 3 accepted reps at every batch size, **0 quarantined at
  every batch size**, fingerprint divergence rate 0.0%, 0 reps starting on a non-clean device
  (observed foreign floor 104 MiB), one physical GPU throughout.
* Stage 2 re-pinned report: **bit-exact 10/10 at every one of bs 1/2/4/8/16** against the
  committed `results/dumps_final`; agreement >= 90% **10/10 at every bs**, worst 0.9375
  (`p06-poem`); repetition ok.
* The single reference divergence is `p06-poem` position 60 at every bs,
  `engine=40581 ref=31000 baseline=40581` — the same token the committed baseline emits, i.e.
  the M2-adjudicated tie, reported as `same-as-baseline [known-adjudicated]`. `RUN_AC3_EXIT=1`
  is that pre-existing tie under the old strict-token harness and is superseded by the
  2026-07-29 re-pin, exactly as in M4-I5, M4-I7 and M4-I8. `REPIN_EXIT=0`.

So the bit-exactness claim is not only the construction argument and the unit test: it is
measured through 40 real layers on the real checkpoint at all five batch sizes, with the
megakernel's own `-use_fast_math`, against the committed dumps. This is also the only gate that
exercises the GRAPH change — one fewer op per layer, a new task type, a task at a finer grid —
so it covers the annotated-graph rewrite, the event fan-in counts and the task-to-worker
assignment as well as the arithmetic. (`gates/ac3/`.)

### Gate 4 — e2e A/B

Geometry B (synthetic 256-token prompts, msl=353, 96 decode steps, mbt=16), 3 reps per cell,
arms interleaved per `(bs, rep)` inside one GPU claim, a kernel dir per `(arm, bs)` — mandatory,
because the arm changes the generated TU and two arms sharing a dir under `--reuse-kernel` would
run one binary and report themselves identical (M3-I7 defect 3). **30 runs, 0 dirty, 0
unauditable**, each run's audit derived from its own `cuda_visible_devices` + `gpu_before` line,
observed pinned-device floor 120 MiB.

| bs | A per-rep (ms) | A med | F per-rep (ms) | F med | F/A | paired delta per rep (ms) | us/step |
|---|---|---|---|---|---|---|---|
| 1 | 783.4 / 786.0 / 787.2 | 786.0 | 761.7 / 761.7 / 771.9 | 761.7 | **1.0319x** | +21.7 / +24.3 / +15.3 | +213 |
| 2 | 950.0 / 964.3 / 955.7 | 955.7 | 930.3 / 943.5 / 936.0 | 936.0 | **1.0210x** | +19.7 / +20.7 / +19.7 | +209 |
| 4 | 1274.0 / 1313.5 / 1305.7 | 1305.7 | 1244.0 / 1306.7 / 1356.3 | 1306.7 | 0.9993x | +30.0 / +6.8 / −50.6 | −48 |
| 8 | 2136.6 / 2062.6 / 2092.7 | 2092.7 | 2129.1 / 2054.7 / 2086.4 | 2086.4 | **1.0030x** | +7.4 / +7.8 / +6.3 | +75 |
| 16 | 3265.0 / 3304.2 / 3250.6 | 3265.0 | 3246.1 / 3286.7 / 3229.7 | 3246.1 | **1.0058x** | +19.0 / +17.6 / +20.9 | +199 |

**14 of 15 paired reps favour the fused arm.** Paired is the right statistic: the arms are
interleaved per `(bs, rep)` with the same seed inside one claim, so the rep is a block, and the
*unpaired* per-rep range reaches 112 ms at bs4 while the effect is ~20 ms. bs4's single negative
rep (F 1356.3 against its own 1244.0) is what flips that row; every other cell is unanimous.

A separate **same-window control** at bs1/bs16 on the corrected base reproduces it rep for rep
(A 786.7 / 785.7 / 786.6 against F 762.1 / 762.0 / 762.5 at bs1; A 3265.8 / 3304.2 / 3250.4
against F 3246.9 / 3286.9 / 3230.8 at bs16), i.e. mean paired +24.6 and +19.2 ms.

Tokens: `tokens_sha256` identical for **15 of 15** A/F pairs — bit-exactness measured through 40
real layers with the megakernel's own `-use_fast_math`, not only in the unit test.

### Gate 5 — the chain actually got shorter

An e2e win with an unchanged chain would mean something else moved. cp_exact was re-derived on
arm F's **own** profiled buffers at the same geometry, the same windows and with the same
instrument, two consecutive iterations per cell (cp_exact is a max over chains of one
iteration's realized durations, so it needs the second iteration as a noise check). Every cell:
`assign_qc PASS`, `identity_error 0 ns`, `anchor_qc PASS` with `worst_rel_err 0.0`.

| bs | it | A cp | F cp | Δcp | A work | F work | Δwork | floor/vLLM A → F |
|---|---|---|---|---|---|---|---|---|
| 1 | 288 | 4130.7 | 3750.1 | **−380.6** | 2039.7 | 1955.0 | −84.7 | 1.179 → **1.071** |
| 1 | 289 | 4110.7 | 3728.8 | **−381.9** | 2033.4 | 1949.4 | −84.0 | 1.173 → **1.064** |
| 8 | 365 | 5275.7 | 5032.1 | **−243.6** | 4280.2 | 4211.9 | −68.3 | 1.116 → **1.065** |
| 8 | 366 | 5319.1 | 5054.5 | **−264.6** | 4287.4 | 4211.2 | −76.3 | 1.125 → **1.069** |
| 16 | 720 | 5638.7 | 5404.3 | **−234.4** | 5976.0 | 5890.6 | −85.4 | 1.127 → **1.111** |
| 16 | 721 | 5776.1 | 5542.8 | **−233.3** | 6112.8 | 6024.9 | −87.9 | 1.153 → **1.137** |

The two iterations agree on Δcp within 0.3–8%. `width.py`, the independent instrument, agrees on
the work bound over the whole window: −4.5 / −1.7 / −1.5%.

**The two geometries disagree on the STEP at bs8/bs16, and that is worth stating rather than
burying.** `width.py`'s window-averaged step moves −3.59% / +0.03% / +0.13% here (msl=897, late
context, profiled), against the geometry-B paired e2e of +213 / +75 / +199 us per step (msl=353,
unprofiled). The chain reduction is unambiguous and consistent in both; the *step* reduction is
only unambiguous at bs1. That is what the decomposition predicts: PATH work is 52.2% of the step
at bs1 but 43.5% at bs8 and 34.2% at bs16, while QUEUE work grows to 41.3% and 52.0%, so at
larger batch the realized step is packing-and-throughput-bound and a shorter chain has less to
convert. It also means the profiled instrument is not a fair step comparator between these arms:
arm F has 640 fewer profiled records per step, so it pays *less* of the 1.4–2.7% profiler
overhead M4-I8 measured, which biases the profiled step comparison in arm F's favour — and it
still comes out flat at bs8/bs16. The e2e A/B, which carries no profiler, is the step evidence;
this section is the chain evidence.

**The measured reduction is about twice what the site-local model predicted** (−177.3 / −189.0 /
−184.4). The extra is not the fused site: arm F's graph has 640 fewer tasks and 640 fewer events
per step, and the tasks the fusion does not touch measured faster for it — `w13` on the chain
581.3 → 569.5 us, attention 588.4 → 577.3 us at bs1. That is why the work bound fell 85 us when
the removed quantize work alone is only 19 us of it. The model holds every other duration fixed,
so it *understates* fusion; the honest consequence is recorded in §6.

Arm F's graph, from its own compiled `task_graph_rank0.json`: `t118` (silu) 5280 → 160 (only the
shared-expert site is left), `t275` (quantize) 3840 → 3200, `t243` (fused) 0 → 5120, tasks 55528
→ 54888, events 2278 → 1638.

---

## 6. Terminal disposition

**Arm F: WORKS, KEEP, STAY DEFAULT-OFF pending the milestone's integration decision.** It is
register-neutral, byte-identical in both nvcc lanes and at all five batch sizes end-to-end, the
shipped default TU is byte-identical to pre-M4-I9, and it moves the binding AC-4 floor by 108 /
51 / 16 thousandths of vLLM's step. Its e2e win is 1.032x / 1.021x / 1.003x / 1.006x — real at
bs1/bs2, and at bs4/bs8 inside the per-rep noise of this geometry even though the paired sign is
consistent.

**The AC-4 finding, which is the deciding one.** Modelled on arm F's own measured durations,
every remaining admissible fusion in the graph is worth another 0.054 / 0.041 / 0.006:

| set (on top of arm F) | records | bs1 floor | x vLLM | bs8 floor | x vLLM | bs16 floor | x vLLM |
|---|---|---|---|---|---|---|---|
| arm F as shipped | — | 3750.1 | 1.071 | 5032.1 | 1.065 | 5890.6 | 1.111 |
| + `recurrent/attn + quant` | 40 | 3621.4 | 1.034 | 4902.6 | 1.037 | 5878.5 | 1.109 |
| + `combine + rms` (3-way at attn layers) | 50 | 3692.0 | 1.054 | 4969.9 | 1.051 | 5879.1 | 1.109 |
| **+ every remaining admissible fusion** | **90** | **3563.3** | **1.017** | **4840.4** | **1.024** | **5855.3** | **1.105** |
| *+ router pair free* (inadmissible) | 170 | 2960.7 | 0.845 | 4077.4 | 0.863 | 5758.8 | 1.086 |
| *+ combine free too* (inadmissible) | 210 | 2711.3 | 0.774 | 4050.0 | 0.857 | 5736.2 | 1.082 |

So the bounded result, stated the way it should be:

* **bs16: fusion is finished.** The work bound binds, it is exactly additive, and every
  non-GEMM stage in the graph made free leaves it at 1.030x. bs16 needs faster w13/w2/GDN
  kernels — M4-I8's conclusion, unchanged.
* **bs1 and bs8: fusion lands at 1.017 / 1.024 and cannot be pushed past that by fusion alone**,
  because the two biggest tiny stages left on the chain (the router pair, the combine) are
  cross-task reductions. The honest caveat cuts the other way too: this model understated arm
  F's own effect by about 2x, so the true value of the full admissible set could plausibly reach
  ~1.00 at bs1/bs8. That is not a claim, it is a reason to measure — and getting there needs
  four more fusions, of which one touches `gdn_recurrent`'s split-epilogue, one touches the
  attention epilogue the parity loop owns, and the 3-way `combine + rms + quant` is blocked at
  GDN layers by `MAX_OUTPUTS_PER_TASK = 3` (it would need 4 outputs:
  `moe_out`, `pre_norm` for the `gdn_ba` linear, `xq`, `xs`).

**Backlog, ranked by measured cp per unit of risk:**

1. `gdn_recurrent + quantize` and `attention + quantize` (F4): −128.7 / −129.5 / −107.2 us of
   cp on top of arm F, 40 chain records. Tile-compatible (a v-head owns exactly one 128-group;
   a kv-head owns 16). Needs care around `grid.z > 1`'s arrival-counter epilogue and it touches
   the attention kernel the parity loop owns.
2. `moe_mul_sum_add + rms_norm` (F3, plus the 3-way at attention layers): −58.1 / −62.2 / −89.4
   us. Trivially tile-compatible; the GDN-layer 3-way needs `MAX_OUTPUTS_PER_TASK` raised from
   3, which changes `sizeof(TaskDesc)` and hence `TASK_DESCS_BUFFER_LENGTH` — price that first.
3. **Not** the router pair. Priced and rejected here: the only fusion is a 32x serialization of
   a 1 MiB weight read onto one CTA. If the router is worth attacking it is as a **width**
   lever (`topk_softmax` runs at `live/lvl = 1.0`), which is a different issue.

---

## 8. The continuation — two more flags, their stack, and the mechanism

The coordinator asked for the remaining admissible set, easiest-first, each behind
its own flag, with the stacked floor as the deciding number. Two were implemented;
one was deliberately not.

### What was built

**Flag A / arm N — `MPK_FUSE_NORM_QUANT=1`** (new task type 244). Each layer's
pre-norm fused into the quantize that consumes it. Bit-exactness here is
*structural* rather than arithmetic: `rms_norm_hopper_impl` already stages its
bf16 output in shared memory, so the fused task runs the norm half unchanged and
then **calls `per_token_group_quantize_fp8_task_impl`** — the same function the
standalone task calls, at the same instantiation — with only the input pointer's
address space changed. No rounding position moves, so the CAST-POSITION RULE does
not apply. No extra barrier either: the `__syncthreads()` the norm already runs
between filling the staging buffer and copying it out is exactly the one the
quantize needs. Three outputs at GDN layers (the bf16 norm still feeds the `ba`
projection), two at attention layers.

Not applied at the post-norm site, and that is a graph constraint rather than a
scope choice: `post_norm -> moe_xq` would make the fused task a fork-producer
whose consumer `w13` is a join-consumer (it also waits on the router), which
`build_annotated_graph` rejects as case 3 — the same rule that already forces the
routed and shared MoE branches to carry separate quantizes.

**Flag C / arm G — `MPK_FUSE_RECUR_QUANT=1`** (a new *variant* of task 237, not a
new type — the runtime dispatches on `(task_type, variant_id)`, so no new id had
to be threaded through the five tables). The GDN recurrence's gated-RMSNorm
epilogue already holds one whole v-head of bf16 results, and
`linear_value_head_dim == 128` is **exactly one fp8 scale group**, so the quantize
needs no regrouping. In the split configuration only the last-arriving task
reaches the epilogue, elected by M3-I3's `__threadfence()` + `atomicAdd` counter —
which is not a barrier, no task waits on a peer, and the fusion inherits that
property unchanged. It needs **no new input**, which is what makes it possible at
all: the recurrence already has 7 and `MAX_INPUTS_PER_TASK` is 7.

This is the one flag whose quantize arithmetic is hand-written rather than a call
to the shared function: the 128 values live one-per-thread across four warps, so
the amax needs a block reduction instead of the impl's warp shuffle. That is
admissible because `fmaxf` is exact and order-independent — a different reduction
*shape* cannot move a bit — and it is the claim `test_gdn_recurrent_fusedq.py`
falsifies. It is applied to **both** the decode-split and the prefill epilogue:
with the flag on, `out`'s consumer reads the fp8 pair, so a prefill chunk
producing only bf16 would hand the projection stale bytes.

**The attention sibling was skipped, on purpose.** `o_proj_xq` into
`paged_attention` is worth 32.4 / 32.9 / 26.4 us of cp_exact, and the attention
body is being rewritten by a concurrent parity loop — a fusion that lives inside a
kernel someone else is replacing is a merge conflict by construction, and the
composable alternative (a separate quantize task) is what already ships. The
1.5-2% of the deciding number it would have moved does not buy that.

**The 3-way `combine + rms + quant` was skipped, and the limit was priced.**
`MAX_OUTPUTS_PER_TASK` is 3 and the 3-way needs 4 at GDN layers (`moe_out`,
`pre_norm` for the `ba` linear, `xq`, `xs`). Lifting it 3 -> 4 grows `TaskDesc`
from 352 to 384 B (`output_ptrs[4]` +8 B, `output_tma_desc_ptrs[4][3]` +24 B),
which takes `TASK_DESCS_BUFFER_LENGTH = min((3072 - 56) / (sizeof(TaskDesc) +
sizeof(TaskId)), 16)` from **8 to 7** — a prefetch-depth change on *every task in
the kernel*, needing its own A/B before anything could be attributed to the
fusion. Against that, the 3-way buys the GDN layers' second record, ~48 us of cp
at bs1. Not cheap, and not worth it. Recorded as a bounded negative rather than
attempted.

Flags A and B from the original plan are also **mutually exclusive on the same
record**, for the same reason: A fuses `rms -> quant` and B would fuse
`combine -> rms`, and both consume the pre-norm rms. With `MAX_OUTPUTS_PER_TASK`
at 3 they cannot both apply, so B's marginal value in the stack is ~0 (their solo
cp deltas are 62.0 and 47.5 us at bs1). B was therefore not implemented and flag
C — which composes cleanly and is worth twice as much — was done in its place.
That reordering is a deviation from the requested sequence and the reason is
exactly this: the stacked number is what decides the issue, and B contributes
nothing to it.

### Gates, per flag

Every flag got the full cycle. **The register probe was the first gate after each
implementation**, and it cannot precede one: unlike a `-D` arm, a fusion changes
the *generated TU*, so the two arms are two different translation units and there
is nothing to compile before the code exists. (Recorded here so the next reader
does not re-derive that.)

| gate | arm N (norm+quant) | arm G (recur+quant) | arm S (stack) |
|---|---|---|---|
| registers / barriers / stack / spill / smem | **255 / 16 / 96 B / 0-0 / 5856 B** | identical | identical |
| gate 1c: arm-A TU vs pristine pre-M4-I9 | byte-identical (`sha256 30fba221`) | idem | idem |
| unit/oracle, no `-use_fast_math` | **PASS, 56/56 byte-identical** | **PASS, 20/20 byte-identical** | (covered by its parts) |
| unit/oracle, `-use_fast_math` | **PASS, 56/56** | **PASS, 20/20** | (idem) |
| pre-existing regression test in both lanes | quantize test: PASS/known | split-vs-golden: PASS both lanes | — |
| AC-3 stage 1, all five bs | **STABLE, 3 accepted / 0 quarantined** | **STABLE, 3/0** | **STABLE, 3/0** |
| AC-3 stage 2, all five bs | **bit-exact 10/10**, agreement 10/10 | **bit-exact 10/10** | **bit-exact 10/10** |
| tokens vs base, every e2e pair | identical 9/9 | identical 9/9 | identical 15/15 |

Arm G's unit test checks four things per case, because its quantize arithmetic is
the one that is hand-written: the bf16 `out` and the fp32 `state` must be
byte-identical to the UNFUSED split kernel (the fusion perturbs nothing), the fp8
bytes and fp32 scales must equal the standalone quantize run over that same bf16
in the same TU with the same flags, and the shipped no-store form
(`WRITE_OUT=false`) must produce the same fp8 as the store-keeping form. 20 cases
= 5 (slots, split, depth) configurations x 4 value scales, both nvcc lanes. The
`if constexpr (DV == 128)` guard in its dispatch is load-bearing: the impl
`static_assert`s that a v-head is one scale group, and the test extension also
instantiates 32- and 64-wide shapes that flag C is not defined for.

All three AC-3 runs report the same single reference divergence as every prior
issue — `p06-poem` position 60, `engine=40581 ref=31000 baseline=40581`, the
M2-adjudicated tie, `same-as-baseline [known-adjudicated]`. `RUN_AC3_EXIT=1` is
that pre-existing tie under the old strict-token harness; `REPIN_EXIT=0`.

Arm N's unit test (`test_rmsnorm_quant_fused.py`) runs at **256 threads**, the
megakernel's real block size: `rms_norm_hopper_impl`'s cp.async warm-up covers
exactly `NUM_THREADS * CHUNK` elements with no loop, so the template parameter and
`blockDim` must agree. It covers 7 shapes x 4 value scales x both output forms and
checks the fp8 bytes, the fp32 scales *and* the bf16 norm where it is kept.

New template parameters on the shared impls are **defaulted off**, which is what
keeps the pre-M4-I9 call sites emitting identical generated text and lets gate 1c
keep asserting the strong form after three flags' worth of edits.

### 8.3 The measured floors, and why the model was wrong

Same geometry, same windows, same instrument, two consecutive iterations per
cell; `assign_qc PASS`, `identity_error 0 ns`, `anchor_qc PASS worst_rel_err 0.0`
everywhere. Full table: `tables/cp_stack.txt`.

| flag | Δcp bs1 | Δwork bs1 | Δcp bs8 | Δwork bs8 | Δcp bs16 | Δwork bs16 |
|---|---|---|---|---|---|---|
| F silu+quant | **−380.6** | **−84.7** | **−243.6** | **−68.3** | **−234.4** | **−85.4** |
| N norm+quant | −56.2 | +3.8 | **+41.9** | **+26.9** | **+63.7** | **+62.8** |
| G recur+quant | −127.5 | −4.9 | −42.7 | **+73.0** | **+24.7** | **+107.1** |
| S all three | −333.1 | −35.1 | −339.0 | −36.7 | −241.4 | **+18.4** |

Against the inc=0 model:

| flag | bs1 modelled → measured | bs8 | bs16 |
|---|---|---|---|
| F | −177.3 → **−380.6** | −189.0 → −243.6 | −184.4 → −234.4 |
| N | −62.0 → −56.2 | −63.3 → **+41.9** | −67.8 → **+63.7** |
| G | −128.7 → −127.5 | −129.5 → −42.7 | −107.2 → **+24.7** |

**At bs1 the model is accurate to within 5% for all three flags. At bs8/bs16 it
has the wrong sign for two of them.** The discriminator is whether the machine has
slack: at bs1 the work bound is 0.58x of vLLM's step and the fused host absorbs
its victim in idle time it already had; at bs8/bs16 the work bound is at or above
cp, the host is busy, and the victim's work is paid in full — on a task that is
*on the chain*, so cp grows. `inc` is a function of machine occupancy, not a
property of the fusion.

That also explains the stack's non-additivity in both directions: at bs8 the stack
beats every single flag (−339.0 us of cp against F's −243.6) because F's widening
buys back the slack N and G consume; at bs1 the stack is *worse* than F alone
(−333.1 against −380.6) because there was already enough slack and N and G only
add work.

### 8.4 The revised answer

The first half of this issue projected that every remaining admissible fusion
would take the floor to 1.017 / 1.024 / 1.105. **That projection is refuted by
measurement**: the best floor any arm achieves is 1.071 (bs1, flag F), 1.044 (bs8,
the stack) and 1.111 (bs16, flag F). The gap is entirely the inc=0 assumption, and
the correction is not a tuning matter — it is that fusion's benefit is bounded by
the machine's spare capacity, which is exactly what AC-4's hard batch sizes do not
have.

So the honest verdict, replacing the earlier one:

* **No admissible fusion, and no stack of them, reaches 1.0 at any batch size.**
* **bs16 is finished for this lever twice over**: the work bound binds, it is
  exactly additive, all tiny-stage work is 194 us of 5976, and two of the three
  fusions make it *worse* there.
* **bs1 and bs8 are 4.4-7.1% short**, not the 1.7-2.4% the model claimed. The
  remaining unfused candidates (the attention quantize, the 3-way, flag B) are
  each worth less than the flags already measured, and the two big items (the
  router pair, the combine) are structurally unfusable.
* **What the flags ARE worth is a real decode win**: the stack is 1.026 / 1.034 /
  1.038 / 1.021 / 1.004x at bs 1/2/4/8/16, 15/15 paired reps, bit-exact
  everywhere. That is worth shipping on its own terms; it is not an AC-4 lever.

### 8.5 Disposition per flag

| flag | keep? | why |
|---|---|---|
| `MPK_FUSE_SILU_QUANT` (F) | **yes** | the only one that widens as well as shortens; work bound falls at every bs; best arm at bs1 and bs16 |
| `MPK_FUSE_NORM_QUANT` (N) | **yes, but only in the stack** | alone it regresses 0.3% at bs8 and 1.5% at bs16; inside the stack it is part of the best arm at bs 2/4/8 |
| `MPK_FUSE_RECUR_QUANT` (G) | **yes, but only in the stack** | alone it is +1.8% at bs1 and neutral above; its work-bound cost is the largest of the three (+107 us at bs16) |
| the stack (all three) | **yes — this is the artifact** | 1.026-1.038x at bs 1-8, 15/15 paired reps, AC-3 STABLE and bit-exact 10/10 at all five bs |

All four stay **default-off** pending the milestone's integration decision. If only
one ships, ship the stack: its worst cell (bs16, +0.36%) is still a win and its
best (bs4, +3.8%) is the largest single decode improvement this issue produced.

## 7. Reproducing

```bash
# on the coordinator box
bash opt/m4i9/scripts/setup_m4i9.sh                       # own clone at HEAD + fresh extension
                                                          # NOTE: purge build/lib.* and build/temp.*
                                                          # or setup silently reuses a cached core.so
python opt/m4i9/scripts/fuse_model.py <raw> <meta> <names> \
    --graph <task_graph_rank0.json> --window 288,384 \
    --sets opt/m4i9/scripts/fusion_sets.json              # the floor model, CPU only
bash opt/m4i9/scripts/gpu_guard_m4i9.sh 3,5,6,0,7 -- \
     bash opt/m4i9/scripts/sweep_m4i9.sh                  # e2e A/B, arms interleaved
bash opt/m4i9/scripts/mk_ptxas_m4i9.sh                    # registers, both arms, no GPU
bash opt/m4i9/scripts/check_default_unchanged_m4i9.sh      # gate 1c, no GPU
bash opt/m4i9/scripts/gpu_guard_m4i9.sh 3,5,6,0,7 -- \
     bash opt/m4i9/scripts/gate_unit_m4i9.sh              # unit/oracle, both nvcc lanes
bash opt/m4i9/scripts/gpu_guard_m4i9.sh 3,5,6,0,7 -- \
     bash opt/m4i9/scripts/prof_m4i9.sh                   # arm-F profiled capture
bash opt/m4i9/scripts/gpu_guard_m4i9.sh 3,5,6,0,7 -- \
     bash opt/m4i9/scripts/gate_ac3_m4i9.sh               # AC-3, all five bs
```

Raw profiler buffers are 0.9–1.5 GB per cell and are regenerable; they are not committed. The
derived per-iteration JSONs are in `raw/fuse/` (arm A) and `raw/fuseF/` (arm F).
