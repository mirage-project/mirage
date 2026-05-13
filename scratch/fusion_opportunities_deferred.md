# Fusion opportunities deferred until View API lands

User decision 2026-05-13: the DTensor view/slice API (B3) is being deferred
indefinitely (requires runtime + repo-wide architecture work). Until that
lands, any fusion that would have been clean with View API has to be
implemented via **manual stride/offset params on every downstream kernel**
(the pattern used for QKV-a fusion at 2026-05-12). This file catalogs
fusion opportunities that fit the View API pattern, so we can revisit them
when View API is back on the table — or do them now with the manual-stride
approach if individually high-value.

## Status of fusion work to date

| Fusion | Status | Mechanism | Notes |
|---|---|---|---|
| **Q rope/nope split → fused q_b_proj_unabsorbed** (row-swap) | LANDED 2026-05-12 | Manual stride (qfused_mode kernel branch) | Env-gated `MPK_DSV3_QB_FUSED=1`, default OFF. Correctness verified bit-equal to baseline; not a perf win at current scale. |
| **QKV-a fused (q_a + kv_a → 2176)** | LANDED 2026-05-13 (this session) | Manual stride on 5 downstream kernels + OUTPUT_STRIDE fix on quantize | Env-gated `MPK_DSV3_QKV_A_FUSED=1`, default OFF until e2e perf measured. |

## Open fusion opportunities — would benefit from View API, can also do manually

### O1. kv_a + kv_rope fusion ("Path 1" in earlier work)

**What:** Merge the `kv_a_proj_with_mqa` and the rope projection so the FP8
GEMM emits both c_latent (512) and k_pe (128) in one shot. Currently QKV-a
fusion already does this — kv_a_latent and k_pe live in the same fused
qkv_a_out at offsets 1536..2047 and 2048..2111. So **O1 is subsumed by
QKV-a fusion** now that QKV-a fusion is correctness-correct. Not a separate
TODO.

### O2. Q split-K rope+nope fusion

**What:** When `_use_prefill=True`, the unabsorbed q_b emits `q_b_prefill_fused`
of shape `[S, H, 128+64]`. Currently the rope kernel walks q_pe at one
stride and the chunked_prefill walks q_nope at another. The kernel
already handles this via `qfused_mode=1` (row-swap branch). **Status:**
LANDED, env-gated. To turn on by default needs a perf re-measurement after
QKV-a fix.

### O3. Attention output write-into-residual

**What:** The MLA decode/prefill kernel emits its output into a separate
buffer, which is then added to the residual stream by `elementwise_add`.
Fusing the attention epilogue with the residual add would save one task
dispatch + one buffer write+read pair.

**Why View API helps:** the attention output is naturally a slice of the
residual stream (same shape, in-place ADD). View API lets us alias the two
without copy. Manual-stride alternative: extend the attention kernel to
optionally read `output[i]` and add to it (fused-residual epilogue), with
an `ADD_TO_OUTPUT=true` template flag.

**Cost (manual):** modest — 1 kernel change (attention) + 1 builder
rewiring. **Benefit:** ~one elementwise_add dispatch per layer (~3-5 μs
× 60 layers = 180-300 μs/token).

### O4. RMSNorm + Linear fusion (rmsnorm_linear)

**What:** Currently a dedicated `rmsnorm_linear_layer` exists in MPK but is
unused for the FP8 path. The pattern is:
  hidden → rmsnorm → quantize_fp8 → linear_fp8
Fusing rmsnorm into the quantize step would save 1 task dispatch + 1 buffer
write+read. Quantize already needs full-row scan to find abs-max; rmsnorm
is the previous full-row pass. Combining them saves one full hidden-state
read.

**Why View API helps:** the rmsnorm output is the quantize input. With View
API the buffer aliasing is direct; with manual stride it's straightforward
too. **Cost (manual):** modest — extend quantize_fp8 with a "fused rmsnorm
preamble". **Benefit:** ~one rmsnorm task dispatch + one full-row write per
layer.

### O5. Quantize + Linear fusion (rmsnorm_linear_FP8)

**What:** Beyond O4, fuse the quantize step INTO the linear kernel itself
(produce FP8 input on-the-fly inside the GEMM). The GEMM's first SMEM load
of A would absorb the quantize.

**Why deferred:** the GEMM kernel is complex and tightly tuned. Quantize
fusion would require deeper kernel changes. **Cost:** high. **Benefit:**
saves 1 full hidden-state R/W pair per FP8 linear (~6 RW/layer × O(60) layers).

### O6. MoE silu_mul fused with W2 GroupGEMM input

**What:** `moe_silu_mul` produces a tensor that's immediately consumed by
W2 GroupGEMM. Fusing the silu+mul as the W2 GEMM's load-A preamble saves
one buffer write+read pair per layer.

**Why deferred:** changes the W2 GEMM kernel signature. **Cost:** moderate
(1 kernel change). **Benefit:** ~one silu_mul dispatch per layer (and one
of the LARGE buffers in MoE — the intermediate is `[N, 9216, T]`).

### O7. AllReduce + Residual + Norm fusion

**What:** Currently after MLP, the pattern is `mlp_out` → AllReduce →
+residual → norm. The AllReduce kernel could fuse the residual add and
potentially the norm preamble into its epilogue.

**Why deferred:** AllReduce kernel is shared infra; changes affect all
TP paths. **Cost:** moderate-high. **Benefit:** ~one elementwise_add + one
rmsnorm per layer.

## Recommendation

Of the above, **O3 (attention + residual fuse)** and **O4 (rmsnorm fused
with quantize)** are the highest value-per-effort candidates without a
View API. Each is a single-kernel change + builder rewiring, no manual
stride proliferation needed.

**O6 (silu_mul + W2 fuse)** is a clear MoE-side win.

**O5, O7** are deferred — bigger kernel changes, lower priority until the
ones above are measured.

When View API does land, **all of the above become significantly cleaner**
(no need for new template params or kernel signatures; just alias the
buffer at the DTensor level). That is the long-term architectural fix.
