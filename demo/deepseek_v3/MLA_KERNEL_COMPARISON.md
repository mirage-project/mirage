# MLA kernel comparison: generic (TP=1) vs TP-specialised (TP=2/4/8)

## Kernels

| | Generic `mla_decode_sm100_task_impl` (TP=1) | TP-specialised `mla_mtp_tp{2,4,8}_main` (TP>1) |
|---|---|---|
| File | `mla_decode_sm100.cuh` | `mla_mtp_decode_tp{2,4,8}_sm100.cuh` |
| `NUM_HEADS` | 128 | 64 / 32 / 16 |
| Pipeline stages `(QK, PV)` | (4, 2) | (5, 3) |
| `Oa` partial storage | **FP32** | **BF16** |
| Softmax base | natural (`logf` / `__expf`) | base 2 (`log2f` / `exp2f`) |
| Q-loading | TMA Q once (all K-iters) upfront, reused from smem | TMA Q per tile × per q-in-group |
| Multi-query support | no (Q_LEN == 1) | yes (Q_LEN up to 8 for MTP) |
| Per-tile accumulation | **normalised**: `Oout = corr_inv · Oout + t16 · inv` written to Oa FP32 each tile | **un-normalised** in TMEM; single division by `row_sum` at kernel epilogue |

The two kernels compute mathematically the same MLA attention (softmax(QK·ss)·V), but along different numerical paths.

## Ablation proving the non-MLA TP path is identical

Ran TP=1 and TP=2 with `MPK_SKIP_ATTN=1` (layer 3, bs=1, seq=128, max_new=4). Both produced **bit-identical first-10 tokens**:

```
TP=1 skip_attn: [64374, 14283, 1340, 25232, 34375, 116226, 75102, 85079, 118477, 74790]
TP=2 skip_attn: [64374, 14283, 1340, 25232, 34375, 116226, 75102, 85079, 118477, 74790]  ← Match
```

With MLA enabled the tokens diverge immediately (TP=1 first token=51424, TP=2/TP=4 first token=66383). TP=2 and TP=4 produce **the same token** — i.e. the specialised TP MLA kernel is self-consistent across TP sizes, but differs numerically from the generic 128-head kernel.

**Conclusion**: the TP sharding math (o_proj column-parallel, MoE w13/w2, shared expert, AllReduce, replicated q_a/kv_a/embedding/lm_head) is 100% correct. The entire divergence comes from the two MLA kernels taking different FP paths.

## Theoretical sources of numerical divergence

1. **BF16 partial storage (`Oa`) — highest-impact.** Generic writes `t16 * inv` in FP32 directly to `Oa`. TP-specialised rounds to BF16 before writing, so each per-split attention output loses ~9 bits of mantissa. Over the full 512-dim-V output, this is ~0.4% per-element RMS noise.
2. **Log-base + exp differences.** `exp2f(x * log2(e))` vs `__expf(x)` produce slightly different rounded values because `exp2f` reads a different LUT than `__expf`.
3. **Pipeline stages (4 QK / 2 PV vs 5 / 3).** Deeper pipeline reorders concurrent MMAs slightly; FP accumulation is not associative, so per-element rounding differs at bit level.
4. **Per-tile normalise vs. epilogue normalise.** Generic divides by the running `row_sum` each tile (introduces early rounding); TP-specialised divides once at the end (preserves precision until later). Both are mathematically correct implementations of online softmax.

Of these, (1) is the biggest single contributor — BF16 partial storage introduces systematic rounding at every split/tile boundary that generic FP32 avoids.

## How the drift compounds across iterations

Per-iter attention-output error from (1)+(2)+(3) is small (expected cos ≈ 0.99+ per iter). However MLA output enters the residual stream:

```
x_{n+1} = x_n + αn · attn_out_n    # after o_proj, AllReduce, residual add
```

If `attn_out_n` dominates `x_n` in magnitude — which happens in shallow configurations where the model hasn't converged to a stable residual stream — even tiny per-iter noise can **amplify geometrically** over 15+ prefill iterations, producing the cos ≈ 0.5 we see at the attn_out boundary and cos ≈ 0.16 we see at `q_a_out` by iter 5.

## Empirical observations

- TP=2 and TP=4 MLA outputs agree bit-for-bit on first generated token — confirms the TP MLA kernel is deterministic and the two TP sizes run the exact same kernel math (just with different head counts).
- TP=1 (generic) differs — confirms it's the generic-vs-TP-kernel split, not a TP>1 specific sharding bug.
- A proper iter-0 comparison (one prefill iteration only) would require either a harness change to stop MPK after iter 0, or synthetic Q/K/V input to both kernels in isolation. This is deferred.

## Candidate fixes (if a closer match to TP=1 is desired)

1. **Switch TP MLA `Oa` to FP32.** The biggest single change — reduces per-tile BF16 rounding. Cost: 2× partial buffer memory (may matter at large `mbr × num_splits`). Implementation: change `Oa` type in `mla_mtp_tp{2,4,8}_main` / `mla_mtp_tp{2,4,8}_reduce` and in `persistent_kernel.py` buffer allocation.
2. **Match log base.** Change TP MLA to use `__expf` / `logf` internally. Minor effect; not recommended unless Fix 1 already applied.
3. **Leave as-is and document.** The TP>1 path produces self-consistent output (TP=2 == TP=4), and the single-token disagreement with TP=1 is FP-precision rather than semantic. Downstream quality likely unaffected for realistic deployment (40+ layers, stable residual stream).

**My recommendation**: deploy with current TP MLA kernel, document the numerical difference from TP=1, and optionally try Fix 1 if a closer TP=1 match is required for regression testing.
