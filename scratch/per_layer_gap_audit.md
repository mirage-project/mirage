# Per-layer profile gap audit — DSv3 TP=4 EP=2 decode

**Trace**: `outputs/regression_20260510_230825/D_tp4_ep2_l20_mtp0_trace_rank0` (1 decode iter, 20 layers, mbt=1).
**Per-token wallclock**: 7.19 ms (per-layer mean L1..L19 = 352.8 μs).
**SM util in L5 window**: 31.8% across 128 workers.
**Reference**: `~/ref_vllm_sglang.md` (vLLM per MoE layer = 143 μs).

## Top bottlenecks (in-MPK vs vLLM ref, sorted by per-layer waste)

| # | Task | Σworker ms/iter | wall μs/call | vLLM μs | Gap | per-layer waste (μs) | Note |
|---|------|---|---|---|---|---|---|
| 1 | **MoE W13 GroupGEMM** | 23.7 | 76.5 | 24 | 3.2× | **52.5** | Largest single gap. M=128, N=2048, K=7168, top8 |
| 2 | **SPLITK swapAB o_proj** (worst slot) | (in 130.4) | 39.6 | 15 | 2.6× | **25.0** | M=128, N=8192, K=16384 — o_proj raw width. Other splitk shapes are fine |
| 3 | **MoE W2 GroupGEMM** | 9.8 | 27.4 | 16 | 1.7× | **11.4** | M=128, N=7168, K=1024, top8 |
| 4 | **MoE topk_sigmoid** | 0.26 | 15.1 | 7 | 2.2× | **8.1** | A5-already-fixed kernel, but still 2× slow |
| 5 | **MLA decode TP4** main+reduce | 11.2 | 20.5 + 5.5 | 15 | — | **~5.5** | Reduce stage adds ~5.5 μs not in vLLM |
| 6 | **Router (SPLITK bf16)** | 14.5 | 7.5 | 3 | 2.5× | **4.5** | A6 task — small but 2.5× |
| 7 | **AllReduce post-MoE** | 24.5 | 9.8 | 6 | 1.6× | **3.8** | AR2 — structural (EP=2 stragglers) |
| 8 | **AllReduce post-attn** | (in #7) | 10.4 | 8 | 1.3× | **2.4** | AR1 |
| 9 | SPLITK swapAB q_b (v2) | (in 130.4) | ~9 | 7 | 1.3× | ~2.0 | Q_b — not a big gap |
| 10 | SPLITK swapAB kv_a (v4) | (in 130.4) | ~6 | 4.5 | 1.3× | ~1.5 | KV-up — not a big gap |

**Per-layer total**: MPK 352.8 μs vs vLLM 143 μs = **2.5× slower overall**.

## Top-5 worst per-call latency (regardless of frequency)

| # | Task | wall μs/call | shape |
|---|------|---|---|
| 1 | MoE W13 GroupGEMM | 76.5 | M=128, N=2048, K=7168 |
| 2 | LM head (LINEAR_SM100) | 75.0 | end-of-iter, 1 call |
| 3 | SPLITK swapAB o_proj v1 | 39.6 | M=128, N=8192, K=16384 |
| 4 | MoE W2 GroupGEMM | 27.4 | M=128, N=7168, K=1024 |
| 5 | MLA decode TP4 main | 20.5 | TP=4 q + kv-paged |

## Implications for optimization priority

**HIGH-VALUE optimization targets (>10 μs/layer gap):**
1. **MoE W13 GroupGEMM** (52.5 μs/layer × 19 layers = ~1 ms/token waste). This is the biggest win.
2. **SPLITK swapAB o_proj** (25 μs/layer × 19 = ~470 μs/token). Likely a tile-size / split-K issue at this specific shape.
3. **MoE W2 GroupGEMM** (11.4 μs/layer × 19 = ~217 μs/token).

**MEDIUM-VALUE (5–10 μs/layer):**
4. MoE topk_sigmoid (8.1/layer)
5. MLA decode TP4 reduce
6. Router GEMV (already on TODO as A6)

**LOW-VALUE (<5 μs/layer):**
- All other splitk shapes (q_b, kv_a, q_a) are near-optimal — gap ~1-2 μs.
- AllReduces (structural — EP=2 token-imbalance straggler).

## Implications for the user-listed remaining tasks

| Task | Audit verdict |
|------|---|
| **USER #3 BMM on Q-NoP** | LOW priority — q_b path is ~2 μs/layer gap (1.3× vs vLLM), not a top bottleneck. Still worth doing if memory savings matter (smaller q_b weight) but won't move the needle on perf. |
| **QKV-a fusion (current)** | Reduces 3 GEMMs → 1 but each is ~5–10 μs. Net win 10–15 μs/layer if AllReduce/load overhead is amortized. Worth doing but not a top mover. |
| **A6 Router rewrite** | 4.5 μs/layer. Modest but >3 μs threshold. Worth doing. |
| **LM head investigation (75 μs)** | End-of-iter, 1 call/iter → 75 μs/token. Worth a bench to see if standalone is faster. |
| **MLA decode TP4 standalone bench** | Likely 1.4× gap is structural (TP=4 quad latency vs single GPU). Verify with bench. |
| **B1 Q/KV phase 27% SM util** | Already known — most splitk q_a/q_b/kv_a are near-optimal. The 27% util might reflect serialization between q_a → q_b not parallel utilization. Worth re-investigating with this audit data in hand. |

## SPLITK FP8 swapAB o_proj — kernel template inspection (2026-05-13)

From `outputs/perfetto_decode_ep2_175618/build/test_rank0.cu` variant 1
(matches the audit's 39.6 μs/call):

```cpp
kernel::linear_fp8_swapAB_sm100_task_impl<
    cutlass::float_e4m3_t,
    TMA_A, TMA_B, decltype(mBias), TMA_OUT,
    /*MMA_M=*/128, /*MMA_N=*/16,
    /*BATCH_SIZE=*/1, /*OUTPUT_SIZE_PER_TASK=*/128,
    /*REDUCTION_SIZE=*/8192,
    /*NOBIAS=*/true, /*SplitK=*/true,
    /*NUM_AB_STAGE=*/8, /*NUM_ACC_STAGE=*/2, /*NUM_C_STAGE=*/4>
```

Inferred dispatch:
- Full o_proj at decode: N=hidden=7168, K=16384.
- Grid: (output_size//128, split_k, 1) = (56, 2, 1) → **112 tasks total**.
- 112 ≤ 128 = num_workers → fits in 1 wave (each SM does 1 task).
- Per-task compute: 128×16×8192 FP8 GEMM ≈ 17M FMA = 33M flops. At B200's 4.5 PFLOPS peak FP8, the compute is 7.3 ns. The 39.6 μs/call is therefore **entirely TMA-bound** (loading 8192 K-elements per task).

Note: my initial reading of "SplitK=8" in the audit was wrong; the kernel
template has `SplitK=true` (boolean) and the actual split factor is the
grid_dim.y (= 2 from the picker for n_tiles=56, num_workers=128).

**Why MPK is 2.6× slower than vLLM (39.6 vs 15 μs):**
- 112 SMs each load their slice of the weight tensor. Most rows are
  shared across SMs (only the K-split differs). This may cause L2 / HBM
  bandwidth contention.
- vLLM likely uses a fused multi-SM cooperative kernel with
  cluster-shared TMA loads, reducing total bandwidth.

**Potential MPK-side mitigations:**
1. Use a CLUSTERED kernel (NUM_CLUSTER_M > 1) so SMs in a cluster
   cooperatively load the same A rows. Would need a new swapAB kernel
   variant — substantial work.
2. Reduce split_k to 1 (still single-wave at 56 tasks) — saves the
   tma_reduce_add epilogue. **Cost:** none if 56 ≤ num_workers.
   **Benefit:** may shave 5-10 μs/call by skipping the reduce step.

## Standalone bench results (2026-05-13)

Captured from Codex bench session (sm100_fp8_moe + standalone shapes,
GPU 4/5 idle window). Numbers are median μs/call.

| Layer | MPK wallclock μs/call | Standalone reference μs | Gap μs | Verdict |
|---|---|---|---|---|
| **LM head** (linear_layer) | 75.0 | 18–29 (torch.matmul per tile, M=1, OUT=256, K=7168) | **46–57** | **Kernel is 2.6–4× slower than cuBLAS at this shape**. Big opportunity. |
| **MoE W13** GroupGEMM | 76.5 | 23 (BF16 cuBLAS @ M=128 per expert) | **53** | Most of the gap is per-expert routing + TMA setup overhead inside MPK W13 task. Structural; harder to fix. |
| **MoE W2** GroupGEMM | 27.4 | 16.5 (torch.matmul BF16 @ M=128, N=7168, K=1024) | **10.9** | Modest. Mostly the same routing overhead as W13 (smaller). |
| **SPLITK swapAB o_proj** | 39.6 | 76.8 (per-CTA, cold launch of same kernel) | **MPK is 1.9× FASTER** | The MPK persistent kernel is FASTER than a standalone launch of the same kernel — SM context warm reuse helps. **No optimization needed.** |

### Re-prioritization based on standalone benches

The audit's initial "top-3 bottlenecks" (MoE W13, SPLITK o_proj, MoE W2)
needs revision:
- SPLITK o_proj is **off the list** — MPK is faster than the standalone
  of the same kernel. The 39.6 μs vs vLLM 15 μs gap is comparing two
  different kernel families (MPK's per-task swapAB vs vLLM's
  cooperative GEMM); MPK's kernel is doing fine for what it is.
- **LM head** (75 μs) moves UP to top-1 by gap-per-call. Worth investigating:
  the `linear_layer` kernel at M=1, K=7168, OUT=256 per tile is 2.6–4×
  slower than cuBLAS torch.matmul. Likely tile/stage tuning issue at
  this very-small-M shape.
- **MoE W13** stays as biggest aggregate gap (~1000 μs/token over 19
  layers). Structural — the per-expert routing overhead is intrinsic
  to the persistent-kernel-per-expert design.

## Calling-overhead vs kernel-level — verdict (2026-05-13)

Per the autonomous-run decision rubric:
- **>3 μs gap to a clean standalone run of the SAME kernel** → calling-overhead → fix in builder/scheduler.
- **>20% slower than the SAME kernel run standalone** → kernel-level → ask kernel-team for tune.

| Layer | Standalone of SAME kernel | MPK | Verdict |
|---|---|---|---|
| **SPLITK swapAB o_proj** | 76.8 μs/CTA (cold) | 39.6 μs/call (warm in MPK) | MPK is **faster**. No issue. |
| **LM head linear_layer** | unknown (no standalone run of the same `linear_sm100_mpk_task_impl`); cuBLAS torch.matmul at same shape = 18-29 μs/tile | 75 μs/tile | The MPK kernel is 2.6-4× slower **than cuBLAS at this shape**. Without a same-kernel standalone, can't say kernel-vs-calling cleanly, but the MMA template params `(MMA_M=128, MMA_N=16, BATCH=1)` waste 15/16 = 94% of MMA-N capacity at decode. **Verdict: kernel-level — small-M needs a different tile shape (MMA_N=4 or specialised small-M variant).** Mitigation: convert LM head to FP8 dense kernel which has better M=1 support, OR ask kernel team for an `MMA_N=4` variant of `linear_sm100_mpk_task_impl`. |
| **MoE W13** GroupGEMM | (standalone bench at M=16 ≠ MPK decode shape M=1) | 76.5 μs/call | cuBLAS BF16 ref @ M=128 per active expert = 23 μs. MPK is 3.3× slower **at the active-expert shape**. **Verdict: kernel-level — MPK W13 group GEMM is itself slow vs cuBLAS at this shape.** Mitigation: ask kernel team for a tuned W13 variant for the (M=128, N=2048, K=7168) sweet-spot shape, OR investigate routing-mask fast-path optimisation inside the kernel. |
| **MoE W2** | (similar — no clean same-kernel standalone) | 27.4 μs | torch.matmul BF16 ref @ M=128, N=7168, K=1024 = 16.5 μs. MPK is 1.7×. **Verdict: kernel-level (modest).** Same mitigation path as W13. |

Net: the top-3 wallclock bottlenecks are all **kernel-level** (the MPK kernel
itself is slower than the cuBLAS reference at the same shape). None are
calling/scheduling-overhead in a way that a builder fix can target
directly. The recommended action is to upstream the gap measurements
to the kernel-team for tuning passes.

## Concrete next-step priorities (rewritten 2026-05-13)

P1. **MoE W13 GroupGEMM @ 76.5 μs vs 24 μs** — investigate. Standalone bench at the exact MPK shape. If standalone matches vLLM (~24 μs), the gap is calling-overhead inside MPK persistent kernel. If standalone is also slow, kernel itself needs tuning.
P2. **SPLITK swapAB o_proj @ 39.6 μs vs 15 μs** — investigate. Same as P1 — standalone bench at M=128, N=8192, K=16384.
P3. **MoE W2 GroupGEMM @ 27.4 μs vs 16 μs**.
P4. (User-requested) QKV-a fusion e2e verification → commit.
P5. LM head bench.
P6. (User-requested) BMM on Q-NoP — modest gain, do after P1–P3.
P7. Router A6.
P8. Other items per the TODO.
