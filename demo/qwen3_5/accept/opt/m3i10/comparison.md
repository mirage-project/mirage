# M3-I10 — vLLM vs MPK, per kernel

What each decode step costs in vLLM 0.25.1 at our exact workload and shapes, mapped stage by
stage onto MPK task families, so the remaining 4× gap can be spent as per-kernel optimization
targets instead of guesses.

**Headline.** MPK is slower than the corresponding vLLM kernel in **13 of 15 stages** at bs16
(12 at bs1, 13 at bs8, 12 at every batch size, 14 at at least one). The gap is not concentrated in
one place — it is 2×–11× spread across quantize, the MoE GEMMs, the dense fp8 GEMMs, the GDN
recurrent kernel and attention. MPK wins decisively in exactly one stage (norms/RoPE/glue, 3–4×
faster, because it fuses them) and is at parity in two more. The one gap that *grows* with batch
size is the GDN recurrent kernel: 7.4× at bs1, 10.8× at bs16.

| | bs1 | bs8 | bs16 |
|---|---:|---:|---:|
| vLLM decode step (this profile, union of GPU intervals) | 3567 µs | 4818 µs | 5363 µs |
| vLLM decode step (from the binding tok/s baseline) | 3502 µs | 4727 µs | 5301 µs |
| MPK decode step (M3-I1) | 15264 µs | 18618 µs | 22005 µs |
| ratio | **4.28×** | **3.86×** | **4.10×** |

That 4.28/3.86/4.10 reproduces M3-I1's 4.36/3.94/4.43 from an independent measurement (M3-I1
derived vLLM's step from throughput; this derives it from the kernel timeline).

---

## 1. The profiled engine is the binding baseline

Every hard check the baseline capture asserts passed on the profiled engine, in the same process
that produced the traces:

```
quantization_is_fp8 ................... true
has_flashinfer_trtllm_fused_moe ....... true
log_deepgemm_autodisabled ............. true   "Auto-disabled DeepGemm ... Falling back to CUTLASS"
log_dense_kernel_cutlass .............. true   "Selected CutlassFp8BlockScaledMMKernel"
log_moe_backend_flashinfer_trtllm ..... true   "Using FLASHINFER_TRTLLM Fp8 MoE backend"
tensor/pipeline/data_parallel_size .... 1/1/1
language_model_only ................... False
```

The kernel names in the traces confirm it independently — `cutlass_3x_gemm_fp8_blockwise` for
dense, `bmm_*E4m3*_sm100f` (TRT-LLM) for the MoE, `fmhaSm100fKernel_...H256PagedKv...` for
attention, KV block 1056, GDN state fp32. No silent backend fallback.

Throughput of the profiling process, unprofiled, against the binding table
(`baselines/vllm-0.25.1-20260725/README.md`):

| bs | binding tok/s | this run (median of 3) | delta |
|---|---:|---:|---:|
| 1 | 285.5 | 284.27 | −0.43 % |
| 8 | 1692.5 | 1748.18 | +3.29 % |
| 16 | 3018.1 | 2988.47 | −0.98 % |

All inside the baseline's own dispersion (IQR 3.1–3.4 % at bs8/16). The one deliberate deviation
from `bench_vllm.py` is `VLLM_ENABLE_V1_MULTIPROCESSING=0`, which moves the EngineCore in-process
so a step-driven profiler can run in the process that launches the kernels. Those three rows are
the evidence that it changes nothing that matters.

---

## 2. Method

**Capture.** `scripts/profile_vllm_decode.py` imports the repo's `bench_vllm.py` and reuses its
engine construction, prompt construction, fp8 assertions and decode-window timing verbatim.
Workload 256 in / 1024 out, greedy, fp8, CUDA graphs on. Per batch size: 1 warmup generate,
3 unprofiled timed reps, then 2 profiled generates. `GPUModelRunner.execute_model` is wrapped to
drive `torch.profiler`'s schedule, so windows are counted in engine steps, not wall time:
`skip_first=300, wait=60, warmup=3, active=50, repeat=3` — **6 windows × 50 consecutive steady
decode steps per batch size**, all taken between steps 300 and 640 of a 1024-token decode.

**Normalisation.** Kineto's `ProfilerStep#` markers live on the CPU timeline; with CUDA graphs the
GPU timeline lags by about a step, so a nominally 10-step window contains 10 ± 1 steps of GPU work
(≈9 % error). Instead each window is integrated over `[first, last)` occurrence of an **anchor
kernel that fires exactly once per decode step** (the lm_head GEMM), which contains exactly
`n_anchor − 1` complete steps. QC: with the correct anchor every per-step call count must be an
integer. **It is — `max_calls_int_dev = 0.0000` in all 18 decode windows**, and the counts come out
at exactly 160 / 80 / 40 / 30 / 20 / 10 / 1, matching the architecture (40 layers = 30 GDN +
10 full-attention, MoE on all 40).

**Two streams.** vLLM runs the shared expert on a second CUDA stream, overlapped with the routed
MoE — exactly 8 kernels × 40 layers on the side stream. So the sum of kernel durations
(5172 µs/step at bs1) is not the step time. The union of all GPU intervals is, and it closes
against the unprofiled step to **1.4 % at bs1, 5.3 % at bs8, 0.2 % at bs16**. Within a stage,
union == sum everywhere except the shared expert, so per-stage sums are still the right
per-stage cost; they just do not add up to the step.

**Coverage.** The stage map (`scripts/stage_map.py`, explicit substring + stream rules) leaves
**0 µs/step unmapped at bs1** and 2.6 µs/step at bs16 (one gather kernel).

**Per-site costs.** A trace gives one name for all 160 CUTLASS GEMMs in a step, but those are six
different shapes. Within a step the launch order is deterministic and each stream is in order, so
the ordinal position within (step, name, stream) is a stable site identity. Aggregating duration
by ordinal over 150 steps recovers per-site cost (`scripts/ordinal_profile.py`).

**Triton names.** vLLM's fused kernels are inductor-generated, and inductor numbers kernels *per
subgraph*, so the same short name can mean different things in different layers. Each name was
resolved against the compile cache's `# Topologically Sorted Source Nodes:` provenance and checked
for collisions across all 33 subgraph artifacts (`scripts/dump_triton.py`). No collisions found.

**The MPK side was not re-measured.** All MPK numbers are the committed M3-I1 per-task wall spans
in `opt/pertask_by_bs.csv` — the union of the time a task family is executing inside the
persistent kernel. M3-I1 established that summing those wall spans accounts for 109–114 % of the
step, so a family's wall span is the fair estimate of the step time it costs. NCU cannot
decompose MPK: it is a single persistent kernel, so per-task detail can only come from the
committed perfetto tables.

---

## 3. Correspondence table

`sites` is how many model sites the stage covers per step (both implementations run the same
model, so this is the same on both sides). vLLM = sum of that stage's kernel durations per step;
MPK = that task family's wall span per step.

### bs1

| vLLM µs/step | MPK µs/step | ratio | gap µs | vLLM µs/layer | MPK µs/layer | stage |
|---:|---:|---:|---:|---:|---:|---|
| 559.3 | 4540.0 | **8.12×** | 3981 | 2.80 | 22.70 | quantize / fp8 casts |
| 336.6 | 3084.0 | **9.16×** | 2747 | 8.41 | 77.10 | MoE routed GEMM w13 |
| 1353.5 | 2936.0 | **2.17×** | 1582 | 33.84 | 73.40 | dense projections (fp8 blockscale) |
| 300.7 | 1702.0 | **5.66×** | 1401 | 7.52 | 42.55 | MoE routed GEMM w2 |
| 163.7 | 1217.0 | **7.44×** | 1053 | 5.46 | 40.57 | GDN recurrent |
| 147.9 | 565.0 | **3.82×** | 417 | 3.70 | 14.12 | MoE router top-k/softmax |
| 133.6 | 513.0 | **3.84×** | 379 | 13.36 | 51.30 | full attention (incl. KV write) |
| 609.3 | 838.0 | **1.38×** | 229 | 8.58 | 11.80 | dense bf16 small + lm_head |
| 89.6 | 193.0 | **2.15×** | 103 | 2.99 | 6.43 | GDN conv1d |
| 2.5 | 54.0 | **21.9×** | 52 | 2.47 | 54.00 | embedding |
| 174.1 | 222.0 | **1.27×** | 48 | 4.35 | 5.55 | MoE combine |
| 8.7 | 37.0 | **4.26×** | 28 | 8.68 | 37.00 | sampling / argmax |
| 429.6 | 428.0 | 1.00× | −2 | 10.74 | 10.70 | shared-expert gate |
| 273.6 | 238.0 | 0.87× | −36 | 3.42 | 2.98 | MoE / shared SiLU-mul |
| 557.8 | 162.0 | **0.29×** | −396 | 6.89 | 2.00 | norms / RoPE / glue |

### bs16

| vLLM µs/step | MPK µs/step | ratio | gap µs | vLLM µs/layer | MPK µs/layer | stage |
|---:|---:|---:|---:|---:|---:|---|
| 462.7 | 4979.0 | **10.76×** | 4516 | 15.43 | 165.97 | GDN recurrent |
| 1123.2 | 5009.0 | **4.46×** | 3886 | 28.08 | 125.22 | MoE routed GEMM w13 |
| 606.2 | 4244.0 | **7.00×** | 3638 | 3.03 | 21.22 | quantize / fp8 casts |
| 755.9 | 2641.0 | **3.49×** | 1885 | 18.90 | 66.03 | MoE routed GEMM w2 |
| 1371.0 | 2973.0 | **2.17×** | 1602 | 34.27 | 74.33 | dense projections (fp8 blockscale) |
| 238.2 | 632.0 | **2.65×** | 394 | 5.96 | 15.80 | MoE router top-k/softmax |
| 148.7 | 461.0 | **3.10×** | 312 | 14.87 | 46.10 | full attention (incl. KV write) |
| 632.2 | 921.0 | **1.46×** | 289 | 8.90 | 12.97 | dense bf16 small + lm_head |
| 95.6 | 212.0 | **2.22×** | 116 | 3.19 | 7.07 | GDN conv1d |
| 13.5 | 115.0 | **8.54×** | 102 | 13.47 | 115.00 | sampling / argmax |
| 390.2 | 479.0 | **1.23×** | 89 | 4.88 | 5.99 | MoE / shared SiLU-mul |
| 194.3 | 243.0 | **1.25×** | 49 | 4.86 | 6.08 | MoE combine |
| 8.4 | 50.0 | **5.97×** | 42 | 8.37 | 50.00 | embedding |
| 455.1 | 432.0 | 0.95× | −23 | 11.38 | 10.80 | shared-expert gate |
| 680.4 | 162.0 | **0.24×** | −518 | 8.40 | 2.00 | norms / RoPE / glue |

bs8 is in `tables/comparison_by_stage.csv` along with per-layer columns for all three batch sizes.

### What maps to what

| MPK task family | vLLM kernel(s) | sites/step | mapping confidence |
|---|---|---|---|
| 279 `LINEAR_FP8_BLOCKSCALE_SM100` | `cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<…>>` | 160 / 160 | **exact** — same 6 GEMM shapes, same site counts |
| 241 `MOE_W13_FP8_BLOCKSCALE_SM100` | `bmm_E4m3_E4m3E4m3_Fp32_…_sm100f` (TRT-LLM MoE gemm1) | 40 / 40 | **exact** |
| 242 `MOE_W2_FP8_BLOCKSCALE_SM100` | `bmm_Bfloat16_E4m3E4m3_Fp32_…_sm100f` (gemm2) | 40 / 40 | **exact** |
| 275 `QUANTIZE_FP8_SM100` | `per_token_group_quant_8bit_kernel<BFloat16, Float8_e4m3fn, …>` ×2 variants | 200 / 200 | **exact** — 160 dense + 40 MoE (column-major scales) |
| 234 `GDN_CONV1D_SM100` | `_causal_conv1d_update_kernel` | 30 / 30 | **exact** |
| 237 `GDN_RECURRENT_SM100` | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 30 / 30 | **exact** |
| 257 `ATTN_SM100` | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausal…ForGen` + `reshape_and_cache_flash_kernel` | 10 / 10 | exact kernels, **context differs** (see caveats) |
| 260 `MOE_TOPK_SOFTMAX_SM100` | `routingIndicesBlockKernel` (bs1) / `routingIndicesDynBlockKernel` (bs8/16) | 40 / 40 | **exact** |
| 261 `MOE_MUL_SUM_ADD_SM100` | `moe::dev::finalize::finalizeKernel` | 40 / 40 | **exact** |
| 238 `SIGMOID_GATE_MUL_ADD_SM100` | shared-expert gate GEMM + `sigmoid_kernel_cuda` + `BinaryFunctor` mul (+ splitK reduce at bs8/16) | 40 / 40 | **exact stage**, 1 MPK task vs 3–4 vLLM kernels |
| 118 `SILU_MUL` | `activationDeepSeekKernel` (routed) + `triton_poi_fused_mul_silu_slice_0` (shared) | 80 / 80 | **APPROXIMATE** — vLLM's routed kernel also re-quantises to fp8, which MPK bills to task 275 |
| 253 `LINEAR_SM100` (bf16) | `nvjet_…32x64_64x16_2x1_splitK` (in_proj_ba ×30) + `nvjet_…32x64_64x16_4x1_splitK` (router ×40) + `splitKreduce_kernel` + `nvjet_…192x*_TNT` (lm_head ×1) | 71 / 71 | **exact at the family level**; the committed MPK table cannot split ba / router / lm_head, so only the sum is comparable |
| 154 `RMS_NORM_HOPPER` | the residual `triton_*` / elementwise kernels | 81 / — | **NOT like-for-like** — MPK fuses most norms, RoPE and L2-norm into its GEMM / attention / recurrent tasks; task 154 is only the standalone RMSNorm |
| 259 + 258 argmax | `reduce_kernel<…ArgMaxOps…>` | 1 / 1 | **exact** |
| 101 `EMBEDDING` | `indexSelectSmallIndex` | 1 / 1 | **exact** |

Two ambiguities were resolved from evidence rather than assumed. The MoE **router gate**
`[256,2048]` and the **shared-expert gate** `[1,2048]` are both bf16 GEMMs firing 40×/step, and
their kernel names differ by batch size. They separate cleanly by **stream**: the shared-expert
gate is always on the overlapped side stream, together with the shared expert's own quantize,
GEMMs, SiLU-mul, sigmoid and multiply — exactly 8 kernels per layer at bs1. The router is always
on the main stream. The inductor `triton_*` names were resolved from compile-cache provenance,
e.g. `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_per_token_group_fp8_quant_permute_pow_rsqrt_silu_view_1`
→ `[reshape, pow, mean, add, rsqrt, mul, silu, per_token_group_fp8_quant]`, i.e. the GDN gated
RMSNorm(o)·w ⊙ silu(z) with the fp8 quantize fused in.

---

## 4. Where MPK is slower, ranked

Ranked by absolute step time recoverable, worst first. Full detail with shapes and per-call
targets in `ferret_targets.json`.

`ferret_targets.json` covers **all 15 stages**, with no stage left implicit: **9 real target
specs** (shapes, per-call baseline µs, proposed target µs, roofline reading, expected step gain)
and **6 machine-readable disposition rows** — `below-threshold` for sampling/argmax, MoE combine
and the bs16-only SiLU-mul deficit, `structural-not-kernel` for the embedding, and `mpk-ahead` for
the two stages MPK already wins. Its `coverage` block asserts that targets ∪ dispositions equals
every row of `tables/comparison_by_stage.csv` exactly once, and
`scripts/extend_ferret.py` re-checks that assertion (and every µs in the added rows, read from the
CSV rather than typed) on each regeneration.

| # | MPK task | ratio bs1 / bs8 / bs16 | gap µs/step bs1 / bs16 | character |
|---|---|---|---|---|
| 1 | 237 GDN recurrent | 7.44 / 9.10 / **10.76** | 1053 / **4516** | the only gap that grows with batch |
| 2 | 275 quantize fp8 | **8.12** / 7.10 / 7.00 | **3981** / 3638 | biggest bs1 gap; 3840 MPK tasks vs 200 vLLM launches |
| 3 | 241 MoE w13 | **9.16** / 5.34 / 4.46 | 2747 / 3886 | MPK streams ~7× the expert weight it uses at bs1 |
| 4 | 279 dense fp8 | 2.17 / 2.05 / 2.17 | 1582 / 1602 | flattest and most predictable; 160 sites |
| 5 | 242 MoE w2 | 5.66 / 3.57 / 3.49 | 1401 / 1885 | |
| 6 | 260 MoE router top-k | 3.82 / 3.19 / 2.65 | 417 / 394 | MPK concurrency 9 of 128 workers — width, not just kernel |
| 7 | 257 attention | 3.84 / 3.68 / 3.10 | 379 / 312 | MPK concurrency 2.0 — the narrowest stage |
| 8 | 253 dense bf16 + lm_head | 1.38 / 1.41 / 1.46 | 229 / 289 | lm_head is near the memory roof on both sides |
| 9 | 234 GDN conv1d | 2.15 / 2.24 / 2.22 | 103 / 116 | |
| 10 | 259+258 argmax | 4.26 / 7.86 / 8.54 | 28 / 102 | small but scales badly |
| 11 | 101 embedding | 21.9 / 9.06 / 5.97 | 52 / 42 | tiny absolute cost |
| 12 | 261 MoE combine | 1.27 / 1.20 / 1.25 | 48 / 49 | |
| 13 | 118 SiLU-mul | 0.87 / 0.99 / **1.23** | −36 / 89 | only slower at bs16; approximate mapping |

**Do not ferret these.** `norms / RoPE / glue` — MPK is **3–4× faster** and that is structural:
vLLM spends 558–680 µs/step across 273–312 separate elementwise and Triton launches for work MPK
folds into its GEMM, attention and recurrent tasks. `shared-expert gate` — parity (1.00× / 0.95×),
and again MPK does in one task what vLLM does in 3–4 kernels. `lm_head` — vLLM moves 1.017 GB of
bf16 weights in 150.7 µs = **6.75 TB/s, ~84 % of B200 HBM peak**; that is a roofline, not an
implementation gap.

---

## 5. Two mechanisms worth naming

**vLLM's dense fp8 GEMMs are latency-bound at M=1, not bandwidth-bound.** From the ordinal
analysis at bs1: `in_proj_qkvz` `[12288,2048]` (25.2 MB of weights) costs **9.4 µs**, while
`out_proj` `[2048,4096]` (8.4 MB, 3× less data) costs **10.7 µs**. Cost tracks the K-loop depth
(K=2048 → 16 blocks; K=4096 → 32 blocks), not bytes. At 8 TB/s, 25.2 MB should take 3.1 µs — so
vLLM itself is leaving ~3× on the table here, and a ferret kernel that is bandwidth-bound rather
than K-loop-bound could beat vLLM by far more than the 20–30 % bar. The shared-expert GEMMs make
this starker: `down` `[2048,512]` is 0.5 MB of weights and still costs 5.8 µs.

**Attention's context caveat is small enough to quantify.** MPK's numbers are at the AC-3 geometry
(`max_seq_length` 132) and vLLM's at 256/1024, which only matters for attention and KV traffic.
The same FMHA kernel costs **8.706 µs/call at ctx ≈ 260** and **9.425 µs/call at ctx 556–896**
(+8.3 %) — both measured here. So the 3.84× attention ratio is robust to within ~10 % at matched
context, and it is a *lower* bound on MPK's disadvantage, since vLLM is being charged the longer
context.

---

## 6. SGLang — it works, and it is a *different* baseline

**Yes.** SGLang 0.5.16 serves `Qwen/Qwen3.5-35B-A3B-FP8` on one B200, TP=1, at the pinned
256/1024 greedy workload, first try, inside the timebox. It registers
`Qwen3_5MoeForConditionalGeneration` natively, loads the fp8 checkpoint in 19.9 s (34.38 GB), and
allocates the GDN mamba cache and paged KV without help.

| | value |
|---|---|
| version | sglang 0.5.16, torch 2.11.0+cu130 (fresh venv) |
| engine boot | 447.9 s (dominated by DeepGEMM JIT pre-compile of our 5 dense shapes) |
| bs1 e2e throughput | 329.7 / 328.2 tok/s over 2 reps (1022 tokens ÷ wall, prefill included) |
| profiled decode step | 3299 µs → ≈303 tok/s under the profiler |

For reference vLLM's binding bs1 numbers are 285.5 tok/s steady-decode and 284.4 tok/s e2e, so
**SGLang is roughly 15 % faster than vLLM at bs1**. That number is *not binding*: it is 2 reps in
one boot, e2e rather than decode-window, and the GPU had an idle co-tenant context (~1 GB, 0 %
util) at the time. Take it as a signal, not a baseline.

The interesting part is that SGLang is **not the same kernel identity**, so it is a second
independent opinion on several of our targets rather than a replication:

| component | vLLM 0.25.1 | SGLang 0.5.16 |
|---|---|---|
| dense fp8 | CUTLASS block-scale (`cutlass_3x_gemm_fp8_blockwise`) — vLLM **auto-disables DeepGEMM** for `qwen3_5_moe_text` on Blackwell, citing E8M0 accuracy degradation | **DeepGEMM** (`deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl`), JIT-compiled for exactly our 5 shapes |
| MoE | FlashInfer TRT-LLM fp8 block-scale | **same** (`moe_runner_backend=flashinfer_trtllm`) |
| GDN | Triton `fused_recurrent_gated_delta_rule_packed_decode_kernel` | **same kernel** |
| full attention | FlashInfer trtllm-gen FMHA (`fmhaSm100fKernel_…H256PagedKv…`) | Triton flash-decoding (`_fwd_grouped_kernel_stage1` + `_fwd_kernel_stage2`) |
| norms / gating | inductor-generated `triton_*` | hand-fused CUDA (`FusedAddRMSNormKernel`, `_fused_gate_sigmoid_mul_add_kernel`, `_fused_qk_rmsnorm_rope_gate_kernel`, `fused_qkvzba_split_reshape_cat_contiguous_kernel`) |

One profiled window of 49 consecutive decode steps at bs1, same anchor method, same 160/40/30/10/1
site structure:

| vLLM µs/step | SGLang µs/step | SGL/vLLM | stage |
|---:|---:|---:|---|
| 1353.5 | 1178.9 | **0.87** | dense fp8, 160 sites (DeepGEMM 7.37 µs/site vs CUTLASS 8.46) |
| 559.3 | 381.1 | **0.68** | quantize fp8 (`_v2_kernel`, 1.90 µs/call vs 2.70) |
| 336.6 | 351.5 | 1.04 | MoE w13 |
| 300.7 | 227.3 | **0.76** | MoE w2 (`_rM_TN_transOut_noShfl_` variant vs `_rM_BN_transOut_`) |
| 201.9 | 146.1 | **0.72** | MoE activation / SiLU + requant |
| 163.6 | 164.9 | 1.01 | GDN recurrent (identical kernel, identical time) |
| 89.6 | 105.3 | 1.18 | GDN conv1d |
| 94.2 | 139.0 | **1.47** | full attention (Triton flash-decode is slower than trtllm-gen FMHA) |
| 150.7 | 149.3 | 0.99 | lm_head |
| 5173.8 | 4566.6 | **0.88** | all kernels (GPU busy) |

Three things this buys us:

1. **The GDN recurrent kernel is the same code running at the same speed in both engines**
   (163.6 vs 164.9 µs/step, 5.50 µs/call). So MPK's 7.4–10.8× gap there is measured against a
   genuinely stable reference, not a vLLM quirk.
2. **The dense fp8 and quantize targets have 13–32 % of slack that a second production engine
   already collects.** A ferret kernel does not have to invent that headroom; DeepGEMM and
   SGLang's `_v2` quantize kernel demonstrate it. Note the accuracy caveat: vLLM refuses DeepGEMM
   on this architecture on E8M0 grounds, so a DeepGEMM-shaped win would need AC-3 to confirm it.
3. **vLLM's FlashInfer trtllm-gen FMHA is the right attention reference** — it beats SGLang's
   Triton flash-decoding by 32 %, so keeping vLLM as the attention baseline is correct.

Artifacts: `sglang/probe_bs1.json`, `sglang/sglang_bs1_kernels.csv`. Trace (1.7 MB gz) on the box
at `~/mpk-qwen35/m3i10-profile/sglang/traces/`.

---

## 7. Kernel detail on the top offenders

**Nsight Compute does not work on catalyst-B200.** It fails with
`Failed to prepare kernel for profiling / Unknown error on device 0` on *every* kernel including a
three-line `torch.randn` control, on all three installed paths (all the same Nsight 2026.1.0.0
build), with profiling permissions open (`RmProfilingAdminOnly: 0`) and an exclusively free GPU.
Full diagnosis in `ncu/NCU_UNAVAILABLE.md`. Separately, NCU could never have decomposed MPK
anyway — it is one persistent kernel, so per-task detail can only come from the committed perfetto
tables.

The substitute is an analytic memory roofline (`ncu/roofline.csv`, `scripts/roofline.py`): bytes
moved per step from the exact shapes in `vllm-graph.md` §3.3/§4.1/§4.2, divided by the measured
median µs from this issue's tables, against B200 HBM3e at 8 TB/s. Its calibration check is
`lm_head` — a kernel we know must be at the roof — which lands at 84 % of peak.

### bs1

| MB/step | µs/step | TB/s | % of roof | roofline µs | × off roof | kernel / stage |
|---:|---:|---:|---:|---:|---:|---|
| 970 | 150.7 | 6.75 | **84.3** | 127.1 | 1.2 | lm_head `[248320,2048]` bf16 — *the calibration* |
| 640 | 336.6 | 1.99 | 24.9 | 83.9 | 4.0 | MoE routed w13 |
| 720 | 283.5 | 2.66 | 33.3 | 94.4 | 3.0 | in_proj_qkvz `[12288,2048]` ×30 |
| 320 | 300.7 | 1.12 | 13.9 | 41.9 | 7.2 | MoE routed w2 |
| 1340 | 1353.5 | 1.04 | 13.0 | 175.6 | **7.7** | dense fp8, all 160 sites |
| 240 | 322.5 | 0.78 | 9.8 | 31.5 | 10.2 | gdn out_proj `[2048,4096]` ×30 |
| 120 | 163.7 | 0.77 | 9.6 | 15.7 | 10.4 | GDN recurrent state rd+wr |
| 80 | 312.0 | 0.27 | 3.4 | 10.5 | **29.8** | shared gate_up `[1024,2048]` ×40 |
| 40 | 232.0 | 0.18 | 2.3 | 5.2 | **44.2** | shared down `[2048,512]` ×40 |
| 2.8 | 89.6 | 0.03 | 0.4 | 0.4 | 243 | GDN conv1d state rd+wr |
| 1.0 | 559.3 | ~0 | ~0 | 0.1 | **4284** | quantize / fp8 casts, 200 sites |

### bs16

| MB/step | µs/step | TB/s | % of roof | × off roof | kernel / stage |
|---:|---:|---:|---:|---:|---|
| 970 | 155.1 | 6.56 | 82.0 | 1.2 | lm_head |
| 1920 | 462.7 | 4.35 | **54.4** | 1.8 | GDN recurrent state rd+wr |
| 1340 | 1371.0 | 1.02 | 12.8 | 7.8 | dense fp8, 160 sites |
| 640 | 1123.2 | 0.60 | 7.5 | 13.4 | MoE routed w13 |
| 320 | 755.9 | 0.44 | 5.5 | 18.0 | MoE routed w2 |
| 15.9 | 606.2 | 0.03 | 0.3 | 290 | quantize / fp8 casts |

### What this changes about the targets

**GDN recurrent is the one target where MPK is losing to physics, not to an implementation.** At
bs16 vLLM's Triton kernel runs at **54 % of the HBM roof** — genuinely bandwidth-bound, only 1.8×
off the wall — and MPK is 10.76× slower than that, i.e. **about 5 % of roof**. There is no
"vLLM is leaving room" story here: MPK has to move 1.92 GB of fp32 state per step and is not
moving it at anything like memory speed. This is the highest-confidence, highest-value ferret task
in the list, and the target is a bandwidth target, not a 20–30 % target.

**Quantize should not be a kernel target at all.** At bs1 the entire 200-site quantize stage moves
**about 1 MB**, and vLLM spends 559 µs on it — 4284× off roofline. It is pure launch overhead;
MPK's 4540 µs across 3840 tasks is the same disease, worse. Ferreting a faster quantize kernel
recovers almost nothing. The mechanism is **fusion** — fold the quantize into the producing task,
which is M3-I1 backlog rank 1 — plus not computing `mbt=16` rows when only 1 token is live.

**The dense fp8 GEMMs have far more than 20–30 % in them.** vLLM's whole 160-site dense stage runs
at 13 % of roof. The shared-expert GEMMs are the extreme: `down [2048,512]` moves 1 MB of weights
per site and takes 5.8 µs — **44× off roofline**. The ordinal analysis says why: cost tracks
K-loop depth, not bytes (K=4096 `out_proj` at 8.4 MB costs 10.7 µs while K=2048 `qkvz` at 25.2 MB
costs 9.4 µs). A ferret kernel that is actually bandwidth-bound at M≤16 could beat vLLM by 3–7×
here, not 1.3×. SGLang's DeepGEMM already takes 13 % of it with a different algorithm.

**lm_head is closed.** 84 % of roof on both engines. Do not spend a ferret run on it.

---

## 8. Caveats

1. **The MPK column is stale in two ways, and a re-measure is specified but not yet run.** It is
   M3-I1's capture at the AC-3 geometry (24–68 input tokens, `max_seq_length` 132) **and** it
   predates both the M3-I8 MoE router gate — now `MOE_GATE_PADDING_ROWS = True`, default-ON at
   HEAD since `96eff01` — and the M3-I2b quantize/width fixes. **`remeasure_spec.md`** is the run
   matrix that closes it: arm A at matched 256/1024 geometry plus arm B at M3-I1's exact geometry
   (so a moved number can be attributed to the code rather than to the geometry), bs {1, 8, 16},
   3 profiled + 3 unprofiled reps, **~75–90 GPU-minutes** for the required tier. It also lists what
   the running M3-I9 window's traces would have to contain to be harvested instead of scheduling
   new time.

   **Robust to the re-measure** — these should not move: the GDN-recurrent growth ratio
   7.44 → 9.10 → 10.76×, since nothing in I8 or I2b touches task 237 and SGLang independently runs
   the identical kernel at 164.9 vs vLLM's 163.6 µs/step; the dense-fp8 flat ratio ~2.1× at every
   batch size across 160 untouched call sites; lm_head at 84 % of the HBM roof (arithmetic from
   shapes plus a vLLM-side measurement, no MPK term at all); norms/RoPE/glue as an MPK win.

   **May reshuffle:** the quantize rank (M3-I2b targeted exactly that stage and found 93.75 % of
   its work redundant), the MoE w13/w2 ranks (the I8 gate cuts per-layer `moe_w13` wall span
   76.8 → 34.8 µs at bs1, and much less at bs16), the shared-expert-gate bs8 point, and every
   absolute `step_gain_if_met_us` — and with them the overall 4.28× / 3.86× / 4.10×, since the MPK
   step denominator moves too. The *mechanism* conclusions survive either way, because they are
   statements about vLLM's side and about physics: quantize moving ~1 MB at 4284× off roofline
   stays a fusion/width problem rather than a kernel problem whatever MPK's number becomes.
2. **Only attention is context-sensitive.** §5 bounds it at +8.3 % from ctx ≈ 260 to ctx 556–896.
   Every other stage depends on token count and batch size, not context.
3. **MPK's dense and MoE stages already do bs16-worth of work at bs1** (`max_num_batched_tokens`
   = 16). That is a property of the current MPK scheduler, not of the kernels, and it is M3-I1
   backlog items 2 and 4. The per-kernel ratios at bs1 therefore mix a kernel-quality gap with a
   padding gap; at bs16 they are closer to pure kernel quality.
4. **The two sides use different overlap conventions.** vLLM per-stage numbers are sums of kernel
   durations (== union within every stage except the shared expert); MPK numbers are unions across
   128 workers. Neither sums to the step: vLLM's stage sums are 145 % of its step (two streams plus
   graph-level concurrency), MPK's are 109–114 % of its step.
5. **The SiLU-mul row is approximate** — vLLM's `activationDeepSeekKernel` also re-quantises the
   intermediate to fp8; MPK bills that to task 275. The row flatters vLLM's 118-equivalent and
   correspondingly penalises MPK's 275.
6. **Norms/RoPE is not like-for-like** and is reported only to show where MPK's fusion advantage
   already lives.
7. **The profiler slows the host, not the kernels.** Profiled decode ran at 155–2006 tok/s vs
   284–2988 unprofiled, but that is launch-gap inflation: kernel durations are CUPTI hardware
   timestamps, and the union-of-intervals closure against the unprofiled step (§2) is 0.2–5.3 %.

---

## 9. Artifacts

In this directory:

| path | what |
|---|---|
| `tables/comparison_by_stage.csv` | the §3 table, all three batch sizes, with per-layer columns |
| `tables/bs{1,8,16}_kernels.csv` | every kernel: calls/step, µs/step median + min/max over 6 windows, µs/call, % of GPU busy, stream |
| `tables/bs{1,8,16}_stages.csv` | per-stage sums and unions |
| `tables/prefill_bs{1,8,16}_kernels.csv` | short-context window (ctx ≈ 260), used for the attention context bound |
| `tables/ordinal_bs1_cutlass.json` | per-GEMM-site cost of all 160 CUTLASS fp8 calls |
| `tables/profile_meta.json` | engine assertions, versions, per-rep throughput, GPU clocks, co-tenant checks |
| `ferret_targets.json` | the dispatch list, total coverage of all 15 stages: 9 target specs (kernel, shapes, baseline µs, target µs, roofline, expected step gain) + 6 disposition rows + a `coverage` assertion |
| `remeasure_spec.md` | the GPU window needed to regenerate the MPK column at matched geometry on current HEAD: run matrix, capture invocation, normalisation, analysis commands, M3-I9 harvest checklist, ~75–90 GPU-min |
| `ncu/NCU_UNAVAILABLE.md` | why Nsight Compute could not run on this box, and what was tried |
| `ncu/roofline.csv`, `ncu/roofline.json` | the analytic memory-roofline substitute for NCU's SOL section |
| `sglang/probe_bs1.json` | SGLang feasibility result: versions, boot time, throughput reps |
| `sglang/sglang_bs1_kernels.csv` | SGLang's per-kernel decode table at bs1, same anchor method |
| `scripts/` | the capture and analysis pipeline |

On catalyst-B200 under `~/mpk-qwen35/m3i10-profile/` (too large for the repo):

| path | size |
|---|---|
| `traces/main/decode_bs{1,8,16}_win{0..5}.json` | 18 chrome traces, 57–62 MB each, **1.1 GB total** |
| `traces/main/prefill_bs{1,8,16}_win0.json` | 3 traces, 10–13 MB each |
| `logs/main.log` | the capture log, including the engine assertion line |
| `sglang/` | the SGLang venv and probe output |

### Reproduce

```bash
# on catalyst-B200, with an exclusively free GPU <id>
cd ~/mpk-qwen35/m3i10-profile
bash scripts/run_profile.sh <id> main --batch-sizes 1,16,8 --output-len 1024 \
     --skip-first 300 --wait 60 --warmup 3 --active 50 --repeat 3 \
     --profiled-gens 2 --timed-reps 3 --prefill-trace
bash scripts/run_analysis.sh
bash scripts/run_stages.sh
# then, in the repo
python3 opt/m3i10/scripts/build_comparison.py
```
