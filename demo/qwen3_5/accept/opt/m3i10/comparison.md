# M3-I10 — vLLM vs MPK, per kernel

What each decode step costs in vLLM 0.25.1 at our exact workload and shapes, mapped stage by
stage onto MPK task families, so the remaining gap can be spent as per-kernel optimization
targets instead of guesses.

**STATUS (2026-07-27): the MPK column has been RE-MEASURED at matched geometry on current HEAD.**
This document's main body is the CURRENT correspondence — matched-geometry MPK (`msl=353` =
256-token prompt + 96 decode steps + 1, gate_padding_rows ON, post-M3-I2b), with attention's
own primary basis further corrected to a late-context closure capture (`msl=897`, ctx ≈801–896,
matching the vLLM reference table's own 556–896 sampling band — see §5). The original M3-I1-era
capture (AC-3 geometry, pre-I8/I2b) that this document originally reported is preserved verbatim,
for provenance, in **§10 Historical appendix** — nothing below silently overwrites it, but nothing
in §§1–9 should be read as that old capture either. Full methodology: `remeasure_spec.md`. Full
regenerated data + every script used: `remeasure/`. Ranking + per-target detail: `ferret_targets.json`
(schema 2.0).

**Headline.** MPK is slower than the corresponding vLLM kernel in **11 of 15 stages at every batch
size** (12 at bs1, 11 at bs8, 11 at bs16; 12 at at least one batch size) — down from the old
capture's 12/13/13 (14 at at least one), because **quantize has been resolved** (see below) and
dropped out of the slower set almost entirely. Of the 15 stages: **8 are live ferret targets**,
**1 is resolved** (quantize, by M3-I2b — no target left), **2 are below-threshold** (absolute gap
too small to be worth a dispatch), **1 is structural** (embedding — a graph-width problem, not a
kernel problem), and **3 are MPK-ahead** (norms/RoPE/glue, shared-expert gate, MoE/shared SiLU-mul
— MPK wins all three now, up from one in the old capture). The gap is spread 1.3×–10.7× across
the MoE GEMMs, the dense fp8 GEMMs, GDN recurrent, and attention. The two gaps that *grow* with
batch size are GDN recurrent (7.4× → 10.7×) and attention (8.1× → 10.1×, at its corrected
late-context basis).

| | bs1 | bs8 | bs16 |
|---|---:|---:|---:|
| vLLM decode step (this profile, union of GPU intervals) | 3567 µs | 4818 µs | 5363 µs |
| vLLM decode step (from the binding tok/s baseline) | 3502 µs | 4727 µs | 5301 µs |
| MPK decode step (matched geometry, current HEAD) | 10964 µs | 12255 µs | 14715 µs |
| ratio | **3.07×** | **2.54×** | **2.74×** |

Down from the old capture's 4.28×/3.86×/4.10× (§10) — **not because MPK got proportionally
faster everywhere** (most per-stage ratios are similar or worse; attention in particular is worse
at its corrected basis), but because **quantize's collapse (4540→560 µs/step at bs1, an 87.6 %
reduction) shrinks the MPK step denominator by more than everything else combined moved it**. See
§4/§8 for the per-stage code-delta vs. geometry-delta attribution that separates "the I8/I2b code
landed" from "the geometry changed."

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
attention, KV block 1056, GDN state fp32. No silent backend fallback. **This section is about the
vLLM side and is unaffected by the MPK re-measure** — vLLM was not re-captured in M3-I10 closure;
only the MPK column moved.

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
10 full-attention, MoE on all 40). **The MPK side transplanted this exact QC method** for its own
re-measure — anchor = `TASK_BEGIN_TASK_GRAPH`, same integer-count assertion — and it is
independently clean there too (`max_frac_err = 0.0000` at every batch size; `remeasure_spec.md` §5,
`ferret_targets.json`'s generation log).

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

**The MPK side HAS been re-measured** (superseding the sentence this section originally had).
Matched-geometry MPK numbers are `opt/m3i10/remeasure/armA/pertask_by_bs.csv`'s wall spans at
`max_seq_length=353` (256-token synthetic prompt + 96 decode steps + 1), current HEAD
(`gate_padding_rows=True`, post-M3-I2b) — the union of the time a task family is executing inside
the persistent kernel, same convention M3-I1 established (sums to 109–114 % of the step, still the
fair per-stage estimate). Attention's own primary number is a further, dedicated late-context
capture (`msl=897`, §5). NCU still cannot decompose MPK (one persistent kernel); per-task detail
still only comes from the profiler-buffer tables, now regenerated rather than the M3-I1 ones.

---

## 3. Correspondence table (matched geometry, current HEAD)

`sites` is how many model sites the stage covers per step (both implementations run the same
model, so this is the same on both sides). vLLM = sum of that stage's kernel durations per step
(unchanged from the original capture — not re-measured); MPK = that task family's wall span per
step, regenerated at matched geometry (`remeasure/armA_m3i10/tables/comparison_by_stage.csv`).
**The `full attention` row here is the matched-geometry number (ctx 257–352); `ferret_targets.json`
uses a further-corrected late-context number (ctx ≈801–896) as ITS primary basis for that stage —
see §5 and the note under the table.** Ranked tightly against `ferret_targets.json` schema 2.0;
§4 gives the ranked/dispositioned view.

### bs1

| vLLM µs/step | MPK µs/step | ratio | gap µs | vLLM µs/layer | MPK µs/layer | stage |
|---:|---:|---:|---:|---:|---:|---|
| 336.6 | 2388.0 | **7.10×** | 2051 | 8.41 | 59.70 | MoE routed GEMM w13 |
| 1353.5 | 2933.0 | **2.17×** | 1580 | 33.84 | 73.33 | dense projections (fp8 blockscale) |
| 163.7 | 1218.0 | **7.44×** | 1054 | 5.46 | 40.60 | GDN recurrent |
| 300.7 | 1347.0 | **4.48×** | 1046 | 7.52 | 33.68 | MoE routed GEMM w2 |
| 133.6 | 757.0 (matched) / **1080.5 (late-ctx, primary)** | 5.67× (matched) / **8.09× (primary)** | 623 / **980** | 13.36 | 75.70 / 108.05 | full attention (incl. KV write) |
| 147.9 | 497.0 | **3.36×** | 349 | 3.70 | 12.43 | MoE router top-k/softmax |
| 609.3 | 788.0 | **1.29×** | 179 | 8.58 | 11.10 | dense bf16 small + lm_head |
| 174.1 | 282.0 | **1.62×** | 108 | 4.35 | 7.05 | MoE combine |
| 89.6 | 185.0 | **2.06×** | 95 | 2.99 | 6.17 | GDN conv1d |
| 2.5 | 65.0 | **26.3×** | 62 | 2.47 | 65.00 | embedding |
| 8.7 | 39.0 | **4.49×** | 30 | 8.68 | 39.00 | sampling / argmax |
| 559.3 | 560.0 | 1.00× | 1 | 8.58 | 8.58 | quantize / fp8 casts — **resolved by I2b, no longer a target (was 8.12×/3981 µs)** |
| 273.6 | 238.0 | 0.87× | −36 | 3.42 | 2.98 | MoE / shared SiLU-mul |
| 429.6 | 293.0 | **0.68×** | −137 | 10.74 | 7.33 | shared-expert gate |
| 557.8 | 184.0 | **0.33×** | −374 | 6.89 | 2.27 | norms / RoPE / glue |

### bs16

| vLLM µs/step | MPK µs/step | ratio | gap µs | vLLM µs/layer | MPK µs/layer | stage |
|---:|---:|---:|---:|---:|---:|---|
| 462.7 | 4938.0 | **10.67×** | 4475 | 15.43 | 164.60 | GDN recurrent |
| 1371.0 | 3005.0 | **2.19×** | 1634 | 34.27 | 75.12 | dense projections (fp8 blockscale) |
| 1123.2 | 2505.0 | **2.23×** | 1382 | 28.08 | 62.62 | MoE routed GEMM w13 |
| 755.9 | 1454.0 | **1.92×** | 698 | 18.90 | 36.35 | MoE routed GEMM w2 |
| 148.7 | 787.0 (matched) / **1509.3 (late-ctx, primary, low-confidence — see §5)** | 5.29× (matched) / **10.15× (primary)** | 638 / **1398** | 14.87 | 78.70 / 150.93 | full attention (incl. KV write) |
| 238.2 | 604.0 | **2.54×** | 366 | 5.96 | 15.10 | MoE router top-k/softmax |
| 632.2 | 905.0 | **1.43×** | 273 | 8.90 | 12.75 | dense bf16 small + lm_head |
| 95.6 | 207.0 | **2.17×** | 111 | 3.19 | 6.90 | GDN conv1d |
| 194.3 | 294.0 | **1.51×** | 100 | 4.86 | 7.35 | MoE combine |
| 13.5 | 58.0 | **4.31×** | 45 | 13.47 | 58.00 | sampling / argmax |
| 8.4 | 52.0 | **6.21×** | 44 | 8.37 | 52.00 | embedding |
| 606.2 | 571.0 | 0.94× | −35 | 3.03 | 2.85 | quantize / fp8 casts — **resolved by I2b (was 7.00×/3638 µs)** |
| 390.2 | 301.0 | 0.77× | −89 | 4.88 | 3.76 | MoE/shared SiLU-mul |
| 455.1 | 363.0 | **0.80×** | −92 | 11.38 | 9.07 | shared-expert gate |
| 680.4 | 203.0 | **0.30×** | −477 | 8.40 | 2.51 | norms / RoPE / glue |

bs8 is in `remeasure/armA_m3i10/tables/comparison_by_stage.csv` along with per-layer columns for
all three batch sizes and the matched-geometry `full attention` figure that table itself carries
as primary (the late-context correction lives in `ferret_targets.json`, not in that CSV).

**Code-delta vs. geometry-delta.** Because arm B (continuity, `msl=132`, current HEAD) and the
original M3-I1 capture (`msl=132`, pre-I8/I2b) share the same geometry, and arm A/A-late share the
same code as arm B, every stage's move factors cleanly into "what the I8 gate / I2b landed" vs.
"what the longer/later context changed" (wall span µs/step, bs1):

| task | M3-I1 (pre-I8/I2b, msl132) | arm B (code delta, msl132) | arm A (+ geometry, msl353) | arm A-late (+ further context, msl897) |
|---|---:|---:|---:|---:|
| quantize (275) | 4540.0 | 562.0 | 560.0 | — |
| MoE w13 (241) | 3084.0 | 2381.0 | 2388.0 | — |
| MoE w2 (242) | 1702.0 | 1345.0 | 1347.0 | — |
| GDN recurrent (237) | 1217.0 | 1219.0 | 1218.0 | 1221.0 (flat, +0.2%) |
| dense fp8 (279) | 2936.0 | 2936.0 | 2933.0 | 2939.2 (flat, +0.2%) |
| attention (257) | 513.0 | 510.0 | 757.0 | **1080.5 (+42.7% vs matched)** |
| shared-expert gate (238) | 428.0 | 295.0 | 293.0 | — |

Quantize's collapse is ~100 % code delta (4540→562, geometry then adds nothing: 562→560).
GDN recurrent and dense-fp8 are flat under BOTH deltas — genuinely robust, confirmed at the
late-context capture too. Attention is flat under the code delta (513→510, I8/I2b do not touch
it) and the ENTIRE ratio increase is geometry: 510→757 (matched) →1080.5 (late-context) — see §5.
The shared-expert-gate bs8 anomaly (428/**596**/432 in the old capture) is resolved mostly by the
code delta alone (428/**596**/432 → 295/**373**/379) with geometry finishing the smoothing to
293/360/363 — confirmed capture artifact, not a real bs8 regression (§4, §8).

### What maps to what

| MPK task family | vLLM kernel(s) | sites/step | mapping confidence |
|---|---|---|---|
| 279 `LINEAR_FP8_BLOCKSCALE_SM100` | `cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<…>>` | 160 / 160 | **exact** — same 6 GEMM shapes, same site counts |
| 241 `MOE_W13_FP8_BLOCKSCALE_SM100` | `bmm_E4m3_E4m3E4m3_Fp32_…_sm100f` (TRT-LLM MoE gemm1) | 40 / 40 | **exact** |
| 242 `MOE_W2_FP8_BLOCKSCALE_SM100` | `bmm_Bfloat16_E4m3E4m3_Fp32_…_sm100f` (gemm2) | 40 / 40 | **exact** |
| 275 `QUANTIZE_FP8_SM100` | `per_token_group_quant_8bit_kernel<BFloat16, Float8_e4m3fn, …>` ×2 variants | 200 / 200 | **exact** — 160 dense + 40 MoE (column-major scales) |
| 234 `GDN_CONV1D_SM100` | `_causal_conv1d_update_kernel` | 30 / 30 | **exact** |
| 237 `GDN_RECURRENT_SM100` | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 30 / 30 | **exact** |
| 257 `ATTN_SM100` | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausal…ForGen` + `reshape_and_cache_flash_kernel` | 10 / 10 | exact kernels; **context now matched via the late-context capture, see §5** |
| 260 `MOE_TOPK_SOFTMAX_SM100` | `routingIndicesBlockKernel` (bs1) / `routingIndicesDynBlockKernel` (bs8/16) | 40 / 40 | **exact** |
| 261 `MOE_MUL_SUM_ADD_SM100` | `moe::dev::finalize::finalizeKernel` | 40 / 40 | **exact** |
| 238 `SIGMOID_GATE_MUL_ADD_SM100` | shared-expert gate GEMM + `sigmoid_kernel_cuda` + `BinaryFunctor` mul (+ splitK reduce at bs8/16) | 40 / 40 | **exact stage**, 1 MPK task vs 3–4 vLLM kernels |
| 118 `SILU_MUL` | `activationDeepSeekKernel` (routed) + `triton_poi_fused_mul_silu_slice_0` (shared) | 80 / 80 | **APPROXIMATE** — vLLM's routed kernel also re-quantises to fp8, which MPK bills to task 275 |
| 253 `LINEAR_SM100` (bf16) | `nvjet_…32x64_64x16_2x1_splitK` (in_proj_ba ×30) + `nvjet_…32x64_64x16_4x1_splitK` (router ×40) + `splitKreduce_kernel` + `nvjet_…192x*_TNT` (lm_head ×1) | 71 / 71 | **exact at the family level; NOW ALSO split per call site** — lm_head 1.17× (parity, skip), MoE router-gate 2.28×, in_proj_ba 2.79× (the real targets) — `ferret_targets.json`'s rank-9-era `held_status_resolution` |
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

## 4. Where MPK is slower, ranked (schema 2.0, matched geometry + late-context attention)

Ranked by absolute step time recoverable at bs1, worst first (`ferret_targets.json`'s
`basis.rank_rule`; the CSV this generates from carries no rank column, so this file is the one
place an ordering choice gets made — stated so it is reproducible). Full detail with shapes,
per-call targets, and the code-delta/geometry-delta isolation: `ferret_targets.json`.

`ferret_targets.json` (schema 2.0, `generator: opt/m3i10/scripts/regenerate_ferret_v2.py`,
machine-derived + coverage-asserted, idempotent) covers **all 15 stages** with no stage left
implicit: **8 real target specs** and **7 disposition rows** — `resolved-by-I2b` (quantize, new
category — collapsed, no target left), `below-threshold` (sampling/argmax, MoE combine),
`structural-not-kernel` (embedding), and `mpk-ahead` (norms/RoPE/glue, shared-expert gate,
**and now MoE/shared SiLU-mul**, up from one mpk-ahead stage in the original capture). Every
row's ORIGINAL M3-I1-era numbers are preserved in that row's own `history_m3i1` sub-object —
nothing is silently overwritten. `coverage` asserts targets ∪ dispositions equals every row of
`remeasure/armA_m3i10/tables/comparison_by_stage.csv` exactly once, checked by the generator at
every run (verified idempotent: two consecutive runs produce byte-identical output).

| rank | MPK task | ratio bs1 / bs8 / bs16 | gap µs/step bs1 / bs8 / bs16 | character |
|---|---|---|---|---|
| 1 | 241 MoE w13 | 7.10 / 2.82 / 2.23 | **2136** / 1818 / 1663 | I8 gate lands a big code-delta cut (was 9.16/5.34/4.46); still the biggest bs1 gap |
| 2 | 279 dense fp8 | 2.17 / 2.09 / 2.19 | 1918 / 1919 / 1977 | flattest and most predictable; 160 sites; **robust to the re-measure exactly as predicted** |
| 3 | 242 MoE w2 | 4.48 / 2.08 / 1.92 | 1122 / 929 / 887 | tracks MoE w13's I8-gate improvement |
| 4 | 237 GDN recurrent | 7.44 / 9.12 / **10.67** | 1095 / 2264 / **4591** | the only gap besides attention that grows with batch; wall span flat under both code- and geometry-delta — **robust exactly as predicted**; dominant at bs16 by a wide margin |
| 5 | 257 attention (late-ctx primary) | **8.09** / 9.16 / 10.15 | 980 / 1121 / 1398 | **NOT robust — the re-measure's biggest surprise.** Flat under code-delta; the entire ratio increase (5.67→8.09 at bs1) is the corrected late-context basis, itself larger than the old +8.3% single-kernel estimate. bs16 point carries a real caveat (see §5) |
| 6 | 260 MoE router top-k | 3.36 / 3.06 / 2.54 | 386 / 425 / 425 | modest code-delta improvement |
| 7 | 253 dense bf16 + lm_head | 1.29 / 1.40 / 1.43 | 331 / 409 / 431 | **HELD status now resolved**: per-call-site split shows lm_head at parity (1.17×, both sides roofline-bound, skip) — the target is the router-gate (2.28×) and in_proj_ba (2.79×) sub-sites only |
| 8 | 234 GDN conv1d | 2.06 / 2.19 / 2.17 | 118 / 132 / 135 | unchanged in character |

**Resolved — no longer a target.** `quantize / fp8 casts`: ratio 8.12/7.10/7.00 → **1.00/0.93/0.94**,
gap 3981/3785/3638 µs → **0.7/−44.5/−35.2 µs**. Code-delta isolation (arm B vs. the original M3-I1
capture, same `msl=132` geometry) attributes essentially 100 % of the collapse to M3-I2b
(4540→562 µs/step at bs1); geometry then adds nothing further (562→560). This is the M3-I2b
"93.75 % of that stage's work was redundant" prediction landing almost exactly.

**Do not ferret these — MPK already wins, or the gap is not worth it.** `norms / RoPE / glue` —
MPK is **3–4× faster**, structural (vLLM's 558–680 µs/step across 273–312 separate launches, MPK
folds it into GEMM/attention/recurrent tasks). `shared-expert gate` — **now ahead at every batch
size** (0.68×/0.81×/0.80×; the bs8 point that was 1.34× in the old capture is a **confirmed
capture artifact**, resolved mostly by the I8 code-delta and finished by the geometry-delta — see
§3's code-delta table). `MoE/shared SiLU-mul` — **newly ahead at every batch size** (0.87×/0.73×/
0.77×; was slower only at bs16 in the old capture, 1.23×, which flips too at the corrected
geometry). `lm_head` — parity (1.17× at the per-call-site split), vLLM moves 1.017 GB of bf16
weights in 150.7 µs = **6.75 TB/s, ~84 % of B200 HBM peak**; that is a roofline, not an
implementation gap, on either side. `sampling / argmax`, `MoE combine` — real but small absolute
gaps (below-threshold). `embedding` — structural graph-width problem, not a kernel problem.

---

## 5. Two mechanisms worth naming

**vLLM's dense fp8 GEMMs are latency-bound at M=1, not bandwidth-bound.** From the ordinal
analysis at bs1: `in_proj_qkvz` `[12288,2048]` (25.2 MB of weights) costs **9.4 µs**, while
`out_proj` `[2048,4096]` (8.4 MB, 3× less data) costs **10.7 µs**. Cost tracks the K-loop depth
(K=2048 → 16 blocks; K=4096 → 32 blocks), not bytes. At 8 TB/s, 25.2 MB should take 3.1 µs — so
vLLM itself is leaving ~3× on the table here, and a ferret kernel that is bandwidth-bound rather
than K-loop-bound could beat vLLM by far more than the 20–30 % bar. The shared-expert GEMMs make
this starker: `down` `[2048,512]` is 0.5 MB of weights and still costs 5.8 µs. **Robust to the
re-measure** — this is a statement about vLLM's own kernel behavior and physics, and dense-fp8's
wall span on the MPK side is confirmed flat under both code- and geometry-delta (§3).

**Attention's context sensitivity was UNDERESTIMATED by the original +8.3 % single-kernel
correction — this is the re-measure's one real surprise, now closed with a dedicated capture
rather than an extrapolation.** The original document estimated the ctx-260-vs-556–896 gap from
one FMHA kernel call (8.706 → 9.425 µs/call, +8.3 %) because MPK's own capture sat at ctx ≈257–352
and a full re-measure at vLLM's context seemed to need 16× the profiler events. It doesn't: a
closure capture (`remeasure_spec.md` §4(d), `remeasure/armAlate/`) runs the wave to `msl=897`
(256-token prompt + 640 decode steps) and takes only the FINAL 96 iterations — landing at MPK's
own context ≈801–896, inside the vLLM reference table's sampled band, at a fraction of the event
cost of a full 1024-step capture. Result: MPK's attention wallspan (µs/step) grows **+42.7 % (bs1),
+51.9 % (bs8), +91.8 % (bs16)** moving from ctx 257–352 to ctx ≈801–896 — 5–10× the old
single-kernel correction. bs1/bs8 are clean measurements (bs1 = one request; bs8 = a full
8-concurrent `decode_full` window). **bs16 carries a real caveat**: MPK's admission is
slot-staggered, so no full-bs16, prefill-free window exists at ANY context (the same structural
reason the original capture already noted for bs16's steady state) — the 12 concurrent survivors
at the chosen bs16 snapshot span per-slot context 263–890, not a tight band, so treat that point as
directionally right, not as precise as bs1/bs8. This is now `ferret_targets.json`'s `full
attention` row's PRIMARY basis (`context_band` / `matched_window` sub-objects document both the
corrected number and the demoted matched-geometry one side by side).

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
util) at the time. Take it as a signal, not a baseline. **Unaffected by the MPK re-measure** — this
section compares two vLLM-family engines, no MPK term.

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
   (163.6 vs 164.9 µs/step, 5.50 µs/call). So MPK's 7.4–10.7× gap there is measured against a
   genuinely stable reference, not a vLLM quirk — and the re-measure confirms MPK's own side is
   just as stable (§3, §4).
2. **The dense fp8 and quantize targets have 13–32 % of slack that a second production engine
   already collects.** A ferret kernel does not have to invent that headroom; DeepGEMM and
   SGLang's `_v2` quantize kernel demonstrate it. Note the accuracy caveat: vLLM refuses DeepGEMM
   on this architecture on E8M0 grounds, so a DeepGEMM-shaped win would need AC-3 to confirm it.
   (Quantize itself is resolved on the MPK side now — §4 — but this vLLM-side headroom
   observation stands regardless.)
3. **vLLM's FlashInfer trtllm-gen FMHA is the right attention reference** — it beats SGLang's
   Triton flash-decoding by 32 %, so keeping vLLM as the attention baseline is correct, including
   for the corrected late-context number in §5.

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
`lm_head` — a kernel we know must be at the roof — which lands at 84 % of peak. **This whole
section is vLLM-side shape/physics arithmetic and is unaffected by the MPK re-measure** except
where noted below.

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
| 1.0 | 559.3 | ~0 | ~0 | 0.1 | **4284** | quantize / fp8 casts, 200 sites (vLLM side — MPK side now resolved, §4) |

### bs16

| MB/step | µs/step | TB/s | % of roof | × off roof | kernel / stage |
|---:|---:|---:|---:|---:|---|
| 970 | 155.1 | 6.56 | 82.0 | 1.2 | lm_head |
| 1920 | 462.7 | 4.35 | **54.4** | 1.8 | GDN recurrent state rd+wr |
| 1340 | 1371.0 | 1.02 | 12.8 | 7.8 | dense fp8, 160 sites |
| 640 | 1123.2 | 0.60 | 7.5 | 13.4 | MoE routed w13 |
| 320 | 755.9 | 0.44 | 5.5 | 18.0 | MoE routed w2 |
| 15.9 | 606.2 | 0.03 | 0.3 | 290 | quantize / fp8 casts (vLLM side) |

### What this changes about the targets

**GDN recurrent is the one target where MPK is losing to physics, not to an implementation —
confirmed, not just still-believed, by the re-measure.** At bs16 vLLM's Triton kernel runs at
**54 % of the HBM roof** — genuinely bandwidth-bound, only 1.8× off the wall — and MPK is 10.67×
slower than that (§4), essentially unchanged from the old capture's 10.76× and flat under both the
code-delta and geometry-delta (§3). There is no "vLLM is leaving room" story here: MPK has to move
1.92 GB of fp32 state per step and is not moving it at anything like memory speed. This remains
the highest-confidence, highest-value ferret task in the list, and the target is a bandwidth
target, not a 20–30 % target.

**Quantize is no longer a kernel target — confirmed, not merely predicted.** At bs1 the entire
200-site quantize stage moves **about 1 MB**, and vLLM spends 559 µs on it — 4284× off roofline,
pure launch overhead, unaffected by the MPK re-measure (it's a statement about vLLM's own kernel).
On the MPK side, M3-I2b's fusion fix landed: wall span collapsed from 4540 to 560 µs/step at bs1
(§4), which is what the roofline argument always implied should happen — the mechanism was never
"quantize needs a faster kernel," it was "fuse it into the producing task, and stop computing
`mbt=16` rows when only 1 token is live," and that is now done. Do not dispatch a ferret run here.

**The dense fp8 GEMMs still have far more than 20–30 % in them, confirmed flat by the re-measure.**
vLLM's whole 160-site dense stage runs at 13 % of roof, and MPK's own wall span for this family is
confirmed flat under both code- and geometry-delta (§3) — this remains rank 2. The shared-expert
GEMMs are the extreme: `down [2048,512]` moves 1 MB of weights per site and takes 5.8 µs —
**44× off roofline**. The ordinal analysis says why: cost tracks K-loop depth, not bytes (K=4096
`out_proj` at 8.4 MB costs 10.7 µs while K=2048 `qkvz` at 25.2 MB costs 9.4 µs). A ferret kernel
that is actually bandwidth-bound at M≤16 could beat vLLM by 3–7× here, not 1.3×. SGLang's DeepGEMM
already takes 13 % of it with a different algorithm.

**lm_head is closed on both sides.** 84 % of roof on vLLM; the corrected per-call-site split
(§3, §4) puts MPK at 1.17× there — parity, both roofline-bound. Do not spend a ferret run on it;
the family's real target is the router-gate and in_proj_ba sub-sites (§4 rank 7).

---

## 8. Caveats

1. **The MPK column WAS stale in two ways; it has now been re-measured (M3-I10 closure,
   2026-07-27) and the outcomes below are measured, not predicted.** The old column was M3-I1's
   capture at the AC-3 geometry (24–68 input tokens, `max_seq_length` 132), predating both the
   M3-I8 MoE router gate (`MOE_GATE_PADDING_ROWS = True`, default-ON at HEAD since `96eff01`) and
   the M3-I2b quantize/width fixes. `remeasure_spec.md` is the executed methodology: arm A at
   matched geometry (`msl=353`, corrected from an initial `msl=1280` slip that would have produced
   ~1023 decode steps instead of 96 — see that file's status block and
   `remeasure/logs/ROOT_CAUSE_msl.txt`) + arm B at M3-I1's exact geometry (code-delta isolation) +
   a late-context closure capture (`msl=897`, attention only). ~61 GPU-minutes actually spent
   across all three (vs. the spec's ~75–90 estimate for tier 1 alone), including two GPU-residency
   wedges (a known, documented MPK hazard on this shared box, not a code bug — see the memory
   note) that cost ~29 of those minutes unproductively.

   **Confirmed robust, exactly as predicted:** the GDN-recurrent growth ratio, now 7.44 → 9.12 →
   10.67× (was 7.44→9.10→10.76 — essentially unchanged), flat under both code- and geometry-delta,
   consistent with SGLang independently running the identical kernel at 164.9 vs vLLM's
   163.6 µs/step; the dense-fp8 ratio, flat at ~2.1–2.2× across all three batch sizes and both
   deltas; lm_head at 84 % of the HBM roof (vLLM-side arithmetic, no MPK term); norms/RoPE/glue as
   an MPK win.

   **Reshuffled, as predicted:** quantize collapsed from rank 2 (8.12×/3981 µs at bs1) to
   `resolved-by-I2b` (1.00×/0.7 µs) — ~100 % code-delta, matching M3-I2b's own "93.75 % redundant"
   finding almost exactly. MoE w13/w2 ranks fell (9.16→7.10× and 5.66→4.48× at bs1; much larger
   moves at bs8/16, consistent with the I8 gate helping more there). The shared-expert-gate bs8
   point (1.34× in the old capture) is a **confirmed capture artifact** — new wall spans are
   293/360/363 µs, smooth and monotonic, resolved mostly by the code-delta. Every absolute
   `step_gain_if_met_us` moved, and with them the overall step ratio (now 3.07×/2.54×/2.74×, down
   from 4.28×/3.86×/4.10×, driven mostly by quantize's collapse — see the Headline).

   **NOT robust — the one real surprise:** attention. The old document's own +8.3 % single-kernel
   context correction UNDERSTATED the effect; the dedicated late-context capture measures
   +42.7 %/+51.9 %/+91.8 % (§5), moving attention's ratio from 3.84/3.68/3.10× to
   8.09/9.16/10.15× and its rank from 7 to 5. This is now flagged as its own live consideration for
   ferret prioritization, not folded quietly into a footnote.

   AC-3 non-regression was checked on arm B's captures at current HEAD: 48/57 positions byte-exact;
   the sole recurring divergence (p06-poem, position 60, all 18 profiled+unprofiled instances) is
   the pre-existing, already-adjudicated M2-era logit tie (`top1_logit == top2_logit == 21.0`
   exactly — a genuine numerical tie in the reference itself, documented in
   `opt/m3i8/results/VALIDATION.md`), not a new regression.

2. **Attention's context sensitivity is no longer estimated from one kernel call — it is measured
   directly** (§5), and it is larger than the old estimate. Every other stage still depends on
   token count and batch size, not context, and this is now empirically confirmed for GDN
   recurrent and dense-fp8 specifically (§3's code-delta table) rather than assumed.
3. **MPK's dense and MoE stages already do bs16-worth of work at bs1** (`max_num_batched_tokens`
   = 16). That is a property of the current MPK scheduler, not of the kernels, and it is M3-I1
   backlog items 2 and 4 — unaffected by the re-measure. The per-kernel ratios at bs1 therefore
   still mix a kernel-quality gap with a padding gap; at bs16 they are closer to pure kernel
   quality.
4. **The two sides use different overlap conventions.** vLLM per-stage numbers are sums of kernel
   durations (== union within every stage except the shared expert); MPK numbers are unions across
   128 workers. Neither sums to the step: vLLM's stage sums are ~145 % of its step (two streams
   plus graph-level concurrency); MPK's matched-geometry sums remain 109–114 % of its own step,
   confirmed unchanged by the re-measure's own anchor-QC.
5. **The SiLU-mul row is approximate** — vLLM's `activationDeepSeekKernel` also re-quantises the
   intermediate to fp8; MPK bills that to task 275. The row flatters vLLM's 118-equivalent and
   correspondingly penalises MPK's 275. At the re-measure this row flipped to MPK-ahead at every
   batch size (§4); read the flip as a direction, not a precise measurement, for the same reason.
6. **Norms/RoPE is not like-for-like** and is reported only to show where MPK's fusion advantage
   already lives.
7. **The profiler slows the host, not the kernels.** Profiled decode ran at 155–2006 tok/s vs
   284–2988 unprofiled, but that is launch-gap inflation: kernel durations are CUPTI hardware
   timestamps, and the union-of-intervals closure against the unprofiled step (§2) is 0.2–5.3 %.
   The MPK side re-measure checked the analogous thing directly: profiled vs. unprofiled wall time
   for the same (bs, geometry) config, required <5 % difference, as M3-I1 established.

---

## 9. Artifacts

In this directory:

| path | what |
|---|---|
| `tables/comparison_by_stage.csv` | the ORIGINAL (M3-I1-era) §10 table, all three batch sizes, with per-layer columns — vLLM side only, still current |
| `tables/bs{1,8,16}_kernels.csv` | every kernel: calls/step, µs/step median + min/max over 6 windows, µs/call, % of GPU busy, stream |
| `tables/bs{1,8,16}_stages.csv` | per-stage sums and unions |
| `tables/prefill_bs{1,8,16}_kernels.csv` | short-context window (ctx ≈ 260), the ORIGINAL attention context bound (superseded by §5's direct measurement) |
| `tables/ordinal_bs1_cutlass.json` | per-GEMM-site cost of all 160 CUTLASS fp8 calls |
| `tables/profile_meta.json` | engine assertions, versions, per-rep throughput, GPU clocks, co-tenant checks |
| `ferret_targets.json` | **schema 2.0** — the dispatch list, total coverage of all 15 stages: 8 target specs + 7 disposition rows (incl. `resolved-by-I2b`) + a `coverage` assertion; every row's M3-I1-era numbers preserved in `history_m3i1`; attention's primary basis is the late-context capture (`context_band`/`matched_window`) |
| `remeasure_spec.md` | the EXECUTED methodology: run matrix as actually run (msl corrected to 353, late-context addendum §4(d)), capture invocation, normalisation, analysis commands, the two bugs found and fixed (msl semantics, `schedule_sim` tie-break) |
| `remeasure/` | **all regenerated data**: `armA/`, `armB/`, `armAlate/` (tables, pertask_by_bs.csv, attribution.csv), `armA_m3i10/tables/comparison_by_stage.csv` (the CURRENT §3 table), `qc/` (anchor-QC + call-site-split + AC-3 spot-check JSON), `scripts/` (every analysis tool used, for reproducibility), `patch/` (the one `profile_wave.py` code change), `logs/` (every run log + `ROOT_CAUSE_msl.txt`) — raw npz (>20 MB) pointed to `/home/catalyst/mpk-artifacts/m3i10-remeasure/` |
| `ncu/NCU_UNAVAILABLE.md` | why Nsight Compute could not run on this box, and what was tried |
| `ncu/roofline.csv`, `ncu/roofline.json` | the analytic memory-roofline substitute for NCU's SOL section |
| `sglang/probe_bs1.json` | SGLang feasibility result: versions, boot time, throughput reps |
| `sglang/sglang_bs1_kernels.csv` | SGLang's per-kernel decode table at bs1, same anchor method |
| `scripts/` | the vLLM-side capture and analysis pipeline (unchanged by the re-measure) |

On catalyst-B200 under `~/mpk-qwen35/m3i10-profile/` (too large for the repo):

| path | size |
|---|---|
| `traces/main/decode_bs{1,8,16}_win{0..5}.json` | 18 chrome traces, 57–62 MB each, **1.1 GB total** |
| `traces/main/prefill_bs{1,8,16}_win0.json` | 3 traces, 10–13 MB each |
| `logs/main.log` | the capture log, including the engine assertion line |
| `sglang/` | the SGLang venv and probe output |

### Reproduce

**vLLM side (unchanged):**
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

**MPK side (the re-measure, see `remeasure_spec.md` for the full matrix):**
```bash
# an isolated mirage-rm clone + venv-rm, patch/profile_wave_synthetic_prompts.patch applied
bash remeasure/scripts/gpu_guard_m3i10rm.sh <candidates> -- \
     bash remeasure/scripts/run_m3i10rm.sh A all   # arm A, matched geometry
bash remeasure/scripts/gpu_guard_m3i10rm.sh <candidates> -- \
     bash remeasure/scripts/run_m3i10rm.sh B all   # arm B, continuity
bash remeasure/scripts/run_armA_latectx.sh          # late-context closure, attention only
# analysis (all CPU): parse_profile.py / concurrency.py / anchor_qc.py per bs,
# then analyze.py -> pertask_by_bs.csv, build_comparison_armA.py -> comparison_by_stage.csv
python3 opt/m3i10/scripts/regenerate_ferret_v2.py    # -> ferret_targets.json (idempotent)
```

---

## 10. Historical appendix — the original M3-I1-era analysis (superseded 2026-07-27)

**Preserved verbatim for provenance.** Everything in this section is the document's ORIGINAL
content before the M3-I10 re-measure closure — the MPK numbers here are M3-I1's capture at the
AC-3 geometry (`max_seq_length=132`, 24–68-token prompts), predating the M3-I8 gate and the
M3-I2b quantize fix. **Do not read anything below as current** — §§1–9 above are the current
correspondence. This section exists only so the original analysis is not lost.

### 10.1 Original headline

> MPK is slower than the corresponding vLLM kernel in **13 of 15 stages** at bs16
> (12 at bs1, 13 at bs8, 12 at every batch size, 14 at at least one). The gap is not concentrated
> in one place — it is 2×–11× spread across quantize, the MoE GEMMs, the dense fp8 GEMMs, the GDN
> recurrent kernel and attention. MPK wins decisively in exactly one stage (norms/RoPE/glue, 3–4×
> faster, because it fuses them) and is at parity in two more. The one gap that *grows* with batch
> size is the GDN recurrent kernel: 7.4× at bs1, 10.8× at bs16.

| | bs1 | bs8 | bs16 |
|---|---:|---:|---:|
| vLLM decode step (this profile, union of GPU intervals) | 3567 µs | 4818 µs | 5363 µs |
| vLLM decode step (from the binding tok/s baseline) | 3502 µs | 4727 µs | 5301 µs |
| MPK decode step (M3-I1) | 15264 µs | 18618 µs | 22005 µs |
| ratio | **4.28×** | **3.86×** | **4.10×** |

That 4.28/3.86/4.10 reproduces M3-I1's 4.36/3.94/4.43 from an independent measurement (M3-I1
derived vLLM's step from throughput; this derives it from the kernel timeline).

### 10.2 Original correspondence table (M3-I1 capture, AC-3 geometry `msl=132`)

#### bs1

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

#### bs16

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

### 10.3 Original ranking (schema 1.0, M3-I1 capture — 9 targets / 6 dispositions)

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

*(§10 ends here; §§1–9 above are the current document.)*
