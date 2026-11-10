# What SGLang runs for the Kimi-K3 MoE layer on AMD MI35x (decode)

Written 2026-10-09. Read-only research. No GPU code was run. Every claim cites a file and line,
a URL, or is marked **inferred** (reasoned from code, not read directly) or **not found**.

## 0. Sources and versions

| Name used below | What | Commit / date |
|---|---|---|
| `sglang@main` | github.com/sgl-project/sglang, shallow clone | `86624f14` (2026-10-08) |
| `sglang@v0.5.19` | same repo, tag `v0.5.19` | `0bcd8223` (2026-09-03) |
| `aiter@main` | github.com/ROCm/aiter, shallow clone | `e6ded216` (2026-10-08) |
| AITER pinned by `sglang@v0.5.19` | `docker/rocm.Dockerfile:113-119` (stage `gfx950-rocm720`) | `c16d44b9` (2026-08-18) |
| AITER pinned by `sglang@main` between 2026-09-07 and 2026-09-16 | sglang commit `b9b39f22` "[AMD] Update ROCm AITER pin to 4ad9983 (#37784)" | `4ad99832` (2026-08-29) |
| AITER pinned by `sglang@main` today | `docker/rocm.Dockerfile` (`AITER_COMMIT_DEFAULT`) | `e7d2453f` (2026-09-25) |

The cookbook image `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916` (cookbook
`docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx:33`) was built 13 days after the `v0.5.19` tag.
Which SGLang commit and which AITER commit it contains: **not found**. Inferred: SGLang `main`
of about 2026-09-16 with AITER `4ad99832` (the pin in effect on that date).

File paths below are relative to the repo root of the named project. Line numbers are for
`sglang@main` and `aiter@main` unless a tag is named.

Run conditions assumed in this note (the cookbook / CI recipe): `SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1`, TP8, no DP attention, no
`--moe-a2a-backend`, MXFP4 checkpoint `moonshotai/Kimi-K3`
(`test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py:82-117`,
`test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py:63-91`). `--dcp-size 8` shards only
the MLA KV cache (cookbook line 98-106); the MoE code (`KimiK3MoE`) has no DCP branch (inferred
from reading `models/kimi_k3.py:422-1497`).

## 1. Short answer

- The MoE layer is `KimiK3MoE` in `python/sglang/srt/models/kimi_k3.py:422`. On ROCm with a
  bf16 shared expert it takes the **fused-front** path `_forward_fused` (`kimi_k3.py:1324-1456`).
- At decode (T tokens, T <= 16), one MoE layer on one GPU launches about **14 kernels**, in this
  order: front GEMM (hipBLASLt), copy, SiTU-and-mul (SGLang JIT), shared down GEMM (hipBLASLt),
  top-k (AITER HIP), MoE sort (AITER Opus HIP), MXFP8 activation quant (AITER HIP), expert GEMM1
  W13 + SiTU (AITER FlyDSL), expert GEMM2 W2 (AITER FlyDSL, or Opus for T = 8/16), copy,
  one all-reduce (AITER custom all-reduce), RMSNorm (AITER), latent up GEMM (AITER tuned GEMM,
  FlyDSL), 3-way add (SGLang JIT). Section 3 has the table with file:line for each.
- The expert GEMMs are **FlyDSL** kernels (a Python/MLIR DSL that AITER JIT-compiles), not
  assembly `.co` and not CK. Activations are MXFP8 (fp8 e4m3 + one e8m0 scale per 32);
  weights are MXFP4 (fp4 e2m1 + one e8m0 scale per 32), preshuffled.
- AITER's own tuner measured GEMM1+GEMM2 at **17.73 us for T=1**, 21.80 us for T=2, 33.69 us
  for T=4, 53.84 us for T=8, 91.86 us for T=16 (gfx950, 256 CU, inter_dim 384). Section 7 has
  the conditions.
- Decode is captured as one full HIP graph per batch size (`torch.cuda.CUDAGraph`). There is no
  PDL on ROCm: SGLang's JIT kernels turn PDL off under HIP.
- Routing is **not** fused into the expert kernels. Router logits come out of the merged front
  GEMM; top-k is a separate AITER kernel; sorting is a separate kernel.
- The cleanest plug-in boundary for a whole-layer kernel is `KimiK3MoE.forward(hidden_states,
  prefix_sum=...)` (`kimi_k3.py:1458`). For the routed experts only, `AiterRunnerCore.run`
  (`python/sglang/srt/layers/moe/moe_runner/aiter.py:259`) already has an intercept for an
  external kernel (the gfx950 small-M MoE kernel, lines 336-367) that can serve as the pattern.

## 2. Which code path is taken, and why

| Decision | Result on MI35x with the recipe | Evidence |
|---|---|---|
| MoE quant method | `Mxfp4Config` replaces the compressed-tensors config for the experts only | `kimi_k3.py:464-473` |
| Dense linears under `Mxfp4Config` on HIP | `UnquantizedLinearMethod` (bf16) | `layers/quantization/mxfp4.py:356-372` |
| Shared expert dtype | bf16 (the SBO comment gives "shared experts are bf16 ... ~264 MB/layer/rank") | `kimi_k3.py:619-628` (comment) |
| Latent down / up projections | `ReplicatedLinear`, bf16 unless the quant config sets `supports_kimi_k3_quantized_latent_projections` | `kimi_k3.py:630-662` |
| MoE runner backend | `auto` resolves to `AITER` on HIP | `mxfp4.py:1438-1454`; the K3 override to `flashinfer_mxfp4` is SM100/103/107 only, `arg_groups/model_overrides/kimi_k3.py:186-205` |
| Merged front weight `[shared gate_up | router gate | latent down]` | built on CUDA **and** HIP (main); built whenever not NPU (v0.5.19) | `kimi_k3.py:688-736` (main, HIP check at line 706); `sglang@v0.5.19 kimi_k3.py` `_merge_front_weights` (`if _is_npu: return`) |
| `_eligible_for_fused_front` | true: latent MoE, bf16 shared down weight, a2a none | `kimi_k3.py:741-755`, used at `kimi_k3.py:1487-1488` |
| Fused-front output dtype | bf16 on HIP (`_front_fp32` is `not _is_hip and ...`) | `kimi_k3.py:758-768` |
| Side stream (`alt_stream`) | `None` on HIP | `kimi_k3.py:2866-2871` ("Disable on HIP code path") |
| K3 fused all-reduce (`k3_ar_fusion`) | off: auto-probe only enables on SM 100/103 | `layers/communication/k3_ar_fusion.py:59-72`; `get_device_sm()` returns `major*10+minor` (`srt/utils/common.py:734-744`); inferred: gfx950 reports capability (9,5) |
| Therefore the tail | "single collective over the flat [latent | shared] pair" | `kimi_k3.py:1433-1446` |
| Merged-front GEMM call | `_k3_bf16_gemm` calls `torch.nn.functional.linear` / `torch.mm` directly, not AITER `tgemm` | `kimi_k3.py:188-221`; confirmed by open PR #41867 ("calls torch.mm directly and lands on hipBLASLt", trace shows `Cijk_Alik_Bljk_..._MT32x16x1024`) |

Unfused fallback (`_forward_unfused`, `kimi_k3.py:1060-1240`) runs only when the merge is not
possible (for example a quantized shared expert). It issues the gate GEMM, top-k, latent down
GEMM, experts, a latent all-reduce, norm, up GEMM, the shared MLP and a second all-reduce for
the shared output. Not the recipe path; not detailed further here.

## 3. Kernel list for one MoE layer at decode, in execution order

Per GPU, TP8, T tokens (T <= 16). Sizes are per rank. Shared intermediate is 6144 total, 768 per
rank, so the merged front has `2*768 + 896 + 3584 = 6016` rows (inferred from the merge order at
`kimi_k3.py:708-713` and `_merge_weights_as_views`, `kimi_k3.py:238-263`; AITER's K3 bf16 GEMM
table has a tuned row for exactly N=6016, K=7168, `aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv`).

Input to the layer: `hidden_states [T,7168] bf16` (output of the attention-residual aggregation
and `post_attention_layernorm`, `kimi_k3.py:2809-2822`) and `prefix_sum [T,7168] bf16` (the
residual stream, added at the end).

| # | Operation | Function (file:line) | Library / kernel | Shape (per rank) |
|---|---|---|---|---|
| 1 | Merged front GEMM: shared gate_up + router logits + latent down, one GEMM | `_k3_bf16_gemm(hidden_states, self._front_w)` `kimi_k3.py:1345-1349` -> `F.linear` `kimi_k3.py:216-217` | torch -> hipBLASLt (PR #41867 trace) | `[T,7168] x [6016,7168]^T -> [T,6016]` bf16 |
| - | Split into `gate_up [T,1536]`, `router_logits [T,896]`, `routed_input [T,3584]` (views, no kernel) | `kimi_k3.py:1350-1352` | - | - |
| - | With `SGLANG_AITER_K3_OPT=1`, `router_logits` is **not** made contiguous (else one copy kernel when T>1) | `kimi_k3.py:1353-1354` | - | - |
| 2 | Densify `routed_input` (`.to(bf16).contiguous()`); always taken on ROCm | `kimi_k3.py:1355-1358`, condition `_moe_front_needs_dense_bf16` `kimi_k3.py:1243-1262` | torch elementwise copy (PR #41848 trace: `at::native::elementwise_kernel_manual_unroll`) | `[T,3584]` bf16 |
| - | Allocate flat `buf = [latent T*3584 | shared T*7168]` | `kimi_k3.py:1370-1376` | allocation only | `T*10752` bf16 |
| 3 | Shared expert activation SiTU-and-mul | `_forward_shared` `kimi_k3.py:1306-1318` -> `SituAndMul.forward_cuda` (`forward_hip` defaults to `forward_cuda`, `layers/utils/multi_platform.py:47-48`) -> `sglang.kernels.ops.activation.situ_and_mul` `layers/activation.py:209-212` | SGLang JIT HIP kernel (`sglang::situ_and_mul_kernel` in PR #41848 trace) | `[T,1536] -> [T,768]` |
| 4 | Shared expert down GEMM, written into `buf` slice via `out=` | `_k3_bf16_gemm(..., out=shared_output)` `kimi_k3.py:1313-1318` -> `torch.mm(..., out=)` `kimi_k3.py:221` | torch -> hipBLASLt (inferred; PR #41867 says the shared down projection is left on torch) | `[T,768] x [7168,768]^T -> [T,7168]` |
| 5 | Top-k: sigmoid + bias + top-16 + renorm + routed scaling | `_forward_routed` `kimi_k3.py:1281-1291` -> `TopK.forward_cuda` `layers/moe/topk.py:704` -> `select_experts` `topk.py:2545` -> `biased_grouped_topk_gpu` `topk.py:1777` -> aiter branch `topk.py:1919-1966` -> `aiter.biased_grouped_topk` `aiter/ops/topk.py:178-216` -> `biased_grouped_topk_hip` | AITER HIP C++ (`csrc/kernels/topk_softmax_kernels_group.cu`, module `module_moe_asm`, `aiter/jit/optCompilerConfig.json:519-535`) | in `[T,896]` bf16 logits + bf16 bias; out `[T,16]` fp32 weights, int32 ids |
| - | FusedMoE -> StandardDispatcher (no kernel at TP) -> `Mxfp4MoEMethod.apply` -> `_apply_aiter` (no pad: hidden 3584 is already a multiple of 256) | `fused_moe_triton/layer.py:1546-1585`, `mxfp4.py:1701-1702`, `mxfp4.py:1738-1802` | - | - |
| - | `pre_permute_standard_to_aiter`: `.to(float32)` / `.to(int32)` (no-ops for these dtypes, inferred) | `moe_runner/aiter.py:406-429` | - | - |
| - | SGLang gfx950 small-M MoE kernel check: rejected for K3 (needs SiLU, hidden 4096, top-k 10/11) | `moe_runner/aiter.py:336-367`, `kernels/ops/moe/smallm_moe_gfx950/__init__.py:264-313` | - | - |
| 6 | MoE sort (token -> expert blocks, block_m 32) and zero of `moe_buf` | `aiter.fused_moe` `aiter/fused_moe.py:819` -> `_fused_moe_impl` `fused_moe.py:1101` -> `moe_sorting` `fused_moe.py:1484-1499` -> `_moe_sorting_impl` -> `aiter.moe_sorting_opus_fwd` `fused_moe.py:510` | AITER Opus HIP C++ (`csrc/include/moe_sorting_opus.h`) | `[T,16]` routes |
| 7 | MXFP8 quant of `routed_input` into sorted order (fp8 e4m3 + e8m0 per 32) | `fused_moe_2stages` `fused_moe.py:4196-4219` -> `fused_dynamic_mxfp8_quant_moe_sort` `aiter/ops/quant.py:1242` -> `fused_dynamic_mx_quant_moe_sort` (HIP, `module_quant`) | AITER HIP C++ | `[T,3584]` -> fp8 + scales |
| 8 | Expert GEMM1 (W13 gate+up) + SiTU v2 + fp8 re-quant of the intermediate | `metadata.stage1` -> `_flydsl_stage1_wrapper` `fused_moe.py:1948-2028` -> `flydsl_moe_stage1` `aiter/ops/flydsl/moe_kernels.py:1888` -> `compile_mixed_moe_gemm1_common` `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:113` | AITER **FlyDSL** (uses `rocdl.mfma_scale_f32_16x16x128_f8f6f4`, `mixed_moe_gemm_2stage_common.py:705`) | kernel name for T=1: `flydsl_moe1_afp8_wfp4_bf16_t32x64x256_gui_kw2_fp8` |
| 9 | Expert GEMM2 (W2 down), route-weighted, accumulated into `moe_buf` | `_flydsl_v2_stage2_wrapper` `fused_moe.py:2657-2852` -> `mxfp4_moe_gemm2` `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:772` (MFMA in `mxfp4_gemm2.py:509-528`) | AITER **FlyDSL** for T=1,2,4; AITER **Opus** HIP C++ (`csrc/opus_moe/opus_moe.cu`) for T=8,16 | T=1: `flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_nt_sbm32`, out `[T,3584]` bf16 |
| 9b | Only when the tuned row says `reduce` (T=3/4 bucket): extra reduction kernel | `fused_moe.py:2838-2851` | AITER FlyDSL | `[T,16,3584] -> [T,3584]` |
| 10 | Copy expert output into the `buf` latent slice (AITER runner ignores the zero-copy output slot) | `kimi_k3.py:1290-1291`; zero-copy is read only by marlin and flashinfer runners (`fused_marlin_moe.py:412`, `moe_runner/flashinfer_trtllm.py:1120-1122`) | torch copy | `[T,3584]` |
| 11 | All-reduce of the flat `[latent | shared]` buffer (TP-partial sums) | `tensor_model_parallel_all_reduce(buf)` `kimi_k3.py:1446` -> `GroupCoordinator.all_reduce` `distributed/parallel_state.py:761-799` -> method `"ca"` `parallel_state.py:962-967`, `1048-1050` | AITER custom all-reduce (`aiter/dist/device_communicators/custom_all_reduce.py`, kernels in `csrc/include/custom_all_reduce.cuh`), selected by `dispatch_custom_allreduce` `device_communicators/custom_all_reduce.py:392-405` (`SGLANG_USE_AITER_AR` default true) | `T*10752` bf16 = `T*21,504` bytes |
| 12 | Latent RMSNorm | `self._latent_norm(latent)` `kimi_k3.py:1450-1451` -> `RMSNorm.forward_hip` `layers/layernorm.py:757-791` -> `aiter.rmsnorm2d_fwd` (`layernorm.py:111-124`) | AITER | `[T,3584]` |
| 13 | Latent up projection | `self.routed_expert_up_proj(latent)` `kimi_k3.py:1452` -> `UnquantizedLinearMethod.apply` `layers/quantization/unquant.py:514-517` -> `tgemm.mm` (K=3584 > 2048, so not the Triton a16w16 branch, `unquant.py:79-107`) | AITER tuned GEMM; tuned row gfx950 M=1 N=7168 K=3584 is a FlyDSL hgemm (`kimik3_bf16_tuned_gemm.csv`) | `[T,3584] x [7168,3584]^T -> [T,7168]` |
| 14 | `out + shared_output + prefix_sum` | `_add3(..., prefetch_bc=True)` `kimi_k3.py:1456`, `kimi_k3.py:383-406` -> `sglang.kernels.ops.elementwise.add3` (`kernels/ops/elementwise/add3.py:26-69`) | SGLang JIT kernel, PDL flag false on HIP (`kernels/jit/utils/arch.py:178-181`) | `[T,7168]` bf16, the layer output |

Count: 14 kernels for T=1 or 2 (15 when the GEMM2 row is a `reduce` row). Inferred from the
code path above; not checked against a ROCm trace except steps 1-3, which PR #41848's trace shows.

Not present on ROCm (CUDA-only in this file): route+quant fusion (`_route_quant_fuse_eligible`
needs flashinfer, `kimi_k3.py:1264-1279`), deferred finalize, `k3_ar_fusion` push/norm/gemm_ag
tail (`kimi_k3.py:1378-1432`), side-stream overlap of shared and routed experts.

### Optional SGLang patch that changes steps 6-7 (not in the 2026-09-16 image)

`sglang@main` installs a runtime patch on `aiter.fused_moe` when gfx950: for `T*topk <= 256`
(so T <= 16 for K3), one Triton launch does the sort, the `moe_buf` zeroing and the MXFP8
activation quant; AITER's own sort and quant calls are then skipped
(`python/sglang/kernels/ops/moe/moe_sorting_small.py:421-560`, gates at lines 297-306 and
442-449; installed from `moe_runner/aiter.py:250-257`). The file was added 2026-09-24 (sglang
commit `36f59982`, PR #36559), after the cookbook image date, and is absent in `sglang@v0.5.19`.
With it, steps 6+7 become one kernel (13 kernels total at T <= 2, inferred).

## 4. What the environment variables change

| Variable | Read where | Effect on the MoE layer |
|---|---|---|
| `SGLANG_USE_AITER=1` | `mxfp4.py:200`, `topk.py:136`, `unquant.py:70`, `layernorm.py:54`, `deepseek_v2.py` | Turns on the AITER branches: AITER MoE runner, AITER top-k, AITER `tgemm` for module linears, AITER RMSNorm. `SGLANG_AITER_K3_OPT` is gated behind it in `mxfp4.py:201` and `quark_w4a4_mxfp4_moe.py:50` |
| `SGLANG_AITER_K3_OPT=1` | `kimi_k3.py:145,1353`; `mxfp4.py:201,519-525`; `quark/schemes/quark_w4a4_mxfp4_moe.py:50,214` | (a) Pads the per-rank expert intermediate to a multiple of **128** instead of 256, so 384 stays 384 instead of 512. The comment says 256 "costs +33% mem" and that 128 "lets the TP8 decode CUDA graph range grow from 20 to 120" (`mxfp4.py:517-525`). (b) Skips `router_logits.contiguous()` for T>1 (`kimi_k3.py:1353-1354`). (c) Same padding rule for Quark checkpoints. Open PR #43130 renames it to `SGLANG_ROCM_K3_AITER_OPT` |
| `AITER_SITUV2_A8W4=1` | AITER `fused_moe.py:785-797` (`resolve_activation_dtype`); SGLang `mxfp4.py:150-154` (main) / `mxfp4.py:953` (v0.5.19) | AITER: SiTU v2 MoE quantizes activations to **fp8** (A8W4) instead of the default bf16 (A16W4); A8W4 wins if `AITER_SITUV2_A4W4` is also set. SGLang: selects the gate/up-interleaved preshuffle (`shuffle_weight_a16w4`, `shuffle_scale_a16w4`) at load (`mxfp4.py:979-1004`) |
| `AITER_FLYDSL_FORCE=1` | AITER `fused_moe.py:3718-3719` | Default is already "1" in `aiter@main`; the comment says "for debug use". It only matters when **no tuned row** is found: it lets per_1x32 fp8/fp4 MoE fall to the FlyDSL heuristic path (`fused_moe.py:3748-3764`). With the K3 tuned rows present (section 5) it has no effect on decode (inferred) |

`SGLANG_K3_FUSED_FRONT` (default true, `environ.py:1838`) matters only for the EP a2a two-way
merge. `SGLANG_K3_RADIX4_TOPK` (default false, `environ.py:1840`) would replace the AITER top-k
with a SGLang HIP radix-select kernel written for 896 experts / top-16
(`kernels/ops/moe/moe_route_radix4.py:1-25`, dispatch at `topk.py:1929-1940`).

## 5. AITER expert kernels in detail

### 5.1 Kernel technology and files

| Piece | File | Technology |
|---|---|---|
| Dispatcher, config lookup | `aiter/fused_moe.py` (`fused_moe` 819, `_fused_moe_impl` 1101, `get_2stage_cfgs` 2934, `fused_moe_2stages` 4054) | Python |
| GEMM1 (fp8 A x fp4 W, SiTU, fp8 out) | `aiter/ops/flydsl/moe_kernels.py` (`flydsl_moe_stage1` 1888, `compile_flydsl_moe_stage1` 622) -> `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:30` -> `mixed_moe_gemm_2stage_common.py:113` | FlyDSL (`import flydsl.compiler as flyc`, `flydsl._mlir`, lines 24-32), JIT compiled |
| GEMM2 "layout v2" | `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:772`, `mxmoe_gemm_v2.py`, `mxfp4_gemm2.py` | FlyDSL |
| GEMM2 Opus variant (T=8,16) | `aiter/ops/opus/moe_stage2_a8w4.py`, `csrc/opus_moe/opus_moe.cu`, header library `csrc/include/opus/opus.hpp` ("OPUS, AI Operator Micro STD") | HIP C++ templates |
| Sort | `aiter.moe_sorting_opus_fwd` (`fused_moe.py:510`), `csrc/include/moe_sorting_opus.h` | HIP C++ |
| Activation quant + sort of scales | `aiter/ops/quant.py:1087-1280` | HIP C++ (`module_quant`) |
| SiTU v2 reference | `fused_moe.py:4676-4684`: `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)`; defaults `beta=4.0`, `linear_beta=25.0` (`aiter/ops/flydsl/moe_common.py:10-11`) | - |

None of the K3 decode MoE kernels is an assembly `.co` or a CK kernel (inferred from the kernel
names in the tuned rows below; the asm 1-stage path is only taken when a row has `run_1stage=1`,
`fused_moe.py:3364-3375`, and no K3 A8W4 row does).

SGLang passes the SiTU constants from the checkpoint config as `beta=gemm1_alpha` and
`linear_beta=gemm1_clamp_limit` (`moe_runner/aiter.py:302-308`, `kimi_k3.py:486-487`). The
MegaMoE path asserts these are (4.0, 25.0) for K3 (`kimi_k3.py:538-545`).

### 5.2 Tuned configs for small M (gfx950, 256 CU)

File: `aiter/configs/model_configs/kimik3_a8w4_tuned_fmoe.csv`. It is merged into the main
`tuned_fmoe.csv` lookup at import (`aiter/jit/core.py:322-325, 573-598`). Key: (gfx, cu_num,
padded token, model_dim 3584, inter_dim, 896 experts, top-16, Situv2, bf16, fp8 A, fp4x2 W,
per_1x32, g1u1). Token is padded with `get_padded_M` = next power of two below 32768
(`fused_moe.py:1864-1879`), so T=3 uses the T=4 row, T=5..8 the T=8 row, T=9..16 the T=16 row
(inferred from that function).

Rows for **inter_dim 384** (TP8 with `SGLANG_AITER_K3_OPT=1`). These rows are byte-identical in
`aiter@main`, `c16d44b9` and `4ad99832` (checked by diffing the three files).

| T | block_m | GEMM1 kernel | us1 | GEMM2 kernel | us2 | total us |
|---|---|---|---|---|---|---|
| 1 | 32 | `flydsl_moe1_afp8_wfp4_bf16_t32x64x256_gui_kw2_fp8` | 10.10 | `flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 7.63 | 17.73 |
| 2 | 32 | `..._t32x64x256_w2_gui_kw2_fp8` | 12.56 | `flydsl_moe2_layout_..._t32x128x128_atomic_sbm32` | 9.24 | 21.80 |
| 4 | 32 | `..._t32x64x256_w3_gui_xcd4_kw2_fp8` | 18.76 | `flydsl_moe2_layout_..._t32x128x128_atomic_nt_sbm32` | 14.93 | 33.69 |
| 8 | 32 | `..._t32x128x256_w3_gui_fp8` | 31.12 | `opus_moe2_afp8_wfp4_atomic_t32x128x256_sbm32_occ1_cache_b2_ws2` | 22.71 | 53.84 |
| 16 | 32 | `..._t32x128x256_w4_gui_xcd4_fp8` | 56.87 | `opus_moe2_afp8_wfp4_atomic_t32x128x256_sbm32_occ1_cache_b3_ws2` | 34.98 | 91.86 |

(A T=3 row exists with a `reduce` GEMM2, but per the padding rule above it is never looked up.)
Rows for inter_dim 512 (without `SGLANG_AITER_K3_OPT`) are also present; at T=1 they total
18.39 us in `aiter@main`.

Name decoding (from `moe_kernels.py:54-71, 162-307`): `t{M}x{N}x{K}` = tile; `_w{n}` =
waves_per_eu; `_kw{n}` = intra-block K split across waves; `_xcd{n}` = XCD swizzle; `_gui` =
gate/up interleaved (fp8-A stage1 kernels use `gate_mode="interleave"`); trailing `_fp8` = GEMM1
writes an fp8 intermediate (fused re-quant); `_atomic` / `_reduce` = GEMM2 epilogue;
`_sbm32` = sort block_m 32; `_nt` = non-temporal loads.

Computed (not measured): at T=1 the 16 selected experts hold
`16 x (768x1792 + 3584x192) B = 33.0 MB` of fp4 weights plus about 2.1 MB of e8m0 scales, so
17.73 us is about 2.0 TB/s of weight traffic.

### 5.3 Weight layout the kernels expect (MXFP4 experts)

Shapes after load, per rank (from `mxfp4.py:517-600`, hidden 3584, inter 384):

- `w13_weight [896, 768, 1792]` uint8 (two fp4 e2m1 per byte), rows = `[gate 0..383 | up 0..383]`
  as loaded (`gate_up_interleaved=False`, `kimi_k3.py:488`, so SGLang's own row interleave at
  `mxfp4.py:957-976` is skipped).
- `w2_weight [896, 3584, 192]` uint8.
- `w13_weight_scale [896, 768, 112]`, `w2_weight_scale [896, 3584, 12]` uint8 e8m0, one scale per
  32 elements along K. No bias on HIP (`mxfp4.py:585`).

Preshuffle applied at load under A8W4 (`mxfp4.py:991-1004`):

- `shuffle_weight_a16w4(w13, 16, gate_up=True)` = `shuffle_weight(x, layout=(16,16),
  is_guinterleave=True, gate_up=True)` (`aiter/ops/shuffle.py:213-217`). Code at
  `shuffle.py:173-187`: with `NLane=16`, `KPack=16` bytes, `KLane=4`, `N0=inter/16`,
  `K0=(K/2)/64`, the tensor is viewed as `[E, 2(gate/up), N0, 16, K0, 4, 16B]` and permuted to
  `[E, N0, 2(gate/up), K0, KLane=4, NLane=16, 16B]`. So for each 16-row block of the intermediate,
  the 16 gate rows and then the 16 matching up rows are stored, each as K chunks of 64 bytes (128
  fp4 values) laid out lane-major (4 K-lanes x 16 N-lanes x 16 bytes). This matches the 64-lane
  operand of the 16x16x128 scaled MFMA (inferred).
- `shuffle_weight_a16w4(w2, 16, gate_up=False)`: `[E, N0=3584/16, 16, K0, 4, 16B]` ->
  `[E, N0, K0, KLane, NLane, 16B]` (`shuffle.py:183-185`).
- Scales: `shuffle_scale_a16w4(scale.view(-1, K/32), 896, gate_up)` = `shuffle_scale(...,
  is_guinterleave=True)` (`shuffle.py:444-450`, body `shuffle.py:385-419`). K/32 is first padded
  to a multiple of 8 with `0x7F` (= 1.0): w13 stays 112, w2 goes 12 -> 16. Then with
  `K_Pack=2, N_Pack=2, N_Lane=16, K_Lane=4`:
  - gate_up: view `[E, 2(gate/up), N1=inter/16, 16, K1=Kscale/8, 2, 4]`, permute to
    `[E, N1, K1, K_Lane, N_Lane, K_Pack, gate/up]`.
  - down: view `[E, N1=3584/32, 2, 16, K1, 2, 4]`, permute to `[E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]`.
  So each lane reads 4 consecutive e8m0 bytes (one dword) per 256-element K step (inferred from
  the permute order).
- The tensors are tagged `is_shuffled = True` (`mxfp4.py:1029-1030`); `_apply_aiter` re-tags the
  `float4_e2m1fn_x2` views (`mxfp4.py:1759-1764`). AITER discards Opus rows if weights are not
  tagged (`fused_moe.py:3184-3197`).

Nibble order inside a byte: **not found** (not checked).

### 5.4 Activation quant format

- GEMM1 input: fp8 e4m3fn + e8m0 scale per 1x32 group (`QuantType.per_1x32`, `q_dtype_a =
  dtypes.fp8` under A8W4, `fused_moe.py:785-797`). Produced by
  `fused_dynamic_mxfp8_quant_moe_sort`, which writes scales "(pad32(M_o), N_o) fp8_e8m0 -- the
  same byte layout the FlyDSL stage1/stage2 GEMM consumes" in sorted-token order
  (`aiter/ops/quant.py:1242-1283`). SGLang's later Triton replacement documents the scale rule
  as "e8m0 RoundUp scale byte per (sorted_row, group) at aiter's mx_scale_shuffle_idx address"
  (`moe_sorting_small.py:1-11`).
- GEMM2 input: the fp8 + e8m0 intermediate written by GEMM1 itself (`_fp8` suffix,
  `fuse_quant="fp8"`, `skip_inter_quant` for `moe2_layout` kernels, `fused_moe.py:3614-3634`,
  `4389-4405`). No separate inter-stage quant kernel.
- Output: bf16 `[T,3584]`, accumulated with atomics into the zeroed sort buffer for `atomic` rows.

## 6. Graph capture, PDL, all-reduce, routing fusion

- **Graph**: decode uses `FullCudaGraphBackend` by default (`model_executor/runner_backend/utils.py:54-64`),
  which records `torch.cuda.CUDAGraph()` per batch size (`runner_backend/full_cuda_graph_backend.py:161-182`).
  On ROCm this is a HIP graph (inferred: PyTorch's ROCm build maps the CUDA graph API to HIP).
  The recipe captures up to `--cuda-graph-max-bs-decode` (256 in issue #32548, 64 in the CI tests).
  The merged front weight is built before capture (`kimi_k3.py:3517-3530`).
- **PDL**: SGLang JIT kernels set PDL off when the runtime is HIP (`kernels/jit/utils/arch.py:178-181`).
  An equivalent mechanism on ROCm in this path: **not found**.
- **Side streams**: off on HIP (`kimi_k3.py:2866-2871`), so shared and routed experts run
  serially on one stream.
- **All-reduce**: AITER `CustomAllreduce` (`aiter/dist/device_communicators/custom_all_reduce.py`,
  kernels `csrc/kernels/custom_all_reduce.cu`, `csrc/include/custom_all_reduce.cuh`), chosen in
  `sglang/srt/distributed/device_communicators/custom_all_reduce.py:344-411`. It is registered for
  graph capture (`enable_register_for_capturing`, line 401-404). SGLang also builds a
  `QuickAllReduce` on ROCm (`parallel_state.py:527-538`), but `"ca"` is tried first
  (`parallel_state.py:962-985`). Which of AITER's 1-stage / 2-stage variants runs at 21.5 KB:
  **not found** (selection is inside AITER C++, not read).
- **RCCL**: not used for this all-reduce when the custom all-reduce accepts the size (inferred
  from the dispatch order above).
- **Fused routing**: no. Logits come from the merged front GEMM (bf16), top-k is a separate AITER
  kernel, sort is a separate kernel, and the expert GEMMs read the sorted ids. SGLang's fused
  gate+top-k front (`kernels/ops/moe/moe_front.py`) is only used for the EP a2a two-way merge
  (`kimi_k3.py:906-922`).

## 7. Published timing numbers

| Number | Conditions | Source |
|---|---|---|
| Expert GEMM1+GEMM2: 17.73 / 21.80 / 33.69 / 53.84 / 91.86 us for T=1/2/4/8/16 | AITER tuner rows, gfx950, 256 CU, inter 384, A8W4. Tuner times kernels with `run_perftest` (torch profiler device time, averaged, default 101 iters) and rotates deep copies of all arguments to defeat cache reuse (`aiter/test_common.py:50-80, 242-265`; tuner call `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py:4843-4873`). Inferred: weights are mostly not L2/MALL-resident. Sort and quant kernels are not included | `aiter/configs/model_configs/kimik3_a8w4_tuned_fmoe.csv` |
| Merged front GEMM [6016x7168] with AITER FlyDSL hgemm: 16.36 us (T=1), 17.76 us (T=16); hipBLASLt row at T=16: 19.28 us | AITER bf16 GEMM tuner rows, gfx950. SGLang main does **not** use the AITER row for this GEMM (it calls torch) | `kimik3_bf16_tuned_gemm.csv` |
| Latent up GEMM [7168x3584]: 10.69 us (T=1); shared down [7168x768]: 4.61 us (T=1) | same tuner, gfx950. Up GEMM goes through `tgemm` so this row applies (inferred); shared down goes through torch in SGLang | `kimik3_bf16_tuned_gemm.csv` |
| Routing the front GEMM through AITER saves "about 0.5-0.7 ms of decode time per token"; TPOT 18.57 -> 18.06 ms at 1024/1024 conc 4 | MI355X, ROCm 7.2.4, SGLang 0.5.20, TP8, bench_serving random, single run per point; PR open, not merged | PR #41867 |
| Removing the step-2 copy saves "about 0.3-0.4 ms" TPOT; 18.57 -> 18.20 ms at conc 4 | same setup, `AITER_SITUV2_A8W4=1`; PR open | PR #41848 |
| End-to-end decode: median TPOT 20.86 ms at conc 2, 22.62 at 4, 27.57 at 8, 34.66 at 16, 48.73 at 32 | MI355 TP8, day-0 image `rocm720-mi35x-k3-20260727`, non-speculative, same env vars, triton attention | issue #32548 |
| Fused KDA decode boundary 9.20 -> 8.38 us/layer | 69-layer graph; KDA, not MoE | cookbook `Kimi-K3.mdx:227` |
| A4W4 vs A8W4: 530.8 vs 537.3 tok/s median output | 8x MI35x; no other conditions given | cookbook `Kimi-K3.mdx:223` |

Per-layer split inferred from the PR deltas: dividing 0.5-0.7 ms and 0.3-0.4 ms by the 92 MoE
layers (count from `test_kimi_k3_perf_mi35x.py` comment "93 attention + 92 MoE layers") gives
about 5-8 us per layer for hipBLASLt vs AITER on the front GEMM and about 3-4 us per layer for
the copy. These are derived numbers, not measurements of a single kernel.

A per-kernel trace breakdown of the whole K3 MoE layer on MI35x: **not found**.

## 8. Where an external fused MoE kernel could plug in

| Boundary | Input | Output | Notes |
|---|---|---|---|
| **A. Whole MoE layer**: `KimiK3MoE.forward(hidden_states, *, prefix_sum, forward_batch)` `kimi_k3.py:1458-1497` (or `_forward_fused`, `kimi_k3.py:1324`) | `hidden_states [T,7168]` bf16; `prefix_sum [T,7168]` bf16 or `None` | `[T,7168]` bf16 = `up(RMSNorm(AR(latent_partial))) + AR(shared_partial) + prefix_sum` | Covers steps 1-14, including the all-reduce. Weights available as `self._front_w` (merged bf16), `self.shared_experts.down_proj.weight`, `self.experts.w13_weight/w2_weight` + scales (preshuffled), `self.routed_expert_norm.weight`, `self.routed_expert_up_proj.weight`, `self.gate.e_score_correction_bias` (fp32 after load, `kimi_k3.py:3524-3529`). Open PR #43130 adds ROCm-only `kimi_k3_rocm_*.py` modules hooked from `kimi_k3.py` with `_is_hip` guards; that is the shape upstream is moving toward |
| **B. Routed experts only**: `AiterRunnerCore.run` `moe_runner/aiter.py:259-394` | `hidden_states [T,3584]` bf16 (dense), `topk_weights [T,16]` fp32, `topk_ids [T,16]` int32, `quant_info` with preshuffled `w13/w2` and scales | `AiterRunnerOutput(hidden_states=[T,3584] bf16)`, TP-partial sum over this rank's 384-wide intermediate shard | Precedent: the gfx950 small-M MoE intercept (`aiter.py:336-367`) calls an external kernel guarded by a `*_supported(...)` check and returns early. Steps 6-9 would be replaced; the all-reduce stays outside |
| **C. Inside AITER**: `aiter.fused_moe.fused_moe` (`aiter/fused_moe.py:819`) | same as B | `[T,3584]` bf16 | SGLang already monkey-patches `aiter.fused_moe.fused_moe`, `_moe_sorting_impl` and `fused_dynamic_mxfp8_quant_moe_sort` at runtime (`moe_sorting_small.py:421-560`), so a patch at this level has precedent |

For a megakernel that also owns routing, the communication and the residual add, boundary A is
the only one that covers all of it. Boundaries B/C keep SGLang's front GEMM, top-k, all-reduce
and tail.

## 9. Differences between versions that matter

- `sglang@v0.5.19` vs `main` for this path: same fused-front logic and same ROCm tail
  (`_forward_fused` diff shows only comment and `get_tp_group()` -> `get_parallel().tp_group`
  changes). v0.5.19 has no `moe_sorting_small` patch and no `smallm_moe_gfx950` kernel
  (`ls python/sglang/kernels/ops/moe/` in the tag).
- `sglang@v0.5.19` chooses the GU-interleaved shuffle only when `AITER_SITUV2_A8W4=1`
  (`mxfp4.py:951-958` in the tag); `main` uses it unless only `AITER_SITUV2_A4W4` is set
  (`mxfp4.py:150-154, 983-990`).
- Open, unmerged ROCm K3 PRs that would change the kernel list: #41867 (front GEMM via AITER),
  #41848 (drop step-2 copy), #43130 (fused AR+RMSNorm, AR+residual, split MoE AR, FP8 decode
  projections), #41794 (Quark checkpoint fusions), #35287 (cooperative "preroute" MoE producer
  for exact M=2/M=4).

## 10. Not found / open

- Exact SGLang and AITER commits inside `v0.5.19-rocm720-mi35x-20260916`.
- Which AITER custom all-reduce kernel variant runs for 21.5 KB at TP8, and its latency.
- Measured runtime (not tuner) per-kernel times for the K3 MoE layer on MI35x.
- Latency of the AITER top-k, sort and MXFP8 quant kernels at T=1..16.
- fp4 nibble order inside a packed byte as consumed by the FlyDSL kernels.
- Whether `add3` (step 14) JIT-builds on HIP in the shipped image (code path reaches it when
  `prefix_sum` is not `None`; a ROCm trace was not read).
