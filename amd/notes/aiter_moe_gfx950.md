# AITER's MoE kernels for K3 decode on MI355X (gfx950), read 2026-10-11

Source: the AITER tree in `lmsysorg/sglang-rocm:v0.5.21-rocm10-mi35x-20261009` (`/sgl-workspace/aiter`, commit `e7d2453f`,
2026-09-25), the pin SGLang main uses. Read only; nothing of it was run. Paths below are relative to that tree; `common.py` =
`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py`.

## What runs at T = 8 (K = 3584, IR = 384, E = 896)

From the tuned table `aiter/configs/model_configs/kimik3_a8w4_tuned_fmoe.csv`:

- W13 (stage 1): `flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8`, 31.12 us
- W2 (stage 2): `opus_moe2_afp8_wfp4_atomic_t32x128x256_sbm32_occ1_cache_b2_ws2`, 22.71 us
- both: 53.8 us, two launches (plus routing, sorting and quantization kernels before them). `t32x64x256_gui_kw2_fp8` is the
  T = 1 row (10.10 us).
- How the tuner routes tokens was not found. At 90 experts stage 1 reads ~124 MB: 4.0 TB/s; at ~120 experts (random) ~5.3.

## Stage 1 (FlyDSL W13)

- Tile 32 (sorted token rows) x 128 (N: 64 gate + 64 up, interleaved per 16 rows, "gui") x 256 (K). Grid 6 N tiles x 128 M
  blocks, 256 threads (4 waves); ~540 workgroups have work at 90 experts, the rest exit at once. No split K across
  workgroups at T = 8 (`kw2` at T = 1: two wave groups split K inside a workgroup and add through LDS).
- `w3`: LDS padded to >= 40,961 B so at most 3 workgroups (3 waves per SIMD) per CU (`common.py:325-334`).
- Weights: `buffer_load_dwordx4` into VGPRs, 16 B per lane, nt; one K tile (4 KB per wave) ahead (`common.py:896-958,
  1504-1736`). About 32-48 KB in flight per CU.
- Tokens: `raw_ptr_buffer_load_lds` (direct to LDS), XOR-swizzled ping/pong, two tiles ahead (`common.py:1139-1196`).
- Scheduling: `sched_barrier(0)` between phases; MFMAs inside `s_setprio(1)`/`s_setprio(0)`; an explicit `s_waitcnt
  vmcnt(5)`; an inline-asm barrier so LLVM inserts no `vmcnt(0)` at barriers (`common.py:75-98, 1534-1672`).
- Epilogue: SiTU v2 in registers, fp32 through LDS, 32-column amax by 3 shuffles, e8m0 exponent lowered by 8 (headroom),
  `cvt_pk_fp8_f32`, nt stores into `[token, topk, 384]`; the scale byte written in stage 2's layout (`common.py:2243-2474`).

## Stage 2 (Opus W2)

- Tile 32 x 128, K step 128 (3 steps), 128 threads (2 waves); grid 28 x 128, ~2,520 CTAs with work (~10 per CU);
  `__launch_bounds__(128, 1)`. Tokens: all 3 K steps into LDS up front. Weights per tile into VGPRs, nt (`cache_b2`),
  scales nt (`ws2`); `s_setprio(1)` around MFMAs.
- Sum over the 16 routed experts: routing weight applied, bf16 in LDS, then `raw_buffer_atomic_fadd_v2bf16` into a zeroed
  `out[token, hidden]`; no reduce pass.

## Other

- `aiter/ops/flydsl/kernels/mega_moe/`: an EP8 operator for DeepSeek-V4-Pro shapes (a8w4, gfx95x): several launches
  (prepare, dispatch + GEMM1, GEMM2 + scatter, combine), not one kernel; atomic ticket queues sharded 64 B apart; W13 -> W2
  ordered by the stream.
- XCDs: the T = 8 kernels do nothing per die. Stage 1 has an optional `xcd_swizzle` (used at T = 16); `splitk_epilogue.py`
  notes the dies do not share L2 and uses sc0|sc1 write-through stores.
- Weight layout: `(N/16, K/128, 4 k-lanes, 16 rows, 16 B)`, one 1 KB coalesced wave load per block; scales
  `(MN/32, K/256, 4, 16)` dwords.

## Compared with ours (amd/notes/moe_experts_gfx950.md)

Their expert kernels run at about the same speed as ours (our standalone W13 + W2: ~50 us at 80 experts). What differs: many
small independent work items balanced by the hardware dispatcher (little tail) instead of one wave per item from a queue;
up to 3 waves per SIMD with a shallow prefetch instead of 1 with a deep one (our probe 10-11: 2 waves per SIMD did not speed
up our W13); atomics for the W2 sum; tokens loaded straight to LDS.
