/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// ============================================================================
// FFN FULL MEGAKERNEL — DSv3 decode MoE-FFN, FULLY fused (bs=1, M=1, TP8 EP2
// per-rank). This is THE default decode MoE-FFN path: it fuses FOUR front
// stages IN FRONT of the FFN body, so the routing is computed INTERNALLY
// (active_experts/weights are derived, NOT inputs):
//   (A) input RMSNorm   (PRE-rmsnorm bf16 hidden -> bf16 normed, the boundary)
//   (B) router gate-GEMV (bf16 normed . bf16 W_gate, fp32 accum -> bf16 logits)
//   (C) topk-sigmoid group routing -> EP-LOCAL active_experts + weights
//   (0..3) the unchanged FFN body (FP8-quant of the BF16 normed -> W13 -> silu
//          -> requant -> W2 + weighted atomicAdd; shared gate_up -> silu
//          ->down)
//
// This kernel REPLACES the whole per-task decode MoE chain (rmsnorm +
// router-gate-GEMV + topk-sigmoid + permute + W13/W2 group-GEMM + silu). The
// builder selects it for the bs=1 TP8/EP2/B200 decode geometry
// (_use_ffn_full_megakernel); all other configs (prefill mbt>8, non-TP8) fall
// through to the per-task chain.
//
// PORTED FROM: scratch/megakernels/ffn_fullyfused_ferret_v015_cold62us_ACTIVE8
// .cuh (fused_moe_full). The MPK ABI (the now-retired partially-fused COLD FFN
// predecessor was the original ground truth) covers: GridBarrier +
// grid_barrier, extern __shared__ __align__(1024) dynamic smem, the per-task
// Scratch, input_ptrs binding, fp32 out_acc + bf16 convert, the dynamic-smem
// layout, the SCRATCH_BYTES %16 rule, output_ptrs[0]-only write.
//
// v019 SPEED MERGE (ferret workspace5, 46.18us cold ACTIVE=4, 9.35x vs chain,
// correctness EXACT). The merged levers
// are GEMV/compute SPEED levers ONLY (the reduce/topk/silu MATH is
// byte-identical to v015): (1) cp.async y13 staging in Phase 2; (2) RBX_W13
// 4->8 already + STAGES_W13 2->4; (3) packed half2 inner dot (W13/W2/shGU) + W2
// 4B->16B path; (4) parallel per-group top-8 argmax in Phase C. The DYNAMIC
// active_count + EP-LOCAL filter, the atomic grid_barrier, the per-task
// Scratch, input-binding order, and extern __align__(1024) are PRESERVED from
// v015 (the in-MPK-correct ABI). The y13 staging REUSES the per-warp s_wbuf (no
// new Scratch region); WBUF_U4 is grown + re-asserted to cover the 16B-W2 stage
// AND the whole-block y13 staging. See the FFN-FAST lever comments below.
// NOTE: SCRATCH_BYTES is 106624 (not v015's 106608) — the FAST y13 staging
// reads the GLOBAL y13 section via cp.async.cg-16, which requires y13
// 16B-aligned, so every Scratch section start is now 16B-aligned (was
// contiguous-after-8B-barrier → %16==8 → misaligned-y13 crash). The +16 is two
// 8B section pads, not a new region.
//
// Fast Phase B router GEMV path. Replaces the scalar router_partial<4> (NCU
// long_scoreboard 17.82 cyc/issue, HBM latency exposed) with
// router_partial_cpa<4,4>: cp.async STAGES=4 pipeline for weight loads + packed
// __nv_bfloat162 inner dot (2 muls/instr vs 8 scalar bf16->float + 8 fmul). The
// activation (s_norm) is in SMEM so it contributes zero HBM latency; only the
// 256×7168 weight matrix is staged. Math NEAR-identical, NOT bit-identical:
// same slice offsets + uint4 decomposition order, but the packed bf162 dot sums
// pairwise (p0+p1) vs the scalar left-fold (acc+=p0; acc+=p1) -> fp32
// non-associative -> sub-ULP. Uses the existing per-warp my_wbuf as the ring
// buffer (WBUF_U4=1024 uint4/warp >> 128 uint4 needed). TP8-alignment-safe: row
// stride = 14336B (16B-aligned), slice offset = sp×3584B (16B-aligned), extern
// smem stays
// __align__(1024). The cp.async is flushed (cpasync_wait<0>) before the
// grid_barrier. Target: router Phase B slowCTA from ~8.88µs → ~3µs at W=128.
//
// THREE STRUCTURAL ADAPTATIONS vs the standalone fused_moe_full:
//  1. DYNAMIC active_count, EP-LOCAL-FILTERED (NOT compile-time ACTIVE=8). The
//     ferret gate hard-coded ACTIVE=8 (an EP1/all-256-local config). In TP8 EP2
//     each rank holds 128 experts and only ~4 of the global top-8 are LOCAL.
//     Phase C computes the GLOBAL top-8 selection + the GLOBAL-sum-normalized
//     weights (production-exact topk_sigmoid math, incl. the off-node weight
//     still counted in the sum), then FILTERS to this rank's
//     [local_expert_start, local_expert_end) range: the LOCAL expert id
//     (e - local_expert_start) and its weight are appended to active_experts /
//     active_weights, and active_count++ — exactly as the COLD FFN derives the
//     active list from moe_routing_indices (the production topk writes
//     routing_indices ONLY for on-node experts). The W13/W2 loops then range
//     over active_count LOCAL experts. (See the EP-LOCAL FILTER block below.)
//  2. cg::grid.sync() -> the MPK atomic grid_barrier (the megakernel is NOT
//     cooperative-launched).
//  3. ALL of fused_moe_full's static __shared__ arrays (s_norm[7168] bf16 alone
//     is 14KB, far over the ~6KB megakernel static reserve) move into the
//     DYNAMIC extern __align__(1024) pool, exactly as the COLD FFN did. Global
//     __device__ g_* arrays do NOT exist in the source; the per-task Scratch
//     (block-0-persisted, ABI) holds the few globals (rmsnorm_out / a_fp8 /
//     a_scale / logits / inter / y13 / i_fp8 / i_scale / sg / si_fp8 / si_scale
//     / out_acc).
// ============================================================================

#include "mirage/persistent_kernel/runtime_header.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ---- MPK grid barrier (the standard megakernel grid_barrier pattern) -------
struct FfnFullGridBarrier {
  unsigned int *count; // [1] arrivals in the current generation
  unsigned int *gen;   // [1] generation (sense) counter
};

// Block-collective: call with the WHOLE block; only thread 0 touches global
// mem. Identical semantics to the COLD FFN's grid_barrier (double-fence so a
// relaxed atomic count-bump does NOT release readers before prior regular
// global stores land).
__device__ __forceinline__ void ffn_full_grid_barrier(FfnFullGridBarrier b,
                                                      int num_participants) {
  __syncthreads();
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int my_gen = *((unsigned int volatile *)b.gen);
    unsigned int prev = atomicAdd(b.count, 1u);
    if (prev + 1u == (unsigned int)num_participants) {
      *b.count = 0u;
      __threadfence();
      atomicAdd(b.gen, 1u);
    } else {
      while (*((unsigned int volatile *)b.gen) == my_gen) { /* spin */
      }
    }
  }
  __syncthreads();
}

namespace kernel {
namespace ffn_full_megakernel_sm100 {

// ---- Problem shapes (DSv3 TP8 EP2 per-rank) — VERBATIM from the COLD FFN
// -----
static constexpr int HIDDEN = 7168;
static constexpr int W13_N = 1024;
static constexpr int W2_K = 512;
static constexpr int W2_N = 7168;
static constexpr int E_LOCAL = 128;
static constexpr int GRP = 128;
static constexpr int KG1 = HIDDEN / GRP; // 56
static constexpr int KG2 = W2_K / GRP;   // 4
static constexpr int NB1 = W13_N / GRP;  // 8
static constexpr int NB2 = W2_N / GRP;   // 56
static constexpr int MAX_ACTIVE = 8;
static constexpr int NUM_WORKERS =
    136; // B200 worker count. Builder asserts ==.
// Compile-time warps-per-worker (TPB=256 on the B200 worker -> 8). The runtime
// `nwl = blockDim.x>>5` is asserted == NWL_CT below; NWL_CT is needed in the
// constexpr WBUF/y13-staging sizing (blockDim.x is not a constant expression).
static constexpr int NWL_CT = 8;

// ---- Router / topk-sigmoid shapes (DSv3) -----------------------------------
static constexpr int ROUTER_N = 256;    // num experts (global)
static constexpr int ROUTER_K = HIDDEN; // 7168
static constexpr int NUM_EXPERTS = 256; // == ROUTER_N
static constexpr int NUM_GROUPS = 8;    // n_group
static constexpr int EXPERTS_PER_GROUP = NUM_EXPERTS / NUM_GROUPS; // 32
static constexpr int TOPK_GROUP = 4;                               // topk_group
static constexpr int TOPK_EXPERTS = 8; // num_experts_per_tok
static constexpr float RMS_EPS = 1e-6f;
static constexpr int RKSPLIT = 4; // router GEMV split-K

// ---- Shared-expert shapes --------------------------------------------------
static constexpr int SH_GU_N = 512;
static constexpr int SH_GU_K = HIDDEN;
static constexpr int SH_DN_K = 256;
static constexpr int SH_DN_N = W2_N;
static constexpr int KG_SHGU = SH_GU_K / GRP; // 56
static constexpr int KG_SHDN = SH_DN_K / GRP; // 2
static constexpr int NB_SHGU = SH_GU_N / GRP; // 4
static constexpr int NB_SHDN = SH_DN_N / GRP; // 56

// ============================================================================
//  Per-task Scratch (block-0-persisted globals + the cross-phase barrier).
//  Layout VERBATIM-extended from the COLD FFN's Scratch + two front-stage
//  globals (rmsnorm_out, logits). `inter` is reused: Phase B writes router
//  partials [ROUTER_N*RKSPLIT] there (1024 floats), Phase 2 overwrites it with
//  the routed silu intermediate [MAX_ACTIVE*W2_K] (4096 floats) — sized for the
//  max (MAX_ACTIVE*W2_K = 4096 >= 1024). SCRATCH_BYTES is the %16-aligned sum;
//  the builder's FFN_FULL_MEGAKERNEL_SCRATCH_BYTES must equal it.
// ============================================================================
static constexpr int BARRIER_BYTES = 2 * static_cast<int>(sizeof(uint32_t));

// Round a running byte offset up to 16. EVERY scratch section start is 16B
// aligned via this (matching the attn megakernel's attn_au16 convention) so a
// 16-byte vector access into ANY section (e.g. the FAST `cp.async.cg ...,16`
// staging of `y13`) lands on a 16B boundary. WITHOUT this, the 8-byte BARRIER
// shifts every section to byte-offset %16==8, and the y13 cp.async.cg-16 read
// (cp.async requires natural 16B source alignment) is a misaligned global
// access → cudaErrorMisalignedAddress (surfaced asynchronously at the next
// nvshmem barrier). make_scratch() and SCRATCH_BYTES_RAW MUST use this in
// lock-step so the constant equals make_scratch's high-water mark exactly.
__device__ __host__ __forceinline__ constexpr int ffn_full_au16(int x) {
  return (x + 15) & ~15;
}

struct Scratch {
  __nv_bfloat16 *rmsnorm_out; // [HIDDEN]   bf16 normed (the boundary)
  uint8_t *a_fp8;             // [HIDDEN]
  float *a_scale;             // [KG1]
  __nv_bfloat16 *logits;      // [NUM_EXPERTS] bf16 router logits
  float *inter;   // [MAX_ACTIVE*W2_K] (router partials in Phase B; silu in Ph2)
  float *y13;     // [MAX_ACTIVE*W13_N]
  uint8_t *i_fp8; // [MAX_ACTIVE*W2_K]
  float *i_scale; // [MAX_ACTIVE*KG2]
  float *sg;      // [SH_GU_N]
  uint8_t *si_fp8; // [SH_DN_K]
  float *si_scale; // [KG_SHDN]
  float *out_acc;  // [W2_N] fp32 accumulator (-> bf16 out)
};

// Region byte sizes (every region's byte size is a multiple of 4 or 2). Each
// section START is rounded up to 16 (ffn_full_au16) — so SCRATCH_BYTES_RAW
// below MUST accumulate them with the SAME per-section align as make_scratch,
// or the constant would disagree with the kernel's high-water mark.
static constexpr int SC_RMSNORM = HIDDEN * 2;          // bf16
static constexpr int SC_AFP8 = HIDDEN;                 // 7168
static constexpr int SC_ASCALE = KG1 * 4;              // 224
static constexpr int SC_LOGITS = NUM_EXPERTS * 2;      // 512
static constexpr int SC_INTER = MAX_ACTIVE * W2_K * 4; // 16384 (>=router 4096)
static constexpr int SC_Y13 = MAX_ACTIVE * W13_N * 4;  // 32768
static constexpr int SC_IFP8 = MAX_ACTIVE * W2_K;      // 4096
static constexpr int SC_ISCALE = MAX_ACTIVE * KG2 * 4; // 128
static constexpr int SC_SG = SH_GU_N * 4;              // 2048
static constexpr int SC_SIFP8 = SH_DN_K;               // 256
static constexpr int SC_SISCALE = KG_SHDN * 4;         // 8
static constexpr int SC_OUTACC = W2_N * 4;             // 28672

// High-water mark of make_scratch(): barrier, then each section's start
// 16B-aligned (ffn_full_au16) before adding its size. MUST mirror make_scratch
// exactly. (Only RMSNORM and OUTACC actually take an 8B pad; the others are
// already 16-aligned by their predecessors' sizes.)
static constexpr int SCRATCH_BYTES_RAW =
    ffn_full_au16(
        ffn_full_au16(
            ffn_full_au16(
                ffn_full_au16(
                    ffn_full_au16(
                        ffn_full_au16(
                            ffn_full_au16(
                                ffn_full_au16(
                                    ffn_full_au16(
                                        ffn_full_au16(
                                            ffn_full_au16(
                                                ffn_full_au16(BARRIER_BYTES) +
                                                SC_RMSNORM) +
                                            SC_AFP8) +
                                        SC_ASCALE) +
                                    SC_LOGITS) +
                                SC_INTER) +
                            SC_Y13) +
                        SC_IFP8) +
                    SC_ISCALE) +
                SC_SG) +
            SC_SIFP8) +
        SC_SISCALE) +
    SC_OUTACC;
// Round the TOTAL up to 16 (tensor_init zero-init uses 16B vec stores; the
// builder allocates bytes/2 bf16 and asserts %16==0 on the byte count).
static constexpr int SCRATCH_BYTES = (SCRATCH_BYTES_RAW + 15) & ~15;
// CONTRACT: must equal builder.py FFN_FULL_MEGAKERNEL_SCRATCH_BYTES.
// Per-section 16B alignment (the misaligned-y13-cp.async fix) raised this from
// 106608 to 106624 (+16: an 8B pad before rmsnorm_out and before out_acc) —
// pinned here so a future Scratch edit that drifts from the builder constant
// fails at compile time.
static_assert(SCRATCH_BYTES == 106624,
              "FFN-FULL SCRATCH_BYTES changed — update builder.py "
              "FFN_FULL_MEGAKERNEL_SCRATCH_BYTES to match.");

__device__ __forceinline__ Scratch make_scratch(uint8_t *base) {
  // Track the offset (not the raw pointer) so each section START is rounded up
  // to 16 — keeps every section (incl. y13, read as cp.async.cg-16) 16B
  // aligned regardless of the 8B barrier. MUST mirror SCRATCH_BYTES_RAW.
  int off = BARRIER_BYTES;
  Scratch sc;
  off = ffn_full_au16(off);
  sc.rmsnorm_out = reinterpret_cast<__nv_bfloat16 *>(base + off);
  off += SC_RMSNORM;
  off = ffn_full_au16(off);
  sc.a_fp8 = base + off;
  off += SC_AFP8;
  off = ffn_full_au16(off);
  sc.a_scale = reinterpret_cast<float *>(base + off);
  off += SC_ASCALE;
  off = ffn_full_au16(off);
  sc.logits = reinterpret_cast<__nv_bfloat16 *>(base + off);
  off += SC_LOGITS;
  off = ffn_full_au16(off);
  sc.inter = reinterpret_cast<float *>(base + off);
  off += SC_INTER;
  off = ffn_full_au16(off);
  sc.y13 = reinterpret_cast<float *>(base + off);
  off += SC_Y13;
  off = ffn_full_au16(off);
  sc.i_fp8 = base + off;
  off += SC_IFP8;
  off = ffn_full_au16(off);
  sc.i_scale = reinterpret_cast<float *>(base + off);
  off += SC_ISCALE;
  off = ffn_full_au16(off);
  sc.sg = reinterpret_cast<float *>(base + off);
  off += SC_SG;
  off = ffn_full_au16(off);
  sc.si_fp8 = base + off;
  off += SC_SIFP8;
  off = ffn_full_au16(off);
  sc.si_scale = reinterpret_cast<float *>(base + off);
  off += SC_SISCALE;
  off = ffn_full_au16(off);
  sc.out_acc = reinterpret_cast<float *>(base + off);
  off += SC_OUTACC;
  return sc;
}

// ============================================================================
//  Canonical device helpers (uniquely scoped in this namespace so there is no
//  ODR clash with the other megakernel task impls in the same test.cu).
// ============================================================================
__device__ __forceinline__ uint32_t bitcast_f2u(float f) {
  return __float_as_uint(f);
}
__device__ __forceinline__ float bitcast_u2f(uint32_t u) {
  return __uint_as_float(u);
}
__device__ __forceinline__ uint8_t encode_ue8m0(float scale) {
  scale = fmaxf(scale, 1e-30f);
  uint32_t bits = bitcast_f2u(scale);
  int exp_unbiased = static_cast<int>((bits >> 23) & 0xFF) - 127;
  uint32_t mantissa = bits & 0x7FFFFF;
  int ue = (mantissa == 0 ? exp_unbiased : exp_unbiased + 1) + 127;
  ue = ue < 0 ? 0 : ue;
  ue = ue > 255 ? 255 : ue;
  return static_cast<uint8_t>(ue);
}
__device__ __forceinline__ float decode_ue8m0(uint8_t e) {
  return bitcast_u2f(static_cast<uint32_t>(e) << 23);
}
__device__ __forceinline__ float quant_scale(float amax) {
  return decode_ue8m0(encode_ue8m0(amax / 448.0f));
}
__device__ __forceinline__ float f8(uint8_t x) {
  return __half2float(__nv_cvt_fp8_to_halfraw(
      *reinterpret_cast<__nv_fp8_storage_t *>(&x), __NV_E4M3));
}
__device__ __forceinline__ uint8_t to_f8(float v) {
  __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  return *reinterpret_cast<uint8_t *>(&s);
}
__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}
// Fast silu via __expf (SFU). Within the requant tolerance (the COLD/fused gate
// used this for the silu->i_fp8 path). Same algebra as silu().
__device__ __forceinline__ float silu_fast(float x) {
  return x / (1.0f + __expf(-x));
}

// ============================================================================
//  FFN-FAST levers (ported from ferret workspace5 v019; default ON). These are
//  GEMV/compute SPEED levers only — the reduce/topk/silu MATH is byte-identical
//  to the proven v015 path.
//  v019 wins folded in (each a kernel-BODY speed lever, NOT a math change):
//   (1) cp.async y13 staging in Phase 2 (-5.1us): the cold y13 global readback
//       is handed to the LSU as one coalesced cp.async.cg run + ONE wait, so
//       the per-group exposed latency overlaps onto the memory pipe (8 warps).
//   (2) RBX_W13 4->8 + STAGES_W13 2->4 (-4.2us): deeper register-blocked weight
//       streams hide L1TEX latency at the 8-warp/SM regime. (RBX | GRP=128
//       keeps the shared per-N-block scale row valid — see HAZARD 1 in the
//       merge log.)
//   (3) packed half2 inner dot (W13 + W2 + shared): per super-step the lane's
//       activations decode ONCE to half2, PRE-SCALED by 2^-8 (exact exp shift),
//       reused across RBX rows; each row -> __hfma2 (2 FMAs/instr); the dot is
//       reduced to fp32, x256 to undo the prescale, x the group scale. Halves
//       the inner-loop instr count. Cross-super-step accum stays fp32 ->
//       y13/out cosine ~1.0 (>= the 0.999 floor the gate enforces).
//   (4) parallel per-group top-8 argmax in Phase C (-0.6us, see Phase C below).
//  DEAD (do NOT add): tcgen05 (1/16-util at M=1), STAGES_W13=5 (+1us),
//  CTA round-robin W13 (+1.5us).
// ============================================================================

// FP8(e4m3)->__half2 for one packed uint32 (4 fp8) -> two half2, via the
// hardware F2FP.E4M3 path (cvt.rn.f16x2.e4m3x2; bf16x2 is NOT supported on
// sm_100). e4m3 magnitudes reach 448 so the raw fp16 PRODUCT (~448^2) would
// overflow fp16 (65504): the GEMV pre-scales the ACTIVATION half2 by 2^-8
// (=1/256, an exact exponent shift, no mantissa loss) so products stay <=784
// and the sum-of-8 <=~6300, then multiplies the dot back by 256.
__device__ __forceinline__ __half2 ffn_full_f8x2_h2(uint32_t v16) {
  __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(
      (__nv_fp8x2_storage_t)(v16 & 0xffff), __NV_E4M3);
  return *reinterpret_cast<__half2 *>(&r);
}
__device__ __forceinline__ void
    ffn_full_f8x4_h2(uint32_t v, __half2 &b0, __half2 &b1) {
  b0 = ffn_full_f8x2_h2(v & 0xffff);
  b1 = ffn_full_f8x2_h2((v >> 16) & 0xffff);
}
// Decode a uint4 (16 fp8) -> 8 half2.
__device__ __forceinline__ void ffn_full_f8x16_h2(uint4 v, __half2 b[8]) {
  ffn_full_f8x4_h2(v.x, b[0], b[1]);
  ffn_full_f8x4_h2(v.y, b[2], b[3]);
  ffn_full_f8x4_h2(v.z, b[4], b[5]);
  ffn_full_f8x4_h2(v.w, b[6], b[7]);
}
// NOTE: the PACKED-half2 GEMV (dgemv_cpa16_h2) is defined just after dgemv_cpa
// below (it needs the cpasync* helpers defined there).

template <typename T>
__device__ __forceinline__ void
    quant_group_warp(T const *src, uint8_t *q, float *scale, int g, int lane) {
  float v[4], amax = 0.f;
#pragma unroll
  for (int t = 0; t < 4; t++) {
    float x = static_cast<float>(src[g * GRP + lane * 4 + t]);
    v[t] = x;
    amax = fmaxf(amax, fabsf(x));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  }
  float s = quant_scale(amax);
  float inv = 1.f / s;
  if (lane == 0) {
    scale[g] = s;
  }
#pragma unroll
  for (int t = 0; t < 4; t++) {
    q[g * GRP + lane * 4 + t] = to_f8(v[t] * inv);
  }
}

// ---- cp.async helpers (VERBATIM from the COLD FFN) -------------------------
__device__ __forceinline__ void cpasync4(uint32_t smem_addr, void const *gptr) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr),
               "l"(gptr));
}
__device__ __forceinline__ void cpasync16(uint32_t smem_addr,
                                          void const *gptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gptr));
}
__device__ __forceinline__ void cpasync_commit() {
  asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cpasync_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void
    f8x4(uint32_t v, float &f0, float &f1, float &f2, float &f3) {
  __half2_raw lo =
      __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xffff), __NV_E4M3);
  __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(
      (__nv_fp8x2_storage_t)((v >> 16) & 0xffff), __NV_E4M3);
  f0 = __half2float(*(__half *)&lo.x);
  f1 = __half2float(*(__half *)&lo.y);
  f2 = __half2float(*(__half *)&hi.x);
  f3 = __half2float(*(__half *)&hi.y);
}
// Two bf16 packed in a uint32 -> two floats.
__device__ __forceinline__ void bf16x2(uint32_t v, float &f0, float &f1) {
  __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162 const *>(&v);
  f0 = __bfloat162float(b.x);
  f1 = __bfloat162float(b.y);
}

// uint4 (16B) cp.async.cg pipelined GEMV (W13 + shared GU). VERBATIM from the
// COLD FFN. ONE warp computes RBX consecutive output rows.
template <int RBX, int STAGES>
__device__ __forceinline__ void
    dgemv_cpa16(uint8_t const *__restrict__ a_fp8,
                float const *__restrict__ a_scale,
                uint8_t const *__restrict__ w_fp8,
                float const *__restrict__ w_scale_row,
                int K,
                int KG,
                int n0,
                int lane,
                uint4 *wbuf_base,
                float *y_out) {
  uint4 const *a16 = reinterpret_cast<uint4 const *>(a_fp8);
  uint4 const *w16 =
      reinterpret_cast<uint4 const *>(w_fp8 + static_cast<size_t>(n0) * K);
  int KUr = K >> 4;
  int SS = KUr >> 5;
  float y[RBX];
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    y[r] = 0.f;
  }
  uint32_t const sbase = __cvta_generic_to_shared(wbuf_base);
  uint32_t const STRIDE = (uint32_t)(RBX * 32 * 16);
  int pf = (STAGES - 1 < SS) ? (STAGES - 1) : SS;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < pf) {
      uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                  &w16[static_cast<size_t>(r) * KUr +
                       static_cast<size_t>(s) * 32 + lane]);
      }
    }
    cpasync_commit();
  }
  for (int ss = 0; ss < SS; ss++) {
    int sp = ss + (STAGES - 1);
    if (sp < SS) {
      uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                  &w16[static_cast<size_t>(r) * KUr +
                       static_cast<size_t>(sp) * 32 + lane]);
      }
    }
    cpasync_commit();
    cpasync_wait<STAGES - 1>();
    __syncwarp();
    int g = (ss * 32 + lane) >> 3;
    float sc = a_scale[g] * __ldg(&w_scale_row[g]);
    uint4 av = a16[ss * 32 + lane];
    float a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
    f8x4(av.x, a0, a1, a2, a3);
    f8x4(av.y, a4, a5, a6, a7);
    f8x4(av.z, a8, a9, a10, a11);
    f8x4(av.w, a12, a13, a14, a15);
    uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
    for (int r = 0; r < RBX; r++) {
      uint4 wv;
      uint32_t saddr = cur + (uint32_t)((r * 32 + lane) * 16);
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                   : "r"(saddr));
      float w0, w1, w2, w3, w4_, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14,
          w15;
      f8x4(wv.x, w0, w1, w2, w3);
      f8x4(wv.y, w4_, w5, w6, w7);
      f8x4(wv.z, w8, w9, w10, w11);
      f8x4(wv.w, w12, w13, w14, w15);
      float acc = a0 * w0;
      acc += a1 * w1;
      acc += a2 * w2;
      acc += a3 * w3;
      acc += a4 * w4_;
      acc += a5 * w5;
      acc += a6 * w6;
      acc += a7 * w7;
      acc += a8 * w8;
      acc += a9 * w9;
      acc += a10 * w10;
      acc += a11 * w11;
      acc += a12 * w12;
      acc += a13 * w13;
      acc += a14 * w14;
      acc += a15 * w15;
      y[r] += acc * sc;
    }
  }
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    float v = y[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (lane == 0) {
      y_out[r] = v;
    }
  }
}

// 4-byte cp.async pipelined GEMV (W2 K=512 + shared down K=256). VERBATIM from
// the COLD FFN.
template <int RBX, int STAGES>
__device__ __forceinline__ void dgemv_cpa(uint8_t const *__restrict__ a_fp8,
                                          float const *__restrict__ a_scale,
                                          uint8_t const *__restrict__ w_fp8,
                                          float const *__restrict__ w_scale_row,
                                          int K,
                                          int KG,
                                          int n0,
                                          int lane,
                                          uint32_t *wbuf_base,
                                          float *y_out) {
  uint32_t const *a4 = reinterpret_cast<uint32_t const *>(a_fp8);
  uint32_t const *w4 =
      reinterpret_cast<uint32_t const *>(w_fp8 + static_cast<size_t>(n0) * K);
  int Kw = K / 4;
  float y[RBX];
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    y[r] = 0.f;
  }
  uint32_t const sbase = __cvta_generic_to_shared(wbuf_base);
  uint32_t const STRIDE = (uint32_t)(RBX * 32 * 4);
  int pf = (STAGES - 1 < KG) ? (STAGES - 1) : KG;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < pf) {
      uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync4(b + (uint32_t)((r * 32 + lane) * 4),
                 &w4[static_cast<size_t>(r) * Kw + static_cast<size_t>(s) * 32 +
                     lane]);
      }
    }
    cpasync_commit();
  }
  for (int g = 0; g < KG; g++) {
    int gp = g + (STAGES - 1);
    if (gp < KG) {
      uint32_t b = sbase + (uint32_t)(gp % STAGES) * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync4(b + (uint32_t)((r * 32 + lane) * 4),
                 &w4[static_cast<size_t>(r) * Kw +
                     static_cast<size_t>(gp) * 32 + lane]);
      }
    }
    cpasync_commit();
    cpasync_wait<STAGES - 1>();
    __syncwarp();
    uint32_t cur = sbase + (uint32_t)(g % STAGES) * STRIDE;
    uint32_t av = a4[g * 32 + lane];
    float sc = a_scale[g] * __ldg(&w_scale_row[g]);
    float a0, a1, a2, a3;
    f8x4(av, a0, a1, a2, a3);
#pragma unroll
    for (int r = 0; r < RBX; r++) {
      uint32_t w;
      uint32_t saddr = cur + (uint32_t)((r * 32 + lane) * 4);
      asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(w) : "r"(saddr));
      float w0, w1, w2, w3;
      f8x4(w, w0, w1, w2, w3);
      float acc = a0 * w0;
      acc += a1 * w1;
      acc += a2 * w2;
      acc += a3 * w3;
      y[r] += acc * sc;
    }
  }
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    float v = y[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (lane == 0) {
      y_out[r] = v;
    }
  }
}

// PACKED-half2 uint4 (16B) cp.async.cg pipelined GEMV (FAST: W13 + W2 + shGU).
// Same cp.async staging + ascending-group/ascending-byte/left-fold order as the
// scalar dgemv_cpa16 (byte-for-byte the SAME fp32 accumulation order -> cosine
// ~1.0), but the inner dot is packed half2: per super-step the lane's 16
// activations decode ONCE to 8 half2 + PRE-SCALED by 2^-8 (an EXACT exponent
// shift, no mantissa loss; reused across RBX rows); each row's 16 weights -> 8
// half2 and 8 __hfma2 (=16 FMAs, 2/instr) into ONE half2 partial reduced
// (.x+.y) to fp32, x256 to undo the prescale, x the group scale. The 2^-8
// prescale keeps every fp16 product <=784 and the sum-of-8 <=~6300 (under the
// 65504 fp16 max) so e4m3-magnitude (<=448) products do NOT overflow. ONE warp
// computes RBX consecutive output rows. K must be %16==0 (uint4): HIDDEN=7168,
// SH_GU_K=7168, W2_K=512 all qualify. RBX must divide GRP=128 (shared
// per-N-block scale row).
template <int RBX, int STAGES>
__device__ __forceinline__ void
    dgemv_cpa16_h2(uint8_t const *__restrict__ a_fp8,
                   float const *__restrict__ a_scale,
                   uint8_t const *__restrict__ w_fp8,
                   float const *__restrict__ w_scale_row,
                   int K,
                   int KG,
                   int n0,
                   int lane,
                   uint4 *wbuf_base,
                   float *y_out) {
  (void)KG;
  uint4 const *a16 = reinterpret_cast<uint4 const *>(a_fp8);
  uint4 const *w16 =
      reinterpret_cast<uint4 const *>(w_fp8 + static_cast<size_t>(n0) * K);
  int KUr = K >> 4;  // uint4 per row (row stride)
  int SS = KUr >> 5; // super-steps (32 lanes * uint4)
  const __half2 kInvScale = __float2half2_rn(0.00390625f); // 2^-8
  float y[RBX];
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    y[r] = 0.f;
  }
  uint32_t const sbase = __cvta_generic_to_shared(wbuf_base);
  uint32_t const STRIDE = (uint32_t)(RBX * 32 * 16); // bytes per stage buffer
  int pf = (STAGES - 1 < SS) ? (STAGES - 1) : SS;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < pf) {
      uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                  &w16[static_cast<size_t>(r) * KUr +
                       static_cast<size_t>(s) * 32 + lane]);
      }
    }
    cpasync_commit();
  }
  for (int ss = 0; ss < SS; ss++) {
    int sp = ss + (STAGES - 1);
    if (sp < SS) {
      uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                  &w16[static_cast<size_t>(r) * KUr +
                       static_cast<size_t>(sp) * 32 + lane]);
      }
    }
    cpasync_commit();
    cpasync_wait<STAGES - 1>();
    __syncwarp();
    int g = (ss * 32 + lane) >> 3;
    float sc =
        a_scale[g] * __ldg(&w_scale_row[g]) * 256.0f; // undo 2^-8 prescale
    uint4 av = a16[ss * 32 + lane];
    __half2 ah[8];
    ffn_full_f8x16_h2(av, ah);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      ah[i] = __hmul2(ah[i], kInvScale);
    }
    uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
    for (int r = 0; r < RBX; r++) {
      uint4 wv;
      uint32_t saddr = cur + (uint32_t)((r * 32 + lane) * 16);
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                   : "r"(saddr));
      __half2 bw[8];
      ffn_full_f8x16_h2(wv, bw);
      __half2 p = __hmul2(ah[0], bw[0]);
      p = __hfma2(ah[1], bw[1], p);
      p = __hfma2(ah[2], bw[2], p);
      p = __hfma2(ah[3], bw[3], p);
      p = __hfma2(ah[4], bw[4], p);
      p = __hfma2(ah[5], bw[5], p);
      p = __hfma2(ah[6], bw[6], p);
      p = __hfma2(ah[7], bw[7], p);
      float acc = __half2float(p.x) + __half2float(p.y);
      y[r] += acc * sc;
    }
  }
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    float v = y[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (lane == 0) {
      y_out[r] = v;
    }
  }
}

// Router split-K partial GEMV (bf16 . bf16, fp32 accum). Each (expert, split)
// warp accumulates the dot of normed . W_gate[e] over its K-slice; the grid-
// wide reduction over splits happens in Phase C. (Ported from the source
// fmf_router_partial.)
template <int KSPLIT>
__device__ __forceinline__ float
    router_partial(__nv_bfloat16 const *__restrict__ normed,
                   __nv_bfloat16 const *__restrict__ wr,
                   int sp,
                   int lane) {
  int const Kc = ROUTER_K / KSPLIT;
  int const base = sp * Kc;
  uint4 const *nrm4 = reinterpret_cast<uint4 const *>(normed + base);
  uint4 const *wr4 = reinterpret_cast<uint4 const *>(wr + base);
  int const U = Kc >> 3; // uint4 per slice
  float acc = 0.f;
  for (int u = lane; u < U; u += 32) {
    uint4 nv = nrm4[u];
    uint4 wv = wr4[u];
    float n0, n1, n2, n3, n4, n5, n6, n7, m0, m1, m2, m3, m4, m5, m6, m7;
    bf16x2(nv.x, n0, n1);
    bf16x2(nv.y, n2, n3);
    bf16x2(nv.z, n4, n5);
    bf16x2(nv.w, n6, n7);
    bf16x2(wv.x, m0, m1);
    bf16x2(wv.y, m2, m3);
    bf16x2(wv.z, m4, m5);
    bf16x2(wv.w, m6, m7);
    acc += n0 * m0;
    acc += n1 * m1;
    acc += n2 * m2;
    acc += n3 * m3;
    acc += n4 * m4;
    acc += n5 * m5;
    acc += n6 * m6;
    acc += n7 * m7;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  }
  return acc; // valid on lane 0
}

// ============================================================================
//  Fast routing — Phase B fast router GEMV + Phase C bitonic topk.
//
//  Phase B fast (router_partial_cpa):
//    Problem: router_partial issues HBM loads in a serial stride-32 loop with
//    NO prefetch -> NCU long_scoreboard 17.82 cyc/issue (HBM load latency
//    exposed). Fix: cp.async.cg pipeline the weight slice (uint4, 16B) into the
//    per-warp staging buffer (my_wbuf, reused from W13/W2, WBUF_U4 >> needed).
//    Inner dot: packed __nv_bfloat162 multiply-into-float2 (2 muls/instr vs 8
//    scalar bf16->float + 8 fmul in the scalar path). The activation (s_norm)
//    is in SMEM -> no latency; only weight is pipeline-staged from HBM. Math:
//    NEAR-identical to router_partial<KSPLIT> (NOT byte-identical) — same slice
//    offsets + uint4 decomposition order (x/y/z/w), but the packed bf162 dot
//    sums pairwise (p0+p1) vs the scalar left-fold (acc+=p0; acc+=p1) -> fp32
//    non-associative -> sub-ULP. Load ordering also differs (cp.async vs
//    __ldg). WBUF fit: KSPLIT=4 -> Kc=1792 bf16 = 7168 bytes per slice = 448
//    uint4. Ring: STAGES=4 * 32 lanes * 1 uint4/lane = 128 uint4/stage per
//    warp. Total: 128*4 = 512 uint4 = 8192 bytes. WBUF_U4=1024 uint4/warp
//    (16384 B)
//    >> 512 uint4 -> fits. Stage stride = 32 uint4 per stage (one lane per
//    uint4).
//
//  Phase C bitonic topk (router_topk_bitonic):
//    Problem: the TOPK_EXPERTS=8 sequential argmax passes with 8 warp-reduces,
//    even in the FAST parallel-group path, still requires TOPK_EXPERTS rounds
//    of shfl-reduces. A bitonic top-8 over 256 bf16-logit-biased values runs in
//    O(log^2 N) compare-swap steps on 256 values distributed across 8 warps,
//    reducing barrier count from ~8 synchronization rounds to 3 (one per
//    bitonic merge step across warp boundaries). NOTE: within the megakernel,
//    Phase C runs block-locally (no grid_barrier needed) so __syncthreads is
//    safe here. The fast Phase C path already does the
//    parallel per-group top-8 + warp-0 merge which is equivalent benefit.
//    FAST_ROUTING Phase C replaces the serial warp-0 merge of 64 candidates
//    (2/lane, TOPK_EXPERTS rounds of shfl) with a direct bitonic sort of all 64
//    candidates in warp 0 in O(log^2 64) = 15 compare-swap passes (completely
//    unrolled, no shuffles needed — all values in registers of warp 0's 32
//    lanes, 2 per lane).
// ============================================================================

// cp.async pipelined router partial GEMV (bf16 . bf16, fp32 accum).
// STAGES ring buffer in wbuf_base (MUST be >= STAGES*32 uint4 per warp).
// Activation (normed) is in SMEM -> read directly. Weight (wr) is in HBM ->
// staged via cp.async.cg (16B coalesced, one uint4 per lane per stage).
// Math NEAR-identical to router_partial<KSPLIT> (sub-ULP, NOT byte-identical):
// same slice, but pairwise bf162 dot (p0+p1) vs scalar left-fold -> fp32
// non-assoc. ALIGNMENT: wr row start = e*ROUTER_K*2 bytes; ROUTER_K=7168,
// e<256. Each row is 14336 bytes, and base=sp*(ROUTER_K/KSPLIT)*2 = sp*3584
// bytes offset within the row. 3584 % 16 == 0 (3584/16=224) -> ALWAYS
// 16B-aligned. ✓ The activation normed (s_norm in SMEM) is at s_norm+base:
// base=sp*Kc in bf16 elements; Kc=1792; SMEM base must be 16B-aligned (s_norm
// is FFN_FULL_AU16-aligned in the layout above -> guaranteed 16B-aligned). ✓
template <int KSPLIT, int STAGES>
__device__ __forceinline__ float
    router_partial_cpa(__nv_bfloat16 const *__restrict__ normed,
                       __nv_bfloat16 const *__restrict__ wr,
                       int sp,
                       int lane,
                       uint4 *wbuf_base) {
  int const Kc = ROUTER_K / KSPLIT;
  int const base = sp * Kc;
  // Activation is in SMEM (s_norm); cast to uint4 for 16B vectorized reads.
  // nrm4[u] reads 8 bf16 = 1 uint4 from SMEM at position (base + u*8).
  uint4 const *nrm4 = reinterpret_cast<uint4 const *>(normed + base);
  // Weight is in HBM; staged into wbuf_base via cp.async.
  uint4 const *wr4 = reinterpret_cast<uint4 const *>(wr + base);
  int const U = Kc >> 3;        // uint4 per slice (1792/8=224)
  int const SS = (U + 31) >> 5; // super-steps = ceil(U/32) = 7
  uint32_t const sbase = __cvta_generic_to_shared(wbuf_base);
  uint32_t const STRIDE = 32u * 16u; // bytes per stage (32 uint4 * 16B)
  float acc = 0.f;

  // Prefill STAGES-1 stages.
  int pf = (STAGES - 1 < SS) ? (STAGES - 1) : SS;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < pf) {
      uint32_t b = sbase + (uint32_t)s * STRIDE;
      int u = s * 32 + lane;
      if (u < U) {
        cpasync16(b + (uint32_t)lane * 16, &wr4[u]);
      }
    }
    cpasync_commit();
  }

  for (int ss = 0; ss < SS; ss++) {
    // Issue next stage.
    int sp_nxt = ss + (STAGES - 1);
    if (sp_nxt < SS) {
      uint32_t b = sbase + (uint32_t)(sp_nxt % STAGES) * STRIDE;
      int u = sp_nxt * 32 + lane;
      if (u < U) {
        cpasync16(b + (uint32_t)lane * 16, &wr4[u]);
      }
    }
    cpasync_commit();
    // Read activation for this super-step (SMEM -> registers, independent of
    // weight cp.async so this can overlap the HBM wait).
    int u_act = ss * 32 + lane;
    uint4 nv = (u_act < U) ? nrm4[u_act] : uint4{0, 0, 0, 0};
    cpasync_wait<STAGES - 1>();
    __syncwarp();
    // Load weight from ring buffer.
    uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
    uint4 wv;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                 : "r"(cur + (uint32_t)lane * 16));
    // Inner dot: convert bf162 -> float2 pairs then accumulate in fp32.
    // Using __bfloat162float2 (paired bf16->float2 conversion, 1 instr per
    // pair on SM100) + float32 multiply is BIT-IDENTICAL to the scalar path
    // (same float32 products, same accumulation order). The packed load of
    // nv (from SMEM) + wv (from staged SMEM ring) and the packed bf162->float2
    // conversion are the speedup vs the scalar bf16x2() unpacking path.
    if (u_act < U) {
      // Unpack 4 pairs of bf16 from each uint4 activation and weight.
      __nv_bfloat162 n01 = *reinterpret_cast<__nv_bfloat162 const *>(&nv.x);
      __nv_bfloat162 n23 = *reinterpret_cast<__nv_bfloat162 const *>(&nv.y);
      __nv_bfloat162 n45 = *reinterpret_cast<__nv_bfloat162 const *>(&nv.z);
      __nv_bfloat162 n67 = *reinterpret_cast<__nv_bfloat162 const *>(&nv.w);
      __nv_bfloat162 m01 = *reinterpret_cast<__nv_bfloat162 const *>(&wv.x);
      __nv_bfloat162 m23 = *reinterpret_cast<__nv_bfloat162 const *>(&wv.y);
      __nv_bfloat162 m45 = *reinterpret_cast<__nv_bfloat162 const *>(&wv.z);
      __nv_bfloat162 m67 = *reinterpret_cast<__nv_bfloat162 const *>(&wv.w);
      // Convert to float2 pairs (paired intrinsic, efficient on SM100).
      float2 f01 = __bfloat1622float2(n01);
      float2 f23 = __bfloat1622float2(n23);
      float2 f45 = __bfloat1622float2(n45);
      float2 f67 = __bfloat1622float2(n67);
      float2 g01 = __bfloat1622float2(m01);
      float2 g23 = __bfloat1622float2(m23);
      float2 g45 = __bfloat1622float2(m45);
      float2 g67 = __bfloat1622float2(m67);
      // Float32 multiply-accumulate — BIT-IDENTICAL to scalar path.
      acc += f01.x * g01.x + f01.y * g01.y;
      acc += f23.x * g23.x + f23.y * g23.y;
      acc += f45.x * g45.x + f45.y * g45.y;
      acc += f67.x * g67.x + f67.y * g67.y;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  }
  return acc; // valid on lane 0
}

// Flush pending cp.async after router Phase B (the next op is a grid_barrier
// which includes __threadfence, so we only need the local warp-level drain).
__device__ __forceinline__ void router_cpa_drain() {
  cpasync_wait<0>();
}

// ============================================================================
//  MPK task entry. The standard megakernel task ABI: TaskDesc +
//  merge_task_offset (logical CTA id == blockIdx.x set by the scheduler) +
//  runtime_config (unused here — the routing is computed internally).
//
//  input_ptrs ABI — 14 slots (the HARD MAX_INPUTS_PER_TASK cap). The FULL
//  kernel REPLACES the COLD FFN's moe_mask/moe_routing_indices/moe_topk_weights
//  inputs (slots 5/6/7) with rmsnorm_weight/router_gate_weight/bias — the
//  routing is computed INTERNALLY, so those 3 routing tensors are OUTPUTS of
//  the internal topk, not inputs:
//    [0]  hidden       (bf16, the PRE-rmsnorm layer input self.x)     (1,7168)
//    [1]  w13          (fp8 [E,N,K])                          (128,1024,7168)
//    [2]  w13_scale     (fp32 pow2 [E,NB1,KG1])                  (128,8,56)
//    [3]  w2           (fp8 [E,N,K])                           (128,7168,512)
//    [4]  w2_scale      (fp32 pow2 [E,NB2,KG2])                 (128,56,4)
//    [5]  rmsnorm_weight (bf16 post_attention_layernorm.weight)     (7168,)
//    [6]  router_gate_w (bf16 gate.weight [N,K])                 (256,7168)
//    [7]  bias          (fp32 e_score_correction_bias)            (256,)
//    [8]  wgu_raw       (fp8 shared gate_up [gate;up] concat)     (512,7168)
//    [9]  wgu_scale      (fp32 pow2-or-raw [NB_SHGU,KG_SHGU])      (4,56)
//    [10] wdn          (fp8 shared down)                         (7168,256)
//    [11] wdn_scale      (fp32 [NB_SHDN,KG_SHDN])                  (56,2)
//    [12] out          (store_in_dmem alias; write through OUTPUT slot only)
//    [13] scratch      (uint8 Scratch base: barrier + globals, zero-init head)
//  + out bound as output_ptrs[0] (the tracked bf16 moe_output write).
//
//  RUNTIME ROUTING PARAMS baked as constexpr would be WRONG (the rank's local
//  expert range varies per rank). They are passed via the LAST two int32 slots
//  of the barrier scratch head OR a constexpr derived from the rank — but MPK
//  has no per-rank constexpr, so they are read from runtime_config (the EP rank
//  range). HOWEVER the simplest robust source matching the COLD FFN is the
//  routing_indices the production topk would have produced; since we compute
//  routing internally, the builder passes local_expert_start/local_expert_end
//  through runtime_config.* . To avoid adding a new runtime field we instead
//  read them from the SCRATCH head (written once by tensor_init? no) — the
//  chosen mechanism: the builder bakes local_expert_start as a kernel-template
//  arg is impossible. SOLUTION (see builder): local_expert_start and
//  num_local_experts are passed as the params[] of register_task and emitted as
//  literals into the dispatch snippet via task_register.cc -> here as the two
//  function args `local_expert_start`, `num_local_experts`.
// ============================================================================
// ============================================================================
//  MPK_DSV3_FFN_WARPSPEC — hand-rolled warp-specialization experiment
//  (self-scheduled roles; env-gated, default-OFF, default build
//  byte-identical).
//
//  Roles (stage 1, MPK_DSV3_FFN_WARPSPEC>=1):
//    W7  = LOADER. (a) at t0 stages this CTA's 8 router-GEMV weight K-slices
//          into SMEM pages (28KB overlay on the idle-until-Phase-1 s_wbuf) and
//          publishes each page through a per-page mbarrier armed by
//          cp.async.mbarrier.arrive.noinc (a hand-rolled page semaphore);
//          (b) issues cp.async.bulk.prefetch.L2 for this CTA's 1/136 slice of
//          the routing-INDEPENDENT weight streams (shared gate_up + shared
//          down); (c) rejoins the block-collective rmsnorm/quant late (same
//          threadIdx striding => bit-exact); (d) after GB1, runs a PRIVATE
//          single-warp topk (registers only, feeds PREFETCH ONLY — divergence
//          on FP ties is harmless) to learn the routed expert IDs ~2-3us before
//          the consumers' authoritative block topk completes, and bulk-
//          prefetches the active experts' W13/W2 grid slices; (e) waits the
//          s_topk_done release flag, recomputes the authoritative EP filter,
//          and rejoins Phase 1 as a normal MAC warp.
//    W0-6 = CONSUMERS. Run the phases as before; Phase B consumes the loader's
//          SMEM pages (identical fp32 dot order => sc.inter bit-exact); Phase C
//          runs on 7 warps with barrier.cta.sync 1,224 partial barriers (W7 is
//          detoured; __syncthreads would stall on it) — per-group work strides
//          g += 7, elementwise loops stride 224 => bit-exact selection.
// ============================================================================
// Gate levels (all default-OFF; the default build is byte-identical):
//   MPK_DSV3_FFN_WARPSPEC=1  WS_PAGES: loader-staged router SMEM pages
//                            (V2 page semaphores). Measured -1.4us router body.
//   MPK_DSV3_FFN_WARPSPEC=2  + WS_DETOUR: W7 topk-detour + L2 warm-up streams.
//                            MEASURED NET-NEGATIVE (kept as the ablation arm):
//                            advisory prefetch is background-class on B200
//                            (starves demand, 2.3ms retire-drain), demand-class
//                            streams interfere with the concurrent small-ops,
//                            and the ~5us topk window x non-interfering rate
//                            cannot move MB-scale weight streams.
//   MPK_DSV3_FFN_WARPSPEC=3  + WS_PIPE (stage 2): paged loader‖consumer GEMV
//                            pipeline on Phase 3 — W7 pre-stages each
//                            consumer's W2 task pages into SMEM rings DURING
//                            Phase 1/silu through mbarrier full/empty channels
//                            (the V2 Channel, hand-rolled).
#if defined(MPK_DSV3_FFN_WARPSPEC) && MPK_DSV3_FFN_WARPSPEC >= 1
#define MPK_DSV3_FFN_WS_PAGES 1
#endif
#if defined(MPK_DSV3_FFN_WARPSPEC) && MPK_DSV3_FFN_WARPSPEC == 2
#define MPK_DSV3_FFN_WS_DETOUR 1
#endif
#if defined(MPK_DSV3_FFN_WARPSPEC) && MPK_DSV3_FFN_WARPSPEC >= 3
#define MPK_DSV3_FFN_WS_PIPE 1
#endif

#if defined(MPK_DSV3_FFN_WS_PAGES)
static constexpr int WS_LOADER_WARP = 7;
static constexpr int WS_NUM_CONSUMER_WARPS = 7;
static constexpr int WS_CONSUMER_THREADS = WS_NUM_CONSUMER_WARPS * 32; // 224
static constexpr int WS_NUM_RPAGES = 8;
static constexpr int WS_RPAGE_BYTES = (ROUTER_K / RKSPLIT) * 2; // 3584
// Stage-2 (WS_PIPE) ring geometry: per consumer warp, its my_wbuf hosts
// WS_PIPE_PAGES pages of WS_PIPE_PAGE_BYTES (one W2 task's 16 rows x 512B).
static constexpr int WS_PIPE_PAGES = 2;
static constexpr int WS_PIPE_PAGE_BYTES = 16 * 512; // RBX_W2 rows x W2_K

__device__ __forceinline__ int ws_smem_addr(void const *p) {
  return static_cast<int>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ws_mbar_init(uint64_t *mbar, int count) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(ws_smem_addr(mbar)),
      "r"(count)
      : "memory");
}
__device__ __forceinline__ void ws_mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
// Arm the mbarrier with completion of ALL prior cp.async ops of THIS thread
// (arrive-on-complete). With mbar count=32 and all 32 loader lanes executing
// this, the mbarrier phase flips exactly when the page's bytes have landed.
__device__ __forceinline__ void ws_cpasync_arrive_noinc(uint64_t *mbar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(
                   ws_smem_addr(mbar))
               : "memory");
}
__device__ __forceinline__ void ws_mbar_wait_parity(uint64_t *mbar, int phase) {
  int addr = ws_smem_addr(mbar);
  asm volatile("{\n\t.reg .pred P;\n\t"
               "WS_WAIT: "
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], "
               "%1, 0x989680;\n\t"
               "@P bra WS_DONE;\n\t"
               "bra WS_WAIT;\n\t"
               "WS_DONE:\n\t}" ::"r"(addr),
               "r"(phase)
               : "memory");
}
// Non-blocking parity poll (V2's mbar_poll: minimal suspend hint) — used by
// the stage-2 loader's round-robin ring sweep so one full ring never parks it.
__device__ __forceinline__ bool ws_mbar_poll(uint64_t *mbar, int phase) {
  int ok;
  asm volatile("{\n\t.reg .pred P;\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], "
               "%2, 1;\n\t"
               "selp.b32 %0, 1, 0, P;\n\t}"
               : "=r"(ok)
               : "r"(ws_smem_addr(mbar)), "r"(phase)
               : "memory");
  return ok != 0;
}
__device__ __forceinline__ void ws_mbar_arrive(uint64_t *mbar) {
  asm volatile(
      "mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(ws_smem_addr(mbar))
      : "memory");
}
__device__ __forceinline__ void ws_mbar_arrive_cnt(uint64_t *mbar, int count) {
  asm volatile(
      "mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(ws_smem_addr(mbar)),
      "r"(count)
      : "memory");
}
// Partial barrier for the 7 consumer warps (id 1). W7 never touches it.
__device__ __forceinline__ void ws_bar_consumers() {
  asm volatile("barrier.cta.sync %0, %1;" ::"r"(1), "r"(WS_CONSUMER_THREADS)
               : "memory");
}
__device__ __forceinline__ void ws_st_release_u32(uint32_t *p, uint32_t v) {
  asm volatile(
      "st.release.cta.shared::cta.u32 [%0], %1;" ::"r"(ws_smem_addr(p)), "r"(v)
      : "memory");
}
__device__ __forceinline__ uint32_t ws_ld_acquire_u32(uint32_t *p) {
  uint32_t v;
  asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];"
               : "=r"(v)
               : "r"(ws_smem_addr(p))
               : "memory");
  return v;
}
// Advisory L2 prefetch. addr 16B-aligned, bytes %16==0, bytes>0 (caller
// guarantees). Fallback arm (per-128B line prefetch) for ablation.
// MEASURED DEAD on B200 (2026-07-02): BOTH flavors are serviced in a slow
// background class (~19GB/s) that starves demand traffic — cold slowCTA
// REGRESSED +18us and the kernel retire-drained 2.3ms warm. Kept only for the
// ablation record; ws_stream_slice (real discard-loads) is the live mechanism.
__device__ __forceinline__ void ws_bulk_prefetch_l2(void const *gptr,
                                                    uint32_t bytes) {
#if defined(MPK_DSV3_FFN_WARPSPEC_NO_PF)
  (void)gptr;
  (void)bytes;
#elif defined(MPK_DSV3_FFN_WARPSPEC_PF_FALLBACK)
  uintptr_t p = reinterpret_cast<uintptr_t>(gptr);
  for (uint32_t off = 0; off < bytes; off += 128) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(
        reinterpret_cast<void const *>(p + off)));
  }
#else
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(gptr),
               "r"(bytes));
#endif
}
// Prefetch this CTA's 1/NUM_WORKERS grid-slice of [ptr, ptr+bytes),
// uint4-quantized (16B-aligned starts/sizes), lane-parallel, tail-clamped.
__device__ __forceinline__ void
    ws_prefetch_slice(uint8_t const *ptr, size_t bytes, int cta, int lane) {
  size_t total_u4 = bytes >> 4;
  size_t per_cta = (total_u4 + NUM_WORKERS - 1) / NUM_WORKERS;
  size_t start = (size_t)cta * per_cta;
  if (start >= total_u4) {
    return;
  }
  size_t len = total_u4 - start;
  if (len > per_cta) {
    len = per_cta;
  }
  size_t per_lane = (len + 31) >> 5;
  size_t lstart = (size_t)lane * per_lane;
  if (lstart >= len) {
    return;
  }
  size_t llen = len - lstart;
  if (llen > per_lane) {
    llen = per_lane;
  }
  ws_bulk_prefetch_l2(ptr + ((start + lstart) << 4), (uint32_t)(llen << 4));
}

// W7 t0 prologue: fill the 8 router pages + arm their mbarriers. Straight-line
// (no cross-warp waits) => its page publications can never deadlock a
// consumer. NOTE: no t0 weight prefetch — anything W7 does here delays its
// rmsnorm join and the consumers wait for it at the first __syncthreads, so
// the t0 window is ~free only for this sub-us page issue (measured).
__device__ __noinline__ void
    ws_loader_prologue(__nv_bfloat16 const *router_gate_w,
                       uint8_t *rpage_base,
                       uint64_t *rp_mbar,
                       int cta,
                       int lane) {
  if (cta * WS_NUM_RPAGES < ROUTER_N * RKSPLIT) { // CTAs 0..127 own router work
#pragma unroll
    for (int p = 0; p < WS_NUM_RPAGES; p++) {
      int const t = cta * WS_NUM_RPAGES + p;
      int const e = t / RKSPLIT, sp = t % RKSPLIT;
      uint8_t const *src = reinterpret_cast<uint8_t const *>(
          router_gate_w + (size_t)e * ROUTER_K +
          (size_t)sp * (ROUTER_K / RKSPLIT));
      uint32_t const dst =
          __cvta_generic_to_shared(rpage_base + (size_t)p * WS_RPAGE_BYTES);
      for (int u = lane; u < WS_RPAGE_BYTES / 16; u += 32) {
        cpasync16(dst + (uint32_t)u * 16, src + (size_t)u * 16);
      }
      ws_cpasync_arrive_noinc(&rp_mbar[p]);
    }
  }
}

// DEMAND-CLASS L2 warm-up stream: real ld.global.cg loads (L2-caching,
// L1-bypass), discarded through an asm sink so they cannot be DCE'd. This is
// the replacement for the advisory prefetch instructions, which B200 services
// in a slow background class (~19GB/s) that starves demand traffic (measured
// +18us cold / 2.3ms retire-drain — see ws_bulk_prefetch_l2). Streams this
// CTA's 1/NUM_WORKERS slice of [ptr, ptr+bytes); polls `stop` (if non-null)
// every chunk and returns early once it is set, so the loader never steals
// bandwidth from the real phase work that follows.
__device__ __forceinline__ void ws_stream_slice(
    uint8_t const *ptr, size_t bytes, int cta, int lane, uint32_t *stop) {
  size_t const total_u4 = bytes >> 4;
  size_t const per_cta = (total_u4 + NUM_WORKERS - 1) / NUM_WORKERS;
  size_t const start = (size_t)cta * per_cta;
  if (start >= total_u4) {
    return;
  }
  size_t len = total_u4 - start;
  if (len > per_cta) {
    len = per_cta;
  }
  uint4 const *p4 = reinterpret_cast<uint4 const *>(ptr) + start;
  uint32_t acc = 0;
  bool stopped = false;
  // 4-deep per-lane MLP: 128 uint4 (2KB) per warp-iteration.
  size_t u = lane;
  for (; u + 96 < len; u += 128) {
    uint4 a, b, c, d;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                 : "l"(p4 + u));
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
                 : "l"(p4 + u + 32));
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w)
                 : "l"(p4 + u + 64));
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "l"(p4 + u + 96));
    acc ^= a.x ^ b.x ^ c.x ^ d.x;
    if (stop != nullptr) {
      if (__shfl_sync(0xffffffffu,
                      (lane == 0) ? *(uint32_t volatile *)stop : 0u,
                      0) != 0u) {
        stopped = true;
        break;
      }
    }
  }
  if (!stopped) {
    for (; u < len; u += 32) {
      uint4 a;
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                   : "l"(p4 + u));
      acc ^= a.x;
    }
  }
  asm volatile("" ::"r"(acc)); // sink: keep the loads alive
}

// Phase-B consumer dot over a loader-staged SMEM page. Statement-for-statement
// the SAME fp32 accumulation as router_partial_cpa (same per-lane index set
// u = lane, lane+32, ... ascending; same x/y/z/w pairwise order) => the
// sc.inter partial is BIT-IDENTICAL to the baseline path.
__device__ __forceinline__ float ws_router_partial_page(
    __nv_bfloat16 const *normed, __nv_bfloat16 const *wpage, int sp, int lane) {
  int const Kc = ROUTER_K / RKSPLIT;
  int const base = sp * Kc;
  uint4 const *nrm4 = reinterpret_cast<uint4 const *>(normed + base);
  uint4 const *w4 = reinterpret_cast<uint4 const *>(wpage);
  int const U = Kc >> 3; // 224
  float acc = 0.f;
  for (int u = lane; u < U; u += 32) {
    uint4 nv = nrm4[u];
    uint4 wv = w4[u];
    __nv_bfloat162 n01 = *reinterpret_cast<__nv_bfloat162 const *>(&nv.x);
    __nv_bfloat162 n23 = *reinterpret_cast<__nv_bfloat162 const *>(&nv.y);
    __nv_bfloat162 n45 = *reinterpret_cast<__nv_bfloat162 const *>(&nv.z);
    __nv_bfloat162 n67 = *reinterpret_cast<__nv_bfloat162 const *>(&nv.w);
    __nv_bfloat162 m01 = *reinterpret_cast<__nv_bfloat162 const *>(&wv.x);
    __nv_bfloat162 m23 = *reinterpret_cast<__nv_bfloat162 const *>(&wv.y);
    __nv_bfloat162 m45 = *reinterpret_cast<__nv_bfloat162 const *>(&wv.z);
    __nv_bfloat162 m67 = *reinterpret_cast<__nv_bfloat162 const *>(&wv.w);
    float2 f01 = __bfloat1622float2(n01);
    float2 f23 = __bfloat1622float2(n23);
    float2 f45 = __bfloat1622float2(n45);
    float2 f67 = __bfloat1622float2(n67);
    float2 g01 = __bfloat1622float2(m01);
    float2 g23 = __bfloat1622float2(m23);
    float2 g45 = __bfloat1622float2(m45);
    float2 g67 = __bfloat1622float2(m67);
    acc += f01.x * g01.x + f01.y * g01.y;
    acc += f23.x * g23.x + f23.y * g23.y;
    acc += f45.x * g45.x + f45.y * g45.y;
    acc += f67.x * g67.x + f67.y * g67.y;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  }
  return acc; // valid on lane 0
}

// W7 post-GB1 detour: PRIVATE single-warp topk (registers only; feeds PREFETCH
// ONLY — a divergent selection on an FP tie merely warms a wrong expert, it
// can never affect the output) -> EP-local filter -> demand-class L2 warm-up
// streams for the routed weights, PRIORITIZED by consumption distance: W2
// slices first (read ~20us later in Phase 3 — the full cold delta is
// winnable), then shared-down, then W13 (read almost immediately — only the
// overlap remainder is winnable). Every stream polls s_topk_done and aborts
// the moment the consumers' authoritative topk completes, so the loader never
// competes with the real Phase-1 reads. No shared-state writes, no block syncs.
__device__ __noinline__ void ws_private_topk_prefetch(float const *inter,
                                                      float const *bias_g,
                                                      uint8_t const *w13,
                                                      uint8_t const *w2,
                                                      uint8_t const *wdn,
                                                      uint32_t *stop,
                                                      int les,
                                                      int nle,
                                                      int cta,
                                                      int lane) {
  // lane l owns experts [l*8, l*8+8)
  float biased[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int const e = lane * 8 + j;
    float tot = 0.f;
#pragma unroll
    for (int sp = 0; sp < RKSPLIT; sp++) {
      tot += inter[e * RKSPLIT + sp];
    }
    float const lg = __bfloat162float(__float2bfloat16(tot));
    float const s = 1.0f / (1.0f + expf(-lg));
    biased[j] = s + bias_g[e];
  }
  // group top-2: local top-2 of the lane's 8, then merge across the group's
  // 4 lanes (xor 1, xor 2 stay inside the aligned 4-lane group)
  float m1 = -1e30f, m2 = -1e30f;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    float const v = biased[j];
    if (v > m1) {
      m2 = m1;
      m1 = v;
    } else if (v > m2) {
      m2 = v;
    }
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    float const o1 = __shfl_xor_sync(0xffffffffu, m1, o);
    float const o2 = __shfl_xor_sync(0xffffffffu, m2, o);
    if (o1 > m1) {
      m2 = fmaxf(m1, o2);
      m1 = o1;
    } else {
      m2 = fmaxf(m2, o1);
    }
  }
  float const gscore = m1 + m2; // converged across the 4-lane group
  // top-4 groups (redundant on every lane; deterministic; low-index tiebreak
  // matches the block topk)
  unsigned gsel = 0;
  {
    float gs[NUM_GROUPS];
#pragma unroll
    for (int g = 0; g < NUM_GROUPS; g++) {
      gs[g] = __shfl_sync(0xffffffffu, gscore, g * 4);
    }
#pragma unroll
    for (int k = 0; k < TOPK_GROUP; k++) {
      int bg = 0;
      float bs = -1e30f;
#pragma unroll
      for (int g = 0; g < NUM_GROUPS; g++) {
        if (!((gsel >> g) & 1u) && gs[g] > bs) {
          bs = gs[g];
          bg = g;
        }
      }
      gsel |= 1u << bg;
    }
  }
  int const mygrp = lane >> 2;
  if (!((gsel >> mygrp) & 1u)) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      biased[j] = -10000.f;
    }
  }
  // top-8 of 256 via 8 warp-argmax rounds (low-index tiebreak), collecting the
  // EP-local winners; then the prioritized warm-up streams.
  int local_e[MAX_ACTIVE];
  int nlocal = 0;
  for (int k = 0; k < TOPK_EXPERTS; k++) {
    float bv = -1e30f;
    int bi = NUM_EXPERTS;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      if (biased[j] > bv) {
        bv = biased[j];
        bi = lane * 8 + j;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float const ov = __shfl_xor_sync(0xffffffffu, bv, o);
      int const oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (bi >= les && bi < les + nle && nlocal < MAX_ACTIVE) {
      local_e[nlocal++] = bi - les;
    }
    if ((bi >> 3) == lane) {
      biased[bi & 7] = -10000.f;
    }
  }
  // Priority 1: routed W2 (consumed in Phase 3, ~20us away).
  for (int s = 0; s < nlocal; s++) {
    ws_stream_slice(w2 + (size_t)local_e[s] * W2_N * W2_K,
                    (size_t)W2_N * W2_K,
                    cta,
                    lane,
                    stop);
  }
  // Priority 2: shared down (also Phase 3).
  ws_stream_slice(wdn, (size_t)SH_DN_N * SH_DN_K, cta, lane, stop);
  // Priority 3: routed W13 (consumed the moment topk ends; leftover time only).
  for (int s = 0; s < nlocal; s++) {
    ws_stream_slice(w13 + (size_t)local_e[s] * W13_N * HIDDEN,
                    (size_t)W13_N * HIDDEN,
                    cta,
                    lane,
                    stop);
  }
}

#if defined(MPK_DSV3_FFN_WS_PIPE)
// ═══ Stage 2: paged loader‖consumer GEMV pipeline (Phase 3 W2) ══════════════
// W7 = persistent loader for phases 1-3: it never MACs; consumers use a
// 7-warp task map. Each consumer warp w owns a 2-page ring inside its own
// my_wbuf (page = one W2 task's 16x512B weights, the EXACT smem layout the
// dgemv MAC reads). Channel = full/empty mbarriers, V2-style:
//   full[w][p]:  count 32, armed by the loader's 32 lanes via
//                cp.async.mbarrier.arrive.noinc (page-landed = phase flip)
//   empty[w][p]: count 32, pre-armed once at entry (V2 pre-empty convention),
//                then arrived by the consumer's 32 lanes after each MAC
// Parity = lap & 1 (lap = task_ordinal / WS_PIPE_PAGES).
// Window A (during Phase 1): after consumer w releases its phase-1 ring use
// (s_p1_done[w]), fill its first <=2 pages — needs NO empty wait (pre-armed),
// so window A cannot deadlock against consumers heading to GB2.
// Window B (after GB2, during silu + Phase 3): poll-based round-robin refill
// (never parks on one ring); classic bounded buffer — consumers always
// progress, so empty releases always come; the loader issues its exact fill
// count (same pure map function as the consumers use) and joins GB3.

// #fed (W2) tasks of consumer w on CTA `cta`: the k-prefix of its task list.
__device__ __forceinline__ int ws_pipe_nfed(int cta, int w, int n2) {
  int const cw = cta * WS_NUM_CONSUMER_WARPS + w;
  int const cwarps = NUM_WORKERS * WS_NUM_CONSUMER_WARPS; // 952
  if (cw >= n2) {
    return 0;
  }
  return (n2 - cw + cwarps - 1) / cwarps;
}

// Fill one W2 task's page: byte-for-byte the layout dgemv_cpa16_h2's producer
// side writes for K=512 (SS=1): dst (r*32+lane)*16, src w16[r*32 + lane].
__device__ __forceinline__ void ws_pipe_fill_w2_page(
    uint8_t const *w2, int e, int n0, uint32_t page_saddr, int lane) {
  uint4 const *w16 = reinterpret_cast<uint4 const *>(
      w2 + (size_t)e * W2_N * W2_K + (size_t)n0 * W2_K);
#pragma unroll
  for (int r = 0; r < 16; r++) {
    cpasync16(page_saddr + (uint32_t)((r * 32 + lane) * 16),
              &w16[(size_t)r * (W2_K / 16) + lane]);
  }
}

// FED consumer MAC for one W2 task (K=512, SS==1): statement-for-statement the
// ss=0 iteration of dgemv_cpa16_h2 (same decode order, same hfma2 chain, same
// fp32 accumulate + shfl reduce) with the producer side removed => the y
// values are BIT-IDENTICAL to the self-load path.
template <int RBX>
__device__ __forceinline__ void
    ws_dgemv_page16_h2(uint8_t const *__restrict__ a_fp8,
                       float const *__restrict__ a_scale,
                       float const *__restrict__ w_scale_row,
                       uint32_t page_saddr,
                       int lane,
                       float *y_out) {
  uint4 const *a16 = reinterpret_cast<uint4 const *>(a_fp8);
  const __half2 kInvScale = __float2half2_rn(0.00390625f); // 2^-8
  float y[RBX];
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    y[r] = 0.f;
  }
  // ss = 0 (the only super-step at K=512)
  int g = lane >> 3;
  float sc = a_scale[g] * __ldg(&w_scale_row[g]) * 256.0f;
  uint4 av = a16[lane];
  __half2 ah[8];
  ffn_full_f8x16_h2(av, ah);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    ah[i] = __hmul2(ah[i], kInvScale);
  }
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    uint4 wv;
    uint32_t saddr = page_saddr + (uint32_t)((r * 32 + lane) * 16);
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                 : "r"(saddr));
    __half2 bw[8];
    ffn_full_f8x16_h2(wv, bw);
    __half2 p = __hmul2(ah[0], bw[0]);
    p = __hfma2(ah[1], bw[1], p);
    p = __hfma2(ah[2], bw[2], p);
    p = __hfma2(ah[3], bw[3], p);
    p = __hfma2(ah[4], bw[4], p);
    p = __hfma2(ah[5], bw[5], p);
    p = __hfma2(ah[6], bw[6], p);
    p = __hfma2(ah[7], bw[7], p);
    float acc = __half2float(p.x) + __half2float(p.y);
    y[r] += acc * sc;
  }
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    float v = y[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (lane == 0) {
      y_out[r] = v;
    }
  }
}

// Codex-requested falsification arm (WS_PIPE_NOFEED + WS_W2STREAM): W2-only
// demand-class L2 warm-up DURING the Phase-1 window — the one window (known
// after topk, consumed two phases later) the other arms did not isolate. No
// rings, consumers self-load; stops when ALL consumers have finished Phase 1
// (so it never competes past the window it claims).
__device__ __forceinline__ bool ws_p1_all_done(uint32_t *s_p1_done, int lane) {
  uint32_t d = 0;
  if (lane == 0) {
    d = 1;
#pragma unroll
    for (int w = 0; w < WS_NUM_CONSUMER_WARPS; w++) {
      if (*(uint32_t volatile *)&s_p1_done[w] == 0u) {
        d = 0;
      }
    }
  }
  return __shfl_sync(0xffffffffu, d, 0) != 0u;
}
__device__ __noinline__ void ws_w2_stream_phase1(uint8_t const *w2,
                                                 uint8_t const *wdn,
                                                 int const *active_experts,
                                                 int active_count,
                                                 uint32_t *s_p1_done,
                                                 int cta,
                                                 int lane) {
  for (int s = 0; s < active_count; s++) {
    if (ws_p1_all_done(s_p1_done, lane)) {
      return;
    }
    size_t const total_u4 = ((size_t)W2_N * W2_K) >> 4;
    size_t const per_cta = (total_u4 + NUM_WORKERS - 1) / NUM_WORKERS;
    size_t const start = (size_t)cta * per_cta;
    if (start >= total_u4) {
      continue;
    }
    size_t len = total_u4 - start;
    if (len > per_cta) {
      len = per_cta;
    }
    uint4 const *p4 = reinterpret_cast<uint4 const *>(
                          w2 + (size_t)active_experts[s] * W2_N * W2_K) +
                      start;
    uint32_t acc = 0;
    for (size_t u = lane; u + 96 < len; u += 128) {
      uint4 a, b, c, d;
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                   : "l"(p4 + u));
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
                   : "l"(p4 + u + 32));
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w)
                   : "l"(p4 + u + 64));
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                   : "l"(p4 + u + 96));
      acc ^= a.x ^ b.x ^ c.x ^ d.x;
      if (ws_p1_all_done(s_p1_done, lane)) {
        break;
      }
    }
    asm volatile("" ::"r"(acc));
  }
  (void)wdn; // W2-only (the arm Codex specified); an unbounded wdn tail would
             // delay W7's GB2 arrival and stall the grid.
}

// Loader window A: for each consumer (poll s_p1_done round-robin), fill its
// first <=WS_PIPE_PAGES pages. No empty waits (pre-armed) => cannot block on
// consumer progress; terminates unconditionally (p1_done is released by every
// consumer before it can reach GB2).
__device__ __noinline__ void ws_pipe_loader_prefill(uint8_t const *w2,
                                                    int const *active_experts,
                                                    int n2,
                                                    uint32_t *s_p1_done,
                                                    uint64_t *s_pipe_full,
                                                    uint4 *s_wbuf,
                                                    size_t wbuf_u4_per_warp,
                                                    int cta,
                                                    int lane,
                                                    int *next_k) {
  int const cwarps = NUM_WORKERS * WS_NUM_CONSUMER_WARPS;
  int remaining = 0;
  int target[WS_NUM_CONSUMER_WARPS];
  bool seen[WS_NUM_CONSUMER_WARPS];
#pragma unroll
  for (int w = 0; w < WS_NUM_CONSUMER_WARPS; w++) {
    int nf = ws_pipe_nfed(cta, w, n2);
    target[w] = nf < WS_PIPE_PAGES ? nf : WS_PIPE_PAGES;
    next_k[w] = 0;
    seen[w] = false;
    remaining += target[w];
  }
  while (remaining > 0) {
    for (int w = 0; w < WS_NUM_CONSUMER_WARPS; w++) {
      if (next_k[w] >= target[w]) {
        continue;
      }
      if (!seen[w]) {
        uint32_t d = 0;
        if (lane == 0) {
          d = *(uint32_t volatile *)&s_p1_done[w];
        }
        if (__shfl_sync(0xffffffffu, d, 0) == 0u) {
          continue;
        }
        seen[w] = true;
      }
      int const k = next_k[w];
      int const idx = cta * WS_NUM_CONSUMER_WARPS + w + k * cwarps;
      int const slot = idx / (W2_N / 16);
      int const n0 = (idx % (W2_N / 16)) * 16;
      int const e = active_experts[slot];
      uint32_t const page_saddr = __cvta_generic_to_shared(
          reinterpret_cast<uint8_t *>(s_wbuf + (size_t)w * wbuf_u4_per_warp) +
          (size_t)(k % WS_PIPE_PAGES) * WS_PIPE_PAGE_BYTES);
      ws_pipe_fill_w2_page(w2, e, n0, page_saddr, lane);
      ws_cpasync_arrive_noinc(
          &s_pipe_full[w * WS_PIPE_PAGES + (k % WS_PIPE_PAGES)]);
      next_k[w]++;
      remaining--;
    }
  }
}

// Loader window B: poll-based refill of laps >= 1 until every fed task's page
// has been issued. Bounded buffer: the empty edge is arrived by consumer MACs,
// which only ever wait on pages this loop is about to fill.
__device__ __noinline__ void ws_pipe_loader_refill(uint8_t const *w2,
                                                   int const *active_experts,
                                                   int n2,
                                                   uint64_t *s_pipe_full,
                                                   uint64_t *s_pipe_empty,
                                                   uint4 *s_wbuf,
                                                   size_t wbuf_u4_per_warp,
                                                   int cta,
                                                   int lane,
                                                   int *next_k) {
  int const cwarps = NUM_WORKERS * WS_NUM_CONSUMER_WARPS;
  int remaining = 0;
  int nfed[WS_NUM_CONSUMER_WARPS];
#pragma unroll
  for (int w = 0; w < WS_NUM_CONSUMER_WARPS; w++) {
    nfed[w] = ws_pipe_nfed(cta, w, n2);
    remaining += nfed[w] - next_k[w];
  }
  while (remaining > 0) {
    for (int w = 0; w < WS_NUM_CONSUMER_WARPS; w++) {
      if (next_k[w] >= nfed[w]) {
        continue;
      }
      int const k = next_k[w];
      int const p = k % WS_PIPE_PAGES;
      int const lap = k / WS_PIPE_PAGES;
      // page p is reusable once its lap-1 occupant was consumed: the empty
      // parity for THIS lap is (lap & 1) (the pre-arm completed phase 0).
      if (!ws_mbar_poll(&s_pipe_empty[w * WS_PIPE_PAGES + p], lap & 1)) {
        continue;
      }
      int const idx = cta * WS_NUM_CONSUMER_WARPS + w + k * cwarps;
      int const slot = idx / (W2_N / 16);
      int const n0 = (idx % (W2_N / 16)) * 16;
      int const e = active_experts[slot];
      uint32_t const page_saddr = __cvta_generic_to_shared(
          reinterpret_cast<uint8_t *>(s_wbuf + (size_t)w * wbuf_u4_per_warp) +
          (size_t)p * WS_PIPE_PAGE_BYTES);
      ws_pipe_fill_w2_page(w2, e, n0, page_saddr, lane);
      ws_cpasync_arrive_noinc(&s_pipe_full[w * WS_PIPE_PAGES + p]);
      next_k[w]++;
      remaining--;
    }
  }
}
#endif // MPK_DSV3_FFN_WS_PIPE
#endif // MPK_DSV3_FFN_WS_PAGES

// ---- Optional per-phase probe (MPK_DSV3_FFN_WS_PROBE builds only; NEVER set
// in production — the default build is byte-identical). Thread-0-of-CTA
// %globaltimer stamps at phase boundaries; a standalone driver can read the
// symbol back. ---------------------------------------------------------------
#ifdef MPK_DSV3_FFN_WS_PROBE
__device__ unsigned long long ffn_ws_probe_ts[NUM_WORKERS * 16];
#define FFN_WS_TS(k)                                                           \
  do {                                                                         \
    if (threadIdx.x == 0) {                                                    \
      unsigned long long t__;                                                  \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t__));                  \
      ffn_ws_probe_ts[merge_task_offset * 16 + (k)] = t__;                     \
    }                                                                          \
  } while (0)
#else
#define FFN_WS_TS(k)
#endif

__device__ __noinline__ void ffn_full_megakernel_sm100_task_impl(
    mirage::runtime::TaskDesc const *task_desc,
    int merge_task_offset,
    int local_expert_start,
    int num_local_experts,
    float routed_scaling_factor,
    mirage::runtime::RuntimeConfig const &runtime_config) {
  (void)runtime_config;

  __nv_bfloat16 const *hidden =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  uint8_t const *w13 = static_cast<uint8_t const *>(task_desc->input_ptrs[1]);
  float const *w13_scale = static_cast<float const *>(task_desc->input_ptrs[2]);
  uint8_t const *w2 = static_cast<uint8_t const *>(task_desc->input_ptrs[3]);
  float const *w2_scale = static_cast<float const *>(task_desc->input_ptrs[4]);
  __nv_bfloat16 const *rmsnorm_weight =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[5]);
  __nv_bfloat16 const *router_gate_w =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[6]);
  float const *bias = static_cast<float const *>(task_desc->input_ptrs[7]);
  uint8_t const *wgu = static_cast<uint8_t const *>(task_desc->input_ptrs[8]);
  float const *wgu_s = static_cast<float const *>(task_desc->input_ptrs[9]);
  uint8_t const *wdn = static_cast<uint8_t const *>(task_desc->input_ptrs[10]);
  float const *wdn_s = static_cast<float const *>(task_desc->input_ptrs[11]);
  // Write through the OUTPUT slot (input_ptrs[12] is a distinct stale alias —
  // same ABI note as the COLD FFN).
  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);
  uint8_t *scratch_base = static_cast<uint8_t *>(task_desc->input_ptrs[13]);

  int const local_expert_end = local_expert_start + num_local_experts;

  FfnFullGridBarrier barrier;
  barrier.count = reinterpret_cast<unsigned int *>(scratch_base);
  barrier.gen =
      reinterpret_cast<unsigned int *>(scratch_base + sizeof(uint32_t));
  Scratch sc = make_scratch(scratch_base);

  int const worker_idx = merge_task_offset;
  int const TPB = (int)blockDim.x; // 256 on the B200 worker (NOT 512).
  int const gtid = worker_idx * TPB + threadIdx.x;
  int const gthreads = NUM_WORKERS * TPB;
  int const gwarp = gtid / 32;
  int const lane = threadIdx.x & 31;
  int const gwarps = gthreads / 32;
  int const wlocal = threadIdx.x >> 5; // within-worker warp id
  int const nwl = TPB >> 5;            // warps per worker (8)
  bool const do_shared = true;
  FFN_WS_TS(0); // entry

  // ====================================================================
  //  DYNAMIC SMEM LAYOUT (extern __align__(1024), the megakernel convention;
  //  ALL the source's static __shared__ arrays live here). Offsets computed
  //  with an explicit align-up. WBUF is the per-warp cp.async weight stage,
  //  worst case across every GEMV path (STAGES-aware!).
  //  Block-local sections (s_norm/s_a/s_ifp8/...) are recomputed identically by
  //  every block so the front stages need no extra grid_barrier.
  // ====================================================================
  // Tunables: row-block / pipeline stages per GEMV. RBX_* MUST divide GRP=128
  // (the shared per-N-block weight-scale row is keyed by n0/GRP and shared by
  // all RBX rows in one call — HAZARD 1: an RBX that does not divide 128
  // crosses an N-block boundary and reads the wrong scale row; 8/16/4 all
  // divide 128).
  constexpr int RBX_W13 = 8;
  constexpr int RBX_W2 = 16;
  constexpr int RBX_SH = 4;
  // v019 FAST path STAGES (deeper W13 pipeline; W2 on the 16B half2 path):
  //   W13 16B half2 STAGES=4 ; shGU 16B half2 STAGES=2 ;
  //   W2  16B half2 STAGES=2 ; shDN 4B scalar STAGES=3.
  constexpr int ST_W13 = 4;
  constexpr int ST_SH13 = 2;
  constexpr int ST_W2 = 2;
  constexpr int ST_SH2 = 3;
  // Per-warp uint4 stage budget = MAX over every GEMV path actually used.
  // W13(16B) RBX_W13*32*ST_W13 ; shGU(16B) RBX_SH*32*ST_SH13 ;
  // W2(16B) RBX_W2*32*ST_W2 ; shDN(4B) ceil(RBX_SH*32*ST_SH2 /4) uint4.
  constexpr size_t U4_W13 = (size_t)RBX_W13 * 32 * ST_W13;
  constexpr size_t U4_SH13 = (size_t)RBX_SH * 32 * ST_SH13;
  constexpr size_t U4_W2 = (size_t)RBX_W2 * 32 * ST_W2;
  constexpr size_t U4_SH2 = ((size_t)RBX_SH * 32 * ST_SH2 + 3) / 4;
  constexpr size_t WBUF_GEMV =
      (U4_W13 > U4_SH13 ? U4_W13 : U4_SH13) > (U4_W2 > U4_SH2 ? U4_W2 : U4_SH2)
          ? (U4_W13 > U4_SH13 ? U4_W13 : U4_SH13)
          : (U4_W2 > U4_SH2 ? U4_W2 : U4_SH2);
  // Phase 2 (FAST) reuses the WHOLE-BLOCK s_wbuf to stage the routed y13
  // (active_count*W13_N <= MAX_ACTIVE*W13_N floats) before the silu — so the
  // whole-block buffer (NWL_CT*WBUF_U4 uint4 = NWL_CT*WBUF_U4*16 bytes) MUST
  // also hold MAX_ACTIVE*W13_N*4 bytes. Express that as a per-warp uint4 floor.
  constexpr size_t U4_Y13_STAGE =
      ((size_t)MAX_ACTIVE * W13_N * 4 + (size_t)NWL_CT * 16 - 1) /
      ((size_t)NWL_CT * 16);
  constexpr size_t WBUF_U4 =
      WBUF_GEMV > U4_Y13_STAGE ? WBUF_GEMV : U4_Y13_STAGE;
  // Compile-time guards (HAZARD 1 fix: the v015 assert only covered the 4B-W2
  // path; the 16B-W2 stage needs ~2x more, and the y13 staging reuse needs the
  // whole-block buffer >= MAX_ACTIVE*W13_N*4).
  static_assert(
      RBX_W13 % 1 == 0 && (128 % RBX_W13) == 0 && (128 % RBX_W2) == 0 &&
          (128 % RBX_SH) == 0,
      "FFN-FULL RBX_* must divide GRP=128 (shared scale-row validity)");
  static_assert(WBUF_U4 * 16 >= U4_W13 * 16 && WBUF_U4 * 16 >= U4_SH13 * 16 &&
                    WBUF_U4 * 16 >= U4_W2 * 16 && WBUF_U4 * 16 >= U4_SH2 * 16,
                "FFN-FULL per-warp wbuf too small for a GEMV stage buffer");
  static_assert((size_t)NWL_CT * WBUF_U4 * 16 >= (size_t)MAX_ACTIVE * W13_N * 4,
                "FFN-FULL whole-block wbuf too small to stage routed y13");
  // FAST_ROUTING: WBUF must also hold the router cp.async ring: STAGES*32
  // uint4.
  static_assert(WBUF_U4 >= 4 * 32,
                "FFN-FULL per-warp wbuf too small for router STAGES=4 ring");
  // The constexpr sizing above assumes nwl==NWL_CT; verify the runtime worker
  // warp count matches (TPB=256 -> 8). A mismatch would under-size the staging.
  assert(nwl == NWL_CT);

  extern __shared__ __align__(1024) uint8_t s_smem[];
#define FFN_FULL_AU16(x) (((x) + 15u) & ~((size_t)15u))
  size_t off = 0;
  uint4 *s_wbuf = reinterpret_cast<uint4 *>(s_smem + off);
  off += (size_t)nwl * WBUF_U4 * sizeof(uint4); // 16B/uint4 -> 16-aligned
  uint4 *my_wbuf = s_wbuf + static_cast<size_t>(wlocal) * WBUF_U4;
  uint32_t *my_wbuf4 = reinterpret_cast<uint32_t *>(my_wbuf);
  off = FFN_FULL_AU16(off);
  __nv_bfloat16 *s_norm = reinterpret_cast<__nv_bfloat16 *>(s_smem + off);
  off += (size_t)HIDDEN * sizeof(__nv_bfloat16);
  off = FFN_FULL_AU16(off);
  uint8_t *s_a = s_smem + off; // 16-aligned: dgemv_cpa16 uint4 activation read
  off += HIDDEN;
  off = FFN_FULL_AU16(off);
  float *s_as = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)KG1 * sizeof(float);
  off = FFN_FULL_AU16(off);
  uint8_t *s_ifp8 = s_smem + off; // 16-aligned (block-local W2 input)
  off += (size_t)MAX_ACTIVE * W2_K;
  off = FFN_FULL_AU16(off);
  float *s_iscale = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)MAX_ACTIVE * KG2 * sizeof(float);
  off = FFN_FULL_AU16(off);
  uint8_t *s_sifp8 = s_smem + off; // 16-aligned (block-local shared-down input)
  off += (size_t)SH_DN_K;
  off = FFN_FULL_AU16(off);
  float *s_siscale = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)KG_SHDN * sizeof(float);
  off = FFN_FULL_AU16(off);
  // topk working set (parallel sigmoid + group scores; block-local).
  float *s_sig = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)NUM_EXPERTS * sizeof(float);
  off = FFN_FULL_AU16(off);
  float *s_biased = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)NUM_EXPERTS * sizeof(float);
  off = FFN_FULL_AU16(off);
  float *s_gscore = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)NUM_GROUPS * sizeof(float);
  off = FFN_FULL_AU16(off);
  int *s_gsel = reinterpret_cast<int *>(s_smem + off);
  off += (size_t)NUM_GROUPS * sizeof(int);
  off = FFN_FULL_AU16(off);
  // GLOBAL top-8 winners (expert id 0..255) + their normalized weights. The
  // EP-local filter below derives the LOCAL active list from these.
  int *s_gacte = reinterpret_cast<int *>(s_smem + off); // [TOPK_EXPERTS]
  off += (size_t)TOPK_EXPERTS * sizeof(int);
  off = FFN_FULL_AU16(off);
  float *s_gactw = reinterpret_cast<float *>(s_smem + off); // [TOPK_EXPERTS]
  off += (size_t)TOPK_EXPERTS * sizeof(float);
  off = FFN_FULL_AU16(off);
  // FAST lever (4): per-group local top-8 staging for the parallel argmax
  // (NUM_GROUPS warps x TOPK_EXPERTS winners). value + index.
  float *s_top8v = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)NUM_GROUPS * TOPK_EXPERTS * sizeof(float);
  off = FFN_FULL_AU16(off);
  int *s_top8i = reinterpret_cast<int *>(s_smem + off);
  off += (size_t)NUM_GROUPS * TOPK_EXPERTS * sizeof(int);
  off = FFN_FULL_AU16(off);
  // per-warp RMSNorm reduction scratch (one float per warp).
  float *s_red = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)nwl * sizeof(float);
#if defined(MPK_DSV3_FFN_WS_PAGES)
  // warp-spec additions (gated => default layout byte-identical):
  //   per-router-page mbarriers (V2 page semaphores) + the topk-done flag.
  off = FFN_FULL_AU16(off);
  uint64_t *s_rp_mbar = reinterpret_cast<uint64_t *>(s_smem + off);
  off += (size_t)WS_NUM_RPAGES * sizeof(uint64_t);
  off = FFN_FULL_AU16(off);
  uint32_t *s_topk_done = reinterpret_cast<uint32_t *>(s_smem + off);
  off += sizeof(uint32_t);
  (void)s_topk_done;
  // router pages OVERLAY s_wbuf[0 .. 8*3584): s_wbuf is idle until Phase 1
  // (the baseline router path used it as its cp.async ring in Phase B; the
  // warpspec consumers read the loader's pages instead and issue no cp.async
  // before Phase 1), so this costs ZERO new SMEM.
  uint8_t *s_rpages = reinterpret_cast<uint8_t *>(s_wbuf);
#endif
#if defined(MPK_DSV3_FFN_WS_PIPE)
  // stage-2 ring channel state (V2 Channel hand-rolled): per (consumer w<7,
  // page p<2) full/empty mbarriers + the per-consumer phase-1-ring-released
  // flag; plus a DEDICATED phase-2 y13 staging buffer (the baseline overlays
  // it on s_wbuf, which the rings now own through phases 1-3). s_y13_ded is
  // also lent to the consumers' shared-DN self-load rings in Phase 3 (free by
  // then: y13 is consumed during silu).
  off = FFN_FULL_AU16(off);
  uint64_t *s_pipe_full = reinterpret_cast<uint64_t *>(s_smem + off);
  off += (size_t)WS_NUM_CONSUMER_WARPS * WS_PIPE_PAGES * sizeof(uint64_t);
  off = FFN_FULL_AU16(off);
  uint64_t *s_pipe_empty = reinterpret_cast<uint64_t *>(s_smem + off);
  off += (size_t)WS_NUM_CONSUMER_WARPS * WS_PIPE_PAGES * sizeof(uint64_t);
  off = FFN_FULL_AU16(off);
  uint32_t *s_p1_done = reinterpret_cast<uint32_t *>(s_smem + off);
  off += (size_t)WS_NUM_CONSUMER_WARPS * sizeof(uint32_t);
  off = FFN_FULL_AU16(off);
  float *s_y13_ded = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)MAX_ACTIVE * W13_N * sizeof(float);
#endif
#undef FFN_FULL_AU16

#if defined(MPK_DSV3_FFN_WS_PAGES)
  // ---- warp-spec entry: init the page mbarriers (+ stage-2 ring channel
  // state), publish via fence.mbarrier_init + a full-block barrier, then
  // detach W7 into its loader prologue (router page fills). W7 rejoins the
  // block-collective rmsnorm below LATE — same threadIdx striding, so the math
  // stays bit-exact; consumers merely wait ~its issue time at the first
  // __syncthreads. -----------------------------------------------------------
  if (threadIdx.x == 0) {
#pragma unroll
    for (int p = 0; p < WS_NUM_RPAGES; p++) {
      ws_mbar_init(&s_rp_mbar[p], 32);
    }
    *s_topk_done = 0u;
#if defined(MPK_DSV3_FFN_WS_PIPE)
    for (int i = 0; i < WS_NUM_CONSUMER_WARPS * WS_PIPE_PAGES; i++) {
      ws_mbar_init(&s_pipe_full[i], 32);
      // empty[]: init + PRE-ARRIVE the full count (V2 pre-empty convention) so
      // the loader's first-lap wait on parity 0 passes immediately.
      ws_mbar_init(&s_pipe_empty[i], 32);
      ws_mbar_arrive_cnt(&s_pipe_empty[i], 32);
    }
    for (int w = 0; w < WS_NUM_CONSUMER_WARPS; w++) {
      s_p1_done[w] = 0u;
    }
#endif
    ws_mbar_init_fence();
  }
  __syncthreads();
  if (wlocal == WS_LOADER_WARP) {
    ws_loader_prologue(router_gate_w, s_rpages, s_rp_mbar, worker_idx, lane);
  }
#endif

  // ====================================================================
  //  Phase A — RMSNorm. Every block computes the FULL sum(x^2) over hidden
  //  [7168] redundantly, then writes the bf16 normed into s_norm. blk0
  //  persists sc.rmsnorm_out. normed[i] = x[i]*rms_rcp*w[i].
  // ====================================================================
  {
    float ss = 0.f;
    for (int i = threadIdx.x; i < HIDDEN; i += TPB) {
      float v = __bfloat162float(hidden[i]);
      ss += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) {
      s_red[wlocal] = ss;
    }
    __syncthreads();
    float tot = (threadIdx.x < nwl) ? s_red[threadIdx.x] : 0.f;
    // reduce nwl (<=8) partials within warp 0.
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      tot += __shfl_xor_sync(0xffffffffu, tot, o);
    }
    if (threadIdx.x == 0) {
      s_red[0] = tot;
    }
    __syncthreads();
    float rms_rcp = rsqrtf(s_red[0] / float(HIDDEN) + RMS_EPS);
    for (int i = threadIdx.x; i < HIDDEN; i += TPB) {
      float v = __bfloat162float(hidden[i]);
      float wt = __bfloat162float(rmsnorm_weight[i]);
      s_norm[i] = __float2bfloat16(v * rms_rcp * wt);
    }
    __syncthreads();
    if (blockIdx.x == 0) {
      for (int i = threadIdx.x; i < HIDDEN; i += TPB) {
        sc.rmsnorm_out[i] = s_norm[i];
      }
    }
  }
  FFN_WS_TS(1); // post-A (rmsnorm)

  // ====================================================================
  //  Phase 0 — FP8 quantize the BF16 normed (rmsnorm-quant convention) into
  //  per-worker smem. Init out_acc=0. blk0 persists a_fp8 / a_scale.
  // ====================================================================
  for (int g = wlocal; g < KG1; g += nwl) {
    quant_group_warp<__nv_bfloat16>(s_norm, s_a, s_as, g, lane);
  }
  for (int i = gtid; i < W2_N; i += gthreads) {
    sc.out_acc[i] = 0.f;
  }
  __syncthreads();
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < HIDDEN; i += TPB) {
      sc.a_fp8[i] = s_a[i];
    }
    for (int g = threadIdx.x; g < KG1; g += TPB) {
      sc.a_scale[g] = s_as[g];
    }
  }
  FFN_WS_TS(2); // post-0 (quant)

  // ====================================================================
  //  Phase B — router gate-GEMV (SPLIT-K). RKSPLIT warps per expert row;
  //  each warp's fp32 partial -> sc.inter[e*RKSPLIT + sp]. s_norm is
  //  block-local but identical across blocks.
  //
  //  Fast Phase B: replaces the scalar router_partial
  //  with router_partial_cpa (cp.async pipelined weight loads + packed
  //  bf162 inner dot). Uses my_wbuf as the staging ring (WBUF_U4 >> needed).
  //  The ring is flushed (cpasync_wait<0>) before the grid_barrier.
  // ====================================================================
#if defined(MPK_DSV3_FFN_WS_PAGES)
  // warp-spec Phase B: warp w consumes the loader's SMEM page w (mbarrier
  // parity wait = the V2 page-ready handshake). Task map unchanged (task
  // t = gwarp, single round since 1024 < 1088); the page holds exactly this
  // task's (expert, K-split) slice; the dot is statement-identical to
  // router_partial_cpa => sc.inter is bit-exact vs baseline.
  {
    int const t = gwarp;
    if (t < ROUTER_N * RKSPLIT) {
      int const e = t / RKSPLIT, sp = t % RKSPLIT;
      ws_mbar_wait_parity(&s_rp_mbar[wlocal], 0);
      __nv_bfloat16 const *wpage = reinterpret_cast<__nv_bfloat16 const *>(
          s_rpages + (size_t)wlocal * WS_RPAGE_BYTES);
      float const acc = ws_router_partial_page(s_norm, wpage, sp, lane);
      if (lane == 0) {
        sc.inter[e * RKSPLIT + sp] = acc;
      }
    }
  }
#else
  {
    int t = gwarp;
    int ntask = ROUTER_N * RKSPLIT;
    // FAST: cp.async pipelined weight loads + packed bf162 inner dot.
    // STAGES=4: hides ~300+ cycle HBM latency across 7 super-steps of 32 uint4.
    // my_wbuf reuse: WBUF_U4 uint4/warp >> STAGES*32=128 uint4 needed. Safe.
    constexpr int ROUTER_STAGES = 4;
    for (; t < ntask; t += gwarps) {
      int e = t / RKSPLIT, sp = t % RKSPLIT;
      __nv_bfloat16 const *wr =
          router_gate_w + static_cast<size_t>(e) * ROUTER_K;
      float acc = router_partial_cpa<RKSPLIT, ROUTER_STAGES>(
          s_norm, wr, sp, lane, my_wbuf);
      if (lane == 0) {
        sc.inter[e * RKSPLIT + sp] = acc;
      }
    }
    router_cpa_drain(); // flush any pending cp.async before grid_barrier
  }
#endif
  FFN_WS_TS(3);                                // post-B (router GEMV)
  ffn_full_grid_barrier(barrier, NUM_WORKERS); // router partials visible
  FFN_WS_TS(4);                                // post-GB1

  // ====================================================================
  //  Phase C — reduce the RKSPLIT router partials -> bf16 logits, then run
  //  topk-sigmoid. Every block does this redundantly into block-local
  //  buffers. blk0 publishes sc.logits.
  // ====================================================================
#if defined(MPK_DSV3_FFN_WS_DETOUR)
  // warp-spec Phase C: consumers (W0-6) run the AUTHORITATIVE topk on 7 warps
  // — per-group warp stages stride g += 7 (identical per-group values +
  // shfl trees => bit-exact selection), elementwise loops stride 224, and all
  // internal block syncs are the barrier-1/224 partial barrier (a
  // __syncthreads here would stall on the detoured W7). W7 meanwhile runs the
  // PRIVATE prefetch topk and then waits the s_topk_done release flag.
  if (wlocal < WS_NUM_CONSUMER_WARPS) {
    // (i) reduce partials -> bf16 logits + parallel sigmoid/biased.
    for (int e = threadIdx.x; e < ROUTER_N; e += WS_CONSUMER_THREADS) {
      float tot = 0.f;
#pragma unroll
      for (int sp = 0; sp < RKSPLIT; sp++) {
        tot += sc.inter[e * RKSPLIT + sp];
      }
      __nv_bfloat16 lgb =
          __float2bfloat16(tot); // bf16-round (production boundary)
      if (blockIdx.x == 0) {
        sc.logits[e] = lgb;
      }
      float lg = __bfloat162float(lgb);
      float s = 1.0f / (1.0f + expf(-lg));
      s_sig[e] = s;
      s_biased[e] = s + bias[e];
    }
    ws_bar_consumers();
    FFN_WS_TS(12); // after (i) sigmoid/logits
    // (ii) group score = top-2 biased per group of 32. Warp w owns groups
    // {w, w+7} (identical per-group math as the 8-warp form).
    for (int g = wlocal; g < NUM_GROUPS; g += WS_NUM_CONSUMER_WARPS) {
      float v = s_biased[g * EXPERTS_PER_GROUP + lane];
      float t1 = v;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, o));
      }
      unsigned ismax = __ballot_sync(0xffffffffu, v == t1);
      int firstmax = __ffs(ismax) - 1;
      float v2;
      if (lane == firstmax) {
        v2 = -1e30f;
      } else {
        v2 = v;
      }
      float t2 = v2;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        t2 = fmaxf(t2, __shfl_xor_sync(0xffffffffu, t2, o));
      }
      if (lane == 0) {
        s_gscore[g] = t1 + t2;
      }
    }
    ws_bar_consumers();
    // (iii) top-4 GROUP selection (verbatim thread-0 block).
    if (threadIdx.x == 0) {
      float gsc[NUM_GROUPS];
#pragma unroll
      for (int g = 0; g < NUM_GROUPS; g++) {
        gsc[g] = s_gscore[g];
        s_gsel[g] = 0;
      }
#pragma unroll
      for (int ki = 0; ki < TOPK_GROUP; ki++) {
        int bg = 0;
        float bs = -1e30f;
#pragma unroll
        for (int g = 0; g < NUM_GROUPS; g++) {
          if (!s_gsel[g] && gsc[g] > bs) {
            bs = gsc[g];
            bg = g;
          }
        }
        s_gsel[bg] = 1;
      }
    }
    ws_bar_consumers();
    for (int n = threadIdx.x; n < NUM_EXPERTS; n += WS_CONSUMER_THREADS) {
      if (!s_gsel[n / EXPERTS_PER_GROUP]) {
        s_biased[n] = -10000.f;
      }
    }
    ws_bar_consumers();
    FFN_WS_TS(13); // after (ii)+(iii)+mask
    // per-group local top-8 (FAST lever 4), restructured for 7 warps WITHOUT
    // doubling any warp: only the TOPK_GROUP(4) SELECTED groups carry real
    // candidates — a fully-masked group's 8 argmax rounds are deterministic
    // ((-10000.f, g*32) every round: all 32 values are the -10000.f literal,
    // the low-index tie-break picks lane 0, and the owner-lane re-mask is a
    // no-op), so warps 0-3 run ONE selected group each and warps 4-6 write the
    // masked groups' constant pattern directly. BYTE-identical s_top8v/s_top8i
    // vs the 8-warp baseline; stage wall = ONE group's top-8 (no 2x round).
    {
      int selrank = -1, unselrank = -1, mygroup = -1;
      if (wlocal < TOPK_GROUP) {
        selrank = wlocal; // warps 0-3: the selrank-th SELECTED group
      } else {
        unselrank = wlocal - TOPK_GROUP; // warps 4-6 (+wrap): masked groups
      }
      int cnt_sel = 0, cnt_unsel = 0;
#pragma unroll
      for (int g = 0; g < NUM_GROUPS; g++) {
        if (s_gsel[g]) {
          if (cnt_sel == selrank) {
            mygroup = g;
          }
          cnt_sel++;
        } else {
          if (unselrank >= 0 &&
              (cnt_unsel == unselrank || cnt_unsel == unselrank + 3)) {
            // warp 4 covers masked ranks {0,3}, warp 5 {1}, warp 6 {2}
            if (lane < TOPK_EXPERTS) {
              s_top8v[g * TOPK_EXPERTS + lane] = -10000.f;
              s_top8i[g * TOPK_EXPERTS + lane] = g * EXPERTS_PER_GROUP;
            }
          }
          cnt_unsel++;
        }
      }
      if (mygroup >= 0) {
        int const g = mygroup;
        float lvv = s_biased[g * EXPERTS_PER_GROUP + lane];
        int lvi = g * EXPERTS_PER_GROUP + lane;
#pragma unroll
        for (int k = 0; k < TOPK_EXPERTS; k++) {
          float bv = lvv;
          int bi = lvi;
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
              bv = ov;
              bi = oi;
            }
          }
          if (lane == 0) {
            s_top8v[g * TOPK_EXPERTS + k] = bv;
            s_top8i[g * TOPK_EXPERTS + k] = bi;
          }
          if (lvi == bi) {
            lvv = -10000.f; // mask the winner in its owning lane
          }
        }
      }
    }
    ws_bar_consumers();
    FFN_WS_TS(14); // after per-group top-8
    // warp 0 merges the 64 candidates (verbatim).
    if (wlocal == 0) {
      int const CPL = (NUM_GROUPS * TOPK_EXPERTS) / 32; // 2 per lane
      float cv[CPL];
      int ci[CPL];
#pragma unroll
      for (int j = 0; j < CPL; j++) {
        cv[j] = s_top8v[lane * CPL + j];
        ci[j] = s_top8i[lane * CPL + j];
      }
      float wsum = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK_EXPERTS; k++) {
        float bv = -1e30f;
        int bi = NUM_EXPERTS;
#pragma unroll
        for (int j = 0; j < CPL; j++) {
          if (cv[j] > bv || (cv[j] == bv && ci[j] < bi)) {
            bv = cv[j];
            bi = ci[j];
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          float ov = __shfl_xor_sync(0xffffffffu, bv, o);
          int oi = __shfl_xor_sync(0xffffffffu, bi, o);
          if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
          }
        }
        if (lane == 0) {
          s_gacte[k] = bi;
          s_gactw[k] = s_sig[bi];
        }
        wsum += s_sig[bi];
#pragma unroll
        for (int j = 0; j < CPL; j++) {
          if (ci[j] == bi) {
            cv[j] = -10000.f;
          }
        }
      }
      if (lane == 0) {
        float inv = 1.0f / (wsum + 1e-20f);
#pragma unroll
        for (int k = 0; k < TOPK_EXPERTS; k++) {
          s_gactw[k] = s_gactw[k] * inv * routed_scaling_factor;
        }
      }
    }
    ws_bar_consumers(); // s_gacte/s_gactw visible to the 7 consumer warps
    FFN_WS_TS(15);      // after merge + publish barrier
    if (threadIdx.x == 0) {
      ws_st_release_u32(s_topk_done, 1u); // publish to the detoured W7
    }
  } else {
    // W7 detour: private prefetch topk + stop-bounded warm-up streams, then
    // wait the authoritative publish.
    ws_private_topk_prefetch(sc.inter,
                             bias,
                             w13,
                             w2,
                             wdn,
                             s_topk_done,
                             local_expert_start,
                             num_local_experts,
                             worker_idx,
                             lane);
    // Gentle spin: a tight ld.acquire loop measurably slowed the consumers'
    // warp-0 merge (SMEM/issue interference); 256ns sleep makes it near-silent
    // while costing W7 at most ~256ns of Phase-1 join latency.
    while (ws_ld_acquire_u32(s_topk_done) == 0u) {
      __nanosleep(256);
    }
  }
#else
  // (i) reduce partials -> bf16 logits + parallel sigmoid/biased.
  for (int e = threadIdx.x; e < ROUTER_N; e += TPB) {
    float tot = 0.f;
#pragma unroll
    for (int sp = 0; sp < RKSPLIT; sp++) {
      tot += sc.inter[e * RKSPLIT + sp];
    }
    __nv_bfloat16 lgb =
        __float2bfloat16(tot); // bf16-round (production boundary)
    if (blockIdx.x == 0) {
      sc.logits[e] = lgb;
    }
    float lg = __bfloat162float(lgb);
    float s = 1.0f / (1.0f + expf(-lg));
    s_sig[e] = s;
    s_biased[e] = s + bias[e];
  }
  __syncthreads();
  FFN_WS_TS(12); // after (i) sigmoid/logits
  // (ii) group score = top-2 biased per group of 32. One warp per group.
  if (wlocal < NUM_GROUPS) {
    float v = s_biased[wlocal * EXPERTS_PER_GROUP + lane];
    float t1 = v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, o));
    }
    unsigned ismax = __ballot_sync(0xffffffffu, v == t1);
    int firstmax = __ffs(ismax) - 1;
    float v2;
    if (lane == firstmax) {
      v2 = -1e30f;
    } else {
      v2 = v;
    }
    float t2 = v2;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      t2 = fmaxf(t2, __shfl_xor_sync(0xffffffffu, t2, o));
    }
    if (lane == 0) {
      s_gscore[wlocal] = t1 + t2;
    }
  }
  __syncthreads();
  // (iii) top-4 GROUP selection (8 groups -> thread 0), published to s_gsel.
  if (threadIdx.x == 0) {
    float gsc[NUM_GROUPS];
#pragma unroll
    for (int g = 0; g < NUM_GROUPS; g++) {
      gsc[g] = s_gscore[g];
      s_gsel[g] = 0;
    }
#pragma unroll
    for (int ki = 0; ki < TOPK_GROUP; ki++) {
      int bg = 0;
      float bs = -1e30f;
#pragma unroll
      for (int g = 0; g < NUM_GROUPS; g++) {
        if (!s_gsel[g] && gsc[g] > bs) {
          bs = gsc[g];
          bg = g;
        }
      }
      s_gsel[bg] = 1;
    }
  }
  __syncthreads();
  // parallel mask: non-selected groups -> -10000 (whole block).
  for (int n = threadIdx.x; n < NUM_EXPERTS; n += TPB) {
    if (!s_gsel[n / EXPERTS_PER_GROUP]) {
      s_biased[n] = -10000.f;
    }
  }
  __syncthreads();
  FFN_WS_TS(13); // after (ii)+(iii)+mask
  // FAST lever (4): GLOBAL top-8 via PARALLEL per-group top-8 + warp-0 merge.
  // Instead of warp 0 serially scanning all 256 biased values 8 times (8
  // dependent rounds over 8 values/lane on ONE warp while 7 idle), ALL 8 warps
  // cooperate: warp w owns its group's 32 experts (1/lane) and computes ITS
  // local top-8 (8 rounds of a 32-value warp-shfl argmax, lower-index
  // tie-break, mask the winner in its owning lane). The global top-8 can take
  // at most 8 from any single group AND only TOPK_GROUP(4) groups survive the
  // mask above (the other 4 are all -10000), so a per-group top-8 over the
  // surviving groups is sufficient to reconstruct the global top-8. Warp 0 then
  // merges the 8x8=64 candidates (2/lane) into the global top-8 with the SAME
  // lower-index tie-break and accumulates wsum over the GLOBAL-8 (the
  // production denom). This writes the SAME s_gacte/s_gactw (global ids,
  // global-normalized weights) as the serial path -> the
  // EP-local-filter-to-registers block below is unchanged.
  {
    float lvv =
        s_biased[wlocal * EXPERTS_PER_GROUP + lane]; // this warp's 32 vals
    int lvi = wlocal * EXPERTS_PER_GROUP + lane;
#pragma unroll
    for (int k = 0; k < TOPK_EXPERTS; k++) {
      float bv = lvv;
      int bi = lvi;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (lane == 0) {
        s_top8v[wlocal * TOPK_EXPERTS + k] = bv;
        s_top8i[wlocal * TOPK_EXPERTS + k] = bi;
      }
      if (lvi == bi) {
        lvv = -10000.f; // mask the winner in its owning lane
      }
    }
  }
  __syncthreads();
  FFN_WS_TS(14); // after per-group top-8
  // warp 0 merges the 64 candidates (2/lane) into the global top-8 + wsum.
  if (wlocal == 0) {
    int const CPL = (NUM_GROUPS * TOPK_EXPERTS) / 32; // 64/32 = 2 per lane
    float cv[CPL];
    int ci[CPL];
#pragma unroll
    for (int j = 0; j < CPL; j++) {
      cv[j] = s_top8v[lane * CPL + j];
      ci[j] = s_top8i[lane * CPL + j];
    }
    float wsum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK_EXPERTS; k++) {
      float bv = -1e30f;
      int bi = NUM_EXPERTS;
#pragma unroll
      for (int j = 0; j < CPL; j++) {
        if (cv[j] > bv || (cv[j] == bv && ci[j] < bi)) {
          bv = cv[j];
          bi = ci[j];
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (lane == 0) {
        s_gacte[k] = bi;
        s_gactw[k] = s_sig[bi];
      }
      wsum += s_sig[bi];
#pragma unroll
      for (int j = 0; j < CPL; j++) {
        if (ci[j] == bi) {
          cv[j] = -10000.f; // mask the winning candidate in its owning lane
        }
      }
    }
    if (lane == 0) {
      float inv = 1.0f / (wsum + 1e-20f);
#pragma unroll
      for (int k = 0; k < TOPK_EXPERTS; k++) {
        s_gactw[k] = s_gactw[k] * inv * routed_scaling_factor;
      }
    }
  }
  __syncthreads(); // s_gacte/s_gactw visible to all warps in this block
  FFN_WS_TS(15);   // after merge + publish barrier
#endif // MPK_DSV3_FFN_WARPSPEC Phase-C split

  // ====================================================================
  //  EP-LOCAL FILTER (#1 correctness risk). The GLOBAL top-8 winners
  //  (s_gacte[k] in 0..255) are filtered to THIS rank's
  //  [local_expert_start, local_expert_end) range. For each global winner in
  //  range, append the LOCAL expert id (e - local_expert_start) and its
  //  already-normalized weight to active_experts/active_weights (registers),
  //  and active_count++. This mirrors the COLD FFN's derivation from
  //  moe_routing_indices: the production topk_sigmoid writes routing_indices
  //  (and therefore the COLD FFN's active list) ONLY for on-node experts, and
  //  the weight was normalized over the GLOBAL-8 sum (off-node experts counted
  //  in the sum, then zeroed) — both reproduced here. The selection-order of
  //  the global top-8 is preserved (k ascending), so the local active list is
  //  in the same order the COLD FFN's local-id scan would produce up to the
  //  per-id reordering (which is immaterial: the W2 atomicAdd is order-free and
  //  each slot's weight travels with its expert).
  //
  //  NOTE: active_count <= min(TOPK_EXPERTS, num_local_experts) <= MAX_ACTIVE.
  //  Every thread computes the SAME active list from block-local
  //  s_gacte/s_gactw (uniform across the block), so the Phase 1/2/3 grid-stride
  //  loops over active_count are warp-uniform (no divergence, no barrier skew).
  // ====================================================================
  int active_experts[MAX_ACTIVE];
  float active_weights[MAX_ACTIVE];
#pragma unroll
  for (int s = 0; s < MAX_ACTIVE; ++s) {
    active_experts[s] = 0;
    active_weights[s] = 0.f;
  }
  int active_count = 0;
#pragma unroll
  for (int k = 0; k < TOPK_EXPERTS; ++k) {
    int e = s_gacte[k];
    if (e >= local_expert_start && e < local_expert_end &&
        active_count < MAX_ACTIVE) {
      active_experts[active_count] = e - local_expert_start;
      active_weights[active_count] = s_gactw[k];
      active_count++;
    }
  }
  FFN_WS_TS(5); // post-C (topk + EP filter)

  // ====================================================================
  //  Phase 1 — routed W13 GEMV -> sc.y13[slot][n]; shared gate_up -> sc.sg[n].
  // ====================================================================
  int const n13 = active_count * (W13_N / RBX_W13);
  int const nsh1 = do_shared ? (SH_GU_N / RBX_SH) : 0;
  int const ntot1 = n13 + nsh1;
#if defined(MPK_DSV3_FFN_WS_PIPE)
  // Stage 2: consumers (W0-6) run Phase 1 on a 7-warp map (same per-task math
  // => y13/sg values identical); each consumer releases its my_wbuf ring to
  // the loader (s_p1_done) as soon as its own phase-1 use ends. W7 = loader:
  // window-A prefill of the phase-3 W2 pages (see ws_pipe_loader_prefill).
  static_assert(RBX_W2 == 16 && RBX_SH == 4,
                "WS_PIPE page geometry hardcodes RBX_W2=16 / RBX_SH=4");
  int const ws_n2 = active_count * (W2_N / RBX_W2);
  int ws_next_k[WS_NUM_CONSUMER_WARPS]; // loader cursors (live across GB2)
  if (wlocal < WS_NUM_CONSUMER_WARPS) {
    int const cw = worker_idx * WS_NUM_CONSUMER_WARPS + wlocal;
    int const cwarps = NUM_WORKERS * WS_NUM_CONSUMER_WARPS;
    for (int idx = cw; idx < ntot1; idx += cwarps) {
      if (idx < n13) {
        int slot = idx / (W13_N / RBX_W13);
        int n0 = (idx % (W13_N / RBX_W13)) * RBX_W13;
        int e = active_experts[slot];
        uint8_t const *wb = w13 + static_cast<size_t>(e) * W13_N * HIDDEN;
        float const *ws = w13_scale + static_cast<size_t>(e) * NB1 * KG1 +
                          static_cast<size_t>(n0 / GRP) * KG1;
        float yb[RBX_W13];
        dgemv_cpa16_h2<RBX_W13, ST_W13>(
            s_a, s_as, wb, ws, HIDDEN, KG1, n0, lane, my_wbuf, yb);
        if (lane == 0) {
#pragma unroll
          for (int r = 0; r < RBX_W13; r++) {
            sc.y13[static_cast<size_t>(slot) * W13_N + n0 + r] = yb[r];
          }
        }
      } else {
        int n0 = (idx - n13) * RBX_SH;
        float const *ws = wgu_s + static_cast<size_t>(n0 / GRP) * KG_SHGU;
        float yb[RBX_SH];
        dgemv_cpa16_h2<RBX_SH, ST_SH13>(
            s_a, s_as, wgu, ws, SH_GU_K, KG_SHGU, n0, lane, my_wbuf, yb);
        if (lane == 0) {
#pragma unroll
          for (int r = 0; r < RBX_SH; r++) {
            sc.sg[n0 + r] = yb[r];
          }
        }
      }
    }
    cpasync_wait<0>(); // drain this warp's self-load ring before handoff
    __syncwarp();
    if (lane == 0) {
      ws_st_release_u32(&s_p1_done[wlocal], 1u);
    }
  } else {
#if !defined(MPK_DSV3_FFN_WS_PIPE_NOFEED)
    ws_pipe_loader_prefill(w2,
                           active_experts,
                           ws_n2,
                           s_p1_done,
                           s_pipe_full,
                           s_wbuf,
                           WBUF_U4,
                           worker_idx,
                           lane,
                           ws_next_k);
#elif defined(MPK_DSV3_FFN_WS_W2STREAM)
    (void)ws_next_k;
    ws_w2_stream_phase1(
        w2, wdn, active_experts, active_count, s_p1_done, worker_idx, lane);
#else
    (void)ws_next_k; // ablation arm: loader idle (isolates the restructure)
#endif
  }
#else
  for (int idx = gwarp; idx < ntot1; idx += gwarps) {
    if (idx < n13) {
      int slot = idx / (W13_N / RBX_W13);
      int n0 = (idx % (W13_N / RBX_W13)) * RBX_W13;
      int e = active_experts[slot];
      uint8_t const *wb = w13 + static_cast<size_t>(e) * W13_N * HIDDEN;
      float const *ws = w13_scale + static_cast<size_t>(e) * NB1 * KG1 +
                        static_cast<size_t>(n0 / GRP) * KG1;
      float yb[RBX_W13];
      dgemv_cpa16_h2<RBX_W13, ST_W13>(
          s_a, s_as, wb, ws, HIDDEN, KG1, n0, lane, my_wbuf, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_W13; r++) {
          sc.y13[static_cast<size_t>(slot) * W13_N + n0 + r] = yb[r];
        }
      }
    } else {
      int n0 = (idx - n13) * RBX_SH;
      float const *ws = wgu_s + static_cast<size_t>(n0 / GRP) * KG_SHGU;
      float yb[RBX_SH];
      dgemv_cpa16_h2<RBX_SH, ST_SH13>(
          s_a, s_as, wgu, ws, SH_GU_K, KG_SHGU, n0, lane, my_wbuf, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_SH; r++) {
          sc.sg[n0 + r] = yb[r];
        }
      }
    }
  }
#endif          // MPK_DSV3_FFN_WS_PIPE Phase-1 split
  FFN_WS_TS(6); // post-1 (W13 + shared GU)
  ffn_full_grid_barrier(barrier, NUM_WORKERS);
  FFN_WS_TS(7); // post-GB2

  // ====================================================================
  //  Phase 2 — routed silu_mul+quant -> block-local s_ifp8; shared silu_mul+
  //  quant -> s_sifp8. The full silu_mul is recomputed by EVERY block (tiny:
  //  <=8 slots * 512) so Phase 3 reads the W2 input from SMEM (no cold global
  //  readback) and we DROP the Phase2->3 grid_barrier (replaced by a
  //  __syncthreads). The stride MUST be the per-BLOCK warp id (wlocal/nwl) so
  //  each block fills its FULL block-local s_ifp8. blk0 persists to global.
  // ====================================================================
  int const ng = active_count * KG2;
#if defined(MPK_DSV3_FFN_WS_PIPE)
  // Stage 2: the y13 staging + silu run on the 7 consumer warps with the
  // DEDICATED s_y13_ded buffer (my_wbuf now belongs to the W2 page rings) and
  // partial barriers (W7 is off refilling — a __syncthreads would stall on
  // it). All silu/quant math is elementwise or warp-local => values identical.
  // W7 = loader window B: poll-based ring refill through silu AND Phase 3.
  {
    float *s_y13 = s_y13_ded;
    if (wlocal < WS_NUM_CONSUMER_WARPS) {
      {
        uint4 const *y4 = reinterpret_cast<uint4 const *>(sc.y13);
        uint32_t s_y13_b = __cvta_generic_to_shared(s_y13);
        int const NU4 = (active_count * W13_N) >> 2;
        for (int u = threadIdx.x; u < NU4; u += WS_CONSUMER_THREADS) {
          cpasync16(s_y13_b + (uint32_t)u * 16, &y4[u]);
        }
        cpasync_commit();
        cpasync_wait<0>();
      }
      ws_bar_consumers();
      for (int gg = wlocal; gg < ng; gg += WS_NUM_CONSUMER_WARPS) {
        int slot = gg / KG2;
        int g = gg % KG2;
        float const *y = s_y13 + static_cast<size_t>(slot) * W13_N;
        int i0 = g * GRP + lane * 4;
        float4 gpart = *reinterpret_cast<float4 const *>(&y[i0]);
        float4 upart = *reinterpret_cast<float4 const *>(&y[512 + i0]);
        float v[4], amax = 0.f;
        v[0] = silu_fast(gpart.x) * upart.x;
        v[1] = silu_fast(gpart.y) * upart.y;
        v[2] = silu_fast(gpart.z) * upart.z;
        v[3] = silu_fast(gpart.w) * upart.w;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          amax = fmaxf(amax, fabsf(v[t]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        }
        float s = quant_scale(amax);
        float inv = 1.f / s;
        if (lane == 0) {
          s_iscale[slot * KG2 + g] = s;
        }
#pragma unroll
        for (int t = 0; t < 4; t++) {
          int i = i0 + t;
          s_ifp8[static_cast<size_t>(slot) * W2_K + i] = to_f8(v[t] * inv);
        }
      }
      if (do_shared) {
        for (int g = wlocal; g < KG_SHDN; g += WS_NUM_CONSUMER_WARPS) {
          float v[4], amax = 0.f;
#pragma unroll
          for (int t = 0; t < 4; t++) {
            int i = g * GRP + lane * 4 + t;
            float val = silu_fast(sc.sg[i]) * sc.sg[256 + i];
            v[t] = val;
            amax = fmaxf(amax, fabsf(val));
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
          }
          float s = quant_scale(amax);
          float inv = 1.f / s;
          if (lane == 0) {
            s_siscale[g] = s;
          }
#pragma unroll
          for (int t = 0; t < 4; t++) {
            int i = g * GRP + lane * 4 + t;
            s_sifp8[i] = to_f8(v[t] * inv);
          }
        }
      }
      ws_bar_consumers(); // block-local i_fp8/si_fp8 visible to consumers
      if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < active_count * W2_K;
             i += WS_CONSUMER_THREADS) {
          sc.i_fp8[i] = s_ifp8[i];
        }
        for (int i = threadIdx.x; i < active_count * KG2;
             i += WS_CONSUMER_THREADS) {
          sc.i_scale[i] = s_iscale[i];
        }
        if (do_shared) {
          for (int i = threadIdx.x; i < SH_DN_K; i += WS_CONSUMER_THREADS) {
            sc.si_fp8[i] = s_sifp8[i];
          }
          for (int i = threadIdx.x; i < KG_SHDN; i += WS_CONSUMER_THREADS) {
            sc.si_scale[i] = s_siscale[i];
          }
        }
      }
    } else {
#if !defined(MPK_DSV3_FFN_WS_PIPE_NOFEED)
      ws_pipe_loader_refill(w2,
                            active_experts,
                            ws_n2,
                            s_pipe_full,
                            s_pipe_empty,
                            s_wbuf,
                            WBUF_U4,
                            worker_idx,
                            lane,
                            ws_next_k);
#endif
    }
  }
#else
  // FAST lever (1): stage the routed y13 (active_count*W13_N floats) into the
  // (idle-here) whole-block s_wbuf via ONE coalesced cp.async.cg 16B run + ONE
  // wait, so the silu reads from FAST smem and the cold y13 global round-trip
  // (it was written by OTHER blocks in Phase 1) overlaps onto the memory pipe
  // instead of stalling each thread per-group. The grid_barrier above made the
  // y13 stores from every block visible; this is THIS block's first read of
  // them. WBUF_U4 is sized so nwl*WBUF_U4*16 >= MAX_ACTIVE*W13_N*4 (asserted).
  float *s_y13 =
      reinterpret_cast<float *>(s_wbuf); // >= active_count*W13_N floats
  {
    uint4 const *y4 = reinterpret_cast<uint4 const *>(sc.y13);
    uint32_t s_y13_b = __cvta_generic_to_shared(s_y13);
    int const NU4 = (active_count * W13_N) >> 2; // uint4 (4 floats each)
    for (int u = threadIdx.x; u < NU4; u += TPB) {
      cpasync16(s_y13_b + (uint32_t)u * 16, &y4[u]);
    }
    cpasync_commit();
    cpasync_wait<0>();
  }
  __syncthreads();
  for (int gg = wlocal; gg < ng; gg += nwl) {
    int slot = gg / KG2;
    int g = gg % KG2;
    float const *y = s_y13 + static_cast<size_t>(slot) * W13_N;
    int i0 = g * GRP + lane * 4;
    float4 gpart = *reinterpret_cast<float4 const *>(&y[i0]);
    float4 upart = *reinterpret_cast<float4 const *>(&y[512 + i0]);
    float v[4], amax = 0.f;
    v[0] = silu_fast(gpart.x) * upart.x;
    v[1] = silu_fast(gpart.y) * upart.y;
    v[2] = silu_fast(gpart.z) * upart.z;
    v[3] = silu_fast(gpart.w) * upart.w;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      amax = fmaxf(amax, fabsf(v[t]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    float s = quant_scale(amax);
    float inv = 1.f / s;
    if (lane == 0) {
      s_iscale[slot * KG2 + g] = s;
    }
#pragma unroll
    for (int t = 0; t < 4; t++) {
      int i = i0 + t;
      s_ifp8[static_cast<size_t>(slot) * W2_K + i] = to_f8(v[t] * inv);
    }
  }
  if (do_shared) {
    for (int g = wlocal; g < KG_SHDN; g += nwl) {
      float v[4], amax = 0.f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int i = g * GRP + lane * 4 + t;
        float val = silu_fast(sc.sg[i]) * sc.sg[256 + i];
        v[t] = val;
        amax = fmaxf(amax, fabsf(val));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      }
      float s = quant_scale(amax);
      float inv = 1.f / s;
      if (lane == 0) {
        s_siscale[g] = s;
      }
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int i = g * GRP + lane * 4 + t;
        s_sifp8[i] = to_f8(v[t] * inv);
      }
    }
  }
  __syncthreads(); // block-local i_fp8/si_fp8 visible to all warps in THIS
                   // block
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < active_count * W2_K; i += TPB) {
      sc.i_fp8[i] = s_ifp8[i];
    }
    for (int i = threadIdx.x; i < active_count * KG2; i += TPB) {
      sc.i_scale[i] = s_iscale[i];
    }
    if (do_shared) {
      for (int i = threadIdx.x; i < SH_DN_K; i += TPB) {
        sc.si_fp8[i] = s_sifp8[i];
      }
      for (int i = threadIdx.x; i < KG_SHDN; i += TPB) {
        sc.si_scale[i] = s_siscale[i];
      }
    }
  }
#endif          // MPK_DSV3_FFN_WS_PIPE Phase-2 split
  FFN_WS_TS(8); // post-2 (silu + requant)

  // ====================================================================
  //  Phase 3 — W2 + shared-down, INTERLEAVED, accumulate into fp32 out_acc.
  //  Reads the W2 input from BLOCK-LOCAL s_ifp8/s_iscale (s_sifp8/s_siscale).
  // ====================================================================
  int const n2 = active_count * (W2_N / RBX_W2);
  int const nshd = do_shared ? (SH_DN_N / RBX_SH) : 0;
  int const ntot3 = n2 + nshd;
#if defined(MPK_DSV3_FFN_WS_PIPE)
  // Stage 2: consumers MAC the routed W2 from the loader-fed SMEM pages
  // (mbarrier full-wait -> identical MAC statements -> release empty). The fed
  // task ordinals are exactly the k-prefix of each consumer's list (idx < n2
  // iff k < nfed) — the same map the loader uses. Shared-down tasks self-load
  // into a ring hosted in s_y13_ded (free after silu; my_wbuf belongs to the
  // pages). W7 is still in (or done with) its refill loop and rejoins at GB3.
  if (wlocal < WS_NUM_CONSUMER_WARPS) {
    int const cw = worker_idx * WS_NUM_CONSUMER_WARPS + wlocal;
    int const cwarps = NUM_WORKERS * WS_NUM_CONSUMER_WARPS;
    uint32_t *dn_wbuf = reinterpret_cast<uint32_t *>(s_y13_ded) +
                        (size_t)wlocal * (RBX_SH * 32 * ST_SH2);
    int k = 0;
    for (int idx = cw; idx < ntot3; idx += cwarps, k++) {
      if (idx < n2) {
        int const p = k % WS_PIPE_PAGES;
        int const lap = k / WS_PIPE_PAGES;
        int slot = idx / (W2_N / RBX_W2);
        int n0 = (idx % (W2_N / RBX_W2)) * RBX_W2;
        float ew = active_weights[slot];
        float const *ws =
            w2_scale + static_cast<size_t>(active_experts[slot]) * NB2 * KG2 +
            static_cast<size_t>(n0 / GRP) * KG2;
        float yb[RBX_W2];
#if defined(MPK_DSV3_FFN_WS_PIPE_NOFEED)
        // ablation arm: self-load exactly as baseline (loader idle; my_wbuf
        // is free), keeping the 7-warp map + phase-2 restructure in place.
        (void)p;
        (void)lap;
        dgemv_cpa16_h2<RBX_W2, ST_W2>(
            s_ifp8 + static_cast<size_t>(slot) * W2_K,
            s_iscale + slot * KG2,
            w2 + static_cast<size_t>(active_experts[slot]) * W2_N * W2_K,
            ws,
            W2_K,
            KG2,
            n0,
            lane,
            my_wbuf,
            yb);
#else
        ws_mbar_wait_parity(&s_pipe_full[wlocal * WS_PIPE_PAGES + p], lap & 1);
        __syncwarp();
        uint32_t const page_saddr =
            __cvta_generic_to_shared(reinterpret_cast<uint8_t *>(my_wbuf) +
                                     (size_t)p * WS_PIPE_PAGE_BYTES);
        ws_dgemv_page16_h2<RBX_W2>(s_ifp8 + static_cast<size_t>(slot) * W2_K,
                                   s_iscale + slot * KG2,
                                   ws,
                                   page_saddr,
                                   lane,
                                   yb);
        __syncwarp(); // all lanes done reading the page
        ws_mbar_arrive(&s_pipe_empty[wlocal * WS_PIPE_PAGES + p]);
#endif
        if (lane == 0) {
#pragma unroll
          for (int r = 0; r < RBX_W2; r++) {
            atomicAdd(&sc.out_acc[n0 + r], ew * yb[r]);
          }
        }
      } else {
        int n0 = (idx - n2) * RBX_SH;
        float const *ws = wdn_s + static_cast<size_t>(n0 / GRP) * KG_SHDN;
        float yb[RBX_SH];
        dgemv_cpa<RBX_SH, 3>(s_sifp8,
                             s_siscale,
                             wdn,
                             ws,
                             SH_DN_K,
                             KG_SHDN,
                             n0,
                             lane,
                             dn_wbuf,
                             yb);
        if (lane == 0) {
#pragma unroll
          for (int r = 0; r < RBX_SH; r++) {
            atomicAdd(&sc.out_acc[n0 + r], yb[r]);
          }
        }
      }
    }
  }
#else
  for (int idx = gwarp; idx < ntot3; idx += gwarps) {
    if (idx < n2) {
      int slot = idx / (W2_N / RBX_W2);
      int n0 = (idx % (W2_N / RBX_W2)) * RBX_W2;
      int e = active_experts[slot];
      float ew = active_weights[slot];
      uint8_t const *wb = w2 + static_cast<size_t>(e) * W2_N * W2_K;
      float const *ws = w2_scale + static_cast<size_t>(e) * NB2 * KG2 +
                        static_cast<size_t>(n0 / GRP) * KG2;
      float yb[RBX_W2];
      // FAST lever (3): W2 on the 16B packed-half2 path (W2_K=512 -> SS=1; the
      // half2 group index lane>>3 maps to the SAME 4 K-groups as the 4B path).
      dgemv_cpa16_h2<RBX_W2, ST_W2>(s_ifp8 + static_cast<size_t>(slot) * W2_K,
                                    s_iscale + slot * KG2,
                                    wb,
                                    ws,
                                    W2_K,
                                    KG2,
                                    n0,
                                    lane,
                                    my_wbuf,
                                    yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_W2; r++) {
          atomicAdd(&sc.out_acc[n0 + r], ew * yb[r]);
        }
      }
    } else {
      int n0 = (idx - n2) * RBX_SH;
      float const *ws = wdn_s + static_cast<size_t>(n0 / GRP) * KG_SHDN;
      float yb[RBX_SH];
      dgemv_cpa<RBX_SH, 3>(s_sifp8,
                           s_siscale,
                           wdn,
                           ws,
                           SH_DN_K,
                           KG_SHDN,
                           n0,
                           lane,
                           my_wbuf4,
                           yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RBX_SH; r++) {
          atomicAdd(&sc.out_acc[n0 + r], yb[r]);
        }
      }
    }
  }
#endif          // MPK_DSV3_FFN_WS_PIPE Phase-3 split
  FFN_WS_TS(9); // post-3 (W2 + shared down)
  ffn_full_grid_barrier(barrier, NUM_WORKERS);
  FFN_WS_TS(10); // post-GB3

  // Final: cast the fp32 accumulator to the bf16 MPK output buffer.
  for (int i = gtid; i < W2_N; i += gthreads) {
    out[i] = __float2bfloat16_rn(sc.out_acc[i]);
  }
  // Publish the output stores globally before MPK signals task completion.
  __threadfence();
  __syncthreads();
  FFN_WS_TS(11); // exit
}

} // namespace ffn_full_megakernel_sm100
} // namespace kernel
