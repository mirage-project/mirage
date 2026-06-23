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

#include "mirage/persistent_kernel/runtime_header.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

struct GridBarrier {
  unsigned int *count; // [1] arrivals in the current generation
  unsigned int *gen;   // [1] generation (sense) counter
};

// Block-collective: call with the WHOLE block; only thread 0 touches global
// mem.
__device__ __forceinline__ void grid_barrier(GridBarrier b,
                                             int num_participants) {
  __syncthreads(); // all threads of THIS worker arrive first
  // FIX: publish THIS worker's global writes (y13 / i_fp8 / out) to all CTAs
  // BEFORE arriving — a relaxed atomic count-bump does NOT order prior regular
  // global stores, so without this the post-barrier readers raced on stale
  // intermediates (invisible to the simple standalone barrier test).
  __threadfence();
  __syncthreads(); // ensure EVERY thread's __threadfence has retired before thread
                   // 0 bumps the count / flips the gen. A per-thread fence only
                   // orders the CALLING thread's writes, so thread 0 signalling
                   // early could release readers before other warps' stores land.
  if (threadIdx.x == 0) {
    unsigned int my_gen = *((unsigned int volatile *)b.gen);
    // arrive; the one that bumps count to num_participants flips the gen.
    unsigned int prev = atomicAdd(b.count, 1u);
    if (prev + 1u == (unsigned int)num_participants) {
      *b.count = 0u;        // reset for the next generation
      __threadfence();      // publish the reset before the flip
      atomicAdd(b.gen, 1u); // release: flip the sense
    } else {
      // spin until the sense flips
      while (*((unsigned int volatile *)b.gen) == my_gen) { /* spin */
      }
    }
  }
  __syncthreads(); // re-converge this worker's threads, acquire the new gen
}

namespace kernel {
namespace ffn_mlp_megakernel_sm100 {

// ---- Problem shapes (DSv3 TP8 EP2 per-rank) -------------------------------
static constexpr int HIDDEN = 7168;
static constexpr int W13_N = 1024;
static constexpr int W2_K = 512;
static constexpr int W2_N = 7168;
static constexpr int E_LOCAL = 128;
static constexpr int GRP = 128;
static constexpr int KG1 = HIDDEN / GRP;
static constexpr int KG2 = W2_K / GRP;
static constexpr int NB1 = W13_N / GRP;
static constexpr int NB2 = W2_N / GRP;
static constexpr int MAX_ACTIVE = 8;
// NOTE: TPB (threads per worker CTA) is the RUNTIME blockDim.x (256 on the MPK
// Blackwell worker), derived locally in the task body below — NOT a compile
// constant. The old `static constexpr TPB=512` (carried verbatim from the
// standalone 512-thread cooperative bench) left 50% of every output uncomputed:
// each 256-thread worker only fills [0,256) of a 512-wide gtid stride.
static constexpr int NUM_WORKERS = 136; // MPK worker count on B200 (148 SM). Builder asserts num_workers==136.

// ---- Shared-expert shapes --------------------------------------------------
static constexpr int SH_GU_N = 512;
static constexpr int SH_GU_K = HIDDEN;
static constexpr int SH_DN_K = 256;
static constexpr int SH_DN_N = W2_N;
static constexpr int KG_SHGU = SH_GU_K / GRP;
static constexpr int KG_SHDN = SH_DN_K / GRP;
static constexpr int NB_SHGU = SH_GU_N / GRP;
static constexpr int NB_SHDN = SH_DN_N / GRP;
static constexpr int RB = 2;

static constexpr int BARRIER_BYTES = 2 * static_cast<int>(sizeof(uint32_t));
static constexpr int SCRATCH_BYTES =
    BARRIER_BYTES + HIDDEN + KG1 * 4 + MAX_ACTIVE * W13_N * 4 +
    MAX_ACTIVE * W2_K + MAX_ACTIVE * W2_K + MAX_ACTIVE * KG2 * 4 + SH_GU_N * 4 +
    SH_DN_K + KG_SHDN * 4 + W2_N * 4;

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

template <int ROW_BLOCK>
__device__ __forceinline__ void dgemv_block_warp(uint8_t const *a_fp8,
                                                 float const *a_scale,
                                                 uint8_t const *w_fp8,
                                                 float const *w_scale_row,
                                                 int K,
                                                 int KG,
                                                 int n0,
                                                 int lane,
                                                 float *y_out) {
  uint32_t const *a4 = reinterpret_cast<uint32_t const *>(a_fp8);
  uint32_t const *w4 =
      reinterpret_cast<uint32_t const *>(w_fp8 + static_cast<size_t>(n0) * K);
  int Kw = K / 4;
  float y[ROW_BLOCK];
#pragma unroll
  for (int r = 0; r < ROW_BLOCK; r++) {
    y[r] = 0.f;
  }
  for (int g = 0; g < KG; g++) {
    uint32_t av = a4[g * 32 + lane];
    float sc = a_scale[g] * w_scale_row[g];
#pragma unroll
    for (int r = 0; r < ROW_BLOCK; r++) {
      uint32_t wv = w4[static_cast<size_t>(r) * Kw + g * 32 + lane];
      float acc = 0.f;
#pragma unroll
      for (int b = 0; b < 4; b++) {
        acc += f8((av >> (b * 8)) & 0xff) * f8((wv >> (b * 8)) & 0xff);
      }
      y[r] += acc * sc;
    }
  }
#pragma unroll
  for (int r = 0; r < ROW_BLOCK; r++) {
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

struct Scratch {
  uint8_t *a_fp8;
  float *a_scale;
  float *y13;
  float *inter;
  uint8_t *i_fp8;
  float *i_scale;
  float *sg;
  uint8_t *si_fp8;
  float *si_scale;
  float *out_acc; // W2_N fp32 accumulator for the interleaved phase-3 atomicAdd
                  // (converted to bf16 `out` in a final pass — matches the
                  // chain's fp32-accumulate / bf16-store precision).
};

__device__ __forceinline__ Scratch make_scratch(uint8_t *base) {
  uint8_t *p = base + BARRIER_BYTES;
  Scratch sc;
  sc.a_fp8 = p;
  p += HIDDEN;
  sc.a_scale = reinterpret_cast<float *>(p);
  p += KG1 * 4;
  sc.y13 = reinterpret_cast<float *>(p);
  p += MAX_ACTIVE * W13_N * 4;
  sc.inter = reinterpret_cast<float *>(p);
  p += MAX_ACTIVE * W2_K;
  sc.i_fp8 = p;
  p += MAX_ACTIVE * W2_K;
  sc.i_scale = reinterpret_cast<float *>(p);
  p += MAX_ACTIVE * KG2 * 4;
  sc.sg = reinterpret_cast<float *>(p);
  p += SH_GU_N * 4;
  sc.si_fp8 = p;
  p += SH_DN_K;
  sc.si_scale = reinterpret_cast<float *>(p);
  p += KG_SHDN * 4;
  sc.out_acc = reinterpret_cast<float *>(p);
  p += W2_N * 4;
  return sc;
}


// Register-blocked + K-group-unrolled warp GEMV (ferret workspace1, faithful
// 256-thread gate: 88.5us -> 58.75us, cosine 1.0 byte-exact vs dgemv_block_warp).
// RBX consecutive output rows per warp (same n-block -> one w_scale_row); the
// activation uint is loaded ONCE per K-group and reused across RBX rows; GU groups
// are loaded (GU*RBX weight uints) BEFORE any MAC -> deep in-flight load window
// that hides the FP8-weight-load latency (NCU: 56% of stalls). MACs run in the
// SAME ascending-group/ascending-byte order as dgemv_block_warp -> bit-identical.
// CONSTRAINT (gate-caught bug): RBX must divide GRP=128 so all RBX rows share one
// w_scale_row; a scalar tail covers KG % GU.
#ifndef MPK_FFN_RBX_W13
#define MPK_FFN_RBX_W13 8
#endif
#ifndef MPK_FFN_RBX_W2
#define MPK_FFN_RBX_W2 8
#endif
#ifndef MPK_FFN_RBX_SH
#define MPK_FFN_RBX_SH 4
#endif
#ifndef MPK_FFN_GU_W13
#define MPK_FFN_GU_W13 8
#endif
#ifndef MPK_FFN_GU_W2
#define MPK_FFN_GU_W2 4
#endif
#ifndef MPK_FFN_GU_SHGU
#define MPK_FFN_GU_SHGU 8
#endif
#ifndef MPK_FFN_GU_SHDN
#define MPK_FFN_GU_SHDN 2
#endif

template <int RBX, int GU>
__device__ __forceinline__ void
    dgemv_blk(uint8_t const *__restrict__ a_fp8,
              float const *__restrict__ a_scale,
              uint8_t const *__restrict__ w_fp8, // expert base
              float const *__restrict__ w_scale_row,
              int K, int KG, int n0, int lane, float *y_out) {
  uint32_t const *a4 = reinterpret_cast<uint32_t const *>(a_fp8);
  uint32_t const *w4 =
      reinterpret_cast<uint32_t const *>(w_fp8 + static_cast<size_t>(n0) * K);
  int Kw = K / 4;
  float y[RBX];
#pragma unroll
  for (int r = 0; r < RBX; r++)
    y[r] = 0.f;
  int g = 0;
  for (; g + GU <= KG; g += GU) {
    uint32_t av[GU];
    float sc[GU];
#pragma unroll
    for (int u = 0; u < GU; u++) {
      av[u] = a4[(g + u) * 32 + lane];
      sc[u] = a_scale[g + u] * __ldg(&w_scale_row[g + u]);
    }
    uint32_t wv[GU][RBX];
#pragma unroll
    for (int u = 0; u < GU; u++)
#pragma unroll
      for (int r = 0; r < RBX; r++)
        wv[u][r] = __ldg(&w4[static_cast<size_t>(r) * Kw + (g + u) * 32 + lane]);
#pragma unroll
    for (int r = 0; r < RBX; r++) {
#pragma unroll
      for (int u = 0; u < GU; u++) {
        uint32_t w = wv[u][r], a = av[u];
        float acc = 0.f;
#pragma unroll
        for (int b = 0; b < 4; b++)
          acc += f8((a >> (b * 8)) & 0xff) * f8((w >> (b * 8)) & 0xff);
        y[r] += acc * sc[u];
      }
    }
  }
  for (; g < KG; g++) {
    uint32_t av = a4[g * 32 + lane];
    float sc = a_scale[g] * __ldg(&w_scale_row[g]);
    uint32_t wv[RBX];
#pragma unroll
    for (int r = 0; r < RBX; r++)
      wv[r] = __ldg(&w4[static_cast<size_t>(r) * Kw + g * 32 + lane]);
#pragma unroll
    for (int r = 0; r < RBX; r++) {
      uint32_t w = wv[r];
      float acc = 0.f;
#pragma unroll
      for (int b = 0; b < 4; b++)
        acc += f8((av >> (b * 8)) & 0xff) * f8((w >> (b * 8)) & 0xff);
      y[r] += acc * sc;
    }
  }
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    float v = y[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0)
      y_out[r] = v;
  }
}

__device__ __noinline__ void ffn_mlp_megakernel_sm100_task_impl(
    mirage::runtime::TaskDesc const *task_desc,
    int merge_task_offset,
    mirage::runtime::RuntimeConfig const &runtime_config) {
  (void)runtime_config;

  __nv_bfloat16 const *hidden =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  uint8_t const *w13 = static_cast<uint8_t const *>(task_desc->input_ptrs[1]);
  float const *w13_scale = static_cast<float const *>(task_desc->input_ptrs[2]);
  uint8_t const *w2 = static_cast<uint8_t const *>(task_desc->input_ptrs[3]);
  float const *w2_scale = static_cast<float const *>(task_desc->input_ptrs[4]);
  int const *moe_mask = static_cast<int const *>(task_desc->input_ptrs[5]);
  int const *moe_routing_indices =
      static_cast<int const *>(task_desc->input_ptrs[6]);
  float const *moe_topk_weights =
      static_cast<float const *>(task_desc->input_ptrs[7]);
  uint8_t const *wgu = static_cast<uint8_t const *>(task_desc->input_ptrs[8]);
  float const *wgu_s = static_cast<float const *>(task_desc->input_ptrs[9]);
  uint8_t const *wdn = static_cast<uint8_t const *>(task_desc->input_ptrs[10]);
  float const *wdn_s = static_cast<float const *>(task_desc->input_ptrs[11]);
  // The task is registered (14 inputs, 1 output); the wrapper passes `out` BOTH
  // as input slot 12 (binding-map ABI) AND as output slot 0 (what MPK tracks for
  // the downstream dependency). Write through the OUTPUT slot — input_ptrs[12] is
  // a distinct stale alias, so writing it left downstream reading garbage while
  // the in-kernel value looked sane.
  __nv_bfloat16 *out =
      static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);
  uint8_t *scratch_base = static_cast<uint8_t *>(task_desc->input_ptrs[13]);

  GridBarrier barrier;
  barrier.count = reinterpret_cast<unsigned int *>(scratch_base);
  barrier.gen =
      reinterpret_cast<unsigned int *>(scratch_base + sizeof(uint32_t));
  Scratch sc = make_scratch(scratch_base);

  int const worker_idx = merge_task_offset;
  // TPB = the REAL worker thread count (256), NOT the bench's 512 — using 512
  // skips 50% of gtid (each 256-thread worker fills only [0,256) of a 512 stride,
  // leaving every other warp-residue's output uncomputed = silent wrong result).
  int const TPB = (int)blockDim.x;
  int const gtid = worker_idx * TPB + threadIdx.x;
  int const gthreads = NUM_WORKERS * TPB;
  int const gwarp = gtid / 32;
  int const lane = threadIdx.x & 31;
  int const gwarps = gthreads / 32;
  int const wlocal = threadIdx.x >> 5; // within-worker warp id (for smem quant)
  int const nwl = TPB >> 5;            // warps per worker
  bool const do_shared = true;

  // topk_sigmoid compact_active_experts_ballot layout:
  //   moe_mask[0..count-1]  = dense active local expert IDs (compacted in-place)
  //   moe_mask[E_LOCAL]     = count  (at index 128)
  // Note: BINDING_MAP.md described a different convention; this matches the
  // actual MPK topk_sigmoid output as confirmed by topk_sigmoid_sm100.cuh.
  // ROOT CAUSE FIX: moe_mask[E_LOCAL] (the topk's compacted active-count) is NOT
  // populated on the decode path — moe_mask was an UNUSED topk output before this
  // task (the chain reads the permute meta, not moe_mask), so the compaction
  // (topk active_ids[LOCAL_EXPERTS]=count) never lands and it stays at the init 0.
  // Derive the active experts directly from moe_routing_indices, which IS
  // populated (the chain's permute consumes it): routing_indices[le] = topk_slot+1
  // (0 = expert le not selected for this token).
  int active_experts[MAX_ACTIVE];
  float active_weights[MAX_ACTIVE];
#pragma unroll
  for (int s = 0; s < MAX_ACTIVE; ++s) {
    active_experts[s] = 0;
    active_weights[s] = 0.f;
  }
  int active_count = 0;
  for (int le = 0; le < E_LOCAL && active_count < MAX_ACTIVE; ++le) {
    int slot1 = moe_routing_indices[le];
    if (slot1 != 0) {
      int topk_slot = slot1 - 1;
      active_experts[active_count] = le;
      active_weights[active_count] =
          (topk_slot >= 0 && topk_slot < MAX_ACTIVE)
              ? moe_topk_weights[topk_slot]
              : 0.f;
      active_count++;
    }
  }

  // Phase 0: quantize hidden into PER-WORKER smem (each worker keeps the full FP8
  // activation in smem -> phase-1 reads it with NO global round-trip, saving ~20us
  // and avoiding an extra grid barrier). Zero the fp32 output accumulator.
  // Per-worker activation staging in the DYNAMIC smem pool (extern-alias, the MPK
  // convention) — NOT static __shared__: 7392B static would exceed the runtime's
  // ~6KB static reserve and fail cudaFuncSetAttribute. The 221KB dynamic pool is
  // already allocated; one task runs per worker so aliasing its start is safe.
  extern __shared__ __align__(1024) uint8_t s_smem[]; // match the megakernel
  // dynamic-smem convention (other tasks declare __align__(1024)); a smaller
  // alignment on this aliased extern array can lower the shared base alignment and
  // misalign other tasks' 1024-aligned (TMA / AR) accesses.
  uint8_t *s_a = s_smem;
  float *s_as = reinterpret_cast<float *>(s_smem + HIDDEN);
  for (int g = wlocal; g < KG1; g += nwl) {
    quant_group_warp<__nv_bfloat16>(hidden, s_a, s_as, g, lane);
  }
  for (int i = gtid; i < W2_N; i += gthreads) {
    sc.out_acc[i] = 0.f;
  }
  __syncthreads(); // per-worker: s_a ready for this worker's phase-1 reads. The
                   // out_acc zeroes get globally published by the phase-1->2
                   // grid_barrier below, well before phase-3's atomicAdds.

  // Phase 1: routed W13 GEMV -> y13[slot][n]; shared gate_up GEMV -> sg[n].
  // Interleave routed + shared in one warp-idx space for inter-phase ILP; read the
  // activation from per-worker smem. dgemv_blk is register-blocked (RBX independent
  // weight-load streams) to hide FP8-weight-load latency. RBX must divide GRP=128.
  int const n13 = active_count * (W13_N / MPK_FFN_RBX_W13);
  int const nsh1 = do_shared ? (SH_GU_N / MPK_FFN_RBX_SH) : 0;
  int const ntot1 = n13 + nsh1;
  for (int idx = gwarp; idx < ntot1; idx += gwarps) {
    if (idx < n13) {
      int slot = idx / (W13_N / MPK_FFN_RBX_W13);
      int n0 = (idx % (W13_N / MPK_FFN_RBX_W13)) * MPK_FFN_RBX_W13;
      int e = active_experts[slot];
      uint8_t const *wb = w13 + static_cast<size_t>(e) * W13_N * HIDDEN;
      float const *ws = w13_scale + static_cast<size_t>(e) * NB1 * KG1 +
                        static_cast<size_t>(n0 / GRP) * KG1;
      float yb[MPK_FFN_RBX_W13];
      dgemv_blk<MPK_FFN_RBX_W13, MPK_FFN_GU_W13>(
          s_a, s_as, wb, ws, HIDDEN, KG1, n0, lane, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < MPK_FFN_RBX_W13; r++) {
          sc.y13[static_cast<size_t>(slot) * W13_N + n0 + r] = yb[r];
        }
      }
    } else {
      int n0 = (idx - n13) * MPK_FFN_RBX_SH;
      float const *ws = wgu_s + static_cast<size_t>(n0 / GRP) * KG_SHGU;
      float yb[MPK_FFN_RBX_SH];
      dgemv_blk<MPK_FFN_RBX_SH, MPK_FFN_GU_SHGU>(
          s_a, s_as, wgu, ws, SH_GU_K, KG_SHGU, n0, lane, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < MPK_FFN_RBX_SH; r++) {
          sc.sg[n0 + r] = yb[r];
        }
      }
    }
  }
  grid_barrier(barrier, NUM_WORKERS);

  // Phase 2: routed silu_mul+quant -> i_fp8; shared silu_mul+quant -> si_fp8.
  int const ng = active_count * KG2;
  for (int gg = gwarp; gg < ng; gg += gwarps) {
    int slot = gg / KG2;
    int g = gg % KG2;
    float const *y = sc.y13 + static_cast<size_t>(slot) * W13_N;
    float v[4], amax = 0.f;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      int i = g * GRP + lane * 4 + t;
      float val = silu(y[i]) * y[512 + i];
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
      sc.i_scale[slot * KG2 + g] = s;
    }
#pragma unroll
    for (int t = 0; t < 4; t++) {
      int i = g * GRP + lane * 4 + t;
      sc.i_fp8[static_cast<size_t>(slot) * W2_K + i] = to_f8(v[t] * inv);
    }
  }
  if (do_shared) {
    for (int g = gwarp; g < KG_SHDN; g += gwarps) {
      float v[4], amax = 0.f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int i = g * GRP + lane * 4 + t;
        float val = silu(sc.sg[i]) * sc.sg[256 + i];
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
        sc.si_scale[g] = s;
      }
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int i = g * GRP + lane * 4 + t;
        sc.si_fp8[i] = to_f8(v[t] * inv);
      }
    }
  }
  grid_barrier(barrier, NUM_WORKERS);

  // Phase 3: W2 + shared-down, INTERLEAVED (one warp per routed (expert,n0-tile)
  // or shared n0-tile) for max occupancy, accumulating into the fp32 sc.out_acc
  // via atomicAdd. A final pass casts out_acc -> bf16 out (chain-precision
  // fp32-accumulate / bf16-store). The per-output-tile alternative under-occupies
  // (896 tiles) and ran 93us vs this structure's ~62us on the faithful gate.
  int const n2 = active_count * (W2_N / MPK_FFN_RBX_W2);
  int const nshd = do_shared ? (SH_DN_N / MPK_FFN_RBX_SH) : 0;
  int const ntot3 = n2 + nshd;
  for (int idx = gwarp; idx < ntot3; idx += gwarps) {
    if (idx < n2) {
      int slot = idx / (W2_N / MPK_FFN_RBX_W2);
      int n0 = (idx % (W2_N / MPK_FFN_RBX_W2)) * MPK_FFN_RBX_W2;
      int e = active_experts[slot];
      float ew = active_weights[slot];
      uint8_t const *wb = w2 + static_cast<size_t>(e) * W2_N * W2_K;
      float const *ws = w2_scale + static_cast<size_t>(e) * NB2 * KG2 +
                        static_cast<size_t>(n0 / GRP) * KG2;
      float yb[MPK_FFN_RBX_W2];
      dgemv_blk<MPK_FFN_RBX_W2, MPK_FFN_GU_W2>(
          sc.i_fp8 + static_cast<size_t>(slot) * W2_K, sc.i_scale + slot * KG2,
          wb, ws, W2_K, KG2, n0, lane, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < MPK_FFN_RBX_W2; r++) {
          atomicAdd(&sc.out_acc[n0 + r], ew * yb[r]);
        }
      }
    } else {
      int n0 = (idx - n2) * MPK_FFN_RBX_SH;
      float const *ws = wdn_s + static_cast<size_t>(n0 / GRP) * KG_SHDN;
      float yb[MPK_FFN_RBX_SH];
      dgemv_blk<MPK_FFN_RBX_SH, MPK_FFN_GU_SHDN>(
          sc.si_fp8, sc.si_scale, wdn, ws, SH_DN_K, KG_SHDN, n0, lane, yb);
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < MPK_FFN_RBX_SH; r++) {
          atomicAdd(&sc.out_acc[n0 + r], yb[r]);
        }
      }
    }
  }
  grid_barrier(barrier, NUM_WORKERS);

  // Final: cast the fp32 accumulator to the bf16 MPK output buffer.
  for (int i = gtid; i < W2_N; i += gthreads) {
    out[i] = __float2bfloat16_rn(sc.out_acc[i]);
  }
  // Publish the output stores globally before MPK signals task completion (the
  // post-task block-sync alone does NOT order other threads' global writes — same
  // class as the topk_sigmoid membar.gl fix; downstream AR would read stale out).
  __threadfence();
  __syncthreads();
}

} // namespace ffn_mlp_megakernel_sm100
} // namespace kernel
