/* Copyright 2026 CMU
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

// mHC pre: the full prenorm pipeline in one file, in two stages.
//
// pre_k1 -- prenorm GEMM (mixes = residual @ fn.T, mix_hc=24 wide) + per-token
//   L2 (sqrsum). Two implementations; the host wrapper dispatches between them
//   by a (tokens, hidden) heuristic:
//     * mHC_pre_k1_cuda_core   -- CUDA-core FFMA + split-k, NO tensor cores.
//     Wins at
//                            low token count (decode): no MMA/TMA/TMEM setup to
//                            amortize, split-k fills the grid.
//     * mHC_pre_k1_tensor_core -- raw-PTX tcgen05 (kind::f16 MMA), no
//     CUTLASS/CuTe.
//                            Wins at higher token count (prefill).
//   Both produce identical outputs (mixes_pad bf16 [tokens,128] + sqrsum fp32),
//   so the k2 tail consumes either interchangeably. (The earlier CUTLASS/CuTe
//   tcgen05 path was removed; mHC_pre_k1_tensor_core is bit-identical to it.)
//
// pre_k2 -- the tail (mhc_pre_big_fuse): RMS-fold of the gemm output + pre/post
//   sigmoid affines + Sinkhorn(4x4) comb + pre_mix-weighted residual sum, all
//   in smem, producing f_pre / h_post / comb.
//
// (Previously split across mHC_pre_k1.cuh and mHC_pre_k2.cuh; merged here with
// no logic change.)

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "blackwell/sm100_ptx.cuh"
#include "tasks/common/common_header.cuh" // pre_k2
// Borrow ONLY the MMA instruction-descriptor constant from CuTe (its kind::f16
// bit layout is fiddly to hand-encode). The MMA/TMA/TMEM pipeline is raw PTX.
#include <cute/arch/mma_sm100_desc.hpp>

// ============================================================================
// CUDA-core implementation (FFMA + split-k). One block per (token, k-split);
// each block reduces its K-slice into per-split partials, a reduce kernel folds
// them into mixes_pad + sqrsum.
// ============================================================================

namespace kernel {

template <typename T,
          int MIX_HC, // N*N + 2*N (24 for N=4)
          int K,      // reduction dim = N*C
          int BLOCK_THREADS,
          int SPLIT_K,
          int MIX_PAD = 128, // padded mixes_pad column count
          int TPB = 1> // tokens processed per CTA (amortizes the fn reload)
__device__ __forceinline__ void mHC_pre_k1_cuda_core_task_impl(
    T const *__restrict__ residual,  // [tokens, K]
    __nv_bfloat16 const *__restrict__ fn, // [MIX_HC, K]  (weight, bf16)
    float *__restrict__ out_partial, // [SPLIT_K, tokens, MIX_HC] (SPLIT_K>1)
    float *__restrict__ sqr_partial, // [SPLIT_K, tokens]         (SPLIT_K>1)
    void *__restrict__ mixes_pad,    // [tokens, MIX_PAD] bf16    (SPLIT_K==1)
    float *__restrict__ sqrsum,      // [tokens]                  (SPLIT_K==1)
    int num_tokens,
    int token0, // first token of this CTA's TPB-group
    int i_ks) {
  // When SPLIT_K==1 each block already holds the complete reduction for its
  // token(s), so it writes the FINAL bf16 mixes_pad + fp32 sqrsum directly,
  // folding the separate reduce kernel into this epilogue (one fewer launch --
  // matters at low token count where launch overhead dominates). SPLIT_K>1
  // still writes fp32 partials for the reduce kernel to fold.
  constexpr bool DIRECT = (SPLIT_K == 1);
  constexpr int K_PER_SPLIT = K / SPLIT_K;
  static_assert(K % SPLIT_K == 0, "K must be divisible by SPLIT_K");
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp_id = tid >> 5;
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;

  // Per-token accumulators. fn is loaded ONCE per k-vec and reused across all
  // TPB tokens, so the (dominant) fn L1 traffic is divided by TPB.
  float acc[TPB][MIX_HC];
  float sqr[TPB];
#pragma unroll
  for (int t = 0; t < TPB; ++t) {
    sqr[t] = 0.0f;
#pragma unroll
    for (int o = 0; o < MIX_HC; ++o) {
      acc[t][o] = 0.0f;
    }
  }

  int const ntok = (num_tokens - token0) < TPB ? (num_tokens - token0) : TPB;
  T const *res0 = residual + (int64_t)token0 * K;
  int const k_base = i_ks * K_PER_SPLIT;

  constexpr int VEC = 8;
  bool const aligned = (K_PER_SPLIT % VEC) == 0 &&
                       ((reinterpret_cast<uintptr_t>(res0) & 0xF) == 0);
  if (aligned) {
    int const vec_count = K_PER_SPLIT / VEC;
    int const k_vec_base = k_base / VEC;
    for (int v = tid; v < vec_count; v += BLOCK_THREADS) {
      int const k = (k_vec_base + v) * VEC;
      // Load all TPB tokens' residual vecs for this k.
      float rv[TPB][VEC];
#pragma unroll
      for (int t = 0; t < TPB; ++t) {
        if (t < ntok) {
          uint4 r_raw =
              reinterpret_cast<uint4 const *>(res0 + (int64_t)t * K + k)[0];
          __nv_bfloat162 const *rb =
              reinterpret_cast<__nv_bfloat162 const *>(&r_raw);
#pragma unroll
          for (int e2 = 0; e2 < VEC / 2; ++e2) {
            float2 f = __bfloat1622float2(rb[e2]);
            rv[t][2 * e2 + 0] = f.x;
            rv[t][2 * e2 + 1] = f.y;
          }
#pragma unroll
          for (int e = 0; e < VEC; ++e) {
            sqr[t] += rv[t][e] * rv[t][e];
          }
        }
      }
#pragma unroll
      for (int o = 0; o < MIX_HC; ++o) {
        __nv_bfloat16 const *fn_o = fn + (int64_t)o * K + k;
        uint4 fw0 = reinterpret_cast<uint4 const *>(fn_o)[0];
        uint4 fw1 = reinterpret_cast<uint4 const *>(fn_o)[1];
        __nv_bfloat162 const *fb0 = reinterpret_cast<__nv_bfloat162 const *>(&fw0);
        __nv_bfloat162 const *fb1 = reinterpret_cast<__nv_bfloat162 const *>(&fw1);
        float wf[VEC];
#pragma unroll
        for (int e2 = 0; e2 < VEC / 2; ++e2) {
          float2 f = __bfloat1622float2(e2 < 2 ? fb0[e2] : fb1[e2 - 2]);
          wf[2 * e2 + 0] = f.x;
          wf[2 * e2 + 1] = f.y;
        }
#pragma unroll
        for (int t = 0; t < TPB; ++t) {
          if (t < ntok) {
            acc[t][o] += wf[0] * rv[t][0] + wf[1] * rv[t][1] + wf[2] * rv[t][2] +
                         wf[3] * rv[t][3] + wf[4] * rv[t][4] + wf[5] * rv[t][5] +
                         wf[6] * rv[t][6] + wf[7] * rv[t][7];
          }
        }
      }
    }
  } else {
    for (int it = tid; it < K_PER_SPLIT; it += BLOCK_THREADS) {
      int const k = k_base + it;
#pragma unroll
      for (int t = 0; t < TPB; ++t) {
        if (t < ntok) {
          float const rv = static_cast<float>(res0[(int64_t)t * K + k]);
          sqr[t] += rv * rv;
#pragma unroll
          for (int o = 0; o < MIX_HC; ++o) {
            acc[t][o] += __bfloat162float(fn[(int64_t)o * K + k]) * rv;
          }
        }
      }
    }
  }

  // Warp-reduce each token's accumulators.
#pragma unroll
  for (int t = 0; t < TPB; ++t) {
#pragma unroll
    for (int o = 0; o < MIX_HC; ++o) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        acc[t][o] += __shfl_xor_sync(0xffffffff, acc[t][o], off);
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sqr[t] += __shfl_xor_sync(0xffffffff, sqr[t], off);
    }
  }

  __shared__ float s_acc[TPB][NUM_WARPS][MIX_HC];
  __shared__ float s_sqr[TPB][NUM_WARPS];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
#pragma unroll
      for (int o = 0; o < MIX_HC; ++o) {
        s_acc[t][warp_id][o] = acc[t][o];
      }
      s_sqr[t][warp_id] = sqr[t];
    }
  }
  __syncthreads();

  // warp 0: cross-warp reduce + write. SPLIT_K==1 -> final bf16/fp32 outputs;
  // SPLIT_K>1 -> fp32 partials for the reduce kernel.
  if (warp_id == 0) {
    __nv_bfloat16 *mixes = static_cast<__nv_bfloat16 *>(mixes_pad);
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
      if (t >= ntok) {
        continue;
      }
      int const token = token0 + t;
      if (lane < MIX_HC) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
          v += s_acc[t][w][lane];
        }
        if (DIRECT) {
          mixes[(int64_t)token * MIX_PAD + lane] = __float2bfloat16(v);
        } else {
          out_partial[((int64_t)i_ks * num_tokens + token) * MIX_HC + lane] = v;
        }
      }
      if (lane == 0) {
        float v2 = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
          v2 += s_sqr[t][w];
        }
        if (DIRECT) {
          sqrsum[token] = v2;
        } else {
          sqr_partial[(int64_t)i_ks * num_tokens + token] = v2;
        }
      }
    }
  }
}

// Reduce SPLIT_K partials -> mixes_pad (bf16, padded to MIX_PAD cols) + sqrsum.
// One block per token with >= MIX_HC threads.
template <int MIX_HC, int MIX_PAD, int SPLIT_K>
__device__ __forceinline__ void mHC_pre_k1_cuda_core_reduce_impl(
    float const *__restrict__ out_partial, // [SPLIT_K, tokens, MIX_HC]
    float const *__restrict__ sqr_partial, // [SPLIT_K, tokens]
    void *__restrict__ mixes_pad,          // [tokens, MIX_PAD] bf16
    float *__restrict__ sqrsum,            // [tokens]
    int num_tokens,
    int token) {
  __nv_bfloat16 *mixes = static_cast<__nv_bfloat16 *>(mixes_pad);
  int const o = threadIdx.x;
  if (o < MIX_HC) {
    float v = 0.0f;
#pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
      v += out_partial[((int64_t)s * num_tokens + token) * MIX_HC + o];
    }
    mixes[(int64_t)token * MIX_PAD + o] = __float2bfloat16(v);
  }
  if (o == 0) {
    float sq = 0.0f;
#pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
      sq += sqr_partial[(int64_t)s * num_tokens + token];
    }
    sqrsum[token] = sq;
  }
}

} // namespace kernel

// ============================================================================
// Raw-PTX tcgen05 implementation (kind::f16 MMA + sqrsum). One CTA per BLOCK_N
// batch tile; loops K in BLOCK_K chunks through a NUM_STAGES smem ring; reads
// the TMEM accumulator in the epilogue. bf16, NOT block-scaled.
// ============================================================================

// 2D K-major bf16 TMA descriptor for a [rows, K] tile with 128B swizzle.
inline void init_2d_bf16_tmap(CUtensorMap *tmap,
                              void const *ptr,
                              uint64_t rows,
                              uint64_t K,
                              uint32_t block_k,
                              uint32_t block_rows) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank] = {K, rows};
  uint64_t globalStrides[rank - 1] = {K * 2}; // bytes per row
  uint32_t boxDim[rank] = {block_k, block_rows};
  uint32_t elementStrides[rank] = {1, 1};
  CUresult err = cuTensorMapEncodeTiled(tmap,
                                        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                        rank,
                                        const_cast<void *>(ptr),
                                        globalDim,
                                        globalStrides,
                                        boxDim,
                                        elementStrides,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                                        CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(
      err == CUDA_SUCCESS, "init_2d_bf16_tmap encode failed: ", (int)err);
}

namespace kernel {
namespace pre_k1_tensor_core {

using namespace ::kernel::sm100_ptx;

constexpr int MMA_K = 16; // 16-bit tcgen05 MMA-K

__device__ __forceinline__ uint64_t make_desc_bf16(int smem_addr) {
  constexpr uint64_t SBO = 8ULL * 128;
  return desc_enc((uint64_t)smem_addr) | (desc_enc(SBO) << 32) | (1ULL << 46) |
         (2ULL << 61);
}

template <int REDUCTION_SIZE,
          int OUTPUT_PAD,
          int BLOCK_N,
          int BLOCK_K,
          int MIX_HC,
          int NUM_STAGES>
__global__
    __launch_bounds__(OUTPUT_PAD + 2 * 32) void mHC_pre_k1_tensor_core_kernel(
        const __grid_constant__ CUtensorMap A_tmap, // weight fn [OUTPUT_PAD, K]
        const __grid_constant__ CUtensorMap B_tmap, // residual  [BATCH,     K]
        __nv_bfloat16 const
            *__restrict__ residual,            // [BATCH, K] gmem (for sqrsum)
        __nv_bfloat16 *__restrict__ mixes_pad, // [BATCH, OUTPUT_PAD] bf16
        float *__restrict__ sqrsum,            // [BATCH] fp32
        int BATCH) {
  static_assert(OUTPUT_PAD == 128, "tcgen05 MMA uses M=128");
  static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K divisible by MMA_K");
  static_assert(REDUCTION_SIZE % BLOCK_K == 0, "K divisible by BLOCK_K");
  static_assert(BLOCK_N == 16 || BLOCK_N == 32 || BLOCK_N == 64 ||
                    BLOCK_N == 128,
                "BLOCK_N in {16,32,64,128}");

  int const tid = threadIdx.x;
  int const warp_id = tid / 32;
  int const lane = tid & 31;
  constexpr int NUM_WARPS = OUTPUT_PAD / 32 + 2;

  int const bid_n = blockIdx.x;
  int const off_n = bid_n * BLOCK_N;

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_bytes = OUTPUT_PAD * BLOCK_K * 2;
  constexpr int B_bytes = BLOCK_N * BLOCK_K * 2;
  constexpr int STAGE_BYTES = A_bytes + B_bytes;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  int const tma_mbar = static_cast<int>(__cvta_generic_to_shared(mbars));
  int const mma_mbar = tma_mbar + NUM_STAGES * 8;
  int const mainloop_mbar = mma_mbar + NUM_STAGES * 8;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++) {
      mbar_init(tma_mbar + i * 8, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tmem_alloc<1, OUTPUT_PAD>(smem);
  }
  __syncthreads();

  constexpr int NUM_ITERS = REDUCTION_SIZE / BLOCK_K;
  int const tx_bytes = STAGE_BYTES;

  // 128B swizzle pins the TMA box's contiguous-K dim to 64 bf16 (=128 B), so a
  // BLOCK_K>64 stage is assembled from BLOCK_K/64 sub-loads of 64-wide tiles,
  // laid out linearly in smem (each is a self-contained 128B-swizzle atom that
  // the MMA descriptor offsets into by 64-elem steps).
  constexpr int K_ATOM = 64;
  constexpr int NUM_KSUB = BLOCK_K / K_ATOM;
  static_assert(BLOCK_K % K_ATOM == 0, "BLOCK_K must be a multiple of 64");
  constexpr int A_atom_bytes = OUTPUT_PAD * K_ATOM * 2;
  constexpr int B_atom_bytes = BLOCK_N * K_ATOM * 2;

  auto issue_tma = [&](int iter_k, int stage_id) {
    const int mbar = tma_mbar + stage_id * 8;
    const int A_smem = smem + stage_id * STAGE_BYTES;
    const int B_smem = A_smem + A_bytes;
    const int off_k = iter_k * BLOCK_K;
#pragma unroll
    for (int s = 0; s < NUM_KSUB; ++s) {
      const int ka = off_k + s * K_ATOM;
      tma_load_2d(A_smem + s * A_atom_bytes, &A_tmap, ka, 0, mbar, EVICT_LAST);
      tma_load_2d(
          B_smem + s * B_atom_bytes, &B_tmap, ka, off_n, mbar, EVICT_FIRST);
    }
    mbarrier_arrive_expect_tx_tile_local(mbar, tx_bytes);
  };

  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    // ---- TMA producer warp ----
    constexpr int PREFETCH = (NUM_ITERS < NUM_STAGES) ? NUM_ITERS : NUM_STAGES;
    for (int iter_k = 0; iter_k < PREFETCH; iter_k++) {
      issue_tma(iter_k, iter_k);
    }
    for (int iter_k = NUM_STAGES; iter_k < NUM_ITERS; iter_k++) {
      int const stage_id = iter_k % NUM_STAGES;
      int const mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      mbar_wait(mma_mbar + stage_id * 8, mma_phase);
      issue_tma(iter_k, stage_id);
    }
  } else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    // ---- MMA warp ----
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = OUTPUT_PAD;
    uint64_t const idescE =
        cute::UMMA::make_runtime_instr_desc<cute::bfloat16_t,
                                            cute::bfloat16_t,
                                            float,
                                            MMA_M,
                                            MMA_N,
                                            cute::UMMA::Major::K,
                                            cute::UMMA::Major::K>();
    uint32_t const i_desc = (uint32_t)(idescE >> 32);
    for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
      int const stage_id = iter_k % NUM_STAGES;
      int const tma_phase = (iter_k / NUM_STAGES) % 2;
      int const A_smem = smem + stage_id * STAGE_BYTES;
      int const B_smem = A_smem + A_bytes;

      mbar_wait(tma_mbar + stage_id * 8, tma_phase);

      // K-atoms are laid out as separate 64-wide swizzle tiles; within each,
      // 64/MMA_K MMAs step by MMA_K*2 bytes.
      constexpr int MMA_PER_ATOM = K_ATOM / MMA_K;
#pragma unroll
      for (int s = 0; s < NUM_KSUB; ++s) {
        int const A_sub = A_smem + s * A_atom_bytes;
        int const B_sub = B_smem + s * B_atom_bytes;
#pragma unroll
        for (int k = 0; k < MMA_PER_ATOM; k++) {
          uint64_t a_desc = make_desc_bf16(A_sub + k * MMA_K * 2);
          uint64_t b_desc = make_desc_bf16(B_sub + k * MMA_K * 2);
          int const acc = (iter_k == 0 && s == 0 && k == 0) ? 0 : 1;
          tcgen05_mma(/*taddr=*/0, a_desc, b_desc, i_desc, acc);
        }
      }
      tcgen05_commit_arrive<1>(mma_mbar + stage_id * 8);
    }
    tcgen05_commit_arrive<1>(mainloop_mbar);
  } else if (tid < OUTPUT_PAD) {
    // ---- Epilogue warps + sqrsum ----
    constexpr int NUM_EPI_WARPS = OUTPUT_PAD / 32;
    for (int n = warp_id; n < BLOCK_N; n += NUM_EPI_WARPS) {
      int const token = off_n + n;
      if (token >= BATCH) {
        continue;
      }
      __nv_bfloat16 const *row = residual + (int64_t)token * REDUCTION_SIZE;
      float local = 0.0f;
      for (int k = lane; k < REDUCTION_SIZE; k += 32) {
        float v = __bfloat162float(row[k]);
        local += v * v;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_xor_sync(0xffffffff, local, off);
      }
      if (lane == 0) {
        sqrsum[token] = local;
      }
    }

    mbar_wait(mainloop_mbar, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    constexpr int WIDTH = (BLOCK_N < 64) ? BLOCK_N : 64;
    constexpr int NUM_SUB = BLOCK_N / WIDTH;
    for (int g = 0; g < NUM_SUB; g++) {
      float tmp[WIDTH];
      if constexpr (WIDTH == 128) {
        tcgen05_ld_32x32bx128(tmp, warp_id * 32, g * WIDTH);
      } else if constexpr (WIDTH == 64) {
        tcgen05_ld_32x32bx64(tmp, warp_id * 32, g * WIDTH);
      } else {
        tcgen05_ld_32x32bx32(tmp, warp_id * 32, g * WIDTH);
      }
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      int const m = warp_id * 32 + lane; // output (mix) index
      for (int i = 0; i < WIDTH; i++) {
        int const n = g * WIDTH + i; // batch token within tile
        int const token = off_n + n;
        if (token < BATCH && m < OUTPUT_PAD) {
          mixes_pad[(int64_t)token * OUTPUT_PAD + m] = __float2bfloat16(tmp[i]);
        }
      }
    }
    asm volatile("bar.sync 1, %0;" : : "r"(OUTPUT_PAD) : "memory");
    if (warp_id == 0) {
      tmem_dealloc<1, OUTPUT_PAD>(0);
    }
  }
}

} // namespace pre_k1_tensor_core
} // namespace kernel

// ============================================================================
// pre K2 (mhc_pre_big_fuse): the prenorm tail.
// ============================================================================
// mHC pre K2 (vLLM-style split): mhc_pre_big_fuse.
//
// Consumes the un-normalized GEMM output `mixes` and the per-token `sqrsum`
// produced by pre K1, and fuses:
//   * RMS normalization of the gemm output (applied as a per-token scalar
//     s[t] = rsqrt(sqrsum[t] / RMS_HIDDEN + rms_eps), folded into the K2
//     affine since the GEMM is linear),
//   * pre_mix / post_mix (sigmoid affines),
//   * comb_mix via Sinkhorn (4x4),
//   * pre_mix-weighted residual sum (layer_input).
//
// This is the K2+K3+K4 tail of mHC_hc_pre_post_fused.cuh, with the single
// addition of the RMS scale folded into the K2 read so the tail sees the
// normalized projection. Intermediate tensors (h_pre, h_res, comb) live
// entirely in smem; `dyn_smem` sizing is identical to the fused tail.

namespace kernel {

template <typename T_in,
          int N,
          int C,
          int RMS_HIDDEN,
          int TOKENS_PER_CTA = 32,
          int BLOCK_THREADS = 256,
          int MIX_STRIDE = 0>
__device__ __forceinline__ void mHC_pre_k2_task_impl(
    void const *mixes_ptr,
    void const *sqrsum_ptr,
    void const *scale_ptr,
    void const *base_ptr,
    void const *x_ptr,
    void *f_pre_ptr,
    void *h_post_out_ptr,
    void *comb_out_ptr,
    int sinkhorn_repeat,
    float sinkhorn_eps,
    float rms_eps,
    int num_tokens,
    char *dyn_smem,
    // MPK-mode override: if >=0, run a single iteration at this fixed
    // token_base (input pointers must already be offset by the caller).
    // Standalone callers pass -1 for normal grid-stride behavior.
    int token_base_override = -1) {
  static_assert(N == 4, "pre K2 hardcoded to n=4");
  static_assert(BLOCK_THREADS % 32 == 0, "block size must be a warp multiple");
  static_assert(TOKENS_PER_CTA % 32 == 0,
                "TOKENS_PER_CTA must be a warp multiple");
  static_assert(C % 8 == 0, "C must be a multiple of 8");
  constexpr int MIX_HC = N * N + 2 * N;
  constexpr int MIX_ROW_STRIDE = (MIX_STRIDE == 0) ? MIX_HC : MIX_STRIDE;
  constexpr int SINKHORN_WARPS = TOKENS_PER_CTA / 32;
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;
  static_assert(SINKHORN_WARPS <= NUM_WARPS,
                "not enough warps to cover the sinkhorn batch");

  // Lay out the 4 smem buffers contiguously inside `dyn_smem`.
  // Use 16-byte alignment for each block so float4 stores work.
  uintptr_t base_ptr_addr = reinterpret_cast<uintptr_t>(dyn_smem);
  base_ptr_addr = (base_ptr_addr + 15u) & ~uintptr_t(15);
  float *h_pre_arr = reinterpret_cast<float *>(base_ptr_addr);
  float *h_post_arr = h_pre_arr + TOKENS_PER_CTA * N;
  float *h_res_arr = h_post_arr + TOKENS_PER_CTA * N;
  float *comb_arr = h_res_arr + TOKENS_PER_CTA * N * N;
  // Index helpers (row-major: arr[t][j] == arr[t * COLS + j]).
  auto h_pre = [h_pre_arr](int t, int j) -> float & {
    return h_pre_arr[t * N + j];
  };
  auto h_post = [h_post_arr](int t, int j) -> float & {
    return h_post_arr[t * N + j];
  };
  auto h_res = [h_res_arr](int t, int j) -> float & {
    return h_res_arr[t * (N * N) + j];
  };
  auto comb = [comb_arr](int t, int j) -> float & {
    return comb_arr[t * (N * N) + j];
  };

  T_in const *mixes = static_cast<T_in const *>(mixes_ptr);
  float const *sqrsum = static_cast<float const *>(sqrsum_ptr);
  float const *scale = static_cast<float const *>(scale_ptr);
  float const *base = static_cast<float const *>(base_ptr);
  T_in const *x = static_cast<T_in const *>(x_ptr);
  T_in *f_pre = static_cast<T_in *>(f_pre_ptr);
  float *h_post_out_g = static_cast<float *>(h_post_out_ptr);
  float *comb_out = static_cast<float *>(comb_out_ptr);

  float const alpha_pre = scale[0];
  float const alpha_post = scale[1];
  float const alpha_res = scale[2];
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;

  int const _tb_start = (token_base_override >= 0)
                            ? token_base_override
                            : (int)(blockIdx.x * TOKENS_PER_CTA);
  int const _tb_step = (token_base_override >= 0)
                           ? num_tokens // single-iteration in MPK mode
                           : (int)(gridDim.x * TOKENS_PER_CTA);
  for (int token_base = _tb_start; token_base < num_tokens;
       token_base += _tb_step) {
    int const tokens_this_iter = TOKENS_PER_CTA < num_tokens - token_base
                                     ? TOKENS_PER_CTA
                                     : num_tokens - token_base;

    // Per-token RMS scale s[t] = rsqrt(mean(residual^2) + eps), shared by all
    // mix columns of the token. Computed once into smem (reusing h_post's
    // slot is unsafe, so use a tiny dedicated stack of registers via smem).
    __shared__ float rms_scale[TOKENS_PER_CTA];
    for (int t = threadIdx.x; t < tokens_this_iter; t += BLOCK_THREADS) {
      float const ms = sqrsum[token_base + t] / static_cast<float>(RMS_HIDDEN);
      rms_scale[t] = rsqrtf(ms + rms_eps);
    }
    __syncthreads();

    // ---- Stage K2 (with RMS scale folded in) ----
    int const total_k2 = tokens_this_iter * MIX_HC;
    for (int idx = threadIdx.x; idx < total_k2; idx += BLOCK_THREADS) {
      int const t = idx / MIX_HC;
      int const j = idx % MIX_HC;
      int const token = token_base + t;
      // Normalize the gemm output before the affine: y = (mix * s) * alpha
      // + bias. Folding s here is exact because the GEMM is linear.
      float const mix =
          static_cast<float>(mixes[token * MIX_ROW_STRIDE + j]) * rms_scale[t];
      float const bias = base[j];
      float alpha;
      int region, local;
      if (j < N) {
        alpha = alpha_pre;
        region = 0;
        local = j;
      } else if (j < 2 * N) {
        alpha = alpha_post;
        region = 1;
        local = j - N;
      } else {
        alpha = alpha_res;
        region = 2;
        local = j - 2 * N;
      }
      float const y = mix * alpha + bias;
      if (region == 0) {
        h_pre(t, local) = 1.0f / (1.0f + __expf(-y));
      } else if (region == 1) {
        h_post(t, local) = 2.0f / (1.0f + __expf(-y));
      } else {
        h_res(t, local) = y;
      }
    }
    __syncthreads();

    // h_post coalesced gmem flush.
    {
      int const total_h_post = tokens_this_iter * N;
      for (int idx = threadIdx.x; idx < total_h_post; idx += BLOCK_THREADS) {
        int const t = idx / N;
        int const j = idx % N;
        h_post_out_g[(token_base + t) * N + j] = h_post(t, j);
      }
    }

    // ---- Stage K3 (sinkhorn 4x4) ----
    int const k3_token = warp * 32 + lane;
    if (warp < SINKHORN_WARPS && k3_token < tokens_this_iter) {
      int const t = k3_token;
      float m00 = h_res(t, 0), m01 = h_res(t, 1);
      float m02 = h_res(t, 2), m03 = h_res(t, 3);
      float m10 = h_res(t, 4), m11 = h_res(t, 5);
      float m12 = h_res(t, 6), m13 = h_res(t, 7);
      float m20 = h_res(t, 8), m21 = h_res(t, 9);
      float m22 = h_res(t, 10), m23 = h_res(t, 11);
      float m30 = h_res(t, 12), m31 = h_res(t, 13);
      float m32 = h_res(t, 14), m33 = h_res(t, 15);

      float const rmax0 = fmaxf(fmaxf(m00, m01), fmaxf(m02, m03));
      float const rmax1 = fmaxf(fmaxf(m10, m11), fmaxf(m12, m13));
      float const rmax2 = fmaxf(fmaxf(m20, m21), fmaxf(m22, m23));
      float const rmax3 = fmaxf(fmaxf(m30, m31), fmaxf(m32, m33));
      m00 = __expf(m00 - rmax0);
      m01 = __expf(m01 - rmax0);
      m02 = __expf(m02 - rmax0);
      m03 = __expf(m03 - rmax0);
      m10 = __expf(m10 - rmax1);
      m11 = __expf(m11 - rmax1);
      m12 = __expf(m12 - rmax1);
      m13 = __expf(m13 - rmax1);
      m20 = __expf(m20 - rmax2);
      m21 = __expf(m21 - rmax2);
      m22 = __expf(m22 - rmax2);
      m23 = __expf(m23 - rmax2);
      m30 = __expf(m30 - rmax3);
      m31 = __expf(m31 - rmax3);
      m32 = __expf(m32 - rmax3);
      m33 = __expf(m33 - rmax3);

      float const rs0 = m00 + m01 + m02 + m03;
      float const rs1 = m10 + m11 + m12 + m13;
      float const rs2 = m20 + m21 + m22 + m23;
      float const rs3 = m30 + m31 + m32 + m33;
      float const ri0 = __frcp_rn(rs0);
      float const ri1 = __frcp_rn(rs1);
      float const ri2 = __frcp_rn(rs2);
      float const ri3 = __frcp_rn(rs3);
      m00 = m00 * ri0 + sinkhorn_eps;
      m01 = m01 * ri0 + sinkhorn_eps;
      m02 = m02 * ri0 + sinkhorn_eps;
      m03 = m03 * ri0 + sinkhorn_eps;
      m10 = m10 * ri1 + sinkhorn_eps;
      m11 = m11 * ri1 + sinkhorn_eps;
      m12 = m12 * ri1 + sinkhorn_eps;
      m13 = m13 * ri1 + sinkhorn_eps;
      m20 = m20 * ri2 + sinkhorn_eps;
      m21 = m21 * ri2 + sinkhorn_eps;
      m22 = m22 * ri2 + sinkhorn_eps;
      m23 = m23 * ri2 + sinkhorn_eps;
      m30 = m30 * ri3 + sinkhorn_eps;
      m31 = m31 * ri3 + sinkhorn_eps;
      m32 = m32 * ri3 + sinkhorn_eps;
      m33 = m33 * ri3 + sinkhorn_eps;

      int const steps = sinkhorn_repeat > 0 ? sinkhorn_repeat : 1;
      int const dyn_steps = steps; // alias so the loop-exit check compiles in both paths
#pragma unroll 1
      for (int it = 0; it < steps; ++it) {
        float const cs0 = m00 + m10 + m20 + m30 + sinkhorn_eps;
        float const cs1 = m01 + m11 + m21 + m31 + sinkhorn_eps;
        float const cs2 = m02 + m12 + m22 + m32 + sinkhorn_eps;
        float const cs3 = m03 + m13 + m23 + m33 + sinkhorn_eps;
        float const ci0 = __frcp_rn(cs0);
        float const ci1 = __frcp_rn(cs1);
        float const ci2 = __frcp_rn(cs2);
        float const ci3 = __frcp_rn(cs3);
        m00 *= ci0;
        m10 *= ci0;
        m20 *= ci0;
        m30 *= ci0;
        m01 *= ci1;
        m11 *= ci1;
        m21 *= ci1;
        m31 *= ci1;
        m02 *= ci2;
        m12 *= ci2;
        m22 *= ci2;
        m32 *= ci2;
        m03 *= ci3;
        m13 *= ci3;
        m23 *= ci3;
        m33 *= ci3;
        if (it == dyn_steps - 1) {
          break;
        }
        float const rs0i = m00 + m01 + m02 + m03 + sinkhorn_eps;
        float const rs1i = m10 + m11 + m12 + m13 + sinkhorn_eps;
        float const rs2i = m20 + m21 + m22 + m23 + sinkhorn_eps;
        float const rs3i = m30 + m31 + m32 + m33 + sinkhorn_eps;
        float const ri0i = __frcp_rn(rs0i);
        float const ri1i = __frcp_rn(rs1i);
        float const ri2i = __frcp_rn(rs2i);
        float const ri3i = __frcp_rn(rs3i);
        m00 *= ri0i;
        m01 *= ri0i;
        m02 *= ri0i;
        m03 *= ri0i;
        m10 *= ri1i;
        m11 *= ri1i;
        m12 *= ri1i;
        m13 *= ri1i;
        m20 *= ri2i;
        m21 *= ri2i;
        m22 *= ri2i;
        m23 *= ri2i;
        m30 *= ri3i;
        m31 *= ri3i;
        m32 *= ri3i;
        m33 *= ri3i;
      }

      int const token = token_base + t;
      *reinterpret_cast<float4 *>(&comb(t, 0)) =
          make_float4(m00, m01, m02, m03);
      *reinterpret_cast<float4 *>(&comb(t, 4)) =
          make_float4(m10, m11, m12, m13);
      *reinterpret_cast<float4 *>(&comb(t, 8)) =
          make_float4(m20, m21, m22, m23);
      *reinterpret_cast<float4 *>(&comb(t, 12)) =
          make_float4(m30, m31, m32, m33);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 0) =
          make_float4(m00, m01, m02, m03);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 4) =
          make_float4(m10, m11, m12, m13);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 8) =
          make_float4(m20, m21, m22, m23);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 12) =
          make_float4(m30, m31, m32, m33);
    }
    // No sync between K3 and K4 (see static-smem variant for rationale).

    // ---- Stage K4 (pre_mix-weighted residual sum) ----
    constexpr int VEC = 8;
    static_assert(C % VEC == 0, "C must be a multiple of 8");
    int const c_vec_count = C / VEC;
    int const total_work = tokens_this_iter * c_vec_count;

    auto issue_loads = [&](int li, uint4 &r0, uint4 &r1, uint4 &r2, uint4 &r3) {
      int const t = li / c_vec_count;
      int const v = li % c_vec_count;
      int const token = token_base + t;
      T_in const *x_t = x + token * N * C;
      uint4 const *__restrict__ x_v0 =
          reinterpret_cast<uint4 const *>(x_t + 0 * C);
      uint4 const *__restrict__ x_v1 =
          reinterpret_cast<uint4 const *>(x_t + 1 * C);
      uint4 const *__restrict__ x_v2 =
          reinterpret_cast<uint4 const *>(x_t + 2 * C);
      uint4 const *__restrict__ x_v3 =
          reinterpret_cast<uint4 const *>(x_t + 3 * C);
      r0 = x_v0[v];
      r1 = x_v1[v];
      r2 = x_v2[v];
      r3 = x_v3[v];
    };

    auto compute_store = [&](int li, uint4 r0, uint4 r1, uint4 r2, uint4 r3) {
      int const t = li / c_vec_count;
      int const v = li % c_vec_count;
      int const token = token_base + t;
      T_in *f_pre_t = f_pre + token * C;
      uint4 *__restrict__ f_v = reinterpret_cast<uint4 *>(f_pre_t);
      float const w0 = h_pre(t, 0);
      float const w1 = h_pre(t, 1);
      float const w2 = h_pre(t, 2);
      float const w3 = h_pre(t, 3);
      __nv_bfloat162 const *b0 = reinterpret_cast<__nv_bfloat162 const *>(&r0);
      __nv_bfloat162 const *b1 = reinterpret_cast<__nv_bfloat162 const *>(&r1);
      __nv_bfloat162 const *b2 = reinterpret_cast<__nv_bfloat162 const *>(&r2);
      __nv_bfloat162 const *b3 = reinterpret_cast<__nv_bfloat162 const *>(&r3);
      float out_f[VEC];
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        float2 v0 = __bfloat1622float2(b0[k]);
        float2 v1 = __bfloat1622float2(b1[k]);
        float2 v2 = __bfloat1622float2(b2[k]);
        float2 v3 = __bfloat1622float2(b3[k]);
        out_f[2 * k + 0] = w0 * v0.x + w1 * v1.x + w2 * v2.x + w3 * v3.x;
        out_f[2 * k + 1] = w0 * v0.y + w1 * v1.y + w2 * v2.y + w3 * v3.y;
      }
      uint4 packed;
      __nv_bfloat162 *p = reinterpret_cast<__nv_bfloat162 *>(&packed);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        p[k] = __floats2bfloat162_rn(out_f[2 * k + 0], out_f[2 * k + 1]);
      }
      f_v[v] = packed;
    };

    int const my_first = threadIdx.x;
    if (my_first < total_work) {
      uint4 r0_cur, r1_cur, r2_cur, r3_cur;
      issue_loads(my_first, r0_cur, r1_cur, r2_cur, r3_cur);
      for (int li = my_first; li < total_work; li += BLOCK_THREADS) {
        int const li_next = li + BLOCK_THREADS;
        uint4 r0_next, r1_next, r2_next, r3_next;
        if (li_next < total_work) {
          issue_loads(li_next, r0_next, r1_next, r2_next, r3_next);
        }
        compute_store(li, r0_cur, r1_cur, r2_cur, r3_cur);
        r0_cur = r0_next;
        r1_cur = r1_next;
        r2_cur = r2_next;
        r3_cur = r3_next;
      }
    }
    __syncthreads();
  }
}

// ----------------------------------------------------------------------------
// Low-t k2: ONE CTA per token (grid = num_tokens), so the grid fills the SMs at
// small batch -- the default k2 packs 32 tokens/CTA, giving only ceil(t/32)
// blocks (e.g. 2 blocks at t=64 -> 12.5% occupancy, launch/serialization
// bound). Here warp 0 does the token's affine + 4x4 sinkhorn into smem, then
// the whole block vectorizes that token's C-wide pre_mix-weighted residual sum.
// Identical math to mHC_pre_k2_task_impl; only the token->block mapping
// differs.
//
// RDSPLIT_K: when 0, reads the pre-reduced mixes_pad (bf16) + sqrsum as usual.
// When >0, reads the k1 GEMM's fp32 partials (out_partial[RDSPLIT_K,tokens,
// MIX_HC] + sqr_partial[RDSPLIT_K,tokens]) and reduces them INLINE in warp 0 --
// folding the separate reduce kernel into this k2, so the low-t pre runs in 2
// launches (GEMM + this) instead of 3. mixes_ptr/sqrsum_ptr then point at
// out_partial/sqr_partial.
// ----------------------------------------------------------------------------
template <typename T_in,
          int N,
          int C,
          int RMS_HIDDEN,
          int BLOCK_THREADS = 256,
          int MIX_STRIDE = 0,
          int RDSPLIT_K = 0,
          int SINKHORN_REPEAT = 0>
__device__ __forceinline__ void
    mHC_pre_k2_lowt_task_impl(void const *mixes_ptr,
                              void const *sqrsum_ptr,
                              void const *scale_ptr,
                              void const *base_ptr,
                              void const *x_ptr,
                              void *f_pre_ptr,
                              void *h_post_out_ptr,
                              void *comb_out_ptr,
                              int sinkhorn_repeat,
                              float sinkhorn_eps,
                              float rms_eps,
                              int num_tokens) {
  static_assert(N == 4, "pre K2 hardcoded to n=4");
  static_assert(C % 8 == 0, "C must be a multiple of 8");
  constexpr int MIX_HC = N * N + 2 * N;
  constexpr int MIX_ROW_STRIDE = (MIX_STRIDE == 0) ? MIX_HC : MIX_STRIDE;

  int const token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;

  T_in const *mixes = static_cast<T_in const *>(mixes_ptr);
  float const *sqrsum = static_cast<float const *>(sqrsum_ptr);
  float const *scale = static_cast<float const *>(scale_ptr);
  float const *base = static_cast<float const *>(base_ptr);
  T_in const *x = static_cast<T_in const *>(x_ptr);
  T_in *f_pre = static_cast<T_in *>(f_pre_ptr);
  float *h_post_out_g = static_cast<float *>(h_post_out_ptr);
  float *comb_out = static_cast<float *>(comb_out_ptr);

  float const alpha_pre = scale[0];
  float const alpha_post = scale[1];
  float const alpha_res = scale[2];

  // Per-token shared state: pre_mix weights (4) broadcast from warp 0 to K4.
  __shared__ float s_pre[N];

  if (warp == 0) {
    // sqrsum: pre-reduced read, or inline reduce of the k1 sqr_partial.
    float sqr_val;
    if (RDSPLIT_K == 0) {
      sqr_val = sqrsum[token];
    } else {
      float const *sqr_partial = static_cast<float const *>(sqrsum_ptr);
      sqr_val = 0.0f;
#pragma unroll
      for (int s = 0; s < RDSPLIT_K; ++s) {
        sqr_val += sqr_partial[(int64_t)s * num_tokens + token];
      }
    }
    float const ms = sqr_val / static_cast<float>(RMS_HIDDEN);
    float const rms_scale = rsqrtf(ms + rms_eps);

    // Affine over the MIX_HC=24 mixes: lanes 0..23 each handle one column.
    // pre (j<N) -> sigmoid into s_pre; post (N<=j<2N) -> sigmoid*2 -> gmem;
    // comb (j>=2N) -> raw logit into a register, gathered for sinkhorn below.
    float h_res_local = 0.0f; // valid for comb lanes (j in [2N, MIX_HC))
    if (lane < MIX_HC) {
      int const j = lane;
      // mix[j]: pre-reduced bf16 read, or inline fp32 reduce of out_partial.
      float mix_raw;
      if (RDSPLIT_K == 0) {
        mix_raw = static_cast<float>(mixes[token * MIX_ROW_STRIDE + j]);
      } else {
        float const *out_partial = static_cast<float const *>(mixes_ptr);
        mix_raw = 0.0f;
#pragma unroll
        for (int s = 0; s < RDSPLIT_K; ++s) {
          mix_raw +=
              out_partial[((int64_t)s * num_tokens + token) * MIX_HC + j];
        }
      }
      float const mix = mix_raw * rms_scale;
      float const bias = base[j];
      if (j < N) {
        float const y = mix * alpha_pre + bias;
        s_pre[j] = 1.0f / (1.0f + __expf(-y));
      } else if (j < 2 * N) {
        float const y = mix * alpha_post + bias;
        float const hp = 2.0f / (1.0f + __expf(-y));
        h_post_out_g[token * N + (j - N)] = hp;
      } else {
        h_res_local = mix * alpha_res + bias;
      }
    }
    // Gather the 16 comb logits (lanes 2N..2N+15) to lane 0 via shuffle.
    float cm[N * N];
#pragma unroll
    for (int e = 0; e < N * N; ++e) {
      cm[e] = __shfl_sync(0xffffffff, h_res_local, 2 * N + e);
    }

    if (lane == 0) {
      // ---- Sinkhorn (4x4), identical to the batched path ----
      float m00 = cm[0], m01 = cm[1], m02 = cm[2], m03 = cm[3];
      float m10 = cm[4], m11 = cm[5], m12 = cm[6], m13 = cm[7];
      float m20 = cm[8], m21 = cm[9], m22 = cm[10], m23 = cm[11];
      float m30 = cm[12], m31 = cm[13], m32 = cm[14], m33 = cm[15];
      float const rmax0 = fmaxf(fmaxf(m00, m01), fmaxf(m02, m03));
      float const rmax1 = fmaxf(fmaxf(m10, m11), fmaxf(m12, m13));
      float const rmax2 = fmaxf(fmaxf(m20, m21), fmaxf(m22, m23));
      float const rmax3 = fmaxf(fmaxf(m30, m31), fmaxf(m32, m33));
      m00 = __expf(m00 - rmax0);
      m01 = __expf(m01 - rmax0);
      m02 = __expf(m02 - rmax0);
      m03 = __expf(m03 - rmax0);
      m10 = __expf(m10 - rmax1);
      m11 = __expf(m11 - rmax1);
      m12 = __expf(m12 - rmax1);
      m13 = __expf(m13 - rmax1);
      m20 = __expf(m20 - rmax2);
      m21 = __expf(m21 - rmax2);
      m22 = __expf(m22 - rmax2);
      m23 = __expf(m23 - rmax2);
      m30 = __expf(m30 - rmax3);
      m31 = __expf(m31 - rmax3);
      m32 = __expf(m32 - rmax3);
      m33 = __expf(m33 - rmax3);
      float const ri0 = __frcp_rn(m00 + m01 + m02 + m03);
      float const ri1 = __frcp_rn(m10 + m11 + m12 + m13);
      float const ri2 = __frcp_rn(m20 + m21 + m22 + m23);
      float const ri3 = __frcp_rn(m30 + m31 + m32 + m33);
      m00 = m00 * ri0 + sinkhorn_eps;
      m01 = m01 * ri0 + sinkhorn_eps;
      m02 = m02 * ri0 + sinkhorn_eps;
      m03 = m03 * ri0 + sinkhorn_eps;
      m10 = m10 * ri1 + sinkhorn_eps;
      m11 = m11 * ri1 + sinkhorn_eps;
      m12 = m12 * ri1 + sinkhorn_eps;
      m13 = m13 * ri1 + sinkhorn_eps;
      m20 = m20 * ri2 + sinkhorn_eps;
      m21 = m21 * ri2 + sinkhorn_eps;
      m22 = m22 * ri2 + sinkhorn_eps;
      m23 = m23 * ri2 + sinkhorn_eps;
      m30 = m30 * ri3 + sinkhorn_eps;
      m31 = m31 * ri3 + sinkhorn_eps;
      m32 = m32 * ri3 + sinkhorn_eps;
      m33 = m33 * ri3 + sinkhorn_eps;
      // SINKHORN_REPEAT>0: compile-time unroll; 0: runtime count from arg.
      int const dyn_steps = (SINKHORN_REPEAT > 0) ? SINKHORN_REPEAT
                            : (sinkhorn_repeat > 0 ? sinkhorn_repeat : 1);
#pragma unroll 1
      for (int it = 0; it < dyn_steps; ++it) {
        float const ci0 = __frcp_rn(m00 + m10 + m20 + m30 + sinkhorn_eps);
        float const ci1 = __frcp_rn(m01 + m11 + m21 + m31 + sinkhorn_eps);
        float const ci2 = __frcp_rn(m02 + m12 + m22 + m32 + sinkhorn_eps);
        float const ci3 = __frcp_rn(m03 + m13 + m23 + m33 + sinkhorn_eps);
        m00 *= ci0;
        m10 *= ci0;
        m20 *= ci0;
        m30 *= ci0;
        m01 *= ci1;
        m11 *= ci1;
        m21 *= ci1;
        m31 *= ci1;
        m02 *= ci2;
        m12 *= ci2;
        m22 *= ci2;
        m32 *= ci2;
        m03 *= ci3;
        m13 *= ci3;
        m23 *= ci3;
        m33 *= ci3;
        if (it == dyn_steps - 1) {
          break;
        }
        float const ri0i = __frcp_rn(m00 + m01 + m02 + m03 + sinkhorn_eps);
        float const ri1i = __frcp_rn(m10 + m11 + m12 + m13 + sinkhorn_eps);
        float const ri2i = __frcp_rn(m20 + m21 + m22 + m23 + sinkhorn_eps);
        float const ri3i = __frcp_rn(m30 + m31 + m32 + m33 + sinkhorn_eps);
        m00 *= ri0i;
        m01 *= ri0i;
        m02 *= ri0i;
        m03 *= ri0i;
        m10 *= ri1i;
        m11 *= ri1i;
        m12 *= ri1i;
        m13 *= ri1i;
        m20 *= ri2i;
        m21 *= ri2i;
        m22 *= ri2i;
        m23 *= ri2i;
        m30 *= ri3i;
        m31 *= ri3i;
        m32 *= ri3i;
        m33 *= ri3i;
      }
      float cf[N * N] = {m00,
                         m01,
                         m02,
                         m03,
                         m10,
                         m11,
                         m12,
                         m13,
                         m20,
                         m21,
                         m22,
                         m23,
                         m30,
                         m31,
                         m32,
                         m33};
#pragma unroll
      for (int e = 0; e < N * N; ++e) {
        comb_out[token * N * N + e] = cf[e];
      }
    }
  }
  __syncthreads();

  // ---- K4: whole block vectorizes this token's C-wide weighted sum ----
  float const w0 = s_pre[0], w1 = s_pre[1], w2 = s_pre[2], w3 = s_pre[3];
  constexpr int VEC = 8;
  int const c_vec_count = C / VEC;
  T_in const *x_t = x + (int64_t)token * N * C;
  uint4 const *x_v0 = reinterpret_cast<uint4 const *>(x_t + 0 * C);
  uint4 const *x_v1 = reinterpret_cast<uint4 const *>(x_t + 1 * C);
  uint4 const *x_v2 = reinterpret_cast<uint4 const *>(x_t + 2 * C);
  uint4 const *x_v3 = reinterpret_cast<uint4 const *>(x_t + 3 * C);
  uint4 *f_v = reinterpret_cast<uint4 *>(f_pre + (int64_t)token * C);

  for (int v = tid; v < c_vec_count; v += BLOCK_THREADS) {
    uint4 r0 = x_v0[v], r1 = x_v1[v], r2 = x_v2[v], r3 = x_v3[v];
    __nv_bfloat162 const *b0 = reinterpret_cast<__nv_bfloat162 const *>(&r0);
    __nv_bfloat162 const *b1 = reinterpret_cast<__nv_bfloat162 const *>(&r1);
    __nv_bfloat162 const *b2 = reinterpret_cast<__nv_bfloat162 const *>(&r2);
    __nv_bfloat162 const *b3 = reinterpret_cast<__nv_bfloat162 const *>(&r3);
    float out_f[VEC];
#pragma unroll
    for (int k = 0; k < VEC / 2; ++k) {
      float2 v0 = __bfloat1622float2(b0[k]);
      float2 v1 = __bfloat1622float2(b1[k]);
      float2 v2 = __bfloat1622float2(b2[k]);
      float2 v3 = __bfloat1622float2(b3[k]);
      out_f[2 * k + 0] = w0 * v0.x + w1 * v1.x + w2 * v2.x + w3 * v3.x;
      out_f[2 * k + 1] = w0 * v0.y + w1 * v1.y + w2 * v2.y + w3 * v3.y;
    }
    uint4 packed;
    __nv_bfloat162 *p = reinterpret_cast<__nv_bfloat162 *>(&packed);
#pragma unroll
    for (int k = 0; k < VEC / 2; ++k) {
      p[k] = __floats2bfloat162_rn(out_f[2 * k + 0], out_f[2 * k + 1]);
    }
    f_v[v] = packed;
  }
}

} // namespace kernel
