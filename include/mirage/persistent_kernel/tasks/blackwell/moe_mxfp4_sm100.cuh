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

// SM100 grouped GEMM for GPT-OSS MXFP4 experts.
//
// Weights stay packed E2M1 + UE8M0 in HBM (4.25 bits/param). Each K tile is
// dequantized into the bf16 K-major swizzle the tcgen05.kind::f16 MMA reads,
// so the accumulator matches a bf16 matmul of the dequantized weights.
// Activations stay bf16: the checkpoint was trained that way.
//
// swapAB, same tile as moe_linear_sm100: MMA_M is the output column (N),
// MMA_N is the token tile. One MMA stays in flight while the next K tile is
// loaded. GPT-OSS decode/prefill chunks are M<=16, so the run is the MXFP4
// weight load; the tensor-core MMA is overlapped with that load.
//
// blocks  [E, orig_N, K/2]   uint8, this CTA's pointer is at its column slice
// scales  [E, orig_N, K/32]  uint8
// bias    [E, orig_N]        bf16 (checkpoint stores fp32; the loader casts)
// routing [E, B]             int32, value is the topk slot + 1, or 0
// mask    [E+1]              int32, mask[E] is the active-expert count

// cute/tensor.hpp must be included before storage.cuh. storage.cuh pulls
// cooperative_copy.hpp, and nvcc rejects CuTe's pack-not-last templates when
// that header is the first CuTe include.
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "mxfp4.cuh"
#include "storage.cuh"

#include <cstdint>

namespace kernel {

template <int BATCH,
          int OUTPUT_SIZE,
          int ORIG_OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          int OUTPUT_STRIDE,
          bool W13_LINEAR,
          bool NO_BIAS>
__device__ __forceinline__ void moe_mxfp4_sm100_task_impl(
    cute::bfloat16_t const *__restrict__ input,
    uint8_t const *__restrict__ blocks,
    uint8_t const *__restrict__ scales,
    int32_t const *__restrict__ routing,
    int32_t const *__restrict__ mask,
    cute::bfloat16_t const *__restrict__ bias,
    cute::bfloat16_t *__restrict__ output,
    int expert_offset) {
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int BK = 64;
  constexpr int NUM_STAGES = 2;
  constexpr int NUM_ACC_STAGE = 1;
  static_assert(REDUCTION_SIZE % BK == 0, "K must be a multiple of the MMA K tile");
  static_assert(BK == 64, "bf16 tcgen05 K atom is 16; four atoms make the K tile");

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane = cutlass::canonical_lane_idx();

  cute::TiledMMA tiled_mma = cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_SS<cute::bfloat16_t,
                                 cute::bfloat16_t,
                                 float,
                                 MMA_M,
                                 MMA_N,
                                 cute::UMMA::Major::K,
                                 cute::UMMA::Major::K>{});
  auto bK = cute::tile_size<2>(tiled_mma) * cute::Int<4>{};
  auto mma_tiler = cute::make_shape(cute::Int<MMA_M>{}, cute::Int<MMA_N>{}, bK);

  auto mma_shape_A = cute::partition_shape_A(
      tiled_mma,
      cute::make_shape(cute::Int<MMA_M>{}, cute::size<2>(mma_tiler), cute::Int<NUM_STAGES>{}));
  auto mma_shape_B = cute::partition_shape_B(
      tiled_mma,
      cute::make_shape(cute::Int<MMA_N>{}, cute::size<2>(mma_tiler), cute::Int<NUM_STAGES>{}));
  auto sA_layout =
      cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{}, mma_shape_A);
  auto sB_layout =
      cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{}, mma_shape_B);

  // Logical (row, k, 1, stage), K-major, over the same swizzled storage the
  // MMA descriptor walks. Stores through this layout are what the UMMA reads.
  auto sA_cp_layout = cute::composition(
      sA_layout.layout_a(),
      sA_layout.offset(),
      cute::make_layout(cute::make_shape(cute::Int<MMA_M>{}, bK, cute::Int<1>{}, cute::Int<NUM_STAGES>{}),
                        cute::make_stride(bK, cute::Int<1>{}, cute::Int<0>{}, cute::Int<MMA_M>{} * bK)));
  auto sB_cp_layout = cute::composition(
      sB_layout.layout_a(),
      sB_layout.offset(),
      cute::make_layout(cute::make_shape(cute::Int<MMA_N>{}, bK, cute::Int<1>{}, cute::Int<NUM_STAGES>{}),
                        cute::make_stride(bK, cute::Int<1>{}, cute::Int<0>{}, cute::Int<MMA_N>{} * bK)));

  using SharedStorage = MoESharedStorage<cute::bfloat16_t,
                                         cute::bfloat16_t,
                                         decltype(sA_layout),
                                         decltype(sB_layout),
                                         decltype(sB_cp_layout),
                                         NUM_EXPERTS,
                                         NUM_STAGES,
                                         NUM_ACC_STAGE>;
  struct MbarStore {
    alignas(16) uint64_t mma_done;
  };

  extern __shared__ char shared_memory[];
  uintptr_t aligned = (reinterpret_cast<uintptr_t>(shared_memory) + 127) & ~uintptr_t{127};
  SharedStorage &shared = *reinterpret_cast<SharedStorage *>(aligned);
  MbarStore &mbars = *reinterpret_cast<MbarStore *>(aligned + sizeof(SharedStorage));

  cute::Tensor sA = cute::make_tensor(cute::make_smem_ptr(shared.A.begin()), sA_cp_layout);
  cute::Tensor sB = cute::make_tensor(cute::make_smem_ptr(shared.B.begin()), sB_cp_layout);

  cute::ThrMMA cta_mma = tiled_mma.get_slice(0);
  cute::Tensor tCsA = shared.tensor_sA();
  cute::Tensor tCsB = shared.tensor_sB();
  cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  auto acc_shape = cute::partition_shape_C(
      tiled_mma, cute::make_shape(cute::size<0>(mma_tiler), cute::size<1>(mma_tiler), cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);

  constexpr int k_tiles = REDUCTION_SIZE / BK;
  constexpr int m_tiles = (OUTPUT_SIZE + MMA_M - 1) / MMA_M;
  constexpr int n_tiles = (BATCH + MMA_N - 1) / MMA_N;
  constexpr int a_elems = MMA_M * BK;
  constexpr int b_elems = MMA_N * BK;

  if (threadIdx.x == 0) {
    cutlass::arch::ClusterBarrier::init(&mbars.mma_done, 1);
  }
  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (warp_idx == 0) {
    tmem_allocator.allocate(MMA_N * NUM_ACC_STAGE, &shared.tmem_base_ptr);
  }
  __syncthreads();
  tCtAcc.data() = shared.tmem_base_ptr;

  auto load_tile = [&](int expert, int m_tile, int n_tile, int k_tile, int stage, int tid, int nthreads) {
    if (tid < 0) {
      return;
    }
    for (int e = tid; e < a_elems; e += nthreads) {
      int row = e / BK;
      int k = e - row * BK;
      int grow = m_tile * MMA_M + row;
      float val = 0.f;
      if (grow < OUTPUT_SIZE) {
        int gk = k_tile * BK + k;
        size_t row_index = static_cast<size_t>(expert) * ORIG_OUTPUT_SIZE + grow;
        val = mxfp4::dequant(blocks + row_index * (REDUCTION_SIZE / 2),
                             scales + row_index * (REDUCTION_SIZE / 32),
                             gk);
      }
      sA(row, k, 0, stage) = cute::bfloat16_t(val);
    }
    for (int e = tid; e < b_elems; e += nthreads) {
      int n = e / BK;
      int k = e - n * BK;
      int token = n_tile * MMA_N + n;
      float val = 0.f;
      if (token < BATCH) {
        int slot = routing[expert * BATCH + token];
        if (slot > 0) {
          int gk = k_tile * BK + k;
          cute::bfloat16_t src;
          if constexpr (W13_LINEAR) {
            src = input[static_cast<size_t>(token) * REDUCTION_SIZE + gk];
          } else {
            src = input[(static_cast<size_t>(token) * NUM_TOPK + (slot - 1)) * REDUCTION_SIZE + gk];
          }
          val = static_cast<float>(src);
        }
      }
      sB(n, k, 0, stage) = cute::bfloat16_t(val);
    }
  };

  int num_activated = mask[NUM_EXPERTS];
  for (int slot = expert_offset; slot < num_activated; slot += EXPERT_STRIDE) {
    int expert = mask[slot];
    for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
      for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
        load_tile(expert, m_tile, n_tile, 0, 0, threadIdx.x, 256);
        __syncthreads();

        int phase = 0;
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
          int stage = k_tile & 1;
          int next = stage ^ 1;
          if (warp_idx == 4) {
            tiled_mma.accumulate_ =
                (k_tile == 0) ? cute::UMMA::ScaleOut::Zero : cute::UMMA::ScaleOut::One;
            auto acc = tCtAcc(cute::_, cute::_, cute::_, 0);
            CUTE_UNROLL
            for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
              cute::gemm(tiled_mma,
                         tCrA(cute::_, cute::_, k_block, stage),
                         tCrB(cute::_, cute::_, k_block, stage),
                         acc);
              tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(&mbars.mma_done);
          } else if (k_tile + 1 < k_tiles) {
            int loader = threadIdx.x < 128 ? threadIdx.x : threadIdx.x - 32;
            load_tile(expert, m_tile, n_tile, k_tile + 1, next, loader, 224);
          }
          if (warp_idx == 4 && lane == 0) {
            cute::wait_barrier(mbars.mma_done, phase);
          }
          if (warp_idx == 4) {
            __syncwarp();
          }
          __syncthreads();
          phase ^= 1;
        }

        if (warp_idx < 4) {
          using AccType = float;
          cutlass::NumericConverter<cute::bfloat16_t, AccType> to_bf16;
          cute::TiledCopy tiled_copy_t2r =
              cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc(cute::_, cute::_, cute::_, 0));
          cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
          cute::Tensor tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc);
          cute::Tensor tCgC_fake =
              cute::make_tensor<cute::bfloat16_t>(cute::shape(tCtAcc(cute::_, cute::_, cute::_, 0)));
          cute::Tensor tTR_rAcc_fake = thr_copy_t2r.partition_D(tCgC_fake);
          cute::Tensor tTR_rAcc = cute::make_tensor<AccType>(cute::shape(tTR_rAcc_fake));
          cute::copy(tiled_copy_t2r, tTR_tAcc(cute::_, cute::_, cute::_, cute::_, 0), tTR_rAcc);

          int out_col = m_tile * MMA_M + threadIdx.x;
          bool col_ok = out_col < OUTPUT_SIZE;
          float bias_v = 0.f;
          if constexpr (!NO_BIAS) {
            if (col_ok) {
              bias_v = static_cast<float>(bias[static_cast<size_t>(expert) * ORIG_OUTPUT_SIZE + out_col]);
            }
          }
          CUTE_UNROLL
          for (int i = 0; i < MMA_N; ++i) {
            int token = n_tile * MMA_N + i;
            if (token >= BATCH || !col_ok) {
              continue;
            }
            int slot_1 = routing[expert * BATCH + token];
            if (slot_1 <= 0) {
              continue;
            }
            // Same fragment order as moe_linear_sm100: linear i is token i
            // of this column.
            float acc = tTR_rAcc[i];
            if constexpr (!NO_BIAS) {
              acc += bias_v;
            }
            output[(static_cast<size_t>(token) * NUM_TOPK + (slot_1 - 1)) * OUTPUT_STRIDE + out_col] =
                to_bf16(acc);
          }
        }
        __syncthreads();
      }
    }
  }

  __syncthreads();
  if (warp_idx == 0) {
    tmem_allocator.free(shared.tmem_base_ptr, MMA_N * NUM_ACC_STAGE);
  }
}

} // namespace kernel
