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
// Both MMA operands are E2M1. Weights stay packed. Activations are bf16 and
// are quantized to E2M1 + UE8M0 per 32-K block here. The accumulator matches
// a dot product of those dequantized values.
//
// Tile is 128x128x256 (M = output columns, N = token slots). K is padded with
// zeros to a multiple of 256. One K tile is loaded, its scales are copied to
// TMEM, then four block-scaled MMAs consume it. A CTA owns one output slice
// and walks its experts.

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_tmem_helper.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "mxfp4.cuh"

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
  using cute::Int;
  using cute::_;
  using Element = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue8m0_t;

  constexpr int MMA_M = 128;
  constexpr int MMA_N = 128;
  constexpr int BK = 256;
  constexpr int TMEM_COLUMNS = 512;
  constexpr int k_tiles = (REDUCTION_SIZE + BK - 1) / BK;
  constexpr int kSmemA = 0;
  constexpr int kSmemB = 32 * 1024;
  constexpr int kSmemSFA = 64 * 1024;
  constexpr int kSmemSFB = 72 * 1024;
  constexpr int kScaleClear = 4096;
  static_assert(BATCH <= MMA_N, "token tile is 128");
  static_assert(REDUCTION_SIZE % mxfp4::BLOCK == 0, "K must be a multiple of the MX block");

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane = cutlass::canonical_lane_idx();
  int tid = threadIdx.x;

  auto tiled_mma = cute::make_tiled_mma(
      cute::SM100_MMA_MXF4_SS<Element, Element, float, ElementSF, MMA_M, MMA_N, 32,
                              cute::UMMA::Major::K, cute::UMMA::Major::K>{});
  auto mma_shape_A = cute::partition_shape_A(tiled_mma, cute::make_shape(Int<MMA_M>{}, Int<BK>{}));
  auto mma_shape_B = cute::partition_shape_B(tiled_mma, cute::make_shape(Int<MMA_N>{}, Int<BK>{}));
  auto sA_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<Element>{}, cute::append(mma_shape_A, Int<1>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{});
  auto sB_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<Element>{}, cute::append(mma_shape_B, Int<1>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{});
  auto sA_cp = cute::composition(
      sA_layout.layout_a(), sA_layout.offset(),
      cute::make_layout(cute::make_shape(Int<MMA_M>{}, Int<BK>{}, Int<1>{}, Int<1>{}),
                        cute::make_stride(Int<BK>{}, Int<1>{}, Int<0>{}, Int<MMA_M * BK>{})));
  auto sB_cp = cute::composition(
      sB_layout.layout_a(), sB_layout.offset(),
      cute::make_layout(cute::make_shape(Int<MMA_N>{}, Int<BK>{}, Int<1>{}, Int<1>{}),
                        cute::make_stride(Int<BK>{}, Int<1>{}, Int<0>{}, Int<MMA_N * BK>{})));

  using Config = cutlass::detail::Sm1xxBlockScaledConfig<32>;
  auto sSFA_layout = Config::deduce_smem_layoutSFA(tiled_mma, cute::make_shape(Int<MMA_M>{}, Int<MMA_N>{}, Int<BK>{}));
  auto sSFB_layout = Config::deduce_smem_layoutSFB(tiled_mma, cute::make_shape(Int<MMA_M>{}, Int<MMA_N>{}, Int<BK>{}));

  extern __shared__ __align__(1024) char shared_memory[];
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(shared_memory);
  char *raw = shared_memory + ((1024u - (smem_addr & 1023u)) & 1023u);
  auto a_ptr = cute::make_smem_ptr(cute::subbyte_iterator<Element>(reinterpret_cast<uint8_t *>(raw + kSmemA)));
  auto b_ptr = cute::make_smem_ptr(cute::subbyte_iterator<Element>(reinterpret_cast<uint8_t *>(raw + kSmemB)));
  ElementSF *sfa_raw = reinterpret_cast<ElementSF *>(raw + kSmemSFA);
  ElementSF *sfb_raw = reinterpret_cast<ElementSF *>(raw + kSmemSFB);

  auto tCsA = cute::make_tensor(a_ptr, sA_layout);
  auto tCsB = cute::make_tensor(b_ptr, sB_layout);
  auto sA = cute::make_tensor(a_ptr, sA_cp);
  auto sB = cute::make_tensor(b_ptr, sB_cp);

  auto acc_shape = cute::partition_shape_C(tiled_mma, cute::make_shape(Int<MMA_M>{}, Int<MMA_N>{}, Int<1>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);
  auto tCtSFA = cute::make_tensor<typename decltype(tiled_mma)::FrgTypeSFA>(cute::shape(sSFA_layout));
  auto tCtSFB = cute::make_tensor<typename decltype(tiled_mma)::FrgTypeSFB>(cute::shape(sSFB_layout));

  __shared__ uint32_t tmem_base;
  __shared__ uint64_t mma_done;
  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (warp_idx == 0) {
    tmem_allocator.allocate(TMEM_COLUMNS, &tmem_base);
  }
  __syncthreads();
  tCtAcc.data() = tmem_base;
  tCtSFA.data() = tCtAcc.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tCtAcc);
  tCtSFB.data() = tCtSFA.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tCtSFA);

  auto tCrA = tiled_mma.make_fragment_A(tCsA);
  auto tCrB = tiled_mma.make_fragment_B(tCsB);
  auto tCsSFA = cute::make_tensor(cute::make_smem_ptr(sfa_raw), sSFA_layout);
  auto tCsSFB = cute::make_tensor(cute::make_smem_ptr(sfb_raw), sSFB_layout);
  auto tCsSFA_compact = cute::make_tensor(tCsSFA.data(), cute::filter_zeros(tCsSFA.layout()));
  auto tCtSFA_compact = cute::make_tensor(tCtSFA.data(), cute::filter_zeros(tCtSFA.layout()));
  auto tCsSFB_compact = cute::make_tensor(tCsSFB.data(), cute::filter_zeros(tCsSFB.layout()));
  auto tCtSFB_compact = cute::make_tensor(tCtSFB.data(), cute::filter_zeros(tCtSFB.layout()));
  auto tiled_copy_s2t_SFA = cute::make_utccp_copy(cute::SM100_UTCCP_4x32dp128bit_1cta{}, tCtSFA_compact);
  auto tiled_copy_s2t_SFB = cute::make_utccp_copy(cute::SM100_UTCCP_4x32dp128bit_1cta{}, tCtSFB_compact);
  auto thr_copy_s2t_SFA = tiled_copy_s2t_SFA.get_slice(0);
  auto thr_tCsSFA = cute::get_utccp_smem_desc_tensor<cute::SM100_UTCCP_4x32dp128bit_1cta>(
      thr_copy_s2t_SFA.partition_S(tCsSFA_compact));
  auto thr_tCtSFA = thr_copy_s2t_SFA.partition_D(tCtSFA_compact);
  auto thr_copy_s2t_SFB = tiled_copy_s2t_SFB.get_slice(0);
  auto thr_tCsSFB = cute::get_utccp_smem_desc_tensor<cute::SM100_UTCCP_4x32dp128bit_1cta>(
      thr_copy_s2t_SFB.partition_S(tCsSFB_compact));
  auto thr_tCtSFB = thr_copy_s2t_SFB.partition_D(tCtSFB_compact);

  // Scale SMEM atom: 32-row groups, then 4 scales packed per 32-bit word.
  auto sf_offset = [](int row, int s) {
    int row_base = (row & 31) * 16 + (row >> 5) * 4;
    return row_base + (s & 3) + (s >> 2) * 512;
  };

  constexpr int m_tiles = (OUTPUT_SIZE + MMA_M - 1) / MMA_M;
  int num_activated = mask[NUM_EXPERTS];
  for (int slot = expert_offset; slot < num_activated; slot += EXPERT_STRIDE) {
    int expert = mask[slot];
    for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
      for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        if (tid == 0) {
          cutlass::arch::ClusterBarrier::init(&mma_done, 1);
        }
        __syncthreads();
        for (int i = tid; i < kScaleClear; i += blockDim.x) {
          sfa_raw[i] = ElementSF::bitcast(uint8_t(0));
          sfb_raw[i] = ElementSF::bitcast(uint8_t(0));
        }
        __syncthreads();

        // Weights: one thread owns a 32-K block so both nibbles of each byte
        // are stored by the same thread.
        for (int job = tid; job < MMA_M * (BK / 32); job += blockDim.x) {
          int row = job / (BK / 32);
          int s = job - row * (BK / 32);
          int grow = m_tile * MMA_M + row;
          int gk = k_tile * BK + s * 32;
          uint8_t raw_bytes[16];
          uint8_t scale_byte = 127;
          if (grow < OUTPUT_SIZE && gk < REDUCTION_SIZE) {
            size_t row_index = static_cast<size_t>(expert) * ORIG_OUTPUT_SIZE + grow;
            uint8_t const *src = blocks + row_index * (REDUCTION_SIZE / 2) + (gk >> 1);
            int valid = REDUCTION_SIZE - gk;
            if (valid > 32) {
              valid = 32;
            }
            for (int b = 0; b < 16; ++b) {
              raw_bytes[b] = (b * 2 < valid) ? src[b] : 0;
            }
            if (valid > 0) {
              scale_byte = scales[row_index * (REDUCTION_SIZE / 32) + (gk >> 5)];
            }
          } else {
            for (int b = 0; b < 16; ++b) {
              raw_bytes[b] = 0;
            }
          }
          for (int b = 0; b < 16; ++b) {
            int k = s * 32 + b * 2;
            sA(row, k, 0, 0) = Element::bitcast(static_cast<uint8_t>(raw_bytes[b] & 15));
            sA(row, k + 1, 0, 0) = Element::bitcast(static_cast<uint8_t>(raw_bytes[b] >> 4));
          }
          sfa_raw[sf_offset(row, s)] = ElementSF::bitcast(scale_byte);
        }

        // Activations: quantize each live token's 32-K block.
        for (int job = tid; job < BATCH * (BK / 32); job += blockDim.x) {
          int token = job / (BK / 32);
          int s = job - token * (BK / 32);
          int gk = k_tile * BK + s * 32;
          float vals[32];
          float amax = 0.f;
          int slot_1 = 0;
          bool live = token < BATCH && gk < REDUCTION_SIZE;
          if (live) {
            slot_1 = routing[expert * BATCH + token];
            live = slot_1 > 0;
          }
          int valid = 0;
          if (live) {
            valid = REDUCTION_SIZE - gk;
            if (valid > 32) {
              valid = 32;
            }
            for (int k = 0; k < 32; ++k) {
              float v = 0.f;
              if (k < valid) {
                cute::bfloat16_t src;
                if constexpr (W13_LINEAR) {
                  src = input[static_cast<size_t>(token) * REDUCTION_SIZE + gk + k];
                } else {
                  src = input[(static_cast<size_t>(token) * NUM_TOPK + (slot_1 - 1)) * REDUCTION_SIZE + gk + k];
                }
                v = static_cast<float>(src);
              }
              vals[k] = v;
              amax = fmaxf(amax, fabsf(v));
            }
          }
          int scale_byte = mxfp4::quantize_ue8m0(amax);
          float inv = (amax > 0.f) ? 1.f / mxfp4::ue8m0(static_cast<unsigned>(scale_byte)) : 0.f;
          for (int k = 0; k < 32; k += 2) {
            int n0 = 0;
            int n1 = 0;
            if (live && k < valid) {
              n0 = mxfp4::quantize_e2m1(vals[k] * inv);
              if (k + 1 < valid) {
                n1 = mxfp4::quantize_e2m1(vals[k + 1] * inv);
              }
            }
            sB(token, s * 32 + k, 0, 0) = Element::bitcast(static_cast<uint8_t>(n0));
            sB(token, s * 32 + k + 1, 0, 0) = Element::bitcast(static_cast<uint8_t>(n1));
          }
          sfb_raw[sf_offset(token, s)] = ElementSF::bitcast(static_cast<uint8_t>(scale_byte));
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();

        if (warp_idx == 0) {
          if (cute::elect_one_sync()) {
            cute::copy(tiled_copy_s2t_SFA, thr_tCsSFA, thr_tCtSFA);
            cute::copy(tiled_copy_s2t_SFB, thr_tCsSFB, thr_tCtSFB);
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          tiled_mma.accumulate_ =
              (k_tile == 0) ? cute::UMMA::ScaleOut::Zero : cute::UMMA::ScaleOut::One;
          auto acc = tCtAcc(_, _, _, 0);
          CUTE_UNROLL
          for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
            cute::gemm(tiled_mma.with(tiled_mma.accumulate_, tCtSFA(_, _, k_block), tCtSFB(_, _, k_block)),
                       tCrA(_, _, k_block, 0), tCrB(_, _, k_block, 0), acc);
            tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
          }
          cutlass::arch::umma_arrive(&mma_done);
          if (lane == 0) {
            // Phase is re-inited to 0 at the start of this K tile.
            cutlass::arch::ClusterBarrier::wait(&mma_done, 0);
          }
        }
        __syncthreads();
      }

      if (warp_idx < 4) {
        using AccType = float;
        cutlass::NumericConverter<cute::bfloat16_t, AccType> to_bf16;
        cute::TiledCopy tiled_copy_t2r =
            cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc(_, _, _, 0));
        cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(tid);
        cute::Tensor tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc);
        cute::Tensor tCgC_fake = cute::make_tensor<cute::bfloat16_t>(cute::shape(tCtAcc(_, _, _, 0)));
        cute::Tensor tTR_rAcc_fake = thr_copy_t2r.partition_D(tCgC_fake);
        cute::Tensor tTR_rAcc = cute::make_tensor<AccType>(cute::shape(tTR_rAcc_fake));
        cute::copy(tiled_copy_t2r, tTR_tAcc(_, _, _, _, 0), tTR_rAcc);

        int out_col = m_tile * MMA_M + tid;
        bool col_ok = out_col < OUTPUT_SIZE;
        float bias_v = 0.f;
        if constexpr (!NO_BIAS) {
          if (col_ok) {
            bias_v = static_cast<float>(bias[static_cast<size_t>(expert) * ORIG_OUTPUT_SIZE + out_col]);
          }
        }
        CUTE_UNROLL
        for (int i = 0; i < MMA_N; ++i) {
          int token = i;
          if (token >= BATCH || !col_ok) {
            continue;
          }
          int slot_1 = routing[expert * BATCH + token];
          if (slot_1 <= 0) {
            continue;
          }
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

  __syncthreads();
  if (warp_idx == 0) {
    tmem_allocator.free(tmem_base, TMEM_COLUMNS);
  }
}

} // namespace kernel
