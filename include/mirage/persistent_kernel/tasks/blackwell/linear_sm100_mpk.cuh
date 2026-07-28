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

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "../hopper/tma.cuh"

#include "sm100_ptx.cuh"

namespace kernel {

template <typename T,
          int MMA_M,
          int MMA_N,
          int BK,
          int NUM_AB_STAGE,
          int NUM_ACC_STAGE,
          int NUM_C_STAGE>
struct LinearMpkSharedStorage {
  alignas(128) T A[NUM_AB_STAGE * MMA_M * BK];
  alignas(128) T B[NUM_AB_STAGE * MMA_N * BK];
  alignas(128) T C[NUM_C_STAGE * MMA_N * MMA_M];

  alignas(16) uint64_t ab_full_mbar[NUM_AB_STAGE];
  alignas(16) uint64_t ab_empty_mbar[NUM_AB_STAGE];
  alignas(16) uint64_t acc_full_mbar[NUM_ACC_STAGE];
  alignas(16) uint64_t acc_empty_mbar[NUM_ACC_STAGE];

  alignas(16) uint32_t tmem_base;
};

static constexpr int LINEAR_MPK_GROUP_M = 8;

template <typename T_,
          typename TMA_A,
          typename TMA_B,
          typename TMA_OUT,
          int MMA_M,
          int MMA_N,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          bool NOBIAS,
          bool SplitK,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__device__ __noinline__ void
    linear_sm100_mpk_task_impl(TMA_A const &tma_a,
                               TMA_B const &tma_b,
                               T_ const *mBias,
                               int output_stride,
                               TMA_OUT const &tma_out) {
  using namespace ::kernel::sm100_ptx;

  constexpr int BK = 64;
  constexpr int MMA_K = 16;
  constexpr int K_SUB = BK / MMA_K;
  constexpr int K_TILES = REDUCTION_SIZE / BK;
  constexpr int NUM_M_TILES = (OUTPUT_SIZE + MMA_M - 1) / MMA_M;
  constexpr int NUM_N_TILES = (BATCH_SIZE + MMA_N - 1) / MMA_N;
  constexpr int NUM_TILES = NUM_M_TILES * NUM_N_TILES;

  constexpr int A_TILE_ELEMS = MMA_M * BK;
  constexpr int B_TILE_ELEMS = MMA_N * BK;
  constexpr int C_TILE_ELEMS = MMA_N * MMA_M;

  constexpr uint32_t idesc = make_idesc_f16(MMA_M, MMA_N);

  constexpr int kClampedBN = (BATCH_SIZE < MMA_N) ? BATCH_SIZE : MMA_N;
  constexpr int TMA_BYTES = (int)sizeof(T_) * (MMA_M * BK + kClampedBN * BK);

  constexpr int tmem_used_columns = MMA_N * NUM_ACC_STAGE;
  constexpr int num_tmem_columns = (tmem_used_columns <= 128)   ? 128
                                   : (tmem_used_columns <= 256) ? 256
                                                                : 512;
  static_assert(num_tmem_columns <= 512, "TMEM oversubscribed");

  int const tid = threadIdx.x;
  if (tid >= 256) {
    return;
  }
  int const wid = tid / 32;

  int const k_begin = 0;
  int const k_end = K_TILES;
  bool const use_reduce_add = SplitK;

  using SharedStorage = LinearMpkSharedStorage<T_,
                                               MMA_M,
                                               MMA_N,
                                               BK,
                                               NUM_AB_STAGE,
                                               NUM_ACC_STAGE,
                                               NUM_C_STAGE>;
  extern __shared__ __align__(1024) char linear_mpk_smem_raw[];
  uintptr_t aligned =
      (reinterpret_cast<uintptr_t>(linear_mpk_smem_raw) + 1023) / 1024 * 1024;
  SharedStorage &ss = *reinterpret_cast<SharedStorage *>(aligned);

  int const A_smem = __cvta_generic_to_shared(&ss.A[0]);
  int const B_smem = __cvta_generic_to_shared(&ss.B[0]);
  int const ab_full = __cvta_generic_to_shared(&ss.ab_full_mbar[0]);
  int const ab_empty = __cvta_generic_to_shared(&ss.ab_empty_mbar[0]);
  int const acc_full = __cvta_generic_to_shared(&ss.acc_full_mbar[0]);
  int const acc_empty = __cvta_generic_to_shared(&ss.acc_empty_mbar[0]);

  if (wid == 0 && elect_sync()) {
    for (int i = 0; i < NUM_AB_STAGE; i++) {
      mbar_init(ab_full + i * 8, 1);
      mbar_init(ab_empty + i * 8, 1);
    }
    for (int i = 0; i < NUM_ACC_STAGE; i++) {
      mbar_init(acc_full + i * 8, 1);
      mbar_init(acc_empty + i * 8, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (wid == 1) {
    int addr = __cvta_generic_to_shared(&ss.tmem_base);
    tcgen05_alloc(addr, num_tmem_columns);
  }
  __syncthreads();
  int const taddr = 0;

  if (wid == 5) {
    if (elect_sync()) {
      int stage_seq = 0;
      for (int t = 0; t < NUM_TILES; t++) {
        int m_tile, n_tile;
        supergroup_tile_coord(
            t, NUM_M_TILES, NUM_N_TILES, LINEAR_MPK_GROUP_M, m_tile, n_tile);

        for (int k = k_begin; k < k_end; k++, stage_seq++) {
          int stage = stage_seq % NUM_AB_STAGE;
          if (stage_seq >= NUM_AB_STAGE) {
            int phase = (stage_seq / NUM_AB_STAGE - 1) % 2;
            mbar_wait(ab_empty + stage * 8, phase);
          }

          int ab_mbar = ab_full + stage * 8;
          mbar_tx(ab_mbar, TMA_BYTES);
          int a_dst = A_smem + stage * A_TILE_ELEMS * (int)sizeof(T_);
          int b_dst = B_smem + stage * B_TILE_ELEMS * (int)sizeof(T_);
          tma_load_2d_cta(
              a_dst, tma_a.desc_ptr, k * BK, m_tile * MMA_M, ab_mbar);
          tma_load_2d_cta(
              b_dst, tma_b.desc_ptr, k * BK, n_tile * MMA_N, ab_mbar);
        }
      }
    }
  } else if (wid == 4) {
    if (elect_sync()) {
      int stage_seq = 0;
      int local_iter = 0;
      for (int t = 0; t < NUM_TILES; t++, local_iter++) {
        int acc_idx = local_iter % NUM_ACC_STAGE;
        int acc_taddr = taddr + acc_idx * MMA_N;
        if (local_iter >= NUM_ACC_STAGE) {
          int acc_empty_phase = (local_iter / NUM_ACC_STAGE - 1) % 2;
          mbar_wait(acc_empty + acc_idx * 8, acc_empty_phase);
        }
        for (int k = k_begin; k < k_end; k++, stage_seq++) {
          int stage = stage_seq % NUM_AB_STAGE;
          int phase = (stage_seq / NUM_AB_STAGE) % 2;
          mbar_wait(ab_full + stage * 8, phase);
          tcgen05_fence_after();

          int a_base = A_smem + stage * A_TILE_ELEMS * (int)sizeof(T_);
          int b_base = B_smem + stage * B_TILE_ELEMS * (int)sizeof(T_);
          for (int ks = 0; ks < K_SUB; ks++) {
            uint64_t a_desc = make_desc(a_base + ks * 32);
            uint64_t b_desc = make_desc(b_base + ks * 32);
            int acc = (k == k_begin && ks == 0) ? 0 : 1;
            tcgen05_mma(acc_taddr, a_desc, b_desc, idesc, acc);
          }
          tcgen05_commit(ab_empty + stage * 8);
        }
        tcgen05_commit(acc_full + acc_idx * 8);
      }
    }
  } else if (wid < 4) {
    int local_iter = 0;
    for (int t = 0; t < NUM_TILES; t++, local_iter++) {
      int m_tile, n_tile;
      supergroup_tile_coord(
          t, NUM_M_TILES, NUM_N_TILES, LINEAR_MPK_GROUP_M, m_tile, n_tile);

      int acc_idx = local_iter % NUM_ACC_STAGE;
      int acc_full_phase = (local_iter / NUM_ACC_STAGE) % 2;
      int c_stage = local_iter % NUM_C_STAGE;
      int acc_taddr = taddr + acc_idx * MMA_N;
      mbar_wait(acc_full + acc_idx * 8, acc_full_phase);
      tcgen05_fence_after();

      float acc[MMA_N];
      int ld_addr = acc_taddr + (tid << 16);
      tcgen05_ld_cols<MMA_N>(ld_addr, acc);
      tcgen05_ld_wait();

      named_barrier_sync(1, 128);
      if (wid == 0 && elect_sync()) {
        mbar_arrive(acc_empty + acc_idx * 8);
      }

      int out_col = m_tile * MMA_M + tid;
      if constexpr (!NOBIAS) {
        if (out_col < OUTPUT_SIZE) {
#pragma unroll
          for (int j = 0; j < MMA_N; j++) {
            int batch_row = n_tile * MMA_N + j;
            if (batch_row < BATCH_SIZE) {
              uint16_t bits = *reinterpret_cast<uint16_t const *>(
                  &mBias[batch_row * output_stride + out_col]);
              acc[j] += __bfloat162float(
                  *reinterpret_cast<__nv_bfloat16 const *>(&bits));
            }
          }
        }
      }

      int c_base = __cvta_generic_to_shared(&ss.C[c_stage * C_TILE_ELEMS]);
#pragma unroll
      for (int j = 0; j < MMA_N; j++) {
        nv_bfloat16 v = __float2bfloat16(acc[j]);
        int off = (j * MMA_M + tid) * (int)sizeof(T_);
        asm volatile("st.shared.b16 [%0], %1;" ::"r"(c_base + off),
                     "h"(*reinterpret_cast<uint16_t *>(&v)));
      }

      kernel::tma::async_proxy_fence();
      named_barrier_sync(1, 128);

      if (wid == 0 && elect_sync()) {
        int coords[2] = {m_tile * MMA_M, n_tile * MMA_N};
        if (use_reduce_add) {
          tma_out.tma_reduce_add_async(&ss.C[c_stage * C_TILE_ELEMS], coords);
        } else {
          tma_out.tma_store_async(&ss.C[c_stage * C_TILE_ELEMS], coords);
        }
        kernel::tma::store_commit_group();
        kernel::tma::store_async_wait<NUM_C_STAGE - 1>();
      }
    }
    if (wid == 0 && elect_sync()) {
      kernel::tma::store_async_wait<0>();
    }
  }

  __syncthreads();
  if (wid == 1) {
    tcgen05_dealloc(0, num_tmem_columns);
  }
}

} // namespace kernel
