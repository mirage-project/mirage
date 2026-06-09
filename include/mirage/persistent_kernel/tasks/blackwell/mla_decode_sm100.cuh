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

// MLA Decode for DeepSeek V3 on B200 (SM100a)
// Warp-specialized TMA/MMA pipeline for QK and PV GEMMs
// Split-KV: each task handles one tile, separate reduce task merges partials
#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace kernel {

static constexpr int MLA_NUM_HEADS = 128;
static constexpr int MLA_D_K = 576;
static constexpr int MLA_D_V = 512;
static constexpr int MLA_TILE_S = 128;
static constexpr int MLA_BK = 64;
static constexpr int MLA_MMA_K = 16;
static constexpr int MLA_K_ITERS = MLA_D_K / MLA_BK;  // 9
static constexpr int MLA_V_CHUNKS = MLA_D_V / MLA_BK; // 8
static constexpr int MLA_TB = 128;

static constexpr int MLA_NUM_QK_STAGES = 4;
static constexpr int MLA_NUM_PV_STAGES = 2;
static constexpr int MLA_MAX_STAGES = 4;

static constexpr int MLA_TILE_BYTES = MLA_NUM_HEADS * MLA_BK * 2;

namespace mla_ptx {

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t p = 0;
  asm volatile("{\n\t.reg .pred %%px;\n\t"
               "elect.sync _|%%px, 0xFFFFFFFF;\n\t"
               "@%%px mov.s32 %0, 1;\n\t}"
               : "+r"(p));
  return p;
}

__device__ __forceinline__ void mbar_init(int addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(count));
}

__device__ __forceinline__ void mbar_wait(int addr, int phase) {
  asm volatile("{\n\t.reg .pred P;\n\t"
               "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, "
               "[%0], %1, 0x989680;\n\t"
               "@P bra DONE;\n\t"
               "bra WAIT;\n\t"
               "DONE:\n\t}" ::"r"(addr),
               "r"(phase));
}

__device__ __forceinline__ void mbar_tx(int addr, int bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(addr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4;
}

__device__ __forceinline__ uint64_t make_desc(int smem_addr) {
  constexpr uint64_t SBO = 8ULL * 128;
  return desc_enc(smem_addr) | (desc_enc(SBO) << 32) | (1ULL << 46) |
         (2ULL << 61);
}

__device__ __forceinline__ void tcgen05_mma(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, int acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(
          taddr),
      "l"(a_desc),
      "l"(b_desc),
      "r"(idesc),
      "r"(acc));
}

__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(mbar_addr)
               : "memory");
}

} // namespace mla_ptx

// MLA decode device function.
// split_idx (si) and batch_idx (bi) are passed as params.
// TMA descriptors passed by pointer.
__device__ __noinline__ void mla_decode_sm100_task_impl(
    CUtensorMap const *Q_tm_ptr,
    CUtensorMap const *KV_tm_ptr,
    float *__restrict__ Oa,
    float *__restrict__ La,
    float ss,
    int kv_len,
    int sk,
    int si,
    int bi // split_idx, batch_idx (were blockIdx.x, blockIdx.y)
) {
  using namespace mla_ptx;

  int const tid = threadIdx.x;
  if (tid >= MLA_TB) {
    return; // guard for MPK's 256-thread workers
  }
  int const wid = tid / 32;

  // INVARIANT: each split handles exactly 1 tile (tps must be 1).
  // Multi-tile accumulation per split has broken online softmax correction.
  // Caller must set sk = ceil(kv_len / MLA_TILE_S).
  int const kvt = (kv_len + MLA_TILE_S - 1) / MLA_TILE_S;
  int const tps = (kvt + sk - 1) / sk; // must be 1
  int const t0 = si * tps;
  int const t1 = min(t0 + tps, kvt);
  if (t0 >= t1) {
    return;
  }

  extern __shared__ __align__(1024) char smem_buf[];
  int const smem_base = __cvta_generic_to_shared(smem_buf);

  int const Q_smem = smem_base;
  int const work_smem = smem_base + MLA_K_ITERS * MLA_TILE_BYTES;

  __shared__ uint64_t mbar_buf[10];
  __shared__ int tmem_addr_buf[1];
  int const tma_bar = __cvta_generic_to_shared(&mbar_buf[0]);
  int const mma_bar = __cvta_generic_to_shared(&mbar_buf[MLA_MAX_STAGES]);
  int const mainloop_bar =
      __cvta_generic_to_shared(&mbar_buf[2 * MLA_MAX_STAGES]);
  int const q_bar = __cvta_generic_to_shared(&mbar_buf[2 * MLA_MAX_STAGES + 1]);

  if (wid == 0 && elect_sync()) {
    for (int i = 0; i < MLA_MAX_STAGES; i++) {
      mbar_init(tma_bar + i * 8, 1);
      mbar_init(mma_bar + i * 8, 1);
    }
    mbar_init(mainloop_bar, 1);
    mbar_init(q_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (wid == 1) {
    int addr_smem = __cvta_generic_to_shared(tmem_addr_buf);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(addr_smem),
        "r"(MLA_D_V));
  }
  __syncthreads();
  int const taddr = tmem_addr_buf[0];

  // Load Q via TMA
  if (wid == 0 && elect_sync()) {
    mbar_tx(q_bar, MLA_TILE_BYTES * MLA_K_ITERS);
    for (int ki = 0; ki < MLA_K_ITERS; ki++) {
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::"
          "bytes "
          "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(Q_smem + ki * MLA_TILE_BYTES),
          "l"(Q_tm_ptr),
          "r"(0),
          "r"(bi * MLA_NUM_HEADS),
          "r"(ki),
          "r"(q_bar)
          : "memory");
    }
  }
  mbar_wait(q_bar, 0);
  __syncthreads();

  constexpr uint32_t idesc_qk = (1U << 4) | (1U << 7) | (1U << 10) |
                                ((uint32_t)(MLA_TILE_S >> 3) << 17) |
                                ((uint32_t)(MLA_NUM_HEADS >> 4) << 24);
  constexpr uint32_t idesc_pv = (1U << 4) | (1U << 7) | (1U << 10) |
                                (1U << 16) | ((uint32_t)(MLA_BK >> 3) << 17) |
                                ((uint32_t)(MLA_NUM_HEADS >> 4) << 24);

  float *Oout = Oa + (bi * sk + si) * MLA_D_V * MLA_NUM_HEADS;
  float row_max = -1e30f;
  float row_sum = 0.0f;

  for (int tile = t0; tile < t1; tile++) {
    int const kvs = tile * MLA_TILE_S;
    int const tlen = min(MLA_TILE_S, kv_len - kvs);

    // QK Phase
    __syncthreads();
    if (wid == 0 && elect_sync()) {
      for (int i = 0; i < MLA_NUM_QK_STAGES; i++) {
        mbar_init(tma_bar + i * 8, 1);
        mbar_init(mma_bar + i * 8, 1);
      }
      mbar_init(mainloop_bar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    if (wid == 0 && elect_sync()) {
      int phase = 0;
      for (int ki = 0; ki < MLA_K_ITERS; ki++) {
        int stage = ki % MLA_NUM_QK_STAGES;
        mbar_wait(mma_bar + stage * 8, phase ^ 1);
        if (stage == MLA_NUM_QK_STAGES - 1) {
          phase ^= 1;
        }
        int k_smem = work_smem + stage * MLA_TILE_BYTES;
        mbar_tx(tma_bar + stage * 8, MLA_TILE_BYTES);
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::"
                     "complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(k_smem),
                     "l"(KV_tm_ptr),
                     "r"(0),
                     "r"(kvs),
                     "r"(ki),
                     "r"(tma_bar + stage * 8)
                     : "memory");
      }
    } else if (wid == 1 && elect_sync()) {
      int phase = 0;
      for (int ki = 0; ki < MLA_K_ITERS; ki++) {
        int stage = ki % MLA_NUM_QK_STAGES;
        mbar_wait(tma_bar + stage * 8, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (stage == MLA_NUM_QK_STAGES - 1) {
          phase ^= 1;
        }
        int k_smem = work_smem + stage * MLA_TILE_BYTES;
        for (int k2 = 0; k2 < MLA_BK / MLA_MMA_K; k2++) {
          uint64_t a_desc = make_desc(Q_smem + ki * MLA_TILE_BYTES + k2 * 32);
          uint64_t b_desc = make_desc(k_smem + k2 * 32);
          tcgen05_mma(
              taddr, a_desc, b_desc, idesc_qk, (ki == 0 && k2 == 0) ? 0 : 1);
        }
        tcgen05_commit(mma_bar + stage * 8);
      }
      tcgen05_commit(mainloop_bar);
    }

    __syncthreads();
    mbar_wait(mainloop_bar, 0);

    // Softmax Phase
    asm volatile("tcgen05.fence::after_thread_sync;");

    if (wid == 0 && elect_sync()) {
      for (int i = 0; i < MLA_NUM_PV_STAGES; i++) {
        mbar_init(tma_bar + i * 8, 1);
        mbar_init(mma_bar + i * 8, 1);
      }
      mbar_init(mainloop_bar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;");
      int v_smem0 = work_smem + 2 * MLA_TILE_BYTES;
      mbar_tx(tma_bar, MLA_TILE_BYTES);
      asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::"
                   "complete_tx::bytes "
                   "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(v_smem0),
                   "l"(KV_tm_ptr),
                   "r"(0),
                   "r"(kvs),
                   "r"(0),
                   "r"(tma_bar)
                   : "memory");
    }

    float sl[MLA_TILE_S];
    for (int c = 0; c < MLA_TILE_S; c += 16) {
      float t16[16];
      int addr = taddr + (tid << 16) + c;
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
          : "=f"(t16[0]),
            "=f"(t16[1]),
            "=f"(t16[2]),
            "=f"(t16[3]),
            "=f"(t16[4]),
            "=f"(t16[5]),
            "=f"(t16[6]),
            "=f"(t16[7]),
            "=f"(t16[8]),
            "=f"(t16[9]),
            "=f"(t16[10]),
            "=f"(t16[11]),
            "=f"(t16[12]),
            "=f"(t16[13]),
            "=f"(t16[14]),
            "=f"(t16[15])
          : "r"(addr));
      asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
      for (int i = 0; i < 16; i++) {
        sl[c + i] = t16[i];
      }
    }

    float tm = -1e30f;
#pragma unroll
    for (int t = 0; t < MLA_TILE_S; t++) {
      float v = sl[t] * ss;
      sl[t] = v;
      tm = fmaxf(tm, v);
    }
    if (tlen < MLA_TILE_S) {
      for (int t = tlen; t < MLA_TILE_S; t++) {
        sl[t] = -1e30f;
      }
      tm = -1e30f;
      for (int t = 0; t < tlen; t++) {
        tm = fmaxf(tm, sl[t]);
      }
    }
    float nm = fmaxf(row_max, tm);
    float corr = __expf(row_max - nm);
    float ts = 0;
#pragma unroll
    for (int t = 0; t < MLA_TILE_S; t++) {
      float e = __expf(sl[t] - nm);
      sl[t] = e;
      ts += e;
    }
    if (tlen < MLA_TILE_S) {
      ts = 0;
      for (int t = 0; t < tlen; t++) {
        ts += sl[t];
      }
      for (int t = tlen; t < MLA_TILE_S; t++) {
        sl[t] = 0;
      }
    }

    // Write P to SMEM
    int P0_smem = work_smem;
    int P1_smem = work_smem + MLA_TILE_BYTES;
#pragma unroll
    for (int p_tile = 0; p_tile < 2; p_tile++) {
      int p_base = (p_tile == 0) ? P0_smem : P1_smem;
      int row_base = p_base + tid * 128;
#pragma unroll
      for (int g = 0; g < 64; g += 8) {
        int t_off = p_tile * 64 + g;
        uint32_t w0, w1, w2, w3;
        {
          nv_bfloat16 b0 = __float2bfloat16(sl[t_off + 0]);
          nv_bfloat16 b1 = __float2bfloat16(sl[t_off + 1]);
          w0 = (uint32_t)(*(uint16_t *)&b0) |
               ((uint32_t)(*(uint16_t *)&b1) << 16);
        }
        {
          nv_bfloat16 b0 = __float2bfloat16(sl[t_off + 2]);
          nv_bfloat16 b1 = __float2bfloat16(sl[t_off + 3]);
          w1 = (uint32_t)(*(uint16_t *)&b0) |
               ((uint32_t)(*(uint16_t *)&b1) << 16);
        }
        {
          nv_bfloat16 b0 = __float2bfloat16(sl[t_off + 4]);
          nv_bfloat16 b1 = __float2bfloat16(sl[t_off + 5]);
          w2 = (uint32_t)(*(uint16_t *)&b0) |
               ((uint32_t)(*(uint16_t *)&b1) << 16);
        }
        {
          nv_bfloat16 b0 = __float2bfloat16(sl[t_off + 6]);
          nv_bfloat16 b1 = __float2bfloat16(sl[t_off + 7]);
          w3 = (uint32_t)(*(uint16_t *)&b0) |
               ((uint32_t)(*(uint16_t *)&b1) << 16);
        }
        int byte_off = g * 2;
        int swizzled = (byte_off & ~0xF) ^ ((tid & 7) << 4) | (byte_off & 0xF);
        int addr = row_base + swizzled;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr),
                     "r"(w0),
                     "r"(w1),
                     "r"(w2),
                     "r"(w3));
      }
    }
    __syncthreads();

    // PV Phase
    int V_buf_base = work_smem + 2 * MLA_TILE_BYTES;

    if (wid == 0 && elect_sync()) {
      int phase = 0;
      for (int vc = 1; vc < MLA_V_CHUNKS; vc++) {
        int stage = vc % MLA_NUM_PV_STAGES;
        mbar_wait(mma_bar + stage * 8, phase ^ 1);
        if (stage == MLA_NUM_PV_STAGES - 1) {
          phase ^= 1;
        }
        int v_smem = V_buf_base + stage * MLA_TILE_BYTES;
        mbar_tx(tma_bar + stage * 8, MLA_TILE_BYTES);
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::"
                     "complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(v_smem),
                     "l"(KV_tm_ptr),
                     "r"(0),
                     "r"(kvs),
                     "r"(vc),
                     "r"(tma_bar + stage * 8)
                     : "memory");
      }
    } else if (wid == 1 && elect_sync()) {
      int phase = 0;
      for (int vc = 0; vc < MLA_V_CHUNKS; vc++) {
        int stage = vc % MLA_NUM_PV_STAGES;
        mbar_wait(tma_bar + stage * 8, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (stage == MLA_NUM_PV_STAGES - 1) {
          phase ^= 1;
        }
        int v_smem = V_buf_base + stage * MLA_TILE_BYTES;
        int out_taddr = taddr + vc * MLA_BK;
        int first = 1;
        for (int k1 = 0; k1 < 2; k1++) {
          int p_addr = (k1 == 0) ? P0_smem : P1_smem;
          int v_k1_off = k1 * 64 * 128;
          for (int k2 = 0; k2 < MLA_BK / MLA_MMA_K; k2++) {
            uint64_t a_desc = make_desc(p_addr + k2 * 32);
            uint64_t b_desc = make_desc(v_smem + v_k1_off + k2 * 16 * 128);
            tcgen05_mma(out_taddr, a_desc, b_desc, idesc_pv, first ? 0 : 1);
            first = 0;
          }
        }
        tcgen05_commit(mma_bar + stage * 8);
      }
      tcgen05_commit(mainloop_bar);
    }

    __syncthreads();
    mbar_wait(mainloop_bar, 0);

    // Accumulate
    asm volatile("tcgen05.fence::after_thread_sync;");

    row_max = nm;
    row_sum = corr * row_sum + ts;
    float inv = (row_sum > 0) ? 1.0f / row_sum : 0.0f;
    float corr_inv = corr * inv;
    int is_first = (tile == t0);

    for (int vc = 0; vc < MLA_V_CHUNKS; vc++) {
      int out_taddr = taddr + vc * MLA_BK;
      for (int c = 0; c < MLA_BK; c += 16) {
        float t16[16];
        int addr = out_taddr + (tid << 16) + c;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(t16[0]),
              "=f"(t16[1]),
              "=f"(t16[2]),
              "=f"(t16[3]),
              "=f"(t16[4]),
              "=f"(t16[5]),
              "=f"(t16[6]),
              "=f"(t16[7]),
              "=f"(t16[8]),
              "=f"(t16[9]),
              "=f"(t16[10]),
              "=f"(t16[11]),
              "=f"(t16[12]),
              "=f"(t16[13]),
              "=f"(t16[14]),
              "=f"(t16[15])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        int base_d = (vc * MLA_BK + c) * MLA_NUM_HEADS + tid;
        if (is_first) {
#pragma unroll
          for (int i = 0; i < 16; i++) {
            Oout[base_d + i * MLA_NUM_HEADS] = t16[i] * inv;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; i++) {
            int gaddr = base_d + i * MLA_NUM_HEADS;
            Oout[gaddr] = corr_inv * Oout[gaddr] + t16[i] * inv;
          }
        }
      }
    }
  }

  La[(bi * sk + si) * MLA_NUM_HEADS + tid] =
      logf(fmaxf(row_sum, 1e-30f)) + row_max;

  __syncthreads();
  if (wid == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
        "r"(MLA_D_V));
  }
}

} // namespace kernel
