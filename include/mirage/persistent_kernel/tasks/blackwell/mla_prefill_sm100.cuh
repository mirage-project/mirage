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

// MLA Prefill Attention for DeepSeek V3 on B200
// Ampere-style m16n8k16 MMA with cp.async, online softmax, causal masking
// Each task handles one (head, q_block) pair
// 256 threads = 8 warps, ~216KB SMEM
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernel {

namespace mla_prefill {

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

static constexpr int PF_D_CKV = 512;
static constexpr int PF_D_KPE = 64;
static constexpr int PF_D_QK = 576;
static constexpr int PF_D_V = 512;
static constexpr int PF_BM = 64;
static constexpr int PF_BN = 64;
static constexpr int PF_NUM_THREADS = 256;
static constexpr int PF_NUM_WARPS = 8;
static constexpr int PF_WARP_SIZE = 32;
static constexpr int PF_NUM_STAGES = 2;

static constexpr int PF_MMA_M = 16;
static constexpr int PF_MMA_N = 8;
static constexpr int PF_MMA_K = 16;

static constexpr int PF_NUM_MMA_KV = PF_BN / PF_MMA_K;
static constexpr int PF_NUM_MMA_N16 = PF_BN / 16;
static constexpr int PF_NUM_MMA_D_CKV_K = PF_D_CKV / PF_MMA_K;
static constexpr int PF_NUM_MMA_D_KPE_K = PF_D_KPE / PF_MMA_K;
static constexpr int PF_NUM_MMA_D_V_HALF = (PF_D_V / 2) / 16;

static constexpr int PF_Q_NOPE_SIZE = PF_BM * PF_D_CKV * sizeof(bf16);
static constexpr int PF_Q_PE_SIZE = PF_BM * PF_D_KPE * sizeof(bf16);
static constexpr int PF_CKV_STAGE_SIZE = PF_BN * PF_D_CKV * sizeof(bf16);
static constexpr int PF_KPE_STAGE_SIZE = PF_BN * PF_D_KPE * sizeof(bf16);
static constexpr int PF_SMEM_SIZE =
    PF_Q_NOPE_SIZE + PF_Q_PE_SIZE +
    PF_NUM_STAGES * (PF_CKV_STAGE_SIZE + PF_KPE_STAGE_SIZE) +
    2 * PF_BM * sizeof(float);

// ============ PTX Intrinsics ============

template <int STRIDE>
__device__ __forceinline__ int swizzle(int row, int col) {
  if constexpr (STRIDE >= 128) {
    col ^= (row % 8) / (128 / STRIDE > 1 ? 128 / STRIDE : 1);
  }
  return row * STRIDE + col * 16;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t reg[4], int addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t reg[4], int addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
      : "r"(addr));
}

__device__ __forceinline__ void
    mma_m16n8k16_bf16(const uint32_t A[4], const uint32_t B[2], float C[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

__device__ __forceinline__ void mma_m16n8k16_bf16_init(const uint32_t A[4],
                                                       const uint32_t B[2],
                                                       float C[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(A[2]),
        "r"(A[3]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(0.0f),
        "f"(0.0f),
        "f"(0.0f),
        "f"(0.0f));
}

__device__ __forceinline__ void
    mma_m16n16k16_bf16(const uint32_t A[4], const uint32_t B[4], float C[8]) {
  mma_m16n8k16_bf16(A, &B[0], &C[0]);
  mma_m16n8k16_bf16(A, &B[2], &C[4]);
}

__device__ __forceinline__ void mma_m16n16k16_bf16_init(const uint32_t A[4],
                                                        const uint32_t B[4],
                                                        float C[8]) {
  mma_m16n8k16_bf16_init(A, &B[0], &C[0]);
  mma_m16n8k16_bf16_init(A, &B[2], &C[4]);
}

__device__ __forceinline__ void mma_rowsum_bf16(float *d, uint32_t *s_u32) {
  asm volatile("{\n"
               "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, _, %1, _}, {%2, %3, %4, %5}, {%6, %7}, {%0, 0., %1, 0.};\n"
               "}\n"
               : "+f"(d[0]), "+f"(d[1])
               : "r"(s_u32[0]),
                 "r"(s_u32[1]),
                 "r"(s_u32[2]),
                 "r"(s_u32[3]),
                 "r"(1065369472u),
                 "r"(1065369472u));
}

__device__ __forceinline__ void cp_async_128b(int dst_smem_addr,
                                              void const *src_gmem) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem_addr),
      "l"(src_gmem));
}

__device__ __forceinline__ void
    cp_async_128b_pred(int dst_smem_addr, void const *src_gmem, bool pred) {
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %2, 0;\n"
               "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
               "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
               "}\n" ::"r"(dst_smem_addr),
               "l"(src_gmem),
               "r"((int)pred));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ float shfl_xor(float val, int mask) {
  return __shfl_xor_sync(0xffffffff, val, mask);
}

__device__ __forceinline__ float fast_exp2f(float x) {
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ uint32_t float2_to_bf16x2(float a, float b) {
  bf16_2 val = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<uint32_t *>(&val);
}

__host__ __device__ __forceinline__ int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

} // namespace mla_prefill

// ============ MLA Prefill Device Function ============
// Adapted from mla_prefill_host.cu.
// Changes: __device__, blockIdx->{head, q_block} params.
// No TMA — uses cp.async, same as original.
// 256 threads = matches MPK Blackwell worker thread count exactly.
__device__ __noinline__ void mla_prefill_sm100_task_impl(
    __nv_bfloat16 const *__restrict__ Q_nope, // [S, H, D_CKV]
    __nv_bfloat16 const *__restrict__ Q_pe,   // [S, H, D_KPE]
    __nv_bfloat16 const *__restrict__ CKV,    // [S, D_CKV]
    __nv_bfloat16 const *__restrict__ KPE,    // [S, D_KPE]
    __nv_bfloat16 *__restrict__ O,            // [S, H, D_V]
    int const S,
    int const H,
    float const sm_scale_log2,
    int const head,
    int const q_block // were blockIdx.x, blockIdx.y
) {
  using namespace mla_prefill;

  int const q_start = q_block * PF_BM;

  int const tid = threadIdx.x;
  int const warp_id = tid / PF_WARP_SIZE;
  int const lane_id = tid % PF_WARP_SIZE;
  int const warp_m = warp_id / 2;
  int const warp_d = warp_id % 2;

  extern __shared__ __align__(128) uint8_t smem_raw[];

  int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
  int q_nope_smem = smem_base;
  int q_pe_smem = q_nope_smem + PF_Q_NOPE_SIZE;
  int ckv_smem_base = q_pe_smem + PF_Q_PE_SIZE;
  int kpe_smem_base = ckv_smem_base + PF_NUM_STAGES * PF_CKV_STAGE_SIZE;

  auto ckv_smem = [&](int stage) {
    return ckv_smem_base + stage * PF_CKV_STAGE_SIZE;
  };
  auto kpe_smem = [&](int stage) {
    return kpe_smem_base + stage * PF_KPE_STAGE_SIZE;
  };

  // Step 1: Load Q into shared memory
  {
    constexpr int Q_NOPE_LOADS = (PF_BM * PF_D_CKV) / 8;
    constexpr int Q_PE_LOADS = (PF_BM * PF_D_KPE) / 8;
    constexpr int STRIDE_NOPE = PF_D_CKV * sizeof(bf16);
    constexpr int STRIDE_PE = PF_D_KPE * sizeof(bf16);

    for (int i = tid; i < Q_NOPE_LOADS; i += PF_NUM_THREADS) {
      int row = i / (PF_D_CKV / 8);
      int col = i % (PF_D_CKV / 8);
      int q_idx = q_start + row;
      int smem_addr = q_nope_smem + swizzle<STRIDE_NOPE>(row, col);
      bf16 const *gmem_ptr = Q_nope + (long long)q_idx * H * PF_D_CKV +
                             (long long)head * PF_D_CKV + col * 8;
      if (q_idx < S) {
        cp_async_128b(smem_addr, gmem_ptr);
      } else {
        asm volatile("st.shared.v4.u32 [%0], {0, 0, 0, 0};\n" ::"r"(smem_addr));
      }
    }

    for (int i = tid; i < Q_PE_LOADS; i += PF_NUM_THREADS) {
      int row = i / (PF_D_KPE / 8);
      int col = i % (PF_D_KPE / 8);
      int q_idx = q_start + row;
      int smem_addr = q_pe_smem + swizzle<STRIDE_PE>(row, col);
      bf16 const *gmem_ptr = Q_pe + (long long)q_idx * H * PF_D_KPE +
                             (long long)head * PF_D_KPE + col * 8;
      if (q_idx < S) {
        cp_async_128b(smem_addr, gmem_ptr);
      } else {
        asm volatile("st.shared.v4.u32 [%0], {0, 0, 0, 0};\n" ::"r"(smem_addr));
      }
    }

    cp_async_commit();
    mla_prefill::cp_async_wait<0>();
    __syncthreads();
  }

  // Pre-compute ldmatrix base addresses for Q
  constexpr int STRIDE_NOPE_B = PF_D_CKV * sizeof(bf16);
  constexpr int STRIDE_PE_B = PF_D_KPE * sizeof(bf16);
  int q_nope_ldm_base =
      q_nope_smem +
      swizzle<STRIDE_NOPE_B>(warp_m * 16 + (lane_id % 16), lane_id / 16);
  int q_pe_ldm_base =
      q_pe_smem +
      swizzle<STRIDE_PE_B>(warp_m * 16 + (lane_id % 16), lane_id / 16);

  // Initialize accumulators
  float o_frag[PF_NUM_MMA_D_V_HALF][8];
#pragma unroll
  for (int i = 0; i < PF_NUM_MMA_D_V_HALF; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) {
      o_frag[i][j] = 0.0f;
    }

  float m_state[2] = {-INFINITY, -INFINITY};
  float d_state[2] = {1.0f, 1.0f};

  constexpr int NUM_N_SHARD_GLOBAL = PF_NUM_MMA_N16 / 2;
  float s_frag[NUM_N_SHARD_GLOBAL][8];

  // KV tile range with causal masking
  int kv_end = min(S, q_start + PF_BM);
  int num_kv_tiles = cdiv(kv_end, PF_BN);
  int num_safe_tiles = q_start / PF_BN;

  // Prefetch first KV tile
  auto load_kv_tile = [&](int kv_tile, int stage) {
    int kv_base = kv_tile * PF_BN;
    constexpr int CKV_LOADS = (PF_BN * PF_D_CKV) / 8;
    constexpr int KPE_LOADS = (PF_BN * PF_D_KPE) / 8;
    constexpr int STRIDE_CKV = PF_D_CKV * sizeof(bf16);
    constexpr int STRIDE_KPE_B = PF_D_KPE * sizeof(bf16);

    if (kv_base + PF_BN <= S) {
      for (int i = tid; i < CKV_LOADS; i += PF_NUM_THREADS) {
        int row = i / (PF_D_CKV / 8);
        int col = i % (PF_D_CKV / 8);
        int kv_idx = kv_base + row;
        int addr = ckv_smem(stage) + swizzle<STRIDE_CKV>(row, col);
        const bf16 *ptr = CKV + (long long)kv_idx * PF_D_CKV + col * 8;
        cp_async_128b(addr, ptr);
      }
      for (int i = tid; i < KPE_LOADS; i += PF_NUM_THREADS) {
        int row = i / (PF_D_KPE / 8);
        int col = i % (PF_D_KPE / 8);
        int kv_idx = kv_base + row;
        int addr = kpe_smem(stage) + swizzle<STRIDE_KPE_B>(row, col);
        const bf16 *ptr = KPE + (long long)kv_idx * PF_D_KPE + col * 8;
        cp_async_128b(addr, ptr);
      }
    } else {
      for (int i = tid; i < CKV_LOADS; i += PF_NUM_THREADS) {
        int row = i / (PF_D_CKV / 8);
        int col = i % (PF_D_CKV / 8);
        int kv_idx = kv_base + row;
        int addr = ckv_smem(stage) + swizzle<STRIDE_CKV>(row, col);
        const bf16 *ptr = CKV + (long long)kv_idx * PF_D_CKV + col * 8;
        cp_async_128b_pred(addr, ptr, kv_idx < S);
      }
      for (int i = tid; i < KPE_LOADS; i += PF_NUM_THREADS) {
        int row = i / (PF_D_KPE / 8);
        int col = i % (PF_D_KPE / 8);
        int kv_idx = kv_base + row;
        int addr = kpe_smem(stage) + swizzle<STRIDE_KPE_B>(row, col);
        const bf16 *ptr = KPE + (long long)kv_idx * PF_D_KPE + col * 8;
        cp_async_128b_pred(addr, ptr, kv_idx < S);
      }
    }
    cp_async_commit();
  };

  if (num_kv_tiles > 0) {
    load_kv_tile(0, 0);
  }

  // Main loop over KV tiles
  constexpr int STRIDE_CKV_B = PF_D_CKV * sizeof(bf16);
  constexpr int STRIDE_KPE_B2 = PF_D_KPE * sizeof(bf16);
  int const k_row = (lane_id % 8) + (lane_id / 16) * 8;
  int const k_col_base = (lane_id % 16) / 8;
  constexpr int NUM_N_SHARD = PF_NUM_MMA_N16 / 2;
  int const n_offset = warp_d * NUM_N_SHARD;

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    int stage = kv_tile % PF_NUM_STAGES;
    int kv_base = kv_tile * PF_BN;

    mla_prefill::cp_async_wait<0>();
    __syncthreads();

    {
      int next_tile = kv_tile + 1;
      if (next_tile < num_kv_tiles) {
        load_kv_tile(next_tile, next_tile % PF_NUM_STAGES);
      }
    }

    // Fused QK
    {
      int kpe_ldm_base_addr =
          kpe_smem(stage) + swizzle<STRIDE_KPE_B2>(k_row, k_col_base);
      int ckv_ldm_base_addr =
          ckv_smem(stage) + swizzle<STRIDE_CKV_B>(k_row, k_col_base);

#pragma unroll
      for (int mma_k = 0; mma_k < PF_NUM_MMA_D_KPE_K; mma_k++) {
        {
          uint32_t q_reg[4];
          int q_addr = q_pe_ldm_base ^ (mma_k * 32);
          ldmatrix_x4(q_reg, q_addr);
#pragma unroll
          for (int nl = 0; nl < NUM_N_SHARD; nl++) {
            uint32_t k_reg[4];
            int mma_n = n_offset + nl;
            int k_addr = kpe_ldm_base_addr + mma_n * 16 * STRIDE_KPE_B2;
            ldmatrix_x4(k_reg, k_addr ^ (mma_k * 32));
            if (mma_k == 0) {
              mma_m16n16k16_bf16_init(q_reg, k_reg, s_frag[nl]);
            } else {
              mma_m16n16k16_bf16(q_reg, k_reg, s_frag[nl]);
            }
          }
        }
        {
          uint32_t q_reg[4];
          int q_addr = q_nope_ldm_base ^ (mma_k * 32);
          ldmatrix_x4(q_reg, q_addr);
#pragma unroll
          for (int nl = 0; nl < NUM_N_SHARD; nl++) {
            uint32_t k_reg[4];
            int mma_n = n_offset + nl;
            int k_addr = ckv_ldm_base_addr + mma_n * 16 * STRIDE_CKV_B;
            ldmatrix_x4(k_reg, k_addr ^ (mma_k * 32));
            mma_m16n16k16_bf16(q_reg, k_reg, s_frag[nl]);
          }
        }
      }
#pragma unroll
      for (int mma_k = PF_NUM_MMA_D_KPE_K; mma_k < PF_NUM_MMA_D_CKV_K;
           mma_k++) {
        uint32_t q_reg[4];
        int q_addr = q_nope_ldm_base ^ (mma_k * 32);
        ldmatrix_x4(q_reg, q_addr);
#pragma unroll
        for (int nl = 0; nl < NUM_N_SHARD; nl++) {
          uint32_t k_reg[4];
          int mma_n = n_offset + nl;
          int k_addr = ckv_ldm_base_addr + mma_n * 16 * STRIDE_CKV_B;
          ldmatrix_x4(k_reg, k_addr ^ (mma_k * 32));
          mma_m16n16k16_bf16(q_reg, k_reg, s_frag[nl]);
        }
      }
    }

    // Causal Masking
    if (kv_base + PF_BN > q_start) {
      int q_row_base = q_start + warp_m * 16;
#pragma unroll
      for (int nl = 0; nl < NUM_N_SHARD; nl++) {
        int mma_n_global = n_offset + nl;
#pragma unroll
        for (int reg_id = 0; reg_id < 8; reg_id++) {
          int row_in_tile =
              ((reg_id & 2) == 0) ? (lane_id / 4) : (lane_id / 4 + 8);
          int kv_col =
              2 * (lane_id % 4) + ((reg_id & 4) ? 8 : 0) + (reg_id & 1);
          int q_pos = q_row_base + row_in_tile;
          int kv_pos = kv_base + mma_n_global * 16 + kv_col;
          if (!((kv_pos <= q_pos) && (kv_pos < S))) {
            s_frag[nl][reg_id] = -INFINITY;
          }
        }
      }
    }

    // Online Softmax
    constexpr int M_WG_OFF =
        PF_Q_NOPE_SIZE + PF_Q_PE_SIZE +
        PF_NUM_STAGES * (PF_CKV_STAGE_SIZE + PF_KPE_STAGE_SIZE);
    float *m_wg = reinterpret_cast<float *>(smem_raw + M_WG_OFF);
    {
      float m_prev[2] = {m_state[0], m_state[1]};
#pragma unroll
      for (int j = 0; j < 2; j++) {
#pragma unroll
        for (int nl = 0; nl < NUM_N_SHARD; nl++) {
          float local_max =
              fmaxf(fmaxf(s_frag[nl][j * 2 + 0], s_frag[nl][j * 2 + 1]),
                    fmaxf(s_frag[nl][j * 2 + 4], s_frag[nl][j * 2 + 5]));
          m_state[j] = fmaxf(m_state[j], local_max);
        }
        m_state[j] = fmaxf(m_state[j], shfl_xor(m_state[j], 0x2));
        m_state[j] = fmaxf(m_state[j], shfl_xor(m_state[j], 0x1));
        if (lane_id % 4 == 0) {
          m_wg[warp_d * PF_BM + warp_m * 16 + j * 8 + lane_id / 4] = m_state[j];
        }
      }
      asm volatile("bar.sync %0, 64;" ::"r"(1 + warp_m));
#pragma unroll
      for (int j = 0; j < 2; j++) {
        m_state[j] = fmaxf(m_wg[0 * PF_BM + warp_m * 16 + j * 8 + lane_id / 4],
                           m_wg[1 * PF_BM + warp_m * 16 + j * 8 + lane_id / 4]);
        float neg_m_scaled = -(m_state[j] * sm_scale_log2);
        float scale =
            fast_exp2f(__fmaf_rn(m_prev[j], sm_scale_log2, neg_m_scaled));
        d_state[j] *= scale;
#pragma unroll
        for (int mma_d = 0; mma_d < PF_NUM_MMA_D_V_HALF; mma_d++) {
          o_frag[mma_d][j * 2 + 0] *= scale;
          o_frag[mma_d][j * 2 + 1] *= scale;
          o_frag[mma_d][j * 2 + 4] *= scale;
          o_frag[mma_d][j * 2 + 5] *= scale;
        }
#pragma unroll
        for (int nl = 0; nl < NUM_N_SHARD; nl++) {
          s_frag[nl][j * 2 + 0] = fast_exp2f(
              __fmaf_rn(s_frag[nl][j * 2 + 0], sm_scale_log2, neg_m_scaled));
          s_frag[nl][j * 2 + 1] = fast_exp2f(
              __fmaf_rn(s_frag[nl][j * 2 + 1], sm_scale_log2, neg_m_scaled));
          s_frag[nl][j * 2 + 4] = fast_exp2f(
              __fmaf_rn(s_frag[nl][j * 2 + 4], sm_scale_log2, neg_m_scaled));
          s_frag[nl][j * 2 + 5] = fast_exp2f(
              __fmaf_rn(s_frag[nl][j * 2 + 5], sm_scale_log2, neg_m_scaled));
        }
      }
    }

    // Convert S to P (bf16), compute row sum, store P to SMEM
    uint32_t p_f16_local[NUM_N_SHARD][4];
#pragma unroll
    for (int nl = 0; nl < NUM_N_SHARD; nl++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        p_f16_local[nl][i] =
            float2_to_bf16x2(s_frag[nl][i * 2], s_frag[nl][i * 2 + 1]);
      }
      mma_rowsum_bf16(d_state, p_f16_local[nl]);
    }

    constexpr int P_STRIDE = PF_BN * sizeof(bf16);
    int p_smem = kpe_smem(stage);
#pragma unroll
    for (int nl = 0; nl < NUM_N_SHARD; nl++) {
      int mma_n_global = n_offset + nl;
      int p_row = warp_m * 16 + lane_id / 4;
      int p_col = mma_n_global * 16 + 2 * (lane_id % 4);
      int p_col2 = mma_n_global * 16 + 2 * (lane_id % 4) + 8;
      int col_unit = p_col / 8;
      int col_unit2 = p_col2 / 8;
      int col_off = (p_col % 8) * (int)sizeof(bf16);
      int col_off2 = (p_col2 % 8) * (int)sizeof(bf16);
      int a0 = p_smem + swizzle<P_STRIDE>(p_row, col_unit) + col_off;
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(a0),
                   "r"(p_f16_local[nl][0]));
      int a1 = p_smem + swizzle<P_STRIDE>(p_row + 8, col_unit) + col_off;
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(a1),
                   "r"(p_f16_local[nl][1]));
      int a2 = p_smem + swizzle<P_STRIDE>(p_row, col_unit2) + col_off2;
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(a2),
                   "r"(p_f16_local[nl][2]));
      int a3 = p_smem + swizzle<P_STRIDE>(p_row + 8, col_unit2) + col_off2;
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(a3),
                   "r"(p_f16_local[nl][3]));
    }
    asm volatile("bar.sync %0, 64;" ::"r"(1 + warp_m));

    // PV Matmul
    {
      int v_row = lane_id % 16;
      int v_col_base = warp_d * (PF_D_CKV / 2 / 8) + lane_id / 16;
      int p_ldm_row = warp_m * 16 + (lane_id % 16);
      int p_ldm_col = lane_id / 16;
#pragma unroll
      for (int mma_kv = 0; mma_kv < PF_NUM_MMA_KV; mma_kv++) {
        uint32_t p_frag[4];
        int p_addr =
            p_smem + swizzle<P_STRIDE>(p_ldm_row, mma_kv * 2 + p_ldm_col);
        ldmatrix_x4(p_frag, p_addr);
#pragma unroll
        for (int mma_d = 0; mma_d < PF_NUM_MMA_D_V_HALF; mma_d++) {
          uint32_t v_frag[4];
          int v_r = v_row + mma_kv * 16;
          int v_c = v_col_base + mma_d * 2;
          int v_addr = ckv_smem(stage) + swizzle<STRIDE_CKV_B>(v_r, v_c);
          ldmatrix_x4_trans(v_frag, v_addr);
          mma_m16n16k16_bf16(p_frag, v_frag, o_frag[mma_d]);
        }
      }
    }
  }

  // Normalize O
  {
    constexpr int DWG_OFF =
        PF_Q_NOPE_SIZE + PF_Q_PE_SIZE +
        PF_NUM_STAGES * (PF_CKV_STAGE_SIZE + PF_KPE_STAGE_SIZE);
    float *d_wg_smem = reinterpret_cast<float *>(smem_raw + DWG_OFF);
#pragma unroll
    for (int j = 0; j < 2; j++) {
      if (lane_id % 4 == 0) {
        d_wg_smem[warp_d * PF_BM + warp_m * 16 + j * 8 + lane_id / 4] =
            d_state[j];
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 2; j++) {
      d_state[j] = d_wg_smem[0 * PF_BM + warp_m * 16 + j * 8 + lane_id / 4] +
                   d_wg_smem[1 * PF_BM + warp_m * 16 + j * 8 + lane_id / 4];
    }
    float d_rcp[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      if (m_state[j] != -INFINITY) {
        asm volatile("rcp.approx.ftz.f32 %0, %1;"
                     : "=f"(d_rcp[j])
                     : "f"(d_state[j]));
      } else {
        d_rcp[j] = 0.0f;
      }
    }
#pragma unroll
    for (int mma_d = 0; mma_d < PF_NUM_MMA_D_V_HALF; mma_d++)
#pragma unroll
      for (int reg_id = 0; reg_id < 8; reg_id++) {
        int j = (reg_id % 4) / 2;
        o_frag[mma_d][reg_id] *= d_rcp[j];
      }
  }

  // Write O to global memory
  {
    using bf16_2 = __nv_bfloat162;
    int g = lane_id / 4;
    int t = lane_id % 4;
#pragma unroll
    for (int mma_d = 0; mma_d < PF_NUM_MMA_D_V_HALF; mma_d++) {
      int d_base = warp_d * (PF_D_V / 2) + mma_d * 16;
      {
        int q_pos = q_start + warp_m * 16 + g;
        if (q_pos < S) {
          long long base_offset =
              (long long)q_pos * H * PF_D_V + (long long)head * PF_D_V + d_base;
          bf16_2 val0 = __float22bfloat162_rn(
              make_float2(o_frag[mma_d][0], o_frag[mma_d][1]));
          *reinterpret_cast<bf16_2 *>(&O[base_offset + 2 * t]) = val0;
          bf16_2 val4 = __float22bfloat162_rn(
              make_float2(o_frag[mma_d][4], o_frag[mma_d][5]));
          *reinterpret_cast<bf16_2 *>(&O[base_offset + 2 * t + 8]) = val4;
        }
      }
      {
        int q_pos = q_start + warp_m * 16 + g + 8;
        if (q_pos < S) {
          long long base_offset =
              (long long)q_pos * H * PF_D_V + (long long)head * PF_D_V + d_base;
          bf16_2 val2 = __float22bfloat162_rn(
              make_float2(o_frag[mma_d][2], o_frag[mma_d][3]));
          *reinterpret_cast<bf16_2 *>(&O[base_offset + 2 * t]) = val2;
          bf16_2 val6 = __float22bfloat162_rn(
              make_float2(o_frag[mma_d][6], o_frag[mma_d][7]));
          *reinterpret_cast<bf16_2 *>(&O[base_offset + 2 * t + 8]) = val6;
        }
      }
    }
  }
}

} // namespace kernel
