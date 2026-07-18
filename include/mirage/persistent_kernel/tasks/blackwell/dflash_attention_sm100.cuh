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
#include "tasks/ampere/mma.cuh"
#include "tasks/ampere/smem_layout.cuh"
#include "tasks/common/common_header.cuh"

#include <cutlass/arch/barrier.h>

namespace kernel {

// ============================================================================
// DFlash non-causal block attention, split ctx/block KV.
//
// The B block-query tokens attend NON-CAUSALLY to [context (ctx_len) ++
// block (B)] keys. Context K/V come from a cache-like buffer (materialized
// once, NOT token-batched); block K/V are this round's. Optional sliding
// window limits each query to keys within `sliding_window` of its absolute
// position.
//
// Inputs are already projected / normed / roped:
//   q     : [B, NUM_Q_HEADS, HEAD_DIM]        (block queries; q_norm + RoPE)
//   ctx_k : [ctx_len, NUM_KV_HEADS, HEAD_DIM] (k_norm + RoPE at ctx positions)
//   ctx_v : [ctx_len, NUM_KV_HEADS, HEAD_DIM] (raw v)
//   blk_k : [B, NUM_KV_HEADS, HEAD_DIM]       (k_norm + RoPE at block pos)
//   blk_v : [B, NUM_KV_HEADS, HEAD_DIM]       (raw v)
//   out   : [B, NUM_Q_HEADS, HEAD_DIM]
//
// Absolute positions: context key j -> j (0..ctx_len-1); block key/query i ->
// ctx_len + i. So key j position == j, query i position == ctx_len + i.
//
// The head counts are PER-TASK: when the layer is grid-split across kv heads
// (grid.x = G), the runtime offsets each input pointer to this task's head
// column slice and the register passes NUM_Q_HEADS/G, NUM_KV_HEADS/G here.
// Q_STRIDE / KV_STRIDE / O_STRIDE are the FULL row widths (in elements) of the
// underlying tensors; 0 means "dense" (row width == per-task width).
//
// Two implementations:
//   * dflash_attention_sm100_ref: scalar reference (one warp per (query,
//     q_head) pair). Used as fallback for shapes the mma path doesn't cover.
//   * mma path: tensor-core flash attention. Requires
//     B * NUM_QO_PER_KV == 64 rows (e.g. Kimi-K2.6 DFlash: B=8, GQA 8:1),
//     bf16/fp16, HEAD_DIM multiple of 64 and <= 128. Per kv head: the 64
//     (query, head) rows form the MMA M dim; each of the 4 warps owns one
//     16-row m-tile, so softmax and the O accumulation stay warp-local (no
//     cross-warp merge buffers). K/V are streamed through shared memory in
//     double-buffered 64-key tiles with cp.async. For sliding-window layers,
//     whole tiles below the window are skipped.
// ============================================================================
template <typename T,
          int NUM_Q_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int B,
          int Q_STRIDE = 0,
          int KV_STRIDE = 0,
          int O_STRIDE = 0>
__device__ __forceinline__ void
    dflash_attention_sm100_ref(void const *q_ptr,
                               void const *ctx_k_ptr,
                               void const *ctx_v_ptr,
                               void const *blk_k_ptr,
                               void const *blk_v_ptr,
                               void *output_ptr,
                               int ctx_len,
                               int sliding_window) {
  static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of 32");
  constexpr int DPT = HEAD_DIM / 32; // dims per lane
  constexpr int NUM_QO_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS;
  constexpr int WARPS = NUM_THREADS / 32;
  constexpr int QS = Q_STRIDE > 0 ? Q_STRIDE : NUM_Q_HEADS * HEAD_DIM;
  constexpr int KVS = KV_STRIDE > 0 ? KV_STRIDE : NUM_KV_HEADS * HEAD_DIM;
  constexpr int OS = O_STRIDE > 0 ? O_STRIDE : NUM_Q_HEADS * HEAD_DIM;

  // Only the NUM_THREADS consumer threads do work (runtime launches more).
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  T const *q = static_cast<T const *>(q_ptr);
  T const *ctx_k = static_cast<T const *>(ctx_k_ptr);
  T const *ctx_v = static_cast<T const *>(ctx_v_ptr);
  T const *blk_k = static_cast<T const *>(blk_k_ptr);
  T const *blk_v = static_cast<T const *>(blk_v_ptr);
  T *out = static_cast<T *>(output_ptr);

  int const T_kv = ctx_len + B;
  float const scale = rsqrtf(static_cast<float>(HEAD_DIM));

  int const warp = threadIdx.x / 32;
  int const lane = threadIdx.x % 32;
  int const total_pairs = B * NUM_Q_HEADS;

  for (int pair = warp; pair < total_pairs; pair += WARPS) {
    int const qi = pair / NUM_Q_HEADS; // query row 0..B-1
    int const h = pair % NUM_Q_HEADS;  // q head
    int const kvh = h / NUM_QO_PER_KV; // kv head
    int const q_pos = ctx_len + qi;

    float q_reg[DPT];
    T const *q_row = q + qi * QS + h * HEAD_DIM;
#pragma unroll
    for (int e = 0; e < DPT; ++e) {
      q_reg[e] = static_cast<float>(q_row[lane * DPT + e]);
    }

    float m_i = -inf;
    float l_i = 0.0f;
    float acc[DPT];
#pragma unroll
    for (int e = 0; e < DPT; ++e) {
      acc[e] = 0.0f;
    }

    for (int j = 0; j < T_kv; ++j) {
      if (sliding_window > 0) {
        int d = q_pos - j;
        d = d < 0 ? -d : d;
        if (d >= sliding_window) {
          continue;
        }
      }
      // select context vs block key/value rows
      T const *k_row;
      T const *v_row;
      if (j < ctx_len) {
        k_row = ctx_k + j * KVS + kvh * HEAD_DIM;
        v_row = ctx_v + j * KVS + kvh * HEAD_DIM;
      } else {
        int bj = j - ctx_len;
        k_row = blk_k + bj * KVS + kvh * HEAD_DIM;
        v_row = blk_v + bj * KVS + kvh * HEAD_DIM;
      }
      float partial = 0.0f;
#pragma unroll
      for (int e = 0; e < DPT; ++e) {
        partial += q_reg[e] * static_cast<float>(k_row[lane * DPT + e]);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, off);
      }
      float score = partial * scale;

      float m_new = fmaxf(m_i, score);
      float corr = __expf(m_i - m_new);
      float p = __expf(score - m_new);
      l_i = l_i * corr + p;
#pragma unroll
      for (int e = 0; e < DPT; ++e) {
        acc[e] = acc[e] * corr + p * static_cast<float>(v_row[lane * DPT + e]);
      }
      m_i = m_new;
    }

    float invv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    T *o_row = out + qi * OS + h * HEAD_DIM;
#pragma unroll
    for (int e = 0; e < DPT; ++e) {
      o_row[lane * DPT + e] = static_cast<T>(acc[e] * invv);
    }
  }
}

template <typename T,
          int NUM_Q_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int B,
          int Q_STRIDE = 0,
          int KV_STRIDE = 0,
          int O_STRIDE = 0>
__device__ __forceinline__ void dflash_attention_sm100(void const *q_ptr,
                                                       void const *ctx_k_ptr,
                                                       void const *ctx_v_ptr,
                                                       void const *blk_k_ptr,
                                                       void const *blk_v_ptr,
                                                       void *output_ptr,
                                                       int ctx_len,
                                                       int sliding_window) {
  static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0);
  constexpr int NUM_QO_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS;
  // The mma path needs exactly 64 (query, q_head) rows per kv head (4 warps x
  // one 16-row m-tile), 16-bit elements, and HEAD_DIM in {64, 128}.
  constexpr bool USE_MMA = (sizeof(T) == 2) && (B * NUM_QO_PER_KV == 64) &&
                           (HEAD_DIM % 64 == 0) && (HEAD_DIM <= 128);

  if constexpr (!USE_MMA) {
    dflash_attention_sm100_ref<T,
                               NUM_Q_HEADS,
                               NUM_KV_HEADS,
                               HEAD_DIM,
                               B,
                               Q_STRIDE,
                               KV_STRIDE,
                               O_STRIDE>(q_ptr,
                                         ctx_k_ptr,
                                         ctx_v_ptr,
                                         blk_k_ptr,
                                         blk_v_ptr,
                                         output_ptr,
                                         ctx_len,
                                         sliding_window);
    return;
  } else {
    constexpr int QS = Q_STRIDE > 0 ? Q_STRIDE : NUM_Q_HEADS * HEAD_DIM;
    constexpr int KVS = KV_STRIDE > 0 ? KV_STRIDE : NUM_KV_HEADS * HEAD_DIM;
    constexpr int OS = O_STRIDE > 0 ? O_STRIDE : NUM_Q_HEADS * HEAD_DIM;
    constexpr int CP_CHUNK_SIZE = 16 / sizeof(T);
    constexpr int KV_TILE_SIZE = 64;
    constexpr int M_ROWS = B * NUM_QO_PER_KV;      // 64
    constexpr int MMA_N_TILES = KV_TILE_SIZE / 16; // 4 (score cols per tile)
    constexpr int MMA_K_TILES = HEAD_DIM / 16;     // QK k-dim / PV n-dim
    static_assert(M_ROWS == 64);
    static_assert(HEAD_DIM % CP_CHUNK_SIZE == 0);

    constexpr int CONSUMER_WARPGROUP_SYNC_BARRIER_ID = 6;
    cutlass::arch::NamedBarrier wg_barrier(NUM_THREADS,
                                           CONSUMER_WARPGROUP_SYNC_BARRIER_ID);
    // Only the NUM_THREADS consumer threads participate (runtime launches
    // more; the extras never touch smem nor arrive at the named barrier).
    if (threadIdx.x >= NUM_THREADS) {
      return;
    }

    float const sm_scale = rsqrtf(static_cast<float>(HEAD_DIM));
    int const warp_idx = warp_id();
    int const lane_idx = lane_id();

    int const T_kv = ctx_len + B;
    // Sliding window: the smallest query position is ctx_len, so no query can
    // see keys below ctx_len - sliding_window + 1. Skip whole tiles below
    // that; per-element masking below handles the ragged boundary.
    int const kv_start =
        (sliding_window > 0)
            ? max(0,
                  ((ctx_len - sliding_window + 1) / KV_TILE_SIZE) *
                      KV_TILE_SIZE)
            : 0;
    int const total_kv = T_kv - kv_start;
    int const num_iters = (total_kv + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

    // STensors' offsets and sizes
    constexpr size_t ZERO_BUFFER_OFFSET = 0;
    constexpr size_t ZERO_BUFFER_SIZE = sizeof(T) * 8;

    constexpr size_t S_Q_OFFSET =
        (ZERO_BUFFER_OFFSET + ZERO_BUFFER_SIZE + 15) & ~size_t(15);
    constexpr size_t S_Q_SIZE = sizeof(T) * M_ROWS * HEAD_DIM;

    constexpr size_t S_K_OFFSET = S_Q_OFFSET + S_Q_SIZE;
    constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;

    constexpr size_t S_K_BUFFER_OFFSET = S_K_OFFSET + S_K_SIZE;
    constexpr size_t S_V_OFFSET = S_K_BUFFER_OFFSET + S_K_SIZE;
    constexpr size_t S_V_BUFFER_OFFSET = S_V_OFFSET + S_K_SIZE;

    constexpr size_t S_O_OFFSET = S_V_BUFFER_OFFSET + S_K_SIZE;
    constexpr size_t S_O_SIZE = S_Q_SIZE;

    constexpr size_t S_TOTAL = S_O_OFFSET + S_O_SIZE;
    static_assert(S_TOTAL <= mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE);

    extern __shared__ char smem[];

    T *zero_buf = reinterpret_cast<T *>(smem + ZERO_BUFFER_OFFSET);
    clear_smem_buffer<T, 8>(zero_buf);
    T *s_q = reinterpret_cast<T *>(smem + S_Q_OFFSET);
    T *s_k = reinterpret_cast<T *>(smem + S_K_OFFSET);
    T *s_k_buffer = reinterpret_cast<T *>(smem + S_K_BUFFER_OFFSET);
    T *s_v = reinterpret_cast<T *>(smem + S_V_OFFSET);
    T *s_v_buffer = reinterpret_cast<T *>(smem + S_V_BUFFER_OFFSET);
    T *s_o = reinterpret_cast<T *>(smem + S_O_OFFSET);

    // STensors' layouts (swizzled)
    using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
    using QOSmem = smem_row<T, 3, 3, 3, M_ROWS, HEAD_DIM, HEAD_DIM>;
    using KVSmem = smem_row<T, 3, 3, 3, KV_TILE_SIZE, HEAD_DIM, HEAD_DIM>;

    ZeroBufferSmem zero_buffer(zero_buf);
    QOSmem q_smem(s_q), o_smem(s_o);
    KVSmem k_smem(s_k), v_smem(s_v);
    KVSmem k_buffer_smem(s_k_buffer), v_buffer_smem(s_v_buffer);

    // DTensors' layouts (strides are the full row widths of the underlying
    // tensors; the per-task pointers are column slices into them)
    using QDmem = dmem_row_const<T, B, NUM_QO_PER_KV * HEAD_DIM, QS>;
    using KVDmem = dmem_row_const<T, KV_TILE_SIZE, HEAD_DIM, KVS>;
    using ODmem = dmem_row<T, B, NUM_QO_PER_KV * HEAD_DIM, OS>;

    // Process this task's kv heads one at a time; each pass streams all KV
    // tiles for that head.
    for (int g = 0; g < NUM_KV_HEADS; g++) {
      // Protects q_smem/o_smem of the previous pass before reloading.
      wg_barrier.arrive_and_wait();

      QDmem q_dmem(static_cast<T const *>(q_ptr) +
                   g * NUM_QO_PER_KV * HEAD_DIM);
      KVDmem ctx_k_dmem(static_cast<T const *>(ctx_k_ptr) + g * HEAD_DIM);
      KVDmem ctx_v_dmem(static_cast<T const *>(ctx_v_ptr) + g * HEAD_DIM);
      KVDmem blk_k_dmem(static_cast<T const *>(blk_k_ptr) + g * HEAD_DIM);
      KVDmem blk_v_dmem(static_cast<T const *>(blk_v_ptr) + g * HEAD_DIM);
      ODmem o_dmem(static_cast<T *>(output_ptr) + g * NUM_QO_PER_KV * HEAD_DIM);

      k_smem.set_ptr(s_k);
      k_buffer_smem.set_ptr(s_k_buffer);
      v_smem.set_ptr(s_v);
      v_buffer_smem.set_ptr(s_v_buffer);

      int curr_iter_len = min(total_kv, KV_TILE_SIZE);
      int cp_finished = 0; // keys issued so far (relative to kv_start)

      // Load Q: gmem rows are [B x (NUM_QO_PER_KV*HEAD_DIM)] per kv head;
      // smem rows are (query, q_head) pairs of HEAD_DIM.
#pragma unroll
      for (int chunk_idx = threadIdx.x;
           chunk_idx < M_ROWS * HEAD_DIM / CP_CHUNK_SIZE;
           chunk_idx += NUM_THREADS) {
        int src_row = chunk_idx / (NUM_QO_PER_KV * HEAD_DIM / CP_CHUNK_SIZE);
        int src_col = (chunk_idx % (NUM_QO_PER_KV * HEAD_DIM / CP_CHUNK_SIZE)) *
                      CP_CHUNK_SIZE;
        int dst_row = src_row * NUM_QO_PER_KV + src_col / HEAD_DIM;
        int dst_col = src_col % HEAD_DIM;
        load_smem(q_smem(dst_row, dst_col), q_dmem(src_row, src_col));
      }

      // Prologue: load KV tile 0 into the buffers.
#pragma unroll
      for (int chunk_idx = threadIdx.x;
           chunk_idx < curr_iter_len * HEAD_DIM / CP_CHUNK_SIZE;
           chunk_idx += NUM_THREADS) {
        int dst_row = chunk_idx / (HEAD_DIM / CP_CHUNK_SIZE);
        int col = (chunk_idx % (HEAD_DIM / CP_CHUNK_SIZE)) * CP_CHUNK_SIZE;
        int j = kv_start + dst_row;
        if (j < ctx_len) {
          load_smem(k_buffer_smem(dst_row, col), ctx_k_dmem(j, col));
          load_smem(v_buffer_smem(dst_row, col), ctx_v_dmem(j, col));
        } else {
          int bj = j - ctx_len;
          load_smem(k_buffer_smem(dst_row, col), blk_k_dmem(bj, col));
          load_smem(v_buffer_smem(dst_row, col), blk_v_dmem(bj, col));
        }
      }
      cp_async_fence();
      cp_finished += curr_iter_len;

      // Per-thread flash-softmax state. Each warp owns m-tile `warp_idx`
      // (rows [warp_idx*16, warp_idx*16+16)); within it each thread holds 2
      // row-halves, so m/d have 2 entries and o holds this warp's 16 x
      // HEAD_DIM accumulator fragments.
      float m_local[2] = {-inf, -inf};
      float d[2] = {1.f, 1.f};
      float o[MMA_K_TILES][8];
#pragma unroll
      for (int n = 0; n < MMA_K_TILES; n++) {
        clear_8_floats(o[n]);
      }
      uint32_t q_frags[MMA_K_TILES][4];

      for (int iter = 0; iter < num_iters; iter++) {
        int next_iter_len =
            iter + 1 < num_iters ? min(total_kv - cp_finished, KV_TILE_SIZE)
                                 : 0;
        if (next_iter_len > 0) {
          // Prefetch the next tile into the compute buffers of this iter
          // (they become the buffer pair after the rotation below).
#pragma unroll
          for (int chunk_idx = threadIdx.x;
               chunk_idx < next_iter_len * HEAD_DIM / CP_CHUNK_SIZE;
               chunk_idx += NUM_THREADS) {
            int dst_row = chunk_idx / (HEAD_DIM / CP_CHUNK_SIZE);
            int col = (chunk_idx % (HEAD_DIM / CP_CHUNK_SIZE)) * CP_CHUNK_SIZE;
            int j = kv_start + cp_finished + dst_row;
            if (j < ctx_len) {
              load_smem(k_smem(dst_row, col), ctx_k_dmem(j, col));
              load_smem(v_smem(dst_row, col), ctx_v_dmem(j, col));
            } else {
              int bj = j - ctx_len;
              load_smem(k_smem(dst_row, col), blk_k_dmem(bj, col));
              load_smem(v_smem(dst_row, col), blk_v_dmem(bj, col));
            }
          }
          cp_async_fence();
          cp_async_wait<1>();
          cp_finished += next_iter_len;
        } else {
          cp_async_wait<0>();
        }

        // rotate the buffers
        if ((iter & 0x1) == 0) {
          k_smem.set_ptr(s_k_buffer);
          k_buffer_smem.set_ptr(s_k);
          v_smem.set_ptr(s_v_buffer);
          v_buffer_smem.set_ptr(s_v);
        } else {
          k_smem.set_ptr(s_k);
          k_buffer_smem.set_ptr(s_k_buffer);
          v_smem.set_ptr(s_v);
          v_buffer_smem.set_ptr(s_v_buffer);
        }
        wg_barrier.arrive_and_wait();

        // Q never changes across iters: hoist its fragments into registers.
        if (iter == 0) {
          int q_row = (warp_idx << 4) + (lane_idx & 0xF);
#pragma unroll
          for (int k = 0; k < MMA_K_TILES; k++) {
            int q_col = (k << 4) + ((lane_idx >> 4) << 3);
            ldsm(q_smem(q_row, q_col), q_frags[k]);
          }
        }

        // X = Q K^T for this warp's 16 rows x 64 keys (m16n16k16 mma;
        // warp-local: iterate over the 4 n-tiles and the k dim).
        float x_frag_f[MMA_N_TILES][8];
#pragma unroll
        for (int n = 0; n < MMA_N_TILES; n++) {
          clear_8_floats(x_frag_f[n]);
        }
        uint32_t kt_frag[4];
#pragma unroll
        for (int n = 0; n < MMA_N_TILES; n++) {
          int kt_col = (n << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
          for (int k = 0; k < MMA_K_TILES; k++) {
            int kt_row = (k << 4) + (((lane_idx & 0xF) >> 3) << 3);
            T *src_ptr_KT = kt_col < curr_iter_len ? k_smem(kt_col, kt_row)
                                                   : zero_buffer(0, 0);
            ldsm(src_ptr_KT, kt_frag);
            mma_m16n16k16_bf16bf16bf32(
                x_frag_f[n], q_frags[k], kt_frag, x_frag_f[n]);
          }
        }

        // Mask invalid scores and update the running row max.
        float m_prev[2] = {m_local[0], m_local[1]};
#pragma unroll
        for (int n = 0; n < MMA_N_TILES; n++) {
#pragma unroll
          for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
            int row = (warp_idx << 4) + (lane_idx >> 2) +
                      (((frag_idx & 0x3) >> 1) << 3);
            int col = (n << 4) + ((lane_idx & 0x3) << 1) +
                      ((frag_idx >> 2) << 3) + (frag_idx & 0x1);
            bool is_valid = col < curr_iter_len;
            if (sliding_window > 0) {
              int j_abs = kv_start + iter * KV_TILE_SIZE + col;
              int qi = row / NUM_QO_PER_KV;
              int dpos = ctx_len + qi - j_abs;
              dpos = dpos < 0 ? -dpos : dpos;
              is_valid = is_valid && (dpos < sliding_window);
            }
            x_frag_f[n][frag_idx] = is_valid ? x_frag_f[n][frag_idx] : -inf;
            m_local[(frag_idx & 0x3) >> 1] =
                max(m_local[(frag_idx & 0x3) >> 1], x_frag_f[n][frag_idx]);
          }
        }
        // row max across the 4 threads holding each row
        m_local[0] = max(m_local[0], shfl_xor_sync(m_local[0], 0x1));
        m_local[0] = max(m_local[0], shfl_xor_sync(m_local[0], 0x2));
        m_local[1] = max(m_local[1], shfl_xor_sync(m_local[1], 0x1));
        m_local[1] = max(m_local[1], shfl_xor_sync(m_local[1], 0x2));

        float rescale[2];
        rescale[0] = expf(m_prev[0] * sm_scale - m_local[0] * sm_scale);
        rescale[1] = expf(m_prev[1] * sm_scale - m_local[1] * sm_scale);

        // exponentiate and accumulate the row sums
        float d_partial[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < MMA_N_TILES; n++) {
#pragma unroll
          for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
            x_frag_f[n][frag_idx] =
                x_frag_f[n][frag_idx] != -inf
                    ? expf(x_frag_f[n][frag_idx] * sm_scale -
                           m_local[(frag_idx & 0x3) >> 1] * sm_scale)
                    : 0.f;
            d_partial[(frag_idx & 0x3) >> 1] += x_frag_f[n][frag_idx];
          }
        }
        d_partial[0] += shfl_xor_sync(d_partial[0], 0x1);
        d_partial[0] += shfl_xor_sync(d_partial[0], 0x2);
        d_partial[1] += shfl_xor_sync(d_partial[1], 0x1);
        d_partial[1] += shfl_xor_sync(d_partial[1], 0x2);
        d[0] = d[0] * rescale[0] + d_partial[0];
        d[1] = d[1] * rescale[1] + d_partial[1];

        // rescale O
#pragma unroll
        for (int n = 0; n < MMA_K_TILES; n++) {
#pragma unroll
          for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
            o[n][frag_idx] *= rescale[(frag_idx & 0x3) >> 1];
          }
        }

        // O += exp(X - m) V. The QK accumulator fragments double as the mma
        // A operand after f32->bf16 conversion; the 64-key dim is the mma k
        // dim, split into 4 chunks (one per score n-tile).
        uint32_t x_frag[MMA_N_TILES][4], v_frag[4];
#pragma unroll
        for (int kk = 0; kk < MMA_N_TILES; kk++) {
          convert_f32_to_bf16_uint32(x_frag_f[kk], x_frag[kk]);
        }
#pragma unroll
        for (int kk = 0; kk < MMA_N_TILES; kk++) {
          int v_row = (kk << 4) + (lane_idx & 0xF);
#pragma unroll
          for (int n = 0; n < MMA_K_TILES; n++) {
            int v_col = (n << 4) + ((lane_idx >> 4) << 3);
            T *src_ptr_V = v_row < curr_iter_len ? v_smem(v_row, v_col)
                                                 : zero_buffer(0, 0);
            ldsm_t(src_ptr_V, v_frag);
            mma_m16n16k16_bf16bf16bf32(o[n], x_frag[kk], v_frag, o[n]);
          }
        }
        // All warps must be done reading this tile before the next iter's
        // prefetch overwrites it.
        wg_barrier.arrive_and_wait();

        curr_iter_len = next_iter_len;
      }

      // Epilogue: each warp writes its own rows (no cross-warp reduction).
      // If a row saw no valid key, o == 0 and d == 1, so it stores 0 (same
      // as the reference kernel).
#pragma unroll
      for (int n = 0; n < MMA_K_TILES; n++) {
#pragma unroll
        for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
          int row = (warp_idx << 4) + (lane_idx >> 2) +
                    (((frag_idx & 0x3) >> 1) << 3);
          int col = (n << 4) + ((lane_idx & 0x3) << 1) +
                    ((frag_idx >> 2) << 3) + (frag_idx & 0x1);
          o_smem.at(row, col) =
              static_cast<T>(o[n][frag_idx] / d[(frag_idx & 0x3) >> 1]);
        }
      }
      wg_barrier.arrive_and_wait();

      // store the output (smem row (qi, h) -> gmem row qi, col h*HEAD_DIM+)
      for (int elem_idx = threadIdx.x; elem_idx < M_ROWS * HEAD_DIM;
           elem_idx += NUM_THREADS) {
        int src_row = elem_idx / HEAD_DIM;
        int src_col = elem_idx % HEAD_DIM;
        int dst_row = src_row / NUM_QO_PER_KV;
        int dst_col = src_col + (src_row % NUM_QO_PER_KV) * HEAD_DIM;
        o_dmem.at(dst_row, dst_col) = o_smem.at(src_row, src_col);
      }
    } // for g (kv heads)
  }
}

} // namespace kernel
