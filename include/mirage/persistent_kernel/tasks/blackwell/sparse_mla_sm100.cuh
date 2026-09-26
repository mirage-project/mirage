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

// Sparse MLA on an already-populated paged latent cache, SM100.
// One CTA handles one query, 16 heads, and one split of its token indices.
// The same task supports decode and chunked prefill: causality uses original
// token positions, never positions in the selected list. No global KV gather.
#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace kernel {
namespace sparse_mla {

using bf16 = __nv_bfloat16;
static constexpr int LATENT_DIM = 512;
static constexpr int HEAD_TILE = 16;
static constexpr int KV_TILE = 64;
static constexpr int NUM_THREADS = 256;

template <int ROPE_DIM>
struct alignas(32) SharedStorage {
  static_assert(ROPE_DIM == 0 || ROPE_DIM == 64, "unsupported RoPE dimension");
  static constexpr int D_QK = LATENT_DIM + ROPE_DIM;
  bf16 q[HEAD_TILE * D_QK];
  bf16 kv[KV_TILE * D_QK];
  float scores[HEAD_TILE * KV_TILE];
  bf16 probabilities[HEAD_TILE * KV_TILE];
  float output[HEAD_TILE * LATENT_DIM];
  float update[HEAD_TILE * LATENT_DIM];
  float row_max[HEAD_TILE];
  float row_sum[HEAD_TILE];
  float alpha[HEAD_TILE];
  int64_t cache_rows[KV_TILE];
};

} // namespace sparse_mla

template <int NUM_HEADS, int ROPE_DIM, int PAGE_SIZE, int NUM_SPLITS>
__device__ __forceinline__ void
    sparse_mla_sm100_task_impl(void const *q_ptr,
                               void const *cache_ptr,
                               int const *indices,
                               int const *index_counts,
                               void *output_ptr,
                               float *partial_output,
                               float *partial_lse,
                               int const *qo_indptr,
                               int const *kv_indptr,
                               int const *page_indices,
                               int const *last_page_len,
                               int num_requests,
                               int num_pages,
                               float softmax_scale,
                               int index_capacity,
                               int query_idx,
                               int head_group,
                               int split_idx) {
  using namespace sparse_mla;
  namespace wmma = nvcuda::wmma;
  static_assert(NUM_HEADS == 8 || NUM_HEADS == 16 || NUM_HEADS == 32 ||
                    NUM_HEADS == 64,
                "unsupported head count");
  static_assert(NUM_SPLITS == 1 || NUM_SPLITS == 2 || NUM_SPLITS == 4 ||
                    NUM_SPLITS == 8,
                "unsupported split count");
  static_assert(PAGE_SIZE > 0, "invalid page size");
  static_assert(sizeof(SharedStorage<ROPE_DIM>) <= 200 * 1024,
                "sparse MLA exceeds the MPK shared-memory budget");
  constexpr int D_QK = LATENT_DIM + ROPE_DIM;
  int const tid = threadIdx.x;
  int const warp = tid / 32;
  int const head_start = head_group * HEAD_TILE;
  auto const *q = static_cast<bf16 const *>(q_ptr);
  auto const *cache = static_cast<bf16 const *>(cache_ptr);
  auto *output = static_cast<bf16 *>(output_ptr);
  extern __shared__ __align__(1024) char smem_buf[];
  auto &smem = *reinterpret_cast<SharedStorage<ROPE_DIM> *>(smem_buf);

  // Padded query tasks still overwrite their output/partials with zeros.
  int request = 0;
  while (request < num_requests && query_idx >= qo_indptr[request + 1]) {
    request++;
  }
  bool const active = request < num_requests && query_idx >= qo_indptr[request];
  int first_page = 0, seq_len = 0, query_pos = -1, count = 0;
  if (active) {
    first_page = kv_indptr[request];
    int const pages = kv_indptr[request + 1] - first_page;
    seq_len = pages > 0 ? (pages - 1) * PAGE_SIZE + last_page_len[request] : 0;
    query_pos = seq_len - (qo_indptr[request + 1] - qo_indptr[request]) +
                query_idx - qo_indptr[request];
    // Valid callers supply 0 <= count <= capacity. Clamp before any load so
    // malformed metadata cannot index beyond the indices tensor.
    count = min(max(index_counts[query_idx], 0), index_capacity);
  }
  for (int i = tid; i < HEAD_TILE * D_QK / 8; i += NUM_THREADS) {
    int const row = i * 8 / D_QK;
    int const col = i * 8 % D_QK;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (active && head_start + row < NUM_HEADS) {
      value = *reinterpret_cast<uint4 const *>(
          q + (int64_t(query_idx) * NUM_HEADS + head_start + row) * D_QK + col);
    }
    *reinterpret_cast<uint4 *>(smem.q + i * 8) = value;
  }
  for (int i = tid; i < HEAD_TILE * LATENT_DIM; i += NUM_THREADS) {
    smem.output[i] = 0.0f;
  }
  if (tid < HEAD_TILE) {
    smem.row_max[tid] = -INFINITY;
    smem.row_sum[tid] = 0.0f;
  }
  __syncthreads();

  int const num_tiles = (count + KV_TILE - 1) / KV_TILE;
  int const tiles_per_split = (num_tiles + NUM_SPLITS - 1) / NUM_SPLITS;
  int const first_tile = split_idx * tiles_per_split;
  int const last_tile = min(first_tile + tiles_per_split, num_tiles);
  for (int tile = first_tile; tile < last_tile; tile++) {
    if (tid < KV_TILE) {
      int const slot = tile * KV_TILE + tid;
      int64_t cache_row = -1;
      if (slot < count) {
        int const pos = indices[int64_t(query_idx) * index_capacity + slot];
        if (pos >= 0 && pos < seq_len && pos <= query_pos) {
          int const page = page_indices[first_page + pos / PAGE_SIZE];
          if (page >= 0 && page < num_pages) {
            cache_row = int64_t(page) * PAGE_SIZE + pos % PAGE_SIZE;
          }
        }
      }
      smem.cache_rows[tid] = cache_row;
    }
    __syncthreads();
    for (int i = tid; i < KV_TILE * D_QK / 8; i += NUM_THREADS) {
      int const row = i * 8 / D_QK;
      int const col = i * 8 % D_QK;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (smem.cache_rows[row] >= 0) {
        value = *reinterpret_cast<uint4 const *>(
            cache + smem.cache_rows[row] * D_QK + col);
      }
      *reinterpret_cast<uint4 *>(smem.kv + i * 8) = value;
    }
    __syncthreads();

    // Four warps compute disjoint 16-token columns of Q @ KV^T. WMMA uses
    // BF16 Tensor Cores, with explicit row-major stores for row softmax.
    if (warp < KV_TILE / 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int k = 0; k < D_QK; k += 16) {
        wmma::load_matrix_sync(a, smem.q + k, D_QK);
        wmma::load_matrix_sync(b, smem.kv + warp * 16 * D_QK + k, D_QK);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(
          smem.scores + warp * 16, acc, KV_TILE, wmma::mem_row_major);
    }
    __syncthreads();
    if (tid < HEAD_TILE) {
      float tile_max = -INFINITY;
      for (int j = 0; j < KV_TILE; j++) {
        float const score =
            smem.cache_rows[j] >= 0 && head_start + tid < NUM_HEADS
                ? smem.scores[tid * KV_TILE + j] * softmax_scale
                : -INFINITY;
        smem.scores[tid * KV_TILE + j] = score;
        tile_max = fmaxf(tile_max, score);
      }
      float const new_max = fmaxf(smem.row_max[tid], tile_max);
      float const alpha = smem.row_max[tid] == -INFINITY
                              ? 0.0f
                              : expf(smem.row_max[tid] - new_max);
      float sum = 0.0f;
      for (int j = 0; j < KV_TILE; j++) {
        float const score = smem.scores[tid * KV_TILE + j];
        float const p = score == -INFINITY ? 0.0f : expf(score - new_max);
        smem.probabilities[tid * KV_TILE + j] = __float2bfloat16(p);
        sum += p;
      }
      smem.alpha[tid] = alpha;
      smem.row_sum[tid] = alpha * smem.row_sum[tid] + sum;
      smem.row_max[tid] = new_max;
    }
    __syncthreads();

    // Eight warps cover 64 latent columns each. Position-only columns are
    // deliberately excluded from V. FP32 output is rescaled between tiles.
    for (int c = 0; c < 4; c++) {
      int const col = warp * 64 + c * 16;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int k = 0; k < KV_TILE; k += 16) {
        wmma::load_matrix_sync(a, smem.probabilities + k, KV_TILE);
        wmma::load_matrix_sync(b, smem.kv + k * D_QK + col, D_QK);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(
          smem.update + col, acc, LATENT_DIM, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < HEAD_TILE * LATENT_DIM; i += NUM_THREADS) {
      smem.output[i] =
          smem.alpha[i / LATENT_DIM] * smem.output[i] + smem.update[i];
    }
    __syncthreads();
  }

  for (int i = tid; i < HEAD_TILE * LATENT_DIM; i += NUM_THREADS) {
    int const row = i / LATENT_DIM;
    int const head = head_start + row;
    if (head < NUM_HEADS) {
      float const sum = smem.row_sum[row];
      float const value = sum > 0.0f ? smem.output[i] / sum : 0.0f;
      int64_t const offset =
          (int64_t(query_idx) * NUM_HEADS + head) * LATENT_DIM + i % LATENT_DIM;
      if constexpr (NUM_SPLITS == 1) {
        output[offset] = __float2bfloat16(value);
      } else {
        partial_output[((int64_t(query_idx) * NUM_SPLITS + split_idx) *
                            NUM_HEADS +
                        head) *
                           LATENT_DIM +
                       i % LATENT_DIM] = value;
      }
    }
  }
  if constexpr (NUM_SPLITS > 1) {
    if (tid < HEAD_TILE && head_start + tid < NUM_HEADS) {
      partial_lse[(int64_t(query_idx) * NUM_SPLITS + split_idx) * NUM_HEADS +
                  head_start + tid] =
          smem.row_sum[tid] > 0.0f ? smem.row_max[tid] + logf(smem.row_sum[tid])
                                   : -INFINITY;
    }
  }
  __syncthreads();
}

template <int NUM_HEADS, int NUM_SPLITS>
__device__ __forceinline__ void
    sparse_mla_reduce_sm100_task_impl(float const *partial_output,
                                      float const *partial_lse,
                                      void *output_ptr,
                                      int query_idx,
                                      int head_group) {
  using namespace sparse_mla;
  static_assert(NUM_SPLITS == 2 || NUM_SPLITS == 4 || NUM_SPLITS == 8,
                "reduce requires multiple splits");
  int const tid = threadIdx.x;
  int const head_start = head_group * HEAD_TILE;
  auto *output = static_cast<bf16 *>(output_ptr);
  extern __shared__ __align__(1024) char smem_buf[];
  auto *weights = reinterpret_cast<float *>(smem_buf);
  if (tid < HEAD_TILE) {
    int const head = head_start + tid;
    float max_lse = -INFINITY;
    if (head < NUM_HEADS) {
      for (int s = 0; s < NUM_SPLITS; s++) {
        max_lse = fmaxf(
            max_lse,
            partial_lse[(int64_t(query_idx) * NUM_SPLITS + s) * NUM_HEADS +
                        head]);
      }
    }
    float sum = 0.0f;
    for (int s = 0; s < NUM_SPLITS; s++) {
      float const weight =
          max_lse == -INFINITY
              ? 0.0f
              : expf(partial_lse[(int64_t(query_idx) * NUM_SPLITS + s) *
                                     NUM_HEADS +
                                 head] -
                     max_lse);
      weights[tid * NUM_SPLITS + s] = weight;
      sum += weight;
    }
    for (int s = 0; s < NUM_SPLITS; s++) {
      weights[tid * NUM_SPLITS + s] =
          sum > 0.0f ? weights[tid * NUM_SPLITS + s] / sum : 0.0f;
    }
  }
  __syncthreads();
  for (int i = tid; i < HEAD_TILE * LATENT_DIM; i += NUM_THREADS) {
    int const row = i / LATENT_DIM;
    int const head = head_start + row;
    if (head < NUM_HEADS) {
      float value = 0.0f;
      for (int s = 0; s < NUM_SPLITS; s++) {
        value +=
            weights[row * NUM_SPLITS + s] *
            partial_output[((int64_t(query_idx) * NUM_SPLITS + s) * NUM_HEADS +
                            head) *
                               LATENT_DIM +
                           i % LATENT_DIM];
      }
      output[(int64_t(query_idx) * NUM_HEADS + head) * LATENT_DIM +
             i % LATENT_DIM] = __float2bfloat16(value);
    }
  }
  __syncthreads();
}

} // namespace kernel
