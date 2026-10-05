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
// Shared decode/chunked-prefill task with split partial-output reduction.
// Per-query indices are gathered directly into SMEM with a double-buffered
// cp.async pipeline; no dense global gather. Index lookahead, warp-local page
// lookup sharing, stable tile-local compaction, and empty-tile skipping reduce
// gather and MMA work according to the valid token count.
#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernel {
namespace sparse_mla {

using bf16 = __nv_bfloat16;
static constexpr int LATENT_DIM = 512;
static constexpr int HEAD_TILE = 16;
static constexpr int KV_TILE = 64;
static constexpr int NUM_THREADS = 256;
static constexpr int NUM_STAGES = 2;

template <int ROPE_DIM>
struct alignas(128) SharedStorage {
  static_assert(ROPE_DIM == 0 || ROPE_DIM == 64, "unsupported RoPE dimension");
  static constexpr int D_QK = LATENT_DIM + ROPE_DIM;
  bf16 q[HEAD_TILE * D_QK];
  bf16 kv[NUM_STAGES][KV_TILE * D_QK];
  float scores[HEAD_TILE * KV_TILE];
  bf16 probabilities[HEAD_TILE * KV_TILE];
  float row_max[HEAD_TILE];
  float row_sum[HEAD_TILE];
  float alpha[HEAD_TILE];
  int64_t cache_rows[NUM_STAGES][KV_TILE];
  unsigned valid_masks[NUM_STAGES][KV_TILE / 32];
};

// XOR 16-byte chunks within each 128-byte segment. All strides below are
// multiples of 64 BF16 elements. Global Q/cache layouts remain unchanged.
template <int STRIDE>
__device__ __forceinline__ int swizzle(int row, int col) {
  static_assert(STRIDE % 64 == 0, "swizzle requires 128-byte row alignment");
  return row * STRIDE + (col ^ ((row & 7) * 8));
}

__device__ __forceinline__ void
    copy_async(bf16 *dst, bf16 const *src, bool valid) {
  // src is in bounds even for a masked row; src-size=0 zero-fills all 16 B.
  unsigned const address = __cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address),
               "l"(src),
               "r"(valid ? 16 : 0)
               : "memory");
}

__device__ __forceinline__ void copy_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void copy_wait() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void load_a(uint32_t (&r)[4], bf16 const *ptr) {
  unsigned const address = __cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
               "{%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void load_k(uint32_t (&r)[2], bf16 const *ptr) {
  unsigned const address = __cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void load_v(uint32_t (&r)[4], bf16 const *ptr) {
  unsigned const address = __cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
               "{%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void
    mma(uint32_t const *a, uint32_t const *b, float *c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Scalar loads allow arbitrary index row alignment/capacity (including 2051).
__device__ __forceinline__ int load_index(int const *indices,
                                          int64_t index_offset,
                                          int count,
                                          int tile,
                                          int last_tile) {
  int const tid = threadIdx.x;
  int64_t const slot = int64_t(tile) * KV_TILE + tid;
  return tid < KV_TILE && tile < last_tile && slot < count
             ? __ldg(indices + index_offset + slot)
             : -1;
}

template <int ROPE_DIM>
__device__ __forceinline__ int
    tile_valid_count(SharedStorage<ROPE_DIM> const &smem, int stage) {
  return __popc(smem.valid_masks[stage][0]) +
         __popc(smem.valid_masks[stage][1]);
}

template <int ROPE_DIM, int PAGE_SIZE>
__device__ __forceinline__ void prefetch_tile(SharedStorage<ROPE_DIM> &smem,
                                              bf16 const *cache,
                                              int const *indices,
                                              int const *page_indices,
                                              int64_t index_offset,
                                              int count,
                                              int first_page,
                                              int seq_len,
                                              int query_pos,
                                              int num_pages,
                                              int tile,
                                              int stage,
                                              int prefetched_pos) {
  constexpr int D_QK = LATENT_DIM + ROPE_DIM;
  int const tid = threadIdx.x;
  int const lane = tid % 32;
  int64_t cache_row = -1;
  unsigned valid_mask = 0;
  if (tid < KV_TILE) {
    int const slot = tile * KV_TILE + tid;

    int const pos = prefetched_pos;
    bool const valid =
        slot < count && pos >= 0 && pos < seq_len && pos <= query_pos;
    int const logical_page = valid ? pos / PAGE_SIZE : -1;
    // One page-table LDG per distinct logical page in this warp. This
    // targets pooled/adjacent selections without sorting or changing their
    // order. All lanes (including invalid ones) participate in match/shfl.
    unsigned const peers = __match_any_sync(0xffffffff, logical_page);
    int const leader = __ffs(peers) - 1;
    int page = -1;
    if (valid && lane == leader) {
      page = __ldg(page_indices + first_page + logical_page);
    }
    page = __shfl_sync(0xffffffff, page, leader);
    if (valid && page >= 0 && page < num_pages) {
      cache_row = int64_t(page) * PAGE_SIZE + pos % PAGE_SIZE;
    }

    smem.cache_rows[stage][tid] = cache_row;

    valid_mask = __ballot_sync(0xffffffff, cache_row >= 0);
    if (lane == 0) {
      smem.valid_masks[stage][tid / 32] = valid_mask;
    }
  }
  __syncthreads();
  int rows_to_load = KV_TILE;
  int live_rows = KV_TILE;

  live_rows = tile_valid_count(smem, stage);
  if (live_rows == 0) {
    // CTA-uniform: no KV copy/zero-fill, no async group to wait for.
    return;
  }
  if (live_rows < KV_TILE) {
    // Every lane retained its original row in a register before the barrier,
    // so scattering into the same SMEM array cannot clobber another source.
    if (tid < KV_TILE && cache_row >= 0) {
      unsigned const lower_lanes = (1u << lane) - 1u;
      int const prefix = __popc(valid_mask & lower_lanes) +
                         (tid >= 32 ? __popc(smem.valid_masks[stage][0]) : 0);
      smem.cache_rows[stage][prefix] = cache_row;
    }
    __syncthreads();
  }
  // PV consumes 16 K rows at a time; only clear the final MMA's padding.
  rows_to_load = (live_rows + 15) / 16 * 16;

  for (int i = tid; i < rows_to_load * D_QK / 8; i += NUM_THREADS) {
    int const row = i / (D_QK / 8);
    int const col = i % (D_QK / 8) * 8;
    int64_t const cache_row =
        row < live_rows ? smem.cache_rows[stage][row] : -1;
    copy_async(smem.kv[stage] + swizzle<D_QK>(row, col),
               cache + (cache_row >= 0 ? cache_row * D_QK : 0) + col,
               cache_row >= 0);
  }
  copy_commit();
}

template <int NUM_HEADS, int NUM_SPLITS>
__device__ __forceinline__ void clear_split(void *output_ptr,
                                            float *partial_output,
                                            float *partial_lse,
                                            int query_idx,
                                            int head_start,
                                            int split_idx) {
  auto *output = static_cast<bf16 *>(output_ptr);
  for (int i = threadIdx.x; i < HEAD_TILE * LATENT_DIM; i += NUM_THREADS) {
    int const head = head_start + i / LATENT_DIM;
    if (head < NUM_HEADS) {
      if constexpr (NUM_SPLITS == 1) {
        output[(int64_t(query_idx) * NUM_HEADS + head) * LATENT_DIM +
               i % LATENT_DIM] = __float2bfloat16(0.0f);
      } else {
        partial_output[((int64_t(query_idx) * NUM_SPLITS + split_idx) *
                            NUM_HEADS +
                        head) *
                           LATENT_DIM +
                       i % LATENT_DIM] = 0.0f;
      }
    }
  }
  if constexpr (NUM_SPLITS > 1) {
    if (threadIdx.x < HEAD_TILE && head_start + threadIdx.x < NUM_HEADS) {
      partial_lse[(int64_t(query_idx) * NUM_SPLITS + split_idx) * NUM_HEADS +
                  head_start + threadIdx.x] = -INFINITY;
    }
  }
  __syncthreads();
}

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
  int const lane = tid % 32;
  int const head_start = head_group * HEAD_TILE;
  auto const *q = static_cast<bf16 const *>(q_ptr);
  auto const *cache = static_cast<bf16 const *>(cache_ptr);
  auto *output = static_cast<bf16 *>(output_ptr);
  extern __shared__ __align__(1024) char smem_buf[];
  auto &smem = *reinterpret_cast<SharedStorage<ROPE_DIM> *>(smem_buf);

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
    count = min(max(index_counts[query_idx], 0), index_capacity);
  }

  int const num_tiles = count / KV_TILE + (count % KV_TILE != 0);
  int const tiles_per_split = (num_tiles + NUM_SPLITS - 1) / NUM_SPLITS;
  int const first_tile = split_idx * tiles_per_split;
  int const last_tile = min(first_tile + tiles_per_split, num_tiles);

  if (first_tile >= last_tile) {
    // Empty splits/padded queries overwrite outputs before returning, and
    // issue no Q loads or cp.async operations into the worker's SMEM.
    clear_split<NUM_HEADS, NUM_SPLITS>(output_ptr,
                                       partial_output,
                                       partial_lse,
                                       query_idx,
                                       head_start,
                                       split_idx);
    return;
  }

#pragma unroll
  for (int i = tid; i < HEAD_TILE * D_QK / 8; i += NUM_THREADS) {
    int const row = i / (D_QK / 8);
    int const col = i % (D_QK / 8) * 8;
    bool const valid = active && head_start + row < NUM_HEADS;
    int64_t const offset =
        valid ? (int64_t(query_idx) * NUM_HEADS + head_start + row) * D_QK : 0;
    copy_async(smem.q + swizzle<D_QK>(row, col), q + offset + col, valid);
  }
  copy_commit();
  if (tid < HEAD_TILE) {
    smem.row_max[tid] = -INFINITY;
    smem.row_sum[tid] = 0.0f;
  }

  // Each warp owns 64 latent columns; each lane retains 32 FP32 output
  // elements across tiles. No SMEM output/update round trip in the mainloop.
  float acc[4][8] = {};
  int next_pos = -1;
  if (first_tile < last_tile) {
    int first_pos = -1;

    first_pos = load_index(indices,
                           int64_t(query_idx) * index_capacity,
                           count,
                           first_tile,
                           last_tile);
    next_pos = load_index(indices,
                          int64_t(query_idx) * index_capacity,
                          count,
                          first_tile + 1,
                          last_tile);

    prefetch_tile<ROPE_DIM, PAGE_SIZE>(smem,
                                       cache,
                                       indices,
                                       page_indices,
                                       int64_t(query_idx) * index_capacity,
                                       count,
                                       first_page,
                                       seq_len,
                                       query_pos,
                                       num_pages,
                                       first_tile,
                                       0,
                                       first_pos);
  }
  copy_wait();
  __syncthreads();

  for (int tile = first_tile; tile < last_tile; tile++) {
    // Parity is relative to this split, whose first tile may be odd.
    int const stage = (tile - first_tile) & 1;
    if (tile + 1 < last_tile) {
      int following_pos = -1;

      // Keep the following index in a register while translating/gathering
      // the next tile and computing QK/softmax/PV for the current tile.
      following_pos = load_index(indices,
                                 int64_t(query_idx) * index_capacity,
                                 count,
                                 tile + 2,
                                 last_tile);

      prefetch_tile<ROPE_DIM, PAGE_SIZE>(smem,
                                         cache,
                                         indices,
                                         page_indices,
                                         int64_t(query_idx) * index_capacity,
                                         count,
                                         first_page,
                                         seq_len,
                                         query_pos,
                                         num_pages,
                                         tile + 1,
                                         stage ^ 1,
                                         next_pos);
      next_pos = following_pos;
    }
    int const live_rows = tile_valid_count(smem, stage);

    if (live_rows == 0) {
      // Do not change online max/sum/acc for an empty tile. Drain next-tile
      // copies and keep the same stage rotation/barrier protocol as PV.
      copy_wait();
      __syncthreads();
      continue;
    }

    auto const *kv = smem.kv[stage];

    // All eight warps compute QK: 16 heads x 8 selected tokens per warp.
    // A: [16,16] row-major. B: eight KV rows interpreted as column-major.
    if (warp * 8 < live_rows) {
      float scores[4] = {};
#pragma unroll
      for (int k = 0; k < D_QK; k += 16) {
        uint32_t a[4], b[2];
        load_a(a, smem.q + swizzle<D_QK>(lane % 16, k + lane / 16 * 8));
        load_k(b,
               kv + swizzle<D_QK>(warp * 8 + lane % 8, k + (lane / 8 % 2) * 8));
        mma(a, b, scores);
      }
#pragma unroll
      for (int r = 0; r < 2; r++) {
        int const row = lane / 4 + r * 8;
        int const col = warp * 8 + (lane % 4) * 2;
        *reinterpret_cast<float2 *>(smem.scores + row * KV_TILE + col) =
            make_float2(scores[r * 2], scores[r * 2 + 1]);
      }
    }
    __syncthreads();

    // 16 lanes per head, four scores per lane. Pair stores avoid two lanes
    // writing different BF16 halves of the same shared-memory word.
    int const row = tid / 16;
    int const row_lane = tid % 16;
    float p[4];
    float tile_max = -INFINITY;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int const col = row_lane * 2 + (i / 2) * 32 + i % 2;
      bool const valid = col < live_rows;
      p[i] = valid && head_start + row < NUM_HEADS
                 ? smem.scores[row * KV_TILE + col] * softmax_scale
                 : -INFINITY;
      tile_max = fmaxf(tile_max, p[i]);
    }
#pragma unroll
    for (int mask = 8; mask > 0; mask /= 2) {
      tile_max =
          fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, mask, 16));
    }
    float const old_max = smem.row_max[row];
    float const new_max = fmaxf(old_max, tile_max);
    float const alpha = old_max == -INFINITY ? 0.0f : __expf(old_max - new_max);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      p[i] = p[i] == -INFINITY ? 0.0f : __expf(p[i] - new_max);
      sum += p[i];
    }
#pragma unroll
    for (int mask = 8; mask > 0; mask /= 2) {
      sum += __shfl_xor_sync(0xffffffff, sum, mask, 16);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int const col = row_lane * 2 + i * 32;
      *reinterpret_cast<__nv_bfloat162 *>(smem.probabilities +
                                          swizzle<KV_TILE>(row, col)) =
          __float22bfloat162_rn(make_float2(p[i * 2], p[i * 2 + 1]));
    }
    // Every lane must finish reading the old row_max before its row leader
    // overwrites it. Shuffle collectives do not order shared-memory accesses.
    __syncwarp();
    if (row_lane == 0) {
      smem.alpha[row] = alpha;
      smem.row_sum[row] = alpha * smem.row_sum[row] + sum;
      smem.row_max[row] = new_max;
    }
    __syncthreads();

    // Explicit mma.m16n8k16 accumulator layout (as in the TP8 prefill MLA):
    // registers 0,1,4,5 hold row lane/4; 2,3,6,7 hold row lane/4+8.
    float const alpha0 = smem.alpha[lane / 4];
    float const alpha1 = smem.alpha[lane / 4 + 8];
#pragma unroll
    for (int c = 0; c < 4; c++) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc[c][i] *= (i & 2) ? alpha1 : alpha0;
      }
    }
#pragma unroll
    for (int k = 0; k < KV_TILE; k += 16) {
      if (k < live_rows) {
        uint32_t a[4];
        load_a(a,
               smem.probabilities +
                   swizzle<KV_TILE>(lane % 16, k + lane / 16 * 8));
#pragma unroll
        for (int c = 0; c < 4; c++) {
          uint32_t b[4];
          load_v(b,
                 kv + swizzle<D_QK>(k + lane % 16,
                                    warp * 64 + c * 16 + lane / 16 * 8));
          mma(a, b, acc[c]);
          mma(a, b + 2, acc[c] + 4);
        }
      }
    }
    // Finish the prefetch before consuming it; the CTA barrier also prevents
    // overwriting this tile while another warp is still reading it for PV.
    copy_wait();
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; r++) {
    int const row = lane / 4 + r * 8;
    int const head = head_start + row;
    float const sum = smem.row_sum[row];
    float const inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    if (head < NUM_HEADS) {
#pragma unroll
      for (int c = 0; c < 4; c++) {
#pragma unroll
        for (int half = 0; half < 2; half++) {
          int const col = warp * 64 + c * 16 + half * 8 + (lane % 4) * 2;
          int const reg = half * 4 + r * 2;
          float2 const value =
              make_float2(acc[c][reg] * inv_sum, acc[c][reg + 1] * inv_sum);
          if constexpr (NUM_SPLITS == 1) {
            int64_t const offset =
                (int64_t(query_idx) * NUM_HEADS + head) * LATENT_DIM + col;
            *reinterpret_cast<__nv_bfloat162 *>(output + offset) =
                __float22bfloat162_rn(value);
          } else {
            int64_t const offset =
                ((int64_t(query_idx) * NUM_SPLITS + split_idx) * NUM_HEADS +
                 head) *
                    LATENT_DIM +
                col;
            *reinterpret_cast<float2 *>(partial_output + offset) = value;
          }
        }
      }
      if constexpr (NUM_SPLITS > 1) {
        if (warp == 0 && lane % 4 == 0) {
          partial_lse[(int64_t(query_idx) * NUM_SPLITS + split_idx) *
                          NUM_HEADS +
                      head] =
              sum > 0.0f ? smem.row_max[row] + logf(sum) : -INFINITY;
        }
      }
    }
  }
  // MPK may immediately reuse the same worker's shared memory for any task.
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
