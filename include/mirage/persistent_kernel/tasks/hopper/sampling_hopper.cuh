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

/*
 * Temperature / top-k / top-p sampling for Hopper: a partial task per vocab
 * chunk and a reduce task. The buffers exchanged between the two, the RNG
 * stream and the sampling semantics are those of sampling_sm100.cuh (whose
 * Gumbel / Philox helpers are reused here), so the two backends draw the same
 * token for the same logits, seed and step.
 *
 * Instead of popping candidates one argmax round at a time, both kernels
 * select them in parallel:
 *
 *   1. threshold  a cheap lower bound L on the k-th largest value: every warp
 *                 takes the ceil(k / 8)-th largest of a sample it owns (one
 *                 representative per thread or per list, each of which stands
 *                 for at least one distinct element >= it), and L is the
 *                 minimum over the 8 warps, so at least k elements are >= L.
 *   2. compact    a deterministic block scan gathers the elements > L and,
 *                 capped at k, the elements == L (ties at the threshold, e.g.
 *                 a flat logit floor, never blow up the candidate set).
 *   3. rank       each candidate > L counts the candidates above it (value,
 *                 then position) and lands directly in its output slot; the
 *                 elements == L fill the slots after them.
 *
 * partial  all 256 worker threads load their strided slice of the chunk at
 *          once; the representative of a thread is the max of its slice.
 * reduce   all 256 threads work on one row at a time. A TMA bulk copy stages
 *          the row's candidates in shared memory; the representatives are
 *          the leading entries of the per-chunk lists (already sorted by the
 *          partial stage). Warp 0 then applies the top-p cut with a prefix
 *          sum and draws with the Gumbel-Max trick.
 *
 * Elements with equal values are ordered by position.
 */

#pragma once
#include "runtime_header.h"
#include "tasks/blackwell/sampling_sm100.cuh"
#include "tasks/common/utils.cuh"
#include "tasks/hopper/barrier.cuh"

namespace kernel {

namespace sampling_hopper_detail {

int constexpr WARP = NUM_THREADS_PER_WARP;
int constexpr NUM_WORKER_WARPS = WORKER_NUM_THREADS / WARP;
// Largest per-thread slice the partial stage loads into registers in one go.
int constexpr MAX_REGISTER_SLICE = 16;

// Monotonic float -> uint32 map. -inf (and only -inf, among non-NaN values)
// is mapped to 0, which therefore also means "no element".
__device__ __forceinline__ uint32_t key_of(float f) {
  uint32_t const bits = __float_as_uint(f);
  uint32_t const key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  return f == -INFINITY ? 0u : key;
}

__device__ __forceinline__ float value_of(uint32_t key) {
  uint32_t const bits = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
  return key == 0u ? -INFINITY : __uint_as_float(bits);
}

// Total order used for ranking: larger value first, then smaller position.
__device__ __forceinline__ uint64_t rank_key(uint32_t key, int pos) {
  return (static_cast<uint64_t>(key) << 32) |
         (0xffffffffu - static_cast<uint32_t>(pos));
}

// Merge two (max, sum-of-exp relative to max) pairs.
__device__ __forceinline__ void
    combine_max_sum(float &m, float &s, float om, float os) {
  if (om == -INFINITY) {
    return;
  }
  if (m == -INFINITY) {
    m = om;
    s = os;
    return;
  }
  float const nm = fmaxf(m, om);
  s = s * __expf(m - nm) + os * __expf(om - nm);
  m = nm;
}

__device__ __forceinline__ void warp_combine_max_sum(float &m, float &s) {
#pragma unroll
  for (int mask = WARP / 2; mask > 0; mask /= 2) {
    float const om = __shfl_xor_sync(0xffffffff, m, mask);
    float const os = __shfl_xor_sync(0xffffffff, s, mask);
    combine_max_sum(m, s, om, os);
  }
}

// q-th largest (1-based, q <= 32) of one key per lane.
__device__ __forceinline__ uint32_t warp_qth_largest(uint32_t key, int q) {
  int const lane = threadIdx.x % WARP;
  int rank = 0;
#pragma unroll
  for (int j = 0; j < WARP; j++) {
    uint32_t const other = __shfl_sync(0xffffffff, key, j);
    rank += (other > key) || (other == key && j < lane);
  }
  int const src = __ffs(__ballot_sync(0xffffffff, rank == q - 1)) - 1;
  return __shfl_sync(0xffffffff, key, src);
}

// Exclusive block-wide scan of two counters (each total < 65536), packed
// into one word so a single scan serves both. Contains one __syncthreads;
// `scratch` (NUM_WORKER_WARPS words) must not be reused before the next one.
__device__ __forceinline__ void block_scan_pair(int a,
                                                int b,
                                                int &a_off,
                                                int &b_off,
                                                int &a_total,
                                                int &b_total,
                                                uint32_t *scratch) {
  int const lane = threadIdx.x % WARP;
  int const warp = threadIdx.x / WARP;
  uint32_t const packed = (static_cast<uint32_t>(a) << 16) | b;
  uint32_t incl = packed;
#pragma unroll
  for (int d = 1; d < WARP; d *= 2) {
    uint32_t const o = __shfl_up_sync(0xffffffff, incl, d);
    incl += lane >= d ? o : 0u;
  }
  if (lane == WARP - 1) {
    scratch[warp] = incl;
  }
  __syncthreads();
  uint32_t base = 0, total = 0;
#pragma unroll
  for (int w = 0; w < NUM_WORKER_WARPS; w++) {
    uint32_t const t = scratch[w];
    base += w < warp ? t : 0u;
    total += t;
  }
  uint32_t const excl = base + incl - packed;
  a_off = excl >> 16;
  b_off = excl & 0xffff;
  a_total = total >> 16;
  b_total = total & 0xffff;
}

// Warp-parallel version of sampling_philox_offset_for_row (same result): all
// request slots are fetched at once and a ballot picks the first one whose
// [qo_indptr[r], qo_indptr[r + 1]) range holds the row, instead of one
// dependent pair of global loads per slot.
__device__ __forceinline__ unsigned long long
    warp_philox_offset_for_row(int batch_idx,
                               int const *steps,
                               int const *request_ids,
                               int const *qo_indptr,
                               unsigned long long fallback_offset) {
  int const lane = threadIdx.x % WARP;
  int found = -1;
  for (int r0 = 0; r0 < MPK_MAX_NUM_BATCHED_REQUESTS && found < 0; r0 += WARP) {
    int const r = r0 + lane;
    bool const hit = r < MPK_MAX_NUM_BATCHED_REQUESTS &&
                     batch_idx >= qo_indptr[r] && batch_idx < qo_indptr[r + 1];
    unsigned const ballot = __ballot_sync(0xffffffff, hit);
    found = ballot != 0u ? r0 + __ffs(ballot) - 1 : -1;
  }
  if (found >= 0) {
    int const rid = request_ids[found];
    if (rid >= 0) {
      return (unsigned long long)steps[rid] * MPK_MAX_NUM_BATCHED_TOKENS +
             (unsigned long long)batch_idx;
    }
  }
  return fallback_offset + (unsigned long long)batch_idx;
}

// 1-D TMA bulk copy global -> shared, completing on `bar`'s transaction count.
__device__ __forceinline__ void
    bulk_copy_g2s(void *dst, void const *src, uint32_t bytes, Barrier &bar) {
  uint32_t const d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t const b = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3];\n" ::"r"(d),
               "l"(src),
               "r"(bytes),
               "r"(b)
               : "memory");
}

} // namespace sampling_hopper_detail

/*
 * Per-chunk stage. Output layout is identical to sampling_partial_sm100_kernel:
 *   output_val[0 .. TOPK_MAX-1] : TOPK_MAX largest scaled logits, descending
 *   output_val[TOPK_MAX]        : chunk max of the scaled logits
 *   output_val[TOPK_MAX + 1]    : sum of exp(scaled logit - chunk max)
 *   output_idx[0 .. TOPK_MAX-1] : vocabulary ids, -1 when the slot is unused
 * Positions at or after VOCAB_SIZE are lm_head padding and never selected.
 */
template <typename T,
          int BATCH_SIZE,
          int CHUNK_SIZE,
          int NUM_PARTIAL_TASKS,
          int TOPK_MAX,
          int VOCAB_SIZE = CHUNK_SIZE *NUM_PARTIAL_TASKS>
__device__ __forceinline__ void
    sampling_partial_hopper_kernel(void const *__restrict__ input_ptr,
                                   void *__restrict__ output_val_ptr,
                                   void *__restrict__ output_idx_ptr,
                                   int num_active_tokens,
                                   float inv_temperature,
                                   int chunk_start) {
  using namespace sampling_hopper_detail;
  int constexpr PER_THREAD =
      (CHUNK_SIZE + WORKER_NUM_THREADS - 1) / WORKER_NUM_THREADS;
  // Each warp's threshold sample is its 32 thread maxima.
  int constexpr Q = (TOPK_MAX + NUM_WORKER_WARPS - 1) / NUM_WORKER_WARPS;
  static_assert(TOPK_MAX > 0 && TOPK_MAX <= 128,
                "TOPK_MAX must be in (0, 128] (see sampling_partial_layer)");
  static_assert(CHUNK_SIZE < 65536, "candidate counts are scanned as 16 bits");
  static_assert(Q <= WARP, "threshold sample too small");
  // scratch, warp max/sum/threshold, keys, candidates > L, candidates == L
  static_assert(4 * 4 * NUM_WORKER_WARPS + 4 * CHUNK_SIZE + 8 * CHUNK_SIZE +
                        4 * TOPK_MAX + 256 <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "sampling_partial_hopper smem exceeds the dynamic budget");

  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  float *__restrict__ output_val = static_cast<float *>(output_val_ptr);
  long long *__restrict__ output_idx = static_cast<long long *>(output_idx_ptr);

  int const val_stride = NUM_PARTIAL_TASKS * (TOPK_MAX + 2);
  int const idx_stride = NUM_PARTIAL_TASKS * TOPK_MAX;
  int const valid_len = max(0, min(CHUNK_SIZE, VOCAB_SIZE - chunk_start));

  extern __shared__ char smem[];
  char *base = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(smem) + 127) / 128 * 128);
  uint64_t *gt_rank_key = reinterpret_cast<uint64_t *>(base);
  uint32_t *chunk_key = reinterpret_cast<uint32_t *>(gt_rank_key + CHUNK_SIZE);
  int *eq_pos = reinterpret_cast<int *>(chunk_key + CHUNK_SIZE);
  uint32_t *scratch = reinterpret_cast<uint32_t *>(eq_pos + TOPK_MAX);
  float *warp_max = reinterpret_cast<float *>(scratch + NUM_WORKER_WARPS);
  float *warp_sum = warp_max + NUM_WORKER_WARPS;
  uint32_t *warp_thresh =
      reinterpret_cast<uint32_t *>(warp_sum + NUM_WORKER_WARPS);

  int const tid = threadIdx.x;
  int const lane = tid % WARP;
  int const warp = tid / WARP;

  for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
    T const *row = input + (size_t)batch_idx * CHUNK_SIZE * NUM_PARTIAL_TASKS;
    float *row_val = output_val + batch_idx * val_stride;
    long long *row_idx = output_idx + batch_idx * idx_stride;

    // ---- Load + scale; online (max, exp-sum); keys; thread max ----
    // Thread tid owns positions tid + j * WORKER_NUM_THREADS.
    float m = -INFINITY, s = 0.0f;
    uint32_t my_max = 0;
    auto consume = [&](int i, float x) {
      uint32_t const k = key_of(x);
      if (x != -INFINITY) {
        if (x > m) {
          s = s * __expf(m - x) + 1.0f;
          m = x;
        } else {
          s += __expf(x - m);
        }
      }
      chunk_key[i] = k;
      my_max = max(my_max, k);
    };
    if constexpr (PER_THREAD <= MAX_REGISTER_SLICE) {
      T raw[PER_THREAD];
#pragma unroll
      for (int j = 0; j < PER_THREAD; j++) {
        int const i = tid + j * WORKER_NUM_THREADS;
        raw[j] = i < valid_len ? row[i] : static_cast<T>(0.0f);
      }
#pragma unroll
      for (int j = 0; j < PER_THREAD; j++) {
        int const i = tid + j * WORKER_NUM_THREADS;
        if (i < CHUNK_SIZE) {
          consume(i,
                  i < valid_len ? static_cast<float>(raw[j]) * inv_temperature
                                : -INFINITY);
        }
      }
    } else {
      for (int i = tid; i < CHUNK_SIZE; i += WORKER_NUM_THREADS) {
        consume(i,
                i < valid_len ? static_cast<float>(row[i]) * inv_temperature
                              : -INFINITY);
      }
    }
    warp_combine_max_sum(m, s);
    uint32_t const warp_q = warp_qth_largest(my_max, Q);
    if (lane == 0) {
      warp_max[warp] = m;
      warp_sum[warp] = s;
      warp_thresh[warp] = warp_q;
    }
    __syncthreads();

    // ---- Threshold and compaction ----
    uint32_t thresh = 0xffffffffu;
#pragma unroll
    for (int w = 0; w < NUM_WORKER_WARPS; w++) {
      thresh = min(thresh, warp_thresh[w]);
    }
    int n_gt = 0, n_eq = 0;
    for (int i = tid; i < CHUNK_SIZE; i += WORKER_NUM_THREADS) {
      uint32_t const k = chunk_key[i];
      n_gt += k > thresh;
      n_eq += k == thresh && k != 0u;
    }
    int gt_off, eq_off, total_gt, total_eq;
    block_scan_pair(n_gt, n_eq, gt_off, eq_off, total_gt, total_eq, scratch);
    for (int i = tid; i < CHUNK_SIZE; i += WORKER_NUM_THREADS) {
      uint32_t const k = chunk_key[i];
      if (k > thresh) {
        gt_rank_key[gt_off++] = rank_key(k, i);
      } else if (k == thresh && k != 0u) {
        if (eq_off < TOPK_MAX) {
          eq_pos[eq_off] = i;
        }
        eq_off++;
      }
    }
    __syncthreads();

    // ---- Rank the candidates above the threshold; fill the rest ----
    for (int c = tid; c < total_gt; c += WORKER_NUM_THREADS) {
      uint64_t const mine = gt_rank_key[c];
      int rank = 0;
      for (int j = 0; j < total_gt; j++) {
        rank += gt_rank_key[j] > mine;
      }
      if (rank < TOPK_MAX) {
        int const pos = 0xffffffffu - static_cast<uint32_t>(mine);
        row_val[rank] = value_of(static_cast<uint32_t>(mine >> 32));
        row_idx[rank] = static_cast<long long>(chunk_start + pos);
      }
    }
    for (int e = tid; e < total_eq && total_gt + e < TOPK_MAX;
         e += WORKER_NUM_THREADS) {
      row_val[total_gt + e] = value_of(thresh);
      row_idx[total_gt + e] = static_cast<long long>(chunk_start + eq_pos[e]);
    }
    for (int r = total_gt + total_eq + tid; r < TOPK_MAX;
         r += WORKER_NUM_THREADS) {
      row_val[r] = -INFINITY;
      row_idx[r] = -1LL;
    }
    if (warp == 0) {
      float bm = lane < NUM_WORKER_WARPS ? warp_max[lane] : -INFINITY;
      float bs = lane < NUM_WORKER_WARPS ? warp_sum[lane] : 0.0f;
      warp_combine_max_sum(bm, bs);
      if (lane == 0) {
        row_val[TOPK_MAX] = bm;
        row_val[TOPK_MAX + 1] = bm == -INFINITY ? 0.0f : bs;
      }
    }
    // Keys and candidate buffers are reused by the next row.
    __syncthreads();
  }
}

/*
 * Reduce stage. Same inputs, output and semantics as
 * sampling_reduce_sm100_kernel: merge the per-chunk candidates, keep at most
 * top_k of them and stop once their probability mass reaches top_p, then draw
 * one with the Gumbel-Max trick (greedy picks the largest).
 */
template <typename T, int BATCH_SIZE, int NUM_PARTIAL_TASKS, int TOPK_MAX>
__device__ __forceinline__ void
    sampling_reduce_hopper_kernel(void const *__restrict__ input_val_ptr,
                                  void const *__restrict__ input_idx_ptr,
                                  void *__restrict__ final_output_ptr,
                                  int num_active_tokens,
                                  float top_p,
                                  int top_k,
                                  bool greedy,
                                  unsigned long long philox_seed,
                                  unsigned long long philox_fallback_offset,
                                  int const *steps,
                                  int const *request_ids,
                                  int const *qo_indptr) {
  using namespace sampling_hopper_detail;
  int constexpr NT = NUM_PARTIAL_TASKS;
  int constexpr LIST = TOPK_MAX + 2; // stride of one chunk's entries
  int constexpr NUM_CANDIDATES = NT * TOPK_MAX;
  int constexpr ROW_FLOATS = NT * LIST;
  int constexpr ROW_BYTES = ROW_FLOATS * sizeof(float);
  // Lists are dealt round-robin over the warps: list c -> warp c % 8.
  int constexpr MIN_LISTS_PER_WARP = NT / NUM_WORKER_WARPS;
  static_assert(TOPK_MAX > 0 && TOPK_MAX <= 128,
                "TOPK_MAX must be in (0, 128]");
  static_assert(NT >= NUM_WORKER_WARPS && NT <= WORKER_NUM_THREADS,
                "one per-chunk list per thread, every warp holding one");
  static_assert(NUM_CANDIDATES < 65536,
                "candidate counts are scanned as 16 bits");
  // barrier + small arrays, staging area, candidates > L
  int constexpr SMALL_BYTES = 1024;
  static_assert(SMALL_BYTES >= 16 + 4 * 4 * NUM_WORKER_WARPS + 3 * 4 * TOPK_MAX,
                "small arrays overflow their region");
  static_assert(SMALL_BYTES + (ROW_BYTES + 15) / 16 * 16 + 8 * NUM_CANDIDATES +
                        128 <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "sampling_reduce_hopper smem exceeds the dynamic budget");

  float const *__restrict__ partial_val =
      static_cast<float const *>(input_val_ptr);
  long long const *__restrict__ partial_idx =
      static_cast<long long const *>(input_idx_ptr);
  long long *__restrict__ final_output =
      static_cast<long long *>(final_output_ptr);
  int const idx_stride = NUM_CANDIDATES;

  extern __shared__ char smem[];
  char *base = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(smem) + 127) / 128 * 128);
  Barrier &bar = *reinterpret_cast<Barrier *>(base);
  uint32_t *scratch = reinterpret_cast<uint32_t *>(base + 16);
  float *warp_max = reinterpret_cast<float *>(scratch + NUM_WORKER_WARPS);
  float *warp_sum = warp_max + NUM_WORKER_WARPS;
  uint32_t *warp_thresh =
      reinterpret_cast<uint32_t *>(warp_sum + NUM_WORKER_WARPS);
  float *sel_val = reinterpret_cast<float *>(warp_thresh + NUM_WORKER_WARPS);
  int *sel_pos = reinterpret_cast<int *>(sel_val + TOPK_MAX);
  int *eq_pos = sel_pos + TOPK_MAX;
  float *stage = reinterpret_cast<float *>(base + SMALL_BYTES);
  uint64_t *gt_rank_key = reinterpret_cast<uint64_t *>(
      base + SMALL_BYTES + (ROW_BYTES + 15) / 16 * 16);

  int const tid = threadIdx.x;
  int const lane = tid % WARP;
  int const warp = tid / WARP;
  int const list = lane * NUM_WORKER_WARPS + warp;
  bool const has_list = list < NT;
  float const *my_list = stage + list * LIST;

  if (tid == 0) {
    initialize_barrier(bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  uint32_t phase = 0;

  int const k_limit =
      greedy ? 1 : ((top_k > 0 && top_k < TOPK_MAX) ? top_k : TOPK_MAX);
  // Threshold sample: the first `lead` entries of each list, enough for every
  // warp to hold at least q representatives.
  int const q = (k_limit + NUM_WORKER_WARPS - 1) / NUM_WORKER_WARPS;
  int const lead =
      min(TOPK_MAX, (q + MIN_LISTS_PER_WARP - 1) / MIN_LISTS_PER_WARP);

  for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
    // ---- Stage this row's candidate values and chunk stats ----
    float const *src = partial_val + (size_t)batch_idx * ROW_FLOATS;
    if (ROW_BYTES % 16 == 0 && (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
      if (tid == 0) {
        // Order earlier generic-proxy reads of the staging area before the
        // async-proxy write that replaces it.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        set_barrier_transaction_bytes(bar, ROW_BYTES);
        bulk_copy_g2s(stage, src, ROW_BYTES, bar);
      }
      wait(bar, phase);
      phase ^= 1;
    } else {
      for (int i = tid; i < ROW_FLOATS; i += WORKER_NUM_THREADS) {
        stage[i] = src[i];
      }
      __syncthreads();
    }

    // ---- Softmax normalizer and per-warp threshold ----
    float m = -INFINITY, s = 0.0f;
    if (has_list) {
      combine_max_sum(m, s, my_list[TOPK_MAX], my_list[TOPK_MAX + 1]);
    }
    warp_combine_max_sum(m, s);
    uint32_t warp_q;
    if (lead == 1) {
      warp_q = warp_qth_largest(has_list ? key_of(my_list[0]) : 0u, q);
    } else {
      // Bitwise search for the largest t with #(sample keys >= t) >= q.
      warp_q = 0;
      for (int bit = 31; bit >= 0; bit--) {
        uint32_t const cand = warp_q | (1u << bit);
        int cnt = 0;
        if (has_list) {
          for (int p = 0; p < lead && key_of(my_list[p]) >= cand; p++) {
            cnt++;
          }
        }
        if (__reduce_add_sync(0xffffffff, cnt) >= q) {
          warp_q = cand;
        }
      }
    }
    if (lane == 0) {
      warp_max[warp] = m;
      warp_sum[warp] = s;
      warp_thresh[warp] = warp_q;
    }
    __syncthreads();

    float gm = -INFINITY, gs = 0.0f;
    uint32_t thresh = 0xffffffffu;
#pragma unroll
    for (int w = 0; w < NUM_WORKER_WARPS; w++) {
      combine_max_sum(gm, gs, warp_max[w], warp_sum[w]);
      thresh = min(thresh, warp_thresh[w]);
    }

    // ---- Compaction: lists are sorted, so only their heads qualify ----
    int n_gt = 0, n_eq = 0;
    if (has_list) {
      for (int p = 0; p < TOPK_MAX; p++) {
        uint32_t const k = key_of(my_list[p]);
        if (k < thresh || k == 0u) {
          break;
        }
        n_gt += k > thresh;
        n_eq += k == thresh;
      }
    }
    int gt_off, eq_off, total_gt, total_eq;
    block_scan_pair(n_gt, n_eq, gt_off, eq_off, total_gt, total_eq, scratch);
    for (int p = 0; p < n_gt + n_eq; p++) {
      uint32_t const k = key_of(my_list[p]);
      int const pos = list * TOPK_MAX + p;
      if (k > thresh) {
        gt_rank_key[gt_off++] = rank_key(k, pos);
      } else {
        if (eq_off < k_limit) {
          eq_pos[eq_off] = pos;
        }
        eq_off++;
      }
    }
    __syncthreads();

    // ---- Rank into descending order, at most k_limit slots ----
    for (int c = tid; c < total_gt; c += WORKER_NUM_THREADS) {
      uint64_t const mine = gt_rank_key[c];
      int rank = 0;
      for (int j = 0; j < total_gt; j++) {
        rank += gt_rank_key[j] > mine;
      }
      if (rank < k_limit) {
        sel_val[rank] = value_of(static_cast<uint32_t>(mine >> 32));
        sel_pos[rank] = 0xffffffffu - static_cast<uint32_t>(mine);
      }
    }
    for (int e = tid; e < total_eq && total_gt + e < k_limit;
         e += WORKER_NUM_THREADS) {
      sel_val[total_gt + e] = value_of(thresh);
      sel_pos[total_gt + e] = eq_pos[e];
    }
    int const num_cand = min(k_limit, total_gt + total_eq);
    __syncthreads();

    // ---- Warp 0: top-p cut and draw ----
    if (warp == 0) {
      // Smallest prefix whose probability mass reaches top_p.
      int num_selected = num_cand;
      if (top_p < 1.0f) {
        float const target = top_p * gs;
        float carry = 0.0f;
        for (int b = 0; b < num_cand; b += WARP) {
          int const r = b + lane;
          float cum = r < num_cand ? __expf(sel_val[r] - gm) : 0.0f;
#pragma unroll
          for (int d = 1; d < WARP; d *= 2) {
            float const o = __shfl_up_sync(0xffffffff, cum, d);
            cum += lane >= d ? o : 0.0f;
          }
          cum += carry;
          unsigned const hit =
              __ballot_sync(0xffffffff, r < num_cand && cum >= target);
          if (hit != 0u) {
            num_selected = b + __ffs(hit);
            break;
          }
          carry = __shfl_sync(0xffffffff, cum, WARP - 1);
        }
      }

      long long token = -1;
      if (num_selected > 0) {
        long long const *row_idx = partial_idx + (size_t)batch_idx * idx_stride;
        if (greedy) {
          token = row_idx[sel_pos[0]];
        } else {
          unsigned long long const row_offset = warp_philox_offset_for_row(
              batch_idx, steps, request_ids, qo_indptr, philox_fallback_offset);
          float best = -INFINITY;
          int best_r = -1;
          long long best_tok = -1;
          for (int r = lane; r < num_selected; r += WARP) {
            long long const tok = row_idx[sel_pos[r]];
            float const noisy =
                sel_val[r] + sampling_gumbel_noise(philox_seed,
                                                   row_offset,
                                                   static_cast<uint64_t>(tok));
            if (noisy > best) {
              best = noisy;
              best_r = r;
              best_tok = tok;
            }
          }
#pragma unroll
          for (int mask = WARP / 2; mask > 0; mask /= 2) {
            float const ob = __shfl_xor_sync(0xffffffff, best, mask);
            int const orr = __shfl_xor_sync(0xffffffff, best_r, mask);
            long long const ot = __shfl_xor_sync(0xffffffff, best_tok, mask);
            if (ob > best ||
                (ob == best && orr >= 0 && (best_r < 0 || orr < best_r))) {
              best = ob;
              best_r = orr;
              best_tok = ot;
            }
          }
          token = best_tok;
        }
      }
      if (lane == 0) {
        final_output[batch_idx] = token;
      }
    }
    // The staging area and candidate buffers are reused by the next row.
    __syncthreads();
  }
  if (tid == 0) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(&bar)))
                 : "memory");
  }
}

} // namespace kernel
