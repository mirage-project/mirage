/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Attention planner (FlashInfer-style) for SM100.
//
// Reads the current iteration's qo_indptr_buffer / paged_kv_indptr_buffer
// (from runtime_config) and produces a per-bucket work list. Each request is
// classified into PREFILL (packed_qo > PREFILL_THRESHOLD) or DECODE
// (packed_qo ≤ threshold) and tiled along its packed-Q axis at P_Q_TILE /
// D_Q_TILE packed rows respectively. Each (qo_tile, kv_head) pair becomes
// one work item.
//
// Outputs (all int32 device tensors, produced fresh every iter):
//   plan_prefill_indptr[NUM_BUCKETS+1]  : prefill works in bucket b live at
//       worker_*[plan_prefill_indptr[b] : plan_prefill_indptr[b+1])
//   plan_decode_indptr[NUM_BUCKETS+1]   : decode works likewise
//   worker_batch_indices[MAX_WORKS]     : per-work batch (request) index
//   worker_kv_head_indices[MAX_WORKS]   : per-work KV head index
//   worker_qo_tile_indices[MAX_WORKS]   : per-work qo-tile index within request
//   worker_kv_tile_indices[MAX_WORKS]   : per-work KV chunk index within request
//
// Work id layout: prefill works occupy indices [0, P), decode works occupy
// [P, P+D). plan_prefill_indptr values fall in [0, P]; plan_decode_indptr
// values fall in [P, P+D].
//
// Bucket assignment: GREEDY MIN-COST. For each work, find the bucket with the
// lowest running cost and assign the work there. The current heuristic is
// 2*q_tile_size + kv_len, which performed better for Mirage's decode-heavy
// full-model runs than the pure mma-count model.
// Prefill and decode have separate cost arrays since they run on different
// consumer kernels. This matters when:
//   * The number of works of one type exceeds NUM_BUCKETS — earlier works
//     stack into low-numbered buckets, later works pick least-loaded.
//   * Requests have different KV lengths — greedy spreads heavy work across
//     buckets instead of leaving some idle while one is overloaded.
//
// Execution model: SINGLE THREAD. attention_planner_sm100_core() runs the
// whole plan on one thread (no __syncthreads, no atomics) in three logical
// phases — (1) greedy bucket assignment in shared memory, (2) prefix-sum the
// bucket counts into plan_*_indptr, (3) scatter the per-work attributes into
// worker_* global arrays. The scheduler invokes it directly from its
// per-iteration prepare step (which owns a single thread and cannot
// __syncthreads); see mpk_run_attention_planner() in persistent_kernel.cuh.
// The greedy assignment is inherently sequential, so a multi-thread scatter
// bought little; single-thread keeps it simple and lets the planner live on
// the scheduler thread instead of a dedicated worker CTA.
//
// Hard-wired causal=true.

#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

template <int NUM_BUCKETS>
__device__ __forceinline__ bool
    attention_planner_heap_less(int lhs_bucket,
                                int rhs_bucket,
                                long long const *bucket_costs) {
  long long lhs_cost = bucket_costs[lhs_bucket];
  long long rhs_cost = bucket_costs[rhs_bucket];
  return lhs_cost < rhs_cost ||
         (lhs_cost == rhs_cost && lhs_bucket < rhs_bucket);
}

template <int NUM_BUCKETS>
__device__ __forceinline__ void
    attention_planner_heap_sift_down(int *heap,
                                     long long const *bucket_costs) {
  int idx = 0;
  while (true) {
    int left = idx * 2 + 1;
    int right = left + 1;
    int best = idx;
    if (left < NUM_BUCKETS &&
        attention_planner_heap_less<NUM_BUCKETS>(
            heap[left], heap[best], bucket_costs)) {
      best = left;
    }
    if (right < NUM_BUCKETS &&
        attention_planner_heap_less<NUM_BUCKETS>(
            heap[right], heap[best], bucket_costs)) {
      best = right;
    }
    if (best == idx) {
      break;
    }
    int tmp = heap[idx];
    heap[idx] = heap[best];
    heap[best] = tmp;
    idx = best;
  }
}

template <int NUM_BUCKETS>
__device__ __forceinline__ void
    attention_planner_heapify(int *heap, long long const *bucket_costs) {
  for (int start = NUM_BUCKETS / 2 - 1; start >= 0; start--) {
    int idx = start;
    while (true) {
      int left = idx * 2 + 1;
      int right = left + 1;
      int best = idx;
      if (left < NUM_BUCKETS &&
          attention_planner_heap_less<NUM_BUCKETS>(
              heap[left], heap[best], bucket_costs)) {
        best = left;
      }
      if (right < NUM_BUCKETS &&
          attention_planner_heap_less<NUM_BUCKETS>(
              heap[right], heap[best], bucket_costs)) {
        best = right;
      }
      if (best == idx) {
        break;
      }
      int tmp = heap[idx];
      heap[idx] = heap[best];
      heap[best] = tmp;
      idx = best;
    }
  }
}

// Per-work cost for the greedy min-cost bucket assignment. `q_tokens` is the
// packed-Q rows the work covers, `kv_len` is the KV span it scans (a chunk
// length for split works, the full kv_len otherwise). Only the relative
// magnitude matters (greedy compares running per-bucket sums); 2*q + kv weights
// the per-row Q work against the KV traffic and balances Mirage's decode-heavy
// full-model batches well.
__device__ __forceinline__ long long
attention_planner_work_cost(long long q_tokens, long long kv_len) {
  return 2ll * q_tokens + kv_len;
}

// Flat plan buffer layout (one contiguous int32 array):
//   [0 .. NUM_BUCKETS]            : plan_prefill_indptr
//   [NUM_BUCKETS+1 .. 2*NUM_BUCKETS+1] : plan_decode_indptr
//   [2*(NUM_BUCKETS+1) .. + MAX_WORKS) : worker_batch_indices
//   [+ MAX_WORKS .. + MAX_WORKS)        : worker_kv_head_indices
//   [+ MAX_WORKS .. + MAX_WORKS)        : worker_qo_tile_indices
//   [+ MAX_WORKS .. + MAX_WORKS)        : worker_kv_tile_indices
//   [+ MAX_WORKS .. + MAX_WORKS)        : worker_kv_tile_indices
//   [+ 1)                               : planned producer work counter
//   [+ 1)                               : planned merge work counter
// Total working size: 2*(NUM_BUCKETS+1) + 4*MAX_WORKS + 2 ints.

// PREFILL_THRESHOLD: packed_qo strictly greater than this value classifies
// the request as PREFILL. Lower thresholds push more requests into the
// prefill list.
template <int NUM_KV_HEADS,
          int GQA_GROUP,
          int NUM_BUCKETS,
          int MAX_WORKS,
          int P_Q_TILE = 128,
          int D_Q_TILE = 16,
          int PREFILL_THRESHOLD = 16,
          int KV_SPLIT_SIZE = 0,
          int PAGE_SIZE_PARAM = MPK_PAGE_SIZE>
__device__ __forceinline__ void attention_planner_sm100_core(
    int *plan_buffer,
    int const *qo_indptr_buffer,
    int const *paged_kv_indptr_buffer,
    int const *paged_kv_last_page_len_buffer,
    int num_requests) {
  // SINGLE-THREAD planner. Everything below runs on exactly one thread — the
  // caller is responsible for ensuring only one thread enters (the worker-task
  // wrapper guards with threadIdx.x==0; the scheduler calls this from the one
  // thread that owns prepare_next_batch). There is therefore NO __syncthreads()
  // and NO atomics here: __syncthreads() is illegal in the scheduler context,
  // and a lone thread needs neither. Output is byte-for-byte identical to the
  // old 3-phase multi-thread version.
  int *plan_prefill_indptr = plan_buffer;
  int *plan_decode_indptr = plan_buffer + (NUM_BUCKETS + 1);
  int *worker_batch_indices = plan_buffer + 2 * (NUM_BUCKETS + 1);
  int *worker_kv_head_indices = worker_batch_indices + MAX_WORKS;
  int *worker_qo_tile_indices = worker_kv_head_indices + MAX_WORKS;
  int *worker_kv_tile_indices = worker_qo_tile_indices + MAX_WORKS;

  // Per-work assignments produced by phase 1 and consumed by phase 3, all on
  // the same single thread. MAX_WORKS=1024 → ~9KB of shared memory.
  __shared__ int16_t s_work_bucket[MAX_WORKS];
  __shared__ int16_t s_work_batch[MAX_WORKS];
  __shared__ int16_t s_work_kv_head[MAX_WORKS];
  __shared__ int16_t s_work_qo_tile[MAX_WORKS];
  __shared__ int16_t s_work_kv_tile[MAX_WORKS];
  __shared__ int8_t s_work_is_prefill[MAX_WORKS];
  __shared__ int s_total_works;

  // Greedy state. Same fast-path as before: when num_assigned < NUM_BUCKETS,
  // every untouched bucket has cost 0, so greedy degenerates to round-robin
  // (work k → bucket k). Skip the O(NUM_BUCKETS) argmin scan in that regime.
  __shared__ long long bucket_cost_prefill[NUM_BUCKETS];
  __shared__ int bucket_works_prefill[NUM_BUCKETS];
  __shared__ long long bucket_cost_decode[NUM_BUCKETS];
  __shared__ int bucket_works_decode[NUM_BUCKETS];
  __shared__ int bucket_heap_prefill[NUM_BUCKETS];
  __shared__ int bucket_heap_decode[NUM_BUCKETS];

  for (int b = 0; b < NUM_BUCKETS; b++) {
    bucket_cost_prefill[b] = 0;
    bucket_works_prefill[b] = 0;
    bucket_cost_decode[b] = 0;
    bucket_works_decode[b] = 0;
    bucket_heap_prefill[b] = b;
    bucket_heap_decode[b] = b;
  }

  // ---------------------------------------------------------------
  // Phase 1: serial greedy assignment.
  //
  // The greedy decisions are sequentially dependent (each assignment changes
  // the cost vector that the next argmin reads), so this phase is serial.
  // It only does shared-memory accesses — no global writes — so it's ~1µs for
  // typical workloads. Per-work assignments are saved to shared memory for
  // phase 3 to consume.
  // ---------------------------------------------------------------
  // Hoisted out of Phase 1 so the merge split-flags slot (written at the plan
  // tail in Phase 3) can record the decode occupancy-gate decision.
  bool g_decode_split_on = true;
  {
    int num_assigned_prefill = 0;
    int num_assigned_decode = 0;
    bool heap_built_prefill = false;
    bool heap_built_decode = false;
    int w = 0;
    // Decode split-KV occupancy gate (FlashInfer-style). A long-context decode
    // request (kv_len > KV_SPLIT_SIZE) can be tiled into ceil(kv_len/split)
    // chunk works so one heavy request is balanced across many buckets instead
    // of pinning a single straggler worker. But Mirage's decode parallelism is
    // capped at NUM_BUCKETS (one consumer CTA per bucket), so once the base
    // (unsplit) decode work count already fills the buckets, splitting only adds
    // merge + redundant-Q work onto an already-full machine (measured net loss).
    // Gate: only split decode when the unsplit decode works underfill the
    // buckets. The merge (merge_splitkv.cuh) reads this decision via the plan
    // split-flags slot, so it skips a request the gate left unsplit.
    bool decode_split_ok = true;
    {
      int base_decode_q_tiles = 0;
      for (int r = 0; r < num_requests; r++) {
        int qol = qo_indptr_buffer[r + 1] - qo_indptr_buffer[r];
        if (qol <= 0) {
          continue;
        }
        int pq = qol * GQA_GROUP;
        if (pq > PREFILL_THRESHOLD) {
          continue; // prefill request — not a decode work
        }
        base_decode_q_tiles += (pq + D_Q_TILE - 1) / D_Q_TILE;
      }
      int base_decode_works = base_decode_q_tiles * NUM_KV_HEADS;
      if (base_decode_works >= NUM_BUCKETS) {
        decode_split_ok = false;
      }
    }
    g_decode_split_on = decode_split_ok;
    for (int request_iter = 0; request_iter < num_requests; request_iter++) {
      int r = request_iter;
      int qo_len = qo_indptr_buffer[r + 1] - qo_indptr_buffer[r];
      if (qo_len <= 0) {
        continue;
      }
      int packed_qo = qo_len * GQA_GROUP;
      bool is_prefill = packed_qo > PREFILL_THRESHOLD;
      int q_tile = is_prefill ? P_Q_TILE : D_Q_TILE;
      int num_qo_tiles = (packed_qo + q_tile - 1) / q_tile;

      int num_pages =
          paged_kv_indptr_buffer[r + 1] - paged_kv_indptr_buffer[r];
      int kv_len = (num_pages - 1) * PAGE_SIZE_PARAM +
                   paged_kv_last_page_len_buffer[r];
      int history_len = kv_len - qo_len;
      // Split-KV applies to prefill always, and to decode when the occupancy
      // gate (decode_split_ok, computed above) leaves it on.
      bool split_kv_request =
          KV_SPLIT_SIZE > 0 && kv_len > KV_SPLIT_SIZE &&
          (is_prefill || decode_split_ok);
      int num_kv_tiles = split_kv_request
                             ? (kv_len + KV_SPLIT_SIZE - 1) / KV_SPLIT_SIZE
                             : 1;
      num_kv_tiles = max(num_kv_tiles, 1);
      // Select the bucket-accounting arrays for this request's work class;
      // prefill and decode use separate consumers and separate cost vectors.
      long long *sel_cost = is_prefill ? bucket_cost_prefill : bucket_cost_decode;
      int *sel_works = is_prefill ? bucket_works_prefill : bucket_works_decode;
      int *sel_heap = is_prefill ? bucket_heap_prefill : bucket_heap_decode;
      int *sel_nassigned =
          is_prefill ? &num_assigned_prefill : &num_assigned_decode;
      bool *sel_hbuilt = is_prefill ? &heap_built_prefill : &heap_built_decode;

      for (int h = 0; h < NUM_KV_HEADS; h++) {
        if (w >= MAX_WORKS) {
          break;
        }
        for (int qt = 0; qt < num_qo_tiles; qt++) {
          if (w >= MAX_WORKS) {
            break;
          }
          int qtile_start = qt;
          int q_work_tokens = min(q_tile, packed_qo - qt * q_tile);
          if (split_kv_request) {
            int qtile_token_start = (qt * q_tile) / GQA_GROUP;
            int qtile_tokens =
                (q_work_tokens + GQA_GROUP - 1) / GQA_GROUP;
            int qtile_full_seq_len =
                history_len + qtile_token_start + qtile_tokens;
            int kv_tiles_for_q_tile =
                max(1, (qtile_full_seq_len + KV_SPLIT_SIZE - 1) /
                           KV_SPLIT_SIZE);
            kv_tiles_for_q_tile = min(kv_tiles_for_q_tile, num_kv_tiles);
            for (int kt = 0; kt < kv_tiles_for_q_tile && w < MAX_WORKS; kt++) {
              int min_b;
              int chunk_start = kt * KV_SPLIT_SIZE;
              int chunk_len =
                  min(KV_SPLIT_SIZE,
                      max(0, qtile_full_seq_len - chunk_start));
              long long cost = attention_planner_work_cost(
                  (long long)q_work_tokens, (long long)chunk_len);
              if (*sel_nassigned < NUM_BUCKETS) {
                min_b = *sel_nassigned;
                sel_cost[min_b] += cost;
                sel_works[min_b] += 1;
                *sel_nassigned += 1;
              } else {
                if (!*sel_hbuilt) {
                  attention_planner_heapify<NUM_BUCKETS>(sel_heap, sel_cost);
                  *sel_hbuilt = true;
                }
                min_b = sel_heap[0];
                sel_cost[min_b] += cost;
                sel_works[min_b] += 1;
                attention_planner_heap_sift_down<NUM_BUCKETS>(sel_heap, sel_cost);
                *sel_nassigned += 1;
              }
              s_work_bucket[w] = (int16_t)min_b;
              s_work_batch[w] = (int16_t)r;
              s_work_kv_head[w] = (int16_t)h;
              s_work_qo_tile[w] = (int16_t)qtile_start;
              s_work_kv_tile[w] = (int16_t)kt;
              s_work_is_prefill[w] = (int8_t)is_prefill;
              w++;
            }
            continue;
          }
          int kv_tiles_to_emit =
              KV_SPLIT_SIZE > 0 ? min(num_kv_tiles, MAX_WORKS - w) : 1;
          int works_to_emit = kv_tiles_to_emit;
          int min_b;
          long long cost = attention_planner_work_cost(
              (long long)q_work_tokens, (long long)kv_len);
          // Non-split prefill emits one independently scheduled work per Q
          // tile. Decode keeps the legacy request/head grouping because the
          // decode consumer currently processes the whole request for a KV
          // head and does not consume qo_tile.
          if (!is_prefill) {
            works_to_emit = min(num_qo_tiles - qt, MAX_WORKS - w);
            qt += works_to_emit - 1;
            cost *= works_to_emit;
          }
          if (is_prefill) {
            if (num_assigned_prefill < NUM_BUCKETS) {
              min_b = num_assigned_prefill;
              bucket_cost_prefill[min_b] += cost;
              bucket_works_prefill[min_b] += works_to_emit;
              num_assigned_prefill += works_to_emit;
            } else {
              if (!heap_built_prefill) {
                attention_planner_heapify<NUM_BUCKETS>(
                    bucket_heap_prefill, bucket_cost_prefill);
                heap_built_prefill = true;
              }
              min_b = bucket_heap_prefill[0];
              bucket_cost_prefill[min_b] += cost;
              bucket_works_prefill[min_b] += works_to_emit;
              attention_planner_heap_sift_down<NUM_BUCKETS>(
                  bucket_heap_prefill, bucket_cost_prefill);
              num_assigned_prefill += works_to_emit;
            }
          } else {
            if (num_assigned_decode < NUM_BUCKETS) {
              min_b = num_assigned_decode;
              bucket_cost_decode[min_b] += cost;
              bucket_works_decode[min_b] += works_to_emit;
              num_assigned_decode += works_to_emit;
            } else {
              if (!heap_built_decode) {
                attention_planner_heapify<NUM_BUCKETS>(
                    bucket_heap_decode, bucket_cost_decode);
                heap_built_decode = true;
              }
              min_b = bucket_heap_decode[0];
              bucket_cost_decode[min_b] += cost;
              bucket_works_decode[min_b] += works_to_emit;
              attention_planner_heap_sift_down<NUM_BUCKETS>(
                  bucket_heap_decode, bucket_cost_decode);
              num_assigned_decode += works_to_emit;
            }
          }
          for (int kt = 0; kt < works_to_emit; kt++) {
            s_work_bucket[w] = (int16_t)min_b;
            s_work_batch[w] = (int16_t)r;
            s_work_kv_head[w] = (int16_t)h;
            s_work_qo_tile[w] =
                (int16_t)(qtile_start + kt);
            s_work_kv_tile[w] = (int16_t)(KV_SPLIT_SIZE > 0 ? -1 : 0);
            s_work_is_prefill[w] = (int8_t)is_prefill;
            w++;
          }
        }
      }
    }
    s_total_works = w;
  }

  // ---------------------------------------------------------------
  // Phase 2: prefix-sum bucket counts into indptrs, written straight to global
  // plan_buffer. Reset bucket_works_* to 0 so phase 3 can use them as
  // per-bucket scatter cursors.
  // ---------------------------------------------------------------
  {
    plan_prefill_indptr[0] = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
      plan_prefill_indptr[b + 1] =
          plan_prefill_indptr[b] + bucket_works_prefill[b];
      bucket_works_prefill[b] = 0;
    }
    int prefill_total = plan_prefill_indptr[NUM_BUCKETS];
    plan_decode_indptr[0] = prefill_total;
    for (int b = 0; b < NUM_BUCKETS; b++) {
      plan_decode_indptr[b + 1] =
          plan_decode_indptr[b] + bucket_works_decode[b];
      bucket_works_decode[b] = 0;
    }
  }

  // ---------------------------------------------------------------
  // Phase 3: serial write of worker_* arrays to global mem. Each work claims
  // the next slot within its bucket's slice of worker_* (plain increment — no
  // atomics needed in single-thread mode). Order within a bucket doesn't
  // matter — the consumer iterates all works assigned to its bucket.
  // ---------------------------------------------------------------
  int total_works = s_total_works;
  for (int w = 0; w < total_works; w++) {
    int bucket = s_work_bucket[w];
    bool is_prefill = (bool)s_work_is_prefill[w];
    int slot;
    if (is_prefill) {
      int off = bucket_works_prefill[bucket]++;
      slot = plan_prefill_indptr[bucket] + off;
    } else {
      int off = bucket_works_decode[bucket]++;
      slot = plan_decode_indptr[bucket] + off;
    }
    if (slot < MAX_WORKS) {
      worker_batch_indices[slot] = s_work_batch[w];
      worker_kv_head_indices[slot] = s_work_kv_head[w];
      worker_qo_tile_indices[slot] = s_work_qo_tile[w];
      worker_kv_tile_indices[slot] = s_work_kv_tile[w];
    }
  }

  constexpr int PLAN_SIZE = 2 * (NUM_BUCKETS + 1) + 4 * MAX_WORKS;
  // Zero the legacy producer/merge counters (+0,+1) AND the per-layer merge
  // work-claim counters (+2..+2+MAX_MERGE_COUNTERS). Each layer's split-KV
  // merge claims rows via atomicAdd on its OWN counter slot; a single shared
  // counter is only consumed by the first layer per iteration (never reset
  // between layers), which left layers 2..N unmerged.
  constexpr int MAX_MERGE_COUNTERS = 64;
  for (int i = 0; i < 2 + MAX_MERGE_COUNTERS; i++) {
    plan_buffer[PLAN_SIZE + i] = 0;
  }
  // Split-flags slot (one int) right after the merge counters. Records the
  // decode occupancy-gate decision so the merge knows whether a long
  // (seq > chunk) decode request was actually split into partials (needs
  // merging) or left unsplit by the gate and written straight to final (must be
  // skipped). A compile-time is_unsplit (seq <= chunk) check can't capture this
  // runtime, batch-dependent decision. bit0 = decode split on. (Prefill always
  // splits, so the merge keys prefill purely on is_unsplit.)
  constexpr int SPLIT_FLAGS_SLOT = PLAN_SIZE + 2 + MAX_MERGE_COUNTERS;
  plan_buffer[SPLIT_FLAGS_SLOT] = (g_decode_split_on ? 1 : 0);
  // Make every global store (worker_* arrays, indptrs, counters) visible before
  // the caller releases the event/launch that lets consumers read the plan.
  __threadfence();
}

// Helper: read the highest active bucket index from plan_buffer's indptrs.
// A bucket b is "active" if it has at least one prefill OR decode work.
// Buckets > num_active are guaranteed empty; consumers can skip them with
// a single load. This is optimization #5 from issue #627's perf follow-up.
template <int NUM_BUCKETS>
__device__ __forceinline__ bool attention_planner_bucket_has_work(
    int const *plan_buffer, int bucket_idx) {
  int const *plan_prefill_indptr = plan_buffer;
  int const *plan_decode_indptr = plan_buffer + (NUM_BUCKETS + 1);
  int p_works = plan_prefill_indptr[bucket_idx + 1] -
                plan_prefill_indptr[bucket_idx];
  int d_works = plan_decode_indptr[bucket_idx + 1] -
                plan_decode_indptr[bucket_idx];
  return (p_works + d_works) > 0;
}

} // namespace kernel
