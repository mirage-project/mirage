// tasks/sm90_attention_split.cuh -- the sm90_attention_split task type: MPK's
// Hopper paged attention with the KV split in chunks (its PARTITION_KV path)
// as a pool: one task per SM; the SMs claim the step's units from a queue, a
// unit = request r, KV head h, KV positions [c CHUNK, c CHUNK + CHUNK) -- its
// part of the output (normalized over the chunk) and the chunk's
// log2-sum-exp, for sm90_attention_merge to combine (with the sinks). Full
// attention only (no window). Only the chunks a request's sequence reaches are
// units (at a decode position the first few of C): which SM runs which is the
// queue's, not the plan's, so the populated chunks spread over the SMs
// whatever the lengths. The chunk with the new tokens appends their K / V.
#pragma once
#include "sm90_attention.cuh"

namespace static_mk {

// PARAMS {NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, PAGE_SIZE, CHUNK,
// C (chunks), QKV_BIAS, R (requests)}; BUF_SLOTS {qkv, k cache, v cache, cos,
// sin, qo_indptr, kv_indptr, kv_indices, last_page_len (sm90_attention's), the
// qkv bias and qkv's bf16 rows (QKV_BIAS, as sm90_attention's; else -1, -1),
// parts: the chunks' outputs [C, T, Q x D] bf16, then their log2-sum-exps
// [T, C, Q] fp32 (one buffer: the node's one output)}. Counters: the node's,
// then (a line on) the queue.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_attention_split(Maps const &,
                                                         G const &g,
                                                         KernelLocals &L,
                                                         StaticTask const &) {
  constexpr int QH = PARAMS::v[0], KVH = PARAMS::v[1], D = PARAMS::v[2],
                PAGE = PARAMS::v[4], CHUNK = PARAMS::v[5], C = PARAMS::v[6],
                R = PARAMS::v[8];
  constexpr bool QKV_BIAS = PARAMS::v[7] == 1;
  constexpr int QPK = QH / KVH, QUEUE = SELF::counter + 32;
  static_assert(R * C <= 65536, "sm90_attention_split: (request, chunk) ids");
  int const *qo = buf_at<int const>(g, slot_at<BUF_SLOTS, 5>());
  int const *kv_indptr = buf_at<int const>(g, slot_at<BUF_SLOTS, 6>());
  int const *last_page_len = buf_at<int const>(g, slot_at<BUF_SLOTS, 8>());
  // the step's (request, chunk) pairs, chunk-major (the full chunks of every
  // request before the later ones), from the page tables (graph inputs: no
  // producer to wait for); unit u = pair u / KVH, KV head u % KVH
  __shared__ uint16_t pairs[R * C];
  __shared__ int n_pairs, claim[2];
  __syncthreads(); // this SM's earlier tasks are done with shared memory
  if (threadIdx.x == 0) {
    claim[0] = atomicAdd(g.cnt + QUEUE, 1u);
    int nc[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
      int const seq =
          (kv_indptr[r + 1] - kv_indptr[r] - 1) * PAGE + last_page_len[r];
      nc[r] = qo[r + 1] > qo[r] && seq > 0 ? (seq + CHUNK - 1) / CHUNK : 0;
    }
    int n = 0;
    for (int c = 0; c < C; c++) {
#pragma unroll
      for (int r = 0; r < R; r++) {
        if (c < nc[r]) {
          pairs[n++] = (uint16_t)(r * C + c);
        }
      }
    }
    n_pairs = n;
  }
  __syncthreads();
  int const units = n_pairs * KVH;
  kernel::bfloat16 *qkv =
      QKV_BIAS ? buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 10>())
               : buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 0>());
  kernel::bfloat16 *parts =
      buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 11>());
  bool waited = false;
  for (int k = 0;; k++) {
    int const u = claim[k & 1];
    if (u >= units) {
      break;
    }
    int const pr = pairs[u / KVH], request = pr / C, chunk = pr % C,
              head = u % KVH;
    // the page table and the first tile's earlier positions before the wait
    // for the QKV GEMM (once; later units find it done), q's staging after
    auto pre_q = [&]() {
      if (!waited) {
        if (threadIdx.x == 0) {
          sm90_wait_inputs<~0u, IN...>(g);
        }
        __syncthreads();
#ifdef STATIC_TIMING_BUILD
        if (threadIdx.x == 0 && L.stage_stamps) {
          L.stage_stamps[0] = gtime(); // the QKV GEMM's rows ready
        }
#endif
        waited = true;
      }
      if (threadIdx.x == 0) {
        claim[(k + 1) & 1] = atomicAdd(g.cnt + QUEUE, 1u); // the next unit
      }
      if constexpr (QKV_BIAS) {
        sm90_qkv_stage<BUF_SLOTS, 9, QH, KVH, D>(g, request, head);
      }
    };
    sm90_paged_attention<BUF_SLOTS,
                         QH,
                         KVH,
                         D,
                         PARAMS::v[3],
                         PAGE,
                         CHUNK, // SEQ_LEN: a chunk
                         true,
                         C,
                         0, // no window
                         R>(
        g,
        qkv,
        request,
        head,
        parts + (size_t)chunk * T * QH * D + head * QPK * D,
        reinterpret_cast<float *>(parts + (size_t)C * T * QH * D) +
            ((size_t)qo[request] * C + chunk) * QH +
            head * QPK, // the body offsets the output to the request's
                        // first token, not the log2-sum-exps
        chunk,
        nullptr,
        pre_q);
    __syncthreads(); // the body done with shared memory; claim[] written
  }
  if (!waited) { // no unit: the node still ends after its inputs
    if (threadIdx.x == 0) {
      sm90_wait_inputs<~0u, IN...>(g);
    }
    __syncthreads();
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_attention_split, SM90_BODY_SMEM)

} // namespace static_mk
