// tasks/sm90_attention_split.cuh -- the sm90_attention_split task type: MPK's
// Hopper paged attention with the KV split in chunks (its PARTITION_KV path):
// task (x, y, z) of grid (R, KVH, C) = request x, KV head y, KV positions
// [z CHUNK, z CHUNK + CHUNK) -- its part of the output (normalized over the
// chunk) and the chunk's log2-sum-exp, for sm90_attention_merge to combine
// (with the sinks). Full attention only (no window); a chunk past the sequence
// does nothing. The chunk with the new tokens appends their K / V.
#pragma once
#include "sm90_attention.cuh"

namespace static_mk {

// PARAMS {NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, PAGE_SIZE, CHUNK,
// C (chunks), QKV_BIAS}; BUF_SLOTS {qkv, k cache, v cache, cos, sin,
// qo_indptr, kv_indptr, kv_indices, last_page_len (sm90_attention's), the qkv
// bias and qkv's bf16 rows (QKV_BIAS, as sm90_attention's; else -1, -1),
// parts: the chunks' outputs [C, T, Q x D] bf16, then their log2-sum-exps
// [T, C, Q] fp32 (one buffer: the node's one output)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_attention_split(Maps const &,
                                                         G const &g,
                                                         KernelLocals &,
                                                         StaticTask const &tk) {
  constexpr int QH = PARAMS::v[0], KVH = PARAMS::v[1], D = PARAMS::v[2],
                C = PARAMS::v[6];
  constexpr bool QKV_BIAS = PARAMS::v[7] == 1;
  constexpr int QPK = QH / KVH;
  int const head = tk.y, chunk = tk.z;
  // as sm90_attention's: the page table and the first tile's earlier
  // positions before the wait for the QKV GEMM (a chunk past the sequence
  // returns before the hook and waits below)
  __syncthreads();
  kernel::bfloat16 *qkv =
      QKV_BIAS ? buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 10>())
               : buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 0>());
  bool waited = false;
  auto pre_q = [&]() {
    if (threadIdx.x == 0) {
      sm90_wait_inputs<~0u, IN...>(g);
    }
    __syncthreads();
    waited = true;
    if constexpr (QKV_BIAS) {
      sm90_qkv_stage<BUF_SLOTS, 9, QH, KVH, D>(g, tk.x, head);
    }
  };
  kernel::bfloat16 *parts =
      buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 11>());
  sm90_paged_attention<BUF_SLOTS,
                       QH,
                       KVH,
                       D,
                       PARAMS::v[3],
                       PARAMS::v[4],
                       PARAMS::v[5], // SEQ_LEN: a chunk
                       true,
                       C,
                       0,        // no window
                       SELF::x>( // the requests
      g,
      qkv,
      tk.x,
      head,
      parts + (size_t)chunk * T * QH * D + head * QPK * D,
      reinterpret_cast<float *>(parts + (size_t)C * T * QH * D) +
          ((size_t)buf_at<int const>(g, slot_at<BUF_SLOTS, 5>())[tk.x] * C +
           chunk) *
              QH +
          head * QPK, // the body offsets the output to the request's first
                      // token, not the log2-sum-exps
      chunk,
      nullptr,
      pre_q);
  if (!waited) {
    if (threadIdx.x == 0) {
      sm90_wait_inputs<~0u, IN...>(g);
    }
    __syncthreads();
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_attention_split, SM90_BODY_SMEM)

} // namespace static_mk
