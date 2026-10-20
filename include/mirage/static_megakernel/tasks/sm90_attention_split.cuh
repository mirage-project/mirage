// tasks/sm90_attention_split.cuh -- the sm90_attention_split task type: MPK's
// Hopper paged attention with the KV split in chunks (its PARTITION_KV path):
// task (x, y, z) of grid (R, KVH, C) = request x, KV head y, KV positions
// [z CHUNK, z CHUNK + CHUNK) -- its part of the output (normalized over the
// chunk) and the chunk's log2-sum-exp, for sm90_attention_merge to combine
// (with the sinks). Full attention only (no window); a chunk past the sequence
// does nothing. The chunk with the new tokens appends their K / V.
#pragma once
#include "sm90_common.cuh"
#include "tasks/hopper/multitoken_paged_attention_hopper.cuh"

namespace static_mk {

// PARAMS {NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, PAGE_SIZE, CHUNK,
// C (chunks)}; BUF_SLOTS {qkv, k cache, v cache, cos, sin, qo_indptr,
// kv_indptr, kv_indices, last_page_len (sm90_attention's), parts: the chunks'
// outputs [C, T, Q x D] bf16, then their log2-sum-exps [T, C, Q] fp32 (one
// buffer: the node's one output)}
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
                MAX_SEQ = PARAMS::v[3], PAGE = PARAMS::v[4],
                CHUNK = PARAMS::v[5], C = PARAMS::v[6];
  constexpr int QPK = QH / KVH, QKV_STRIDE = KVH * (QPK + 2) * D;
  sm90_task_begin<IN...>(g);
  using bf16 = kernel::bfloat16;
  int const head = tk.y, chunk = tk.z;
  kernel::multitoken_paged_attention_hopper_impl<bf16,
                                                 QPK,
                                                 1,
                                                 KVH,
                                                 KVH * D, // KV_CACHE_STRIDE
                                                 QKV_STRIDE,
                                                 QH * D, // O_STRIDE
                                                 D,
                                                 CHUNK, // SEQ_LEN: a chunk
                                                 MAX_SEQ,
                                                 PAGE,
                                                 T,
                                                 true, // PARTITION_KV
                                                 C,
                                                 D,  // ROTARY_DIM
                                                 0,  // packed pages
                                                 0>( // no window
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 1>()) + head * D,
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 2>()) + head * D,
      buf_at<int const>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 7>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 8>()),
      (int16_t)tk.x,
      false, // qk_norm
      true,  // RoPE
      nullptr,
      nullptr,
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 4>()),
      1e-6f,
      1e-6f,
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 0>()) + head * (QPK + 2) * D,
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 9>()) + (size_t)chunk * T * QH * D +
          head * QPK * D,
      reinterpret_cast<float *>(buf_at<bf16>(g, slot_at<BUF_SLOTS, 9>()) +
                                (size_t)C * T * QH * D) +
          (size_t)chunk * QH + head * QPK,
      chunk);
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_attention_split() {
  return SM90_BODY_SMEM;
}

} // namespace static_mk
