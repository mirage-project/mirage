// tasks/sm90_attention.cuh -- the sm90_attention task type: MPK's Hopper paged
// attention (multitoken_paged_attention_hopper.cuh: RoPE, the new tokens' K / V
// appended to the paged cache, attention with GPT-OSS's per-head sinks and
// sliding window) of one request and KV head. Task (x, y) of grid (R, KVH, 1):
// request x, KV head y. The page tables are graph tensors the host fills
// before each step (MPK's runtime keeps them in its own buffers).
#pragma once
#include "sm90_common.cuh"
#include "tasks/hopper/multitoken_paged_attention_hopper.cuh"

namespace static_mk {

// PARAMS {NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, PAGE_SIZE,
// WINDOW (0: full attention), HAS_SINK}; BUF_SLOTS {qkv [T, KVH x (Q / KVH + 2)
// x D] bf16 (each KV head's q heads, k, v), k cache and v cache [pages, PAGE,
// KVH, D] bf16, cos and sin [MAX_SEQ_LEN, D] bf16, qo_indptr [R + 1] i32,
// kv_indptr [R + 1] i32, kv_indices [pages] i32, last_page_len [R] i32, sinks
// [KVH, Q / KVH] bf16 (-1: none), output [T, Q x D] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_attention(Maps const &,
                                                   G const &g,
                                                   KernelLocals &,
                                                   StaticTask const &tk) {
  constexpr int QH = PARAMS::v[0], KVH = PARAMS::v[1], D = PARAMS::v[2],
                MAX_SEQ = PARAMS::v[3], PAGE = PARAMS::v[4],
                WINDOW = PARAMS::v[5];
  constexpr bool HAS_SINK = PARAMS::v[6] == 1;
  constexpr int QPK = QH / KVH, QKV_STRIDE = KVH * (QPK + 2) * D;
  sm90_task_begin<IN...>(g);
  using bf16 = kernel::bfloat16; // the body's own type (MPK's codegen's)
  int const head = tk.y;
  kernel::multitoken_paged_attention_hopper_impl<bf16,
                                                 QPK,
                                                 1,
                                                 KVH,
                                                 KVH * D, // KV_CACHE_STRIDE
                                                 QKV_STRIDE,
                                                 QH * D, // O_STRIDE
                                                 D,
                                                 -1, // SEQ_LEN (split KV only)
                                                 MAX_SEQ,
                                                 PAGE,
                                                 T,
                                                 false, // PARTITION_KV
                                                 1,
                                                 D, // ROTARY_DIM
                                                 0, // packed pages
                                                 WINDOW>(
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 1>()) + head * D,
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 2>()) + head * D,
      buf_at<int const>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 7>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 8>()),
      (int16_t)tk.x,
      false, // qk_norm (GPT-OSS has none)
      true,  // RoPE
      nullptr,
      nullptr,
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 4>()),
      1e-6f,
      1e-6f,
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 0>()) + head * (QPK + 2) * D,
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 10>()) + head * QPK * D,
      nullptr,
      0,
      HAS_SINK ? buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 9>()) + head * QPK
               : nullptr);
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_attention() {
  return SM90_BODY_SMEM;
}

} // namespace static_mk
