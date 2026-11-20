// Static-megakernel adapter for the native Blackwell paged-attention task.
#pragma once

#include "sm90_common.cuh"
#include "mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh"

namespace static_mk {

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm100_attention(Maps const &,
                                                    G const &g,
                                                    KernelLocals &L,
                                                    StaticTask const &tk) {
  constexpr int QH = PARAMS::v[0], KVH = PARAMS::v[1], D = PARAMS::v[2];
  constexpr int MAX_SEQ = PARAMS::v[3], PAGE = PARAMS::v[4];
  constexpr int WINDOW = PARAMS::v[5], HAS_SINK = PARAMS::v[6];
  constexpr int HAS_BIAS = PARAMS::v[7], MAX_TOKENS = PARAMS::v[8];
  constexpr int QPK = QH / KVH;
  constexpr int W = (QPK + 2) * D, QKV_STRIDE = KVH * W;
  static_assert(QH % KVH == 0 && D == 64,
                "sm100_attention expects GPT-OSS GQA with head_dim=64");
  static_assert(MAX_TOKENS > 0, "sm100_attention needs a positive token bound");

  sm90_wait_inputs<~0u, IN...>(g);
  int const head = tk.y;
  kernel::bfloat16 *qkv =
      HAS_BIAS ? buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 11>())
               : buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 0>());
  if constexpr (HAS_BIAS) {
    float const *src = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>());
    kernel::bfloat16 const *bias =
        buf_at<kernel::bfloat16 const>(g, slot_at<BUF_SLOTS, 10>());
    int const *qo = buf_at<int const>(g, slot_at<BUF_SLOTS, 5>());
    int const t0 = qo[tk.x], rows = qo[tk.x + 1] - t0;
    for (int i = threadIdx.x; i < rows * W; i += blockDim.x) {
      int const t = t0 + i / W, col = i % W;
      qkv[(size_t)t * QKV_STRIDE + head * W + col] =
          kernel::bfloat16(src[(size_t)t * QKV_STRIDE + head * W + col] +
                           float(bias[head * W + col]));
    }
    __syncthreads();
  }

  kernel::multitoken_paged_attention_sm100_task_impl<kernel::bfloat16,
                                                     QPK,
                                                     1,
                                                     KVH * D,
                                                     QKV_STRIDE,
                                                     QH * D,
                                                     D,
                                                     MAX_SEQ,
                                                     PAGE,
                                                     0,
                                                     0,
                                                     MAX_TOKENS,
                                                     D,
                                                     WINDOW>(
      qkv + head * W,
      buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 1>()) + head * D,
      buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 2>()) + head * D,
      buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 12>()) + head * QPK * D,
      buf_at<int const>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 7>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 8>()),
      (int16_t)tk.x,
      false,
      true,
      nullptr,
      nullptr,
      buf_at<kernel::bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<kernel::bfloat16 const>(g, slot_at<BUF_SLOTS, 4>()),
      1e-6f,
      1e-6f,
      HAS_SINK ? buf_at<kernel::bfloat16 const>(g, slot_at<BUF_SLOTS, 9>()) +
                     head * QPK
               : nullptr);
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm100_attention() {
  return mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
}

} // namespace static_mk
