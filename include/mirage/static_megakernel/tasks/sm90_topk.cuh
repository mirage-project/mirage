// tasks/sm90_topk.cuh -- the sm90_topk task type: MPK's top-k softmax router
// (topk_softmax_sm100.cuh, architecture-neutral) over the step's tokens. One
// task.
#pragma once
#include "sm90_common.cuh"
#include "tasks/blackwell/topk_softmax_sm100.cuh"

namespace static_mk {

// PARAMS {E, TOPK}; BUF_SLOTS {logits [T, E] bf16, the step's token count [1]
// i32 (rows past it are padding, left unrouted), weights [T, TOPK] fp32,
// routing indices [E, T] i32, the active-expert list [E + 1] i32}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_topk(Maps const &,
                                              G const &g,
                                              KernelLocals &,
                                              StaticTask const &) {
  constexpr int E = PARAMS::v[0], TOPK = PARAMS::v[1];
  // a row of experts over E / VPT lanes, a half or whole warp (MPK's codegen)
  constexpr int VPT = (E / 8 == 16 || E / 8 == 32) ? 8 : E / 16;
  constexpr int BYTES_PER_LDG = VPT * 2 < 16 ? VPT * 2 : 16;
  sm90_task_begin<IN...>(g);
  kernel::topk_softmax_task_impl<cute::bfloat16_t, VPT, E, 8, BYTES_PER_LDG>(
      buf_at<void>(g, slot_at<BUF_SLOTS, 0>()),
      nullptr,
      buf_at<void>(g, slot_at<BUF_SLOTS, 2>()),
      T,
      TOPK,
      buf_at<void>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<void>(g, slot_at<BUF_SLOTS, 4>()),
      0,
      E,
      true,
      *buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()));
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_topk() {
  return 0;
}

} // namespace static_mk
