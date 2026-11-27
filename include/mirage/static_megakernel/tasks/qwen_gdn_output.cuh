// Publish the final BF16 out-projection buffer to SGLang's output tensor.
#pragma once
#include "../core.cuh"

namespace static_mk {

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS,
          class PROJ, class... REST>
__device__ __forceinline__ void run_qwen_gdn_output(
    Maps const &, G const &g, KernelLocals &, StaticTask const &tk) {
  static_assert(T == 8 && SELF::x == 40 && SELF::y == 1,
                "Qwen GDN static output supports B8 only");
  static_assert(PROJ::buf >= 0 && PROJ::counter >= 0,
                "the out projection must store BF16 and count its tasks");
  if (threadIdx.x == 0)
    cnt_wait(cnt_now(g) + PROJ::counter, PROJ::x * PROJ::y);
  __syncthreads();
  auto src = buf_at<__nv_bfloat16 const>(g, PROJ::buf);
  auto dst = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 0>());
  int const c0 = tk.x * 128;
  for (int i = threadIdx.x; i < T * 128; i += blockDim.x) {
    int const row = i / 128, c = i % 128;
    dst[row * 5120 + c0 + c] = src[row * 5120 + c0 + c];
  }
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS,
          class... IN>
constexpr int smem_qwen_gdn_output() { return 0; }

} // namespace static_mk
