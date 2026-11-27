// Qwen3.8 GDN decode: one 256-channel Conv1D/SiLU tile of one request.
// The state is SGLang's [slot, channel, 3] BF16 cache. Slot zero is padding.
#pragma once
#include "../core.cuh"
#include "mirage/persistent_kernel/tasks/common/qwen35_gdn.cuh"

namespace static_mk {

__device__ __forceinline__ float qwen_gdn_conv_triton_silu(float acc) {
  // SGLang's fused Triton Conv1D emits mul by -log2(e), ex2.approx,
  // add 1, then div.full before the final BF16 conversion.
  float exponent = __fmul_rn(acc, __int_as_float(0xbfb8aa3b));
  float exp_value;
  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(exp_value) : "f"(exponent));
  float const denominator = __fadd_rn(1.f, exp_value);
  float result;
  asm volatile("div.full.f32 %0, %1, %2;"
               : "=f"(result) : "f"(acc), "f"(denominator));
  return result;
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS,
          class QKVZ, class... REST>
__device__ __forceinline__ void run_qwen_gdn_conv(
    Maps const &, G const &g, KernelLocals &, StaticTask const &tk) {
  static_assert(T == 8 && SELF::x == 40 && SELF::y == 8,
                "Qwen GDN static Conv1D supports B8 only");
  static_assert((QKVZ::x == 0 || (QKVZ::buf >= 0 && QKVZ::counter >= 0)) &&
                    SELF::buf >= 0 && SELF::counter >= 0,
                "projection and conv need output buffers and counters");
  if constexpr (QKVZ::x != 0) {
    if (threadIdx.x == 0)
      cnt_wait(cnt_now(g) + QKVZ::counter, QKVZ::x * QKVZ::y);
  }
  __syncthreads();

  int const c0 = tk.x * 256, row = tk.y;
  int const slot = buf_at<int const>(g, slot_at<BUF_SLOTS, 2>())[row];
  auto out = buf_at<__nv_bfloat16>(g, SELF::buf) + row * 10240 + c0;
  if (slot <= 0) {
    out[threadIdx.x] = __float2bfloat16(0.f);
    __syncthreads();
  } else {
    __nv_bfloat16 const *qkvz;
    if constexpr (QKVZ::x == 0)
      qkvz = buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>());
    else
      qkvz = buf_at<__nv_bfloat16 const>(g, QKVZ::buf);
    auto x = qkvz + row * 16384 + c0;
    auto w = buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()) + c0 * 4;
    auto state = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 1>()) +
                 static_cast<size_t>(slot) * 10240 * 3 + c0 * 3;
    // SGLang's fused Triton Conv1D multiplies BF16 operands into a BF16
    // product, widens that product for FP32 accumulation, and applies SiLU
    // before the final BF16 store. Keep each product's rounding boundary.
    int const c = threadIdx.x;
    float acc = 0.f;
    for (int tap = 0; tap < 3; ++tap) {
      float const product = __bfloat162float(__float2bfloat16_rn(
          __fmul_rn(float(state[c * 3 + tap]), float(w[c * 4 + tap]))));
      acc = __fadd_rn(acc, product);
    }
    float const last_product = __bfloat162float(__float2bfloat16_rn(
        __fmul_rn(float(x[c]), float(w[c * 4 + 3]))));
    acc = __fadd_rn(acc, last_product);
    out[c] = __float2bfloat16_rn(qwen_gdn_conv_triton_silu(acc));
    state[c * 3] = state[c * 3 + 1];
    state[c * 3 + 1] = state[c * 3 + 2];
    state[c * 3 + 2] = x[c];
    __syncthreads();
  }
  if (threadIdx.x == 0) cnt_add(cnt_now(g) + SELF::counter + row, 1);
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS,
          class... IN>
constexpr int smem_qwen_gdn_conv() { return 0; }

} // namespace static_mk
