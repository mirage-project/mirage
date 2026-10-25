// tasks/fx_bf16.cuh -- Turn a COMBINE_ADD GEMM output (int64 fixed point
// [T][N], the K parts added by its tasks) into bf16 [T][N]: a layer's final
// output when its last GEMM splits K (GLM-5.3-Flash DSA: o_proj, whose bf16
// output SGLang all-reduces). Task x is columns [128 x, 128 x + 128) of all T
// tokens: it waits for that column block (dsa::fx_wait_cols: the GEMM's counter
// of the block counts its K parts), then thread i converts columns 4 (i % 32)
// .. 4 (i % 32) + 3 of token i / 32 (round to nearest).
#pragma once
#include "../dsa_common.cuh"

namespace static_mk {

// Task x: column block x. No PARAMS. IN: the COMBINE_ADD GEMM node.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class IN,
          class... MORE>
__device__ __forceinline__ void run_fx_bf16(Maps const &,
                                            G const &g,
                                            KernelLocals &L,
                                            StaticTask const &tk) {
  constexpr int N = IN::v[2];
  static_assert(N % 128 == 0 && SELF::x * 128 == N && T * 32 == 256,
                "fx_bf16: a task per 128 columns, a thread per 4 columns of "
                "a token");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0,
                "fx_bf16: its output and its counter");
  int const c0 = 128 * tk.x;
  // The task uses no shared memory, so it starts with no barrier.
  if (threadIdx.x < 32) {
    dsa::fx_wait_cols<IN>(g, c0, c0 + 128);
  }
  __syncthreads();
  dsa::stamp_ready(L);
  int const t = threadIdx.x / 32, c = c0 + 4 * (threadIdx.x % 32);
  auto const *fx =
      buf_at<unsigned long long const>(g, IN::buf) + (size_t)t * N + c;
  ulonglong2 const a = __ldcg(reinterpret_cast<ulonglong2 const *>(fx)),
                   b = __ldcg(reinterpret_cast<ulonglong2 const *>(fx) + 1);
  __nv_bfloat162 const lo = __floats2bfloat162_rn(fx_to_float(a.x),
                                                  fx_to_float(a.y)),
                       hi = __floats2bfloat162_rn(fx_to_float(b.x),
                                                  fx_to_float(b.y));
  uint2 out;
  out.x = *reinterpret_cast<uint32_t const *>(&lo);
  out.y = *reinterpret_cast<uint32_t const *>(&hi);
  *reinterpret_cast<uint2 *>(buf_at<__nv_bfloat16>(g, SELF::buf) +
                             (size_t)t * N + c) = out;
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(g.cnt + SELF::counter, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_fx_bf16() {
  return 0;
}

} // namespace static_mk
