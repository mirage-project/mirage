// tasks/sm90_moe.cuh -- the sm90_moe task type: one routed expert projection
// (W13 or W2) of MPK's Hopper MoE kernel (moe_linear_swapAB_hopper.cuh, packed
// MXFP4 weights decoded into the WGMMA registers). Task (x, y) of grid
// (EXPERT_STRIDE, N / ROWS, 1): the activated experts x, x + EXPERT_STRIDE, ...
// times weight rows [y ROWS, y ROWS + ROWS) of each.
#pragma once
#include "sm90_common.cuh"
#include "tasks/hopper/moe_linear_swapAB_hopper.cuh"

namespace static_mk {

// PARAMS {W13 (1: W13, input [T, K]; 0: W2, input [T, TOPK, K]), ROWS (weight
// rows per task), N (the weight's rows), K, E (experts), TOPK, HAS_BIAS};
// BUF_SLOTS {input, weight scales [E, N, K / 32] u8, routing indices [E, T]
// i32, the active-expert list [E + 1] i32, bias [E, N] bf16 (-1: none), output
// [T, TOPK, N] bf16}; MAP_SLOTS {the packed weight [E N, K / 2] u8, box 64 rows
// x 32 B}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_moe(Maps const &maps,
                                             G const &g,
                                             KernelLocals &,
                                             StaticTask const &tk) {
  constexpr bool W13 = PARAMS::v[0] == 1, HAS_BIAS = PARAMS::v[6] == 1;
  constexpr int ROWS = PARAMS::v[1], N = PARAMS::v[2], K = PARAMS::v[3],
                E = PARAMS::v[4], TOPK = PARAMS::v[5];
  static_assert(N % ROWS == 0 && ROWS % 64 == 0 && K % 64 == 0,
                "sm90_moe: whole 64-row tiles, whole 64-value k tiles");
  sm90_task_begin<IN...>(g);
  int const row0 = tk.y * ROWS;
  using TMA_A = kernel::tma::tma_2d<uint8_t,
                                    0,
                                    0,
                                    0,
                                    E * N,
                                    K / 2,
                                    64,
                                    32,
                                    K / 2,
                                    1,
                                    1,
                                    1,
                                    64 * 32,
                                    true>;
  TMA_A tma_a(const_cast<CUtensorMap *>(&maps.m[slot_at<MAP_SLOTS, 0>()]));
  using bf16 = cute::bfloat16_t;
  auto mInput = [&] {
    bf16 *p = buf_at<bf16>(g, slot_at<BUF_SLOTS, 0>());
    if constexpr (W13) {
      return cute::make_tensor(
          cute::make_gmem_ptr(p),
          cute::make_layout(cute::make_shape(cute::Int<T>{}, cute::Int<K>{}),
                            cute::make_stride(cute::Int<K>{}, cute::Int<1>{})));
    } else {
      return cute::make_tensor(
          cute::make_gmem_ptr(p),
          cute::make_layout(
              cute::make_shape(
                  cute::Int<T>{}, cute::Int<K>{}, cute::Int<TOPK>{}),
              cute::make_stride(
                  cute::Int<TOPK * K>{}, cute::Int<1>{}, cute::Int<K>{})));
    }
  }();
  auto mBias = cute::make_tensor(
      cute::make_gmem_ptr(
          HAS_BIAS ? buf_at<bf16>(g, slot_at<BUF_SLOTS, 4>()) + row0 : nullptr),
      cute::make_layout(
          cute::make_shape(cute::Int<T>{}, cute::Int<ROWS>{}, cute::Int<E>{}),
          cute::make_stride(cute::Int<0>{}, cute::Int<1>{}, cute::Int<N>{})));
  auto mRouting = cute::make_tensor(
      cute::make_gmem_ptr(buf_at<int32_t>(g, slot_at<BUF_SLOTS, 2>())),
      cute::make_layout(cute::make_shape(cute::Int<E>{}, cute::Int<T>{}),
                        cute::make_stride(cute::Int<T>{}, cute::Int<1>{})));
  auto mMask = cute::make_tensor(
      cute::make_gmem_ptr(buf_at<int32_t>(g, slot_at<BUF_SLOTS, 3>())),
      cute::make_layout(cute::make_shape(cute::Int<E>{}),
                        cute::make_stride(cute::Int<1>{})));
  auto mOutput = cute::make_tensor(
      cute::make_gmem_ptr(buf_at<bf16>(g, slot_at<BUF_SLOTS, 5>()) + row0),
      cute::make_layout(
          cute::make_shape(
              cute::Int<T>{}, cute::Int<ROWS>{}, cute::Int<TOPK>{}),
          cute::make_stride(
              cute::Int<TOPK * N>{}, cute::Int<1>{}, cute::Int<N>{})));
  kernel::moe_linear_sm90_task_impl<bf16,
                                    TMA_A,
                                    decltype(mInput),
                                    decltype(mBias),
                                    decltype(mRouting),
                                    decltype(mMask),
                                    decltype(mOutput),
                                    64,
                                    16,
                                    T,
                                    ROWS,
                                    N,
                                    K,
                                    E,
                                    TOPK,
                                    SELF::x, // the stride between experts
                                    W13,
                                    !HAS_BIAS,
                                    true, // MXFP4 weights
                                    8>(
      tma_a,
      mInput,
      mBias,
      mRouting,
      mMask,
      mOutput,
      tk.x,
      buf_at<uint8_t const>(g, slot_at<BUF_SLOTS, 1>()) + row0 * (K / 32),
      row0);
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_moe() {
  return SM90_BODY_SMEM;
}

} // namespace static_mk
