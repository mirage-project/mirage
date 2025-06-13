// element_binary.h - Implementation of element wise binary operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace kn {

enum class ElementBinaryOpType { ADD, MUL, DIV, POW };

template <typename T, ElementBinaryOpType OP>
static __device__ __forceinline__ T perform_element_binary_op(T a, T b) {
  if constexpr (OP == ElementBinaryOpType::ADD) {
    return a + b;
  } else if constexpr (OP == ElementBinaryOpType::MUL) {
    return a * b;
  } else if constexpr (OP == ElementBinaryOpType::DIV) {
    return a / b;
  } else if constexpr (OP == ElementBinaryOpType::POW) {
    return (T)powf((float)a, (float)b);
  } else {
    assert(0);
  }
}

// Elementwise Binary Kernel
// TODO(intlsy): Use half2
template <typename Config>
static __global__ void
    element_binary_kernel_fwd(typename Config::T *__restrict__ out,
                              typename Config::T const *__restrict__ in0,
                              typename Config::T const *__restrict__ in1) {
  using T = typename Config::T;
  using Numel = typename Config::Numel;
  auto src0_in_dst_layout = (typename Config::Src0InDstLayout){};
  auto src1_in_dst_layout = (typename Config::Src1InDstLayout){};
  auto dst_layout = (typename Config::DstLayout){};
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Numel{}) {
    int64_t src0_phy_pos = src0_in_dst_layout(idx);
    int64_t src1_phy_pos = src1_in_dst_layout(idx);
    int64_t dst_phy_pos = dst_layout(idx);
    out[dst_phy_pos] = perform_element_binary_op<T, Config::OP>(
        in0[src0_phy_pos], in1[src1_phy_pos]);
  }
}

class ReplacementFunctionClass {
public:
  template <auto cur_dim_shape, auto cur_dim_stride>
  constexpr Int<cur_dim_shape == 1 ? 0 : cur_dim_stride>
      operator()(Int<cur_dim_shape>, Int<cur_dim_stride>) {
    return {};
  }
};

template <typename T_,
          ElementBinaryOpType OP_,
          typename Src0Layout_,
          typename Src1Layout_,
          typename DstLayout_>
class ElementBinaryKernel {
public:
  using T = T_;
  static constexpr ElementBinaryOpType OP = OP_;
  using Src0Layout = Src0Layout_;
  using Src1Layout = Src1Layout_;
  using DstLayout = DstLayout_;

  using Numel = decltype(cute::size(DstLayout{}));

  // Layouts that take an logical index from dst's layout, and return the
  // physical index in src0 and src1. We use this to implement broadcast: both
  // two layouts have the same shape as DstLayout, while for broadcast
  // dimentions, we set the stride to 0
  using Src0InDstLayout =
      decltype(make_layout(shape(DstLayout{}),
                           transform(shape(Src0Layout{}),
                                     stride(Src0Layout{}),
                                     ReplacementFunctionClass{})));
  using Src1InDstLayout =
      decltype(make_layout(shape(DstLayout{}),
                           transform(shape(Src1Layout{}),
                                     stride(Src1Layout{}),
                                     ReplacementFunctionClass{})));
  static_assert(is_static_v<Src0InDstLayout>);
  static_assert(is_static_v<Src1InDstLayout>);

  static constexpr int BLOCK_SIZE = 512;
  static constexpr dim3 block_shape = {BLOCK_SIZE, 1, 1};
  static constexpr dim3 grid_shape = {ceil_div(Numel::value, BLOCK_SIZE), 1, 1};

  static void run(T *out, T const *in0, T const *in1) {
    element_binary_kernel_fwd<
        ElementBinaryKernel<T, OP, Src0Layout, Src1Layout, DstLayout>>
        <<<grid_shape, block_shape>>>(out, in0, in1);
  }
};

} // namespace kn