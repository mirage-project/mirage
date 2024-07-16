// element_binary.h - Implementation of element wise binary operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>

#include "utils.h"

enum class ElementBinaryOpType { ADD, MUL, DIV };

template <typename T, ElementBinaryOpType OP>
static __device__ __forceinline__ T perform_element_binary_op(T a, T b) {
  if constexpr (OP == ElementBinaryOpType::ADD) {
    return a + b;
  } else if constexpr (OP == ElementBinaryOpType::MUL) {
    return a * b;
  } else if constexpr (OP == ElementBinaryOpType::DIV) {
    return a / b;
  } else {
    assert(0);
  }
}

// TODO(intlsy): Use half2
template <typename Config>
static __global__ void
    element_binary_kernel_fwd(typename Config::T *__restrict__ out,
                              typename Config::T const *__restrict__ in0,
                              typename Config::T const *__restrict__ in1) {
  using T = typename Config::T;
  using Numel = typename Config::Numel;
  auto layout = (typename Config::SrcDstLayout){};
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Numel{}) {
    int64_t phy_pos = layout(idx);
    out[phy_pos] =
        perform_element_binary_op<T, Config::OP>(in0[phy_pos], in1[phy_pos]);
  }
}

template <typename T_, ElementBinaryOpType OP_, typename SrcDstLayout_>
class ElementBinaryKernel {
public:
  using T = T_;
  static constexpr ElementBinaryOpType OP = OP_;
  using SrcDstLayout = SrcDstLayout_;

  using Numel = decltype(cute::size(SrcDstLayout{}));

  static constexpr int BLOCK_SIZE = 512;
  static constexpr dim3 block_shape = {BLOCK_SIZE, 1, 1};
  static constexpr dim3 grid_shape = {CDIV(Numel::value, BLOCK_SIZE), 1, 1};

  static void run(T *out, T const *in0, T const *in1) {
    element_binary_kernel_fwd<ElementBinaryKernel<T, OP, SrcDstLayout>>
        <<<grid_shape, block_shape>>>(out, in0, in1);
  }
};
