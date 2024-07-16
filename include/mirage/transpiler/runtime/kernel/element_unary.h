// element_unary.h - Implementation of element wise unary operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>

#include "utils.h"

enum class ElementUnaryOpType { EXP };

template <typename T, ElementUnaryOpType OP>
static __device__ __forceinline__ T perform_element_unary_op(T a) {
  if constexpr (OP == ElementUnaryOpType::EXP) {
    if constexpr (std::is_same_v<T, cutlass::half_t> ||
                  std::is_same_v<T, __half>) {
      return (T)expf((float)a);
    } else {
      assert(0);
    }
  } else {
    assert(0);
  }
}

// TODO(intlsy): Use half2
template <typename Config>
static __global__ void
    element_unary_kernel_fwd(typename Config::T *__restrict__ out,
                             typename Config::T const *__restrict__ in) {
  using T = typename Config::T;
  using Numel = typename Config::Numel;
  auto layout = (typename Config::SrcDstLayout){};
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Numel{}) {
    int64_t phy_pos = layout(idx);
    out[phy_pos] = perform_element_unary_op<T, Config::OP>(in[phy_pos]);
  }
}

template <typename T_, ElementUnaryOpType OP_, typename SrcDstLayout_>
class ElementUnaryKernel {
public:
  using T = T_;
  static constexpr ElementUnaryOpType OP = OP_;
  using SrcDstLayout = SrcDstLayout_;

  using Numel = decltype(cute::size(SrcDstLayout{}));

  static constexpr int BLOCK_SIZE = 512;
  static constexpr dim3 block_shape = {BLOCK_SIZE, 1, 1};
  static constexpr dim3 grid_shape = {CDIV(Numel::value, BLOCK_SIZE), 1, 1};

  static void run(T *out, T const *in) {
    element_unary_kernel_fwd<ElementUnaryKernel<T, OP, SrcDstLayout>>
        <<<grid_shape, block_shape>>>(out, in);
  }
};
