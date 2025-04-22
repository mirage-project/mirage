// element_unary.h - Implementation of element wise unary operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>

#include "utils.h"

namespace kn {

enum class ElementUnaryOpType { EXP, SILU, GELU, RELU, CLAMP, SQUARE, SQRT };

template <typename T, ElementUnaryOpType OP>
static __device__ __forceinline__ T perform_element_unary_op(T a) {
  if constexpr (!(std::is_same_v<T, cutlass::half_t> ||
                  std::is_same_v<T, cutlass::bfloat16_t> ||
                  std::is_same_v<T, float> || std::is_same_v<T, __half>)) {
    assert(0 && "unsupport datatype in kn elementunary");
  }
  if constexpr (OP == ElementUnaryOpType::EXP) {
    return (T)expf((float)a);
  } else if constexpr (OP == ElementUnaryOpType::SILU) {
    return (T)(((float)a) * (1.0f / (1.0f + expf((float)-a))));
  } else if constexpr (OP == ElementUnaryOpType::GELU) {
    return (T)((((float)a) / 2.0f) * (1.0f + erff(((float)a) / sqrtf(2.0f))));
  } else if constexpr (OP == ElementUnaryOpType::SQUARE) {
    return (T)((float)a * (float)a);
  } else if constexpr (OP == ElementUnaryOpType::SQRT) {
    return (T)(sqrtf((float)a));
  } else if constexpr (OP == ElementUnaryOpType::RELU) {
    return (T)(fmaxf(0.f, (float)a));
  } else if constexpr (OP == ElementUnaryOpType::CLAMP) {
    return (T)(fmaxf(0.f, fminf((float)a, 1.f)));
  } else {
    assert(0 && "unsupport datatype in kn elementunary");
  }
}

// Elementwise Unary Kernel
// A simple kernel that applies a unary operator to each element in the input
// and writes the result to the output.
// Each thread is responsible for one element. Thread #i calculates element
// on address SrcLayout(i) and writes the result to DstLayout(i), so the kernel
// is efficient when the 0-th dimension is the innermost dimension.
// TODO(intlsy): Use half2
template <typename Config>
static __global__ void
    element_unary_kernel_fwd(typename Config::T *__restrict__ out,
                             typename Config::T const *__restrict__ in) {
  using T = typename Config::T;
  using Numel = typename Config::Numel;
  auto src_layout = (typename Config::SrcLayout){};
  auto dst_layout = (typename Config::DstLayout){};
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Numel{}) {
    int64_t phy_pos_src = src_layout(idx);
    int64_t phy_pos_dst = dst_layout(idx);
    out[phy_pos_dst] = perform_element_unary_op<T, Config::OP>(in[phy_pos_src]);
  }
}

template <typename T_,
          ElementUnaryOpType OP_,
          typename SrcLayout_,
          typename DstLayout_>
class ElementUnaryKernel {
public:
  using T = T_;
  static constexpr ElementUnaryOpType OP = OP_;
  using SrcLayout = SrcLayout_;
  using DstLayout = DstLayout_;

  CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}));
  using Numel = decltype(cute::size(SrcLayout{}));

  static constexpr int BLOCK_SIZE = 512;
  static constexpr dim3 block_shape = {BLOCK_SIZE, 1, 1};
  static constexpr dim3 grid_shape = {ceil_div(Numel::value, BLOCK_SIZE), 1, 1};

  static void run(T *out, T const *in) {
    element_unary_kernel_fwd<ElementUnaryKernel<T, OP, SrcLayout, DstLayout>>
        <<<grid_shape, block_shape>>>(out, in);
  }
};

} // namespace kn