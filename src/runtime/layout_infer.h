#pragma once
#include <cute/tensor.hpp>
#include "mirage/type.h"
using namespace cute;

namespace mirage {
namespace layout{

template <OpType Op, typename... InputLayouts>
struct LayoutInfer;

template <typename LayoutIn0, typename LayoutIn1>
struct LayoutInfer<TB_MATMUL_OP, LayoutIn0, LayoutIn1> {
  using LayoutOut = decltype(composition(
      Swizzle<3,3,3>{},
      make_layout(
        make_shape(get<0>(shape(LayoutIn0{})), get<1>(shape(LayoutIn1{}))),
        make_stride(Int<1>{}, get<0>(shape(LayoutIn0{})))
      )));
};

template <typename LayoutIn>
struct LayoutInfer<TB_SQUARE_OP, LayoutIn> {
  using LayoutOut = LayoutIn;
};

template <typename LayoutIn>
struct LayoutInfer<TB_REDUCTION_1_OP, LayoutIn> {
  using LayoutOut = decltype(make_layout(
      make_shape(get<0>(shape(LayoutIn0{})), Int<1>{}),
      make_stride(Int<1>{}, get<0>(shape(LayoutIn0{})))));
};

template <typename LayoutIn0, typename LayoutIn1>
struct LayoutInfer<TB_DIV_OP, LayoutIn0, LayoutIn1> {
  using LayoutOut = LayoutIn0;
};

} // namespace runtime
} // namespace mirage