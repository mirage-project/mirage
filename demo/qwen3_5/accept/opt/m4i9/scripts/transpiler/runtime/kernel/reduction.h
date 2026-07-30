#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace kn {

// The First Reduction Kernel
// Each thread in this kernel is responsible for producing one element in the
// output tensor. In other words, each thread reduces a segment of the input.
// This is efficient when the reduction dim != the innermost dim since global
// memory access can be coalesced.
// TODO(intlsy): Use half2
template <typename Config>
static __global__ void
    reduction_kernel_fwd(typename Config::T *__restrict__ out,
                         typename Config::T const *__restrict__ in) {
  using T = typename Config::T;
  auto dst_layout = (typename Config::DstLayout){};
  auto dst_in_src_layout = (typename Config::DstInSrcLayout){};
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size(dst_layout)) {
    int64_t dst_phy_pos = dst_layout(idx);
    int64_t src_phy_pos = dst_in_src_layout(idx);
    T result = (T)0;
    CUTE_UNROLL
    for (int i = 0; i < Config::REDUCTION_FACTOR; ++i) {
      result += in[src_phy_pos + i * Config::REDUCTION_DIM_STRIDE];
    }
    out[dst_phy_pos] = result;
  }
}

template <typename T_,
          typename SrcLayout_,
          typename DstLayout_,
          int REDUCTION_DIM_>
class ReductionKernel {
public:
  using T = T_;
  using SrcLayout = SrcLayout_;
  using DstLayout = DstLayout_;
  static constexpr int REDUCTION_DIM = REDUCTION_DIM_;

  static constexpr int NUM_DIMS = rank(SrcLayout{});
  CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == rank(DstLayout{}));

  CUTE_STATIC_ASSERT_V(get<REDUCTION_DIM>(shape(SrcLayout{})) %
                           get<REDUCTION_DIM>(shape(DstLayout{})) ==
                       _0{});
  static constexpr int REDUCTION_FACTOR =
      get<REDUCTION_DIM>(shape(SrcLayout{})) /
      get<REDUCTION_DIM>(shape(DstLayout{}));

  static constexpr int REDUCTION_DIM_STRIDE =
      get<REDUCTION_DIM>(stride(SrcLayout{}));

  // When we feed a CuTe layout with an interger, it first translates the int
  // into a (logical) coordinate, and then takes the dot product with the
  // strides to get the physical index.
  // Using the following layout, we can get the physical index of the starting
  // element in a reduction segment by a logical index (i.e. the thread index)
  // in the output tensor.
  using DstInSrcLayout = decltype(make_layout(
      shape(DstLayout{}),
      replace<REDUCTION_DIM>(stride(SrcLayout{}),
                             Int<REDUCTION_DIM_STRIDE * REDUCTION_FACTOR>{})));
  static_assert(is_static_v<DstInSrcLayout>);

  static constexpr int BLOCK_SIZE = 512;
  static constexpr dim3 block_shape = {BLOCK_SIZE, 1, 1};
  static constexpr dim3 grid_shape = {
      ceil_div(size(DstLayout{}), BLOCK_SIZE), 1, 1};

  static void run(T *out, T const *in) {
    // Except the reduction dimension, the other dimensions should be the same.
    constexpr_for<int, 0, NUM_DIMS, 1>([&](auto i) {
      if constexpr (i != REDUCTION_DIM) {
        CUTE_STATIC_ASSERT_V(get<i>(shape(SrcLayout{})) ==
                             get<i>(shape(DstLayout{})));
      }
    });
    reduction_kernel_fwd<
        ReductionKernel<T, SrcLayout, DstLayout, REDUCTION_DIM>>
        <<<grid_shape, block_shape>>>(out, in);
  }
};

} // namespace kn
