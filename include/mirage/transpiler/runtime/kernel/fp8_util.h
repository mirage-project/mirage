// fp8_util.h - H100 FP8 Quantization utility functions
// This file bear much resemblance to element_binary.h and element_unary.h
// The reason we do not compose operators from these two files but implement
// specialized scaling utility functions is to save kernel launch overhead

#include <cassert>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/float8.h>
// #include <type_traits>

#include "utils.h"

using namespace cute;

#define E4M3_RANGE 448.f

namespace kn {

// template <typename Tout, typename Tin>
// static __device__ __forceinline__ Tout fp8_cast(Tin a) {
//   if constexpr (!(std::is_same_v<Tin, cutlass::half_t> ||
//                   std::is_same_v<Tin, cutlass::bfloat16_t> ||
//                   std::is_same_v<Tin, float> || std::is_same_v<Tin, __half>)) {
//     assert(0 && "unsupport input type in kn fp8_rowwise_scaling");
//   }
//   if constexpr (!(std::is_same_v<Tout, cutlass::float_e4m3_t> ||
//                   std::is_same_v<Tout, cutlass::float_e5m2_t>)) {
//     assert(0 && "unsupport output type in kn fp8_rowwise_scaling");
//   }
//   if constexpr (Tout == cutlass::float_e4m3_t) {
//     if constexpr (std::is_same_v<Tin, cutlass::half_t> || std::is_same_v<Tin, __half>) {
//       return cutlass::float_e4m3_t::from_half(a);
//     } else if constexpr (std::is_same_v<Tin, cutlass::bfloat16_t> || std::is_same_v<Tin, float>) {
//       float fv = float(v);
//       return cutlass::float_e4m3_t::from_float(a);
//     } else {
//       assert(0 && "unsupport datatype in kn fp8_rowwise_scaling");
//     }
//   } else if constexpr (Tout == cutlass::float_e5m2_t) {
//     assert(0 && "e5m2 output type current not implemented!")
//   }

// }

template <typename Tout, typename Tin>
static __device__ __forceinline__ Tout fp8_cvt(Tin a, Tin scale) {
  float f = float(a) / float(scale);
  f = (Tin)(fmaxf(-E4M3_RANGE, fminf((float)f, E4M3_RANGE)));

  if constexpr (Tout == cutlass::float_e4m3_t) {
    return cutlass::float_e4m3_t::from_float(f);
  } else if constexpr (Tout == cutlass::float_e5m2_t) {
    assert(0 && "e5m2 output type current not implemented!")
  }
}

// 
template <typename Config> 
static __global__ void 
    fp8_rowwise_scale_and_cvt(typename Config::Tout *__restrict__ out, // [B, X, Y]
                          typename Config::Tin const *__restrict__ in,  // [B, X, Y]
                          typename Config::Tin const *__restrict__ scale) { // [B, X]
    using T = typename Config::T;
    using Numel = typename Config::Numel;
    auto src_layout = (typename Config::SrcLayout){};
    auto dst_layout = (typename Config::DstLayout){};
    auto scale_layout = (typename Config::ScaleLayout){};

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Numel{}) {
      int64_t phy_pos_src = src_layout(idx);
      Tin v_in   = in[phy_pos_src];

      // in shape (B, X, Y) -> scale should have shape (B, X)
      auto coord_full = cute::unflatten(src_layout, idx);
      auto coord_scale = cute::select<0,1>(coord_full);
      int64_t phy_pos_scale = scale_layout(coord_scale);
      Tin v_scale = scale[phy_pos_scale];

      int64_t phy_pos_dst = dst_layout(idx);
      out[phy_pos_dst] = fp8_cvt(v_in, v_scale);
    }
}
                          
template <typename Tout_, 
          typename Tin_, 
          typename SrcLayout_,
          typename ScaleLayout_,
          typename DstLayout_>
class FP8RowwiseScaleAndConvertKernel {
public: 
  using Tout = Tout_; // output fp8 tensor datatype
  static_assert(
    std::is_same_v<Tout, cutlass::float_e4m3_t> ||
    std::is_same_v<Tout, cutlass::float_e5m2_t>,
    "Tout_ must be cutlass::float_e4m3_t or cutlass::float_e5m2_t"
  );

  using Tin = Tin_; // input tensor datatype
  static_assert(
    std::is_same_v<Tin, cutlass::half_t> ||
    std::is_same_v<Tin, cutlass::bfloat16_t> ||
    std::is_same_v<Tin, float> || 
    std::is_same_v<Tin, __half>,
    "Tin_ must be one of supported float type"
  );

  using SrcLayout = SrcLayout_;
  using DstLayout = DstLayout_;
  using ScaleLayout = ScaleLayout_;
  CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}));
  using Numel = decltype(cute::size(SrcLayout{}));

  static constexpr int BLOCK_SIZE = 512;
  static constexpr dim3 block_shape = {BLOCK_SIZE, 1, 1};
  static constexpr dim3 grid_shape = {ceil_div(Numel::value, BLOCK_SIZE), 1, 1};
  static void run(Tout *out, Tin const *in, TinScale const *in_s) {
    fp8_rowwise_scale_and_cvt<
      FP8RowwiseScaleAndConvertKernel<Tout, Tin, TinScale, SrcLayout, DstLayout>>
      <<<grid_shape, block_shape>>>(out, in, in_s);
  }
}; 

} // mirage