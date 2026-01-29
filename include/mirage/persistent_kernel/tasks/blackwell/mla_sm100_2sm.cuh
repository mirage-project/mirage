/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Self-contained MLA (Multi-head Latent Attention) Device Kernel for Blackwell (SM100)
 * 
 * This is a SINGLE FILE that contains the complete MLA kernel implementation.
 * All flashinfer dependencies are inlined - only CUTLASS/CuTe headers are needed.
 *
 * Contents:
 * 1. Pow2 helper for efficient power-of-2 operations
 * 2. CustomStride/Gather utilities for paged attention
 * 3. Tile schedulers (Individual and Persistent)
 * 4. Reduction kernel for split-KV
 * 5. Main MLA kernel (Sm100FmhaMlaKernelTmaWarpspecialized)
 * 6. Device layer (MLA class)
 *
 **************************************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <limits>

// CUTLASS/CuTe includes
#include "cute/arch/simd_sm100.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/device_kernel.h"

#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif


//////////////////////////////////////////////////////////////////////////////
// Section 1: Pow2 Helper
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha {

struct Pow2 {
  int n;
  int log2_n;

  explicit CUTE_DEVICE Pow2(int n) : n(n) {
#ifdef __CUDA_ARCH__
    log2_n = __ffs(n) - 1;
#endif
  }

  template <class T>
  CUTE_HOST_DEVICE T operator*(T const& b) const {
    return n * b;
  }

  template <int N>
  CUTE_HOST_DEVICE auto operator*(cute::Int<N> const&) const {
    if constexpr ((N & (N - 1)) == 0) {
      return Pow2{n * N};
    }
    return n * N;
  }
};

template <class T>
CUTE_HOST_DEVICE auto operator/(T const& a, Pow2 const& b) {
  return a >> b.log2_n;
}

template <class T>
CUTE_HOST_DEVICE auto operator%(T const& a, Pow2 const& b) {
  return a & (b.n - 1);
}

template <class T>
CUTE_HOST_DEVICE bool operator<(T const& a, Pow2 const& b) {
  return a < b.n;
}

CUTE_HOST_DEVICE void print(Pow2 const& a) { printf("2^%d", a.log2_n); }

}  // namespace cutlass::fmha

namespace cute {
template <>
struct is_integral<cutlass::fmha::Pow2> : true_type {};
}  // namespace cute


//////////////////////////////////////////////////////////////////////////////
// Section 2: CustomStride/Gather Utilities
//////////////////////////////////////////////////////////////////////////////

namespace example {

using namespace cute;

struct NoGather {
  template <class... Ts>
  NoGather(Ts...){};
};

template <class Index>
struct IndexedGather {
  CUTE_HOST_DEVICE constexpr IndexedGather(Index const* indices = {}) : indices_(indices) {}

  template <typename I>
  CUTE_HOST_DEVICE constexpr Index operator()(I i) const {
    return indices_[i];
  }

  CUTE_HOST_DEVICE friend void print(IndexedGather const& s) { cute::print("Indexed"); }

  Index const* indices_;
};

template <class Stride>
struct StridedGather {
  CUTE_HOST_DEVICE constexpr StridedGather(Stride stride = {}) : stride_(stride) {}

  template <class I>
  CUTE_HOST_DEVICE constexpr auto operator()(I i) const {
    return i * stride_;
  }

  CUTE_HOST_DEVICE friend void print(StridedGather const& s) {
    cute::print("Strided{");
    print(s.stride_);
    cute::print("}");
  }

  Stride stride_;
};

template <class Func, class Stride>
struct CustomStride {
  CUTE_HOST_DEVICE constexpr CustomStride(Func const& func, Stride const& stride)
      : func_(func), stride_(stride) {}

  template <class I>
  CUTE_HOST_DEVICE constexpr friend auto operator*(I i, CustomStride const& s) {
    return s.func_(i) * s.stride_;
  }

  template <class I>
  CUTE_HOST_DEVICE constexpr friend auto operator*(CustomStride const& s, I i) {
    return s.func_(i) * s.stride_;
  }

  CUTE_HOST_DEVICE friend void print(CustomStride const& s) {
    cute::print("Custom{");
    print(s.func_);
    cute::print(",");
    print(s.stride_);
    cute::print("}");
  }

  template <class Div>
  CUTE_HOST_DEVICE constexpr friend auto safe_div(CustomStride const& s, Div const& div) {
    return CustomStride<Func, decltype(safe_div(s.stride_, div))>(s.func_,
                                                                  safe_div(s.stride_, div));
  }

  template <class Shape>
  CUTE_HOST_DEVICE constexpr friend auto make_layout(Shape const& shape,
                                                     CustomStride const& stride) {
    return Layout<Shape, CustomStride>(shape, stride);
  }

  Func func_;
  Stride stride_;
};

template <class Stride, class Func>
CUTLASS_HOST_DEVICE auto make_custom_stride_layout(Stride const& stride, Func&& func) {
  auto idx = find_if(stride, [](auto x) { return not is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;
  return make_layout(repeat_like(stride, _1{}),
                     replace<I>(stride, CustomStride{static_cast<Func&&>(func), get<I>(stride)}));
}

template <class Iterator, class Shape, class Stride, class Func>
CUTLASS_HOST_DEVICE auto make_gather_tensor(Iterator iter, Shape const& shape, Stride const& stride,
                                            Func&& func) {
  if constexpr (not cutlass::platform::is_same<remove_cvref_t<Func>, NoGather>::value) {
    Layout matrix_layout = make_identity_layout(shape);
    auto offset = as_arithmetic_tuple(repeat_like(shape, _0{}));
    Layout gather_layout = make_custom_stride_layout(stride, static_cast<Func&&>(func));
    return make_tensor(iter, ComposedLayout{gather_layout, offset, matrix_layout});
  } else {
    return make_tensor(iter, shape, stride);
  }
}

}  // namespace example

namespace cute {

template <int N, int I, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(Shape const& shape, Stride const& stride) {
  if constexpr (is_tuple<Shape>::value) {
    return transform_layout(shape, stride,
                            [](auto const& s, auto const& d) { return upcast<N, I>(s, d); });
  } else if constexpr (is_scaled_basis<Stride>::value) {
    if constexpr (Stride::mode() == I) {
      return make_layout(ceil_div(shape, Int<N>{}), ceil_div(stride, Int<N>{}));
    } else {
      return make_layout(shape, stride);
    }
  } else {
    return upcast<N>(shape, stride);
  }
  CUTE_GCC_UNREACHABLE;
}

template <int N, class OuterShape, class OuterStride, class Offset, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(
    ComposedLayout<Layout<OuterShape, OuterStride>, Offset, Layout<Shape, Stride>> const& layout) {
  auto idx = find_if(layout.layout_a().stride(), [](auto x) { return is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;
  auto outer = upcast<N>(layout.layout_a());
  auto offset = as_arithmetic_tuple(replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));
  auto inner = upcast<N, I>(layout.layout_b().shape(), layout.layout_b().stride());
  return composition(outer, offset, inner);
}

}  // namespace cute


//////////////////////////////////////////////////////////////////////////////
// Section 3: Tile Schedulers
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::kernel {

struct Sm100MlaIndividualTileScheduler {
  struct Params {
    dim3 grid;
  };

  bool valid_ = true;

  CUTLASS_DEVICE
  Sm100MlaIndividualTileScheduler(Params const&) {}

  template <class ProblemShape, class ClusterShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape,
                                        KernelHardwareInfo hw_info,
                                        ClusterShape const& cluster_shape, int const& split_kv) {
    using namespace cute;
    dim3 grid(cute::get<0>(cluster_shape), cute::get<3>(problem_shape), split_kv);
    return Params{grid};
  }

  static dim3 get_grid_shape(Params const& params) { return params.grid; }

  CUTLASS_DEVICE bool is_valid() { return valid_; }

  CUTLASS_DEVICE auto get_block_coord() {
    using namespace cute;
    return make_coord(blockIdx.x, _0{}, blockIdx.y, blockIdx.z);
  }

  CUTLASS_DEVICE Sm100MlaIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

struct Sm100MlaPersistentTileScheduler {
  struct Params {
    int num_blocks;
    FastDivmod divmod_m_block;
    FastDivmod divmod_b;
    FastDivmod divmod_split_kv;
    KernelHardwareInfo hw_info;
  };

  int block_idx = 0;
  Params params;

  CUTLASS_DEVICE
  Sm100MlaPersistentTileScheduler(Params const& params) : block_idx(blockIdx.x), params(params) {}

  template <class ProblemShape, class ClusterShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape,
                                        KernelHardwareInfo hw_info,
                                        ClusterShape const& cluster_shape, int const& split_kv) {
    using namespace cute;
    int sm_count = hw_info.sm_count;
    if (sm_count <= 1 || sm_count % cute::size<0>(cluster_shape) != 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    hw_info.sm_count = sm_count;

    int num_m_blocks = cute::size<0>(cluster_shape);
    int num_blocks = num_m_blocks * cute::get<3>(problem_shape) * split_kv;

    return Params{num_blocks, {num_m_blocks}, {cute::get<3>(problem_shape)}, {split_kv}, hw_info};
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(std::min(params.num_blocks, params.hw_info.sm_count), 1, 1);
    return grid;
  }

  CUTLASS_DEVICE bool is_valid() { return block_idx < params.num_blocks; }

  CUTLASS_DEVICE auto get_block_coord() {
    using namespace cute;
    int block_decode = block_idx;
    int m_block, bidb, n_split_kv;
    params.divmod_m_block(block_decode, m_block, block_decode);
    params.divmod_b(block_decode, bidb, block_decode);
    params.divmod_split_kv(block_decode, n_split_kv, block_decode);
    return make_coord(m_block, _0{}, bidb, n_split_kv);
  }

  CUTLASS_DEVICE Sm100MlaPersistentTileScheduler& operator++() {
    block_idx += gridDim.x;
    return *this;
  }
};

}  // namespace cutlass::fmha::kernel


//////////////////////////////////////////////////////////////////////////////
// Section 4: Reduction Kernel
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::kernel {

using namespace cute;
template <class ElementOut, class ElementAcc, class ElementScale, size_t kNumHeads,
          size_t kHeadDimLatent, int kMaxSplits>
struct Sm100FmhaMlaReductionKernel {
  static const int SharedStorageSize = 0;
  static const int MaxThreadsPerBlock = 128;
  static const int MinBlocksPerMultiprocessor = 1;

  using ArchTag = cutlass::arch::Sm100;

  static_assert(kHeadDimLatent % MaxThreadsPerBlock == 0);
  struct Arguments {
    ElementAcc* ptr_oaccum = nullptr;
    ElementOut* ptr_o = nullptr;
    ElementAcc* ptr_lseaccum = nullptr;
    ElementAcc* ptr_lse = nullptr;
    ElementScale scale = 1.f;
    int num_batches = 0;
    int split_kv = -1;
    int dim_k = -1;
    int* ptr_seq = nullptr;
    int* ptr_split_kv = nullptr;
    int tile_shape_s = 128;
  };
  using Params = Arguments;

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {args.ptr_oaccum, args.ptr_o,        args.ptr_lseaccum, args.ptr_lse,
            args.scale,      args.num_batches,  args.split_kv,     args.dim_k,
            args.ptr_seq,    args.ptr_split_kv, args.tile_shape_s};
  }

  static size_t get_workspace_size(Arguments const& /*args*/) { return 0; }

  static Status initialize_workspace(Arguments const& /*args*/, void* /*ws*/,
                                     cudaStream_t /*stream*/) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3(kNumHeads, 1, params.num_batches);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  static bool can_implement(Arguments const& args) {
    if (args.num_batches <= 0) return false;
    if (args.split_kv <= 0) return false;
    return true;
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_raw) {
    if (params.split_kv <= 1) return;
    auto blk_coord = make_coord(blockIdx.x, _0{}, blockIdx.z);

    __shared__ ElementAcc sLseScale[kMaxSplits];
    const size_t offset_lseaccum =
        get<0>(blk_coord) + kNumHeads * params.split_kv * get<2>(blk_coord);
    const size_t offset_lse = get<0>(blk_coord) + kNumHeads * get<2>(blk_coord);

    Tensor gLSEaccum = make_tensor(make_gmem_ptr(params.ptr_lseaccum + offset_lseaccum),
                                   make_shape(params.split_kv), Stride<Int<kNumHeads>>{});

    Tensor gLSE =
        make_tensor(make_gmem_ptr(params.ptr_lse + offset_lse), Shape<_1>{}, Stride<_1>{});

    auto dim_k = params.ptr_seq == nullptr ? params.dim_k : params.ptr_seq[get<2>(blk_coord)];
    auto local_split_kv =
        params.ptr_split_kv == nullptr ? params.split_kv : params.ptr_split_kv[get<2>(blk_coord)];
    auto k_tile_total = ceil_div(dim_k, params.tile_shape_s);
    auto k_tile_per_cta = ceil_div(k_tile_total, local_split_kv);
    local_split_kv = ceil_div(k_tile_total, k_tile_per_cta);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0) {
      constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, 32);

      ElementAcc local_lse[kNLsePerThread];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        const int split = i * 32 + threadIdx.x;
        local_lse[i] = split < local_split_kv ? gLSEaccum(split)
                                              : -std::numeric_limits<ElementAcc>::infinity();
      }

      ElementAcc lse_max = -std::numeric_limits<ElementAcc>::infinity();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        lse_max = max(lse_max, local_lse[i]);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int offset = 16; offset >= 1; offset /= 2) {
        lse_max = max(lse_max, __shfl_xor_sync(0xffffffff, lse_max, offset));
      }
      lse_max = lse_max == -std::numeric_limits<ElementAcc>::infinity()
                    ? 0.0f
                    : lse_max;  // In case all local LSEs are -inf
      lse_max = __shfl_sync(0xffffffff, lse_max, 0);

      ElementAcc sum_lse = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        sum_lse = sum_lse + expf(local_lse[i] - lse_max);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int offset = 16; offset >= 1; offset /= 2) {
        sum_lse = sum_lse + __shfl_xor_sync(0xffffffff, sum_lse, offset);
      }

      sum_lse = __shfl_sync(0xffffffff, sum_lse, 0);

      ElementAcc global_lse = (sum_lse == 0.f || sum_lse != sum_lse)
                                  ? std::numeric_limits<ElementAcc>::infinity()
                                  : logf(sum_lse) + lse_max;
      if (threadIdx.x == 0 and params.ptr_lse != nullptr) {
        gLSE(0) = global_lse;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        const int split = i * 32 + threadIdx.x;
        if (split < local_split_kv) {
          sLseScale[split] = expf(local_lse[i] - global_lse);
        }
      }
    }
    __syncthreads();

    constexpr int Elements = kHeadDimLatent / MaxThreadsPerBlock;
    const size_t offset_oaccum =
        kHeadDimLatent * params.split_kv * (get<0>(blk_coord) + kNumHeads * get<2>(blk_coord));
    Tensor gOaccum = make_tensor(make_gmem_ptr(params.ptr_oaccum + offset_oaccum),
                                 Shape<Int<kHeadDimLatent>>{}, Stride<_1>{});
    ElementAcc local_val[Elements] = {0};
    for (int split = 0; split < local_split_kv; ++split) {
      ElementAcc lse_scale = sLseScale[split];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Elements; ++i) {
        local_val[i] += lse_scale * gOaccum(threadIdx.x + MaxThreadsPerBlock * i);
      }
      gOaccum.data() = gOaccum.data() + kHeadDimLatent;
    }
    auto ptr_o_local =
        params.ptr_o + (get<0>(blk_coord) + get<2>(blk_coord) * kNumHeads) * kHeadDimLatent;
    Tensor gO = make_tensor(make_gmem_ptr(ptr_o_local), Shape<Int<kHeadDimLatent>>{}, Stride<_1>{});

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Elements; ++i) {
      gO(threadIdx.x + MaxThreadsPerBlock * i) = static_cast<ElementOut>(local_val[i]);
    }
  }
};

}  // namespace cutlass::fmha::kernel


//////////////////////////////////////////////////////////////////////////////
// Section 5: Main MLA Kernel
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::kernel {

using namespace cute;

template <class TileShape, class Element_, class ElementAcc_, class ElementOut_, class ElementLSE_,
          class TileScheduler,
#ifdef CPASYNC
          bool kIsCpAsync = true
#else
          bool kIsCpAsync = false
#endif
          >
struct Sm100FmhaMlaKernelTmaWarpspecialized {

  using Element = Element_;
  using ElementAcc = ElementAcc_;
  using ElementOut = ElementOut_;
  using ElementLSE = ElementLSE_;

  // only 2Sm mode is supported
  static const bool kIs2Sm = true;
  static const int MaxThreadsPerBlock = 256;
  static const int MinBlocksPerMultiprocessor = 1;
  static const int TotalSNum = 2;
  static const int TotalPNum = 2;
  using ArchTag = cutlass::arch::Sm100;

  using ClusterShape = cute::conditional_t<kIs2Sm, Shape<_2, _1, _1>, Shape<_1, _1, _1>>;

  using TileShapeH = tuple_element_t<0, TileShape>;
  using TileShapeS = tuple_element_t<1, TileShape>;
  using TileShapeD = tuple_element_t<2, TileShape>;

  using TileShapeL = tuple_element_t<0, TileShapeD>;
  using TileShapeR = tuple_element_t<1, TileShapeD>;
  static_assert(TileShapeL{} % TileShapeR{} == 0, "Rope head dim must divide latent head dim");

  using ProblemShape = Shape<TileShapeH, int, TileShapeD, int>;
  using TensorStride = Stride<int64_t, _1, int64_t>;
  using TmemAllocator =
      cute::conditional_t<kIs2Sm, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

  static_assert(TileShapeH{} == 128);
  static const int kWarpsInN = kIs2Sm ? 2 : 1;

  static const int kNumComputeWarps = 4;
  static const int kNumLoadWarps = kIsCpAsync ? 2 : 1;

  enum class WarpRole {
    kMma = 0x1,
    kLoad = 0x2,
    kCompute = 0x3,
    kLoadPageTable = 0x4,
    kEmpty = 0x0
  };

  static const long long unsigned int kWarpAssignment =
      kIsCpAsync ? 0x4221'3333ull : 0x0021'3333ull;

  static CUTLASS_DEVICE WarpRole warp_idx_to_role(int warp_idx) {
    return static_cast<WarpRole>((kWarpAssignment >> (4 * warp_idx)) & 0xF);
  }

  //kcpsync is true   warp_idx : hex : role
  // --------------------------
  // 0 : 0x3 : kCompute
  // 1 : 0x3 : kCompute
  // 2 : 0x3 : kCompute
  // 3 : 0x3 : kCompute
  // 4 : 0x1 : kMma
  // 5 : 0x2 : kLoad
  // 6 : 0x2 : kLoad
  // 7 : 0x4 : kLoadPageTable


  //kcpsync is false   warp_idx : hex : role
  // --------------------------
  // 0 : 0x3 : kCompute
  // 1 : 0x3 : kCompute
  // 2 : 0x3 : kCompute
  // 3 : 0x3 : kCompute
  // 4 : 0x1 : kMma
  // 5 : 0x2 : kLoad
  // 6 : 0x0 : kEmpty
  // 7 : 0x0 : kEmpty


  static const int Alignment = 128 / sizeof_bits_v<Element>;
  static const int AlignmentOut = 128 / sizeof_bits_v<ElementOut>;

  using TileShapeQK = Shape<TileShapeH, TileShapeS, decltype(TileShapeR{} / _1{})>;
  static const int StagesQK = 24 / sizeof(Element);  // free parameter
  static const int IterationsQKLatent = decltype(TileShapeL{} / get<2>(TileShapeQK{}))::value;
  static const int IterationsQKRope = decltype(TileShapeR{} / get<2>(TileShapeQK{}))::value;
  static const int IterationsQK = IterationsQKLatent + IterationsQKRope;

  using Schedule = cute::conditional_t<kIs2Sm, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                                       cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element, TensorStride, Alignment,
      Element, TensorStride, Alignment, ElementAcc, TileShapeQK, ClusterShape,
      cutlass::gemm::collective::StageCount<StagesQK>, Schedule>::CollectiveOp;
  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  using CtaShapeQK = typename CollectiveMmaQK::CtaShape_MNK;

  // chosen for unified smem staging between K and V
  using TileShapePV = Shape<TileShapeH, _256, _32>;
  using TransposeTensorStride = decltype(select<1, 0, 2>(TensorStride{}));
  static const int StagesPV = StagesQK;  // not sure why, but must be at least two. check pipes
  static const int IterationsPV_K = decltype(TileShapeS{} / get<2>(TileShapePV{}))::value;
  static const int IterationsPV_N = decltype(TileShapeL{} / get<1>(TileShapePV{}))::value;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element, TensorStride, Alignment,
      Element, TransposeTensorStride, Alignment, ElementAcc, TileShapePV, ClusterShape,
      cutlass::gemm::collective::StageCount<StagesPV>, Schedule>::CollectiveOp;
  using CtaShapePV = typename CollectiveMmaPV::CtaShape_MNK;
  static_assert(std::is_same_v<TransposeTensorStride, typename CollectiveMmaPV::StrideB>);

  using TiledMmaPV = typename CollectiveMmaPV::TiledMma;

  using AtomThrShapeMNK = typename CollectiveMmaQK::AtomThrShapeMNK;
  static_assert(typename CollectiveMmaQK::AtomThrShapeMNK{} ==
                    typename CollectiveMmaPV::AtomThrShapeMNK{},
                "schedule must match");
  
  static const int StagesPageTable = kIsCpAsync ? StagesPV : 1;

  // pipelines from load to mma, PipelineTmaUmmaAsync, stages tbd
  // use expect_tx for Q load
  using PipelineLoadQK =
      cute::conditional_t<kIsCpAsync, PipelineUmmaConsumerAsync<StagesQK, AtomThrShapeMNK>,
                          PipelineTmaUmmaAsync<StagesQK, ClusterShape, AtomThrShapeMNK>>;
  using PipelineLoadPV = PipelineLoadQK;
  // pipeline from mma (Q@K) to softmax, PipelineUmmaAsync, 2 stages
  using PipelineS = PipelineUmmaAsync<TotalSNum, AtomThrShapeMNK>;
  // pipeline from softmax (P) to mma (bmm2), PipelineUmmaAsync, 2 stages
  using PipelineP = PipelineUmmaConsumerAsync<TotalPNum, AtomThrShapeMNK>;
  // pipeline from mma to softmax (for rescale), PipelineUmmaAsync, 1 stage
  using PipelineO = PipelineUmmaAsync<1, AtomThrShapeMNK>;

  using PipelinePT = PipelineAsync<StagesPageTable>;

  struct PipelineStorage {
    alignas(16) typename PipelineLoadQK::SharedStorage load_qk;
    alignas(16) typename PipelineS::SharedStorage mma_s;
    alignas(16) typename PipelineP::SharedStorage p_mma;
    alignas(16) typename PipelineO::SharedStorage mma_o;
    alignas(16) typename PipelinePT::SharedStorage load_page_table;
  };

  template <class Layout, class Stages = _1>
  static CUTE_DEVICE constexpr auto unstageSmemLayout(Layout const& layout, Stages stages = {}) {
    return composition(layout, make_tuple(_, _, _, make_layout(stages)));
  }

  using SmemLayoutQ =
      decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<IterationsQK>{}));
  using SmemLayoutKC = typename CollectiveMmaQK::SmemLayoutB;
  using SmemLayoutVC = typename CollectiveMmaPV::SmemLayoutB;
  using SmemLayoutP = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutA{},
                                                 make_shape(Int<IterationsPV_K>{}, _2{})));
  using SmemLayoutA = typename CollectiveMmaQK::SmemLayoutA;

  static const int kBytesLoadQ =
      size(AtomThrShapeMNK{}) *
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);
  static const int kBytesLoadKC =
      size(AtomThrShapeMNK{}) *
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutKC{})) * cute::sizeof_bits_v<Element>);
  static const int kBytesLoadVC =
      size(AtomThrShapeMNK{}) *
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutVC{})) * cute::sizeof_bits_v<Element>);
  // pre-condition for overlapped smem staging
  static_assert(kBytesLoadKC == kBytesLoadVC);
  static_assert(StagesQK == StagesPV);

  static const int kTransactionsBytesLoadQK = kBytesLoadKC;
  static const int kTransactionsBytesLoadExtraQ = kBytesLoadQ;
  static const int kTransactionsBytesLoadPV = kBytesLoadVC;

  static const int kNamedBarrierExchange =
      (int)cutlass::arch::ReservedNamedBarriers::TransformBarrier;
  // This Named Barrier is introduced to solve Q tile loading overwritten issue when enable
  // persistent tile scheduler for FP8 MLA.
  static const int kNamedBarrierEpilogue =
      (int)cutlass::arch::ReservedNamedBarriers::EpilogueBarrier;
  //
  static const int kNamedBarrierTmemDealloc =
      (int)cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier;

  enum class TmemAllocation : uint32_t {
    kSizeS = TileShapeS::value / kWarpsInN,
    // Overall
    kSizeO = TileShapeL::value / kWarpsInN,
    // Between accumulators we loop over
    kSizeAccO = decltype(get<1>(TileShapePV{}))::value / kWarpsInN,
    kNumS = TotalSNum,
    kNumP = TotalPNum,
    kNumO = 1,
    kS0 = 0,
    kS1 = kS0 + kSizeS,
    kO0 = kS1 + kSizeS,
    kTotal = kO0 + kSizeO
  };

  static_assert(static_cast<int>(TmemAllocation::kTotal) <= TmemAllocator::Sm100TmemCapacityColumns,
                "using too much tmem");

  struct TensorStorage {
    // to communicate max and row_sum
    cute::array<ElementAcc, kNumComputeWarps * cutlass::NumThreadsPerWarp> smem_exchange;
    cute::array<int, StagesPageTable * TileShapeS::value> smem_page_table;
    alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutKC>> smem_kc;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutVC>> smem_vc;
    };
    alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutP>> smem_p;
  };

  struct SharedStorage {
    PipelineStorage pipelines;
    TensorStorage tensors;
    uint32_t tmem_base_ptr;
  };

  static const int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes,
                "using too much smem");

  struct MainloopArguments {
    ElementAcc softmax_scale;

    // all tensors strides are (num_heads or seqlen, head_dim, batch)
    // head_dim stride is always 1
    Element* ptr_q_latent;
    TensorStride stride_q_latent;
    Element* ptr_q_rope;
    TensorStride stride_q_rope;

    Element* ptr_c_latent;
    TensorStride stride_c_latent;
    Element* ptr_k_rope;
    TensorStride stride_k_rope;

    // for paged attention, we interpret what was previously [batch, seqlen]
    // as [page_count, page_size], and index according to page_table
    int* ptr_seq = nullptr;
    int* ptr_page_table = nullptr;
    // page table is [batch, seqlen or similar]
    Stride<_1, int> stride_page_table = {};
    int page_count = 0;
    int page_size = TileShapeS{};  // powers of two if kIsCpAsync, otherwise TileShapeS
  };

  struct EpilogueArguments {
    ElementOut* ptr_o = nullptr;
    TensorStride stride_o;
    ElementLSE* ptr_lse = nullptr;
    Stride<_1, int> stride_lse;
    ElementAcc output_scale = 1.0f;
  };

  struct Arguments {
    // (num_heads=128, seqlen, (d_latent=512, d_rope=64), batch_count)
    // for paged attention, seqlen is max seqlen
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
    KernelHardwareInfo hw_info;
    int split_kv = -1;
    int* ptr_split_kv = nullptr;
  };

  using TmaLoadQLatent = typename CollectiveMmaQK::Params::TMA_A;
  using TmaLoadQRope = typename CollectiveMmaQK::Params::TMA_A;
  using TmaLoadCLatent = typename CollectiveMmaQK::Params::TMA_B;
  using TmaLoadKRope = typename CollectiveMmaQK::Params::TMA_B;
  using TmaLoadCLatentTranspose = typename CollectiveMmaPV::Params::TMA_B;

  struct MainloopParams {
    TmaLoadQLatent tma_load_q_latent;
    TmaLoadQRope tma_load_q_rope;
    TmaLoadCLatent tma_load_c_latent;
    TmaLoadKRope tma_load_k_rope;
    TmaLoadCLatentTranspose tma_load_c_latent_transpose;
  };

  struct EpilogueParams {
    ElementOut* ptr_o = nullptr;
    ElementAcc* ptr_o_acc = nullptr;
    TensorStride stride_o;
    TensorStride stride_o_acc;
    ElementLSE* ptr_lse = nullptr;
    ElementLSE* ptr_lse_acc = nullptr;
    Stride<_1, int> stride_lse;
    Stride<_1, int> stride_lse_acc;
    ElementAcc output_scale = 1.0f;
  };

  struct Params {
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueParams epilogue;
    MainloopParams mainloop_params;
    typename TileScheduler::Params tile_scheduler;
    int split_kv = -1;
    int* ptr_split_kv = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    // workspace = nullptr;  // let's get an error if one of these needs workspace

    auto [H, K, D, B] = args.problem_shape;
    auto [L, R] = D;

    int paged_B = B;
    int paged_K = K;
    if (args.mainloop.ptr_page_table != nullptr) {
      paged_B = args.mainloop.page_count;
      paged_K = args.mainloop.page_size;
    }

    auto params_qk_latent =
        CollectiveMmaQK::to_underlying_arguments(make_shape(H, K, L, B),
                                                 typename CollectiveMmaQK::Arguments{
                                                     args.mainloop.ptr_q_latent,
                                                     args.mainloop.stride_q_latent,
                                                     args.mainloop.ptr_c_latent,
                                                     args.mainloop.stride_c_latent,
                                                 },
                                                 nullptr);

    auto params_qk_latent_paged =
        CollectiveMmaQK::to_underlying_arguments(make_shape(H, paged_K, L, paged_B),
                                                 typename CollectiveMmaQK::Arguments{
                                                     args.mainloop.ptr_q_latent,
                                                     args.mainloop.stride_q_latent,
                                                     args.mainloop.ptr_c_latent,
                                                     args.mainloop.stride_c_latent,
                                                 },
                                                 nullptr);

    auto params_qk_rope =
        CollectiveMmaQK::to_underlying_arguments(make_shape(H, K, R, B),
                                                 typename CollectiveMmaQK::Arguments{
                                                     args.mainloop.ptr_q_rope,
                                                     args.mainloop.stride_q_rope,
                                                     args.mainloop.ptr_k_rope,
                                                     args.mainloop.stride_k_rope,
                                                 },
                                                 nullptr);

    auto params_qk_rope_paged =
        CollectiveMmaQK::to_underlying_arguments(make_shape(H, paged_K, R, paged_B),
                                                 typename CollectiveMmaQK::Arguments{
                                                     args.mainloop.ptr_q_rope,
                                                     args.mainloop.stride_q_rope,
                                                     args.mainloop.ptr_k_rope,
                                                     args.mainloop.stride_k_rope,
                                                 },
                                                 nullptr);

    auto stride_c_latent_transpose = select<1, 0, 2>(args.mainloop.stride_c_latent);
    auto params_pv_latent = CollectiveMmaPV::to_underlying_arguments(
        make_shape(H, L, paged_K, paged_B),
        typename CollectiveMmaPV::Arguments{
            args.mainloop.ptr_q_latent,
            args.mainloop.stride_q_latent,  // dummy, never used
            args.mainloop.ptr_c_latent,
            stride_c_latent_transpose,
        },
        nullptr);

    MainloopParams mainloop_params{params_qk_latent.tma_load_a, params_qk_rope.tma_load_a,
                                   params_qk_latent_paged.tma_load_b,
                                   params_qk_rope_paged.tma_load_b, params_pv_latent.tma_load_b};

    EpilogueParams epilogue_params;

    epilogue_params.ptr_o = args.epilogue.ptr_o;
    epilogue_params.stride_o = args.epilogue.stride_o;
    epilogue_params.ptr_lse = args.epilogue.ptr_lse;
    epilogue_params.stride_lse = args.epilogue.stride_lse;
    epilogue_params.output_scale = args.epilogue.output_scale;

    if (args.split_kv > 1) {
      ElementAcc* ptr_o_acc = reinterpret_cast<ElementAcc*>(workspace);
      ElementLSE* ptr_lse_acc =
          reinterpret_cast<ElementLSE*>(ptr_o_acc + H * L * args.split_kv * B);
      epilogue_params.ptr_o_acc = ptr_o_acc;
      epilogue_params.ptr_lse_acc = ptr_lse_acc;

      epilogue_params.stride_o_acc = make_tuple(static_cast<int64_t>(0 + L) * args.split_kv, _1{},
                                                static_cast<int64_t>(0 + H * L) * args.split_kv);
      epilogue_params.stride_lse_acc = make_tuple(_1{}, (0 + H) * args.split_kv);
    }

    return {args.problem_shape,
            args.mainloop,
            epilogue_params,
            mainloop_params,
            TileScheduler::to_underlying_arguments(args.problem_shape, args.hw_info, ClusterShape{},
                                                   args.split_kv),
            args.split_kv,
            args.ptr_split_kv};
  }

  static size_t get_workspace_size(Arguments const& args) {
    ProblemShape problem_shape = args.problem_shape;
    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;
    auto split_kv = args.split_kv;
    return (sizeof(ElementAcc) * D_latent + sizeof(ElementLSE)) * H * split_kv * B;
  }
  static Status initialize_workspace(Arguments const& /*args*/, void* /*ws*/,
                                     cudaStream_t /*stream*/) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.tile_scheduler);
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static bool can_implement(Arguments const& args) {
    if (kIsCpAsync) {
      if ((args.mainloop.page_size & (args.mainloop.page_size - 1)) != 0) {
        return false;
      }
      if (args.mainloop.page_size > TileShapeS{}) {
        return false;
      }
    } else {
      if (args.mainloop.ptr_page_table != nullptr && args.mainloop.page_size != TileShapeS{}) {
        return false;
      }
    }
    if (get<0>(args.problem_shape) != 128) {
      return false;
    }
    if (get<1>(args.problem_shape) <= 0) {
      return false;
    }
    if (args.split_kv <= 0) {
      return false;
    }
    return true;
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_raw) {
    TileScheduler tile_scheduler(params.tile_scheduler);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto role = warp_idx_to_role(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();

    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    int cta_coord_v = cta_rank_in_cluster % size<0>(AtomThrShapeMNK{});
    bool is_mma_leader_cta = cta_coord_v == 0;

    if (role == WarpRole::kLoad && lane_predicate && !kIsCpAsync) {
      prefetch_tma_descriptor(params.mainloop_params.tma_load_q_latent.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_c_latent.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_q_rope.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_k_rope.get_tma_descriptor());
      prefetch_tma_descriptor(
          params.mainloop_params.tma_load_c_latent_transpose.get_tma_descriptor());
    }
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_raw);

    typename PipelineLoadQK::Params pipeline_load_qk_params;
    if (role == WarpRole::kLoad) {
      pipeline_load_qk_params.role = PipelineLoadQK::ThreadCategory::Producer;
    }
    if (role == WarpRole::kMma) {
      pipeline_load_qk_params.role = PipelineLoadQK::ThreadCategory::Consumer;
    }
    if constexpr (kIsCpAsync) {
      // we can make our life easier by unconditionally loading blocks
      // since we know it'll always be legal
      pipeline_load_qk_params.producer_arv_count =
          kNumLoadWarps * cutlass::NumThreadsPerWarp * size(AtomThrShapeMNK{});
    } else {
      pipeline_load_qk_params.is_leader =
          lane_predicate && (role == WarpRole::kLoad) && is_mma_leader_cta;
      pipeline_load_qk_params.transaction_bytes = kTransactionsBytesLoadQK;
    }
    pipeline_load_qk_params.initializing_warp = 0;
    PipelineLoadQK pipeline_load_qk(shared_storage.pipelines.load_qk, pipeline_load_qk_params,
                                    ClusterShape{}, /*barrier init*/ cute::true_type{},
                                    /*mask calc*/ cute::false_type{});

    typename PipelineS::Params pipeline_mma_s_params;
    if (role == WarpRole::kMma) {
      pipeline_mma_s_params.role = PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::kCompute) {
      pipeline_mma_s_params.role = PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s_params.consumer_arv_count =
        kNumComputeWarps * cutlass::NumThreadsPerWarp * size(AtomThrShapeMNK{});
    pipeline_mma_s_params.initializing_warp = 1;
    PipelineS pipeline_mma_s(shared_storage.pipelines.mma_s, pipeline_mma_s_params, ClusterShape{},
                             /*barrier init*/ cute::true_type{}, /*mask calc*/ cute::false_type{});

    typename PipelineP::Params pipeline_p_mma_params;
    if (role == WarpRole::kMma) {
      pipeline_p_mma_params.role = PipelineP::ThreadCategory::Consumer;
    }
    if (role == WarpRole::kCompute) {
      pipeline_p_mma_params.role = PipelineP::ThreadCategory::Producer;
    }
    pipeline_p_mma_params.producer_arv_count =
        kNumComputeWarps * cutlass::NumThreadsPerWarp * size(AtomThrShapeMNK{});
    pipeline_p_mma_params.consumer_arv_count = 1;
    pipeline_p_mma_params.initializing_warp = 2;
    PipelineP pipeline_p_mma(shared_storage.pipelines.p_mma, pipeline_p_mma_params, ClusterShape{},
                             /*barrier init*/ cute::true_type{}, /*mask calc*/ cute::false_type{});

    typename PipelineO::Params pipeline_mma_o_params;
    if (role == WarpRole::kMma) {
      pipeline_mma_o_params.role = PipelineO::ThreadCategory::Producer;
    }
    if (role == WarpRole::kCompute) {
      pipeline_mma_o_params.role = PipelineO::ThreadCategory::Consumer;
    }
    pipeline_mma_o_params.consumer_arv_count =
        kNumComputeWarps * cutlass::NumThreadsPerWarp * size(AtomThrShapeMNK{});
    pipeline_mma_o_params.initializing_warp = 3;
    PipelineO pipeline_mma_o(shared_storage.pipelines.mma_o, pipeline_mma_o_params, ClusterShape{},
                             /*barrier init*/ cute::true_type{}, /*mask calc*/ cute::false_type{});

    typename PipelinePT::Params pipeline_pt_params;
    if (role == WarpRole::kLoad) {
      pipeline_pt_params.role = PipelinePT::ThreadCategory::Consumer;
    }
    if (role == WarpRole::kLoadPageTable) {
      pipeline_pt_params.role = PipelinePT::ThreadCategory::Producer;
    }
    pipeline_pt_params.consumer_arv_count = kNumLoadWarps * cutlass::NumThreadsPerWarp;
    pipeline_pt_params.producer_arv_count = cutlass::NumThreadsPerWarp;
    pipeline_pt_params.initializing_warp = 4;
    PipelinePT pipeline_page_table(shared_storage.pipelines.load_page_table, pipeline_pt_params);

    TmemAllocator tmem_allocator;

    pipeline_init_arrive_relaxed(size(ClusterShape{}));

    pipeline_load_qk.init_masks(ClusterShape{});  // do we need an update here for 2Sm?
    pipeline_mma_s.init_masks(ClusterShape{});
    pipeline_p_mma.init_masks(ClusterShape{});
    pipeline_mma_o.init_masks(ClusterShape{});

    typename PipelineLoadQK::PipelineState pipeline_load_qk_consumer_state;
    typename PipelineLoadQK::PipelineState pipeline_load_qk_producer_state =
        cutlass::make_producer_start_state<PipelineLoadQK>();

    typename PipelineS::PipelineState pipeline_mma_s_consumer_state;
    typename PipelineS::PipelineState pipeline_mma_s_producer_state =
        cutlass::make_producer_start_state<PipelineS>();

    typename PipelineP::PipelineState pipeline_p_mma_consumer_state;
    typename PipelineP::PipelineState pipeline_p_mma_producer_state =
        cutlass::make_producer_start_state<PipelineP>();

    typename PipelineO::PipelineState pipeline_mma_o_consumer_state;
    typename PipelineO::PipelineState pipeline_mma_o_producer_state =
        cutlass::make_producer_start_state<PipelineO>();

    typename PipelinePT::PipelineState pipeline_pt_consumer_state;
    typename PipelinePT::PipelineState pipeline_pt_producer_state =
        cutlass::make_producer_start_state<PipelinePT>();

    pipeline_init_wait(size(ClusterShape{}));
    // if(blockIdx.x == 0 && threadIdx.x == 0){
    //   printf("Sm100FmhaMlaKernelTmaWarpspecialized launched with %d CTAs\n", kIs2Sm);
    // }

    if (role == WarpRole::kLoadPageTable) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();
        auto problem_shape = params.problem_shape;
        auto local_split_kv = params.split_kv;
        if (params.mainloop.ptr_seq != nullptr) {
          get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
          if (params.ptr_split_kv != nullptr) {
            local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
          }
        }
        if (local_split_kv <= get<3>(blk_coord)) continue;
        load_page_table(blk_coord, problem_shape, params.mainloop, shared_storage.tensors,
                        pipeline_page_table, pipeline_pt_producer_state, local_split_kv);
      }
    } else if (role == WarpRole::kLoad) {
      if constexpr (kIsCpAsync) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (; tile_scheduler.is_valid(); ++tile_scheduler) {
          auto blk_coord = tile_scheduler.get_block_coord();
          auto problem_shape = params.problem_shape;
          auto local_split_kv = params.split_kv;
          if (params.mainloop.ptr_seq != nullptr) {
            get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
            if (params.ptr_split_kv != nullptr) {
              local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
            }
          }
          if (local_split_kv <= get<3>(blk_coord)) continue;
          load_cpasync(blk_coord, problem_shape, params.mainloop, params.mainloop_params,
                       shared_storage.tensors, pipeline_load_qk, pipeline_load_qk_producer_state,
                       local_split_kv,
                       /* must be shared pipe */
                       pipeline_page_table, pipeline_pt_consumer_state);
          cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp,
                                      kNamedBarrierEpilogue)
              .arrive_and_wait();
        }
      } else {
        if (params.mainloop.ptr_page_table != nullptr) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (; tile_scheduler.is_valid(); ++tile_scheduler) {
            auto blk_coord = tile_scheduler.get_block_coord();
            auto problem_shape = params.problem_shape;
            auto local_split_kv = params.split_kv;
            if (params.mainloop.ptr_seq != nullptr) {
              get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
              if (params.ptr_split_kv != nullptr) {
                local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
              }
            }
            if (local_split_kv <= get<3>(blk_coord)) continue;
            load_tma</* paged= */ true>(
                blk_coord, problem_shape, params.mainloop, params.mainloop_params,
                shared_storage.tensors, pipeline_load_qk, pipeline_load_qk_producer_state,
                pipeline_load_qk, pipeline_load_qk_producer_state, local_split_kv);
            cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp,
                                        kNamedBarrierEpilogue)
                .arrive_and_wait();
          }
        } else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (; tile_scheduler.is_valid(); ++tile_scheduler) {
            auto blk_coord = tile_scheduler.get_block_coord();
            auto problem_shape = params.problem_shape;
            auto local_split_kv = params.split_kv;
            if (params.mainloop.ptr_seq != nullptr) {
              get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
              if (params.ptr_split_kv != nullptr) {
                local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
              }
            }
            if (local_split_kv <= get<3>(blk_coord)) continue;
            load_tma<false>(blk_coord, problem_shape, params.mainloop, params.mainloop_params,
                            shared_storage.tensors, pipeline_load_qk,
                            pipeline_load_qk_producer_state, pipeline_load_qk,
                            pipeline_load_qk_producer_state, local_split_kv);
            cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp,
                                        kNamedBarrierEpilogue)
                .arrive_and_wait();
          }
        }
      }
    } else if (role == WarpRole::kMma) {
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                              &shared_storage.tmem_base_ptr);
      __syncwarp();

      if (is_mma_leader_cta) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (; tile_scheduler.is_valid(); ++tile_scheduler) {
          auto blk_coord = tile_scheduler.get_block_coord();
          auto problem_shape = params.problem_shape;
          auto local_split_kv = params.split_kv;
          if (params.mainloop.ptr_seq != nullptr) {
            get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
            if (params.ptr_split_kv != nullptr) {
              local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
            }
          }
          if (local_split_kv <= get<3>(blk_coord)) continue;
          mma(blk_coord, problem_shape, shared_storage.tensors, pipeline_load_qk,
              pipeline_load_qk_consumer_state, pipeline_load_qk, pipeline_load_qk_consumer_state,
              pipeline_mma_s, pipeline_mma_s_producer_state, pipeline_p_mma,
              pipeline_p_mma_consumer_state, pipeline_mma_o, pipeline_mma_o_producer_state,
              local_split_kv);
        }
      }

      // cutlass::arch::NamedBarrier((kNumComputeWarps + 1) * NumThreadsPerWarp,
      // kNamedBarrierTmemDealloc).arrive_and_wait();

      // uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
      // tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    } else if (role == WarpRole::kCompute) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();
        auto problem_shape = params.problem_shape;
        auto split_kv = params.split_kv;
        auto local_split_kv = split_kv;
        if (params.mainloop.ptr_seq != nullptr) {
          get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
          if (params.ptr_split_kv != nullptr) {
            local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
          }
        }
        if (local_split_kv <= get<3>(blk_coord)) continue;
        compute(blk_coord, problem_shape,
                params.mainloop,  // for softmax_scale
                params.epilogue,
                shared_storage.tensors,  // for smem_comm
                pipeline_mma_s, pipeline_mma_s_consumer_state, pipeline_p_mma,
                pipeline_p_mma_producer_state, pipeline_mma_o, pipeline_mma_o_consumer_state,
                local_split_kv);
      }

      // cutlass::arch::NamedBarrier((kNumComputeWarps + 1) * NumThreadsPerWarp,
      // kNamedBarrierTmemDealloc).arrive();
    }

    cute::cluster_sync();
    cutlass::arch::NamedBarrier((kNumComputeWarps + 1) * NumThreadsPerWarp,
                                kNamedBarrierTmemDealloc)
        .arrive();
    if (role == WarpRole::kMma) {
      uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
      tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
  }

  template <class BlkCoord>
  CUTLASS_DEVICE void load_page_table(
      BlkCoord const& blk_coord, ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args, TensorStorage& shared_tensors,
      PipelinePT& pipeline_page_table,
      typename PipelinePT::PipelineState& pipeline_pt_producer_state, int const& split_kv) {
    auto [H, K, D, B] = problem_shape;
    int batch_coord = get<2>(blk_coord);

    auto mPT_l =
        make_tensor(make_gmem_ptr(mainloop_args.ptr_page_table),
                    make_shape(mainloop_args.page_count, B), mainloop_args.stride_page_table);
    auto mPT = mPT_l(_, batch_coord);

    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(blk_coord) * k_tile_per_cta;  // lower limit

    // if(threadIdx.x % 32 == 0 && blockIdx.x == 0){ 
    //   printf("blk_coord: ");
    //   print(blk_coord);
    //   printf("\nget 3 blk_coord: %d", get<3>(blk_coord));
    //   printf("\n ktile per cta: %d\n", k_tile_per_cta);
    //   printf("----------------------\n");
    // }
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {
      return;
    }

    auto page_size = Pow2{mainloop_args.page_size};
    auto pages_per_tile = Pow2{TileShapeS{} / page_size};
    int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarp;

#if 1
    for (; k_tile_count > 0; ++k_index, --k_tile_count) {
      pipeline_page_table.producer_acquire(pipeline_pt_producer_state);

      // assume a single warp

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < TileShapeS{}; i += cutlass::NumThreadsPerWarp) {
        int idx = i + thread_idx;
        bool guard = idx < pages_per_tile;
        int smem_idx = pipeline_pt_producer_state.index() * TileShapeS::value + idx;
        int pt_idx = pages_per_tile * k_index + idx;

        cutlass::arch::cp_async_zfill<sizeof(int), cutlass::arch::CacheOperation::Always>(
            &shared_tensors.smem_page_table[smem_idx], &mPT(pt_idx), guard);
      }

      pipeline_page_table.producer_commit(pipeline_pt_producer_state,
                                          cutlass::arch::cpasync_barrier_arrive);
      ++pipeline_pt_producer_state;
    }
#endif
  }

  struct Gather {
    int& page_table_stage;
    Pow2 pages_per_tile;
    const int* __restrict__ smem_page_table;

    CUTLASS_DEVICE int operator()(int idx) const {
      return smem_page_table[page_table_stage * TileShapeS::value + idx % pages_per_tile];
    }

    CUTLASS_DEVICE friend void print(Gather const&) { printf("<gather>"); }
  };

  template <class BlkCoord>
  CUTLASS_DEVICE void load_cpasync(
      BlkCoord const& blk_coord, ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args, MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors, PipelineLoadQK& pipeline_load,
      typename PipelineLoadQK::PipelineState& pipeline_load_producer_state, int const& split_kv,
      PipelinePT& pipeline_page_table,
      typename PipelinePT::PipelineState& pipeline_pt_consumer_state) {
    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    using X = Underscore;

    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(blk_coord) * k_tile_per_cta;  // lower limit
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {
      return;
    }

    // partition all tensors
    auto mQL = make_tensor(make_gmem_ptr(mainloop_args.ptr_q_latent), make_shape(H, D_latent, B),
                           mainloop_args.stride_q_latent);
    auto mQR = make_tensor(make_gmem_ptr(mainloop_args.ptr_q_rope), make_shape(H, D_rope, B),
                           mainloop_args.stride_q_rope);

    int paged_B = mainloop_args.page_count;
    auto paged_K = Pow2{mainloop_args.page_size};
    // auto mPT_l = make_tensor(make_gmem_ptr(mainloop_args.ptr_page_table), make_shape(paged_B, B),
    //                          mainloop_args.stride_page_table);

    int batch_coord = get<2>(blk_coord);
    // auto mPT = mPT_l(_, batch_coord);

    auto gQL = local_tile(mQL, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
    auto gQR = local_tile(mQR, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});

    ThrMMA cta_mma_qk = TiledMmaQK{}.get_slice(get<0>(blk_coord) % size(AtomThrShapeMNK{}));
    ThrMMA cta_mma_pv = TiledMmaPV{}.get_slice(get<0>(blk_coord) % size(AtomThrShapeMNK{}));

    auto tSgQL = cta_mma_qk.partition_A(gQL);
    auto tSgQR = cta_mma_qk.partition_A(gQR);

    Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    Tensor sKC = make_tensor(make_smem_ptr(shared_tensors.smem_kc.begin()), SmemLayoutKC{});
    Tensor sVC = make_tensor(make_smem_ptr(shared_tensors.smem_vc.begin()), SmemLayoutVC{});

    auto make_copy_for = [](auto sT) {
      auto rT_a = sT.layout()(_, _, _, _0{});
      auto rT = make_ordered_layout(shape(rT_a), stride(rT_a));
      auto threads = Int<kNumLoadWarps * cutlass::NumThreadsPerWarp>{};
      auto values = Int<sizeof(uint128_t) / sizeof(Element)>{};
      return make_cotiled_copy(
          Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
          make_ordered_layout(make_shape(threads, values), make_stride(_1{}, _0{})), rT);
    };

    // like cute::copy, but makes sure we do all page table lookups first
    auto copy_split = [](auto atom, auto src, auto dst) {
      auto src_v = group_modes<1, rank_v<decltype(src)>>(src);
      auto dst_v = group_modes<1, rank_v<decltype(dst)>>(dst);

      auto src_v_ptrs = make_tensor<Element*>(size<1>(src_v));
      for (int i = 0; i < size<1>(src_v); i++) {
        src_v_ptrs(i) = &src_v(_0{}, i);
      }

      for (int i = 0; i < size<1>(src_v); i++) {
        auto src_v_i = make_tensor(make_gmem_ptr(src_v_ptrs(i)), make_shape(shape<0>(src_v)),
                                   make_stride(make_stride(_1{}, _0{})));
        atom.call(src_v_i, dst_v(_, i));
      }
    };

    auto tiled_copy_q = make_copy_for(sQ);
    auto tiled_copy_kc = make_copy_for(sKC);
    auto tiled_copy_vc = make_copy_for(sVC);

    auto thr_copy_q =
        tiled_copy_q.get_thread_slice(threadIdx.x % (kNumLoadWarps * cutlass::NumThreadsPerWarp));
    auto thr_copy_kc =
        tiled_copy_kc.get_thread_slice(threadIdx.x % (kNumLoadWarps * cutlass::NumThreadsPerWarp));
    auto thr_copy_vc =
        tiled_copy_vc.get_thread_slice(threadIdx.x % (kNumLoadWarps * cutlass::NumThreadsPerWarp));

    auto tQsQ = thr_copy_q.partition_D(sQ);
    auto tQgQL = thr_copy_q.partition_S(tSgQL);
    auto tQgQR = thr_copy_q.partition_S(tSgQR);

    auto tKCsKC = thr_copy_kc.partition_D(sKC);
    auto tVCsVC = thr_copy_vc.partition_D(sVC);

    auto pipeline_pt_release_state = pipeline_pt_consumer_state;

    int page_table_stage = -1;
    Pow2 pages_per_tile{TileShapeS{} / paged_K};
    const int* __restrict__ smem_page_table = shared_tensors.smem_page_table.begin();
    Gather gather{page_table_stage, pages_per_tile, smem_page_table};

    auto mCL = make_tensor(
        make_gmem_ptr(mainloop_args.ptr_c_latent),
        ComposedLayout{
            make_layout(make_shape(make_shape(paged_K, paged_B), _1{}),
                        make_stride(make_stride(get<0>(mainloop_args.stride_c_latent),
                                                example::CustomStride(
                                                    gather, get<2>(mainloop_args.stride_c_latent))),
                                    get<1>(mainloop_args.stride_c_latent))),
            make_coord(_0{}, _0{}), make_identity_layout(make_shape(paged_K * paged_B, D_latent))});

    auto mKR = make_tensor(
        make_gmem_ptr(mainloop_args.ptr_k_rope),
        ComposedLayout{
            make_layout(make_shape(make_shape(paged_K, paged_B), _1{}),
                        make_stride(make_stride(get<0>(mainloop_args.stride_k_rope),
                                                example::CustomStride(
                                                    gather, get<2>(mainloop_args.stride_k_rope))),
                                    get<1>(mainloop_args.stride_k_rope))),
            make_coord(_0{}, _0{}), make_identity_layout(make_shape(paged_K * paged_B, D_latent))});

    auto mCLT = make_tensor(
        make_gmem_ptr(mainloop_args.ptr_c_latent),
        ComposedLayout{
            make_layout(
                make_shape(_1{}, make_shape(paged_K, paged_B)),
                make_stride(get<1>(mainloop_args.stride_c_latent),
                            make_stride(get<0>(mainloop_args.stride_c_latent),
                                        example::CustomStride(
                                            gather, get<2>(mainloop_args.stride_c_latent))))),
            make_coord(_0{}, _0{}), make_identity_layout(make_shape(D_latent, paged_K * paged_B))});

    auto gCL = local_tile(mCL, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto gKR = local_tile(mKR, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto gCLT = local_tile(mCLT, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});

    auto tSgCL = cta_mma_qk.partition_B(gCL);
    auto tSgKR = cta_mma_qk.partition_B(gKR);
    auto tOgCLT = cta_mma_pv.partition_B(gCLT);

    auto tKCgCL = thr_copy_kc.partition_S(tSgCL);
    auto tKCgKR = thr_copy_kc.partition_S(tSgKR);
    auto tVCgCLT = thr_copy_vc.partition_S(tOgCLT);

    // latent is first in memory, so let's load it first always
    // startup: alternate Q and K, set tx count appropriately, for k_idx = 0
    auto& pipeline_acquire_state = pipeline_load_producer_state;
    auto pipeline_commit_state = pipeline_acquire_state;
    int pipeline_offset = 0;

    for (int i = 0; i < StagesPV; i++) {
      cutlass::arch::cp_async_fence();
    }

    auto load_stage = [&](auto fn) {
      pipeline_load.producer_acquire(pipeline_acquire_state);
      fn(pipeline_acquire_state.index());
      cutlass::arch::cp_async_fence();

      ++pipeline_acquire_state;
      ++pipeline_offset;

      if (pipeline_offset == StagesPV - 1) {
        cutlass::arch::cp_async_wait<StagesPV - 1>();
        pipeline_load.producer_commit(pipeline_commit_state);
        ++pipeline_commit_state;
        --pipeline_offset;
      }
    };

    pipeline_page_table.consumer_wait(pipeline_pt_consumer_state);
    page_table_stage = pipeline_pt_consumer_state.index();
    ++pipeline_pt_consumer_state;

    // each Q/K tile consists of rope and latent
    for (int i = 0; i < IterationsQKLatent; i++) {
      load_stage([&](int index) {
        cute::copy(tiled_copy_q, tQgQL(_, _, _, _, _0{}, i, batch_coord), tQsQ(_, _, _, _, i));
        copy_split(tiled_copy_kc, tKCgCL(_, _, _, _, k_index, i), tKCsKC(_, _, _, _, index));
      });
    }

    for (int i = 0; i < IterationsQKRope; i++) {
      load_stage([&](int index) {
        cute::copy(tiled_copy_q, tQgQR(_, _, _, _, _0{}, i, batch_coord),
                   tQsQ(_, _, _, _, IterationsQKLatent + i));
        copy_split(tiled_copy_kc, tKCgKR(_, _, _, _, k_index, i), tKCsKC(_, _, _, _, index));
      });
    }

    k_index += 1;
    k_tile_count -= 1;

    // assume k_tile_count >= 1
    // perform K+Q load here
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      pipeline_page_table.consumer_wait(pipeline_pt_consumer_state);
      page_table_stage = pipeline_pt_consumer_state.index();
      ++pipeline_pt_consumer_state;

      for (int i = 0; i < IterationsQKLatent; i++) {
        load_stage([&](int index) {
          copy_split(tiled_copy_kc, tKCgCL(_, _, _, _, k_index, i), tKCsKC(_, _, _, _, index));
        });
      }

      for (int i = 0; i < IterationsQKRope; i++) {
        load_stage([&](int index) {
          copy_split(tiled_copy_kc, tKCgKR(_, _, _, _, k_index, i), tKCsKC(_, _, _, _, index));
        });
      }

      page_table_stage = pipeline_pt_release_state.index();

      for (int i = 0; i < IterationsPV_K; i++) {
        for (int j = 0; j < IterationsPV_N; j++) {
          load_stage([&](int index) {
            copy_split(tiled_copy_vc, tVCgCLT(_, _, _, _, j, IterationsPV_K * (k_index - 1) + i),
                       tVCsVC(_, _, _, _, index));
          });
        }
      }

      pipeline_page_table.consumer_release(pipeline_pt_release_state);
      ++pipeline_pt_release_state;

      k_index += 1;
      k_tile_count -= 1;
    }

    page_table_stage = pipeline_pt_release_state.index();

    for (int i = 0; i < IterationsPV_K; i++) {
      for (int j = 0; j < IterationsPV_N; j++) {
        load_stage([&](int index) {
          copy_split(tiled_copy_vc, tVCgCLT(_, _, _, _, j, IterationsPV_K * (k_index - 1) + i),
                     tVCsVC(_, _, _, _, index));
        });
      }
    }

    pipeline_page_table.consumer_release(pipeline_pt_release_state);
    ++pipeline_pt_release_state;

    while (pipeline_offset > 0) {
      cutlass::arch::cp_async_fence();

      cutlass::arch::cp_async_wait<StagesPV - 1>();
      pipeline_load.producer_commit(pipeline_commit_state);
      ++pipeline_commit_state;
      --pipeline_offset;
    }

    cutlass::arch::cp_async_wait<0>();
  }

  template <bool kIsPaged = false, class BlkCoord>
  CUTLASS_DEVICE void load_tma(
      BlkCoord const& blk_coord, ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args, MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors, PipelineLoadQK& pipeline_load_qk,
      typename PipelineLoadQK::PipelineState& pipeline_load_qk_producer_state,
      PipelineLoadPV& pipeline_load_pv,
      typename PipelineLoadPV::PipelineState& pipeline_load_pv_producer_state,
      int const& split_kv) {
    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(blk_coord) * k_tile_per_cta;  // lower limit
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {
      return;
    }

    using X = Underscore;

    // partition all tensors
    auto mQL = mainloop_params.tma_load_q_latent.get_tma_tensor(make_shape(H, D_latent, B));
    auto mQR = mainloop_params.tma_load_q_rope.get_tma_tensor(make_shape(H, D_rope, B));

    int paged_B = B;
    int paged_K = K;
    if constexpr (kIsPaged) {
      paged_B = mainloop_args.page_count;
      paged_K = mainloop_args.page_size;
    }
    auto mPT_l = make_tensor(make_gmem_ptr(mainloop_args.ptr_page_table), make_shape(paged_B, B),
                             mainloop_args.stride_page_table);

    auto mCL =
        mainloop_params.tma_load_c_latent.get_tma_tensor(make_shape(paged_K, D_latent, paged_B));
    auto mKR = mainloop_params.tma_load_k_rope.get_tma_tensor(make_shape(paged_K, D_rope, paged_B));

    auto mCLT = mainloop_params.tma_load_c_latent_transpose.get_tma_tensor(
        make_shape(D_latent, paged_K, paged_B));

    auto gQL = local_tile(mQL, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
    auto gQR = local_tile(mQR, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});

    auto gCL = local_tile(mCL, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto gKR = local_tile(mKR, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto gCLT = local_tile(mCLT, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});

    ThrMMA cta_mma_qk = TiledMmaQK{}.get_slice(get<0>(blk_coord) % size(AtomThrShapeMNK{}));
    ThrMMA cta_mma_pv = TiledMmaPV{}.get_slice(get<0>(blk_coord) % size(AtomThrShapeMNK{}));

    auto tSgQL = cta_mma_qk.partition_A(gQL);
    auto tSgQR = cta_mma_qk.partition_A(gQR);

    auto tSgCL = cta_mma_qk.partition_B(gCL);
    auto tSgKR = cta_mma_qk.partition_B(gKR);

    auto tOgCLT = cta_mma_pv.partition_B(gCLT);

    Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    Tensor sKC = make_tensor(make_smem_ptr(shared_tensors.smem_kc.begin()), SmemLayoutKC{});
    Tensor sVC = make_tensor(make_smem_ptr(shared_tensors.smem_vc.begin()), SmemLayoutVC{});

    auto [tQLgQL_mkl, tQsQ] =
        tma_partition(mainloop_params.tma_load_q_latent, _0{}, make_layout(_1{}),
                      group_modes<0, 3>(sQ), group_modes<0, 3>(tSgQL));

    auto [tQRgQR_mkl, tQsQ_ignore] =
        tma_partition(mainloop_params.tma_load_q_rope, _0{}, make_layout(_1{}),
                      group_modes<0, 3>(sQ), group_modes<0, 3>(tSgQR));

    auto [tCLgCL_nkl, tKCsKC] =
        tma_partition(mainloop_params.tma_load_c_latent, _0{}, make_layout(_1{}),
                      group_modes<0, 3>(sKC), group_modes<0, 3>(tSgCL));

    auto [tKRgKR_nkl, tKCsKC_ignore] =
        tma_partition(mainloop_params.tma_load_k_rope, _0{}, make_layout(_1{}),
                      group_modes<0, 3>(sKC), group_modes<0, 3>(tSgKR));

    auto [tCLTgCLT_nkl, tVCsVC] =
        tma_partition(mainloop_params.tma_load_c_latent_transpose, _0{}, make_layout(_1{}),
                      group_modes<0, 3>(sVC), group_modes<0, 3>(tOgCLT));

    uint16_t mcast_mask = 0;

    int batch_coord = get<2>(blk_coord);
    Tensor tQLgQL = tQLgQL_mkl(_, _, _, batch_coord);
    Tensor tQRgQR = tQRgQR_mkl(_, _, _, batch_coord);

    auto mPT = mPT_l(_, batch_coord);

    Tensor tCLgCL = tCLgCL_nkl(_, _, _, _);
    Tensor tKRgKR = tKRgKR_nkl(_, _, _, _);

    // careful: stage and k are swapped here!
    Tensor tCLTgCLT = tCLTgCLT_nkl(_, _, _, _);

    // latent is first in memory, so let's load it first always
    // startup: alternate Q and K, set tx count appropriately, for k_idx = 0

    // each Q/K tile consists of rope and latent
    for (int i = 0; i < IterationsQKLatent; i++) {
      pipeline_load_qk.producer_expect_transaction(pipeline_load_qk_producer_state,
                                                   kTransactionsBytesLoadExtraQ);
      pipeline_load_qk.producer_acquire(pipeline_load_qk_producer_state);
      auto tma_barrier = pipeline_load_qk.producer_get_barrier(pipeline_load_qk_producer_state);

      if (cute::elect_one_sync()) {
        // expect the extra bytes
        // load_qk ql
        cute::copy(mainloop_params.tma_load_q_latent.with(*tma_barrier, mcast_mask),
                   tQLgQL(_, _0{}, i), tQsQ(_, i));
        // load_qk cl
        if constexpr (kIsPaged) {
          cute::copy(mainloop_params.tma_load_c_latent.with(*tma_barrier, mcast_mask),
                     tCLgCL(_, _0{}, i, mPT(k_index)),
                     tKCsKC(_, pipeline_load_qk_producer_state.index()));
        } else {
          cute::copy(mainloop_params.tma_load_c_latent.with(*tma_barrier, mcast_mask),
                     tCLgCL(_, k_index, i, batch_coord),
                     tKCsKC(_, pipeline_load_qk_producer_state.index()));
        }
      }
      ++pipeline_load_qk_producer_state;
    }

    for (int i = 0; i < IterationsQKRope; i++) {
      pipeline_load_qk.producer_expect_transaction(pipeline_load_qk_producer_state,
                                                   kTransactionsBytesLoadExtraQ);
      pipeline_load_qk.producer_acquire(pipeline_load_qk_producer_state);
      auto tma_barrier = pipeline_load_qk.producer_get_barrier(pipeline_load_qk_producer_state);

      if (cute::elect_one_sync()) {
        // expect the extra bytes
        // load_qk ql
        cute::copy(mainloop_params.tma_load_q_rope.with(*tma_barrier, mcast_mask),
                   tQRgQR(_, _0{}, i), tQsQ(_, i + IterationsQKLatent));
        // load_qk cl
        if constexpr (kIsPaged) {
          cute::copy(mainloop_params.tma_load_k_rope.with(*tma_barrier, mcast_mask),
                     tKRgKR(_, _0{}, i, mPT(k_index)),
                     tKCsKC(_, pipeline_load_qk_producer_state.index()));
        } else {
          cute::copy(mainloop_params.tma_load_k_rope.with(*tma_barrier, mcast_mask),
                     tKRgKR(_, k_index, i, batch_coord),
                     tKCsKC(_, pipeline_load_qk_producer_state.index()));
        }
      }
      ++pipeline_load_qk_producer_state;
    }

    k_index += 1;
    k_tile_count -= 1;

    // assume k_tile_count >= 1
    // perform K+Q load here
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // perform K load
      for (int i = 0; i < IterationsQKLatent; i++) {
        pipeline_load_qk.producer_acquire(pipeline_load_qk_producer_state);
        auto tma_barrier = pipeline_load_qk.producer_get_barrier(pipeline_load_qk_producer_state);

        if (cute::elect_one_sync()) {
          // load_qk cl
          if constexpr (kIsPaged) {
            cute::copy(mainloop_params.tma_load_c_latent.with(*tma_barrier, mcast_mask),
                       tCLgCL(_, _0{}, i, mPT(k_index)),
                       tKCsKC(_, pipeline_load_qk_producer_state.index()));
          } else {
            cute::copy(mainloop_params.tma_load_c_latent.with(*tma_barrier, mcast_mask),
                       tCLgCL(_, k_index, i, batch_coord),
                       tKCsKC(_, pipeline_load_qk_producer_state.index()));
          }
        }
        ++pipeline_load_qk_producer_state;
      }

      for (int i = 0; i < IterationsQKRope; i++) {
        pipeline_load_qk.producer_acquire(pipeline_load_qk_producer_state);
        auto tma_barrier = pipeline_load_qk.producer_get_barrier(pipeline_load_qk_producer_state);

        if (cute::elect_one_sync()) {
          // load_qk cl
          if constexpr (kIsPaged) {
            cute::copy(mainloop_params.tma_load_k_rope.with(*tma_barrier, mcast_mask),
                       tKRgKR(_, _0{}, i, mPT(k_index)),
                       tKCsKC(_, pipeline_load_qk_producer_state.index()));
          } else {
            cute::copy(mainloop_params.tma_load_k_rope.with(*tma_barrier, mcast_mask),
                       tKRgKR(_, k_index, i, batch_coord),
                       tKCsKC(_, pipeline_load_qk_producer_state.index()));
          }
        }
        ++pipeline_load_qk_producer_state;
      }

      // prefetch next K load to keep busy while we transpose-load from cache
      const int kPrefetchDistance = 1;
      for (int i = 0; i < IterationsQKLatent; i++) {
        if (cute::elect_one_sync()) {
          if constexpr (kIsPaged) {
            if (k_tile_count > kPrefetchDistance) {
              cute::prefetch(mainloop_params.tma_load_c_latent,
                             tCLgCL(_, _0{}, i, mPT(k_index + kPrefetchDistance)));
            }
          } else {
            cute::prefetch(mainloop_params.tma_load_c_latent,
                           tCLgCL(_, k_index + kPrefetchDistance, i, batch_coord));
          }
        }
      }

      for (int i = 0; i < IterationsQKRope; i++) {
        if (cute::elect_one_sync()) {
          if constexpr (kIsPaged) {
            if (k_tile_count > kPrefetchDistance) {
              cute::prefetch(mainloop_params.tma_load_k_rope,
                             tKRgKR(_, _0{}, i, mPT(k_index + kPrefetchDistance)));
            }
          } else {
            cute::prefetch(mainloop_params.tma_load_k_rope,
                           tKRgKR(_, k_index + kPrefetchDistance, i, batch_coord));
          }
        }
      }

      // perform V load (k_idx - 1)

      for (int i = 0; i < IterationsPV_K; i++) {
        for (int j = 0; j < IterationsPV_N; j++) {
          pipeline_load_pv.producer_acquire(pipeline_load_pv_producer_state);
          auto tma_barrier = pipeline_load_pv.producer_get_barrier(pipeline_load_pv_producer_state);

          if (cute::elect_one_sync()) {
            // load_pv cl
            // note the transpose in indices!
            // note we are off-by-one on k_index
            if constexpr (kIsPaged) {
              cute::copy(mainloop_params.tma_load_c_latent_transpose.with(
                             *tma_barrier, mcast_mask, cute::TMA::CacheHintSm100::EVICT_FIRST),
                         tCLTgCLT(_, j, i, mPT(k_index - 1)),
                         tVCsVC(_, pipeline_load_pv_producer_state.index()));
            } else {
              cute::copy(mainloop_params.tma_load_c_latent_transpose.with(
                             *tma_barrier, mcast_mask, cute::TMA::CacheHintSm100::EVICT_FIRST),
                         tCLTgCLT(_, j, IterationsPV_K * (k_index - 1) + i, batch_coord),
                         tVCsVC(_, pipeline_load_pv_producer_state.index()));
            }
          }
          ++pipeline_load_pv_producer_state;
        }
      }

      k_index += 1;
      k_tile_count -= 1;
    }

    for (int i = 0; i < IterationsPV_K; i++) {
      for (int j = 0; j < IterationsPV_N; j++) {
        pipeline_load_pv.producer_acquire(pipeline_load_pv_producer_state);
        auto tma_barrier = pipeline_load_pv.producer_get_barrier(pipeline_load_pv_producer_state);

        if (cute::elect_one_sync()) {
          // load_pv cl
          // note the transpose in indices
          // note we are off-by-one on k_index

          if constexpr (kIsPaged) {
            cute::copy(mainloop_params.tma_load_c_latent_transpose.with(
                           *tma_barrier, mcast_mask, cute::TMA::CacheHintSm100::EVICT_FIRST),
                       tCLTgCLT(_, j, i, mPT(k_index - 1)),
                       tVCsVC(_, pipeline_load_pv_producer_state.index()));
          } else {
            cute::copy(mainloop_params.tma_load_c_latent_transpose.with(
                           *tma_barrier, mcast_mask, cute::TMA::CacheHintSm100::EVICT_FIRST),
                       tCLTgCLT(_, j, IterationsPV_K * (k_index - 1) + i, batch_coord),
                       tVCsVC(_, pipeline_load_pv_producer_state.index()));
          }
        }
        ++pipeline_load_pv_producer_state;
      }
    }
  }

  template <class BlkCoord>
  CUTLASS_DEVICE void mma(BlkCoord const& blk_coord, ProblemShape const& problem_shape,
                          TensorStorage& shared_tensors, PipelineLoadQK& pipeline_load_qk,
                          typename PipelineLoadQK::PipelineState& pipeline_load_qk_consumer_state,
                          PipelineLoadPV& pipeline_load_pv,
                          typename PipelineLoadPV::PipelineState& pipeline_load_pv_consumer_state,
                          PipelineS& pipeline_mma_s,
                          typename PipelineS::PipelineState& pipeline_mma_s_producer_state,
                          PipelineP& pipeline_p_mma,
                          typename PipelineP::PipelineState& pipeline_p_mma_consumer_state,
                          PipelineO& pipeline_mma_o,
                          typename PipelineO::PipelineState& pipeline_mma_o_producer_state,
                          int const& split_kv) {
    auto [H, K, D, B] = problem_shape;

    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(blk_coord) * k_tile_per_cta;  // lower limit
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {
      return;
    }

    // mma init
    Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    Tensor sKC = make_tensor(make_smem_ptr(shared_tensors.smem_kc.begin()), SmemLayoutKC{});
    Tensor sVC = make_tensor(make_smem_ptr(shared_tensors.smem_vc.begin()), SmemLayoutVC{});
    Tensor sP = make_tensor(make_smem_ptr((Element*)shared_tensors.smem_p.begin()), SmemLayoutP{});

    Tensor tSrQ = TiledMmaQK::make_fragment_A(sQ);
    Tensor tSrKC = TiledMmaQK::make_fragment_B(sKC);
    Tensor tOrP = TiledMmaPV::make_fragment_A(sP);
    Tensor tOrVC = TiledMmaPV::make_fragment_B(sVC);

    TiledMmaQK tiled_mma_qk;
    AtomThrShapeMNK atom_thr_shape_mnk ;
    SmemLayoutQ smem_layout_q ;
    SmemLayoutA smem_layout_a;
    // if(threadIdx.x %32 == 0 && blockIdx.x == 0){
    //   printf("TiledMmaQK Configuration:\n");
    //   // tiled_mma_qk.print_configuration();
    //   print(tiled_mma_qk);
    //   printf("\nAtomThrShapeMNK:");
    //   print(atom_thr_shape_mnk);
    //   printf("\n smem layout q:");
    //   print(smem_layout_q);
    //   printf("\n smem layout a:");
    //   print(smem_layout_a);

    // }
    TiledMmaPV tiled_mma_pv;

    Tensor tStS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShapeQK{}));
    Tensor tOtO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShapePV{}));

    tiled_mma_pv.accumulate_ = UMMA::ScaleOut::Zero;

    pipeline_mma_s.producer_acquire(pipeline_mma_s_producer_state);

    // Mma           S0 S1 O0 S2 O1 ... Sn On-1 On
    // S0 ownership  --    -----        --      --
    // S1 ownership     --       -----     ----
    // O ownership         --    --        ---- --

    tiled_mma_qk.accumulate_ = UMMA::ScaleOut::Zero;
    for (int i = 0; i < IterationsQK; i++) {
      pipeline_load_qk.consumer_wait(pipeline_load_qk_consumer_state);
      int read_stage = pipeline_load_qk_consumer_state.index();

      tStS.data() = uint32_t(pipeline_mma_s_producer_state.index() == 0 ? TmemAllocation::kS0
                                                                        : TmemAllocation::kS1);

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
        cute::gemm(tiled_mma_qk, tSrQ(_, _, k_block, i), tSrKC(_, _, k_block, read_stage), tStS);
        tiled_mma_qk.accumulate_ = UMMA::ScaleOut::One;
      }

      pipeline_load_qk.consumer_release(pipeline_load_qk_consumer_state);
      ++pipeline_load_qk_consumer_state;
    }

    pipeline_mma_s.producer_commit(pipeline_mma_s_producer_state);
    ++pipeline_mma_s_producer_state;

    k_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      pipeline_mma_s.producer_acquire(pipeline_mma_s_producer_state);
      tiled_mma_qk.accumulate_ = UMMA::ScaleOut::Zero;
      for (int i = 0; i < IterationsQK; i++) {
        pipeline_load_qk.consumer_wait(pipeline_load_qk_consumer_state);
        int read_stage = pipeline_load_qk_consumer_state.index();

        tStS.data() = uint32_t(pipeline_mma_s_producer_state.index() == 0 ? TmemAllocation::kS0
                                                                          : TmemAllocation::kS1);

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
          cute::gemm(tiled_mma_qk, tSrQ(_, _, k_block, i), tSrKC(_, _, k_block, read_stage), tStS);
          tiled_mma_qk.accumulate_ = UMMA::ScaleOut::One;
        }

        pipeline_load_qk.consumer_release(pipeline_load_qk_consumer_state);
        ++pipeline_load_qk_consumer_state;
      }

      pipeline_mma_s.producer_commit(pipeline_mma_s_producer_state);
      ++pipeline_mma_s_producer_state;

      pipeline_mma_o.producer_acquire(pipeline_mma_o_producer_state);
      pipeline_p_mma.consumer_wait(pipeline_p_mma_consumer_state);

      for (int i = 0; i < IterationsPV_K; i++) {
        auto acc_flag = tiled_mma_pv.accumulate_;
        for (int j = 0; j < IterationsPV_N; j++) {
          pipeline_load_pv.consumer_wait(pipeline_load_pv_consumer_state);

          int read_stage = pipeline_load_pv_consumer_state.index();

          tOtO.data() = uint32_t(TmemAllocation::kO0) + j * uint32_t(TmemAllocation::kSizeAccO);
          tiled_mma_pv.accumulate_ = acc_flag;

          CUTLASS_PRAGMA_UNROLL
          for (int k_block = 0; k_block < size<2>(tOrP); ++k_block) {
            cute::gemm(tiled_mma_pv,
                       tOrP(_, _, k_block, make_coord(i, pipeline_p_mma_consumer_state.index())),
                       tOrVC(_, _, k_block, read_stage), tOtO);
            tiled_mma_pv.accumulate_ = UMMA::ScaleOut::One;
          }

          pipeline_load_pv.consumer_release(pipeline_load_pv_consumer_state);
          ++pipeline_load_pv_consumer_state;
        }
      }

      pipeline_p_mma.consumer_release(pipeline_p_mma_consumer_state);
      ++pipeline_p_mma_consumer_state;
      pipeline_mma_o.producer_commit(pipeline_mma_o_producer_state);
      ++pipeline_mma_o_producer_state;

      --k_tile_count;
    }

    pipeline_mma_o.producer_acquire(pipeline_mma_o_producer_state);
    pipeline_p_mma.consumer_wait(pipeline_p_mma_consumer_state);

    for (int i = 0; i < IterationsPV_K; i++) {
      auto acc_flag = tiled_mma_pv.accumulate_;
      for (int j = 0; j < IterationsPV_N; j++) {
        pipeline_load_pv.consumer_wait(pipeline_load_pv_consumer_state);

        int read_stage = pipeline_load_pv_consumer_state.index();

        tOtO.data() = uint32_t(TmemAllocation::kO0) + j * uint32_t(TmemAllocation::kSizeAccO);
        tiled_mma_pv.accumulate_ = acc_flag;

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tOrP); ++k_block) {
          cute::gemm(tiled_mma_pv,
                     tOrP(_, _, k_block, make_coord(i, pipeline_p_mma_consumer_state.index())),
                     tOrVC(_, _, k_block, read_stage), tOtO);
          tiled_mma_pv.accumulate_ = UMMA::ScaleOut::One;
        }

        pipeline_load_pv.consumer_release(pipeline_load_pv_consumer_state);
        ++pipeline_load_pv_consumer_state;
      }
    }

    pipeline_p_mma.consumer_release(pipeline_p_mma_consumer_state);
    ++pipeline_p_mma_consumer_state;
    pipeline_mma_o.producer_commit(pipeline_mma_o_producer_state);
    ++pipeline_mma_o_producer_state;
  }

  template <class IsLastTile>
  CUTLASS_DEVICE void softmax(IsLastTile const& is_last_tile, ElementAcc& row_max,
                              ElementAcc& row_sum, ElementAcc& correction_factor,
                              ProblemShape const& problem_shape,
                              MainloopArguments const& mainloop_args, TensorStorage& shared_tensors,
                              int k_index, uint32_t tmem_s, int smem_p_index) {
    auto load_op = cute::SM100_TMEM_LOAD_32dp32b32x{};

    TiledMmaQK tiled_mma_qk;

    Tensor tStS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShapeQK{}));
    tStS.data() = tmem_s;

    CUTE_STATIC_ASSERT_V(shape<1>(tStS) == _1{});
    CUTE_STATIC_ASSERT_V(shape<2>(tStS) == _1{});
    Tensor tAcc = tStS(make_coord(_, _), _0{}, _0{});

    Tensor cS = make_identity_tensor(take<0, 2>(CtaShapeQK{}));

    auto tiled_t2r = make_tmem_copy(load_op, tAcc);
    auto thread_idx = threadIdx.x % size(tiled_t2r);

    auto thread_t2r = tiled_t2r.get_slice(thread_idx);
    Tensor tTR_cS = thread_t2r.partition_D(cS);
    Tensor tTR_rAcc = make_tensor<ElementAcc>(shape(tTR_cS));

    Tensor tTR_rS_frag = make_tensor<Element>(shape(tTR_rAcc));
    const int AlignmentS = 4;
    Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);
    Tensor tTR_rAcc_vec = recast<Array<ElementAcc, AlignmentS>>(tTR_rAcc);
    Tensor tTR_rS_vec = recast<Array<Element, AlignmentS>>(tTR_rS_frag);

    // load s
    copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

    if (is_last_tile) {
      for (int i = 0; i < size(tTR_rAcc); i++) {
        if (get<1>(tTR_cS(i)) + TileShapeS{} * k_index >= get<1>(problem_shape)) {
          tTR_rAcc(i) = -std::numeric_limits<ElementAcc>::infinity();
        }
      }
    }

    // max
    ElementAcc row_max_new = row_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rAcc); i += 1) {
      row_max_new = ::fmax(row_max_new, tTR_rAcc(i));
    }

    // for 2x2 dp, reduce here
    if constexpr (kWarpsInN > 1) {
      shared_tensors.smem_exchange[threadIdx.x] = row_max_new;
      cutlass::arch::NamedBarrier(kNumComputeWarps * NumThreadsPerWarp, kNamedBarrierExchange)
          .sync();
      // (64, 2) shape
      int peer_index = (threadIdx.x + 64) % 128;
      row_max_new = cutlass::max(row_max_new, shared_tensors.smem_exchange[peer_index]);
    }

#ifndef B2B
    // find correction factor
    ElementAcc softmax_scale_log2 = mainloop_args.softmax_scale * static_cast<ElementAcc>(M_LOG2E);
    correction_factor = ::exp2f(softmax_scale_log2 * (row_max - row_max_new));
    row_max = row_max_new;

    // softmax
    ElementAcc row_max_scale_log2 = row_max * softmax_scale_log2;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rAcc); i++) {
      tTR_rAcc(i) = ::exp2f(softmax_scale_log2 * tTR_rAcc(i) - row_max_scale_log2);
    }
#endif

    // quantize
    cutlass::NumericArrayConverter<Element, ElementAcc, AlignmentS> epilogue_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rAcc_vec); i++) {
      tTR_rS_vec(i) = epilogue_op(tTR_rAcc_vec(i));
    }

    Tensor sP = make_tensor(make_smem_ptr((Element*)shared_tensors.smem_p.begin()), SmemLayoutP{})(
        _, _, _, make_coord(_, smem_p_index));

    Tensor tOcP = TiledMmaPV{}.get_slice(_0{}).partition_A(cS);

    // have a mapping for each thread to coord
    // find identical mapping to coords for the MMA
    auto l = make_ordered_layout(
        make_shape(make_shape(_64{}, _2{}), make_shape(_16{}, TileShapeS{} / _32{})),
        make_stride(make_stride(_0{}, _3{}), make_stride(_1{}, _2{})));
    auto sP_ = as_position_independent_swizzle_tensor(sP);
    copy_aligned(tTR_rS_frag, sP_.compose(l)(threadIdx.x, _));

    // sum
    row_sum *= correction_factor;

    static_assert(cute::is_same_v<ElementAcc, float>);
    auto tTR_rAcc_float2 = recast<float2>(tTR_rAcc);
    auto sums = make_tensor<float2>(_4{});
    static_assert(size(tTR_rAcc_float2) % size(sums) == 0);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(sums); i++) {
      sums(i) = tTR_rAcc_float2(i);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = size(sums); i < size(tTR_rAcc_float2); i += size(sums)) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(sums); j++) {
        cute::add(sums(j), sums(j), tTR_rAcc_float2(i + j));
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < size(sums); i *= 2) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(sums); j += 2 * i) {
        cute::add(sums(j), sums(j), sums(j + i));
      }
    }
    row_sum += sums(0).x + sums(0).y;
  }

  CUTLASS_DEVICE void rescale(ElementAcc correction_factor, uint32_t tmem_o) {
    // for b2b gemm, do nothing
#ifndef B2B
    auto load_op = cute::SM100_TMEM_LOAD_32dp32b32x{};
    auto store_op = TMEM::tmem_load_to_store(load_op);

    TiledMmaPV tiled_mma_pv;

    Tensor tOtO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShapePV{}));
    tOtO.data() = tmem_o;

    CUTE_STATIC_ASSERT_V(shape<1>(tOtO) == _1{});
    CUTE_STATIC_ASSERT_V(shape<2>(tOtO) == _1{});
    Tensor tAcc = tOtO(make_coord(_, _), _0{}, _0{});

    auto cta_tiler_pv = take<0, 2>(typename CollectiveMmaPV::CtaShape_MNK{});
    Tensor gO = make_tensor(make_gmem_ptr((ElementAcc*)nullptr), cta_tiler_pv, make_stride(0, 0));

    auto tiled_t2r = make_tmem_copy(load_op, tAcc);
    auto tiled_r2t = make_tmem_copy(store_op, tAcc);
    auto thread_idx = threadIdx.x % size(tiled_t2r);

    auto thread_t2r = tiled_t2r.get_slice(thread_idx);
    auto thread_r2t = tiled_r2t.get_slice(thread_idx);
    Tensor tTR_gO = thread_t2r.partition_D(gO);
    Tensor tTR_rAcc = make_tensor<ElementAcc>(shape(tTR_gO));

    Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);

    // load o
    copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

    // multiply by correction factor
    float2 correction_factor_vec = make_float2(correction_factor, correction_factor);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rAcc); i += 2) {
      float2 in = make_float2(tTR_rAcc(i + 0), tTR_rAcc(i + 1));
      float2 out;
      cute::mul(out, in, correction_factor_vec);
      tTR_rAcc(i + 0) = out.x;
      tTR_rAcc(i + 1) = out.y;
    }

    // store o
    copy(tiled_r2t, tTR_rAcc, tTR_tAcc);
#endif
  }

  template <class BlkCoord>
  CUTLASS_DEVICE void epilogue(ElementAcc& row_max, ElementAcc& row_sum, BlkCoord const& cta_coord,
                               ProblemShape const& problem_shape,
                               MainloopArguments const& mainloop_args,
                               EpilogueParams const& epilogue_args, TensorStorage& shared_tensors,
                               uint32_t tmem_o, int const& split_kv) {
    auto load_op = cute::SM100_TMEM_LOAD_32dp32b32x{};

    TiledMmaPV tiled_mma_pv;

    Tensor tOtO =
        TiledMmaPV::make_fragment_C(partition_shape_C(TiledMmaPV{}, take<0, 2>(TileShapePV{})));
    tOtO.data() = tmem_o;

    CUTE_STATIC_ASSERT_V(shape<1>(tOtO) == _1{});
    CUTE_STATIC_ASSERT_V(shape<2>(tOtO) == _1{});
    Tensor tAcc = tOtO(make_coord(_, _), _0{}, _0{});

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;
    if (epilogue_args.ptr_o_acc != nullptr) {
      using ElementOutAcc = ElementAcc;
      constexpr auto AlignmentOutAcc = 128 / cute::sizeof_bits_v<ElementOutAcc>;
      Tensor mO = make_tensor(make_gmem_ptr(epilogue_args.ptr_o_acc + get<3>(cta_coord) * D_latent),
                              make_shape(H, D_latent, B), epilogue_args.stride_o_acc);
      auto cta_tiler_pv = take<0, 2>(typename CollectiveMmaPV::CtaShape_MNK{});
      Tensor gO = local_tile(mO, cta_tiler_pv, take<0, 3>(cta_coord));

      auto tiled_t2r = make_tmem_copy(load_op, tAcc);
      auto thread_idx = threadIdx.x % size(tiled_t2r);

      auto thread_t2r = tiled_t2r.get_slice(thread_idx);
      Tensor tTR_gO = thread_t2r.partition_D(gO);
      Tensor tTR_rAcc = make_tensor<ElementAcc>(shape(tTR_gO));

      Tensor tTR_rO_frag = make_tensor<ElementOutAcc>(shape(tTR_rAcc));
      Tensor tTR_rO_src = recast<Array<ElementOutAcc, AlignmentOutAcc>>(coalesce(tTR_rO_frag));
      Tensor tR2G_rO_dst = recast<Array<ElementOutAcc, AlignmentOutAcc>>(coalesce(tTR_gO));
      Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);

      copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

      cutlass::epilogue::thread::LinearCombination<
          ElementOutAcc, 1, ElementAcc, ElementAcc,
          cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>
          epilogue_op({epilogue_args.output_scale / row_sum});
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rAcc); i++) {
        tTR_rO_frag(i) = epilogue_op(tTR_rAcc(i));
      }

      copy(tTR_rO_src, tR2G_rO_dst);

#ifndef B2B

      // compute LSE
      ElementAcc lse = cutlass::fast_log(row_sum) + mainloop_args.softmax_scale * row_max;

      // store LSE
      Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_args.ptr_lse_acc + H * get<3>(cta_coord)),
                                make_shape(H, B), epilogue_args.stride_lse_acc);
      Tensor gLSE = local_tile(mLSE, append<3>(cta_tiler_pv, _1{}), take<0, 3>(cta_coord),
                               Step<_1, Underscore, _1>{});
      // for 2x2 dp, this must be conditional and the index is wrong
      if (!kIs2Sm || (threadIdx.x < 64)) {
        gLSE(threadIdx.x) = lse;
      }
#endif
    } else {
      Tensor mO = make_tensor(make_gmem_ptr(epilogue_args.ptr_o), make_shape(H, D_latent, B),
                              epilogue_args.stride_o);
      auto cta_tiler_pv = take<0, 2>(typename CollectiveMmaPV::CtaShape_MNK{});
      Tensor gO = local_tile(mO, cta_tiler_pv, take<0, 3>(cta_coord));

      auto tiled_t2r = make_tmem_copy(load_op, tAcc);
      auto thread_idx = threadIdx.x % size(tiled_t2r);

      auto thread_t2r = tiled_t2r.get_slice(thread_idx);
      Tensor tTR_gO = thread_t2r.partition_D(gO);
      Tensor tTR_rAcc = make_tensor<ElementAcc>(shape(tTR_gO));

      Tensor tTR_rO_frag = make_tensor<ElementOut>(shape(tTR_rAcc));
      Tensor tTR_rO_src = recast<Array<ElementOut, AlignmentOut>>(coalesce(tTR_rO_frag));
      Tensor tR2G_rO_dst = recast<Array<ElementOut, AlignmentOut>>(coalesce(tTR_gO));
      Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);

      copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

      cutlass::epilogue::thread::LinearCombination<
          ElementOut, 1, ElementAcc, ElementAcc,
          cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>
          epilogue_op({epilogue_args.output_scale / row_sum});
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rAcc); i++) {
        tTR_rO_frag(i) = epilogue_op(tTR_rAcc(i));
      }

      copy(tTR_rO_src, tR2G_rO_dst);

#ifndef B2B
      if (epilogue_args.ptr_lse != nullptr) {
        // compute LSE
        ElementAcc lse = cutlass::fast_log(row_sum) + mainloop_args.softmax_scale * row_max;

        // store LSE
        Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_args.ptr_lse), make_shape(H, B),
                                  epilogue_args.stride_lse);
        Tensor gLSE = local_tile(mLSE, append<3>(cta_tiler_pv, _1{}), take<0, 3>(cta_coord),
                                 Step<_1, Underscore, _1>{});

        // for 2x2 dp, this must be conditional and the index is wrong
        if (!kIs2Sm || (threadIdx.x < 64)) {
          gLSE(threadIdx.x) = lse;
        }
      }
#endif
    }
  }

  template <class CtaCoord>
  CUTLASS_DEVICE void compute(
      CtaCoord const& cta_coord, ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args, EpilogueParams const& epilogue_args,
      TensorStorage& shared_tensors, PipelineS& pipeline_mma_s,
      typename PipelineS::PipelineState& pipeline_mma_s_consumer_state, PipelineP& pipeline_p_mma,
      typename PipelineP::PipelineState& pipeline_p_mma_producer_state, PipelineO& pipeline_mma_o,
      typename PipelineO::PipelineState& pipeline_mma_o_consumer_state, int const& split_kv) {
    auto [H, K, D, B] = problem_shape;

    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(cta_coord) * k_tile_per_cta;  // lower limit
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {
      // if we return early, we have to make sure we release the load warp
      cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp,
                                  kNamedBarrierEpilogue)
          .arrive();

      return;
    }
    int k_index_final = k_tile_total - 1;

    ElementAcc row_max = -std::numeric_limits<ElementAcc>::infinity();
    ElementAcc row_sum = 0;
    ElementAcc correction_factor = 1;

    pipeline_p_mma.producer_acquire(pipeline_p_mma_producer_state);
    pipeline_mma_s.consumer_wait(pipeline_mma_s_consumer_state);

    auto dispatch_bool = [](bool b, auto fn) {
      if (b) {
        fn(cute::true_type{});
      } else {
        fn(cute::false_type{});
      }
    };

    // softmax s0 -> p0
    dispatch_bool(k_index == k_index_final, [&](auto is_last_tile) {
      softmax(is_last_tile, row_max, row_sum, correction_factor, problem_shape, mainloop_args,
              shared_tensors, k_index,
              uint32_t(pipeline_mma_s_consumer_state.index() == 0 ? TmemAllocation::kS0
                                                                  : TmemAllocation::kS1),
              pipeline_p_mma_producer_state.index());
    });

    k_index += 1;

    cutlass::arch::fence_view_async_tmem_load();
    cutlass::arch::fence_view_async_shared();
    pipeline_mma_s.consumer_release(pipeline_mma_s_consumer_state);
    ++pipeline_mma_s_consumer_state;
    pipeline_p_mma.producer_commit(pipeline_p_mma_producer_state);
    ++pipeline_p_mma_producer_state;

    k_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      pipeline_p_mma.producer_acquire(pipeline_p_mma_producer_state);
      pipeline_mma_s.consumer_wait(pipeline_mma_s_consumer_state);

      // softmax s1 -> p1
      dispatch_bool(k_index == k_index_final, [&](auto is_last_tile) {
        softmax(is_last_tile, row_max, row_sum, correction_factor, problem_shape, mainloop_args,
                shared_tensors, k_index,
                uint32_t(pipeline_mma_s_consumer_state.index() == 0 ? TmemAllocation::kS0
                                                                    : TmemAllocation::kS1),
                pipeline_p_mma_producer_state.index());
      });

      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::fence_view_async_shared();
      pipeline_mma_s.consumer_release(pipeline_mma_s_consumer_state);
      ++pipeline_mma_s_consumer_state;
      pipeline_p_mma.producer_commit(pipeline_p_mma_producer_state);
      ++pipeline_p_mma_producer_state;

      pipeline_mma_o.consumer_wait(pipeline_mma_o_consumer_state);

      // rescale
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < IterationsPV_N; j++) {
        rescale(correction_factor,
                uint32_t(TmemAllocation::kO0) + j * uint32_t(TmemAllocation::kSizeAccO));
      }

      cutlass::arch::fence_view_async_tmem_store();
      pipeline_mma_o.consumer_release(pipeline_mma_o_consumer_state);
      ++pipeline_mma_o_consumer_state;

      --k_tile_count;
      k_index += 1;
    }

    pipeline_mma_o.consumer_wait(pipeline_mma_o_consumer_state);

#ifdef B2B
    row_sum = 1;
#else
    if constexpr (kWarpsInN > 1) {
      // reduce row_sum if needed (for 2x2 dp)
      shared_tensors.smem_exchange[threadIdx.x] = row_sum;
      cutlass::arch::NamedBarrier(kNumComputeWarps * NumThreadsPerWarp, kNamedBarrierExchange)
          .sync();
      // (64, 2) shape
      int peer_index = (threadIdx.x + 64) % 128;
      row_sum += shared_tensors.smem_exchange[peer_index];
    }
#endif

    cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp,
                                kNamedBarrierEpilogue)
        .arrive();

    // epilogue
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < IterationsPV_N; j++) {
      epilogue(row_max, row_sum, replace<1>(cta_coord, j), problem_shape, mainloop_args,
               epilogue_args, shared_tensors,
               uint32_t(TmemAllocation::kO0) + j * uint32_t(TmemAllocation::kSizeAccO), split_kv);
    }

    cutlass::arch::fence_view_async_tmem_load();
    pipeline_mma_o.consumer_release(pipeline_mma_o_consumer_state);
    ++pipeline_mma_o_consumer_state;
  }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::fmha::kernel


//////////////////////////////////////////////////////////////////////////////
// Section 6: Device Layer
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::device {

using namespace cute;
using namespace cutlass::fmha::kernel;

template <class Kernel_>
class MLA {
 public:
  using Kernel = Kernel_;

  using ReductionKernel = cutlass::fmha::kernel::Sm100FmhaMlaReductionKernel<
      typename Kernel::ElementOut, typename Kernel::ElementAcc, typename Kernel::ElementAcc,
      Kernel::TileShapeH::value, Kernel::TileShapeL::value, 256>;

  using KernelArguments = typename Kernel::Arguments;
  using ReductionArguments = typename ReductionKernel::Arguments;
  using Arguments = KernelArguments;

  using KernelParams = typename Kernel::Params;
  using ReductionParams = typename ReductionKernel::Params;
  struct Params {
    KernelParams fmha_params;
    ReductionParams reduction_params;
  };

 private:
  Params params_;

  bool is_initialized(bool set = false) {
    static bool initialized = false;
    if (set) initialized = true;
    return initialized;
  }

  static ReductionArguments to_reduction_args(Arguments const& args) {
    auto [H, K, D, B] = args.problem_shape;
    return ReductionArguments{nullptr, args.epilogue.ptr_o, nullptr, args.epilogue.ptr_lse,
                              args.mainloop.softmax_scale, B, args.split_kv, K,
                              args.mainloop.ptr_seq, args.ptr_split_kv, Kernel::TileShapeS::value};
  }

 public:
  Params const& params() const { return params_; }

  static void set_split_kv(KernelArguments& args) {
    if (args.split_kv >= 1) return;
    auto [H, K, D, B] = args.problem_shape;
    int sm_count = args.hw_info.sm_count;
    int max_splits = ceil_div(K, 128);
    int sms_per_batch = max(1, sm_count / B);
    int split_heur = min(max_splits, sms_per_batch);
    int waves = ceil_div(B * split_heur, sm_count);
    int k_waves = ceil_div(max_splits, split_heur);
    int split_wave_aware = ceil_div(max_splits, k_waves);
    args.split_kv = split_wave_aware;
  }

  static Status can_implement(Arguments const& args) {
    if (!Kernel::can_implement(args)) return Status::kInvalid;
    if (!ReductionKernel::can_implement(to_reduction_args(args))) return Status::kInvalid;
    return Status::kSuccess;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += Kernel::get_workspace_size(args);
    workspace_bytes += ReductionKernel::get_workspace_size(to_reduction_args(args));
    return workspace_bytes;
  }

  static int maximum_active_blocks(int = -1) {
    CUTLASS_TRACE_HOST("MLA::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = Kernel::SharedStorageSize;

    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = cudaFuncSetAttribute(device_kernel<Kernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError();
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
        return -1;
      }
    }

    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, device_kernel<Kernel>, Kernel::MaxThreadsPerBlock, smem_size);

    if (cudaSuccess != result) {
      result = cudaGetLastError();
      CUTLASS_TRACE_HOST("  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
                         << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("MLA::initialize() - workspace " << workspace);

    Status status = Kernel::initialize_workspace(args, workspace, stream);
    if (status != Status::kSuccess) return status;
    status = ReductionKernel::initialize_workspace(to_reduction_args(args), workspace, stream);
    if (status != Status::kSuccess) return status;

    KernelParams kernel_params = Kernel::to_underlying_arguments(args, workspace);

    ReductionArguments reduction_args = to_reduction_args(args);
    if (reduction_args.split_kv > 1) {
      reduction_args.ptr_oaccum = kernel_params.epilogue.ptr_o_acc;
      reduction_args.ptr_lseaccum = kernel_params.epilogue.ptr_lse_acc;
    }
    ReductionParams reduction_params = ReductionKernel::to_underlying_arguments(reduction_args, workspace);
    params_ = Params{kernel_params, reduction_params};

    if (is_initialized()) return Status::kSuccess;

    int smem_size = Kernel::SharedStorageSize;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      cudaError_t result = cudaFuncSetAttribute(
          device_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError();
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }
    }

    is_initialized(true);
    return Status::kSuccess;
  }

  Status update(Arguments const& args, void* workspace = nullptr) {
    CUTLASS_TRACE_HOST("MLA()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);
    if (workspace_bytes > 0 && nullptr == workspace) return Status::kErrorWorkspaceNull;

    auto fmha_params = Kernel::to_underlying_arguments(args, workspace);

    ReductionArguments reduction_args = to_reduction_args(args);
    if (reduction_args.split_kv > 1) {
      reduction_args.ptr_oaccum = fmha_params.epilogue.ptr_o_acc;
      reduction_args.ptr_lseaccum = fmha_params.epilogue.ptr_lse_acc;
    }
    ReductionParams reduction_params = ReductionKernel::to_underlying_arguments(reduction_args, workspace);
    params_ = Params{fmha_params, reduction_params};

    return Status::kSuccess;
  }

  static Status run(Params& params, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("MLA::run()");
    dim3 const block = Kernel::get_block_shape();
    dim3 const grid = Kernel::get_grid_shape(params.fmha_params);
    int smem_size = Kernel::SharedStorageSize;

    Status launch_result;
    if constexpr (Kernel::ArchTag::kMinComputeCapability >= 90) {
      dim3 cluster(cute::size<0>(typename Kernel::ClusterShape{}),
                   cute::size<1>(typename Kernel::ClusterShape{}),
                   cute::size<2>(typename Kernel::ClusterShape{}));
      void const* kernel = (void const*)device_kernel<Kernel>;
      void* kernel_params[] = {&params.fmha_params};
      launch_result = ClusterLauncher::launch(grid, cluster, block, smem_size, stream, kernel, kernel_params);
    } else {
      launch_result = Status::kSuccess;
      device_kernel<Kernel><<<grid, block, smem_size, stream>>>(params.fmha_params);
    }

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess != result || Status::kSuccess != launch_result) {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }

    if (params.reduction_params.split_kv > 1) {
      dim3 const block = ReductionKernel::get_block_shape();
      dim3 const grid = ReductionKernel::get_grid_shape(params.reduction_params);
      device_kernel<ReductionKernel><<<grid, block, 0, stream>>>(params.reduction_params);
      cudaError_t result = cudaGetLastError();
      if (cudaSuccess == result) {
        return Status::kSuccess;
      } else {
        CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
        return Status::kErrorInternal;
      }
    } else {
      return Status::kSuccess;
    }
  }

  Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (Status::kSuccess == status) {
      status = run(params_, stream);
    }
    return status;
  }

  Status operator()(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    return run(args, workspace, stream);
  }

  Status run(cudaStream_t stream = nullptr) { return run(params_, stream); }

  Status operator()(cudaStream_t stream = nullptr) { return run(params_, stream); }
};

}  // namespace cutlass::fmha::device

