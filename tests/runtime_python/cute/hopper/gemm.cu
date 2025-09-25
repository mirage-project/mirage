/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "epilogue.cuh"
#include "gemm_ws.cuh"
#include "kernel_traits.cuh"
#include "mma_tma_ws_mainloop.cuh"

template <typename CollectiveMainloop,
          typename CollectiveEpilogue,
          typename Ktraits>
__global__ __launch_bounds__(256, 1) void linear_kernel_hopper_cute_wrapper(
    CUTE_GRID_CONSTANT
    typename CollectiveMainloop::Params const mainloop_params,
    CUTE_GRID_CONSTANT
    typename CollectiveEpilogue::Params const epilogue_params) {
  kernel::gemm_kernel_tma_warp_specialized<CollectiveMainloop,
                                           CollectiveEpilogue,
                                           Ktraits>()(mainloop_params,
                                                      epilogue_params);
}

template <int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_hopper_cute(void *input_ptr,
                               void *weight_ptr,
                               void *residual_ptr,
                               void *output_ptr,
                               cudaStream_t stream) {

  using namespace cute;
  using T = bfloat16_t;
  using KernelTraits =
      kernel::MMAKernelTraits<T,
                              BATCH_SIZE,
                              OUTPUT_SIZE,
                              REDUCTION_SIZE,
                              cutlass::layout::RowMajor,    // GmemLayoutATag
                              cutlass::layout::ColumnMajor, // GmemLayoutBTag
                              cutlass::layout::ColumnMajor, // GmemLayoutCTag
                              cutlass::layout::ColumnMajor, // GmemLayoutDTag
                              8,                            // NUM_WARPS
                              128,                          // M
                              128,                          // N
                              32>;                          // K

  using Mainloop = kernel::CollectiveMainloop<KernelTraits>;
  using Epilogue = kernel::CollectiveEpilogue<KernelTraits>;

  auto problem_shape = {BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE};

  using StrideA =
      cutlass::detail::TagToStrideA_t<typename KernelTraits::GmemLayoutATag>;
  using StrideB =
      cutlass::detail::TagToStrideB_t<typename KernelTraits::GmemLayoutATag>;
  using StrideC =
      cutlass::detail::TagToStrideC_t<typename KernelTraits::GmemLayoutCTag>;
  using StrideD =
      cutlass::detail::TagToStrideC_t<typename KernelTraits::GmemLayoutDTag>;
  StrideA stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, {KernelTraits::M, KernelTraits::K, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(
      StrideB{}, {KernelTraits::N, KernelTraits::K, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, {KernelTraits::M, KernelTraits::N, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, {KernelTraits::M, KernelTraits::N, 1});

  typename Mainloop::Arguments mainloop_args{
      static_cast<T const *>(input_ptr),  // ptr_A
      stride_A,                           // dA
      static_cast<T const *>(weight_ptr), // ptr_B
      stride_B,                           // dB
  };

  typename Epilogue::Arguments epilogue_args{
      static_cast<T const *>(residual_ptr), // ptr_C
      stride_C,                             // dC
      static_cast<T *>(output_ptr),         // ptr_D
      stride_C,                             // dD
  };

  typename Mainloop::Params mainloop_params =
      Mainloop::to_underlying_arguments(problem_shape, mainloop_args);
  typename Epilogue::Params epilogue_params =
      Epilogue::to_underlying_arguments();

  dim3 grid(1);
  dim3 block(256);

  size_t shared_mem_size = 1000000;
  linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue, KernelTraits>
      <<<grid, block, shared_mem_size>>>(mainloop_params, epilogue_params);
}