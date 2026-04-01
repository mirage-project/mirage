/* Copyright 2026 CMU
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

#include "../common/sm100_fp8_runtime_registry.h"
#include "blackwell/linear_fp8_1d2d_sm100.cuh"
#include "hopper/tma_2d.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>
#include <cutlass/util/print_error.hpp>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>

using bfloat16 = cute::bfloat16_t;
namespace fp8_runtime = mirage::blackwell::sm100_fp8_runtime;
// sm100_linear_fp8_1d2d
namespace {

struct LinearFp8RuntimeKey {
  int batch_size;
  int output_size;
  int reduction_size;
  bool has_residual;

  bool operator<(LinearFp8RuntimeKey const &other) const {
    return std::tie(batch_size, output_size, reduction_size, has_residual) <
           std::tie(other.batch_size,
                    other.output_size,
                    other.reduction_size,
                    other.has_residual);
  }
};

struct LinearFp8TmaDescriptorCache {
  CUtensorMap host_input_desc{};
  CUtensorMap host_weight_desc{};
  CUtensorMap host_output_desc{};
  CUtensorMap *device_input_desc = nullptr;
  CUtensorMap *device_weight_desc = nullptr;
  CUtensorMap *device_output_desc = nullptr;
  void *last_input_ptr = nullptr;
  void *last_weight_ptr = nullptr;
  void *last_output_ptr = nullptr;
  bool initialized = false;
  bool kernel_configured = false;
  std::mutex mutex;
};

LinearFp8TmaDescriptorCache &
    get_linear_fp8_tma_descriptor_cache(LinearFp8RuntimeKey const &key) {
  static std::map<LinearFp8RuntimeKey,
                  std::unique_ptr<LinearFp8TmaDescriptorCache>>
      caches;
  static std::mutex caches_mutex;

  std::lock_guard<std::mutex> guard(caches_mutex);
  auto &cache = caches[key];
  if (!cache) {
    cache = std::make_unique<LinearFp8TmaDescriptorCache>();
  }
  return *cache;
}

template <typename T>
void check_driver_success(CUresult result, char const *what) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  char const *error_name = nullptr;
  char const *error_string = nullptr;
  cuGetErrorName(result, &error_name);
  cuGetErrorString(result, &error_string);
  TORCH_CHECK(false,
              what,
              " failed with ",
              (error_name ? error_name : "unknown"),
              ": ",
              (error_string ? error_string : "unknown"));
}

template <typename KernelPtr, typename... Args>
void launch_kernel_ex_cluster(dim3 grid_dim,
                              dim3 block_dim,
                              dim3 cluster_dim,
                              int smem_bytes,
                              cudaStream_t cuda_stream,
                              KernelPtr kernel_ptr,
                              Args... args) {
  cudaLaunchAttribute launch_attribute{};
  launch_attribute.id = cudaLaunchAttributeClusterDimension;
  launch_attribute.val.clusterDim = {
      cluster_dim.x, cluster_dim.y, cluster_dim.z};

  cudaLaunchConfig_t launch_config{};
  launch_config.gridDim = grid_dim;
  launch_config.blockDim = block_dim;
  launch_config.dynamicSmemBytes = smem_bytes;
  launch_config.stream = cuda_stream;
  launch_config.numAttrs = 1;
  launch_config.attrs = &launch_attribute;

  void *kernel_params[] = {const_cast<void *>(
      reinterpret_cast<void const *>(std::addressof(args)))...};
  if (cluster_dim.x == 1 && cluster_dim.y == 1 && cluster_dim.z == 1) {
    CUTE_CHECK_ERROR(cudaLaunchKernel((void const *)kernel_ptr,
                                      grid_dim,
                                      block_dim,
                                      kernel_params,
                                      smem_bytes,
                                      cuda_stream));
  } else {
    CUTE_CHECK_ERROR(
        cudaLaunchKernelExC(&launch_config, (void *)kernel_ptr, kernel_params));
  }
}

} // namespace

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          class BiasTensor,
          int MMA_M,
          int MMA_N,
          bool NoBias,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__global__ __launch_bounds__(256, 1) void linear_fp8_1d2d_sm100_wrapper(
    void *tma_a_desc_ptr,
    void *tma_b_desc_ptr,
    uint32_t const *weight_scale_ptr,
    uint32_t const *input_scale_ptr,
    BiasTensor mBias,
    void *tma_out_desc_ptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 128;
  constexpr int TILE_SIZE = 128;
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  using TMA_B =
      kernel::tma::tma_2d<cutlass::float_e4m3_t,
                          B,
                          M,
                          S,
                          BATCH_SIZE,                /*GMEM_ROW_*/
                          REDUCTION_SIZE,            /*GMEM_COL_*/
                          MMA_N,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                          REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                         /*GMEM_STRIDE_COL_*/
                          1,                         /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                          MMA_N * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;

  using TMA_A =
      kernel::tma::tma_2d<cutlass::float_e4m3_t,
                          B,
                          M,
                          S,
                          OUTPUT_SIZE,               /*GMEM_ROW_*/
                          REDUCTION_SIZE,            /*GMEM_COL_*/
                          MMA_M,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                          REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                         /*GMEM_STRIDE_COL_*/
                          1,                         /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                          MMA_M * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;

  using TMA_OUT = kernel::tma::tma_2d<bfloat16,
                                      0,
                                      M,
                                      S,
                                      BATCH_SIZE,    /*GMEM_ROW_*/
                                      OUTPUT_SIZE,   /*GMEM_COL_*/
                                      MMA_N,         /*SMEM_ROW_*/
                                      MMA_M,         /*SMEM_COL_*/
                                      OUTPUT_SIZE,   /*GMEM_STRIDE_ROW_*/
                                      1,             /*GMEM_STRIDE_COL_*/
                                      1,             /*SMEM_REPEAT_ROW_*/
                                      1,             /*SMEM_REPEAT_COL_*/
                                      MMA_N * MMA_M, /*SMEM_STRIDE_*/
                                      true>;

  TMA_A tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

  kernel::linear_fp8_1d2d_sm100_task_impl<T,
                                          TMA_A,
                                          TMA_B,
                                          BiasTensor,
                                          TMA_OUT,
                                          MMA_M,
                                          MMA_N,
                                          BATCH_SIZE,
                                          OUTPUT_SIZE,
                                          REDUCTION_SIZE,
                                          NoBias,
                                          /*SplitK=*/false,
                                          NUM_AB_STAGE,
                                          NUM_ACC_STAGE,
                                          NUM_C_STAGE>(
      tma_a, tma_b, weight_scale_ptr, input_scale_ptr, mBias, tma_out);
}

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          bool HasResidual>
void launch_linear_fp8_1d2d_sm100(void *input_ptr,        // b_ptr
                                  void *input_scale_ptr,  // sfb_ptr
                                  void *weight_ptr,       // a_ptr
                                  void *weight_scale_ptr, // sfa_ptr
                                  void *output_ptr,
                                  void *residual_ptr = nullptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;

  constexpr int TMA_CP_ASYNC_SIZE = 128;
  constexpr int TILE_SIZE = 128;
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  // TMA_A tma_a(weight_ptr);
  // TMA_B tma_b(input_ptr);
  // TMA_OUT tma_out(output_ptr);

  auto &cache = get_linear_fp8_tma_descriptor_cache(LinearFp8RuntimeKey{
      BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, HasResidual});
  std::lock_guard<std::mutex> lock(cache.mutex);

  if (!cache.initialized) {
    uint64_t i_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                                static_cast<uint64_t>(REDUCTION_SIZE)};
    uint64_t i_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE)};
    uint32_t i_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                                static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};

    uint64_t w_gmem_shape[2] = {static_cast<uint64_t>(OUTPUT_SIZE),
                                static_cast<uint64_t>(REDUCTION_SIZE)};
    uint64_t w_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE)};
    uint32_t w_smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                                static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};

    int const output_stride = OUTPUT_SIZE;
    uint64_t o_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                                static_cast<uint64_t>(OUTPUT_SIZE)};
    uint64_t o_gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
    uint32_t o_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                                static_cast<uint32_t>(MMA_M)};
    size_t o_smem_repeat_col = 1;

    mirage::runtime::fill_tma_desc<cutlass::float_e4m3_t, B, M, S, 2>(
        &cache.host_input_desc,
        static_cast<cutlass::float_e4m3_t *>(input_ptr),
        i_gmem_shape,
        i_gmem_stride,
        i_smem_shape,
        1,
        TMA_CP_ASYNC_REPEAT_COL);
    mirage::runtime::fill_tma_desc<cutlass::float_e4m3_t, B, M, S, 2>(
        &cache.host_weight_desc,
        static_cast<cutlass::float_e4m3_t *>(weight_ptr),
        w_gmem_shape,
        w_gmem_stride,
        w_smem_shape,
        1,
        TMA_CP_ASYNC_REPEAT_COL);
    mirage::runtime::fill_tma_desc<bfloat16, 0, M, S, 2>(
        &cache.host_output_desc,
        static_cast<bfloat16 *>(output_ptr),
        o_gmem_shape,
        o_gmem_stride,
        o_smem_shape,
        1,
        o_smem_repeat_col);

    CUTE_CHECK_ERROR(cudaMalloc(&cache.device_input_desc, sizeof(CUtensorMap)));
    CUTE_CHECK_ERROR(
        cudaMalloc(&cache.device_weight_desc, sizeof(CUtensorMap)));
    CUTE_CHECK_ERROR(
        cudaMalloc(&cache.device_output_desc, sizeof(CUtensorMap)));

    CUTE_CHECK_ERROR(cudaMemcpy(cache.device_input_desc,
                                &cache.host_input_desc,
                                sizeof(CUtensorMap),
                                cudaMemcpyHostToDevice));
    CUTE_CHECK_ERROR(cudaMemcpy(cache.device_weight_desc,
                                &cache.host_weight_desc,
                                sizeof(CUtensorMap),
                                cudaMemcpyHostToDevice));
    CUTE_CHECK_ERROR(cudaMemcpy(cache.device_output_desc,
                                &cache.host_output_desc,
                                sizeof(CUtensorMap),
                                cudaMemcpyHostToDevice));

    cache.last_input_ptr = input_ptr;
    cache.last_weight_ptr = weight_ptr;
    cache.last_output_ptr = output_ptr;
    cache.initialized = true;
  } else {
    if (input_ptr != cache.last_input_ptr) {
      check_driver_success<T>(
          cuTensorMapReplaceAddress(&cache.host_input_desc, input_ptr),
          "cuTensorMapReplaceAddress(input)");
      CUTE_CHECK_ERROR(cudaMemcpy(cache.device_input_desc,
                                  &cache.host_input_desc,
                                  sizeof(CUtensorMap),
                                  cudaMemcpyHostToDevice));
      cache.last_input_ptr = input_ptr;
    }
    if (weight_ptr != cache.last_weight_ptr) {
      check_driver_success<T>(
          cuTensorMapReplaceAddress(&cache.host_weight_desc, weight_ptr),
          "cuTensorMapReplaceAddress(weight)");
      CUTE_CHECK_ERROR(cudaMemcpy(cache.device_weight_desc,
                                  &cache.host_weight_desc,
                                  sizeof(CUtensorMap),
                                  cudaMemcpyHostToDevice));
      cache.last_weight_ptr = weight_ptr;
    }
    if (output_ptr != cache.last_output_ptr) {
      check_driver_success<T>(
          cuTensorMapReplaceAddress(&cache.host_output_desc, output_ptr),
          "cuTensorMapReplaceAddress(output)");
      CUTE_CHECK_ERROR(cudaMemcpy(cache.device_output_desc,
                                  &cache.host_output_desc,
                                  sizeof(CUtensorMap),
                                  cudaMemcpyHostToDevice));
      cache.last_output_ptr = output_ptr;
    }
  }

  // Residual
  cute::Layout layout_Bias = cute::make_layout(
      cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
      cute::make_stride(OUTPUT_SIZE,
                        cute::Int<1>{})); // (Gemm_M,Gemm_N):(Gemm_N,_1)
  bfloat16 *bias_ptr = residual_ptr ? static_cast<bfloat16 *>(residual_ptr)
                                    : static_cast<bfloat16 *>(output_ptr);
  cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(bias_ptr),
                                         layout_Bias); // (Gemm_N, Gemm_M)

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  int smemBytes = 224 * 1024;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();

  if constexpr (HasResidual) {
    auto *kernel_ptr = &linear_fp8_1d2d_sm100_wrapper<T,
                                                      BATCH_SIZE,
                                                      OUTPUT_SIZE,
                                                      REDUCTION_SIZE,
                                                      decltype(mBias),
                                                      MMA_M,
                                                      MMA_N,
                                                      false>;
    if (!cache.kernel_configured) {
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      cache.kernel_configured = true;
    }
    launch_kernel_ex_cluster(grid_dim,
                             block_dim,
                             cluster_dim,
                             smemBytes,
                             cuda_stream,
                             kernel_ptr,
                             cache.device_weight_desc,
                             cache.device_input_desc,
                             static_cast<uint32_t const *>(weight_scale_ptr),
                             static_cast<uint32_t const *>(input_scale_ptr),
                             mBias,
                             cache.device_output_desc);
  } else {
    auto *kernel_ptr = &linear_fp8_1d2d_sm100_wrapper<T,
                                                      BATCH_SIZE,
                                                      OUTPUT_SIZE,
                                                      REDUCTION_SIZE,
                                                      decltype(mBias),
                                                      MMA_M,
                                                      MMA_N,
                                                      true>;
    if (!cache.kernel_configured) {
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      cache.kernel_configured = true;
    }
    launch_kernel_ex_cluster(grid_dim,
                             block_dim,
                             cluster_dim,
                             smemBytes,
                             cuda_stream,
                             kernel_ptr,
                             cache.device_weight_desc,
                             cache.device_input_desc,
                             static_cast<uint32_t const *>(weight_scale_ptr),
                             static_cast<uint32_t const *>(input_scale_ptr),
                             mBias,
                             cache.device_output_desc);
  }
  CUTE_CHECK_LAST();
}

void linear_fp8_1d2d_sm100_kernel(torch::Tensor input_q,
                                  torch::Tensor input_scale,
                                  torch::Tensor weight_q,
                                  torch::Tensor weight_scale,
                                  c10::optional<at::Tensor> residual,
                                  torch::Tensor output) {
  TORCH_CHECK(input_q.dim() == 2, "input_q must be 2D");
  TORCH_CHECK(input_scale.dim() == 2, "input_scale must be 2D");
  TORCH_CHECK(weight_q.dim() == 2, "weight_q must be 2D");
  TORCH_CHECK(weight_scale.dim() == 2, "weight_scale must be 2D");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(input_q.is_contiguous(), "input_q must be contiguous");
  TORCH_CHECK(input_scale.is_contiguous(), "input_scale must be contiguous");
  TORCH_CHECK(weight_q.is_contiguous(), "weight_q must be contiguous");
  TORCH_CHECK(weight_scale.is_contiguous(), "weight_scale must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(input_q.scalar_type() == at::kFloat8_e4m3fn,
              "input_q must be float8_e4m3fn");
  TORCH_CHECK(weight_q.scalar_type() == at::kFloat8_e4m3fn,
              "weight_q must be float8_e4m3fn");
  TORCH_CHECK(input_scale.scalar_type() == at::kInt ||
                  input_scale.scalar_type() == at::kUInt32,
              "input_scale must be uint32-compatible");
  TORCH_CHECK(weight_scale.scalar_type() == at::kInt ||
                  weight_scale.scalar_type() == at::kUInt32,
              "weight_scale must be uint32-compatible");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");

  int const batch_size = static_cast<int>(input_q.size(0));
  int const reduction_size = static_cast<int>(input_q.size(1));
  int const output_size = static_cast<int>(weight_q.size(0));
  int const expected_padded_scale_k =
      fp8_runtime::padded_scale_k_for_reduction_size(reduction_size);

  TORCH_CHECK(weight_q.size(1) == reduction_size,
              "weight_q shape mismatch: expected reduction_size ",
              reduction_size,
              " in dim-1 but got ",
              weight_q.size(1));
  TORCH_CHECK(output.size(0) == batch_size && output.size(1) == output_size,
              "output shape mismatch: expected [",
              batch_size,
              ", ",
              output_size,
              "] but got [",
              output.size(0),
              ", ",
              output.size(1),
              "]");
  TORCH_CHECK(input_scale.size(0) == batch_size &&
                  input_scale.size(1) == expected_padded_scale_k,
              "input_scale shape mismatch: expected [",
              batch_size,
              ", ",
              expected_padded_scale_k,
              "] but got [",
              input_scale.size(0),
              ", ",
              input_scale.size(1),
              "]");
  TORCH_CHECK(weight_scale.size(0) == output_size &&
                  weight_scale.size(1) == expected_padded_scale_k,
              "weight_scale shape mismatch: expected [",
              output_size,
              ", ",
              expected_padded_scale_k,
              "] but got [",
              weight_scale.size(0),
              ", ",
              weight_scale.size(1),
              "]");

  TORCH_CHECK(fp8_runtime::is_supported_dense_gemm_shape(
                  batch_size, output_size, reduction_size),
              "Unsupported linear_fp8_1d2d_sm100 shape [B=",
              batch_size,
              ", N=",
              output_size,
              ", K=",
              reduction_size,
              "]. Supported batch sizes: {",
              fp8_runtime::supported_batch_sizes_string(),
              "}, output sizes: {",
              fp8_runtime::supported_output_sizes_string(),
              "}, reduction sizes: {",
              fp8_runtime::supported_reduction_sizes_string(),
              "}");

  void *input_ptr = input_q.data_ptr();
  void *input_scale_ptr = input_scale.data_ptr();
  void *weight_ptr = weight_q.data_ptr();
  void *weight_scale_ptr = weight_scale.data_ptr();
  void *output_ptr = output.data_ptr();
  bool const has_residual = residual.has_value();
  if (has_residual) {
    TORCH_CHECK(residual->dim() == 2, "residual must be 2D");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(residual->scalar_type() == at::kBFloat16,
                "residual must be bfloat16");
    TORCH_CHECK(residual->size(0) == batch_size &&
                    residual->size(1) == output_size,
                "residual shape mismatch: expected [",
                batch_size,
                ", ",
                output_size,
                "] but got [",
                residual->size(0),
                ", ",
                residual->size(1),
                "]");
  }
  void *residual_ptr = has_residual ? residual->data_ptr() : nullptr;

#define DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                         \
    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, HAS_RESIDUAL)                     \
  case REDUCTION_SIZE:                                                         \
    launch_linear_fp8_1d2d_sm100<cutlass::float_e4m3_t,                        \
                                 BATCH_SIZE,                                   \
                                 OUTPUT_SIZE,                                  \
                                 REDUCTION_SIZE,                               \
                                 HAS_RESIDUAL>(input_ptr,                      \
                                               input_scale_ptr,                \
                                               weight_ptr,                     \
                                               weight_scale_ptr,               \
                                               output_ptr,                     \
                                               residual_ptr);                  \
    break;

#define DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(BATCH_SIZE, HAS_RESIDUAL)    \
  case BATCH_SIZE:                                                             \
    switch (reduction_size) {                                                  \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 128, HAS_RESIDUAL)                                  \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 256, HAS_RESIDUAL)                                  \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 384, HAS_RESIDUAL)                                  \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 512, HAS_RESIDUAL)                                  \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 768, HAS_RESIDUAL)                                  \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 1024, HAS_RESIDUAL)                                 \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 1536, HAS_RESIDUAL)                                 \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 2048, HAS_RESIDUAL)                                 \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 4096, HAS_RESIDUAL)                                 \
      DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE(                           \
          BATCH_SIZE, 128, 7168, HAS_RESIDUAL)                                 \
      default:                                                                 \
        TORCH_CHECK(false, "Unsupported reduction_size dispatch");             \
    }                                                                          \
    break;

  if (has_residual) {
    switch (batch_size) {
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(1, true)
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(2, true)
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(4, true)
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(8, true)
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(16, true)
      default:
        TORCH_CHECK(false, "Unsupported batch_size dispatch");
    }
  } else {
    switch (batch_size) {
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(1, false)
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(2, false)
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(4, false)
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(8, false)
      DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE(16, false)
      default:
        TORCH_CHECK(false, "Unsupported batch_size dispatch");
    }
  }

#undef DISPATCH_LINEAR_FP8_SM100_BATCH_SIZE_CASE
#undef DISPATCH_LINEAR_FP8_SM100_REDUCTION_SIZE_CASE
}

std::vector<std::vector<int64_t>> supported_dense_gemm_shapes() {
  return fp8_runtime::supported_dense_gemm_shapes_vector();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_fp8_1d2d_sm100",
        &linear_fp8_1d2d_sm100_kernel,
        "Linear kernel SM100 FP8 1D2D");
  m.def("supported_dense_gemm_shapes",
        &supported_dense_gemm_shapes,
        "Supported SM100 FP8 dense GEMM shapes");
}
