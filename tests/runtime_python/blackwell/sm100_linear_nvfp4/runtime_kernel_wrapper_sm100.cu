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

#include "blackwell/linear_nvfp4_1d2d_sm100.cuh"
#include "blackwell/linear_nvfp4_swapAB_sm100.cuh"
#include "blackwell/quantize_nvfp4_sm100.cuh"
#include "hopper/tma_2d_nvfp4.cuh"
#include "hopper/tma_3d.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/util/print_error.hpp>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>

using float_e2m1  = cute::float_e2m1_t;
using float_ue4m3 = cute::float_ue4m3_t;

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          class BiasTensor,
          class OutputTensor,
          int MMA_M,
          int MMA_N,
          bool NoBias,
          int NUM_AB_STAGE  = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE   = 4>
__global__ __launch_bounds__(256, 1)
void linear_nvfp4_1d2d_sm100_wrapper(void *tma_a_desc_ptr,
                                     void *tma_b_desc_ptr,
                                     void *tma_sfa_desc_ptr,
                                     void *tma_sfb_desc_ptr,
                                     BiasTensor mBias,
                                     OutputTensor mOutput,
                                     void *tma_out_desc_ptr) {
  constexpr int MMA_K          = 64;
  constexpr int NUM_MMA_K      = 4;
  constexpr int bK             = MMA_K * NUM_MMA_K;
  constexpr int SCALE_VECTOR_SIZE = 16;
  constexpr int EPI_PIPE_DEPTH = 4;
  constexpr int EPI_N          = MMA_N / EPI_PIPE_DEPTH;
  constexpr int B_FP4 = 3;
  constexpr int B_SF  = 0;
  constexpr int B_OUT = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;
  constexpr int SF_COL_B = MMA_N * MMA_K / SCALE_VECTOR_SIZE / 2;
  constexpr int SWIZZLE_SIZE = 128 / sizeof(float);

  using TMA_A = kernel::tma::tma_2d_nvfp4<cute::float_e2m1_t, B_FP4, M, S,
      BATCH_SIZE, REDUCTION_SIZE, MMA_M, bK, REDUCTION_SIZE, 1, 1, 1, MMA_M * bK / 2, true>;
  using TMA_B = kernel::tma::tma_2d_nvfp4<cute::float_e2m1_t, B_FP4, M, S,
      OUTPUT_SIZE, REDUCTION_SIZE, MMA_N, bK, REDUCTION_SIZE, 1, 1, 1, MMA_N * bK / 2, true>;
  using TMA_SFA = kernel::tma::tma_3d<cute::half_t, B_SF, M, S,
      BATCH_SIZE / MMA_M, REDUCTION_SIZE / MMA_K, SF_COL_A,
      1, 1, SF_COL_A,
      SF_COL_A * (REDUCTION_SIZE / MMA_K), SF_COL_A, 1,
      NUM_MMA_K, 1, SF_COL_A, true>;
  using TMA_SFB = kernel::tma::tma_3d<cute::half_t, B_SF, M, S,
      OUTPUT_SIZE / MMA_N, REDUCTION_SIZE / MMA_K, SF_COL_B,
      1, 1, SF_COL_B,
      SF_COL_B * (REDUCTION_SIZE / MMA_K), SF_COL_B, 1,
      NUM_MMA_K, 1, SF_COL_B, true>;
  using TMA_OUT = kernel::tma::tma_3d<float, B_OUT, M, S,
      BATCH_SIZE, OUTPUT_SIZE / SWIZZLE_SIZE, SWIZZLE_SIZE,
      MMA_M, EPI_N / SWIZZLE_SIZE, SWIZZLE_SIZE,
      OUTPUT_SIZE, SWIZZLE_SIZE, 1,
      1, 1, MMA_M * EPI_N, true>;

  TMA_A   tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B   tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_SFA tma_sfa(static_cast<CUtensorMap *>(tma_sfa_desc_ptr));
  TMA_SFB tma_sfb(static_cast<CUtensorMap *>(tma_sfb_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

  kernel::linear_nvfp4_1d2d_sm100_task_impl<T, TMA_A, TMA_B, TMA_SFA, TMA_SFB,
                                             BiasTensor, OutputTensor, TMA_OUT,
                                             MMA_M, MMA_N,
                                             BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                                             SCALE_VECTOR_SIZE, NoBias, /*SplitK=*/false,
                                             NUM_AB_STAGE, NUM_ACC_STAGE, NUM_C_STAGE>(
      tma_a, tma_b, tma_sfa, tma_sfb, mBias, mOutput, tma_out);
}

struct LinearNVFP4DescriptorCache {
  CUtensorMap host_i_desc{};
  CUtensorMap host_i_sf_desc{};
  CUtensorMap host_w_desc{};
  CUtensorMap host_w_sf_desc{};
  CUtensorMap host_o_desc{};

  CUtensorMap *desc_i_ptr    = nullptr;
  CUtensorMap *desc_i_sf_ptr = nullptr;
  CUtensorMap *desc_w_ptr    = nullptr;
  CUtensorMap *desc_w_sf_ptr = nullptr;
  CUtensorMap *desc_o_ptr    = nullptr;

  void *last_input_ptr     = nullptr;
  void *last_input_sf_ptr  = nullptr;
  void *last_weight_ptr    = nullptr;
  void *last_weight_sf_ptr = nullptr;
  void *last_output_ptr    = nullptr;

  bool initialized                = false;
  bool kernel_configured_no_bias  = false;  
  bool kernel_configured_bias     = false;  
};

static std::map<std::tuple<int,int>, std::unique_ptr<LinearNVFP4DescriptorCache>> s_1d2d_caches;
static std::mutex s_1d2d_mutex;

static LinearNVFP4DescriptorCache &get_linear_nvfp4_descriptor_cache(int output_size, int reduction_size) {
  auto key = std::make_tuple(output_size, reduction_size);
  std::lock_guard<std::mutex> guard(s_1d2d_mutex);
  auto &entry = s_1d2d_caches[key];
  if (!entry) {
    entry = std::make_unique<LinearNVFP4DescriptorCache>();
  }
  return *entry;
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_nvfp4_1d2d_sm100(void *input_ptr,
                                    void *input_sf_ptr,
                                    void *weight_ptr,
                                    void *weight_sf_ptr,
                                    void *output_ptr,
                                    void *residual_ptr = nullptr) {
  using namespace cute;
  using namespace cutlass;

  constexpr int B_FP4 = 3;
  constexpr int B_SF  = 0;
  constexpr int B_OUT = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int SCALE_VECTOR_SIZE = 16;
  constexpr int MMA_M    = 128;
  constexpr int MMA_N    = 128;
  constexpr int MMA_K    = 64;
  constexpr int NUM_MMA_K = 4;
  constexpr int bK = MMA_K * NUM_MMA_K;
  constexpr int EPI_PIPE_DEPTH = 4;
  constexpr int EPI_N = MMA_N / EPI_PIPE_DEPTH;

  constexpr int NUM_M_TILE_PER_CTA = 1;
  constexpr int NUM_N_TILE_PER_CTA = 1;
  constexpr int bM = MMA_M;
  constexpr int bN = MMA_N;

  LinearNVFP4DescriptorCache &cache = get_linear_nvfp4_descriptor_cache(OUTPUT_SIZE, REDUCTION_SIZE);

  if (!cache.initialized) {
    {
      uint64_t gmem_shape[2]  = {(uint64_t)BATCH_SIZE, (uint64_t)REDUCTION_SIZE};
      uint64_t gmem_stride[2] = {1, (uint64_t)REDUCTION_SIZE};
      uint32_t smem_shape[2]  = {(uint32_t)MMA_M, (uint32_t)bK};
      mirage::runtime::fill_tma_desc<cute::float_e2m1_t, B_FP4, M, S, 2>(
          &cache.host_i_desc, static_cast<cute::float_e2m1_t *>(input_ptr),
          gmem_shape, gmem_stride, smem_shape, 1, 1);
    }
    {
      constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;
      uint64_t gmem_shape[3]  = {(uint64_t)(BATCH_SIZE / MMA_M), (uint64_t)(REDUCTION_SIZE / MMA_K), (uint64_t)SF_COL_A};
      uint64_t gmem_stride[3] = {1, (uint64_t)SF_COL_A, (uint64_t)(SF_COL_A * (REDUCTION_SIZE / MMA_K))};
      uint32_t smem_shape[3]  = {1, 1, (uint32_t)SF_COL_A};
      mirage::runtime::fill_tma_desc<cute::half_t, B_SF, M, S, 3>(
          &cache.host_i_sf_desc, static_cast<cute::half_t *>(input_sf_ptr),
          gmem_shape, gmem_stride, smem_shape, 1, 1);
    }
    {
      uint64_t gmem_shape[2]  = {(uint64_t)OUTPUT_SIZE, (uint64_t)REDUCTION_SIZE};
      uint64_t gmem_stride[2] = {1, (uint64_t)REDUCTION_SIZE};
      uint32_t smem_shape[2]  = {(uint32_t)MMA_N, (uint32_t)bK};
      mirage::runtime::fill_tma_desc<cute::float_e2m1_t, B_FP4, M, S, 2>(
          &cache.host_w_desc, static_cast<cute::float_e2m1_t *>(weight_ptr),
          gmem_shape, gmem_stride, smem_shape, 1, 1);
    }
    {
      constexpr int SF_COL_B = MMA_N * MMA_K / SCALE_VECTOR_SIZE / 2;
      uint64_t gmem_shape[3]  = {(uint64_t)(OUTPUT_SIZE / MMA_N), (uint64_t)(REDUCTION_SIZE / MMA_K), (uint64_t)SF_COL_B};
      uint64_t gmem_stride[3] = {1, (uint64_t)SF_COL_B, (uint64_t)(SF_COL_B * (REDUCTION_SIZE / MMA_K))};
      uint32_t smem_shape[3]  = {1, 1, (uint32_t)SF_COL_B};
      mirage::runtime::fill_tma_desc<cute::half_t, B_SF, M, S, 3>(
          &cache.host_w_sf_desc, static_cast<cute::half_t *>(weight_sf_ptr),
          gmem_shape, gmem_stride, smem_shape, 1, 1);
    }
    {
      constexpr int SWIZZLE_SIZE = 128 / sizeof(float);
      uint64_t gmem_shape[3]  = {(uint64_t)BATCH_SIZE, (uint64_t)(OUTPUT_SIZE / SWIZZLE_SIZE), (uint64_t)SWIZZLE_SIZE};
      uint64_t gmem_stride[3] = {1, (uint64_t)SWIZZLE_SIZE, (uint64_t)OUTPUT_SIZE};
      uint32_t smem_shape[3]  = {(uint32_t)MMA_M, (uint32_t)(EPI_N / SWIZZLE_SIZE), (uint32_t)SWIZZLE_SIZE};
      mirage::runtime::fill_tma_desc<float, B_OUT, M, S, 3>(
          &cache.host_o_desc, static_cast<float *>(output_ptr),
          gmem_shape, gmem_stride, smem_shape, 1, 1);
    }

    cudaMalloc(&cache.desc_i_ptr,    sizeof(CUtensorMap));
    cudaMalloc(&cache.desc_i_sf_ptr, sizeof(CUtensorMap));
    cudaMalloc(&cache.desc_w_ptr,    sizeof(CUtensorMap));
    cudaMalloc(&cache.desc_w_sf_ptr, sizeof(CUtensorMap));
    cudaMalloc(&cache.desc_o_ptr,    sizeof(CUtensorMap));

    cudaMemcpy(cache.desc_i_ptr,    &cache.host_i_desc,    sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(cache.desc_i_sf_ptr, &cache.host_i_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(cache.desc_w_ptr,    &cache.host_w_desc,    sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(cache.desc_w_sf_ptr, &cache.host_w_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(cache.desc_o_ptr,    &cache.host_o_desc,    sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    cache.last_input_ptr     = input_ptr;
    cache.last_input_sf_ptr  = input_sf_ptr;
    cache.last_weight_ptr    = weight_ptr;
    cache.last_weight_sf_ptr = weight_sf_ptr;
    cache.last_output_ptr    = output_ptr;
    cache.initialized        = true;
  } else {
    // Subsequent calls: patch host descriptor and async-upload only changed pointers.
    if (input_ptr != cache.last_input_ptr) {
      cuTensorMapReplaceAddress(&cache.host_i_desc, input_ptr);
      cudaMemcpyAsync(cache.desc_i_ptr, &cache.host_i_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
      cache.last_input_ptr = input_ptr;
    }
    if (input_sf_ptr != cache.last_input_sf_ptr) {
      cuTensorMapReplaceAddress(&cache.host_i_sf_desc, input_sf_ptr);
      cudaMemcpyAsync(cache.desc_i_sf_ptr, &cache.host_i_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
      cache.last_input_sf_ptr = input_sf_ptr;
    }
    if (weight_ptr != cache.last_weight_ptr) {
      cuTensorMapReplaceAddress(&cache.host_w_desc, weight_ptr);
      cudaMemcpyAsync(cache.desc_w_ptr, &cache.host_w_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
      cache.last_weight_ptr = weight_ptr;
    }
    if (weight_sf_ptr != cache.last_weight_sf_ptr) {
      cuTensorMapReplaceAddress(&cache.host_w_sf_desc, weight_sf_ptr);
      cudaMemcpyAsync(cache.desc_w_sf_ptr, &cache.host_w_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
      cache.last_weight_sf_ptr = weight_sf_ptr;
    }
    if (output_ptr != cache.last_output_ptr) {
      cuTensorMapReplaceAddress(&cache.host_o_desc, output_ptr);
      cudaMemcpyAsync(cache.desc_o_ptr, &cache.host_o_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
      cache.last_output_ptr = output_ptr;
    }
  }

  cute::Layout layout_Bias = cute::make_layout(
      cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
      cute::make_stride(OUTPUT_SIZE, cute::Int<1>{}));
  cute::Tensor mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<float *>(residual_ptr)), layout_Bias);
  cute::Tensor mOutput = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<float *>(output_ptr)), layout_Bias);

  constexpr int num_tiles_m = BATCH_SIZE / bM / NUM_M_TILE_PER_CTA;
  constexpr int num_tiles_n = OUTPUT_SIZE / bN / NUM_N_TILE_PER_CTA;
  dim3 grid_dim(num_tiles_m, num_tiles_n, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  constexpr int NUM_C_STAGE_LAUNCH = 4;
  int smemBytes = 224 * 1024;

  bool has_residual = (residual_ptr != nullptr);
  auto launch = [&](auto *kernel_ptr, bool &configured_flag) {
    if (!configured_flag) {
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      configured_flag = true;
    }
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, (void const *)kernel_ptr,
        (void *)cache.desc_i_ptr, (void *)cache.desc_w_ptr,
        (void *)cache.desc_i_sf_ptr, (void *)cache.desc_w_sf_ptr,
        mBias, mOutput, (void *)cache.desc_o_ptr);
    CUTE_CHECK_ERROR(cudaGetLastError());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel launch" << std::endl;
    }
  };

  if (has_residual) {
    launch(&linear_nvfp4_1d2d_sm100_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                                            decltype(mBias), decltype(mOutput), MMA_M, MMA_N, false,
                                            /*NUM_AB_STAGE=*/4, /*NUM_ACC_STAGE=*/2,
                                            NUM_C_STAGE_LAUNCH>,
           cache.kernel_configured_bias);
  } else {
    launch(&linear_nvfp4_1d2d_sm100_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                                            decltype(mBias), decltype(mOutput), MMA_M, MMA_N, true,
                                            /*NUM_AB_STAGE=*/4, /*NUM_ACC_STAGE=*/2,
                                            NUM_C_STAGE_LAUNCH>,
           cache.kernel_configured_no_bias);
  }
}

template <typename T,
          int MMA_N,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          class BiasTensor,
          class OutputTensor,
          bool NoBias,
          int NUM_AB_STAGE = 4,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__global__ __launch_bounds__(256, 1)
void linear_nvfp4_swapAB_sm100_wrapper(void *tma_a_desc_ptr,
                                       void *tma_b_desc_ptr,
                                       void *tma_sfa_desc_ptr,
                                       void *tma_sfb_desc_ptr,
                                       BiasTensor mBias,
                                       OutputTensor mOutput,
                                       int logical_batch_size) {
  constexpr int MMA_K = 64;
  constexpr int NUM_MMA_K = 4;
  constexpr int bK = MMA_K * NUM_MMA_K;
  constexpr int SCALE_VECTOR_SIZE = 16;
  constexpr int MMA_M = 128;

  constexpr int B_FP4 = 3;
  constexpr int B_SF  = 0;
  constexpr int B_OUT = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int MMA_N_SFB = 128;
  constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;        // 256
  constexpr int SF_COL_B = MMA_N_SFB * MMA_K / SCALE_VECTOR_SIZE / 2;    // 256

  using TMA_A = kernel::tma::tma_2d_nvfp4<cute::float_e2m1_t, B_FP4, M, S,
      OUTPUT_SIZE, REDUCTION_SIZE, MMA_M, bK, REDUCTION_SIZE, 1, 1, 1, MMA_M * bK / 2, true>;
  using TMA_B = kernel::tma::tma_2d_nvfp4<cute::float_e2m1_t, B_FP4, M, S,
      MMA_N, REDUCTION_SIZE, MMA_N, bK, REDUCTION_SIZE, 1, 1, 1, MMA_N * bK / 2, true>;
  using TMA_SFA = kernel::tma::tma_3d<cute::half_t, B_SF, M, S,
      OUTPUT_SIZE / MMA_M, REDUCTION_SIZE / MMA_K, SF_COL_A,
      1, 1, SF_COL_A,
      SF_COL_A * (REDUCTION_SIZE / MMA_K), SF_COL_A, 1,
      NUM_MMA_K, 1, SF_COL_A, true>;
  using TMA_SFB = kernel::tma::tma_3d<cute::half_t, B_SF, M, S,
      1, REDUCTION_SIZE / MMA_K, SF_COL_B,
      1, 1, SF_COL_B,
      SF_COL_B * (REDUCTION_SIZE / MMA_K), SF_COL_B, 1,
      NUM_MMA_K, 1, SF_COL_B, true>;

  TMA_A   tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B   tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_SFA tma_sfa(static_cast<CUtensorMap *>(tma_sfa_desc_ptr));
  TMA_SFB tma_sfb(static_cast<CUtensorMap *>(tma_sfb_desc_ptr));

  kernel::linear_nvfp4_smallm_swapAB_sm100_task_impl<T, TMA_A, TMA_B, TMA_SFA, TMA_SFB,
                                                     BiasTensor, OutputTensor,
                                                     /*MMA_M=*/128, MMA_N,
                                                     OUTPUT_SIZE, REDUCTION_SIZE,
                                                     SCALE_VECTOR_SIZE, NoBias,
                                                     NUM_AB_STAGE, NUM_ACC_STAGE>(
      tma_a, tma_b, tma_sfa, tma_sfb, mBias, mOutput, logical_batch_size);
}

struct LinearNVFP4SwapABDescriptorCache {
  CUtensorMap host_w_desc{};
  CUtensorMap host_x_desc{};
  CUtensorMap host_w_sf_desc{};
  CUtensorMap host_x_sf_desc{};

  CUtensorMap *desc_w_ptr    = nullptr;
  CUtensorMap *desc_x_ptr    = nullptr;
  CUtensorMap *desc_w_sf_ptr = nullptr;
  CUtensorMap *desc_x_sf_ptr = nullptr;

  void *last_weight_ptr    = nullptr;
  void *last_input_ptr     = nullptr;
  void *last_weight_sf_ptr = nullptr;
  void *last_input_sf_ptr  = nullptr;
  int   last_padded_m      = 0;

  bool initialized                = false;
  bool kernel_configured_no_bias  = false; 
  bool kernel_configured_bias     = false; 
};

static std::map<std::tuple<int,int,int>, std::unique_ptr<LinearNVFP4SwapABDescriptorCache>> s_swapAB_caches;
static std::mutex s_swapAB_mutex;

static LinearNVFP4SwapABDescriptorCache &get_linear_nvfp4_swapAB_descriptor_cache(int mma_n, int output_size, int reduction_size) {
  auto key = std::make_tuple(mma_n, output_size, reduction_size);
  std::lock_guard<std::mutex> guard(s_swapAB_mutex);
  auto &entry = s_swapAB_caches[key];
  if (!entry) {
    entry = std::make_unique<LinearNVFP4SwapABDescriptorCache>();
  }
  return *entry;
}


template <typename T, int MMA_N, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_nvfp4_swapAB_sm100(void *input_ptr,
                                      void *input_sf_ptr,
                                      void *weight_ptr,
                                      void *weight_sf_ptr,
                                      void *output_ptr,
                                      void *residual_ptr,
                                      int logical_batch_size) {
  using namespace cute;
  using namespace cutlass;

  constexpr int B_FP4 = 3;   // 128B swizzle
  constexpr int B_SF  = 0;   // no SF swizzle
  constexpr int MMA_K = 64;
  constexpr int NUM_MMA_K = 4;
  constexpr int bK = MMA_K * NUM_MMA_K;
  constexpr int SCALE_VECTOR_SIZE = 16;
  constexpr int MMA_M = 128;
  constexpr int MMA_N_SFB = 128;
  constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;      // 256
  constexpr int SF_COL_B = MMA_N_SFB * MMA_K / SCALE_VECTOR_SIZE / 2;  // 256
  constexpr int M = 3;
  constexpr int S = 3;

  LinearNVFP4SwapABDescriptorCache &cache = get_linear_nvfp4_swapAB_descriptor_cache(MMA_N, OUTPUT_SIZE, REDUCTION_SIZE);

  const int num_n_tiles = (logical_batch_size + MMA_N - 1) / MMA_N;
  auto rebuild_x_descs = [&]() {
    {
      uint64_t gmem_shape[2]  = {(uint64_t)logical_batch_size, (uint64_t)REDUCTION_SIZE};
      uint64_t gmem_stride[2] = {1, (uint64_t)REDUCTION_SIZE};
      uint32_t smem_shape[2]  = {(uint32_t)MMA_N, (uint32_t)bK};
      mirage::runtime::fill_tma_desc<cute::float_e2m1_t, B_FP4, M, S, 2>(
          &cache.host_x_desc, static_cast<cute::float_e2m1_t *>(input_ptr),
          gmem_shape, gmem_stride, smem_shape, 1, 1);
    }
    {
      uint64_t gmem_shape[3]  = {(uint64_t)num_n_tiles,
                                 (uint64_t)(REDUCTION_SIZE / MMA_K),
                                 (uint64_t)SF_COL_B};
      uint64_t gmem_stride[3] = {1,
                                 (uint64_t)SF_COL_B,
                                 (uint64_t)(SF_COL_B * (REDUCTION_SIZE / MMA_K))};
      uint32_t smem_shape[3]  = {1, 1, (uint32_t)SF_COL_B};
      mirage::runtime::fill_tma_desc<cute::half_t, B_SF, M, S, 3>(
          &cache.host_x_sf_desc, static_cast<cute::half_t *>(input_sf_ptr),
          gmem_shape, gmem_stride, smem_shape, 1, 1);
    }
    cudaMemcpyAsync(cache.desc_x_ptr,    &cache.host_x_desc,    sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(cache.desc_x_sf_ptr, &cache.host_x_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cache.last_input_ptr    = input_ptr;
    cache.last_input_sf_ptr = input_sf_ptr;
    cache.last_padded_m     = logical_batch_size;
  };

  if (!cache.initialized) {
    {
      uint64_t gmem_shape[2]  = {(uint64_t)OUTPUT_SIZE, (uint64_t)REDUCTION_SIZE};
      uint64_t gmem_stride[2] = {1, (uint64_t)REDUCTION_SIZE};
      uint32_t smem_shape[2]  = {(uint32_t)MMA_M, (uint32_t)bK};
      mirage::runtime::fill_tma_desc<cute::float_e2m1_t, B_FP4, M, S, 2>(
          &cache.host_w_desc, static_cast<cute::float_e2m1_t *>(weight_ptr),
          gmem_shape, gmem_stride, smem_shape, 1, 1);
    }
    {
      uint64_t gmem_shape[3]  = {(uint64_t)(OUTPUT_SIZE / MMA_M),
                                 (uint64_t)(REDUCTION_SIZE / MMA_K),
                                 (uint64_t)SF_COL_A};
      uint64_t gmem_stride[3] = {1,
                                 (uint64_t)SF_COL_A,
                                 (uint64_t)(SF_COL_A * (REDUCTION_SIZE / MMA_K))};
      uint32_t smem_shape[3]  = {1, 1, (uint32_t)SF_COL_A};
      mirage::runtime::fill_tma_desc<cute::half_t, B_SF, M, S, 3>(
          &cache.host_w_sf_desc, static_cast<cute::half_t *>(weight_sf_ptr),
          gmem_shape, gmem_stride, smem_shape, 1, 1);
    }
    cudaMalloc(&cache.desc_w_ptr,    sizeof(CUtensorMap));
    cudaMalloc(&cache.desc_x_ptr,    sizeof(CUtensorMap));
    cudaMalloc(&cache.desc_w_sf_ptr, sizeof(CUtensorMap));
    cudaMalloc(&cache.desc_x_sf_ptr, sizeof(CUtensorMap));
    cudaMemcpy(cache.desc_w_ptr,    &cache.host_w_desc,    sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(cache.desc_w_sf_ptr, &cache.host_w_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cache.last_weight_ptr    = weight_ptr;
    cache.last_weight_sf_ptr = weight_sf_ptr;
    rebuild_x_descs(); 
    cache.initialized = true;
  } else {
    if (weight_ptr != cache.last_weight_ptr) {
      cuTensorMapReplaceAddress(&cache.host_w_desc, weight_ptr);
      cudaMemcpyAsync(cache.desc_w_ptr, &cache.host_w_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
      cache.last_weight_ptr = weight_ptr;
    }
    if (weight_sf_ptr != cache.last_weight_sf_ptr) {
      cuTensorMapReplaceAddress(&cache.host_w_sf_desc, weight_sf_ptr);
      cudaMemcpyAsync(cache.desc_w_sf_ptr, &cache.host_w_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
      cache.last_weight_sf_ptr = weight_sf_ptr;
    }
    if (logical_batch_size != cache.last_padded_m) {
      rebuild_x_descs();
    } else {
      if (input_ptr != cache.last_input_ptr) {
        cuTensorMapReplaceAddress(&cache.host_x_desc, input_ptr);
        cudaMemcpyAsync(cache.desc_x_ptr, &cache.host_x_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        cache.last_input_ptr = input_ptr;
      }
      if (input_sf_ptr != cache.last_input_sf_ptr) {
        cuTensorMapReplaceAddress(&cache.host_x_sf_desc, input_sf_ptr);
        cudaMemcpyAsync(cache.desc_x_sf_ptr, &cache.host_x_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        cache.last_input_sf_ptr = input_sf_ptr;
      }
    }
  }

  constexpr int num_output_tiles = OUTPUT_SIZE / MMA_M;
  dim3 grid_dim(num_output_tiles, num_n_tiles, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  int smemBytes = 224 * 1024;

  cute::Layout layout_Out = cute::make_layout(
      cute::make_shape(MMA_N, OUTPUT_SIZE),
      cute::make_stride((int)OUTPUT_SIZE, cute::Int<1>{}));
  cute::Tensor mOutput = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<float *>(output_ptr)), layout_Out);
  cute::Tensor mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<float *>(residual_ptr)), layout_Out);

  bool has_residual = (residual_ptr != nullptr);
  auto do_launch = [&](auto *kernel_ptr, auto mBias, auto mOutput, bool &configured_flag) {
    if (!configured_flag) {
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      configured_flag = true;
    }
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, (void const *)kernel_ptr,
        (void *)cache.desc_w_ptr, (void *)cache.desc_x_ptr,
        (void *)cache.desc_w_sf_ptr, (void *)cache.desc_x_sf_ptr,
        mBias, mOutput, logical_batch_size);
    CUTE_CHECK_ERROR(cudaGetLastError());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: SwapAB kernel launch failed" << std::endl;
    }
  };

  if (has_residual) {
    do_launch(&linear_nvfp4_swapAB_sm100_wrapper<T, MMA_N, OUTPUT_SIZE, REDUCTION_SIZE,
                                                 decltype(mBias), decltype(mOutput), false,
                                                 /*NUM_AB_STAGE=*/4, /*NUM_ACC_STAGE=*/2,
                                                 /*NUM_C_STAGE=*/4>,
              mBias, mOutput, cache.kernel_configured_bias);
  } else {
    do_launch(&linear_nvfp4_swapAB_sm100_wrapper<T, MMA_N, OUTPUT_SIZE, REDUCTION_SIZE,
                                                 decltype(mBias), decltype(mOutput), true,
                                                 /*NUM_AB_STAGE=*/4, /*NUM_ACC_STAGE=*/2,
                                                 /*NUM_C_STAGE=*/4>,
              mBias, mOutput, cache.kernel_configured_no_bias);
  }
}

// Select MMA_N based on batch_size to minimize grid size (following trtllm heuristic).
// Each tile covers MMA_N rows; grid Y = ceil(batch_size/MMA_N).
static int select_mma_n(int batch_size) {
  if (batch_size <=   8) return   8;
  if (batch_size <=  16) return  16;
  if (batch_size <=  32) return  32;
  if (batch_size <=  64) return  64;
  return 128;
}

template <typename T, int OUTPUT_SIZE, int MMA_N>
void dispatch_linear_nvfp4_swapAB_by_k(int reduction_size,
                                       int batch_size,
                                       void *input_ptr,
                                       void *input_sf_ptr,
                                       void *weight_ptr,
                                       void *weight_sf_ptr,
                                       void *output_ptr,
                                       void *residual_ptr) {
  switch (reduction_size) {
    case 256:
      launch_linear_nvfp4_swapAB_sm100<T, MMA_N, OUTPUT_SIZE, 256>(
          input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr, batch_size);
      break;
    case 512:
      launch_linear_nvfp4_swapAB_sm100<T, MMA_N, OUTPUT_SIZE, 512>(
          input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr, batch_size);
      break;
    case 768:
      launch_linear_nvfp4_swapAB_sm100<T, MMA_N, OUTPUT_SIZE, 768>(
          input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr, batch_size);
      break;
    case 1024:
      launch_linear_nvfp4_swapAB_sm100<T, MMA_N, OUTPUT_SIZE, 1024>(
          input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr, batch_size);
      break;
    case 1536:
      launch_linear_nvfp4_swapAB_sm100<T, MMA_N, OUTPUT_SIZE, 1536>(
          input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr, batch_size);
      break;
    case 2048:
      launch_linear_nvfp4_swapAB_sm100<T, MMA_N, OUTPUT_SIZE, 2048>(
          input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr, batch_size);
      break;
    case 4096:
      launch_linear_nvfp4_swapAB_sm100<T, MMA_N, OUTPUT_SIZE, 4096>(
          input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr, batch_size);
      break;
    case 7168:
      launch_linear_nvfp4_swapAB_sm100<T, MMA_N, OUTPUT_SIZE, 7168>(
          input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr, batch_size);
      break;
    default:
      TORCH_CHECK(
          false,
          "Small-M SM100 NVFP4 custom kernel supports K in {256, 512, 768, 1024, 1536, 2048, 4096, 7168}. "
          "Got K=",
          reduction_size);
  }
}

template <typename T, int MMA_N>
void dispatch_linear_nvfp4_swapAB_by_n(int output_size,
                                       int reduction_size,
                                       int batch_size,
                                       void *input_ptr,
                                       void *input_sf_ptr,
                                       void *weight_ptr,
                                       void *weight_sf_ptr,
                                       void *output_ptr,
                                       void *residual_ptr) {
  switch (output_size) {
    case 128:
      dispatch_linear_nvfp4_swapAB_by_k<T, 128, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 256:
      dispatch_linear_nvfp4_swapAB_by_k<T, 256, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 384:
      dispatch_linear_nvfp4_swapAB_by_k<T, 384, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 512:
      dispatch_linear_nvfp4_swapAB_by_k<T, 512, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 768:
      dispatch_linear_nvfp4_swapAB_by_k<T, 768, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 1024:
      dispatch_linear_nvfp4_swapAB_by_k<T, 1024, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 1536:
      dispatch_linear_nvfp4_swapAB_by_k<T, 1536, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 2048:
      dispatch_linear_nvfp4_swapAB_by_k<T, 2048, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 4096:
      dispatch_linear_nvfp4_swapAB_by_k<T, 4096, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 7168:
      dispatch_linear_nvfp4_swapAB_by_k<T, 7168, MMA_N>(
          reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    default:
      TORCH_CHECK(
          false,
          "Small-M SM100 NVFP4 custom kernel supports N in {128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 7168}. "
          "Got N=",
          output_size);
  }
}

template <typename T>
void dispatch_linear_nvfp4_swapAB(int output_size,
                                  int reduction_size,
                                  int batch_size,
                                  void *input_ptr,
                                  void *input_sf_ptr,
                                  void *weight_ptr,
                                  void *weight_sf_ptr,
                                  void *output_ptr,
                                  void *residual_ptr) {
  int mma_n = select_mma_n(batch_size);
  switch (mma_n) {
    case 8:
      dispatch_linear_nvfp4_swapAB_by_n<T, 8>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 16:
      dispatch_linear_nvfp4_swapAB_by_n<T, 16>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 32:
      dispatch_linear_nvfp4_swapAB_by_n<T, 32>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 64:
      dispatch_linear_nvfp4_swapAB_by_n<T, 64>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 128:
      dispatch_linear_nvfp4_swapAB_by_n<T, 128>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    default:
      TORCH_CHECK(false, "Unexpected mma_n=", mma_n);
  }
}

// ============================================================
// Python entry points
// ============================================================

namespace {

constexpr int OUTPUT_SIZE = 128;
constexpr int REDUCTION_SIZE = 768;
constexpr int BATCH_SIZE = 4096;
constexpr int SCALE_VEC_SIZE = 16;
constexpr int QUANTIZE_THREADS = 128;

template <typename T, int HIDDEN_SIZE>
__global__ __launch_bounds__(QUANTIZE_THREADS, 1)
void quantize_nvfp4_sm100_wrapper(T const *input_ptr,
                                  uint8_t *output_q_ptr,
                                  uint8_t *output_s_ptr,
                                  int batch_size,
                                  int mma_n) {
  kernel::quantize_nvfp4_sm100_task_impl<HIDDEN_SIZE,
                                         SCALE_VEC_SIZE,
                                         HIDDEN_SIZE,
                                         T>(
      input_ptr, output_q_ptr, output_s_ptr, batch_size, 1.0e-6f,
      /*min_4bit=*/-6.0f, /*max_4bit=*/6.0f,
      /*scale_outer_stride=*/32 * 4 * 4, mma_n);
}

// mma_n == 0  → interleaved layout [padded/128, K/64, 32, 4, 4]  (for 1d2d path)
// mma_n >  0  → per-tile swapAB layout [ceil(batch/mma_n), K/64, 32, 4, 4]
template <int HIDDEN_SIZE>
std::vector<torch::Tensor> launch_quantize_nvfp4_sm100(torch::Tensor const &input,
                                                       int mma_n) {
  const int batch_size = static_cast<int>(input.size(0));
  const int padded_batch_size = ((batch_size + 127) / 128) * 128;
  const int sf_k_outer = HIDDEN_SIZE / 64;

  auto output_q = torch::empty({padded_batch_size, HIDDEN_SIZE / 2},
                               input.options().dtype(torch::kUInt8));

  at::Tensor output_s;
  if (mma_n > 0) {
    // layout for swapAB
    const int num_n_tiles = (batch_size + mma_n - 1) / mma_n;
    output_s = torch::empty({num_n_tiles, sf_k_outer, 32, 4, 4},
                            input.options().dtype(torch::kUInt8));
  } else {
    // layout for 1d2d path
    output_s = torch::empty({padded_batch_size / 128, sf_k_outer, 32, 4, 4},
                            input.options().dtype(torch::kUInt8));
  }

  quantize_nvfp4_sm100_wrapper<float, HIDDEN_SIZE>
      <<<dim3(padded_batch_size), dim3(QUANTIZE_THREADS)>>>(
          static_cast<float const *>(input.data_ptr()),
          static_cast<uint8_t *>(output_q.data_ptr()),
          static_cast<uint8_t *>(output_s.data_ptr()),
          batch_size, mma_n);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "quantize_nvfp4_sm100 launch failed: ",
              cudaGetErrorString(err));
  return {output_q, output_s};
}

std::vector<torch::Tensor> dispatch_quantize_nvfp4_sm100(torch::Tensor const &input,
                                                         int mma_n = 0) {
  const int hidden_size = static_cast<int>(input.size(1));
  TORCH_CHECK(hidden_size % 64 == 0,
              "input.shape[1] must be divisible by 64");

  switch (hidden_size) {
    case 128:  return launch_quantize_nvfp4_sm100<128>(input, mma_n);
    case 256:  return launch_quantize_nvfp4_sm100<256>(input, mma_n);
    case 384:  return launch_quantize_nvfp4_sm100<384>(input, mma_n);
    case 512:  return launch_quantize_nvfp4_sm100<512>(input, mma_n);
    case 768:  return launch_quantize_nvfp4_sm100<768>(input, mma_n);
    case 1024: return launch_quantize_nvfp4_sm100<1024>(input, mma_n);
    case 1536: return launch_quantize_nvfp4_sm100<1536>(input, mma_n);
    case 2048: return launch_quantize_nvfp4_sm100<2048>(input, mma_n);
    case 4096: return launch_quantize_nvfp4_sm100<4096>(input, mma_n);
    case 7168: return launch_quantize_nvfp4_sm100<7168>(input, mma_n);
    default:
      TORCH_CHECK(
          false,
          "quantize_nvfp4_sm100 supports K in {128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 7168}. Got K=",
          hidden_size);
  }
}

void launch_linear_nvfp4_small_batch(torch::Tensor const& input,
                                     torch::Tensor const& input_sf,
                                     torch::Tensor const& weight,
                                     torch::Tensor const& weight_sf,
                                     c10::optional<at::Tensor> const& residual,
                                     torch::Tensor const& output,
                                     int reduction_size,
                                     int batch_size) {
  TORCH_CHECK(batch_size >= 1 && batch_size <= 128,
              "launch_linear_nvfp4_small_batch supports 1 <= batch_size <= 128, got ", batch_size);
  const int output_size = static_cast<int>(weight.size(0));

  // input_sf is already in per-tile swapAB layout [num_n_tiles, sf_k_outer, 32, 4, 4]
  // produced directly by the quantizer — no restructuring needed.
  dispatch_linear_nvfp4_swapAB<cute::float_e2m1_t>(
      output_size,
      reduction_size,
      batch_size,
      input.data_ptr(),
      input_sf.data_ptr(),
      weight.data_ptr(),
      weight_sf.data_ptr(),
      output.data_ptr(),
      residual.has_value() ? residual->data_ptr() : nullptr);
}

void launch_linear_nvfp4(torch::Tensor const& input,
                         torch::Tensor const& input_sf,
                         torch::Tensor const& weight,
                         torch::Tensor const& weight_sf,
                         c10::optional<at::Tensor> const& residual,
                         torch::Tensor const& output,
                         int batch_size,
                         int output_size,
                         int reduction_size) {
  TORCH_CHECK(output_size == OUTPUT_SIZE && reduction_size == REDUCTION_SIZE,
              "The normal SM100 NVFP4 path currently supports only N=",
              OUTPUT_SIZE, " and K=", REDUCTION_SIZE,
              ". Got N=", output_size, ", K=", reduction_size);
  TORCH_CHECK(batch_size == BATCH_SIZE,
              "Unified SM100 entry point currently supports batch_size <= 128 "
              "via SwapAB or batch_size == ", BATCH_SIZE,
              " via the normal kernel. Got ", batch_size);

  launch_linear_nvfp4_1d2d_sm100<cute::float_e2m1_t,
                                 BATCH_SIZE,
                                 OUTPUT_SIZE,
                                 REDUCTION_SIZE>(
      input.data_ptr(),
      input_sf.data_ptr(),
      weight.data_ptr(),
      weight_sf.data_ptr(),
      output.data_ptr(),
      residual.has_value() ? residual->data_ptr() : nullptr);
}

void dispatch_linear_nvfp4(torch::Tensor const& input,
                                  torch::Tensor const& input_sf,
                                  torch::Tensor const& weight,
                                  torch::Tensor const& weight_sf,
                                  c10::optional<at::Tensor> const& residual,
                                  torch::Tensor const& output,
                                  int batch_size,
                                  int output_size,
                                  int reduction_size) {
  if (batch_size <= 128) {
    launch_linear_nvfp4_small_batch(
        input, input_sf, weight, weight_sf, residual, output, reduction_size, batch_size);
  } else {
    launch_linear_nvfp4(
        input, input_sf, weight, weight_sf, residual, output, batch_size, output_size, reduction_size);
  }
}

void validate_linear_tensors(torch::Tensor const& weight,
                             torch::Tensor const& weight_sf,
                             c10::optional<at::Tensor> const& residual,
                             torch::Tensor const& output,
                             int batch_size) {
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(weight_sf.is_cuda(), "weight_sf must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(weight.dim() == 2, "weight must be rank-2");
  TORCH_CHECK(output.dim() == 2, "output must be rank-2");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(weight_sf.is_contiguous(), "weight_sf must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8,
              "weight must have dtype uint8");
  TORCH_CHECK(weight_sf.scalar_type() == torch::kUInt8,
              "weight_sf must have dtype uint8");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32,
              "output must have dtype float32");
  TORCH_CHECK(output.size(0) == batch_size,
              "output.shape[0] must equal the logical batch size");
  TORCH_CHECK(output.size(1) == weight.size(0),
              "output.shape[1] must equal weight.shape[0]");
  if (residual.has_value()) {
    TORCH_CHECK(residual->is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(residual->scalar_type() == torch::kFloat32,
                "residual must have dtype float32");
    TORCH_CHECK(residual->sizes() == output.sizes(),
                "residual must have the same shape as output");
  }
}

void check_cuda_sync(char const *label) {
  cudaError_t err = cudaPeekAtLastError();
  TORCH_CHECK(err == cudaSuccess, label, ": ", cudaGetErrorString(err));
}

}  // namespace

std::vector<torch::Tensor> quantize_nvfp4_sm100_kernel(torch::Tensor input,
                                                       int64_t mma_n = 0) {
  return dispatch_quantize_nvfp4_sm100(input, static_cast<int>(mma_n));
}

void linear_nvfp4_sm100_no_quantization_kernel(torch::Tensor input,
                                               torch::Tensor input_sf,
                                               torch::Tensor weight,
                                               torch::Tensor weight_sf,
                                               c10::optional<at::Tensor> residual,
                                               torch::Tensor output) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input_sf.is_cuda(), "input_sf must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input_sf.is_contiguous(), "input_sf must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kUInt8,
              "input must have dtype uint8");
  TORCH_CHECK(input_sf.scalar_type() == torch::kUInt8,
              "input_sf must have dtype uint8");
  TORCH_CHECK(input.size(1) == weight.size(1),
              "input.shape[1] and weight.shape[1] must match");

  const int batch_size = static_cast<int>(output.size(0));
  const int output_size = static_cast<int>(weight.size(0));
  const int reduction_size = static_cast<int>(input.size(1) * 2);
  TORCH_CHECK(input.size(0) >= batch_size,
              "input must provide at least output.shape[0] rows");
  validate_linear_tensors(weight, weight_sf, residual, output, batch_size);
  dispatch_linear_nvfp4(
      input, input_sf, weight, weight_sf, residual, output,
      batch_size, output_size, reduction_size);
  check_cuda_sync("linear_nvfp4_sm100_no_quantization");
}

// Auto-quantizing Python entry point.
// - input:     [M, K]    fp32 row-major
// - weight:    [N, K/2]  uint8
// - weight_sf: interleaved weight scale factors
// - output:    [M, N]    fp32 row-major
void linear_nvfp4_sm100_kernel(torch::Tensor input,
                               torch::Tensor weight,
                               torch::Tensor weight_sf,
                               c10::optional<at::Tensor> residual,
                               torch::Tensor output) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "input must have dtype float32");
  validate_linear_tensors(
      weight, weight_sf, residual, output, static_cast<int>(input.size(0)));
  TORCH_CHECK(weight.size(1) * 2 == input.size(1),
              "weight.shape[1] must equal input.shape[1] / 2");

  const int batch_size = static_cast<int>(input.size(0));
  const int output_size = static_cast<int>(weight.size(0));
  const int reduction_size = static_cast<int>(input.size(1));
  const int mma_n = (batch_size <= 128) ? select_mma_n(batch_size) : 0;
  auto quantized_input = dispatch_quantize_nvfp4_sm100(input, mma_n);
  dispatch_linear_nvfp4(
      quantized_input[0], quantized_input[1], weight, weight_sf, residual, output,
      batch_size, output_size, reduction_size);
  check_cuda_sync("linear_nvfp4_sm100");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_nvfp4_sm100", &quantize_nvfp4_sm100_kernel,
        "SM100 NVFP4 quantize. mma_n=0 → interleaved layout for 1d2d path; "
        "mma_n>0 → per-tile swapAB layout [ceil(M/mma_n), K/64, 32, 4, 4].",
        pybind11::arg("input"), pybind11::arg("mma_n") = 0);
  m.def("linear_nvfp4_sm100_no_quantization", &linear_nvfp4_sm100_no_quantization_kernel,
        "SM100 NVFP4 linear entry point expecting uint8 activations plus interleaved activation scale factors.");
  m.def("linear_nvfp4_sm100", &linear_nvfp4_sm100_kernel,
        "SM100 NVFP4 linear entry point that quantizes float32 activations before dispatching to the no-quantization kernel.");
}
