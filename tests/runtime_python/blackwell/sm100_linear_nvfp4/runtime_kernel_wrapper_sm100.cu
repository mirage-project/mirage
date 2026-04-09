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
#include "hopper/tma_2d_nvfp4.cuh"
#include "hopper/tma_3d.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>

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
                                     void *tma_out_desc_ptr) {
  constexpr int MMA_K          = 64;
  constexpr int NUM_MMA_K      = 4;
  constexpr int bK             = MMA_K * NUM_MMA_K;
  constexpr int SCALE_VECTOR_SIZE = 16;
  constexpr int B_FP4 = 3;
  constexpr int B_SF  = 0;
  constexpr int B_OUT = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  using TMA_A = kernel::tma::tma_2d_nvfp4<cute::float_e2m1_t, B_FP4, M, S,
      BATCH_SIZE, REDUCTION_SIZE, MMA_M, bK, REDUCTION_SIZE, 1, 1, 1, MMA_M * bK / 2, true>;
  using TMA_B = kernel::tma::tma_2d_nvfp4<cute::float_e2m1_t, B_FP4, M, S,
      OUTPUT_SIZE, REDUCTION_SIZE, MMA_N, bK, REDUCTION_SIZE, 1, 1, 1, MMA_N * bK / 2, true>;

  constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;
  constexpr int SF_COL_B = MMA_N * MMA_K / SCALE_VECTOR_SIZE / 2;

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

  constexpr int SWIZZLE_SIZE = 128 / sizeof(float);
  using TMA_OUT = kernel::tma::tma_3d<float, B_OUT, M, S,
      BATCH_SIZE, OUTPUT_SIZE / SWIZZLE_SIZE, SWIZZLE_SIZE,
      MMA_M, MMA_N / SWIZZLE_SIZE, SWIZZLE_SIZE,
      OUTPUT_SIZE, SWIZZLE_SIZE, 1,
      1, 1, MMA_M * MMA_N, true>;

  TMA_A   tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B   tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_SFA tma_sfa(static_cast<CUtensorMap *>(tma_sfa_desc_ptr));
  TMA_SFB tma_sfb(static_cast<CUtensorMap *>(tma_sfb_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

  kernel::linear_nvfp4_1d2d_sm100_task_impl<T, TMA_A, TMA_B, TMA_SFA, TMA_SFB,
                                             BiasTensor, TMA_OUT,
                                             MMA_M, MMA_N,
                                             BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                                             SCALE_VECTOR_SIZE, NoBias, /*SplitK=*/false,
                                             NUM_AB_STAGE, NUM_ACC_STAGE, NUM_C_STAGE>(
      tma_a, tma_b, tma_sfa, tma_sfb, mBias, tma_out);
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

  constexpr int NUM_M_TILE_PER_CTA = 1;
  constexpr int NUM_N_TILE_PER_CTA = 1;
  constexpr int bM = MMA_M;
  constexpr int bN = MMA_N;

  CUtensorMap host_i_desc, host_i_sf_desc, host_w_desc, host_w_sf_desc, host_o_desc;
  CUtensorMap *desc_i_ptr, *desc_i_sf_ptr, *desc_w_ptr, *desc_w_sf_ptr, *desc_o_ptr;

  // TMA A (input): MMA_M rows x bK=256 FP4 cols per transaction (128B, fits 128B swizzle)
  {
    uint64_t gmem_shape[2]  = {(uint64_t)BATCH_SIZE, (uint64_t)REDUCTION_SIZE};
    uint64_t gmem_stride[2] = {1, (uint64_t)REDUCTION_SIZE};
    uint32_t smem_shape[2]  = {(uint32_t)MMA_M, (uint32_t)bK};
    mirage::runtime::fill_tma_desc<cute::float_e2m1_t, B_FP4, M, S, 2>(
        &host_i_desc, static_cast<cute::float_e2m1_t *>(input_ptr),
        gmem_shape, gmem_stride, smem_shape, 1, 1);
  }

  // TMA SFA (input scale factors)
  {
    constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;
    uint64_t gmem_shape[3]  = {(uint64_t)(BATCH_SIZE / MMA_M), (uint64_t)(REDUCTION_SIZE / MMA_K), (uint64_t)SF_COL_A};
    uint64_t gmem_stride[3] = {1, (uint64_t)SF_COL_A, (uint64_t)(SF_COL_A * (REDUCTION_SIZE / MMA_K))};
    uint32_t smem_shape[3]  = {1, 1, (uint32_t)SF_COL_A};
    mirage::runtime::fill_tma_desc<cute::half_t, B_SF, M, S, 3>(
        &host_i_sf_desc, static_cast<cute::half_t *>(input_sf_ptr),
        gmem_shape, gmem_stride, smem_shape, 1, 1);
  }

  // TMA B (weight): MMA_N rows x bK=256 FP4 cols per transaction
  {
    uint64_t gmem_shape[2]  = {(uint64_t)OUTPUT_SIZE, (uint64_t)REDUCTION_SIZE};
    uint64_t gmem_stride[2] = {1, (uint64_t)REDUCTION_SIZE};
    uint32_t smem_shape[2]  = {(uint32_t)MMA_N, (uint32_t)bK};
    mirage::runtime::fill_tma_desc<cute::float_e2m1_t, B_FP4, M, S, 2>(
        &host_w_desc, static_cast<cute::float_e2m1_t *>(weight_ptr),
        gmem_shape, gmem_stride, smem_shape, 1, 1);
  }

  // TMA SFB (weight scale factors)
  {
    constexpr int SF_COL_B = MMA_N * MMA_K / SCALE_VECTOR_SIZE / 2;
    uint64_t gmem_shape[3]  = {(uint64_t)(OUTPUT_SIZE / MMA_N), (uint64_t)(REDUCTION_SIZE / MMA_K), (uint64_t)SF_COL_B};
    uint64_t gmem_stride[3] = {1, (uint64_t)SF_COL_B, (uint64_t)(SF_COL_B * (REDUCTION_SIZE / MMA_K))};
    uint32_t smem_shape[3]  = {1, 1, (uint32_t)SF_COL_B};
    mirage::runtime::fill_tma_desc<cute::half_t, B_SF, M, S, 3>(
        &host_w_sf_desc, static_cast<cute::half_t *>(weight_sf_ptr),
        gmem_shape, gmem_stride, smem_shape, 1, 1);
  }

  // TMA output
  {
    constexpr int SWIZZLE_SIZE = 128 / sizeof(float);
    uint64_t gmem_shape[3]  = {(uint64_t)BATCH_SIZE, (uint64_t)(OUTPUT_SIZE / SWIZZLE_SIZE), (uint64_t)SWIZZLE_SIZE};
    uint64_t gmem_stride[3] = {1, (uint64_t)SWIZZLE_SIZE, (uint64_t)OUTPUT_SIZE};
    uint32_t smem_shape[3]  = {(uint32_t)MMA_M, (uint32_t)(MMA_N / SWIZZLE_SIZE), (uint32_t)SWIZZLE_SIZE};
    mirage::runtime::fill_tma_desc<float, B_OUT, M, S, 3>(
        &host_o_desc, static_cast<float *>(output_ptr),
        gmem_shape, gmem_stride, smem_shape, 1, 1);
  }

  cudaMalloc(&desc_i_ptr,    sizeof(CUtensorMap));
  cudaMalloc(&desc_i_sf_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_w_ptr,    sizeof(CUtensorMap));
  cudaMalloc(&desc_w_sf_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_o_ptr,    sizeof(CUtensorMap));

  cudaMemcpy(desc_i_ptr,    &host_i_desc,    sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_i_sf_ptr, &host_i_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_w_ptr,    &host_w_desc,    sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_w_sf_ptr, &host_w_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_o_ptr,    &host_o_desc,    sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  cute::Layout layout_Bias = cute::make_layout(
      cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
      cute::make_stride(OUTPUT_SIZE, cute::Int<1>{}));
  cute::Tensor mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<float *>(residual_ptr)), layout_Bias);

  constexpr int num_tiles_m = BATCH_SIZE / bM / NUM_M_TILE_PER_CTA;
  constexpr int num_tiles_n = OUTPUT_SIZE / bN / NUM_N_TILE_PER_CTA;
  dim3 grid_dim(num_tiles_m, num_tiles_n, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  constexpr int NUM_C_STAGE_LAUNCH = 1;
  int smemBytes = 224 * 1024;

  bool has_residual = (residual_ptr != nullptr);
  auto launch = [&](auto *kernel_ptr) {
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, (void const *)kernel_ptr,
        (void *)desc_i_ptr, (void *)desc_w_ptr,
        (void *)desc_i_sf_ptr, (void *)desc_w_sf_ptr,
        mBias, (void *)desc_o_ptr);
    CUTE_CHECK_LAST();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel launch" << std::endl;
    }
  };

  if (has_residual) {
    launch(&linear_nvfp4_1d2d_sm100_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                                            decltype(mBias), MMA_M, MMA_N, false,
                                            /*NUM_AB_STAGE=*/4, /*NUM_ACC_STAGE=*/2,
                                            NUM_C_STAGE_LAUNCH>);
  } else {
    launch(&linear_nvfp4_1d2d_sm100_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                                            decltype(mBias), MMA_M, MMA_N, true,
                                            /*NUM_AB_STAGE=*/4, /*NUM_ACC_STAGE=*/2,
                                            NUM_C_STAGE_LAUNCH>);
  }
}

void linear_nvfp4_1d2d_sm100_kernel(torch::Tensor input,
                                    torch::Tensor input_sf,
                                    torch::Tensor weight,
                                    torch::Tensor weight_sf,
                                    c10::optional<at::Tensor> residual,
                                    torch::Tensor output) {
  void *input_ptr    = input.data_ptr();
  void *input_sf_ptr = input_sf.data_ptr();
  void *weight_ptr   = weight.data_ptr();
  void *weight_sf_ptr = weight_sf.data_ptr();
  void *residual_ptr = residual.has_value() ? residual->data_ptr() : nullptr;
  void *output_ptr   = output.data_ptr();

  constexpr int BATCH_SIZE     = 1024 * 4;
  constexpr int OUTPUT_SIZE    = 1024 * 4;
  constexpr int REDUCTION_SIZE = 1024 * 4;

  assert(input.size(1)  == REDUCTION_SIZE / 2);
  assert(weight.size(0) == OUTPUT_SIZE);

  launch_linear_nvfp4_1d2d_sm100<cute::float_e2m1_t, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
      input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_nvfp4_1d2d_sm100", &linear_nvfp4_1d2d_sm100_kernel, "Linear kernel SM100 nvfp4 1D2D");
}
