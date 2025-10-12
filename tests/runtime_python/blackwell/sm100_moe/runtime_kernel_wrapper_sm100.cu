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
#include "runtime_header.h"
#include "blackwell/task_header.cuh"
#include "hopper/tma_2d.cuh"
#include "tma.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <cstdio>

// Cutlass includes
#include <cutlass/half.h>                       // F16 data type
#include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

// CuTe includes
#include <cute/tensor.hpp>                      // CuTe tensor implementation
#include <cute/arch/cluster_sm90.hpp>           // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp>   // Compile time in constants such as _1, _256 etc.
#include <cute/algorithm/cooperative_copy.hpp>  // Auto vectorized copy operation
#include <cute/arch/tmem_allocator_sm100.hpp>   // TMEM allocator for SM100
#include <cute/pointer_flagged.hpp> 

using bfloat16 = cute::bfloat16_t;

// gate_topk_sm100

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int NUM_TOPK,
          class BiasTensor,
          class IndicesTensor,
          class WeightsTensor,
          int MMA_M,
          int MMA_N,
          bool NoBias, 
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__global__ __launch_bounds__(256, 1) void gate_topk_sm100_wrapper(
    void * tma_a_desc_ptr,
    void * tma_b_desc_ptr,
    BiasTensor mBias,
    IndicesTensor mIndices,
    WeightsTensor mWeights) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  using TypeAcc = float;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size
  constexpr int TILE_SIZE =
      64; // we should modify this param if we want larger tile size
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  
  using TMA_B =
      kernel::tma::tma_2d<bfloat16,
                          B,
                          M,
                          S,
                          BATCH_SIZE,                      /*GMEM_ROW_*/
                          REDUCTION_SIZE,                  /*GMEM_COL_*/
                          MMA_N,                           /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,               /*SMEM_COL_*/
                          REDUCTION_SIZE,                  /*GMEM_STRIDE_ROW_*/
                          1,                               /*GMEM_STRIDE_COL_*/
                          1,                               /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,         /*SMEM_REPEAT_COL_*/
                          MMA_N * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  using TMA_A =
      kernel::tma::tma_2d<bfloat16,
                          B,
                          M,
                          S,
                          OUTPUT_SIZE,             /*GMEM_ROW_*/
                          REDUCTION_SIZE,          /*GMEM_COL_*/
                          MMA_M,                   /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,       /*SMEM_COL_*/
                          REDUCTION_SIZE,          /*GMEM_STRIDE_ROW_*/
                          1,                       /*GMEM_STRIDE_COL_*/
                          1,                       /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL, /*SMEM_REPEAT_COL_*/
                          MMA_M * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;

  TMA_A tma_a(static_cast<CUtensorMap*>(tma_a_desc_ptr));
  TMA_B tma_b(static_cast<CUtensorMap*>(tma_b_desc_ptr));

  kernel::gate_topk_sm100_task_impl<
          T, 
          TMA_A,
          TMA_B,
          BiasTensor,
          IndicesTensor,
          WeightsTensor,
          MMA_M,
          MMA_N,
          BATCH_SIZE,
          OUTPUT_SIZE,
          REDUCTION_SIZE,
          NUM_TOPK,
          NoBias,
          NUM_AB_STAGE,
          NUM_ACC_STAGE,
          NUM_C_STAGE>(tma_a, tma_b, mBias, mIndices, mWeights);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE, int NUM_TOPK>
void launch_gate_topk_sm100(void *input_ptr,
                          void *weight_ptr,
                          void *topk_indices_ptr,
                          void *topk_weights_ptr,
                          void *residual_ptr = nullptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  using TypeAcc = float;

  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size
  constexpr int TILE_SIZE =
      64; // we should modify this param if we want larger tile size

  // TMA_A tma_a(weight_ptr);
  // TMA_B tma_b(input_ptr);
  // TMA_OUT tma_out(output_ptr);

  CUtensorMap host_i_desc;
  CUtensorMap host_w_desc;
  CUtensorMap *desc_i_ptr;
  CUtensorMap *desc_w_ptr;

  // TMA_INPUT
  uint64_t i_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                            static_cast<uint64_t>(REDUCTION_SIZE)};
  uint64_t i_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE)};
  uint32_t i_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                            static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};

  size_t i_smem_repeat_col =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 2>(&host_i_desc,
                                      static_cast<bfloat16 *>(input_ptr),
                                      i_gmem_shape,
                                      i_gmem_stride,
                                      i_smem_shape,
                                      1,
                                      i_smem_repeat_col);

  // TMA_WEIGHT
  uint64_t w_gmem_shape[2] = {static_cast<uint64_t>(OUTPUT_SIZE),
                            static_cast<uint64_t>(REDUCTION_SIZE)};
  uint64_t w_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE)};
  uint32_t w_smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                            static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  size_t w_smem_repeat_col =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 2>(&host_w_desc,
                                      static_cast<bfloat16 *>(weight_ptr),
                                      w_gmem_shape,
                                      w_gmem_stride,
                                      w_smem_shape,
                                      1,
                                      w_smem_repeat_col);
  
  cudaMalloc(&desc_i_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_w_ptr, sizeof(CUtensorMap));

  cudaMemcpy(desc_i_ptr, &host_i_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_w_ptr, &host_w_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  void *tma_desc_input;
  void *tma_desc_weight;

  tma_desc_input = desc_i_ptr;
  tma_desc_weight = desc_w_ptr;

  // Residual
  cute::Layout layout_Bias = cute::make_layout(cute::make_shape(BATCH_SIZE, OUTPUT_SIZE), cute::make_stride(OUTPUT_SIZE, cute::Int<1>{}));
  cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<T*>(residual_ptr)), layout_Bias);

  // Topk_indices
  cute::Layout layout_indices = cute::make_layout(cute::make_shape(BATCH_SIZE, NUM_TOPK), cute::make_stride(NUM_TOPK, cute::Int<1>{}));
  cute::Tensor mIndices = cute::make_tensor(cute::make_gmem_ptr(static_cast<int32_t*>(topk_indices_ptr)), layout_indices);

  // Topk_weights
  cute::Layout layout_weights = cute::make_layout(cute::make_shape(BATCH_SIZE, NUM_TOPK), cute::make_stride(NUM_TOPK, cute::Int<1>{}));
  cute::Tensor mWeights = cute::make_tensor(cute::make_gmem_ptr(static_cast<float*>(topk_weights_ptr)), layout_weights);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  int smemBytes = 224 * 1024;

  if(residual_ptr != nullptr){
    auto* kernel_ptr = &gate_topk_sm100_wrapper<T,
                                BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, NUM_TOPK,
                                decltype(mBias), decltype(mIndices), decltype(mWeights),
                                MMA_M, MMA_N,
                                false>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smemBytes));
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                              tma_desc_weight, tma_desc_input, mBias, mIndices, mWeights);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
  } else {
    auto* kernel_ptr = &gate_topk_sm100_wrapper<T,
                                BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, NUM_TOPK,
                                decltype(mBias), decltype(mIndices), decltype(mWeights),
                                MMA_M, MMA_N,
                                true>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smemBytes));
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                              tma_desc_weight, tma_desc_input, mBias, mIndices, mWeights);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
  }

}

void gate_topk_sm100_kernel(torch::Tensor input,
                          torch::Tensor weight,
                          c10::optional<at::Tensor> residual,
                          torch::Tensor topk_indices,
                          torch::Tensor topk_weights) {

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  bool has_residual = residual.has_value();
  void *residual_ptr = has_residual ? residual->data_ptr() : nullptr;
  void *topk_indices_ptr = topk_indices.data_ptr();
  void *topk_weights_ptr = topk_weights.data_ptr();

  // const int BATCH_SIZE = input.size(0);
  // const int OUTPUT_SIZE = output.size(1);
  // const int REDUCTION_SIZE = weight.size(1);

  constexpr int BATCH_SIZE = 8;
  constexpr int OUTPUT_SIZE = 128;
  constexpr int REDUCTION_SIZE = 2048;
  constexpr int NUM_TOPK = 8;

  assert(input.size(1) == REDUCTION_SIZE);
  assert(weight.size(0) == OUTPUT_SIZE);
  assert(topk_indices.size(0) == BATCH_SIZE && topk_indices.size(1) == NUM_TOPK);
  assert(topk_weights.size(0) == BATCH_SIZE && topk_weights.size(1) == NUM_TOPK);
  assert(!has_residual);

  launch_gate_topk_sm100<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, NUM_TOPK>(input_ptr, weight_ptr, topk_indices_ptr, topk_weights_ptr, residual_ptr);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_topk_sm100", &gate_topk_sm100_kernel, "Gate TopK kernel SM100");
}