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
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>

// Cutlass includes
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h> // F16 data type
#include <cutlass/util/print_error.hpp>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp> // CuTe tensor implementation

#include "runtime_header.h"
#include "tasks/hopper/moe_linear_swapAB_hopper.cuh"
#include "tasks/hopper/tma_2d.cuh"
#include "tma.cuh"

using bfloat16 = cute::bfloat16_t;

// moe_linear_sm90

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          bool W13_LINEAR,
          class InputTensor,
          class BiasTensor,
          class IndicesTensor,
          class MaskTensor,
          class OutputTensor,
          int MMA_M,
          int MMA_N,
          bool NoBias,
          int NUM_AB_STAGE = 8>
__global__ __launch_bounds__(256, 1) void moe_linear_sm90_wrapper(
    void *tma_w_desc_ptr,
    InputTensor mInput,
    BiasTensor mBias,
    IndicesTensor mRoutingIndices,
    MaskTensor mMask,
    OutputTensor mOutput,
    int const expert_offset) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size
  constexpr int TILE_SIZE =
      64; // we should modify this param if we want larger tile size
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  using TMA_A =
      kernel::tma::tma_2d<bfloat16,
                          B,
                          M,
                          S,
                          NUM_EXPERTS * OUTPUT_SIZE, /*GMEM_ROW_*/
                          REDUCTION_SIZE,            /*GMEM_COL_*/
                          MMA_M,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                          REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                         /*GMEM_STRIDE_COL_*/
                          1,                         /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                          MMA_M * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;

  TMA_A tma_a(static_cast<CUtensorMap *>(tma_w_desc_ptr));

  kernel::moe_linear_sm90_task_impl<T,
                                    TMA_A,
                                    InputTensor,
                                    BiasTensor,
                                    IndicesTensor,
                                    MaskTensor,
                                    OutputTensor,
                                    MMA_M,
                                    MMA_N,
                                    BATCH_SIZE,
                                    OUTPUT_SIZE,
                                    OUTPUT_SIZE, // ORIG_OUTPUT_SIZE
                                    REDUCTION_SIZE,
                                    NUM_EXPERTS,
                                    NUM_TOPK,
                                    EXPERT_STRIDE,
                                    W13_LINEAR,
                                    NoBias,
                                    NUM_AB_STAGE>(
      tma_a, mInput, mBias, mRoutingIndices, mMask, mOutput, expert_offset);
}

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          bool W13_LINEAR = true>
void launch_moe_linear_sm90(int const expert_offset,
                            void *input_ptr,
                            void *weight_ptr,
                            void *mpk_routing_indices_ptr,
                            void *mpk_expert_mask_ptr,
                            void *output_ptr,
                            void *residual_ptr = nullptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int MMA_M = 64;
  constexpr int MMA_N = 16;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size
  constexpr int TILE_SIZE =
      64; // we should modify this param if we want larger tile size

  CUtensorMap host_w_desc;
  CUtensorMap *desc_w_ptr;

  // TMA_WEIGHT, for input we do cp_async in the kernel
  uint64_t w_gmem_shape[2] = {static_cast<uint64_t>(NUM_EXPERTS * OUTPUT_SIZE),
                              static_cast<uint64_t>(REDUCTION_SIZE)};
  uint64_t w_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE)};
  uint32_t w_smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                              static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  size_t w_smem_repeat_col =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 2>(
      &host_w_desc,
      static_cast<bfloat16 *>(weight_ptr),
      w_gmem_shape,
      w_gmem_stride,
      w_smem_shape,
      1,
      w_smem_repeat_col);

  cudaMalloc(&desc_w_ptr, sizeof(CUtensorMap));
  cudaMemcpy(
      desc_w_ptr, &host_w_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  void *tma_desc_weight;
  tma_desc_weight = desc_w_ptr;

  // Residual
  cute::Layout layout_bias = cute::make_layout(
      cute::make_shape(BATCH_SIZE, OUTPUT_SIZE, NUM_EXPERTS),
      cute::make_stride(OUTPUT_SIZE, cute::Int<1>{}, BATCH_SIZE * OUTPUT_SIZE));
  cute::Tensor mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<T *>(residual_ptr)), layout_bias);

  // Topk_indices
  cute::Layout layout_routing_indices =
      cute::make_layout(cute::make_shape(NUM_EXPERTS, BATCH_SIZE),
                        cute::make_stride(BATCH_SIZE, cute::Int<1>{}));
  cute::Tensor mRoutingIndices = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<int32_t *>(mpk_routing_indices_ptr)),
      layout_routing_indices);

  // Topk_weights
  cute::Layout layout_expert_mask = cute::make_layout(
      cute::make_shape(NUM_EXPERTS + 1), cute::make_stride(cute::Int<1>{}));
  cute::Tensor mMask = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<int32_t *>(mpk_expert_mask_ptr)),
      layout_expert_mask);

  // Output
  // NOTE(Yu): (B,K,O) will coalesce, so we use (B,O,K)
  //  cute::Layout layout_output =
  //  cute::make_layout(cute::make_shape(BATCH_SIZE, NUM_TOPK, OUTPUT_SIZE),
  //  cute::make_stride(NUM_TOPK * OUTPUT_SIZE, OUTPUT_SIZE, cute::Int<1>{}));
  cute::Layout layout_output = cute::make_layout(
      cute::make_shape(BATCH_SIZE, OUTPUT_SIZE, NUM_TOPK),
      cute::make_stride(NUM_TOPK * OUTPUT_SIZE, cute::Int<1>{}, OUTPUT_SIZE));
  cute::Tensor mOutput = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<T *>(output_ptr)), layout_output);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  int smemBytes = 224 * 1024;

  // Input
  if constexpr (W13_LINEAR) {
    cute::Layout layout_input =
        cute::make_layout(cute::make_shape(BATCH_SIZE, REDUCTION_SIZE),
                          cute::make_stride(REDUCTION_SIZE, cute::Int<1>{}));
    cute::Tensor mInput = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<T *>(input_ptr)), layout_input);
    if (residual_ptr != nullptr) {
      auto *kernel_ptr = &moe_linear_sm90_wrapper<T,
                                                  BATCH_SIZE,
                                                  OUTPUT_SIZE,
                                                  REDUCTION_SIZE,
                                                  NUM_EXPERTS,
                                                  NUM_TOPK,
                                                  EXPERT_STRIDE,
                                                  W13_LINEAR,
                                                  decltype(mInput),
                                                  decltype(mBias),
                                                  decltype(mRoutingIndices),
                                                  decltype(mMask),
                                                  decltype(mOutput),
                                                  MMA_M,
                                                  MMA_N,
                                                  false>;
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      cutlass::ClusterLaunchParams params = {
          grid_dim, block_dim, cluster_dim, smemBytes};
      cutlass::Status status =
          cutlass::launch_kernel_on_cluster(params,
                                            (void const *)kernel_ptr,
                                            tma_desc_weight,
                                            mInput,
                                            mBias,
                                            mRoutingIndices,
                                            mMask,
                                            mOutput,
                                            expert_offset);
      CUTE_CHECK_LAST();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
      }
    } else {
      auto *kernel_ptr = &moe_linear_sm90_wrapper<T,
                                                  BATCH_SIZE,
                                                  OUTPUT_SIZE,
                                                  REDUCTION_SIZE,
                                                  NUM_EXPERTS,
                                                  NUM_TOPK,
                                                  EXPERT_STRIDE,
                                                  W13_LINEAR,
                                                  decltype(mInput),
                                                  decltype(mBias),
                                                  decltype(mRoutingIndices),
                                                  decltype(mMask),
                                                  decltype(mOutput),
                                                  MMA_M,
                                                  MMA_N,
                                                  true>;
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      cutlass::ClusterLaunchParams params = {
          grid_dim, block_dim, cluster_dim, smemBytes};
      cutlass::Status status =
          cutlass::launch_kernel_on_cluster(params,
                                            (void const *)kernel_ptr,
                                            tma_desc_weight,
                                            mInput,
                                            mBias,
                                            mRoutingIndices,
                                            mMask,
                                            mOutput,
                                            expert_offset);
      CUTE_CHECK_LAST();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
      }
    }
  } else {
    cute::Layout layout_input = cute::make_layout(
        cute::make_shape(BATCH_SIZE, REDUCTION_SIZE, NUM_TOPK),
        cute::make_stride(
            REDUCTION_SIZE * NUM_TOPK, cute::Int<1>{}, REDUCTION_SIZE));
    cute::Tensor mInput = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<T *>(input_ptr)), layout_input);
    if (residual_ptr != nullptr) {
      auto *kernel_ptr = &moe_linear_sm90_wrapper<T,
                                                  BATCH_SIZE,
                                                  OUTPUT_SIZE,
                                                  REDUCTION_SIZE,
                                                  NUM_EXPERTS,
                                                  NUM_TOPK,
                                                  EXPERT_STRIDE,
                                                  W13_LINEAR,
                                                  decltype(mInput),
                                                  decltype(mBias),
                                                  decltype(mRoutingIndices),
                                                  decltype(mMask),
                                                  decltype(mOutput),
                                                  MMA_M,
                                                  MMA_N,
                                                  false>;
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      cutlass::ClusterLaunchParams params = {
          grid_dim, block_dim, cluster_dim, smemBytes};
      cutlass::Status status =
          cutlass::launch_kernel_on_cluster(params,
                                            (void const *)kernel_ptr,
                                            tma_desc_weight,
                                            mInput,
                                            mBias,
                                            mRoutingIndices,
                                            mMask,
                                            mOutput,
                                            expert_offset);
      CUTE_CHECK_LAST();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
      }
    } else {
      auto *kernel_ptr = &moe_linear_sm90_wrapper<T,
                                                  BATCH_SIZE,
                                                  OUTPUT_SIZE,
                                                  REDUCTION_SIZE,
                                                  NUM_EXPERTS,
                                                  NUM_TOPK,
                                                  EXPERT_STRIDE,
                                                  W13_LINEAR,
                                                  decltype(mInput),
                                                  decltype(mBias),
                                                  decltype(mRoutingIndices),
                                                  decltype(mMask),
                                                  decltype(mOutput),
                                                  MMA_M,
                                                  MMA_N,
                                                  true>;
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      cutlass::ClusterLaunchParams params = {
          grid_dim, block_dim, cluster_dim, smemBytes};
      cutlass::Status status =
          cutlass::launch_kernel_on_cluster(params,
                                            (void const *)kernel_ptr,
                                            tma_desc_weight,
                                            mInput,
                                            mBias,
                                            mRoutingIndices,
                                            mMask,
                                            mOutput,
                                            expert_offset);
      CUTE_CHECK_LAST();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
      }
    }
  }
}

void moe_w13_linear_sm90_kernel(torch::Tensor input,
                                torch::Tensor weight,
                                c10::optional<at::Tensor> residual,
                                torch::Tensor mpk_routing_indices,
                                torch::Tensor mpk_expert_mask,
                                torch::Tensor output) {

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  bool has_residual = residual.has_value();
  void *residual_ptr = has_residual ? residual->data_ptr() : nullptr;
  void *mpk_routing_indices_ptr = mpk_routing_indices.data_ptr();
  void *mpk_expert_mask_ptr = mpk_expert_mask.data_ptr();
  void *output_ptr = output.data_ptr();

  // const int BATCH_SIZE = input.size(0);
  // const int OUTPUT_SIZE = output.size(1);
  // const int REDUCTION_SIZE = weight.size(1);

  constexpr int BATCH_SIZE = 8;
  constexpr int OUTPUT_SIZE = 64;
  constexpr int REDUCTION_SIZE = 2048;
  constexpr int NUM_EXPERTS = 128;
  constexpr int NUM_TOPK = 8;
  constexpr int EXPERT_STRIDE = 12;
  int const expert_offset = 0;

  assert(input.size(1) == REDUCTION_SIZE);
  assert(weight.size(0) == NUM_EXPERTS && weight.size(1) == OUTPUT_SIZE &&
         weight.size(2) == REDUCTION_SIZE);
  assert(mpk_routing_indices.size(0) == NUM_EXPERTS &&
         mpk_routing_indices.size(1) == BATCH_SIZE);
  assert(mpk_expert_mask.size(0) == NUM_EXPERTS + 1);
  //   assert(!has_residual);

  launch_moe_linear_sm90<bfloat16,
                         BATCH_SIZE,
                         OUTPUT_SIZE,
                         REDUCTION_SIZE,
                         NUM_EXPERTS,
                         NUM_TOPK,
                         EXPERT_STRIDE,
                         true>(expert_offset,
                               input_ptr,
                               weight_ptr,
                               mpk_routing_indices_ptr,
                               mpk_expert_mask_ptr,
                               output_ptr,
                               residual_ptr);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

void moe_w2_linear_sm90_kernel(torch::Tensor input,
                               torch::Tensor weight,
                               c10::optional<at::Tensor> residual,
                               torch::Tensor mpk_routing_indices,
                               torch::Tensor mpk_expert_mask,
                               torch::Tensor output) {

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  bool has_residual = residual.has_value();
  void *residual_ptr = has_residual ? residual->data_ptr() : nullptr;
  void *mpk_routing_indices_ptr = mpk_routing_indices.data_ptr();
  void *mpk_expert_mask_ptr = mpk_expert_mask.data_ptr();
  void *output_ptr = output.data_ptr();

  // const int BATCH_SIZE = input.size(0);
  // const int OUTPUT_SIZE = output.size(1);
  // const int REDUCTION_SIZE = weight.size(1);

  constexpr int BATCH_SIZE = 8;
  constexpr int OUTPUT_SIZE = 64;
  constexpr int REDUCTION_SIZE = 2048;
  constexpr int NUM_EXPERTS = 128;
  constexpr int NUM_TOPK = 8;
  constexpr int EXPERT_STRIDE = 12;
  int const expert_offset = 0;

  assert(input.size(0) == BATCH_SIZE && input.size(1) == NUM_TOPK &&
         input.size(2) == REDUCTION_SIZE);
  assert(weight.size(0) == NUM_EXPERTS && weight.size(1) == OUTPUT_SIZE &&
         weight.size(2) == REDUCTION_SIZE);
  assert(mpk_routing_indices.size(0) == NUM_EXPERTS &&
         mpk_routing_indices.size(1) == BATCH_SIZE);
  assert(mpk_expert_mask.size(0) == NUM_EXPERTS + 1);
  //   assert(!has_residual);

  launch_moe_linear_sm90<bfloat16,
                         BATCH_SIZE,
                         OUTPUT_SIZE,
                         REDUCTION_SIZE,
                         NUM_EXPERTS,
                         NUM_TOPK,
                         EXPERT_STRIDE,
                         false>(expert_offset,
                                input_ptr,
                                weight_ptr,
                                mpk_routing_indices_ptr,
                                mpk_expert_mask_ptr,
                                output_ptr,
                                residual_ptr);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_w13_linear_sm90",
        &moe_w13_linear_sm90_kernel,
        "MoE W13 Linear kernel sm90");
  m.def("moe_w2_linear_sm90",
        &moe_w2_linear_sm90_kernel,
        "MoE W2 Linear kernel sm90");
}