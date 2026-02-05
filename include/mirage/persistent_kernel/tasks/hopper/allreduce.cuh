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

#pragma once
#include "device_host/nvshmem_types.h"
#include "tasks/common/common_header.cuh"

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace kernel {

#ifdef USE_NVSHMEM

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void nvshmem_tile_allreduce(void *input_ptr,
                                                       void *output_ptr,
                                                       void *_teams,
                                                       int task_offset,
                                                       int active_tokens) {
  // Output stride is the same as hidden size
  using c_hidden = ConstInt<OUTPUT_STRIDE>;
  using c_output = ConstInt<OUTPUT_SIZE>;
  using c_batch = ConstInt<BATCH_SIZE>;
  using c_1 = ConstInt<1>;
  auto tile_shape =
      nvshmemx::make_shape<c_output, int>(c_output{}, active_tokens);
  auto tile_stride = nvshmemx::make_stride<c_1, c_hidden>(c_1{}, c_hidden{});
  auto tile_layout = nvshmemx::make_layout(tile_shape, tile_stride);
  auto src_tensor =
      nvshmemx::Tensor<T, decltype(tile_layout)>(reinterpret_cast<T*>(input_ptr), tile_layout);
  auto dst_tensor =
      nvshmemx::Tensor<T, decltype(tile_layout)>(reinterpret_cast<T*>(output_ptr), tile_layout);
  
  nvshmem_team_t* teams = reinterpret_cast<nvshmem_team_t*>(_teams);

  struct empty {};
  nvshmemx::tile_sum_reduce_block<
      decltype(src_tensor),
      decltype(dst_tensor),
      empty,
      nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(
      teams[task_offset], src_tensor, dst_tensor, empty{}, empty{}, 0, 0);
  
  // // Ensure completion of the NBI tile collective before reusing dst.
  // nvshmemx::tile_collective_wait_block<
  //     nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(teams[task_offset], 0);
}

/**
 * @brief Warp level nvshmem tile allreduce.
 * 
 * @note To use this function, we need a huge number of teams (one per warp).
 *       This would introduce significant setup overhead.
 * @tparam T
 * @tparam BATCH_SIZE 
 * @tparam OUTPUT_SIZE 
 * @tparam OUTPUT_STRIDE 
 * @param input_ptr 
 * @param output_ptr 
 * @param _teams NVSHMEM teams pointer. Must have at least one team per warp.
 * @param task_offset The task offset (thread block level).
 * @param active_tokens 
 */
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void nvshmem_tile_allreduce_warp(void *input_ptr,
                                                       void *output_ptr,
                                                       void *_teams,
                                                       int task_offset,
                                                       int active_tokens) {
  // Output stride is the same as hidden size
  using c_hidden = ConstInt<OUTPUT_STRIDE>;
  using c_output = ConstInt<OUTPUT_SIZE>;
  using c_batch = ConstInt<BATCH_SIZE>;
  using c_1 = ConstInt<1>;

  constexpr int WARP_SIZE = 32;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  const int tokens_per_warp = active_tokens + num_warps - 1;
  const int token_start = warp_id * tokens_per_warp;
  int tokens_this_warp = 0;
  if (tokens_per_warp > 0) {
    int remaining = active_tokens - token_start;
    if (remaining > 0) {
      tokens_this_warp = remaining < tokens_per_warp ? remaining : tokens_per_warp;
    }
  }

  // Skip warps that are not mapped to any active token.
  if (tokens_this_warp <= 0) {
    return;
  }

  auto tile_shape =
      nvshmemx::make_shape<c_output, int>(c_output{}, tokens_this_warp);
  auto tile_stride = nvshmemx::make_stride<c_1, c_hidden>(c_1{}, c_hidden{});
  auto tile_layout = nvshmemx::make_layout(tile_shape, tile_stride);

  T *src_base = reinterpret_cast<T*>(input_ptr) + token_start * OUTPUT_STRIDE;
  T *dst_base = reinterpret_cast<T*>(output_ptr) + token_start * OUTPUT_STRIDE;
  auto src_tensor = nvshmemx::Tensor<T, decltype(tile_layout)>(src_base, tile_layout);
  auto dst_tensor = nvshmemx::Tensor<T, decltype(tile_layout)>(dst_base, tile_layout);
  
  nvshmem_team_t* teams = reinterpret_cast<nvshmem_team_t*>(_teams);

  struct empty {};
  nvshmemx::tile_sum_reduce_warp<
      decltype(src_tensor),
      decltype(dst_tensor),
      empty,
      nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(
      teams[task_offset], src_tensor, dst_tensor, empty{}, empty{}, 0, 0);

  // Ensure completion of the NBI tile collective before reusing dst.
  nvshmemx::tile_collective_wait_warp<
      nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(teams[task_offset], 0);
}

#endif // USE_NVSHMEM

} // namespace kernel