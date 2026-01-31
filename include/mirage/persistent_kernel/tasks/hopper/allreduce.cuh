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
                                                       int task_offset) {
  // TODO(Zepeng): Avoid transferring inactive tokens
  // Output stride is the same as hidden size
  using c_hidden = ConstInt<OUTPUT_STRIDE>;
  using c_output = ConstInt<OUTPUT_SIZE>;
  using c_batch = ConstInt<BATCH_SIZE>;
  using c_1 = ConstInt<1>;
  auto tile_shape =
      nvshmemx::make_shape<c_output, c_batch>(c_output{}, c_batch{});
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

  // Ensure completion of the NBI tile collective before reusing dst.
  nvshmemx::tile_collective_wait_block<
      nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(teams[task_offset], 0);
}

#endif // USE_NVSHMEM

} // namespace kernel