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

#pragma once
#include <cuda.h>
#include "tasks/common.h"
#include "runtime_header.h"

using bfloat16 = type::bfloat16_t;

template <typename TaskDesc, typename TensorDesc>
__host__ inline CUtensorMap *
    create_tma_desc_from_tensor(TaskDesc const &task_desc,
                                TensorDesc const &tensor_desc, uint16_t const param_id) {
  CUtensorMap host_desc;
  CUtensorMap *desc_ptr;
  fill_tma_desc_by_task(&host_desc, task_desc, tensor_desc, param_id); // host-only function
  cudaMalloc(&desc_ptr, sizeof(CUtensorMap));
  cudaMemcpy(desc_ptr, &host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  return desc_ptr;
}

template <typename T, int B, int M, int S, int NDIM>
__host__ static inline void fill_tma_desc(CUtensorMap *tma_desc, void * const src, uint64_t const (&gmem_shape)[NDIM], uint64_t const (&gmem_stride)[NDIM], uint32_t const (&smem_shape)[NDIM], size_t smem_repeat_row, size_t smem_repeat_col) {
  constexpr uint32_t tma_dim = 5;
  void *global_addr = src;

  constexpr CUtensorMapDataType tma_format = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  //  (std::is_same_v<T, type::bfloat16_t> ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
  //                                       : CUtensorMapDataType(-1));
  constexpr CUtensorMapInterleave tma_interleave =
      CU_TENSOR_MAP_INTERLEAVE_NONE;
  constexpr CUtensorMapL2promotion tma_l2Promotion =
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  constexpr CUtensorMapFloatOOBfill tma_oobFill =
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
  constexpr CUtensorMapSwizzle tma_swizzle =
      (B == 1   ? CU_TENSOR_MAP_SWIZZLE_32B
      : B == 2 ? CU_TENSOR_MAP_SWIZZLE_64B
      : B == 3 ? CU_TENSOR_MAP_SWIZZLE_128B
                : CU_TENSOR_MAP_SWIZZLE_NONE);

  uint64_t gmem_prob_shape[5];
  uint64_t gmem_prob_stride[5];
  uint32_t smem_box_shape[5];
  uint32_t smem_box_stride[5];

  if constexpr (NDIM == 2) {
      gmem_prob_shape[0] = gmem_shape[1];
      gmem_prob_shape[1] = gmem_shape[0];
      gmem_prob_shape[2] = 1;
      gmem_prob_shape[3] = 1;
      gmem_prob_shape[4] = 1;
      gmem_prob_stride[0] = sizeof(T);
      gmem_prob_stride[1] = gmem_stride[1] * sizeof(T);
      gmem_prob_stride[2] = 0;
      gmem_prob_stride[3] = 0;
      gmem_prob_stride[4] = 0;
  }
  else if constexpr (NDIM == 3) {
    gmem_prob_shape[0] = gmem_shape[2];
    gmem_prob_shape[1] = gmem_shape[1];
    gmem_prob_shape[2] = gmem_shape[0];
    gmem_prob_shape[3] = 1;
    gmem_prob_shape[4] = 1;
    gmem_prob_stride[0] = sizeof(T);
    gmem_prob_stride[1] = gmem_stride[1] * sizeof(T);
    gmem_prob_stride[2] = gmem_stride[0] * sizeof(T);
    gmem_prob_stride[3] = 0;
    gmem_prob_stride[4] = 0;
  }
  else {
    assert(false);
  }

  assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) ==
        0); // Address must be 16B-aligned

  // assert(gmem_prob_shape[0] >= (uint64_t(1)));       // Size must be min 1
  // assert(gmem_prob_shape[0] <= (uint64_t(1) << 32)); // Size must be max 2^32
  // assert(gmem_prob_shape[1] >= (uint64_t(1)));       // Size must be min 1
  // assert(gmem_prob_shape[1] <= (uint64_t(1) << 32)); // Size must be max 2^32
  // assert(gmem_prob_shape[2] >= (uint64_t(1)));       // Size must be min 1
  // assert(gmem_prob_shape[2] <= (uint64_t(1) << 32)); // Size must be max 2^32
  // assert(gmem_prob_shape[3] >= (uint64_t(1)));       // Size must be min 1
  // assert(gmem_prob_shape[3] <= (uint64_t(1) << 32)); // Size must be max 2^32
  // assert(gmem_prob_shape[4] >= (uint64_t(1)));       // Size must be min 1
  // assert(gmem_prob_shape[4] <= (uint64_t(1) << 32)); // Size must be max 2^32

  // // Assert the byte strides. Tma Descriptor uses byte strides
  // assert((gmem_prob_stride[1]) <
  //       (uint64_t(1) << 40)); // Stride must be max 2^40
  // assert((gmem_prob_stride[1] & 0b1111) ==
  //       0); // Stride must be multiple of 16B (128b)
  // assert((gmem_prob_stride[2]) <
  //       (uint64_t(1) << 40)); // Stride must be max 2^40
  // assert((gmem_prob_stride[2] & 0b1111) ==
  //       0); // Stride must be multiple of 16B (128b)
  // assert((gmem_prob_stride[3]) <
  //       (uint64_t(1) << 40)); // Stride must be max 2^40
  // assert((gmem_prob_stride[3] & 0b1111) ==
  //       0); // Stride must be multiple of 16B (128b)
  // assert((gmem_prob_stride[4]) <
  //       (uint64_t(1) << 40)); // Stride must be max 2^40
  // assert((gmem_prob_stride[4] & 0b1111) ==
  //       0); // Stride must be multiple of 16B (128b)

  if constexpr (NDIM == 2) {
    smem_box_shape[0] = smem_shape[1];
    smem_box_shape[1] = smem_shape[0];
    smem_box_shape[2] = 1;
    smem_box_shape[3] = 1;
    smem_box_shape[4] = 1;
    smem_box_stride[0] = 1;
    smem_box_stride[1] = 1;
    smem_box_stride[2] = 1;
    smem_box_stride[3] = 1;
    smem_box_stride[4] = 1;
  }
  else if constexpr (NDIM == 3) {
    smem_box_shape[0] = smem_shape[2];
    smem_box_shape[1] = smem_shape[1];
    smem_box_shape[2] = smem_shape[0];
    smem_box_shape[3] = 1;
    smem_box_shape[4] = 1;
    smem_box_stride[0] = 1;
    smem_box_stride[1] = 1;
    smem_box_stride[2] = 1;
    smem_box_stride[3] = 1;
    smem_box_stride[4] = 1;
  }
  else {
    assert(false);
  }

#if 0
printf("gmem_prob_shape: %lu, %lu, %lu, %lu, %lu\n",
gmem_prob_shape[0],
gmem_prob_shape[1],
gmem_prob_shape[2],
gmem_prob_shape[3],
gmem_prob_shape[4]);
printf("gmem_prob_stride: %lu, %lu, %lu, %lu, %lu\n",
gmem_prob_stride[0],
gmem_prob_stride[1],
gmem_prob_stride[2],
gmem_prob_stride[3],
gmem_prob_stride[4]);
printf("smem_box_shape: %d, %d, %d, %d, %d\n",
smem_box_shape[0],
smem_box_shape[1],
smem_box_shape[2],
smem_box_shape[3],
smem_box_shape[4]);
printf("smem_box_stride: %d, %d, %d, %d, %d\n",
smem_box_stride[0],
smem_box_stride[1],
smem_box_stride[2],
smem_box_stride[3],
smem_box_stride[4]);
#endif

  // assert(smem_box_shape[0] >= (uint32_t(1)));      // Size must be min 1
  // assert(smem_box_shape[0] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256
  // assert(smem_box_shape[1] >= (uint32_t(1)));      // Size must be min 1
  // assert(smem_box_shape[1] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256
  // assert(smem_box_shape[2] >= (uint32_t(1)));      // Size must be min 1
  // assert(smem_box_shape[2] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256
  // assert(smem_box_shape[3] >= (uint32_t(1)));      // Size must be min 1
  // assert(smem_box_shape[3] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256
  // assert(smem_box_shape[4] >= (uint32_t(1)));      // Size must be min 1
  // assert(smem_box_shape[4] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256

  // assert(smem_box_stride[0] >= (uint32_t(1))); // Stride must be min 1
  // assert(smem_box_stride[0] <= (uint32_t(8))); // Stride must be max 2^3 = 8
  // assert(smem_box_stride[1] >= (uint32_t(1))); // Stride must be min 1
  // assert(smem_box_stride[1] <= (uint32_t(8))); // Stride must be max 2^3 = 8
  // assert(smem_box_stride[2] >= (uint32_t(1))); // Stride must be min 1
  // assert(smem_box_stride[2] <= (uint32_t(8))); // Stride must be max 2^3 = 8
  // assert(smem_box_stride[3] >= (uint32_t(1))); // Stride must be min 1
  // assert(smem_box_stride[3] <= (uint32_t(8))); // Stride must be max 2^3 = 8
  // assert(smem_box_stride[4] >= (uint32_t(1))); // Stride must be min 1
  // assert(smem_box_stride[4] <= (uint32_t(8))); // Stride must be max 2^3 = 8

  uint64_t const *gmem_shape_ptr = &gmem_prob_shape[0];
  uint64_t const *gmem_stride_ptr = &gmem_prob_stride[0];
  uint32_t const *smem_box_shape_ptr = &smem_box_shape[0];
  uint32_t const *smem_box_stride_ptr = &smem_box_stride[0];

  CUresult result = cuTensorMapEncodeTiled(tma_desc,
                                          tma_format,
                                          tma_dim,
                                          global_addr,
                                          gmem_shape_ptr,
                                          gmem_stride_ptr + 1,
                                          smem_box_shape_ptr,
                                          smem_box_stride_ptr,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          tma_swizzle,
                                          CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  char const *error_string;
  CUresult res = cuGetErrorString(result, &error_string);
  if (result != CUDA_SUCCESS) {
    std::cerr << "TMA Desc Addr:   " << &tma_desc << "\nformat         "
              << tma_format << "\ndim            " << tma_dim
              << "\ngmem_address   " << global_addr << "\nglobalDim      "
              << gmem_prob_shape << "\nglobalStrides  " << gmem_prob_stride
              << "\nboxDim         " << smem_box_shape << "\nelementStrides "
              << smem_box_stride << "\ninterleave     " << tma_interleave
              << "\nswizzle        " << tma_swizzle << "\nl2Promotion    "
              << tma_l2Promotion << "\noobFill        " << tma_oobFill
              << std::endl;
    std::cerr << "Error in tile TMA descriptor creation: " << error_string
              << std::endl;
    assert(false);
  }
}

template <typename TaskDesc, typename TensorDesc>
__host__ inline void
  fill_tma_desc_by_task(CUtensorMap *tma_desc, TaskDesc const &task_desc,
                                TensorDesc const &tensor_desc, uint16_t const param_id) {
  switch (task_desc.task_type) {
    case mirage::runtime::TASK_LINEAR_HOPPER:
    case mirage::runtime::TASK_LINEAR_WITH_RESIDUAL_HOPPER:
    {
      const int cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;

      if (param_id == 0) {
        // TMA_INPUT
        const int batch_size = tensor_desc.dim[0];
        const int reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size), static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(cp_async_size)};
        constexpr int TILE_SIZE = 128;

        size_t smem_repeat_col = (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc, tensor_desc.base_ptr, gmem_shape, gmem_stride, smem_shape, smem_repeat_row, smem_repeat_col);
      } else if (param_id == 1) {
        // TMA_WEIGHT
        const int output_size = tensor_desc.dim[0];
        const int output_atom_size =
    (output_size >= 256) ? 256 :
    (output_size >= 128) ? 128 :
    (output_size >=  64) ?  64 :
    (output_size >=  32) ?  32 : 16;
        const int reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(output_size), static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(output_atom_size), static_cast<uint32_t>(cp_async_size)};
        constexpr int TILE_SIZE = 128;
        size_t smem_repeat_col = (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc, tensor_desc.base_ptr, gmem_shape, gmem_stride, smem_shape, smem_repeat_row, smem_repeat_col);
      } else if (param_id == 2 && task_desc.task_type == mirage::runtime::TASK_LINEAR_WITH_RESIDUAL_HOPPER) {
        // TMA_RESIDUAL
        const int batch_size = tensor_desc.dim[0];
        const int output_size = tensor_desc.dim[1];
        const int output_stride = (tensor_desc.stride[0]);
        // printf("output_size: %d, output stride: %d\n", output_size, stride);
        const int output_atom_size = (output_size >= 256) ? 256 :
        (output_size >= 128) ? 128 :
        (output_size >=  64) ?  64 :
        (output_size >=  32) ?  32 : 16;
        // printf("output_atom_size: %d\n", output_atom_size);
        const int output_tma_cp_size = output_atom_size < 64 ? output_atom_size : 64;
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size), static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(output_tma_cp_size)};
        size_t smem_repeat_col = (output_atom_size + output_tma_cp_size - 1) / output_tma_cp_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc, tensor_desc.base_ptr, gmem_shape, gmem_stride, smem_shape, smem_repeat_row, smem_repeat_col);
      } else if (param_id == 3 && task_desc.task_type == mirage::runtime::TASK_LINEAR_WITH_RESIDUAL_HOPPER || param_id == 2 && task_desc.task_type == mirage::runtime::TASK_LINEAR_HOPPER) {
        // TMA_OUT
        const int batch_size = tensor_desc.dim[0];
        const int output_size = tensor_desc.dim[1];
        const int output_stride = (tensor_desc.stride[0]);
        // printf("output_size: %d, output stride: %d\n", output_size, stride);
        const int output_atom_size = (output_size >= 256) ? 256 :
        (output_size >= 128) ? 128 :
        (output_size >=  64) ?  64 :
        (output_size >=  32) ?  32 : 16;
        const int output_tma_cp_size = output_atom_size < 64 ? output_atom_size : 64;
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size), static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(output_tma_cp_size)};
        size_t smem_repeat_col = (output_atom_size + output_tma_cp_size - 1) / output_tma_cp_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc, tensor_desc.base_ptr, gmem_shape, gmem_stride, smem_shape, smem_repeat_row, smem_repeat_col);
      }
      break;
    }
    default:
      assert(false);
  }
}
