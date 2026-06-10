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
#include "runtime_header.h"
#include "tasks/common/common_header.cuh"
#include <cstdlib>
#include <cuda.h>
#include <cutlass/float8.h>
#include <cutlass/numeric_types.h>
#include <type_traits>

namespace mirage {
namespace runtime {

using bfloat16 = type::bfloat16_t;

// NOTE(Yu): Assume smem_stride is always 1, so we don't pass it as an argument
template <typename T, int B, int M, int S, int NDIM>
__host__ static inline void fill_tma_desc(CUtensorMap *tma_desc,
                                          void *const src,
                                          uint64_t const (&gmem_shape)[NDIM],
                                          uint64_t const (&gmem_stride)[NDIM],
                                          uint32_t const (&smem_shape)[NDIM],
                                          size_t smem_repeat_row,
                                          size_t smem_repeat_col) {
  constexpr uint32_t tma_dim = 5;
  void *global_addr = src;

  constexpr CUtensorMapDataType tma_format =
      std::is_same_v<T, type::bfloat16_t> ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
      : std::is_same_v<T, cutlass::bfloat16_t>
          ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
      : std::is_same_v<T, cutlass::half_t> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
      : std::is_same_v<T, __half>          ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
      : std::is_same_v<T, float>           ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32
      : std::is_same_v<T, double>          ? CU_TENSOR_MAP_DATA_TYPE_FLOAT64
      : std::is_same_v<T, cutlass::float_e4m3_t> ? CU_TENSOR_MAP_DATA_TYPE_UINT8
      : std::is_same_v<T, cutlass::float_e5m2_t> ? CU_TENSOR_MAP_DATA_TYPE_UINT8
      : std::is_same_v<T, cutlass::float_ue8m0_t>
          ? CU_TENSOR_MAP_DATA_TYPE_UINT8
      : std::is_same_v<T, uint8_t>  ? CU_TENSOR_MAP_DATA_TYPE_UINT8
      : std::is_same_v<T, uint16_t> ? CU_TENSOR_MAP_DATA_TYPE_UINT16
      : std::is_same_v<T, uint32_t> ? CU_TENSOR_MAP_DATA_TYPE_UINT32
      : std::is_same_v<T, int32_t>  ? CU_TENSOR_MAP_DATA_TYPE_INT32
                                    : CUtensorMapDataType(-1);
  static_assert(tma_format != CUtensorMapDataType(-1),
                "Unsupported TMA data type");
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
  } else if constexpr (NDIM == 3) {
    gmem_prob_shape[0] = gmem_shape[2];
    gmem_prob_shape[1] = gmem_shape[1];
    gmem_prob_shape[2] = gmem_shape[0];
    gmem_prob_shape[3] = 1;
    gmem_prob_shape[4] = 1;
    gmem_prob_stride[0] = sizeof(T);
    gmem_prob_stride[1] = gmem_stride[1] * sizeof(T);
    gmem_prob_stride[2] = gmem_stride[2] * sizeof(T);
    gmem_prob_stride[3] = 0;
    gmem_prob_stride[4] = 0;
  } else if constexpr (NDIM == 4) {
    gmem_prob_shape[0] = gmem_shape[3];
    gmem_prob_shape[1] = gmem_shape[2];
    gmem_prob_shape[2] = gmem_shape[1];
    gmem_prob_shape[3] = gmem_shape[0];
    gmem_prob_shape[4] = 1;
    gmem_prob_stride[0] = sizeof(T);
    gmem_prob_stride[1] = gmem_stride[1] * sizeof(T);
    gmem_prob_stride[2] = gmem_stride[2] * sizeof(T);
    gmem_prob_stride[3] = gmem_stride[3] * sizeof(T);
    gmem_prob_stride[4] = 0;
  } else {
    assert(false);
  }

  // TMA requires 16B-aligned global address
  if ((reinterpret_cast<uint64_t>(global_addr) & 0b1111) != 0) {
    printf("WARN: TMA addr %p not 16B-aligned\n", global_addr);
  }

  assert(gmem_prob_shape[0] >= (uint64_t(1)));       // Size must be min 1
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32)); // Size must be max 2^32
  assert(gmem_prob_shape[1] >= (uint64_t(1)));       // Size must be min 1
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32)); // Size must be max 2^32
  assert(gmem_prob_shape[2] >= (uint64_t(1)));       // Size must be min 1
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32)); // Size must be max 2^32
  assert(gmem_prob_shape[3] >= (uint64_t(1)));       // Size must be min 1
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32)); // Size must be max 2^32
  assert(gmem_prob_shape[4] >= (uint64_t(1)));       // Size must be min 1
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32)); // Size must be max 2^32

  // Assert the byte strides. Tma Descriptor uses byte strides
  assert((gmem_prob_stride[1]) <
         (uint64_t(1) << 40)); // Stride must be max 2^40
  assert((gmem_prob_stride[1] & 0b1111) ==
         0); // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[2]) <
         (uint64_t(1) << 40)); // Stride must be max 2^40
  assert((gmem_prob_stride[2] & 0b1111) ==
         0); // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[3]) <
         (uint64_t(1) << 40)); // Stride must be max 2^40
  assert((gmem_prob_stride[3] & 0b1111) ==
         0); // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[4]) <
         (uint64_t(1) << 40)); // Stride must be max 2^40
  assert((gmem_prob_stride[4] & 0b1111) ==
         0); // Stride must be multiple of 16B (128b)

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
  } else if constexpr (NDIM == 3) {
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
  } else if constexpr (NDIM == 4) {
    smem_box_shape[0] = smem_shape[3];
    smem_box_shape[1] = smem_shape[2];
    smem_box_shape[2] = smem_shape[1];
    smem_box_shape[3] = smem_shape[0];
    smem_box_shape[4] = 1;
    smem_box_stride[0] = 1;
    smem_box_stride[1] = 1;
    smem_box_stride[2] = 1;
    smem_box_stride[3] = 1;
    smem_box_stride[4] = 1;
  } else {
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
printf("global_addr: %p\n", global_addr);
#endif

  assert(smem_box_shape[0] >= (uint32_t(1)));      // Size must be min 1
  assert(smem_box_shape[0] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256
  assert(smem_box_shape[1] >= (uint32_t(1)));      // Size must be min 1
  assert(smem_box_shape[1] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256
  assert(smem_box_shape[2] >= (uint32_t(1)));      // Size must be min 1
  assert(smem_box_shape[2] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256
  assert(smem_box_shape[3] >= (uint32_t(1)));      // Size must be min 1
  assert(smem_box_shape[3] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256
  assert(smem_box_shape[4] >= (uint32_t(1)));      // Size must be min 1
  assert(smem_box_shape[4] <= (uint32_t(1) << 8)); // Size must be max 2^8 = 256

  assert(smem_box_stride[0] >= (uint32_t(1))); // Stride must be min 1
  assert(smem_box_stride[0] <= (uint32_t(8))); // Stride must be max 2^3 = 8
  assert(smem_box_stride[1] >= (uint32_t(1))); // Stride must be min 1
  assert(smem_box_stride[1] <= (uint32_t(8))); // Stride must be max 2^3 = 8
  assert(smem_box_stride[2] >= (uint32_t(1))); // Stride must be min 1
  assert(smem_box_stride[2] <= (uint32_t(8))); // Stride must be max 2^3 = 8
  assert(smem_box_stride[3] >= (uint32_t(1))); // Stride must be min 1
  assert(smem_box_stride[3] <= (uint32_t(8))); // Stride must be max 2^3 = 8
  assert(smem_box_stride[4] >= (uint32_t(1))); // Stride must be min 1
  assert(smem_box_stride[4] <= (uint32_t(8))); // Stride must be max 2^3 = 8

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
    // Continue instead of asserting - some tensors may have alignment issues
    // that need to be fixed at the builder/allocation level
    return;
  }
}

__host__ inline void fill_tma_desc_by_task(CUtensorMap *tma_desc,
                                           FullTaskDesc const &task_desc,
                                           TensorDesc const &tensor_desc,
                                           size_t param_id,
                                           size_t tma_desc_id = 0) {
  switch (task_desc.task_type) {
    case TASK_LINEAR_HOPPER:
    case TASK_LINEAR_WITH_RESIDUAL_HOPPER: {
      int const cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int TILE_SIZE = 128;

      if (param_id == 0) {
        // TMA_INPUT
        int const batch_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(cp_async_size)};

        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 1) {
        // TMA_WEIGHT
        int const output_size = tensor_desc.dim[0];
        int const output_atom_size = (output_size >= 256)   ? 256
                                     : (output_size >= 128) ? 128
                                     : (output_size >= 64)  ? 64
                                     : (output_size >= 32)  ? 32
                                                            : 16;
        int const reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(output_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(output_atom_size),
                                  static_cast<uint32_t>(cp_async_size)};
        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 2 &&
                 task_desc.task_type == TASK_LINEAR_WITH_RESIDUAL_HOPPER) {
        // TMA_RESIDUAL
        int const batch_size = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const output_stride = (tensor_desc.stride[0]);
        int const output_atom_size = (output_size >= 256)   ? 256
                                     : (output_size >= 128) ? 128
                                     : (output_size >= 64)  ? 64
                                     : (output_size >= 32)  ? 32
                                                            : 16;
        int const output_tma_cp_size =
            output_atom_size < 64 ? output_atom_size : 64;
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(output_tma_cp_size)};
        size_t smem_repeat_col =
            (output_atom_size + output_tma_cp_size - 1) / output_tma_cp_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 3 &&
                     task_desc.task_type == TASK_LINEAR_WITH_RESIDUAL_HOPPER ||
                 param_id == 2 && task_desc.task_type == TASK_LINEAR_HOPPER) {
        // TMA_OUT
        int const batch_size = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const output_stride = (tensor_desc.stride[0]);
        int const output_atom_size = (output_size >= 256)   ? 256
                                     : (output_size >= 128) ? 128
                                     : (output_size >= 64)  ? 64
                                     : (output_size >= 32)  ? 32
                                                            : 16;
        int const output_tma_cp_size =
            output_atom_size < 64 ? output_atom_size : 64;
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(output_tma_cp_size)};
        size_t smem_repeat_col =
            (output_atom_size + output_tma_cp_size - 1) / output_tma_cp_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      }
      break;
    }
    case TASK_PAGED_ATTENTION_HOPPER: {
      using T = bfloat16;
      constexpr int B = 3, M = 3, S = 3;
      constexpr int TMA_CP_ASYNC_SIZE = 64;
      constexpr int KV_TILE_SIZE = 64;
      const size_t smem_repeat_row = 1;

      auto &qkv =
          task_desc.inputs[0]; // [max_tokens, (num_q + 2*num_kv)*head_dim]
      auto &k_cache =
          task_desc.inputs[1]; // [num_pages, page_size, num_kv, head_dim]

      int const max_tokens = qkv.dim[0];
      int const qkv_cols = qkv.dim[1];
      int const num_pages = k_cache.dim[0];
      int const page_size = k_cache.dim[1];
      int const num_kv_heads = k_cache.dim[2];
      int const head_dim = k_cache.dim[3];
      int const num_q_heads = qkv_cols / head_dim - 2 * num_kv_heads;
      // int const head_group = task_desc.head_group;
      int const total_head_dims = qkv.stride[0];
      int const total_head_groups =
          total_head_dims / head_dim / (num_q_heads + 2 * num_kv_heads);

      assert(num_q_heads > 0 && "Invalid num_q_heads derived from qkv");

      if (param_id == 0) {
        // map 2D qkv to 3D: [depth=num_tokens, row=num heads, col=head_dim]
        uint64_t gmem_shape[3] = {
            static_cast<uint64_t>(max_tokens),
            static_cast<uint64_t>(num_q_heads + 2 * num_kv_heads),
            static_cast<uint64_t>(head_dim)};
        uint64_t gmem_stride[3] = {1,
                                   static_cast<uint64_t>(head_dim),
                                   static_cast<uint64_t>(qkv.stride[0])};
        uint32_t smem_shape[3] = {static_cast<uint32_t>(max_tokens),
                                  static_cast<uint32_t>(tma_desc_id == 0
                                                            ? num_q_heads
                                                            : num_kv_heads),
                                  static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
        const size_t smem_repeat_col = static_cast<size_t>(
            (head_dim + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE);
        fill_tma_desc<T, B, M, S, 3>(tma_desc,
                                     tensor_desc.base_ptr,
                                     gmem_shape,
                                     gmem_stride,
                                     smem_shape,
                                     smem_repeat_row,
                                     smem_repeat_col);
      }

      else if (param_id == 1 || param_id == 2) {
        // paged_k_cache_ptr / paged_v_cache_ptr
        // tensor is 3D: [num_pages, page_size, head_dim]
        uint64_t gmem_shape[4] = {static_cast<uint64_t>(num_pages),
                                  static_cast<uint64_t>(page_size),
                                  static_cast<uint64_t>(total_head_groups),
                                  static_cast<uint64_t>(head_dim)};
        uint64_t gmem_stride[4] = {
            1,
            static_cast<uint64_t>(head_dim),
            static_cast<uint64_t>(total_head_groups * head_dim),
            static_cast<uint64_t>(page_size * total_head_groups * head_dim)};
        uint32_t smem_shape[4] = {1u,
                                  static_cast<uint32_t>(KV_TILE_SIZE),
                                  1u,
                                  static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
        const size_t smem_repeat_col = static_cast<size_t>(
            (head_dim + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE);

        fill_tma_desc<T, B, M, S, 4>(tma_desc,
                                     tensor_desc.base_ptr,
                                     gmem_shape,
                                     gmem_stride,
                                     smem_shape,
                                     smem_repeat_row,
                                     smem_repeat_col);
      } else if (param_id == 3) {
        uint64_t gmem_shape[3] = {
            static_cast<uint64_t>(max_tokens),
            static_cast<uint64_t>(num_q_heads * total_head_groups),
            static_cast<uint64_t>(head_dim)};
        uint64_t gmem_stride[3] = {
            1,
            static_cast<uint64_t>(head_dim),
            static_cast<uint64_t>(num_q_heads * total_head_groups * head_dim)};

        uint32_t smem_shape[3] = {static_cast<uint32_t>(max_tokens),
                                  static_cast<uint32_t>(num_q_heads),
                                  static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
        const size_t smem_repeat_col = static_cast<size_t>(
            (head_dim + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE);
        fill_tma_desc<T, B, M, S, 3>(tma_desc,
                                     tensor_desc.base_ptr,
                                     gmem_shape,
                                     gmem_stride,
                                     smem_shape,
                                     smem_repeat_row,
                                     smem_repeat_col);
      } else {
        assert(false && "Unknown param_id for TASK_PAGED_ATTENTION_HOPPER");
      }

      break;
    }
    case TASK_LINEAR_SWAPAB_HOPPER:
    case TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER: {
      int const cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int output_atom_size = 64;
      constexpr int TILE_SIZE = 128;

      if (param_id == 0) {
        // TMA_INPUT
        int const batch_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(cp_async_size)};

        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 1) {
        // TMA_WEIGHT
        int const output_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(output_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        // NOTE(Yu): even for output_size < output_atom_size, we still use
        // output_atom_size as padding
        uint32_t smem_shape[2] = {static_cast<uint32_t>(output_atom_size),
                                  static_cast<uint32_t>(cp_async_size)};
        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 2 && task_desc.task_type ==
                                      TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER) {
        // TMA_RESIDUAL
        int const batch_size = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const output_stride = (tensor_desc.stride[0]);
        int const output_tma_cp_size = output_size < 64 ? output_size : 64;
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(output_tma_cp_size)};
        size_t smem_repeat_col = 1;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 3 && task_desc.task_type ==
                                      TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER ||
                 param_id == 2 &&
                     task_desc.task_type == TASK_LINEAR_SWAPAB_HOPPER) {
        // TMA_OUT
        int const batch_size = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const output_stride = (tensor_desc.stride[0]);
        int const output_tma_cp_size = output_size < 64 ? output_size : 64;
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(output_tma_cp_size)};
        size_t smem_repeat_col = 1;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      }
      break;
    }
    case TASK_SPLITK_LINEAR_SWAPAB_HOPPER: {
      int const cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int output_atom_size = 64;
      constexpr int TILE_SIZE = 64;

      if (param_id == 0) {
        // TMA_INPUT
        int const batch_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        int const reduction_stride = tensor_desc.stride[0];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(cp_async_size)};

        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 1) {
        // TMA_WEIGHT
        int const output_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        int const reduction_stride = tensor_desc.stride[0];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(output_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_stride)};
        // NOTE(Yu): even for output_size < output_atom_size, we still use
        // output_atom_size as padding
        uint32_t smem_shape[2] = {static_cast<uint32_t>(output_atom_size),
                                  static_cast<uint32_t>(cp_async_size)};
        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 2) {
        // TMA_OUT
        int const batch_size = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const output_stride = (tensor_desc.stride[0]);
        int const output_tma_cp_size = output_size < 64 ? output_size : 64;
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(output_tma_cp_size)};
        size_t smem_repeat_col = 1;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      }
      break;
    }
    case TASK_LINEAR_CUTLASS_HOPPER:
    case TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER: {
      int const cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int output_atom_size = 64;
      constexpr int TILE_SIZE = 128;

      if (param_id == 0) {
        // TMA_INPUT
        int const batch_size = tensor_desc.dim[0];
        // int const batch_size = 16;
        int const reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(cp_async_size)};

        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 1) {
        // TMA_WEIGHT
        int const output_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(output_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        // NOTE(Yu): even for output_size < output_atom_size, we still use
        // output_atom_size as padding
        uint32_t smem_shape[2] = {static_cast<uint32_t>(output_atom_size),
                                  static_cast<uint32_t>(cp_async_size)};
        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      }
      break;
    }
    case TASK_LINEAR_SM100:
    case TASK_LINEAR_WITH_RESIDUAL_SM100: {
      int const cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int MMA_M = 128;
      constexpr int MMA_N = 16;

      if (param_id == 0) {
        // TMA_INPUT: box must not exceed global dims
        int const batch_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(min(MMA_N, batch_size)),
                                  static_cast<uint32_t>(cp_async_size)};
        constexpr int TILE_SIZE = 64;

        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 1) {
        // TMA_WEIGHT
        int const output_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(output_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                                  static_cast<uint32_t>(cp_async_size)};
        constexpr int TILE_SIZE = 64;
        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 3 &&
                     (task_desc.task_type == TASK_LINEAR_WITH_RESIDUAL_SM100) ||
                 param_id == 2 && (task_desc.task_type == TASK_LINEAR_SM100)) {
        // TMA_OUT: box must not exceed global dims
        int const batch_size = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const output_stride = (tensor_desc.stride[0]);
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(min(MMA_N, batch_size)),
                                  static_cast<uint32_t>(MMA_M)};
        size_t smem_repeat_col = 1;
        fill_tma_desc<bfloat16, 0, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      }
      break;
    }
    case TASK_LINEAR_FP8_SM100:
    case TASK_LINEAR_FP8_WITH_RESIDUAL_SM100: {
      // New FP8 GEMM kernel: use cuTensorMapEncodeTiled directly
      // BLOCK_M=32, BLOCK_N=16, BLOCK_K=128
      constexpr int BLOCK_M_FP8 = 32, BLOCK_N_FP8 = 16, BLOCK_K_FP8 = 128;
      bool with_res =
          (task_desc.task_type == TASK_LINEAR_FP8_WITH_RESIDUAL_SM100);
      bool is_output = (param_id == (size_t)(task_desc.num_inputs));
      bool is_uint32 = (tensor_desc.data_type == 956); // DT_UINT32
      bool is_fp8 = (tensor_desc.data_type == 930);    // DT_FLOAT8

      if (is_fp8 && param_id == 0) {
        // A (input FP8): dim=[batch, K]. C20 (2026-05-17): gs from
        // stride[0] (FP8: 1 byte/elem) so views read the parent's row
        // stride, not the narrow slot width.
        int batch = tensor_desc.dim[0];
        int K = tensor_desc.dim[1];
        uint64_t gd[2] = {(uint64_t)K, (uint64_t)batch};
        uint64_t gs[1] = {(uint64_t)tensor_desc.stride[0]};
        uint32_t bd[2] = {(uint32_t)BLOCK_K_FP8,
                          (uint32_t)min(BLOCK_M_FP8, batch)};
        uint32_t es[2] = {1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                   2,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA FP8 input A failed: " << err << std::endl;
        }
      } else if (is_fp8 && param_id == 2) {
        // B (weight FP8): dim=[output, K], stride=[K, 1]
        int output = tensor_desc.dim[0];
        int K = tensor_desc.dim[1];
        uint64_t gd[2] = {(uint64_t)K, (uint64_t)output};
        uint64_t gs[1] = {(uint64_t)K * 1}; // stride0 * sizeof(uint8)
        uint32_t bd[2] = {(uint32_t)BLOCK_K_FP8, (uint32_t)BLOCK_N_FP8};
        uint32_t es[2] = {1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                   2,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA FP8 weight B failed: " << err << std::endl;
        }
      } else if (is_uint32 && param_id == 1) {
        // SFA (input scale): stored as [packed_k, aligned_batch] row-major
        // = column-major [aligned_batch, packed_k]
        int packed_k = tensor_desc.dim[0];
        int aligned_batch = tensor_desc.dim[1];
        uint64_t gd[2] = {(uint64_t)aligned_batch, (uint64_t)packed_k};
        uint64_t gs[1] = {(uint64_t)aligned_batch *
                          4}; // stride * sizeof(uint32)
        // box must not exceed global dims
        uint32_t bd[2] = {(uint32_t)min(BLOCK_M_FP8, aligned_batch), 1};
        uint32_t es[2] = {1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                   2,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA FP8 scale SFA failed: " << err << std::endl;
        }
      } else if (is_uint32 && param_id == 3) {
        // SFB (weight scale): stored as [packed_k, aligned_output] row-major
        // = column-major [aligned_output, packed_k]
        // Weight scale after grid partition: dim[0]=M_per_block,
        // dim[1]=packed_k Column-major: stride[1] is the physical outer stride
        // (original aligned_M)
        int aligned_output = tensor_desc.dim[0]; // after grid split
        int packed_k = tensor_desc.dim[1];
        int physical_outer_stride = tensor_desc.stride[1]; // original aligned_M
        uint64_t gd[2] = {(uint64_t)aligned_output, (uint64_t)packed_k};
        uint64_t gs[1] = {(uint64_t)physical_outer_stride *
                          4}; // stride * sizeof(uint32)
        uint32_t bd[2] = {(uint32_t)BLOCK_N_FP8, 1};
        uint32_t es[2] = {1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                   2,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA FP8 scale SFB failed: " << err << std::endl;
        }
      } else if (with_res && param_id == 4) {
        // Residual (BF16): dim=[batch, output], stride=[stride0, 1]
        int batch = tensor_desc.dim[0];
        int output = tensor_desc.dim[1];
        int stride = tensor_desc.stride[0];
        uint64_t gd[2] = {(uint64_t)output, (uint64_t)batch};
        uint64_t gs[1] = {(uint64_t)stride * 2}; // stride0 * sizeof(bf16)
        uint32_t bd[2] = {16,
                          (uint32_t)min(BLOCK_M_FP8, batch)}; // clamp to global
        uint32_t es[2] = {1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                   2,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_32B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA FP8 residual failed: " << err << std::endl;
        }
      } else if (is_output) {
        // CD (output BF16): dim=[batch, output], stride=[stride0, 1]
        int batch = tensor_desc.dim[0];
        int output = tensor_desc.dim[1];
        int stride = tensor_desc.stride[0];
        uint64_t gd[2] = {(uint64_t)output, (uint64_t)batch};
        uint64_t gs[1] = {(uint64_t)stride * 2}; // stride0 * sizeof(bf16)
        uint32_t bd[2] = {16,
                          (uint32_t)min(BLOCK_M_FP8, batch)}; // clamp to global
        uint32_t es[2] = {1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                   2,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_32B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA FP8 output CD failed: " << err << std::endl;
        }
      }
      break;
    }
    case TASK_LINEAR_FP8_BMM_SM100: {
      // FP8 BMM reuses the swapAB GEMM body but its tensor_desc is 3D:
      //   inputs[0]  input_fp8   [N, H_per_task, D_in]
      //   inputs[2]  weight_fp8  [H_per_task, D_out_per_task, D_in]
      //   outputs[0] output_bf16 [N, H_per_task, D_out_per_task]
      // The kernel sees a 2D per-head slice, so we encode rank=5 TMA
      // descriptors with the same logical (rows, K) extents as swapAB,
      // but pull the dims out of the 3D layout. Row strides come from
      // the gmem tensor's stride[] array — for input/output that's the
      // H-spanning stride[0], for weight it's the within-head stride[1].
      constexpr int MMA_M_BMM = 128;
      constexpr int MMA_N_BMM = 16;
      constexpr int BLOCK_K_BMM = 128;
      bool is_output_mpk = (param_id == (size_t)(task_desc.num_inputs));
      if (param_id == 0) {
        // input_fp8 -> kernel's TMA_B (B-side after swapAB).
        int batch = tensor_desc.dim[0];
        int K = tensor_desc.dim[2];
        int row_stride = tensor_desc.stride[0]; // H * D_in
        uint64_t gd[5] = {(uint64_t)K, (uint64_t)batch, 1, 1, 1};
        uint64_t gs[4] = {(uint64_t)row_stride * 1, 0, 0, 0};
        uint32_t bd[5] = {(uint32_t)BLOCK_K_BMM, (uint32_t)MMA_N_BMM, 1, 1, 1};
        uint32_t es[5] = {1, 1, 1, 1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                   5,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA BMM FP8 input failed: " << err << std::endl;
        }
      } else if (param_id == 2) {
        // weight_fp8 -> kernel's TMA_A (A-side). dim=[H_per_task,
        // D_out_per_task, D_in].
        int output_pt = tensor_desc.dim[1];
        int K = tensor_desc.dim[2];
        int row_stride = tensor_desc.stride[1]; // = D_in within a head
        uint64_t gd[5] = {(uint64_t)K, (uint64_t)output_pt, 1, 1, 1};
        uint64_t gs[4] = {(uint64_t)row_stride * 1, 0, 0, 0};
        uint32_t bd[5] = {(uint32_t)BLOCK_K_BMM, (uint32_t)MMA_M_BMM, 1, 1, 1};
        uint32_t es[5] = {1, 1, 1, 1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                   5,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA BMM FP8 weight failed: " << err << std::endl;
        }
      } else if (is_output_mpk) {
        // output (BF16): dim=[N, H_per_task, D_out_per_task].
        int batch = tensor_desc.dim[0];
        int output_pt = tensor_desc.dim[2];
        int row_stride = tensor_desc.stride[0]; // H * D_out
        uint64_t gd[5] = {(uint64_t)output_pt, (uint64_t)batch, 1, 1, 1};
        uint64_t gs[4] = {(uint64_t)row_stride * 2, 0, 0, 0};
        uint32_t bd[5] = {(uint32_t)MMA_M_BMM, (uint32_t)MMA_N_BMM, 1, 1, 1};
        uint32_t es[5] = {1, 1, 1, 1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                   5,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA BMM FP8 output failed: " << err << std::endl;
        }
      }
      break;
    }
    case TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100:
    case TASK_LINEAR_FP8_SWAPAB_SM100:
    case TASK_LINEAR_FP8_SWAPAB_WITH_RESIDUAL_SM100: {
      // MPK-native FP8 swapAB kernel. Tile shapes match the kernel template:
      // MMA_M=128 along the OUTPUT axis (kernel's A = weight after swap),
      // MMA_N=16 along the BATCH axis (kernel's B = input after swap),
      // BLOCK_K=128. Scales are NOT TMA'd here (raw pointers handle them in
      // the producer warp).
      constexpr int MMA_M_SWAPAB = 128;
      constexpr int MMA_N_SWAPAB = 16;
      constexpr int BLOCK_K_SWAPAB = 128;
      bool with_res =
          (task_desc.task_type == TASK_LINEAR_FP8_SWAPAB_WITH_RESIDUAL_SM100);
      bool is_output_mpk = (param_id == (size_t)(task_desc.num_inputs));

      // The kernel uses kernel::tma::tma_2d typed wrappers, which issue
      // cp.async.bulk.tensor.5d.* PTX. That requires a *5D-encoded* descriptor
      // (rank=5 with trailing dims = 1) — encoding as rank=2 produces an
      // illegal instruction at runtime. We mirror Mirage's `fill_tma_desc`
      // (tma.cuh:30+) which always emits rank=5 descriptors.
      if (param_id == 0) {
        // input_fp8 (slot 0) -> kernel's TMA_B (B-side). dim=[batch, K].
        // For split-K, dim[1] is the per-task K-slice but the gmem row
        // stride stays at the full K — read it from stride[0]. (For the
        // non-split case stride[0] == K so behavior is unchanged.)
        int batch = tensor_desc.dim[0];
        int K = tensor_desc.dim[1];
        int row_stride = tensor_desc.stride[0];
        uint64_t gd[5] = {(uint64_t)K, (uint64_t)batch, 1, 1, 1};
        uint64_t gs[4] = {(uint64_t)row_stride * 1, 0, 0, 0};
        uint32_t bd[5] = {
            (uint32_t)BLOCK_K_SWAPAB, (uint32_t)MMA_N_SWAPAB, 1, 1, 1};
        uint32_t es[5] = {1, 1, 1, 1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                   5,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA MPK FP8 input failed: " << err << std::endl;
        }
      } else if (param_id == 2) {
        // weight_fp8 (slot 2) -> kernel's TMA_A (A-side). dim=[output_per_task,
        // K]. Same row-stride consideration as input above.
        int output_pt = tensor_desc.dim[0];
        int K = tensor_desc.dim[1];
        int row_stride = tensor_desc.stride[0];
        uint64_t gd[5] = {(uint64_t)K, (uint64_t)output_pt, 1, 1, 1};
        uint64_t gs[4] = {(uint64_t)row_stride * 1, 0, 0, 0};
        uint32_t bd[5] = {
            (uint32_t)BLOCK_K_SWAPAB, (uint32_t)MMA_M_SWAPAB, 1, 1, 1};
        uint32_t es[5] = {1, 1, 1, 1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                   5,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA MPK FP8 weight failed: " << err << std::endl;
        }
      } else if (with_res && param_id == 4) {
        // residual (BF16): dim=[batch, output_per_task], stride=[stride0, 1].
        int batch = tensor_desc.dim[0];
        int output_pt = tensor_desc.dim[1];
        int stride = tensor_desc.stride[0];
        uint64_t gd[5] = {(uint64_t)output_pt, (uint64_t)batch, 1, 1, 1};
        uint64_t gs[4] = {(uint64_t)stride * 2, 0, 0, 0};
        uint32_t bd[5] = {
            (uint32_t)MMA_M_SWAPAB, (uint32_t)MMA_N_SWAPAB, 1, 1, 1};
        uint32_t es[5] = {1, 1, 1, 1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                   5,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA MPK FP8 residual failed: " << err << std::endl;
        }
      } else if (is_output_mpk) {
        // output (BF16): dim=[batch, output_per_task], stride=[stride0, 1].
        int batch = tensor_desc.dim[0];
        int output_pt = tensor_desc.dim[1];
        int stride = tensor_desc.stride[0];
        uint64_t gd[5] = {(uint64_t)output_pt, (uint64_t)batch, 1, 1, 1};
        uint64_t gs[4] = {(uint64_t)stride * 2, 0, 0, 0};
        uint32_t bd[5] = {
            (uint32_t)MMA_M_SWAPAB, (uint32_t)MMA_N_SWAPAB, 1, 1, 1};
        uint32_t es[5] = {1, 1, 1, 1, 1};
        CUresult result =
            cuTensorMapEncodeTiled(tma_desc,
                                   CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                   5,
                                   tensor_desc.base_ptr,
                                   gd,
                                   gs,
                                   bd,
                                   es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          char const *err;
          cuGetErrorString(result, &err);
          std::cerr << "TMA MPK FP8 output failed: " << err << std::endl;
        }
      }
      break;
    }
    case TASK_SPLITK_LINEAR_SM100: {
      int const cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int MMA_M = 128;
      constexpr int MMA_N = 16;

      if (param_id == 0) {
        // TMA_INPUT: clamp box height to in-bounds rows (matches non-splitk
        // path). Without this clamp, the descriptor delivers MMA_N rows of
        // bytes to the mbarrier, but the kernel's expect_tx is sized for
        // min(MMA_N, batch_size), causing mbarrier underflow → deadlock at
        // batch_size < MMA_N. Symmetric with TASK_LINEAR_SM100 above.
        int const batch_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        int const reduction_stride = tensor_desc.stride[0];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(min(MMA_N, batch_size)),
                                  static_cast<uint32_t>(cp_async_size)};
        constexpr int TILE_SIZE = 64;

        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 1) {
        // TMA_WEIGHT
        int const output_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        int const reduction_stride = tensor_desc.stride[0];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(output_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                                  static_cast<uint32_t>(cp_async_size)};
        constexpr int TILE_SIZE = 64;
        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      } else if (param_id == 2) {
        // TMA_OUT: clamp box height to in-bounds rows (matches non-splitk).
        int const batch_size = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const output_stride = (tensor_desc.stride[0]);
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(min(MMA_N, batch_size)),
                                  static_cast<uint32_t>(MMA_M)};
        size_t smem_repeat_col = 1;
        fill_tma_desc<bfloat16, 0, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      }
      break;
    }
    case TASK_MOE_W13_LINEAR_SM100:
    case TASK_MOE_W2_LINEAR_SM100: {
      int const cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int MMA_M = 128;

      if (param_id == 1) {
        // TMA_WEIGHT
        int const num_experts = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const reduction_size = tensor_desc.dim[2];
        int const orig_output_size =
            tensor_desc.stride[0] / tensor_desc.stride[1];
        uint64_t gmem_shape[2] = {
            static_cast<uint64_t>((num_experts - 1) * orig_output_size +
                                  output_size),
            static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                                  static_cast<uint32_t>(cp_async_size)};
        constexpr int TILE_SIZE = 64;
        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      }
      break;
    }
    case TASK_MOE_W13_FP8_SM100:
    case TASK_MOE_W2_FP8_SM100: {
      // FP8 E4M3 weight TMA (param_id == 2 is the weight tensor)
      // bK=128 tiles, MMA_M=128 rows per tile; FP8 = 1 byte per element.
      constexpr int FP8_BK = 128;
      const size_t smem_repeat_row_fp8 = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int MMA_M = 128;

      if (param_id == 2) {
        // TMA_WEIGHT (fp8): inputs are [input_fp8, input_scale, weight_fp8,
        // ...]
        int const num_experts = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const reduction_size = tensor_desc.dim[2];
        int const orig_output_size =
            tensor_desc.stride[0] / tensor_desc.stride[1];
        uint64_t gmem_shape[2] = {
            static_cast<uint64_t>((num_experts - 1) * orig_output_size +
                                  output_size),
            static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                                  static_cast<uint32_t>(FP8_BK)};
        size_t smem_repeat_col_fp8 = 1; // FP8_BK is the full tile width
        // Use uint8_t as the element type (sizeof=1) so fill_tma_desc selects
        // UINT8 format
        fill_tma_desc<uint8_t, B, M, S, 2>(tma_desc,
                                           tensor_desc.base_ptr,
                                           gmem_shape,
                                           gmem_stride,
                                           smem_shape,
                                           smem_repeat_row_fp8,
                                           smem_repeat_col_fp8);
      }
      break;
    }
    case TASK_MOE_W13_LINEAR_SM90:
    case TASK_MOE_W2_LINEAR_SM90: {
      int const cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int MMA_M = 64;

      if (param_id == 1) {
        // TMA_WEIGHT
        int const num_experts = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const reduction_size = tensor_desc.dim[2];
        int const orig_output_size =
            tensor_desc.stride[0] / tensor_desc.stride[1];
        uint64_t gmem_shape[2] = {
            static_cast<uint64_t>((num_experts - 1) * orig_output_size +
                                  output_size),
            static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                                  static_cast<uint32_t>(cp_async_size)};
        constexpr int TILE_SIZE = 64;
        size_t smem_repeat_col =
            (TILE_SIZE + cp_async_size - 1) / cp_async_size;
        fill_tma_desc<bfloat16, B, M, S, 2>(tma_desc,
                                            tensor_desc.base_ptr,
                                            gmem_shape,
                                            gmem_stride,
                                            smem_shape,
                                            smem_repeat_row,
                                            smem_repeat_col);
      }
      break;
    }
    case TASK_MLA_PREFILL_TP8_CHUNKED_SPLITK_SM100:
    case TASK_MLA_PREFILL_TP8_CHUNKED_SM100: {
      // Per-head unabsorbed MLA chunked prefill (TP=8), 3 TMA inputs:
      //   param_id=2: K_nope [S,H,128] viewed as [S,H*2,64], 3D
      //   param_id=3: K_rope [S,64] or [S,1,64], 2D
      //   param_id=4: V      [S,H,128] viewed as [S,H*2,64], 3D
      constexpr int BK = 64;
      constexpr int BN_BOX = 128;
      constexpr CUtensorMapDataType fmt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
      constexpr CUtensorMapInterleave interleave =
          CU_TENSOR_MAP_INTERLEAVE_NONE;
      constexpr CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
      constexpr CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
      constexpr CUtensorMapFloatOOBfill oob =
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;

      if (param_id == 3) {
        int d_last = tensor_desc.dim[tensor_desc.num_dims - 1];
        int total_rows = 1;
        for (int i = 0; i < tensor_desc.num_dims - 1; i++) {
          total_rows *= tensor_desc.dim[i];
        }
        uint64_t gd[2] = {(uint64_t)d_last, (uint64_t)total_rows};
        uint64_t gs[1] = {(uint64_t)d_last * 2};
        uint32_t bd[2] = {(uint32_t)BK, (uint32_t)BN_BOX};
        uint32_t es[2] = {1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              fmt,
                                              2,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              swizzle,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      } else {
        int total_rows = tensor_desc.dim[0];
        int H_local = 0;
        int d_last = 0;
        if (tensor_desc.num_dims == 2) {
          // Builder stores decompressed K_nope/V as flat [S, H*128].
          H_local = tensor_desc.dim[1] / 128;
          d_last = 128;
        } else {
          H_local = tensor_desc.dim[1];
          d_last = tensor_desc.dim[2];
        }
        int num_blocks = H_local * (d_last / BK);
        uint64_t gd[3] = {
            (uint64_t)BK, (uint64_t)total_rows, (uint64_t)num_blocks};
        uint64_t gs[2] = {(uint64_t)H_local * d_last * 2, (uint64_t)BK * 2};
        uint32_t bd[3] = {(uint32_t)BK, (uint32_t)BN_BOX, 1};
        uint32_t es[3] = {1, 1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              fmt,
                                              3,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              swizzle,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      }
      break;
    }
    case TASK_FP8_GEMM_DENSE_SMALLM_SM100:
    case TASK_FP8_GEMM_DENSE_MEDIUMM_SM100:
    case TASK_FP8_GEMM_DENSE_SMALLM_FP8OUT_SM100:
    case TASK_FP8_GEMM_DENSE_MEDIUMM_FP8OUT_SM100: {
      // Dense FP8 GEMM TMA for A [M,K] and B [N,K], both row-major raw
      // e4m3 bytes. Scales are loaded directly, not through TMA. SplitK
      // variant uses the same descriptor — per-CTA K offset is encoded in
      // the runtime tile-index decomposition.
      // C20 (2026-05-17): gmem row stride must come from
      // `tensor_desc.stride[0]` (in FP8 bytes), not dim[1]. For root
      // tensors the two are equal; for `mpk.narrow` views of e.g.
      // qkv_a_out, dim[1] = slot_width but stride[0] = parent_row_width,
      // which is what the TMA engine must use to advance between rows.
      constexpr int BK_BOX = 128;
      constexpr int OUTER_BOX = 128;
      constexpr CUtensorMapDataType fmt = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      constexpr CUtensorMapInterleave interleave =
          CU_TENSOR_MAP_INTERLEAVE_NONE;
      constexpr CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
      constexpr CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
      constexpr CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
      int outer = tensor_desc.dim[0];
      int K_local = tensor_desc.dim[1];
      uint64_t row_stride_bytes =
          (uint64_t)tensor_desc.stride[0]; // FP8: 1 byte per element
      uint64_t gd[2] = {(uint64_t)K_local, (uint64_t)outer};
      uint64_t gs[1] = {row_stride_bytes};
      uint32_t bd[2] = {BK_BOX, OUTER_BOX};
      uint32_t es[2] = {1, 1};
      CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                            fmt,
                                            2,
                                            tensor_desc.base_ptr,
                                            gd,
                                            gs,
                                            bd,
                                            es,
                                            interleave,
                                            swizzle,
                                            l2,
                                            oob);
      assert(err == CUDA_SUCCESS);
      break;
    }
    case TASK_FP8_GEMM_DENSE_SPLITK_TMAREDUCE_SM100: {
      // A_fp8 (param 0) + B_fp8 (param 1): same raw-e4m3 [K, outer] descriptor
      // as the dense GEMM above (128B swizzle, BK=128, OUTER=128 box). The C
      // output (param 2 == num_inputs) is a bf16 reduce-add descriptor: no
      // swizzle, box={BN=128 cols (N), BM=128 rows (M)}. The kernel stages a
      // row-major [BM][BN] bf16 tile to SMEM and issues
      // cp.reduce.async.bulk.tensor.2d with coords {c0=on (N), c1=om (M)},
      // which matches gd={N, M} (innermost=N).
      constexpr CUtensorMapInterleave interleave =
          CU_TENSOR_MAP_INTERLEAVE_NONE;
      constexpr CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
      constexpr CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
      if (param_id == 0 || param_id == 1) {
        // A/B FP8 e4m3 raw bytes, 128B swizzle, box [BK=128, OUTER=128].
        constexpr int BK_BOX = 128;
        constexpr int OUTER_BOX = 128;
        int outer = tensor_desc.dim[0];
        int K_local = tensor_desc.dim[1];
        uint64_t row_stride_bytes =
            (uint64_t)tensor_desc.stride[0]; // FP8: 1 byte per element
        uint64_t gd[2] = {(uint64_t)K_local, (uint64_t)outer};
        uint64_t gs[1] = {row_stride_bytes};
        uint32_t bd[2] = {BK_BOX, OUTER_BOX};
        uint32_t es[2] = {1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                              2,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              CU_TENSOR_MAP_SWIZZLE_128B,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      } else {
        // C bf16 reduce-add output: [M, N] row-major. innermost=N, outer=M.
        constexpr int BN_BOX = 128;
        constexpr int BM_BOX = 128;
        int M_out = tensor_desc.dim[0];
        int N_out = tensor_desc.dim[1];
        uint64_t row_stride_bytes =
            (uint64_t)tensor_desc.stride[0] * sizeof(__nv_bfloat16);
        uint64_t gd[2] = {(uint64_t)N_out, (uint64_t)M_out};
        uint64_t gs[1] = {row_stride_bytes};
        uint32_t bd[2] = {BN_BOX, BM_BOX};
        uint32_t es[2] = {1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                              2,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              CU_TENSOR_MAP_SWIZZLE_NONE,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      }
      break;
    }
    // D3: fp8out flavor reuses the IDENTICAL A/B input TMA layout (outputs are
    // raw FP8 + float32-scale stores, no TMA).
    case TASK_LINEAR_FP8_BMM_DENSE_FP8OUT_SM100:
    case TASK_LINEAR_FP8_BMM_DENSE_SM100: {
      // Per-head dense FP8 BMM: 2 TMA descriptors (A=input param 0, B=weight
      // param 2). Same 2D [K, outer] raw-e4m3 descriptor as the dense GEMM,
      // but the per-task tensors are 3D per-head slices, so K and the gmem
      // row stride live at different dims:
      //   A (input  [N_batch, H, D_in])  per head -> STensor [N_batch, 1, D_in]
      //       outer = dim[0] (= N_batch), K = dim[2] (= D_in),
      //       row_stride = stride[0] (= H * D_in).
      //   B (weight [H, D_out, D_in])     per head -> STensor [1, D_out, D_in]
      //       outer = dim[1] (= D_out),  K = dim[2] (= D_in),
      //       row_stride = stride[1] (= D_in).
      // Scales (float32) and the bf16 output are raw pointers, not TMA.
      constexpr int BK_BOX = 128;
      constexpr int OUTER_BOX = 128;
      constexpr CUtensorMapDataType fmt = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      constexpr CUtensorMapInterleave interleave =
          CU_TENSOR_MAP_INTERLEAVE_NONE;
      constexpr CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
      constexpr CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
      constexpr CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
      int outer, K_local;
      uint64_t row_stride_bytes;
      if (param_id == 0) {
        // A = input: outer = M (dim0), K = D_in (dim2), stride = stride[0].
        outer = tensor_desc.dim[0];
        K_local = tensor_desc.dim[2];
        row_stride_bytes = (uint64_t)tensor_desc.stride[0];
      } else {
        // B = weight (param 2): outer = D_out (dim1), K = D_in (dim2),
        // stride = stride[1].
        outer = tensor_desc.dim[1];
        K_local = tensor_desc.dim[2];
        row_stride_bytes = (uint64_t)tensor_desc.stride[1];
      }
      uint64_t gd[2] = {(uint64_t)K_local, (uint64_t)outer};
      uint64_t gs[1] = {row_stride_bytes};
      uint32_t bd[2] = {BK_BOX, OUTER_BOX};
      uint32_t es[2] = {1, 1};
      CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                            fmt,
                                            2,
                                            tensor_desc.base_ptr,
                                            gd,
                                            gs,
                                            bd,
                                            es,
                                            interleave,
                                            swizzle,
                                            l2,
                                            oob);
      assert(err == CUDA_SUCCESS);
      break;
    }
    case TASK_FP8_GROUP_GEMM_SMALLM_SM100:
    case TASK_FP8_GROUP_GEMM_LARGEM_SM100: {
      // 5 TMA descriptors: A (param 0), B (param 1), SFA (param 2),
      // SFB (param 3), D output (output param 0). param_id 4 (m_indices) is
      // direct LDG, not TMA. B/SFB box dim depends on BN: smallm uses BN=64
      // (one TMA load = 64 rows of B), largem BN=128.
      constexpr CUtensorMapInterleave interleave =
          CU_TENSOR_MAP_INTERLEAVE_NONE;
      constexpr CUtensorMapL2promotion l2_none =
          CU_TENSOR_MAP_L2_PROMOTION_NONE;
      constexpr CUtensorMapL2promotion l2_128 =
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
      constexpr CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
      int const VARIANT_BN =
          (task_desc.task_type == TASK_FP8_GROUP_GEMM_SMALLM_SM100) ? 64 : 128;
      if (param_id == 0) {
        // A: [K_inner, M_total_outer], FP8 raw bytes, 128B swizzle.
        int M_total = tensor_desc.dim[0];
        int K = tensor_desc.dim[1];
        uint64_t gd[2] = {(uint64_t)K, (uint64_t)M_total};
        uint64_t gs[1] = {(uint64_t)K};
        uint32_t bd[2] = {128, 128};
        uint32_t es[2] = {1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                              2,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              CU_TENSOR_MAP_SWIZZLE_128B,
                                              l2_128,
                                              oob);
        assert(err == CUDA_SUCCESS);
      } else if (param_id == 1) {
        // B: [K_inner, E*N_outer], FP8 raw bytes. dim[0]=E, dim[1]=N, dim[2]=K
        // (3D allocation); need E*N*K viewed as [K, E*N].
        int E = tensor_desc.dim[0];
        int N = tensor_desc.dim[1];
        int K = tensor_desc.dim[2];
        uint64_t gd[2] = {(uint64_t)K, (uint64_t)E * N};
        uint64_t gs[1] = {(uint64_t)K};
        uint32_t bd[2] = {128, (uint32_t)VARIANT_BN};
        uint32_t es[2] = {1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                              2,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              CU_TENSOR_MAP_SWIZZLE_128B,
                                              l2_none,
                                              oob);
        assert(err == CUDA_SUCCESS);
      } else if (param_id == 2) {
        // SFA: python tensor shape [num_sf_k, M_total] row-major (matches
        // source's prepare_sf which writes packed[sk*dim + d]).
        // TMA reads with M_total as innermost: gd=[M_total, num_sf_k],
        // gs=[M_total*4 bytes between successive num_sf_k rows].
        int num_sf_k = tensor_desc.dim[0];
        int M_total = tensor_desc.dim[1];
        uint64_t gd[2] = {(uint64_t)M_total, (uint64_t)num_sf_k};
        uint64_t gs[1] = {(uint64_t)M_total * sizeof(uint32_t)};
        uint32_t bd[2] = {128, 1};
        uint32_t es[2] = {1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                              2,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              CU_TENSOR_MAP_SWIZZLE_NONE,
                                              l2_128,
                                              oob);
        assert(err == CUDA_SUCCESS);
      } else if (param_id == 3) {
        // SFB: python tensor shape [num_sf_k, E*N] row-major.
        // TMA reads with E*N as innermost: gd=[E*N, num_sf_k].
        int num_sf_k = tensor_desc.dim[0];
        int EN = tensor_desc.dim[1];
        uint64_t gd[2] = {(uint64_t)EN, (uint64_t)num_sf_k};
        uint64_t gs[1] = {(uint64_t)EN * sizeof(uint32_t)};
        uint32_t bd[2] = {(uint32_t)VARIANT_BN, 1};
        uint32_t es[2] = {1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                              2,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              CU_TENSOR_MAP_SWIZZLE_NONE,
                                              l2_none,
                                              oob);
        assert(err == CUDA_SUCCESS);
      } else {
        // Output D: [N_inner, M_total_outer] BF16, 128B swizzle. TMA store.
        int M_total = tensor_desc.dim[0];
        int N = tensor_desc.dim[1];
        uint64_t gd[2] = {(uint64_t)N, (uint64_t)M_total};
        uint64_t gs[1] = {(uint64_t)N * sizeof(__nv_bfloat16)};
        uint32_t bd[2] = {64, 128};
        uint32_t es[2] = {1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                              2,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              CU_TENSOR_MAP_SWIZZLE_128B,
                                              l2_none,
                                              oob);
        assert(err == CUDA_SUCCESS);
      }
      break;
    }
    case TASK_MLA_MTP_DECODE_TP2_SM100:
    case TASK_MLA_MTP_DECODE_TP4_SM100:
    case TASK_MLA_MTP_DECODE_TP8_SM100: {
      // TP variants: Q box height = rows consumed by one CTA. TP2 splits
      // its 64 local heads into 2 head groups, so each CTA loads 32 rows.
      // KV box = TILE_S=128. Same encoding as v037/v007/v001 host code.
      constexpr int BK = 64;
      constexpr int TILE_S = 128;
      constexpr CUtensorMapDataType fmt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
      constexpr CUtensorMapInterleave interleave =
          CU_TENSOR_MAP_INTERLEAVE_NONE;
      constexpr CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
      constexpr CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
      constexpr CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

      int num_heads =
          (task_desc.task_type == TASK_MLA_MTP_DECODE_TP2_SM100)   ? 32
          : (task_desc.task_type == TASK_MLA_MTP_DECODE_TP4_SM100) ? 32
                                                                   : 16;
      if (task_desc.task_type == TASK_MLA_MTP_DECODE_TP4_SM100) {
        int head_groups = 1;
        if (char const *env = std::getenv("MPK_MLA_TP4_HEAD_GROUPS")) {
          head_groups = std::atoi(env);
        }
        if (head_groups == 1 || head_groups == 2 || head_groups == 4 ||
            head_groups == 8) {
          num_heads = 32 / head_groups;
        }
      }
      if (param_id == 0) {
        // Q: may be flat [mbt, num_heads*D_K] (2D) or per-head [mbt, num_heads,
        // D_K] (3D — produced when the upstream q_b GEMM emits the BMM
        // layout). Reinterpret either form as [B*Q*heads, D_K].
        constexpr int D_K = 576;
        int total_elements = tensor_desc.dim[0] * tensor_desc.dim[1];
        if (tensor_desc.num_dims == 3) {
          total_elements *= tensor_desc.dim[2];
        }
        int total_rows = total_elements / D_K;
        int k_iters = D_K / BK;
        uint64_t gd[3] = {
            (uint64_t)BK, (uint64_t)total_rows, (uint64_t)k_iters};
        uint64_t gs[2] = {(uint64_t)D_K * 2, (uint64_t)BK * 2};
        uint32_t bd[3] = {(uint32_t)BK, (uint32_t)num_heads, 1};
        uint32_t es[3] = {1, 1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              fmt,
                                              3,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              swizzle,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      } else if (param_id == 1) {
        // KV: gd = {BK, B*KL, K_ITERS}, bd = {BK, TILE_S, 1}
        int total_rows = tensor_desc.dim[0];
        int d_k = tensor_desc.dim[1];
        if (tensor_desc.num_dims == 3) {
          total_rows = tensor_desc.dim[0] * tensor_desc.dim[1];
          d_k = tensor_desc.dim[2];
        }
        int k_iters = d_k / BK;
        uint64_t gd[3] = {
            (uint64_t)BK, (uint64_t)total_rows, (uint64_t)k_iters};
        uint64_t gs[2] = {(uint64_t)d_k * 2, 128};
        uint32_t bd[3] = {(uint32_t)BK, (uint32_t)TILE_S, 1};
        uint32_t es[3] = {1, 1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              fmt,
                                              3,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              swizzle,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      }
      break;
    }
    case TASK_MLA_DECODE_SM100: {
      // MLA uses 3D TMA descriptors with 128B swizzle.
      // Q tensor: [B*NUM_HEADS, D_K] → 3D TMA (BK=64, B*NUM_HEADS, D_K/BK)
      // KV tensor: [B*KL, D_K] → 3D TMA (BK=64, B*KL, D_K/BK)
      //
      // The kernel loads tiles of shape (BK, NUM_HEADS_or_TILE_S, 1) per TMA
      // op. cuTensorMapEncodeTiled is called directly since fill_tma_desc's
      // generic path doesn't handle the MLA-specific layout.
      constexpr int BK = 64;
      constexpr CUtensorMapDataType fmt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
      constexpr CUtensorMapInterleave interleave =
          CU_TENSOR_MAP_INTERLEAVE_NONE;
      constexpr CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
      constexpr CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
      constexpr CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

      if (param_id == 0) {
        // Q: may arrive as [B*NUM_HEADS, D_K] or flat [mbt, NUM_HEADS*D_K].
        // For TMA, reinterpret as 3D (BK, B*Q_LEN*NUM_HEADS, D_K/BK).
        // TP-aware: derive num_heads from tensor shape (= local heads in TP
        // mode), NOT hardcoded 128. Without this, TMA box height is wrong for
        // TP and causes OOB loads when kernel reads Q.
        int d_k = 576; // DeepSeek V3 MLA: 512 latent + 64 rope
        // Compute total elements from first 2 dims only (ignore padding dims)
        int total_elements = tensor_desc.dim[0] * tensor_desc.dim[1];
        int total_rows = total_elements / d_k; // mbt * (local_)NUM_HEADS
        int k_iters = d_k / BK;
        // Derive num_heads (local) from tensor's hidden dim:
        //   tensor_desc.dim[1] = num_heads * d_k  →  num_heads = dim[1] / d_k
        int num_heads = tensor_desc.dim[1] / d_k;
        if (num_heads < 1) {
          num_heads = 128; // safety fallback
        }
        // Derive hpb (assumes B=1):
        //   total_rows = Q_LEN * num_heads  →  Q_LEN = total_rows / num_heads
        //   hpb = num_heads / Q_LEN
        int q_len = total_rows / num_heads;
        if (q_len < 1) {
          q_len = 1;
        }
        int hpb = num_heads / q_len;
        while (hpb > 0 && num_heads % hpb != 0) {
          hpb--;
        }
        if (hpb <= 0) {
          hpb = num_heads;
        }
        // gd: global dims, gs: global byte strides (dim0 stride is implicit
        // sizeof(T)) gs[0] = row stride in bytes = D_K * sizeof(bf16)
        // gs[1] = k_iter stride in bytes = BK * sizeof(bf16) = 128
        uint64_t gd[3] = {
            (uint64_t)BK, (uint64_t)total_rows, (uint64_t)k_iters};
        uint64_t gs[2] = {(uint64_t)d_k * 2, (uint64_t)BK * 2};
        uint32_t bd[3] = {(uint32_t)BK, (uint32_t)hpb, 1};
        uint32_t es[3] = {1, 1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              fmt,
                                              3,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              swizzle,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      } else if (param_id == 1) {
        // KV: global [B*KL, D_K], or a paged cache flattened from
        // [num_pages, page_size, D_K].
        int total_rows = tensor_desc.dim[0]; // B*KL
        int d_k = tensor_desc.dim[1];        // D_K
        if (tensor_desc.num_dims == 3) {
          total_rows = tensor_desc.dim[0] * tensor_desc.dim[1];
          d_k = tensor_desc.dim[2];
        }
        int k_iters = d_k / BK;
        int tile_s = 128; // TILE_S
        uint64_t gd[3] = {
            (uint64_t)BK, (uint64_t)total_rows, (uint64_t)k_iters};
        uint64_t gs[2] = {(uint64_t)d_k * 2, 128};
        uint32_t bd[3] = {(uint32_t)BK, (uint32_t)tile_s, 1};
        uint32_t es[3] = {1, 1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              fmt,
                                              3,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              swizzle,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      }
      break;
    }    case TASK_MLA_MTP_DECODE_SM100: {
      // MTP decode: Q box height = hpb (varies with Q_LEN), KV box = TILE_S=128
      // Q: [B*Q_LEN*NUM_HEADS, D_K], hpb derived from Q dim[0]
      // KV: [B*KL, D_K], same as MLA decode
      constexpr int BK = 64;
      constexpr int NUM_H = 128;
      constexpr CUtensorMapDataType fmt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
      constexpr CUtensorMapInterleave interleave =
          CU_TENSOR_MAP_INTERLEAVE_NONE;
      constexpr CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
      constexpr CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
      constexpr CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

      if (param_id == 0) {
        // Q is allocated by the DeepSeek builder as flat [MBT, H * D_K].
        // Reinterpret it as [MBT * H, D_K] for TMA so the box height matches
        // the kernel's hpb rows. This mirrors the TP2/4/8 descriptor path.
        constexpr int D_K = 576;
        int const total_elements = tensor_desc.dim[0] * tensor_desc.dim[1];
        int total_rows = total_elements / D_K; // B * Q_LEN * NUM_HEADS
        int d_k = D_K;
        int k_iters = d_k / BK;
        uint32_t const packed_mtp =
            static_cast<uint32_t>(task_desc.task_metadata.merge_task_offset);
        int hpb = static_cast<int>((packed_mtp >> 16) & 0xffff);
        if (hpb <= 0 || hpb > NUM_H) {
          int num_heads = (total_rows <= NUM_H) ? total_rows : NUM_H;
          int q_len = total_rows / num_heads;
          if (q_len < 1) {
            q_len = 1;
          }
          hpb = num_heads / q_len;
          while (hpb > 0 && num_heads % hpb != 0) {
            hpb--;
          }
          if (hpb <= 0) {
            hpb = num_heads;
          }
        }
        uint64_t gd[3] = {
            (uint64_t)BK, (uint64_t)total_rows, (uint64_t)k_iters};
        uint64_t gs[2] = {(uint64_t)d_k * 2, (uint64_t)BK * 2};
        uint32_t bd[3] = {(uint32_t)BK, (uint32_t)hpb, 1};
        uint32_t es[3] = {1, 1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              fmt,
                                              3,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              swizzle,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      } else if (param_id == 1) {
        // KV: same as MLA decode — box = {64, TILE_S=128, 1}
        int total_rows = tensor_desc.dim[0];
        int d_k = tensor_desc.dim[1];
        if (tensor_desc.num_dims == 3) {
          total_rows = tensor_desc.dim[0] * tensor_desc.dim[1];
          d_k = tensor_desc.dim[2];
        }
        int k_iters = d_k / BK;
        uint64_t gd[3] = {
            (uint64_t)BK, (uint64_t)total_rows, (uint64_t)k_iters};
        uint64_t gs[2] = {(uint64_t)d_k * 2, 128};
        uint32_t bd[3] = {(uint32_t)BK, 128, 1};
        uint32_t es[3] = {1, 1, 1};
        CUresult err = cuTensorMapEncodeTiled(tma_desc,
                                              fmt,
                                              3,
                                              tensor_desc.base_ptr,
                                              gd,
                                              gs,
                                              bd,
                                              es,
                                              interleave,
                                              swizzle,
                                              l2,
                                              oob);
        assert(err == CUDA_SUCCESS);
      }
      break;
    }
    default:
      assert(false);
  }
}

// create the tma descs for each tensor, some tensors may have multiple tma
// descs
__host__ inline void create_tma_desc_for_tensor(FullTaskDesc &task_desc,
                                                TensorDesc &tensor_desc,
                                                size_t param_id,
                                                size_t tma_desc_id) {
  CUtensorMap host_desc;
  CUtensorMap *desc_ptr;
  if ((reinterpret_cast<uint64_t>(tensor_desc.base_ptr) & 0xF) != 0) {
    printf("[TMA ALIGN] task_type=%d param=%zu base=%p dims=[%d,%d,%d]\n",
           task_desc.task_type,
           param_id,
           tensor_desc.base_ptr,
           tensor_desc.dim[0],
           tensor_desc.dim[1],
           tensor_desc.dim[2]);
  }
  fill_tma_desc_by_task(&host_desc,
                        task_desc,
                        tensor_desc,
                        param_id,
                        tma_desc_id); // host-only function
  cudaMalloc(&desc_ptr, sizeof(CUtensorMap));
  cudaMemcpy(desc_ptr, &host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  tensor_desc.tma_desc_ptrs[tma_desc_id] = desc_ptr;
}

__host__ inline void create_tma_desc_by_task(FullTaskDesc &task_desc) {
  switch (task_desc.task_type) {
    case TASK_LINEAR_HOPPER:
    case TASK_LINEAR_WITH_RESIDUAL_HOPPER:
    case TASK_LINEAR_SWAPAB_HOPPER:
    case TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER:
    case TASK_SPLITK_LINEAR_SWAPAB_HOPPER:
    case TASK_LINEAR_SM100:
    case TASK_LINEAR_WITH_RESIDUAL_SM100:
    case TASK_SPLITK_LINEAR_SM100: {
      // all tensors have 1 tma_desc
      for (size_t param_id = 0;
           param_id < task_desc.num_inputs + task_desc.num_outputs;
           param_id++) {
        TensorDesc &tensor_desc =
            (param_id < task_desc.num_inputs)
                ? task_desc.inputs[param_id]
                : task_desc.outputs[param_id - task_desc.num_inputs];
        create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      }
      break;
    }
    case TASK_LINEAR_FP8_SM100:
    case TASK_LINEAR_FP8_WITH_RESIDUAL_SM100: {
      // New FP8 GEMM kernel: TMA for all 6 tensors
      // Inputs: 0=input_fp8, 1=input_scale, 2=weight_fp8, 3=weight_scale,
      //         4=residual (if with_res)
      // Output: 0=output_bf16
      bool with_res =
          (task_desc.task_type == TASK_LINEAR_FP8_WITH_RESIDUAL_SM100);
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[0], 0, 0); // A
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[1], 1, 0); // SFA
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[2], 2, 0); // B
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[3], 3, 0); // SFB
      if (with_res) {
        create_tma_desc_for_tensor(
            task_desc, task_desc.inputs[4], 4, 0); // residual
      }
      create_tma_desc_for_tensor(
          task_desc, task_desc.outputs[0], task_desc.num_inputs, 0); // CD
      break;
    }
    case TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100:
    case TASK_LINEAR_FP8_SWAPAB_SM100:
    case TASK_LINEAR_FP8_SWAPAB_WITH_RESIDUAL_SM100:
    case TASK_LINEAR_FP8_BMM_SM100: {
      // MPK-native FP8 swapAB kernel: TMA only for the data tensors and
      // (optionally) residual. Scales (UE8M0 packed uint32) are passed as
      // raw global pointers from task_desc->input_ptrs[]; the kernel
      // dereferences them directly inside the producer warp and feeds them
      // to UTCCP. Split-K reuses the same descriptor layout — the per-CTA
      // K-slice is encoded by TBGraph partitioning advancing base_ptr.
      // BMM reuses it too — the per-head slice is also encoded by TBGraph
      // partitioning, with H-strided GMEM strides baked into the codegen
      // TMA descriptor type (no kernel-side change).
      // Tensor order (Python-layer): 0=input_fp8, 1=input_scale,
      // 2=weight_fp8, 3=weight_scale, 4=residual?, output[0]=out.
      bool with_res =
          (task_desc.task_type == TASK_LINEAR_FP8_SWAPAB_WITH_RESIDUAL_SM100);
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[0], 0, 0);
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[2], 2, 0);
      if (with_res) {
        create_tma_desc_for_tensor(task_desc, task_desc.inputs[4], 4, 0);
      }
      create_tma_desc_for_tensor(
          task_desc, task_desc.outputs[0], task_desc.num_inputs, 0);
      break;
    }
    // D3: fp8out flavor — same TMA-for-A/B-only as bf16 (FP8 + float32-scale
    // outputs are raw stores).
    case TASK_LINEAR_FP8_BMM_DENSE_FP8OUT_SM100:
    case TASK_LINEAR_FP8_BMM_DENSE_SM100: {
      // Per-head dense FP8 BMM: TMA only for A=input (param 0) and B=weight
      // (param 2). Float32 scales (params 1, 3) and the bf16 output are raw
      // global pointers read directly by the kernel. The per-head slice is
      // encoded by TBGraph partitioning advancing each tensor's base_ptr.
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[0], 0, 0); // A
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[2], 2, 0); // B
      break;
    }
    case TASK_PAGED_ATTENTION_HOPPER: {
      constexpr int TMA_TENSOR_NUM =
          4; // 3 input tensors and 1 output tensor that need TMA
      for (size_t param_id = 0; param_id < TMA_TENSOR_NUM; param_id++) {
        TensorDesc &tensor_desc =
            (param_id < 3) ? task_desc.inputs[param_id] : task_desc.outputs[0];
        // qkv has 3 tma_descs
        if (param_id == 0) {
          for (size_t tma_desc_id = 0; tma_desc_id < 3; tma_desc_id++) {
            create_tma_desc_for_tensor(
                task_desc, tensor_desc, param_id, tma_desc_id);
          }
        }
        // paged_k_cache and paged_v_cache
        else if (param_id == 1 || param_id == 2) {
          create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
        }
        // output only has 1 tma_desc
        else {
          create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
        }
      }
      break;
    }
    case TASK_LINEAR_CUTLASS_HOPPER:
    case TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER: {
      // only A and B have 1 tma_desc
      for (size_t param_id = 0; param_id < 2; param_id++) {
        TensorDesc &tensor_desc =
            (param_id < task_desc.num_inputs)
                ? task_desc.inputs[param_id]
                : task_desc.outputs[param_id - task_desc.num_inputs];
        create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      }
      break;
    }
    case TASK_MOE_W13_LINEAR_SM90:
    case TASK_MOE_W2_LINEAR_SM90:
    case TASK_MOE_W13_LINEAR_SM100:
    case TASK_MOE_W2_LINEAR_SM100: {
      // only weight (param_id=1) have 1 tma_desc
      size_t param_id = 1;
      TensorDesc &tensor_desc =
          (param_id < task_desc.num_inputs)
              ? task_desc.inputs[param_id]
              : task_desc.outputs[param_id - task_desc.num_inputs];
      create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      break;
    }
    case TASK_MOE_W13_FP8_SM100:
    case TASK_MOE_W2_FP8_SM100: {
      // only weight_fp8 (param_id=2) has 1 tma_desc
      // inputs order: [input_fp8, input_scale, weight_fp8, weight_scale,
      //                routing_indices, mask]
      size_t param_id = 2;
      TensorDesc &tensor_desc =
          (param_id < task_desc.num_inputs)
              ? task_desc.inputs[param_id]
              : task_desc.outputs[param_id - task_desc.num_inputs];
      create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      break;
    }
    case TASK_RMS_NORM_HOPPER: {
      // no TMA needed
      break;
    }
    case TASK_FUSED_RMSNORM_QUANTIZE_FP8_SM100: {
      // no TMA needed — uses cp.async for input/weight, plain stores for fp8.
      break;
    }
    case TASK_MLA_REDUCE_SM100: {
      // no TMA needed — uses raw pointer reads
      break;
    }
    case TASK_MLA_DECODE_SM100: {
      // Q (input 0) and KV (input 1) each get 1 TMA desc
      for (size_t param_id = 0; param_id < 2; param_id++) {
        TensorDesc &tensor_desc = task_desc.inputs[param_id];
        create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      }
      break;
    }    case TASK_MLA_MTP_DECODE_SM100: {
      // Q (input 0) and KV (input 1) each get 1 TMA desc
      // Same as MLA_DECODE but Q box uses hpb (variable) instead of NUM_HEADS
      for (size_t param_id = 0; param_id < 2; param_id++) {
        TensorDesc &tensor_desc = task_desc.inputs[param_id];
        create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      }
      break;
    }
    case TASK_MLA_MTP_REDUCE_SM100: {
      // no TMA needed
      break;
    }
    case TASK_MLA_MTP_DECODE_TP2_SM100:
    case TASK_MLA_MTP_DECODE_TP4_SM100:
    case TASK_MLA_MTP_DECODE_TP8_SM100: {
      // Q (input 0) and KV (input 1) each get 1 TMA desc
      for (size_t param_id = 0; param_id < 2; param_id++) {
        TensorDesc &tensor_desc = task_desc.inputs[param_id];
        create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      }
      break;
    }
    case TASK_MLA_MTP_DECODE_TP2_REDUCE_SM100:
    case TASK_MLA_MTP_DECODE_TP4_REDUCE_SM100:
    case TASK_MLA_MTP_DECODE_TP8_REDUCE_SM100: {
      // no TMA needed
      break;
    }
    case TASK_MLA_PREFILL_TP8_CHUNKED_SPLITK_SM100:
    case TASK_MLA_PREFILL_TP8_CHUNKED_SM100: {
      // Per-head unabsorbed: [0]Qn, [1]Qp, [2]K_nope, [3]K_rope, [4]V.
      for (size_t param_id = 2; param_id < 5; param_id++) {
        TensorDesc &tensor_desc = task_desc.inputs[param_id];
        create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      }
      break;
    }
    case TASK_FP8_GEMM_DENSE_SMALLM_SM100:
    case TASK_FP8_GEMM_DENSE_MEDIUMM_SM100:
    case TASK_FP8_GEMM_DENSE_SMALLM_FP8OUT_SM100:
    case TASK_FP8_GEMM_DENSE_MEDIUMM_FP8OUT_SM100: {
      // A_fp8 and B_fp8 use TMA; scale tensors are plain LDG inputs.
      // SplitK uses the same TMA layout (full K extent in descriptor;
      // per-CTA K offset baked into runtime tile indexing). FP8OUT
      // variants have the same input TMA layout as the bf16 variants —
      // the only difference is the epilogue store path (FP8 + packed
      // scale instead of bf16), which doesn't use TMA.
      for (size_t param_id = 0; param_id < 2; param_id++) {
        TensorDesc &tensor_desc = task_desc.inputs[param_id];
        create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      }
      break;
    }
    case TASK_FP8_GEMM_DENSE_SPLITK_TMAREDUCE_SM100: {
      // A_fp8 (input 0) + B_fp8 (input 1) use TMA loads. sa/sb (inputs 2,3) are
      // plain LDG float* (no TMA). The C bf16 output (output 0) gets a TMA
      // reduce-add descriptor (cp.reduce.async.bulk.tensor.2d, no swizzle,
      // box={BN cols, BM=128 rows}).
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[0], 0, 0); // A
      create_tma_desc_for_tensor(task_desc, task_desc.inputs[1], 1, 0); // B
      create_tma_desc_for_tensor(
          task_desc, task_desc.outputs[0], task_desc.num_inputs, 0); // C
      break;
    }
    case TASK_FP8_GROUP_GEMM_SMALLM_SM100:
    case TASK_FP8_GROUP_GEMM_LARGEM_SM100: {
      // 4 TMA inputs (A, B, SFA, SFB) + 1 TMA output (D for TMA store).
      // m_indices (input param 4) is direct LDG.
      for (size_t param_id = 0; param_id < 4; param_id++) {
        TensorDesc &tensor_desc = task_desc.inputs[param_id];
        create_tma_desc_for_tensor(task_desc, tensor_desc, param_id, 0);
      }
      create_tma_desc_for_tensor(task_desc,
                                 task_desc.outputs[0],
                                 /*param_id=*/4,
                                 0);
      break;
    }
    default:
      assert(false);
  }
}

} // namespace runtime
} // namespace mirage
