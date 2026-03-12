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
#include <cuda.h>

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

  assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) ==
         0); // Address must be 16B-aligned

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
    assert(false);
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
        // TMA_INPUT
        int const batch_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_size)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(MMA_N),
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
        // TMA_OUT
        int const batch_size = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const output_stride = (tensor_desc.stride[0]);
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(MMA_N),
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
    case TASK_SPLITK_LINEAR_SM100: {
      int const cp_async_size = 64;
      const size_t smem_repeat_row = 1;
      constexpr int B = 3;
      constexpr int M = 3;
      constexpr int S = 3;
      constexpr int MMA_M = 128;
      constexpr int MMA_N = 16;

      if (param_id == 0) {
        // TMA_INPUT
        int const batch_size = tensor_desc.dim[0];
        int const reduction_size = tensor_desc.dim[1];
        int const reduction_stride = tensor_desc.stride[0];
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(reduction_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(reduction_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(MMA_N),
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
        // TMA_OUT
        int const batch_size = tensor_desc.dim[0];
        int const output_size = tensor_desc.dim[1];
        int const output_stride = (tensor_desc.stride[0]);
        uint64_t gmem_shape[2] = {static_cast<uint64_t>(batch_size),
                                  static_cast<uint64_t>(output_size)};
        uint64_t gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
        uint32_t smem_shape[2] = {static_cast<uint32_t>(MMA_N),
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
    case TASK_RMS_NORM_HOPPER: {
      // no TMA needed
      break;
    }
    default:
      assert(false);
  }
}

} // namespace runtime
} // namespace mirage
