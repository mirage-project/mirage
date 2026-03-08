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
// #include "../common/utils.h"
#include "barrier.cuh"
#include <cuda.h>
namespace kernel {
namespace tma {

template <typename T,
          int B,
          int M,
          int S,
          size_t GMEM_ROW_,
          size_t GMEM_COL_,
          size_t SMEM_ROW_,
          size_t SMEM_COL_,
          size_t GMEM_STRIDE_ROW_ = 1,
          size_t GMEM_STRIDE_COL_ = 1,
          size_t SMEM_REPEAT_ROW_ = 1,
          size_t SMEM_REPEAT_COL_ = 1,
          size_t SMEM_STRIDE_ = 1,
          bool ROW_MAJOR = true>
struct tma_2d {

  CUtensorMap *desc_ptr;

  static constexpr size_t GMEM_ROW = GMEM_ROW_;
  static constexpr size_t GMEM_COL = GMEM_COL_;
  static constexpr size_t SMEM_ROW = SMEM_ROW_;
  static constexpr size_t SMEM_COL = SMEM_COL_;

  static constexpr size_t SMEM_REPEAT_COL = SMEM_REPEAT_COL_;
  static constexpr size_t SMEM_REPEAT_ROW = SMEM_REPEAT_ROW_;

  __device__ inline tma_2d(CUtensorMap *desc_ptr) {
    this->desc_ptr = desc_ptr;
  }

  __host__ inline tma_2d(void *src) {
    CUtensorMap host_desc;
    create_tma_desc(&host_desc, src); // host-only function
    cudaMalloc(&desc_ptr, sizeof(CUtensorMap));
    cudaMemcpy(
        desc_ptr, &host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, desc_ptr);
#ifdef MIRAGE_DEBUG_HOPPER
    std::cout << "Memory type: " << attr.type << std::endl;
#endif
  }

public:
  template <int NDIM, typename Barrier>
  __device__ inline void tma_cp_async(Barrier &mbar,
                                      T *smem_ptr,
                                      int const (&tma_coords)[NDIM]) const {
#pragma unroll
    for (size_t i = 0; i < SMEM_REPEAT_ROW; i++) {
      for (size_t j = 0; j < SMEM_REPEAT_COL; j++) {
        int smem_offset = SMEM_STRIDE_ * j;
        int const tma_coords_local[NDIM] = {
            tma_coords[0] + static_cast<int>(j * SMEM_COL),
            tma_coords[1] + static_cast<int>(i * SMEM_ROW)};
#if 0
        printf("tma_coords: %d, %d\n", tma_coords[0], tma_coords[1]);
        printf("tma_coords_local: %d, %d\n",
              tma_coords_local[0],
              tma_coords_local[1]);
        printf("smem_offset: %d\n", smem_offset);
        printf("smem_ptr: %p\n", smem_ptr);
        printf("smem_ptr + smem_offset: %p\n", smem_ptr + smem_offset);
#endif
        launch_tma_cp_async(mbar, smem_ptr + smem_offset, tma_coords_local);
      }
    }
  }

  template <int NDIM, typename Barrier>
  __device__ inline void launch_tma_cp_async(
      Barrier &mbar, T *smem_ptr, int const (&tma_coords)[NDIM]) const {
#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar =
        static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
    uint32_t smem_int_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    if constexpr (NDIM > 0) {
      c0 = tma_coords[0];
    }
    if constexpr (NDIM > 1) {
      c1 = tma_coords[1];
    }
    if constexpr (NDIM > 2) {
      c2 = tma_coords[2];
    }
    if constexpr (NDIM > 3) {
      c3 = tma_coords[3];
    }
    if constexpr (NDIM > 4) {
      c4 = tma_coords[4];
    }

    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier:"
                 ":complete_tx::bytes"
                 " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
                 :
                 : "r"(smem_int_ptr),
                   "l"(gmem_int_desc),
                   "r"(smem_int_mbar),
                   "r"(c0),
                   "r"(c1),
                   "r"(c2),
                   "r"(c3),
                   "r"(c4)
                 : "memory");
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  template <int NDIM>
  __device__ inline void tma_store_async(T *smem_ptr,
                                         int const (&tma_coords)[NDIM]) const {
#pragma unroll
    for (size_t i = 0; i < SMEM_REPEAT_ROW; i++) {
      for (size_t j = 0; j < SMEM_REPEAT_COL; j++) {
        int smem_offset = SMEM_STRIDE_ * j;
        int const tma_coords_local[NDIM] = {
            tma_coords[0] + static_cast<int>(j * SMEM_COL),
            tma_coords[1] + static_cast<int>(i * SMEM_ROW)};
        launch_tma_store_async(smem_ptr + smem_offset, tma_coords_local);
      }
    }
  }

  template <int NDIM>
  __device__ inline void
      tma_reduce_add_async(T *smem_ptr, int const (&tma_coords)[NDIM]) const {
#pragma unroll
    for (size_t i = 0; i < SMEM_REPEAT_ROW; i++) {
      for (size_t j = 0; j < SMEM_REPEAT_COL; j++) {
        int smem_offset = SMEM_STRIDE_ * j;
        int const tma_coords_local[NDIM] = {
            tma_coords[0] + static_cast<int>(j * SMEM_COL),
            tma_coords[1] + static_cast<int>(i * SMEM_ROW)};
        launch_tma_reduce_add_async(smem_ptr + smem_offset, tma_coords_local);
      }
    }
  }

  template <int NDIM>
  __device__ inline void
      launch_tma_store_async(void *smem_ptr,
                             int const (&tma_coords)[NDIM]) const {
#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    if constexpr (NDIM > 0) {
      c0 = tma_coords[0];
    }
    if constexpr (NDIM > 1) {
      c1 = tma_coords[1];
    }
    if constexpr (NDIM > 2) {
      c2 = tma_coords[2];
    }
    if constexpr (NDIM > 3) {
      c3 = tma_coords[3];
    }
    if constexpr (NDIM > 4) {
      c4 = tma_coords[4];
    }

    asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, "
                 "{%2, %3, %4, %5, %6}], [%1];"
                 :
                 : "l"(gmem_int_desc),
                   "r"(smem_int_ptr),
                   "r"(c0),
                   "r"(c1),
                   "r"(c2),
                   "r"(c3),
                   "r"(c4)
                 : "memory");
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  template <int NDIM>
  __device__ inline void
      launch_tma_reduce_add_async(void *smem_ptr,
                                  int const (&tma_coords)[NDIM]) const {
#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    if constexpr (NDIM > 0) {
      c0 = tma_coords[0];
    }
    if constexpr (NDIM > 1) {
      c1 = tma_coords[1];
    }
    if constexpr (NDIM > 2) {
      c2 = tma_coords[2];
    }
    if constexpr (NDIM > 3) {
      c3 = tma_coords[3];
    }
    if constexpr (NDIM > 4) {
      c4 = tma_coords[4];
    }

    asm volatile(
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.bulk_group [%0, "
        "{%2, %3, %4, %5, %6}], [%1];"
        :
        : "l"(gmem_int_desc),
          "r"(smem_int_ptr),
          "r"(c0),
          "r"(c1),
          "r"(c2),
          "r"(c3),
          "r"(c4)
        : "memory");
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

private:
  __host__ static inline void create_tma_desc(CUtensorMap *tma_desc,
                                              void *src) {
    static_assert(ROW_MAJOR == true);
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

    uint64_t gmem_prob_shape[5] = {GMEM_COL, GMEM_ROW, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {
        sizeof(T), GMEM_STRIDE_ROW_ * sizeof(T), 0, 0, 0};

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

    uint32_t smem_box_shape[5] = {SMEM_COL, SMEM_ROW, 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    assert(smem_box_shape[0] >= (uint32_t(1))); // Size must be min 1
    assert(smem_box_shape[0] <=
           (uint32_t(1) << 8));                 // Size must be max 2^8 = 256
    assert(smem_box_shape[1] >= (uint32_t(1))); // Size must be min 1
    assert(smem_box_shape[1] <=
           (uint32_t(1) << 8));                 // Size must be max 2^8 = 256
    assert(smem_box_shape[2] >= (uint32_t(1))); // Size must be min 1
    assert(smem_box_shape[2] <=
           (uint32_t(1) << 8));                 // Size must be max 2^8 = 256
    assert(smem_box_shape[3] >= (uint32_t(1))); // Size must be min 1
    assert(smem_box_shape[3] <=
           (uint32_t(1) << 8));                 // Size must be max 2^8 = 256
    assert(smem_box_shape[4] >= (uint32_t(1))); // Size must be min 1
    assert(smem_box_shape[4] <=
           (uint32_t(1) << 8)); // Size must be max 2^8 = 256

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
};
// cutlass/include/cute/atom/copy_traits_sm90_tma.hpp
}; // namespace tma

} // namespace kernel
