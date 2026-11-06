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
#include "common/bfloat16.h"
#include <c10/util/Exception.h>
#include <cstdint>
#include <cuda.h>
#include <iostream>
namespace kernel {
namespace tma {

inline void check_cu_tma(CUresult err) {
  if (err == CUDA_SUCCESS) {
    return;
  }
  char const *error_msg_ptr = nullptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS) {
    error_msg_ptr = "unable to get error string";
  }
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

inline void init_AB_tmap_fp4(CUtensorMap *tmap,
                             char const *ptr,
                             uint64_t global_height,
                             uint64_t global_width,
                             uint32_t shared_height,
                             uint32_t shared_width) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank] = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank - 1] = {global_width / 2, 128};
  uint32_t boxDim[rank] = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank] = {1, 1, 1};

  check_cu_tma(cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
      rank,
      const_cast<char *>(ptr),
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

inline void init_SF_tmap_fp4(CUtensorMap *tmap,
                             char const *ptr,
                             uint64_t rows,
                             uint64_t reduction_size,
                             uint32_t shared_k_blocks) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank] = {256, 2 * (reduction_size / 64), rows / 128};
  uint64_t globalStrides[rank - 1] = {256, 512 * (reduction_size / 64)};
  uint32_t boxDim[rank] = {256, 2 * shared_k_blocks, 1};
  uint32_t elementStrides[rank] = {1, 1, 1};

  check_cu_tma(cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
      rank,
      const_cast<char *>(ptr),
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

inline void init_C_tmap_fp4(CUtensorMap *tmap,
                            void *ptr,
                            uint64_t batch_size,
                            uint64_t output_size,
                            uint32_t tile_rows,
                            uint32_t tile_cols) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank] = {output_size, batch_size};
  uint64_t globalStrides[rank - 1] = {output_size * sizeof(type::bfloat16_t)};
  uint32_t boxDim[rank] = {tile_cols, tile_rows};
  uint32_t elementStrides[rank] = {1, 1};

  check_cu_tma(cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      ptr,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

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
struct tma_fp4 {

  CUtensorMap *desc_ptr;

  static constexpr size_t GMEM_ROW = GMEM_ROW_;
  static constexpr size_t GMEM_COL = GMEM_COL_;
  static constexpr size_t SMEM_ROW = SMEM_ROW_;
  static constexpr size_t SMEM_COL = SMEM_COL_;

  static constexpr size_t SMEM_REPEAT_COL = SMEM_REPEAT_COL_;
  static constexpr size_t SMEM_REPEAT_ROW = SMEM_REPEAT_ROW_;

  __device__ inline tma_fp4(CUtensorMap *desc_ptr) {
    this->desc_ptr = desc_ptr;
  }

  __host__ inline tma_fp4(void *src) {
    CUtensorMap host_desc;
    create_tma_desc_nvfp4(&host_desc, src); // host-only function
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
  __host__ __device__ inline CUtensorMap *get_tma_descriptor() const {
    return desc_ptr;
  }

  template <int NDIM>
  __device__ inline void prefetch(int const (&tma_coords)[NDIM]) const {
#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);

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
        "cp.async.bulk.prefetch.tensor.5d.L2.global.tile "
        "[%0, {%1, %2, %3, %4, %5}];"
        :
        : "l"(gmem_int_desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4)
        : "memory");
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  template <int NDIM, typename Barrier>
  __device__ inline void tma_cp_async(Barrier &mbar,
                                      void *smem_ptr,
                                      int const (&tma_coords)[NDIM]) const {
#pragma unroll
    for (size_t i = 0; i < SMEM_REPEAT_ROW; i++) {
      for (size_t j = 0; j < SMEM_REPEAT_COL; j++) {
        int smem_offset = SMEM_STRIDE_ * j;
        int const tma_coords_local[NDIM] = {
            tma_coords[0] + static_cast<int>(j * SMEM_COL),
            tma_coords[1] + static_cast<int>(i * SMEM_ROW)};
#if 1
#endif
        launch_tma_cp_async(
            mbar, static_cast<T *>(smem_ptr) + smem_offset, tma_coords_local);
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

  __host__ static inline void create_tma_desc_nvfp4(CUtensorMap *tma_desc,
                                                    void *src) {
    static_assert(ROW_MAJOR == true);
    constexpr uint32_t tma_dim = 5;
    void *global_addr = src;

    constexpr CUtensorMapDataType tma_format =
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
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
