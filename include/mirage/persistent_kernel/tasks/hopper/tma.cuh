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
#include "../common.h"
#include "barrier.cuh"
#include <cuda.h>
namespace kernel {
namespace tma {

__device__ static inline void async_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;");
}

__device__ static inline void store_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int N = 0>
__device__ static inline void store_async_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
}

template <typename T,
          int B,
          int M,
          int S,
          size_t GMEM_ROW, // 64
          size_t GMEM_COL, // 4096
          size_t SMEM_ROW, // 64
          size_t SMEM_COL, // 64
          bool ROW_MAJOR = true>
struct tma {

  CUtensorMap *desc_ptr;

  __host__ inline tma(void *src) {
    CUtensorMap host_desc;
    create_tma_desc(&host_desc, src); // host-only function
    cudaMalloc(&desc_ptr, sizeof(CUtensorMap));
    cudaMemcpy(
        desc_ptr, &host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, desc_ptr);
    std::cout << "Memory type: " << attr.type << std::endl;
  }

public:
  __device__ inline void
      tma_cp_async(Barrier &mbar, T *smem_ptr, int2 const &tma_coords) const {
#ifdef MIRAGE_GRACE_HOPPER
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar =
        static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
    uint32_t smem_int_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    // if (threadIdx.x == 128 && threadIdx.y == 0 && threadIdx.z == 0) {
    //   printf("tma_cp_async: crd0: %d, crd1: %d", tma_coords.x, tma_coords.y);
    // }

    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier:"
                 ":complete_tx::bytes"
                 " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
                 :
                 : "r"(smem_int_ptr),
                   "l"(gmem_int_desc),
                   "r"(smem_int_mbar),
                   "r"(tma_coords.x),
                   "r"(tma_coords.y),
                   "r"(0),
                   "r"(0),
                   "r"(0)
                 : "memory");
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  __device__ inline void tma_store_async(void *smem_ptr,
                                         int2 const &tma_coords) const {
#ifdef MIRAGE_GRACE_HOPPER
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    // not sure what this line means
    //  cutlass::arch::synclog_emit_tma_load(
    asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, "
                 "{%2, %3, %4, %5, %6}], [%1];"
                 :
                 : "l"(gmem_int_desc),
                   "r"(smem_int_ptr),
                   "r"(tma_coords.x),
                   "r"(tma_coords.y),
                   "r"(0),
                   "r"(0) "r"(0)
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
    constexpr CUtensorMapSwizzle tma_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    // (B == 1    ? CU_TENSOR_MAP_SWIZZLE_32B
    //  : B == 2  ? CU_TENSOR_MAP_SWIZZLE_64B
    //  : B == 3 ? CU_TENSOR_MAP_SWIZZLE_128B
    //             : CU_TENSOR_MAP_SWIZZLE_NONE);
    //  constexpr CUtensorMapSwizzle tma_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;

    //     printf("GMEM_ROW: %zu, GMEM_COL: %zu\n", GMEM_ROW, GMEM_COL);

    uint64_t gmem_shape[5] = {GMEM_COL, GMEM_ROW, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(T), GMEM_COL * sizeof(T), 0, 0, 0};
    uint32_t smem_box_shape[5] = {SMEM_COL, SMEM_ROW, 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    uint64_t const *gmem_shape_ptr = &gmem_shape[0];
    uint64_t const *gmem_stride_ptr = &gmem_prob_stride[0];
    uint32_t const *smem_box_shape_ptr = &smem_box_shape[0];
    uint32_t const *smem_box_stride_ptr = &smem_box_stride[0];

    // //     TMA descriptor does not store the zeroth stride and assumes it is
    // 1 (TmaInternalType element).
    //     assert(gmem_prob_stride[0] == 1 && "Majorness of smem doesn't match
    //     majorness of gmem");

    //     // ensure that the global address is always 16-byte aligned
    //     assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    //     assert((gmem_prob_stride[1]) <
    //            (uint64_t(1) << 40)); // Stride must be max 2^40
    //     assert((gmem_prob_stride[1] & 0b1111) ==
    //            0); // Stride must be multiple of 16B (128b)
    //     assert((gmem_prob_stride[2]) <
    //            (uint64_t(1) << 40)); // Stride must be max 2^40
    //     assert((gmem_prob_stride[2] & 0b1111) ==
    //            0); // Stride must be multiple of 16B (128b)
    //     assert((gmem_prob_stride[3]) <
    //            (uint64_t(1) << 40)); // Stride must be max 2^40
    //     assert((gmem_prob_stride[3] & 0b1111) ==
    //            0); // Stride must be multiple of 16B (128b)
    //     assert((gmem_prob_stride[4]) <
    //            (uint64_t(1) << 40)); // Stride must be max 2^40
    //     assert((gmem_prob_stride[4] & 0b1111) ==
    //            0); // Stride must be multiple of 16B (128b)

    //     assert(smem_box_shape[0] >= (uint32_t(1))); // Size must be min 1
    //     assert(smem_box_shape[0] <=
    //            (uint32_t(1) << 8));                 // Size must be max 2^8 =
    //            256
    //     assert(smem_box_shape[1] >= (uint32_t(1))); // Size must be min 1
    //     assert(smem_box_shape[1] <=
    //            (uint32_t(1) << 8));                 // Size must be max 2^8 =
    //            256
    //     assert(smem_box_shape[2] >= (uint32_t(1))); // Size must be min 1
    //     assert(smem_box_shape[2] <=
    //            (uint32_t(1) << 8));                 // Size must be max 2^8 =
    //            256
    //     assert(smem_box_shape[3] >= (uint32_t(1))); // Size must be min 1
    //     assert(smem_box_shape[3] <=
    //            (uint32_t(1) << 8));                 // Size must be max 2^8 =
    //            256
    //     assert(smem_box_shape[4] >= (uint32_t(1))); // Size must be min 1
    //     assert(smem_box_shape[4] <=
    //            (uint32_t(1) << 8)); // Size must be max 2^8 = 256

    //     assert(smem_box_stride[0] >= (uint32_t(1))); // Stride must be min 1
    //     assert(smem_box_stride[0] <= (uint32_t(8))); // Stride must be max
    //     2^3 = 8 assert(smem_box_stride[1] >= (uint32_t(1))); // Stride must
    //     be min 1 assert(smem_box_stride[1] <= (uint32_t(8))); // Stride must
    //     be max 2^3 = 8 assert(smem_box_stride[2] >= (uint32_t(1))); // Stride
    //     must be min 1 assert(smem_box_stride[2] <= (uint32_t(8))); // Stride
    //     must be max 2^3 = 8 assert(smem_box_stride[3] >= (uint32_t(1))); //
    //     Stride must be min 1 assert(smem_box_stride[3] <= (uint32_t(8))); //
    //     Stride must be max 2^3 = 8 assert(smem_box_stride[4] >=
    //     (uint32_t(1))); // Stride must be min 1 assert(smem_box_stride[4] <=
    //     (uint32_t(8))); // Stride must be max 2^3 = 8

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
                << gmem_shape << "\nglobalStrides  " << gmem_prob_stride
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
