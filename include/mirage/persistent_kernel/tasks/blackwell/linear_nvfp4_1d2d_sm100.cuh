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

#include <cstdint>

#include "common/bfloat16.h"

#include <c10/util/Exception.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

inline void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) {
    return;
  }
  const char *error_msg_ptr = nullptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS) {
    error_msg_ptr = "unable to get error string";
  }
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

inline void check_cuda(cudaError_t err) {
  if (err == cudaSuccess) {
    return;
  }
  TORCH_CHECK(false, cudaGetErrorString(err));
}

inline void init_AB_tmap(CUtensorMap *tmap,
                         const char *ptr,
                         uint64_t global_height,
                         uint64_t global_width,
                         uint32_t shared_height,
                         uint32_t shared_width) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank] = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank - 1] = {global_width / 2, 128};
  uint32_t boxDim[rank] = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank] = {1, 1, 1};

  CUresult err = cuTensorMapEncodeTiled(
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
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  check_cu(err);
}

namespace kernel {
namespace nvfp4_1d2d_detail {

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ULL;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000ULL;

__device__ __forceinline__ constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}"
      :
      : "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ __forceinline__ void
tma_gmem2smem(int dst, const void *src, int size, int mbar_addr, uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;"
      :
      : "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

__device__ __forceinline__ void tma_3d_gmem2smem(int dst,
                                                 const void *tmap_ptr,
                                                 int x,
                                                 int y,
                                                 int z,
                                                 int mbar_addr,
                                                 uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes."
      "cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
      :
      : "r"(dst),
        "l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(mbar_addr),
        "l"(cache_policy)
      : "memory");
}

__device__ __forceinline__ void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ __forceinline__ void tcgen05_mma_nvfp4(uint64_t a_desc,
                                                  uint64_t b_desc,
                                                  uint32_t i_desc,
                                                  int scale_A_tmem,
                                                  int scale_B_tmem,
                                                  int enable_input_d) {
  const int d_tmem = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}"
      :
      : "r"(d_tmem),
        "l"(a_desc),
        "l"(b_desc),
        "r"(i_desc),
        "r"(scale_A_tmem),
        "r"(scale_B_tmem),
        "r"(enable_input_d));
}

struct SHAPE {
  static constexpr char _32x32b[] = ".32x32b";
  static constexpr char _16x128b[] = ".16x128b";
  static constexpr char _16x256b[] = ".16x256b";
};

struct NUM {
  static constexpr char x4[] = ".x4";
  static constexpr char x8[] = ".x8";
  static constexpr char x16[] = ".x16";
  static constexpr char x32[] = ".x32";
  static constexpr char x64[] = ".x64";
  static constexpr char x128[] = ".x128";
};

template <const char *SHAPE_NAME, const char *NUM_NAME>
__device__ __forceinline__ void tcgen05_ld_16regs(float *tmp, int row, int col) {
  asm volatile(
      "tcgen05.ld.sync.aligned%17%18.b32 "
      "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
      "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
      : "=f"(tmp[0]),
        "=f"(tmp[1]),
        "=f"(tmp[2]),
        "=f"(tmp[3]),
        "=f"(tmp[4]),
        "=f"(tmp[5]),
        "=f"(tmp[6]),
        "=f"(tmp[7]),
        "=f"(tmp[8]),
        "=f"(tmp[9]),
        "=f"(tmp[10]),
        "=f"(tmp[11]),
        "=f"(tmp[12]),
        "=f"(tmp[13]),
        "=f"(tmp[14]),
        "=f"(tmp[15])
      : "r"((row << 16) | col), "C"(SHAPE_NAME), "C"(NUM_NAME));
}

template <const char *SHAPE_NAME, const char *NUM_NAME>
__device__ __forceinline__ void tcgen05_ld_32regs(float *tmp, int row, int col) {
  asm volatile(
      "tcgen05.ld.sync.aligned%33%34.b32 "
      "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
      "  %8,  %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=f"(tmp[0]),
        "=f"(tmp[1]),
        "=f"(tmp[2]),
        "=f"(tmp[3]),
        "=f"(tmp[4]),
        "=f"(tmp[5]),
        "=f"(tmp[6]),
        "=f"(tmp[7]),
        "=f"(tmp[8]),
        "=f"(tmp[9]),
        "=f"(tmp[10]),
        "=f"(tmp[11]),
        "=f"(tmp[12]),
        "=f"(tmp[13]),
        "=f"(tmp[14]),
        "=f"(tmp[15]),
        "=f"(tmp[16]),
        "=f"(tmp[17]),
        "=f"(tmp[18]),
        "=f"(tmp[19]),
        "=f"(tmp[20]),
        "=f"(tmp[21]),
        "=f"(tmp[22]),
        "=f"(tmp[23]),
        "=f"(tmp[24]),
        "=f"(tmp[25]),
        "=f"(tmp[26]),
        "=f"(tmp[27]),
        "=f"(tmp[28]),
        "=f"(tmp[29]),
        "=f"(tmp[30]),
        "=f"(tmp[31])
      : "r"((row << 16) | col), "C"(SHAPE_NAME), "C"(NUM_NAME));
}

template <const char *SHAPE_NAME, const char *NUM_NAME>
__device__ __forceinline__ void tcgen05_ld_64regs(float *tmp, int row, int col) {
  asm volatile(
      "tcgen05.ld.sync.aligned%65%66.b32 "
      "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
      "  %8,  %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31, "
      " %32, %33, %34, %35, %36, %37, %38, %39, "
      " %40, %41, %42, %43, %44, %45, %46, %47, "
      " %48, %49, %50, %51, %52, %53, %54, %55, "
      " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
      : "=f"(tmp[0]),
        "=f"(tmp[1]),
        "=f"(tmp[2]),
        "=f"(tmp[3]),
        "=f"(tmp[4]),
        "=f"(tmp[5]),
        "=f"(tmp[6]),
        "=f"(tmp[7]),
        "=f"(tmp[8]),
        "=f"(tmp[9]),
        "=f"(tmp[10]),
        "=f"(tmp[11]),
        "=f"(tmp[12]),
        "=f"(tmp[13]),
        "=f"(tmp[14]),
        "=f"(tmp[15]),
        "=f"(tmp[16]),
        "=f"(tmp[17]),
        "=f"(tmp[18]),
        "=f"(tmp[19]),
        "=f"(tmp[20]),
        "=f"(tmp[21]),
        "=f"(tmp[22]),
        "=f"(tmp[23]),
        "=f"(tmp[24]),
        "=f"(tmp[25]),
        "=f"(tmp[26]),
        "=f"(tmp[27]),
        "=f"(tmp[28]),
        "=f"(tmp[29]),
        "=f"(tmp[30]),
        "=f"(tmp[31]),
        "=f"(tmp[32]),
        "=f"(tmp[33]),
        "=f"(tmp[34]),
        "=f"(tmp[35]),
        "=f"(tmp[36]),
        "=f"(tmp[37]),
        "=f"(tmp[38]),
        "=f"(tmp[39]),
        "=f"(tmp[40]),
        "=f"(tmp[41]),
        "=f"(tmp[42]),
        "=f"(tmp[43]),
        "=f"(tmp[44]),
        "=f"(tmp[45]),
        "=f"(tmp[46]),
        "=f"(tmp[47]),
        "=f"(tmp[48]),
        "=f"(tmp[49]),
        "=f"(tmp[50]),
        "=f"(tmp[51]),
        "=f"(tmp[52]),
        "=f"(tmp[53]),
        "=f"(tmp[54]),
        "=f"(tmp[55]),
        "=f"(tmp[56]),
        "=f"(tmp[57]),
        "=f"(tmp[58]),
        "=f"(tmp[59]),
        "=f"(tmp[60]),
        "=f"(tmp[61]),
        "=f"(tmp[62]),
        "=f"(tmp[63])
      : "r"((row << 16) | col), "C"(SHAPE_NAME), "C"(NUM_NAME));
}

template <const char *SHAPE_NAME, const char *NUM_NAME>
__device__ __forceinline__ void tcgen05_ld_128regs(float *tmp, int row, int col) {
  asm volatile(
      "tcgen05.ld.sync.aligned%129%130.b32 "
      "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
      "  %8,  %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31, "
      " %32, %33, %34, %35, %36, %37, %38, %39, "
      " %40, %41, %42, %43, %44, %45, %46, %47, "
      " %48, %49, %50, %51, %52, %53, %54, %55, "
      " %56, %57, %58, %59, %60, %61, %62, %63, "
      " %64, %65, %66, %67, %68, %69, %70, %71, "
      " %72, %73, %74, %75, %76, %77, %78, %79, "
      " %80, %81, %82, %83, %84, %85, %86, %87, "
      " %88, %89, %90, %91, %92, %93, %94, %95, "
      " %96, %97, %98, %99,%100,%101,%102,%103, "
      "%104,%105,%106,%107,%108,%109,%110,%111, "
      "%112,%113,%114,%115,%116,%117,%118,%119, "
      "%120,%121,%122,%123,%124,%125,%126,%127}, [%128];"
      : "=f"(tmp[0]),
        "=f"(tmp[1]),
        "=f"(tmp[2]),
        "=f"(tmp[3]),
        "=f"(tmp[4]),
        "=f"(tmp[5]),
        "=f"(tmp[6]),
        "=f"(tmp[7]),
        "=f"(tmp[8]),
        "=f"(tmp[9]),
        "=f"(tmp[10]),
        "=f"(tmp[11]),
        "=f"(tmp[12]),
        "=f"(tmp[13]),
        "=f"(tmp[14]),
        "=f"(tmp[15]),
        "=f"(tmp[16]),
        "=f"(tmp[17]),
        "=f"(tmp[18]),
        "=f"(tmp[19]),
        "=f"(tmp[20]),
        "=f"(tmp[21]),
        "=f"(tmp[22]),
        "=f"(tmp[23]),
        "=f"(tmp[24]),
        "=f"(tmp[25]),
        "=f"(tmp[26]),
        "=f"(tmp[27]),
        "=f"(tmp[28]),
        "=f"(tmp[29]),
        "=f"(tmp[30]),
        "=f"(tmp[31]),
        "=f"(tmp[32]),
        "=f"(tmp[33]),
        "=f"(tmp[34]),
        "=f"(tmp[35]),
        "=f"(tmp[36]),
        "=f"(tmp[37]),
        "=f"(tmp[38]),
        "=f"(tmp[39]),
        "=f"(tmp[40]),
        "=f"(tmp[41]),
        "=f"(tmp[42]),
        "=f"(tmp[43]),
        "=f"(tmp[44]),
        "=f"(tmp[45]),
        "=f"(tmp[46]),
        "=f"(tmp[47]),
        "=f"(tmp[48]),
        "=f"(tmp[49]),
        "=f"(tmp[50]),
        "=f"(tmp[51]),
        "=f"(tmp[52]),
        "=f"(tmp[53]),
        "=f"(tmp[54]),
        "=f"(tmp[55]),
        "=f"(tmp[56]),
        "=f"(tmp[57]),
        "=f"(tmp[58]),
        "=f"(tmp[59]),
        "=f"(tmp[60]),
        "=f"(tmp[61]),
        "=f"(tmp[62]),
        "=f"(tmp[63]),
        "=f"(tmp[64]),
        "=f"(tmp[65]),
        "=f"(tmp[66]),
        "=f"(tmp[67]),
        "=f"(tmp[68]),
        "=f"(tmp[69]),
        "=f"(tmp[70]),
        "=f"(tmp[71]),
        "=f"(tmp[72]),
        "=f"(tmp[73]),
        "=f"(tmp[74]),
        "=f"(tmp[75]),
        "=f"(tmp[76]),
        "=f"(tmp[77]),
        "=f"(tmp[78]),
        "=f"(tmp[79]),
        "=f"(tmp[80]),
        "=f"(tmp[81]),
        "=f"(tmp[82]),
        "=f"(tmp[83]),
        "=f"(tmp[84]),
        "=f"(tmp[85]),
        "=f"(tmp[86]),
        "=f"(tmp[87]),
        "=f"(tmp[88]),
        "=f"(tmp[89]),
        "=f"(tmp[90]),
        "=f"(tmp[91]),
        "=f"(tmp[92]),
        "=f"(tmp[93]),
        "=f"(tmp[94]),
        "=f"(tmp[95]),
        "=f"(tmp[96]),
        "=f"(tmp[97]),
        "=f"(tmp[98]),
        "=f"(tmp[99]),
        "=f"(tmp[100]),
        "=f"(tmp[101]),
        "=f"(tmp[102]),
        "=f"(tmp[103]),
        "=f"(tmp[104]),
        "=f"(tmp[105]),
        "=f"(tmp[106]),
        "=f"(tmp[107]),
        "=f"(tmp[108]),
        "=f"(tmp[109]),
        "=f"(tmp[110]),
        "=f"(tmp[111]),
        "=f"(tmp[112]),
        "=f"(tmp[113]),
        "=f"(tmp[114]),
        "=f"(tmp[115]),
        "=f"(tmp[116]),
        "=f"(tmp[117]),
        "=f"(tmp[118]),
        "=f"(tmp[119]),
        "=f"(tmp[120]),
        "=f"(tmp[121]),
        "=f"(tmp[122]),
        "=f"(tmp[123]),
        "=f"(tmp[124]),
        "=f"(tmp[125]),
        "=f"(tmp[126]),
        "=f"(tmp[127])
      : "r"((row << 16) | col), "C"(SHAPE_NAME), "C"(NUM_NAME));
}

__device__ __forceinline__ void tcgen05_ld_32x32bx32(float *tmp, int row, int col) {
  tcgen05_ld_32regs<SHAPE::_32x32b, NUM::x32>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_32x32bx64(float *tmp, int row, int col) {
  tcgen05_ld_64regs<SHAPE::_32x32b, NUM::x64>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_32x32bx128(float *tmp, int row, int col) {
  tcgen05_ld_128regs<SHAPE::_32x32b, NUM::x128>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_16x128bx8(float *tmp, int row, int col) {
  tcgen05_ld_16regs<SHAPE::_16x128b, NUM::x8>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_16x128bx16(float *tmp, int row, int col) {
  tcgen05_ld_32regs<SHAPE::_16x128b, NUM::x16>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_16x128bx32(float *tmp, int row, int col) {
  tcgen05_ld_64regs<SHAPE::_16x128b, NUM::x32>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_16x256bx4(float *tmp, int row, int col) {
  tcgen05_ld_16regs<SHAPE::_16x256b, NUM::x4>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_16x256bx8(float *tmp, int row, int col) {
  tcgen05_ld_32regs<SHAPE::_16x256b, NUM::x8>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_16x256bx16(float *tmp, int row, int col) {
  tcgen05_ld_64regs<SHAPE::_16x256b, NUM::x16>(tmp, row, col);
}

__device__ __forceinline__ void tcgen05_commit_arrive_cluster(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
          "[%0];"
          :
          : "r"(mbar_addr)
          : "memory");
}

}  // namespace nvfp4_1d2d_detail

template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int NUM_STAGES,
          bool C_N_MAJOR,
          int EPI_BATCH_LA = 1>
__global__ __launch_bounds__(BLOCK_M + 2 * 32) void
linear_nvfp4_1d2d_sm100_kernel(const __grid_constant__ CUtensorMap A_tmap,
                               const __grid_constant__ CUtensorMap B_tmap,
                               const char *SFA_ptr,
                               const char *SFB_ptr,
                               type::bfloat16_t *C_ptr,
                               const type::bfloat16_t *bias_ptr,
                               int M,
                               int N) {
  using namespace nvfp4_1d2d_detail;

  static_assert(BLOCK_M == 128, "SM100 NVFP4 tcgen05 MMA uses BLOCK_M == 128");
  static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K must be divisible by MMA_K");
  static_assert(REDUCTION_SIZE % BLOCK_K == 0, "K must be divisible by BLOCK_K");
  static_assert(BLOCK_N == 32 || BLOCK_N == 64 || BLOCK_N == 128, "BLOCK_N must be 32, 64, or 128");

  const int tid = threadIdx.x;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int bid_m = blockIdx.x;
  const int bid_n = blockIdx.y;
  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :
                 : "r"(smem), "r"(BLOCK_N * 2));
  }
  __syncthreads();

  constexpr int NUM_ITERS = REDUCTION_SIZE / BLOCK_K;

  auto make_desc_AB = [](int addr) -> uint64_t {
    const int SBO = 8 * 128;
    return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
  };
  
  auto make_desc_SF = [](int addr) -> uint64_t {
    const int SBO = 8 * 16;
    return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
  };

  auto issue_tma = [&](int iter_k, int stage_id) {
    uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
    uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

    const int mbar_addr = tma_mbar_addr + stage_id * 8;
    const int A_smem = smem + stage_id * STAGE_SIZE;
    const int B_smem = A_smem + A_size;
    const int SFA_smem = B_smem + B_size;
    const int SFB_smem = SFA_smem + SFA_size;
    const int off_k = iter_k * BLOCK_K;

    tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
    tma_3d_gmem2smem(B_smem, &B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

    const int rest_k = REDUCTION_SIZE / 64;
    const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / 64) * 512;
    const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / 64) * 512;

    tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
    tma_gmem2smem(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");
  };

  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    constexpr int PREFETCH_ITERS = (NUM_ITERS < NUM_STAGES) ? NUM_ITERS : NUM_STAGES;
    
    for (int iter_k = 0; iter_k < PREFETCH_ITERS; iter_k++) {
      issue_tma(iter_k, iter_k);
    }
    
    for (int iter_k = NUM_STAGES; iter_k < NUM_ITERS; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
      issue_tma(iter_k, stage_id);
    }

  } else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t)MMA_N >> 3U << 17U) | ((uint32_t)MMA_M >> 7U << 27U);

    for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) % 2;

      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      const uint64_t SF_desc = make_desc_SF(0);
      const uint64_t SFA_desc = SF_desc + (static_cast<uint64_t>(SFA_smem) >> 4ULL);
      const uint64_t SFB_desc = SF_desc + (static_cast<uint64_t>(SFB_smem) >> 4ULL);

      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc = SFA_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
        uint64_t sfb_desc = SFB_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
      }

      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        const int k1 = k / 4;
        const int k2 = k % 4;

        uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
        uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
        const int scale_A_tmem = SFA_tmem + k * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
        const int scale_B_tmem = SFB_tmem + k * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);
        const int enable_input_d = (k == 0) ? iter_k : 1;

        tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
      }
      tcgen05_commit_arrive_cluster(mma_mbar_addr + stage_id * 8);
    }
    tcgen05_commit_arrive_cluster(mainloop_mbar_addr);
  } else if (tid < BLOCK_M) {
    mbarrier_wait(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    auto epilogue_M_major = [&]() {
      constexpr int WIDTH = (BLOCK_N < 64) ? BLOCK_N : 64;
      constexpr int NUM_SUBTILES = BLOCK_N / WIDTH;
      constexpr int BATCH = (EPI_BATCH_LA <= NUM_SUBTILES && NUM_SUBTILES % EPI_BATCH_LA == 0)
                           ? EPI_BATCH_LA
                           : 1;

      auto load_subtile = [&](float *dst, int n) {
        if constexpr (WIDTH == 128) {
          tcgen05_ld_32x32bx128(dst, warp_id * 32, n * WIDTH);
        }
        if constexpr (WIDTH == 64) {
          tcgen05_ld_32x32bx64(dst, warp_id * 32, n * WIDTH);
        }
        if constexpr (WIDTH == 32) {
          tcgen05_ld_32x32bx32(dst, warp_id * 32, n * WIDTH);
        }
      };

      auto store_subtile = [&](const float *src, int n) {
        for (int i = 0; i < WIDTH; i++) {
          const int row = off_n + n * WIDTH + i;
          const int col = off_m + tid;
          const int offset = row * M + col;
          type::bfloat16_t acc_bf16(src[i]);
          if (bias_ptr != nullptr) {
            C_ptr[offset] = acc_bf16 + bias_ptr[offset];
          } else {
            C_ptr[offset] = acc_bf16;
          }
        }
      };

      for (int g = 0; g < NUM_SUBTILES; g += BATCH) {
        float tmp_batch[BATCH][WIDTH];
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
          load_subtile(tmp_batch[b], g + b);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
          store_subtile(tmp_batch[b], g + b);
        }
      }
    };

    auto epilogue_N_major = [&]() {
      for (int m = 0; m < 32 / 16; m++) {
        float tmp[BLOCK_N / 2];
        if constexpr (BLOCK_N == 128) {
          tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, 0);
        }
        if constexpr (BLOCK_N == 64) {
          tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, 0);
        }
        if constexpr (BLOCK_N == 32) {
          tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, 0);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        for (int i = 0; i < BLOCK_N / 8; i++) {
          const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
          const int col = off_n + i * 8 + (lane_id % 4) * 2;
          const int off0 = (row + 0) * N + col;
          const int off1 = (row + 8) * N + col;

          type::bfloat16_t a00(tmp[i * 4 + 0]), a01(tmp[i * 4 + 1]);
          type::bfloat16_t a10(tmp[i * 4 + 2]), a11(tmp[i * 4 + 3]);
          if (bias_ptr != nullptr) {
            C_ptr[off0 + 0] = type::bfloat16_t(float(a00) + float(bias_ptr[off0 + 0]));
            C_ptr[off0 + 1] = type::bfloat16_t(float(a01) + float(bias_ptr[off0 + 1]));
            C_ptr[off1 + 0] = type::bfloat16_t(float(a10) + float(bias_ptr[off1 + 0]));
            C_ptr[off1 + 1] = type::bfloat16_t(float(a11) + float(bias_ptr[off1 + 1]));
          } else {
            C_ptr[off0 + 0] = a00;
            C_ptr[off0 + 1] = a01;
            C_ptr[off1 + 0] = a10;
            C_ptr[off1 + 1] = a11;
          }
        }
      }
    };

    if constexpr (C_N_MAJOR) {
      epilogue_N_major();
    } else {
      epilogue_M_major();
    }

    asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
    if (warp_id == 0) {
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                   :
                   : "r"(0), "r"(BLOCK_N * 2));
    }
  }
}

}  // namespace kernel
