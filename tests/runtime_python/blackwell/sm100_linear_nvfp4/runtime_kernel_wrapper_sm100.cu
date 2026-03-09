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


#include "blackwell/linear_nvfp4_1d2d_sm100.cuh"
#include "hopper/tma_2d_nvfp4.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>
#include <iostream>
#include <vector>

// Cutlass includes
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
// #include <cutlass/half.h> // F16 data type
#include <cutlass/util/print_error.hpp>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp> // CuTe tensor implementation

using float_e2m1 = cute::float_e2m1_t;
using float_ue4m3 = cute::float_ue4m3_t;

// sm100_linear_nvfp4_1d2d

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          class BiasTensor,
          int MMA_M,
          int MMA_N,
          bool NoBias,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__global__
    __launch_bounds__(256, 1)
        void linear_nvfp4_1d2d_sm100_wrapper(void *tma_a_desc_ptr,
                                             void *tma_b_desc_ptr,
                                             void *tma_sfa_desc_ptr,
                                             void *tma_sfb_desc_ptr,
                                             BiasTensor mBias,
                                             void *tma_out_desc_ptr,
                                             uint32_t* debug_smem_sfa,
                                             uint32_t* debug_smem_sfb,
                                             uint32_t* debug_sfa_out,
                                             uint32_t* debug_sfb_out) {

  constexpr int B = 0; // 3=CU_TENSOR_MAP_SWIZZLE_128B
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int SCALE_VECTOR_SIZE = 16;

  using TypeAcc = float;

  // If swizzle 128, max cp size = 64
  // TODO: Caclulate these values for tile size
  constexpr int TMA_CP_ASYNC_SIZE   = 32; // 32 bytes = 64 elements == MMA_K
  constexpr int TILE_SIZE           = 32; // modify param for larger tile size
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
      
  using TMA_A =
      kernel::tma::tma_2d_nvfp4<float_e2m1,
                          B,
                          M,
                          S,
                          OUTPUT_SIZE,               /*GMEM_ROW_*/
                          REDUCTION_SIZE,            /*GMEM_COL_*/
                          MMA_M,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                          REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                         /*GMEM_STRIDE_COL_*/
                          1,                         /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                          MMA_M * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;

  using TMA_B =
      kernel::tma::tma_2d_nvfp4<float_e2m1,
                          B,
                          M,
                          S,
                          BATCH_SIZE,                /*GMEM_ROW_*/
                          REDUCTION_SIZE,            /*GMEM_COL_*/
                          MMA_N,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                          REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                         /*GMEM_STRIDE_COL_*/
                          1,                         /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                          MMA_N * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  // fake
  // TODO: Caclulate these values for tile size
  // In the kernel wrapper, fix TMA_SFA and TMA_SFB:
  constexpr int SF_TILE_SIZE = TILE_SIZE / SCALE_VECTOR_SIZE; // 64/16 = 4
  constexpr int SF_SMEM_COL  = 16; // minimum for UINT8 TMA (16 bytes)

  using TMA_SFA =
      kernel::tma::tma_2d<float_ue4m3,
                          0, M, S,
                          OUTPUT_SIZE,
                          REDUCTION_SIZE / SCALE_VECTOR_SIZE,
                          MMA_M,                               // weight SF tile rows = MMA_M (not MMA_N)
                          SF_SMEM_COL,
                          REDUCTION_SIZE / SCALE_VECTOR_SIZE,
                          1,
                          1,
                          (SF_TILE_SIZE + SF_SMEM_COL - 1) / SF_SMEM_COL,
                          MMA_M * SF_SMEM_COL,                 // stride updated to match MMA_M rows
                          true>;
 
  using TMA_SFB =
      kernel::tma::tma_2d<float_ue4m3,
                          0, M, S,
                          BATCH_SIZE,
                          REDUCTION_SIZE / SCALE_VECTOR_SIZE,
                          MMA_N,
                          SF_SMEM_COL,
                          REDUCTION_SIZE / SCALE_VECTOR_SIZE,
                          1,
                          1,
                          (SF_TILE_SIZE + SF_SMEM_COL - 1) / SF_SMEM_COL,
                          MMA_N * SF_SMEM_COL,
                          true>;

  using TMA_OUT =
      kernel::tma::tma_2d<float,
                          0,
                          M,
                          S,
                          BATCH_SIZE,             /*GMEM_ROW_*/
                          OUTPUT_SIZE,            /*GMEM_COL_*/
                          MMA_N,                  /*SMEM_ROW_*/
                          MMA_M,                  /*SMEM_COL_*/
                          OUTPUT_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                      /*GMEM_STRIDE_COL_*/
                          1,                      /*SMEM_REPEAT_ROW_*/
                          1,                      /*SMEM_REPEAT_COL_*/
                          MMA_N * MMA_M,          /*SMEM_STRIDE_*/
                          true>;

  TMA_A tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_SFA tma_sfa(static_cast<CUtensorMap *>(tma_sfa_desc_ptr));
  TMA_SFB tma_sfb(static_cast<CUtensorMap *>(tma_sfb_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

  kernel::linear_nvfp4_1d2d_sm100_task_impl<T,
                                            TMA_A,
                                            TMA_B,
                                            TMA_SFA,
                                            TMA_SFB,
                                            BiasTensor,
                                            TMA_OUT,
                                            MMA_M,
                                            MMA_N,
                                            BATCH_SIZE,
                                            OUTPUT_SIZE,
                                            REDUCTION_SIZE,
                                            SCALE_VECTOR_SIZE,
                                            NoBias,
                                            /*SplitK=*/false,
                                            NUM_AB_STAGE,
                                            NUM_ACC_STAGE,
                                            NUM_C_STAGE>(tma_a,
                                                         tma_b,
                                                         tma_sfa,
                                                         tma_sfb,
                                                         mBias,
                                                         tma_out,
                                                         debug_smem_sfa,
                                                         debug_smem_sfb,
                                                         debug_sfa_out,
                                                         debug_sfb_out);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_nvfp4_1d2d_sm100(void *input_ptr,
                                    void *input_sf_ptr,
                                    void *weight_ptr,
                                    void *weight_sf_ptr,
                                    void *output_ptr,
                                    void *residual_ptr = nullptr) {
    //printf("\tlaunch_linear_nvfp4_1d2d_sm100 STARTED\n");
    //fflush(stdout);


  using namespace cute;
  using namespace cutlass;

  constexpr int B = 0;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int SCALE_VECTOR_SIZE = 16;
  using TypeAcc = float;

  constexpr int MMA_M = 128;
  constexpr int MMA_N = 128; // 16
  constexpr int MMA_K = 64;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size
  constexpr int TILE_SIZE =
      64; // we should modify this param if we want larger tile size

  CUtensorMap host_i_desc;
  CUtensorMap host_i_sf_desc;
  CUtensorMap host_w_desc;
  CUtensorMap host_w_sf_desc;
  CUtensorMap host_o_desc;
  CUtensorMap *desc_i_ptr;
  CUtensorMap *desc_i_sf_ptr;
  CUtensorMap *desc_w_ptr;
  CUtensorMap *desc_w_sf_ptr;
  CUtensorMap *desc_o_ptr;

    //printf("\tTMA_INPUT STARTED\n");
//fflush(stdout);


  // TMA_INPUT
  uint64_t i_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                              static_cast<uint64_t>(REDUCTION_SIZE)};
  uint64_t i_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE / 2)};
  uint32_t i_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                              static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  size_t i_smem_repeat_col = (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<float_e2m1, B, M, S, 2>(&host_i_desc,
                                                         static_cast<float_e2m1 *>(input_ptr),
                                                         i_gmem_shape,
                                                         i_gmem_stride,
                                                         i_smem_shape,
                                                         1,
                                                         i_smem_repeat_col);
    printf("\nTMA_INPUT Completed\n");
    //fflush(stdout);

    // TMA_INPUT_SF
    uint64_t i_sf_cols = static_cast<uint64_t>(REDUCTION_SIZE) / SCALE_VECTOR_SIZE;
    uint64_t i_sf_stride = ((i_sf_cols + 15) / 16) * 16; // pad stride to multiple of 16
    uint64_t i_sf_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE), i_sf_cols};
    uint64_t i_sf_gmem_stride[2] = {1, i_sf_stride};
    uint32_t i_sf_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                                    static_cast<uint32_t>(16)}; // min 16 for UINT8 TMA otherwise MMA_K / SCALE_VECTOR_SIZE
    size_t i_sf_smem_repeat_col = (TILE_SIZE / SCALE_VECTOR_SIZE + 16 - 1) / 16;
    mirage::runtime::fill_tma_desc<cute::float_ue4m3_t, 0, M, S, 2>(&host_i_sf_desc,
                                                            static_cast<cute::float_ue4m3_t *>(input_sf_ptr),
                                                            i_sf_gmem_shape,
                                                            i_sf_gmem_stride,
                                                            i_sf_smem_shape,
                                                            1,
                                                            i_sf_smem_repeat_col);
    printf("\nTMA_INPUT_SF Completed\n");
    //fflush(stdout);

    // TMA_WEIGHT
    uint64_t w_gmem_shape[2] = {static_cast<uint64_t>(OUTPUT_SIZE),
                                static_cast<uint64_t>(REDUCTION_SIZE)};
    uint64_t w_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE / 2)};
    uint32_t w_smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                                static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
    size_t w_smem_repeat_col =
        (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
    mirage::runtime::fill_tma_desc<float_e2m1, B, M, S, 2>(&host_w_desc,
                                                            static_cast<float_e2m1 *>(weight_ptr),
                                                            w_gmem_shape,
                                                            w_gmem_stride,
                                                            w_smem_shape,
                                                            1,
                                                            w_smem_repeat_col);

    //printf("\nTMA_WEIGHT Completed\n");
    //fflush(stdout);

    // TMA_WEIGHT_SF
    uint64_t w_sf_cols = static_cast<uint64_t>(REDUCTION_SIZE) / SCALE_VECTOR_SIZE;
    uint64_t w_sf_stride = ((w_sf_cols + 15) / 16) * 16;
    uint64_t w_sf_gmem_shape[2] = {static_cast<uint64_t>(OUTPUT_SIZE), w_sf_cols};
    uint64_t w_sf_gmem_stride[2] = {1, w_sf_stride};
    uint32_t w_sf_smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                                    static_cast<uint32_t>(16)}; // MMA_K / SCALE_VECTOR_SIZE
    size_t w_sf_smem_repeat_col = (TILE_SIZE / SCALE_VECTOR_SIZE + 16 - 1) / 16;
    mirage::runtime::fill_tma_desc<cute::float_ue4m3_t, 0, M, S, 2>(&host_w_sf_desc,
                                                            static_cast<cute::float_ue4m3_t *>(weight_sf_ptr),
                                                            w_sf_gmem_shape,
                                                            w_sf_gmem_stride,
                                                            w_sf_smem_shape,
                                                            1,
                                                            w_sf_smem_repeat_col);

    //printf("\nTMA_WEIGHT_SF Completed\n");
    //fflush(stdout);

//   //printf("o_gmem_shape: %d, %d\n", o_gmem_shape[0], o_gmem_shape[1]);
//   //printf("o_gmem_stride: %d, %d\n", o_gmem_stride[0], o_gmem_stride[1]);
//   //printf("o_smem_shape: %d, %d\n", o_smem_shape[0], o_smem_shape[1]);
//   //printf("o_smem_repeat_col: %zu\n", o_smem_repeat_col);

  // TMA_OUT
  int const output_stride = OUTPUT_SIZE;
  uint64_t o_gmem_shape[2]  = {static_cast<uint64_t>(BATCH_SIZE),
                               static_cast<uint64_t>(OUTPUT_SIZE)};
  uint64_t o_gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
  uint32_t o_smem_shape[2]  = {static_cast<uint32_t>(MMA_N),
                               static_cast<uint32_t>(MMA_M)};
  size_t o_smem_repeat_col = 1;

  mirage::runtime::fill_tma_desc<float, 0, M, S, 2>(&host_o_desc,
                                                       static_cast<float *>(output_ptr),
                                                       o_gmem_shape,
                                                       o_gmem_stride,
                                                       o_smem_shape,
                                                       1,
                                                       o_smem_repeat_col);

    //printf("\ncudaMalloc start\n");
    //fflush(stdout);
  cudaMalloc(&desc_i_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_i_sf_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_w_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_w_sf_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_o_ptr, sizeof(CUtensorMap));

    //printf("\ncudaMemcpy start\n");
        //fflush(stdout);
  cudaMemcpy(desc_i_ptr, &host_i_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_i_sf_ptr, &host_i_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_w_ptr, &host_w_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_w_sf_ptr, &host_w_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_o_ptr, &host_o_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  void *tma_desc_input;
  void *tma_desc_input_sf;
  void *tma_desc_weight;
  void *tma_desc_weight_sf;
  void *tma_desc_output;
//   void *tma_desc_sfa;
//   void *tma_desc_sfb;

  tma_desc_input = desc_i_ptr;
  tma_desc_input_sf = desc_i_sf_ptr;
  tma_desc_weight = desc_w_ptr;
  tma_desc_weight_sf = desc_w_sf_ptr;
  tma_desc_output = desc_o_ptr;
//   tma_desc_sfa = tma_desc_output;
//   tma_desc_sfb = tma_desc_output;

  // Residual
  cute::Layout layout_Bias = cute::make_layout(
      cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
      cute::make_stride(OUTPUT_SIZE,
                        cute::Int<1>{})); // (Gemm_M,Gemm_N):(Gemm_N,_1)
  cute::Tensor mBias =
      cute::make_tensor(cute::make_gmem_ptr(static_cast<float *>(residual_ptr)),
                        layout_Bias); // (Gemm_N, Gemm_M)

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  // NUM_C_STAGE=1 keeps C-buffer = MMA_N*MMA_M*sizeof(float)*1 stage.
  // With MMA_N=128, MMA_M=128: C=64KB, A+B+SFA+SFB≈72KB → ~138KB total,
  // well within the 224KB limit. NUM_C_STAGE=4 would make C alone 256KB.
  constexpr int NUM_C_STAGE_LAUNCH = 1;
  int smemBytes = 224 * 1024;

  // Debug buffers for SMEM→TMEM verification
  constexpr int kUTCCPSFCols_host = 16;
  constexpr int kNumSFKCols_host  = MMA_K / SCALE_VECTOR_SIZE;  // 4
  constexpr int kSFASmemElems     = MMA_M * kNumSFKCols_host / 4;  // 128
  constexpr int kSFBSmemElems     = 128   * kNumSFKCols_host / 4;  // 128
  constexpr int kTmemSFElems      = kUTCCPSFCols_host * 32;        // 512
  uint32_t *debug_smem_sfa, *debug_smem_sfb, *debug_sfa_out, *debug_sfb_out;
  cudaMalloc(&debug_smem_sfa, kSFASmemElems * sizeof(uint32_t));
  cudaMalloc(&debug_smem_sfb, kSFBSmemElems * sizeof(uint32_t));
  cudaMalloc(&debug_sfa_out,  kTmemSFElems  * sizeof(uint32_t));
  cudaMalloc(&debug_sfb_out,  kTmemSFElems  * sizeof(uint32_t));

  if (residual_ptr != nullptr) {
    auto *kernel_ptr = &linear_nvfp4_1d2d_sm100_wrapper<T,
                                                        BATCH_SIZE,
                                                        OUTPUT_SIZE,
                                                        REDUCTION_SIZE,
                                                        decltype(mBias),
                                                        MMA_M,
                                                        MMA_N,
                                                        false,
                                                        /*NUM_AB_STAGE=*/8,
                                                        /*NUM_ACC_STAGE=*/2,
                                                        NUM_C_STAGE_LAUNCH>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    cutlass::ClusterLaunchParams params = {
        grid_dim, block_dim, cluster_dim, smemBytes};
    //printf("\nLaunching kernel...\n"); //fflush(stdout);
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(params,
                                          (void const *)kernel_ptr,
                                          tma_desc_weight,
                                          tma_desc_input,
                                          tma_desc_weight_sf,
                                          tma_desc_input_sf,
                                          mBias,
                                          tma_desc_output,
                                          debug_smem_sfa,
                                          debug_smem_sfb,
                                          debug_sfa_out,
                                          debug_sfb_out);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
  } else {
    auto *kernel_ptr = &linear_nvfp4_1d2d_sm100_wrapper<T,
                                                        BATCH_SIZE,
                                                        OUTPUT_SIZE,
                                                        REDUCTION_SIZE,
                                                        decltype(mBias),
                                                        MMA_M,
                                                        MMA_N,
                                                        true,
                                                        /*NUM_AB_STAGE=*/8,
                                                        /*NUM_ACC_STAGE=*/2,
                                                        NUM_C_STAGE_LAUNCH>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    cutlass::ClusterLaunchParams params = {
        grid_dim, block_dim, cluster_dim, smemBytes};
    printf("\nLaunching kernel...\n");     //fflush(stdout);
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(params,
                                          (void const *)kernel_ptr,
                                          tma_desc_weight,
                                          tma_desc_input,
                                          tma_desc_weight_sf,
                                          tma_desc_input_sf,
                                          mBias,
                                          tma_desc_output,
                                          debug_smem_sfa,
                                          debug_smem_sfb,
                                          debug_sfa_out,
                                          debug_sfb_out);
    printf("\nKernel launched...\n");     //fflush(stdout);

    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
  }

  // ── Host-side SMEM→TMEM verification ─────────────────────────────────────
  cudaDeviceSynchronize();

  std::vector<uint32_t> h_smem_sfa(kSFASmemElems);
  std::vector<uint32_t> h_smem_sfb(kSFBSmemElems);
  std::vector<uint32_t> h_tmem_sfa(kTmemSFElems);
  std::vector<uint32_t> h_tmem_sfb(kTmemSFElems);

  cudaMemcpy(h_smem_sfa.data(), debug_smem_sfa, kSFASmemElems * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_smem_sfb.data(), debug_smem_sfb, kSFBSmemElems * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_tmem_sfa.data(), debug_sfa_out,  kTmemSFElems  * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_tmem_sfb.data(), debug_sfb_out,  kTmemSFElems  * 4, cudaMemcpyDeviceToHost);

  // Print SMEM raw bytes for SFA (first 32 rows × 4 cols = 128 bytes)
  printf("\n=== SMEM SFA (first 8 uint32 words = 32 bytes) ===\n");
  for (int i = 0; i < 8 && i < kSFASmemElems; ++i)
      printf("  smem_sfa[%d] = 0x%08x\n", i, h_smem_sfa[i]);
  printf("=== SMEM SFB (first 8 uint32 words = 32 bytes) ===\n");
  for (int i = 0; i < 8 && i < kSFBSmemElems; ++i)
      printf("  smem_sfb[%d] = 0x%08x\n", i, h_smem_sfb[i]);

  // SMEM layout: sfa_smem_base is MMA_M*num_sf_k_cols bytes (128*4=512 bytes),
  // stored as 128 uint32 words. Word [row] holds k-cols 0..3 packed as bytes.
  // UTCCP copies the full 512-byte block identically into each group of 4 TMEM cols.
  // TMEM readback: h_tmem_sfa[col*32 + lane] = uint32 at TMEM col 'col', DP 'lane'.
  // Expected: TMEM[col][dp] bytes == SMEM[dp][col%4] bytes
  // i.e., h_tmem_sfa[col*32+lane] == h_smem_sfa[lane] with matching byte position col%4.
  // Since tcgen05.ld reads 32 bits and SF are 1-byte, we compare byte-by-byte.

  // Print first few TMEM SFA values
  printf("\n=== TMEM SFA (first 16 cols, lanes 0-3) ===\n");
  for (int col = 0; col < 16; ++col)
    printf("  tmem_sfa[col=%d, lane=0] = 0x%08x\n", col, h_tmem_sfa[col * 32 + 0]);

  printf("\n=== TMEM SFB (first 16 cols, lanes 0-3) ===\n");
  for (int col = 0; col < 16; ++col)
    printf("  tmem_sfb[col=%d, lane=0] = 0x%08x\n", col, h_tmem_sfb[col * 32 + 0]);

  // Compare SFA: all 4 UTCCP iterations write the same SMEM block → col%4 maps to smem k-col
  // h_smem_sfa[lane] = packed uint32 of 4 SF bytes for SMEM row 'lane' (k-cols 0..3)
  // h_tmem_sfa[col*32+lane] should equal h_smem_sfa[lane] (full word, since TMEM stores 4 bytes per DP per 4-col group)
  // bool sfa_ok = true;
  // for (int col_group = 0; col_group < kUTCCPSFCols_host / 4; ++col_group) {
  //   for (int lane = 0; lane < 32; ++lane) {
  //     // The 4 TMEM cols in this group correspond to 4 bytes of smem word
  //     uint32_t smem_word = h_smem_sfa[lane];  // bytes k0,k1,k2,k3 for SMEM row=lane
  //     for (int c = 0; c < 4; ++c) {
  //       int col = col_group * 4 + c;
  //       uint32_t tmem_val = h_tmem_sfa[col * 32 + lane];
  //       uint8_t smem_byte = (smem_word >> (8 * c)) & 0xFF;
  //       uint8_t tmem_byte = tmem_val & 0xFF;  // low byte of TMEM word
  //       if (smem_byte != tmem_byte) {
  //         printf("SFA MISMATCH col=%d lane=%d : smem_byte=0x%02x tmem_byte=0x%02x (smem_word=0x%08x tmem_word=0x%08x)\n",
  //                col, lane, smem_byte, tmem_byte, smem_word, tmem_val);
  //         sfa_ok = false;
  //       }
  //     }
  //   }
  // }
  // if (sfa_ok) printf("SFA: all %d byte comparisons match\n", kUTCCPSFCols_host * 32);

  // Compare SFB
  // bool sfb_ok = true;
  // for (int col_group = 0; col_group < kUTCCPSFCols_host / 4; ++col_group) {
  //   for (int lane = 0; lane < 32; ++lane) {
  //     uint32_t smem_word = h_smem_sfb[lane];
  //     for (int c = 0; c < 4; ++c) {
  //       int col = col_group * 4 + c;
  //       uint32_t tmem_val = h_tmem_sfb[col * 32 + lane];
  //       uint8_t smem_byte = (smem_word >> (8 * c)) & 0xFF;
  //       uint8_t tmem_byte = tmem_val & 0xFF;
  //       // if (smem_byte != tmem_byte) {
  //         printf("SFB MISMATCH col=%d lane=%d : smem_byte=0x%02x tmem_byte=0x%02x (smem_word=0x%08x tmem_word=0x%08x)\n",
  //                col, lane, smem_byte, tmem_byte, smem_word, tmem_val);
  //         // sfb_ok = false;
  //       // }
  //     }
  //   }
  // }
  // if (sfb_ok) printf("SFB: all %d byte comparisons match\n", kUTCCPSFCols_host * 32);

  cudaFree(debug_smem_sfa);
  cudaFree(debug_smem_sfb);
  cudaFree(debug_sfa_out);
  cudaFree(debug_sfb_out);
}


void linear_nvfp4_1d2d_sm100_kernel(torch::Tensor input,
                                    torch::Tensor input_sf,
                                    torch::Tensor weight,
                                    torch::Tensor weight_sf,
                                    c10::optional<at::Tensor> residual,
                                    torch::Tensor output) {

  void *input_ptr = input.data_ptr();
  void *input_sf_ptr = input_sf.data_ptr();
  void *weight_ptr = weight.data_ptr();
  void *weight_sf_ptr = weight_sf.data_ptr();
  bool has_residual = residual.has_value();
  void *residual_ptr = has_residual ? residual->data_ptr() : nullptr;
  void *output_ptr = output.data_ptr();

  constexpr int BATCH_SIZE = 1;
  constexpr int OUTPUT_SIZE = 128;
  constexpr int REDUCTION_SIZE = 768;

  // Divide by 2 for packed representation of fp4 in 8 bits
  assert(input.size(1) == REDUCTION_SIZE / 2); 
  assert(weight.size(0) == OUTPUT_SIZE);

  cudaDeviceSetLimit(cudaLimitStackSize, 4096);  // or 8192

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  // printf("Free: %zu MB, Total: %zu MB\n", free_mem/1024/1024, total_mem/1024/1024);

  // cudaError_t err2 = cudaDeviceSetLimit(cudaLimitStackSize, 8192);
  // if (err2 != cudaSuccess) {
  //     printf("SetLimit failed: %s\n", cudaGetErrorString(err2));
  // }

  cudaMemGetInfo(&free_mem, &total_mem);
  // printf("Free after: %zu MB\n", free_mem/1024/1024);

  //printf("\tLaunching launch_linear_nvfp4_1d2d_sm100\n");
  launch_linear_nvfp4_1d2d_sm100<float_e2m1, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
      input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr
  );
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    //printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "linear_nvfp4_1d2d_sm100", &linear_nvfp4_1d2d_sm100_kernel, "Linear kernel SM100 nvfp4 1D2D");
}
