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

#include <iostream>

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
#include <cute/numeric/numeric_types.hpp>
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
                                             void *tma_out_desc_ptr) {
  constexpr int MMA_K = 64;
  constexpr int NUM_MMA_K = 4;
  constexpr int bK = MMA_K * NUM_MMA_K;  // = 256, full K reduction per stage

  constexpr int B_FP4 = 1; // 32-byte swizzle to match Layout_K_SW32_Atom
  constexpr int B_SF  = 0; // no swizzle for scale factors
  constexpr int B_OUT = 0; // no swizzle for output
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int SCALE_VECTOR_SIZE = 16;

  // Max TMA box = 64 FP4 elements along K; repeat 4× to cover bK=256
  constexpr int TMA_CP_ASYNC_SIZE    = 64;
  constexpr int TMA_CP_ASYNC_REPEAT_COL = bK / TMA_CP_ASYNC_SIZE;  // = 4

  // TMA_A = input (activation), MMA_M rows
  // SMEM_STRIDE_ is in sizeof(float_e2m1_t)=1 byte units; each K-block is MMA_M*TMA_CP_ASYNC_SIZE/2 bytes (FP4 packed)
  constexpr int FP4_SMEM_STRIDE_A = MMA_M * TMA_CP_ASYNC_SIZE / 2;
  using TMA_A =
      kernel::tma::tma_2d_nvfp4<cute::float_e2m1_t,
                                B_FP4, M, S,
                                BATCH_SIZE,                /*GMEM_ROW_*/
                                REDUCTION_SIZE,            /*GMEM_COL_*/
                                MMA_M,                     /*SMEM_ROW_*/
                                TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                                REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                                1,                         /*GMEM_STRIDE_COL_*/
                                1,                         /*SMEM_REPEAT_ROW_*/
                                TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_ = 4*/
                                FP4_SMEM_STRIDE_A,         /*SMEM_STRIDE_ = 4096 bytes*/
                                true>;

  // TMA_B = weight, MMA_N rows
  constexpr int FP4_SMEM_STRIDE_B = MMA_N * TMA_CP_ASYNC_SIZE / 2;
  using TMA_B =
      kernel::tma::tma_2d_nvfp4<cute::float_e2m1_t,
                                B_FP4, M, S,
                                OUTPUT_SIZE,               /*GMEM_ROW_*/
                                REDUCTION_SIZE,            /*GMEM_COL_*/
                                MMA_N,                     /*SMEM_ROW_*/
                                TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                                REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                                1,                         /*GMEM_STRIDE_COL_*/
                                1,                         /*SMEM_REPEAT_ROW_*/
                                TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_ = 4*/
                                FP4_SMEM_STRIDE_B,         /*SMEM_STRIDE_ = 4096 bytes*/
                                true>;

  // SF TMA: load all NUM_MMA_K=4 K-blocks per stage via SMEM_REPEAT_ROW=4
  constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;  // = 16 half_t per K-block
  constexpr int SF_COL_B = MMA_N * MMA_K / SCALE_VECTOR_SIZE / 2;  // = 16 half_t per K-block

  // TMA_SFA = input SF, MMA_M-side
  using TMA_SFA = kernel::tma::tma_3d<cute::half_t,
                          B_SF, M, S,
                          BATCH_SIZE / MMA_M,                              /*GMEM_DEPTH_ = 1*/
                          REDUCTION_SIZE / MMA_K,                          /*GMEM_ROW_ = 4*/
                          SF_COL_A,                                        /*GMEM_COL_ = 16*/
                          1,                                               /*SMEM_DEPTH_*/
                          1,                                               /*SMEM_ROW_*/
                          SF_COL_A,                                        /*SMEM_COL_ = 16*/
                          SF_COL_A * (REDUCTION_SIZE / MMA_K),             /*GMEM_STRIDE_DEPTH_*/
                          SF_COL_A,                                        /*GMEM_STRIDE_ROW_*/
                          1,                                               /*GMEM_STRIDE_COL_*/
                          NUM_MMA_K,                                       /*SMEM_REPEAT_ROW_ = 4*/
                          1,                                               /*SMEM_REPEAT_COL_*/
                          SF_COL_A,                                        /*SMEM_STRIDE_ = 16*/
                          true>;

  // TMA_SFB = weight SF, MMA_N-side
  using TMA_SFB = kernel::tma::tma_3d<cute::half_t,
                          B_SF, M, S,
                          OUTPUT_SIZE / MMA_N,                             /*GMEM_DEPTH_ = 1*/
                          REDUCTION_SIZE / MMA_K,                          /*GMEM_ROW_ = 4*/
                          SF_COL_B,                                        /*GMEM_COL_ = 16*/
                          1,                                               /*SMEM_DEPTH_*/
                          1,                                               /*SMEM_ROW_*/
                          SF_COL_B,                                        /*SMEM_COL_ = 16*/
                          SF_COL_B * (REDUCTION_SIZE / MMA_K),             /*GMEM_STRIDE_DEPTH_*/
                          SF_COL_B,                                        /*GMEM_STRIDE_ROW_*/
                          1,                                               /*GMEM_STRIDE_COL_*/
                          NUM_MMA_K,                                       /*SMEM_REPEAT_ROW_ = 4*/
                          1,                                               /*SMEM_REPEAT_COL_*/
                          SF_COL_B,                                        /*SMEM_STRIDE_ = 16*/
                          true>;

  using TMA_OUT =
      kernel::tma::tma_2d<float,
                          B_OUT,
                          M,
                          S,
                          BATCH_SIZE,             /*GMEM_ROW_*/
                          OUTPUT_SIZE,            /*GMEM_COL_*/
                          MMA_M,                  /*SMEM_ROW_*/
                          MMA_N,                  /*SMEM_COL_*/
                          OUTPUT_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                      /*GMEM_STRIDE_COL_*/
                          1,                      /*SMEM_REPEAT_ROW_*/
                          1,                      /*SMEM_REPEAT_COL_*/
                          MMA_M * MMA_N,          /*SMEM_STRIDE_*/
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
                                                         tma_out);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_nvfp4_1d2d_sm100(void *input_ptr,
                                    void *input_sf_ptr,
                                    void *weight_ptr,
                                    void *weight_sf_ptr,
                                    void *output_ptr,
                                    void *residual_ptr = nullptr) {

  using namespace cute;
  using namespace cutlass;

  constexpr int B_FP4 = 1;
  constexpr int B_SF  = 0;
  constexpr int B_OUT = 0;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int SCALE_VECTOR_SIZE = 16;

  constexpr int MMA_M = 128;
  constexpr int MMA_N = 128;
  constexpr int MMA_K = 64;

  constexpr int NUM_MMA_K = 4;
  constexpr int bK = MMA_K * NUM_MMA_K;  // = 256
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TMA_CP_ASYNC_REPEAT_COL = bK / TMA_CP_ASYNC_SIZE;  // = 4

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

  // TMA_A = input
  uint64_t i_gmem_shape[2] = {
    static_cast<uint64_t>(BATCH_SIZE),
    static_cast<uint64_t>(REDUCTION_SIZE)
  };
  uint64_t i_gmem_stride[2] = {
    1,
    static_cast<uint64_t>(REDUCTION_SIZE)
  };
  uint32_t i_smem_shape[2] = {
    static_cast<uint32_t>(MMA_M),
    static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)
  };
  mirage::runtime::fill_tma_desc<cute::float_e2m1_t, B_FP4, M, S, 2>(
    &host_i_desc,
    static_cast<cute::float_e2m1_t *>(input_ptr),
    i_gmem_shape, i_gmem_stride, i_smem_shape,
    1, TMA_CP_ASYNC_REPEAT_COL);

  // TMA_SFA = input SF
  constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;  // = 16 half_t per K-block
  uint64_t i_sf_gmem_shape[3] = {
    static_cast<uint64_t>(BATCH_SIZE / MMA_M),
    static_cast<uint64_t>(REDUCTION_SIZE / MMA_K),
    static_cast<uint64_t>(SF_COL_A)
  };
  uint64_t i_sf_gmem_stride[3] = {
    1,
    static_cast<uint64_t>(SF_COL_A),
    static_cast<uint64_t>(SF_COL_A * (REDUCTION_SIZE / MMA_K))
  };
  uint32_t i_sf_smem_shape[3] = {
    static_cast<uint32_t>(1),
    static_cast<uint32_t>(1),
    static_cast<uint32_t>(SF_COL_A)
  };
  mirage::runtime::fill_tma_desc<cute::half_t, B_SF, M, S, 3>(
    &host_i_sf_desc,
    static_cast<cute::half_t *>(input_sf_ptr),
    i_sf_gmem_shape, i_sf_gmem_stride, i_sf_smem_shape,
    1, 1);

  // TMA_B = weight, MMA_N rows, 4 K-repeats
  uint64_t w_gmem_shape[2] = {
    static_cast<uint64_t>(OUTPUT_SIZE),
    static_cast<uint64_t>(REDUCTION_SIZE)
  };
  uint64_t w_gmem_stride[2] = {
    1,
    static_cast<uint64_t>(REDUCTION_SIZE)
  };
  uint32_t w_smem_shape[2] = {
    static_cast<uint32_t>(MMA_N),
    static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)
  };
  mirage::runtime::fill_tma_desc<cute::float_e2m1_t, B_FP4, M, S, 2>(
    &host_w_desc,
    static_cast<cute::float_e2m1_t *>(weight_ptr),
    w_gmem_shape, w_gmem_stride, w_smem_shape,
    1, TMA_CP_ASYNC_REPEAT_COL);

  // TMA_SFB = weight SF
  constexpr int SF_COL_B = MMA_N * MMA_K / SCALE_VECTOR_SIZE / 2;  // = 16 half_t per K-block
  uint64_t w_sf_gmem_shape[3] = {
    static_cast<uint64_t>(OUTPUT_SIZE / MMA_N),
    static_cast<uint64_t>(REDUCTION_SIZE / MMA_K),
    static_cast<uint64_t>(SF_COL_B)
  };
  uint64_t w_sf_gmem_stride[3] = {
    1,
    static_cast<uint64_t>(SF_COL_B),
    static_cast<uint64_t>(SF_COL_B * (REDUCTION_SIZE / MMA_K))
  };
  uint32_t w_sf_smem_shape[3] = {
    static_cast<uint32_t>(1),
    static_cast<uint32_t>(1),
    static_cast<uint32_t>(SF_COL_B)
  };
  mirage::runtime::fill_tma_desc<cute::half_t, B_SF, M, S, 3>(
    &host_w_sf_desc,
    static_cast<cute::half_t *>(weight_sf_ptr),
    w_sf_gmem_shape, w_sf_gmem_stride, w_sf_smem_shape,
    1, 1);

  // TMA_OUT
  uint64_t o_gmem_shape[2]  = {
    static_cast<uint64_t>(BATCH_SIZE),
    static_cast<uint64_t>(OUTPUT_SIZE)
  };
  uint64_t o_gmem_stride[2] = {
    1, 
    static_cast<uint64_t>(OUTPUT_SIZE)
  };
  uint32_t o_smem_shape[2]  = {
    static_cast<uint32_t>(MMA_M),
    static_cast<uint32_t>(MMA_N)
  };
  size_t o_smem_repeat_col = 1;
  mirage::runtime::fill_tma_desc<float, B_OUT, M, S, 2>(&host_o_desc,
                                                       static_cast<float *>(output_ptr),
                                                       o_gmem_shape,
                                                       o_gmem_stride,
                                                       o_smem_shape,
                                                       1,
                                                       o_smem_repeat_col);

  cudaMalloc(&desc_i_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_i_sf_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_w_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_w_sf_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_o_ptr, sizeof(CUtensorMap));

  cudaMemcpy(desc_i_ptr, &host_i_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_i_sf_ptr, &host_i_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_w_ptr, &host_w_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_w_sf_ptr, &host_w_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_o_ptr, &host_o_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice); 

  void *tma_desc_input      = desc_i_ptr;
  void *tma_desc_input_sf   = desc_i_sf_ptr;
  void *tma_desc_weight     = desc_w_ptr;
  void *tma_desc_weight_sf  = desc_w_sf_ptr;
  void *tma_desc_output     = desc_o_ptr;

  // Residual
  cute::Layout layout_Bias = cute::make_layout(
      cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
      cute::make_stride(OUTPUT_SIZE, cute::Int<1>{})
    ); // (Gemm_M,Gemm_N):(Gemm_N,_1)
  cute::Tensor mBias =
      cute::make_tensor(cute::make_gmem_ptr(static_cast<float *>(residual_ptr)),
                        layout_Bias); // (Gemm_N, Gemm_M)

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  constexpr int NUM_C_STAGE_LAUNCH = 1;
  int smemBytes = 224 * 1024;

  if (residual_ptr != nullptr) {
    auto *kernel_ptr = &linear_nvfp4_1d2d_sm100_wrapper<T,
                                                        BATCH_SIZE,
                                                        OUTPUT_SIZE,
                                                        REDUCTION_SIZE,
                                                        decltype(mBias),
                                                        MMA_M,
                                                        MMA_N,
                                                        false,
                                                        /*NUM_AB_STAGE=*/2,
                                                        /*NUM_ACC_STAGE=*/2,
                                                        NUM_C_STAGE_LAUNCH>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(params,
                                          (void const *)kernel_ptr,
                                          tma_desc_input,
                                          tma_desc_weight,
                                          tma_desc_input_sf,
                                          tma_desc_weight_sf,
                                          mBias,
                                          tma_desc_output);
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
                                                        /*NUM_AB_STAGE=*/2,
                                                        /*NUM_ACC_STAGE=*/2,
                                                        NUM_C_STAGE_LAUNCH>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    cutlass::ClusterLaunchParams params = {
        grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(params,
                                          (void const *)kernel_ptr,
                                          tma_desc_input,
                                          tma_desc_weight,
                                          tma_desc_input_sf,
                                          tma_desc_weight_sf,
                                          mBias,
                                          tma_desc_output);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
  }
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

  constexpr int BATCH_SIZE = 256*32;
  constexpr int OUTPUT_SIZE = 128*2;
  constexpr int REDUCTION_SIZE = 128*7;

  assert(input.size(1)  == REDUCTION_SIZE / 2); 
  assert(weight.size(0) == OUTPUT_SIZE);

  cudaDeviceSetLimit(cudaLimitStackSize, 4096);  // or 8192
  launch_linear_nvfp4_1d2d_sm100<cute::float_e2m1_t, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
      input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr, residual_ptr
  );
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_nvfp4_1d2d_sm100", &linear_nvfp4_1d2d_sm100_kernel, "Linear kernel SM100 nvfp4 1D2D");
}
