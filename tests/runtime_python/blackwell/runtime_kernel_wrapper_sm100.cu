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
#include "blackwell/linear_sm100_warp_specialized.cuh"
#include "blackwell/linear_sm100_mpk.cuh"
#include "blackwell/utils.cuh"
#include "hopper/tma_2d.cuh"
#include "tma.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <cstdio>

// Use Thrust to handle host/device allocations
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Cutlass includes
#include <cutlass/half.h>                       // F16 data type
#include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

// CuTe includes
#include <cute/tensor.hpp>                      // CuTe tensor implementation
#include <cute/arch/cluster_sm90.hpp>           // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp>   // Compile time in constants such as _1, _256 etc.
#include <cute/algorithm/cooperative_copy.hpp>  // Auto vectorized copy operation
#include <cute/arch/tmem_allocator_sm100.hpp>   // TMEM allocator for SM100
#include <cute/pointer_flagged.hpp> 

using bfloat16 = cute::bfloat16_t;

// Linear SM100 Warp Specialized

template <class SharedStorage,
          class ATensor, class BTensor, class BiasTensor, class CTensor,
          class MmaTiler_MNK, class EpiTiler_MN, class TiledMMA, 
          class TmaAtomA, class TmaAtomB, class TmaAtomC, 
          int Num_AB_Stage, int Num_ACC_Stage, int Num_C_Stage, bool NoBias>
__global__ __launch_bounds__(256, 1) void linear_kernel_ws_sm100_wrapper(
            ATensor mA,                      // (Gemm_M, Gemm_K)
            BTensor mB,                      // (Gemm_N, Gemm_K)
            BiasTensor mBias,                      // (Gemm_M, Gemm_N)
            CTensor mC,                     // (Gemm_M, Gemm_N)
            MmaTiler_MNK mma_tiler,          // <MmaTile_M, MmaTile_N, MmaTile_K>
            EpiTiler_MN epi_tiler,           // <EpiTile_M, EpiTile_N>
            TiledMMA tiled_mma,              // <    Mma_M,     Mma_N,     Mma_K>
            CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
            CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
            CUTE_GRID_CONSTANT TmaAtomC const tma_atom_C,
            const int num_tmem_columns
) {

  kernel::linear_kernel_ws_sm100<SharedStorage, ATensor, BTensor, BiasTensor, CTensor,
                      MmaTiler_MNK, EpiTiler_MN, TiledMMA,
                      TmaAtomA, TmaAtomB, TmaAtomC, Num_AB_Stage, Num_ACC_Stage, Num_C_Stage, NoBias>(
                      mA, mB, mBias, mC,
                      mma_tiler, epi_tiler, tiled_mma,
                      &tma_atom_A, &tma_atom_B, &tma_atom_C, num_tmem_columns);

}


template <class TypeA, class TypeB, class TypeBias, class TypeC, class LayoutA, class LayoutB, class LayoutBias, class LayoutC>
void launch_linear_sm100_warp_specialized(TypeA const* device_ptr_input,
                                          TypeB const* device_ptr_weight,
                                          TypeBias const* device_ptr_bias,
                                          TypeC      * device_ptr_output,
                                          LayoutA layout_A,
                                          LayoutB layout_B,
                                          LayoutBias layout_Bias,
                                          LayoutC layout_C) {
  // Ensure the input/output tensor dimensions match for GEMM
  assert(cute::shape<0>(layout_A) == cute::shape<1>(layout_C));  // Gemm_M
  assert(cute::shape<0>(layout_B) == cute::shape<0>(layout_C));  // Gemm_N
  assert(cute::shape<1>(layout_A) == cute::shape<1>(layout_B));  // Gemm_K

  if (device_ptr_bias != nullptr) {
    assert(cute::shape<0>(layout_A) == cute::shape<1>(layout_Bias));  // Gemm_M
    assert(cute::shape<0>(layout_B) == cute::shape<0>(layout_Bias));  // Gemm_N
  }

  using TypeAcc = float;
  // SwapAB configuration
  const int mma_m = 128;
  const int mma_n = 32;

  const int num_ab_stage = 8;
  const int num_c_stage = 4;
  const int num_acc_stage = 2;

  // Represent the full tensors in global memory
  cute::Tensor mA = cute::make_tensor(cute::make_gmem_ptr(device_ptr_weight), layout_A);      // (Gemm_M, Gemm_K)
  cute::Tensor mB = cute::make_tensor(cute::make_gmem_ptr(device_ptr_input), layout_B);      // (Gemm_N, Gemm_K)
  cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(device_ptr_bias), layout_Bias);      // (Gemm_N, Gemm_M)
  cute::Tensor mC = cute::make_tensor(cute::make_gmem_ptr(device_ptr_output), layout_C);      // (Gemm_N, Gemm_M)

  // // TODO(Zhihao): add dispatch for residual

  // Get M, N, K dimensions of the GEMM we are running
  auto Gemm_M = cute::shape<0>(layout_A);
  auto Gemm_N = cute::shape<0>(layout_B);
  auto Gemm_K = cute::shape<1>(layout_A);

  cute::TiledMMA tiled_mma = cute::make_tiled_mma(cute::SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeAcc,           // Mma's A, B, and Accumulator types
                                                                 mma_m, mma_n,                            // Mma M and N dimensions
                                                                 cute::UMMA::Major::K, cute::UMMA::Major::K>{});  // A and B layouts

  // cute::print(tiled_mma);
  // Define MMA tiler sizes (static)
  auto bM = cute::tile_size<0>(tiled_mma);             // MMA Tile M. We'll use 1 MMAs per MMA Tile M.
  auto bN = cute::tile_size<1>(tiled_mma);             // MMA Tile N. We'll use 1 MMAs per MMA Tile N.
  auto bK = cute::tile_size<2>(tiled_mma) * cute::Int<4>{};  // MMA Tile K. We'll use 4 MMAs per MMA Tile K. For 16b types, tcgen05.mma has K16.
  auto mma_tiler = cute::make_shape(bM, bN, bK);       // (MMA_M, MMA_N, MMA_K)

  const int num_tmem_columns = bN * num_acc_stage;
  assert(num_tmem_columns <= 512);

  // Pre-partitioned Tile Shape (MmaTile_M, MmaTile_K) to post-partitioned (MmaA, NumMma_M, NumMma_K)
  auto mma_shape_A = cute::partition_shape_A(tiled_mma, cute::make_shape(cute::size<0>(mma_tiler), cute::size<2>(mma_tiler), cute::Int<num_ab_stage>{}));
  // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_K) to post-partitioned (MmaB, NumMma_N, NumMma_K)
  auto mma_shape_B = cute::partition_shape_B(tiled_mma, cute::make_shape(cute::size<1>(mma_tiler), cute::size<2>(mma_tiler), cute::Int<num_ab_stage>{}));
  // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_M) to post-partitioned (MmaC, NumMma_N, NumMma_K)
  auto mma_shape_C = cute::make_shape(cute::make_shape(cute::size<1>(mma_tiler), cute::size<0>(mma_tiler)), cute::Int<1>{}, cute::Int<1>{}, cute::Int<num_c_stage>{});

  // Print and inspect mma_shape_A, and mma_shape_B for this example.
  // cute::print("mma_shape_A:\t"); cute::print(mma_shape_A); cute::print("\n");  // mma_shape_A:  ((_128,_16),_1,_4,_8)
  // cute::print("mma_shape_B:\t"); cute::print(mma_shape_B); cute::print("\n");  // mma_shape_B:  ((_32,_16),_1,_4,_8)
  // cute::print("mma_shape_C:\t"); cute::print(mma_shape_C); cute::print("\n");  // mma_shape_C:  ((_32,_128),_1,_1,_4)

  auto epi_tiler = cute::make_tile(cute::size<0,0>(mma_shape_C), cute::size<0,1>(mma_shape_C));

  auto sA_layout = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);
  // constuct unswizzled layout for C
  auto sC_layout_fake = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_INTER_Atom<TypeC>{}, mma_shape_C);
  auto sC_shape = cute::make_shape(cute::make_shape(cute::Int<mma_n>{}, cute::Int<mma_m>{}), cute::Int<1>{}, cute::Int<1>{}, cute::make_shape(cute::Int<1>{}, cute::Int<num_c_stage>{}));
  auto sC_stride = cute::make_stride(cute::make_stride(cute::Int<mma_m>{}, cute::Int<1>{}), cute::Int<0>{}, cute::Int<0>{}, cute::make_stride(cute::Int<0>{}, cute::Int<mma_m * mma_n>{}));
  auto sC_layout = cute::composition(sC_layout_fake.layout_a(), sC_layout_fake.offset(), cute::make_layout(sC_shape, sC_stride));

  // Print and inspect sA_layout and sB_layout for this example.
  // cute::print("sA_layout:\t"); cute::print(sA_layout); cute::print("\n");      // sA_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
  // cute::print("sB_layout:\t"); cute::print(sB_layout); cute::print("\n");      // sB_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_256,_16),_1,_4):((_64,_1),_0,_16)
  // cute::print("sC_layout:\t"); cute::print(sC_layout); cute::print("\n");      // sC_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_256,_16),_1,_4):((_64,_1),_0,_16)
  
  // The cluster shape and layout
  auto cluster_shape = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, cute::Int<1>{});
  cute::Layout cluster_layout_vmnk = cute::tiled_divide(cute::make_layout(cluster_shape),
                                            cute::make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  // Create TMA descriptors for A and B matrices
  cute::Copy_Atom tma_atom_A = cute::make_tma_atom_A_sm100(
    cute::SM90_TMA_LOAD{},        // TMA Load Op
    mA,                     // Source GMEM tensor
    sA_layout(cute::_, cute::_, cute::_, cute::Int<0>{}),              // Destination SMEM layout
    mma_tiler,
    tiled_mma,
    cluster_layout_vmnk  // MK Tiler for TMA operation
  );
  cute::Tensor mA_tma = tma_atom_A.get_tma_tensor(cute::shape(mA));   // (Gemm_M, Gemm_K)

  // cute::print("tma_atom_A:\t"); cute::print(tma_atom_A); cute::print("\n");

  cute::Copy_Atom tma_atom_B = cute::make_tma_atom_B_sm100(
      cute::SM90_TMA_LOAD{},        // TMA Load Op
      mB,                     // Source GMEM tensor
      sB_layout(cute::_, cute::_, cute::_, cute::Int<0>{}),             // Destination SMEM layout
      mma_tiler,
      tiled_mma,
      cluster_layout_vmnk  // NK Tiler for TMA operation
    );
  cute::Tensor mB_tma = tma_atom_B.get_tma_tensor(cute::shape(mB));   // (Gemm_N, Gemm_K)

  // cute::print("tma_atom_B:\t"); cute::print(tma_atom_B); cute::print("\n");

  cute::Copy_Atom tma_atom_C = cute::make_tma_atom(
      cute::SM90_TMA_STORE{},        // TMA Store Op
      mC,                     // Source GMEM tensor
      sC_layout(cute::_, cute::_, cute::_, cute::Int<0>{}),             // Destination SMEM layout
      epi_tiler
    );
  cute::Tensor mC_tma = tma_atom_C.get_tma_tensor(cute::shape(mC));   // (Gemm_N, Gemm_K)

  using SMEMStorage = kernel::PipedSharedStorage<TypeA, TypeB, TypeC, decltype(sA_layout), decltype(sB_layout), decltype(sC_layout), num_ab_stage, num_acc_stage>;

  int smemBytes = sizeof(SMEMStorage);

  // printf("SMEM Bytes: %d\n", smemBytes);

  dim3 grid_dim(1, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);

  if(device_ptr_bias != nullptr){
    auto* kernel_ptr = &linear_kernel_ws_sm100_wrapper<SMEMStorage,
                                decltype(mA_tma), decltype(mB_tma), decltype(mBias), decltype(mC_tma),
                                decltype(mma_tiler), decltype(epi_tiler), decltype(tiled_mma),
                                decltype(tma_atom_A), decltype(tma_atom_B), decltype(tma_atom_C), num_ab_stage, num_acc_stage, num_c_stage, false>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smemBytes));
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                              mA_tma, mB_tma, mBias, mC_tma,
                                                              mma_tiler, epi_tiler, tiled_mma, 
                                                              tma_atom_A, tma_atom_B, tma_atom_C, num_tmem_columns);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
  } else {
    auto* kernel_ptr = &linear_kernel_ws_sm100_wrapper<SMEMStorage,
                                decltype(mA_tma), decltype(mB_tma), decltype(mBias), decltype(mC_tma),
                                decltype(mma_tiler), decltype(epi_tiler), decltype(tiled_mma),
                                decltype(tma_atom_A), decltype(tma_atom_B), decltype(tma_atom_C), num_ab_stage, num_acc_stage, num_c_stage, true>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smemBytes));
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                              mA_tma, mB_tma, mBias, mC_tma,
                                                              mma_tiler, epi_tiler, tiled_mma,
                                                              tma_atom_A, tma_atom_B, tma_atom_C, num_tmem_columns);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
  }
}


void linear_kernel_warp_specialized(torch::Tensor input,
                   torch::Tensor weight,
                   c10::optional<at::Tensor> residual,
                   torch::Tensor output) {

  auto const* input_ptr = static_cast<const bfloat16*>(input.data_ptr());
  auto const* weight_ptr = static_cast<const bfloat16*>(weight.data_ptr());
  bool has_residual = residual.has_value();
  auto const* residual_ptr = has_residual ? static_cast<const bfloat16*>(residual->data_ptr()) : nullptr;
  auto *output_ptr = static_cast<bfloat16*>(output.data_ptr());
  
  // A tensor MxK K-major (Layout T = Row-Major)
  auto layout_A = cute::make_layout(cute::make_shape(weight.size(0), weight.size(1)), cute::make_stride(weight.size(1), cute::Int<1>{}));   // (Gemm_M,Gemm_K):(Gemm_K,_1)
  // B tensor NxK K-major (Layout N = Column-Major)
  auto layout_B = cute::make_layout(cute::make_shape(input.size(0), input.size(1)), cute::make_stride(input.size(1), cute::Int<1>{}));   // (Gemm_N,Gemm_K):(Gemm_K,_1)
  // C tensor MxN N-major (Layout T = Row-Major)
  auto layout_C = cute::make_layout(cute::make_shape(input.size(0), weight.size(0)), cute::make_stride(weight.size(0), cute::Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
  // Bias tensor MxN N-major (Layout T = Row-Major)
  auto layout_Bias = cute::make_layout(cute::make_shape(input.size(0), weight.size(0)), cute::make_stride(weight.size(0), cute::Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)

  launch_linear_sm100_warp_specialized(
    input_ptr,
    weight_ptr,
    residual_ptr,
    output_ptr,
    layout_A,
    layout_B,
    layout_Bias,
    layout_C
  );  

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// sm100_linear_mpk

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
__global__ __launch_bounds__(256, 1) void linear_sm100_mpk_wrapper(
    void * tma_a_desc_ptr,
    void * tma_b_desc_ptr,
    BiasTensor mBias,
    void * tma_out_desc_ptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  using TypeAcc = float;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size
  constexpr int TILE_SIZE =
      64; // we should modify this param if we want larger tile size
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  constexpr int OUTPUT_ATOM_SIZE = 128; // this is padded
  constexpr int OUTPUT_TMA_CP_SIZE = 128;
  constexpr int OUTPUT_ATOM_REPEAT_COL = 1;
  
  using TMA_B =
      kernel::tma::tma_2d<bfloat16,
                          B,
                          M,
                          S,
                          BATCH_SIZE,                      /*GMEM_ROW_*/
                          REDUCTION_SIZE,                  /*GMEM_COL_*/
                          MMA_N,                           /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,               /*SMEM_COL_*/
                          REDUCTION_SIZE,                  /*GMEM_STRIDE_ROW_*/
                          1,                               /*GMEM_STRIDE_COL_*/
                          1,                               /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,         /*SMEM_REPEAT_COL_*/
                          MMA_N * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  using TMA_A =
      kernel::tma::tma_2d<bfloat16,
                          B,
                          M,
                          S,
                          OUTPUT_SIZE,             /*GMEM_ROW_*/
                          REDUCTION_SIZE,          /*GMEM_COL_*/
                          MMA_M,                   /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,       /*SMEM_COL_*/
                          REDUCTION_SIZE,          /*GMEM_STRIDE_ROW_*/
                          1,                       /*GMEM_STRIDE_COL_*/
                          1,                       /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL, /*SMEM_REPEAT_COL_*/
                          MMA_M * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;

  using TMA_OUT = kernel::tma::tma_2d<bfloat16,
                                      0,
                                      M,
                                      S,
                                      BATCH_SIZE,              /*GMEM_ROW_*/
                                      OUTPUT_SIZE,             /*GMEM_COL_*/
                                      MMA_N,                   /*SMEM_ROW_*/
                                      MMA_M,                   /*SMEM_COL_*/
                                      OUTPUT_SIZE,             /*GMEM_STRIDE_ROW_*/
                                      1,                       /*GMEM_STRIDE_COL_*/
                                      1,                       /*SMEM_REPEAT_ROW_*/
                                      OUTPUT_ATOM_REPEAT_COL,  /*SMEM_REPEAT_COL_*/
                                      MMA_N * MMA_M,           /*SMEM_STRIDE_*/
                                      true>;
  
  TMA_A tma_a(static_cast<CUtensorMap*>(tma_a_desc_ptr));
  TMA_B tma_b(static_cast<CUtensorMap*>(tma_b_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap*>(tma_out_desc_ptr));

  kernel::linear_sm100_mpk_task_impl<
          T, 
          TMA_A,
          TMA_B,
          BiasTensor,
          TMA_OUT,
          MMA_M,
          MMA_N,
          BATCH_SIZE,
          OUTPUT_SIZE,
          REDUCTION_SIZE,
          NoBias,
          NUM_AB_STAGE,
          NUM_ACC_STAGE,
          NUM_C_STAGE>(tma_a, tma_b, mBias, tma_out);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_sm100_mpk(void *input_ptr,
                          void *weight_ptr,
                          void *output_ptr,
                          void *residual_ptr = nullptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  using TypeAcc = float;

  constexpr int MMA_M = 128;
  constexpr int MMA_N = 32;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size
  constexpr int TILE_SIZE =
      64; // we should modify this param if we want larger tile size
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  constexpr int OUTPUT_ATOM_SIZE = 128; // this is padded
  constexpr int OUTPUT_TMA_CP_SIZE = 128;
  constexpr int OUTPUT_ATOM_REPEAT_COL = 1;

  // TMA_A tma_a(weight_ptr);
  // TMA_B tma_b(input_ptr);
  // TMA_OUT tma_out(output_ptr);

  CUtensorMap host_i_desc;
  CUtensorMap host_w_desc;
  CUtensorMap host_o_desc;
  CUtensorMap *desc_i_ptr;
  CUtensorMap *desc_w_ptr;
  CUtensorMap *desc_o_ptr;

  // TMA_INPUT
  uint64_t i_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                            static_cast<uint64_t>(REDUCTION_SIZE)};
  uint64_t i_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE)};
  uint32_t i_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                            static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};

  size_t i_smem_repeat_col =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 2>(&host_i_desc,
                                      static_cast<bfloat16 *>(input_ptr),
                                      i_gmem_shape,
                                      i_gmem_stride,
                                      i_smem_shape,
                                      1,
                                      i_smem_repeat_col);

  // TMA_WEIGHT
  uint64_t w_gmem_shape[2] = {static_cast<uint64_t>(OUTPUT_SIZE),
                            static_cast<uint64_t>(REDUCTION_SIZE)};
  uint64_t w_gmem_stride[2] = {1, static_cast<uint64_t>(REDUCTION_SIZE)};
  uint32_t w_smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                            static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  size_t w_smem_repeat_col =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 2>(&host_w_desc,
                                      static_cast<bfloat16 *>(weight_ptr),
                                      w_gmem_shape,
                                      w_gmem_stride,
                                      w_smem_shape,
                                      1,
                                      w_smem_repeat_col);

  // TMA_OUT
  int const output_stride = OUTPUT_SIZE;
  uint64_t o_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                            static_cast<uint64_t>(OUTPUT_SIZE)};
  uint64_t o_gmem_stride[2] = {1, static_cast<uint64_t>(output_stride)};
  uint32_t o_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                            static_cast<uint32_t>(MMA_M)};
  size_t o_smem_repeat_col = 1;
  mirage::runtime::fill_tma_desc<bfloat16, 0, M, S, 2>(&host_o_desc,
                                      static_cast<bfloat16 *>(output_ptr),
                                      o_gmem_shape,
                                      o_gmem_stride,
                                      o_smem_shape,
                                      1,
                                      o_smem_repeat_col);
  
  cudaMalloc(&desc_i_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_w_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_o_ptr, sizeof(CUtensorMap));

  cudaMemcpy(desc_i_ptr, &host_i_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_w_ptr, &host_w_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(desc_o_ptr, &host_o_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  void *tma_desc_input;
  void *tma_desc_weight;
  void *tma_desc_output;

  tma_desc_input = desc_i_ptr;
  tma_desc_weight = desc_w_ptr;
  tma_desc_output = desc_o_ptr;

  // Residual
  cute::Layout layout_Bias = cute::make_layout(cute::make_shape(BATCH_SIZE, OUTPUT_SIZE), cute::make_stride(OUTPUT_SIZE, cute::Int<1>{}));   // (Gemm_M,Gemm_N):(Gemm_N,_1)
  cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<T*>(residual_ptr)), layout_Bias);      // (Gemm_N, Gemm_M)

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  int smemBytes = 224 * 1024;

  if(residual_ptr != nullptr){
    auto* kernel_ptr = &linear_sm100_mpk_wrapper<T,
                                BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                                decltype(mBias), 
                                MMA_M, MMA_N,
                                false>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smemBytes));
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                              tma_desc_weight, tma_desc_input, mBias, tma_desc_output);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
  } else {
    auto* kernel_ptr = &linear_sm100_mpk_wrapper<T,
                                BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                                decltype(mBias),
                                MMA_M, MMA_N,
                                true>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smemBytes));
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                              tma_desc_weight, tma_desc_input, mBias, tma_desc_output);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
  }

}

void linear_sm100_mpk_kernel(torch::Tensor input,
                          torch::Tensor weight,
                          c10::optional<at::Tensor> residual,
                          torch::Tensor output) {

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  bool has_residual = residual.has_value();
  void *residual_ptr = has_residual ? residual->data_ptr() : nullptr;
  void *output_ptr = output.data_ptr();

  // const int BATCH_SIZE = input.size(0);
  // const int OUTPUT_SIZE = output.size(1);
  // const int REDUCTION_SIZE = weight.size(1);

  constexpr int BATCH_SIZE = 8;
  constexpr int OUTPUT_SIZE = 64*96;
  constexpr int REDUCTION_SIZE = 4096;

  assert(input.size(1) == REDUCTION_SIZE);
  assert(weight.size(0) == OUTPUT_SIZE);

  launch_linear_sm100_mpk<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(input_ptr, weight_ptr, output_ptr, residual_ptr);      
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_warp_specialized", &linear_kernel_warp_specialized, "Linear kernel Warp Specialized");
  m.def("linear_sm100_mpk", &linear_sm100_mpk_kernel, "Linear kernel SM100 MPK");
}