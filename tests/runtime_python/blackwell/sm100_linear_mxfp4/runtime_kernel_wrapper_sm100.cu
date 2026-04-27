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

#include "blackwell/linear_mxfp4_1d2d_sm100.cuh"
#include "blackwell/linear_mxfp4_swapAB_sm100.cuh"
#include "blackwell/quantize_mxfp4_sm100.cuh"
#include "hopper/tma_2d_mxfp4.cuh"
#include "hopper/tma_3d.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/util/print_error.hpp>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>

using float_e2m1  = cute::float_e2m1_t;
using float_ue8m0 = cute::float_ue8m0_t;  // MXFP4 uses 8-bit exponent scale factor

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

constexpr int align_up_bytes(int bytes, int alignment) {
  return ((bytes + alignment - 1) / alignment) * alignment;
}

template <int MMA_M,
          int MMA_N,
          int NUM_AB_STAGE,
          int NUM_ACC_STAGE,
          int NUM_C_STAGE,
          bool PAD_SFB_TO_128,
          bool FULL_C_TILE>
constexpr int compute_smem_bytes() {
  constexpr int MMA_K = 64;
  constexpr int NUM_MMA_K = 4;
  constexpr int SCALE_VEC_SIZE = 32;
  constexpr int SMEM_ALIGNMENT_BYTES = 128;
  constexpr int MEMBER_ALIGNMENT_BYTES = 16;

  constexpr int bK = MMA_K * NUM_MMA_K;
  constexpr int MMA_N_SFB = PAD_SFB_TO_128 ? ((MMA_N + 127) / 128) * 128 : MMA_N;
  constexpr int A_BYTES = NUM_AB_STAGE * (MMA_M * bK / 2);
  constexpr int B_BYTES = NUM_AB_STAGE * (MMA_N * bK / 2);
  constexpr int C_STAGE_BYTES =
      FULL_C_TILE ? (MMA_M * MMA_N * static_cast<int>(sizeof(float)))
                  : (MMA_M * MMA_N);
  constexpr int C_BYTES = NUM_C_STAGE * C_STAGE_BYTES;
  constexpr int SFA_BYTES = NUM_AB_STAGE * (MMA_M * bK / SCALE_VEC_SIZE);
  constexpr int SFB_BYTES = NUM_AB_STAGE * (MMA_N_SFB * bK / SCALE_VEC_SIZE);
  constexpr int AB_BARRIER_BYTES = 4 * NUM_AB_STAGE * static_cast<int>(sizeof(cute::uint64_t));
  constexpr int ACC_BARRIER_BYTES = 2 * NUM_ACC_STAGE * static_cast<int>(sizeof(cute::uint64_t));
  constexpr int TMEM_PTR_BYTES = 3 * static_cast<int>(sizeof(cute::uint32_t));

  int total_bytes = 0;
  total_bytes = align_up_bytes(total_bytes, SMEM_ALIGNMENT_BYTES) + A_BYTES;
  total_bytes = align_up_bytes(total_bytes, SMEM_ALIGNMENT_BYTES) + B_BYTES;
  total_bytes = align_up_bytes(total_bytes, SMEM_ALIGNMENT_BYTES) + C_BYTES;
  total_bytes = align_up_bytes(total_bytes, SMEM_ALIGNMENT_BYTES) + SFA_BYTES;
  total_bytes = align_up_bytes(total_bytes, SMEM_ALIGNMENT_BYTES) + SFB_BYTES;
  total_bytes = align_up_bytes(total_bytes, MEMBER_ALIGNMENT_BYTES) + AB_BARRIER_BYTES;
  total_bytes = align_up_bytes(total_bytes, MEMBER_ALIGNMENT_BYTES) + ACC_BARRIER_BYTES;
  total_bytes = align_up_bytes(total_bytes, MEMBER_ALIGNMENT_BYTES) + TMEM_PTR_BYTES;
  return align_up_bytes(total_bytes, SMEM_ALIGNMENT_BYTES);
}

template <int MMA_M, int MMA_N, int OUTPUT_SIZE, int M, int S>
static inline void
fill_linear_mxfp4_swapAB_output_tma_desc(CUtensorMap *tma_desc,
                                         void *output_ptr,
                                         int logical_batch_size) {
  using OutputTma =
      kernel::detail::LinearMxfp4SwapABOutputTma<float,
                                                 MMA_M,
                                                 MMA_N,
                                                 OUTPUT_SIZE>;
  uint64_t gmem_shape[3] = {
      (uint64_t)logical_batch_size,
      (uint64_t)(OUTPUT_SIZE / OutputTma::kVectorElems),
      (uint64_t)OutputTma::kVectorElems};
  uint64_t gmem_stride[3] = {1,
                             (uint64_t)OutputTma::kVectorElems,
                             (uint64_t)OUTPUT_SIZE};
  uint32_t smem_shape[3] = {
      (uint32_t)MMA_N,
      (uint32_t)(MMA_M / OutputTma::kVectorElems),
      (uint32_t)OutputTma::kVectorElems};
  mirage::runtime::fill_tma_desc<float, OutputTma::kBOut, M, S, 3>(
      tma_desc, static_cast<float *>(output_ptr),
      gmem_shape, gmem_stride, smem_shape, 1, 1);
}

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          class BiasTensor,
          class OutputTensor,
          int MMA_M,
          int MMA_N,
          bool NoBias,
          int NUM_AB_STAGE  = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE   = 4>
__global__ __launch_bounds__(256, 1)
void linear_mxfp4_1d2d_sm100_wrapper(void *tma_a_desc_ptr,
                                     void *tma_b_desc_ptr,
                                     void *tma_sfa_desc_ptr,
                                     void *tma_sfb_desc_ptr,
                                     BiasTensor mBias,
                                     OutputTensor mOutput,
                                     void *tma_out_desc_ptr) {
  constexpr int MMA_K          = 64;
  constexpr int NUM_MMA_K      = 4;
  constexpr int bK             = MMA_K * NUM_MMA_K;
  constexpr int SCALE_VECTOR_SIZE = 32;
  constexpr int EPI_PIPE_DEPTH = 4;
  constexpr int EPI_N          = MMA_N / EPI_PIPE_DEPTH;
  constexpr int B_FP4 = 3;
  constexpr int B_SF  = 0;
  constexpr int B_OUT = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;
  constexpr int SF_COL_B = MMA_N * MMA_K / SCALE_VECTOR_SIZE / 2;
  constexpr int SWIZZLE_SIZE = 128 / sizeof(float);

  using TMA_A = kernel::tma::tma_2d_mxfp4<cute::float_e2m1_t, B_FP4, M, S,
      BATCH_SIZE, REDUCTION_SIZE, MMA_M, bK, REDUCTION_SIZE, 1, 1, 1, MMA_M * bK / 2, true>;
  using TMA_B = kernel::tma::tma_2d_mxfp4<cute::float_e2m1_t, B_FP4, M, S,
      OUTPUT_SIZE, REDUCTION_SIZE, MMA_N, bK, REDUCTION_SIZE, 1, 1, 1, MMA_N * bK / 2, true>;
  using TMA_SFA = kernel::tma::tma_3d<cute::half_t, B_SF, M, S,
      BATCH_SIZE / MMA_M, REDUCTION_SIZE / MMA_K, SF_COL_A,
      1, 1, SF_COL_A,
      SF_COL_A * (REDUCTION_SIZE / MMA_K), SF_COL_A, 1,
      NUM_MMA_K, 1, SF_COL_A, true>;
  using TMA_SFB = kernel::tma::tma_3d<cute::half_t, B_SF, M, S,
      OUTPUT_SIZE / MMA_N, REDUCTION_SIZE / MMA_K, SF_COL_B,
      1, 1, SF_COL_B,
      SF_COL_B * (REDUCTION_SIZE / MMA_K), SF_COL_B, 1,
      NUM_MMA_K, 1, SF_COL_B, true>;
  using TMA_OUT = kernel::tma::tma_3d<float, B_OUT, M, S,
      BATCH_SIZE, OUTPUT_SIZE / SWIZZLE_SIZE, SWIZZLE_SIZE,
      MMA_M, EPI_N / SWIZZLE_SIZE, SWIZZLE_SIZE,
      OUTPUT_SIZE, SWIZZLE_SIZE, 1,
      1, 1, MMA_M * EPI_N, true>;

  TMA_A   tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B   tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_SFA tma_sfa(static_cast<CUtensorMap *>(tma_sfa_desc_ptr));
  TMA_SFB tma_sfb(static_cast<CUtensorMap *>(tma_sfb_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

  kernel::linear_mxfp4_1d2d_sm100_task_impl<T, TMA_A, TMA_B, TMA_SFA, TMA_SFB,
                                             BiasTensor, OutputTensor, TMA_OUT,
                                             MMA_M, MMA_N,
                                             BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                                             SCALE_VECTOR_SIZE, NoBias, /*SplitK=*/false,
                                             NUM_AB_STAGE, NUM_ACC_STAGE, NUM_C_STAGE>(
      tma_a, tma_b, tma_sfa, tma_sfb, mBias, mOutput, tma_out);
}

struct LinearMXFP4DescriptorCache {
  CUtensorMap host_i_desc{};
  CUtensorMap host_i_sf_desc{};
  CUtensorMap host_w_desc{};
  CUtensorMap host_w_sf_desc{};
  CUtensorMap host_o_desc{};

  CUtensorMap *desc_i_ptr    = nullptr;
  CUtensorMap *desc_i_sf_ptr = nullptr;
  CUtensorMap *desc_w_ptr    = nullptr;
  CUtensorMap *desc_w_sf_ptr = nullptr;
  CUtensorMap *desc_o_ptr    = nullptr;

  void *last_input_ptr     = nullptr;
  void *last_input_sf_ptr  = nullptr;
  void *last_weight_ptr    = nullptr;
  void *last_weight_sf_ptr = nullptr;
  void *last_output_ptr    = nullptr;

  bool initialized                = false;
  bool kernel_configured_no_bias  = false;  
  bool kernel_configured_bias     = false;  
};

static std::map<std::tuple<int,int,int>, std::unique_ptr<LinearMXFP4DescriptorCache>> s_1d2d_caches;
static std::mutex s_1d2d_mutex;

static LinearMXFP4DescriptorCache &get_linear_mxfp4_descriptor_cache(int batch_size, int output_size, int reduction_size) {
  auto key = std::make_tuple(batch_size, output_size, reduction_size);
  std::lock_guard<std::mutex> guard(s_1d2d_mutex);
  auto &entry = s_1d2d_caches[key];
  if (!entry) {
    entry = std::make_unique<LinearMXFP4DescriptorCache>();
  }
  return *entry;
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_mxfp4_1d2d_sm100(void *input_ptr,
                                    void *input_sf_ptr,
                                    void *weight_ptr,
                                    void *weight_sf_ptr,
                                    void *output_ptr,
                                    void *residual_ptr = nullptr) {
  using namespace cute;
  using namespace cutlass;

  constexpr int B_FP4 = 3;
  constexpr int B_SF  = 0;
  constexpr int B_OUT = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int SCALE_VECTOR_SIZE = 32;
  constexpr int MMA_M    = 128;
  constexpr int MMA_N    = 128;
  constexpr int MMA_K    = 64;
  constexpr int NUM_MMA_K = 4;
  constexpr int bK = MMA_K * NUM_MMA_K;
  constexpr int EPI_PIPE_DEPTH = 4;
  constexpr int EPI_N = MMA_N / EPI_PIPE_DEPTH;

  constexpr int NUM_M_TILE_PER_CTA = 1;
  constexpr int NUM_N_TILE_PER_CTA = 1;
  constexpr int bM = MMA_M;
  constexpr int bN = MMA_N;
  constexpr int NUM_AB_STAGE_LAUNCH = 4;
  constexpr int NUM_ACC_STAGE_LAUNCH = 2;
  constexpr int NUM_C_STAGE_LAUNCH = 4;

  auto &cache = get_linear_mxfp4_descriptor_cache(BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto build_2d_desc = [](auto fp_tag, int b, CUtensorMap *host_desc,
                          CUtensorMap **device_desc_pp, void *src,
                          uint64_t rows, uint64_t cols, uint32_t srows,
                          uint32_t scols) {
    using FP = decltype(fp_tag);
    uint64_t gmem_shape[2]  = {rows, cols};
    uint64_t gmem_stride[2] = {1, cols};
    uint32_t smem_shape[2]  = {srows, scols};
    auto dispatch = [&](auto B_tag) {
      mirage::runtime::fill_tma_desc<FP, decltype(B_tag)::value, M, S, 2>(
          host_desc, static_cast<FP *>(src), gmem_shape, gmem_stride, smem_shape, 1, 1);
    };
    if (b == B_FP4) dispatch(std::integral_constant<int, B_FP4>{});
    else            dispatch(std::integral_constant<int, B_SF >{});
    cudaMalloc(device_desc_pp, sizeof(CUtensorMap));
    cudaMemcpy(*device_desc_pp, host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  };

  auto build_3d_desc = [](auto fp_tag, int b, CUtensorMap *host_desc,
                          CUtensorMap **device_desc_pp, void *src,
                          uint64_t s0, uint64_t s1, uint64_t s2,
                          uint64_t st0, uint64_t st1, uint64_t st2,
                          uint32_t b0, uint32_t b1, uint32_t b2) {
    using FP = decltype(fp_tag);
    uint64_t gmem_shape[3]  = {s0, s1, s2};
    uint64_t gmem_stride[3] = {st0, st1, st2};
    uint32_t smem_shape[3]  = {b0, b1, b2};
    auto dispatch = [&](auto B_tag) {
      mirage::runtime::fill_tma_desc<FP, decltype(B_tag)::value, M, S, 3>(
          host_desc, static_cast<FP *>(src), gmem_shape, gmem_stride, smem_shape, 1, 1);
    };
    if (b == B_FP4)      dispatch(std::integral_constant<int, B_FP4>{});
    else if (b == B_OUT) dispatch(std::integral_constant<int, B_OUT>{});
    else                 dispatch(std::integral_constant<int, B_SF >{});
    cudaMalloc(device_desc_pp, sizeof(CUtensorMap));
    cudaMemcpy(*device_desc_pp, host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  };

  auto refresh_desc = [&](void *new_ptr, void *&last_ptr,
                          CUtensorMap *host_desc, CUtensorMap *device_desc) {
    if (new_ptr == last_ptr) return;
    cuTensorMapReplaceAddress(host_desc, new_ptr);
    cudaMemcpyAsync(device_desc, host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    last_ptr = new_ptr;
  };

  constexpr int SF_COL_A     = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;
  constexpr int SF_COL_B     = MMA_N * MMA_K / SCALE_VECTOR_SIZE / 2;
  constexpr int SWIZZLE_SIZE = 128 / sizeof(float);

  if (!cache.initialized) {
    build_2d_desc(cute::float_e2m1_t{}, B_FP4, &cache.host_i_desc, &cache.desc_i_ptr,
                  input_ptr,  BATCH_SIZE,  REDUCTION_SIZE, MMA_M, bK);
    build_2d_desc(cute::float_e2m1_t{}, B_FP4, &cache.host_w_desc, &cache.desc_w_ptr,
                  weight_ptr, OUTPUT_SIZE, REDUCTION_SIZE, MMA_N, bK);
    build_3d_desc(cute::half_t{}, B_SF, &cache.host_i_sf_desc, &cache.desc_i_sf_ptr,
                  input_sf_ptr,
                  BATCH_SIZE / MMA_M, REDUCTION_SIZE / MMA_K, SF_COL_A,
                  1, SF_COL_A, SF_COL_A * (REDUCTION_SIZE / MMA_K),
                  1, 1, SF_COL_A);
    build_3d_desc(cute::half_t{}, B_SF, &cache.host_w_sf_desc, &cache.desc_w_sf_ptr,
                  weight_sf_ptr,
                  OUTPUT_SIZE / MMA_N, REDUCTION_SIZE / MMA_K, SF_COL_B,
                  1, SF_COL_B, SF_COL_B * (REDUCTION_SIZE / MMA_K),
                  1, 1, SF_COL_B);
    build_3d_desc(float{}, B_OUT, &cache.host_o_desc, &cache.desc_o_ptr,
                  output_ptr,
                  BATCH_SIZE, OUTPUT_SIZE / SWIZZLE_SIZE, SWIZZLE_SIZE,
                  1, SWIZZLE_SIZE, OUTPUT_SIZE,
                  MMA_M, EPI_N / SWIZZLE_SIZE, SWIZZLE_SIZE);

    cache.last_input_ptr     = input_ptr;
    cache.last_input_sf_ptr  = input_sf_ptr;
    cache.last_weight_ptr    = weight_ptr;
    cache.last_weight_sf_ptr = weight_sf_ptr;
    cache.last_output_ptr    = output_ptr;
    cache.initialized        = true;
  } else {
    refresh_desc(input_ptr,     cache.last_input_ptr,     &cache.host_i_desc,    cache.desc_i_ptr);
    refresh_desc(input_sf_ptr,  cache.last_input_sf_ptr,  &cache.host_i_sf_desc, cache.desc_i_sf_ptr);
    refresh_desc(weight_ptr,    cache.last_weight_ptr,    &cache.host_w_desc,    cache.desc_w_ptr);
    refresh_desc(weight_sf_ptr, cache.last_weight_sf_ptr, &cache.host_w_sf_desc, cache.desc_w_sf_ptr);
    refresh_desc(output_ptr,    cache.last_output_ptr,    &cache.host_o_desc,    cache.desc_o_ptr);
  }

  cute::Layout layout_Bias = cute::make_layout(
      cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
      cute::make_stride(OUTPUT_SIZE, cute::Int<1>{}));
  cute::Tensor mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<float *>(residual_ptr)), layout_Bias);
  cute::Tensor mOutput = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<float *>(output_ptr)), layout_Bias);

  constexpr int num_tiles_m = BATCH_SIZE / bM / NUM_M_TILE_PER_CTA;
  constexpr int num_tiles_n = OUTPUT_SIZE / bN / NUM_N_TILE_PER_CTA;
  dim3 grid_dim(num_tiles_m, num_tiles_n, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  constexpr int smemBytes =
      compute_smem_bytes<MMA_M,
                         MMA_N,
                         NUM_AB_STAGE_LAUNCH,
                         NUM_ACC_STAGE_LAUNCH,
                         NUM_C_STAGE_LAUNCH,
                         /*PAD_SFB_TO_128=*/false,
                         /*FULL_C_TILE=*/false>();
  static_assert(
      smemBytes <= 224 * 1024,
      "SM100 MXFP4 1d2d launch exceeds the 224 KiB shared-memory budget");
  // std::cout << "SMEM BYTES: " << smemBytes << "B" << std::endl;

  auto launch = [&](auto *kernel_ptr, bool &configured_flag) {
    if (!configured_flag) {
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      configured_flag = true;
    }
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes, stream};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, (void const *)kernel_ptr,
        (void *)cache.desc_i_ptr, (void *)cache.desc_w_ptr,
        (void *)cache.desc_i_sf_ptr, (void *)cache.desc_w_sf_ptr,
        mBias, mOutput, (void *)cache.desc_o_ptr);
    CUTE_CHECK_ERROR(cudaGetLastError());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel launch" << std::endl;
    }
  };

  auto kernel_for = [&](auto NoBiasTag) {
    return &linear_mxfp4_1d2d_sm100_wrapper<
        T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
        decltype(mBias), decltype(mOutput), MMA_M, MMA_N,
        decltype(NoBiasTag)::value,
        NUM_AB_STAGE_LAUNCH, NUM_ACC_STAGE_LAUNCH, NUM_C_STAGE_LAUNCH>;
  };

  if (residual_ptr != nullptr) {
    launch(kernel_for(std::false_type{}), cache.kernel_configured_bias);
  } else {
    launch(kernel_for(std::true_type{}),  cache.kernel_configured_no_bias);
  }
}

template <typename T,
          int MMA_N,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          class BiasTensor,
          bool NoBias,
          int NUM_AB_STAGE = 4,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 1>
__global__ __launch_bounds__(256, 1)
void linear_mxfp4_swapAB_sm100_wrapper(void *tma_a_desc_ptr,
                                       void *tma_b_desc_ptr,
                                       void *tma_sfa_desc_ptr,
                                       void *tma_sfb_desc_ptr,
                                       void *tma_out_desc_ptr,
                                       BiasTensor mBias) {
  constexpr int MMA_K = 64;
  constexpr int NUM_MMA_K = 4;
  constexpr int bK = MMA_K * NUM_MMA_K;
  constexpr int SCALE_VECTOR_SIZE = 32;
  constexpr int MMA_M = 128;

  constexpr int B_FP4 = 3;
  constexpr int B_SF  = 0;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int MMA_N_SFB = 128;
  constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;        // 128
  constexpr int SF_COL_B = MMA_N_SFB * MMA_K / SCALE_VECTOR_SIZE / 2;    // 128
  using OutputTma =
      kernel::detail::LinearMxfp4SwapABOutputTma<float,
                                                 MMA_M,
                                                 MMA_N,
                                                 OUTPUT_SIZE>;

  using TMA_A = kernel::tma::tma_2d_mxfp4<cute::float_e2m1_t, B_FP4, M, S,
      OUTPUT_SIZE, REDUCTION_SIZE, MMA_M, bK, REDUCTION_SIZE, 1, 1, 1, MMA_M * bK / 2, true>;
  using TMA_B = kernel::tma::tma_2d_mxfp4<cute::float_e2m1_t, B_FP4, M, S,
      MMA_N, REDUCTION_SIZE, MMA_N, bK, REDUCTION_SIZE, 1, 1, 1, MMA_N * bK / 2, true>;
  using TMA_SFA = kernel::tma::tma_3d<cute::half_t, B_SF, M, S,
      OUTPUT_SIZE / MMA_M, REDUCTION_SIZE / MMA_K, SF_COL_A,
      1, 1, SF_COL_A,
      SF_COL_A * (REDUCTION_SIZE / MMA_K), SF_COL_A, 1,
      NUM_MMA_K, 1, SF_COL_A, true>;
  using TMA_SFB = kernel::tma::tma_3d<cute::half_t, B_SF, M, S,
      1, REDUCTION_SIZE / MMA_K, SF_COL_B,
      1, 1, SF_COL_B,
      SF_COL_B * (REDUCTION_SIZE / MMA_K), SF_COL_B, 1,
      NUM_MMA_K, 1, SF_COL_B, true>;
  using TMA_OUT = typename OutputTma::template Tma<M, S>;

  TMA_A   tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B   tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_SFA tma_sfa(static_cast<CUtensorMap *>(tma_sfa_desc_ptr));
  TMA_SFB tma_sfb(static_cast<CUtensorMap *>(tma_sfb_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

  kernel::linear_mxfp4_smallm_swapAB_sm100_task_impl<T, TMA_A, TMA_B, TMA_SFA, TMA_SFB,
                                                     TMA_OUT, BiasTensor,
                                                     /*MMA_M=*/128, MMA_N,
                                                     OUTPUT_SIZE, REDUCTION_SIZE,
                                                     SCALE_VECTOR_SIZE, NoBias,
                                                     NUM_AB_STAGE, NUM_ACC_STAGE, NUM_C_STAGE>(
      tma_a, tma_b, tma_sfa, tma_sfb, tma_out, mBias);
}

struct LinearMXFP4SwapABDescriptorCache {
  CUtensorMap host_w_desc{};
  CUtensorMap host_x_desc{};
  CUtensorMap host_w_sf_desc{};
  CUtensorMap host_x_sf_desc{};
  CUtensorMap host_o_desc{};

  CUtensorMap *desc_w_ptr    = nullptr;
  CUtensorMap *desc_x_ptr    = nullptr;
  CUtensorMap *desc_w_sf_ptr = nullptr;
  CUtensorMap *desc_x_sf_ptr = nullptr;
  CUtensorMap *desc_o_ptr    = nullptr;

  void *last_weight_ptr    = nullptr;
  void *last_input_ptr     = nullptr;
  void *last_weight_sf_ptr = nullptr;
  void *last_input_sf_ptr  = nullptr;
  void *last_output_ptr    = nullptr;
  int   last_logical_batch_size = 0;

  bool initialized                = false;
  bool kernel_configured_no_bias  = false;
  bool kernel_configured_bias     = false;
};

static std::map<std::tuple<int,int,int>, std::unique_ptr<LinearMXFP4SwapABDescriptorCache>> s_swapAB_caches;
static std::mutex s_swapAB_mutex;

static LinearMXFP4SwapABDescriptorCache &get_linear_mxfp4_swapAB_descriptor_cache(int mma_n,
                                                                                   int output_size,
                                                                                   int reduction_size) {
  auto key = std::make_tuple(mma_n, output_size, reduction_size);
  std::lock_guard<std::mutex> guard(s_swapAB_mutex);
  auto &entry = s_swapAB_caches[key];
  if (!entry) {
    entry = std::make_unique<LinearMXFP4SwapABDescriptorCache>();
  }
  return *entry;
}

template <typename T, int MMA_N, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_mxfp4_swapAB_sm100(void *input_ptr,
                                      void *input_sf_ptr,
                                      void *weight_ptr,
                                      void *weight_sf_ptr,
                                      void *output_ptr,
                                      void *residual_ptr,
                                      int logical_batch_size) {
  using namespace cute;
  using namespace cutlass;

  constexpr int B_FP4 = 3;
  constexpr int B_SF  = 0;
  constexpr int MMA_K = 64;
  constexpr int NUM_MMA_K = 4;
  constexpr int bK = MMA_K * NUM_MMA_K;
  constexpr int SCALE_VECTOR_SIZE = 32;
  constexpr int MMA_M = 128;
  constexpr int MMA_N_SFB = 128;
  constexpr int SF_COL_A = MMA_M * MMA_K / SCALE_VECTOR_SIZE / 2;      // 128
  constexpr int SF_COL_B = MMA_N_SFB * MMA_K / SCALE_VECTOR_SIZE / 2;  // 128
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int NUM_AB_STAGE_LAUNCH = 4;
  constexpr int NUM_ACC_STAGE_LAUNCH = 2;
  // The swapAB template is instantiated for every supported MMA_N in the
  // dispatch table. Four C stages fit for MMA_N <= 32, but 64/128 must fall
  // back to a single C stage to stay within the 224 KiB SMEM budget.
  constexpr int NUM_C_STAGE_LAUNCH = (MMA_N <= 32) ? 4 : 1;

  auto &cache = get_linear_mxfp4_swapAB_descriptor_cache(MMA_N, OUTPUT_SIZE, REDUCTION_SIZE);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_n_tiles = (logical_batch_size + MMA_N - 1) / MMA_N;

  auto fill_2d_desc = [&](auto fp_tag, int b, CUtensorMap *host_desc,
                          CUtensorMap **device_desc_pp, void *src,
                          uint64_t rows, uint64_t cols, uint32_t srows,
                          uint32_t scols) {
    using FP = decltype(fp_tag);
    uint64_t gmem_shape[2]  = {rows, cols};
    uint64_t gmem_stride[2] = {1, cols};
    uint32_t smem_shape[2]  = {srows, scols};
    auto dispatch = [&](auto B_tag) {
      mirage::runtime::fill_tma_desc<FP, decltype(B_tag)::value, M, S, 2>(
          host_desc, static_cast<FP *>(src), gmem_shape, gmem_stride, smem_shape, 1, 1);
    };
    if (b == B_FP4) dispatch(std::integral_constant<int, B_FP4>{});
    else            dispatch(std::integral_constant<int, B_SF>{});
    if (device_desc_pp) {
      cudaMalloc(device_desc_pp, sizeof(CUtensorMap));
      cudaMemcpy(*device_desc_pp, host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    }
  };

  auto fill_3d_desc = [&](auto fp_tag, int b, CUtensorMap *host_desc,
                          CUtensorMap **device_desc_pp, void *src,
                          uint64_t s0, uint64_t s1, uint64_t s2,
                          uint64_t st0, uint64_t st1, uint64_t st2,
                          uint32_t b0, uint32_t b1, uint32_t b2) {
    using FP = decltype(fp_tag);
    uint64_t gmem_shape[3]  = {s0, s1, s2};
    uint64_t gmem_stride[3] = {st0, st1, st2};
    uint32_t smem_shape[3]  = {b0, b1, b2};
    auto dispatch = [&](auto B_tag) {
      mirage::runtime::fill_tma_desc<FP, decltype(B_tag)::value, M, S, 3>(
          host_desc, static_cast<FP *>(src), gmem_shape, gmem_stride, smem_shape, 1, 1);
    };
    if (b == B_FP4) dispatch(std::integral_constant<int, B_FP4>{});
    else            dispatch(std::integral_constant<int, B_SF>{});
    if (device_desc_pp) {
      cudaMalloc(device_desc_pp, sizeof(CUtensorMap));
      cudaMemcpy(*device_desc_pp, host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    }
  };

  auto refresh_desc = [&](void *new_ptr, void *&last_ptr,
                          CUtensorMap *host_desc, CUtensorMap *device_desc) {
    if (new_ptr == last_ptr) return;
    cuTensorMapReplaceAddress(host_desc, new_ptr);
    cudaMemcpyAsync(device_desc, host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    last_ptr = new_ptr;
  };

  auto rebuild_x_descs = [&](CUtensorMap **x_pp, CUtensorMap **x_sf_pp) {
    fill_2d_desc(cute::float_e2m1_t{}, B_FP4, &cache.host_x_desc, x_pp,
                 input_ptr, (uint64_t)logical_batch_size, REDUCTION_SIZE, MMA_N, bK);
    fill_3d_desc(cute::half_t{}, B_SF, &cache.host_x_sf_desc, x_sf_pp,
                 input_sf_ptr,
                 (uint64_t)num_n_tiles, REDUCTION_SIZE / MMA_K, SF_COL_B,
                 1, SF_COL_B, SF_COL_B * (REDUCTION_SIZE / MMA_K),
                 1, 1, SF_COL_B);
    if (!x_pp) {
      cudaMemcpyAsync(cache.desc_x_ptr,    &cache.host_x_desc,    sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(cache.desc_x_sf_ptr, &cache.host_x_sf_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    }
    cache.last_input_ptr          = input_ptr;
    cache.last_input_sf_ptr       = input_sf_ptr;
    cache.last_logical_batch_size = logical_batch_size;
  };

  auto rebuild_output_desc = [&](bool first_init) {
    fill_linear_mxfp4_swapAB_output_tma_desc<MMA_M, MMA_N, OUTPUT_SIZE, M, S>(
        &cache.host_o_desc, output_ptr, logical_batch_size);
    if (first_init) {
      cudaMalloc(&cache.desc_o_ptr, sizeof(CUtensorMap));
      cudaMemcpy(cache.desc_o_ptr, &cache.host_o_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(cache.desc_o_ptr, &cache.host_o_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    }
    cache.last_output_ptr = output_ptr;
  };

  if (!cache.initialized) {
    fill_2d_desc(cute::float_e2m1_t{}, B_FP4, &cache.host_w_desc, &cache.desc_w_ptr,
                 weight_ptr, OUTPUT_SIZE, REDUCTION_SIZE, MMA_M, bK);
    fill_3d_desc(cute::half_t{}, B_SF, &cache.host_w_sf_desc, &cache.desc_w_sf_ptr,
                 weight_sf_ptr,
                 OUTPUT_SIZE / MMA_M, REDUCTION_SIZE / MMA_K, SF_COL_A,
                 1, SF_COL_A, SF_COL_A * (REDUCTION_SIZE / MMA_K),
                 1, 1, SF_COL_A);
    cache.last_weight_ptr    = weight_ptr;
    cache.last_weight_sf_ptr = weight_sf_ptr;
    rebuild_x_descs(&cache.desc_x_ptr, &cache.desc_x_sf_ptr);
    rebuild_output_desc(/*first_init=*/true);
    cache.initialized = true;
  } else {
    refresh_desc(weight_ptr,    cache.last_weight_ptr,    &cache.host_w_desc,    cache.desc_w_ptr);
    refresh_desc(weight_sf_ptr, cache.last_weight_sf_ptr, &cache.host_w_sf_desc, cache.desc_w_sf_ptr);
    if (output_ptr != cache.last_output_ptr || logical_batch_size != cache.last_logical_batch_size) {
      rebuild_output_desc(/*first_init=*/false);
    }
    if (logical_batch_size != cache.last_logical_batch_size) {
      rebuild_x_descs(/*x_pp=*/nullptr, /*x_sf_pp=*/nullptr);
    } else {
      refresh_desc(input_ptr,    cache.last_input_ptr,    &cache.host_x_desc,    cache.desc_x_ptr);
      refresh_desc(input_sf_ptr, cache.last_input_sf_ptr, &cache.host_x_sf_desc, cache.desc_x_sf_ptr);
    }
  }

  constexpr int num_output_tiles = OUTPUT_SIZE / MMA_M;
  dim3 grid_dim(num_output_tiles, num_n_tiles, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  constexpr int smemBytes =
      compute_smem_bytes<MMA_M,
                         MMA_N,
                         NUM_AB_STAGE_LAUNCH,
                         NUM_ACC_STAGE_LAUNCH,
                         NUM_C_STAGE_LAUNCH,
                         /*PAD_SFB_TO_128=*/true,
                         /*FULL_C_TILE=*/true>();
  static_assert(
      smemBytes <= 224 * 1024,
      "SM100 MXFP4 swapAB launch exceeds the 224 KiB shared-memory budget");
  // std::cout << "SMEM BYTES: " << smemBytes << "B" << std::endl;

  cute::Layout layout_Out = cute::make_layout(
      cute::make_shape(logical_batch_size, OUTPUT_SIZE),
      cute::make_stride((int)OUTPUT_SIZE, cute::Int<1>{}));
  cute::Tensor mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<float *>(residual_ptr)), layout_Out);

  auto do_launch = [&](auto *kernel_ptr, bool &configured_flag) {
    if (!configured_flag) {
      CUTE_CHECK_ERROR(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
      configured_flag = true;
    }
    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smemBytes, stream};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, (void const *)kernel_ptr,
        (void *)cache.desc_w_ptr, (void *)cache.desc_x_ptr,
        (void *)cache.desc_w_sf_ptr, (void *)cache.desc_x_sf_ptr,
        (void *)cache.desc_o_ptr,
        mBias);
    CUTE_CHECK_ERROR(cudaGetLastError());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: MXFP4 swapAB kernel launch failed" << std::endl;
    }
  };

  auto kernel_for = [&](auto NoBiasTag) {
    return &linear_mxfp4_swapAB_sm100_wrapper<
        T, MMA_N, OUTPUT_SIZE, REDUCTION_SIZE,
        decltype(mBias), decltype(NoBiasTag)::value,
        NUM_AB_STAGE_LAUNCH, NUM_ACC_STAGE_LAUNCH, NUM_C_STAGE_LAUNCH>;
  };

  if (residual_ptr != nullptr) {
    do_launch(kernel_for(std::false_type{}), cache.kernel_configured_bias);
  } else {
    do_launch(kernel_for(std::true_type{}),  cache.kernel_configured_no_bias);
  }
}

#define MIRAGE_MXFP4_SWAPAB_LAUNCH(OUT, K)                                      \
  case K:                                                                       \
    launch_linear_mxfp4_swapAB_sm100<T, MMA_N, OUT, K>(                        \
        input_ptr, input_sf_ptr, weight_ptr, weight_sf_ptr, output_ptr,         \
        residual_ptr, batch_size);                                              \
    return true;

#define MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(OUT)                                    \
  case OUT:                                                                     \
    switch (reduction_size) {                                                   \
      MIRAGE_MXFP4_SWAPAB_LAUNCH(OUT, 256)                                      \
      MIRAGE_MXFP4_SWAPAB_LAUNCH(OUT, 512)                                      \
      MIRAGE_MXFP4_SWAPAB_LAUNCH(OUT, 768)                                      \
      MIRAGE_MXFP4_SWAPAB_LAUNCH(OUT, 1024)                                     \
      MIRAGE_MXFP4_SWAPAB_LAUNCH(OUT, 1536)                                     \
      MIRAGE_MXFP4_SWAPAB_LAUNCH(OUT, 2048)                                     \
      MIRAGE_MXFP4_SWAPAB_LAUNCH(OUT, 4096)                                     \
      MIRAGE_MXFP4_SWAPAB_LAUNCH(OUT, 7168)                                     \
      default:                                                                  \
        return false;                                                           \
    }

template <typename T, int MMA_N>
bool dispatch_linear_mxfp4_swapAB_shape(int output_size,
                                        int reduction_size,
                                        int batch_size,
                                        void *input_ptr,
                                        void *input_sf_ptr,
                                        void *weight_ptr,
                                        void *weight_sf_ptr,
                                        void *output_ptr,
                                        void *residual_ptr) {
  switch (output_size) {
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(128)
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(256)
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(384)
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(512)
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(768)
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(1024)
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(1536)
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(2048)
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(4096)
    MIRAGE_MXFP4_SWAPAB_DISPATCH_KS(7168)
    default:
      return false;
  }
}

#undef MIRAGE_MXFP4_SWAPAB_DISPATCH_KS
#undef MIRAGE_MXFP4_SWAPAB_LAUNCH

template <typename T>
void dispatch_linear_mxfp4_swapAB(int output_size,
                                  int reduction_size,
                                  int batch_size,
                                  void *input_ptr,
                                  void *input_sf_ptr,
                                  void *weight_ptr,
                                  void *weight_sf_ptr,
                                  void *output_ptr,
                                  void *residual_ptr) {
  const int mma_n = 8;
  bool dispatched = false;
  switch (mma_n) {
    case 8:
      dispatched = dispatch_linear_mxfp4_swapAB_shape<T, 8>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr,
          weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 16:
      dispatched = dispatch_linear_mxfp4_swapAB_shape<T, 16>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr,
          weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 32:
      dispatched = dispatch_linear_mxfp4_swapAB_shape<T, 32>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr,
          weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 64:
      dispatched = dispatch_linear_mxfp4_swapAB_shape<T, 64>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr,
          weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    case 128:
      dispatched = dispatch_linear_mxfp4_swapAB_shape<T, 128>(
          output_size, reduction_size, batch_size, input_ptr, input_sf_ptr,
          weight_ptr, weight_sf_ptr, output_ptr, residual_ptr);
      break;
    default:
      break;
  }

  TORCH_CHECK(dispatched,
              "Small-M SM100 MXFP4 swapAB: unsupported shape (mma_n=", mma_n,
              ", N=", output_size, ", K=", reduction_size, ")");
}

namespace {

constexpr int SCALE_VEC_SIZE = 32;
constexpr int QUANTIZE_THREADS = 128;

template <typename T, int HIDDEN_SIZE>
__global__ __launch_bounds__(QUANTIZE_THREADS, 1)
void quantize_mxfp4_sm100_wrapper(T const *input_ptr,
                                  uint8_t *output_q_ptr,
                                  uint8_t *output_s_ptr,
                                  int batch_size,
                                  int mma_n) {
  kernel::quantize_mxfp4_sm100_task_impl<HIDDEN_SIZE,
                                         SCALE_VEC_SIZE,
                                         HIDDEN_SIZE,
                                         T>(
      input_ptr, output_q_ptr, output_s_ptr, batch_size, 1.0e-6f,
      /*min_4bit=*/-6.0f, /*max_4bit=*/6.0f,
      /*scale_outer_stride=*/32 * 4 * 4, mma_n);
}

template <int HIDDEN_SIZE>
std::vector<torch::Tensor> launch_quantize_mxfp4_sm100(torch::Tensor const &input,
                                                       int mma_n) {
  const int batch_size = static_cast<int>(input.size(0));
  const int padded_batch_size = ((batch_size + 127) / 128) * 128;
  const int sf_k_outer = HIDDEN_SIZE / 128;

  auto output_q = torch::empty({padded_batch_size, HIDDEN_SIZE / 2},
                               input.options().dtype(torch::kUInt8));

  at::Tensor output_s;
  if (mma_n > 0) {
    // swapAB path: per-tile scale factors [ceil(batch/mma_n), K/128, 32, 4, 4]
    const int num_n_tiles = (batch_size + mma_n - 1) / mma_n;
    output_s = torch::empty({num_n_tiles, sf_k_outer, 32, 4, 4},
                            input.options().dtype(torch::kUInt8));
  } else {
    // 1d2d path: interleaved scale factors [padded/128, K/128, 32, 4, 4]
    output_s = torch::empty({padded_batch_size / 128, sf_k_outer, 32, 4, 4},
                            input.options().dtype(torch::kUInt8));
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  quantize_mxfp4_sm100_wrapper<float, HIDDEN_SIZE>
      <<<dim3(padded_batch_size), dim3(QUANTIZE_THREADS), 0, stream>>>(
          static_cast<float const *>(input.data_ptr()),
          static_cast<uint8_t *>(output_q.data_ptr()),
          static_cast<uint8_t *>(output_s.data_ptr()),
          batch_size, mma_n);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "quantize_mxfp4_sm100 launch failed: ",
              cudaGetErrorString(err));
  return {output_q, output_s};
}

std::vector<torch::Tensor> dispatch_quantize_mxfp4_sm100(torch::Tensor const &input,
                                                         int mma_n = 0) {
  const int hidden_size = static_cast<int>(input.size(1));
  TORCH_CHECK(hidden_size % 64 == 0,
              "input.shape[1] must be divisible by 64");

  switch (hidden_size) {
    case 128:  return launch_quantize_mxfp4_sm100<128>(input, mma_n);
    case 256:  return launch_quantize_mxfp4_sm100<256>(input, mma_n);
    case 384:  return launch_quantize_mxfp4_sm100<384>(input, mma_n);
    case 512:  return launch_quantize_mxfp4_sm100<512>(input, mma_n);
    case 768:  return launch_quantize_mxfp4_sm100<768>(input, mma_n);
    case 1024: return launch_quantize_mxfp4_sm100<1024>(input, mma_n);
    case 1536: return launch_quantize_mxfp4_sm100<1536>(input, mma_n);
    case 2048: return launch_quantize_mxfp4_sm100<2048>(input, mma_n);
    case 4096: return launch_quantize_mxfp4_sm100<4096>(input, mma_n);
    case 7168: return launch_quantize_mxfp4_sm100<7168>(input, mma_n);
    default:
      TORCH_CHECK(
          false,
          "quantize_mxfp4_sm100 supports K in {128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 7168}. Got K=",
          hidden_size);
  }
}

template <int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void dispatch_one_shape(torch::Tensor const& input,
                        torch::Tensor const& input_sf,
                        torch::Tensor const& weight,
                        torch::Tensor const& weight_sf,
                        c10::optional<at::Tensor> const& residual,
                        torch::Tensor const& output) {
  launch_linear_mxfp4_1d2d_sm100<cute::float_e2m1_t,
                                 BATCH_SIZE,
                                 OUTPUT_SIZE,
                                 REDUCTION_SIZE>(
      input.data_ptr(),
      input_sf.data_ptr(),
      weight.data_ptr(),
      weight_sf.data_ptr(),
      output.data_ptr(),
      residual.has_value() ? residual->data_ptr() : nullptr);
}

// Compile-time supported (M, N, K) shapes. M and N must be multiples of 128,
// K must be a multiple of 256 (= MMA_K * NUM_MMA_K).
template <int M_, int N_, int K_>
struct mxfp4_shape { static constexpr int M = M_, N = N_, K = K_; };
using mxfp4_supported_shapes = std::tuple<
    mxfp4_shape<4096,  128,  768>, mxfp4_shape<4096,  128, 1024>,
    mxfp4_shape<4096,  128, 2048>, mxfp4_shape<4096,  128, 4096>,
    mxfp4_shape<4096,  256,  768>, mxfp4_shape<4096,  256, 1024>,
    mxfp4_shape<4096,  256, 2048>, mxfp4_shape<4096,  256, 4096>,
    mxfp4_shape<4096,  512, 1024>, mxfp4_shape<4096,  512, 2048>,
    mxfp4_shape<4096,  512, 4096>, mxfp4_shape<4096, 1024, 1024>,
    mxfp4_shape<4096, 1024, 2048>, mxfp4_shape<4096, 1024, 4096>,
    mxfp4_shape<4096, 2048, 2048>, mxfp4_shape<4096, 2048, 4096>,
    mxfp4_shape<4096, 4096, 4096>, mxfp4_shape<2048, 2048, 2048>,
    mxfp4_shape<2048, 4096, 4096>, mxfp4_shape<8192, 2048, 2048>,
    mxfp4_shape<8192, 4096, 4096>, mxfp4_shape<1024, 1024, 1024>,
    mxfp4_shape<1024, 2048, 2048>, mxfp4_shape<1024, 4096, 4096>>;

void launch_linear_mxfp4_small_batch(torch::Tensor const& input,
                                     torch::Tensor const& input_sf,
                                     torch::Tensor const& weight,
                                     torch::Tensor const& weight_sf,
                                     c10::optional<at::Tensor> const& residual,
                                     torch::Tensor const& output,
                                     int reduction_size,
                                     int batch_size) {
  TORCH_CHECK(batch_size >= 1 && batch_size <= 128,
              "launch_linear_mxfp4_small_batch supports 1 <= batch_size <= 128, got ", batch_size);
  const int output_size = static_cast<int>(weight.size(0));
  dispatch_linear_mxfp4_swapAB<cute::float_e2m1_t>(
      output_size,
      reduction_size,
      batch_size,
      input.data_ptr(),
      input_sf.data_ptr(),
      weight.data_ptr(),
      weight_sf.data_ptr(),
      output.data_ptr(),
      residual.has_value() ? residual->data_ptr() : nullptr);
}

void launch_linear_mxfp4_large_batch(torch::Tensor const& input,
                                     torch::Tensor const& input_sf,
                                     torch::Tensor const& weight,
                                     torch::Tensor const& weight_sf,
                                     c10::optional<at::Tensor> const& residual,
                                     torch::Tensor const& output,
                                     int batch_size,
                                     int output_size,
                                     int reduction_size) {
  bool dispatched = false;
  auto try_one = [&](auto shape_tag) {
    using S = decltype(shape_tag);
    constexpr int M_VAL = S::M, N_VAL = S::N, K_VAL = S::K;
    if (!dispatched && batch_size == M_VAL && output_size == N_VAL &&
        reduction_size == K_VAL) {
      dispatch_one_shape<M_VAL, N_VAL, K_VAL>(
          input, input_sf, weight, weight_sf, residual, output);
      dispatched = true;
    }
  };
  std::apply([&](auto... shape) { (try_one(shape), ...); },
             mxfp4_supported_shapes{});

  TORCH_CHECK(dispatched,
              "Large-batch SM100 MXFP4 path: unsupported shape (M=", batch_size,
              ", N=", output_size, ", K=", reduction_size, ")");
}

void dispatch_linear_mxfp4(torch::Tensor const& input,
                           torch::Tensor const& input_sf,
                           torch::Tensor const& weight,
                           torch::Tensor const& weight_sf,
                           c10::optional<at::Tensor> const& residual,
                           torch::Tensor const& output,
                           int batch_size,
                           int output_size,
                           int reduction_size) {
  if (batch_size <= 128) {
    launch_linear_mxfp4_small_batch(
        input, input_sf, weight, weight_sf, residual, output, reduction_size, batch_size);
  } else {
    launch_linear_mxfp4_large_batch(
        input, input_sf, weight, weight_sf, residual, output,
        batch_size, output_size, reduction_size);
  }
}

void validate_linear_tensors(torch::Tensor const& weight,
                             torch::Tensor const& weight_sf,
                             c10::optional<at::Tensor> const& residual,
                             torch::Tensor const& output,
                             int batch_size) {
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(weight_sf.is_cuda(), "weight_sf must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(weight.dim() == 2, "weight must be rank-2");
  TORCH_CHECK(output.dim() == 2, "output must be rank-2");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(weight_sf.is_contiguous(), "weight_sf must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8,
              "weight must have dtype uint8");
  TORCH_CHECK(weight_sf.scalar_type() == torch::kUInt8,
              "weight_sf must have dtype uint8");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32,
              "output must have dtype float32");
  TORCH_CHECK(output.size(0) == batch_size,
              "output.shape[0] must equal the logical batch size");
  TORCH_CHECK(output.size(1) == weight.size(0),
              "output.shape[1] must equal weight.shape[0]");
  if (residual.has_value()) {
    TORCH_CHECK(residual->is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(residual->scalar_type() == torch::kFloat32,
                "residual must have dtype float32");
    TORCH_CHECK(residual->sizes() == output.sizes(),
                "residual must have the same shape as output");
  }
}

void check_cuda_sync(char const *label) {
  cudaError_t err = cudaPeekAtLastError();
  TORCH_CHECK(err == cudaSuccess, label, ": ", cudaGetErrorString(err));
}

}  // namespace

std::vector<torch::Tensor> quantize_mxfp4_sm100_kernel(torch::Tensor input,
                                                       int64_t mma_n = 0) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "input must have dtype float32");
  return dispatch_quantize_mxfp4_sm100(input, static_cast<int>(mma_n));
}

void linear_mxfp4_sm100_no_quantization_kernel(torch::Tensor input,
                                               torch::Tensor input_sf,
                                               torch::Tensor weight,
                                               torch::Tensor weight_sf,
                                               c10::optional<at::Tensor> residual,
                                               torch::Tensor output) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input_sf.is_cuda(), "input_sf must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input_sf.is_contiguous(), "input_sf must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kUInt8,
              "input must have dtype uint8");
  TORCH_CHECK(input_sf.scalar_type() == torch::kUInt8,
              "input_sf must have dtype uint8");
  TORCH_CHECK(input.size(1) == weight.size(1),
              "input.shape[1] and weight.shape[1] must match");

  const int batch_size = static_cast<int>(output.size(0));
  const int output_size = static_cast<int>(weight.size(0));
  const int reduction_size = static_cast<int>(input.size(1) * 2);
  TORCH_CHECK(input.size(0) >= batch_size,
              "input must provide at least output.shape[0] rows");
  validate_linear_tensors(weight, weight_sf, residual, output, batch_size);
  dispatch_linear_mxfp4(
      input, input_sf, weight, weight_sf, residual, output,
      batch_size, output_size, reduction_size);
  check_cuda_sync("linear_mxfp4_sm100_no_quantization");
}

void linear_mxfp4_sm100_kernel(torch::Tensor input,
                               torch::Tensor weight,
                               torch::Tensor weight_sf,
                               c10::optional<at::Tensor> residual,
                               torch::Tensor output) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "input must have dtype float32");
  validate_linear_tensors(
      weight, weight_sf, residual, output, static_cast<int>(input.size(0)));
  TORCH_CHECK(weight.size(1) * 2 == input.size(1),
              "weight.shape[1] must equal input.shape[1] / 2");

  const int batch_size = static_cast<int>(input.size(0));
  const int output_size = static_cast<int>(weight.size(0));
  const int reduction_size = static_cast<int>(input.size(1));
  const int mma_n = (batch_size <= 128) ? 8 : 0;
  auto quantized_input = dispatch_quantize_mxfp4_sm100(input, mma_n);
  dispatch_linear_mxfp4(
      quantized_input[0], quantized_input[1], weight, weight_sf, residual, output,
      batch_size, output_size, reduction_size);
  check_cuda_sync("linear_mxfp4_sm100");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_mxfp4_sm100", &quantize_mxfp4_sm100_kernel,
        "SM100 MXFP4 quantize. mma_n=0 -> interleaved layout [padded/128, K/128, 32, 4, 4]; mma_n>0 -> per-tile swapAB layout [ceil(batch/mma_n), K/128, 32, 4, 4].",
        pybind11::arg("input"), pybind11::arg("mma_n") = 0);
  m.def("linear_mxfp4_sm100_no_quantization", &linear_mxfp4_sm100_no_quantization_kernel,
        "SM100 MXFP4 linear entry point expecting uint8 activations plus activation scale factors in the layout produced by quantize_mxfp4_sm100 for the target batch size.");
  m.def("linear_mxfp4_sm100", &linear_mxfp4_sm100_kernel,
        "SM100 MXFP4 linear entry point that quantizes float32 activations before dispatching to the no-quantization kernel.");
}

#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
