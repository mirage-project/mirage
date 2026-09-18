// BF16 MoE Group GEMM — DeepSeek V3 Configuration Test Wrapper
//
// Wraps the existing BF16 moe_linear_sm100 kernel with DeepSeek V3 dims
// and 2D grid support (expert_stride x n_splits), matching the FP8 wrapper.
//
// This file is standalone — does not modify the original sm100_moe test.

#include <cassert>
#include <cstdio>

// task_header.cuh includes all SM100 kernels and sets up the CuTe include
// chain correctly (Copy_Atom, SM80_CP_ASYNC, etc.) for moe_linear_sm100.cuh.
// NOTE: If this fails to compile due to mla_prefill_sm100.cuh errors, the
// BF16 module can be safely skipped — the benchmark handles HAS_BF16=False.
#include "runtime_header.h"
#include "tasks/blackwell/task_header.cuh"
#include "tasks/hopper/tma_2d.cuh"
#include "tma.cuh"
#include <cuda_runtime.h>

#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/util/print_error.hpp>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

using bfloat16 = cute::bfloat16_t;

// ================================================================
// DeepSeek V3 W13 dimensions
// ================================================================
constexpr int DSV3_MMA_M = 128;
constexpr int DSV3_MMA_N = 16; // BF16 kernel uses MMA_N=16
// BF16 kernel's CuTe tiled copy requires BATCH_SIZE <= MMA_N (single n_tile).
// Use 16 to match MMA_N. Actual token count (1-16) is bounded by this.
constexpr int DSV3_BATCH_SIZE = 16;
constexpr int DSV3_OUTPUT_SIZE = 4096;
constexpr int DSV3_REDUCTION_SIZE = 7168;
constexpr int DSV3_NUM_EXPERTS = 256;
constexpr int DSV3_NUM_TOPK = 8;
constexpr int DSV3_NUM_AB_STAGE = 8;
constexpr int DSV3_NUM_ACC_STAGE = 2;
constexpr int DSV3_NUM_C_STAGE = 4;

constexpr int MAX_N_SPLITS = 32;

// ================================================================
// 2D grid BF16 kernel wrapper
//
// grid = (expert_stride, n_splits, 1)
// Each CTA adjusts weight/output pointers based on blockIdx.y.
// ================================================================
template <int EXPERT_STRIDE, int OUTPUT_SIZE_PER_CTA, int ORIG_OUTPUT_SIZE>
__global__ __launch_bounds__(256, 1) void dsv3_bf16_moe_2d_kernel(
    CUtensorMap const *const *__restrict__ tma_desc_array,
    bfloat16 const *input,
    bfloat16 const *bias_ptr, // nullptr for no-bias
    int32_t const *routing_indices,
    int32_t const *mask,
    bfloat16 *output_base) {
  constexpr int B = 3, M = 3, S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  using TMA_A = kernel::tma::tma_2d<bfloat16,
                                    B,
                                    M,
                                    S,
                                    (DSV3_NUM_EXPERTS - 1) * ORIG_OUTPUT_SIZE +
                                        OUTPUT_SIZE_PER_CTA,
                                    DSV3_REDUCTION_SIZE,
                                    DSV3_MMA_M,
                                    TMA_CP_ASYNC_SIZE,
                                    DSV3_REDUCTION_SIZE,
                                    1,
                                    1,
                                    TMA_CP_ASYNC_REPEAT_COL,
                                    DSV3_MMA_M * TMA_CP_ASYNC_SIZE,
                                    true>;

  TMA_A tma_a(const_cast<CUtensorMap *>(tma_desc_array[blockIdx.y]));

  // Offset output by blockIdx.y's N-slice
  bfloat16 *output = output_base + (size_t)OUTPUT_SIZE_PER_CTA * blockIdx.y;

  // Input: [BATCH, K] — no offset (broadcast across y)
  auto mInput = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<bfloat16 *>(input)),
      cute::make_layout(
          cute::make_shape(cute::Int<DSV3_BATCH_SIZE>{},
                           cute::Int<DSV3_REDUCTION_SIZE>{}),
          cute::make_stride(cute::Int<DSV3_REDUCTION_SIZE>{}, cute::Int<1>{})));

  // Bias: nullptr (NoBias=true)
  auto mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<bfloat16 *>(nullptr)),
      cute::make_layout(
          cute::make_shape(cute::Int<DSV3_BATCH_SIZE>{},
                           cute::Int<OUTPUT_SIZE_PER_CTA>{},
                           cute::Int<DSV3_NUM_EXPERTS>{}),
          cute::make_stride(
              cute::Int<OUTPUT_SIZE_PER_CTA>{},
              cute::Int<1>{},
              cute::Int<DSV3_BATCH_SIZE * OUTPUT_SIZE_PER_CTA>{})));

  // Routing indices: [experts, batch]
  auto mRoutingIndices = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<int32_t *>(routing_indices)),
      cute::make_layout(
          cute::make_shape(cute::Int<DSV3_NUM_EXPERTS>{},
                           cute::Int<DSV3_BATCH_SIZE>{}),
          cute::make_stride(cute::Int<DSV3_BATCH_SIZE>{}, cute::Int<1>{})));

  // Mask: [experts+1]
  auto mMask = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<int32_t *>(mask)),
      cute::make_layout(cute::make_shape(cute::Int<DSV3_NUM_EXPERTS + 1>{}),
                        cute::make_stride(cute::Int<1>{})));

  // Output: [BATCH, TOPK, OUTPUT_PER_CTA] with stride [TOPK*ORIG, ORIG, 1]
  auto mOutput = cute::make_tensor(
      cute::make_gmem_ptr(output),
      cute::make_layout(
          cute::make_shape(cute::Int<DSV3_BATCH_SIZE>{},
                           cute::Int<DSV3_NUM_TOPK>{},
                           cute::Int<OUTPUT_SIZE_PER_CTA>{}),
          cute::make_stride(cute::Int<DSV3_NUM_TOPK * ORIG_OUTPUT_SIZE>{},
                            cute::Int<ORIG_OUTPUT_SIZE>{},
                            cute::Int<1>{})));

  kernel::moe_linear_sm100_task_impl<bfloat16,
                                     TMA_A,
                                     decltype(mInput),
                                     decltype(mBias),
                                     decltype(mRoutingIndices),
                                     decltype(mMask),
                                     decltype(mOutput),
                                     DSV3_MMA_M,
                                     DSV3_MMA_N,
                                     DSV3_BATCH_SIZE,
                                     OUTPUT_SIZE_PER_CTA,
                                     ORIG_OUTPUT_SIZE,
                                     DSV3_REDUCTION_SIZE,
                                     DSV3_NUM_EXPERTS,
                                     DSV3_NUM_TOPK,
                                     EXPERT_STRIDE,
                                     true, // W13_LINEAR
                                     true, // NoBias
                                     DSV3_NUM_AB_STAGE,
                                     DSV3_NUM_ACC_STAGE,
                                     DSV3_NUM_C_STAGE>(
      tma_a,
      mInput,
      mBias,
      mRoutingIndices,
      mMask,
      mOutput,
      static_cast<int>(blockIdx.x));
}

// ================================================================
// TMA descriptor creation for BF16
// ================================================================
static CUtensorMap **create_bf16_tma_desc_array(void *weight_base_ptr,
                                                int n_splits,
                                                int output_per_cta,
                                                int orig_output_size) {
  constexpr int B = 3, M = 3, S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  size_t w_smem_repeat_col =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  int total_rows = (DSV3_NUM_EXPERTS - 1) * orig_output_size + output_per_cta;

  CUtensorMap host_descs[MAX_N_SPLITS];
  for (int y = 0; y < n_splits; ++y) {
    bfloat16 *w_ptr = reinterpret_cast<bfloat16 *>(weight_base_ptr) +
                      (size_t)y * output_per_cta * DSV3_REDUCTION_SIZE;

    uint64_t gmem_shape[2] = {(uint64_t)total_rows,
                              (uint64_t)DSV3_REDUCTION_SIZE};
    uint64_t gmem_stride[2] = {1, (uint64_t)DSV3_REDUCTION_SIZE};
    uint32_t smem_shape[2] = {(uint32_t)DSV3_MMA_M,
                              (uint32_t)TMA_CP_ASYNC_SIZE};

    mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 2>(&host_descs[y],
                                                         w_ptr,
                                                         gmem_shape,
                                                         gmem_stride,
                                                         smem_shape,
                                                         1,
                                                         w_smem_repeat_col);
  }

  CUtensorMap *dev_descs;
  cudaMalloc(&dev_descs, n_splits * sizeof(CUtensorMap));
  cudaMemcpy(dev_descs,
             host_descs,
             n_splits * sizeof(CUtensorMap),
             cudaMemcpyHostToDevice);

  CUtensorMap *host_ptr_array[MAX_N_SPLITS];
  for (int y = 0; y < n_splits; ++y) {
    host_ptr_array[y] = dev_descs + y;
  }

  CUtensorMap **dev_ptr_array;
  cudaMalloc(&dev_ptr_array, n_splits * sizeof(CUtensorMap *));
  cudaMemcpy(dev_ptr_array,
             host_ptr_array,
             n_splits * sizeof(CUtensorMap *),
             cudaMemcpyHostToDevice);

  return dev_ptr_array;
}

// ================================================================
// Benchmark API: setup / launch / cleanup
// ================================================================
static CUtensorMap **g_bf16_tma_array = nullptr;
static int g_bf16_n_splits = 0;
static int g_bf16_expert_stride = 0;

template <int EXPERT_STRIDE, int N_SPLITS>
void bf16_bench_setup_impl(torch::Tensor weight) {
  constexpr int OUTPUT_PER_CTA = DSV3_OUTPUT_SIZE / N_SPLITS;
  constexpr int smem_size = 224 * 1024;

  g_bf16_tma_array = create_bf16_tma_desc_array(
      weight.data_ptr(), N_SPLITS, OUTPUT_PER_CTA, DSV3_OUTPUT_SIZE);
  g_bf16_n_splits = N_SPLITS;
  g_bf16_expert_stride = EXPERT_STRIDE;

  cudaFuncSetAttribute(
      dsv3_bf16_moe_2d_kernel<EXPERT_STRIDE, OUTPUT_PER_CTA, DSV3_OUTPUT_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
}

void bf16_moe_bench_setup(torch::Tensor weight,
                          int expert_stride,
                          int n_splits) {
  c10::cuda::CUDAGuard guard(weight.device());
  if (expert_stride == 8 && n_splits == 1) {
    bf16_bench_setup_impl<8, 1>(weight);
  } else if (expert_stride == 8 && n_splits == 16) {
    bf16_bench_setup_impl<8, 16>(weight);
  } else if (expert_stride == 16 && n_splits == 8) {
    bf16_bench_setup_impl<16, 8>(weight);
  } else if (expert_stride == 32 && n_splits == 4) {
    bf16_bench_setup_impl<32, 4>(weight);
  } else {
    printf("ERROR: bf16 bench_setup not supported for (%d, %d)\n",
           expert_stride,
           n_splits);
  }
}

template <int EXPERT_STRIDE, int N_SPLITS>
void bf16_bench_launch_impl(torch::Tensor input,
                            torch::Tensor routing,
                            torch::Tensor mask,
                            torch::Tensor output) {
  constexpr int OUTPUT_PER_CTA = DSV3_OUTPUT_SIZE / N_SPLITS;
  constexpr int smem_size = 224 * 1024;

  dsv3_bf16_moe_2d_kernel<EXPERT_STRIDE, OUTPUT_PER_CTA, DSV3_OUTPUT_SIZE>
      <<<dim3(EXPERT_STRIDE, N_SPLITS, 1), dim3(256, 1, 1), smem_size>>>(
          g_bf16_tma_array,
          reinterpret_cast<bfloat16 *>(input.data_ptr()),
          nullptr,
          reinterpret_cast<int32_t *>(routing.data_ptr()),
          reinterpret_cast<int32_t *>(mask.data_ptr()),
          reinterpret_cast<bfloat16 *>(output.data_ptr()));
}

void bf16_moe_bench_launch(torch::Tensor input,
                           torch::Tensor routing,
                           torch::Tensor mask,
                           torch::Tensor output) {
  assert(g_bf16_tma_array && "Call bf16_moe_bench_setup first");
  if (g_bf16_expert_stride == 8 && g_bf16_n_splits == 1) {
    bf16_bench_launch_impl<8, 1>(input, routing, mask, output);
  } else if (g_bf16_expert_stride == 8 && g_bf16_n_splits == 16) {
    bf16_bench_launch_impl<8, 16>(input, routing, mask, output);
  } else if (g_bf16_expert_stride == 16 && g_bf16_n_splits == 8) {
    bf16_bench_launch_impl<16, 8>(input, routing, mask, output);
  } else if (g_bf16_expert_stride == 32 && g_bf16_n_splits == 4) {
    bf16_bench_launch_impl<32, 4>(input, routing, mask, output);
  }
}

void bf16_moe_bench_cleanup() {
  if (g_bf16_tma_array) {
    CUtensorMap *first_desc;
    cudaMemcpy(&first_desc,
               g_bf16_tma_array,
               sizeof(CUtensorMap *),
               cudaMemcpyDeviceToHost);
    cudaFree(first_desc);
    cudaFree(g_bf16_tma_array);
    g_bf16_tma_array = nullptr;
  }
}

// ================================================================
// Correctness test: single CTA
// ================================================================
void bf16_moe_gemm_test(torch::Tensor input,
                        torch::Tensor weight,
                        torch::Tensor routing,
                        torch::Tensor mask,
                        torch::Tensor output) {
  c10::cuda::CUDAGuard guard(input.device());

  CUtensorMap **tma_array = create_bf16_tma_desc_array(
      weight.data_ptr(), 1, DSV3_OUTPUT_SIZE, DSV3_OUTPUT_SIZE);

  constexpr int smem_size = 224 * 1024;
  cudaFuncSetAttribute(
      dsv3_bf16_moe_2d_kernel<1, DSV3_OUTPUT_SIZE, DSV3_OUTPUT_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  dsv3_bf16_moe_2d_kernel<1, DSV3_OUTPUT_SIZE, DSV3_OUTPUT_SIZE>
      <<<dim3(1, 1, 1), dim3(256, 1, 1), smem_size>>>(
          tma_array,
          reinterpret_cast<bfloat16 *>(input.data_ptr()),
          nullptr,
          reinterpret_cast<int32_t *>(routing.data_ptr()),
          reinterpret_cast<int32_t *>(mask.data_ptr()),
          reinterpret_cast<bfloat16 *>(output.data_ptr()));

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  CUtensorMap *first;
  cudaMemcpy(&first, tma_array, sizeof(CUtensorMap *), cudaMemcpyDeviceToHost);
  cudaFree(first);
  cudaFree(tma_array);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_moe_gemm_test",
        &bf16_moe_gemm_test,
        "BF16 MoE GEMM - single CTA test");
  m.def("bf16_moe_bench_setup",
        &bf16_moe_bench_setup,
        "Setup TMA for BF16 benchmark");
  m.def("bf16_moe_bench_launch",
        &bf16_moe_bench_launch,
        "Async launch BF16 MoE benchmark");
  m.def("bf16_moe_bench_cleanup",
        &bf16_moe_bench_cleanup,
        "Cleanup BF16 benchmark TMA");
}
