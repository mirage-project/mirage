// FP8 MoE Group GEMM — DeepSeek V3 Configuration Test Wrapper
//
// DeepSeek V3 MoE:
//   num_experts = 256, top_k = 8
//   hidden_size = 7168 (K for W13, N for W2)
//   intermediate_size = 2048
//   W13: [M, 7168] @ [256, 4096, 7168] -> [M, topk, 4096]
//
// 2D grid launch: grid=(expert_stride, n_splits, 1)
//   blockIdx.x → expert_offset (which experts this CTA handles)
//   blockIdx.y → N-slice index (which output rows this CTA computes)
//
// Each CTA adjusts its own weight/scale/output pointers based on blockIdx.y,
// emulating the MPK runtime's per-CTA pointer offset logic from runtime.cc.

#include <cassert>
#include <cstdio>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "tasks/blackwell/fp8_group_gemm_sm100.cuh"
#include "tasks/hopper/smem_layout_tma.cuh"
#include "tasks/hopper/tma.cuh"
#include "tma.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// ================================================================
// DeepSeek V3 W13 dimensions
// ================================================================
constexpr int DSV3_MMA_M = 128;
constexpr int DSV3_MMA_N = 16; // matches production task_register (MMA_N=16)
constexpr int DSV3_BATCH_SIZE = 16;    // actual max tokens (no padding waste)
constexpr int DSV3_OUTPUT_SIZE = 4096; // full N = 2 * intermediate_size
constexpr int DSV3_REDUCTION_SIZE = 7168; // hidden_size = K
constexpr int DSV3_NUM_EXPERTS = 256;
constexpr int DSV3_NUM_TOPK = 8;
constexpr int DSV3_NUM_AB_STAGE = 4; // matches BF16 kernel for fair comparison
constexpr int DSV3_NUM_ACC_STAGE = 2;
constexpr int DSV3_NUM_C_STAGE = 4;
constexpr int DSV3_K_SCALE = DSV3_REDUCTION_SIZE / 128; // 56

// Maximum number of N-splits (for TMA descriptor array sizing)
constexpr int MAX_N_SPLITS = 32;

// ================================================================
// CuTe type aliases (shared across all kernels)
// ================================================================

template <int OUTPUT_SIZE_, int ORIG_OUTPUT_SIZE_>
using FP8TMA = kernel::tma::tma_2d<uint8_t,
                                   3,
                                   3,
                                   3,
                                   (DSV3_NUM_EXPERTS - 1) * ORIG_OUTPUT_SIZE_ +
                                       OUTPUT_SIZE_,
                                   DSV3_REDUCTION_SIZE,
                                   DSV3_MMA_M,
                                   128,
                                   DSV3_REDUCTION_SIZE,
                                   1,
                                   1,
                                   1,
                                   DSV3_MMA_M * 128,
                                   true>;

using InputLayout = cute::Layout<
    cute::Shape<cute::Int<DSV3_BATCH_SIZE>, cute::Int<DSV3_REDUCTION_SIZE>>,
    cute::Stride<cute::Int<DSV3_REDUCTION_SIZE>, cute::Int<1>>>;
using InputTensor =
    cute::Tensor<cute::ViewEngine<cute::gmem_ptr<uint8_t *>>, InputLayout>;

using InputScaleLayout = cute::Layout<
    cute::Shape<cute::Int<DSV3_BATCH_SIZE>, cute::Int<DSV3_K_SCALE>>,
    cute::Stride<cute::Int<DSV3_K_SCALE>, cute::Int<1>>>;
using InputScaleTensor =
    cute::Tensor<cute::ViewEngine<cute::gmem_ptr<float *>>, InputScaleLayout>;

using IndicesLayout = cute::Layout<
    cute::Shape<cute::Int<DSV3_NUM_EXPERTS>, cute::Int<DSV3_BATCH_SIZE>>,
    cute::Stride<cute::Int<DSV3_BATCH_SIZE>, cute::Int<1>>>;
using IndicesTensor =
    cute::Tensor<cute::ViewEngine<cute::gmem_ptr<cute::int32_t *>>,
                 IndicesLayout>;

using MaskLayout = cute::Layout<cute::Shape<cute::Int<DSV3_NUM_EXPERTS + 1>>,
                                cute::Stride<cute::Int<1>>>;
using MaskTensor =
    cute::Tensor<cute::ViewEngine<cute::gmem_ptr<cute::int32_t *>>, MaskLayout>;

// ================================================================
// 2D grid kernel wrapper
//
// grid = (expert_stride, n_splits, 1)
// Each CTA:
//   - Uses blockIdx.x as expert_offset (EXPERT_STRIDE = gridDim.x)
//   - Uses blockIdx.y to offset weight/scale/output pointers (N-split)
//   - Reads its TMA descriptor from a per-y-slice array
//
// This emulates the MPK runtime's per-CTA pointer adjustment logic
// from runtime.cc lines 1026-1042.
// ================================================================
template <int EXPERT_STRIDE, int OUTPUT_SIZE_PER_CTA, int ORIG_OUTPUT_SIZE>
__global__ __launch_bounds__(256, 1) void dsv3_fp8_moe_2d_kernel(
    CUtensorMap const *const *__restrict__ tma_desc_array, // [n_splits]
    uint8_t const *input_fp8,
    float const *input_scale,
    float const *weight_scale_base, // base pointer (before y-offset)
    cute::int32_t const *routing_indices,
    cute::int32_t const *mask,
    cute::bfloat16_t *output_base, // base pointer (before y-offset)
    uint8_t const
        *weight_fp8_base) // only used for reference, TMA handles actual loads
{
  // ---- Per-CTA pointer offsets based on blockIdx.y ----
  // This mirrors MPK runtime's offset logic:
  //   offset = block_size * bid.y * stride[mapped_dim]
  //
  // weight_fp8 [E, N, K]: input_map.y=1 (split dim 1=N)
  //   block_size = N / n_splits = OUTPUT_SIZE_PER_CTA
  //   stride[1] = K (row-major)
  //   offset = OUTPUT_SIZE_PER_CTA * blockIdx.y * K  (in elements = bytes for
  //   uint8) (Handled by per-y TMA descriptor, so no explicit offset needed
  //   here)
  //
  // weight_scale [E, N, K/128]: input_map.y=1 (split dim 1=N)
  //   block_size = N / n_splits = OUTPUT_SIZE_PER_CTA
  //   stride[1] = K_SCALE
  //   offset = OUTPUT_SIZE_PER_CTA * blockIdx.y * K_SCALE  (in float elements)
  float const *weight_scale = weight_scale_base + (size_t)OUTPUT_SIZE_PER_CTA *
                                                      blockIdx.y * DSV3_K_SCALE;

  // output [B, topk, N]: input_map.y=2 (split dim 2=N)
  //   block_size = N / n_splits = OUTPUT_SIZE_PER_CTA
  //   stride[2] = 1
  //   offset = OUTPUT_SIZE_PER_CTA * blockIdx.y * 1  (in bfloat16 elements)
  cute::bfloat16_t *output =
      output_base + (size_t)OUTPUT_SIZE_PER_CTA * blockIdx.y;

  // ---- Build CuTe tensor views with adjusted pointers ----
  using TMA_t = FP8TMA<OUTPUT_SIZE_PER_CTA, ORIG_OUTPUT_SIZE>;
  TMA_t tma_weight(const_cast<CUtensorMap *>(tma_desc_array[blockIdx.y]));

  auto mInput = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<uint8_t *>(input_fp8)), InputLayout{});
  auto mInputScale =
      cute::make_tensor(cute::make_gmem_ptr(const_cast<float *>(input_scale)),
                        InputScaleLayout{});

  constexpr int WS_ROWS =
      (DSV3_NUM_EXPERTS - 1) * ORIG_OUTPUT_SIZE + OUTPUT_SIZE_PER_CTA;
  auto mWeightScale = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<float *>(weight_scale)),
      cute::make_layout(
          cute::make_shape(cute::Int<WS_ROWS>{}, cute::Int<DSV3_K_SCALE>{}),
          cute::make_stride(cute::Int<DSV3_K_SCALE>{}, cute::Int<1>{})));

  auto mRoutingIndices = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t *>(routing_indices)),
      IndicesLayout{});
  auto mMask = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t *>(mask)), MaskLayout{});

  // Output layout: kernel indexes mOutput(n_idx, topk_idx-1, m_idx)
  //   dim 0 = batch (stride topk*ORIG)
  //   dim 1 = topk  (stride ORIG)
  //   dim 2 = output_row (stride 1, contiguous)
  auto mOutput = cute::make_tensor(
      cute::make_gmem_ptr(output),
      cute::make_layout(
          cute::make_shape(cute::Int<DSV3_BATCH_SIZE>{},
                           cute::Int<DSV3_NUM_TOPK>{},
                           cute::Int<OUTPUT_SIZE_PER_CTA>{}),
          cute::make_stride(cute::Int<DSV3_NUM_TOPK * ORIG_OUTPUT_SIZE>{},
                            cute::Int<ORIG_OUTPUT_SIZE>{},
                            cute::Int<1>{})));

  // expert_offset = blockIdx.x (EXPERT_STRIDE = gridDim.x)
  kernel::fp8_moe_group_gemm_sm100_task_impl<TMA_t,
                                             decltype(mInput),
                                             decltype(mInputScale),
                                             decltype(mWeightScale),
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
                                             DSV3_NUM_AB_STAGE,
                                             DSV3_NUM_ACC_STAGE,
                                             DSV3_NUM_C_STAGE>(
      tma_weight,
      mInput,
      mInputScale,
      mWeightScale,
      mRoutingIndices,
      mMask,
      mOutput,
      static_cast<int>(blockIdx.x));
}

// ================================================================
// Single-CTA kernel (no N-split, for correctness baseline)
// ================================================================
__global__ __launch_bounds__(256, 1) void dsv3_fp8_moe_1cta_kernel(
    CUtensorMap const *__restrict__ tma_weight_desc,
    uint8_t const *input_fp8,
    float const *input_scale,
    float const *weight_scale,
    cute::int32_t const *routing_indices,
    cute::int32_t const *mask,
    cute::bfloat16_t *output,
    int expert_offset) {
  using TMA_t = FP8TMA<DSV3_OUTPUT_SIZE, DSV3_OUTPUT_SIZE>;
  TMA_t tma_weight(const_cast<CUtensorMap *>(tma_weight_desc));

  auto mInput = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<uint8_t *>(input_fp8)), InputLayout{});
  auto mInputScale =
      cute::make_tensor(cute::make_gmem_ptr(const_cast<float *>(input_scale)),
                        InputScaleLayout{});

  constexpr int WS_ROWS = DSV3_NUM_EXPERTS * DSV3_OUTPUT_SIZE;
  auto mWeightScale = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<float *>(weight_scale)),
      cute::make_layout(
          cute::make_shape(cute::Int<WS_ROWS>{}, cute::Int<DSV3_K_SCALE>{}),
          cute::make_stride(cute::Int<DSV3_K_SCALE>{}, cute::Int<1>{})));

  auto mRoutingIndices = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t *>(routing_indices)),
      IndicesLayout{});
  auto mMask = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t *>(mask)), MaskLayout{});

  // Output layout: kernel indexes mOutput(n_idx, topk_idx-1, m_idx)
  auto mOutput = cute::make_tensor(
      cute::make_gmem_ptr(output),
      cute::make_layout(
          cute::make_shape(cute::Int<DSV3_BATCH_SIZE>{},
                           cute::Int<DSV3_NUM_TOPK>{},
                           cute::Int<DSV3_OUTPUT_SIZE>{}),
          cute::make_stride(cute::Int<DSV3_NUM_TOPK * DSV3_OUTPUT_SIZE>{},
                            cute::Int<DSV3_OUTPUT_SIZE>{},
                            cute::Int<1>{})));

  kernel::fp8_moe_group_gemm_sm100_task_impl<TMA_t,
                                             decltype(mInput),
                                             decltype(mInputScale),
                                             decltype(mWeightScale),
                                             decltype(mRoutingIndices),
                                             decltype(mMask),
                                             decltype(mOutput),
                                             DSV3_MMA_M,
                                             DSV3_MMA_N,
                                             DSV3_BATCH_SIZE,
                                             DSV3_OUTPUT_SIZE,
                                             DSV3_OUTPUT_SIZE,
                                             DSV3_REDUCTION_SIZE,
                                             DSV3_NUM_EXPERTS,
                                             DSV3_NUM_TOPK,
                                             1, // EXPERT_STRIDE=1 (single CTA)
                                             true, // W13_LINEAR
                                             DSV3_NUM_AB_STAGE,
                                             DSV3_NUM_ACC_STAGE,
                                             DSV3_NUM_C_STAGE>(tma_weight,
                                                               mInput,
                                                               mInputScale,
                                                               mWeightScale,
                                                               mRoutingIndices,
                                                               mMask,
                                                               mOutput,
                                                               expert_offset);
}

// ================================================================
// TMA descriptor creation
// ================================================================
CUtensorMap *create_tma_desc(void *weight_ptr, int total_rows, int cols) {
  constexpr int B = 3, M = 3, S = 3;
  constexpr int bK = 128;
  uint64_t gmem_shape[2] = {(uint64_t)total_rows, (uint64_t)cols};
  uint64_t gmem_stride[2] = {1, (uint64_t)cols};
  uint32_t smem_shape[2] = {(uint32_t)DSV3_MMA_M, (uint32_t)bK};

  CUtensorMap host_desc;
  mirage::runtime::fill_tma_desc<uint8_t, B, M, S, 2>(
      &host_desc, weight_ptr, gmem_shape, gmem_stride, smem_shape, 1, 1);

  CUtensorMap *dev_desc;
  cudaMalloc(&dev_desc, sizeof(CUtensorMap));
  cudaMemcpy(dev_desc, &host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  return dev_desc;
}

// Create an array of TMA descriptors, one per N-split slice.
// Each descriptor has a different base pointer (offset by y * output_per_cta *
// K).
CUtensorMap **create_tma_desc_array(void *weight_base_ptr,
                                    int n_splits,
                                    int output_per_cta,
                                    int orig_output_size) {
  CUtensorMap host_descs[MAX_N_SPLITS];
  int total_rows = (DSV3_NUM_EXPERTS - 1) * orig_output_size + output_per_cta;

  for (int y = 0; y < n_splits; ++y) {
    uint8_t *w_ptr = reinterpret_cast<uint8_t *>(weight_base_ptr) +
                     (size_t)y * output_per_cta * DSV3_REDUCTION_SIZE;

    constexpr int B = 3, M = 3, S = 3;
    constexpr int bK = 128;
    uint64_t gmem_shape[2] = {(uint64_t)total_rows,
                              (uint64_t)DSV3_REDUCTION_SIZE};
    uint64_t gmem_stride[2] = {1, (uint64_t)DSV3_REDUCTION_SIZE};
    uint32_t smem_shape[2] = {(uint32_t)DSV3_MMA_M, (uint32_t)bK};

    mirage::runtime::fill_tma_desc<uint8_t, B, M, S, 2>(
        &host_descs[y], w_ptr, gmem_shape, gmem_stride, smem_shape, 1, 1);
  }

  // Copy all descriptors to device as a contiguous array
  CUtensorMap *dev_descs;
  cudaMalloc(&dev_descs, n_splits * sizeof(CUtensorMap));
  cudaMemcpy(dev_descs,
             host_descs,
             n_splits * sizeof(CUtensorMap),
             cudaMemcpyHostToDevice);

  // Create device array of pointers (each pointing to the corresponding
  // descriptor)
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
// Shared memory size
// ================================================================
static constexpr int compute_smem_size() {
  constexpr int bK = 128;
  constexpr int smem_A = DSV3_NUM_AB_STAGE * DSV3_MMA_M * bK;
  constexpr int smem_B = DSV3_NUM_AB_STAGE * DSV3_MMA_N * bK;
  constexpr int smem_SF = 2 * DSV3_NUM_AB_STAGE * 128 * 4;
  constexpr int smem_barriers = 8 * DSV3_NUM_AB_STAGE * 8 +
                                2 * DSV3_NUM_ACC_STAGE * 8 +
                                DSV3_NUM_EXPERTS * 4 + 4 + 128;
  return smem_A + smem_B + smem_SF + smem_barriers + 4096;
}

// ================================================================
// Python API: single-CTA correctness test
// ================================================================
void fp8_moe_gemm_test(torch::Tensor input_fp8,
                       torch::Tensor input_scale,
                       torch::Tensor weight_fp8,
                       torch::Tensor weight_scale,
                       torch::Tensor routing_indices,
                       torch::Tensor mask,
                       torch::Tensor output,
                       int expert_offset) {
  c10::cuda::CUDAGuard guard(input_fp8.device());
  int total_rows = weight_fp8.size(0) * weight_fp8.size(1);
  int cols = weight_fp8.size(2);

  CUtensorMap *tma = create_tma_desc(weight_fp8.data_ptr(), total_rows, cols);
  constexpr int smem_size = compute_smem_size();
  cudaFuncSetAttribute(dsv3_fp8_moe_1cta_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  dsv3_fp8_moe_1cta_kernel<<<dim3(1, 1, 1), dim3(256, 1, 1), smem_size>>>(
      tma,
      reinterpret_cast<uint8_t *>(input_fp8.data_ptr()),
      reinterpret_cast<float *>(input_scale.data_ptr()),
      reinterpret_cast<float *>(weight_scale.data_ptr()),
      reinterpret_cast<cute::int32_t *>(routing_indices.data_ptr()),
      reinterpret_cast<cute::int32_t *>(mask.data_ptr()),
      reinterpret_cast<cute::bfloat16_t *>(output.data_ptr()),
      expert_offset);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
  cudaFree(tma);
}

// ================================================================
// Python API: 2D grid launch (single kernel, proper CTA parallelism)
//
// Launches grid=(expert_stride, n_splits, 1) CTAs in a SINGLE kernel.
// Each CTA adjusts its own pointers based on blockIdx.y.
// TMA descriptors are pre-created per y-slice.
// ================================================================

// Dispatch helper: instantiate and launch for given (expert_stride, n_splits)
template <int EXPERT_STRIDE, int N_SPLITS>
void launch_2d_impl(torch::Tensor input_fp8,
                    torch::Tensor input_scale,
                    torch::Tensor weight_fp8,
                    torch::Tensor weight_scale,
                    torch::Tensor routing_indices,
                    torch::Tensor mask,
                    torch::Tensor output) {
  constexpr int OUTPUT_PER_CTA = DSV3_OUTPUT_SIZE / N_SPLITS;
  constexpr int smem_size = compute_smem_size();

  // Create per-y TMA descriptor array
  CUtensorMap **tma_array = create_tma_desc_array(
      weight_fp8.data_ptr(), N_SPLITS, OUTPUT_PER_CTA, DSV3_OUTPUT_SIZE);

  cudaFuncSetAttribute(
      dsv3_fp8_moe_2d_kernel<EXPERT_STRIDE, OUTPUT_PER_CTA, DSV3_OUTPUT_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  // Single launch with full 2D grid
  dsv3_fp8_moe_2d_kernel<EXPERT_STRIDE, OUTPUT_PER_CTA, DSV3_OUTPUT_SIZE>
      <<<dim3(EXPERT_STRIDE, N_SPLITS, 1), dim3(256, 1, 1), smem_size>>>(
          tma_array,
          reinterpret_cast<uint8_t *>(input_fp8.data_ptr()),
          reinterpret_cast<float *>(input_scale.data_ptr()),
          reinterpret_cast<float *>(weight_scale.data_ptr()),
          reinterpret_cast<cute::int32_t *>(routing_indices.data_ptr()),
          reinterpret_cast<cute::int32_t *>(mask.data_ptr()),
          reinterpret_cast<cute::bfloat16_t *>(output.data_ptr()),
          reinterpret_cast<uint8_t *>(weight_fp8.data_ptr()));

  // Free the TMA descriptor arrays after kernel completes
  // (fp8_moe_gemm_2d calls cudaDeviceSynchronize after this returns)
  CUtensorMap *first_desc;
  cudaMemcpy(
      &first_desc, tma_array, sizeof(CUtensorMap *), cudaMemcpyDeviceToHost);
  cudaFree(first_desc); // free contiguous descriptor block
  cudaFree(tma_array);  // free pointer array
}

// Dispatch macro for (expert_stride, n_splits) compile-time pairs
#define DISPATCH_2D_CASE(ES, NS)                                               \
  if (expert_stride == (ES) && n_splits == (NS)) {                             \
    launch_2d_impl<(ES), (NS)>(input_fp8,                                      \
                               input_scale,                                    \
                               weight_fp8,                                     \
                               weight_scale,                                   \
                               routing_indices,                                \
                               mask,                                           \
                               output);                                        \
    dispatched = true;                                                         \
  }

void fp8_moe_gemm_2d(torch::Tensor input_fp8,
                     torch::Tensor input_scale,
                     torch::Tensor weight_fp8,
                     torch::Tensor weight_scale,
                     torch::Tensor routing_indices,
                     torch::Tensor mask,
                     torch::Tensor output,
                     int expert_stride,
                     int n_splits) {
  c10::cuda::CUDAGuard guard(input_fp8.device());
  bool dispatched = false;

  // Common configs: expert_stride=8, various n_splits
  DISPATCH_2D_CASE(8, 1);
  DISPATCH_2D_CASE(8, 2);
  DISPATCH_2D_CASE(8, 4);
  DISPATCH_2D_CASE(8, 8);
  DISPATCH_2D_CASE(8, 16);
  DISPATCH_2D_CASE(8, 32);
  // Alternative grid configs: more expert CTAs, fewer N-splits
  DISPATCH_2D_CASE(4, 32);
  DISPATCH_2D_CASE(16, 8);
  DISPATCH_2D_CASE(32, 4);
  // Single expert CTA (for debugging)
  DISPATCH_2D_CASE(1, 1);
  DISPATCH_2D_CASE(1, 16);

  if (!dispatched) {
    printf("ERROR: (expert_stride=%d, n_splits=%d) not supported\n",
           expert_stride,
           n_splits);
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

#undef DISPATCH_2D_CASE

// ================================================================
// Benchmark API (async launch, no sync)
// ================================================================
static CUtensorMap **g_bench_tma_array = nullptr;
static int g_bench_n_splits = 0;
static int g_bench_expert_stride = 0;

template <int EXPERT_STRIDE, int N_SPLITS>
void bench_setup_impl(torch::Tensor weight_fp8) {
  constexpr int OUTPUT_PER_CTA = DSV3_OUTPUT_SIZE / N_SPLITS;
  constexpr int smem_size = compute_smem_size();

  // Free any previously allocated descriptors to avoid GPU memory leaks
  if (g_bench_tma_array) {
    CUtensorMap *first_desc;
    cudaMemcpy(&first_desc,
               g_bench_tma_array,
               sizeof(CUtensorMap *),
               cudaMemcpyDeviceToHost);
    cudaFree(first_desc);
    cudaFree(g_bench_tma_array);
    g_bench_tma_array = nullptr;
  }

  g_bench_tma_array = create_tma_desc_array(
      weight_fp8.data_ptr(), N_SPLITS, OUTPUT_PER_CTA, DSV3_OUTPUT_SIZE);
  g_bench_n_splits = N_SPLITS;
  g_bench_expert_stride = EXPERT_STRIDE;

  cudaFuncSetAttribute(
      dsv3_fp8_moe_2d_kernel<EXPERT_STRIDE, OUTPUT_PER_CTA, DSV3_OUTPUT_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
}

void fp8_moe_gemm_bench_setup(torch::Tensor weight_fp8,
                              int expert_stride,
                              int n_splits) {
  c10::cuda::CUDAGuard guard(weight_fp8.device());

  // Dispatch to correct template
  if (expert_stride == 8 && n_splits == 16) {
    bench_setup_impl<8, 16>(weight_fp8);
  } else if (expert_stride == 8 && n_splits == 1) {
    bench_setup_impl<8, 1>(weight_fp8);
  } else if (expert_stride == 8 && n_splits == 32) {
    bench_setup_impl<8, 32>(weight_fp8);
  } else if (expert_stride == 4 && n_splits == 32) {
    bench_setup_impl<4, 32>(weight_fp8);
  } else if (expert_stride == 16 && n_splits == 8) {
    bench_setup_impl<16, 8>(weight_fp8);
  } else if (expert_stride == 32 && n_splits == 4) {
    bench_setup_impl<32, 4>(weight_fp8);
  } else {
    printf("ERROR: bench_setup not supported for (%d, %d)\n",
           expert_stride,
           n_splits);
  }
}

template <int EXPERT_STRIDE, int N_SPLITS>
void bench_launch_impl(torch::Tensor input_fp8,
                       torch::Tensor input_scale,
                       torch::Tensor weight_scale,
                       torch::Tensor routing_indices,
                       torch::Tensor mask,
                       torch::Tensor output) {
  constexpr int OUTPUT_PER_CTA = DSV3_OUTPUT_SIZE / N_SPLITS;
  constexpr int smem_size = compute_smem_size();

  dsv3_fp8_moe_2d_kernel<EXPERT_STRIDE, OUTPUT_PER_CTA, DSV3_OUTPUT_SIZE>
      <<<dim3(EXPERT_STRIDE, N_SPLITS, 1), dim3(256, 1, 1), smem_size>>>(
          g_bench_tma_array,
          reinterpret_cast<uint8_t *>(input_fp8.data_ptr()),
          reinterpret_cast<float *>(input_scale.data_ptr()),
          reinterpret_cast<float *>(weight_scale.data_ptr()),
          reinterpret_cast<cute::int32_t *>(routing_indices.data_ptr()),
          reinterpret_cast<cute::int32_t *>(mask.data_ptr()),
          reinterpret_cast<cute::bfloat16_t *>(output.data_ptr()),
          nullptr); // weight_fp8_base not needed for bench
}

void fp8_moe_gemm_bench_launch(torch::Tensor input_fp8,
                               torch::Tensor input_scale,
                               torch::Tensor weight_scale,
                               torch::Tensor routing_indices,
                               torch::Tensor mask,
                               torch::Tensor output) {
  assert(g_bench_tma_array && "Call fp8_moe_gemm_bench_setup first");
  if (g_bench_expert_stride == 8 && g_bench_n_splits == 16) {
    bench_launch_impl<8, 16>(
        input_fp8, input_scale, weight_scale, routing_indices, mask, output);
  } else if (g_bench_expert_stride == 8 && g_bench_n_splits == 1) {
    bench_launch_impl<8, 1>(
        input_fp8, input_scale, weight_scale, routing_indices, mask, output);
  } else if (g_bench_expert_stride == 8 && g_bench_n_splits == 32) {
    bench_launch_impl<8, 32>(
        input_fp8, input_scale, weight_scale, routing_indices, mask, output);
  } else if (g_bench_expert_stride == 4 && g_bench_n_splits == 32) {
    bench_launch_impl<4, 32>(
        input_fp8, input_scale, weight_scale, routing_indices, mask, output);
  } else if (g_bench_expert_stride == 16 && g_bench_n_splits == 8) {
    bench_launch_impl<16, 8>(
        input_fp8, input_scale, weight_scale, routing_indices, mask, output);
  } else if (g_bench_expert_stride == 32 && g_bench_n_splits == 4) {
    bench_launch_impl<32, 4>(
        input_fp8, input_scale, weight_scale, routing_indices, mask, output);
  }
}

void fp8_moe_gemm_bench_cleanup() {
  if (g_bench_tma_array) {
    // Free the descriptor array and pointer array
    CUtensorMap *first_desc;
    cudaMemcpy(&first_desc,
               g_bench_tma_array,
               sizeof(CUtensorMap *),
               cudaMemcpyDeviceToHost);
    cudaFree(first_desc);        // free contiguous descriptor block
    cudaFree(g_bench_tma_array); // free pointer array
    g_bench_tma_array = nullptr;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_moe_gemm_test",
        &fp8_moe_gemm_test,
        "FP8 MoE GEMM - single CTA correctness test");
  m.def("fp8_moe_gemm_2d",
        &fp8_moe_gemm_2d,
        "FP8 MoE GEMM - 2D grid (expert_stride x n_splits), single launch");
  m.def("fp8_moe_gemm_bench_setup",
        &fp8_moe_gemm_bench_setup,
        "Setup TMA descriptors for benchmark (expert_stride, n_splits)");
  m.def("fp8_moe_gemm_bench_launch",
        &fp8_moe_gemm_bench_launch,
        "Async launch for benchmark (no sync)");
  m.def("fp8_moe_gemm_bench_cleanup",
        &fp8_moe_gemm_bench_cleanup,
        "Cleanup benchmark resources");
}
