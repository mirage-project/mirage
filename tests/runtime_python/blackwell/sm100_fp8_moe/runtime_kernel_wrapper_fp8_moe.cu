#include <cstdio>
#include <cassert>

// Cutlass / CuTe
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>

// MPK headers
#include "tasks/blackwell/fp8_group_gemm_sm100.cuh"
#include "tasks/hopper/smem_layout_tma.cuh"
#include "tasks/hopper/tma.cuh"
#include "tma.cuh"  // mirage::runtime::fill_tma_desc

// PyTorch / pybind11
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// ----------------------------------------------------------------
// Concrete template instantiation for testing.
//
// DeepSeek V3 shapes (W13 variant):
//   BATCH_SIZE    = up to 16
//   OUTPUT_SIZE   = 4096  (2 * intermediate_size for W13)
//   REDUCTION_SIZE= 7168  (hidden_size = K)
//   NUM_EXPERTS   = 256
//   NUM_TOPK      = 8
//
// We test a smaller shape for unit tests but keep the same types.
// ----------------------------------------------------------------

// Small test dimensions (keep compilation fast)
constexpr int TEST_MMA_M          = 128;
constexpr int TEST_MMA_N          = 128;  // Align to DeepGEMM BLOCK_N
constexpr int TEST_BATCH_SIZE     = 128;  // MMA_N (padded; actual tokens ≤ 16)
constexpr int TEST_OUTPUT_SIZE    = 256;
constexpr int TEST_REDUCTION_SIZE = 256;
constexpr int TEST_NUM_EXPERTS    = 8;
constexpr int TEST_NUM_TOPK       = 2;
constexpr int TEST_NUM_AB_STAGE   = 2;  // reduced for debugging
constexpr int TEST_NUM_ACC_STAGE  = 2;
constexpr int TEST_NUM_C_STAGE    = 4;

// TMA type for FP8 weights (uint8_t → UINT8 TMA format)
using FP8TMA_W13 = kernel::tma::tma_2d<
    uint8_t,
    3,   // B (swizzle 128B for TMA weight loading)
    3,   // M
    3,   // S
    (TEST_NUM_EXPERTS - 1) * TEST_OUTPUT_SIZE + TEST_OUTPUT_SIZE, // GMEM_ROW_
    TEST_REDUCTION_SIZE,   // GMEM_COL_
    TEST_MMA_M,            // SMEM_ROW_
    128,                   // SMEM_COL_ = bK
    TEST_REDUCTION_SIZE,   // GMEM_STRIDE_ROW_
    1,                     // GMEM_STRIDE_COL_
    1,                     // SMEM_REPEAT_ROW_
    1,                     // SMEM_REPEAT_COL_
    TEST_MMA_M * 128,      // SMEM_STRIDE_
    true>;                 // row_major

// NOTE: cute::gmem_ptr<T> takes the *pointer type* T as template arg.
// make_gmem_ptr(float* p) returns gmem_ptr<float*>, NOT gmem_ptr<float>.

// Input tensor layout: [batch, K]  (FP8 = 1 byte, use uint8_t*)
using InputLayout = cute::Layout<
    cute::Shape<cute::Int<TEST_BATCH_SIZE>, cute::Int<TEST_REDUCTION_SIZE>>,
    cute::Stride<cute::Int<TEST_REDUCTION_SIZE>, cute::Int<1>>>;
using InputTensor = cute::Tensor<
    cute::ViewEngine<cute::gmem_ptr<uint8_t*>>, InputLayout>;

// Input scale layout: [batch, K/128]
constexpr int TEST_K_SCALE = TEST_REDUCTION_SIZE / 128;
using InputScaleLayout = cute::Layout<
    cute::Shape<cute::Int<TEST_BATCH_SIZE>, cute::Int<TEST_K_SCALE>>,
    cute::Stride<cute::Int<TEST_K_SCALE>, cute::Int<1>>>;
using InputScaleTensor = cute::Tensor<
    cute::ViewEngine<cute::gmem_ptr<float*>>, InputScaleLayout>;

// Weight scale layout: [num_experts * ORIG_OUTPUT_SIZE, K/128]
using WeightScaleLayout = cute::Layout<
    cute::Shape<cute::Int<TEST_NUM_EXPERTS * TEST_OUTPUT_SIZE>,
                cute::Int<TEST_K_SCALE>>,
    cute::Stride<cute::Int<TEST_K_SCALE>, cute::Int<1>>>;
using WeightScaleTensor = cute::Tensor<
    cute::ViewEngine<cute::gmem_ptr<float*>>, WeightScaleLayout>;

// Routing indices: [num_experts, batch]
using IndicesLayout = cute::Layout<
    cute::Shape<cute::Int<TEST_NUM_EXPERTS>, cute::Int<TEST_BATCH_SIZE>>,
    cute::Stride<cute::Int<TEST_BATCH_SIZE>, cute::Int<1>>>;
using IndicesTensor = cute::Tensor<
    cute::ViewEngine<cute::gmem_ptr<cute::int32_t*>>, IndicesLayout>;

// Mask: [num_experts + 1]
using MaskLayout = cute::Layout<
    cute::Shape<cute::Int<TEST_NUM_EXPERTS + 1>>,
    cute::Stride<cute::Int<1>>>;
using MaskTensor = cute::Tensor<
    cute::ViewEngine<cute::gmem_ptr<cute::int32_t*>>, MaskLayout>;

// Output: mOutput(n_idx, topk_idx, m_idx) → maps to out[batch, topk, output_row]
// Python out shape [B, topk, N] strides [topk*N, N, 1].
// C++ tensor shape (BATCH, OUTPUT_SIZE, TOPK) with stride (TOPK*N, 1, N)
// → mOutput(n_idx, topk, m_idx) = n_idx*(topk*N) + topk*N + m_idx*1
using OutputLayout = cute::Layout<
    cute::Shape<cute::Int<TEST_BATCH_SIZE>,
                cute::Int<TEST_OUTPUT_SIZE>,
                cute::Int<TEST_NUM_TOPK>>,
    cute::Stride<cute::Int<TEST_NUM_TOPK * TEST_OUTPUT_SIZE>,
                 cute::Int<TEST_OUTPUT_SIZE>,
                 cute::Int<1>>>;
using OutputTensor = cute::Tensor<
    cute::ViewEngine<cute::gmem_ptr<cute::bfloat16_t*>>, OutputLayout>;

// ----------------------------------------------------------------
// Global kernel wrapper.
//
// Multi-CTA mode: launch with grid=(num_ctas, 1, 1).
//   Each CTA uses blockIdx.x as expert_offset and gridDim.x as expert_stride.
//   CTA i processes experts i, i+gridDim.x, i+2*gridDim.x, ...
//
// Single-CTA mode: launch with grid=(1, 1, 1), expert_offset=0.
//   The single CTA processes all experts sequentially (stride=1).
// ----------------------------------------------------------------
template <int EXPERT_STRIDE>
__global__
__launch_bounds__(256, 1)
void fp8_moe_w13_test_kernel(
    CUtensorMap const *__restrict__ tma_weight_desc,
    uint8_t const *input_fp8,
    float const *input_scale,
    float const *weight_scale,
    cute::int32_t const *routing_indices,
    cute::int32_t const *mask,
    cute::bfloat16_t *output,
    int expert_offset)
{
  // Rebuild TMA wrapper
  FP8TMA_W13 tma_weight(const_cast<CUtensorMap*>(tma_weight_desc));

  // Rebuild CuTe tensor views
  InputTensor mInput = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input_fp8))),
      InputLayout{});
  InputScaleTensor mInputScale = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<float*>(input_scale)),
      InputScaleLayout{});
  WeightScaleTensor mWeightScale = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<float*>(weight_scale)),
      WeightScaleLayout{});
  IndicesTensor mRoutingIndices = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t*>(routing_indices)),
      IndicesLayout{});
  MaskTensor mMask = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t*>(mask)),
      MaskLayout{});
  OutputTensor mOutput = cute::make_tensor(
      cute::make_gmem_ptr(output),
      OutputLayout{});

  // In multi-CTA mode (EXPERT_STRIDE > 1), use blockIdx.x as expert_offset
  // so each CTA handles a different expert. In single-CTA mode (EXPERT_STRIDE=1),
  // use the provided expert_offset argument.
  int actual_expert_offset = (EXPERT_STRIDE > 1) ? blockIdx.x : expert_offset;

  kernel::fp8_moe_group_gemm_sm100_task_impl<
      FP8TMA_W13,
      InputTensor,
      InputScaleTensor,
      WeightScaleTensor,
      IndicesTensor,
      MaskTensor,
      OutputTensor,
      TEST_MMA_M,
      TEST_MMA_N,
      TEST_BATCH_SIZE,
      TEST_OUTPUT_SIZE,
      TEST_OUTPUT_SIZE,    // ORIG_OUTPUT_SIZE
      TEST_REDUCTION_SIZE,
      TEST_NUM_EXPERTS,
      TEST_NUM_TOPK,
      EXPERT_STRIDE,
      true,                // W13_LINEAR
      TEST_NUM_AB_STAGE,
      TEST_NUM_ACC_STAGE,
      TEST_NUM_C_STAGE>(
      tma_weight,
      mInput,
      mInputScale,
      mWeightScale,
      mRoutingIndices,
      mMask,
      mOutput,
      actual_expert_offset);
}

// ----------------------------------------------------------------
// Helper: build host-side CUtensorMap using fill_tma_desc from tma.cuh
// ----------------------------------------------------------------
CUtensorMap* create_fp8_weight_tma_desc(void *weight_ptr,
                                         int total_rows,
                                         int cols) {
  // fill_tma_desc<T, B, M, S, NDIM> convention:
  //   gmem_shape[0] = rows (outermost), gmem_shape[1] = cols (innermost K)
  //   gmem_stride[0] = col-stride (= 1), gmem_stride[1] = row-stride (= cols)
  //   smem_shape[0] = SMEM_ROW = MMA_M, smem_shape[1] = SMEM_COL = bK
  constexpr int B = 3, M = 3, S = 3;
  constexpr int bK = 128;
  uint64_t gmem_shape[2]  = {(uint64_t)total_rows, (uint64_t)cols};
  uint64_t gmem_stride[2] = {1, (uint64_t)cols};  // col-stride=1, row-stride=cols
  uint32_t smem_shape[2]  = {(uint32_t)TEST_MMA_M,  // SMEM_ROW = 128
                              (uint32_t)bK};          // SMEM_COL = 128

  CUtensorMap host_desc;
  mirage::runtime::fill_tma_desc<uint8_t, B, M, S, 2>(
      &host_desc,
      weight_ptr,
      gmem_shape,
      gmem_stride,
      smem_shape,
      1,   // smem_repeat_row
      1);  // smem_repeat_col

  CUtensorMap *dev_desc;
  cudaMalloc(&dev_desc, sizeof(CUtensorMap));
  cudaMemcpy(dev_desc, &host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  return dev_desc;
}

// ----------------------------------------------------------------
// Shared smem size computation
// ----------------------------------------------------------------
static constexpr int compute_smem_size() {
  constexpr int smem_A = TEST_NUM_AB_STAGE * TEST_MMA_M * 128 * 1;
  constexpr int smem_B = TEST_NUM_AB_STAGE * TEST_MMA_N * 128 * 1;
  constexpr int smem_SF = 2 * TEST_NUM_AB_STAGE * 128 * 4;
  constexpr int smem_barriers = 8 * TEST_NUM_AB_STAGE * 8
                               + 2 * TEST_NUM_ACC_STAGE * 8
                               + TEST_NUM_EXPERTS * 4 + 4 + 128;
  return smem_A + smem_B + smem_SF + smem_barriers + 4096;
}

// ----------------------------------------------------------------
// Python-facing launch: single CTA (for correctness tests)
// ----------------------------------------------------------------
void fp8_moe_w13_gemm(
    torch::Tensor input_fp8,
    torch::Tensor input_scale,
    torch::Tensor weight_fp8,
    torch::Tensor weight_scale,
    torch::Tensor routing_indices,
    torch::Tensor mask,
    torch::Tensor output,
    int expert_offset)
{
  c10::cuda::CUDAGuard guard(input_fp8.device());

  int num_experts = weight_fp8.size(0);
  int output_size = weight_fp8.size(1);
  int reduction_size = weight_fp8.size(2);
  int total_rows = num_experts * output_size;

  CUtensorMap *tma_desc = create_fp8_weight_tma_desc(
      weight_fp8.data_ptr(), total_rows, reduction_size);

  constexpr int smem_size = compute_smem_size();
  dim3 grid(1, 1, 1);
  dim3 block(256, 1, 1);

  cudaFuncSetAttribute(fp8_moe_w13_test_kernel<1>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  fp8_moe_w13_test_kernel<1><<<grid, block, smem_size>>>(
      tma_desc,
      reinterpret_cast<uint8_t*>(input_fp8.data_ptr()),
      reinterpret_cast<float*>(input_scale.data_ptr()),
      reinterpret_cast<float*>(weight_scale.data_ptr()),
      reinterpret_cast<cute::int32_t*>(routing_indices.data_ptr()),
      reinterpret_cast<cute::int32_t*>(mask.data_ptr()),
      reinterpret_cast<cute::bfloat16_t*>(output.data_ptr()),
      expert_offset);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
  }
  cudaFree(tma_desc);
}

// ----------------------------------------------------------------
// Python-facing launch: multi-CTA (for realistic benchmarking)
//
// Launches num_ctas thread blocks. CTA i handles experts
// i, i+num_ctas, i+2*num_ctas, ...
// Does NOT synchronize — caller is responsible for sync/timing.
// ----------------------------------------------------------------
// Persistent TMA descriptor handle for benchmark mode
static CUtensorMap *g_tma_desc = nullptr;

void fp8_moe_w13_gemm_setup(
    torch::Tensor weight_fp8)
{
  c10::cuda::CUDAGuard guard(weight_fp8.device());
  int num_experts = weight_fp8.size(0);
  int output_size = weight_fp8.size(1);
  int reduction_size = weight_fp8.size(2);
  int total_rows = num_experts * output_size;

  if (g_tma_desc) cudaFree(g_tma_desc);
  g_tma_desc = create_fp8_weight_tma_desc(
      weight_fp8.data_ptr(), total_rows, reduction_size);

  // Set max dynamic smem for both kernel variants
  constexpr int smem_size = compute_smem_size();
  cudaFuncSetAttribute(fp8_moe_w13_test_kernel<1>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  cudaFuncSetAttribute(fp8_moe_w13_test_kernel<TEST_NUM_EXPERTS>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
}

void fp8_moe_w13_gemm_launch(
    torch::Tensor input_fp8,
    torch::Tensor input_scale,
    torch::Tensor weight_scale,
    torch::Tensor routing_indices,
    torch::Tensor mask,
    torch::Tensor output,
    int num_ctas)
{
  assert(g_tma_desc != nullptr && "Call fp8_moe_w13_gemm_setup first");
  constexpr int smem_size = compute_smem_size();

  if (num_ctas == 1) {
    // Single CTA: EXPERT_STRIDE=1, expert_offset=0
    // One CTA processes all experts sequentially
    fp8_moe_w13_test_kernel<1><<<dim3(1,1,1), dim3(256,1,1), smem_size>>>(
        g_tma_desc,
        reinterpret_cast<uint8_t*>(input_fp8.data_ptr()),
        reinterpret_cast<float*>(input_scale.data_ptr()),
        reinterpret_cast<float*>(weight_scale.data_ptr()),
        reinterpret_cast<cute::int32_t*>(routing_indices.data_ptr()),
        reinterpret_cast<cute::int32_t*>(mask.data_ptr()),
        reinterpret_cast<cute::bfloat16_t*>(output.data_ptr()),
        0);
  } else {
    // Multi-CTA: EXPERT_STRIDE=num_ctas, expert_offset=blockIdx.x
    // Each CTA processes experts blockIdx.x, blockIdx.x+num_ctas, ...
    // The expert_offset arg is ignored; blockIdx.x is used instead.
    dim3 grid(num_ctas, 1, 1);
    dim3 block(256, 1, 1);
    fp8_moe_w13_test_kernel<TEST_NUM_EXPERTS><<<grid, block, smem_size>>>(
        g_tma_desc,
        reinterpret_cast<uint8_t*>(input_fp8.data_ptr()),
        reinterpret_cast<float*>(input_scale.data_ptr()),
        reinterpret_cast<float*>(weight_scale.data_ptr()),
        reinterpret_cast<cute::int32_t*>(routing_indices.data_ptr()),
        reinterpret_cast<cute::int32_t*>(mask.data_ptr()),
        reinterpret_cast<cute::bfloat16_t*>(output.data_ptr()),
        -1);  // unused; kernel uses blockIdx.x
  }
  // No sync — caller handles timing
}

void fp8_moe_w13_gemm_cleanup()
{
  if (g_tma_desc) {
    cudaFree(g_tma_desc);
    g_tma_desc = nullptr;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_moe_w13_gemm", &fp8_moe_w13_gemm,
        "FP8 block-scaled MoE W13 group GEMM - single CTA (SM100/Blackwell)");
  m.def("fp8_moe_w13_gemm_setup", &fp8_moe_w13_gemm_setup,
        "Setup TMA descriptor for benchmark mode");
  m.def("fp8_moe_w13_gemm_launch", &fp8_moe_w13_gemm_launch,
        "Launch FP8 MoE GEMM without sync (for benchmarking)");
  m.def("fp8_moe_w13_gemm_cleanup", &fp8_moe_w13_gemm_cleanup,
        "Free TMA descriptor");
}
