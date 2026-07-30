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

// FP8 MoE group-GEMM harness at QWEN3.5-35B-A3B shapes (probe P2, M2-I13).
//
// The in-tree harness
// (tests/runtime_python/blackwell/sm100_fp8_moe/runtime_kernel_wrapper.cu)
// hardcodes DeepSeek-V3 shapes (N=4096, K=7168 / w2 N=7168, K=2048)
// [docs/qwen35/mpk-gaps.md 2.3]. This file is the same single-CTA driver
// re-parameterized to OUR routed-expert shapes
//
//   w13: [256, 1024, 2048]   (K = 2048 -> fp8_k_tile_count = 16)
//   w2 : [256, 2048,  512]   (K =  512 -> fp8_k_tile_count =  4)
//
// so P2 can (a) run the 4-k-tile w2 regime that Gap 7 flags as never-executed
// and (b) measure the numeric effect of the grouped kernel's INTERNAL UE8M0
// scale conversion (fp8_group_gemm_sm100.cuh warp 6) on real checkpoint
// scales.
//
// It also drives the fp32-scale grouped fallback
// (blackwell/moe_fp8_blockscale_sm100.cuh) at the same shapes, so both scale
// semantics are measured through one harness on identical bytes.
//
// `num_ab_stages` is a runtime argument (4 or 8) because Gap 7's open question
// is exactly whether the 4-k-tile w2 regime survives the pipeline depth that
// task_register.cc:2818 raised to 8 for a DIFFERENT regime.

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

#include "blackwell/moe_fp8_blockscale_sm100.cuh"
#include "blackwell/moe_silu_mul_quantize_fp8_sm100.cuh"
#include "blackwell/rmsnorm_quantize_fp8_sm100.cuh"
#include "blackwell/per_token_group_quantize_fp8.cuh"
#include "tasks/ampere/silu_mul.cuh"
#include "tasks/hopper/rmsnorm_hopper.cuh"
#include "runtime_header.h"
#include "tasks/blackwell/fp8_group_gemm_sm100.cuh"
#include "tasks/hopper/smem_layout_tma.cuh"
#include "tasks/hopper/tma.cuh"
#include "tma.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

namespace {

// ================================================================
// Qwen3.5-35B-A3B routed-expert shapes (docs/qwen35/v1-architecture.md 5, 6.1)
// ================================================================
constexpr int Q35_MMA_M = 128;
constexpr int Q35_MMA_N = 16; // matches production task_register (MMA_N=16)
constexpr int Q35_BATCH_SIZE = 16;
constexpr int Q35_NUM_EXPERTS = 256;
constexpr int Q35_NUM_TOPK = 8;
constexpr int Q35_NUM_ACC_STAGE = 2;
constexpr int Q35_NUM_C_STAGE = 4;

constexpr int W13_OUTPUT_SIZE = 1024;    // 2 * moe_intermediate_size
constexpr int W13_REDUCTION_SIZE = 2048; // hidden_size
constexpr int W13_K_SCALE = W13_REDUCTION_SIZE / 128; // 16 k-tiles

constexpr int W2_OUTPUT_SIZE = 2048;                // hidden_size
constexpr int W2_REDUCTION_SIZE = 512;              // moe_intermediate_size
constexpr int W2_K_SCALE = W2_REDUCTION_SIZE / 128; // 4 k-tiles (Gap 7)

// ================================================================
// CuTe views
// ================================================================
template <int OUTPUT_SIZE_, int REDUCTION_SIZE_>
using Q35TMA = kernel::tma::tma_2d<uint8_t,
                                   3,
                                   3,
                                   3,
                                   Q35_NUM_EXPERTS * OUTPUT_SIZE_,
                                   REDUCTION_SIZE_,
                                   Q35_MMA_M,
                                   128,
                                   REDUCTION_SIZE_,
                                   1,
                                   1,
                                   1,
                                   Q35_MMA_M * 128,
                                   true>;

using IndicesLayout = cute::Layout<
    cute::Shape<cute::Int<Q35_NUM_EXPERTS>, cute::Int<Q35_BATCH_SIZE>>,
    cute::Stride<cute::Int<Q35_BATCH_SIZE>, cute::Int<1>>>;
using MaskLayout = cute::Layout<cute::Shape<cute::Int<Q35_NUM_EXPERTS + 1>>,
                                cute::Stride<cute::Int<1>>>;

// Weight scale as the grouped kernel wants it: flat [E*N, K/128] float32,
// i.e. the checkpoint's [E, N/128, K/128] block scale after the builder's
// repeat_interleave(128, dim=1) (deepseek_v3/builder.py:983).
template <int OUTPUT_SIZE_, int K_SCALE_>
__device__ __forceinline__ auto make_weight_scale_view(float const *p) {
  return cute::make_tensor(
      cute::make_gmem_ptr(const_cast<float *>(p)),
      cute::make_layout(
          cute::make_shape(cute::Int<Q35_NUM_EXPERTS * OUTPUT_SIZE_>{},
                           cute::Int<K_SCALE_>{}),
          cute::make_stride(cute::Int<K_SCALE_>{}, cute::Int<1>{})));
}

// ================================================================
// W13 single-CTA kernel: [B, K] @ [E, N, K]^T -> [B, topk, N]
// ================================================================
template <int NUM_AB_STAGE>
__global__ __launch_bounds__(256, 1) void q35_moe_w13_ue8m0_kernel(
    CUtensorMap const *__restrict__ tma_weight_desc,
    uint8_t const *input_fp8,
    float const *input_scale,
    float const *weight_scale,
    cute::int32_t const *routing_indices,
    cute::int32_t const *mask,
    cute::bfloat16_t *output) {
  using TMA_t = Q35TMA<W13_OUTPUT_SIZE, W13_REDUCTION_SIZE>;
  TMA_t tma_weight(const_cast<CUtensorMap *>(tma_weight_desc));

  auto mInput = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<uint8_t *>(input_fp8)),
      cute::make_layout(
          cute::make_shape(cute::Int<Q35_BATCH_SIZE>{},
                           cute::Int<W13_REDUCTION_SIZE>{}),
          cute::make_stride(cute::Int<W13_REDUCTION_SIZE>{}, cute::Int<1>{})));
  auto mInputScale = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<float *>(input_scale)),
      cute::make_layout(
          cute::make_shape(cute::Int<Q35_BATCH_SIZE>{},
                           cute::Int<W13_K_SCALE>{}),
          cute::make_stride(cute::Int<W13_K_SCALE>{}, cute::Int<1>{})));
  auto mWeightScale =
      make_weight_scale_view<W13_OUTPUT_SIZE, W13_K_SCALE>(weight_scale);
  auto mRoutingIndices = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t *>(routing_indices)),
      IndicesLayout{});
  auto mMask = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t *>(mask)), MaskLayout{});
  auto mOutput = cute::make_tensor(
      cute::make_gmem_ptr(output),
      cute::make_layout(
          cute::make_shape(cute::Int<Q35_BATCH_SIZE>{},
                           cute::Int<Q35_NUM_TOPK>{},
                           cute::Int<W13_OUTPUT_SIZE>{}),
          cute::make_stride(cute::Int<Q35_NUM_TOPK * W13_OUTPUT_SIZE>{},
                            cute::Int<W13_OUTPUT_SIZE>{},
                            cute::Int<1>{})));

  kernel::fp8_moe_group_gemm_sm100_task_impl<TMA_t,
                                             decltype(mInput),
                                             decltype(mInputScale),
                                             decltype(mWeightScale),
                                             decltype(mRoutingIndices),
                                             decltype(mMask),
                                             decltype(mOutput),
                                             Q35_MMA_M,
                                             Q35_MMA_N,
                                             Q35_BATCH_SIZE,
                                             W13_OUTPUT_SIZE,
                                             W13_OUTPUT_SIZE,
                                             W13_REDUCTION_SIZE,
                                             Q35_NUM_EXPERTS,
                                             Q35_NUM_TOPK,
                                             1,    // EXPERT_STRIDE
                                             true, // W13_LINEAR
                                             NUM_AB_STAGE,
                                             Q35_NUM_ACC_STAGE,
                                             Q35_NUM_C_STAGE>(tma_weight,
                                                              mInput,
                                                              mInputScale,
                                                              mWeightScale,
                                                              mRoutingIndices,
                                                              mMask,
                                                              mOutput,
                                                              0);
}

// ================================================================
// W2 single-CTA kernel: [B, topk, I] @ [E, N, I]^T -> [B, topk, N]
// ================================================================
template <int NUM_AB_STAGE>
__global__ __launch_bounds__(256, 1) void q35_moe_w2_ue8m0_kernel(
    CUtensorMap const *__restrict__ tma_weight_desc,
    uint8_t const *input_fp8,
    float const *input_scale,
    float const *weight_scale,
    cute::int32_t const *routing_indices,
    cute::int32_t const *mask,
    cute::bfloat16_t *output) {
  using TMA_t = Q35TMA<W2_OUTPUT_SIZE, W2_REDUCTION_SIZE>;
  TMA_t tma_weight(const_cast<CUtensorMap *>(tma_weight_desc));

  auto mInput = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<uint8_t *>(input_fp8)),
      cute::make_layout(
          cute::make_shape(cute::Int<Q35_BATCH_SIZE>{},
                           cute::Int<Q35_NUM_TOPK>{},
                           cute::Int<W2_REDUCTION_SIZE>{}),
          cute::make_stride(cute::Int<Q35_NUM_TOPK * W2_REDUCTION_SIZE>{},
                            cute::Int<W2_REDUCTION_SIZE>{},
                            cute::Int<1>{})));
  auto mInputScale = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<float *>(input_scale)),
      cute::make_layout(
          cute::make_shape(cute::Int<Q35_BATCH_SIZE>{},
                           cute::Int<Q35_NUM_TOPK>{},
                           cute::Int<W2_K_SCALE>{}),
          cute::make_stride(cute::Int<Q35_NUM_TOPK * W2_K_SCALE>{},
                            cute::Int<W2_K_SCALE>{},
                            cute::Int<1>{})));
  auto mWeightScale =
      make_weight_scale_view<W2_OUTPUT_SIZE, W2_K_SCALE>(weight_scale);
  auto mRoutingIndices = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t *>(routing_indices)),
      IndicesLayout{});
  auto mMask = cute::make_tensor(
      cute::make_gmem_ptr(const_cast<cute::int32_t *>(mask)), MaskLayout{});
  auto mOutput = cute::make_tensor(
      cute::make_gmem_ptr(output),
      cute::make_layout(
          cute::make_shape(cute::Int<Q35_BATCH_SIZE>{},
                           cute::Int<Q35_NUM_TOPK>{},
                           cute::Int<W2_OUTPUT_SIZE>{}),
          cute::make_stride(cute::Int<Q35_NUM_TOPK * W2_OUTPUT_SIZE>{},
                            cute::Int<W2_OUTPUT_SIZE>{},
                            cute::Int<1>{})));

  kernel::fp8_moe_group_gemm_sm100_task_impl<TMA_t,
                                             decltype(mInput),
                                             decltype(mInputScale),
                                             decltype(mWeightScale),
                                             decltype(mRoutingIndices),
                                             decltype(mMask),
                                             decltype(mOutput),
                                             Q35_MMA_M,
                                             Q35_MMA_N,
                                             Q35_BATCH_SIZE,
                                             W2_OUTPUT_SIZE,
                                             W2_OUTPUT_SIZE,
                                             W2_REDUCTION_SIZE,
                                             Q35_NUM_EXPERTS,
                                             Q35_NUM_TOPK,
                                             1,     // EXPERT_STRIDE
                                             false, // W13_LINEAR (w2)
                                             NUM_AB_STAGE,
                                             Q35_NUM_ACC_STAGE,
                                             Q35_NUM_C_STAGE>(tma_weight,
                                                              mInput,
                                                              mInputScale,
                                                              mWeightScale,
                                                              mRoutingIndices,
                                                              mMask,
                                                              mOutput,
                                                              0);
}

// ================================================================
// fp32-scale fallback kernels (moe_fp8_blockscale_sm100.cuh)
//
// Same I/O contract as above except weight_scale is the checkpoint's
// UNEXPANDED [E, N/128, K/128] block scale -- the fallback indexes the block
// directly, so no repeat_interleave is needed.
// ================================================================
template <bool W13_LINEAR, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    q35_moe_blockscale_kernel(void const *input_fp8,
                              void const *input_scale,
                              void const *weight_fp8,
                              void const *weight_scale,
                              void const *routing_indices,
                              void const *mask,
                              void *output) {
  kernel::moe_fp8_blockscale_task_impl<bfloat16,
                                       Q35_BATCH_SIZE,
                                       Q35_NUM_TOPK,
                                       Q35_NUM_EXPERTS,
                                       OUTPUT_SIZE,
                                       OUTPUT_SIZE,
                                       REDUCTION_SIZE,
                                       W13_LINEAR>(input_fp8,
                                                   input_scale,
                                                   weight_fp8,
                                                   weight_scale,
                                                   routing_indices,
                                                   mask,
                                                   output,
                                                   0,
                                                   1);
}

// ================================================================
// fp32-scale activation quantizer (the MoE variant)
//
// per_token_group_quantize_fp8_task_impl with SCALE_UE8M0 = false is the
// primitive the MoE path needs: E4M3 values plus a FLOAT32 [rows, K/128]
// scale, i.e. vLLM's QuantFP8 with use_ue8m0=False (vllm-graph.md 3.4). The
// in-tree quantize test only covers the packed-UE8M0 variant, and only 2D
// inputs; the w2 activation is 3D [batch, topk, I], which flattens to
// rows = batch*topk.
//
// The impl processes ALL BATCH_SIZE rows per instance (the MPK row-identity
// fix, per_token_group_quantize_fp8.cuh:87-92), so one CTA is the whole task.
// ================================================================
template <int ROWS, int HIDDEN>
__global__ void __launch_bounds__(128)
    q35_quantize_f32scale_kernel(void const *__restrict__ input,
                                 void *__restrict__ output_q,
                                 void *__restrict__ output_s) {
  kernel::per_token_group_quantize_fp8_task_impl</*BATCH_SIZE=*/ROWS,
                                                 /*HIDDEN_SIZE=*/HIDDEN,
                                                 /*GROUP_SIZE=*/128,
                                                 /*GLOBAL_STRIDE=*/HIDDEN,
                                                 cute::bfloat16_t,
                                                 __nv_fp8_e4m3,
                                                 /*SCALE_UE8M0=*/false>(
      input, output_q, output_s, 1e-10f, -448.0f, 448.0f, /*unused=*/1);
}

#define Q35_QUANT_ROWS(F, ...)                                                 \
  F(1, __VA_ARGS__)                                                            \
  F(2, __VA_ARGS__)                                                            \
  F(4, __VA_ARGS__)                                                            \
  F(8, __VA_ARGS__)                                                            \
  F(16, __VA_ARGS__)                                                           \
  F(64, __VA_ARGS__)                                                           \
  F(128, __VA_ARGS__)

#define Q35_QUANT_DISPATCH_HIDDEN(H, R)                                        \
  if (hidden == H) {                                                           \
    q35_quantize_f32scale_kernel<R, H>                                         \
        <<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(                     \
            input, output_q, output_s);                                        \
    return true;                                                               \
  }

#define Q35_QUANT_DISPATCH_ROWS(R, UNUSED)                                     \
  if (rows == R) {                                                             \
    Q35_QUANT_DISPATCH_HIDDEN(512, R)                                          \
    Q35_QUANT_DISPATCH_HIDDEN(2048, R)                                         \
    /* M4-I9 added 256/1024 so the fusion's off-shape cases have a REFERENCE   \
       to compare against; the shipped Qwen3.5 sites are still 512 and 2048. */\
    Q35_QUANT_DISPATCH_HIDDEN(256, R)                                          \
    Q35_QUANT_DISPATCH_HIDDEN(1024, R)                                         \
    return false;                                                              \
  }

bool dispatch_quantize_f32scale(
    int rows, int hidden, void const *input, void *output_q, void *output_s) {
  Q35_QUANT_ROWS(Q35_QUANT_DISPATCH_ROWS, 0)
  return false;
}

// ================================================================
// M4-I9 -- the two halves of the fusion, side by side.
//
// The claim under test is BIT-EXACTNESS BY CONSTRUCTION: the fused task's fp8
// bytes and fp32 block scales must be byte-identical to running
// `silu_mul_task_impl` into a bf16 buffer and then
// `per_token_group_quantize_fp8_task_impl` over that buffer -- which is exactly
// what HEAD's unfused pair does. Both arms are compiled in the SAME TU with the
// SAME flags, and the test runs in BOTH nvcc lanes (MOE_TEST_FAST_MATH), because
// -use_fast_math changes `expf` and `/` and the megakernel ships it.
//
// `silu_mul_task_impl` is the ampere-file impl the SM100 registration emits
// (task_register.cc:register_moe_silu_mul_task), not the *_hopper variant.
// ================================================================
template <int ROWS, int OUT>
__global__ void __launch_bounds__(128)
    q35_silu_ref_kernel(void const *__restrict__ input,
                        void *__restrict__ output) {
  kernel::silu_mul_task_impl<cute::bfloat16_t,
                             /*BATCH_SIZE=*/ROWS,
                             /*OUTPUT_SIZE=*/OUT,
                             /*I_STRIDE=*/2 * OUT,
                             /*O_STRIDE=*/OUT>(input, output, ROWS);
}

template <int ROWS, int OUT>
__global__ void __launch_bounds__(128)
    q35_silu_quant_fused_kernel(void const *__restrict__ input,
                                void *__restrict__ output_q,
                                void *__restrict__ output_s) {
  kernel::moe_silu_mul_quantize_fp8_task_impl</*NUM_ROWS=*/ROWS,
                                              /*OUTPUT_SIZE=*/OUT,
                                              /*GROUP_SIZE=*/128,
                                              /*I_STRIDE=*/2 * OUT,
                                              /*O_STRIDE=*/OUT,
                                              /*S_STRIDE=*/OUT / 128,
                                              cute::bfloat16_t,
                                              __nv_fp8_e4m3>(
      input, output_q, output_s, 1e-10f, -448.0f, 448.0f, ROWS);
}

#define Q35_SQ_DISPATCH(R, OUT)                                                \
  if (rows == R && out == OUT) {                                               \
    if (fused) {                                                               \
      q35_silu_quant_fused_kernel<R, OUT>                                      \
          <<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(input, a, b);       \
    } else {                                                                   \
      q35_silu_ref_kernel<R, OUT>                                              \
          <<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(input, a);          \
    }                                                                          \
    return true;                                                               \
  }

bool dispatch_silu_quant(bool fused,
                         int rows,
                         int out,
                         void const *input,
                         void *a,
                         void *b) {
  Q35_SQ_DISPATCH(1, 512)
  Q35_SQ_DISPATCH(2, 512)
  Q35_SQ_DISPATCH(8, 512)
  Q35_SQ_DISPATCH(16, 512)
  Q35_SQ_DISPATCH(128, 512)
  Q35_SQ_DISPATCH(1, 256)
  Q35_SQ_DISPATCH(1, 1024)
  Q35_SQ_DISPATCH(8, 1024)
  return false;
}

#undef Q35_SQ_DISPATCH

// ================================================================
// M4-I9 flag A -- the two halves of the norm+quantize fusion.
//
// Same contract as the silu fusion above: the reference is the SHIPPED PAIR
// (`rms_norm_hopper_impl` into a bf16 buffer, then
// `per_token_group_quantize_fp8_task_impl` over that buffer), not a torch ideal.
//
// LAUNCHED WITH 256 THREADS, which is the megakernel's real block size
// (WORKER_NUM_THREADS = 256) and what `rms_norm_hopper_impl`'s NUM_THREADS
// default assumes -- its cp.async warm-up has no loop, so the template parameter
// and blockDim MUST agree. Dynamic shared memory is
// 3 * HIDDEN * sizeof(T) + NUM_WARPS * 4, the arena the impl carves up.
// ================================================================
template <int ROWS, int HIDDEN>
__global__ void __launch_bounds__(256)
    q35_rmsnorm_ref_kernel(void const *__restrict__ input,
                           void const *__restrict__ weight,
                           void *__restrict__ output) {
  kernel::rms_norm_hopper_impl<cute::bfloat16_t, ROWS, HIDDEN, 256>(
      input, weight, output, 1e-6f);
}

template <int ROWS, int HIDDEN, bool WRITE_NORM>
__global__ void __launch_bounds__(256)
    q35_rmsnorm_quant_fused_kernel(void const *__restrict__ input,
                                   void const *__restrict__ weight,
                                   void *__restrict__ norm_out,
                                   void *__restrict__ out_q,
                                   void *__restrict__ out_s) {
  kernel::rms_norm_quantize_fp8_task_impl</*BATCH_SIZE=*/ROWS,
                                          /*HIDDEN_DIM=*/HIDDEN,
                                          /*GROUP_SIZE=*/128,
                                          WRITE_NORM,
                                          cute::bfloat16_t,
                                          __nv_fp8_e4m3,
                                          /*NUM_THREADS=*/256>(
      input, weight, norm_out, out_q, out_s, 1e-6f, 1e-10f, -448.0f, 448.0f);
}

template <int HIDDEN>
constexpr int q35_norm_smem() {
  return 3 * HIDDEN * (int)sizeof(cute::bfloat16_t) + (256 / 32) * 4;
}

#define Q35_NQ_DISPATCH(R, HID)                                                \
  if (rows == R && hidden == HID) {                                            \
    if (mode == 0) {                                                           \
      q35_rmsnorm_ref_kernel<R, HID>                                           \
          <<<1, 256, q35_norm_smem<HID>(),                                     \
             at::cuda::getCurrentCUDAStream()>>>(input, weight, a);             \
    } else if (mode == 1) {                                                    \
      q35_rmsnorm_quant_fused_kernel<R, HID, true>                             \
          <<<1, 256, q35_norm_smem<HID>(),                                     \
             at::cuda::getCurrentCUDAStream()>>>(input, weight, a, b, c);       \
    } else {                                                                   \
      q35_rmsnorm_quant_fused_kernel<R, HID, false>                            \
          <<<1, 256, q35_norm_smem<HID>(),                                     \
             at::cuda::getCurrentCUDAStream()>>>(input, weight, nullptr, b, c); \
    }                                                                          \
    return true;                                                               \
  }

bool dispatch_norm_quant(int mode,
                         int rows,
                         int hidden,
                         void const *input,
                         void const *weight,
                         void *a,
                         void *b,
                         void *c) {
  Q35_NQ_DISPATCH(1, 2048)
  Q35_NQ_DISPATCH(2, 2048)
  Q35_NQ_DISPATCH(4, 2048)
  Q35_NQ_DISPATCH(16, 2048)
  Q35_NQ_DISPATCH(1, 512)
  Q35_NQ_DISPATCH(1, 1024)
  Q35_NQ_DISPATCH(4, 1024)
  return false;
}

#undef Q35_NQ_DISPATCH

#undef Q35_QUANT_DISPATCH_ROWS
#undef Q35_QUANT_DISPATCH_HIDDEN

// ================================================================
// TMA descriptor for the [E*N, K] flattened weight
// ================================================================
CUtensorMap *create_tma_desc(void *weight_ptr, int total_rows, int cols) {
  constexpr int B = 3, M = 3, S = 3;
  constexpr int bK = 128;
  uint64_t gmem_shape[2] = {(uint64_t)total_rows, (uint64_t)cols};
  uint64_t gmem_stride[2] = {1, (uint64_t)cols};
  uint32_t smem_shape[2] = {(uint32_t)Q35_MMA_M, (uint32_t)bK};

  CUtensorMap host_desc;
  mirage::runtime::fill_tma_desc<uint8_t, B, M, S, 2>(
      &host_desc, weight_ptr, gmem_shape, gmem_stride, smem_shape, 1, 1);

  CUtensorMap *dev_desc;
  cudaMalloc(&dev_desc, sizeof(CUtensorMap));
  cudaMemcpy(dev_desc, &host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  return dev_desc;
}

template <int NUM_AB_STAGE>
constexpr int grouped_smem_size() {
  constexpr int bK = 128;
  constexpr int smem_A = NUM_AB_STAGE * Q35_MMA_M * bK;
  constexpr int smem_B = NUM_AB_STAGE * Q35_MMA_N * bK;
  constexpr int smem_SF = 2 * NUM_AB_STAGE * 128 * 4;
  constexpr int smem_barriers = 8 * NUM_AB_STAGE * 8 +
                                2 * Q35_NUM_ACC_STAGE * 8 +
                                Q35_NUM_EXPERTS * 4 + 4 + 128;
  return smem_A + smem_B + smem_SF + smem_barriers + 4096;
}

void check_common(torch::Tensor const &routing_indices,
                  torch::Tensor const &mask) {
  TORCH_CHECK(routing_indices.dtype() == torch::kInt32 &&
                  routing_indices.size(0) == Q35_NUM_EXPERTS &&
                  routing_indices.size(1) == Q35_BATCH_SIZE,
              "routing_indices must be int32 [256, 16]");
  TORCH_CHECK(mask.dtype() == torch::kInt32 &&
                  mask.size(0) == Q35_NUM_EXPERTS + 1,
              "mask must be int32 [257]");
}

} // namespace

// ================================================================
// Python API
// ================================================================
void moe_w13_ue8m0_sm100(torch::Tensor input_fp8,
                         torch::Tensor input_scale,
                         torch::Tensor weight_fp8,
                         torch::Tensor weight_scale,
                         torch::Tensor routing_indices,
                         torch::Tensor mask,
                         torch::Tensor output,
                         int64_t num_ab_stages) {
  c10::cuda::CUDAGuard guard(input_fp8.device());
  check_common(routing_indices, mask);
  TORCH_CHECK(input_fp8.size(0) == Q35_BATCH_SIZE &&
                  input_fp8.size(1) == W13_REDUCTION_SIZE,
              "w13 input must be [16, 2048]");
  TORCH_CHECK(weight_fp8.size(0) == Q35_NUM_EXPERTS &&
                  weight_fp8.size(1) == W13_OUTPUT_SIZE &&
                  weight_fp8.size(2) == W13_REDUCTION_SIZE,
              "w13 weight must be [256, 1024, 2048]");
  TORCH_CHECK(weight_scale.dtype() == torch::kFloat &&
                  weight_scale.numel() ==
                      (int64_t)Q35_NUM_EXPERTS * W13_OUTPUT_SIZE * W13_K_SCALE,
              "w13 weight_scale must be the per-ROW expanded float32 scale "
              "[256*1024, 16] (builder repeat_interleave form)");

  CUtensorMap *tma = create_tma_desc(weight_fp8.data_ptr(),
                                     Q35_NUM_EXPERTS * W13_OUTPUT_SIZE,
                                     W13_REDUCTION_SIZE);
  auto launch = [&](auto stages) {
    constexpr int S = decltype(stages)::value;
    constexpr int smem = grouped_smem_size<S>();
    cudaFuncSetAttribute(q35_moe_w13_ue8m0_kernel<S>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
    q35_moe_w13_ue8m0_kernel<S><<<dim3(1, 1, 1),
                                  dim3(256, 1, 1),
                                  smem,
                                  at::cuda::getCurrentCUDAStream()>>>(
        tma,
        reinterpret_cast<uint8_t *>(input_fp8.data_ptr()),
        reinterpret_cast<float *>(input_scale.data_ptr()),
        reinterpret_cast<float *>(weight_scale.data_ptr()),
        reinterpret_cast<cute::int32_t *>(routing_indices.data_ptr()),
        reinterpret_cast<cute::int32_t *>(mask.data_ptr()),
        reinterpret_cast<cute::bfloat16_t *>(output.data_ptr()));
  };
  if (num_ab_stages == 4) {
    launch(std::integral_constant<int, 4>{});
  } else if (num_ab_stages == 8) {
    launch(std::integral_constant<int, 8>{});
  } else {
    cudaFree(tma);
    TORCH_CHECK(false, "num_ab_stages must be 4 or 8");
  }
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  cudaFree(tma);
}

void moe_w2_ue8m0_sm100(torch::Tensor input_fp8,
                        torch::Tensor input_scale,
                        torch::Tensor weight_fp8,
                        torch::Tensor weight_scale,
                        torch::Tensor routing_indices,
                        torch::Tensor mask,
                        torch::Tensor output,
                        int64_t num_ab_stages) {
  c10::cuda::CUDAGuard guard(input_fp8.device());
  check_common(routing_indices, mask);
  TORCH_CHECK(input_fp8.size(0) == Q35_BATCH_SIZE &&
                  input_fp8.size(1) == Q35_NUM_TOPK &&
                  input_fp8.size(2) == W2_REDUCTION_SIZE,
              "w2 input must be [16, 8, 512]");
  TORCH_CHECK(weight_fp8.size(0) == Q35_NUM_EXPERTS &&
                  weight_fp8.size(1) == W2_OUTPUT_SIZE &&
                  weight_fp8.size(2) == W2_REDUCTION_SIZE,
              "w2 weight must be [256, 2048, 512]");
  TORCH_CHECK(weight_scale.dtype() == torch::kFloat &&
                  weight_scale.numel() ==
                      (int64_t)Q35_NUM_EXPERTS * W2_OUTPUT_SIZE * W2_K_SCALE,
              "w2 weight_scale must be the per-ROW expanded float32 scale "
              "[256*2048, 4]");

  CUtensorMap *tma = create_tma_desc(weight_fp8.data_ptr(),
                                     Q35_NUM_EXPERTS * W2_OUTPUT_SIZE,
                                     W2_REDUCTION_SIZE);
  auto launch = [&](auto stages) {
    constexpr int S = decltype(stages)::value;
    constexpr int smem = grouped_smem_size<S>();
    cudaFuncSetAttribute(q35_moe_w2_ue8m0_kernel<S>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
    q35_moe_w2_ue8m0_kernel<S><<<dim3(1, 1, 1),
                                 dim3(256, 1, 1),
                                 smem,
                                 at::cuda::getCurrentCUDAStream()>>>(
        tma,
        reinterpret_cast<uint8_t *>(input_fp8.data_ptr()),
        reinterpret_cast<float *>(input_scale.data_ptr()),
        reinterpret_cast<float *>(weight_scale.data_ptr()),
        reinterpret_cast<cute::int32_t *>(routing_indices.data_ptr()),
        reinterpret_cast<cute::int32_t *>(mask.data_ptr()),
        reinterpret_cast<cute::bfloat16_t *>(output.data_ptr()));
  };
  if (num_ab_stages == 4) {
    launch(std::integral_constant<int, 4>{});
  } else if (num_ab_stages == 8) {
    launch(std::integral_constant<int, 8>{});
  } else {
    cudaFree(tma);
    TORCH_CHECK(false, "num_ab_stages must be 4 or 8");
  }
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  cudaFree(tma);
}

void moe_w13_blockscale_sm100(torch::Tensor input_fp8,
                              torch::Tensor input_scale,
                              torch::Tensor weight_fp8,
                              torch::Tensor weight_scale,
                              torch::Tensor routing_indices,
                              torch::Tensor mask,
                              torch::Tensor output) {
  c10::cuda::CUDAGuard guard(input_fp8.device());
  check_common(routing_indices, mask);
  TORCH_CHECK(weight_scale.dtype() == torch::kFloat &&
                  weight_scale.numel() == (int64_t)Q35_NUM_EXPERTS *
                                              (W13_OUTPUT_SIZE / 128) *
                                              W13_K_SCALE,
              "w13 fallback weight_scale must be the CHECKPOINT block scale "
              "[256, 8, 16] float32");
  // M4-I7: size the launch off the DISPATCHER's requirement, not the golden
  // layout alone -- the fast paths stage up to 8 K tiles at once. The kernel
  // also re-checks %dynamic_smem_size and degrades to the narrow path rather
  // than reading past a short allocation, so this is performance, not safety.
  constexpr int smem = kernel::moe_fp8_blockscale_fast::launch_smem_bytes(
      Q35_BATCH_SIZE, W13_OUTPUT_SIZE, W13_REDUCTION_SIZE, /*w13=*/true);
  auto *entry =
      q35_moe_blockscale_kernel<true, W13_OUTPUT_SIZE, W13_REDUCTION_SIZE>;
  cudaFuncSetAttribute(
      entry, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  entry<<<dim3(1, 1, 1),
          dim3(WORKER_NUM_THREADS, 1, 1),
          smem,
          at::cuda::getCurrentCUDAStream()>>>(input_fp8.data_ptr(),
                                              input_scale.data_ptr(),
                                              weight_fp8.data_ptr(),
                                              weight_scale.data_ptr(),
                                              routing_indices.data_ptr(),
                                              mask.data_ptr(),
                                              output.data_ptr());
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}

void moe_w2_blockscale_sm100(torch::Tensor input_fp8,
                             torch::Tensor input_scale,
                             torch::Tensor weight_fp8,
                             torch::Tensor weight_scale,
                             torch::Tensor routing_indices,
                             torch::Tensor mask,
                             torch::Tensor output) {
  c10::cuda::CUDAGuard guard(input_fp8.device());
  check_common(routing_indices, mask);
  TORCH_CHECK(weight_scale.dtype() == torch::kFloat &&
                  weight_scale.numel() == (int64_t)Q35_NUM_EXPERTS *
                                              (W2_OUTPUT_SIZE / 128) *
                                              W2_K_SCALE,
              "w2 fallback weight_scale must be the CHECKPOINT block scale "
              "[256, 16, 4] float32");
  constexpr int smem = kernel::moe_fp8_blockscale_fast::launch_smem_bytes(
      Q35_BATCH_SIZE, W2_OUTPUT_SIZE, W2_REDUCTION_SIZE, /*w13=*/false);
  auto *entry =
      q35_moe_blockscale_kernel<false, W2_OUTPUT_SIZE, W2_REDUCTION_SIZE>;
  cudaFuncSetAttribute(
      entry, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  entry<<<dim3(1, 1, 1),
          dim3(WORKER_NUM_THREADS, 1, 1),
          smem,
          at::cuda::getCurrentCUDAStream()>>>(input_fp8.data_ptr(),
                                              input_scale.data_ptr(),
                                              weight_fp8.data_ptr(),
                                              weight_scale.data_ptr(),
                                              routing_indices.data_ptr(),
                                              mask.data_ptr(),
                                              output.data_ptr());
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}

void quantize_fp8_f32scale_sm100(torch::Tensor input,
                                 torch::Tensor output_q,
                                 torch::Tensor output_s) {
  c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(input.is_contiguous() && output_q.is_contiguous() &&
                  output_s.is_contiguous(),
              "all quantize tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(output_q.scalar_type() == at::kFloat8_e4m3fn,
              "output_q must be float8_e4m3fn");
  TORCH_CHECK(output_s.scalar_type() == at::kFloat,
              "output_s must be float32 (the MoE scale variant)");
  TORCH_CHECK(output_q.sizes() == input.sizes(), "output_q shape mismatch");

  int const hidden = static_cast<int>(input.size(input.dim() - 1));
  int const rows = static_cast<int>(input.numel() / hidden);
  TORCH_CHECK(hidden % 128 == 0, "hidden must be a multiple of 128");
  TORCH_CHECK(output_s.numel() == (int64_t)rows * (hidden / 128),
              "output_s must be [rows, hidden/128] float32");

  bool const ok = dispatch_quantize_f32scale(
      rows, hidden, input.data_ptr(), output_q.data_ptr(), output_s.data_ptr());
  TORCH_CHECK(
      ok, "Unsupported quantize shape [rows=", rows, ", hidden=", hidden, "]");
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}

void rmsnorm_sm100(torch::Tensor input,
                   torch::Tensor weight,
                   torch::Tensor output) {
  c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous() &&
                  weight.is_contiguous(),
              "rmsnorm tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "rmsnorm is bf16");
  int const hidden = static_cast<int>(input.size(input.dim() - 1));
  int const rows = static_cast<int>(input.numel() / hidden);
  TORCH_CHECK(weight.numel() == hidden, "weight must be [hidden]");
  TORCH_CHECK(dispatch_norm_quant(0, rows, hidden, input.data_ptr(),
                                  weight.data_ptr(), output.data_ptr(),
                                  nullptr, nullptr),
              "unsupported rmsnorm shape [rows=", rows, ", hidden=", hidden,
              "]");
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}

void rmsnorm_quantize_fp8_sm100(torch::Tensor input,
                                torch::Tensor weight,
                                torch::Tensor norm_out,
                                torch::Tensor output_q,
                                torch::Tensor output_s,
                                bool write_norm) {
  c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(input.is_contiguous() && output_q.is_contiguous() &&
                  output_s.is_contiguous() && weight.is_contiguous(),
              "fused norm+quant tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(output_q.scalar_type() == at::kFloat8_e4m3fn,
              "output_q must be float8_e4m3fn");
  TORCH_CHECK(output_s.scalar_type() == at::kFloat, "output_s must be float32");
  int const hidden = static_cast<int>(input.size(input.dim() - 1));
  int const rows = static_cast<int>(input.numel() / hidden);
  TORCH_CHECK(weight.numel() == hidden, "weight must be [hidden]");
  TORCH_CHECK(hidden % 128 == 0, "hidden must be a multiple of 128");
  TORCH_CHECK(output_s.numel() == (int64_t)rows * (hidden / 128),
              "output_s must be [rows, hidden/128] float32");
  TORCH_CHECK(dispatch_norm_quant(write_norm ? 1 : 2, rows, hidden,
                                  input.data_ptr(), weight.data_ptr(),
                                  write_norm ? norm_out.data_ptr() : nullptr,
                                  output_q.data_ptr(), output_s.data_ptr()),
              "unsupported fused shape [rows=", rows, ", hidden=", hidden, "]");
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}

void moe_silu_mul_sm100(torch::Tensor input, torch::Tensor output) {
  c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "silu tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "silu is bf16 -> bf16");
  int const out = static_cast<int>(output.size(output.dim() - 1));
  int const rows = static_cast<int>(output.numel() / out);
  TORCH_CHECK(input.numel() == output.numel() * 2, "input must be gate|up");
  TORCH_CHECK(dispatch_silu_quant(false, rows, out, input.data_ptr(),
                                  output.data_ptr(), nullptr),
              "unsupported silu shape [rows=", rows, ", out=", out, "]");
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}

void moe_silu_mul_quantize_fp8_sm100(torch::Tensor input,
                                    torch::Tensor output_q,
                                    torch::Tensor output_s) {
  c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(input.is_contiguous() && output_q.is_contiguous() &&
                  output_s.is_contiguous(),
              "fused silu+quant tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(output_q.scalar_type() == at::kFloat8_e4m3fn,
              "output_q must be float8_e4m3fn");
  TORCH_CHECK(output_s.scalar_type() == at::kFloat, "output_s must be float32");
  int const out = static_cast<int>(output_q.size(output_q.dim() - 1));
  int const rows = static_cast<int>(output_q.numel() / out);
  TORCH_CHECK(input.numel() == output_q.numel() * 2, "input must be gate|up");
  TORCH_CHECK(out % 128 == 0, "intermediate size must be a multiple of 128");
  TORCH_CHECK(output_s.numel() == (int64_t)rows * (out / 128),
              "output_s must be [rows, out/128] float32");
  TORCH_CHECK(dispatch_silu_quant(true, rows, out, input.data_ptr(),
                                  output_q.data_ptr(), output_s.data_ptr()),
              "unsupported fused shape [rows=", rows, ", out=", out, "]");
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_fp8_f32scale_sm100",
        &quantize_fp8_f32scale_sm100,
        "fp32-scale per-token per-128-group FP8 quantization (the MoE "
        "activation variant), 2D or 3D input");
  m.def("moe_w13_ue8m0_sm100",
        &moe_w13_ue8m0_sm100,
        "Existing DSV3 grouped FP8 MoE GEMM (internal UE8M0 scales) at "
        "Qwen3.5 w13 shapes");
  m.def("moe_w2_ue8m0_sm100",
        &moe_w2_ue8m0_sm100,
        "Existing DSV3 grouped FP8 MoE GEMM (internal UE8M0 scales) at "
        "Qwen3.5 w2 shapes");
  m.def("moe_w13_blockscale_sm100",
        &moe_w13_blockscale_sm100,
        "fp32-block-scale grouped FP8 MoE GEMM at Qwen3.5 w13 shapes");
  m.def("moe_w2_blockscale_sm100",
        &moe_w2_blockscale_sm100,
        "fp32-block-scale grouped FP8 MoE GEMM at Qwen3.5 w2 shapes");
  m.def("moe_silu_mul_sm100",
        &moe_silu_mul_sm100,
        "MoE activation SwiGLU alone (the unfused reference half)");
  m.def("moe_silu_mul_quantize_fp8_sm100",
        &moe_silu_mul_quantize_fp8_sm100,
        "M4-I9: MoE activation SwiGLU fused with the fp32-block-scale FP8 "
        "quantize (one task)");
  m.def("rmsnorm_sm100",
        &rmsnorm_sm100,
        "RMS norm alone (the unfused reference half of M4-I9 flag A)");
  m.def("rmsnorm_quantize_fp8_sm100",
        &rmsnorm_quantize_fp8_sm100,
        "M4-I9 flag A: RMS norm fused with the fp32-block-scale FP8 quantize "
        "(one task); write_norm selects the 3-output or 2-output form");
}
