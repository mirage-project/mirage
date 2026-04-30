// Direct kernel-wrapper test for the MPK FP8 swap-AB Linear kernel.
//
// Bypasses MPK's persistent-kernel runtime entirely and launches a single
// CTA via __global__ wrapper. Used to debug the kernel body in isolation
// with compute-sanitizer / cuda-gdb (build with SM100_LINEAR_FP8_MPK_DEBUG=1
// to enable -G device debug info), and to benchmark the kernel directly
// across DeepSeek V3 representative shapes via bench_linear_fp8_swapAB.py.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <tuple>

#include "tasks/blackwell/linear_fp8_swapAB_sm100.cuh"
#include "tasks/hopper/tma_2d.cuh"
#include <cute/tensor.hpp>

namespace {

// =========================================================================
// Per-shape compile-time configuration. The kernel needs (BATCH, OUTPUT, K)
// as template parameters because cute partition layouts and TMEM column
// counts are derived from them. Each entry below produces one nvcc
// instantiation; keep the set small to bound compile time.
// =========================================================================

constexpr int MMA_M = 128;
constexpr int MMA_N = 16;
constexpr int BLOCK_K = 128;
constexpr int NUM_AB_STAGE = 8;
constexpr int NUM_ACC_STAGE = 2;
constexpr int NUM_C_STAGE = 4;

constexpr int B_swz = 3;
constexpr int M_swz = 3;
constexpr int S_swz = 3;
constexpr int TMA_CP_ASYNC_SIZE = 128;
constexpr int TILE_SIZE = 128;

template <int BATCH, int OUTPUT_SIZE, int K_>
struct ShapeCfg {
  static_assert(OUTPUT_SIZE % 128 == 0, "OUTPUT_SIZE must be multiple of 128");
  static_assert(BATCH > 0 && BATCH <= 16, "BATCH must be in (0, 16]");
  static_assert(K_ % 128 == 0, "K must be multiple of 128");

  using TMA_A = kernel::tma::tma_2d<cutlass::float_e4m3_t,
                                    B_swz, M_swz, S_swz,
                                    /*GMEM_ROW=*/OUTPUT_SIZE,
                                    /*GMEM_COL=*/K_,
                                    /*SMEM_ROW=*/MMA_M,
                                    /*SMEM_COL=*/TMA_CP_ASYNC_SIZE,
                                    /*GMEM_STRIDE_ROW=*/K_,
                                    /*GMEM_STRIDE_COL=*/1,
                                    /*SMEM_REPEAT_ROW=*/1,
                                    /*SMEM_REPEAT_COL=*/(TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE,
                                    /*SMEM_STRIDE=*/MMA_M * TMA_CP_ASYNC_SIZE,
                                    /*ROW_MAJOR=*/true>;

  using TMA_B = kernel::tma::tma_2d<cutlass::float_e4m3_t,
                                    B_swz, M_swz, S_swz,
                                    /*GMEM_ROW=*/BATCH,
                                    /*GMEM_COL=*/K_,
                                    /*SMEM_ROW=*/MMA_N,
                                    /*SMEM_COL=*/TMA_CP_ASYNC_SIZE,
                                    /*GMEM_STRIDE_ROW=*/K_,
                                    /*GMEM_STRIDE_COL=*/1,
                                    /*SMEM_REPEAT_ROW=*/1,
                                    /*SMEM_REPEAT_COL=*/(TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE,
                                    /*SMEM_STRIDE=*/MMA_N * TMA_CP_ASYNC_SIZE,
                                    /*ROW_MAJOR=*/true>;

  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t,
                                      /*B=*/0, M_swz, S_swz,
                                      /*GMEM_ROW=*/BATCH,
                                      /*GMEM_COL=*/OUTPUT_SIZE,
                                      /*SMEM_ROW=*/MMA_N,
                                      /*SMEM_COL=*/MMA_M,
                                      /*GMEM_STRIDE_ROW=*/OUTPUT_SIZE,
                                      /*GMEM_STRIDE_COL=*/1,
                                      /*SMEM_REPEAT_ROW=*/1,
                                      /*SMEM_REPEAT_COL=*/1,
                                      /*SMEM_STRIDE=*/MMA_N * MMA_M,
                                      /*ROW_MAJOR=*/true>;
};

// =========================================================================
// CUtensorMap construction (host). The kernel uses kernel::tma::tma_2d
// which issues cp.async.bulk.tensor.5d, so the descriptor MUST be encoded
// rank=5 with trailing dims = 1. Encoding rank=2 is an illegal-instruction
// trap at runtime.
// =========================================================================

template <int BATCH, int K_>
CUtensorMap make_input_desc(void *gmem_ptr) {
  CUtensorMap desc;
  uint64_t gd[5] = {(uint64_t)K_, (uint64_t)BATCH, 1, 1, 1};
  uint64_t gs[4] = {(uint64_t)K_ * 1, 0, 0, 0};
  uint32_t bd[5] = {(uint32_t)BLOCK_K, (uint32_t)MMA_N, 1, 1, 1};
  uint32_t es[5] = {1, 1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, gmem_ptr, gd, gs, bd, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "input cuTensorMapEncodeTiled failed");
  return desc;
}

template <int OUTPUT_SIZE, int K_>
CUtensorMap make_weight_desc(void *gmem_ptr) {
  CUtensorMap desc;
  uint64_t gd[5] = {(uint64_t)K_, (uint64_t)OUTPUT_SIZE, 1, 1, 1};
  uint64_t gs[4] = {(uint64_t)K_ * 1, 0, 0, 0};
  uint32_t bd[5] = {(uint32_t)BLOCK_K, (uint32_t)MMA_M, 1, 1, 1};
  uint32_t es[5] = {1, 1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, gmem_ptr, gd, gs, bd, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "weight cuTensorMapEncodeTiled failed");
  return desc;
}

template <int BATCH, int OUTPUT_SIZE>
CUtensorMap make_output_desc(void *gmem_ptr) {
  CUtensorMap desc;
  uint64_t gd[5] = {(uint64_t)OUTPUT_SIZE, (uint64_t)BATCH, 1, 1, 1};
  uint64_t gs[4] = {(uint64_t)OUTPUT_SIZE * sizeof(cute::bfloat16_t), 0, 0, 0};
  uint32_t bd[5] = {(uint32_t)MMA_M, (uint32_t)MMA_N, 1, 1, 1};
  uint32_t es[5] = {1, 1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, gmem_ptr, gd, gs, bd, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "output cuTensorMapEncodeTiled failed");
  return desc;
}

// =========================================================================
// Templated __global__ kernel wrapper. One CTA, 256 threads.
// =========================================================================
template <int BATCH, int OUTPUT_SIZE, int K_>
__global__ void linear_fp8_swapAB_kernel_wrapper(
    CUtensorMap *input_desc,
    CUtensorMap *weight_desc,
    CUtensorMap *output_desc,
    uint32_t const *weight_scale_ptr,
    uint32_t const *input_scale_ptr) {
  using TMA_A = typename ShapeCfg<BATCH, OUTPUT_SIZE, K_>::TMA_A;
  using TMA_B = typename ShapeCfg<BATCH, OUTPUT_SIZE, K_>::TMA_B;
  using TMA_OUT = typename ShapeCfg<BATCH, OUTPUT_SIZE, K_>::TMA_OUT;

  TMA_A tma_a(weight_desc);   // A-side = weight (after swap)
  TMA_B tma_b(input_desc);    // B-side = input
  TMA_OUT tma_out(output_desc);

  // Dummy bias tensor (NOBIAS=true → never dereferenced).
  auto layout_bias = cute::make_layout(
      cute::make_shape(BATCH, OUTPUT_SIZE),
      cute::make_stride(OUTPUT_SIZE, cute::Int<1>{}));
  auto mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<cute::bfloat16_t *>(nullptr)),
      layout_bias);

  constexpr int packed_scale_k = (K_ + 511) / 512;

  kernel::linear_fp8_swapAB_sm100_task_impl<cutlass::float_e4m3_t,
                                         TMA_A, TMA_B, decltype(mBias), TMA_OUT,
                                         MMA_M, MMA_N, BATCH, OUTPUT_SIZE, K_,
                                         /*NOBIAS=*/true,
                                         /*SplitK=*/false,
                                         NUM_AB_STAGE, NUM_ACC_STAGE, NUM_C_STAGE>(
      tma_a, tma_b,
      weight_scale_ptr, input_scale_ptr,
      /*weight_scale_row_stride=*/packed_scale_k,
      /*input_scale_row_stride=*/packed_scale_k,
      mBias, tma_out);
}

// =========================================================================
// SMEM bytes computation (mirrors PipedSharedStorageWithSF size).
// =========================================================================
template <int BATCH, int OUTPUT_SIZE, int K_>
size_t compute_smem_bytes() {
  using T = cutlass::float_e4m3_t;
  using TypeC = cute::bfloat16_t;
  using Scale = uint32_t;
  constexpr size_t kSmemA = NUM_AB_STAGE * MMA_M * BLOCK_K * sizeof(T);
  constexpr size_t kSmemB = NUM_AB_STAGE * MMA_N * BLOCK_K * sizeof(T);
  constexpr size_t kSmemC = NUM_C_STAGE * MMA_N * MMA_M * sizeof(TypeC);
  constexpr size_t kSF_BLOCK_M = 128;
  constexpr size_t kSF_BLOCK_N = 128;
  constexpr size_t kSmemSFA = NUM_AB_STAGE * kSF_BLOCK_M * sizeof(Scale);
  constexpr size_t kSmemSFB = NUM_AB_STAGE * kSF_BLOCK_N * sizeof(Scale);
  constexpr size_t kBarriers = (NUM_AB_STAGE * 2 + NUM_ACC_STAGE * 2) * 8;
  constexpr size_t kTmemBookkeeping = 16;
  constexpr size_t kAlignSlack = 256;
  return kSmemA + kSmemB + kSmemC + kSmemSFA + kSmemSFB + kBarriers +
         kTmemBookkeeping + kAlignSlack;
}

// =========================================================================
// Per-shape launcher. Runs the kernel `repeat` times back-to-back on the
// current CUDA stream, sharing one allocation of the device descriptor
// memory across iterations. For repeat=1 this matches the original
// per-call cost; bench_linear_fp8_swapAB.py uses repeat>1 to amortize the
// cudaMalloc/cudaMemcpy/cudaLaunchKernelEx per-call overhead (~150-200µs)
// so the timed signal is closer to actual kernel execution.
// =========================================================================
template <int BATCH, int OUTPUT_SIZE, int K_>
void launch_linear_fp8_swapAB(torch::Tensor &input_q,
                           torch::Tensor &input_scale,
                           torch::Tensor &weight_q,
                           torch::Tensor &weight_scale,
                           torch::Tensor &output,
                           int repeat) {
  // Encode descriptors host-side, upload to device.
  CUtensorMap host_in_desc = make_input_desc<BATCH, K_>(input_q.data_ptr());
  CUtensorMap host_w_desc = make_weight_desc<OUTPUT_SIZE, K_>(weight_q.data_ptr());
  CUtensorMap host_out_desc = make_output_desc<BATCH, OUTPUT_SIZE>(output.data_ptr());

  CUtensorMap *d_in_desc = nullptr;
  CUtensorMap *d_w_desc = nullptr;
  CUtensorMap *d_out_desc = nullptr;
  TORCH_CHECK(cudaMalloc(&d_in_desc, sizeof(CUtensorMap)) == cudaSuccess);
  TORCH_CHECK(cudaMalloc(&d_w_desc, sizeof(CUtensorMap)) == cudaSuccess);
  TORCH_CHECK(cudaMalloc(&d_out_desc, sizeof(CUtensorMap)) == cudaSuccess);
  TORCH_CHECK(cudaMemcpy(d_in_desc, &host_in_desc, sizeof(CUtensorMap),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  TORCH_CHECK(cudaMemcpy(d_w_desc, &host_w_desc, sizeof(CUtensorMap),
                         cudaMemcpyHostToDevice) == cudaSuccess);
  TORCH_CHECK(cudaMemcpy(d_out_desc, &host_out_desc, sizeof(CUtensorMap),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  size_t smem_bytes = compute_smem_bytes<BATCH, OUTPUT_SIZE, K_>();
  TORCH_CHECK(
      cudaFuncSetAttribute(linear_fp8_swapAB_kernel_wrapper<BATCH, OUTPUT_SIZE, K_>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem_bytes)) == cudaSuccess,
      "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed");

  // Bump per-thread stack — CuTe + tcgen05 in debug builds (and even some
  // release builds) overflows the default 1024-byte stack. compute-sanitizer
  // synccheck reported "Stack overflow" inside linear_fp8_swapAB_sm100_task_impl.
  TORCH_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024) == cudaSuccess,
              "cudaDeviceSetLimit(StackSize) failed");

  dim3 grid(1, 1, 1);
  dim3 block(256, 1, 1);
  dim3 cluster(1, 1, 1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = cluster.x;
  attrs[0].val.clusterDim.y = cluster.y;
  attrs[0].val.clusterDim.z = cluster.z;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  uint32_t const *w_scale_ptr =
      static_cast<uint32_t const *>(weight_scale.data_ptr());
  uint32_t const *i_scale_ptr =
      static_cast<uint32_t const *>(input_scale.data_ptr());
  for (int i = 0; i < repeat; ++i) {
    cudaError_t le = cudaLaunchKernelEx(&cfg,
                                        linear_fp8_swapAB_kernel_wrapper<BATCH, OUTPUT_SIZE, K_>,
                                        d_in_desc,
                                        d_w_desc,
                                        d_out_desc,
                                        w_scale_ptr,
                                        i_scale_ptr);
    TORCH_CHECK(le == cudaSuccess,
                "cudaLaunchKernelEx failed: ", cudaGetErrorString(le));
  }

  // Free descriptor memory after kernel completion to avoid leaks. cudaFree
  // is stream-ordered on default stream (waits for prior kernels).
  cudaFree(d_in_desc);
  cudaFree(d_w_desc);
  cudaFree(d_out_desc);
}

// =========================================================================
// Supported shape table. Each (BATCH, OUTPUT_SIZE, K) here produces one
// nvcc instantiation, so keep the set small. Targeted at DeepSeek V3 dense
// FP8 layers under TP4 + decode-batch ≤ 16:
//   K = 1536   : q_b family (input from q_lora_rank)
//   K = 4608   : down (input from intermediate_size_per_tp = 18432/4)
//   K = 7168   : q_a / kv_a / o_proj input (hidden_size)
//   K = 16384  : o_proj raw (num_heads * v_head_dim = 128 * 128, before TP)
// =========================================================================
#define DISPATCH_SHAPE(b, n, k)                                              \
  if (BATCH == (b) && OUTPUT_SIZE == (n) && K == (k)) {                      \
    launch_linear_fp8_swapAB<(b), (n), (k)>(                                    \
        input_q, input_scale, weight_q, weight_scale, output, repeat);       \
    return;                                                                  \
  }

#define DISPATCH_FOR_BATCH(b)                                                \
  DISPATCH_SHAPE(b, 128, 128)                                                \
  DISPATCH_SHAPE(b, 128, 1536)                                               \
  DISPATCH_SHAPE(b, 128, 4608)                                               \
  DISPATCH_SHAPE(b, 128, 7168)                                               \
  DISPATCH_SHAPE(b, 128, 16384)                                              \
  DISPATCH_SHAPE(b, 256, 1536)                                               \
  DISPATCH_SHAPE(b, 256, 4608)                                               \
  DISPATCH_SHAPE(b, 256, 7168)                                               \
  DISPATCH_SHAPE(b, 512, 1536)                                               \
  DISPATCH_SHAPE(b, 512, 7168)

} // anonymous namespace

// =========================================================================
// Python-facing entry point. Dispatches on (BATCH, OUTPUT, K) at runtime.
// =========================================================================
void linear_fp8_swapAB_sm100(torch::Tensor input_q,      // [BATCH, K]    fp8_e4m3
                          torch::Tensor input_scale,  // [BATCH, ceil(K/512)] uint32 packed
                          torch::Tensor weight_q,     // [OUTPUT, K]   fp8_e4m3
                          torch::Tensor weight_scale, // [OUTPUT, ceil(K/512)] uint32 packed
                          torch::Tensor output,       // [BATCH, OUTPUT] bf16
                          int64_t repeat = 1) {
  TORCH_CHECK(input_q.dim() == 2 && weight_q.dim() == 2 && output.dim() == 2);
  TORCH_CHECK(input_q.is_contiguous() && weight_q.is_contiguous() &&
                  output.is_contiguous(),
              "all data tensors must be contiguous");
  TORCH_CHECK(input_q.scalar_type() == at::kFloat8_e4m3fn);
  TORCH_CHECK(weight_q.scalar_type() == at::kFloat8_e4m3fn);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16);
  TORCH_CHECK(input_scale.scalar_type() == at::kUInt32 ||
                  input_scale.scalar_type() == at::kInt);
  TORCH_CHECK(weight_scale.scalar_type() == at::kUInt32 ||
                  weight_scale.scalar_type() == at::kInt);

  int const BATCH = static_cast<int>(input_q.size(0));
  int const K = static_cast<int>(input_q.size(1));
  int const OUTPUT_SIZE = static_cast<int>(weight_q.size(0));
  TORCH_CHECK(weight_q.size(1) == K);
  TORCH_CHECK(output.size(0) == BATCH && output.size(1) == OUTPUT_SIZE);

  DISPATCH_FOR_BATCH(1)
  DISPATCH_FOR_BATCH(4)
  DISPATCH_FOR_BATCH(8)
  DISPATCH_FOR_BATCH(16)

  TORCH_CHECK(false, "Unsupported (BATCH=", BATCH, ", OUTPUT=", OUTPUT_SIZE,
              ", K=", K, ") — extend DISPATCH_FOR_BATCH in runtime_kernel_wrapper.cu");
}

// List of supported (BATCH, OUTPUT, K) triples — Python uses this to pick
// shapes that are pre-instantiated.
std::vector<std::tuple<int, int, int>> supported_shapes() {
  std::vector<std::tuple<int, int, int>> shapes;
  static constexpr int kNs[] = {128, 128, 128, 128, 128,
                                256, 256, 256, 512, 512};
  static constexpr int kKs[] = {128, 1536, 4608, 7168, 16384,
                                1536, 4608, 7168, 1536, 7168};
  for (int b : {1, 4, 8, 16}) {
    for (size_t i = 0; i < sizeof(kNs) / sizeof(int); ++i) {
      shapes.emplace_back(b, kNs[i], kKs[i]);
    }
  }
  return shapes;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_fp8_swapAB_sm100", &linear_fp8_swapAB_sm100,
        "MPK FP8 swapAB Linear (multi-shape dispatch). Optional `repeat` "
        "argument runs the kernel that many times back-to-back on the "
        "current stream, sharing one descriptor allocation — used by the "
        "benchmark to amortize per-call overhead.",
        pybind11::arg("input_q"), pybind11::arg("input_scale"),
        pybind11::arg("weight_q"), pybind11::arg("weight_scale"),
        pybind11::arg("output"), pybind11::arg("repeat") = 1);
  m.def("supported_shapes", &supported_shapes,
        "List of pre-instantiated (BATCH, OUTPUT, K) triples");
}
