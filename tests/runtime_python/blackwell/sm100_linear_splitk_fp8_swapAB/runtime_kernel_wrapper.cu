// Direct kernel-wrapper test for the split-K FP8 swap-AB Linear kernel.
//
// Each CTA consumes one K-slice of size K_per_task and reduce-adds its
// partial result into the shared output tile. The wrapper itself launches
// `split_k_factor` CTAs serially (each with its own per-slice TMA
// descriptors); the kernel template is instantiated with SplitK=true so
// the epilogue uses tma_reduce_add_async instead of tma_store_async.
//
// The MPK runtime would launch the same kernels concurrently across many
// SMs; this wrapper is just for direct correctness verification —
// equivalent to one M-shard worth of output and `split_k_factor` parallel
// K-shards collapsed onto the same SM.

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

// =========================================================================
// Per-shape TMA-type aliases. The kernel sees the per-task K only — the
// underlying gmem extent is full_K (a runtime value).
// =========================================================================
template <int BATCH, int OUTPUT_SIZE, int K_PER_TASK>
struct ShapeCfg {
  static_assert(OUTPUT_SIZE % 128 == 0, "OUTPUT_SIZE must be a multiple of 128");
  static_assert(BATCH > 0 && BATCH <= 16, "BATCH must be in (0, 16]");
  static_assert(K_PER_TASK % 128 == 0, "K_PER_TASK must be a multiple of 128");

  using TMA_A = kernel::tma::tma_2d<cutlass::float_e4m3_t,
                                    B_swz, M_swz, S_swz,
                                    /*GMEM_ROW=*/OUTPUT_SIZE,
                                    /*GMEM_COL=*/K_PER_TASK,
                                    /*SMEM_ROW=*/MMA_M,
                                    /*SMEM_COL=*/TMA_CP_ASYNC_SIZE,
                                    /*GMEM_STRIDE_ROW=*/K_PER_TASK,
                                    /*GMEM_STRIDE_COL=*/1,
                                    /*SMEM_REPEAT_ROW=*/1,
                                    /*SMEM_REPEAT_COL=*/(TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE,
                                    /*SMEM_STRIDE=*/MMA_M * TMA_CP_ASYNC_SIZE,
                                    /*ROW_MAJOR=*/true>;

  using TMA_B = kernel::tma::tma_2d<cutlass::float_e4m3_t,
                                    B_swz, M_swz, S_swz,
                                    /*GMEM_ROW=*/BATCH,
                                    /*GMEM_COL=*/K_PER_TASK,
                                    /*SMEM_ROW=*/MMA_N,
                                    /*SMEM_COL=*/TMA_CP_ASYNC_SIZE,
                                    /*GMEM_STRIDE_ROW=*/K_PER_TASK,
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
// Per-slice CUtensorMap construction. Each K-slice is encoded as a tensor
// view of shape [outer, K_per_task] with row stride = full_K bytes (so
// rows are spaced full_K bytes apart, but only K_per_task is in-extent).
// =========================================================================

CUtensorMap make_input_desc(void *gmem_base_for_slice,
                            int batch, int k_per_task, int full_k) {
  CUtensorMap desc;
  uint64_t gd[5] = {(uint64_t)k_per_task, (uint64_t)batch, 1, 1, 1};
  uint64_t gs[4] = {(uint64_t)full_k * 1 /*FP8 = 1 byte*/, 0, 0, 0};
  uint32_t bd[5] = {(uint32_t)BLOCK_K, (uint32_t)MMA_N, 1, 1, 1};
  uint32_t es[5] = {1, 1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, gmem_base_for_slice, gd, gs, bd, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "input cuTensorMapEncodeTiled failed");
  return desc;
}

CUtensorMap make_weight_desc(void *gmem_base_for_slice,
                             int output_size, int k_per_task, int full_k) {
  CUtensorMap desc;
  uint64_t gd[5] = {(uint64_t)k_per_task, (uint64_t)output_size, 1, 1, 1};
  uint64_t gs[4] = {(uint64_t)full_k * 1, 0, 0, 0};
  uint32_t bd[5] = {(uint32_t)BLOCK_K, (uint32_t)MMA_M, 1, 1, 1};
  uint32_t es[5] = {1, 1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, gmem_base_for_slice, gd, gs, bd, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "weight cuTensorMapEncodeTiled failed");
  return desc;
}

CUtensorMap make_output_desc(void *gmem_ptr,
                             int batch, int output_size) {
  CUtensorMap desc;
  uint64_t gd[5] = {(uint64_t)output_size, (uint64_t)batch, 1, 1, 1};
  uint64_t gs[4] = {(uint64_t)output_size * sizeof(cute::bfloat16_t), 0, 0, 0};
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
// Templated __global__ kernel wrapper. One CTA, 256 threads. SplitK=true.
// =========================================================================
template <int BATCH, int OUTPUT_SIZE, int K_PER_TASK>
__global__ void linear_splitk_fp8_swapAB_kernel_wrapper(
    CUtensorMap *input_desc,
    CUtensorMap *weight_desc,
    CUtensorMap *output_desc,
    uint32_t const *weight_scale_ptr,
    uint32_t const *input_scale_ptr,
    int weight_scale_row_stride,
    int input_scale_row_stride) {
  using TMA_A = typename ShapeCfg<BATCH, OUTPUT_SIZE, K_PER_TASK>::TMA_A;
  using TMA_B = typename ShapeCfg<BATCH, OUTPUT_SIZE, K_PER_TASK>::TMA_B;
  using TMA_OUT = typename ShapeCfg<BATCH, OUTPUT_SIZE, K_PER_TASK>::TMA_OUT;

  TMA_A tma_a(weight_desc);
  TMA_B tma_b(input_desc);
  TMA_OUT tma_out(output_desc);

  auto layout_bias = cute::make_layout(
      cute::make_shape(BATCH, OUTPUT_SIZE),
      cute::make_stride(OUTPUT_SIZE, cute::Int<1>{}));
  auto mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<cute::bfloat16_t *>(nullptr)),
      layout_bias);

  kernel::linear_fp8_swapAB_sm100_task_impl<cutlass::float_e4m3_t,
                                            TMA_A, TMA_B, decltype(mBias), TMA_OUT,
                                            MMA_M, MMA_N, BATCH, OUTPUT_SIZE, K_PER_TASK,
                                            /*NOBIAS=*/true,
                                            /*SplitK=*/true,
                                            NUM_AB_STAGE, NUM_ACC_STAGE, NUM_C_STAGE>(
      tma_a, tma_b,
      weight_scale_ptr, input_scale_ptr,
      weight_scale_row_stride, input_scale_row_stride,
      mBias, tma_out);
}

template <int BATCH, int OUTPUT_SIZE, int K_PER_TASK>
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
// Per-shape launcher. Loops over the `split_k_factor` slices, building a
// fresh per-slice TMA descriptor for input/weight and offsetting the scale
// pointer. Caller must pre-zero `output` — the kernel reduce-adds.
// `repeat` runs the whole sweep that many times to amortize per-call cost
// when benchmarking.
// =========================================================================
template <int BATCH, int OUTPUT_SIZE, int K_PER_TASK>
void launch_linear_splitk_fp8_swapAB(torch::Tensor &input_q,
                                     torch::Tensor &input_scale,
                                     torch::Tensor &weight_q,
                                     torch::Tensor &weight_scale,
                                     torch::Tensor &output,
                                     int split_k_factor,
                                     int repeat) {
  int const full_K = K_PER_TASK * split_k_factor;
  TORCH_CHECK(input_q.size(0) == BATCH && input_q.size(1) == full_K,
              "input_q shape mismatch with K_per_task * split_k_factor");
  TORCH_CHECK(weight_q.size(0) == OUTPUT_SIZE && weight_q.size(1) == full_K,
              "weight_q shape mismatch");
  TORCH_CHECK(output.size(0) == BATCH && output.size(1) == OUTPUT_SIZE);

  int const packed_k_per_task = (K_PER_TASK + 511) / 512;
  int const packed_k_full = (full_K + 511) / 512;

  // Allocate descriptor memory once and reuse per slice (only the contents
  // change). 3 descriptors per slice × split_k_factor slices.
  std::vector<CUtensorMap *> d_in_descs(split_k_factor, nullptr);
  std::vector<CUtensorMap *> d_w_descs(split_k_factor, nullptr);
  CUtensorMap *d_out_desc = nullptr;
  TORCH_CHECK(cudaMalloc(&d_out_desc, sizeof(CUtensorMap)) == cudaSuccess);
  CUtensorMap host_out_desc = make_output_desc(output.data_ptr(), BATCH, OUTPUT_SIZE);
  TORCH_CHECK(cudaMemcpy(d_out_desc, &host_out_desc, sizeof(CUtensorMap),
                         cudaMemcpyHostToDevice) == cudaSuccess);

  uint8_t *input_base = static_cast<uint8_t *>(input_q.data_ptr());
  uint8_t *weight_base = static_cast<uint8_t *>(weight_q.data_ptr());
  uint32_t const *iscale_base =
      static_cast<uint32_t const *>(input_scale.data_ptr());
  uint32_t const *wscale_base =
      static_cast<uint32_t const *>(weight_scale.data_ptr());

  for (int s = 0; s < split_k_factor; ++s) {
    void *in_slice = input_base + s * K_PER_TASK; // 1 byte per FP8 elem
    void *w_slice = weight_base + s * K_PER_TASK;
    CUtensorMap host_in = make_input_desc(in_slice, BATCH, K_PER_TASK, full_K);
    CUtensorMap host_w = make_weight_desc(w_slice, OUTPUT_SIZE, K_PER_TASK, full_K);
    TORCH_CHECK(cudaMalloc(&d_in_descs[s], sizeof(CUtensorMap)) == cudaSuccess);
    TORCH_CHECK(cudaMalloc(&d_w_descs[s], sizeof(CUtensorMap)) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(d_in_descs[s], &host_in, sizeof(CUtensorMap),
                           cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(d_w_descs[s], &host_w, sizeof(CUtensorMap),
                           cudaMemcpyHostToDevice) == cudaSuccess);
  }

  size_t smem_bytes = compute_smem_bytes<BATCH, OUTPUT_SIZE, K_PER_TASK>();
  TORCH_CHECK(
      cudaFuncSetAttribute(
          linear_splitk_fp8_swapAB_kernel_wrapper<BATCH, OUTPUT_SIZE, K_PER_TASK>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)) == cudaSuccess,
      "cudaFuncSetAttribute failed");
  TORCH_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024) == cudaSuccess);

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

  for (int r = 0; r < repeat; ++r) {
    for (int s = 0; s < split_k_factor; ++s) {
      // Scale pointers advance by `s * packed_k_per_task` per row; the row
      // stride passed to the kernel is `packed_k_full` so subsequent rows
      // step into the next batch/output row at the right slice.
      uint32_t const *w_scale_slice =
          wscale_base + s * packed_k_per_task;
      uint32_t const *i_scale_slice =
          iscale_base + s * packed_k_per_task;
      cudaError_t le = cudaLaunchKernelEx(
          &cfg,
          linear_splitk_fp8_swapAB_kernel_wrapper<BATCH, OUTPUT_SIZE, K_PER_TASK>,
          d_in_descs[s], d_w_descs[s], d_out_desc,
          w_scale_slice, i_scale_slice,
          packed_k_full, packed_k_full);
      TORCH_CHECK(le == cudaSuccess,
                  "cudaLaunchKernelEx failed: ", cudaGetErrorString(le));
    }
  }

  for (int s = 0; s < split_k_factor; ++s) {
    cudaFree(d_in_descs[s]);
    cudaFree(d_w_descs[s]);
  }
  cudaFree(d_out_desc);
}

// =========================================================================
// Supported (BATCH, OUTPUT, K_PER_TASK) shape table.
// =========================================================================
#define DISPATCH_SHAPE(b, n, k)                                              \
  if (BATCH == (b) && OUTPUT_SIZE == (n) && K_PER_TASK == (k)) {             \
    launch_linear_splitk_fp8_swapAB<(b), (n), (k)>(                          \
        input_q, input_scale, weight_q, weight_scale, output,                \
        split_k_factor, repeat);                                             \
    return;                                                                  \
  }

// K_per_task MUST be a multiple of 512 so the per-slice base offset on the
// packed UE8M0 scale buffer (4 logical scales per uint32, each spanning
// 128 K-elements → 512 K per uint32) lands on a uint32 boundary. Picking a
// split_k_factor that violates this would have us read scale bytes from
// the wrong slice. Caller-side check is in linear_splitk_fp8_swapAB_sm100.
#define DISPATCH_FOR_BATCH(b)                                                \
  DISPATCH_SHAPE(b, 128, 512)                                                \
  DISPATCH_SHAPE(b, 128, 1024)                                               \
  DISPATCH_SHAPE(b, 128, 1536)                                               \
  DISPATCH_SHAPE(b, 128, 2048)                                               \
  DISPATCH_SHAPE(b, 128, 3584)                                               \
  DISPATCH_SHAPE(b, 128, 4096)                                               \
  DISPATCH_SHAPE(b, 128, 4608)                                               \
  DISPATCH_SHAPE(b, 128, 7168)                                               \
  DISPATCH_SHAPE(b, 256, 2048)                                               \
  DISPATCH_SHAPE(b, 256, 4096)

} // anonymous namespace

// =========================================================================
// Python-facing entry point. Dispatches on (BATCH, OUTPUT, K_per_task).
// =========================================================================
void linear_splitk_fp8_swapAB_sm100(
    torch::Tensor input_q,      // [BATCH, full_K]    fp8_e4m3
    torch::Tensor input_scale,  // [BATCH, packed_full_K] uint32 packed UE8M0
    torch::Tensor weight_q,     // [OUTPUT, full_K]   fp8_e4m3
    torch::Tensor weight_scale, // [OUTPUT, packed_full_K] uint32 packed UE8M0
    torch::Tensor output,       // [BATCH, OUTPUT] bf16, MUST be pre-zeroed
    int64_t split_k_factor,
    int64_t repeat = 1) {
  TORCH_CHECK(input_q.dim() == 2 && weight_q.dim() == 2 && output.dim() == 2);
  TORCH_CHECK(input_q.is_contiguous() && weight_q.is_contiguous() &&
                  output.is_contiguous());
  TORCH_CHECK(input_q.scalar_type() == at::kFloat8_e4m3fn);
  TORCH_CHECK(weight_q.scalar_type() == at::kFloat8_e4m3fn);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16);
  TORCH_CHECK(input_scale.scalar_type() == at::kUInt32 ||
                  input_scale.scalar_type() == at::kInt);
  TORCH_CHECK(weight_scale.scalar_type() == at::kUInt32 ||
                  weight_scale.scalar_type() == at::kInt);
  TORCH_CHECK(split_k_factor >= 1, "split_k_factor must be >= 1");

  int const BATCH = static_cast<int>(input_q.size(0));
  int const full_K = static_cast<int>(input_q.size(1));
  int const OUTPUT_SIZE = static_cast<int>(weight_q.size(0));
  TORCH_CHECK(weight_q.size(1) == full_K);
  TORCH_CHECK(output.size(0) == BATCH && output.size(1) == OUTPUT_SIZE);
  TORCH_CHECK(full_K % split_k_factor == 0,
              "full_K must be divisible by split_k_factor");
  int const K_PER_TASK = full_K / static_cast<int>(split_k_factor);
  // K_per_task must be a multiple of 512 (= BLOCK_K * SCALES_PER_UINT32 =
  // 128 * 4) so the per-slice scale-pointer advance lands on a uint32
  // boundary. Picking a split factor that violates this misaligns scales.
  TORCH_CHECK(K_PER_TASK % 512 == 0,
              "K_per_task must be a multiple of 512 (split_k_factor must "
              "divide full_K / 512 evenly): got K_per_task=", K_PER_TASK);

  DISPATCH_FOR_BATCH(1)
  DISPATCH_FOR_BATCH(4)
  DISPATCH_FOR_BATCH(16)

  TORCH_CHECK(false, "Unsupported (BATCH=", BATCH, ", OUTPUT=", OUTPUT_SIZE,
              ", K_per_task=", K_PER_TASK,
              ") — extend DISPATCH_FOR_BATCH in runtime_kernel_wrapper.cu");
}

std::vector<std::tuple<int, int, int>> supported_shapes() {
  std::vector<std::tuple<int, int, int>> shapes;
  static constexpr int kNs[] = {128, 128, 128, 128, 128, 128, 128, 128, 256, 256};
  static constexpr int kKs[] = {512, 1024, 1536, 2048, 3584, 4096, 4608, 7168, 2048, 4096};
  for (int b : {1, 4, 16}) {
    for (size_t i = 0; i < sizeof(kNs) / sizeof(int); ++i) {
      shapes.emplace_back(b, kNs[i], kKs[i]);
    }
  }
  return shapes;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_splitk_fp8_swapAB_sm100", &linear_splitk_fp8_swapAB_sm100,
        "Split-K FP8 swapAB Linear (multi-shape dispatch). Output must be "
        "pre-zeroed by the caller. `split_k_factor` controls how many CTAs "
        "split the K dimension; `repeat` runs the full sweep that many "
        "times for benchmarking.",
        pybind11::arg("input_q"), pybind11::arg("input_scale"),
        pybind11::arg("weight_q"), pybind11::arg("weight_scale"),
        pybind11::arg("output"), pybind11::arg("split_k_factor"),
        pybind11::arg("repeat") = 1);
  m.def("supported_shapes", &supported_shapes,
        "List of pre-instantiated (BATCH, OUTPUT, K_per_task) triples");
}
