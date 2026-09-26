// Standalone launcher for the same device tasks used by MPK.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "mirage/persistent_kernel/tasks/blackwell/sparse_mla_sm100.cuh"

struct LaunchArgs {
  void const *q;
  void const *cache;
  int const *indices;
  int const *counts;
  int const *qo;
  int const *ki;
  int const *pages;
  int const *last;
  void *output;
  float *partial;
  float *lse;
  int tokens, requests, num_pages, capacity, splits;
  float scale;
};

template <int H, int R, int PAGE, int SPLITS>
__global__ __launch_bounds__(256) void sparse_mla_wrapper(LaunchArgs a) {
  kernel::sparse_mla_sm100_task_impl<H, R, PAGE, SPLITS>(a.q,
                                                         a.cache,
                                                         a.indices,
                                                         a.counts,
                                                         a.output,
                                                         a.partial,
                                                         a.lse,
                                                         a.qo,
                                                         a.ki,
                                                         a.pages,
                                                         a.last,
                                                         a.requests,
                                                         a.num_pages,
                                                         a.scale,
                                                         a.capacity,
                                                         blockIdx.x,
                                                         blockIdx.y,
                                                         blockIdx.z);
}

template <int H, int SPLITS>
__global__ __launch_bounds__(256) void sparse_mla_reduce_wrapper(LaunchArgs a) {
  kernel::sparse_mla_reduce_sm100_task_impl<H, SPLITS>(
      a.partial, a.lse, a.output, blockIdx.x, blockIdx.y);
}

template <int H, int R, int PAGE, int SPLITS>
void launch(LaunchArgs a, cudaStream_t stream) {
  constexpr int smem = sizeof(kernel::sparse_mla::SharedStorage<R>);
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(sparse_mla_wrapper<H, R, PAGE, SPLITS>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem));
  dim3 grid(a.tokens, (H + 15) / 16, SPLITS);
  sparse_mla_wrapper<H, R, PAGE, SPLITS><<<grid, 256, smem, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if constexpr (SPLITS > 1) {
    dim3 reduce_grid(a.tokens, (H + 15) / 16, 1);
    sparse_mla_reduce_wrapper<H, SPLITS>
        <<<reduce_grid, 256, 16 * SPLITS * sizeof(float), stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <int H, int R, int PAGE>
void launch_splits(LaunchArgs a, cudaStream_t stream) {
  switch (a.splits) {
    case 1:
      launch<H, R, PAGE, 1>(a, stream);
      break;
    case 2:
      launch<H, R, PAGE, 2>(a, stream);
      break;
    case 4:
      launch<H, R, PAGE, 4>(a, stream);
      break;
    case 8:
      launch<H, R, PAGE, 8>(a, stream);
      break;
    default:
      TORCH_CHECK(false, "splits must be 1/2/4/8");
  }
}

template <int H>
void launch_shape(LaunchArgs a, int rope, int page_size, cudaStream_t stream) {
  if (rope == 0 && page_size == 64) {
    launch_splits<H, 0, 64>(a, stream);
  } else if (rope == 64 && page_size == 64) {
    launch_splits<H, 64, 64>(a, stream);
  } else if (rope == 0 && page_size == 128) {
    launch_splits<H, 0, 128>(a, stream);
  } else if (rope == 64 && page_size == 128) {
    launch_splits<H, 64, 128>(a, stream);
  } else {
    TORCH_CHECK(false, "standalone launcher supports R=0/64, page_size=64/128");
  }
}

void run(torch::Tensor q,
         torch::Tensor cache,
         torch::Tensor indices,
         torch::Tensor counts,
         torch::Tensor qo,
         torch::Tensor ki,
         torch::Tensor pages,
         torch::Tensor last,
         torch::Tensor output,
         torch::Tensor partial,
         torch::Tensor lse,
         double scale,
         int splits) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  c10::cuda::CUDAGuard guard(q.device());
  for (auto const &t :
       {q, cache, indices, counts, qo, ki, pages, last, output, partial, lse}) {
    TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.is_contiguous(),
                "all tensors must be contiguous on the same CUDA device");
  }
  for (auto const &t : {q, cache, output}) {
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16 && t.dim() == 3,
                "q/cache/output must be rank-3 BF16");
  }
  for (auto const &t : {indices, counts, qo, ki, pages, last}) {
    TORCH_CHECK(t.scalar_type() == torch::kInt32, "metadata must be int32");
  }
  for (auto const &t : {counts, qo, ki, pages, last}) {
    TORCH_CHECK(t.dim() == 1, "metadata must be rank 1");
  }
  TORCH_CHECK(partial.scalar_type() == torch::kFloat32 &&
                  lse.scalar_type() == torch::kFloat32,
              "workspace must be FP32");
  TORCH_CHECK(q.size(0) > 0 && q.size(0) <= INT32_MAX && cache.size(0) > 0 &&
                  cache.size(0) <= INT32_MAX,
              "invalid tensor capacities");
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == q.size(0) &&
                  indices.size(1) > 0 && indices.size(1) <= INT32_MAX,
              "indices must be [T, K]");
  TORCH_CHECK(counts.size(0) == q.size(0) && qo.numel() >= 2 &&
                  ki.numel() == qo.numel() && last.numel() == qo.numel() - 1,
              "invalid metadata shapes");
  TORCH_CHECK(cache.size(2) == q.size(2) && output.size(0) == q.size(0) &&
                  output.size(1) == q.size(1) && output.size(2) == 512,
              "invalid cache or output shape");
  TORCH_CHECK(output.data_ptr() != q.data_ptr() &&
                  output.data_ptr() != cache.data_ptr(),
              "output must not alias inputs");
  TORCH_CHECK(std::isfinite(scale) && std::isfinite(float(scale)) &&
                  float(scale) > 0,
              "scale must be positive finite float32");
  TORCH_CHECK(splits == 1 || splits == 2 || splits == 4 || splits == 8,
              "splits must be 1/2/4/8");
  if (splits > 1) {
    TORCH_CHECK(partial.numel() == q.size(0) * splits * q.size(1) * 512 &&
                    lse.numel() == q.size(0) * splits * q.size(1),
                "incorrect split workspace size");
  }
  LaunchArgs a{q.data_ptr(),
               cache.data_ptr(),
               indices.data_ptr<int>(),
               counts.data_ptr<int>(),
               qo.data_ptr<int>(),
               ki.data_ptr<int>(),
               pages.data_ptr<int>(),
               last.data_ptr<int>(),
               output.data_ptr(),
               partial.data_ptr<float>(),
               lse.data_ptr<float>(),
               int(q.size(0)),
               int(qo.numel() - 1),
               int(cache.size(0)),
               int(indices.size(1)),
               splits,
               float(scale)};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int const rope = q.size(2) - 512;
  int const page_size = cache.size(1);
  switch (q.size(1)) {
    case 8:
      launch_shape<8>(a, rope, page_size, stream);
      break;
    case 16:
      launch_shape<16>(a, rope, page_size, stream);
      break;
    case 32:
      launch_shape<32>(a, rope, page_size, stream);
      break;
    case 64:
      launch_shape<64>(a, rope, page_size, stream);
      break;
    default:
      TORCH_CHECK(false, "H must be 8/16/32/64");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "Sparse MLA and optional split reduction (SM100)");
}
