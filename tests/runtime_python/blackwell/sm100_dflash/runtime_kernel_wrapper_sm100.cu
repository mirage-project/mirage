// Standalone pybind wrapper for the DFlash non-causal attention core kernel.
// Builds as `runtime_kernel_dflash`. The kernel is a plain (non-cluster,
// non-TMA) kernel so a direct __global__ launch is safe.
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "tasks/blackwell/dflash_attention_sm100.cuh"
#include "tasks/blackwell/dflash_norm_rope_sm100.cuh"

using bfloat16 = type::bfloat16_t;

template <typename T, int NH, int D>
__global__ void __launch_bounds__(256) dflash_norm_rope_k(void const *x,
                                                          void const *w,
                                                          void const *cos,
                                                          void const *sin,
                                                          void *o,
                                                          int n,
                                                          float eps) {
  kernel::dflash_norm_rope_sm100<T, NH, D>(x, w, cos, sin, o, n, eps);
}

// x:[N,NH,D] w:[D] cos/sin:[N,D] o:[N,NH,D]
void dflash_norm_rope(torch::Tensor x,
                      torch::Tensor w,
                      torch::Tensor cos,
                      torch::Tensor sin,
                      torch::Tensor o,
                      float eps) {
  int N = x.size(0);
  int NH = x.size(1);
  int D = x.size(2);
  dim3 grid(1, 1, 1), block(256, 1, 1);
  if (D == 128 && NH == 64) {
    dflash_norm_rope_k<bfloat16, 64, 128><<<grid, block>>>(x.data_ptr(),
                                                           w.data_ptr(),
                                                           cos.data_ptr(),
                                                           sin.data_ptr(),
                                                           o.data_ptr(),
                                                           N,
                                                           eps);
  } else if (D == 128 && NH == 8) {
    dflash_norm_rope_k<bfloat16, 8, 128><<<grid, block>>>(x.data_ptr(),
                                                          w.data_ptr(),
                                                          cos.data_ptr(),
                                                          sin.data_ptr(),
                                                          o.data_ptr(),
                                                          N,
                                                          eps);
  } else {
    printf("dflash_norm_rope: unsupported NH=%d D=%d\n", NH, D);
  }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("dflash_norm_rope launch error: %s\n", cudaGetErrorString(err));
  }
}

template <typename T, int NQ, int NKV, int D, int B>
__global__ void __launch_bounds__(256) dflash_attn_kernel(void const *q,
                                                          void const *k,
                                                          void const *v,
                                                          void *o,
                                                          int total_kv,
                                                          int sliding_window) {
  kernel::dflash_attention_sm100<T, NQ, NKV, D, B>(
      q, k, v, o, total_kv, sliding_window);
}

// q:[B,NQ,D] k:[T,NKV,D] v:[T,NKV,D] o:[B,NQ,D]; one request -> grid (1,1,1).
void dflash_attn(torch::Tensor q,
                 torch::Tensor k,
                 torch::Tensor v,
                 torch::Tensor o,
                 int sliding_window) {
  int B = q.size(0);
  int NQ = q.size(1);
  int D = q.size(2);
  int T = k.size(0);
  int NKV = k.size(1);
  dim3 grid(1, 1, 1);
  dim3 block(256, 1, 1);
  auto launch = [&](auto bct) {
    constexpr int BB = decltype(bct)::value;
    if (NQ == 64 && NKV == 8 && D == 128) {
      dflash_attn_kernel<bfloat16, 64, 8, 128, BB>
          <<<grid, block>>>(q.data_ptr(),
                            k.data_ptr(),
                            v.data_ptr(),
                            o.data_ptr(),
                            T,
                            sliding_window);
    } else {
      printf("dflash_attn: unsupported NQ=%d NKV=%d D=%d\n", NQ, NKV, D);
    }
  };
  if (B == 8) {
    launch(std::integral_constant<int, 8>{});
  } else if (B == 1) {
    launch(std::integral_constant<int, 1>{});
  } else {
    printf("dflash_attn: unsupported B=%d\n", B);
  }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("dflash_attn launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "dflash_attn", &dflash_attn, "DFlash non-causal attention core (SM100)");
  m.def("dflash_norm_rope",
        &dflash_norm_rope,
        "DFlash per-head RMSNorm + RoPE (SM100)");
}
