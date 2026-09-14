// Poison shared memory before invoking the production attention device function.
#include "tasks/common/common_header.cuh"
#include "tasks/hopper/tma.cuh"
namespace kernel {
using namespace tma;
}
#include "tasks/hopper/multitoken_paged_attention_hopper.cuh"

constexpr int SMEM_BYTES = 205824;

template <int MAX_TOKENS>
__global__ void run_attention(void *qkv, void *kc, void *vc, void *out,
                              int *qo, int *pi, int *pages, int *last) {
  extern __shared__ int poison[];
  for (int i = threadIdx.x; i < SMEM_BYTES / 4; i += blockDim.x) {
    poison[i] = 0x7fffffff; // NaN in both bf16 and float32 interpretations.
  }
  __syncthreads();
  kernel::multitoken_paged_attention_hopper_impl<
      type::bfloat16_t, 8, 1, 1, 128, 1280, 1024, 128, -1, 256, 256,
      MAX_TOKENS, false>(kc, vc, qo, pi, pages, last, 0, false, false,
                        nullptr, nullptr, nullptr, nullptr, 1e-6f, 1e-6f,
                        qkv, out);
}

template <int MAX_TOKENS>
int launch_attention(void *qkv, void *kc, void *vc, void *out,
                     int *qo, int *pi, int *pages, int *last) {
  cudaError_t result = cudaFuncSetAttribute(
      run_attention<MAX_TOKENS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      SMEM_BYTES);
  if (result != cudaSuccess) {
    return result;
  }
  run_attention<MAX_TOKENS><<<1, 256, SMEM_BYTES>>>(
      qkv, kc, vc, out, qo, pi, pages, last);
  return cudaDeviceSynchronize();
}

extern "C" int launch(void *qkv, void *kc, void *vc, void *out,
                      int *qo, int *pi, int *pages, int *last, int max_tokens) {
  if (max_tokens == 1) {
    return launch_attention<1>(qkv, kc, vc, out, qo, pi, pages, last);
  }
  return launch_attention<8>(qkv, kc, vc, out, qo, pi, pages, last);
}
