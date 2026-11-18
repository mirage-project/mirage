#include "tasks/common/common_header.cuh"
#include "tasks/hopper/tma.cuh"
namespace kernel {
using namespace tma;
}
#include "tasks/hopper/multitoken_paged_attention_hopper.cuh"

constexpr int SMEM_BYTES = 205824;
constexpr int D = TEST_HEAD_DIM;
constexpr int GROUP = TEST_QO_PER_KV;
constexpr int HEADS = TEST_KV_HEADS;
constexpr int TOKENS = TEST_MAX_TOKENS;
constexpr int PAGE_STRIDE = TEST_PAGE_STRIDE;
constexpr int ROTARY = TEST_ROTARY_DIM;

template <bool NORMALIZE, bool ROPE, bool POISON>
__global__ void attention(void *qkv,
                          void *kc,
                          void *vc,
                          void *out,
                          void *q_scale,
                          void *k_scale,
                          void *cos,
                          void *sin,
                          int *qo,
                          int *pi,
                          int *pages,
                          int *last) {
  if constexpr (POISON) {
    extern __shared__ int poison[];
    for (int i = threadIdx.x; i < SMEM_BYTES / 4; i += blockDim.x) {
      poison[i] = 0x7fffffff;
    }
    __syncthreads();
  }
  auto offset = [](void *p, int elements) {
    return static_cast<type::bfloat16_t *>(p) + elements;
  };
  int head = blockIdx.x;
  kernel::multitoken_paged_attention_hopper_impl<type::bfloat16_t,
                                                 GROUP,
                                                 1,
                                                 HEADS,
                                                 HEADS * D,
                                                 HEADS *(GROUP + 2) * D,
                                                 HEADS * GROUP * D,
                                                 D,
                                                 -1,
                                                 64,
                                                 64,
                                                 TOKENS,
                                                 false,
                                                 1,
                                                 ROTARY,
                                                 PAGE_STRIDE,
                                                 TEST_FINITE_MASK>(
      offset(kc, head * D),
      offset(vc, head * D),
      qo,
      pi,
      pages,
      last,
      blockIdx.y,
      NORMALIZE,
      ROPE,
      q_scale,
      k_scale,
      cos,
      sin,
      1e-6f,
      1e-6f,
      offset(qkv, head * (GROUP + 2) * D),
      offset(out, head * GROUP * D));
}

void const *attention_kernel(int normalize, int poison) {
  static void const *kernels[4][2] = {
      {(void const *)attention<false, false, false>,
       (void const *)attention<false, false, true>},
      {(void const *)attention<true, true, false>,
       (void const *)attention<true, true, true>},
      {(void const *)attention<false, true, false>,
       (void const *)attention<false, true, true>},
      {(void const *)attention<true, false, false>,
       (void const *)attention<true, false, true>}};
  return kernels[normalize][poison];
}

extern "C" int configure() {
  for (int normalize = 0; normalize < 4; ++normalize) {
    for (int poison = 0; poison < 2; ++poison) {
      auto error =
          cudaFuncSetAttribute(attention_kernel(normalize, poison),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               SMEM_BYTES);
      if (error != cudaSuccess) {
        return error;
      }
    }
  }
  return cudaSuccess;
}

extern "C" int enqueue(void *qkv,
                       void *kc,
                       void *vc,
                       void *out,
                       void *q_scale,
                       void *k_scale,
                       void *cos,
                       void *sin,
                       int *qo,
                       int *pi,
                       int *pages,
                       int *last,
                       int batch,
                       int normalize,
                       int poison,
                       void *stream) {
  void *args[] = {&qkv,
                  &kc,
                  &vc,
                  &out,
                  &q_scale,
                  &k_scale,
                  &cos,
                  &sin,
                  &qo,
                  &pi,
                  &pages,
                  &last};
  return cudaLaunchKernel(attention_kernel(normalize, poison),
                          dim3(HEADS, batch),
                          dim3(256),
                          args,
                          SMEM_BYTES,
                          static_cast<cudaStream_t>(stream));
}

extern "C" int resources(int normalize, int *values) {
  cudaFuncAttributes attributes;
  auto error =
      cudaFuncGetAttributes(&attributes, attention_kernel(normalize, 0));
  if (error == cudaSuccess) {
    values[0] = attributes.numRegs;
    values[1] = attributes.localSizeBytes;
  }
  return error;
}
