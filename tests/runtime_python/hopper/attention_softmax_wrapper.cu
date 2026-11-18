#include "tasks/common/common_header.cuh"
#include "tasks/hopper/tma.cuh"
namespace kernel {
using namespace tma;
}
#include "tasks/ampere/merge_splitkv.cuh"
#include "tasks/hopper/multitoken_paged_attention_hopper.cuh"

constexpr int SMEM_BYTES = 205824;

#ifndef MIRAGE_FINITE_REFERENCE_MASK
#define MIRAGE_FINITE_REFERENCE_MASK 0
#endif

template <bool SPLIT, bool NORMALIZE, bool POISON>
__global__ void attention(void *qkv,
                          void *kc,
                          void *vc,
                          void *out,
                          void *lse,
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
  constexpr int CHUNKS = SPLIT ? 2 : 1;
  constexpr bool FINITE_REFERENCE = MIRAGE_FINITE_REFERENCE_MASK && !SPLIT;
  int head = blockIdx.x;
  int chunk = blockIdx.y;
  auto offset = [](void *p, int elements) {
    return static_cast<type::bfloat16_t *>(p) + elements;
  };
  kernel::multitoken_paged_attention_hopper_impl<type::bfloat16_t,
                                                 2,
                                                 1,
                                                 8,
                                                 1024,
                                                 4096,
                                                 2048 * CHUNKS,
                                                 128,
                                                 256,
                                                 FINITE_REFERENCE ? 64 : 512,
                                                 512,
                                                 8,
                                                 SPLIT,
                                                 CHUNKS,
                                                 128,
                                                 0,
                                                 FINITE_REFERENCE>(
      offset(kc, head * 128),
      offset(vc, head * 128),
      qo,
      pi,
      pages,
      last,
      0,
      NORMALIZE,
      NORMALIZE,
      q_scale,
      k_scale,
      cos,
      sin,
      1e-6f,
      1e-6f,
      offset(qkv, head * 512),
      offset(out, (head * CHUNKS + chunk) * 256),
      static_cast<float *>(lse) + (head * CHUNKS + chunk) * 2,
      chunk);
}

__global__ void
    merge(void *lse, void *partial, int *qo, int *pi, int *last, void *output) {
  int head = blockIdx.x;
  kernel::merge_splitkv<type::bfloat16_t, 2, 1, 8, 128, 8, true, 2, 256, 512>(
      static_cast<float *>(lse) + head * 4,
      static_cast<type::bfloat16_t *>(partial) + head * 512,
      qo,
      pi,
      last,
      0,
      static_cast<type::bfloat16_t *>(output) + head * 256,
      0);
}

void const *attention_kernel(int split, int normalize, int poison) {
  static void const *kernels[2][2][2] = {
      {{(void const *)attention<false, false, false>,
        (void const *)attention<false, false, true>},
       {(void const *)attention<false, true, false>,
        (void const *)attention<false, true, true>}},
      {{(void const *)attention<true, false, false>,
        (void const *)attention<true, false, true>},
       {(void const *)attention<true, true, false>,
        (void const *)attention<true, true, true>}}};
  return kernels[split][normalize][poison];
}

extern "C" int configure() {
  for (int split = 0; split < 2; ++split) {
    for (int normalize = 0; normalize < 2; ++normalize) {
      for (int poison = 0; poison < 2; ++poison) {
        auto error =
            cudaFuncSetAttribute(attention_kernel(split, normalize, poison),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 SMEM_BYTES);
        if (error != cudaSuccess) {
          return error;
        }
      }
    }
  }
  return cudaSuccess;
}

extern "C" int merge_only(void *lse,
                          void *partial,
                          int *qo,
                          int *pi,
                          int *last,
                          void *output,
                          void *stream_ptr) {
  merge<<<8, 128, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
      lse, partial, qo, pi, last, output);
  return cudaGetLastError();
}

extern "C" int enqueue(void *qkv,
                       void *kc,
                       void *vc,
                       void *output,
                       void *partial,
                       void *lse,
                       void *q_scale,
                       void *k_scale,
                       void *cos,
                       void *sin,
                       int *qo,
                       int *pi,
                       int *pages,
                       int *last,
                       int split,
                       int normalize,
                       int poison,
                       void *stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  void *out = split ? partial : output;
  void *args[] = {&qkv,
                  &kc,
                  &vc,
                  &out,
                  &lse,
                  &q_scale,
                  &k_scale,
                  &cos,
                  &sin,
                  &qo,
                  &pi,
                  &pages,
                  &last};
  auto error = cudaLaunchKernel(attention_kernel(split, normalize, poison),
                                dim3(8, split ? 2 : 1),
                                dim3(256),
                                args,
                                SMEM_BYTES,
                                stream);
  if (error != cudaSuccess || !split) {
    return error;
  }
  return merge_only(lse, partial, qo, pi, last, output, stream_ptr);
}

extern "C" int resources(int split, int normalize, int *values) {
  cudaFuncAttributes attributes;
  auto error = cudaFuncGetAttributes(
      &attributes,
      split == 2 ? (void const *)merge : attention_kernel(split, normalize, 0));
  if (error != cudaSuccess) {
    return error;
  }
  values[0] = attributes.numRegs;
  values[1] = attributes.sharedSizeBytes;
  values[2] = attributes.localSizeBytes;
  return cudaSuccess;
}

__global__ void
    score_transform(float const *input, float *output, int count, int masks) {
  int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (i + 1 < count) {
    float2 scores = kernel::finite_mask_bf16_pair(
        input[i], input[i + 1], 1.f / sqrtf(128.f), masks & 1, masks & 2);
    output[i] = scores.x;
    output[i + 1] = scores.y;
  }
}

extern "C" int finite_scores(
    void const *input, void *output, int count, int masks, void *stream_ptr) {
  score_transform<<<(count + 511) / 512,
                    256,
                    0,
                    static_cast<cudaStream_t>(stream_ptr)>>>(
      static_cast<float const *>(input),
      static_cast<float *>(output),
      count,
      masks);
  return cudaGetLastError();
}
