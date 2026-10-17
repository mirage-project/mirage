#include "tasks/ampere/argmax.cuh"
#include "tasks/blackwell/argmax_sm100.cuh"

constexpr int BATCH = 8;
constexpr int CHUNK = 1600;
constexpr int PARTIALS = 96;
constexpr int VOCAB = 151936;

template <bool SM100>
__global__ void
    partial(void const *input, void *values, long long *indices, int active) {
  int chunk = blockIdx.x;
  auto *in = static_cast<kernel::bfloat16 const *>(input) + chunk * CHUNK;
  auto *out = static_cast<kernel::bfloat16 *>(values) + chunk;
  if constexpr (SM100) {
    kernel::argmax_partial_sm100_kernel<kernel::bfloat16,
                                        BATCH,
                                        CHUNK,
                                        PARTIALS,
                                        VOCAB>(
        in, out, indices + chunk, active, chunk * CHUNK);
  } else {
    kernel::
        argmax_partial_kernel<kernel::bfloat16, BATCH, CHUNK, PARTIALS, VOCAB>(
            in, out, indices + chunk, active, chunk * CHUNK);
  }
}

template <bool SM100>
__global__ void finish(void const *values,
                       long long const *indices,
                       long long *output,
                       int active) {
  if constexpr (SM100) {
    kernel::
        argmax_reduce_sm100_kernel<kernel::bfloat16, BATCH, CHUNK, PARTIALS>(
            values, indices, output, active);
  } else {
    kernel::argmax_reduce_kernel<kernel::bfloat16, BATCH, CHUNK, PARTIALS>(
        values, indices, output, active);
  }
}

extern "C" int launch(void const *input,
                      void *values,
                      long long *indices,
                      long long *output,
                      int active,
                      int implementation,
                      cudaStream_t stream) {
  if (active < 0 || active > BATCH || implementation < 0 ||
      implementation > 2) {
    return cudaErrorInvalidValue;
  }
  if (implementation == 0) {
    partial<false>
        <<<PARTIALS, 128, 512, stream>>>(input, values, indices, active);
  } else {
    partial<true>
        <<<PARTIALS, 128, 512, stream>>>(input, values, indices, active);
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  if (implementation == 2) {
    finish<true><<<1, 256, 512, stream>>>(values, indices, output, active);
  } else {
    finish<false><<<1, 256, 512, stream>>>(values, indices, output, active);
  }
  return cudaGetLastError();
}
