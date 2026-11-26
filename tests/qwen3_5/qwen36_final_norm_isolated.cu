#include "tasks/common/qwen35_gdn.cuh"
#include <cuda_runtime.h>

using BF16 = type::bfloat16_t;

__global__ void qwen36_final_norm(BF16 const *input, float const *weight,
                                  BF16 *output) {
  int row = blockIdx.x;
  kernel::qwen35_norm_task<BF16, 2048, 8>(
      input + row * 2048, weight, output + row * 2048);
}

extern "C" int launch_qwen36_final_norm(void const *input, void const *weight,
                                         void *output) {
  qwen36_final_norm<<<8, 256>>>(static_cast<BF16 const *>(input),
                                static_cast<float const *>(weight),
                                static_cast<BF16 *>(output));
  return int(cudaGetLastError());
}
