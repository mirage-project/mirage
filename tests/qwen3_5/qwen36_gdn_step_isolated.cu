#include "tasks/common/qwen35_gdn.cuh"
#include <cuda_runtime.h>

using BF16 = type::bfloat16_t;

__global__ void qwen36_conv_step(BF16 const *qkv, BF16 const *weight,
                                 BF16 *conv_state, BF16 *conv) {
  int row = blockIdx.y, head = blockIdx.x;
  int channel = row * 8192 + head * 128;
  kernel::qwen35_conv_task<BF16, 128>(
      qkv + channel, weight + head * 128 * 4,
      conv_state + channel * 3, conv + channel);
}

__global__ void qwen36_delta_step(
    BF16 const *conv, BF16 const *z, BF16 const *ab,
    float const *parameters, BF16 const *norm_weight,
    float *state, int64_t const *position, int64_t const *head_ids,
    BF16 *raw, BF16 *gated) {
  int row = blockIdx.y, head = blockIdx.x;
  int value_offset = row * 4096 + head * 128;
  kernel::qwen35_delta_task<BF16, BF16, 32, 128, 16>(
      conv + row * 8192, z + value_offset, ab + row * 64 + head,
      parameters + head, norm_weight,
      state + (row * 32 + head) * 128 * 128, position,
      raw + value_offset, gated + value_offset, head_ids + head);
}

extern "C" int launch_qwen36_conv_step(void const *qkv, void const *weight,
                                        void *conv_state, void *conv) {
  qwen36_conv_step<<<dim3(64, 8), 256>>>(
      static_cast<BF16 const *>(qkv), static_cast<BF16 const *>(weight),
      static_cast<BF16 *>(conv_state), static_cast<BF16 *>(conv));
  return int(cudaGetLastError());
}

extern "C" int launch_qwen36_delta_step(
    void const *conv, void const *z, void const *ab,
    void const *parameters, void const *norm_weight,
    void *state, void const *position, void const *head_ids,
    void *raw, void *gated) {
  qwen36_delta_step<<<dim3(32, 8), 256>>>(
      static_cast<BF16 const *>(conv), static_cast<BF16 const *>(z),
      static_cast<BF16 const *>(ab), static_cast<float const *>(parameters),
      static_cast<BF16 const *>(norm_weight), static_cast<float *>(state),
      static_cast<int64_t const *>(position), static_cast<int64_t const *>(head_ids),
      static_cast<BF16 *>(raw), static_cast<BF16 *>(gated));
  return int(cudaGetLastError());
}
