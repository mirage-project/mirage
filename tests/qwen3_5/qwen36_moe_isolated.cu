#include "tasks/common/qwen35_model.cuh"
#include "tasks/common/qwen36_moe.cuh"
#include "tasks/blackwell/topk_softmax_sm100.cuh"
#include <cuda_runtime.h>

using BF16 = type::bfloat16_t;

template <int BATCH>
__global__ void moe_sum(void const *expert_output, void const *scores,
                        void const *routing, void const *shared, void *output) {
  kernel::qwen36_moe_sum_task<BF16, BATCH, 8, 2048, 256>(
      expert_output, scores, routing, shared, output);
}

__global__ void shared_gate(void const *gate, void const *shared, void *output) {
  auto row = blockIdx.x;
  kernel::qwen36_shared_gate_task<BF16, 2048>(
      static_cast<BF16 const *>(gate) + row * 64,
      static_cast<BF16 const *>(shared) + row * 2048,
      static_cast<BF16 *>(output) + row * 2048);
}

extern "C" int launch_moe_sum(void const *expert_output, void const *scores,
                               void const *routing, void const *shared,
                               void *output, int batch) {
  if (batch == 1)
    moe_sum<1><<<1, 256>>>(expert_output, scores, routing, shared, output);
  else if (batch == 8)
    moe_sum<8><<<1, 256>>>(expert_output, scores, routing, shared, output);
  else if (batch == 16)
    moe_sum<16><<<1, 256>>>(expert_output, scores, routing, shared, output);
  else
    return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}

extern "C" int launch_shared_gate(void const *gate, void const *shared,
                                   void *output, int batch) {
  if (batch != 1 && batch != 8 && batch != 16) return int(cudaErrorInvalidValue);
  shared_gate<<<batch, 256>>>(gate, shared, output);
  return int(cudaGetLastError());
}

__global__ void topk(void *logits, void *scores, void *routing,
                     void *active, int batch) {
  kernel::topk_softmax_task_impl<BF16, 8, 256, 8, 16>(
      logits, nullptr, scores, batch, 8, routing, active, 0, 256,
      true, true);
}

extern "C" int launch_topk(void *logits, void *scores, void *routing,
                            void *active, int batch) {
  if (batch != 1 && batch != 8 && batch != 16) return int(cudaErrorInvalidValue);
  topk<<<1, 256, batch * 256 * sizeof(float)>>>(logits, scores, routing, active, batch);
  return int(cudaGetLastError());
}

__global__ void torch_style_router_softmax(BF16 const *logits, float *probs) {
  int row = blockIdx.x, lane = threadIdx.x;
  float elements[8];
  float peak = -INFINITY;
  for (int i = 0; i < 8; ++i) {
    elements[i] = float(logits[row * 256 + lane + 32 * i]);
    peak = peak > elements[i] ? peak : elements[i];
  }
  for (int offset = 16; offset; offset >>= 1)
    peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, offset));
  float sum = 0.f;
  for (int i = 0; i < 8; ++i) {
    elements[i] = expf(elements[i] - peak);
    sum = __fadd_rn(sum, elements[i]);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
  for (int i = 0; i < 8; ++i)
    probs[row * 256 + lane + 32 * i] = __fdiv_rn(elements[i], sum);
}

extern "C" int launch_router_softmax(void const *logits, void *probs, int batch) {
  if (batch != 1 && batch != 8 && batch != 16) return int(cudaErrorInvalidValue);
  torch_style_router_softmax<<<batch, 32>>>(
      static_cast<BF16 const *>(logits), static_cast<float *>(probs));
  return int(cudaGetLastError());
}
