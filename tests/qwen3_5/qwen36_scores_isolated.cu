#include "tasks/common/qwen35_attention.cuh"

using BF16 = type::bfloat16_t;

template <int MMA_K, bool FORCE>
__global__ void qwen36_scores(BF16 const *query, int64_t const *position,
                              BF16 const *keys, int64_t const *heads,
                              float *output) {
  int head = blockIdx.x, row = blockIdx.y;
  kernel::qwen35_scores_task<BF16, 1024, FORCE, MMA_K, 2, 8>(
      query + (row * 16 + head) * 256, position,
      keys + row * 1024 * 2 * 256, heads + head,
      output + (row * 16 + head) * 1024, int(gridDim.y));
}

extern "C" int launch_qwen36_scores(void const *query, void const *position,
                                     void const *keys, void const *heads,
                                     void *output, int batch) {
  qwen36_scores<16, false><<<dim3(16, batch), 256>>>(
      static_cast<BF16 const *>(query), static_cast<int64_t const *>(position),
      static_cast<BF16 const *>(keys), static_cast<int64_t const *>(heads),
      static_cast<float *>(output));
  return cudaGetLastError();
}

// Diagnostic GEMV schedules for comparing the production task with the
// pinned BF16 matmul. The mode encoding matches blackwell_scores.cu.
__global__ void qwen36_score_mode(BF16 const *query, int64_t const *position,
                                   BF16 const *keys, float *output,
                                   int configuration) {
  int head = blockIdx.x, row = blockIdx.y;
  int parts = configuration & 63;
  bool tiled = configuration & (1 << 15);
  int chunk = 1 << ((configuration >> 6) & 7);
  bool strided = configuration & 64;
  bool serial = tiled ? ((configuration >> 9) & 3) == 2 : configuration & 128;
  bool ascending = tiled ? ((configuration >> 9) & 3) == 1 : configuration & 256;
  auto q = query + (row * 16 + head) * 256;
  auto k = keys + row * 1024 * 2 * 256 + (head / 8) * 256;
  auto y = output + (row * 16 + head) * 1024;
  for (int t = threadIdx.x; t <= *position; t += blockDim.x) {
    float partial[32] = {};
    for (int d = 0; d < 256; ++d) {
      int lane = tiled ? (d / chunk) % parts :
                 strided ? d % parts : d / (256 / parts);
      partial[lane] = __fmaf_rn(float(q[d]),
                                float(k[t * 2 * 256 + d]), partial[lane]);
    }
    if (serial) {
      for (int i = 1; i < parts; ++i)
        partial[0] = __fadd_rn(partial[0], partial[i]);
    } else if (ascending) {
      for (int offset = 1; offset < parts; offset <<= 1)
        for (int i = 0; i < parts; i += 2 * offset)
          partial[i] = __fadd_rn(partial[i], partial[i + offset]);
    } else {
      for (int offset = parts / 2; offset; offset >>= 1)
        for (int i = 0; i < offset; ++i)
          partial[i] = __fadd_rn(partial[i], partial[i + offset]);
    }
    y[t] = float(BF16(__fmul_rn(float(BF16(partial[0])), .0625f)));
  }
}

extern "C" int launch_qwen36_scores_mode(void const *query,
                                           void const *position,
                                           void const *keys,
                                           void const *heads,
                                           void *output, int batch, int mode) {
  if (mode >= 100)
    qwen36_score_mode<<<dim3(16, batch), 256>>>(
        static_cast<BF16 const *>(query), static_cast<int64_t const *>(position),
        static_cast<BF16 const *>(keys), static_cast<float *>(output), mode - 100);
  else if (mode == 8)
    qwen36_scores<8, true><<<dim3(16, batch), 256>>>(
        static_cast<BF16 const *>(query), static_cast<int64_t const *>(position),
        static_cast<BF16 const *>(keys), static_cast<int64_t const *>(heads),
        static_cast<float *>(output));
  else if (mode == 16)
    qwen36_scores<16, true><<<dim3(16, batch), 256>>>(
        static_cast<BF16 const *>(query), static_cast<int64_t const *>(position),
        static_cast<BF16 const *>(keys), static_cast<int64_t const *>(heads),
        static_cast<float *>(output));
  else
    return launch_qwen36_scores(query, position, keys, heads, output, batch);
  return cudaGetLastError();
}
