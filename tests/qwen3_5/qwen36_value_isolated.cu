#include "tasks/common/qwen35_attention.cuh"

using BF16 = type::bfloat16_t;

__global__ void qwen36_value_kernel(BF16 const *probabilities,
                                     BF16 const *gate,
                                     int64_t const *position,
                                     BF16 const *cache,
                                     int64_t const *heads,
                                     BF16 *output,
                                     int batch) {
  int head = blockIdx.x;
  int row = blockIdx.y;
  kernel::qwen35_value_task<BF16, 1024, false, 16, 0, 2, 8>(
      probabilities + (row * 16 + head) * 1024,
      gate + (row * 16 + head) * 256,
      position,
      cache + row * 1024 * 2 * 256,
      heads + head,
      output + (row * 16 + head) * 256,
      0, batch);
}

__global__ void qwen36_gemv_kernel(BF16 const *probabilities,
                                    BF16 const *gate,
                                    int64_t const *position,
                                    BF16 const *cache,
                                    BF16 *output,
                                    int configuration) {
  int head = blockIdx.x;
  int row = blockIdx.y;
  int d = threadIdx.x;
  int length = int(*position) + 1;
  int parts = configuration & 127;
  int chunk_code = (configuration >> 7) & 15;
  int chunk = chunk_code ? 1 << (chunk_code - 1) : (length + parts - 1) / parts;
  int order = (configuration >> 11) & 3;
  int inner = 1 << ((configuration >> 13) & 3);
  int inner_order = (configuration >> 15) & 3;
  bool contiguous_inner = configuration & (1 << 17);
  auto p = probabilities + (row * 16 + head) * 1024;
  auto g = gate + (row * 16 + head) * 256;
  auto v = cache + row * 1024 * 2 * 256 + (head / 8) * 256;
  auto y = output + (row * 16 + head) * 256;
  float partial[64] = {};
  for (int t = 0; t < length; ++t) {
    int lane = (t / chunk) % parts;
    int at = (t / (chunk * parts)) * chunk + t % chunk;
    int sub = contiguous_inner ? (at / ((chunk + inner - 1) / inner)) % inner : at % inner;
    int index = lane * inner + sub;
    partial[index] = __fmaf_rn(float(p[t]), float(v[t * 2 * 256 + d]), partial[index]);
  }
  for (int lane = 0; lane < parts; ++lane) {
    float *s = partial + lane * inner;
    if (inner_order == 2) {
      for (int i = 1; i < inner; ++i) s[0] = __fadd_rn(s[0], s[i]);
    } else if (inner_order == 1) {
      for (int offset = 1; offset < inner; offset <<= 1)
        for (int i = 0; i < inner; i += 2 * offset)
          s[i] = __fadd_rn(s[i], s[i + offset]);
    } else {
      for (int offset = inner / 2; offset; offset >>= 1)
        for (int i = 0; i < offset; ++i)
          s[i] = __fadd_rn(s[i], s[i + offset]);
    }
    partial[lane] = s[0];
  }
  if (order == 2) {
    for (int i = 1; i < parts; ++i) partial[0] = __fadd_rn(partial[0], partial[i]);
  } else if (order == 1) {
    for (int offset = 1; offset < parts; offset <<= 1)
      for (int i = 0; i < parts; i += 2 * offset)
        partial[i] = __fadd_rn(partial[i], partial[i + offset]);
  } else {
    for (int offset = parts / 2; offset; offset >>= 1)
      for (int i = 0; i < offset; ++i)
        partial[i] = __fadd_rn(partial[i], partial[i + offset]);
  }
  BF16 sigmoid = BF16(1.f / (1.f + expf(-float(g[d]))));
  y[d] = BF16(__fmul_rn(float(BF16(partial[0])), float(sigmoid)));
}

__global__ void qwen36_wave64_kernel(BF16 const *probabilities,
                                     BF16 const *gate,
                                     int64_t const *position,
                                     BF16 const *cache,
                                     BF16 *output) {
  int head = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
  int length = int(*position) + 1;
  int chunk = (length + 7) / 8;
  auto p = probabilities + (row * 16 + head) * 1024;
  auto g = gate + (row * 16 + head) * 256;
  auto v = cache + row * 1024 * 2 * 256 + (head / 8) * 256;
  auto y = output + (row * 16 + head) * 256;
  float lane_sum[8] = {};
  for (int lane = 0; lane < 8; ++lane) {
    int end = min((lane + 1) * chunk, length);
    for (int begin = lane * chunk; begin < end; begin += 64) {
      float tile_sum = 0.f;
      for (int t = begin; t < min(begin + 64, end); ++t)
        tile_sum = __fmaf_rn(float(p[t]), float(v[t * 2 * 256 + d]), tile_sum);
      lane_sum[lane] = __fadd_rn(lane_sum[lane], tile_sum);
    }
  }
  for (int lane = 1; lane < 8; ++lane)
    lane_sum[0] = __fadd_rn(lane_sum[0], lane_sum[lane]);
  BF16 sigmoid = BF16(1.f / (1.f + expf(-float(g[d]))));
  y[d] = BF16(__fmul_rn(float(BF16(lane_sum[0])), float(sigmoid)));
}

extern "C" int launch_qwen36_value(void const *probabilities,
                                    void const *gate,
                                    void const *position,
                                    void const *cache,
                                    void const *heads,
                                    void *output,
                                    int mode,
                                    int batch) {
  if (mode == 0) qwen36_value_kernel<<<dim3(16, batch), 256>>>(
      static_cast<BF16 const *>(probabilities),
      static_cast<BF16 const *>(gate),
      static_cast<int64_t const *>(position),
      static_cast<BF16 const *>(cache),
      static_cast<int64_t const *>(heads),
      static_cast<BF16 *>(output), batch);
  else if (mode == 9999) qwen36_wave64_kernel<<<dim3(16, batch), 256>>>(
      static_cast<BF16 const *>(probabilities),
      static_cast<BF16 const *>(gate),
      static_cast<int64_t const *>(position),
      static_cast<BF16 const *>(cache),
      static_cast<BF16 *>(output));
  else qwen36_gemv_kernel<<<dim3(16, batch), 256>>>(
      static_cast<BF16 const *>(probabilities),
      static_cast<BF16 const *>(gate),
      static_cast<int64_t const *>(position),
      static_cast<BF16 const *>(cache),
      static_cast<BF16 *>(output), mode - 100);
  return cudaGetLastError();
}
