/* Copyright 2026 The Mirage Team. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "tasks/common/common_header.cuh"
namespace kernel {
// Multi-query FlashAttention numerical schedule, with reverse 64-key tiles,
// online rescaling, BF16 unnormalized probabilities, and FP32 accumulators.
// Each task computes one query/head. Split count and tile ranges reproduce
// the pinned FlashAttention heuristic for the complete prompt, independent
// of the compute tile currently being launched.
template <typename T, int CAPACITY, int KV_HEADS=4, int Q_PER_KV=6>
__device__ __forceinline__ void qwen35_prefill_sdpa_task(
    void const *query, void const *gate, void const *position,
    void const *key_cache, void const *value_cache, void const *head_id,
    void const *split_config, void *gated_output, void *scratch_ptr) {
  static_assert(CAPACITY == 1024, "SDPA scratch supports exactly 16 key tiles");
  int64_t pos = *static_cast<int64_t const *>(position);
  if (pos < 0 || pos >= CAPACITY) return;
  auto q = static_cast<T const *>(query);
  auto g = static_cast<T const *>(gate);
  auto k = static_cast<T const *>(key_cache);
  auto v = static_cast<T const *>(value_cache);
  auto y = static_cast<T *>(gated_output);
  int kv_head = int(*static_cast<int64_t const *>(head_id)) / Q_PER_KV;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int group = lane / 4, pair = (lane % 4) * 2;
  constexpr float scale_log2 = 0.0625f * 1.4426950408889634f;
  __shared__ float scores[64], row_max, sums[4], inverse_sum;
  __shared__ float previous_max, rescale;
  auto split_output=static_cast<float *>(scratch_ptr);
  auto split_lse=split_output+16*256;
  __shared__ float split_scale[16];
  __shared__ T weights[64];
  auto config=static_cast<int64_t const *>(split_config);
  int splits=int(config[0]), blocks_per_split=int(config[1]);
  for(int split=0;split<splits;++split) {
    int first=split*blocks_per_split;
    int last=min(int(pos)/64,(split+1)*blocks_per_split-1);
    if(first>last) {
      split_output[split*256+threadIdx.x]=0.f;
      if(threadIdx.x==0) split_lse[split]=-INFINITY;
      __syncthreads();
      continue;
    }
    float acc[4][4]={};
    if(threadIdx.x==0) row_max=-INFINITY;
    if(threadIdx.x<4) sums[threadIdx.x]=0.f;
    __syncthreads();
    for(int block=last;block>=first;--block) {
    int token = block * 64 + warp * 8;
    float dot[4] = {};
    #pragma unroll
    for (int base = 0; base < 256; base += 16) {
      uint32_t a[4] = {}, b[2] = {};
      #pragma unroll
      for (int half = 0; half < 2; ++half) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          int d = base + pair + half * 8 + j;
          if (group == 0) a[half * 2] |= uint32_t(__bfloat16_as_ushort(
              __float2bfloat16(float(q[d])))) << (16 * j);
          if (token + group <= pos) b[half] |= uint32_t(__bfloat16_as_ushort(
              __float2bfloat16(float(k[(token+group)*(KV_HEADS*256)+kv_head*256+d])))) << (16*j);
        }
      }
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(dot[0]), "+f"(dot[1]), "+f"(dot[2]), "+f"(dot[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    if (group == 0) {
      scores[warp*8+pair] = token+pair <= pos ? dot[0] : -INFINITY;
      scores[warp*8+pair+1] = token+pair+1 <= pos ? dot[1] : -INFINITY;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      previous_max = row_max;
      for (int i = 0; i < 64; ++i) row_max = fmaxf(row_max, scores[i]);
      rescale = block == last ? 0.f : exp2f(__fmul_rn(previous_max-row_max, scale_log2));
    }
    __syncthreads();
    if (threadIdx.x < 4) {
      float row_sum = __fmul_rn(sums[threadIdx.x], rescale);
      float peak = __fmul_rn(row_max, scale_log2);
      #pragma unroll
      for (int tile = 0; tile < 8; ++tile) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          int i = tile*8 + threadIdx.x*2+j;
          // PyTorch builds FlashAttention with UNFUSE_FMA.
          float p = exp2f(__fadd_rn(__fmul_rn(scores[i], scale_log2), -peak));
          row_sum = __fadd_rn(row_sum, p);
          weights[i] = T(p);
        }
      }
      sums[threadIdx.x] = row_sum;
    }
    __syncthreads();
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) acc[part][j] = __fmul_rn(acc[part][j], rescale);
      int channel = warp*32 + part*8;
      #pragma unroll
      for (int base = 0; base < 64; base += 16) {
        uint32_t a[4] = {}, b[2] = {};
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          int offset = base+pair+half*8;
          #pragma unroll
          for (int j = 0; j < 2; ++j) {
            int t = block*64+offset+j;
            if (group == 0) a[half*2] |= uint32_t(__bfloat16_as_ushort(
                __float2bfloat16(float(weights[offset+j])))) << (16*j);
            if (t <= pos) b[half] |= uint32_t(__bfloat16_as_ushort(
                __float2bfloat16(float(v[t*(KV_HEADS*256)+kv_head*256+channel+group])))) << (16*j);
          }
        }
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[part][0]), "+f"(acc[part][1]), "+f"(acc[part][2]), "+f"(acc[part][3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      }
    }
    __syncthreads();
  }
    if(threadIdx.x==0) {
      float total=__fadd_rn(__fadd_rn(sums[0],sums[2]),__fadd_rn(sums[1],sums[3]));
      inverse_sum=1.f/total;
      split_lse[split]=__fmaf_rn(row_max,.0625f,__logf(total));
    }
    __syncthreads();
    if(group==0) {
      for(int part=0;part<4;++part) for(int j=0;j<2;++j) {
        int d=warp*32+part*8+pair+j;
        split_output[split*256+d]=__fmul_rn(acc[part][j],inverse_sum);
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < 32 && splits > 1) {
    int width = 2;
    while (width < splits) width *= 2;
    float lse = lane < splits ? split_lse[lane] : -INFINITY;
    float peak = lse;
    for (int offset = width/2; offset; offset >>= 1)
      peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, offset));
    float total = expf(lse-peak);
    for (int offset = width/2; offset; offset >>= 1)
      total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, offset));
    float logsum = __fadd_rn(logf(total), peak);
    if (lane < splits) split_scale[lane] = expf(lse-logsum);
  }
  __syncthreads();
  int d = threadIdx.x;
  float result = split_output[d];
  if (splits > 1) {
    result = 0.f;
    for (int i = 0; i < splits; ++i)
      result = __fmaf_rn(split_scale[i], split_output[i*256+d], result);
  }
  T sigmoid = T(1.f / (1.f + expf(-float(g[d]))));
  y[d] = T(__fmul_rn(float(T(result)), float(sigmoid)));
  __syncthreads();
}
} // namespace kernel
