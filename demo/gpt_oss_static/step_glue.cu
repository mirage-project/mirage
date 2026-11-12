// step_glue.cu -- a decode step's glue in one launch (one request): the argmax
// of the last step's logits (torch's: the first index of the maximum), the
// generated token's slot, the step's token (the prompt's, else that argmax),
// its embedding row, and the page table for the position. Replaces argmax +
// index_select + copies + fills (~28 us of small launches a step on H100).
// Built by step_glue.py (static_schedule.compile_command, as the layers are),
// called through ctypes.
#include <climits>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256, PER = 16, BLOCK = THREADS * PER;

// an order-preserving signed key of the value (-0 as +0: torch compares the
// values), then the lower index on a tie
__device__ __forceinline__ long long key_of(float v, int i) {
  int const b = __float_as_int(v + 0.0f);
  int const hi = b >= 0 ? b : b ^ 0x7FFFFFFF;
  return ((long long)hi << 32) | (long long)(0xFFFFFFFFu - (unsigned)i);
}

__global__ void __launch_bounds__(THREADS) step_glue(float const *logits,
                                                     int V,
                                                     long long *key,
                                                     unsigned *count,
                                                     int nblk,
                                                     long long const *prompt,
                                                     int P,
                                                     int pos,
                                                     int has_prev,
                                                     long long *gen,
                                                     __nv_bfloat16 const *embed,
                                                     __nv_bfloat16 *x0,
                                                     int H,
                                                     int *kv_indptr,
                                                     int *last_page_len,
                                                     int page) {
  __shared__ long long red[THREADS / 32];
  __shared__ long long tok_s;
  __shared__ bool last;
  if (has_prev) {
    long long k = LLONG_MIN;
    int const base = blockIdx.x * BLOCK + threadIdx.x;
#pragma unroll
    for (int j = 0; j < PER; j++) {
      int const i = base + j * THREADS;
      if (i < V) {
        k = max(k, key_of(logits[i], i));
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      k = max(k, __shfl_xor_sync(0xffffffffu, k, o));
    }
    if ((threadIdx.x & 31) == 0) {
      red[threadIdx.x >> 5] = k;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
      for (int w = 1; w < THREADS / 32; w++) {
        k = max(k, red[w]);
      }
      atomicMax(key, k);
    }
  }
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(count, 1u) == (unsigned)nblk - 1;
  }
  __syncthreads();
  if (!last) {
    return;
  }
  if (threadIdx.x == 0) {
    __threadfence();
    long long tok = prompt[min(pos, P - 1)];
    if (has_prev) {
      long long const k = (long long)atomicExch((unsigned long long *)key,
                                                (unsigned long long)LLONG_MIN);
      long long const nxt = 0xFFFFFFFFll - (k & 0xFFFFFFFFll);
      if (pos >= P) {
        gen[pos - P] = nxt;
        tok = nxt;
      }
    }
    *count = 0;
    tok_s = tok;
    int const n = pos / page + 1;
    kv_indptr[1] = n;
    *last_page_len = pos + 1 - (n - 1) * page;
  }
  __syncthreads();
  __nv_bfloat16 const *row = embed + tok_s * H;
  if (H % 8 == 0) {
    for (int h = threadIdx.x; h < H / 8; h += THREADS) {
      ((uint4 *)x0)[h] = ((uint4 const *)row)[h];
    }
  } else {
    for (int h = threadIdx.x; h < H; h += THREADS) {
      x0[h] = row[h];
    }
  }
}

} // namespace

// key: one int64 at LLONG_MIN, count: one u32 at 0 (both left so)
extern "C" int step_glue_launch(float const *logits,
                                int V,
                                long long *key,
                                unsigned *count,
                                long long const *prompt,
                                int P,
                                int pos,
                                int has_prev,
                                long long *gen,
                                void const *embed,
                                void *x0,
                                int H,
                                int *kv_indptr,
                                int *last_page_len,
                                int page,
                                cudaStream_t stream) {
  int const nblk = has_prev ? (V + BLOCK - 1) / BLOCK : 1;
  step_glue<<<nblk, THREADS, 0, stream>>>(logits,
                                          V,
                                          key,
                                          count,
                                          nblk,
                                          prompt,
                                          P,
                                          pos,
                                          has_prev,
                                          gen,
                                          (__nv_bfloat16 const *)embed,
                                          (__nv_bfloat16 *)x0,
                                          H,
                                          kv_indptr,
                                          last_page_len,
                                          page);
  return (int)cudaGetLastError();
}
