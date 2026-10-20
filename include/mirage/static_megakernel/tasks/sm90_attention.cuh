// tasks/sm90_attention.cuh -- the sm90_attention task type: MPK's Hopper paged
// attention (multitoken_paged_attention_hopper.cuh: RoPE, the new tokens' K / V
// appended to the paged cache, attention with GPT-OSS's per-head sinks and
// sliding window) of one request and KV head. Task (x, y) of grid (R, KVH, 1):
// request x, KV head y. The page tables are graph tensors the host fills
// before each step (MPK's runtime keeps them in its own buffers).
#pragma once
#include "sm90_common.cuh"
#include "tasks/hopper/multitoken_paged_attention_hopper.cuh"

namespace static_mk {

// KV head `head`'s columns of the QKV GEMM's fp32 rows (BUF_SLOTS 0) plus
// their bias (BUF_SLOTS B) to bf16 in BUF_SLOTS B + 1, the body's qkv: the
// bias node this saves was a dependency hop. Every task of the head writes the
// same values (the split's chunks).
template <class BUF_SLOTS, int B, int QH, int KVH, int D>
__device__ __forceinline__ kernel::bfloat16 *sm90_qkv_stage(G const &g,
                                                            int head) {
  constexpr int W = (QH / KVH + 2) * D, STRIDE = KVH * W, WV = W / 8;
  static_assert(W % 8 == 0, "sm90_attention: 16-B vectors");
  float const *in =
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)head * W;
  uint4 const *bias =
      buf_at<uint4 const>(g, slot_at<BUF_SLOTS, B>()) + (size_t)head * WV;
  kernel::bfloat16 *qkv =
      buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, B + 1>());
  uint4 *out = reinterpret_cast<uint4 *>(qkv) + (size_t)head * WV;
  for (int i = threadIdx.x; i < T * WV; i += SM90_THREADS) {
    int const t = i / WV, v = i % WV;
    float f[8], b[8];
    f32x8_load(in + (size_t)t * STRIDE, v, f);
    bf16x8_to_f32(bias[v], b);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      f[k] = f[k] + b[k];
    }
    out[(size_t)t * (STRIDE / 8) + v] = f32_to_bf16x8(f);
  }
  asm volatile("fence.proxy.async.global;" ::: "memory"); // (a TMA q load)
  __syncthreads();
  return qkv;
}

// MPK's body for KV head `head` of request `request`: qkv bf16 (every head's),
// BUF_SLOTS 1..8 as sm90_attention's (the caches, cos, sin, the page tables);
// PARTITION_KV: KV chunk kv_idx of C (SEQ_LEN positions), its log2-sum-exps to
// lse
template <class BUF_SLOTS,
          int QH,
          int KVH,
          int D,
          int MAX_SEQ,
          int PAGE,
          int SEQ_LEN,
          bool PARTITION_KV,
          int C,
          int WINDOW>
__device__ __forceinline__ void sm90_paged_attention(G const &g,
                                                     kernel::bfloat16 *qkv,
                                                     int request,
                                                     int head,
                                                     void *output,
                                                     float *lse,
                                                     int kv_idx,
                                                     void const *sinks) {
  constexpr int QPK = QH / KVH;
  using bf16 = kernel::bfloat16; // the body's own type (MPK's codegen's)
  kernel::multitoken_paged_attention_hopper_impl<bf16,
                                                 QPK,
                                                 1,
                                                 KVH,
                                                 KVH * D, // KV_CACHE_STRIDE
                                                 KVH *(QPK + 2) * D, // QKV
                                                 QH * D,             // O_STRIDE
                                                 D,
                                                 SEQ_LEN,
                                                 MAX_SEQ,
                                                 PAGE,
                                                 T,
                                                 PARTITION_KV,
                                                 C,
                                                 D, // ROTARY_DIM
                                                 0, // packed pages
                                                 WINDOW>(
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 1>()) + head * D,
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 2>()) + head * D,
      buf_at<int const>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 7>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 8>()),
      (int16_t)request,
      false, // qk_norm (GPT-OSS has none)
      true,  // RoPE
      nullptr,
      nullptr,
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 4>()),
      1e-6f,
      1e-6f,
      qkv + head * (QPK + 2) * D,
      output,
      lse,
      kv_idx,
      sinks);
}

// PARAMS {NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, PAGE_SIZE,
// WINDOW (0: full attention), HAS_SINK, QKV_BIAS}; BUF_SLOTS {qkv [T, KVH x
// (Q / KVH + 2) x D] (each KV head's q heads, k, v): bf16, or (QKV_BIAS) the
// GEMM's fp32, k cache and v cache [pages, PAGE, KVH, D] bf16, cos and sin
// [MAX_SEQ_LEN, D] bf16, qo_indptr [R + 1] i32, kv_indptr [R + 1] i32,
// kv_indices [pages] i32, last_page_len [R] i32, sinks [KVH, Q / KVH] bf16
// (-1: none), QKV_BIAS: the bias [KVH x (Q / KVH + 2) x D] bf16 and qkv's bf16
// rows (sm90_qkv_stage; else -1, -1), output [T, Q x D] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_attention(Maps const &,
                                                   G const &g,
                                                   KernelLocals &,
                                                   StaticTask const &tk) {
  constexpr int QH = PARAMS::v[0], KVH = PARAMS::v[1], D = PARAMS::v[2];
  constexpr bool HAS_SINK = PARAMS::v[6] == 1, QKV_BIAS = PARAMS::v[7] == 1;
  constexpr int QPK = QH / KVH;
  sm90_task_begin<IN...>(g);
  int const head = tk.y;
  kernel::bfloat16 *qkv =
      QKV_BIAS ? sm90_qkv_stage<BUF_SLOTS, 10, QH, KVH, D>(g, head)
               : buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 0>());
  sm90_paged_attention<BUF_SLOTS,
                       QH,
                       KVH,
                       D,
                       PARAMS::v[3],
                       PARAMS::v[4],
                       -1, // SEQ_LEN (split KV only)
                       false,
                       1,
                       PARAMS::v[5]>(
      g,
      qkv,
      tk.x,
      head,
      buf_at<kernel::bfloat16>(g, slot_at<BUF_SLOTS, 12>()) + head * QPK * D,
      nullptr,
      0,
      HAS_SINK ? buf_at<kernel::bfloat16 const>(g, slot_at<BUF_SLOTS, 9>()) +
                     head * QPK
               : nullptr);
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_attention, SM90_BODY_SMEM)

} // namespace static_mk
