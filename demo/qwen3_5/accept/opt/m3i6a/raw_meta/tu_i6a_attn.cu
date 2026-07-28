// M3-I6a compile-only resource probe for TASK_ATTN_SM100 at the REAL Qwen3.5
// full-attention shape.  NO CUDA API call, NO launch, NO GPU claim: this TU
// exists only so `ptxas -v` reports registers / spill / smem as a function of
// the pre-registered `max_tokens_per_pass` knob (= Q_PASS_SIZE, and, because
// task_register.cc overwrites `max_tokens` with it, also MAX_TOKENS).
//
// Shape comes from src/kernel/task_register.cc::register_paged_attention_sm100_task
// applied to python/mirage/mpk/models/qwen3_5/builder.py::_build_attention:
//   num_q_heads=16, num_kv_heads=2  -> NUM_QO_HEADS(template) = 16/2 = 8
//                                      NUM_KV_HEADS(template) = 1
//   head_dim=256                    -> HEAD_DIM = 256
//   KV_CACHE_STRIDE = head_dim*num_kv_heads         = 512
//   QKV_STRIDE      = (2*8+2)*256*2                 = 9216   (qkvg_dim)
//   O_STRIDE        = num_q_heads*head_dim          = 4096
//   PAGE_SIZE = 256, MAX_SEQ_LEN = 897 (unused by the body), gate = 1
//
// MMA_ITERS_M = ceil(MAX_TOKENS*8/16) = ceil(Q_PASS/2), and the per-thread
// accumulator is `float o[MMA_ITERS_M][HEAD_DIM/16][8]` = MMA_ITERS_M*128
// floats.  Q_PASS=4 (what ships) therefore asks for 256 floats = 1024 B/thread
// of accumulator alone, above the 255-register ceiling.  This TU measures what
// ptxas actually does about that.
#include "blackwell/attention_sm100.cuh"

using bf16 = kernel::bfloat16;

template <int Q_PASS>
__global__ __launch_bounds__(WORKER_NUM_THREADS, 1) void k_attn_qwen35(
    void const *qkv,
    void *kc,
    void *vc,
    void *out,
    int const *qo_indptr,
    int const *kv_indptr,
    int const *kv_indices,
    int const *kv_last,
    int16_t rid,
    void const *qn,
    void const *kn,
    void const *cos,
    void const *sin) {
  kernel::multitoken_paged_attention_sm100_task_impl<
      bf16,
      /*NUM_QO_HEADS=*/8,
      /*NUM_KV_HEADS=*/1,
      /*KV_CACHE_STRIDE=*/512,
      /*QKV_STRIDE=*/9216,
      /*O_STRIDE=*/4096,
      /*HEAD_DIM=*/256,
      /*MAX_SEQ_LEN=*/897,
      /*PAGE_SIZE=*/256,
      /*Q_LEN_OVERRIDE=*/0,
      /*TAIL_OFFSET=*/0,
      /*MAX_TOKENS=*/Q_PASS,
      /*ATTN_OUTPUT_GATE=*/1,
      /*Q_PASS_SIZE=*/Q_PASS>(qkv,
                              kc,
                              vc,
                              out,
                              qo_indptr,
                              kv_indptr,
                              kv_indices,
                              kv_last,
                              rid,
                              /*qk_norm=*/true,
                              /*rope=*/true,
                              qn,
                              kn,
                              cos,
                              sin,
                              1e-6f,
                              1e-6f);
}

#define INST(Q)                                                                \
  template __global__ void k_attn_qwen35<Q>(void const *,                      \
                                            void *,                            \
                                            void *,                            \
                                            void *,                            \
                                            int const *,                       \
                                            int const *,                       \
                                            int const *,                       \
                                            int const *,                       \
                                            int16_t,                           \
                                            void const *,                      \
                                            void const *,                      \
                                            void const *,                      \
                                            void const *);

#ifdef I6A_QPASS
INST(I6A_QPASS)
#else
INST(1)
INST(2)
INST(3)
INST(4)
#endif
