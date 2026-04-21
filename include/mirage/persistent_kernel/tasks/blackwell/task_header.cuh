// Ampere task impls
#include "tasks/ampere/embedding.cuh"
#include "tasks/ampere/merge_splitkv.cuh"
#include "tasks/ampere/multitoken_paged_attention_split_kv.cuh"
#include "tasks/ampere/silu_mul.cuh"
#ifdef USE_NVSHMEM
#include "tasks/ampere/allreduce.cuh"
#endif // USE_NVSHMEM
// Hopper task impls
#include "tasks/cute/hopper/gemm_ws.cuh"
#include "tasks/cute/hopper/gemm_ws_cooperative.cuh"
#include "tasks/cute/hopper/gemm_ws_mpk.cuh"
#include "tasks/hopper/linear_hopper.cuh"
#include "tasks/hopper/linear_swapAB_hopper.cuh"
#include "tasks/hopper/multitoken_paged_attention_hopper.cuh"
#include "tasks/hopper/rmsnorm_hopper.cuh"
#include "tasks/hopper/rotary_embedding_hopper.cuh"
#include "tasks/hopper/silu_mul_hopper.cuh"
#if defined(USE_NVSHMEM) && !defined(MIRAGE_GRACE_BLACKWELL)
#include "tasks/hopper/allreduce.cuh"
#endif
// Blackwell task impls
#if defined(USE_NVSHMEM) && defined(MIRAGE_GRACE_BLACKWELL)
#include "allreduce.cuh"
#endif
#include "argmax_sm100.cuh"
#include "attention_sm100.cuh"
#include "fp8_group_gemm_sm100.cuh"
#include "linear_fp8_1d2d_sm100.cuh"
#include "linear_fp8_sm100.cuh"
#include "linear_sm100_mpk.cuh"
#include "mla_dispatch_sm100.cuh"
#include "mla_kv_cache_gather_sm100.cuh"
#include "mla_kv_cache_gather_split_sm100.cuh"
// sm100_ptx.cuh must be included BEFORE mla_mtp_decode_sm100.cuh at top level
// so kernel::sm100_ptx is defined in the correct namespace
#include "elementwise_add_sm100.cuh"
#include "mla_mtp_decode_sm100.cuh"
#include "mla_mtp_decode_tp2_sm100.cuh"
#include "mla_mtp_decode_tp4_sm100.cuh"
#include "mla_mtp_decode_tp8_sm100.cuh"
#include "mla_prefill_sm100.cuh"
#include "mla_reduce_sm100.cuh"
#include "mla_sm100_2sm.cuh"
#include "moe_linear_sm100.cuh"
#include "mul_sum_add_sm100.cuh"
#include "per_token_group_quantize_fp8.cuh"
#include "prob_scatter_sm100.cuh"
#include "sm100_ptx.cuh"
#include "softmax_gather_sm100.cuh"
#include "tasks/common/sampling.cuh"
#include "tasks/speculative_decoding/mtp_token_ops.cuh"
#include "tasks/speculative_decoding/target_verify_mtp.cuh"
#include "tensor_init.cuh"
#include "topk_sigmoid_sm100.cuh"
#include "topk_softmax_sm100.cuh"
