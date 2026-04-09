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
#ifdef USE_NVSHMEM
#include "tasks/hopper/allreduce.cuh"
#endif // USE_NVSHMEM
// Blackwell task impls
#include "argmax_sm100.cuh"
#include "attention_sm100.cuh"
#include "fp8_group_gemm_sm100.cuh"
#include "linear_sm100_mpk.cuh"
#include "mla_dispatch_sm100.cuh"
#include "mla_prefill_sm100.cuh"
#include "mla_reduce_sm100.cuh"
#include "moe_linear_sm100.cuh"
#include "mul_sum_add_sm100.cuh"
#include "tasks/common/sampling.cuh"
#include "tensor_init.cuh"
#include "topk_sigmoid_sm100.cuh"
#include "topk_softmax_sm100.cuh"
