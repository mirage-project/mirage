#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>

namespace kernel {
namespace mla_unified_sm100 {

static constexpr int kPrefillMinQLen = 9;
static constexpr int kDecodeMaxQLen = 8;

template <bool SINGLE_TILE, int TP_SIZE>
__device__ __noinline__ void
    mla_unified_sm100_task_impl(nv_bfloat16 const *__restrict__ q_nope,
                                nv_bfloat16 const *__restrict__ q_pe,
                                nv_bfloat16 const *__restrict__ ckv,
                                nv_bfloat16 const *__restrict__ kpe,
                                nv_bfloat16 *__restrict__ out,
                                CUtensorMap const *q_tm_ptr,
                                CUtensorMap const *kv_tm_ptr,
                                nv_bfloat16 *__restrict__ partial_o,
                                float *__restrict__ partial_lse,
                                int prefill_s,
                                int decode_kv_len,
                                int prefill_q_len,
                                int decode_q_len,
                                int decode_q_len_padded,
                                int num_heads,
                                float sm_scale_log2,
                                float sm_scale,
                                int num_splits,
                                int num_decode_groups,
                                int qpg,
                                int const *__restrict__ page_indices,
                                int first_page_pos,
                                int meta_x,
                                int meta_y,
                                int meta_z) {
  if (prefill_q_len >= kPrefillMinQLen) {
    if (meta_x < num_heads) {
      kernel::mla_prefill_sm100_task_impl(q_nope,
                                          q_pe,
                                          ckv,
                                          kpe,
                                          out,
                                          prefill_s,
                                          prefill_q_len,
                                          num_heads,
                                          sm_scale_log2,
                                          meta_x,
                                          meta_y);
    }
    return;
  }

  if (meta_z != 0 || decode_q_len < 1 || decode_q_len > kDecodeMaxQLen) {
    return;
  }

  if constexpr (TP_SIZE == 1) {
    int const gi = meta_x / num_splits;
    int const si = meta_x % num_splits;
    kernel::mla_mtp_decode_sm100_task_impl<SINGLE_TILE>(q_tm_ptr,
                                                        kv_tm_ptr,
                                                        partial_o,
                                                        partial_lse,
                                                        sm_scale,
                                                        decode_kv_len,
                                                        num_splits,
                                                        num_decode_groups,
                                                        decode_q_len,
                                                        gi,
                                                        si,
                                                        meta_y,
                                                        num_heads);
  } else if constexpr (TP_SIZE == 2) {
    kernel::mla_mtp_tp2::mla_mtp_tp2_main<SINGLE_TILE>(q_tm_ptr,
                                                       kv_tm_ptr,
                                                       partial_o,
                                                       partial_lse,
                                                       sm_scale,
                                                       decode_kv_len,
                                                       num_splits,
                                                       decode_q_len,
                                                       qpg,
                                                       page_indices,
                                                       first_page_pos,
                                                       meta_x,
                                                       meta_y);
  } else if constexpr (TP_SIZE == 4) {
    kernel::mla_mtp_tp4::mla_mtp_tp4_main<SINGLE_TILE>(q_tm_ptr,
                                                       kv_tm_ptr,
                                                       partial_o,
                                                       partial_lse,
                                                       sm_scale,
                                                       decode_kv_len,
                                                       num_splits,
                                                       decode_q_len,
                                                       qpg,
                                                       page_indices,
                                                       first_page_pos,
                                                       meta_x,
                                                       meta_y);
  } else if constexpr (TP_SIZE == 8) {
    kernel::mla_mtp_tp8::mla_mtp_tp8_main<SINGLE_TILE>(q_tm_ptr,
                                                       kv_tm_ptr,
                                                       partial_o,
                                                       partial_lse,
                                                       sm_scale,
                                                       decode_kv_len,
                                                       num_splits,
                                                       decode_q_len_padded,
                                                       qpg,
                                                       page_indices,
                                                       first_page_pos,
                                                       decode_q_len,
                                                       meta_x,
                                                       meta_y);
  }
}

} // namespace mla_unified_sm100
} // namespace kernel
