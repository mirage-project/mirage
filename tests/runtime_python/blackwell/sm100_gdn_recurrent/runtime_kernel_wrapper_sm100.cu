/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Kernel-wrapper test harness for the Gated-DeltaNet recurrence task
// (include/mirage/persistent_kernel/tasks/blackwell/gdn_recurrent_sm100.cuh).
//
// One CUDA block per (V-HEAD, REQUEST SLOT), mirroring the persistent runtime
// where the task graph emits grid (NUM_V_HEADS, slots, 1) and each task is
// executed by one worker CTA.  The wrapper does exactly the pointer arithmetic
// that `TaskRegister::register_gdn_recurrent_sm100_task` emits into the
// generated `_execute_task()`:
//
//   hv      = kv_idx                       (blockIdx.x here)
//   slot    = request_id                   (blockIdx.y here)
//   tok0    = qo_indptr[slot],  q_len = qo_indptr[slot+1] - tok0
//   qkv    += tok0 * QKV_STRIDE            (head offsets are done in-kernel:
//                                           q, k and v live at three different
//                                           bases inside the packed row)
//   ba     += tok0 * BA_STRIDE
//   state  += (slot * NUM_V_HEADS + hv) * HEAD_V_DIM * HEAD_K_DIM
//   z      += tok0 * Z_STRIDE   + hv * HEAD_V_DIM
//   out    += tok0 * OUT_STRIDE + hv * HEAD_V_DIM
//
// `zero_state` is supplied per slot instead of being derived from
// `runtime_config.step[request_ids[slot]]`, so one launch can cover "fresh
// request" and "carried state" slots at once - that predicate's only observable
// effect.
//
// A second entry point, `gdn_gating_probe`, evaluates ONLY the gating scalars
// (`beta` and `g`) with the exact expressions the task uses.  They are the two
// intermediates the M2-I3 oracle dumps directly but the fused task never
// materialises, so this is how they get checked bit-for-bit against HF.

#include "blackwell/gdn_recurrent_sm100.cuh"
#include "blackwell/per_token_group_quantize_fp8.cuh"
#include <cuda_fp8.h>
#include "runtime_header.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

namespace {

template <int NUM_V_HEADS,
          int NUM_K_HEADS,
          int HEAD_K_DIM,
          int HEAD_V_DIM,
          int QKV_STRIDE,
          int BA_STRIDE,
          int Z_STRIDE,
          int OUT_STRIDE>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    gdn_recurrent_wrapper(void const *qkv,
                          void const *ba,
                          void const *alog_dtbias,
                          void *state,
                          void const *z,
                          void const *norm_w,
                          void *out,
                          int const *qo_indptr,
                          uint8_t const *zero_state,
                          void *o_debug) {
  int const hv = blockIdx.x;
  int const slot = blockIdx.y;
  int const tok0 = qo_indptr[slot];
  int const q_len = qo_indptr[slot + 1] - tok0;
  constexpr int STATE_ELEMS = HEAD_V_DIM * HEAD_K_DIM;
  kernel::gdn_recurrent_sm100_task_impl<bfloat16,
                                        NUM_V_HEADS,
                                        NUM_K_HEADS,
                                        HEAD_K_DIM,
                                        HEAD_V_DIM,
                                        QKV_STRIDE,
                                        BA_STRIDE,
                                        Z_STRIDE,
                                        OUT_STRIDE>(
      static_cast<bfloat16 const *>(qkv) + (size_t)tok0 * QKV_STRIDE,
      static_cast<bfloat16 const *>(ba) + (size_t)tok0 * BA_STRIDE,
      alog_dtbias,
      static_cast<float *>(state) +
          ((size_t)slot * NUM_V_HEADS + hv) * STATE_ELEMS,
      static_cast<bfloat16 const *>(z) + (size_t)tok0 * Z_STRIDE +
          (size_t)hv * HEAD_V_DIM,
      norm_w,
      static_cast<bfloat16 *>(out) + (size_t)tok0 * OUT_STRIDE +
          (size_t)hv * HEAD_V_DIM,
      hv,
      q_len,
      zero_state[slot] != 0,
      o_debug == nullptr
          ? nullptr
          : static_cast<void *>(static_cast<bfloat16 *>(o_debug) +
                                (size_t)tok0 * OUT_STRIDE +
                                (size_t)hv * HEAD_V_DIM));
}

template <int NUM_V_HEADS,
          int NUM_K_HEADS,
          int HEAD_K_DIM,
          int HEAD_V_DIM,
          int QKV_STRIDE,
          int BA_STRIDE,
          int Z_STRIDE,
          int OUT_STRIDE>
void launch_gdn_recurrent(int num_slots,
                          void const *qkv,
                          void const *ba,
                          void const *alog_dtbias,
                          void *state,
                          void const *z,
                          void const *norm_w,
                          void *out,
                          int const *qo_indptr,
                          uint8_t const *zero_state,
                          void *o_debug) {
  // Matches the smem carve-up in gdn_recurrent_sm100_task_impl:
  // S[HEAD_V_DIM][HEAD_K_DIM] + k + q + o + a 32-slot reduction scratch.
  size_t const smem_size = sizeof(float) * ((size_t)HEAD_V_DIM * HEAD_K_DIM +
                                            2 * HEAD_K_DIM + HEAD_V_DIM + 32);
  auto kern = gdn_recurrent_wrapper<NUM_V_HEADS,
                                    NUM_K_HEADS,
                                    HEAD_K_DIM,
                                    HEAD_V_DIM,
                                    QKV_STRIDE,
                                    BA_STRIDE,
                                    Z_STRIDE,
                                    OUT_STRIDE>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
  dim3 grid(NUM_V_HEADS, num_slots, 1);
  kern<<<grid,
         WORKER_NUM_THREADS,
         smem_size,
         at::cuda::getCurrentCUDAStream()>>>(qkv,
                                             ba,
                                             alog_dtbias,
                                             state,
                                             z,
                                             norm_w,
                                             out,
                                             qo_indptr,
                                             zero_state,
                                             o_debug);
}

// Shapes this harness instantiates:
// (num_v_heads, num_k_heads, head_k_dim, head_v_dim, qkv_stride, ba_stride,
//  z_stride, out_stride).  Qwen3.5's GDN core is the first entry; the second
// covers a strided qkv row (the layout a fused in_proj_qkvz would hand us,
// vllm-graph.md 2.1.2); the rest keep the unit tests fast and cover the GVA
// ratio (1, 2, 4) and a head_k_dim that is exactly one warp wide.
#define GDN_RECURRENT_CASES(F)                                                 \
  F(32, 16, 128, 128, 8192, 64, 4096, 4096)                                    \
  F(32, 16, 128, 128, 12288, 64, 4096, 4096)                                   \
  F(8, 4, 128, 128, 2048, 16, 1024, 1024)                                      \
  F(4, 2, 32, 32, 256, 8, 128, 128)                                            \
  F(4, 4, 32, 32, 384, 8, 128, 128)                                            \
  F(4, 1, 32, 32, 192, 8, 128, 128)                                            \
  F(2, 1, 64, 64, 256, 4, 128, 128)

bool dispatch_gdn_recurrent(int num_v_heads,
                            int num_k_heads,
                            int head_k_dim,
                            int head_v_dim,
                            int qkv_stride,
                            int ba_stride,
                            int z_stride,
                            int out_stride,
                            int num_slots,
                            void const *qkv,
                            void const *ba,
                            void const *alog_dtbias,
                            void *state,
                            void const *z,
                            void const *norm_w,
                            void *out,
                            int const *qo_indptr,
                            uint8_t const *zero_state,
                            void *o_debug) {
#define GDN_RECURRENT_DISPATCH(HV, HK, DK, DV, QS, BS, ZS, OS)                 \
  if (num_v_heads == (HV) && num_k_heads == (HK) && head_k_dim == (DK) &&      \
      head_v_dim == (DV) && qkv_stride == (QS) && ba_stride == (BS) &&         \
      z_stride == (ZS) && out_stride == (OS)) {                                \
    launch_gdn_recurrent<HV, HK, DK, DV, QS, BS, ZS, OS>(num_slots,            \
                                                         qkv,                  \
                                                         ba,                   \
                                                         alog_dtbias,          \
                                                         state,                \
                                                         z,                    \
                                                         norm_w,               \
                                                         out,                  \
                                                         qo_indptr,            \
                                                         zero_state,           \
                                                         o_debug);             \
    return true;                                                               \
  }
  GDN_RECURRENT_CASES(GDN_RECURRENT_DISPATCH)
#undef GDN_RECURRENT_DISPATCH
  return false;
}

// ---------------------------------------------------------------------------
// DECODE SPLIT PATH
// ---------------------------------------------------------------------------
// Same convention, plus grid.z == SPLIT and `split_scratch` pre-offset per
// (slot, hv) - i.e. exactly what `register_gdn_recurrent_sm100_task` emits for
// the q_len == 1 && !zero_state branch.  This is what makes the port's
// bit-exactness claim testable OUTSIDE the megakernel: `test_gdn_recurrent.py`
// runs the golden wrapper and this one on identical inputs and does an integer
// comparison of `out` AND the updated `state`, which is the same gate the
// ferret loop that produced this kernel ran on every iteration.
template <int NUM_V_HEADS,
          int NUM_K_HEADS,
          int HEAD_K_DIM,
          int HEAD_V_DIM,
          int QKV_STRIDE,
          int BA_STRIDE,
          int Z_STRIDE,
          int OUT_STRIDE,
          int SPLIT,
          int DEPTH>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    gdn_decode_split_wrapper(void const *qkv,
                             void const *ba,
                             void const *alog_dtbias,
                             void *state,
                             void const *z,
                             void const *norm_w,
                             void *out,
                             void *split_scratch,
                             int const *qo_indptr) {
  int const hv = blockIdx.x;
  int const slot = blockIdx.y;
  int const tok0 = qo_indptr[slot];
  constexpr int STATE_ELEMS = HEAD_V_DIM * HEAD_K_DIM;
  constexpr int SCRATCH_STRIDE = HEAD_V_DIM + 8;
  kernel::gdn_recurrent_sm100_decode_split_impl<bfloat16,
                                                NUM_V_HEADS,
                                                NUM_K_HEADS,
                                                HEAD_K_DIM,
                                                HEAD_V_DIM,
                                                QKV_STRIDE,
                                                BA_STRIDE,
                                                Z_STRIDE,
                                                OUT_STRIDE,
                                                SPLIT,
                                                DEPTH,
                                                WORKER_NUM_THREADS>(
      static_cast<bfloat16 const *>(qkv) + (size_t)tok0 * QKV_STRIDE,
      static_cast<bfloat16 const *>(ba) + (size_t)tok0 * BA_STRIDE,
      alog_dtbias,
      static_cast<float *>(state) +
          ((size_t)slot * NUM_V_HEADS + hv) * STATE_ELEMS,
      static_cast<bfloat16 const *>(z) + (size_t)tok0 * Z_STRIDE +
          (size_t)hv * HEAD_V_DIM,
      norm_w,
      static_cast<bfloat16 *>(out) + (size_t)tok0 * OUT_STRIDE +
          (size_t)hv * HEAD_V_DIM,
      static_cast<float *>(split_scratch) +
          ((size_t)slot * NUM_V_HEADS + hv) * SCRATCH_STRIDE,
      hv,
      (int)blockIdx.z);
}

// ================================================================
// M4-I9 flag C -- the same decode-split task with the fp32-block-scale FP8
// quantize of `out` FUSED into its gated-RMSNorm epilogue.
//
// WRITE_OUT is a template parameter here so the test can do both jobs:
//   WRITE_OUT=true  -> the bf16 `out` is still stored, so it can be compared
//                      byte-for-byte against the unfused kernel (does the
//                      fusion perturb the recurrence at all?) AND used as the
//                      input to the standalone quantize for the fp8 reference.
//   WRITE_OUT=false -> the form the graph actually ships. Its fp8 must equal
//                      the WRITE_OUT=true form's, i.e. dropping the store
//                      cannot move a byte.
// ================================================================
template <int NUM_V_HEADS,
          int NUM_K_HEADS,
          int HEAD_K_DIM,
          int HEAD_V_DIM,
          int QKV_STRIDE,
          int BA_STRIDE,
          int Z_STRIDE,
          int OUT_STRIDE,
          int SPLIT,
          int DEPTH,
          bool WRITE_OUT>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    gdn_decode_split_fusedq_wrapper(void const *qkv,
                                    void const *ba,
                                    void const *alog_dtbias,
                                    void *state,
                                    void const *z,
                                    void const *norm_w,
                                    void *out,
                                    void *out_q,
                                    void *out_s,
                                    void *split_scratch,
                                    int const *qo_indptr) {
  int const hv = blockIdx.x;
  int const slot = blockIdx.y;
  int const tok0 = qo_indptr[slot];
  constexpr int STATE_ELEMS = HEAD_V_DIM * HEAD_K_DIM;
  constexpr int SCRATCH_STRIDE = HEAD_V_DIM + 8;
  kernel::gdn_recurrent_sm100_decode_split_impl<bfloat16,
                                                NUM_V_HEADS,
                                                NUM_K_HEADS,
                                                HEAD_K_DIM,
                                                HEAD_V_DIM,
                                                QKV_STRIDE,
                                                BA_STRIDE,
                                                Z_STRIDE,
                                                OUT_STRIDE,
                                                SPLIT,
                                                DEPTH,
                                                WORKER_NUM_THREADS,
                                                /*FUSE_QUANT=*/true,
                                                WRITE_OUT>(
      static_cast<bfloat16 const *>(qkv) + (size_t)tok0 * QKV_STRIDE,
      static_cast<bfloat16 const *>(ba) + (size_t)tok0 * BA_STRIDE,
      alog_dtbias,
      static_cast<float *>(state) +
          ((size_t)slot * NUM_V_HEADS + hv) * STATE_ELEMS,
      static_cast<bfloat16 const *>(z) + (size_t)tok0 * Z_STRIDE +
          (size_t)hv * HEAD_V_DIM,
      norm_w,
      WRITE_OUT ? (static_cast<bfloat16 *>(out) + (size_t)tok0 * OUT_STRIDE +
                   (size_t)hv * HEAD_V_DIM)
                : nullptr,
      static_cast<float *>(split_scratch) +
          ((size_t)slot * NUM_V_HEADS + hv) * SCRATCH_STRIDE,
      hv,
      (int)blockIdx.z,
      static_cast<__nv_fp8_e4m3 *>(out_q) + (size_t)tok0 * OUT_STRIDE +
          (size_t)hv * HEAD_V_DIM,
      static_cast<float *>(out_s) + (size_t)tok0 * (OUT_STRIDE / 128) +
          (size_t)hv);
}

// The standalone quantize, for the fp8 reference. Same header, same TU, same
// nvcc flags as the fused arm -- so a torch reference's own rounding never
// enters the comparison.
template <int OUT_STRIDE>
__global__ void __launch_bounds__(128)
    gdn_ref_quantize_kernel(void const *__restrict__ input,
                            void *__restrict__ output_q,
                            void *__restrict__ output_s) {
  kernel::per_token_group_quantize_fp8_task_impl</*BATCH_SIZE=*/1,
                                                /*HIDDEN_SIZE=*/OUT_STRIDE,
                                                /*GROUP_SIZE=*/128,
                                                /*GLOBAL_STRIDE=*/OUT_STRIDE,
                                                bfloat16,
                                                __nv_fp8_e4m3,
                                                /*SCALE_UE8M0=*/false>(
      input, output_q, output_s, 1e-10f, -448.0f, 448.0f, 1);
}

template <int NUM_V_HEADS,
          int NUM_K_HEADS,
          int HEAD_K_DIM,
          int HEAD_V_DIM,
          int QKV_STRIDE,
          int BA_STRIDE,
          int Z_STRIDE,
          int OUT_STRIDE,
          int SPLIT,
          int DEPTH>
void launch_gdn_decode_split(int num_slots,
                             void const *qkv,
                             void const *ba,
                             void const *alog_dtbias,
                             void *state,
                             void const *z,
                             void const *norm_w,
                             void *out,
                             void *split_scratch,
                             int const *qo_indptr) {
  size_t const smem_size =
      kernel::gdn_decode_split_smem_bytes<bfloat16,
                                          HEAD_K_DIM,
                                          HEAD_V_DIM,
                                          DEPTH,
                                          WORKER_NUM_THREADS>();
  auto kern = gdn_decode_split_wrapper<NUM_V_HEADS,
                                       NUM_K_HEADS,
                                       HEAD_K_DIM,
                                       HEAD_V_DIM,
                                       QKV_STRIDE,
                                       BA_STRIDE,
                                       Z_STRIDE,
                                       OUT_STRIDE,
                                       SPLIT,
                                       DEPTH>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
  dim3 grid(NUM_V_HEADS, num_slots, SPLIT);
  kern<<<grid,
         WORKER_NUM_THREADS,
         smem_size,
         at::cuda::getCurrentCUDAStream()>>>(qkv,
                                             ba,
                                             alog_dtbias,
                                             state,
                                             z,
                                             norm_w,
                                             out,
                                             split_scratch,
                                             qo_indptr);
}

template <int NUM_V_HEADS,
          int NUM_K_HEADS,
          int HEAD_K_DIM,
          int HEAD_V_DIM,
          int QKV_STRIDE,
          int BA_STRIDE,
          int Z_STRIDE,
          int OUT_STRIDE,
          int SPLIT,
          int DEPTH>
void launch_gdn_decode_split_fusedq(int num_slots,
                                    bool write_out,
                                    void const *qkv,
                                    void const *ba,
                                    void const *alog_dtbias,
                                    void *state,
                                    void const *z,
                                    void const *norm_w,
                                    void *out,
                                    void *out_q,
                                    void *out_s,
                                    void *split_scratch,
                                    int const *qo_indptr,
                                    int num_tokens) {
  size_t const smem_size =
      kernel::gdn_decode_split_smem_bytes<bfloat16,
                                          HEAD_K_DIM,
                                          HEAD_V_DIM,
                                          DEPTH,
                                          WORKER_NUM_THREADS>();
  dim3 grid(NUM_V_HEADS, num_slots, SPLIT);
  if (write_out) {
    auto kern = gdn_decode_split_fusedq_wrapper<NUM_V_HEADS, NUM_K_HEADS,
                                                HEAD_K_DIM, HEAD_V_DIM,
                                                QKV_STRIDE, BA_STRIDE,
                                                Z_STRIDE, OUT_STRIDE, SPLIT,
                                                DEPTH, true>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
    kern<<<grid, WORKER_NUM_THREADS, smem_size,
           at::cuda::getCurrentCUDAStream()>>>(qkv, ba, alog_dtbias, state, z,
                                               norm_w, out, out_q, out_s,
                                               split_scratch, qo_indptr);
  } else {
    auto kern = gdn_decode_split_fusedq_wrapper<NUM_V_HEADS, NUM_K_HEADS,
                                                HEAD_K_DIM, HEAD_V_DIM,
                                                QKV_STRIDE, BA_STRIDE,
                                                Z_STRIDE, OUT_STRIDE, SPLIT,
                                                DEPTH, false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
    kern<<<grid, WORKER_NUM_THREADS, smem_size,
           at::cuda::getCurrentCUDAStream()>>>(qkv, ba, alog_dtbias, state, z,
                                               norm_w, out, out_q, out_s,
                                               split_scratch, qo_indptr);
  }
}

// The fp8 REFERENCE: the standalone quantize over the bf16 `out` the unfused
// task wrote, one CTA per token row (the impl walks the row's groups itself).
template <int OUT_STRIDE>
void launch_gdn_ref_quantize(int num_tokens,
                             void const *out,
                             void *out_q,
                             void *out_s) {
  for (int t = 0; t < num_tokens; t++) {
    gdn_ref_quantize_kernel<OUT_STRIDE>
        <<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<bfloat16 const *>(out) + (size_t)t * OUT_STRIDE,
            static_cast<__nv_fp8_e4m3 *>(out_q) + (size_t)t * OUT_STRIDE,
            static_cast<float *>(out_s) + (size_t)t * (OUT_STRIDE / 128));
  }
}

// (shape) x (split, depth).  Every shape gets splits 1/2/4 at depth 2; the
// Qwen3.5 production shape additionally gets the wider splits and the deeper
// rings, since that is the one the megakernel actually compiles.
#define GDN_SPLIT_TRIPLE(F, HV, HK, DK, DV, QS, BS, ZS, OS)                    \
  F(HV, HK, DK, DV, QS, BS, ZS, OS, 1, 2)                                      \
  F(HV, HK, DK, DV, QS, BS, ZS, OS, 2, 2)                                      \
  F(HV, HK, DK, DV, QS, BS, ZS, OS, 4, 2)

#define GDN_DECODE_SPLIT_CASES(F)                                              \
  GDN_SPLIT_TRIPLE(F, 32, 16, 128, 128, 8192, 64, 4096, 4096)                  \
  F(32, 16, 128, 128, 8192, 64, 4096, 4096, 8, 2)                              \
  F(32, 16, 128, 128, 8192, 64, 4096, 4096, 16, 2)                             \
  F(32, 16, 128, 128, 8192, 64, 4096, 4096, 32, 2)                             \
  F(32, 16, 128, 128, 8192, 64, 4096, 4096, 1, 3)                              \
  F(32, 16, 128, 128, 8192, 64, 4096, 4096, 1, 4)                              \
  F(32, 16, 128, 128, 8192, 64, 4096, 4096, 2, 4)                              \
  F(32, 16, 128, 128, 8192, 64, 4096, 4096, 4, 4)                              \
  GDN_SPLIT_TRIPLE(F, 32, 16, 128, 128, 12288, 64, 4096, 4096)                 \
  GDN_SPLIT_TRIPLE(F, 8, 4, 128, 128, 2048, 16, 1024, 1024)                    \
  GDN_SPLIT_TRIPLE(F, 4, 2, 32, 32, 256, 8, 128, 128)                          \
  GDN_SPLIT_TRIPLE(F, 4, 4, 32, 32, 384, 8, 128, 128)                          \
  GDN_SPLIT_TRIPLE(F, 4, 1, 32, 32, 192, 8, 128, 128)                          \
  GDN_SPLIT_TRIPLE(F, 2, 1, 64, 64, 256, 4, 128, 128)

bool dispatch_gdn_decode_split(int num_v_heads,
                               int num_k_heads,
                               int head_k_dim,
                               int head_v_dim,
                               int qkv_stride,
                               int ba_stride,
                               int z_stride,
                               int out_stride,
                               int split,
                               int depth,
                               int num_slots,
                               void const *qkv,
                               void const *ba,
                               void const *alog_dtbias,
                               void *state,
                               void const *z,
                               void const *norm_w,
                               void *out,
                               void *split_scratch,
                               int const *qo_indptr) {
#define GDN_DECODE_SPLIT_DISPATCH(HV, HK, DK, DV, QS, BS, ZS, OS, SP, DP)      \
  if (num_v_heads == (HV) && num_k_heads == (HK) && head_k_dim == (DK) &&      \
      head_v_dim == (DV) && qkv_stride == (QS) && ba_stride == (BS) &&         \
      z_stride == (ZS) && out_stride == (OS) && split == (SP) &&               \
      depth == (DP)) {                                                         \
    launch_gdn_decode_split<HV, HK, DK, DV, QS, BS, ZS, OS, SP, DP>(           \
        num_slots,                                                             \
        qkv,                                                                   \
        ba,                                                                    \
        alog_dtbias,                                                           \
        state,                                                                 \
        z,                                                                     \
        norm_w,                                                                \
        out,                                                                   \
        split_scratch,                                                         \
        qo_indptr);                                                            \
    return true;                                                               \
  }
  GDN_DECODE_SPLIT_CASES(GDN_DECODE_SPLIT_DISPATCH)
#undef GDN_DECODE_SPLIT_DISPATCH
  return false;
}

// M4-I9 flag C: the same dispatch for the FUSED arm, plus the fp8 reference.
bool dispatch_gdn_decode_split_fusedq(int num_v_heads,
                                      int num_k_heads,
                                      int head_k_dim,
                                      int head_v_dim,
                                      int qkv_stride,
                                      int ba_stride,
                                      int z_stride,
                                      int out_stride,
                                      int split,
                                      int depth,
                                      int num_slots,
                                      int num_tokens,
                                      bool write_out,
                                      bool ref_only,
                                      void const *qkv,
                                      void const *ba,
                                      void const *alog_dtbias,
                                      void *state,
                                      void const *z,
                                      void const *norm_w,
                                      void *out,
                                      void *out_q,
                                      void *out_s,
                                      void *split_scratch,
                                      int const *qo_indptr) {
#define GDN_FUSEDQ_DISPATCH(HV, HK, DK, DV, QS, BS, ZS, OS, SP, DP)            \
  if (num_v_heads == (HV) && num_k_heads == (HK) && head_k_dim == (DK) &&      \
      head_v_dim == (DV) && qkv_stride == (QS) && ba_stride == (BS) &&         \
      z_stride == (ZS) && out_stride == (OS) && split == (SP) &&               \
      depth == (DP)) {                                                        \
    constexpr int DV_ = (DV);                                                  \
    /* flag C is only defined where a v-head IS one 128-element scale group;    \
       the impl static_asserts that, so the other test shapes must not          \
       instantiate it at all. `if constexpr` keeps them out of the TU. */       \
    if constexpr (DV_ == 128) {                                               \
      if (ref_only) {                                                          \
        launch_gdn_ref_quantize<OS>(num_tokens, out, out_q, out_s);             \
      } else {                                                                 \
        launch_gdn_decode_split_fusedq<HV, HK, DK, DV, QS, BS, ZS, OS, SP,      \
                                       DP>(                                    \
            num_slots, write_out, qkv, ba, alog_dtbias, state, z, norm_w, out,  \
            out_q, out_s, split_scratch, qo_indptr, num_tokens);                \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
    return false;                                                              \
  }
  GDN_DECODE_SPLIT_CASES(GDN_FUSEDQ_DISPATCH)
#undef GDN_FUSEDQ_DISPATCH
  return false;
}

// Gating-scalar probe: the two dumped intermediates the fused task consumes but
// never writes out.  Expressions are copied verbatim from the task impl so a
// mismatch here localises to the CUDA math (expf / log1pf / sigmoid) rather
// than to the recurrence.
__global__ void gdn_gating_probe_kernel(bfloat16 const *__restrict__ ba,
                                        float const *__restrict__ ad,
                                        bfloat16 *__restrict__ beta_out,
                                        float *__restrict__ g_out,
                                        int num_tokens,
                                        int num_v_heads,
                                        int ba_stride) {
  int const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tokens * num_v_heads) {
    return;
  }
  int const t = idx / num_v_heads;
  int const hv = idx - t * num_v_heads;
  float const b_val = static_cast<float>(ba[(size_t)t * ba_stride + hv]);
  float const a_val =
      static_cast<float>(ba[(size_t)t * ba_stride + num_v_heads + hv]);
  float const x = a_val + ad[num_v_heads + hv];
  float const softplus = (x > 20.0f) ? x : log1pf(expf(x));
  beta_out[idx] = bfloat16(1.0f / (1.0f + expf(-b_val)));
  g_out[idx] = -expf(ad[hv]) * softplus;
}

} // namespace

void gdn_recurrent_sm100(torch::Tensor qkv,         // [tokens, qkv_stride] bf16
                         torch::Tensor ba,          // [tokens, ba_stride] bf16
                         torch::Tensor alog_dtbias, // [2, num_v_heads] f32
                         torch::Tensor state,     // [slots,Hv,Dv,Dk] f32 in/out
                         torch::Tensor z,         // [tokens, z_stride] bf16
                         torch::Tensor norm_w,    // [head_v_dim] f32
                         torch::Tensor out,       // [tokens, out_stride] bf16
                         torch::Tensor qo_indptr, // int32 [slots + 1]
                         torch::Tensor zero_state, // uint8 [slots]
                         int64_t num_k_heads,
                         c10::optional<torch::Tensor> o_debug) {
  TORCH_CHECK(qkv.dim() == 2 && qkv.is_contiguous() &&
                  qkv.scalar_type() == at::kBFloat16,
              "qkv must be a contiguous 2D bfloat16 tensor");
  TORCH_CHECK(ba.dim() == 2 && ba.is_contiguous() &&
                  ba.scalar_type() == at::kBFloat16,
              "ba must be a contiguous 2D bfloat16 tensor");
  TORCH_CHECK(z.dim() == 2 && z.is_contiguous() &&
                  z.scalar_type() == at::kBFloat16,
              "z must be a contiguous 2D bfloat16 tensor");
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous() &&
                  out.scalar_type() == at::kBFloat16,
              "out must be a contiguous 2D bfloat16 tensor");
  TORCH_CHECK(alog_dtbias.dim() == 2 && alog_dtbias.is_contiguous() &&
                  alog_dtbias.scalar_type() == at::kFloat &&
                  alog_dtbias.size(0) == 2,
              "alog_dtbias must be a contiguous [2, num_v_heads] fp32 tensor");
  TORCH_CHECK(norm_w.dim() == 1 && norm_w.is_contiguous() &&
                  norm_w.scalar_type() == at::kFloat,
              "norm_w must be a contiguous 1D fp32 tensor");
  TORCH_CHECK(state.dim() == 4 && state.is_contiguous() &&
                  state.scalar_type() == at::kFloat,
              "state must be a contiguous [slots, num_v_heads, head_v_dim, "
              "head_k_dim] fp32 tensor");
  TORCH_CHECK(qo_indptr.dim() == 1 && qo_indptr.is_contiguous() &&
                  qo_indptr.scalar_type() == at::kInt,
              "qo_indptr must be a contiguous 1D int32 tensor");
  TORCH_CHECK(zero_state.dim() == 1 && zero_state.is_contiguous() &&
                  zero_state.scalar_type() == at::kByte,
              "zero_state must be a contiguous 1D uint8 tensor");

  int const num_slots = (int)state.size(0);
  int const num_v_heads = (int)state.size(1);
  int const head_v_dim = (int)state.size(2);
  int const head_k_dim = (int)state.size(3);
  int const qkv_stride = (int)qkv.size(1);
  int const ba_stride = (int)ba.size(1);
  int const z_stride = (int)z.size(1);
  int const out_stride = (int)out.size(1);

  TORCH_CHECK(alog_dtbias.size(1) == num_v_heads,
              "alog_dtbias must have num_v_heads columns");
  TORCH_CHECK(norm_w.size(0) == head_v_dim, "norm_w must be head_v_dim long");
  TORCH_CHECK(qo_indptr.size(0) == num_slots + 1,
              "qo_indptr must have num_slots + 1 entries");
  TORCH_CHECK(zero_state.size(0) == num_slots,
              "zero_state must have num_slots entries");
  TORCH_CHECK(num_k_heads >= 1 && num_v_heads % (int)num_k_heads == 0,
              "num_v_heads must be a multiple of num_k_heads");
  TORCH_CHECK(ba_stride >= 2 * num_v_heads, "ba packs [b | a]");
  TORCH_CHECK(qkv_stride >=
                  2 * (int)num_k_heads * head_k_dim + num_v_heads * head_v_dim,
              "qkv row is too narrow for [q | k | v]");
  TORCH_CHECK(qkv.size(0) == out.size(0) && qkv.size(0) == z.size(0) &&
                  qkv.size(0) == ba.size(0),
              "qkv, ba, z and out must have the same number of token rows");

  void *o_debug_ptr = nullptr;
  if (o_debug.has_value()) {
    torch::Tensor const &od = o_debug.value();
    TORCH_CHECK(od.dim() == 2 && od.is_contiguous() &&
                    od.scalar_type() == at::kBFloat16 &&
                    od.size(0) == out.size(0) && od.size(1) == out_stride,
                "o_debug must match `out` in shape, dtype and layout");
    o_debug_ptr = od.data_ptr();
  }

  bool const dispatched = dispatch_gdn_recurrent(num_v_heads,
                                                 (int)num_k_heads,
                                                 head_k_dim,
                                                 head_v_dim,
                                                 qkv_stride,
                                                 ba_stride,
                                                 z_stride,
                                                 out_stride,
                                                 num_slots,
                                                 qkv.data_ptr(),
                                                 ba.data_ptr(),
                                                 alog_dtbias.data_ptr(),
                                                 state.data_ptr(),
                                                 z.data_ptr(),
                                                 norm_w.data_ptr(),
                                                 out.data_ptr(),
                                                 qo_indptr.data_ptr<int>(),
                                                 zero_state.data_ptr<uint8_t>(),
                                                 o_debug_ptr);
  TORCH_CHECK(dispatched,
              "Unsupported gdn_recurrent_sm100 shape [num_v_heads=",
              num_v_heads,
              ", num_k_heads=",
              num_k_heads,
              ", head_k_dim=",
              head_k_dim,
              ", head_v_dim=",
              head_v_dim,
              ", qkv_stride=",
              qkv_stride,
              ", ba_stride=",
              ba_stride,
              ", z_stride=",
              z_stride,
              ", out_stride=",
              out_stride,
              "]");
  C10_CUDA_CHECK(cudaGetLastError());
}

// Decode fast path (q_len == 1, carried state) with the v-row split.  Same
// tensors as `gdn_recurrent_sm100` plus the [slots, Hv, Dv + 8] fp32 scratch;
// `zero_state` is not a parameter because this entry point IS the
// !zero_state branch.
// M4-I9 flag C host entry. `mode`: 0 = fused with the bf16 store kept,
// 1 = fused with the store dropped (the form the graph ships), 2 = the fp8
// REFERENCE, i.e. the standalone quantize over an already-computed bf16 `out`.
void gdn_recurrent_decode_split_fusedq_sm100(torch::Tensor qkv,
                                            torch::Tensor ba,
                                            torch::Tensor alog_dtbias,
                                            torch::Tensor state,
                                            torch::Tensor z,
                                            torch::Tensor norm_w,
                                            torch::Tensor out,
                                            torch::Tensor out_q,
                                            torch::Tensor out_s,
                                            torch::Tensor split_scratch,
                                            torch::Tensor qo_indptr,
                                            int64_t num_k_heads,
                                            int64_t split,
                                            int64_t depth,
                                            int64_t mode) {
  TORCH_CHECK(out_q.scalar_type() == at::kFloat8_e4m3fn &&
                  out_s.scalar_type() == at::kFloat,
              "out_q must be float8_e4m3fn and out_s float32");
  TORCH_CHECK(out_q.is_contiguous() && out_s.is_contiguous(),
              "fp8 outputs must be contiguous");
  int const num_slots = (int)state.size(0);
  int const num_v_heads = (int)state.size(1);
  int const head_v_dim = (int)state.size(2);
  int const head_k_dim = (int)state.size(3);
  int const num_tokens = (int)out.size(0);
  TORCH_CHECK(out_s.size(1) == out.size(1) / 128,
              "out_s must be [tokens, out_stride/128]");
  bool const dispatched = dispatch_gdn_decode_split_fusedq(
      num_v_heads, (int)num_k_heads, head_k_dim, head_v_dim, (int)qkv.size(1),
      (int)ba.size(1), (int)z.size(1), (int)out.size(1), (int)split,
      (int)depth, num_slots, num_tokens, /*write_out=*/mode == 0,
      /*ref_only=*/mode == 2, qkv.data_ptr(), ba.data_ptr(),
      alog_dtbias.data_ptr(), state.data_ptr(), z.data_ptr(),
      norm_w.data_ptr(), out.data_ptr(), out_q.data_ptr(), out_s.data_ptr(),
      split_scratch.data_ptr(), qo_indptr.data_ptr<int>());
  TORCH_CHECK(dispatched, "Unsupported gdn fused-quantize configuration");
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}

void gdn_recurrent_decode_split_sm100(torch::Tensor qkv,
                                      torch::Tensor ba,
                                      torch::Tensor alog_dtbias,
                                      torch::Tensor state,
                                      torch::Tensor z,
                                      torch::Tensor norm_w,
                                      torch::Tensor out,
                                      torch::Tensor split_scratch,
                                      torch::Tensor qo_indptr,
                                      int64_t num_k_heads,
                                      int64_t split,
                                      int64_t depth) {
  TORCH_CHECK(qkv.dim() == 2 && qkv.is_contiguous() &&
                  qkv.scalar_type() == at::kBFloat16,
              "qkv must be a contiguous 2D bfloat16 tensor");
  TORCH_CHECK(state.dim() == 4 && state.is_contiguous() &&
                  state.scalar_type() == at::kFloat,
              "state must be a contiguous [slots, Hv, Dv, Dk] fp32 tensor");
  TORCH_CHECK(split_scratch.dim() == 3 && split_scratch.is_contiguous() &&
                  split_scratch.scalar_type() == at::kFloat,
              "split_scratch must be a contiguous [slots, Hv, Dv + 8] fp32 "
              "tensor");
  TORCH_CHECK(qo_indptr.dim() == 1 && qo_indptr.is_contiguous() &&
                  qo_indptr.scalar_type() == at::kInt,
              "qo_indptr must be a contiguous 1D int32 tensor");

  int const num_slots = (int)state.size(0);
  int const num_v_heads = (int)state.size(1);
  int const head_v_dim = (int)state.size(2);
  int const head_k_dim = (int)state.size(3);
  TORCH_CHECK(split_scratch.size(0) == num_slots &&
                  split_scratch.size(1) == num_v_heads &&
                  split_scratch.size(2) == head_v_dim + 8,
              "split_scratch must be [slots, num_v_heads, head_v_dim + 8]");
  TORCH_CHECK(split >= 1 && head_v_dim % (int)split == 0,
              "split must divide head_v_dim");

  bool const dispatched =
      dispatch_gdn_decode_split(num_v_heads,
                                (int)num_k_heads,
                                head_k_dim,
                                head_v_dim,
                                (int)qkv.size(1),
                                (int)ba.size(1),
                                (int)z.size(1),
                                (int)out.size(1),
                                (int)split,
                                (int)depth,
                                num_slots,
                                qkv.data_ptr(),
                                ba.data_ptr(),
                                alog_dtbias.data_ptr(),
                                state.data_ptr(),
                                z.data_ptr(),
                                norm_w.data_ptr(),
                                out.data_ptr(),
                                split_scratch.data_ptr(),
                                qo_indptr.data_ptr<int>());
  TORCH_CHECK(dispatched,
              "Unsupported gdn decode-split configuration [num_v_heads=",
              num_v_heads,
              ", num_k_heads=",
              num_k_heads,
              ", head_k_dim=",
              head_k_dim,
              ", head_v_dim=",
              head_v_dim,
              ", qkv_stride=",
              (int)qkv.size(1),
              ", split=",
              split,
              ", depth=",
              depth,
              "]");
  C10_CUDA_CHECK(cudaGetLastError());
}

void gdn_gating_probe(torch::Tensor ba,
                      torch::Tensor alog_dtbias,
                      torch::Tensor beta_out,
                      torch::Tensor g_out) {
  TORCH_CHECK(ba.dim() == 2 && ba.is_contiguous() &&
                  ba.scalar_type() == at::kBFloat16,
              "ba must be a contiguous 2D bfloat16 tensor");
  TORCH_CHECK(alog_dtbias.dim() == 2 && alog_dtbias.size(0) == 2 &&
                  alog_dtbias.is_contiguous() &&
                  alog_dtbias.scalar_type() == at::kFloat,
              "alog_dtbias must be a contiguous [2, num_v_heads] fp32 tensor");
  int const num_tokens = (int)ba.size(0);
  int const ba_stride = (int)ba.size(1);
  int const num_v_heads = (int)alog_dtbias.size(1);
  TORCH_CHECK(beta_out.numel() == (int64_t)num_tokens * num_v_heads &&
                  beta_out.scalar_type() == at::kBFloat16 &&
                  beta_out.is_contiguous(),
              "beta_out must be a contiguous bf16 [tokens, num_v_heads]");
  TORCH_CHECK(g_out.numel() == (int64_t)num_tokens * num_v_heads &&
                  g_out.scalar_type() == at::kFloat && g_out.is_contiguous(),
              "g_out must be a contiguous fp32 [tokens, num_v_heads]");
  int const n = num_tokens * num_v_heads;
  int const threads = 128;
  gdn_gating_probe_kernel<<<(n + threads - 1) / threads,
                            threads,
                            0,
                            at::cuda::getCurrentCUDAStream()>>>(
      static_cast<bfloat16 const *>(ba.data_ptr()),
      alog_dtbias.data_ptr<float>(),
      static_cast<bfloat16 *>(beta_out.data_ptr()),
      g_out.data_ptr<float>(),
      num_tokens,
      num_v_heads,
      ba_stride);
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_recurrent_sm100",
        &gdn_recurrent_sm100,
        "Gated-DeltaNet recurrence + fused gated RMSNorm/SiLU epilogue with a "
        "per-slot fp32 state pool (SM100)",
        pybind11::arg("qkv"),
        pybind11::arg("ba"),
        pybind11::arg("alog_dtbias"),
        pybind11::arg("state"),
        pybind11::arg("z"),
        pybind11::arg("norm_w"),
        pybind11::arg("out"),
        pybind11::arg("qo_indptr"),
        pybind11::arg("zero_state"),
        pybind11::arg("num_k_heads"),
        pybind11::arg("o_debug") = c10::nullopt);
  m.def("gdn_recurrent_decode_split_sm100",
        &gdn_recurrent_decode_split_sm100,
        "Decode fast path (q_len == 1, carried state) with the v-row split "
        "across `split` cooperating tasks (SM100)",
        pybind11::arg("qkv"),
        pybind11::arg("ba"),
        pybind11::arg("alog_dtbias"),
        pybind11::arg("state"),
        pybind11::arg("z"),
        pybind11::arg("norm_w"),
        pybind11::arg("out"),
        pybind11::arg("split_scratch"),
        pybind11::arg("qo_indptr"),
        pybind11::arg("num_k_heads"),
        pybind11::arg("split"),
        pybind11::arg("depth"));
  m.def("gdn_recurrent_decode_split_fusedq_sm100",
        &gdn_recurrent_decode_split_fusedq_sm100,
        "M4-I9 flag C: the decode-split recurrence with the fp32-block-scale "
        "FP8 quantize fused into its epilogue (mode 0/1), or the standalone "
        "quantize as the fp8 reference (mode 2)");
  m.def("gdn_gating_probe",
        &gdn_gating_probe,
        "Evaluate only the GDN gating scalars (beta, g) with the task's own "
        "expressions",
        pybind11::arg("ba"),
        pybind11::arg("alog_dtbias"),
        pybind11::arg("beta_out"),
        pybind11::arg("g_out"));
}
