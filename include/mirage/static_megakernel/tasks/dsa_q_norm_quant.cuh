// tasks/dsa_q_norm_quant.cuh -- Apply the MLA q_a_layernorm of GLM-5.3 / 5.2
// (QL = 2048) or GLM-5.3-Flash (QL = 1536) and quantize the result for the FP8
// GEMMs that read it. Task t is token t:
//   q_c[t] = bf16(RMSNorm(q_a[t]) * gamma)
//   z_q[t] = e4m3(q_c[t] / s), one scale s for each 128 columns:
//            s = max(amax, 1e-10) / 448 (the method of group_quant.cuh)
// q_a is the first QL columns of the fused A projection (q_a, then kv_a). A
// COMBINE_ADD GEMM node makes this projection (the demo uses a
// scaled_gemm_tile). The task rounds q_a to bf16. Then it computes
// x * rsqrt(mean(x^2) + eps) * gamma in fp32 and rounds q_c to bf16 one time.
// The quantization reads this bf16
// value. Thus the output is equal to the output of a group_quant node that
// reads q_c. The output is in the exchange region, in the zq32 layout
// (group_quant.cuh): z_q [T][QL] e4m3, then the scales [QL / 128][T] fp32. A
// scaled_gemm_tile node reads it (q_b, and wq_b in a full layer). That node
// waits for the T tasks of this node. A reader that polls 16-byte units must
// not read this output, because the scales of one unit come from 4 tasks.
// Thus scaled_moe_experts must not read it, and the host does not fill the
// buffer with 0xFF before a launch.
// OUT = 1 (GLM-5.3-Flash: its indexer wq_b is a bf16 GEMM): the task writes
// q_c itself, bf16 [T][QL], to its own node buffer instead (no quantization).
// Thread i takes the 8 columns 8 i .. 8 i + 7 (i < QL / 8; with QL = 1536,
// threads 192 .. 255 take none). The 16 threads of a 128-column group find its
// amax with shuffles.
#pragma once
#include "../dsa_common.cuh"
#include "group_quant.cuh" // zq32_bytes: the layout of the output

namespace static_mk {

// Task x is token x. PARAMS {QL, eps (fp32 bits), OUT (0: z_q, 1: bf16 q_c)}.
// BUF_SLOTS {gamma (fp32 [QL])}. QKVA: the fused A projection (its first QL
// columns are q_a).
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class QKVA,
          class... IN>
__device__ __forceinline__ void run_dsa_q_norm_quant(Maps const &,
                                                     G const &g,
                                                     KernelLocals &L,
                                                     StaticTask const &tk) {
  constexpr int QL = PARAMS::v[0], N = QKVA::v[2], PER = 8, OUT = PARAMS::v[2];
  constexpr int GAMMA = slot_at<BUF_SLOTS, 0>();
  static_assert(QL <= N, "dsa_q_norm_quant: q_a in the A projection");
  static_assert((QL == 2048 || QL == 1536) && N % 8 == 0,
                "dsa_q_norm_quant: 8 columns for each thread, 16 threads for "
                "each 128-column group, aligned vector loads");
  static_assert(SELF::x == T && SELF::buf >= 0 && SELF::counter >= 0 &&
                    (OUT == 1 || exchange_bytes[SELF::buf] == zq32_bytes(QL)),
                "dsa_q_norm_quant: a task per token, its counter, the output "
                "z_q in the exchange region (zq32 layout)");
  int const t = tk.x, tid = threadIdx.x, lane = tid & 31;
  __syncthreads(); // wait until the previous task completes its shared-memory
                   // reads
  if (tid < 32) {
    dsa::fx_wait_cols<QKVA>(g, 0, QL);
  }
  __syncthreads();
  dsa::stamp_ready(L);
  float v[PER];
  dsa::fx_rmsnorm_vals<QL, PER>(buf_at<unsigned long long const>(g, QKVA::buf),
                                N,
                                t,
                                0,
                                buf_at<float const>(g, GAMMA),
                                __int_as_float(PARAMS::v[1]),
                                v);
  // Round q_c to bf16 (the value that a group_quant node reads). Then find the
  // amax of the 128-column group (16 threads) and its scale.
  bool const active = tid < QL / PER; // QL = 1536: threads 192 .. 255 idle
  if constexpr (OUT == 1) {
    if (active) {
      uint32_t w[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        __nv_bfloat162 const h = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
        w[i] = *reinterpret_cast<uint32_t const *>(&h);
      }
      *reinterpret_cast<uint4 *>(buf_at<__nv_bfloat16>(g, SELF::buf) +
                                 (size_t)t * QL + PER * tid) =
          make_uint4(w[0], w[1], w[2], w[3]);
    }
    __syncthreads();
    if (tid == 0) {
      cnt_add(cnt_now(g) + SELF::counter, 1);
    }
    return;
  }
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < PER; i++) {
    v[i] = dsa::bf16r(v[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
#pragma unroll
  for (int d = 8; d > 0; d >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
  }
  float const s = fmaxf(amax, 1.0e-10f) * (1.0f / 448.0f);
  uint32_t q[2];
#pragma unroll
  for (int k = 0; k < 2; k++) {
    q[k] = (uint32_t)cvt_e4m3(__fdiv_rn(v[4 * k], s)) |
           ((uint32_t)cvt_e4m3(__fdiv_rn(v[4 * k + 1], s)) << 8) |
           ((uint32_t)cvt_e4m3(__fdiv_rn(v[4 * k + 2], s)) << 16) |
           ((uint32_t)cvt_e4m3(__fdiv_rn(v[4 * k + 3], s)) << 24);
  }
  // Make each 16-byte unit with one store, as group_quant does: the even
  // thread stores its 8 bytes and the 8 bytes of the odd thread after it.
  uint32_t const q2 = __shfl_down_sync(0xffffffffu, q[0], 1),
                 q3 = __shfl_down_sync(0xffffffffu, q[1], 1);
  unsigned char *const zq =
      g.rv + exchange_offset[SELF::buf] + rg_set(); // this launch's set
  if (active && (lane & 1) == 0) {
    *reinterpret_cast<uint4 *>(zq + (size_t)t * QL + PER * tid) =
        make_uint4(q[0], q[1], q2, q3);
  }
  if (active && (tid & 15) == 0) { // the scale of group tid / 16, token t
    *reinterpret_cast<float *>(zq + (size_t)T * QL +
                               ((size_t)(tid / 16) * T + t) * 4) = s;
  }
  __syncthreads();
  if (tid == 0) {
    cnt_add(cnt_now(g) + SELF::counter, 1); // publish the stores of all threads
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_dsa_q_norm_quant() {
  return 0;
}

} // namespace static_mk
