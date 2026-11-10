// tasks/glm_swiglu_mul.cuh (HIP, gfx950) -- GLM-5.3-Flash's dense MLP activation (layers 0 .. 2; SGLang Glm5NextMLP:
// swiglu_clamped with swiglu_limit): h[t][f] = bf16(SiLU(min(gate, L)) clamp(up, -L, L)) (L = 0: SiLU(gate) up), gate /
// up = bf16 of the gate_up GEMM's K parts added (a gemm_tile node's K parts in slots, [P][T][2 F] fp32 polled: gate in
// columns [0, F), up in [F, 2 F)). Task x = features [64 x, 64 x + 64) of every token; a thread per (token, 8 features);
// -> h [T][F] bf16 (the node's buffer, 0xFF-prefilled: the down projection polls it). One count per task.
#pragma once
#include "../dsa_util.cuh" // dsa::slots_col4

namespace static_mk {

// PARAMS {F, limit (fp32 bits)}; GU: the gate_up gemm_tile (K parts in slots)
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class GU, class... IN>
__device__ __forceinline__ void run_glm_swiglu_mul(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  constexpr int F = PARAMS::v[0], NA = GU::v[2];
  static_assert(GU::buf >= 0 && GU::v[0] == COMBINE_SLOTS && NA == 2 * F && F % 64 == 0,
                "glm_swiglu_mul: gate | up as a gemm_tile node's K parts in slots");
  static_assert(SELF::buf >= 0 && SELF::x * 64 == F && SELF::counter >= 0, "glm_swiglu_mul: a task per 64 features");
  float const lim = __int_as_float(PARAMS::v[1]);
  float const *parts = buf_at<float const>(g, GU::buf);
  __hip_bfloat16 *const h = buf_at<__hip_bfloat16>(g, SELF::buf);
  for (int it = STATIC_TIDX; it < T * 8; it += 256) {
    int const t = it / 8, f0 = 64 * tk.x + 8 * (it % 8);
    float gv[8], uv[8];
#pragma unroll
    for (int q = 0; q < 2; q++) {
      float4 const g4 = dsa::slots_col4<GU::y, NA>(parts, t, f0 + 4 * q);
      float4 const u4 = dsa::slots_col4<GU::y, NA>(parts, t, F + f0 + 4 * q);
      gv[4 * q] = g4.x, gv[4 * q + 1] = g4.y, gv[4 * q + 2] = g4.z, gv[4 * q + 3] = g4.w;
      uv[4 * q] = u4.x, uv[4 * q + 1] = u4.y, uv[4 * q + 2] = u4.z, uv[4 * q + 3] = u4.w;
    }
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      float a = round_bf16(gv[k]), b = round_bf16(uv[k]);
      if (lim > 0.f) {
        a = fminf(a, lim);
        b = fminf(fmaxf(b, -lim), lim);
      }
      o[k] = a / (1.0f + __expf(-a)) * b;
    }
    st16_out(h + (size_t)t * F + f0, make_uint4(pack_bf16x2(o[0], o[1]), pack_bf16x2(o[2], o[3]), pack_bf16x2(o[4], o[5]),
                                                pack_bf16x2(o[6], o[7])));
  }
  task_post(cnt_of(g) + SELF::counter, 1);
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_glm_swiglu_mul() {
  return 0;
}

} // namespace static_mk
