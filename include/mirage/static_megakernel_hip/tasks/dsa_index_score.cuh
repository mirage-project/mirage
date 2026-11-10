// tasks/dsa_index_score.cuh (HIP, gfx950) -- GLM-5.3-Flash's lightning-indexer scores over the pooled keys (SGLang ROCm:
// dsa_indexer_kpool.py _get_topk_paged -> aiter deepgemm_fp8_paged_mqa_logits, its torch reference utils.py
// fp8_paged_mqa_logits_torch; amd/notes/sglang_v0522_glm_dsa_contract.md 1.4 steps 1-9). Task (c, t) of grid (CHUNKS, T)
// = token t, pools [c CHP, c CHP + CHP) below n = seq_lens[t] / 4 (the complete pools):
//   q_h   columns [128 h, + 128) of wq_b(q_c) (a gemm_tile node's K parts in slots), bf16 -> Hadamard-128 -> bf16 -> e4m3
//         with the UE8M0 scale qs_h (act_quant ue8m0)
//   w_h   ((wp_h 32^-1/2) qs_h) 128^-1/2 (fp32), wp_h = column 256 + h of the indexer projection [wk | gate |
//         weights_proj] (K parts in slots), not rounded
//   score[t][j] = s_j sum_h w_h relu(q8_h . k8_j)   (k8_j, s_j: pooled key j in the preshuffled page pooled_bt[t][j / 64])
// The dots by MFMA 16x16x32 bf16 on the e4m3 values turned into bf16 exactly (products exact, fp32 sums): per page 4 tiles
// of 16 pools x 2 tiles of 16 heads x 4 K steps; wave w takes the chunk's pages w, w + 4, ..
//   output  scores [T][MAXP] fp32 (the node's buffer); pools >= n not written. One count on counter + t per task.
#pragma once
#include "../dsa_util.cuh"

namespace static_mk {
namespace dsa {

// 8 e4m3 bytes -> 8 bf16 (exact)
__device__ __forceinline__ v8bf e4m3x8_bf16(uint32_t lo, uint32_t hi) {
  v8bf r;
  uint32_t const w[2] = {lo, hi};
#pragma unroll
  for (int i = 0; i < 2; i++) {
    auto const a = __builtin_amdgcn_cvt_pk_f32_fp8((int)w[i], false);
    auto const b = __builtin_amdgcn_cvt_pk_f32_fp8((int)w[i], true);
    r[4 * i + 0] = (__bf16)a[0];
    r[4 * i + 1] = (__bf16)a[1];
    r[4 * i + 2] = (__bf16)b[0];
    r[4 * i + 3] = (__bf16)b[1];
  }
  return r;
}

} // namespace dsa

// PARAMS {32 (heads), MAXP (pools in the output row), 64 (pools a page), MAXPP (pooled block table columns)}; BUF_SLOTS
// {seq_lens (int32), pooled_bt (int32 [T][MAXPP]), the pooled cache (uint8 [pages][8448])}; IQ: wq_b (gemm_tile, K parts
// in slots, [P][T][4096]); KW: the indexer projection (K parts in slots); IC: the pooled cache's producer
// (dsa_kpool_update: its count of T tasks)
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class IQ, class KW, class IC, class... IN>
__device__ __forceinline__ void run_dsa_index_score(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  constexpr int NH = PARAMS::v[0], MAXP = PARAMS::v[1], PAGE = PARAMS::v[2], MAXPP = PARAMS::v[3];
  constexpr int NAQ = IQ::v[2], NAK = KW::v[2], CHP = MAXP / SELF::x, NPG = CHP / PAGE;
  static_assert(NH == 32 && PAGE == dsa::IDX_PAGE_KEYS && CHP % PAGE == 0 && SELF::y == T,
                "dsa_index_score: 32 heads, pages of 64 pools, whole pages per chunk, a task row per token");
  static_assert((IQ::v[0] == COMBINE_SLOTS || IQ::v[0] == COMBINE_LOCAL) && KW::v[0] == COMBINE_SLOTS && NAQ >= 32 * 128 &&
                    NAK >= 256 + 32,
                "dsa_index_score: wq_b (K parts in slots, or a local bf16 gemm_tile) and the indexer projection (slots)");
  static_assert(IC::counter >= 0 && SELF::buf >= 0 && SELF::counter >= 0, "dsa_index_score: the pooled keys' count");
  int const c = tk.x, t = tk.y, tid = STATIC_TIDX, lane = tid & 63, wave = __builtin_amdgcn_readfirstlane(tid >> 6);
  int const r16 = lane & 15, grp = lane >> 4;
  int const n = buf_at<int const>(g, slot_at<BUF_SLOTS, 0>())[t] / 4; // complete pools
  // (params[4] = the top-k K (dsa_index_score_layer skip_k): a token with n <= K pools
  // computes no scores - dsa_index_topk takes every pool in order without them and the sparse attention reads no list; the task
  // still counts itself)
  constexpr int SKIPK = param_or0<PARAMS, 4>();
  int const p_lo = c * CHP, p_hi = (SKIPK > 0 && n <= SKIPK) ? p_lo : min(n, p_lo + CHP);
  unsigned char *const q8s = reinterpret_cast<unsigned char *>(L.sm); // [32][128] e4m3
  float *const ws = reinterpret_cast<float *>(L.sm + NH * 128);       // [32]
  __syncthreads(); // the LDS may still be read by the previous task
  if (p_lo < p_hi) {
    // 1. q8 and w of the 32 heads: each half-wave one head per round (lane l % 32: dims 4 (l % 32) .. + 3)
    float const *iq = buf_at<float const>(g, IQ::buf), *kw = buf_at<float const>(g, KW::buf);
    int const hl = lane & 31;
#pragma unroll 1
    for (int rr = 0; rr < NH / 8; rr++) {
      int const h = 8 * wave + 2 * rr + (lane >> 5);
      float v[1][4];
      if constexpr (IQ::v[0] == COMBINE_LOCAL) { // bf16 [T][NAQ], 0xFF-prefilled: polled 8 B (4 dims) a lane
        __hip_bfloat16 const *iqb = buf_at<__hip_bfloat16 const>(g, IQ::buf);
        v2u w;
        for (;;) {
          w = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(
                                          buf_rsrc(iqb), (uint32_t)(((size_t)t * NAQ + 128 * h + 4 * hl) * 2), 0, AUX_DEV));
          if (w.x != 0xFFFFFFFFu && w.y != 0xFFFFFFFFu) {
            break;
          }
          tail_pause();
        }
        v[0][0] = bf16_lo(w.x);
        v[0][1] = bf16_hi(w.x);
        v[0][2] = bf16_lo(w.y);
        v[0][3] = bf16_hi(w.y);
      } else {
        float4 const q4 = dsa::slots_col4<IQ::y, NAQ>(iq, t, 128 * h + 4 * hl);
        v[0][0] = round_bf16(q4.x);
        v[0][1] = round_bf16(q4.y);
        v[0][2] = round_bf16(q4.z);
        v[0][3] = round_bf16(q4.w);
      }
      dsa::hadamard128<1>(v, hl);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        v[0][j] = round_bf16(v[0][j]);
      }
      uint32_t q[1];
      int e[1];
      dsa::quant128<1>(v, q, e);
      *reinterpret_cast<uint32_t *>(q8s + h * 128 + 4 * hl) = q[0];
      if (hl == 0) {
        float4 const w4 = dsa::slots_col4<KW::y, NAK>(kw, t, 256 + 4 * (h / 4));
        float const wpv = (h & 3) == 0 ? w4.x : (h & 3) == 1 ? w4.y : (h & 3) == 2 ? w4.z : w4.w;
        ws[h] = ((wpv * 0.17677669529663687f) * dsa::exp2_int(e[0])) * 0.08838834764831845f;
      }
    }
  }
  __syncthreads();
  if (p_lo < p_hi) {
    // 2. the B fragments (q8 of heads 16 ht + r16, dims 32 st + 8 grp ..) for the whole task
    v8bf bq[2][4];
#pragma unroll
    for (int ht = 0; ht < 2; ht++)
#pragma unroll
      for (int st = 0; st < 4; st++) {
        uint2 const w = *reinterpret_cast<uint2 const *>(q8s + (16 * ht + r16) * 128 + 32 * st + 8 * grp);
        bq[ht][st] = dsa::e4m3x8_bf16(w.x, w.y);
      }
    float const wh0 = ws[r16], wh1 = ws[16 + r16];
    // the pooled keys are complete once dsa_kpool_update counted all its tasks (the page of this step's pool)
    if (lane == 0) {
      cnt_wait_in(cnt_of(g) + IC::counter, (uint32_t)(IC::x * IC::y * IC::z));
    }
    asm volatile("" ::: "memory");
    int const *const pbt = buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()) + (size_t)t * MAXPP;
    unsigned char const *const pc = buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 2>());
    float *const out = buf_at<float>(g, SELF::buf) + (size_t)t * MAXP;
    int const pg0 = p_lo / PAGE, npg = (p_hi - p_lo + PAGE - 1) / PAGE;
#pragma unroll 1
    for (int pi = wave; pi < npg; pi += 4) {
      int const pg = pg0 + pi;
      unsigned char const *const page = pc + (size_t)pbt[pg] * dsa::IDX_PAGE_BYTES;
      // A fragments of the page's 4 tiles x 4 K steps (pool 16 rt + r16, dims 32 st + 8 grp ..): device-scope loads (the
      // step's pool was written by another CU)
      uint2 ka[4][4];
#pragma unroll
      for (int rt = 0; rt < 4; rt++)
#pragma unroll
        for (int st = 0; st < 4; st++) {
          uint32_t const off = (uint32_t)dsa::idx_key_off(16 * rt + r16, 32 * st + 8 * grp);
          v2u const w = __builtin_amdgcn_raw_buffer_load_b64(buf_rsrc(page), off, 0, AUX_DEV);
          ka[rt][st] = make_uint2(w.x, w.y);
        }
#pragma unroll
      for (int rt = 0; rt < 4; rt++) {
        v4f d0 = {0.f, 0.f, 0.f, 0.f}, d1 = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int st = 0; st < 4; st++) {
          v8bf const a = dsa::e4m3x8_bf16(ka[rt][st].x, ka[rt][st].y);
          d0 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, bq[0][st], d0, 0, 0, 0);
          d1 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, bq[1][st], d1, 0, 0, 0);
        }
        // D: lane -> head column r16 (+ 16 for d1), register r -> pool 16 rt + 4 grp + r; relu, x w_h, the 32 heads added
        float s[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          s[r] = fmaxf(d0[r], 0.f) * wh0 + fmaxf(d1[r], 0.f) * wh1;
#pragma unroll
          for (int m = 1; m < 16; m <<= 1) {
            s[r] += __shfl_xor(s[r], m, 16);
          }
        }
        if (r16 == 0) {
          int const o = 16 * rt + 4 * grp; // the 4 pools of this lane group, within the page
          float4 const sj = as_float4(ld16_at<AUX_DEV>(page, (uint32_t)(dsa::IDX_SCALE_OFF + 4 * o)));
          int const j = PAGE * pg + o;
          if (j < p_hi) { // (the 4 pools in one store; those >= n are never read)
            st16_out(out + j, make_uint4(__float_as_uint(s[0] * sj.x), __float_as_uint(s[1] * sj.y),
                                         __float_as_uint(s[2] * sj.z), __float_as_uint(s[3] * sj.w)));
          }
        }
      }
    }
  }
  task_post(cnt_of(g) + SELF::counter + t, 1);
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_dsa_index_score() {
  return 32 * 128 + 32 * 4;
}

} // namespace static_mk
