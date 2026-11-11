// tasks/glm53_sparse_attn.cuh (HIP, gfx950) -- GLM-5.3 DSA's sparse absorbed MLA attention in plain decode (SGLang v0.5.22
// ROCm: dsa_backend.py _forward_tilelang, sparse_mla_fwd_decode_partial_fp8 + combine; amd/notes/sglang_v0522_glm53_layer_contract.md
// D5), from dsa_sparse_attn.cuh (GLM-5.3-Flash: pools, NoPE). Task (x, y) of grid (T, NS, 1) = request x, split y of its rows,
// all HEADS heads at once (the MFMA's 16 columns):
//   rows       idx[x][i], i < min(seq_lens[x], K): physical rows of SGLang's fp8 MLA cache [rows][KV + RP] (e4m3, no scale; the
//              indexer's top-K slots, or every position's slot in order when seq_len <= K), turned into bf16 exactly in LDS;
//              the step's own row (out_loc[x], written this step by glm53_mla_norm) by device-scope loads after its count,
//              wherever it is in the list
//   query      glm53_q_absorb's output [T][HEADS][KV + RP] bf16 (polled)
//   s = scale (query . row) over KV + RP; online softmax over 64-row chunks (e rounded to bf16 for the product); attn = bf16(sum e
//   row[:KV] / sum e) -> attn [T][HEADS x KV] bf16 (0xFF-prefilled: o absorb polls it)
// Splits (NS > 1): split s > 0 writes its partial and counts on request x's counter; split 0 adds them.
#pragma once
#include "../core.cuh"
#include "mla_attn.cuh" // mla_a, mla_st8_dev, mla_ld8_dev, mla_kt_chunk

namespace static_mk {

// (SPREAD: every split combines a share of the partials after the count, mla_attn_decode.cuh's combine; else split 0 combines)
template <int HEADS, int KV, int RP, int PS, int NS, int K, bool SPREAD = false>
__device__ __forceinline__ void glm53_sparse_attn_task(KernelLocals &L,
                                                     __hip_bfloat16 const *query, // [T][HEADS][KV + RP]
                                                     uint8_t const *cache,        // [rows][KV + RP] e4m3
                                                     int const *seq_lens,         // int32 [T]
                                                     int const *idx,              // [T][K] physical rows (-1 past the valid)
                                                     uint32_t const *topk_cnt,    // (or null: idx is a graph tensor)
                                                     uint32_t topk_need,
                                                     long long const *out_loc,
                                                     uint32_t const *norm_cnt,
                                                     uint32_t norm_need, // dsa_norm's tasks
                                                     float scale,
                                                     __hip_bfloat16 *attn, // [T][HEADS x KV]
                                                     float *part,          // NS > 1: [T][NS][HEADS KV + 2 HEADS]
                                                     uint32_t *scnt,       // NS > 1: [T] split counts
                                                     int x,
                                                     int s,
                                                     int const *dtab = nullptr, // (direct list: req_to_token [reqs][tw])
                                                     long long const *dreq = nullptr, int dtw = 0 STAGE_STAMP_PARAM) {
  using namespace mla_a;
  constexpr int D = KV + RP;
  constexpr int DL = D + mla_a::DPAD;
  static_assert(KV == 512 && (RP == 0 || RP == 64) && PS == 1 && HEADS <= 16 && HEADS % 4 == 0,
                "glm53_sparse_attn: GLM-5.3's shapes (thread maps below)");
  int const tid = task_tid_all();
  int const npos = min(max(seq_lens[x], 0), K); // (the valid entries of the list)
  // (dtab, 11-11: a full layer's list read straight from req_to_token - glm53_index_k's seq_len <= K list is
  // req_to_token[req][p] for p < seq_len: the same rows, no wait for the indexer's k side)
  int const *const ix = dtab != nullptr ? dtab + (size_t)dreq[x] * dtw : idx + (size_t)x * K;
  long long const myloc = out_loc[x];
  if (topk_cnt != nullptr && tid == 0) {
    cnt_wait_in(topk_cnt, topk_need);
  }
  __syncthreads();
  int const nch = (npos + CH - 1) / CH, act = NS < nch ? NS : nch;
  if (s >= act) {
    return; // no rows for this split
  }
  int const c_lo = s * nch / act, c_hi = (s + 1) * nch / act;
  // LDS: the K chunk [CH][DL] bf16, its row indices [CH], the query [HEADS][D] bf16, kT [KV][CH], e [HEADS][CH] (bf16);
  // scores [HEADS][CH], per head m, l, the chunk's correction
  char *const sm = L.sm;
  __hip_bfloat16 *const kc = reinterpret_cast<__hip_bfloat16 *>(sm);
  long long *const rowi = reinterpret_cast<long long *>(sm + CH * DL * 2);
  __hip_bfloat16 *const qb16 = reinterpret_cast<__hip_bfloat16 *>(sm + CH * DL * 2 + CH * 8);
  __hip_bfloat16 *const kT = qb16 + HEADS * D;
  __hip_bfloat16 *const pb16 = kT + KV * CH;
  float *const sc = reinterpret_cast<float *>(pb16 + HEADS * CH);
  float *const mm = sc + HEADS * CH, *const ll = mm + HEADS, *const cr = ll + HEADS, *const fq = cr + HEADS; // fq [NS][HEADS]
  int const lane = tid & 63, wave = tid >> 6, tok = lane % 16, grp = lane / 16;
  __syncthreads(); // the LDS may still be read by the previous task
  if (tid < HEADS) {
    mm[tid] = -3.0e38f;
    ll[tid] = 0.f;
  }
  // the chunk at j0: its row indices, then its rows of earlier steps (plain loads, all in flight)
  // 16 e4m3 bytes -> 16 bf16 (exact) at dst
  auto cvt16 = [](uint4 const &u, __hip_bfloat16 *dst) {
    uint32_t const w[4] = {u.x, u.y, u.z, u.w};
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      auto const a = __builtin_amdgcn_cvt_pk_f32_fp8((int)w[i], false);
      auto const b = __builtin_amdgcn_cvt_pk_f32_fp8((int)w[i], true);
      o[2 * i] = pack_bf16x2(a[0], a[1]); // (exact: e4m3 values are bf16 values)
      o[2 * i + 1] = pack_bf16x2(b[0], b[1]);
    }
    *reinterpret_cast<uint4 *>(dst) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4 *>(dst + 8) = make_uint4(o[4], o[5], o[6], o[7]);
  };
  // the chunk at j0: its rows (the list's entries), then the rows of earlier steps (plain loads, all in flight); returns the
  // chunk index of the step's own row (out_loc), or -1
  auto load_old = [&](int j0, int nr) {
    if (tid < CH) {
      int const j = j0 + tid;
      rowi[tid] = j < npos ? (long long)max(ix[j], 0) : 0;
    }
    __syncthreads();
    int nj = -1;
    for (int r = 0; r < nr; r++) {
      nj = rowi[r] == myloc ? r : nj;
    }
    // every load issued before the first is used (MI355X 11-10 g53f8b, T 2, L2 emptied: with a guarded loop unrolled by 4 a
    // 64-row chunk's 9 loads per thread went out in 3 rounds - its query seen at +13.2 us vs +7.9 for a 9-row chunk); lanes
    // past the chunk load row 0 again and store nothing (no branch around the loads)
    constexpr int NV = (CH * (D / 16) + 255) / 256;
    uint4 rv[NV];
#pragma unroll
    for (int i = 0; i < NV; i++) {
      int const v = tid + 256 * i, r = min(v / (D / 16), max(nr - 1, 0)), c = v % (D / 16);
      rv[i] = *reinterpret_cast<uint4 const *>(cache + (size_t)rowi[r] * D + 16 * c);
    }
#pragma unroll
    for (int i = 0; i < NV; i++) {
      int const v = tid + 256 * i, r = v / (D / 16), c = v % (D / 16);
      if (v < nr * (D / 16) && r != nj) {
        cvt16(rv[i], kc + r * DL + 16 * c);
      }
    }
    return nj;
  };
  // the step's row (glm53_mla_norm's device-scope stores on another CU; after its count): agent-scope loads
  auto load_new = [&](int nj) {
    if (tid < D / 16) {
      cvt16(ld16_at<AUX_DEV>(cache, (uint32_t)((size_t)rowi[nj] * D + 16 * tid)), kc + nj * DL + 16 * tid);
    }
  };
  // 1. the first chunk's rows of earlier steps (no input needed)
  int const first = c_lo * CH;
  int nj_first = load_old(first, min(CH, npos - first));
  // 2. the request's query, every head (polled: HEADS x D / 8 vectors, rounds of 256 in flight together), after the first
  // chunk's own row and its kT transpose, which need no query (MI355X 11-10 g53prof17b, T 1: the splits started ~10 us before
  // their query landed; g53ekt1 pair L7 T 1 83.6 81.5 -> 80.5 80.3, T 8 125.9 127.4 -> 124.9 125.2 us)
  auto poll_query = [&]() {
    constexpr int NQV = HEADS * D / 8, NQR = (NQV + 255) / 256;
    uint4 qv[NQR];
    for (;;) {
      bool all = true;
#pragma unroll
      for (int i = 0; i < NQR; i++) {
        int const v = tid + 256 * i;
        if (v < NQV) {
          qv[i] = ld16_at<AUX_DEV>(query, (uint32_t)(((size_t)x * HEADS * D + 8 * v) * 2));
          all = all && valid16w(qv[i]);
        }
      }
      if (all) {
        break;
      }
      tail_pause();
    }
#pragma unroll
    for (int i = 0; i < NQR; i++) {
      int const v = tid + 256 * i;
      if (v < NQV) {
        *reinterpret_cast<uint4 *>(qb16 + 8 * v) = qv[i];
      }
    }
    if (tid == 0) { // (timing build: stage stamp 0 = the query landed)
      STAGE_STAMP_AT(0)
    }
  };
  // attn^T by MFMA: wave w holds latent rows l = 128 w + 16 i + 4 (l / 16) + r (register r of tile i) of head l % 16
  v4f acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    acc[i] = v4f{0.f, 0.f, 0.f, 0.f};
  }
  for (int j0 = first; j0 < c_hi * CH && j0 < npos; j0 += CH) {
    int const nr = min(CH, npos - j0);
    int const nj = j0 == first ? nj_first : load_old(j0, nr);
    if (nj >= 0) { // the step's row: after glm53_mla_norm counted
      if (tid == 0) {
        cnt_wait_in(norm_cnt, norm_need);
      }
      __syncthreads();
      load_new(nj);
    }
    __syncthreads();
    // the chunk's latent columns transposed into kT (as mla_attn.cuh); rows >= nr as zeros
#pragma unroll
    for (int it = tid; it < (KV / 8) * (CH / 8); it += 256) {
      int const jg = it / (KV / 8), cg = it % (KV / 8);
      uint4 rv[8];
#pragma unroll
      for (int jj = 0; jj < 8; jj++) {
        int const j = 8 * jg + jj;
        rv[jj] = j < nr ? *reinterpret_cast<uint4 const *>(kc + j * DL + 8 * cg) : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {
        uint32_t w4[4];
#pragma unroll
        for (int jj = 0; jj < 8; jj += 2) {
          uint32_t const a = (&rv[jj].x)[e / 2], b = (&rv[jj + 1].x)[e / 2];
          uint32_t const lo = (e & 1) ? (a >> 16) : (a & 0xFFFFu), hi = (e & 1) ? (b >> 16) : (b & 0xFFFFu);
          w4[jj / 2] = lo | (hi << 16);
        }
        int const d = 8 * cg + e;
        *reinterpret_cast<uint4 *>(kT + d * CH + 8 * mla_kt_chunk(d, jg)) = make_uint4(w4[0], w4[1], w4[2], w4[3]);
      }
    }
    if (j0 == first) { // (the query, after the first chunk's transpose)
      poll_query();
      __syncthreads();
    }
    // b. scores S[h][j] = scale (query[h] . row j) by MFMA (wave w: chunk rows 16 w .. + 15)
    {
      v4f sa = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 6
      for (int st = 0; st < D / 32; st++) {
        v4f b = {0.f, 0.f, 0.f, 0.f};
        if (tok < HEADS) {
          b = *reinterpret_cast<v4f const *>(qb16 + tok * D + 32 * st + 8 * grp);
        }
        v4f const a = *reinterpret_cast<v4f const *>(kc + (16 * wave + tok) * DL + 32 * st + 8 * grp);
        sa = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, b), sa, 0, 0, 0);
      }
      if (tok < HEADS) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
          int const j = 16 * wave + 4 * grp + r;
          sc[tok * CH + j] = j < nr ? sa[r] * scale : -3.0e38f;
        }
      }
    }
    __syncthreads();
    // c. per head (wave w: heads w, w + 4, ..; lane = row): the online softmax
    for (int t = wave; t < HEADS; t += 4) {
      float const sv = sc[t * CH + lane];
      float mx = wave_max64(sv);
      float const mo = mm[t], mn = fmaxf(mo, mx);
      float const e = sv > -1.0e38f ? __expf(sv - mn) : 0.f;
      float const se = wave_sum64(e);
      pb16[t * CH + lane] = __float2bfloat16(e); // (bf16 for the product)
      if (lane == 0) {
        float const corr = mo > -1.0e38f ? __expf(mo - mn) : 0.f;
        cr[t] = corr;
        ll[t] = ll[t] * corr + se;
        mm[t] = mn;
      }
    }
    __syncthreads();
    // d. attn^T += K[:, :KV]^T . e^T by MFMA, after the old sums rescaled by the chunk's correction
    {
      float const c = tok < HEADS ? cr[tok] : 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc[i] *= c;
      }
#pragma unroll
      for (int st = 0; st < CH / 32; st++) {
        v4f b = {0.f, 0.f, 0.f, 0.f};
        if (tok < HEADS) {
          b = *reinterpret_cast<v4f const *>(pb16 + tok * CH + 32 * st + 8 * grp);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
          int const d = 128 * wave + 16 * i + tok;
          v4f const a = *reinterpret_cast<v4f const *>(kT + d * CH + 8 * mla_kt_chunk(d, 4 * st + grp));
          acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, b),
                                                           acc[i], 0, 0, 0);
        }
      }
    }
    __syncthreads(); // the chunk's LDS reused by the next one
  }
  if (tid == 0) { // (stage stamp 1: the split's chunks done, before its partial / the combine)
    STAGE_STAMP_AT(1)
  }
  int const lrow = 128 * wave + 4 * grp; // + 16 i: this lane's first latent row of tile i
  if constexpr (NS > 1 && SPREAD) {
    static_assert(KV % 4 == 0, "mla_attn_decode SPREAD: counted partials");
  }
  if (NS > 1 && SPREAD && act > 1) { // (one active split: split 0 alone below, its result in registers)
    constexpr int PW = HEADS * KV + 2 * HEADS, NP = HEADS * KV / 4;
    float *const pp = part + ((size_t)x * NS + s) * PW;
    if (tok < HEADS) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        st16_dev_acc(pp + tok * KV + lrow + 16 * i,
                     make_uint4(__float_as_uint(acc[i][0]), __float_as_uint(acc[i][1]), __float_as_uint(acc[i][2]),
                                __float_as_uint(acc[i][3])));
      }
    }
    if (tid < HEADS) {
      mla_st8_dev(pp + HEADS * KV + 2 * tid, mm[tid], ll[tid]);
    }
    task_post(scnt + x, 1);
    if (tid == 0) {
      cnt_wait_in(scnt + x, act);
    }
    __syncthreads();
    __amdgpu_buffer_rsrc_t const rp = buf_rsrc(part);
    int const p0 = s * NP / act, p1 = (s + 1) * NP / act;
    constexpr int PPT = NS <= 8 ? 2 : 1; // pieces per thread per round, all their loads in flight together
#pragma unroll 1
    for (int v0 = p0 + tid; v0 < p1; v0 += 256 * PPT) {
      v2u mlwa[PPT][NS];
      uint4 pva[PPT][NS];
#pragma unroll
      for (int u = 0; u < PPT; u++) {
        int const v = v0 + 256 * u, h = v / (KV / 4), c = 4 * (v % (KV / 4));
#pragma unroll
        for (int q = 0; q < NS; q++) {
          if (q < act && v < p1) {
            mlwa[u][q] = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(
                                                     rp, (uint32_t)((((size_t)x * NS + q) * PW + HEADS * KV + 2 * h) * 4), 0,
                                                     AUX_DEV));
            pva[u][q] = ld16_at<AUX_DEV>(part, (uint32_t)((((size_t)x * NS + q) * PW + h * KV + c) * 4));
          }
        }
      }
#pragma unroll
      for (int u = 0; u < PPT; u++) {
      int const v = v0 + 256 * u, h = v / (KV / 4), c = 4 * (v % (KV / 4));
      if (v >= p1) {
        break;
      }
      v2u const *const mlw = mlwa[u];
      uint4 const *const pv = pva[u];
      float M = __uint_as_float(mlw[0].x);
#pragma unroll
      for (int q = 1; q < NS; q++) {
        if (q < act) {
          M = fmaxf(M, __uint_as_float(mlw[q].x));
        }
      }
      float const m0 = __uint_as_float(mlw[0].x), f0 = m0 > -1.0e38f ? __expf(m0 - M) : 0.f;
      float l = __uint_as_float(mlw[0].y) * f0;
      float o0 = __uint_as_float(pv[0].x) * f0, o1 = __uint_as_float(pv[0].y) * f0, o2 = __uint_as_float(pv[0].z) * f0,
            o3 = __uint_as_float(pv[0].w) * f0;
#pragma unroll
      for (int q = 1; q < NS; q++) {
        if (q < act) {
          float const mq = __uint_as_float(mlw[q].x), f = mq > -1.0e38f ? __expf(mq - M) : 0.f;
          l += __uint_as_float(mlw[q].y) * f;
          o0 = fmaf(__uint_as_float(pv[q].x), f, o0);
          o1 = fmaf(__uint_as_float(pv[q].y), f, o1);
          o2 = fmaf(__uint_as_float(pv[q].z), f, o2);
          o3 = fmaf(__uint_as_float(pv[q].w), f, o3);
        }
      }
      float const inv = 1.f / l;
      v2u const w = {pack_bf16x2(o0 * inv, o1 * inv), pack_bf16x2(o2 * inv, o3 * inv)};
      asm volatile("global_store_dwordx2 %0, %1, off sc1\n s_nop 1" ::"v"(attn + (size_t)x * HEADS * KV + h * KV + c), "v"(w)
                   : "memory");
      }
    }
    __syncthreads();
    return;
  }
  if constexpr (NS > 1) {
    constexpr int PW = HEADS * KV + 2 * HEADS;
    if (s > 0) { // this split's partial, then one count on request x's counter
      float *const pp = part + ((size_t)x * NS + s) * PW;
      if (tok < HEADS) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          st16_dev_acc(pp + tok * KV + lrow + 16 * i,
                   make_uint4(__float_as_uint(acc[i][0]), __float_as_uint(acc[i][1]), __float_as_uint(acc[i][2]),
                              __float_as_uint(acc[i][3])));
        }
      }
      if (tid < HEADS) {
        mla_st8_dev(pp + HEADS * KV + 2 * tid, mm[tid], ll[tid]);
      }
      task_post(scnt + x, 1);
      return;
    }
    if (tid == 0) {
      cnt_wait_in(scnt + x, act - 1);
    }
    __syncthreads();
    // every split's (m, l) loaded at once (tracked loads), then the factors; the partials two splits at a time (MI355X dec51,
    // T = 1: one load-and-wait per split, three times over, made the combine ~10 us; dec52: T = 1, 700 rows 75.4 / 76.8 vs
    // 80.4 / 81.2 us)
    if (tid < HEADS) {
      __amdgpu_buffer_rsrc_t const rp = buf_rsrc(part);
      v2u mlw[NS];
#pragma unroll
      for (int q = 1; q < NS; q++) {
        if (q < act) {
          mlw[q] = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(
                                               rp, (uint32_t)((((size_t)x * NS + q) * PW + HEADS * KV + 2 * tid) * 4), 0,
                                               AUX_DEV));
        }
      }
      float M = mm[tid];
#pragma unroll
      for (int q = 1; q < NS; q++) {
        if (q < act) {
          M = fmaxf(M, __uint_as_float(mlw[q].x));
        }
      }
      float const f0 = mm[tid] > -1.0e38f ? __expf(mm[tid] - M) : 0.f;
      float l = ll[tid] * f0;
#pragma unroll
      for (int q = 1; q < NS; q++) {
        if (q < act) {
          float const mq = __uint_as_float(mlw[q].x), f = mq > -1.0e38f ? __expf(mq - M) : 0.f;
          l += __uint_as_float(mlw[q].y) * f;
          fq[q * HEADS + tid] = f;
        }
      }
      cr[tid] = f0;
      mm[tid] = M;
      ll[tid] = l;
    }
    __syncthreads();
    if (tok < HEADS) {
      float const c0 = cr[tok];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc[i] *= c0;
      }
#pragma unroll 1
      for (int q0 = 1; q0 < act; q0 += 2) {
        uint4 vq[2][8];
#pragma unroll
        for (int k = 0; k < 2; k++) {
          if (q0 + k < act) {
            float const *const pp = part + ((size_t)x * NS + q0 + k) * PW;
#pragma unroll
            for (int i = 0; i < 8; i++) {
              vq[k][i] = ld16_at<AUX_DEV>(pp, (uint32_t)((tok * KV + lrow + 16 * i) * 4));
            }
          }
        }
#pragma unroll
        for (int k = 0; k < 2; k++) {
          if (q0 + k < act) {
            float const f = fq[(q0 + k) * HEADS + tok];
#pragma unroll
            for (int i = 0; i < 8; i++) {
              acc[i][0] = fmaf(__uint_as_float(vq[k][i].x), f, acc[i][0]);
              acc[i][1] = fmaf(__uint_as_float(vq[k][i].y), f, acc[i][1]);
              acc[i][2] = fmaf(__uint_as_float(vq[k][i].z), f, acc[i][2]);
              acc[i][3] = fmaf(__uint_as_float(vq[k][i].w), f, acc[i][3]);
            }
          }
        }
      }
    }
  }
  __syncthreads(); // the chunk buffer (kc) free: the request's output staged there, [HEADS][KV] bf16
  if (tok < HEADS) {
    float const inv = 1.f / ll[tok];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      *reinterpret_cast<uint2 *>(kc + tok * KV + lrow + 16 * i) =
          make_uint2(pack_bf16x2(acc[i][0] * inv, acc[i][1] * inv), pack_bf16x2(acc[i][2] * inv, acc[i][3] * inv));
    }
  }
  __syncthreads();
  for (int v = tid; v < HEADS * KV / 8; v += 256) { // the request's HEADS x KV x 2 B in 16-B stores
    st16_dev(attn + (size_t)x * HEADS * KV + 8 * v, *reinterpret_cast<uint4 const *>(kc + 8 * v));
  }
  __syncthreads();
}

// (the optional input after Q and N: the indexer node whose buffer holds the rows' list)
struct g53_no_idx {
  static constexpr int buf = -1, counter = -1, x = 0, y = 0, z = 0;
};
template <class... R_>
struct g53_first_in {
  using type = g53_no_idx;
};
template <class A_, class... R_>
struct g53_first_in<A_, R_...> {
  using type = A_;
};
// task (x, y) = request x, split y; Q = glm53_q_absorb (its buffer: the query), N = glm53_mla_norm (its counter), I (optional) =
// the indexer node (its buffer: the rows' list [T][K] int32; its counter); params = [HEADS, KV, RP, PS (1), scale bits, K];
// buffer slots = the fp8 cache, seq_lens (int32), out_loc (int64), the rows' list as a graph tensor (no I: a shared layer's,
// written by an earlier launch) or -1, (splits: the partials' scratch); counters (splits): one per request
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class Q, class N, class... IN>
__device__ __forceinline__ void run_glm53_sparse_attn(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  using I = typename g53_first_in<IN...>::type;
  static_assert(Q::buf >= 0, "glm53_sparse_attn: glm53_q_absorb's output (the query)");
  static_assert(N::counter >= 0, "glm53_sparse_attn: glm53_mla_norm's counter");
  static_assert(I::buf < 0 || I::counter >= 0, "glm53_sparse_attn: the indexer node's counter");
  static_assert(SELF::buf >= 0 && SELF::x == T, "glm53_sparse_attn: one task per request (and split), its output");
  static_assert(SELF::y == 1 || SELF::counter >= 0, "glm53_sparse_attn: splits count on the node's counters");
  float *part = nullptr;
  uint32_t *scnt = nullptr;
  if constexpr (SELF::y > 1) {
    part = buf_at<float>(g, slot_at<BUF_SLOTS, 4>());
    scnt = cnt_of(g) + SELF::counter;
  }
  int const *rows;
  uint32_t const *icnt = nullptr;
  if constexpr (I::buf >= 0) {
    rows = buf_at<int const>(g, I::buf);
    icnt = cnt_of(g) + I::counter;
  } else {
    rows = buf_at<int const>(g, slot_at<BUF_SLOTS, 3>());
  }
  // (params[6]: the splits' combine, 2 SPREAD (static_megakernel.glm53_sparse_attn_layer), 0 split 0 alone)
  constexpr bool DIRECT = (int)(sizeof(PARAMS::v) / sizeof(int)) > 8 && PARAMS::v[7] == 1; // (params[7] 1: the list from
  int const *dtab = nullptr;                                                                  //  req_to_token, slots 5 / 6)
  long long const *dreq = nullptr;
  if constexpr (DIRECT) {
    dtab = buf_at<int const>(g, slot_at<BUF_SLOTS, 5>());
    dreq = buf_at<long long const>(g, slot_at<BUF_SLOTS, 6>());
    icnt = nullptr;
  }
  glm53_sparse_attn_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], SELF::y, PARAMS::v[5], PARAMS::v[6] == 2>(
      L,
      buf_at<__hip_bfloat16 const>(g, Q::buf),
      buf_at<uint8_t const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()),
      rows,
      icnt,
      (uint32_t)(I::x * I::y * I::z),
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 2>()),
      cnt_of(g) + N::counter,
      (uint32_t)(N::x * N::y * N::z),
      __int_as_float(PARAMS::v[4]),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      part,
      scnt,
      tk.x,
      tk.y,
      dtab,
      dreq,
      DIRECT ? PARAMS::v[8] : 0 STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_glm53_sparse_attn() {
  // the K chunk and its row indices, the query, kT, e (bf16); scores, m / l / corr
  return mla_a::CH * (PARAMS::v[1] + PARAMS::v[2] + mla_a::DPAD) * 2 + mla_a::CH * 8 +
         PARAMS::v[0] * (PARAMS::v[1] + PARAMS::v[2]) * 2 + PARAMS::v[1] * mla_a::CH * 2 + PARAMS::v[0] * mla_a::CH * 2 +
         (PARAMS::v[0] * mla_a::CH + 3 * PARAMS::v[0] + SELF::y * PARAMS::v[0]) * 4;
}

} // namespace static_mk
