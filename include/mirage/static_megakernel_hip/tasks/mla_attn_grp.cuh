// tasks/mla_attn_grp.cuh (HIP, gfx950) -- K3 MLA's absorbed attention for a
// group of HG heads over one KV split (the math of mla_attn.cuh; B300 design:
// mirage_kda tasks/mla_attn_part.cuh). Task (g, s) of grid (heads / HG, NS, 1) =
// heads [HG g, HG g + HG), KV split s:
//   queries  qi = hh T + t (head HG g + hh, token t): mla_absorb's output
//            [T][heads][KV + RP] bf16 (0xFF-prefilled, polled): NQ = HG T = 16
//            rows, held in registers as the score MFMAs' B fragments (wave w: the
//            K steps st = w mod 4 of the D / 32; its partial scores of all 64 rows
//            to LDS, the 4 waves' partials added in wave order: no atomics)
//   rows     as mla_attn.cuh: j < seq_lens[0] from the cache through the token
//            table, this step's T rows at out_loc (written by mla_norm: read
//            after its count, device scope); 64-row chunks, nch of them over act =
//            min(NS, nch) splits (split s: chunks s nch / act .. (s + 1) nch / act)
//   per chunk: S [rows][NQ] = scale (rows . queries) by MFMA (wave w: rows 16 w ..
//            + 15); online softmax per query (e rounded to bf16 for the product,
//            the sum of the unrounded e); O^T [KV][NQ] += kT . P^T by MFMA (wave w:
//            latent rows 128 w .. + 127)
//   act = 1: attn = bf16(O / l) straight to the node's output [T][heads x KV]
//            (0xFF-prefilled: mla_vgate polls it)
//   act > 1: every split writes its partial (O [NQ][KV], then m, l [NQ]; fp32,
//            device scope) and counts on the group's counter, waits for the act
//            counts and combines its 1 / act share of the group's NQ x KV outputs
//            (weights l_s exp(m_s - M) / sum, M = max m_s). The splits of a group
//            run at the same time (the node's concurrent axis); at most one of the
//            node's tasks per CU (the plan: groups x NS <= CUs), so no split waits
//            behind another.
// The pipeline: chunk c + 1's rows of earlier steps are loaded into registers
// (18 x 16 B per thread, all in flight) while chunk c is computed; the token
// table's entries for chunk c + 2 are loaded with them. Nothing between the
// loads and the next chunk's top uses them: every load unconditional (rows
// without data read cache row 0, written to LDS as zeros), the table entry kept
// raw (MI355X 10-13: a branch per load and the entry's row computed right after
// its load made the compiler wait for all 19 loads there: no overlap)
#pragma once
#include "../core.cuh"
#include "mla_norm.cuh" // mla_st4_dev

namespace static_mk {

// kT [KV][CH = 64] in 16-B chunks XOR-swizzled by row
// (chunk c of row d at mla_kt_chunk(d, c)) and the transpose's items along the
// columns: P.V's reads (16 rows of one column per 8 lanes: 2 bank groups, 4-way
// conflicts) and the transpose's (8 rows 9 KB apart) without conflicts
// No kT transpose: the O MFMAs' A operands (8 chunk
// rows of one latent column per lane) read from the chunk by two transposing LDS
// reads (ds_read_b64_tr_b16: in each 16-lane group, lane s gives 4 values M[s][0..3];
// lane i gets M[4 e + i / 4][i % 4], e = 0 .. 3; amd/bench/lds_tr_check.hip, MI355X
// 10-15). Lane s points at chunk row j0 + s / 4, columns d0 + 4 (s % 4) .. + 3: lane i
// gets rows j0 .. j0 + 3 of column d0 + i (MI355X 10-15 kda126: the transpose ~2 us
// per chunk, on the path at 37 rows with the queries late)
typedef short mla_v4s __attribute__((ext_vector_type(4)));
__device__ __forceinline__ mla_v4s mla_ld_tr16(__hip_bfloat16 const *p) { // (LDS; the whole wave active)
  return __builtin_amdgcn_ds_read_tr16_b64_v4i16(
      (__attribute__((address_space(3))) mla_v4s *)((__attribute__((address_space(3))) char *)(p)));
}
// The online softmax with lane -> (query 4 w + l / 16, rows 4 (l % 16) .. + 3):
// each query's max and sum over its 16 lanes by 4 DPP steps, the 4 queries of a
// wave at once (before: one query at a time, lane = row, two 64-lane reductions)
template <bool MAX>
__device__ __forceinline__ float mla_row16_red(float v) { // over the 16 lanes of a DPP row (every lane gets it)
  auto op = [](float a, float b) { return MAX ? fmaxf(a, b) : a + b; };
  v = op(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false)));  // quad_perm [1,0,3,2]
  v = op(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, false)));  // quad_perm [2,3,0,1]
  v = op(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x141, 0xF, 0xF, false))); // row_half_mirror
  v = op(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x140, 0xF, 0xF, false))); // row_mirror
  return v;
}
// The queries are polled after the first chunk is in LDS (that work needs no
// queries: it overlaps mla_absorb's tail)
__device__ __forceinline__ int mla_kt_chunk(int d, int c) {
  return c ^ ((d >> 3) & 7) ^ (d & 7);
}

namespace mla_g {
constexpr int HG = 2;   // heads per task
constexpr int CH = 64;  // KV rows per chunk
constexpr int DPAD = 8; // the chunk's LDS row stride: D + 8 bf16
} // namespace mla_g

__device__ __forceinline__ void mla_g_st8_dev(void *p, uint2 v) {
  v2u const w = {v.x, v.y};
  asm volatile("global_store_dwordx2 %0, %1, off sc1" ::"v"(p), "v"(w) : "memory");
}

template <int HEADS, int KV, int RP, int PS, int NS>
__device__ __forceinline__ void mla_attn_grp_task(KernelLocals &L,
                                                  __hip_bfloat16 const *query, // [T][HEADS][KV + RP]
                                                  __hip_bfloat16 const *cache, // [rows][KV + RP]
                                                  int const *table,
                                                  int const *info,
                                                  long long const *seq_lens,
                                                  long long const *req_idx,
                                                  long long const *out_loc,
                                                  uint32_t const *norm_cnt,
                                                  uint32_t norm_need,  // mla_norm's tasks
                                                  float scale,
                                                  __hip_bfloat16 *attn,        // [T][HEADS x KV]
                                                  float *part,                 // NS > 1: [HEADS / HG][NS][NQ KV + 2 NQ]
                                                  uint32_t *scnt,              // NS > 1: [HEADS / HG] split counts
                                                  int g,
                                                  int s STAGE_STAMP_PARAM) {
  using namespace mla_g;
  constexpr int D = KV + RP, DL = D + DPAD, NQ = HG * T, VR = D / 8, PV = CH * VR / 256, KS = D / 32,
                KW = (KS + 3) / 4; // K steps of the scores; per wave at most
  static_assert(CH == 64, "mla_attn_grp: kT rows of 8 16-B chunks (mla_kt_chunk)");
  static_assert(KV == 512 && RP == 64 && T == 8 && NQ == 16 && (CH * VR) % 256 == 0 && HEADS % HG == 0,
                "mla_attn_grp: K3's shapes (thread maps below)");
  int const tid = task_tid_all(), lane = tid & 63, wave = tid >> 6, ql = lane % 16, grp = lane / 16;
  int const Lp = (int)seq_lens[0], seq_k = Lp + T;
  int const *const rt = table + (size_t)req_idx[0] * (size_t)info[3];
  int const nch = (seq_k + CH - 1) / CH, act = NS < nch ? NS : nch;
  if (s >= act) {
    return; // no rows for this split
  }
  int const c_lo = s * nch / act, c_hi = (s + 1) * nch / act;
  // LDS: the chunk [CH][DL] bf16 | kT [KV][CH] bf16 | scores [4][NQ][CH] fp32 | e
  // [NQ][CH] bf16 | cache rows [CH] int64 | m, l, corr [NQ] fp32
  char *const sm = L.sm;
  __hip_bfloat16 *const kc = reinterpret_cast<__hip_bfloat16 *>(sm);
  __hip_bfloat16 *const kT = kc + CH * DL;
  float *const sc = reinterpret_cast<float *>(kT + KV * CH);
  __hip_bfloat16 *const pb = reinterpret_cast<__hip_bfloat16 *>(sc + 4 * NQ * CH);
  long long *const rowi = reinterpret_cast<long long *>(pb + NQ * CH);
  float *const mm = reinterpret_cast<float *>(rowi + CH), *const ll = mm + NQ, *const cr = ll + NQ;
  auto row_of = [&](int j) -> long long { // a row of an earlier step: its cache row
    return (long long)rt[j / PS] * PS + j % PS;
  };
  // the table entry of row j, loaded whatever j is (j clamped to the request's
  // rows: no branch around the load); entry_row makes the cache row of it
  auto entry_of = [&](int j) -> int { return rt[min(max(j, 0), max(Lp - 1, 0)) / PS]; };
  auto entry_row = [&](int j, int e) -> long long { return j < Lp ? (long long)e * PS + j % PS : -1; };
  __syncthreads(); // the LDS may still be read by the previous task
  if (tid < NQ) {
    mm[tid] = -3.0e38f;
    ll[tid] = 0.f;
  }
  // 1. chunk c_lo's cache rows, its rows of earlier steps in flight, the table's
  // entries of chunk c_lo + 1 (no input needed)
  if (tid < CH) {
    int const j = c_lo * CH + tid;
    rowi[tid] = j < Lp ? row_of(j) : -1;
  }
  __syncthreads();
  uint4 pre[PV];
  auto issue = [&](int j0) { // the chunk at j0: its rows of earlier steps (rowi; -1: row 0, unused) into pre
#pragma unroll
    for (int i = 0; i < PV; i++) {
      int const v = tid + 256 * i, r = v / VR;
      long long const cr_ = rowi[r];
      pre[i] = *reinterpret_cast<uint4 const *>(cache + (size_t)(cr_ >= 0 ? cr_ : 0) * D + 8 * (v % VR));
    }
  };
  issue(c_lo * CH);
  int enext = 0; // (tid < CH) the raw table entry of row tid of the next chunk
  if (tid < CH) {
    enext = entry_of((c_lo + 1) * CH + tid);
  }
  // 2. the queries (polled): lane -> query ql (head HG g + ql / T, token ql % T),
  // its columns 32 st + 8 grp .. + 7 for this wave's K steps st = wave + 4 k
  // (the score MFMAs' B fragments)
  uint4 qf[KW];
#if STATIC_STAMP_SET <= 1 // (timing build: stage stamp 0 = the queries landed)
#define MLA_GRP_Q_STAMP                                                                                                          \
  if (tid == 0) {                                                                                                                \
    STAGE_STAMP_AT(0)                                                                                                            \
  }
#else
#define MLA_GRP_Q_STAMP
#endif
#define MLA_GRP_POLL_Q                                                                                                           \
  {                                                                                                                              \
    int const t = ql % T, hd = HG * g + ql / T;                                                                                  \
    uint32_t const qo = (uint32_t)((((size_t)t * HEADS + hd) * D + 8 * grp) * 2);                                                \
    for (;;) {                                                                                                                   \
      bool all = true;                                                                                                           \
      _Pragma("unroll") for (int k = 0; k < KW; k++) {                                                                           \
        int const st = wave + 4 * k;                                                                                             \
        qf[k] = st < KS ? ld16_at<AUX_DEV>(query, qo + (uint32_t)(64 * st)) : make_uint4(0u, 0u, 0u, 0u);                        \
        all = all && (st >= KS || valid16w(qf[k]));                                                                              \
      }                                                                                                                          \
      if (all) {                                                                                                                 \
        break;                                                                                                                   \
      }                                                                                                                          \
      tail_pause();                                                                                                              \
    }                                                                                                                            \
  }                                                                                                                              \
  MLA_GRP_Q_STAMP
  v4f acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    acc[i] = v4f{0.f, 0.f, 0.f, 0.f};
  }
  for (int c = c_lo; c < c_hi; c++) {
    int const j0 = c * CH, nr = min(CH, seq_k - j0), nold = max(0, min(nr, Lp - j0));
    __syncthreads(); // the previous chunk's LDS reads done
    // a. the chunk into LDS: its rows of earlier steps from pre, rows >= nr as
    // zeros (their e is 0; stale values must not be NaN)
#pragma unroll
    for (int i = 0; i < PV; i++) {
      int const v = tid + 256 * i, r = v / VR;
      if (r < nold || r >= nr) {
        *reinterpret_cast<uint4 *>(kc + r * DL + 8 * (v % VR)) = r < nold ? pre[i] : make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if (tid < CH) { // the next chunk's cache rows
      rowi[tid] = entry_row(j0 + CH + tid, enext);
    }
    if (nold < nr) { // this step's rows: after mla_norm counted T (device scope)
      if (tid == 0) {
        cnt_wait_in(norm_cnt, norm_need);
      }
      __syncthreads();
      constexpr int NNEW = (T * VR + 255) / 256;
      int const nnew = (nr - nold) * VR;
      unsigned long long w[NNEW][2] = {};
#pragma unroll
      for (int i = 0; i < NNEW; i++) {
        int const v = tid + 256 * i;
        if (v < nnew) {
          unsigned long long const *const src = reinterpret_cast<unsigned long long const *>(
              cache + (size_t)out_loc[j0 + nold + v / VR - Lp] * D + 8 * (v % VR));
          w[i][0] = __hip_atomic_load(src, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          w[i][1] = __hip_atomic_load(src + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
#pragma unroll
      for (int i = 0; i < NNEW; i++) {
        int const v = tid + 256 * i;
        if (v < nnew) {
          unsigned long long *const dst = reinterpret_cast<unsigned long long *>(kc + (nold + v / VR) * DL + 8 * (v % VR));
          dst[0] = w[i][0];
          dst[1] = w[i][1];
        }
      }
    }
    __syncthreads();
#if STATIC_STAMP_SET <= 1
    if (tid == 0 && c == c_lo) { // (stage stamp 1: the first chunk in LDS)
      STAGE_STAMP_AT(1)
    }
#elif STATIC_STAMP_SET == 3
    if (tid == 0 && c == c_lo + 1) { // (stamp set 3: stamp 0 = the second chunk in LDS, 1 = the third)
      STAGE_STAMP_AT(0)
    }
    if (tid == 0 && c == c_lo + 2) {
      STAGE_STAMP_AT(1)
    }
#endif
    // b. kT: the chunk's latent columns transposed (item = 8 columns x 4 rows)
    // c. the next chunk's rows in flight, the table's entries of the one after
    // (after the transpose: the compiler's wait at the transpose's first LDS
    // read, MI355X 10-13, then finds no load outstanding)
    if (c + 1 < c_hi) {
      issue(j0 + CH);
      if (tid < CH) {
        enext = entry_of(j0 + 2 * CH + tid);
      }
    }
    if (c == c_lo) {
      MLA_GRP_POLL_Q
    }
    // d. partial scores (wave w: its K steps, the 4 tiles of 16 rows; D: lane ->
    // query ql, register r -> row 16 i + 4 grp + r)
    {
      v4f sa[CH / 16];
#pragma unroll
      for (int i = 0; i < CH / 16; i++) {
        sa[i] = v4f{0.f, 0.f, 0.f, 0.f};
      }
#pragma unroll
      for (int k = 0; k < KW; k++) {
        int const st = wave + 4 * k;
        if (st < KS) {
#pragma unroll
          for (int i = 0; i < CH / 16; i++) {
            v4f const a = *reinterpret_cast<v4f const *>(kc + (16 * i + ql) * DL + 32 * st + 8 * grp);
            sa[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, qf[k]), sa[i],
                                                            0, 0, 0);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CH / 16; i++) {
        *reinterpret_cast<float4 *>(sc + (wave * NQ + ql) * CH + 16 * i + 4 * grp) =
            make_float4(sa[i][0], sa[i][1], sa[i][2], sa[i][3]);
      }
    }
    __syncthreads();
#if STATIC_STAMP_SET == 4
    if (tid == 0 && c == c_lo) { // (stamp set 4: stamp 0 = the first chunk's transpose and scores done, 1 = its softmax)
      STAGE_STAMP_AT(0)
    }
#endif
    // e. the online softmax (wave w: queries 4 w .. + 3; lane = row): the score
    // = scale x the waves' partials added in wave order; rows past the chunk or
    // after the query's token masked
    {
      static_assert(NQ == 16 && CH == 64, "mla_attn_grp softmax: 4 queries per wave, 16 lanes x 4 rows each");
      int const q = (NQ / 4) * wave + (lane >> 4), r0 = 4 * (lane & 15);
      float4 pw[4];
#pragma unroll
      for (int w4 = 0; w4 < 4; w4++) {
        pw[w4] = *reinterpret_cast<float4 const *>(sc + (w4 * NQ + q) * CH + r0);
      }
      float sv[4], mx = -3.0e38f;
#pragma unroll
      for (int e = 0; e < 4; e++) {
        int const row = r0 + e;
        float const raw = (((&pw[0].x)[e] + (&pw[1].x)[e]) + (&pw[2].x)[e]) + (&pw[3].x)[e];
        sv[e] = (row < nr && j0 + row <= Lp + q % T) ? raw * scale : -3.0e38f;
        mx = fmaxf(mx, sv[e]);
      }
      mx = mla_row16_red<true>(mx);
      float const mo = mm[q], mn = fmaxf(mo, mx);
      float ev[4], se = 0.f;
#pragma unroll
      for (int e = 0; e < 4; e++) {
        ev[e] = sv[e] > -1.0e38f ? __expf(sv[e] - mn) : 0.f;
        se += ev[e];
      }
      se = mla_row16_red<false>(se);
      *reinterpret_cast<uint2 *>(pb + q * CH + r0) = make_uint2(pack_bf16x2(ev[0], ev[1]), pack_bf16x2(ev[2], ev[3]));
      if ((lane & 15) == 0) {
        float const corr = mo > -1.0e38f ? __expf(mo - mn) : 0.f;
        cr[q] = corr;
        ll[q] = ll[q] * corr + se;
        mm[q] = mn;
      }
    }
    __syncthreads();
#if STATIC_STAMP_SET == 4
    if (tid == 0 && c == c_lo) {
      STAGE_STAMP_AT(1)
    }
#endif
    // f. O^T += kT . P^T (A: kT rows 128 w + 16 i + l % 16; B: query ql's e),
    // after the old sums rescaled by the chunk's correction
    {
      float const cf = cr[ql];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc[i] *= cf;
      }
#pragma unroll
      for (int st = 0; st < CH / 32; st++) {
        v4f const b = *reinterpret_cast<v4f const *>(pb + ql * CH + 32 * st + 8 * grp);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          __hip_bfloat16 const *const src = kc + (32 * st + 8 * grp + (ql >> 2)) * DL + 128 * wave + 16 * i + 4 * (ql & 3);
          mla_v4s const lo = mla_ld_tr16(src), hi = mla_ld_tr16(src + 4 * DL);
          v4f const a = __builtin_bit_cast(v4f, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
          acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, b), acc[i], 0,
                                                           0, 0);
        }
      }
    }
  }
  __syncthreads();
#if STATIC_STAMP_SET == 2
  if (tid == 0) { // (stamp set 2: stamp 0 = the chunk loop done, 1 = the group's partials all counted)
    STAGE_STAMP_AT(0)
  }
#endif
  // lane: query ql, latent rows lrow + 16 i .. + 3
  int const lrow = 128 * wave + 4 * grp, t = ql % T, hd = HG * g + ql / T;
  if (act == 1) {
    float const inv = 1.f / ll[ql];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      __hip_bfloat16 *const dst = attn + ((size_t)t * HEADS + hd) * KV + lrow + 16 * i;
      mla_g_st8_dev(dst, make_uint2(pack_bf16x2(acc[i][0] * inv, acc[i][1] * inv), pack_bf16x2(acc[i][2] * inv, acc[i][3] * inv)));
    }
    __syncthreads();
    return;
  }
  if constexpr (NS > 1) {
    constexpr int PW = NQ * KV + 2 * NQ, U = NQ * KV / 4; // partial floats per split; 4-float units of the outputs
    float *const pg = part + (size_t)g * NS * PW;
    {
      float *const pp = pg + (size_t)s * PW;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        st16_dev_acc(pp + ql * KV + lrow + 16 * i, make_uint4(__float_as_uint(acc[i][0]), __float_as_uint(acc[i][1]),
                                                          __float_as_uint(acc[i][2]), __float_as_uint(acc[i][3])));
      }
      if (tid < NQ) {
        mla_g_st8_dev(pp + NQ * KV + 2 * tid, make_uint2(__float_as_uint(mm[tid]), __float_as_uint(ll[tid])));
      }
    }
    task_post(scnt + g, 1);
    if (tid == 0) {
      cnt_wait_in(scnt + g, act);
#if STATIC_STAMP_SET == 2
      STAGE_STAMP_AT(1)
#endif
    }
    __syncthreads();
    // this split's share of the outputs: units u_lo .. u_hi - 1 (4 floats each);
    // LDS: wt [act][NQ] (the weights), then the loaded units [act][nu] float4
    int const u_lo = U * s / act, u_hi = U * (s + 1) / act, nu = u_hi - u_lo;
    float *const wt = reinterpret_cast<float *>(sm);
    float *const mls = wt + NS * NQ;                          // [act][NQ][2]
    uint4 *const slot = reinterpret_cast<uint4 *>(mls + NS * NQ * 2);
    { // every split's m / l and this share's units: all loads in flight together (one round trip)
      constexpr int NML = (NS * NQ + 255) / 256, NSL = (U + NS + 255) / 256; // (act x nu <= U + act units)
      float2 vm[NML];
      uint4 vs[NSL];
#pragma unroll
      for (int k = 0; k < NML; k++) {
        int const i = tid + 256 * k;
        if (i < act * NQ) {
          uint32_t const off = (uint32_t)((((size_t)(i / NQ)) * PW + NQ * KV + 2 * (i % NQ)) * 4);
          vm[k] = make_float2(__uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(buf_rsrc(pg), off, 0, AUX_DEV)),
                              __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(buf_rsrc(pg), off + 4, 0, AUX_DEV)));
        }
      }
#pragma unroll
      for (int k = 0; k < NSL; k++) {
        int const i = tid + 256 * k;
        if (i < act * nu) {
          vs[k] = ld16_at<AUX_DEV>(pg, (uint32_t)(((size_t)(i / nu) * PW + 4 * (size_t)(u_lo + i % nu)) * 4));
        }
      }
#pragma unroll
      for (int k = 0; k < NML; k++) {
        int const i = tid + 256 * k;
        if (i < act * NQ) {
          mls[2 * i] = vm[k].x;
          mls[2 * i + 1] = vm[k].y;
        }
      }
#pragma unroll
      for (int k = 0; k < NSL; k++) {
        int const i = tid + 256 * k;
        if (i < act * nu) {
          slot[i] = vs[k];
        }
      }
    }
    __syncthreads();
    { // the weights: query qi's 16 lanes (lanes 16 j .. + 15 of wave qi / 4), split q at lane q % 16 (MI355X 10-13: 16
      // threads looping over the act splits, one LDS load after the other, ~3-4 us)
      int const qi = tid / 16, sl = tid % 16;
      float m = -3.0e38f;
      for (int q = sl; q < act; q += 16) {
        m = fmaxf(m, mls[2 * (q * NQ + qi)]);
      }
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) {
        m = fmaxf(m, __shfl_xor(m, o));
      }
      float den = 0.f;
      for (int q = sl; q < act; q += 16) {
        float const mq = mls[2 * (q * NQ + qi)];
        den += mq > -1.0e38f ? mls[2 * (q * NQ + qi) + 1] * __expf(mq - m) : 0.f;
      }
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) {
        den += __shfl_xor(den, o);
      }
      float const inv = 1.f / den;
      for (int q = sl; q < act; q += 16) {
        float const mq = mls[2 * (q * NQ + qi)];
        wt[q * NQ + qi] = mq > -1.0e38f ? __expf(mq - m) * inv : 0.f;
      }
    }
    __syncthreads();
    // the sums: tpu threads per unit (a power of 2 up to 8 with tpu x nu <= 256 where
    // possible): lane sub adds splits sub, sub + tpu, ...; then the tpu partials
    // by xor shuffles (a fixed order: the same bits every launch)
    int tpu = 1;
    while (tpu < 8 && 2 * tpu * nu <= 256) {
      tpu *= 2;
    }
    for (int base = 0; base < nu * tpu; base += 256) { // (every thread in each round: the shuffles)
      int const t = base + tid, k = t / tpu, sub = t % tpu;
      bool const live = k < nu;
      int const u = u_lo + (live ? k : 0), qi = 4 * u / KV, dim = 4 * u % KV;
      float o[4] = {0.f, 0.f, 0.f, 0.f};
      if (live) {
        for (int q = sub; q < act; q += tpu) {
          uint4 const v = slot[q * nu + k];
          float const w = wt[q * NQ + qi];
          o[0] = fmaf(__uint_as_float(v.x), w, o[0]);
          o[1] = fmaf(__uint_as_float(v.y), w, o[1]);
          o[2] = fmaf(__uint_as_float(v.z), w, o[2]);
          o[3] = fmaf(__uint_as_float(v.w), w, o[3]);
        }
      }
      for (int x = 1; x < tpu; x <<= 1) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
          o[e] += __shfl_xor(o[e], x);
        }
      }
      if (live && sub == 0) {
        __hip_bfloat16 *const dst = attn + ((size_t)(qi % T) * HEADS + HG * g + qi / T) * KV + dim;
        mla_g_st8_dev(dst, make_uint2(pack_bf16x2(o[0], o[1]), pack_bf16x2(o[2], o[3])));
      }
    }
    __syncthreads();
  }
}

#undef MLA_GRP_POLL_Q
#undef MLA_GRP_Q_STAMP

template <int KV, int RP, int NS>
constexpr int smem_mla_attn_grp() {
  using namespace mla_g;
  constexpr int NQ = HG * T;
  constexpr int loop = CH * (KV + RP + DPAD) * 2 + KV * CH * 2 + 4 * NQ * CH * 4 + NQ * CH * 2 + CH * 8 + 3 * NQ * 4;
  constexpr int comb = NS * NQ * 4 * 3 + (NQ * KV / 4 + NS) * 16;
  return loop > comb ? loop : comb;
}

} // namespace static_mk
