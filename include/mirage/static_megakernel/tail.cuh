// tail.cuh -- the layer after the expert queue, on every SM:
//   1. wait until every SM is done with the expert queue (counter C_EXPDONE)
//   2. send this SM's slice of [R | S] (this GPU's routed and shared sums, fp32) to every GPU: one bulk copy per GPU
//   3. SMs 0..31: add R over the GPUs for (token, 896-column quarter), RMSNorm -> Rn;
//      SMs 32..95: add S over the GPUs for (token, 896-column eighth) -> Ssum
//   4. SMs 0..97: one latent_up tile each (7 row tiles x 14 K parts of 2 K tiles of this GPU's 896 output rows): the weights are
//      loaded into the idle ring while R lands, Rn is polled straight into the ring's activation slots; the K parts go to opart;
//      each K-part SM adds the 14 parts of 1/14 of its tile and sends them to every GPU
//   5. every SM: y = bf16(bf16(o + S) + prefix) for its slice of y
// After 1 there are no counters: every hand-off is a 0xFF-prefilled buffer polled by its reader.
#pragma once
#include "runtime.cuh"
#include "moe_types.cuh"
#include "gemm_tile.cuh"

namespace static_mk {

constexpr int TAIL_COLS = 896;       // columns per landing SM: a quarter of R, an eighth of S
constexpr int UP_KPARTS = 14;        // latent_up K parts (2 K tiles each)
static_assert(LAT / 4 == TAIL_COLS && H / 8 == TAIL_COLS && KT_LAT == 2 * UP_KPARTS && N_UPTILE == 7 * UP_KPARTS, "tail layout");

__device__ __forceinline__ void tail_task(G const &g, Maps const &maps, uint32_t base, uint32_t tb, char *sm, int &gl, int &gi0, int &gi1,
                                          int &wt, int sm_id) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // 1. every SM's red.adds into Racc and Sout are done
  if (threadIdx.x == 0) { cnt_add(g.cnt + C_EXPDONE, 1); cnt_wait(g.cnt + C_EXPDONE, NSM); }
  __syncthreads();
  {  // 2. this SM's slice of [R | S] (T x (LAT + H) fp32 = 21504 16-B vectors per GPU) -> shared memory (the idle ring), then
     //    one bulk copy per GPU, one GPU per thread (the copies stream at the same time)
    int const nv = T * (LAT + H) / 4, lo = (int)((long long)nv * sm_id / NSM), hi = (int)((long long)nv * (sm_id + 1) / NSM);
    int const per_t = (LAT + H) / 4;
    float4 *stg = reinterpret_cast<float4 *>(sm + OFF_W);
    for (int i = lo + threadIdx.x; i < hi; i += 256) {
      int const t = i / per_t, c4 = i - t * per_t;
      stg[i - lo] = (c4 < LAT / 4) ? reinterpret_cast<float4 const *>(g.Racc)[t * (LAT / 4) + c4]
                                   : reinterpret_cast<float4 const *>(g.Sout)[t * (H / 4) + (c4 - LAT / 4)];
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x < g.tp && hi > lo) {
      uint32_t const src = su32(stg), bytes = (uint32_t)(hi - lo) * 16u;
      int const r = threadIdx.x;
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                   :: "l"(g.rs_all[r] + (size_t)g.rank * RS_RANK + (size_t)lo * 16), "r"(src), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    __syncthreads();
  }
  // 3. add [R | S] over the GPUs
  if (sm_id < N_RSM + N_SSM) {
    bool const isR = sm_id < N_RSM;
    int const t = isR ? sm_id / 4 : (sm_id - N_RSM) / 8, q = isR ? sm_id % 4 : (sm_id - N_RSM) % 8;
    unsigned char const *rs = g.rs_all[g.rank] + (size_t)t * (LAT + H) * 4 + (isR ? 0 : (size_t)LAT * 4) + (size_t)q * TAIL_COLS * 4;
    int const v = threadIdx.x;   // thread v < 224: columns 4 v .. 4 v + 3 of this SM's 896
    float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
    if (v < TAIL_COLS / 4) {
      uint4 u[TPMAX]; bool ok[TPMAX]; bool all;
#pragma unroll
      for (int r = 0; r < TPMAX; r++) ok[r] = r >= g.tp;
      do {
        all = true;
#pragma unroll
        for (int r = 0; r < TPMAX; r++) if (!ok[r]) u[r] = ld16_poll(rs + (size_t)r * RS_RANK + (size_t)v * 16);
#pragma unroll
        for (int r = 0; r < TPMAX; r++) { if (!ok[r]) ok[r] = valid16(u[r]); all = all && ok[r]; }
      } while (!all);
      for (int r = 0; r < g.tp; r++) {
        a4.x += __uint_as_float(u[r].x); a4.y += __uint_as_float(u[r].y); a4.z += __uint_as_float(u[r].z); a4.w += __uint_as_float(u[r].w);
      }
    }
    if (isR) {   // RMSNorm of token t over LAT: the 4 quarter SMs each write a sum of squares into ss_part[t][q] and poll all 4
      float ss = (v < TAIL_COLS / 4) ? a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w : 0.f;
      for (int d = 16; d > 0; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
      __shared__ float red_s[8];
      if (lane == 0) red_s[warp] = ss;
      __syncthreads();
      if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int w_ = 0; w_ < 8; w_++) tot += red_s[w_];
        g.ss_part[t * 4 + q] = tot;
        float4 q4;
        do { q4 = ldf4_relaxed(g.ss_part + t * 4); if (!validf4(q4)) __nanosleep(64); } while (!validf4(q4));
        red_s[0] = rsqrtf((q4.x + q4.y + q4.z + q4.w) / LAT + g.eps);
      }
      __syncthreads();
      float const rstd = red_s[0];
      if (v < TAIL_COLS / 4) {   // Rn: two threads' 4 columns = one 16-B vector, written by the even thread
        int const c = q * TAIL_COLS + v * 4;
        float const gm[4] = {__bfloat162float(g.gamma[c]), __bfloat162float(g.gamma[c + 1]), __bfloat162float(g.gamma[c + 2]),
                             __bfloat162float(g.gamma[c + 3])};
        __nv_bfloat162 const a = __floats2bfloat162_rn(a4.x * rstd * gm[0], a4.y * rstd * gm[1]);
        __nv_bfloat162 const b = __floats2bfloat162_rn(a4.z * rstd * gm[2], a4.w * rstd * gm[3]);
        uint32_t const wa = *reinterpret_cast<uint32_t const *>(&a), wb = *reinterpret_cast<uint32_t const *>(&b);
        uint32_t const na = __shfl_down_sync(0xffffffffu, wa, 1), nb = __shfl_down_sync(0xffffffffu, wb, 1);
        if ((v & 1) == 0) *reinterpret_cast<uint4 *>(g.Rn + (size_t)t * LAT + c) = make_uint4(wa, wb, na, nb);
      }
    } else if (v < TAIL_COLS / 4) {
      int const c = q * TAIL_COLS + v * 4;
      __nv_bfloat162 const a = __floats2bfloat162_rn(a4.x, a4.y), b = __floats2bfloat162_rn(a4.z, a4.w);
      *reinterpret_cast<__nv_bfloat162 *>(g.Ssum + (size_t)t * H + c) = a;
      *reinterpret_cast<__nv_bfloat162 *>(g.Ssum + (size_t)t * H + c + 2) = b;
    }
  }
  // 4. latent_up: rows [rank * 896, +896) of W_up (this GPU's part of y), tile (m, ks) on SM 14 m + ks
  if (threadIdx.x == 0) ring_reinit(true);
  gl = gi0 = gi1 = 0; wt = 0;
  __syncthreads();
  if (sm_id < N_UPTILE) {
    int const m = sm_id / UP_KPARTS, ks = sm_id % UP_KPARTS, c0 = ks * 256;
    TileJob j = make_bf16_tile(&maps.wup, nullptr, g.rank * (H / TPMAX) + m * 128, ks * 2, 2);
    if (warp == 0 && lane == 0) load_job(j, base, gl);   // the two weight boxes load now, while R is still landing
    {  // thread v: token v / 32, columns c0 + (v % 32) * 8 .. + 8 = one 16-B vector of Rn (written whole by an R SM) -> the stage
      int const t = threadIdx.x >> 5, cg = threadIdx.x & 31, c = c0 + cg * 8;
      uint4 out;
      do { out = ld16_poll(reinterpret_cast<unsigned char const *>(g.Rn + (size_t)t * LAT + c)); } while (!valid16(out));
      // stage s_ (K tile ks * 2 + s_), 64-column half a_, 16-B chunk j_ of row t; 128-B swizzle: chunk j_ ^ (t % 8)
      int const s_ = cg >> 4, a_ = (cg >> 3) & 1, j_ = cg & 7;
      *reinterpret_cast<uint4 *>(sm + s_ * FSTAGE + W_STAGE + a_ * 1024 + t * 128 + ((j_ ^ (t & 7)) << 4)) = out;
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // these stores -> visible to the tensor core
    }
    __syncthreads();
    if (warp == 1 || warp == 6) {
      if (lane == 0) { int &gi = (warp == 1) ? gi0 : gi1; issue_job(j, base, tb, warp == 1 ? 0 : 1, gi, 0, 0); }
    } else if (warp >= 2 && warp <= 5) {
      float v[8];
      int const row = drain_acc(tb, 0, 0, v);
      for (int t = 0; t < T; t++) g.opart[((size_t)ks * T + t) * (H / TPMAX) + m * 128 + row] = v[t];   // this K part
      // every K-part SM adds the 14 parts of ITS 1/14 of the tile's 128 16-B vectors (token t, 8 rows seg) and sends them
      int const i = threadIdx.x - 64, t = i >> 4, seg = i & 15, v0 = ks * 128 / UP_KPARTS, v1 = (ks + 1) * 128 / UP_KPARTS;
      if (i >= v0 && i < v1) {
        float o8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float4 x0[UP_KPARTS], x1[UP_KPARTS]; bool ok[UP_KPARTS]; bool all;   // all 28 loads in flight at once
#pragma unroll
        for (int p_ = 0; p_ < UP_KPARTS; p_++) ok[p_] = false;
        do {
          all = true;
#pragma unroll
          for (int p_ = 0; p_ < UP_KPARTS; p_++)
            if (!ok[p_]) {
              float const *src = g.opart + ((size_t)p_ * T + t) * (H / TPMAX) + m * 128 + seg * 8;
              x0[p_] = ldf4_poll(src); x1[p_] = ldf4_poll(src + 4);
            }
#pragma unroll
          for (int p_ = 0; p_ < UP_KPARTS; p_++) { if (!ok[p_]) ok[p_] = validf4(x0[p_]) && validf4(x1[p_]); all = all && ok[p_]; }
          if (!all) __nanosleep(64);
        } while (!all);
#pragma unroll
        for (int p_ = 0; p_ < UP_KPARTS; p_++) {   // fixed order
          o8[0] += x0[p_].x; o8[1] += x0[p_].y; o8[2] += x0[p_].z; o8[3] += x0[p_].w;
          o8[4] += x1[p_].x; o8[5] += x1[p_].y; o8[6] += x1[p_].z; o8[7] += x1[p_].w;
        }
        __nv_bfloat162 const b0 = __floats2bfloat162_rn(o8[0], o8[1]), b1 = __floats2bfloat162_rn(o8[2], o8[3]);
        __nv_bfloat162 const b2 = __floats2bfloat162_rn(o8[4], o8[5]), b3 = __floats2bfloat162_rn(o8[6], o8[7]);
        push16(g, RG_O + (size_t)g.rank * O_RANK + (size_t)t * (H / TPMAX) * 2 + (size_t)(m * 128 + seg * 8) * 2,
               make_uint4(*reinterpret_cast<uint32_t const *>(&b0), *reinterpret_cast<uint32_t const *>(&b1),
                          *reinterpret_cast<uint32_t const *>(&b2), *reinterpret_cast<uint32_t const *>(&b3)));
      }
    }
  }
  __syncthreads();
  {  // 5. y = bf16(bf16(o + S) + prefix) for this SM's slice: o from the GPU that owns the columns, S from Ssum (both polled)
    int const nv = T * H / 8, lo = (int)((long long)nv * sm_id / NSM), hi = (int)((long long)nv * (sm_id + 1) / NSM);
    for (int i = lo + threadIdx.x; i < hi; i += 256) {
      int const t = i / (H / 8), c = (i - t * (H / 8)) * 8, r = c / (H / TPMAX), cc = c - r * (H / TPMAX);
      uint4 const ou = (r < g.tp) ? poll16(g.rv + RG_O + (size_t)r * O_RANK + (size_t)t * (H / TPMAX) * 2 + cc * 2)
                                  : make_uint4(0u, 0u, 0u, 0u);   // fewer than 8 GPUs (tests): the absent GPUs' columns are 0
      uint4 su;
      do {
        su = ld16_relaxed(reinterpret_cast<unsigned char const *>(g.Ssum + (size_t)t * H + c));
        if (!valid16w(su)) __nanosleep(64);
      } while (!valid16w(su));
      uint4 const pu = *reinterpret_cast<uint4 const *>(g.prefix + (size_t)t * H + c);
      __nv_bfloat16 const *ob = reinterpret_cast<__nv_bfloat16 const *>(&ou), *sb = reinterpret_cast<__nv_bfloat16 const *>(&su);
      __nv_bfloat16 const *pb = reinterpret_cast<__nv_bfloat16 const *>(&pu);
      uint4 yu; __nv_bfloat16 *yb = reinterpret_cast<__nv_bfloat16 *>(&yu);
      for (int k = 0; k < 8; k++) {
        float const a = __bfloat162float(__float2bfloat16(__bfloat162float(ob[k]) + __bfloat162float(sb[k])));
        yb[k] = __float2bfloat16(a + __bfloat162float(pb[k]));
      }
      *reinterpret_cast<uint4 *>(g.y + (size_t)t * H + c) = yu;
    }
  }
  __syncthreads();
}

}  // namespace static_mk
