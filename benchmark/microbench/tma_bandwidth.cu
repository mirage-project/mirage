/* Copyright 2025 CMU
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

// Cost of TMA loads as the megakernel issues them.
//
// A task's loader streams its operands with cp.async.bulk.tensor into a
// shared-memory stage ring and waits on one mbarrier per stage. How long one
// tile takes to land, how many stages an SM needs to reach its bandwidth, and
// how many SMs it takes to saturate HBM decide how tasks should be sized and
// spread -- so they should be measured rather than assumed.
//
// The instruction is copied from tasks/hopper/tma_2d.cuh, which the SM100
// linear task uses, and the tensor map is encoded as tma.cuh's fill_tma_desc
// builds the runtime's descriptors: a 5-D tile-mode load into shared::cluster
// memory from a bf16 tensor map in global memory, 128B swizzle, no L2
// promotion, a box 64 elements (128 B) wide. The mbarrier helpers are
// barrier.cuh's, which issue the same instructions as the CuTe helpers the
// linear task calls. Tiles walk a row-major matrix HIDDEN elements wide along K
// first, the order the linear task's loader uses.

#include "microbench_common.cuh"

#include <cuda.h>

#define CU_CHECK(x)                                                            \
  do {                                                                         \
    CUresult _r = (x);                                                         \
    if (_r != CUDA_SUCCESS) {                                                  \
      char const *_s = nullptr;                                                \
      cuGetErrorString(_r, &_s);                                               \
      printf("CUDA driver error %s at line %d\n", _s ? _s : "?", __LINE__);    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

constexpr int HIDDEN = 7168; // DeepSeek-V3 hidden size
constexpr size_t ROW_BYTES = HIDDEN * 2;
constexpr int BOX_COLS = 64; // 128 B of bf16: the 128B-swizzle span
constexpr int K_CHUNKS = HIDDEN / BOX_COLS;
constexpr int MAX_STAGES = 6;
constexpr int MAX_BOX_ROWS = 256; // TMA box limit per dimension
constexpr int STAGE_BYTES = MAX_BOX_ROWS * BOX_COLS * 2;
// Matrix size, well past L2 so a fresh tile is an HBM read.
constexpr size_t MATRIX_BYTES = 4ull << 30;
constexpr size_t ROWS = MATRIX_BYTES / ROW_BYTES;

// Tile heights measured: 1 KiB to 32 KiB. The linear task's weight tiles
// are 128 rows (16 KiB); decode-sized activation tiles are 8-16 rows.
constexpr int BOX_ROWS[] = {8, 16, 32, 64, 128, 256};
constexpr int NBOX = sizeof(BOX_ROWS) / sizeof(BOX_ROWS[0]);

// ---------------------------------------------------------------------------
// Copied from tasks/hopper/tma_2d.cuh (launch_tma_cp_async) and
// tasks/hopper/barrier.cuh, with the pointer-to-address conversions hoisted
// to the caller.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tma_load_5d(uint32_t smem_int_ptr,
                                            uint64_t gmem_int_desc,
                                            uint32_t smem_int_mbar,
                                            int c0,
                                            int c1) {
  int c2 = 0, c3 = 0, c4 = 0;
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier:"
               ":complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
               :
               : "r"(smem_int_ptr),
                 "l"(gmem_int_desc),
                 "r"(smem_int_mbar),
                 "r"(c0),
                 "r"(c1),
                 "r"(c2),
                 "r"(c3),
                 "r"(c4)
               : "memory");
}

__device__ __forceinline__ void initialize_barrier(uint32_t smem_int_ptr,
                                                   int thread_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr),
               "r"(thread_count));
}

__device__ __forceinline__ void
    set_barrier_transaction_bytes(uint32_t smem_int_ptr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                   smem_int_ptr),
               "r"(bytes));
}

__device__ __forceinline__ void wait_barrier(uint32_t mbar_ptr,
                                             uint32_t phase) {
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(phase));
}

// cutlass::arch::fence_barrier_init(), as the linear task calls it.
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// The copied wait has no memory clobber, so without this the compiler could
// move a plain shared-memory read of the tile above it. Kept out of the copied
// helper so the timed instructions stay as MPK issues them.
__device__ __forceinline__ void compiler_fence() {
  asm volatile("" ::: "memory");
}

// ---------------------------------------------------------------------------
// Data pattern: every element of a tile is distinct, so a misplaced element
// cannot pass the check by coincidence.
// ---------------------------------------------------------------------------
__host__ __device__ inline uint16_t pattern(size_t row, int col) {
  return (uint16_t)(((row & 0xFF) << 8) | ((col & 0x3F) << 2) |
                    ((col >> 6) & 3));
}

__global__ void fill_pattern(uint16_t *m, size_t rows) {
  size_t const n = rows * HIDDEN;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    m[i] = pattern(i / HIDDEN, (int)(i % HIDDEN));
  }
}

// Reads a buffer through L2 (ld.global.cg), keeping the result live.
__global__ void
    read_flush(uint4 const *buf, size_t n, unsigned long long *sink) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldcg(buf + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) {
    *sink = acc;
  }
}

// Tile t of a walk along K: which 128 B column chunk, which row block.
__host__ __device__ inline void
    tile_coords(long long t, int box_rows, int *c0, int *c1) {
  *c0 = (int)(t % K_CHUNKS) * BOX_COLS;
  *c1 = (int)(t / K_CHUNKS) * box_rows;
}

// 128B swizzle needs a 1024-byte aligned destination; align the dynamic
// shared memory base explicitly rather than trusting the declaration.
extern __shared__ __align__(1024) unsigned char dsmem[];

struct Smem {
  uint32_t tiles, bars; // shared-space addresses
  unsigned char *tiles_ptr;
};

__device__ __forceinline__ Smem smem_layout() {
  uint32_t raw = (uint32_t)__cvta_generic_to_shared(dsmem);
  uint32_t tiles = (raw + 1023u) & ~1023u;
  return Smem{tiles, tiles + MAX_STAGES * STAGE_BYTES, dsmem + (tiles - raw)};
}

constexpr int MAX_BARRIERS = 32;
constexpr int SMEM_BYTES = 1024 + MAX_STAGES * STAGE_BYTES + 8 * MAX_BARRIERS;

// The element a tile's first 16 bytes must hold: row 0, chunk 0 is never
// moved by the swizzle.
__device__ __forceinline__ bool
    first_element_ok(unsigned char const *tile, int c0, int c1) {
  return *(uint16_t const *)tile == pattern((size_t)c1, c0);
}

// ---------------------------------------------------------------------------
// Correctness: one tile through the same path, copied out raw so the host can
// check every element against the 128B-swizzle placement.
// ---------------------------------------------------------------------------
__global__ void tma_check(
    CUtensorMap const *tm, int box_rows, int c0, int c1, uint16_t *out) {
  if (threadIdx.x) {
    return;
  }
  Smem s = smem_layout();
  initialize_barrier(s.bars, 1);
  fence_barrier_init();
  set_barrier_transaction_bytes(s.bars, box_rows * BOX_COLS * 2);
  tma_load_5d(s.tiles, (uint64_t)tm, s.bars, c0, c1);
  wait_barrier(s.bars, 0);
  compiler_fence();
  uint16_t const *t = (uint16_t const *)s.tiles_ptr;
  for (int i = 0; i < box_rows * BOX_COLS; i++) {
    out[i] = t[i];
  }
}

// ---------------------------------------------------------------------------
// Latency: one tile in flight, from issuing the load to the mbarrier wait
// returning -- what a loader waiting on its first stage sees. Every SM in
// turn. From HBM each sample is a tile nobody has read since the last L2
// flush; from L2 it is the same tile, loaded once untimed first.
// ---------------------------------------------------------------------------
__global__ void tma_latency(CUtensorMap const *tm,
                            int box_rows,
                            int samples,
                            int from_l2,
                            unsigned long long *ticket,
                            double *out,
                            int *smid_out,
                            int *errors) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  Smem s = smem_layout();
  initialize_barrier(s.bars, 1);
  fence_barrier_init();
  uint32_t const bytes = box_rows * BOX_COLS * 2;
  long long const first = (long long)blockIdx.x * (samples + 1);
  uint32_t phase = 0;
  int c0, c1;
  if (from_l2) {
    tile_coords(first, box_rows, &c0, &c1);
    set_barrier_transaction_bytes(s.bars, bytes);
    tma_load_5d(s.tiles, (uint64_t)tm, s.bars, c0, c1);
    wait_barrier(s.bars, phase);
    phase ^= 1;
  }
  for (int j = 0; j < samples; j++) {
    tile_coords(from_l2 ? first : first + 1 + j, box_rows, &c0, &c1);
    long long t0 = clock64();
    set_barrier_transaction_bytes(s.bars, bytes);
    tma_load_5d(s.tiles, (uint64_t)tm, s.bars, c0, c1);
    wait_barrier(s.bars, phase);
    long long t1 = clock64();
    compiler_fence();
    phase ^= 1;
    out[blockIdx.x * samples + j] = (double)(t1 - t0);
    if (!first_element_ok(s.tiles_ptr, c0, c1)) {
      atomicAdd(errors, 1);
    }
  }
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// ---------------------------------------------------------------------------
// Streaming through a ring of `stages` buffers, the loader's steady state:
// wait for a stage to land, reissue it for the tile `stages` ahead. Stage
// index, phase and coordinates advance incrementally, as in the linear task's
// loader; computing them by division on a runtime stage count costs enough
// per iteration to show up in the result. Nothing consumes the data, so this
// is the rate TMA can deliver, an upper bound for a real loader.
//
// `warps` loader warps each stream their own ring over their own tiles, one
// elected lane issuing, like the linear task's TMA warp. The block's window
// runs from a __syncthreads before any warp starts to one after all finish.
//
//   TURNS=true   blocks take turns: one SM's own throughput, every SM.
//   TURNS=false  all blocks at once after an on-device rendezvous: the
//                aggregate. The window uses %globaltimer, which all SMs
//                share, so it spans the whole machine.
// ---------------------------------------------------------------------------
template <bool TURNS>
__global__ void tma_stream(CUtensorMap const *tm,
                           int box_rows,
                           int stages,
                           int warps,
                           long long ntiles,
                           unsigned long long *sync,
                           long long *cycles_out,
                           unsigned long long *window_out,
                           int *smid_out,
                           int *errors) {
  int const w = threadIdx.x / 32;
  bool const leader = threadIdx.x % 32 == 0 && w < warps;
  Smem s = smem_layout();
  uint32_t const bytes = box_rows * BOX_COLS * 2;
  uint32_t const my_tiles = s.tiles + w * stages * bytes;
  uint32_t const my_bars = s.bars + 8 * w * stages;
  long long const first = ((long long)blockIdx.x * warps + w) * ntiles;
  if (leader) {
    for (int i = 0; i < stages; i++) {
      initialize_barrier(my_bars + 8 * i, 1);
    }
    fence_barrier_init();
  }
  if (threadIdx.x == 0) {
    if (TURNS) {
      wait_turn(sync);
    } else {
      atom_add_release_gpu_u64(sync, 1ull);
      while (ld_acquire_gpu_u64(sync) < (unsigned long long)gridDim.x) {
      }
    }
  }
  __syncthreads();
  unsigned long long g0 = gtimer_ns();
  long long t0 = clock64();
  int end_c0 = 0, end_c1 = 0;
  if (leader) {
    int c0, c1, issue_st = 0, wait_st = 0;
    uint32_t phase = 0;
    tile_coords(first, box_rows, &c0, &c1);
    auto issue = [&]() {
      uint32_t bar = my_bars + 8 * issue_st;
      set_barrier_transaction_bytes(bar, bytes);
      tma_load_5d(my_tiles + issue_st * bytes, (uint64_t)tm, bar, c0, c1);
      c0 += BOX_COLS;
      if (c0 == HIDDEN) {
        c0 = 0;
        c1 += box_rows;
      }
      if (++issue_st == stages) {
        issue_st = 0;
      }
    };
    long long issued = 0;
    for (; issued < stages && issued < ntiles; issued++) {
      issue();
    }
    for (long long i = 0; i < ntiles; i++) {
      wait_barrier(my_bars + 8 * wait_st, phase);
      if (++wait_st == stages) {
        wait_st = 0;
        phase ^= 1;
      }
      if (issued < ntiles) {
        issue();
        issued++;
      }
    }
    end_c0 = c0;
    end_c1 = c1;
  }
  __syncthreads();
  long long t1 = clock64();
  unsigned long long g1 = gtimer_ns();
  if (leader) {
    // The coordinates advanced incrementally; they must end one tile past
    // the last, where the division-based mapping puts it, or some tile was
    // loaded from the wrong place.
    int want_c0, want_c1;
    tile_coords(first + ntiles, box_rows, &want_c0, &want_c1);
    if (end_c0 != want_c0 || end_c1 != want_c1) {
      atomicAdd(errors, 1);
    }
    // Spot-check the last tile landed in each stage.
    for (long long i = ntiles - 1; i >= 0 && i >= ntiles - stages; i--) {
      int c0, c1;
      tile_coords(first + i, box_rows, &c0, &c1);
      if (!first_element_ok(s.tiles_ptr + (my_tiles - s.tiles) +
                                (i % stages) * bytes,
                            c0,
                            c1)) {
        atomicAdd(errors, 1);
      }
    }
  }
  if (threadIdx.x == 0) {
    cycles_out[blockIdx.x] = t1 - t0;
    window_out[2 * blockIdx.x] = g0;
    window_out[2 * blockIdx.x + 1] = g1;
    smid_out[blockIdx.x] = sm_id();
    if (TURNS) {
      pass_turn(sync);
    }
  }
}

// ---------------------------------------------------------------------------

static CUtensorMap *make_tensor_map(void *gmem, int box_rows) {
  // Same encoding as fill_tma_desc<bfloat16, B = 3, ..., 2> in tma.cuh.
  uint64_t shape[5] = {(uint64_t)HIDDEN, (uint64_t)ROWS, 1, 1, 1};
  uint64_t stride[5] = {2, ROW_BYTES, 0, 0, 0};
  uint32_t box[5] = {BOX_COLS, (uint32_t)box_rows, 1, 1, 1};
  uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
  CUtensorMap host;
  CU_CHECK(cuTensorMapEncodeTiled(&host,
                                  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                  5,
                                  gmem,
                                  shape,
                                  stride + 1,
                                  box,
                                  elem_stride,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CUtensorMap *dev;
  CHECK(cudaMalloc(&dev, sizeof(CUtensorMap)));
  CHECK(cudaMemcpy(dev, &host, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
  return dev;
}

static void require_no_errors(int *d_errors, char const *what) {
  int e = 0;
  CHECK(cudaMemcpy(&e, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
  if (e) {
    printf("FAIL: %d tiles held the wrong data (%s)\n", e, what);
    exit(1);
  }
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  char const *json_path = (argc > 1) ? argv[1] : nullptr;

  Guard guard;
  guard.start = probe_gpu_sharing();

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver = 0;
  CHECK(cudaDriverGetVersion(&driver));
  int const nsm = prop.multiProcessorCount;
  require_sm_count_fits(nsm);
  int const smem = std::max(one_block_per_sm_smem(), SMEM_BYTES);
  double const ghz = measure_sm_clock_ghz();
  guard.ghz_start = ghz;
  double const peak = hbm_peak_gbps();
  guard.spread_start = alu_sentinel_spread(nsm, one_block_per_sm_smem());

  printf("device       : %s (SM %d.%d, %d SMs)\n",
         prop.name,
         prop.major,
         prop.minor,
         nsm);
  printf("driver       : %d\n", driver);
  printf("SM clock     : %.3f GHz (measured)\n", ghz);
  printf("L2           : %.0f MiB\n", prop.l2CacheSize / 1048576.0);
  printf("HBM peak     : %.0f GB/s (memory clock x bus width)\n\n", peak);

  uint16_t *d_matrix;
  unsigned char *d_flush;
  size_t const flush_bytes = 2 * (size_t)prop.l2CacheSize;
  CHECK(cudaMalloc(&d_matrix, ROWS * ROW_BYTES));
  CHECK(cudaMalloc(&d_flush, flush_bytes));
  CHECK(cudaMemset(d_flush, 0x5a, flush_bytes));
  unsigned long long *d_flush_sink;
  CHECK(cudaMalloc(&d_flush_sink, sizeof(unsigned long long)));
  fill_pattern<<<4 * nsm, 256>>>(d_matrix, ROWS);
  CHECK_LAUNCH();
  // Reading twice L2's size evicts whatever a previous pass left, and leaves
  // L2 holding clean lines, so the timed reads cause no write-backs.
  auto flush_l2 = [&]() {
    read_flush<<<4 * nsm, 256>>>(
        (uint4 const *)d_flush, flush_bytes / sizeof(uint4), d_flush_sink);
    CHECK_LAUNCH();
  };

  CUtensorMap *tm[NBOX];
  for (int b = 0; b < NBOX; b++) {
    tm[b] = make_tensor_map(d_matrix, BOX_ROWS[b]);
  }

  unsigned long long *d_sync, *d_window;
  long long *d_cycles;
  double *d_out;
  int *d_smid, *d_errors;
  uint16_t *d_tile;
  int const LAT_SAMPLES = 16;
  CHECK(cudaMalloc(&d_sync, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_window, 2 * MAX_SMS * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_cycles, MAX_SMS * sizeof(long long)));
  CHECK(cudaMalloc(&d_out, MAX_SMS * LAT_SAMPLES * sizeof(double)));
  CHECK(cudaMalloc(&d_smid, MAX_SMS * sizeof(int)));
  CHECK(cudaMalloc(&d_errors, sizeof(int)));
  CHECK(cudaMalloc(&d_tile, STAGE_BYTES));
  CHECK(cudaMemset(d_errors, 0, sizeof(int)));
  allow_dynamic_smem(tma_check, smem);
  allow_dynamic_smem(tma_latency, smem);
  allow_dynamic_smem(tma_stream<true>, smem);
  allow_dynamic_smem(tma_stream<false>, smem);

  Json js;
  js.open(json_path);
  js.kvs("device", prop.name);
  js.kv("sm_count", nsm);
  js.kv("driver_version", driver);
  js.kv("sm_clock_ghz", ghz);
  js.kv("hbm_peak_gbps", peak);

  // --- correctness: every element where the 128B swizzle puts it ---
  {
    int checked = 0;
    std::vector<uint16_t> got(MAX_BOX_ROWS * BOX_COLS);
    for (int b = 0; b < NBOX; b++) {
      int const r = BOX_ROWS[b];
      // an aligned tile, and one at an odd row deep in the matrix
      int const starts[2][2] = {{0, 0}, {5 * BOX_COLS, (int)(ROWS / 2) + 7}};
      for (auto const &st : starts) {
        tma_check<<<1, 32, smem>>>(tm[b], r, st[0], st[1], d_tile);
        CHECK_LAUNCH();
        CHECK(cudaMemcpy(got.data(),
                         d_tile,
                         r * BOX_COLS * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost));
        for (int row = 0; row < r; row++) {
          for (int col = 0; col < BOX_COLS; col++) {
            int chunk = col / 8, phys_chunk = chunk ^ (row & 7);
            uint16_t v = got[row * BOX_COLS + phys_chunk * 8 + col % 8];
            uint16_t want = pattern((size_t)st[1] + row, st[0] + col);
            if (v != want) {
              printf("FAIL: %d-row tile at (%d, %d): element (%d, %d) is "
                     "0x%04x, expected 0x%04x\n",
                     r,
                     st[0],
                     st[1],
                     row,
                     col,
                     v,
                     want);
              exit(1);
            }
          }
        }
        checked++;
      }
    }
    printf("data check: %d tiles, every element in its 128B-swizzled place\n\n",
           checked);
  }

  auto tile_label = [](char *buf, size_t n, int r) {
    snprintf(buf, n, "%3d rows (%2d KiB)", r, r * BOX_COLS * 2 / 1024);
  };

  // --- latency, one tile in flight ---
  printf("TMA load latency, one tile in flight, every SM in turn (ns)\n");
  printf("  %-24s %-6s %8s %8s %8s   %-18s %s\n",
         "tile",
         "from",
         "median",
         "p10",
         "p90",
         "[min .. max]",
         "samples");
  auto print_row = [&](char const *tile,
                       char const *from,
                       std::vector<double> const &v,
                       char const *key) {
    Dist d = distribution(v);
    printf("  %-24s %-6s %8.1f %8.1f %8.1f   [%6.1f .. %6.1f]   %zu\n",
           tile,
           from,
           d.med,
           d.p10,
           d.p90,
           d.lo,
           d.hi,
           v.size());
    js.dist(key, d);
  };
  for (int b = 0; b < NBOX; b++) {
    for (int from_l2 = 0; from_l2 < 2; from_l2++) {
      if (!from_l2) {
        flush_l2();
      }
      CHECK(cudaMemset(d_sync, 0, sizeof(unsigned long long)));
      tma_latency<<<nsm, 32, smem>>>(tm[b],
                                     BOX_ROWS[b],
                                     LAT_SAMPLES,
                                     from_l2,
                                     d_sync,
                                     d_out,
                                     d_smid,
                                     d_errors);
      CHECK_LAUNCH();
      require_no_errors(d_errors, "latency");
      std::vector<int> sm(nsm);
      CHECK(cudaMemcpy(
          sm.data(), d_smid, nsm * sizeof(int), cudaMemcpyDeviceToHost));
      require_every_sm_once(sm, nsm);
      std::vector<double> v(nsm * LAT_SAMPLES);
      CHECK(cudaMemcpy(
          v.data(), d_out, v.size() * sizeof(double), cudaMemcpyDeviceToHost));
      for (double &x : v) {
        x /= ghz;
      }
      char tile[48], key[64];
      tile_label(tile, sizeof(tile), BOX_ROWS[b]);
      char const *from = from_l2 ? "L2" : "HBM";
      double cut;
      if (find_split(v, &cut)) {
        std::vector<double> near, far;
        for (double x : v) {
          (x < cut ? near : far).push_back(x);
        }
        snprintf(
            key, sizeof(key), "latency_%s_%d_rows_near_ns", from, BOX_ROWS[b]);
        char t2[64];
        snprintf(t2, sizeof(t2), "%s near", tile);
        print_row(t2, from, near, key);
        snprintf(
            key, sizeof(key), "latency_%s_%d_rows_far_ns", from, BOX_ROWS[b]);
        snprintf(t2, sizeof(t2), "%s far", tile);
        print_row(t2, from, far, key);
      } else {
        snprintf(key, sizeof(key), "latency_%s_%d_rows_ns", from, BOX_ROWS[b]);
        print_row(tile, from, v, key);
      }
    }
  }

  // --- one SM's throughput: pipeline depth, then loader warps ---
  // Every SM in turn; the result is GB/s per SM. At least 1 MiB per loader,
  // so ramp-up is a small part of the window.
  auto per_sm_gbps = [&](int b, int stages, int warps) {
    int const r = BOX_ROWS[b];
    long long const bytes = (long long)r * BOX_COLS * 2;
    long long const ntiles = std::max(64ll, (1ll << 20) / bytes);
    if (warps * stages * bytes > (long long)MAX_STAGES * STAGE_BYTES ||
        warps * stages > MAX_BARRIERS) {
      printf("FAIL: %d warps x %d stages of %lld B do not fit\n",
             warps,
             stages,
             bytes);
      exit(1);
    }
    flush_l2();
    CHECK(cudaMemset(d_sync, 0, sizeof(unsigned long long)));
    tma_stream<true><<<nsm, 32 * warps, smem>>>(tm[b],
                                                r,
                                                stages,
                                                warps,
                                                ntiles,
                                                d_sync,
                                                d_cycles,
                                                d_window,
                                                d_smid,
                                                d_errors);
    CHECK_LAUNCH();
    require_no_errors(d_errors, "per-SM stream");
    std::vector<int> sm(nsm);
    std::vector<long long> cyc(nsm);
    CHECK(cudaMemcpy(
        sm.data(), d_smid, nsm * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        cyc.data(), d_cycles, nsm * sizeof(long long), cudaMemcpyDeviceToHost));
    require_every_sm_once(sm, nsm);
    std::vector<double> gbps(nsm);
    for (int i = 0; i < nsm; i++) {
      gbps[i] = (double)(warps * ntiles * bytes) / (cyc[i] / ghz); // B/ns
    }
    return distribution(gbps);
  };

  int const STAGES[] = {1, 2, 4, 6};
  printf("\nper-SM TMA throughput from HBM, one loader warp, one SM at a time, "
         "every SM\n(GB/s: median [p10 .. p90])\n");
  printf("  %-24s", "tile \\ stages");
  for (int st : STAGES) {
    printf("  %20d", st);
  }
  printf("\n");
  for (int b = 0; b < NBOX; b++) {
    int const r = BOX_ROWS[b];
    char tile[48];
    tile_label(tile, sizeof(tile), r);
    printf("  %-24s", tile);
    for (int st : STAGES) {
      Dist d = per_sm_gbps(b, st, 1);
      printf("  %6.1f [%5.1f .. %5.1f]", d.med, d.p10, d.p90);
      char key[64];
      snprintf(key, sizeof(key), "per_sm_gbps_%d_rows_%d_stages", r, st);
      js.dist(key, d);
    }
    printf("\n");
  }

  // If throughput is set by how many tiles are in flight, it should not
  // matter whether they come from one loader warp's stages or several warps'.
  // Several loader warps, each with its own ring, check that: a limit in one
  // warp's wait-and-reissue loop would make more warps beat more stages.
  int const WARPS[] = {1, 2, 4};
  int const LOADER_STAGES = 4;
  printf("\nper-SM TMA throughput from HBM vs loader warps, %d stages each "
         "(GB/s: median [p10 .. p90])\n",
         LOADER_STAGES);
  printf("  %-24s", "tile \\ loader warps");
  for (int w : WARPS) {
    printf("  %20d", w);
  }
  printf("\n");
  for (int r : {16, 64}) {
    int const b = (int)(std::find(BOX_ROWS, BOX_ROWS + NBOX, r) - BOX_ROWS);
    char tile[48];
    tile_label(tile, sizeof(tile), r);
    printf("  %-24s", tile);
    for (int w : WARPS) {
      Dist d = per_sm_gbps(b, LOADER_STAGES, w);
      printf("  %6.1f [%5.1f .. %5.1f]", d.med, d.p10, d.p90);
      char key[64];
      snprintf(key, sizeof(key), "per_sm_gbps_%d_rows_%d_warps", r, w);
      js.dist(key, d);
    }
    printf("\n");
  }

  // --- aggregate bandwidth vs SM count ---
  struct AggConfig {
    int box_rows, stages, warps;
  };
  AggConfig const configs[] = {
      {128, 4, 1}, {256, 6, 1}, {16, 6, 1}, {16, 4, 4}};
  std::vector<int> const counts = sm_counts(nsm);
  for (AggConfig const &cfg : configs) {
    int const b =
        (int)(std::find(BOX_ROWS, BOX_ROWS + NBOX, cfg.box_rows) - BOX_ROWS);
    long long const bytes = (long long)cfg.box_rows * BOX_COLS * 2;
    long long const total_tiles = (long long)(ROWS / cfg.box_rows) * K_CHUNKS;
    printf("\naggregate TMA bandwidth from HBM, %d KiB tiles, %d loader "
           "warp%s x %d stages per SM,\neach SM streaming its own slice of "
           "%.1f GiB\n",
           (int)(bytes / 1024),
           cfg.warps,
           cfg.warps > 1 ? "s" : "",
           cfg.stages,
           total_tiles * bytes / 1073741824.0);
    printf("  %-6s %12s   %-20s %12s %14s\n",
           "SMs",
           "GB/s",
           "[min .. max]",
           "% of peak",
           "event GB/s");
    for (int n : counts) {
      long long const per_loader = total_tiles / (n * cfg.warps);
      double const moved = (double)per_loader * n * cfg.warps * bytes;
      double v[REPS], ev[REPS];
      for (int rep = 0; rep < REPS; rep++) {
        flush_l2();
        CHECK(cudaMemset(d_sync, 0, sizeof(unsigned long long)));
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        CHECK(cudaEventRecord(e0));
        tma_stream<false><<<n, 32 * cfg.warps, smem>>>(tm[b],
                                                       cfg.box_rows,
                                                       cfg.stages,
                                                       cfg.warps,
                                                       per_loader,
                                                       d_sync,
                                                       d_cycles,
                                                       d_window,
                                                       d_smid,
                                                       d_errors);
        CHECK(cudaEventRecord(e1));
        CHECK_LAUNCH();
        require_no_errors(d_errors, "aggregate stream");
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CHECK(cudaEventDestroy(e0));
        CHECK(cudaEventDestroy(e1));
        std::vector<unsigned long long> w(2 * n);
        std::vector<int> sm(n);
        CHECK(cudaMemcpy(w.data(),
                         d_window,
                         w.size() * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(
            sm.data(), d_smid, n * sizeof(int), cudaMemcpyDeviceToHost));
        require_every_sm_once(sm, n);
        unsigned long long start = ~0ull, end = 0;
        for (int i = 0; i < n; i++) {
          start = std::min(start, w[2 * i]);
          end = std::max(end, w[2 * i + 1]);
        }
        v[rep] = moved / (double)(end - start); // bytes/ns = GB/s
        ev[rep] = moved / (ms * 1e6);
        // The event window contains the device window, so it can only be
        // slower (1% allows for two clocks); and nothing read from HBM can
        // beat the HBM peak.
        if (ev[rep] > v[rep] * 1.01) {
          printf("FAIL: event-timed %.1f GB/s exceeds device-timed %.1f GB/s\n",
                 ev[rep],
                 v[rep]);
          exit(1);
        }
        if (peak > 0 && v[rep] > peak * 1.02) {
          printf("FAIL: %.1f GB/s exceeds the HBM peak %.1f GB/s -- data "
                 "was not coming from HBM\n",
                 v[rep],
                 peak);
          exit(1);
        }
      }
      Stat s = summarize(v, REPS), se = summarize(ev, REPS);
      printf("  %-6d %12.1f   [%7.1f .. %7.1f] %11.1f%% %14.1f\n",
             n,
             s.med,
             s.lo,
             s.hi,
             peak > 0 ? 100.0 * s.med / peak : 0.0,
             se.med);
      char key[80];
      snprintf(key,
               sizeof(key),
               "aggregate_gbps_%d_rows_%d_stages_%d_warps_%d_sms",
               cfg.box_rows,
               cfg.stages,
               cfg.warps,
               n);
      js.kv(key, s.med);
    }
  }

  guard.end = probe_gpu_sharing();
  guard.spread_end = alu_sentinel_spread(nsm, one_block_per_sm_smem());
  guard.ghz_end = measure_sm_clock_ghz();
  printf("\n");
  guard.print();
  js.kv("other_gpu_processes",
        std::max(guard.start.other_procs, guard.end.other_procs));
  js.kv("alu_sentinel_spread", std::max(guard.spread_start, guard.spread_end));
  js.kv("provisional", guard.provisional() ? 1 : 0);
  js.close();
  if (json_path) {
    printf("wrote %s\n", json_path);
  }
  return 0;
}
