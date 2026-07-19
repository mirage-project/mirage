#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace kernel {
namespace simple_linear {

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int TILE_K = 64;
static constexpr int MMA_K = 16;
static constexpr int NUM_STAGES = 4;
static constexpr int TILE_BYTES_A = TILE_M * TILE_K * 2;
static constexpr int TILE_BYTES_B = TILE_N * TILE_K * 2;

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t p = 0;
  asm volatile("{\n\t.reg .pred %%px;\n\telect.sync _|%%px, "
               "0xFFFFFFFF;\n\t@%%px mov.s32 %0, 1;\n\t}"
               : "+r"(p));
  return p;
}
__device__ __forceinline__ void mbar_inval(int addr) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(addr));
}
__device__ __forceinline__ void mbar_init(int addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(count));
}
__device__ __forceinline__ void mbar_wait(int addr, int phase) {
  asm volatile(
      "{\n\t.reg .pred P;\n\tWAIT: "
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, "
      "0x989680;\n\t@P bra DONE;\n\tbra WAIT;\n\tDONE:\n\t}" ::"r"(addr),
      "r"(phase));
}
__device__ __forceinline__ void mbar_tx(int addr, int bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(addr),
      "r"(bytes)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_one(int addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr)
               : "memory");
}
__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4;
}
__device__ __forceinline__ uint64_t make_desc(int smem_addr) {
  constexpr uint64_t SBO = 8ULL * 128;
  return desc_enc(smem_addr) | (desc_enc(SBO) << 32) | (1ULL << 46) |
         (2ULL << 61);
}
__device__ __forceinline__ void tcgen05_mma(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, int acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, "
      "0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::
          "r"(taddr),
      "l"(a_desc),
      "l"(b_desc),
      "r"(idesc),
      "r"(acc));
}
__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(mbar_addr)
               : "memory");
}

// Task-internal sync using a simple atomic counter
// Safe for repeated calls, no phase tracking needed
__device__ __forceinline__ void
    task_sync_atomic(int volatile *counter, int tid, int num_warps) {
  __syncwarp();
  if (tid % 32 == 0 && tid / 32 < num_warps) {
    // Arrive
    int old = atomicAdd((int *)counter, 1);
    // Wait for all warps
    while (*(int volatile *)counter < (old / num_warps + 1) * num_warps) {
      __nanosleep(10);
    }
  }
  __syncwarp();
}

__device__ __noinline__ void simple_linear_task_dummy(float *C, int M, int N) {
  int const tid = threadIdx.x;
  if (tid / 32 >= 4) {
    return;
  }
  float a = 1.0001f + (float)tid * 0.00001f;
  float b = 0.9999f, c = 0.00001f;
  for (int i = 0; i < 100; i++) {
    a = fmaf(a, b, c);
  }
  if (tid < M * N) {
    C[tid] = a;
  }
}

__device__ __noinline__ void simple_linear_task(CUtensorMap const &A_tm,
                                                CUtensorMap const &B_tm,
                                                float *C,
                                                int M,
                                                int N,
                                                int K) {
  int const tid = threadIdx.x;
  int const wid = tid / 32;
  if (wid >= 4) {
    return;
  }

  int const num_k_tiles = K / TILE_K;

  extern __shared__ __align__(1024) char smem_buf[];
  constexpr int SMEM_OFFSET =
      0; // dynamic SMEM starts after static; no skip needed
  int const smem_base = __cvta_generic_to_shared(smem_buf) + SMEM_OFFSET;
  int const A_smem = smem_base;
  int const B_smem = smem_base + NUM_STAGES * TILE_BYTES_A;

  __shared__ uint64_t mbar_buf[2 * NUM_STAGES + 1];
  __shared__ int tmem_addr_buf[1];
  __shared__ int sync_counter; // atomic counter for task-internal sync
  int const tma_bar = __cvta_generic_to_shared(&mbar_buf[0]);
  int const mma_bar = __cvta_generic_to_shared(&mbar_buf[NUM_STAGES]);
  int const done_bar = __cvta_generic_to_shared(&mbar_buf[2 * NUM_STAGES]);

  // Warp 0: inval + init. Warp 1: TMEM alloc. Others: skip to sync.
  // task_sync_atomic AFTER this block ensures all warps see init results.
  if (wid == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbar_inval(tma_bar + i * 8);
      mbar_inval(mma_bar + i * 8);
    }
    mbar_inval(done_bar);
    for (int i = 0; i < NUM_STAGES; i++) {
      mbar_init(tma_bar + i * 8, 1);
      mbar_init(mma_bar + i * 8, 1);
    }
    mbar_init(done_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (wid == 1) {
    int addr_smem = __cvta_generic_to_shared(tmem_addr_buf);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(addr_smem),
        "r"(TILE_N));
  }

  // SYNC POINT 0: barrier init + TMEM alloc complete
  task_sync_atomic(&sync_counter, tid, 4);
  int const taddr = tmem_addr_buf[0];

  constexpr uint32_t idesc = (1U << 4) | (1U << 7) | (1U << 10) |
                             ((uint32_t)(TILE_N >> 3) << 17) |
                             ((uint32_t)(TILE_M >> 4) << 24);

  // In persistent kernel, blockIdx.x = worker ID, not tile index.
  // Tile indices should come from task metadata. For single-tile tasks, use 0.
  int m_tile = 0;
  int n_tile = 0;

  if (wid == 0 && elect_sync()) {
    int phase = 0;
    for (int ki = 0; ki < num_k_tiles; ki++) {
      int stage = ki % NUM_STAGES;
      mbar_wait(mma_bar + stage * 8, phase ^ 1);
      if (stage == NUM_STAGES - 1) {
        phase ^= 1;
      }
      int a_smem = A_smem + stage * TILE_BYTES_A;
      int b_smem = B_smem + stage * TILE_BYTES_B;
      mbar_tx(tma_bar + stage * 8, TILE_BYTES_A + TILE_BYTES_B);
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(a_smem),
          "l"(&A_tm),
          "r"(ki * TILE_K),
          "r"(m_tile * TILE_M),
          "r"(tma_bar + stage * 8)
          : "memory");
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::"
          "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(b_smem),
          "l"(&B_tm),
          "r"(ki * TILE_K),
          "r"(n_tile * TILE_N),
          "r"(tma_bar + stage * 8)
          : "memory");
    }
  } else if (wid == 1 && elect_sync()) {
    int phase = 0;
    for (int ki = 0; ki < num_k_tiles; ki++) {
      int stage = ki % NUM_STAGES;
      mbar_wait(tma_bar + stage * 8, phase);
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (stage == NUM_STAGES - 1) {
        phase ^= 1;
      }
      int a_smem = A_smem + stage * TILE_BYTES_A;
      int b_smem = B_smem + stage * TILE_BYTES_B;
      for (int k2 = 0; k2 < TILE_K / MMA_K; k2++) {
        uint64_t a_desc = make_desc(a_smem + k2 * 32);
        uint64_t b_desc = make_desc(b_smem + k2 * 32);
        tcgen05_mma(taddr, a_desc, b_desc, idesc, (ki == 0 && k2 == 0) ? 0 : 1);
      }
      tcgen05_commit(mma_bar + stage * 8);
    }
    tcgen05_commit(done_bar);
  }

  // SYNC POINT 1: MMA pipeline complete
  task_sync_atomic(&sync_counter, tid, 4);
  mbar_wait(done_bar, 0);
  asm volatile("tcgen05.fence::after_thread_sync;");

  // mbar_inval at start of next call handles cleanup

  // Epilogue: read TMEM → store GMEM
  float *C_out = C + m_tile * TILE_M * N + n_tile * TILE_N;
  for (int col = 0; col < TILE_N; col += 16) {
    float t16[16];
    int addr = taddr + (tid << 16) + col;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=f"(t16[0]),
          "=f"(t16[1]),
          "=f"(t16[2]),
          "=f"(t16[3]),
          "=f"(t16[4]),
          "=f"(t16[5]),
          "=f"(t16[6]),
          "=f"(t16[7]),
          "=f"(t16[8]),
          "=f"(t16[9]),
          "=f"(t16[10]),
          "=f"(t16[11]),
          "=f"(t16[12]),
          "=f"(t16[13]),
          "=f"(t16[14]),
          "=f"(t16[15])
        : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
    for (int i = 0; i < 16; i++) {
      C_out[tid * N + col + i] = t16[i];
    }
  }

  // SYNC POINT 2: before TMEM dealloc
  task_sync_atomic(&sync_counter, tid, 4);
  // Ensure all async ops complete before returning
  asm volatile("cp.async.bulk.wait_group 0;");
  asm volatile("tcgen05.fence::before_thread_sync;");
  task_sync_atomic(&sync_counter, tid, 4);
  if (wid == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
        "r"(TILE_N));
  }
}

} // namespace simple_linear
} // namespace kernel
