// config.cuh -- what the kernel templates share: the kernel-wide sizes, which
// the generated layer (static_schedule.header_code) defines from the graph and
// the schedules before it includes core.cuh:
//   STATIC_TOKENS   tokens per step (StaticMegakernel.tokens: the graph's
//   tensors' first dimension); also the MMA N STATIC_GPUS     GPUs the layer is
//   divided over (StaticMegakernel.num_gpus) STATIC_SMS      task lists per GPU
//   = CTAs of the launch, one per SM (the schedules)
// and the ring's shared-memory layout. Every other size (the hidden, the
// experts, ...) is a node's: its params or its producers' (StaticNode,
// StaticParams), checked by its task type's file. Where each buffer is, and the
// cross-GPU exchange region's layout, come from the layers
// (static_megakernel.py) through the generated layer too.
#pragma once
#include <cstddef>
#include <cstdint>

#if !defined(STATIC_TOKENS) || !defined(STATIC_GPUS) || !defined(STATIC_SMS)
#error                                                                         \
    "config.cuh: the generated layer defines STATIC_TOKENS, STATIC_GPUS, STATIC_SMS"
#endif

namespace static_mk {

// ---- kernel-wide sizes (the generated layer's) ----
constexpr int T = STATIC_TOKENS; // tokens per step; also the MMA N
constexpr int GPUS =
    STATIC_GPUS; // GPUs (an exchange buffer has one slot per GPU)
constexpr int NSM = STATIC_SMS; // CTAs of the launch, one per SM
static_assert(T == 8,
              "the GEMM ring (runtime.cuh) is written for 8 tokens: MMA N = 8, "
              "8 accumulator columns per issuer");

// ---- pipeline depths ----
constexpr int SMAX = 5; // stages of the weight ring (6 do not fit in shared
                        // memory next to moe_experts' shared tables)
constexpr int W2QD = 4; // entries in moe_experts' shared-memory copy (loader ->
                        // issuers / epilogue; runtime.cuh Rt)

// ---- the ring's shared memory (offsets from the 1024-aligned dynamic base); a
// task type that needs more declares its own after
//      RING_BYTES (its smem_<name>) ----
constexpr int W_STAGE = 32768, A_STAGE = T * 256,
              SF_CHUNK = 512; // SF_CHUNK: one MX scale chunk (128 rows x 4 B)
constexpr int FSTAGE = W_STAGE + A_STAGE; // one ring stage: a 32 KB weight tile
                                          // + the activation tile (T x 256 B)
constexpr int OFF_W = 0;                  // SMAX ring stages
constexpr int OFF_WSF = SMAX * FSTAGE;    // SMAX x 2 weight scale chunks
constexpr int RING_BYTES = OFF_WSF + SMAX * 2 * SF_CHUNK;
// the launch's dynamic shared memory for a need of `need` bytes from the base:
// + 1024, the base is 1024-aligned at run time
__host__ __device__ constexpr int smem_launch_bytes(int need) {
  return ((need + 1023) / 1024) * 1024 + 1024;
}

// ---- MMA instruction descriptors ----
// kind::f16: bf16 x bf16 -> fp32, M = m, N = T
__host__ __device__ constexpr uint32_t idesc_bf16(uint32_t m) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((T / 8u) << 17) |
         ((m / 16u) << 24);
}
constexpr uint32_t IDESC_BF16 = idesc_bf16(128);
constexpr uint32_t IDESC_MX =
    0x08820280u; // kind::mxf8f6f4: e2m1 x e4m3 -> fp32, ue8m0 scales, M128 N8
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ull,
                   EVICT_LAST = 0x14F0000000000000ull; // L2 cache hints

// ---- the generated layer's task table, its slots ----
constexpr int MAX_TASKS_PER_SM =
    64;                      // entries per SM (the list, then an end entry)
constexpr int MAX_BUFS = 24; // buffer slots (G::buf): node outputs, scratch
                             // buffers, graph tensors read by pointer
constexpr int MAX_MAPS = 32; // tensor map slots (Maps::m)

// ---- counters (u32, zeroed before each launch). Counters in one 128-B line
// slow each other down, so each node has its own lines:
//      static_megakernel.py gives each node its lines, the node's first counter
//      is StaticNode::counter ----
constexpr int NODE_COUNTER_LINES = 16;
constexpr int NCNT = 32 * NODE_COUNTER_LINES;

// ---- per-SM time stamps read by the host (host.cuh), g.stamps[sm * NSTAMP +
// i]: the kernel's start and end, and two a task type
//      may write (moe_experts: its queue's start and end) ----
constexpr int STAMP_START = 0, STAMP_TASK0 = 1, STAMP_TASK1 = 2, STAMP_END = 3,
              NSTAMP = 4;
// STATIC_RESET_IN_KERNEL (a build where one launch is a whole run, e.g. for a
// CUDA graph): the buffers other GPUs write into, and the 0xFF-polled ones with
// one reader, are reset by their reader for the next launch (REARM);
// kernel_begin resets this GPU's other buffers. Off: the host resets all before
// a launch
#ifdef STATIC_RESET_IN_KERNEL
constexpr bool REARM = true;
#else
constexpr bool REARM = false;
#endif

} // namespace static_mk
