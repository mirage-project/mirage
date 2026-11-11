// config.cuh (HIP, gfx950) -- the kernel-wide sizes, as in
// include/mirage/static_megakernel/config.cuh: the generated layer
// (static_schedule.header_code) defines STATIC_TOKENS, STATIC_GPUS, STATIC_SMS
// before it includes core.cuh. Here STATIC_SMS = CTAs of the launch = one per CU
// (256 on MI355X), and a CTA is 256 threads = 4 wavefronts of 64.
#pragma once
#include <cstddef>
#include <cstdint>

#if !defined(STATIC_TOKENS) || !defined(STATIC_GPUS) || !defined(STATIC_SMS)
#error                                                                         \
    "config.cuh: the generated layer defines STATIC_TOKENS, STATIC_GPUS, STATIC_SMS"
#endif

namespace static_mk {

// ---- kernel-wide sizes (the generated layer's) ----
constexpr int T = STATIC_TOKENS; // tokens per step
constexpr int GPUS = STATIC_GPUS;
constexpr int NSM = STATIC_SMS;  // CTAs of the launch, one per CU
constexpr int WAVE = 64;         // threads per wavefront (gfx950: wave64 only)

// ---- the weight ring (the Python side checks these names: static_schedule
// config_checks); the gfx950 GEMM's stages are sized with moe_experts ----
constexpr int SMAX = 5;
constexpr int SF_CHUNK = 512;
// the launch's dynamic shared memory for a need of `need` bytes from the base:
// + 1024, the base is 1024-aligned at run time (as on CUDA)
__host__ __device__ constexpr int smem_launch_bytes(int need) {
  return ((need + 1023) / 1024) * 1024 + 1024;
}

// ---- the generated layer's task table, its slots (the same as CUDA's) ----
constexpr int MAX_TASKS_PER_SM = 64;
// buffer slots (G::buf): 24, or the generated layer's STATIC_MAX_BUFS (a graph
// with more slots, e.g. the MLA layer: static_megakernel.py buf_slots)
#ifndef STATIC_MAX_BUFS
#define STATIC_MAX_BUFS 24
#endif
constexpr int MAX_BUFS = STATIC_MAX_BUFS;
#ifndef STATIC_MAX_MAPS
#define STATIC_MAX_MAPS 32
#endif
constexpr int MAX_MAPS = STATIC_MAX_MAPS; // (STATIC_MAX_MAPS: a graph with more maps, e.g. a fused layer)

// ---- counters (u32, zeroed before each launch), each node on its own 128-B
// lines (static_megakernel.py gives each node its lines) ----
#ifndef STATIC_COUNTER_LINES
#define STATIC_COUNTER_LINES 16
#endif
constexpr int NODE_COUNTER_LINES = STATIC_COUNTER_LINES; // (STATIC_COUNTER_LINES: a graph with more)
constexpr int NCNT = 32 * NODE_COUNTER_LINES;

// ---- per-CU time stamps read by the host (ns), g.stamps[cu * NSTAMP + i] ----
constexpr int STAMP_START = 0, STAMP_TASK0 = 1, STAMP_TASK1 = 2, STAMP_END = 3,
#if defined(STATIC_TIMING_BUILD) || defined(STATIC_PRE_STAMPS) || defined(STATIC_EXPERT_STAMPS) || defined(STATIC_NODE_END_STAMPS)
              // (STATIC_EXPERT_STAMPS, a normal build: moe_experts' wave 0 writes [4] its first W13 unit's start, [5] its
              // W13 loop's end, [6] its first W2 load, [7] its W2 loop's end)
              // the timing build (or STATIC_PRE_STAMPS): the CU's entry, its
              // pass of the reset barrier (STATIC_EARLY_PREWAIT: before the go)
              // (STATIC_EARLY_PREWAIT: [6] its resets' stores landed)
              STAMP_ENTRY = 4, STAMP_BAR = 5, STAMP_STORED = 6, STAMP_ISSUED = 7,
#if defined(STATIC_NODE_END_STAMPS)
              NSTAMP = 16; // ([8 + i]: the CU's last end of node i of STATIC_NODE_END_STAMPS, a normal build)
#elif defined(STATIC_EXPERT_STAMPS)
              NSTAMP = 12;
#else
              NSTAMP = 8;
#endif
#else
              NSTAMP = 4;
#endif
// s_memrealtime ticks -> ns (100 MHz on gfx950; host.cuh checks the device's
// rate at init)
constexpr long long NS_PER_TICK = 10;

} // namespace static_mk
