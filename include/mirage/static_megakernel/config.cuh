// config.cuh -- sizes of the Kimi K3 MoE layer (8 GPUs, 8 tokens, sm_100a / sm_103a) and the shared-memory, counter and
// exchange-region layouts derived from them. The K split counts of the router, latent_down and shared gate_up GEMMs are not here:
// the compiler chooses them, and they reach the task functions as template arguments (moe_kernel.cuh).
#pragma once
#include <cstdint>
#include <cstddef>

namespace static_mk {

// ---- model shape ----
constexpr int T = 8;            // tokens per step (speculative verify: 1 + 7 draft tokens); also the MMA N
constexpr int H = 7168;         // hidden
constexpr int LAT = 3584;       // MoE latent width (latent_down output)
constexpr int NE = 896;         // experts
constexpr int IR = 384;         // expert intermediate per GPU (3072 / 8)
constexpr int SHR = 768;        // shared-expert intermediate per GPU (6144 / 8)
constexpr int TPMAX = 8;        // GPUs
constexpr int NSLOT = 128;      // at most T x 16 experts are used in one step: one slot each

// ---- machine ----
constexpr int NSM = 148;

// ---- pipeline depths (measured on the hand-written reference kernel) ----
constexpr int SMAX = 5;         // stages of the weight ring (6 does not fit next to the 17 KB of static shared tables)
constexpr int W2_CHUNK = 4;     // W2 items per expert-queue entry
constexpr int W2QD = 4;         // entries in the expert queue's shared-memory copy (loader -> issuers / epilogue)
constexpr int MAXSEG = 4;       // h_q segments resident per SM in the W2 part

// ---- tile counts (128-wide tiles unless noted) ----
constexpr int KT_H = H / 128;       // 56 K tiles of the hidden
constexpr int KT_LAT = LAT / 128;   // 28 K tiles of the latent
constexpr int KT_SH = SHR / 128;    // 6 K tiles of the shared intermediate
constexpr int MT13 = IR / 64;       // 6 W13 items per expert: 64 features each (64 gate rows + 64 up rows)
constexpr int OT2 = LAT / 128;      // 28 W2 output tiles per expert
constexpr int KT2 = IR / 128;       // 3 W2 K tiles
constexpr int N_SDOWN = KT_H * 2;   // shared down entries in the expert queue: 56 row tiles x 2 K halves
constexpr int N_SGU_TILES = 2 * SHR / 128;   // 12 shared gate_up row tiles
constexpr int N_SACT = N_SGU_TILES;          // sact tasks, one per row tile (C_HS reaches this when h_s is complete)
constexpr int N_UPTILE = 98;        // latent_up tiles in the tail: 7 row tiles x 14 K parts of 2 K tiles
static_assert(OT2 % W2_CHUNK == 0, "W2 chunks must not cross expert slots");

// ---- shared memory (offsets from the 1024-aligned dynamic base) ----
constexpr int W_STAGE = 32768, A_STAGE = 2048, SF_CHUNK = 512;
constexpr int FSTAGE = W_STAGE + A_STAGE;           // one ring stage: 32 KB weight tile + 2 KB activation tile
constexpr int OFF_W = 0;                            // SMAX ring stages
constexpr int OFF_WSF = SMAX * FSTAGE;              // SMAX x 2 weight scale chunks
constexpr int OFF_XSF = OFF_W;                      // z scale chunks, staged once at the phase switch in the (then idle) ring area
constexpr int OFF_HQ = OFF_WSF + SMAX * 2 * SF_CHUNK;   // 3 KB + 1.5 KB not used by any task; kept so the later offsets stay the
constexpr int OFF_HSF = OFF_HQ + KT2 * 1024;            // measured ones (the shared-memory layout alone moved the layer time by 3 us)
constexpr int OFF_ACC = OFF_HSF + KT2 * SF_CHUNK;   // 128 x 8 fp32: a W13 item's accumulator, for SiTU across rows
constexpr int SEGB = 5120;                          // one h_q segment: 3 K tiles (3 KB) + 3 scale chunks (1.5 KB), padded
constexpr int OFF_HQ2 = ((OFF_ACC + 4096 + 1023) / 1024) * 1024;   // MAXSEG h_q segments
constexpr int OFF_GLUE = OFF_W;                     // route's scratch, in the ring area (no load is in flight during route)
constexpr int SMEM_BYTES = OFF_HQ2 + MAXSEG * SEGB + 1024;
static_assert(OFF_HQ % 1024 == 0, "128-B swizzled tiles are 1024-aligned");
static_assert(SMEM_BYTES <= 227 * 1024, "shared memory budget");

// ---- MMA instruction descriptors ----
constexpr uint32_t IDESC_BF16 = (1u << 4) | (1u << 7) | (1u << 10) | ((T / 8u) << 17) | ((128u / 16u) << 24);   // kind::f16: bf16 x bf16 -> fp32, M128 N8
constexpr uint32_t IDESC_MX = 0x08820280u;                                                                        // kind::mxf8f6f4: e2m1 x e4m3 -> fp32, ue8m0, M128 N8
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ull, EVICT_LAST = 0x14F0000000000000ull;                     // L2 cache hints

// ---- the generated layer's task table ----
constexpr int MAX_TASKS_PER_SM = 64;   // entries per SM (the list, then an end entry)

// ---- counters (u32, zeroed before each launch). The positions are the measured ones: counters in one 128-B line contend. ----
constexpr int C_HS = 3;          // sact tasks done; shared down waits for N_SACT
constexpr int C_W2NEXT = 5;      // the expert queue's next entry (atomic add returns it)
constexpr int C_SGU = 40;        // [N_SGU_TILES] per shared gate_up row tile: K parts added; sact waits for all of them
constexpr int C_HQ = 64;         // [NSLOT] per expert slot: W13 items that wrote their h_q; W2 waits for MT13
constexpr int C_EXPDONE = 200;   // SMs done with the expert queue; the tail waits for NSM
constexpr int NCNT = C_EXPDONE + 1;

// ---- exchange region between the GPUs (the same layout in every GPU's copy; a multimem.st lands in all copies) ----
constexpr size_t RG_ZQ = 0;                                          // z_q [T][LAT] e4m3 (quant -> every GPU's expert queue)
constexpr size_t RG_ZSF = RG_ZQ + (size_t)T * LAT;                   // z_q scale chunks [KT_LAT][512]
constexpr size_t O_RANK = (size_t)T * (H / TPMAX) * 2;               // one GPU's latent_up output: [T][H / TPMAX] bf16
constexpr size_t RG_O = ((RG_ZSF + (size_t)KT_LAT * SF_CHUNK + 4095) / 4096) * 4096;   // [TPMAX] latent_up outputs (tail)
constexpr size_t RG_HELLO = RG_O + TPMAX * O_RANK;                   // start barrier: one 16-B slot per GPU holds its launch number
constexpr size_t RG_END = RG_HELLO + TPMAX * 16;
// [R | S] partial sums go through separate peer-mapped buffers (G::rs_all), one [TPMAX][RS_RANK] buffer per GPU
constexpr size_t RS_RANK = (size_t)T * (LAT + H) * 4;

// ---- tail SM groups ----
constexpr int N_RSM = 32;   // SMs 0..31 sum and normalise R: (token, 896-column quarter) each
constexpr int N_SSM = 64;   // SMs 32..95 sum S: (token, 896-column eighth) each

// ---- per-SM time stamps read by the host (moe_host_timing, moe_host_phase_stamps), g.stamps[sm * NSTAMP + i] ----
constexpr int STAMP_START = 0, STAMP_QUEUE_START = 1, STAMP_QUEUE_END = 2, STAMP_END = 3, NSTAMP = 4;

}  // namespace static_mk
