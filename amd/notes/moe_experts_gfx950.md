# moe_experts on gfx950 (MI355X): design

The routed experts of one K3 MoE layer, one GPU of TP8, at decode (T = 8 tokens per step: 1 token + the DSpark block, as on
B300). CUDA version: `include/mirage/static_megakernel/tasks/moe_experts.cuh` (TMA ring, tcgen05, TMEM, warp roles). On gfx950
none of those exist, so this is a new design. It keeps the CUDA version's work split and its outputs.

## What one step does (per GPU)

- Routing: 896 experts, top-16 per token. 8 tokens touch about 80 distinct experts (B300 server, layer 2: 80.5 on average).
- Per expert, this GPU's slice (routed intermediate IR = 384, latent LAT = 3584):
  - W13: gate and up rows, 2 x 384 rows x 3584 K, MXFP4 = 1.38 MB, plus scales (one E8M0 per 32 K) 86 KB.
  - h = SiTU(gate, up) for the expert's tokens, quantized to MXFP8 (one scale per 32 features).
  - W2: 3584 rows x 384 K, MXFP4 = 0.69 MB, plus scales 43 KB.
  - the result rows times the routing weight, one fp32 row per (token, k) pair (allreduce_send adds a token's 16 rows in a
    fixed order, as on CUDA).
- Bytes: about 2.2 MB per expert, about 176 MB per step at 80 experts. At 8 TB/s (MI355X HBM peak) that is 22 us; at 6.5 TB/s
  (a sustained rate, to be measured by amd/bench/read_bw.hip) 27 us.
- SGLang today (amd/notes/sglang_k3_moe_rocm.md): AITER's tuned W13 + W2 kernels take 53.84 us at T = 8 on gfx950 (AITER's
  own tuner, inputs rotated), not counting sort and quantization. Target: the weight-read time, about 27-30 us.

## Compute is small

The scaled MFMA `v_mfma_scale_f32_16x16x128_f8f6f4` with weights as A (FP4, 16 output rows) and tokens as B (FP8, 16 columns,
8 of them real tokens): 32 cycles per instruction (FP8 on one side sets the rate; ISA notes 2.1).
- W13: 768 / 16 x 3584 / 128 = 1344 MFMAs per expert; W2: 3584 / 16 x 384 / 128 = 672. Total 2016 x 32 = 64.5 K SIMD-cycles
  per expert, 5.2 M for 80 experts, over 1024 SIMDs: 5040 cycles = 2.1 us at 2.4 GHz.
- So the kernel is a weight stream: the design question is keeping enough bytes in flight, not the math.

## Data path

- Weights go from HBM straight into registers: each lane's FP4 A fragment of one MFMA is 16 contiguous bytes (one
  `global_load_dwordx4`), if the weights are stored pre-shuffled in lane order (`amd/tools/shuffle_weights.py`). No LDS stage:
  at T = 8 every weight byte is used by exactly one MFMA.
- Bytes in flight: at 8 TB/s and about 1 us memory latency, 8 MB must be in flight over 256 CUs = 32 KB per CU = 128 B per lane
  of 4 waves = 8 MFMA fragments ahead per lane (32 VGPRs). The prefetch depth is a variant (`stages`) for the loop to tune.
- Cache hint on the weight loads: a variant (default / nt / sc1 nt), from amd/bench/read_bw.hip.
- Tokens (B operand): z_q = 8 x 3584 FP8 (28 KB) + scales, one copy per CU in LDS. Each wave reads the B fragment of its K step
  from LDS (32 B per lane). To cut LDS traffic, the 4 waves of a CU split K (wave w takes K steps w, w + 4, ...) and sum their
  16 x 16 partial tiles through LDS at the end of a tile (1 KB per wave); then each wave reuses a B fragment for every row tile
  it holds in flight.
- Weight scales: one byte per lane per MFMA (lane l: row l % 16, K block l / 16), 4 MFMAs' scales in one VGPR (op_sel picks the
  byte), pre-shuffled with the weights.

## W13 -> h_q -> W2

- A W13 item = (expert, 32 features): 2 gate tiles + 2 up tiles of 16 rows (4 MFMA chains over K = 3584). After the K loop the
  gate and up values of one (feature, token) are in the same lane and register (C/D layout: lane -> token l % 16, register r ->
  row 4 (l / 16) + r), so SiTU is element-wise in registers.
- MXFP8 quantization of h: a block = 32 features of one token = 2 tiles x 4 lanes (l % 16 the same) x 4 registers; the block's
  max |h| by a cross-lane reduction over lanes l, l + 16, l + 32, l + 48, then the E8M0 scale and the e4m3 values (cvt_e4m3,
  clamped to 448).
- h_q (8 x 384 FP8 per expert, 3 KB) is stored with device-scope stores (sc1) and counted per expert (cnt_add_bypass); W2 items
  of that expert wait for the count (cnt_wait_bypass) and read h_q with sc1 loads (the bypass protocol, checked by
  amd/bench/handoff_stress.hip on day one).
- A W2 item = (expert, 16-row output tile x some count): K = 384 = 3 MFMA steps per tile, so a W2 item takes many tiles to keep
  bytes in flight.

## Work split

As on CUDA: one queue of items per GPU (W13 items first, then W2), taken with an agent-scope atomic add. Two gfx950 additions,
both variants for the loop:
- per-die queues: the W13 and W2 items of one expert on one die (the h_q hand-off stays in that die's L2).
- the order of the experts in the queue (by token count).

## Unit kernels (amd/bench/expert_unit.hip, compile-only so far)

- w13_kernel: one wave per item, weight fragments STAGES K steps ahead in registers, the item's weight scales in LDS (a
  run-time-indexed register array goes to scratch), the last round of K steps peeled (a branch around the prefetch made the
  compiler wait for all loads at the top of every round); then SiTU, the MXFP8 quantization of h (ceil(log2(amax / 448)), as
  the CUDA version) written in W2's fragment order. STAGES 4: 150 VGPRs, up to 15 loads in flight in the main loop; STAGES 7:
  220 VGPRs, up to 27.
- w2_kernel: items of W2_TILES row tiles, two items in flight; an item's h_q fragments are loaded with its weights (a load issued
  inside the compute made it wait for the next item's prefetch: vmcnt counts in order). Up to 62 loads in flight (the counter's
  limit is 63), 256 VGPRs, no scratch.
- Checks against the CPU: h (sampled items), h_q scales and bytes (every item, from the kernel's own h), W2's y (sampled rows,
  from the kernel's own h_q).

## The task (include/mirage/static_megakernel_hip/tasks/moe_experts.cuh, compile-only so far)

- One task per CU, as on CUDA. z_q is gathered into LDS 16 B at a time (each half of a lane's fp8 fragment is 16 consecutive K
  values in both layouts; a byte scatter made the compiler hoist ~100 addresses out of the task loop and spill), the routing
  table is built with 64-lane ballots, the items come from one ordered queue per GPU (an agent-scope counter; each wave takes
  one item at a time): all W13 items, then the W2 items behind each slot's counter.
- Why a queue, not a static deal of the items over the 1024 waves: moe_experts is not a concurrent node, so a CU can still be
  in an earlier task while others run moe_experts. With a static deal a W2 item could wait for a W13 item of such a CU, whose
  earlier task may wait for a task another CU runs after moe_experts (e.g. the shared-expert chain): a deadlock. With the
  ordered queue a wave takes a W2 item only after every W13 item was taken by a wave already in the task, and W13 items wait
  for nothing (the CUDA version's argument).
- In the full layer kernel (all 9 task types) STATIC_W2_TILES 8 spills 504 B per lane; 4 (the default) does not. The other
  knobs: STATIC_W13_STAGES (4), STATIC_GEMM_STAGES (8, gemm_tile), STATIC_HANDOFF (1 bypass / 0 release), STATIC_FP8_LIN,
  STATIC_FP4_HI.

## Open until measured (day one)

1. The FP8 operand layout (fig vs lin) and the FP4 nibble order: amd/bench/mfma_layout_check.hip.
2. Whether the bypass hand-off is correct across dies: amd/bench/handoff_stress.hip.
3. Sustained read speed by cache hint and number of CUs: amd/bench/read_bw.hip.
4. Whether workgroups land on the same die every launch: amd/bench/sync_latency.hip.
