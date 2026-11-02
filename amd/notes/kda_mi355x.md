# The K3 KDA layer on MI355X: plan and math (2026-10-13)

Goal (user 10-13): KDA, then MLA, as static megakernels on MI355X, all three parts (MoE, KDA, MLA) in SGLang ROCm on the full K3,
each part and the full-model span at most 15% slower than B300's (mirage_kda/docs/k3_results_2026-10-11.md: KDA part 31.2 us
per layer in the server, B300 one-layer harness with the serving contract 35.1 us at 2 bank rows).
B300 design reference: mirage_kda demo/kimi_k3_kda/part.py, tasks/{short_conv_silu,delta_rule_recurrence,attn_residual}.cuh,
reference mpk-kernels/tests/kda_reference.py, numerics docs/numerics_vs_sglang.md.

## The layer per GPU (TP8, T = 8 tokens, 12 heads of 128)

```
x [8, 7168] bf16
 -> projection  P = x . Wproj^T        Wproj [6336 -> padded 6400, 7168] bf16   91.8 MB (q k v g 1536 each, f_a 128, beta 12 of 64)
 -> short conv  q/k/v (4608 channels): 4 taps over [3 carried inputs from the pool slot, 8 new], SiLU   (conv_w 4608 x 4 fp32)
                writes ring_k / ring_v (bf16 after SiLU), the per-token windows icw [8][3][4608]
 -> recurrence  per head h: f_b = f_a . Wfb[128h..+128]^T (Wfb 1536 x 128 bf16, 384 KB); gk = -5 sigmoid(exp(A_log) (f_b + dt_bias));
                beta = sigmoid(b); q, k L2-normalised; 8 tokens in order on S [128 v][128 k] fp32 (from the state pool slot,
                not written back):  S *= exp(gk); pred = S k; d = beta (v - pred); S += d k^T; o = S q
                o_norm (RMS over v, eps 1e-5) x w x sigmoid(g) -> y [8, 1536] bf16; writes ring_g, ring_beta
 -> o_proj      partial = y . Wo^T     Wo [7168, 1536] bf16   22.0 MB; each GPU's partial [8, 7168] to every GPU
 -> attn residual  per token: u = bf16(prefix + sum of the 8 partials) (the next prefix); scores over n bank rows and u
                against bf16(q gain), softmax, mix, RMSNorm x ow -> hidden [8, 7168]
```
Weights read per GPU: ~114 MB. At ~24 GB/s per CU (1 WG / CU, MI355X measured) on 256 CUs = 6.1 TB/s: ~19 us of reading;
the projection alone ~15 us.

## Critical chain and budget (what each step should cost)

1. start (in-kernel resets before the go, like the MoE): ~1 us after the go
2. projection 91.8 MB on ~all CUs: ~15-16 us (the floor at the per-CU read rate)
3. conv after the last K part of a 64-column block: ~1 us (adds the K parts, 4 taps)
4. recurrence: 12 heads; per head 8 sequential tokens of 128 x 128 fp32 (4 x 16K MAC per token); a head split by v rows over
   several CUs (rows are independent except the o_norm's RMS) keeps it ~1-2 us
5. o_proj 22 MB: its weights do not depend on y; the CUs idle during the recurrence load them first (LDS / registers), the
   GEMM itself when y lands: ~1-2 us
6. one xGMI hop of each GPU's partial (115 KB) to every GPU: ~3.7 us (push_latency, >= 32 sender CUs)
7. attention residual per token (8 x 7168): ~2 us
Sum ~25-28 us after the go. Target: <= 35.9 us in the server (1.15 x 31.2).

## Port order (each with its check)

K1. reference + random weights in the real shapes (torch float64, from kda_reference.py; serving contract: state / conv window
    read from pool slots, rings and icw written, pools unchanged); harness amd/tests/test_kda_layer.py on the MoE harness's model.
K2. HIP tasks: short_conv_silu (new), delta_rule_recurrence (new: VALU fp32 first; split by v rows), attn_residual (new: one
    token per task first), gemm_tile combine "push_all" (each GPU's partial of every row to every GPU; new mode, default code
    of the MoE build unchanged), projection through gemm_tile slots.
K3. checks (every step against the reference from the kernel's own inputs), then e2e like for like.
K4. tune (budget above), then the SGLang ROCm A/B (KimiK3DeltaAttention on the same weights), then the server.

## Measured (2026-10-13, MI355X TP8, harness, 1 GB flushed before each launch)

- kda5 (all checks pass): plain launch e2e 123.4 us; like for like with B300 (line-up, in-kernel resets before the go, 5 us
  stand-in) e2e 104.5-105.2 us, span ~101.5. Target <= 35.9.
- Stamped profile (GPU 0, us after the start barrier; rank1/rank2 kda prof): projection ends ~25-27 (92 MB: ~3.4 TB/s);
  conv ends ~33; recurrence: conv outputs complete 33.2 -> device loads done +6.7 -> f_b / decay done +9.5 -> token loop
  done ~+6 -> end +3.5 (27 us after its input); o_proj ends ~64 (2.4 us after y); attn_residual (one task per token): the
  GPUs' partials landed 19 us after o_proj ended, sums / dots +11, softmax +2, output +8.6 (42 us after o_proj).
  Every phase ~= its count of serialized memory round trips x ~1.3 us on one CU (load-and-wait helpers ld16_dev / ldf4_dev,
  one polled vector at a time). The layer kernel had 252 B of scratch (spills), almost all in attn_residual's old arrays.
- Fixes (kda6, pending): attn_residual split into 7 column chunks per token (56 tasks, concurrent), every load of a phase in
  flight together, two cross-CU rounds (scores' sums, h's sum of squares) through a scratch buffer; recurrence: its device
  loads issued together (ld16_at<AUX_DEV>), the state / W_fb loads completed before the conv wait, the gate and norm weight
  loaded early. Scratch now 0 B (KDA and MLA).
- Numerics aligned with SGLang ROCm (no cost): conv + SiLU output and the recurrence's o rounded to bf16 before o_norm
  (fused_norm_gate.py order), aggregation 2's cw in fp32, the all-reduced attention output rounded to bf16 before + prefix.
- Server path: per-rank library (rank_host.cuh, gen_rank_lib.py) bit-identical to the harness (8 processes, HIP IPC
  exchange, eager and HIP graph): rank2 10-13 for KDA and MLA.
- kda6 (attn_residual chunks, recurrence load batching): like for like e2e 72.4 us; kda7 (recurrence LDS reads 16 B): 62.1
  (recurrence 14.2 us after its input; attn_residual 19 us after o_proj).
- 10-13 server (SGLang 0.5.21 ROCm, full K3 + DSpark, all three of ours, rank 1 trace): KDA part 69.6 us/layer (kernel 69.7;
  SGLang alone 124.6), MLA part 89.2 (121.6), MoE kernel 91.4; span 16095 vs 20867 us; 7.48 vs 9.38 ms/token.

## 2026-10-13 evening (MI355X TP8, harness, 1 GB flushed before each launch; e2e medians)
- kda14: recurrence restructured (state blocks 4 v x 16 k per thread, scalars for the norms): like-for-like e2e 57.0 (60.8
  before); token loop 8.1 us (12.3).
- push_pattern bench (amd/bench/push_pattern.hip, 115 KB per GPU, 224 senders): o_proj's 8-B stores (gemm8) one hop 5.6 us,
  16-B row stores 4.7, the task's block contiguous with 16-B stores (tile16) 4.2.
- kda16 A/B (2 rounds, like for like): base 56.8 / 57.2; DPP reductions (STATIC_DRR_DPP + STATIC_WAVE_DPP: ds_bpermute
  shuffles -> DPP moves + readlane) 52.8 / 53.1; residual exchange data-as-flag (STATIC_AR_DATAFLAG) 53.5 / 53.7; tile-major
  push (STATIC_PUSH_TILE) 54.0 / 53.7; all four 46.3 / 46.6 -> all four default 1.
- Lost / no change: ring stores after the count (STATIC_DRR_RING_LATE) 55.7 / 56.2 vs 56.1 / 56.0; projection grids 100x8
  57.7, 100x14 59.8, 50x14 57.8 (100x7 ~56); fast reset (RIK_DIE + STATIC + PAR + new RIK_CUGEN, no line-up) with server
  flags 71.2 vs 68.9 / 69.7.
- Found (kda16 per-task dump, MPK_DUMP_TIMES): attn_residual tasks queued behind an o_proj task on the same CU start ~6.4 us
  after it ends (its system-scope pushes to 8 GPUs are waited for by the next task's first load wait: stores and loads share
  vmcnt on gfx950); 24 of the 56 sit behind o_proj with grid 224 -> every token's exchange waits for them. o_proj grid 112
  puts them on other CUs (kda19 A/B).
- Server trace (all3, rank 1): mpk_kda_kernel 67.7 us, mpk_moe_kernel 100.3, _agg_kernel 5.5 per KDA layer; the harness
  e2e with server flags (69 / 104) predicts them. e2e - span ~11.5 us per launch with server flags (dispatch + start).
- Fused KDA + MoE (amd/tests/test_kda_moe_layer.py): correct, but e2e 201 vs 69 + 104 separately (projection starts 7.3 us
  after the barrier, every stage slower; scratch 468 B); kda18 profiles all three to find why.

## 2026-10-13 night (same setup)
- kda23 (mla_vgate's w_vc by global_load_lds, STATIC_VGATE_DMA default 1): MLA 37 rows like for like 50.6 vs 57.1 / 54.4,
  server flags 65.6 vs 69.0.
- kda25 grouped MLA (2 heads per task, distributed merge): 42 splits 1000 / 4000 / 8000 rows 68.3 / 77.2 / 85.8 (21 splits +
  first 35: 74.8 / 87.8 / 103.6). At 8000 rows from the task's start: chunk loop done 23.1, all splits counted 31.0, end 35.3.
- kda26 kT swizzle (STATIC_MLA_KT_SWZ): no difference (37 rows 51.5 / 51.5 off vs 52.0 / 51.3 on; 8000 rows 85.7 / 87.2 vs
  86.3); kept.
- kda29 parity reset (STATIC_RESET_PARITY, server flags, no line-up): KDA 58.0 / 58.6 vs barrier 62.4 / 61.5; MLA 64.1 /
  63.6 vs 65.9 / 63.6; KDA rank lib with parity bit-identical to the harness (eager 55.0 us per rank, HIP graph 58); MoE
  with parity printed no result (to read).
- kda27 lined-up profiles (GPU 0, us after the start barrier):
  - KDA projection (91.8 MB, 700 tasks of 128 KB): CUs with 3 tasks end 21.2, with 2 tasks 15.8; a CU's 1st task 8.0 us,
    2nd 6.0, 3rd 5.0 (256 CUs x 128 KB in 8 us = 4.2 TB/s; registers hold 32 KB per CU in flight: read_bw 5.1 TB/s at 32 KB,
    6.25 at 64 KB). kda36: whole slices through LDS (STATIC_GEMM_LDSW=1) vs 3.
  - attn_residual (KDA and MLA): o_proj ends 37.5 (MLA 41.0); partials landed and u stored 43.4 .. 45.6; the chunks'
    exchange done 48.5 .. 49.6 (~4 us after the slowest chunk's u; sync_latency's data-flag hand-off is 0.6 us); end 49.8.
    kda34: stamps inside the exchange (STATIC_STAMP_SET 4), u_out stored after the exchange (STATIC_AR_UOUT_LATE), and
    amd/bench/hop_size.hip (one xGMI hop at 115 KB vs 14 KB per peer: whether a reduce-scatter + all-gather beats the
    all-to-all + chunk exchange).
  - MoE: expert tasks end median 65.7, last 73.4; sum_gpus last 80.8; latent_up last 85.5; residual_add last 88.6.
- Grouped MLA, two chunks' rows in flight (two register sets): the kernel already uses 256 VGPRs + 256 AGPRs; the second set
  spills (scratch 132 -> 388 B) and the spill's reload waits vmcnt(0) at the chunk's top: no overlap. Not kept.
- Grouped MLA reads the cache once per head group: 6 x 9.2 MB at 8000 rows. kda35: the head groups of a split on one die
  (plan_one_gpu.die_plan, workgroup w on XCC (w + 2) mod 8, sync_latency) so 5 of 6 reads can hit that die's L2.
- KDA with --ar-chunks 4 (1792-column chunks): u (new prefix) non-finite in normal builds and stamp set 2 (hidden correct;
  launch 1 != launch 0), passes with stamp set 0: under investigation (test_kda_layer.py now prints where).
- kda23 rerun (complete): vgate DMA on 50.7 / 49.7 like for like (off 54.4 / 54.8), server flags 65.4 / 66.6 (off 71.2 / 70.6).
- kda32: mla_norm 4 column groups 57.5 / 57.5 vs 1 group 52.5 / 51.7 (37 rows); 8000 rows grouped 85.9 (no change): stays 1.
- kda33 fused KDA + MoE (server flags): A 186.6 / 187.4, B 189.0 / 190.2, C 187.3 / 189.8 vs KDA ~62 + MoE 104-107 apart.
  Profile (B, timing build): resets 5 us, barrier 5 us, first task 7.3 us after it; the projection's first task 12.7 us
  (7.4 apart) with nothing else running: the slow start is the fused build itself (timing build resets the untrimmed MoE
  scratch; not yet separated from the normal build).
- kda34 hop_size: one xGMI hop (store + poll, 8 GPUs) 4.0-4.3 us from 1 KB to 115 KB per peer: latency, not bytes. A
  reduce-scatter + all-gather (two hops) cannot beat the all-to-all + chunk exchange: dropped (edits reverted).
- kda34: KDA --ar-chunks 4 with u_out stored after the exchange (STATIC_AR_UOUT_LATE=1) passes: 47.8 / 48.4 like for like,
  63.2 / 67.5 server flags. Root cause of the non-finite u (asm, kda_cgar4): st16_dev's inline global_store_dwordx4 was
  followed one SALU later by v_pk_fma writing its data VGPRs; the compiler adds wait states only after its own stores.
  Fix: s_nop 1 inside st16_dev / st16_sys (runtime.cuh). kda39 checks KDA / MLA / MoE with it.
- kda35: grouped MLA, 40 splits with the head groups of a split on one die (--die-group 35) 1000 / 4000 / 8000 rows 63.0 /
  76.2 / 82.8 (63.8 / 76.2 / 83.8), 40 splits alone 64.4 / 77.6 / 84.4, 42 splits 69.0 / 78.0 / 86.2: 40 + die-group.
- kda36: projection weights through LDS (STATIC_GEMM_LDSW=1) 52.6 / 52.4 vs LDSW=3 47.8 / 48.2: stays 3.
- kda38: MLA outputs staged in LDS and stored in 16-B pieces of each token's row (vgate: STATIC_VGATE_ST16; attn / absorb:
  STATIC_MLA_ST16): 37 rows like for like 47.9 / 48.2 (vgate only 48.8 / 48.0, neither 52.6 / 51.3); server flags 60.2 /
  60.3 (60.1-61.6, 63.8 / 65.8). Both default 1.
- kda24 (o_proj 112 vs 224, ar 0): KDA 46.4 / 51.0 vs 47.5 / 48.7: no clear difference.
- kda37: MoE with STATIC_RESET_PARITY stops in host setup: routing_pairs is a reset buffer the harness passes in (parity
  keeps reset buffers in its own two-copy arena). The rank lib allocates it itself: kda40 times / bit-compares MoE parity there.
- kda39 (s_nop fix in st16_dev / st16_sys): every check passes, KDA --ar-chunks 4 with u_out early included. KDA like for
  like: ar4 48.2 / 48.2, ar4 + UOUT_LATE 48.5 / 48.0, default chunks 48.8 / 47.6; server flags 63.7 / 64.2, 64.8 / 62.2,
  61.9 / 60.8: default chunks stay. MLA 37 rows (ar4, first 35) 49.0 / 63.1; grouped 1000 rows 40 splits die-group 59.8;
  MoE 88.5 like for like, 103.2 server flags.
- kda40 MoE rank lib (8 processes, 1 GB flushed, lined up): barrier eager ~98.9 / HIP graph ~103.2 us; parity (RIK_CUGEN +
  RESET_PARITY) eager ~98.6 / graph ~101.4; both bit-identical to the harness (eager and graph). srv3 (14.10 02:04): the
  three libraries with parity, KDA default chunks, MLA grouped 40 splits die-group, vgate 16-B stores, the s_nop fix.
- srv3 (14.10, full K3 in SGLang, all three ours with parity, vgate 16-B stores, grouped MLA 40 splits die-group, s_nop fix):
  span 13934 vs SGLang 20867 us (14646 in srv2), 6.790 ms/token (7.185); per layer KDA 54.5, after KDA 93.2, MLA 62.4;
  2 of 8 answers as base0 (as srv2: the store hazard is not the drift). chk3: KDA / MoE rank libs bit-identical (srv3's
  check step had died in torch.distributed, transient); KDA eager 54.4 / graph 57.6, MoE 96.5 / 100.2 us.
- The rank-lib "HIP graph" e2e is a one-node graph launched per replay (its launch overhead in the window); the server's
  step is one graph of many nodes, and its KDA part (54.5) is below the harness's server-flags e2e (60.8): no graph cost to
  chase there. HIP_FORCE_DEV_KERNARG=1 is already set in the image.
- kda41 GEMM stages 16 vs 8: KDA 51.3 / 51.0 / 50.3 vs 47.6 / 47.7 / 47.4; projection tasks 8.3 / 6.5 / 5.8 vs 7.5 / 5.8 /
  4.6 us (1st / 2nd / 3rd per CU): more in flight per wave is slower. kda36: whole slices via LDS also slower (9.3 / 7.3 / 6.2).
- Residual stamps (kda34 set 4: stores done -> exchange done -> end): ~44.5 -> 45.2-46.5 -> 49.3-50.3: the exchange is
  ~1 us; ~3.5 us after it is every thread computing the weights and the 136 Gram terms of h's sum of squares one after the
  other -> STATIC_AR_WAVE_W (wave 0 computes them in parallel, LDS broadcast), kda42.
- kda42 STATIC_AR_WAVE_W (wave 0 computes the weights / h's sum of squares): KDA 47.3 / 48.8 / 48.0 vs 47.7 / 48.0 / 47.7;
  MLA 46.4 / 47.4 / 47.3 vs 47.1 / 48.6 / 47.8: no clear gain (off).
- kda43 projection grids (64-row x 7 parts now): 32-row 200x7 54.9 / 56.0 / 55.4, 16-row 400x7 59.8 / 59.9 / 60.5 vs
  100x7 47.9 / 47.7 / 48.2: smaller tasks lose.
- kda44 attn_residual knockouts (timing only, STATIC_AR_KO): no wave sums in the after-u dots 48.1 / 48.6 / 48.4, no
  exchange 47.4 / 47.3 / 47.3, no weights 47.6 / 47.4 / 48.1, no u_out store 47.8 / 47.8 / 48.0, all four 45.4 / 46.1 / 45.8
  vs 47.6 / 47.3 / 47.9: the residual's own work is ~2 us; its tail is the hop and the stores / kernel end.
- The register-streamed GEMM loop (asm, kda_cgst8): each round issues its 8 loads at its end and consumes them at the next
  round's start: s_waitcnt vmcnt(7 .. 3) before the MFMAs, ~8 KB per wave in flight, ~17-20 GB/s per CU (the projection's
  ~4.4 TB/s); with 16 stages the compiler keeps ~4 register sets (vmcnt(3-4)) after a vmcnt(0): slower. ->
  STATIC_GEMM_DMA_RING (per-wave LDS ring filled by global_load_lds in asm, waits by count), kda45.
- kda45 STATIC_GEMM_DMA_RING (per-wave LDS ring, global_load_lds builtin, waits by count; asm checked: vmcnt(15) only):
  KDA ring 16 50.9 / 51.1 / 52.5, ring 32 52.3 / 52.8 / 52.4 vs off 47.6 / 48.0 / 47.7; MLA 49.5 / 48.7 / 51.0 vs 47.2 /
  48.0 / 47.7; MoE 90.9 / 91.9 / 91.3 vs 89.3 / 88.9 / 87.9. Projection tasks 8.8 / 7.1 / 6.2 us (registers 7.5 / 5.8 /
  4.6): more bytes in flight per wave is slower here, so in-flight depth is not the projection's limit. Reverted.
- amd/bench/stream_pattern.hip (bw1, 256 WGs x 4 waves, 1 KB global_load_lds per wave step, explicit vmcnt depth, 1 GB
  flushed): 384 KB per CU 18.8-19.7 us (5.1-5.35 TB/s), 128 KB per CU 8.9-9.8 us (3.4-3.75); depth 4..32 per wave and
  per-wave vs interleaved streams within a few %. So ~6.4 TB/s once streaming plus ~4 us ramp + drain per kernel. The
  KDA projection (384 KB on its busiest CUs, last end 20.4-22.7) is at that limit: its room is the 384 vs 358 KB balance
  (~1.3 us), not the loop. MoE W13 (131 MB in ~30 us on ~242 CUs: ~18 GB/s per CU) is below it.
- kda46/47 aggregation 1 in the KDA kernel (attn_aggregate, --agg1; graph indices shift by 7: grids 35 / 38): KDA
  61.9-64.7 vs 47.5-49.0 lined up, 79.2-80.4 vs 62.8-63.0 server flags; agg1 tasks 1.3 -> ~17 us (source in and u
  stored at ~11.6, weights at ~16.5), projection ends 39 (20 without). Off (opt-in harness option).
- kda48 MoE start poll sleep (STATIC_START_SLEEP 1 / 8 / 32): MoE 87.0-88.5 lined up at all three, server flags 102.2-104.4;
  fused 186.3-189.8: the start polls' traffic is not what slows MoE or the fused kernel.
- kda49 STATIC_TAIL_SLEEP 1 / 8 / 32: KDA 48.2-48.8 / 47.1-48.2 / 47.4-49.8, MLA 47.4-48.0 / 47.2-48.4 / 49.4-50.1: no gain.
- GEMM loop asm (cgsb): the fused build's projection loop waits vmcnt(0) at its head (a spill reload in the preheader);
  the KDA-only loop vmcnt(7 .. 3). A sched_barrier after each step's refill (STATIC_GEMM_SCHED_BAR) gives the KDA-only
  loop a steady vmcnt(7) (each MFMA waits for its own load only); kda50 measures it.
- kda50 STATIC_GEMM_SCHED_BAR: MLA 46.8 / 45.9 / 46.3 vs 48.5 / 47.6 / 47.9, KDA 47.1 / 47.2 / 47.4 vs 47.7 / 47.6 / 47.8, MoE 88.0-88.8 both, fused 187.8 / 188.9 vs 188.8 / 186.5: default 1.
- Fused KDA + MoE build spills (cgfused): 584 SGPR / 99 VGPR spills, 88 scratch loads (KDA-only: 46 / 0 / 0; MoE-only: 396
  / 2 / 1). The kernel's G and Maps are by-value arguments: every pointer stays live for the whole task loop.
  STATIC_ARG_RELOAD (static_schedule.py): g / maps through constant-space pointers into the kernarg segment
  (__builtin_amdgcn_kernarg_segment_ptr; &g of a by-value argument is a stack copy: scratch loads) laundered each task:
  fused 259 / 73 / 67, MoE 226 / 0 / 0, KDA 40 / 0 / 0.
- The fused projection loop's vmcnt(0) per round: the activation's LDS address (one VGPR hoisted out of the loop) was
  spilled; its reload before the loop made the loop head wait vmcnt(0). STATIC_GEMM_LANE_ASM (lane from volatile v_mbcnt in
  each step): the fused loop waits a steady vmcnt(7) (asm cgla). kda51 measures fused / KDA / MLA / MoE with both.
- kda51 (server flags) fused with ARG_RELOAD + GEMM_LANE_ASM 182.1 / 182.9 / 183.1 vs 185.4 / 187.5 / 187.6; KDA / MLA /
  MoE alone lined up with them 49.3 / 47.6, 48.2 / 47.0, 89.0 / 83.9 vs 47.3 / 47.3, 46.8 / 46.8, 88.0 / 87.5 (off by default).
- kda52 MoE item traces (timing build): fused W13 full 20.7 us (330), W2 7.1 (3080), halves 11.2 (1980) vs MoE alone 17.9
  (270), 7.6 (2520), 10.6 (1620): the units run about as fast; the fused test routes 110 experts (its x is the attention's
  hidden) vs 90 in the MoE test: ~22 % more expert work. The fused-vs-separate comparisons so far were not like for like.
  test_kda_moe_layer.py --save-moe-in / test_layer_one_gpu.py --moe-in: the MoE alone on the fused run's input (kda53).
- kda53 like for like (server flags; the MoE alone on the fused run's x / prefix: 110 experts in both): KDA 63.0 / 61.5 +
  MoE 110.4 / 112.8 = ~173.6 e2e (spans 49.6 + 98.4 = 148) vs fused with ARG_RELOAD + LANE_ASM 181.4 / 183.2 (span 165-166),
  without 187.4 / 185.9: fused +8 us (+13 without); its span is 17 us above the two spans: the KDA part (residual ends 67
  vs ~50 alone) behind a ~8 us reset of both layers' buffers before the start barrier.
- Harness with STATIC_RESET_PARITY and reset buffers the caller gives (routing_pairs): host.cuh keeps them in the arena and
  static_wait copies the launch's copy ((launch + 1) & 1) to the caller's tensor after the end event (static_host_par_mirror).
  kda54: fused / KDA / MoE all with parity.
- kda57 (server flags, parity in every build, 110 experts in both): fused with ARG_RELOAD + LANE_ASM 180.1 (span 168.2),
  without 185.6; KDA 59.9 (span 51.6) + MoE 111.9 (span 102.2) = 171.8 apart: fused +8 us, its span +14 over the two spans.
  All checks pass (the parity mirror: static_wait / static_done copy the launch's arena copy, gen from CU 0's count).
  kda58: profiles of the three.
- kda58 profiles (parity, timing build, GPU 0): fused projection last end 29.8 vs 21.2 KDA alone; its tasks per CU 11.7 /
  6.7 / 5.5 us vs 7.6 / 5.9 / 4.5 (first start 3.2 vs 1.6); o_proj 48.5 vs 38.4, residual 67.8 vs 60.7; the MoE part then
  takes ~107 vs ~105 alone. The fused loss is the KDA start (first task +4 us) and ~1 us per later projection task.
- kda59 (parity): KARG + TASK prefetch: fused 181.2 / 180.8 vs 181.5 / 179.3 without; KDA 58.5 / 58.6 vs 59.6 / 58.0; MoE 108.5 / 108.8 (same input): fused vs apart ~181 vs ~167: fused line stopped (start penalty not the arguments).
- srv4 (10-15): libraries rebuilt (SCHED_BAR) and bit-identical; ours span 13909, 6.846 ms/token, GSM8K 0.975 (195.7 tok/s); SGLang alone (base1, same session) GSM8K 0.985 (147.7 tok/s), 9.302 ms/token; 4/8 answers as base0.
- kda60 next-layer weight prefetch in the MoE kernel (92 MB, one slice per CU after its MoE tasks, default cache policy): MoE 119.2 / 119.2 / 123.4 vs 101.3 / 101.6 / 100.7 (server flags, parity): +18 us against ~3 us to gain; dropped (code removed).
- kda61/62 aggregation 1 in the MoE kernel's tail (moe_residual_agg: residual_add + next layer's agg1, one task per CU at the
  end of the least-loaded CUs): correct (float64 check), MoE 106.7 / 108.5 / 107.8 vs 99.0 / 99.6 / 99.0 without (+8.5 us).
  Profile GPU 0: agg tasks start 77.2, y stored 100.9, weights known 110.5, end 112.0 (latent_up last end 98.6): the 9.6 us
  from y to the weights (chunk dots, exchange over 8 GPUs, softmax) costs more than SGLang's 7.1 us agg kernel. Code removed.
- kda63 recurrence split (timing build, GPU 0, us after the go): conv last end 24.7; q / k / v loaded 26.9; f_b / decay done
  28.0; token loop 27.6 -> 32.2 (4.6 us, 246 instructions per token per wave); o_norm + y + count ~1.5; o_proj last end 37.9.
- kda64-66 recurrence v split (delta_rule_recurrence_layer vsplit 2 / 4: a head's v rows over 2 / 4 tasks, the o_norm sums
  of squares through a 0xFF scratch): token loop done 4 us earlier, but the o_norm exchange adds ~3 us and the o_proj tasks
  behind the split tasks start late; with the recurrence CUs isolated (isolate_plan, removed) still 47-52 vs 45-47:
  lost (code removed). STATIC_DRR_POLL (the recurrence polls its inputs, no conv count): 48.4-49.7 vs 46.8-47.6: lost
  (removed). STATIC_DRR_FOLD (decay folded into the token vectors: S^ = S / prod exp(gk); 214 instructions per token):
  -0.2..-0.5 us (kda67: 45.0 / 44.7 / 44.8 vs 45.2 / 45.6 / 45.2), checks pass.
- kda64 / 66 / 67 placement (simple_plan round-robin): the conv tasks of blocks 68-99 sat on CUs with 3 projection tasks
  (start ~21 us, inputs ready ~16): plan_one_gpu.light_plan 28:27 (conv off those CUs): conv last end 23.1-23.4 vs
  24.7-25.8. The attention residual tasks of tokens 4-7 sat behind o_proj tasks (start 37-39 us; their partials landed
  2-4 us after tokens 0-3's): o_proj grid 112 + plan_one_gpu.apart_plan 31,30 (residual one per CU, no o_proj on it).
  kda67 lined up: light + apart + o_proj 112 + FOLD 45.0 / 44.7 / 44.8, + o_proj 112 without apart 45.2 / 45.5 / 49.1,
  light alone 50.4 / 47.5 / 46.8 (base ~47.2).
- kda68 on that base: STATIC_AR_WAVE_W 45.4 / 45.5, STATIC_AR_UOUT_LATE 45.1 / 44.9, both 46.2 / 45.9, base 44.3 / 46.9;
  vsplit 2 / 4 + isolate 47.0-49.9: no further gain.
- kda69 MLA (grouped, 40 splits, die-group; lined up): o_proj grid 112 37 rows 51.5-52.0 vs 224 53.4-54.0; 1000 rows
  58.8-59.5 vs 59.4-60.1; apart_plan 38,37 no further change.
- kda70 server flags (R + parity, no line-up; one host thread launches the 8 GPUs 5-8 us apart): KDA new plan 59.0 / 58.2 /
  57.8 vs 58.0 / 58.1 / 59.3; MLA o_proj 112 67.4-67.8 vs 65.8-66.4 (37 rows), 74.3-74.5 vs 72.6-72.7 (1000 rows);
  STATIC_AR_PLAIN_OUT (out / u_out by plain stores) no change in either mode.
- srv4 traces of all 8 ranks (amd/serving rank_skew.py in the scratch): the earliest- and latest-started rank of each launch
  run equally long (KDA 54, MLA 61.6, MoE 86 us median): the ~20 us start spread is the traces' clock offsets, the ranks
  run lined up in the server. So the lined-up harness is the server's predictor; the server adds ~9 us per kernel
  (launch -> CUs, end of kernel) that the lined-up harness hides before its go (KDA 54 vs 44.8 lined-up span).
- Server-like harness mode "SL" (R + parity + STATIC_LINEUP without STATIC_EARLY: a line-up kernel, then the layer
  launched normally, e2e from the line-up's end): MoE 97.0 / 94.7 / 96.4 (rank lib 96.6). kda71 KDA SL: base 55.8 / 56.8 /
  56.7, new plan 54.2 / 54.7 / 54.8, + KARG / TASK prefetch 52.9 / 54.5 / 53.4.
- kda72 KDA SL start (timing build, CU entry stamped before the CUGEN count read): entry -> count read 0.8 us, -> resets
  stored ~1.5-2, -> first task ~1.5 (entry -> first task ~4 us); with KARG / TASK prefetch 2.4-2.8 us.
- kda73 MLA profile (lined up, grouped 40 splits, o_proj 112): at 37 rows only split 0 has rows; its 6 tasks sat on CUs
  behind q_b and the gate GEMM (start 22.8 vs 18.7 us), the empty high splits on light CUs; same at 1000 rows for the low
  splits. plan_one_gpu.rank_plan (per die, low splits on the CUs with the fewest tasks before them; --rank 35,29,34):
  kda75 lined up 37 rows 48.8 / 49.2 / 48.5 vs 51.7 / 52.2 / 51.4; 1000 rows 56.6 / 56.4 / 57.3 vs 59.2 / 59.1 / 59.2;
  4000 rows 69.2 / 69.2 / 70.6 vs 69.2 / 68.8 / 67.9; SL 1000 rows rank + prefetch 69.4 / 69.8 vs 71.4 / 71.7.
- kda74 MoE tail (lined up): 20 of 49 latent_up tasks sat behind allreduce_send tasks (start up to 89.7 vs 84.6 median,
  kda58 profile, amd/tools/trace/queue_an.py); --apart 39,37,38 (latent_up one per CU, no R send / S sum on it): 87.1 / 86.6 /
  86.2 vs 89.3 / 89.4 / 89.5; + sum_gpus apart 86.8 / 86.5 / 86.6; latent_up weights in LDS (STATIC_GEMM_LDSW=2) no change.
- srv6 (queued 05:05): KDA + KARG / TASK prefetch, MLA rank_plan + prefetch, MoE latent_up apart.
- kda76 projection rows ordered q k v f_a beta | gate (the gate's conv tasks not counted, the recurrence polling the gate
  after its token loop; plan_one_gpu urgent ordering, K-split 7 or 8): lined up (with KARG / TASK prefetch) K7 48.1 / 49.1
  / 49.2 vs 49.6 / 49.4 / 50.4, K8 49.8 / 48.3 / 50.2 vs 49.9 / 51.4 / 51.5; SL K8 57.4 / 57.0 vs 53.1 / 53.1. Profile: the
  recurrence starts at 19.6 (vs ~23) but takes 16 us (vs ~10): its loads compete with the gate tiles still streaming.
  Lost (code removed). KARG / TASK prefetch costs ~5 us in the early-launch (lined-up) mode (49.5 vs 44.7) but saves ~1 us
  in SL and in the rank library (srv6 rank lib KDA 52.0-52.6 vs 52.5-53.5): judge it in SL / rank-lib modes only.
- kda77 STATIC_GEMM_LDSW_PART (a polled task whose weight slice does not fit in LDS: the first K steps per wave into LDS
  during the wait; KDA o_proj on 112 CUs: 32 of 48 steps): KDA lined up 48.4 / 45.6 / 48.6 vs 44.6 / 44.3 / 44.8; SL with
  prefetch 54.0 / 53.8 / 55.0 vs 53.7 / 54.4 / 53.5; MLA 1000 rows no change; o_proj still ends 4.7 us after the
  recurrence (its 128 KB per CU of copies run while the projection still streams). Lost (code removed).
- srv6b (10-15 05:28; srv6's MoE rank-lib check had died in torch.distributed, passed on the rerun): span 13495 (srv5
  13567), KDA 49.6, after KDA 92.4, MLA 61.7 (59.2), 6.309 ms/token, GSM8K 0.985 at 200.5 tok/s (SGLang alone 0.985 at
  147.3). Kernel durations over all 8 ranks: KDA 50.4, MLA 62.3, MoE 84.7 us (srv4 54.1 / 61.6 / 86.4). kda78: MLA SL at
  37 / 200 rows, rank_plan and prefetch apart.
- kda78 MLA SL (grouped 40 splits, o_proj 112): 37 rows base 65.1 / 64.6 / 65.1, rank_plan 61.3 / 61.2, prefetch 65.4 /
  65.5 / 64.8, both 65.0 / 65.6 / 65.3; 200 rows base 69.1 / 68.9 / 68.6, rank 66.0 / 65.2 / 65.5, prefetch 66.7 / 66.6 /
  66.8, both 66.4 / 65.8 / 65.2: MLA takes rank_plan without the prefetch. srv5 kernels (all 8 ranks): KDA 51.2, MLA 59.9,
  MoE 85.2 us.
- kda79 MoE SL profile (apart 39): router ends 7.5, topk 14.2, latent_down (24 tasks of 128 rows x 896 K, 229 KB, ~7 us
  each) 9.7, sum_quant_send 9.5 -> 13.2, the z_q hop after it: the front's chain is latent_down -> z_q.
- kda80 latent_down --regrid 28:56,8,1 (64-row tiles, 48 tasks per GPU, before the moe task on otherwise waiting CUs): SL
  91.4 / 90.6 / 90.8 vs 92.8 / 93.2 / 92.2; lined up 85.8 / 86.0 / 85.4 vs 87.4 / 86.8 / 87.7; 28:56,14,1 worse (95.9-96.2
  SL). Profile: latent_down ends 7.1, z_q sent by 10.9, moe last end 70.7 (74.5), residual 85.6 (87.9). srv7 queued
  (06:03): + this, MLA rank_plan without the prefetch.
- srv7 (10-15 06:10): span 13194 us, KDA 49.9, after KDA 89.9, MLA 57.6 per layer, 6.256 ms/token, GSM8K 0.985 at 203.7
  tok/s (SGLang alone base4: 0.990 at 144.6, 9.639 ms/token); rank libs KDA 52.0-52.6, MLA 59.1-60.2, MoE 90.6-91.0.
- kda81 MLA SL profile (rank_plan, 37 / 200 rows): CU entry -> first task ~6.5 us (stored -> first task 3.2-3.6, KDA's
  1.5-2); chain wa 11.4 -> mla_norm 16.9 (8 tasks, 7 K parts x 2112 columns per token) -> q_b 18.8 -> absorb 22.2 ->
  attention 32.2 (37 rows) / 36.8 (200) -> vgate 36.1 -> o_proj 43.2 (its 256 KB per task streamed after o_in) ->
  residual 55.4. MLA builds with / without the prefetch flags: same VGPR / SGPR use (511 VGPR, 6 VGPR spills).
- kda82 SL, KARG and TASK prefetch apart: MLA KARG 60.5 / 60.5 / 60.6 (37 rows), 64.1 / 64.8 / 64.2 (200) vs none 61.5 /
  61.0 / 61.8, 65.7 / 65.7 / 65.8; TASK 67.5-68.1, 68.5-69.2 (+6.5); KDA KARG 53.2 / 53.4 / 52.8, both 54.3 / 53.5 / 53.4,
  TASK 54.1-54.7, none 54.4 / 53.8 / 54.4: KARG_PREFETCH alone for KDA and MLA. kda83: the same split for MoE.
- kda83 MoE SL: KARG prefetch 92.0 / 91.7 / 90.4, both 91.9 / 92.2 / 90.9, TASK 93.3 / 93.0 / 93.6, none 92.6 / 92.5 /
  93.6 -> KARG only for all three layers. kda84 SL: KDA o_proj 112 + apart 53.1-53.9 vs 224 55.4-56.5 (224 + ar-chunks 4
  + apart 55.4-56.0); MLA 200 rows o_proj 224 63.6-64.3 vs 112 64.0-64.6 (kept 112: srv5's MLA 61.8 -> 59.2 came from it).
- srv7 trace sequence (rank 1): _agg_kernel 5.4, mpk_kda_kernel 48.2, mpk_moe_kernel 91.7 us back to back (gaps 0); the
  first layer's dense MLP stays SGLang's (hgemm 29.4 + situ 4.1 + hgemm 14.1 + cross_device_reduce, median 8.7 us).
- srv8 queued (06:40): srv7 with KARG prefetch only (no TASK prefetch) in all three libraries.
- srv8 / srv8b (10-15; srv8's MoE rank-lib check hung to its 900 s timeout as srv6's did, the rerun passed: an
  intermittent hang of the 8-process check, not seen in any server run): KARG prefetch only in all three libraries: span
  13245 (srv7 13194), KDA 49.2 (49.9), after KDA 91.6 (89.9), MLA 55.5 (57.6), 6.365 ms/token, GSM8K 0.980 at 202.7 tok/s
  (SGLang alone base5 0.985 at 147.8, 8.851 ms/token). Keep: KDA / MLA KARG only, MoE KARG + TASK (srv7's).
- kda85 SL (last timed launch checked in each run): STATIC_END_RESET_RANK (the parity end-resets only on SMs without a
  task of the last node; rank / count in each list's end entry, static_schedule.task_tables) KDA 52.6 / 52.2 / 52.8 vs 53.4
  / 53.1 / 53.9; + STATIC_AR_POST_NOWAIT 53.2 / 52.3 / 52.8 (NOWAIT alone 53.5 / 53.9 / 53.8); MLA both 58.4 / 57.7 / 58.4
  vs 60.6 / 60.9 / 61.0 (37 rows), 61.0 / 61.4 / 62.7 vs 64.3 / 64.4 / 64.6 (200). A graph whose every SM holds the last
  node (MoE) keeps every SM in the ranking.
- STATIC_MOE_START_RESET (core.cuh parity_resets_slice): each moe_experts task does its SM's slice of the next launch's
  parity resets before its z_q wait, kernel_end skips it (SL profiles: MoE e2e - span 8.8 us vs KDA's 6.2; its 4.3 MB of
  end resets ran after the last tasks). Build: 2 VGPR spills (12 B scratch). kda86 SL: 93.0 / 91.3 / 92.5 vs 91.7 / 91.1 / 91.9: no gain, code removed. srv9 queued (10:51): KDA +
  END_RESET_RANK, MLA + END_RESET_RANK + AR_POST_NOWAIT, MoE back to KARG + TASK.
- The intermittent rank-lib check hang (srv6 MoE, srv8 MoE, srv9 MLA; reruns pass; output always stops before the first
  eager launch's comparison): test_rank_lib.py launched right after create_layer with no barrier, so a fast rank could
  write into a peer's exchange region before that peer's own set-up cleared it. Now torch.cuda.synchronize + dist.barrier
  after create_layer; sglang_k3 prepare ends with tp.barrier() (SGLang's CPU-group barrier) for the same reason.
- srv9 / srv9b (10-15 11:23; srv9's MLA check hung, srv9b with the barrier passed: MLA rank lib eager 56.2-57.0): KDA +
  END_RESET_RANK, MLA + END_RESET_RANK + AR_POST_NOWAIT, MoE KARG + TASK: span 13080 us (best; srv7 13194), KDA 48.7,
  after KDA 90.5, MLA 54.0 per layer, 6.432 ms/token (accept 2.646), GSM8K 0.980 at 202.7 tok/s (SGLang alone base6: 0.985
  at 147.4, 9.624 ms/token). Job file ~/mpk-amd/logs/job_srv9.sh (+ job_srv9b.sh).
- kda87 MLA SL STATIC_MLA_NORM_SPLIT (norm groups 2 split at the q_a / kv boundary, no exchange): 37 rows 58.6 / 58.7 /
  58.0 vs 58.4 / 58.2 / 59.0, 200 rows 62.5 / 62.0 / 61.9 vs 61.7 / 62.0 / 61.6: no gain, code removed.
- kda88b / kda89 MoE SL: residual_add on 64 tasks (residual_add_layer grid_choices 32 / 64 / 128 / NSM; --regrid
  40:64,1,1:36,37,38,39) + STATIC_END_RESET_RANK (the other ~190 SMs take the parity end-resets): 89.4 / 89.4 / 90.0, 89.7 /
  89.5 / 89.1 vs base 91.4 / 90.4 / 91.9; 64 tasks alone 90.6 / 90.9 / 90.4; 32 tasks + ER 90.0 / 89.5 / 90.6. srv10
  queued (11:45): srv9 + this.
- srv10 (11:48): span 12951 us (best), KDA 49.1, after KDA 88.8, MLA 53.7, 6.453 ms/token, GSM8K 0.980 at 205.5 tok/s
  (SGLang alone base7: 0.990 at 147.7, 9.544 ms/token); rank libs KDA 50.8-51.9, MLA 56.5-57.2, MoE 89.8-90.8.
- kda90 MoE SL router regrid (topk sums 7 K parts instead of 14): 26:14,7,1 89.3 / 89.5 / 89.3 vs 89.2 / 89.4 / 88.9;
  26:28,7,1 92.1-93.1: no gain.
- kda91 MoE SL profile (srv10's config): the shared expert's S send (node 35) sat behind moe_experts tasks on 18 CUs
  (start up to 76.1, S summed by 81.0; residual_add ends 87.0, latent_up 81.9). kda92: the S send right after the shared
  down GEMM on its 14 packed CUs (follow pass): 97.8 / 99.6 / 98.0 vs 89.5 / 89.8 / 89.1 (those CUs entered moe ~10 us
  later, its last unit 84.8): lost, pass removed.
- kda93 / kda94 W2 knockouts (timing only): W2 item median 7.2 us; without its R stores 4.0, but that build also drops the
  MFMAs (results unused) and so their waits for the weight loads: not a measurement of the stores. With plain (L2) R
  stores instead of device-scope ones: 7.2 (same). Without the h_q poll: 7.1. The W2 items are bound by their loads'
  latency at this depth (~23 MB in flight), not by the R stores or the h_q checks. Knockout code removed.
- kda95 KDA SL profile (srv10 config): projection 21.2, conv 24.3, recurrence 34.0, o_proj 40.1; residual partials stored
  44.6-45.7, exchange 46.3, end 50.4. kda96 SL residual-tail flags: base 52.5 / 53.1 / 52.4; WAVE_W 53.8 / 54.2 / 53.0;
  UOUT_LATE 53.5 / 53.0 / 53.3; PLAIN_OUT + POST_NOWAIT 54.0 / 52.2 / 52.9; all three 52.7 / 52.8 / 53.3: no gain.
- kda97 MoE SL W2 pipelines (srv10 config otherwise): LATE_WAIT 90.4 / 90.7 / 90.0, DEEP (two items' loads in flight)
  92.7 / 91.8 / 92.0, ADAPT 91.8 / 92.0 / 91.5: LATE_WAIT stays.
- bw2 (expert_unit.hip, 1 GPU, 110 experts, each kernel incl. its launch ramp): nt weight loads W13 3.9 TB/s, W2 3.3 (8-tile
  items) / 3.5 (16) / 2.3 (4, plain) / 1.66 (28, plain); plain loads ~2.2-2.3 for both; W2 without its small loads or with
  sc1 h_q / y: no change. In the layer W2_TILES 16 spills 234 VGPRs (756 B scratch): not usable.
- kda98 MoE SL STATIC_W13_TAIL_SPLIT (srv10 config): 0 -> 89.7 / 89.2 / 89.7, 4 -> 90.3 / 89.6 / 90.0, 8 -> 92.2 / 92.1 /
  90.6, 12 (default) -> 90.0 / 90.4 / 90.1: 0 is ~0.5 us better (next server build).
- kda99 fused KDA + MoE in SL with 10-15's passes (test_kda_moe_layer.py --today, ARG_RELOAD + LANE_ASM, END_RESET_RANK,
  KARG): 163.6 / 164.1 / 163.9 (with TASK prefetch 166.6 / 165.6 / 165.9) vs KDA 52.3 / 52.9 / 52.7 + MoE 92.3 / 90.3 / 90.6
  apart (the fused MoE routes ~110 experts vs 90: +8 like for like on 10-15 morning): the fused line stays closed.
- srv11 (13:20): srv10 + W13_TAIL_SPLIT=0: span 12917 (best), KDA 49.0, after KDA 88.1, MLA 54.1, 6.197 ms/token, GSM8K
  0.985 at 205.1 tok/s (SGLang alone base8: 0.985 at 147.8, 8.893 ms/token); rank libs MoE 89.1-89.4.
- kda100-102 MLA SL attention (rank_plan, srv10 config): at 37 rows the 6 split-0 tasks start 18.7-22.0, queries land
  24.3 (absorb ended 21.7), the chunk is in LDS at 26.3, the task ends 31.3; at 1000 rows (1 chunk per split) loop done
  ~30, all splits counted 35.3, end 38.5. One 64-row chunk's compute (kT transpose, scores, softmax, O) takes ~4-5 us
  warm: a timing-only second pass over the same chunk (STATIC_MLA_KO_TWICE, removed) took 6.3 us incl. one reload of
  this step's rows. Not cold code; the attention's inner loop is the next MLA item (plus the 5 us count + 3 us merge at
  1000 rows).
- MLA attention asm (grouped, srv10 config build): the O MFMA block reads each A operand into one reused register quad and
  waits lgkmcnt(0) before every MFMA (16 waits for 16 MFMAs; LDS addresses kept in AGPRs; the kernel at 511 VGPRs).
  STATIC_MLA_PRE_LATE (the next chunk's rows loaded at its own iteration): the same code in those blocks. STATIC_MLA_LDS_BATCH
  (a step's operands into an array, sched_barrier, then the MFMAs): the O block waits 3 times, but 16 VGPR spills (124 B
  scratch); kda103 SL 37 rows 60.6 / 60.2 / 61.3 vs 58.4 / 58.8 / 58.0, 1000 rows 69.2 / 68.6 / 69.4 vs 65.3 / 65.5 / 65.4:
  lost, both removed. The attention needs a register-budget rework before its loads can overlap.
- gsm11 (srv11 libraries, full GSM8K 1319 questions): ours 0.955 at 201.0 tok/s, SGLang alone 0.956 at 141.8 tok/s (1.42x);
  span 12901, KDA 48.5, after KDA 88.6, MLA 53.4.
- kda104 SL STATIC_ARG_RELOAD (MLA build: SGPR spills 163 -> 56, scratch 68 -> 0 B): MLA 37 rows 56.9 / 57.2 / 57.3 vs 58.2 /
  58.1 / 57.9, 1000 rows 63.6 / 64.7 / 64.6 vs 65.1 / 65.7 / 65.2 (+ LANE_ASM no better); MoE + LANE_ASM 89.0 / 88.7 / 88.7
  vs 90.1 / 89.6 / 89.6; KDA + LANE_ASM 53.3 / 53.6 / 52.7 vs 52.2 / 52.3 / 52.6 (not taken). srv12 queued (13:57).
- srv12 (14:10): srv11 + ARG_RELOAD (MLA, MoE) + LANE_ASM (MoE): span 12724 (best), KDA 48.8, after KDA 86.7, MLA 53.2,
  6.181 ms/token, GSM8K 0.980 at 207.2 tok/s (SGLang alone base9: 0.990 at 147.7); rank libs MLA 55.2-56.4, MoE 88.1-88.5.
- kda105 KDA SL ARG_RELOAD alone 52.2 / 52.8 / 52.6, LANE_ASM alone 52.6 / 53.0 / 52.6 vs 52.2 / 52.5 / 53.3: no change.
- kda106 KDA SL profiles, o_proj with / without the partial LDS preload (LDSW_PART, re-applied for this, removed again):
  recurrence 33.5 / 33.4 -> o_proj last end 38.3 / 39.4 (4.8 vs 6.0 us), residual end 48.9 / 49.8. o_proj's time after y
  is mostly the y poll, its MFMAs and its pushes to the 8 GPUs (the hop), ~1 us is weight streaming.
- MLA attention as its own function (MPK_NOINLINE_TASKS=mla_attn, + ARG_RELOAD): 1492 B private segment, 30 VGPR spills,
  its O block still one lgkmcnt(0) per MFMA: not pursued.
- W2 tickets: consecutive tickets already go to different waves reading adjacent 24 KB tiles at the same time (one
  contiguous stream across waves), so pairing tickets per wave would not change the memory pattern: not pursued.
- kda107 (10-15, SL, MLA server config + ARG_RELOAD, 30 launches x 3 rounds, 1 GB flushed): STATIC_MLA_O_BATCH (O-block kT
  operands read 2 or 4 at a time + sched_barrier; O-block lgkmcnt(0) waits 17 -> 7 / 4, spills unchanged) is no win: medians
  37 rows base 57.0 / OB4 57.1 / OB2 56.6, 1000 rows 64.0 / 63.9 / 64.2 us (spread ~1 us). The O-block LDS waits are not on
  the critical path. Reverted (file md5 8406fde4 on all trees).
- kda108 (SL, MLA server config + ARG_RELOAD): STATIC_MLA_Q_LATE (mla_attn_grp polls the queries after the first chunk is in
  LDS and transposed; kda100: queries landed 24.3, chunk in LDS 26.3 at 37 rows) 1000 rows 62.8 / 62.9 / 62.8 vs 64.8 /
  64.0 / 64.5 us, 37 rows 56.4 / 56.6 / 56.8 vs 56.6 / 56.7 / 57.4: -1.6 us at 1000 rows, kept for the next server run.
  Default and srv12-config builds md5-identical (macro, not a lambda: the lambda changed the default build).
- mla_norm (kda100 trace): the projection ends 10.8, the norm 16.6 (8 tasks, one per token): each task's 3 rounds of K-part
  loads (528 4-column vectors / 256 threads, 7 parts) polled one round after the other. STATIC_MLA_NORM_ONEPOLL: all
  3 x 7 loads per thread in flight, one poll (no new spills). kda109 (SL, 3 rounds): ONEPOLL 37 rows 56.6 / 57.4 / 57.1,
  1000 rows 64.6 / 64.6 / 65.0 vs base 57.7 / 57.0 / 57.0, 64.7 / 63.6 / 64.2: no gain (the norm's 5.8 us is not its
  sequential polls), removed; ONEPOLL + Q_LATE 56.3 / 56.3 / 56.2, 63.2 / 62.8 / 62.7 (= Q_LATE alone).
- kda110 KDA SL profile (srv12 config), recurrence split (GPU 0, us after the go): conv end 23.7 -> q / k / v, f_a, beta
  loaded 25.8 (stamp set 1; counter + one round trip) -> f_b, decay and the FOLD prefix products done 27.7 (1.9 us) ->
  token loop ~4.6 us (stamp set 2: loop start 28.7, tokens 0-3 done 31.0, i.e. ~0.57 us per token) -> o_norm, y, count:
  end 33.4. STATIC_DRR_CHUNK (the T tokens in chunk form; the state is not written back, so only o is needed: S0 a_m and
  S0 c_m for all tokens at once, 2 T T Gram terms, an 8-step substitution per v row; default build md5-identical, 342
  VGPR, no scratch) is measured in kda111.
- kda111 KDA SL STATIC_DRR_CHUNK: y relRMS vs float64 1e-4 .. 0 (sequential: the same range), all checks pass; e2e 57.8 /
  58.4 / 58.3 vs 52.8 / 52.5 / 52.8 us (+5.5). Profile: q / k / v loaded 25.2 -> loop done 34.3 (9.1 us vs ~6.5 for f_b +
  sequential loop). Not the instruction count (+2.2k lines in the kernel; 231 v_accvgpr moves at 342 VGPR, SGPR spills 226
  vs 99); kda114 stamps the chunk steps (set 3) to find the slow one.
- kda112 MLA SL profile with mla_norm stamps (37 rows, GPU 0): projection last end 11.3 (its K-part stores issued, not
  landed), the norm's K parts landed (thread 0) 14.6-14.7, output stores issued 16.1-16.4 (incl. a w_q / w_kv load round
  trip after the sums of squares), end 17.0-17.2. STATIC_MLA_NORM_WPRE (those weights loaded at the task's start) in
  kda115.
- kda113 MLA SL (+ Q_LATE): o_proj grid 224 (98 KB slice, LDSW: in flight during the wait) 37 rows 58.2 / 57.8 / 57.3,
  1000 rows 64.1 / 65.0 / 64.8 vs 112 56.6 / 56.8 / 57.0, 63.0 / 63.4 / 62.7: 112 stays.
- kda115 MLA SL (+ Q_LATE): STATIC_MLA_NORM_WPRE 37 rows 55.6 / 55.6 / 56.2 vs 57.0 / 56.8 / 56.8, 1000 rows 61.9 / 62.6 /
  62.0 vs 62.8 / 62.8 / 63.0: -1.1 / -0.7 us, kept with Q_LATE for the next server run (norm stores issued 15.7 vs
  16.1-16.4). Timeline (37 rows, GPU 0, last task per node): projection 10.9, norm 16.5, q_b 18.8, absorb 21.6, attention
  30.4, vgate 34.2, o_proj 40.3, attn_residual 57.0 (stamps 50.1 / 55.8).
- kda116 KDA SL: CHUNK v2 (Gram loads 8 at a time, rotated per lane against bank conflicts; next token's vectors
  prefetched; substitution operands loaded together) 57.0 / 56.9 / 55.5 vs 53.0 / 53.4 / 52.8; its S0 products + Gram
  still 4.8 us (28.7 -> 33.5): VALU-bound (a wave64 VALU op takes ~4 cycles on the 16-lane SIMD; ~1.1k ops per wave).
  STATIC_DRR_PIPE 53.9 / 53.4 / 54.4: lost. CHUNK v3: S0 [a | c] by fp32 MFMA 16x16x4 (state loaded in the A layout,
  64 MFMAs per wave), products to LDS, one thread per v row for the substitution: kda118.
- kda117 KDA SL residual stamps (per GPU, from its o_proj's last end): set 0 partials landed +4.4, weights known +10,
  end +11; set 4 own stores landed (vmcnt 0) +7.7, exchange done +8.6, end +13. So the 5 us between landing and the
  weights is mostly the task's after-u work and its device-scope stores landing (~3.3 us), then the exchange (~0.9);
  the weights themselves are not it (WAVE_W lost three times: kda42, kda68, kda96). Not pursued further for now.
- kda118 KDA SL CHUNK v3 (fp32 MFMA 16x16x4f32 for S0 [a | c], 311 VGPR, no spills): y relRMS 1e-4 .. 0 (as sequential),
  e2e 52.2 / 52.7 / 52.1 vs 52.3 / 52.9 / 52.5 (-0.3). Profile: f_b 28.4 -> products + Gram 29.6 -> substitution 30.4 ->
  end 32.0-32.5 (sequential: loop done ~32.3, end 33.4): the recurrence ~1 us shorter, but o_proj (activation in LDS 33.2
  median, end 37.4) takes 4.2 us after y; kda119 stamps o_proj's MFMAs (gemm_tile stage stamp 1, non-NORM tasks).
- kda119 profiles with gemm_tile stage stamp 1 (MFMAs done, non-NORM tasks): KDA o_proj activation in LDS 34.0, MFMAs
  done 37.6, end 38.3 (GPU 0 medians); MLA o_proj 35.8 / 39.4 / 40.0. Both o_proj have K = 1536 (KDA: 12 heads x 128), so
  at grid 112 the slice is 196 KB: LDSW=3 never applies (no LDS DMA in either build); 8 of 48 K steps are in registers
  when the activation lands, the other ~168 KB per CU streams after it (~3.6 us). STATIC_GEMM_POLL_ALLREG (polled tasks
  whose slice does not fit in LDS hold all K steps in registers, issued at the start): KDA build 402 VGPR, no spills;
  kda120 A/B.
- kda120 SL: STATIC_GEMM_POLL_ALLREG KDA (+ CHUNK) 52.7 / 52.8 / 52.8 vs 52.5 / 52.4 / 52.3; MLA (+ Q_LATE + WPRE) 37 rows
  56.9 / 56.6 / 56.9 vs 55.7 / 55.3 / 55.8, 1000 rows 64.6 / 64.5 / 64.4 vs 61.8 / 62.0 / 62.1: lost, removed. Its timing
  build looked better (KDA o_proj MFMAs 2.6 us after the activation vs 3.6, residual landed ~1.5 us earlier), as
  LDSW_PART's did (kda77 / kda106): weight prefetch for o_proj wins in the timing build and loses in the normal one.
- srv13 (15:45, job_srv13.sh = srv12 + MLA Q_LATE + NORM_WPRE, KDA DRR_CHUNK): span 12654 (best; srv12 12724), KDA 48.1,
  after KDA 87.2, MLA 50.3 per layer, 6.381 ms/token (accept 2.653), GSM8K 0.985 at 207.4 tok/s (SGLang alone base10:
  0.985 at 147.3); rank libs KDA 50.6-51.0, MLA 53.4-53.9, MoE 88.4-89.0, all bit-identical to the harness.
- KDA projection (kda119, GPU 0): 188 CUs run 3 tasks of 128 KB, 68 run 2 (700 tasks); task durations 7.4 / 5.6 / 4.7 us
  (the first waits ~2 us for x); 2-task CUs done at 15.2, the rest ~20.8: aggregate ~4.4-5 TB/s vs 5.7 (gemm_load) /
  6.38 (read_bw). No even split exists (7168 = 2^10 x 7, 400 row tiles).
- kda121 MoE SL profile (srv12/13 config, GPU 0): front GEMMs 8.0, route 14.3, shared W13 16.5, situ 23.3, shared down
  (node 34, --pack 34:4: 4 tasks on each of 14 CUs) 16-56.4, expert workers end p50 60.0 / p90 63.1 / max 67.1,
  allreduce 70.0, sum_gpus 73.6, latent_up 78.7, residual_add 80.5 (e2e 88.2, span 79.5). The 12 latest workers are the
  14 node-34 CUs: they enter at ~55 and take W2 items until 65-67 (others end ~58-60). STATIC_W2_LATE_SKIP (exists;
  lost 10-13 in EARLY mode on an older config) re-tested in SL: kda122.
- kda122 MoE SL: STATIC_W2_LATE_SKIP 12 / 14 / 8: 89.1 / 90.8 / 88.6, 88.9 / 89.4 / 88.9, 89.0 / 88.5 / 88.8 vs base 89.1 /
  88.5 / 88.5: no gain. kda123 profile (SKIP 12): the late workers still run 55 -> 62 (no W2 item, ~7 us anyway), experts'
  last end 62.7-65.7 (base 67-68), shared allreduce 35 ~66 (70), but routed allreduce 37 67-70, sum_gpus ~72, latent_up
  ~78, residual_add ~80 as before: the tail after the experts (~16 us) bounds the layer.
- kda124 SL STATIC_GEMM_STAGES=16 (default 8): MLA (+ Q_LATE + WPRE, 37 rows) 54.7 / 54.5 / 54.4 vs 56.2 / 55.9 / 56.2
  (-1.6); KDA (+ CHUNK) 51.6 / 52.4 / 51.8 vs 52.1 / 52.8 / 51.7 (-0.3; the projection's task times unchanged: 7.2 / 5.7
  / 4.6). kda41's loss (10-14) no longer holds. MLA's o_proj (48 K steps, streamed after o_in) has 16 ahead. kda125:
  MLA 24 vs 16, MoE 16 vs 8.
- kda125 (its MLA runs lacked the B flags: KARG / END_RESET_RANK / AR_POST_NOWAIT / ARG_RELOAD, ~4 us slower; compare
  only within): MLA GEMM_STAGES 24 vs 16: 37 rows 58.3 / 60.2 / 59.0 vs 59.0 / 58.8 / 59.4, 1000 rows 65.7 / 65.5 / 65.1
  vs 66.1 / 65.8 / 66.3: equal, 16 kept. MoE GEMM_STAGES 16 vs 8: 88.7 / 88.4 / 89.2 vs 89.6 / 89.2 / 88.6: noise, MoE
  stays 8. Next server run: MLA + KDA STATIC_GEMM_STAGES=16.
- srv14 (16:30, job_srv14.sh = srv13 + GEMM_STAGES=16 for KDA and MLA): span 12607 (best), KDA 47.7, after KDA 86.9,
  MLA 50.9, 6.402 ms/token (accept 2.565), GSM8K 0.980 at 208.8 tok/s (SGLang alone base11 0.985 at 147.3); rank libs
  KDA 50.3-51.6, MLA 53.1-54.2 (srv13 53.4-53.9), MoE 87.7-88.5.
- kda127 amd/bench/lds_tr_check.hip (ds_read_b64_tr_b16, one wave): in each 16-lane group lane s gives the 4 b16 at its
  address, M[s][0..3]; lane i gets M[4 e + i / 4][i % 4], e = 0 .. 3 (the same for any addresses). So lane s at row j0 +
  s / 4, columns d0 + 4 (s % 4): lane i gets rows j0 .. j0 + 3 of column d0 + i; two reads = the 16x16x32 MFMA's A
  fragment of a column-major operand.
- kda126 MLA SL attention stamps (Q_LATE + WPRE + 16 stages, 37 rows, GPU 0): chunk in LDS 24.6 -> transpose done and
  queries polled 26.7 (queries ready ~24.3: the transpose, ~2 us, is on the path) -> scores 27.6 -> softmax 28.6 -> end
  30.3. STATIC_MLA_TR: no transpose, the O MFMAs' A operands by two ds_read_b64_tr_b16 from the chunk (no spills):
  kda128 checks 37 / 1000 / 8000 rows and A/B.
- kda128 MLA SL STATIC_MLA_TR (+ Q_LATE + WPRE + 16 stages): checks pass at 37 / 1000 / 8000 rows; 37 rows 53.1 / 53.3 /
  52.8 vs 55.0 / 54.8 / 55.1 (-2.0), 1000 rows 60.6 / 60.8 / 60.7 vs 62.1 / 61.8 / 62.2 (-1.3): kept (srv15). Profile
  (37 rows, GPU 0): absorb 21.2, attention scores 27.1, softmax 28.3, end 30.1, vgate 34.3, o_proj 40.2, residual 51.1.
- srv15 (17:00, job_srv15.sh = srv14 + MLA_TR): span 12617 (srv14 12607: same within run spread), MLA 50.0 (50.9) per
  layer, MLA parts 1153 vs 1173 us, KDA 48.1, after KDA 86.6; 6.505 ms/token (accept 2.709); GSM8K 0.980 at 207.9 tok/s
  (SGLang alone base12 0.975 at 147.3); rank libs KDA 49.9-50.8, MLA 52.2-52.6 (srv14 53.1-54.2), MoE 88.7-89.5.
- kda129 amd/bench/poll_depth.hip (1 GPU; WG 0 stores a value at a random time, WG 1 polls it at agent scope; delay =
  stamps' difference incl. the store): one poll at a time (s_sleep 1) 500 ns median (p90 620) idle, 420 with 254 WGs
  streaming HBM; 2 or 4 polls in flight 400-540 at no spacing, more with spacing (S 4 / 8 / 16: 540-2300). A flag hop is
  ~0.5 us and more polls in flight do not shorten it: the 2-3 us hops in the layers are the payloads' stores / loads.
- Aggregation 1 fused again considered (10-15 evening): attn_residual's math is SGLang's _agg_kernel (prefix 0, input =
  the stream), but kda46/47 already measured it inside the KDA kernel: +15 us, its loads and exchange stretched to ~17 us
  by the projection's streaming (each dependent round trip ~3 us under that load). Not retried.
- kda130 KDA SL recurrence split (CHUNK + 16 stages, GPU 0): conv end 24.1, q / k / v loaded 26.1, norms + f_b + decay
  done 27.4-28.0 (stamp set 5), FOLD done 28.1-28.6 (0.6 us), end 31.7. STATIC_DRR_EARLY_FB (f_a, beta polled before
  the conv's count; f_b and the decay before q / k / v) in kda132; STATIC_MLA_SMX4 (softmax: 4 queries per wave at once,
  16-lane DPP reductions) in kda131.
- kda131 MLA SL (+ TR) STATIC_MLA_SMX4: checks pass 37 / 1000 / 8000; 37 rows 53.0 / 53.0 / 52.3 vs 53.6 / 53.6 / 53.3,
  1000 rows 59.6 / 59.5 / 59.1 vs 61.0 / 60.0 / 60.2 (-0.6 / -0.9); softmax step 0.7 vs 1.2 us (scores 27.0 -> softmax
  27.7 -> end 29.4): kept for the next server run.
- kda132 KDA SL STATIC_DRR_EARLY_FB (checks pass): 53.1 / 52.6 / 52.8 vs 51.9 / 52.0 / 52.3 (+0.7): q / k / v landed
  later (28.2 vs 26.1): f_a / beta are not ready before the conv's count. Removed.
- srv16 (18:40, job_srv16.sh = srv15 + MLA_SMX4): span 12557 (best), KDA 48.0, after KDA 86.5, MLA 48.7, 6.235 ms/token
  (accept 2.719), GSM8K 0.990 at 207.9 tok/s (SGLang alone base13 0.985 at 147.7); rank libs KDA 50.6-51.2, MLA
  51.6-52.2, MoE 87.5-88.3.
- kda133 SL profiles (all kept flags, GPU 0): MLA 37 rows wa 10.0, norm 16.1, q_b 18.2, absorb 21.3, attention 29.2,
  vgate 33.0, o_proj 38.8, residual 50.9; 1000 rows attention 35.9, vgate 39.0, o_proj 45.0, residual 56.6. KDA
  projection 21.8, conv 24.0, recurrence 32.5, o_proj 38.0, residual 48.8. mla_norm: wa tasks all end 9.8-10.0, the
  norm's K parts landed 13.9-14.1: each of its 8 tasks (one per token) reads 7 x 2112 fp32 = 59 KB (~2.4 us at one CU's
  ~25 GB/s), so the one-poll change could not help. kda134: --norm-groups 2 / 4 again (lost on 10-14, kda32).
- kda134 MLA SL --norm-groups (all kept flags): 4 groups 37 rows 52.2 / 52.7 / 53.2, 1000 rows 59.6 / 59.9 / 59.6; 2
  groups 54.7-55.1, 59.9-61.0; 1 group 53.4 / 53.0 / 53.6, 59.6 / 59.9 / 59.5: not taken (-0.6 at 37 rows only). Profile
  (2 groups): norm end 15.9 vs 16.1: the K parts are visible late (the projection's stores land ~3 us after its tasks
  end, under the q_b / g_proj streams), not the norm's reading.
- MoE read speed over time (kda121 item trace, bw_an.py, GPU 0): W13 ~19-36 at 6.5-6.9 TB/s; W2 36-44 4.2-4.5 (ramp,
  W13 tail), 44-50 4.8-5.5, then the drain 52-60: 3.9 -> 0.2 TB/s. W2 moves 66 MB (2520 items x 26 KB) in ~24 us (~2.75
  TB/s average); a W2 item takes 7.8 us median (6.0-9.1) for 26 KB, so the drain is about one item long. R_BLOCKS is
  already on (allreduce per 128-column block; latent_up still needs all of R). kda135: W2_TILES 4 (shorter items) in SL.
- kda135 MoE SL STATIC_W2_TILES=4: 95.2 / 95.1 / 98.4 vs 88.7 / 88.1 / 88.7 (+7): the per-item fixed cost dominates.
- gsm16 (srv16 libraries, full GSM8K 1319 questions): ours 0.955 at 204.9 tok/s, SGLang alone 0.955 at 142.3 tok/s
  (1.44x, same accuracy); span 12596, KDA 48.1, after KDA 86.7, MLA 49.0.
- kda136 KDA SL STATIC_DRR_FOLD_SCAN (FOLD's products in the f_b MFMA epilogue by a row_shr scan; checks pass): 53.0 /
  53.8 / 53.2 vs 51.5 / 52.2 / 51.5 (+1.5); f_b + fold 2.5 us vs 1.9. Removed.
- kda137 SL GEMM_STAGES 24 vs 16 (full flag sets): MLA 37 rows 52.7 / 52.4 / 52.8 vs 52.8 / 52.9 / 52.8, 1000 rows 60.3 /
  60.0 / 60.2 vs 59.8 / 59.6 / 59.8; KDA 53.0 / 52.5 / 52.2 vs 52.0 / 52.2 / 52.2: 16 stays.
- kda138 MoE SL profile, topk_route per task (8 tokens, all GPUs alike): router GEMM ends ~7.4, its K parts landed ~9.5,
  keys +0.4-1.0, the top-16 pick 2.2-3.9 us, end ~13.9; z_q (sum_quant_send) ends ~10.6-11.3, so the pick holds the
  experts' start by ~2.5 us. kda139: STATIC_TOPK_FAST (exists, never A/B'd in the layer).
- kda139 MoE SL STATIC_TOPK_FAST: 88.0 / 87.9 / 88.5 vs 90.2 / 89.4 / 88.7 (within run spread); its profile shows the pick
  unchanged (3.2-3.9 us): not taken. kda140 amd/bench/topk_select.hip (896 experts, K 16, one CU): binary search 6.28 us,
  candidates 4.76, radix 2.72 (random) / 3.52 (sigmoid + bias); a register bitonic network over 896 keys counts out to
  ~900 instructions per wave, no faster: the pick stays.
- kda141 MLA SL 1000 rows attention (all kept flags, stamp set 2, GPU 0): 96 tasks (16 splits x 6 groups, one chunk
  each), loop done 24.4 / 24.5 / 26.6, all splits counted 27.5 / 31.0 / 31.8, end 31.5 / 34.2 / 35.0 (absorb 20.6,
  vgate 38.2): ~4.4 us from the slowest split's loop to its group's count (32 KB fp32 partial stored, the count after the
  stores complete, the poll), then ~3 us merging.
- kda142 / kda143 MLA SL --kv-splits (all kept flags; e2e, 3 rounds): 37 rows 16: 51.3 / 51.3 / 51.1, 24: 51.3 / 51.2 /
  51.0, 40: 52.5 / 52.3 / 52.7; 1000 rows 57.3-57.8 / 56.6-57.4 / 59.3-59.9; 4000 rows 68.3-68.9 / 64.1-65.4 / 70.9-71.5;
  8000 rows 85.1-85.9 / 76.8-77.2 / 75.7-76.0 (8 splits at 1000: 59.0-59.7). At 1000 rows 16 and 40 both run 16 splits:
  the gain is the plan's 96 (or 144) attention tasks instead of 240 and the smaller combine loops. 24 taken (srv17).
- srv17 (job_srv17.sh = srv16 + MLA --kv-splits 24): MLA 46.9 per layer (srv16 48.7), MLA parts 1079 (1126) us; span
  12568 (srv16 12557: equal within the spread; KDA 48.0, after KDA 86.8); 6.193 ms/token (accept 2.808); GSM8K 0.980 at
  209.2 tok/s (SGLang alone base14 0.985 at 147.2); rank libs KDA 50.2-51.0, MLA 48.9-50.1 (srv16 51.6-52.2), MoE
  88.0-88.6.
- kda144 MLA SL grids (24 splits): base (q_b 144, g_proj 24x7) 37 rows 50.8 / 51.4 / 51.3, 1000 rows 57.2 / 56.7 / 57.8;
  g_proj 12x7 53.1-53.5 / 56.8-57.6; q_b 72 51.7-52.2 / 59.9-60.6; both 50.2-50.9 / 59.0-59.3: base stays.
- kda145 SL --ar-chunks 4 (1792-column residual chunks; checks now pass): KDA 54.5 / 54.2 / 54.0 vs 52.4 / 51.4 / 51.7,
  MLA 37 rows 53.6 / 53.4 / 53.4 vs 50.8 / 51.3 / 50.4: lost.
- kda146 SL --ar-chunks 14 (512-column chunks, 112 residual tasks; checks pass): KDA 52.5 / 52.3 / 52.5 vs 51.8 / 52.5 /
  52.9, MLA 37 rows 50.1 / 50.2 / 51.2 vs 50.9 / 50.7 / 51.2: no change, 7 chunks stay.
- kda147 MLA SL --kv-splits 20 / 24 / 28 / 32: 37 rows all 50.2-51.6; 1000 rows 57.3-57.6 / 56.4-57.3 / 56.3-56.8 /
  56.3-56.8; 4000 rows 68.6-69.4 / 64.6-64.8 / 67.0-67.7 / 66.3-67.2: 24 stays.
- kda148 MoE SL plan sweep: pack 34:4 + residual_add 64 (current) 89.5 / 88.7 / 89.1; pack 34:2 89.5 / 89.1 / 89.8; pack
  34:8 107-108; residual_add 32 88.8 / 89.4 / 89.0; residual_add 128: the plan check rejects it (task lists can
  deadlock). Current stays.
- kda149 MLA SL placement at 24 splits: --rank only / --die-group only / both: 37 rows 50.5-51.1 all, 1000 rows
  56.8-58.4 all, 4000 rows 64.6-66.3 all: equal; both stay.
- attn_residual exchange through the die's L2 (STATIC_AR_L2X, the layer's l2x option, die_plan on the residual node;
  10-15 night): the die check worked (no die placement: device fallback, checks pass), the L2 path hung. kda154
  amd/bench/l2_handoff.hip (a plain store by one CU, polled by a CU on the same XCD): load sc0 and buffer_inv sc0 + load
  never see it (this CU's L1 keeps the stale line); buffer_inv sc1 + load 560 ns, load nt 480, load sc1 (device scope)
  480 ns median. A device-scope hand-off is as fast as one through the die's L2: the residual's ~3.3 us is not hand-off
  latency. All L2X code removed (files back to their state before; the bench stays).
- kda155 KDA SL residual stamps, per GPU from its o_proj's last end: stamp set 5: partials landed +2.2-4.2, after-u
  values issued +1.3-1.7 later, end +8.4-10.8; set 4: own stores done +4.0-6.4, exchange done +0.6-0.8 later; set 0:
  weights known +8.2-10.7, end +0.8-0.9 later. So ~3.5 us go from the exchange to the weights (every thread: 17 scores,
  the softmax, h's sum of squares over the Gram terms, one after the other). STATIC_AR_LANE_W (per wave, lanes in
  parallel, readlane broadcast, no barrier) in kda156.
- kda156 SL STATIC_AR_LANE_W (checks pass): KDA 51.2 / 51.4 / 51.4 vs 51.9 / 52.2 / 51.8 (-0.6), MLA 37 rows 49.5 / 48.8 /
  49.2 vs 50.8 / 50.6 / 50.2 (-1.4); landed -> weights known ~3.2 us (was ~5.4). Kept: srv18.
- srv18 (21:15, job_srv18.sh = srv17 + AR_LANE_W in KDA and MLA): span 12416 (best; srv16 12557), KDA 46.8, after KDA
  87.2, MLA 44.7 per layer; 6.255 ms/token (accept 2.569); GSM8K 0.980 at 209.9 tok/s (SGLang alone base15 0.985 at
  147.5); rank libs KDA 49.6-50.6, MLA 47.3-47.9, MoE 87.9-88.6.
- KDA conv tasks (kda133, GPU 0): median end 20.4, the last are q / k / v blocks 68-71 (v of heads 10-11) at 23.6-24.0 and
  the gate blocks 72-95 at ~22.3-23.5. kda157: an order pass putting the projection's gate tiles in its last round (q /
  k / v first): 50.8 / 51.3 / 51.3 vs 51.1 / 51.6 / 51.4, q / k / v conv tasks still end 23-24.8 (they run after each
  other on the light CUs, ~1.5 per CU): removed.
- kda158 KDA SL conv tasks one per CU (apart_plan 28 from itself, then the residual's): 54.0 / 53.7 / 54.4 vs 50.9 /
  51.2 / 51.5 (+3): the light plan stays.
- kda159 MoE SL profile, moe_experts per GPU: task start ~8 us after the go, z_q in LDS +6.2, the W13 queues empty
  +30-35, latest W13 item end 63.6-66.3, latest task end 67.2-69.9; W13 items average 16.2 us (122 KB on one wave), so
  units taken last run nearly to the end and their experts' W2 items follow; W13 items per CU 4 (~158 CUs) to 8.
  kda160: STATIC_W13_CU 1 / 2 (a CU's waves on one item: ~4x shorter items; only measured standalone before).
- kda161 MoE SL STATIC_W13_CU (needed a `late` declaration ahead of the W13_CU branches: moved, the current build's code
  unchanged, md5 dc543614; W13_CU excludes W13W2_ONEQ): W13_CU 1 98.8 / 99.2 / 99.4, 2 96.0 / 96.4 / 96.6, no ONEQ 89.6 /
  89.6 / 89.8, current (ONEQ) 89.2 / 88.5 / 88.6: per-wave W13 items stay.
- gsm18 (srv18 libraries, full GSM8K): ours 0.956 at 206.2 tok/s, SGLang alone 0.951 at 142.3 (1.45x); span 12415, KDA
  46.7, after KDA 87.4, MLA 45.0 per layer.
- kda162 MLA SL profile, norm groups 4 vs 1 (37 rows, all kept flags, GPU 0 / 5): norm K parts landed 11.1 / 11.3 vs
  13.6, norm end 13.7-15.0 vs 15.0-15.4, q_b 16.8-17.2 vs 17.6-17.8, attention 25.9-26.4 vs 26.3-26.9, but the residual
  ends 46.2 / 46.0 vs 45.8 / 47.0: the ~1 us is absorbed by the o_proj / residual hops. Not taken (e2e -0.6 at 37 rows).
- kda163 KDA SL STATIC_DRR_ONORM_FUSE (o_norm and y in the substitution's threads, 2-B y stores; checks pass): 51.7 /
  51.6 / 52.1 vs 51.4 / 51.5 / 51.3: no gain (recurrence end ~32.7 either way). Removed.
- kda164: W2_DEEP / W2_ADAPT do not build with W2_LATE_WAIT (static assertion; kda97 measured them instead of it: lost).
  MoE W13 granularity (kda159): 1080 W13 items over 1024 waves (~1.05 rounds), 16 us each, so the ~56 second-round
  items end ~50-66 us and hold their experts' W2 items; W13_CU (4x shorter items) and K-part units (span +3-6) lost.
- Aggregation 1 as our own standalone kernel (kda165-168, amd/serving/agg1.hip, removed; copy in the session scratchpad):
  56 workgroups (8 tokens x 7 column chunks), two tagged exchange rounds (dots / sums of squares, then h's sum of squares),
  lane-parallel weights; outputs bit-identical to SGLang's attn_res_hip in every case (nvb 1-8, addend, bank write). Per
  launch in a HIP graph (warm), ours vs SGLang: nvb 1 6.25 vs 3.40, nvb 3 6.55 vs 4.59, nvb 5 6.77 vs 5.82, nvb 8 7.04 vs
  8.10 us (first version with ds_bpermute reductions 7.3-9.5): the launch plus two device-scope exchanges cost ~6 us,
  more than SGLang's one-workgroup-per-token kernel spends below nvb ~6 (K3: nvb 1-8 over the 93 layers, 12 per block).
  Not used.
- kda169-171 fused KDA + MoE layer kernel (one plan, KDA tasks then MoE tasks per CU; checks pass): e2e 164.1-167.4 vs
  separate KDA 51.4 + MoE 89.5 = 140.9 us; START_SLEEP=32 / TOPK_SENTINEL poll throttles no change (164.3-167.4). Timing
  build (kda171, GPU 0): projection first start 5.2 / end 32.5 (separate ~2 / 21.8), residual 31 end 72.5 (separate ~45),
  experts first start 29.7, end 150; MoE after the KDA residual ~94 us (separate 89.5 e2e). Both halves slower; not used.
- kda172-173 MoE SL profiles (trace, GPU 0): expert weights 199 MB streamed 18 -> 62 us; W13 first round 6.6-6.9 TB/s
  (640 units start 18-20, 197 at 24-26 on the CUs that ran the shared gate_up 30 first), the 102 second-round units start
  36-38 and take ~10 us (end 52); the W2 phase 4.1-5.5 TB/s at 36-52, then the drain to 60. W13_CU=2 profile: W13 done
  54-56, W2 end 72 (its W13 rate lower).
- kda174 MoE SL STATIC_W2_STREAM (new, off: an item's h_q and scales first, its tiles one at a time into the slots the
  previous item's tiles free, so ~one item stays in flight; tile loads by buffer descriptor, 0 bytes past the last item, so
  the compiler counts them; checks pass): 89.4 / 89.4 / 89.2 vs 88.7 / 88.1 / 89.0. Trace: W2 at 36-46 5.0-6.1 TB/s (was
  4.1-5.5), but the drain longer; same end. Not used.
- kda176 MoE SL W13_STAGES 14 (W13_PF sized by the stages; 499 VGPRs, no spill): 89.8 / 90.7 / 90.0, with W2_STREAM
  92.2 / 92.7 / 92.0 vs 7 stages 88.6 / 88.4 / 88.1; W13 units 17.2 us mean (16.0): more bytes in flight per wave do not
  read faster once the W13 round reads 6.7 TB/s. Not used.
- Aggregation 1 in the MoE kernel (SGLang's _agg_kernel after every MoE layer: 7.1 us per layer in the srv18 trace, kernel
  durations there abut: agg1 7.1, KDA 47.0, MoE 79.6, agg1 7.1). New, off by default (MPK_MOE_AGG; harness --agg NB,WRITE,KIND;
  KDA / MLA / MoE default builds md5-identical):
  - KIND 1, tasks/moe_attn_residual.cuh (attn_residual_task MOE mode: a = bf16(o + S) polled as residual_add, u = y, the
    7 chunk tasks per token exchange sums): checks pass (agg out bf16 bits 100% equal to float64, y unchanged, bank row
    write), kda177 SL 96.8 / 96.8 / 97.1 vs 90.2 / 89.6 / 89.5 (+7.2); timing build: y in -> weights 5.6 us (the exchange).
  - KIND 2, tasks/moe_agg1.cuh (one task per token, 8 CUs, 28 columns per thread, bank rows in LDS, LDS-only barriers, h's
    sum of squares from h as SGLang): kda178 +10 (h loop re-read LDS per element behind branches: ~224 waited reads),
    kda179 (y stores after out, LDS barriers) +11, kda180 (h row by row, 7 reads in flight) 94.2 / 93.8 / 94.4 vs 91.2 /
    90.8 / 89.9 (+3.4); checks pass for 3 / 5 (write) / 8 rows. Server run srv19 (rank19_moeagg; sglang_k3 hooks
    _aggregate_hip: the next agg1 call returns the kernel's result when prefix ptr / bank / nvb / write / proj match).
- srv19 (10-16 00:00; MPK_K3_MOEAGG, rank19_moeagg + rank19_moe, KDA / MLA srv18's): span 12088 us (srv18 12416), after
  KDA 83.3 (87.2), KDA 46.5, MLA 46.4; GSM8K 200 0.985 at 213.0 tok/s, 6.073 ms/token; every layer's agg1 hook hit.
  step_split.py: our MoE kernel is the layer marker where no _agg_kernel follows it.
- Cross-GPU latency benches (10-16): amd/bench/ping_pong.hip: one-way signal GPU 0 <-> 1 / 4 / 7 0.50-0.70 us (sc0 sc1 or
  sc1 polls; two CUs of one GPU 0.53); fan-out GPU 0 -> 7 GPUs -> GPU 0: 7 store instructions from one lane (or 7 x 7
  lanes) 5.0-6.3 us round trip, ONE instruction with a lane per GPU 1.57, a wave per GPU 1.72: a wave's system-scope store
  instructions to other GPUs go out ~0.6 us apart (sc0 only: never seen). hop_size.hip had polled the 8 slots one after
  the other (8 round trips): 4.0-4.2 us "per hop" was that; with the slots' loads in flight together 2.3-2.6 us (1 KB to
  115 KB per peer, lockstep with the stores' acks). So the old "~4 us per xGMI hop" (and its reduce-scatter verdict) was
  the bench's.
- kda184 STATIC_PUSH_SPREAD (o_proj push_all's 512 stores over 4 waves, 2 instructions each, instead of wave 0's 8): KDA
  52.0-52.2 vs 51.2-51.4, MLA 50.0 vs 49.3-50.3; trace: o_proj ends ~1.2 us later (its barrier waits for the acks), the
  partials land at the same time (~42.9 med): the in-kernel hop is the 115 KB per peer (~1.8 us at ~64 GB/s per link)
  plus latency, not the instruction serialization. Off.
- kda182 / kda183 KDA attn_residual (LANE_W) stages: o_proj last -> landed ~2.5 us (with the GPUs' clock offsets), landed
  -> exchange done ~2.8, -> weights +0.3, -> end ~1.1-2.2. STATIC_AR_LDSBAR (LDS-only barriers; exchange loads on waves
  1-3) +- UOUT_LATE: 51.0-51.4 vs 51.1-51.6, no gain (code kept, off).
- kda181 / kda185 moe_agg1: end barrier LDS-only (AGG1_END_SYNC 0) 93.1-94.1 vs 93.8-94.0; scores lane-parallel 93.9-94.7
  (base 89.6-89.8): no gain, both back to the srv19 code. Timing build: latent_up last -> every wave's y 2.2 us, -> weights
  1.8, -> end 1.1.
- kda186 KDA projection grids (fewer, larger tasks): [50,4] 51.1-51.5, [100,2] 52.0-52.2, [200,1] 52.2-52.4 vs [100,7]
  51.0-51.4: no gain.
- kda187 STATIC_NEXT_PF (a GEMM task DMAs the CU's next task's first 16 weight steps into an LDS scratch after its last
  MFMAs, to warm the L2; LDS-only barriers): 54.5-55.0 vs 51.2-51.3. The compiler does not count LDS DMA in its vmcnt waits
  (DMA before the last round made every step of it wait for one more DMA), and every later load (the next task's list
  entry, its weights) waits behind the DMA in issue order: task boundaries got longer (end -> next start 1.3 vs 0.4 us).
  Code removed (and STATIC_PUSH_SPREAD); default builds md5-identical.
- kda188 STATIC_SQS_SPREAD (sum_quant_send's pushes over its 4 waves, 3 instructions each): MoE 90.7-91.4 vs 89.8-90.8,
  with agg1 94.4-96.0 vs 93.5-94.0; sum_quant_send ends 12.0-13.1 vs 10.1-11.1 (its last barrier waits for the stores'
  acks), z_q in the experts' LDS no earlier. With kda184 (o_proj): spreading a task's remote stores over its waves does
  not bring the data earlier in the kernels; the fan-out bench's ~0.6 us per instruction is most likely the sender's
  next load waiting (in issue order) for each remote store's ack, not the delivery. Code removed.
- kda191 MoE SL --zq-finish (latent_down's last K part quantizes and sends z_q; nodes renumbered): latent_down 28x8
  92.3-92.5, 28x14 90.6-91.0 vs base 89.3-90.0. Not used.
- kda192 MoE + agg1 SL STATIC_TAIL_SLEEP 8 / 32: 94.0-95.0 / 95.4-95.5 vs 1: 93.1-94.2 (the tail's polls, ~3 MB per round
  from latent_up's 49 tasks while W2 streams, are not what slows it). Trace (kda185): experts last end ~71 -> R landed in
  latent_up ~76 -> latent_up end 79-82 -> y in agg1 ~84 -> end ~87.
- kda194 / kda195 latent_up with 14 K parts (gemm_norm_fill now also takes 256-column K parts: 4 columns per lane, 8-B
  loads; default builds md5-identical): MoE + agg1 96.8-98.0 vs 93.5-94.4 (+3.5; with agg1 apart from latent_up the
  same). Trace: R landed in latent_up 3.1 us after the last allreduce_send (4.6 with 7 parts), latent_up's last end -1.5,
  but the experts end ~2 us later and agg1's end ~4 us later. Not used (grid stays 56x7).
- kda196 STATIC_W13_PRE (first W13 unit's loads before the z_q wait; no ONEQ in that job): 96.2-96.9 vs 94.2-94.4. kda197
  router grids (MoE + agg1, ONEQ): [14,7] 91.6-92.0, [28,4] 94.0-94.7 vs [7,14] 91.9-92.2. (kda193-196 ran without
  ONEQ: base 93.5-94.6 there vs 91.9-92.2 with it.)
- Fused KDA + MoE build (kda169 flags): 109 VGPR spills (424 B scratch), 298 SGPR spills. kda198 with
  MPK_NOINLINE_TASKS (no VGPR spills; 692 B of call stack): moe_experts 168.1-168.6, + recurrence, attn_residual, conv
  (95 SGPR spills) 170.1-170.4 vs inlined 164.6-165.0 us (separate kernels ~141). Not the cause; fused stays off.
- kda199 STATIC_W13_PAIR (a W13 unit on a pair of waves of one CU, each its 16 features: gate tile m + up tile m + 2 over
  all K steps, 28 KB in flight per wave; the tokens' |h| maxima swapped through LDS for the MXFP8 scale; checks pass):
  MoE + agg1 91.7-91.9 vs 91.2-91.3. Trace: a pair's unit 8.5-10.7 us (one wave's 16), 3 rounds, the last W13 end 50.8
  (52.1), the last W2 end 66.7 (62.3): per-CU bandwidth stays the limit. Code removed.
- kda200 W13_TAIL_SPLIT=2 (halves of each die queue's last 1/8): 92.1-93.2 vs 91.2-91.8; W13_QUARTER does not build with
  TICKET_IN_UNIT.
- kda201 MoE + agg1 latent_down grid 56x14: 101.8-102.8 vs 56x8 91.4-92.4 (+10); 56x28 does not build (finish: at most
  14 K parts).
- amd/bench/l2_carry.hip (10-16): workgroup w lands on the same XCD in two consecutive kernels (256/256), but data kernel A
  read is not usefully in the L2 for kernel B: B's read of the same slices after A = cold (32 / 128 / 256 KB per CU),
  except 64 KB plain (1.68 vs 2.42 us). So prefetching the next kernel's weights into the L2 is out. Per CU, plain loads
  read 21-31 GB/s, nt loads 17-21 GB/s in this pattern (one CU's contiguous slice, 8 x 16 B per thread in flight).
- kda202 / kda203 load cache bits: MoE + agg1 STATIC_EXPERT_NT=0 (plain expert loads) 93.7-94.0 vs 91.4-91.6 (nt stays),
  GEMM_SHUF_AUX=0 91.9-92.0; KDA GEMM_SHUF_AUX 0 / 1 / 2: 51.4-51.8 / 51.2-51.8 / 51.3-51.4 (no difference).
- amd/bench/overlap.hip (10-16): two kernels on two streams, A = 256 workgroups (a quarter of them busy 40 us, the rest
  10 us), B launched right after. Plain streams: B's workgroups start at ~11 us on the CUs A freed (B claiming slots only
  once A's 256 are resident; those starting earlier exit). In a HIP graph (fork / join by events) B waits for A's end
  (42 us) unless DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 or DEBUG_HIP_GRAPH_CLASSIC_PATH=1: then B starts at ~11 us in the
  graph too. Deadlock-free form: B's stream waits on A's resident count (hipStreamWaitValue32 >= 256, A's workgroups add 1
  on entry; capturable), then B = 256 plain workgroups: 192 start at 10.6-12.4 us, the rest at A's end. This is the AMD
  stand-in for B300's programmatic dependent launch (B300 notes: "our MLA kernel launches our MoE kernel at its start, so
  the MoE loads its first weights during the MLA").
- kda205 MPK_MOE_INPUT_WAIT=1 (tasks/input_wait.cuh: the MoE's x and prefix polled + copied by 2 nodes of 8 tasks first,
  the later nodes + 4): outputs bit-identical, MoE + agg1 SL 99.0-100.3 vs 91.4-92.0 (+8 us sequentially).
- ov1-ov6 (10-16) KDA -> MoE overlap, amd/tests/test_overlap.py (8 processes, the server libraries): the KDA library
  built with STATIC_RESIDENT (core.cuh: each workgroup adds 1 to the global static_mk_resident on entry; a bigger G or a
  second global with the reset in the kernel made the compiler reserve a 132-B private segment, so the reset is a stream
  memset after the wait), the MoE + agg1 library with MPK_MOE_INPUT_WAIT=2 (re-arm) on a second stream after
  hipStreamWaitValue32(count >= 256). Outputs bit-identical in all modes. Pair e2e (max over ranks, graph / eager):
  sequential srv19 libraries 151.2-152.0 / 147.6-148.4, sequential with input_wait 157.1-160.0 / 153.7-156.7, overlapped
  155.0-155.7 / 165.2-166.5. Stamps (graph, us after KDA's first CU start, rank 1): sequential KDA ends 45.0, MoE go 47.5
  (the graph launch gap is only ~2.5 us), MoE end 140.6; overlapped MoE go 21.5 (first) / 40.0 (median), KDA ends 49.9
  (+5: the MoE workgroups' polls / resets on the freed CUs slow KDA's tail), MoE end 140.9. MoE after x ~91 vs ~93 us.
  Not used as is: it pays only if the MoE does real work before x arrives (front GEMM weights) without slowing KDA.
- ov7 STATIC_READY_OUT / STATIC_READY_IN (no input_wait nodes: KDA's 56 attn_residual tasks add 1 to the global
  static_mk_ready after their stores (workgroup 0 zeroes it on entry, before its resident count); the MoE's x readers
  (router, latent_down, shared gate_up gemm_tile: after their first weight loads are issued) and agg1 / residual_add
  (prefix) wait for it with one thread's polls; default builds instruction-identical): pair graph 153.0-153.5 vs
  sequential 150.8-151.1 (sequential with the READY_IN library 151.8-151.9). ov8: poll s_sleep 8 / 32: 153.1-153.4;
  STATIC_READY_GATE (every MoE workgroup waits at entry: the MoE does nothing before x) 158.1-158.3, and still KDA ends
  ~5 us later (rank 1: 52.3 vs 47.4) and the MoE after the gate takes 94.1 vs 90.1 us.
- amd/bench/coresident.hip (bb9 / bb10): kernel B's workgroups resident on the CUs kernel A freed (polling a flag with
  s_sleep 1 / 8 / 64, or sleeping without memory) do not slow A's remaining workgroups (compute, device-scope load chain,
  mixed: within 1 %, shader clock 2300-2420 MHz either way). B launched while A's L2-resident chain (2 MB) runs: median
  +0.3-0.5 us, worst workgroup +1-3 us (40.6 -> 43.8): the launch's L2 invalidate. So an overlapped launch costs the
  running kernel its L2 at that moment, and in a HIP graph the sequential gap it could save is only ~2.5 us: the overlap
  stays off (code kept, flags off).
- kda206 SL grids: MoE + agg1 router [14,14] (196 tasks of 64 rows x 512 K) 96.8-97.5 vs [7,14] 91.0-92.7; KDA projection
  [200,7] (1400 tasks of 32 rows) 57.6-57.7 vs [100,7] 51.0-52.2. Finer tasks lose ~6 us in both.
- kda207 MoE + agg1 SL: the shared down GEMM after moe_experts on its CUs (56 or 112 tasks; S allreduce, sum_gpus, agg1
  after it), no pack: 97.2-99.0 / 98.1-99.3 vs base (--pack 34:4) 91.2-92.0: the S path becomes the critical one.
- kda185 item trace (MoE + agg1 SL timing build, GPU 0, us after the first experts task start): z_q in LDS 12.8 (median),
  routing table 3.5 us (ends 16.4), first W13 ticket ~2.1, W13 items 18.5 -> 37.1 (median end; 1080 items of 115 KB,
  13.5-18.9 us each, 6.3-6.6 TB/s at 26-35), 300-513 waves between their last W13 and first W2 load at 38-44 (4-6 us
  each) while 1400 W2 items of finished experts wait, W2 items (2520 of 25 KB) 7.0 us each, 3.6-5.0 TB/s at 38-56, last
  end 61. (amd/tools/trace/idle_an.py, w2wait_an.py.)
- kda208 STATIC_FAST_TABLE (expert_table in 3 phases with no serial chain; default build identical): outputs bit-identical;
  table 3.52 -> 2.36 us but the first W13 unit starts 18.2 vs 18.5 (the first ticket's round trip, 2.7 us); MoE + agg1 SL
  91.5-92.0 vs 91.9-92.2 (span -0.5).
- kda209 MoE + agg1 SL: STATIC_FAST_TABLE + STATIC_W13_EARLY_TICKET (the first W13 ticket issued at the task's start,
  its round trip under the z_q wait; default 0, never A/B'd with ONEQ): 89.3-90.2 vs FAST_TABLE 91.2-91.6 vs base
  92.3-92.9 (spans 81.5-82.2 vs 84.3-84.5); outputs bit-identical. srv20 (rank20_moe / rank20_moeagg with both, KDA /
  MLA srv18's): rank libraries bit-identical (eager MoE 87.0-87.6, MoE + agg1 91.1-91.5); server 5.977 ms/token (srv19
  6.073), GSM8K 200 0.985 at 213.8 tok/s; the profile's 8 trace files came out empty (server killed while writing): span
  re-measured in srv20b.
- kda210 STATIC_W2_STREAM on FT + ET: 94.4-97.0 vs 88.7-89.3 (off).
- kda211 KDA SL STATIC_CONV_WAVES (short_conv_silu's conv on 4 waves, 2 tokens each; q / k / v stored and counted before
  the ring / window stores; first version's runtime-indexed window array went to scratch -> 5 static values per wave;
  checks pass): 50.8-51.5 vs 51.2-51.4. Profile: a q / k / v conv task 4.80 -> 4.16 us after its start (its start ~13.6:
  the first barrier waits for the projection task's stores, then a load round trip, then the stores' acks); the
  recurrence still sees the conv complete at ~23.8: the g / f_a / beta tiles are in the projection's 3rd round (19.5) and
  100 conv tasks share the 68 CUs with 2 projection tasks. Off (code kept).
- kda212 normal-build expert stamps (STATIC_EXPERT_STAMPS, test_layer_one_gpu --dump-stamps, amd/tools/trace/estamps_an.py;
  wave 0 per CU, us after the go, medians of 30 launches x 8 GPUs), base / FT+ET: z_q in LDS 16.1 / 15.7, first W13
  20.7 / 19.3, W13 end 39.6 / 38.7, first W2 load +0.2 (no gap in the normal build: the timing build's 3 us is its own),
  W2 loop end 58.1 / 57.1 (wave 0's W2 loop 17.8 us for ~3 items), CU end 69.4 / 68.4 (max 85.5 / 84.4).
- kda213 STATIC_W2_LOOKAHEAD (the next W2 item's loads before the wait for the other item: two in flight at every wait;
  outputs bit-identical): 90.5-91.2 vs 89.4-89.8; W2 loop 18.1 us (unchanged): W2 is not limited by one wave's items in
  flight (~1000 waves reach ~4.2-4.6 TB/s in W2). Off.
- kda214 STATIC_DBG_W2_NOHQ (timing probe, wrong outputs: constant h_q, no h_q loads): W2 loop 16.2 vs 17.8 us per wave:
  the h_q loads are ~1.6 us of it. kda215 STATIC_W2_TAKE2 (one atomic for two W2 tickets; outputs bit-identical): 92.7-93.1
  vs 89.4-89.8, W2 loop 18.4 (unchanged), the tail longer (a wave holds two items): ticket atomics are not the W2 limit.
  Off (both kept, default identical).
- kda216-220 stamp probes (STATIC_EXPERT_STAMPS modes 2-4, normal build, median CU, us after the go): z_q in LDS 15.0,
  every routing pair seen 16.5 (the pairs come after z_q), table built 18.4, first W13 unit 19.5. The table (FAST_TABLE 1)
  takes 1.5 us even run a second time (warm code: not instruction fetch); LDS-only barriers (FAST_TABLE 2) change
  nothing; per phase (mode 4): the step reading the 128 firsts (32 16-B LDS reads per thread) 1.6 of 2.3 us.
- kda221 STATIC_FAST_TABLE 3 (presence bitmask: slot = bits below e, 7 16-B reads; 3 phases; outputs bit-identical):
  88.0-90.2 vs FAST_TABLE 1 88.7-89.6 (spans 80.4-81.5 vs 81.1-81.9): ~-0.7 us, within the spread; carried to the next
  server run.
- kda222 STATIC_W13_TAKE_CU (the CU's 4 first W13 tickets by one atomic add from wave 0; outputs bit-identical) on FT3 + ET:
  87.8-89.6 vs 88.6-89.2 (spans -0.2): not taken.
- kda223 MoE + agg1 SL --front (plan_one_gpu.front_plan: router first on the first CUs, latent_down on the rest, shared
  gate_up behind router tasks) on FT3 + ET: 14,14,1 91.9-93.2, 28,7,1 91.6-93.2 vs 88.2-88.6. Not used.
- Fused KDA + MoE kernel, found 10-16 (test_kda_moe_layer.py, SL, today's flags):
  - kda224-227: fused 161.7-164.0 us vs KDA alone 50.7-52.0 + MoE alone 85.1-86.0; with every MoE task gated until the KDA
    output is complete (static_schedule STATIC_FUSED_GATE_NODE / _CNT / _NEED: a probe; attn_residual's counter is the 4th
    StaticNode argument, 64 here, not the 5th) 164.3-164.8: the loss is not the MoE running during the KDA tail.
  - kda228 (STATIC_EXPERT_STAMPS slot 8 = the gate pass): the KDA part ends 67.2 us after the go (alone: span 45), the MoE
    part's W2 loop 24.2 vs 17.6 us per wave.
  - The fused build (compiled with line tables): 512 VGPRs + 103 VGPR spills (400 B scratch, 204 scratch ops): the task
    loop's carried values (layer.cu's loop, L.sm, buf_at's parity offsets) and gemm_tile's activation-fill loop, because
    moe_experts needs ~500 VGPRs in the same function. kda231: moe_experts not inlined: KDA part 53.9 (inline 67.1) but the
    MoE part slower (W2 ends 80 vs 66.5 us after the gate).
  - static_schedule MPK_SPLIT_LOOP=S (two task loops: nodes < S, then the rest): 475 VGPRs, no spill, no scratch; kda232
    fused 141.4-141.7 (gated 145.2: KDA part 52.4) vs apart 51.2 + 85.7; + early front + LDSW 4: 154.5-154.9 (worse).
  - The projection's first activation landed 9.1 vs 4.4 us after its start in the fused build: kernel_begin's reset stores
    of the exchange's next set (STATIC_RESET_PARITY) sit ahead of it in issue order -> STATIC_XRESET_LATE (core.cuh
    xreset_next: those resets at the second loop's start) in test (kda234).
- kda234 STATIC_XRESET_LATE: 140.6-141.6 vs split 138.7-140.3 (off); kda233 profile: with split loops the KDA part's stages
  match KDA alone (projection round 0 activation 4.56 vs 4.24 us: the round-0 loss was the spills), the residual's tail
  +2 (median) / +5.6 (max).
- kda235 fused KDA + MoE + agg1 (test_kda_moe_layer.py --agg 3,0,2, split loops): checks pass (agg1 bits 100 %); 145.3-146.5
  vs KDA 51.3 + MoE + agg1 89.3, but not like for like: the fused MoE routes the KDA output (110 experts vs ~90).
  kda236 like for like (MoE + agg1 alone on the fused run's x / prefix, --save-moe-in / --moe-in, 110 experts in both):
  fused 143.6-144.8 vs 51.3-51.7 + 95.4-96.4 = ~147.5: fused -3 us. -> srv22 (rank kind kdamoeagg, MPK_K3_KDAMOEAGG).
- srv22 (07:30, job_srv22.sh = srv21 + MPK_K3_KDAMOEAGG with rank22_kdamoeagg: the fused KDA + MoE + agg1 kernel for
  the 68 KDA layers): span 11888 vs srv21 11828 (+60), GSM8K 0.980 (srv21 0.985; 200 questions) at 214.8 tok/s.
  Trace kernel medians (TP-1): fused 127.7 us vs srv21's KDA 46.7 + MoE + agg1 80.2 = 126.9. In the server the two
  kernels already ran back to back (graph, no gap between them in the trace), so the ~3 us the harness credited to
  fusion (the launch in between) is not there; the fused kernel's own overhead (residual tail) stays. srv21 stays the base.
- ord1 (10-16 08:15, MoE + aggregation 1 + the NEXT layer's KDA part in one kernel: test_kda_moe_layer.py --order moe;
  moe_agg1_layer(posted=True): agg_out is the node's output, device-scope stores, one counter; the KDA projection waits
  for it, attn_residual's prefix = the MoE's y; split loops at the first KDA node, 72; default builds md5-identical):
  all checks pass (KDA part on the kernel's own x / prefix, aggregation 1 bits 100 %). SL e2e: one kernel 133.8-134.0
  (LDSW 3) vs KDA alone 51.2-51.6 + MoE + agg1 alone 88.6-88.9 = 140.1-140.2: -6.2 us. STATIC_GEMM_LDSW=5 (the
  projection's 128 KB slices into LDS during the MoE tail) 135.8: +2 vs LDSW 3. -> srv23 (rank kind moeaggkda,
  MPK_K3_MOEAGGKDA; sglang_k3: moe() of layer L runs layer L+1's KDA part, its attn() returns the result).
- srv23 (08:18, rank23_moeaggkda, eager 134.6-136.2 in 8 processes, bit-identical to the harness): WRONG answers
  (repetition). Cause: SGLang's AttnBankState attn_readout returns `None if self.writes_block else residual`: after a
  block-write layer's aggregation 1 the attention's prefix is ZERO (y is the bank row), not y; the kernel used y (the
  harness ran write 0 only). Killed. srv23b: the one kernel only where the next layer does not write (7 of 8 write
  layers fall back). Fix: moe_agg1 (posted) writes agg_prefix = write ? 0 : y (a 9th buffer slot, run tensor
  "agg_prefix"), the attention part reads it; harness --order moe checks it (write 0 and --agg 3,1,2). Also: 5 layer
  pairs where SGLang does aggregation 1 itself (addend given; srv21 has the same misses) rerun the attention kernel; the
  pair is switched off after the first miss (mk_off).
- MLA + MoE + agg1 in one kernel (either order) failed at set-up: 25 in-kernel reset buffers, at most 24 (kda237 had the
  same, unseen behind the grep): STATIC_MAX_RESETS (default 24: the list is in the kernel argument G) = 28 for it.
- srv23b (08:35, srv23's rank23_moeaggkda, the one kernel except before block-write layers and the 5 miss pairs):
  span 11645 vs srv21 11828 (-183), MoE + next KDA 122.4 per pair vs 46.4 + 79.9 = 126.3 (-3.9; SL said -6);
  GSM8K 0.990 at 216.8 tok/s; 8 prompts 5.751 ms/token (accept 2.915). Answers sane (prompt 6 = SGLang's text).
- ord3 (08:48, agg_prefix version): checks pass: KDA write 0 (agg_prefix = y), KDA write 1 (--agg 3,1,2: agg_prefix
  zero, bank row = y, the KDA part right on the zero prefix), MLA write 0 (STATIC_MAX_RESETS=28). SL: MoE + agg1 + MLA
  131.5 / 132.7 / 133.1 vs MLA alone + MoE + agg1 alone 137.9 / 136.9 / 139.7 (-4 .. -7); MoE + agg1 + KDA with the
  agg_prefix stores 133.4 (as before). -> srv24 (both kinds on every MoE layer).
- srv24 (08:59, job_srv24.sh: rank24_moeaggkda + rank24_moeaggmla, every MoE layer with the next layer's attention part,
  agg_prefix for block-write layers): span 11600.9 (srv23b 11645, srv21 11828); kernels (TP-1 trace medians):
  moeaggkda 124.4, moeaggmla 123.1 (apart: MLA ~45 + moeagg 81.9), 56 KDA kernels left (the miss pairs);
  GSM8K 0.980 at 215.8 tok/s (200 questions; 0.980-0.990 across today's runs).
- ord4 (09:11, timing build, MoE + agg1 + KDA one kernel, GPU 0, us after the start barrier): experts end 64.5,
  sum_gpus 72.7, latent_up 77.6, agg1 input landed 83.6 / end 87.1; the KDA projection (node 73) first start 57.5
  but last end 107.6 = 20.5 us after agg1 (KDA alone: 19.8 from its start). Per CU (proj_an.py on the task dump): CUs
  free at ~60 with 3 projection tasks still end ~104-105 (18 us after x), 2-task CUs ~98: the projection streams
  after x whatever was loaded before; LDSW 5 (first slice in LDS): the same ends.
- ord5 (09:19, STATIC_GEMM_LDSW_FIRST: only a CU's first task of a waited node takes the LDS path, later ones stream
  through registers; default build .s identical): KDA one kernel 134.8 / 135.4 / 134.1 vs LDSW 3 133.4 / 133.8 /
  134.4; MLA one kernel LDSW 5 135.2-135.7, 5 + FIRST 134.2-134.8 vs 3 132.9-133.3. Lost (code kept, off). With
  kda232 and ord1 the fourth time a weight preload into LDS during another phase lost; likely (not measured) the bulk
  reads slow that phase's device-scope polls / hand-offs (they go to MALL / HBM). Not retried without a measurement of
  the poll latency under load.
- bb12 poll_load.hip (one GPU; a device-scope flag ping-pong between XCDs 6 / 7 while 254 CUs read): round trip 0.91 us
  idle, 1.11-1.20 with 2-6 TB/s of reads (1-8 loads of 1 KB per wave in flight), 1.48 at 7.32 TB/s (16 per wave). bb13
  stream_ladder.hip (96 MB cold, 1 GB read before each, event-timed launch): burst reader 20.8 us, rolling window 19.1,
  per-wave contiguous runs 19.2, + LDS activation reads and MFMAs 19.1, 3 tasks per CU 19.9, next task's loads in the
  drain 20.3: all ~5 TB/s; the projection's ~20 us is the cold-stream time, not the GEMM loop. bb14 icache.hip: a
  kernel's first launch pays 0.34-0.56 us for 4-32 KB of code, later launches <= 0.05 us (1 or 256 CUs): instruction
  fetch is not a hidden per-task cost.
- srv25 (09:41, rank25: + agg_bare, the MoE output without the prefix, for the 5 layer pairs where SGLang's aggregation
  adds the residual itself): span 11611.6 (srv24 11600.9), GSM8K 0.980 at 216.5; the 5 pairs still missed: SGLang
  calls our moe() hook only on _ffn_writes_stream layers, and there written_residual is None (so p = zeros), while the
  aggregation's prefix_sum is another tensor: srv25d logs which (MPK_K3_AGG_DEBUG).
- srv25d (debug): at the 5 pairs SGLang's aggregation 1 gets prefix_sum = another (8, 7168) bf16 tensor (a copy of our
  y), addend None (the old miss message printed `addend is None`). srv26 (09:56): the hook takes such a call when the
  copy's values equal y (checked in the layer's first eager call; S.copy_ok): all 5 equal; span 11572.9, GSM8K 0.985
  at 208.2; _agg_kernel per forward 11 -> 6. The pairs still ran apart: the attention's input there is a copy of agg1's
  output too (attn() pointer check) -> srv27 (the same value check for it, S.copy_ok_h).
- srv27 (10:06, srv25's rank25 libs + sglang_k3 copy checks for y and the attention input): all 5 copies equal; decode
  forward = 67 moeaggkda + 24 moeaggmla + 2 KDA + 1 MoE kernels; span 11547.5 (srv21 11828: -280 today), GSM8K 0.980 at
  209.2 tok/s (200 questions); 4 of 8 prompts the same text as SGLang alone.
- bb15 / bb16 lds_mfma.hip (one CU, the K loop once the weights are in LDS): gemm_wave's WLDS loop 3638 clocks for 24
  steps (~150 per step); the activation rows KS * 2 bytes apart (a multiple of 256 B) put the 8 tokens' fragments on
  the same LDS banks: rows padded by 16 B 2038, + reads in chunks of 8 and 2 accumulators 1354 (32 steps: 4808 /
  2672 / 1730). -> STATIC_GEMM_ACT_PAD (gemm_tile's activation slice rows KP + 8 apart; default build .s identical).
- act2 (10:24, SL, 3 rounds): ACT_PAD KDA 50.4 / 50.4 / 50.8 vs 51.2 / 51.8 / 51.4 (-0.9); MoE + agg1 88.6-89.1 vs
  88.9-89.3 (same); MLA 48.6-49.4 vs 48.7-49.4 (same); MoE + agg1 + KDA one kernel 132.9 / 133.9 / 133.7 vs 134.3 /
  134.7 / 134.5 (-0.9). With the pad still lost: o_proj grid 224 + LDSW (KDA 52.4-53.1, MLA 50.6-51.8), LDSW 5 in the
  one kernel (134.6-135.8), LDSW 2 for latent_up (88.8-89.8). -> srv28 (ACT_PAD in both one-kernel libraries).
- srv28 (10:33, job_srv28.sh: rank28_moeaggkda + rank28_moeaggmla = srv25's + STATIC_GEMM_ACT_PAD, sglang_k3 of srv27):
  span 11447.8 (srv27 11547.5: -100), kernels moeaggkda 123.4 (-1.0), moeaggmla 121.7 (-1.1); GSM8K 0.980 at 210.8.
- ord4 MoE profile (timing build): topk_route per task after its K parts landed: keys 0.6-0.8 us, top 16 picked 3.7-4.4
  (the radix pick 3.1-3.8 us, ~7500 clocks), end 4.0-4.7; the routing pairs come ~1.5 us after z_q (kda216). Next: a
  faster pick (per-wave top 16 by wave argmax, then the 64 candidates ranked).
- bb17 topk_select.hip variant 5 (per-wave top 16 by 16 rounds of a DPP wave max + readlane, then wave 0 ranks the 64
  candidates): correct (0 wrong of 1280) but 9.0 us vs radix 2.72 / 3.52 (sigmoid + bias): the DPP / readlane chains
  cost far more than counted. The radix pick stays.
- ord4 read curve (MoE + agg1 alone, timing build, GPU 0): front GEMMs 0-16 us at 1.8-3.4 TB/s (63-164 tasks); W13 16-34
  (waves enter 333 -> 928 over 16-24; peak 6.8-7.0 TB/s at 928-968 waves); W13 -> W2 34-48 at ~5 TB/s; W2 drain 48-56;
  then latent_up only (0.42 TB/s). Against ~29 us at 6.8 TB/s for 198 MB the experts take ~40: ramp ~3, hand-over ~4,
  drain ~3, the 14 shared-down CUs ~1.
- sh1 (10:51, MoE + agg1 SL, today's flags, 3 rounds): base (--pack 34:4) 88.0-88.9; --shared-late 118.3-118.4; --shared-early
  92.0-92.4; no pack 89.4-89.6: the base stays.
- Built 10-16 12:25 (test ep1, queued): moe_agg1 posted mode with 2 counters: count 0 once `out` (the next attention's x)
  is complete, then y / the bank row / agg_prefix / agg_bare and count 1; attn_residual params [H, eps, counter, need]
  (attn_residual_layer prefix_wait) wait for count 1 before reading the prefix and bank rows. Default builds .s identical
  (cg_ord9); the one kernel: attn_residual waits counter 353 = moe_agg1's 352 + 1, need 8.
- Queued: ts1 (MoE + agg1 and the KDA one kernel, W13_TAIL_SPLIT 0 / 4 / 8 / 16), srv29 (server with TS 16, gated on
  ts1), ts2 (timing-build W13 / W2 traces for TS 0 / 2 / 16, amd/tools/trace/w2_an.py).
- ts1 (12:38, SL, 3 rounds): STATIC_W13_TAIL_SPLIT MoE + agg1: 0 88.1 / 88.8 / 89.1, 4 89.5 / 88.9 / 89.3, 8 90.6 / 90.0
  / 90.3, 16 91.2 / 90.9 / 91.5; MoE + agg1 + KDA one kernel (ACT_PAD): 0 134.4 / 133.3 / 133.0 vs 16 135.0 / 135.4 /
  136.7. Splitting the W13 items costs more than the late second round (ord4 trace: 112 waves x 17 us) saves; 0 stays.
  srv29 (gated on it) skipped; ts2 profiles TS 0 / 2 / 16 for the reason.
- ts2 (12:51, timing build, MoE + agg1, w2_an.py): TS 0: W13 1080 items 17.4 us (p50), starts p90 33.7, W2 starts p10
  36.8, experts end 66.7, sum_gpus 73.2, latent_up 77.6. TS 16: 2160 half units 9.4 us (same per-wave rate), W13 starts
  p90 34.6 (a third round of halves to ~44), W2 starts p10 40.1, experts end 65.5, sum_gpus 73.9, latent_up 78.3. With TS 0
  W2 runs beside the late whole items; splitting delays W2 more than it shortens W13: the late second round is not what
  sets the experts' end (my 10-16 estimate of 6-8 us was wrong).
- ep1 (12:53, SL, 3 rounds, srv28's code vs the early post): checks pass (KDA write 0 / 1, MLA); KDA pair 132.9 / 132.7 /
  132.8 -> 131.8 / 131.4 / 132.5; MLA pair 131.4 / 132.2 / 131.7 -> 130.3 / 130.5 / 130.9: -1 us both. -> srv30.
- mf1 (12:58, MoE + agg1 SL, 3 rounds): base (W13_STAGES 7) 88.6-89.0; W13_STAGES 14 90.0-90.7; TOPK_SENTINEL 88.7-89.3;
  START_SLEEP 4 88.4-88.9: nothing beyond the spread; the base stays.
- kd1 (13:17, KDA SL, ACT_PAD, 3 rounds): o_proj 112 LDSW 3 50.7-51.2, 112 LDSW 0 50.1-51.0, 224 LDSW 0 (register
  path, no --apart) 52.6-52.8, 224 LDSW 0 no pad 53.0-53.4: o_proj 224 loses on either path; 112 stays.
- srv30 (13:24, job_srv30.sh: rank30 = srv28's + moe_agg1's early post): span 11370.8 (srv28 11447.8: -77; srv21 11828:
  -457 today), kernels moeaggkda 122.4 (-1.0), moeaggmla 121.6; GSM8K 0.985 at 212.8 tok/s.
- ar1 (13:49, srv30 code, 3 rounds): STATIC_AR_POST_NOWAIT for KDA: pair 131.6 / 131.8 / 132.7 vs 131.8 / 132.3 / 132.6,
  KDA alone 50.8-51.0 vs 50.2-50.9: no change, not taken. (Already lost before: DRR_RING_LATE, CONV_WAVES' early post.)
- pr1 (13:53, timing build, srv30 code, GPU 0, us after the start barrier). MoE + agg1 + KDA: agg1 landed 80.9, end 86.0;
  projection end 104.8 (107.6 before the early post); conv 106.5; recurrence landed 107.0, end 113.8; o_proj y landed
  114.7, end 118.9; attn_residual partials landed 122.6, end 127.7. MoE + agg1 + MLA: agg1 end 84.6; wa (238 tasks) end
  92.5; mla_norm 95.7; q_b 97.8; g_proj (168 tasks, after wa on their CUs) 102.5; absorb 100.9; attention 105.4; vgate
  (12 tasks; w_vc already copied into LDS at its start) 109.2; o_proj landed 110.7, end 115.6; attn_residual landed
  119.0, end 123.9.
- lu1 / lu2 (13:57-14:00, MoE + agg1 SL, ACT_PAD, 3 rounds): latent_up regrid without an after-set: the plan check finds a
  deadlock (656 of 867 tasks in a circle); with after 36,37,38: 112 x 7 (64 rows) 92.9-94.0, 56 x 14 (256 K) 92.2-93.1 vs
  the solver plan's 56 x 7 88.1-88.7: lost.
- pr1 MoE tail (srv30 code, timing build, GPU 0): R send ends 64.7 here but the slowest GPU's ~68 (experts end 62.8-67.3
  across GPUs, a different GPU each run), so latent_up's R lands at 71.6; latent_up ends ~75; o lands 80.9; agg1 86.0.
- lt1 (14:30, --late-light: the attention lists with the fewest projection tasks on the CUs with the MoE's latest tail
  tasks, within each die; checks pass): KDA pair 132.2 / 132.5 / 132.8 vs 132.2 / 133.0 / 132.0, MLA 131.2-131.6 vs
  131.1-132.0: no change in the normal build (the timing build's per-CU ends suggested ~2 us). Off (option kept).
- ns1 (14:54, STATIC_NODE_END_STAMPS: per-CU last end of listed nodes in the NORMAL build; amd/tools/trace/nodeend_an.py),
  MoE + agg1 + KDA, max over GPUs (us after the go): LDSW 3 / 5 / 5+FIRST: experts 67.2 / 66.8 / 66.7, agg1 85.5 / 86.0
  / 85.6, projection 103.7 / 106.3 / 104.4, residual 126.4 / 129.8 / 127.4: the MoE tail is NOT slowed by the preload
  (my earlier guess); the projection itself is.
- ns3 (15:04, STATIC_P1_NODE: the CU's first task of a node, normal build; p1_an.py): activation landed -> MFMAs done
  KDA 2.92 (LDSW 3) vs 2.12 (LDS path); MLA wa 2.52 vs 2.20: the LDS loop ~150 clocks a step; and stragglers: CUs
  busy to the MoE tail's end start their task at x (nothing preloaded), where the LDS copies stream slower (MLA wa end
  max 99.8 with the LDS path).
- ns4 (15:10, + STATIC_GEMM_WLDS_FAST (8 K steps' LDS reads together, 2 accumulators) + STATIC_GEMM_LDSW_SKIP=38,39,48
  (register path after a MoE-tail task), one round, stamped): KDA pair 132.9 vs 133.4, first task landed -> MFMAs 1.72
  vs 2.88; MLA pair 130.8 vs 133.2 (wa end 92.5 vs 93.6, residual 124.3 vs 125.9); + --late-light worse (134.3 / 132.4).
  -> ns5: 3 rounds without stamps.
- ns5 (15:16, 3 rounds, no stamps): KDA pair LDSW 3 132.2 / 131.8 / 133.0, LDSW 5 + FIRST + WLDS_FAST + LDSW_SKIP 133.6
  / 132.6 / 133.2, LDSW 3 + WLDS_FAST 132.6 / 133.3 / 133.2; MLA pair 130.8 / 130.6 / 131.4, 130.6 / 130.7 / 131.1, 130.0 /
  130.8 / 130.5: all within the spread (ns4's MLA -2.4 was one stamped round). The faster LDS loop shortens the first
  task (2.88 -> 1.72 us) but not the pair. Preloads closed; flags kept, off by default (default builds .s identical).
- bb19 amd/bench/l2_warm.hip (one GPU, ONE kernel, 256 CUs, a slice per CU streamed as gemm_tile does: nt b128 loads, 16
  in flight per wave; 1 GB read before each launch): pre-read with plain loads, 10 us wait, then the timed pass: 64 KB per
  CU (2 MB per XCD) 0.64 us vs 3.04 cold; 128 KB (= the 4 MB L2) 3.08 vs 5.16; 256 KB 5.84 vs 9.76; 384 KB 9.60 vs 14.04.
  An nt pre-read is mostly not kept (64 KB 2.00, 128 KB 4.04). (l2_carry: across kernels nothing is kept.) -> STATIC_GEMM_L2WARM=KB
  + STATIC_GEMM_L2WARM_NODES (gemm_tile: a listed node's first task on a CU reads the first KB of the CU's next task of
  the node into the L2 with 4-B loads, one per 128-B line, before its wait); default builds .s identical (cg_l2w).
- lw1 (15:38, KDA pair SL, ACT_PAD + LDSW 3, 3 rounds): STATIC_GEMM_L2WARM=64 (node 74) 131.0 / 131.8 / 131.5, base 132.7 /
  131.9 / 133.0, L2WARM=128 132.0 / 131.8 / 131.9: 64 KB ~-1.1 us (checks pass). -> srv32 (srv30 + L2WARM 64 in the KDA
  one-kernel library).
- ke1-ke3 (15:58-16:04) STATIC_KEEP_BUFS=routing_pairs,latent_z_mxfp8 (host.cuh, a probe: those buffers never reset in
  the kernel; with the same input every launch, launch n finds launch n - 2's pairs and z_q, so moe_experts starts at the
  go, as if the front had run in the kernel before; the front still runs and rewrites the same bytes): MoE + agg1 SL 93.0 /
  91.9 / 91.9 vs 89.0 / 89.6 / 88.9; KDA pair 137.7 / 134.2 / 135.8 vs 131.9 / 132.7 / 132.4: SLOWER. Node ends (normal
  build, max over GPUs, ke3, two rounds): experts 71.7 / 71.6 vs 67.7 / 68.0, shared down 63.9 / 63.6 vs 60.2 / 60.3,
  sum_gpus 79.2 / 78.2 vs 73.8 / 74.6, agg1 88.3 / 85.9 vs 82.9 / 83.8. The experts' end is set by the bytes and the
  contention, not by their start: an earlier start adds contention with the front and the shared path. So moving the
  MoE's front into the kernel before (router, topk, latent_down, z_q in the previous launch) does not pay with this plan.
- srv32 (16:07, srv30 + L2WARM 64 in the KDA one-kernel library, rank31_*): span 11358.1 (srv30 11370.8: -12.7), kernels
  moeaggkda 121.6 (-0.2), GSM8K 0.980 at 211.9 tok/s; rank libraries bit-identical to the harness.
- st1 (16:18, killed after round 1): STATIC_GEMM_STAGES=32 (global: every waited GEMM holds its whole slice in registers,
  incl. the MoE front's) KDA pair 136.8 vs 132.2 (16), + L2WARM 135.6; MLA pair 133.3 vs 130.9: lost.
- hy1 (16:22, KDA pair SL, 3 rounds): the hybrid projection (gemm_tile_layer hybrid=True, MPK_KDA_PROJ_HYB=1, grid
  [50,7]: 128-row tasks, each wave's first row tile copied into LDS, its second held in registers for all 32 K steps,
  all issued before the activation wait: gemm_tile.cuh gemm_wave HYB; 350 tasks of 256 KB, one per CU on most CUs):
  129.0 / 129.6 / 129.4, + L2WARM 64 128.9 / 128.2 / 129.0 vs base [100,7] 131.9 / 131.9 / 132.0: -3.2 us; checks pass.
  Default builds .s identical (cg_hyb). -> srv33.
- srv33 (16:27, job_srv33.sh = srv32 + the hybrid KDA projection; libraries built on the CPU first, lib33.sh): span
  11213.1 (srv32 11358.1: -145), kernels moeaggkda 120.3 (-1.3), GSM8K 0.975 at 213.7 tok/s; rank libraries bit-identical
  to the harness (moeaggkda eager 131.1 vs 133.4).
- hm1 (16:40, MLA pair, 3 rounds): MLA wa hybrid (MPK_MLA_WA_HYB=1, 29:[17,7,1]) 129.4 / 130.8 / 130.4 vs [34,7] 129.8 /
  130.2 / 130.1: no change (the MLA part is not bound by wa's weights). Not taken.
- hy2 (16:45, KDA pair hybrid, 3 rounds): + L2WARM 256 127.4 / 128.4 / 127.9, 128: 128.2 / 128.2 / 129.8, 64: 128.7 /
  129.4 / 128.4: 256 (the second hybrid task's whole slice) ~-0.7 vs 64.
- ns6 (normal-build node ends, max over GPUs): hybrid + W64 vs base: agg1 86.2 / 86.1, projection 97.6 / 103.3, conv
  100.5 / 104.9, recurrence 109.3 / 112.5, o_proj 114.6 / 117.8, residual 122.9 / 125.8: the projection ends 5.7 us
  earlier but conv after it +2.9 vs +1.6. pr2 (timing build, GPU 0, hybrid + W256): the latest projection tasks are on
  CUs 148-155, the moe_agg1 CUs: after agg1 (~86) each streams a cold 256 KB task (87.6 -> 96.2), then runs a conv task
  (-> 101.3). -> test_kda_moe_layer.py --proj-off (moe_first_plan off): no projection task on the CUs whose MoE list
  holds the given MoE nodes (agg = moe_agg1: 8 tasks per GPU moved; agg,39: 58-60), each moved to the CU with the fewest.
- po1 (16:55, KDA pair hybrid + W256, 3 rounds): --proj-off agg 129.1-129.3 (worse), agg,39 127.0-128.2, none 127.7-128.3:
  no gain (the 94 second tasks still stream after x).
- ca1 (17:05): STATIC_GEMM_CARRY + gemm_tile_layer carry (MPK_KDA_PROJ_CARRY=1, [100,7]: a CU's first projection task
  holds its slice in registers and DMAs its next task's into LDS while waiting; that task reads it there; 444
  consecutive pairs per GPU, so it engaged): 132.4-132.7 = the old base; + L2WARM 128 (warming the task after next)
  133.8-134.1; hybrid + W256 127.6-128.3. Each 64-row task pays its own activation round trip and stores (three per CU
  here); lost (code kept, off).
- gl1 / ns7 / gl2 (17:18-17:25): "gate late" (MPK_KDA_GATE_LATE=1: short_conv_silu_layer gate=(4608, 6144) counts the
  gate columns' 24 tasks on a second counter, delta_rule_recurrence_layer gate_late waits for them only before its
  output's gate; --gate-second puts the gate row blocks' projection tasks after the others in the lists): 128.9 / 129.6 /
  129.5 vs 128.1 / 128.2 / 128.0; node ends: recurrence 108.4 vs 108.5 (no earlier). The non-gate rows (69.7 MB per GPU)
  do not fit the CUs' 64 MB of LDS + register tiles: 18 non-gate tasks stay second and stream after x. + STATIC_GEMM_
  L2WARM_SKIP_X=36,48 (warm only the non-gate second tasks): 130.4 / 129.5 / 129.4 vs 128.0 / 128.2 / 127.8: lost. Code
  kept, off (default builds .s identical, cg_gl).
- srv34 (17:27, = srv33 with L2WARM 256 on the hybrid projection; rank34 libs built on the CPU): span 11125.5 (srv33
  11213.1: -87.6), moeaggkda 119.1 (-1.2), GSM8K 0.985 at 215.7 tok/s; checks / rank libraries pass. BEST.
- pu1 (17:28, tree mirage_x): STATIC_PUSH_ROT (core.cuh push16 / push8: lane l's j-th store to GPU (l + j) % 8, so one
  instruction reaches every GPU): MoE + agg1 113.3 / 112.5 / 112.5 vs 87.8 / 88.2 / 88.0; KDA pair 157.2-157.7 vs
  127.4-128.0; MLA pair 162.2-162.6 vs 129.9-130.2. In the kernel one store instruction spanning GPUs is far slower than
  8 per-GPU instructions (opposite of ping_pong.hip's fan-out). Off.
- pp1-pp5 (17:47-18:13) "x prepoll" (MPK_X_PREPOLL=1: moe_agg1_layer prepoll posts a third count before hidden's stores
  and resets hidden (0xFF) in its own buffer; gemm_tile_layer act_prepoll (KDA projection, MLA wa / g_proj) waits for that
  count, then polls the activation; needs STATIC_MAX_RESETS 28 / 32 and the producer's buffer, not the activation map,
  whose tensor is the caller's): KDA pair 130.0 / 129.8 / 129.0 vs 128.6 / 127.6 / 128.2; MLA pair 133.6 / 132.8 / 133.7
  vs 130.2 / 131.0 / 130.3: lost (the waiting tasks' polls of 16 KB slices). Off. (Server note: a rank library binds run
  outputs to the caller's tensor, and the reset list points at its own copy.)
- kh1 / kh2 (18:17-18:25): KDA o_proj K-split hybrid (gemm_tile_layer lds_ksteps=32, MPK_KDA_OPROJ_KH=32: each row
  tile's first 32 of 48 K steps by DMA into LDS, the rest in registers, before y's poll; kh1's per-wave LDS stride was
  the full slice: wrong outputs, fixed): 128.8 / 127.8 / 129.0 vs 127.9 / 128.3 / 127.9: no gain (o_proj's ~4 us after y
  lands is its compute and the push_all hop, not its weights). Off.
- pr2 read curve (timing build, GPU 0, hybrid + W256): front 0-16 us 1.2-3.5 TB/s; W13 16-34 (ramp to 6.6, peak 6.6-6.9
  at 930-960 waves); W13 -> W2 34-52 at 4.3-6.0 TB/s with 1200-1440 W2 waves (~3.3 GB/s per wave vs ~6.9 in W13); drain
  52-56; then the projection's on-chip preload 58-88 at 0.6-3.4 TB/s; x lands ~88; 90-92 5.6 TB/s (second tasks).
- srv35t (18:27, trace run, MPK_K3_TRACE_EXTRA=1: sglang_k3 logs the Python stack of SGLang ops our kernels do not
  cover): the 3 kernels at 5 layers per step (copyBuffer 4.2 + elementwise 4.6 + _agg_kernel 5.6-11.6 us, ~100 us per
  forward) are SGLang's dspark capture of the stream after the layer (kimi_k3.py _dspark_capture_stream: a snapshot
  copy, then attn_residual.aggregate_stream = attn_res_hip with no output norm: the next layer's pre-norm aggregation 1).
- mx1 / mx3 (18:39-18:52) MPK_AGG_MIX=1 (moe_agg1's 11th slot agg_mix = bf16 of its pre-norm mixture h; demo.build_graph
  agg_mix): agg_mix bits equal 100.0 % to the float64 reference (both pairs), but +1.3 us KDA pair / +2 MLA pair (its
  stores lengthen agg1's task; on agg1's CUs the next task is the last cold projection task). -> only at the dspark
  capture layers (sglang_k3 kind moeaggkdamix, S.captures from dspark_layers_to_capture; _dspark_hook returns agg_mix
  after one eager equality check). Default builds .s identical (cg_mix3; the stores under if constexpr: a runtime null
  check before count 1 had changed the default code).
- srv35 (18:54): the mix kind was loaded, but SGLang sets dspark_layers_to_capture after the load (set_dspark_layers_to_
  capture): no capture layer seen, the mix kernels unused; span 11088.0 (no change from this; run-to-run ~30 us).
- srv36 (19:05, = srv34 + kind moeaggkdamix at the capture layers [7, 23, 51, 67, 83], sglang_k3 wraps set_dspark_layers_
  to_capture and rebuilds those layers' kernels; 4 of 5 captures "values equal" and taken from the kernel): span 11058.6
  (srv34 11125.5: -67), GSM8K 0.985 at 223.6 tok/s. Left at the capture layers: AuxHiddenStatePacker.append's strided copy
  into the [T, 5 H] capture buffer (elementwise ~5 us each) -> next: the kernel writes that slice in place.

## Flag cleanup (10-16 evening)
- Removed every compile flag no server library sets, with its code (unifdef at the defaults, then the C-level branches by
  hand): ~100 STATIC_* switches (MoE W2/W13 variants, RIK modes, PUSH_ROT, KEEP_BUFS, DBUF, READY/RESIDENT overlap,
  CARRY, KHYB, PREPOLL, GATE_LATE, PROJ_SPLIT, LDSW 1/2/4/5, FAST_TABLE 1/2, TOPK_FAST, AR_* probes, HANDOFF 0, ...), the
  Python options carry / act_prepoll / lds_ksteps / gate / input_b / gate_late / prepoll, the harness env options
  (MPK_KDA_PROJ_CARRY, MPK_KDA_OPROJ_KH, MPK_X_PREPOLL, MPK_KDA_GATE_LATE, MPK_KDA_PROJ_SPLIT, MPK_MLA_WA_HYB, --proj-off,
  --gate-second, --late-light), the server overlap mode (MPK_K3_MOEIW / MOEAGGIW), test_overlap.py, trace/p1_an.py.
- Baked in the flags every library that uses the code sets the same way: RESET_IN_KERNEL, NO_REARM, RIK_ALL, RIK_CUGEN,
  RESET_PARITY, KARG_PREFETCH, END_RESET_RANK, RIK_TRIM, the MoE set (W13_STAGES 7, W2_DIE_OPEN, W13_TICKET_IN_UNIT,
  W2_EARLY_TICKET, START_BATCH, W2_LATE_WAIT, W13W2_ONEQ, W13_TAIL_SPLIT 0, FAST_TABLE 3, W13_EARLY_TICKET), DRR_FOLD,
  DRR_CHUNK, MLA_Q_LATE / NORM_WPRE / TR / SMX4, AR_LANE_W. Left as flags (libraries differ): GEMM_LDSW (0 | 3),
  GEMM_STAGES, GEMM_ACT_PAD, GEMM_L2WARM (+ _NODES), GEMM_LANE_ASM, ARG_RELOAD, TASK_PREFETCH, AR_POST_NOWAIT, MAX_RESETS;
  and the measurement switches (TIMING_BUILD, SOLO, PRE / EXPERT / NODE_END_STAMPS, STAMP_SET, EARLY*, LINEUP, LAUNCH_GATE).
  moe_experts.cuh 3085 -> 1190 lines, core.cuh 1043 -> 432; the HIP headers 12443 -> 8468.
- Proof (box logs final_fpre.sh / final_fpost.sh, mkfinal.py): kda and mla stripped lib.s identical (881298f36994,
  0e157f8339f3). The 5 MoE libraries differ by register numbering / spill-lane placement only (6-41 of 16k-34k
  instructions, the post builds fewer v_readlane / v_writelane; same SGPR / VGPR totals, no scratch); leave-one-out builds
  (bis_*): only the removal of an empty inline call in sum_gpus.cuh (allreduce_land_sentinel, its body if constexpr (0))
  and moe_experts.cuh's rewrite move it; all 8 files at their unifdef-only form give the identical lib.s. fin1 (20:22):
  all 7 harness + rank-library checks pass, every harness output bit-identical to the pre-cleanup refs (rank18 / 21 / 34 /
  37 ref.pt, 8-104 tensors each).
- Final comparison, same box session 10-16 (SGLang alone basefin 20:26-20:47, ours allfin 20:52-21:09 with the 7 fpost
  libraries; fin1 had a path typo in its server env, rerun as fin2): target forward span 11081.4 vs 20873.0 us (1.88x;
  srv37 before the cleanup 11028.0), step period 13670.5 vs 24889.8 us (1.82x), client 5.944 vs 9.144 ms/token; full
  GSM8K (1319, parallel 1) 0.950 at 221.1 tok/s (602 s) vs 0.947 at 141.6 tok/s (936 s). B300 span 9035: 1.23x.
- sgcfg1 (10-17 01:00-01:45, SGLang alone, one lease; is our baseline SGLang's fastest config?): srv_run.sh ATTN / REPL
  env (defaults = the old command line). GSM8K 200 (one at a time) tok/s / accuracy, step period, target forward span:
  current (triton, ReplaySSM) 148.2 / 0.990, 24816.5, 20911.1; no ReplaySSM 145.6 / 0.980, +199.4, +130.9; --attention-backend
  aiter (aiter's MLA kernels loaded: mla_a16w16_qh32_qseqlen4, prefill pads 12 heads to 16) 147.3 / 0.990, -50.6, -422.9
  (its MLA part +15, the time outside the target forward +372); both 146.8 / 0.985, +133.1, -93.0. None faster by more than
  run-to-run spread (cgcur vs basefin of 10-16: period 24816.5 vs 24889.8, span 20911.1 vs 20873.0); the current config
  keeps the highest throughput. SGLANG_ROCM_K3_KDA_FUSED_BACKEND=aiter applies to plain decode steps only (kimi_k3.py
  defer_f_b needs forward_mode.is_decode()), not to verify steps: not tested.
