# The K3 MoE layer from its math, on MI355X (2026-10-11)

Per GPU at TP8, decode T = 8 tokens, H = 7168, L = 3584 (latent), E = 896 experts, 16 per token, IR = 384 and SHR = 768 per GPU
(demo/kimi_k3_moe/demo.py). All numbers below are per GPU.

## What the layer computes

```
x [8, 7168] ──┬─ router:      logits = x · Wg^T        Wg   [896, 7168]  bf16  12.85 MB (every GPU, all rows)
              │               topk: sigmoid + bias, top 16 of 896, renormalise -> pairs
              ├─ latent_down: z = x · Wdown^T           Wdown[448, 7168]  bf16   6.42 MB (this GPU's 448 rows)
              │               MXFP8 quantise, ALL-GATHER z_q [8, 3584]                     (comm 1, ~29 KB)
              ├─ shared:      h_s = SiTU(x · Wsgu^T)    Wsgu [1536, 7168] bf16  22.0  MB
              │               S   = h_s · Wsd^T         Wsd  [7168, 768]  bf16  11.0  MB
              │               ALL-REDUCE S [8, 7168]                                       (comm 4, off the chain)
              └─ (needs pairs and z_q from every GPU)
                 experts:     for each of the ~90 experts used: W13 [768, 3584] fp4 + W2 [3584, 384] fp4
                                                         ~2.13 MB per expert          ~191 MB
                              R[t] = sum over the token's 16 experts, ALL-REDUCE over GPUs  (comm 2, 57 KB)
                 RMSNorm(R) x gamma -> Rn [8, 3584]     (needs the whole reduced row)
                 latent_up:   up = Rn · Wup^T            Wup  [896, 3584]  bf16   6.42 MB (this GPU's 896 rows)
                              ALL-GATHER o [8, 7168]                                       (comm 3, 114 KB)
                 y = o + S + prefix
```

Weights read per GPU per launch: ~250 MB (experts 191, everything else 59). At the best read speed measured on this box
(6.38 TB/s, amd/bench/read_bw.hip) that is 39 us; the chain below adds 3 communication steps on the critical path.

## The critical chain and what it should cost

1. router + latent_down: 19.3 MB -> ~3 us if ALL 256 CUs read them
2. topk (~3-4 us, latency) and quantise + all-gather z_q (comm 1)
3. experts: 191 MB -> ~30-38 us on all 256 CUs
4. all-reduce R (comm 2) -> RMSNorm (~1 us) -> latent_up 6.4 MB (~1-2 us on all CUs) -> all-gather o (comm 3) -> y

The shared expert (33 MB, ~5.5 us of reading) is NOT on the chain: it only has to finish before y. Rough target: ~55-65 us
span, vs 119 us measured now (TP8, 8 GPUs, profiled launch, GPU 0).

## Measured (TP8, 8 GPUs, GPU 0, times after the start barrier, logs/tp8.out)

- router 98 tasks 2.4 .. 10.2 us: 12.85 MB in 7.8 us = 1.6 TB/s (98 CUs)
- latent_down 24 tasks 2.4 .. 14.1 us: 6.42 MB in 11.7 us = 0.55 TB/s (24 CUs)
- topk ends 17.7; z_q in moe's LDS at ~19.7 (median CU)
- moe_experts: its 256 tasks start at 2.4 and WAIT until ~20 us for pairs and z_q; work ~20 .. 82 (median CU) / 88.6 (last
  CU): 191 MB in ~62-68 us = ~3 TB/s
- shared gate_up 96 tasks 2.4 .. 21.1, SiTU .. 26.8, shared down 56 tasks 10.1 .. 41.3 (on CUs moe also needs)
- R sent 80.5 .. 91.4; RMSNorm done 110.9 (waits for GPU 6, whose moe ended at 103.3); latent_up 49 tasks .. 116.9 (6.42 MB on
  49 CUs); o gathered .. 121.8; y .. 124.1

## Where the implementation departs from what the math allows (read in the code, 10-11)

1. CUs per GEMM (B300 task sizes). One workgroup per CU with 1 wave per SIMD reads ~25 GB/s per CU on MI355X
   (amd/bench/gemm_load.hip: 51 MB, 224 tasks, ~5.7 TB/s); a B300 SM reads ~54 GB/s. The B300 plan's task counts leave the
   chain's GEMMs on 24-98 CUs: latent_down 0.55 TB/s, router 1.6 TB/s, latent_up ~1 TB/s. On MI355X a GEMM on the chain needs
   ~200+ CUs (gemm_tile now has 16/32-row tasks for this).
2. 256 CUs wait ~17 us at the start: moe_experts tasks are placed first on every CU and sit waiting for pairs / z_q, while
   the work that could run then (the whole shared expert, 33 MB) is placed on other CUs and later competes with moe. From the
   math the first ~8 us should be: all CUs on router + latent_down, then the shared expert on the CUs not running topk /
   quantise, then moe on all CUs.
3. Weight layout: gemm_tile reads row-major weights (a wave load = 16 rows x 64 B, rows 14 KB apart); pre-shuffled weights
   (1 KB per wave load, as the MoE's) are 15-30% faster per GEMM (gemm_load.hip, in-layer solo task times 10-11).
4. Communication built for NVLS multicast: push16 = 8 separate system-scope stores per 16 B (no multicast on MI355X); the
   exchange region is plain hipMalloc (vLLM / AITER use hipDeviceMallocUncached); re-filled with 0xFF before every launch
   (AITER: epoch flags, no refill); readers poll every 16 B of every GPU's slot. A/B written (STATIC_XCHG_MEM, STATIC_XCHG_NT).
5. moe -> allreduce_send: a counter of all 256 moe tasks (every slice waits for the slowest CU), R through a 1.8 MB fp32
   buffer [T x 16][3584] written by W2 and read back by 256 tasks that add the 16 rows.
6. The slowest GPU sets the pace after moe (GPU 6: 103 vs ~87 us), so moe's own tail (median CU 82 vs last 88.6) and the
   GPU-to-GPU spread both land on the chain.

The tasks themselves (sum_quant_send, situ_and_mul, sum_gpus, sum_rmsnorm, sum_send, residual_add, allreduce_send) move a few
KB each; their measured durations are almost all waiting (read 10-11: each is a poll, a few adds, 16-B stores).

## Kernel changes measured 10-11 (solo TP8 on 1 GPU, e2e median of 30, 1 GB flushed, timing build)

- gemm_tile: weights in the shuffled layout (--gemm-shuffled: a wave's load 1 KB contiguous): GEMM tasks 20-30% shorter;
  weight loads issued before the activation wait (fill_act). Alone these did not move e2e (the GEMMs are not on the chain
  in the base plan).
- gemm_tile 16/32-row tasks and hand phase plans (V0-V3): all slower than the base plan. A small task costs ~4-5 us fixed
  (activation, weights, stores: one memory round trip after the other), so more, smaller tasks add round trips.
- moe_experts W13 layout (a step's 4 tiles adjacent): no change. W13 depth 4 vs 7 steps: no change.
- moe_experts W2 take-ahead (the next item's queue ticket issued before the current item's weight loads): W2 takes 71 -> 18
  us per CU (summed over 4 waves), e2e 121 -> 109 us. The same for W13 items: worse (a wave holds two items while CUs that
  start later find the queue empty) -> not kept.
- W2 h_q poll by the scalar unit (s_load glc): worse -> not kept.
- W2 polls h_q itself (0xFF-prefilled, no CHQ counter, no per-item store wait + post in W13): e2e 116 -> 104 us (3 rounds
  each: 116.8 / 117.6 / 113.1 -> 103.3 / 108.3 / 102.1), W2 phase after the last W13 wave ~33 -> ~24 us.
- W2 check moved to the compute (two items' loads in flight per wave, STATIC_W2_DEEP): W2 mid-phase 3.7 -> 4.9 TB/s but
  moe's last CU 70-73 -> 78-84 us (the extra W2 traffic slowed the last W13 items), e2e +5-7 us. Off.
- W13 tail split (STATIC_W13_TAIL_SPLIT=4, default): the last quarter of each die queue's W13 items as two units (2 gate
  tiles / 2 up tiles, full K, 14 steps in flight; the second to finish combines through a 2 KB buffer, same bits). Last W13
  item end 55.4-57.4 -> 52.8-53.5 us; span ~90 -> ~87; e2e ~97 -> ~94 (solo, 3 rounds each).
- W2 items of 4 tiles: worse (span 88-90 -> 93-96): an item costs ~2 load round trips and a take whatever its size.
- After the split, the trace (amd/tests trace dump, solo): every W13 unit done by ~51 us; waves start W2 at ~49 (median)
  and do 2-3 items of ~5.9 us; between a wave's last W13 unit and its first W2 load: 4.7 us (median) of queue reads and
  the W2 take in a row (STATIC_W2_EARLY_TICKET under test); W2 phase ~2.6 TB/s overall.
- Did not help: shared down after moe_experts (S path became critical), front plan (router on 196 CUs: each task ~5 us
  latency whatever its size), LDS weight copy for router / latent_up alone, kernarg prefetch (< 0.3 us of the ~4.4 us
  launch -> first CU), z_q / R / S / topk poll sentinels, 11-bit radix digits for topk (slower), latent_up prefetch plan
  (the displaced R-send slices cost what the prefetch saved).
- latent_up pushing its own rows (gemm_tile COMBINE_PUSH, one K part, no sum_send): rmsnorm end -> y 10.3 -> 6.5 us;
  overall within noise in 3 rounds (5-round rerun pending).
- TP8 (8 GPUs, STATIC_LAUNCH_GATE: one host store releases all 8 GPUs' queued launches; no cross-GPU start barrier): e2e
  135 us, span ~126-130 (solo ~93-95 / ~87). z_q reaches moe_experts at ~26-31 us (solo ~14.5): sum_quant_send tasks
  take ~11 us (2.5 solo); the router GEMM ends ~11-13 us (8.5 solo); the moe from z_q to its median CU end ~62-65 us (~50
  solo). Under test: whether 8 GPUs streaming at once are slower per GPU (read_bw / solo layer on 1 vs 8 GPUs at once).

## The hand-offs after moe_experts (10-11 afternoon; built in mirage_stage, A/B queued)

Solo, after moe_experts' last CU (72.6 us): R sent +4.3, RMSNorm +2.7, latent_up +4.3, sum_send +4.3, residual +1.5 = 17
us, five hand-offs in a row. In TP8 (logs/tp8.out) each one is ~5 us. The math needs only the all-reduce of R, the norm (a
per-token scalar), the latent_up GEMM and the all-gather of its output. Three changes remove hand-offs without changing what
is computed:
- latent_up finishes itself (gemm_tile COMBINE_FINISH, --lup-finish): each K part stores its sums and counts on its row
  block; the part that completes the count adds the parts in sum_send's order and pushes o. No sum_send node; y has the same
  bits as before.
- RMSNorm folded into latent_up (--fold): RMSNorm(R) x gamma . Wup^T = rstd x ((R x gamma) . Wup^T), rstd one scalar per
  token. Each latent_up task reads the GPUs' R rows for its K columns, adds them in GPU order, multiplies by gamma (bf16 into
  LDS) and keeps the sum of squares of its columns; the finishing part adds the K parts' sums of squares and scales by rstd.
  No sum_rmsnorm node. y differs from the unfused version in the bf16 rounding of the GEMM input (before rstd instead of
  after); the TP8 test checks y against that computation.
- R per 128-column block (-DSTATIC_R_BLOCKS=1): W2 units count their tiles per block; an allreduce_send slice waits for its
  blocks, not for all 256 moe_experts tasks. Same bits.
- topk: the pick's last phase (picks in expert order) was ~2100-3000 of its ~6000-8200 cycles (bench, warm or cold code:
  the code fetch is ~6% of the pick); -DSTATIC_TOPK_FAST=1 appends the picks by an LDS counter and ranks the 16 by expert id
  in one DPP row.

## 10-12: TP8 at ~101-103 us e2e (fold + R blocks + W13 tail split 12 + line-up start)

Where the time goes (TP8, timing build, GPU 0, us after the start barrier; B300 09-30 build in brackets):
- launch -> first CU 4.5 (platform: two dependent full-grid kernels are 3.6 us apart, stream or graph, amd/bench/launch_gap.hip)
- z_q in moe's LDS 17.8 (route done 14.0 + z_q 4 later ~18)
- W13: median CU's W13 work ends ~55, ~37 us after z_q (B300: 23)
- W2: median moe end ~77, ~22 us (B300 18.7); TP8 W2 is 5.6 us longer than solo (polls of 8 GPUs' slots by the tail tasks
  on CUs done with moe: STATIC_TAIL_SLEEP under test)
- last moe CU 77-83 across GPUs (B300 last SM 65.4); after it ~18.7 us to the end (B300 17.9)
W13 alone (amd/bench/expert_unit.hip) reads ~3.5-3.9 TB/s whether items go to waves or CUs (STATIC_W13_CU), double-
buffered or not, tile- or step-major (W13_SM): about AITER's tuned W13 (~4 TB/s); inside the layer W13 reads 4.6-6 TB/s
while every wave has a unit, then drains (42-52 us solo: 2-4 TB/s while W2 starts). W2 alone ~4.7 TB/s; the layer's
per-die queue costs ~2.7 us of it, device-scope h_q / R ~0.9.
Measured and off: W13 quarters (STATIC_W13_QUARTER), adaptive / late-wait W2 pipelines, W2 16-tile items (spills),
fast topk pick, cross-GPU warm loads, LDS latent_up weights, poll sentinels, uncached exchange, z_q finish in latent_down
(--zq-finish: 102.0 / 101.6 vs 102.2).

### 10-12 TP8 trace of moe_experts (p12, GPU 0, us after the start barrier; amd/tests trace dump)
- moe moves ~197 MB (W13 ~131, W2 ~66) from 22 to 80 us: ~3.4 TB/s on average; at 5 TB/s it would take ~39 us.
- 56 CUs run the shared expert's down GEMM (node 34, 9.2 .. 33.2 us: it waits for situ_and_mul until ~25) and enter moe at
  ~33; they load z_q (~9 us) and take the last W13 halves at 42-46. Those take 14-21 us (other CUs' halves then: ~11)
  and end last: the last slot's W13 is done at 63.8 (solo: 53.0).
- From its last W13 unit to its first W2 load a wave spends 4.7 us (median: own-queue add, 8 counter reads, W2 add); waves
  that leave W13 at 42-46 wait 8-12 us (W2 only after every W13 queue is taken).
- W13 halves started at 40-46 take 11-21 us in TP8 vs ~7 solo; W2 items take 9-12 us (solo 7-9): ~1000 waves with two W2
  items' loads each in flight while the last W13 units run.
- Late CUs doing W2 only (STATIC_W13_LATE_SKIP=6: the 56 CUs, by the trace): 126.1 (that run overlapped another lease's
  job) and 114.3 vs ~102: their waves wait for h_q with short polls while the last W13 units write it. Off.
- Shared expert first on the CUs that start with moe (--shared-early): 102.2 (r1), profile pending.
- From 10-12 11:14 every GPU run goes through the box's lease queue (/usr/local/lib/cresco/gpu_lease.sh); earlier runs
  raced other users' leased jobs (the overlapped stops, the 51-65 ms launch outliers).
- Read speed over time (p12 trace, 2-us bins, bytes of each unit spread over its duration): 23-45 us W13 at 4.2-4.8 TB/s
  (700-870 of 1024 waves on a W13 unit; the shared-down CUs' waves elsewhere); 47-63 us 1.8-2.9 TB/s (W13 drains, W2
  ramps up: ~400 waves on neither); 63-73 us W2 at ~3.4 TB/s; 73-80 drain. Against 4.6 TB/s the 47-63 dip costs ~7-8 us,
  W2 ~2.5, the ramp at 22-27 ~2, the drain ~2.5.
- Last W13 unit end -> first W2 load, per wave: 10.3 us (waves ending 42-45 us: they race for the last W13 items of the
  other queues until ~46, read-8 + add per try), 6.7 (45-48), 3.6-4.6 (48-66: own queue add, 8 counter reads, W2 add, one
  after the other). STATIC_W13_TICKET_IN_UNIT (1: the next own-queue ticket issued after the unit's last weight loads; 2:
  after its sums) hides the first; with STATIC_W2_EARLY_TICKET the other two are one round trip. Off by default (the
  flag-off build is code-identical to before); A/B queued.
- read_bw (logs/t1.out, 10-10, the corrected bench): chunked and interleaved reads are the same speed (nt, 1 workgroup per
  CU: 3.4 / 5.1 / 6.25 TB/s at 16 / 32 / 64 KB in flight per CU); the read pattern is not what limits W13.
- Lease queue: gpu_lease.sh only grants tickets of the best waiting priority class; our priority-2 tickets wait while any
  priority-1 ticket is queued (10-12 12:44 -> 14:40+ without a grant).
- Per-CU read speed in the W13 main phase (p12): a full item (112 KB) 18.3 us, a half (56 KB) 11 us: ~24 GB/s per CU,
  read_bw's best per CU (6.25 TB/s / 256). moe's read speed is the number of CUs on it: ~180 of 256 at 23-42 us (56 on
  the shared down GEMM until 33, 12 on situ_and_mul until 25, then each loads z_q). Built 10-12 (off by default, the
  flag-off builds code-identical), A/B queued:
  - STATIC_W2_DIE_OPEN: W2 queue x opens once the W13 queues holding its experts are seen empty (exact queue range,
    checked against brute force for 1-129 experts); a wave leaves W13 as soon as its own die's W2 queue is open.
  - --shared-late (plan_one_gpu.shared_late_plan): situ_and_mul, the shared down GEMM and its send after moe_experts'
    R send, on CUs without sum_gpus / latent_up; all 256 CUs then enter moe_experts by ~18-22 us.
  - W2 holds one item's loads in flight per wave by default (the h_q check right after the loads waits for the
    weights: loads complete in order); STATIC_W2_DEEP / ADAPT (two items) lost on 10-11 when the last W13 units were
    the tail; re-measured on top of the above.
- Solo vs TP8 (p12_s / p12_t, same plan and items): W13 starts 19 vs 23 us; main-phase units 15.5 vs 18.3 us (full) and
  9.7 vs 11 (half); W2 items 7.7 vs 9-15 us, the W2 phase 5.2 vs 3.4 TB/s; moe's last unit 68 vs 82 us; after it 11.8
  vs 18.7 us to y. The front's GEMMs take the same time in both (router 6.5 / 6.2 us). 8 solo runs at once (pwr8, 10-11)
  were as fast as one: streaming on all 8 GPUs is not the cause; the TP8 run's exchanges or its clocks are the
  suspects. The profile build now records each wave's shader clock over its moe items (trace kinds 8, 9).
- Outside references (read, not run; 10-12): AITER's tuned K3 decode configs (aiter commit e6ded21,
  kimik3_a16w4_tuned_fmoe.csv, model_dim 3584, inter 384, E 896, topk 16, 256 CUs, 8 tokens): one-stage asm
  fmoe_bf16_pertokenMXfp4_g1u1_flat_situv2_16x192 50.3 us; two-stage FlyDSL a4w4 51.1 us (sum of kernel times under
  torch profiler, random routing: ~119 experts, ~262 MB, so ~5.2 TB/s if those numbers hold). The flat kernel: grid
  (inter / 192, topk, tokens) = 256 workgroups, each a (token, pick, 192-feature slice) doing gate/up and then that
  slice of W2, output by bf16 atomic adds: no W13 -> W2 wait, equal work per workgroup, no queues (it reads an
  expert once per token that picked it). Our B300 kernel: one queue (W13 items, then W2 entries), the next entry taken
  while the current item's last stages are armed (W13_TAKE_AT), each W2 entry waiting on its expert's W13 count.
- 10-12 18:41 TP8 round 1 (40 launches each, 1024 MB flushed, e2e median): base 101.7; STATIC_W13_TICKET_IN_UNIT=1 94.0,
  =2 96.4, +W2_EARLY_TICKET 93.8; DIE_OPEN alone 101.6, DIE_OPEN+TICKET+EARLY 94.8 (+DEEP 95.6, +ADAPT 94.2);
  STATIC_EARLY 101.4; W13_PRE 102.5; --shared-early 102.4; --shared-late 106.8; latent_up regrid 112 / 224 rows 103.9 /
  106.8; LATE_SKIP=6 + POLL_SLEEP=16 112.7.
- Shader clock (profile builds, s_memtime cycles / ns per wave): TP8 2.32 GHz median, solo 2.31: not why TP8 is slower.
- TP8 base trace (GPU 0): 45 of 90 experts' W13 done by 48 us, but 2471 of 2520 W2 items end after the last W13 unit
  (64.3 us); the 1123 W2 items started before it mostly wait for h_q. W13 units start in two bursts (639 at 24-28 us,
  645 at 40-44 us); the second burst's halves took 15.9 us (8 at other times) while ~380 W2 waves re-loaded their slot's
  ~6 KB h_q every round trip (POLL_SLEEP 1) -> STATIC_W2_POLL_LITE (poll the 24 scale bytes, then h_q).
- TP8 moe end differs by GPU with the same work: 77.7 .. 87.2 us (latest task end); the slowest sets the tail. Tail on
  GPU 3: moe end 87.2 -> allreduce_send 90.5 -> gemm_tile#39 (latent up, waits for all 8 partial R) -> residual_add
  106.2 us.
- 10-12 19:00 rounds 2-3 (TP8 e2e median): STATIC_W2_POLL_LITE did not help (li 103.0 / 101.6 vs base 106.0 / 103.4;
  tk1 li 96.3 / 95.6 vs tk1 95.5 / 95.8; dt li 94.2 / 94.2 vs do1t 95.0 / 94.3); POLL_SLEEP 16 with tk1 96.4 / 96.0.
  So the h_q polls' traffic is not what slows the second W13 burst. In the base trace no wave is in W2 at 40-50 us, but
  ~380 waves go through the W13 queues' counters (8 loads + an add per try) -> amd/bench/hot_line.hip (queued).
- Late CUs (TP8 trace): the 56 CUs that run the shared expert's down GEMM (node 34) start moe at ~33 us and their first
  unit at 42.9 (base) / 49.8 (tk1, a W2 item) / solo 38.7. Plan option --pack 34:N (plan_one_gpu.pack_plan): down's 56
  tasks N per CU on 56 / N CUs (its grid cannot go below 56; its output is needed only at the residual).
- moe start phase: z_q pieces, then z_q's scale rows (serial polls), then the routing pairs, then the first take: up to
  four round trips in a row. STATIC_START_BATCH=1 loads the scale rows with the pieces and the pairs before them. Profile
  builds now trace kinds 10 (task start -> z_q in LDS) and 11 (-> table).
- latent_down on more CUs: --regrid 28:28,14,1: compiles (49 tasks per GPU; spills 19 / 46 in builds 1 / 4); 28,28 hits
  gemm_tile's 14-K-part finish limit.
- 10-12 21:33 nx (TP8 e2e median, 40 launches, cold; on top of DIE_OPEN + TICKET_IN_UNIT + W2_EARLY_TICKET = "dt"):
  dt 95.2 / 95.4 / 94.5; + START_BATCH=1 93.0 / 93.0 / 94.0; + --pack 34:4 93.8 / 93.6 / 92.7; + --pack 34:2 94.7 / 94.3 /
  94.3; + W13_TAIL_SPLIT=16 94.5 / 94.6 / 93.9; + POLL_LITE 94.2 / 94.4 / 94.8; + W2_TAIL_SPLIT=4 97.8 / 96.4 / 95.5;
  + W2_TAIL_SPLIT=8 + POLL_LITE 98.4 / 98.1 / 97.8; latent_down --regrid 28:28,14,1 101.6 / 100.4 / 101.4.
- hot_line (1 GPU, 3 waves per CU streaming, 1 per CU hammering): atomic adds on ONE line cut the readers from 2.9 to
  0.87 TB/s (88 adds per us); on 8 or 64 lines, or 8 counter loads per try: ~3% slower. moe's queues use 16 lines.
- xgmi_exchange (8 GPUs, flag-based: stores, system release fence, a flag add per peer): one exchange 5.1 us at 28 KB
  per peer (7 sender workgroups), 8.6 us at 112 KB (14-28); MORE senders are slower (28 KB: 112 senders 14.8 us).
- Start phase (profile build, kinds 10 / 11): z_q -> table took a constant ~4.6 us on every CU, even on the late ones
  whose pairs had landed long before: the timing build's spill reloads (scratch, vmcnt(0)) in expert_table; the
  normal build has none there (only the pair poll). START_BATCH=2: the pairs re-polled with z_q, kept in LDS (sel /
  wsel), so the table loads nothing.
- TP8 vs solo, same build (base, profile): W13 units starting 20-30 us take 16-19 us (whole) / 11-13 (halves) in TP8 vs
  14-15 / 8-9.5 solo; units starting 30-40 the same in both (~8-9 for halves). W2 phase at the same number of items in
  flight (~1700) 3.9 vs 5.2 TB/s. Probe queued: TP8 with z zero from L/8 on (--zero-z-from 0.125, what solo's GPU 0
  computes with) -> whether the data (MFMA power) or the exchanges make TP8 slower.
- Halves vs whole items (TP8, starts 20-30 us): a half (56 KB) 11-13 us, a whole item (112 KB) 16-19: ~3.5 us per half
  above half a whole item: the half's end waits for its stores, then the count, then (the second) the other's sums,
  with no weight loads in flight. STATIC_W13_HALF_DEFER: count and sums go out unwaited, the count read a unit later
  (after the next unit's sums), the other's sums polled (0xFF until stored, re-armed by the second half); trace kinds
  12 / 13 (profile builds): a unit's sums -> its end.
- Tail (TP8 sb profile, every GPU alike): allreduce_send's last slice ends 77-79 us, latent_up's R input lands 84-86
  (median) / 87-89 (last): ~5-7 us per hop although each CU sends 14 vectors. amd/bench/push_latency.hip (the layer's
  data-as-flag pattern, senders 8-256, sys / nt stores, with or without an HBM stream on the other CUs) queued.
- 10-12 23:00 nx2 (TP8 e2e median, 3 rounds; dt = DIE_OPEN + TICKET_IN_UNIT + W2_EARLY_TICKET): + START_BATCH=1 94.5 /
  92.8 / 93.4; + START_BATCH=2 94.5 / 94.4 / 93.9; + SB1 + --pack 34:4 93.3 / 92.3 / 93.7; + SB2 + pack 93.4 / 92.8 / 93.3;
  + SB2 + W13_EARLY_TICKET 94.2 / 95.7 / 94.3; + SB2 + W13_HALF_DEFER 95.8 / 95.6 / 96.0 (+ pack 97.1 / 96.7 / 96.4);
  + SB2 + W2_SPEC (margin 128) 107.6 / 106.0 / 107.6; SB1 + --zero-z-from 0.125 94.6 / 94.2 / 92.8 (= SB1: the data /
  MFMA power is not why TP8's units are slower than solo's).
  HALF_DEFER does what it was built for (profile: a half's sums -> end 2.96 -> 0.16 us median, halves starting 20-30 us
  12.0 -> 9.2 us) and the timing build is no slower, but the normal build loses ~2 us e2e (same spills); off.
- Wave states over time (sb2 + pack profile, GPU 0, 968 waves): at 44-54 us 105-353 waves sit between their last W13 unit
  and their first W2 item (2.3-6 us each; those ending W13 at 39-45 all start W2 at ~47.5) while ~1000 W2 items are ready:
  the race for the other dies' last W13 units -> STATIC_W13_NO_STEAL (own die's W13 queue only, then W2).
- The drain (same profile): the last W2 items start at 64-66 us and end 72-73 (dies 5, 6 last, 3 us after die 0); each
  wave keeps one item's weights in flight and computes the previous item only after the next one's weights land.
- Tail per GPU (same profile): moe end 74.2-79.5 us (5 us spread, same work), allreduce_send ends ~3.5 us after, R lands
  at latent_up ~5 us after the latest send, latent_up 2.6-4 us, residual_add ends ~93.
- 10-13 00:01-00:20 (TP8 e2e median, 40 launches, 3 rounds; "b" = DIE_OPEN + TICKET_IN_UNIT + W2_EARLY_TICKET +
  START_BATCH=1 + --pack 34:4):
  - nx3: b 92.7 / 92.4 / 92.3; + W13_NO_STEAL 92.8 / 91.3 / 90.5; allreduce_send #37 on 64 CUs (grid_choices, --regrid
    37:64,1,1:36) 92.2 / 91.6 / 92.1, on 32 95.4 / 95.2 / 95.2, on 8 119.3 / 119.0 / 119.6 (fewer senders: slower);
    --zq-finish 93.8 / 93.6 / 93.4; probe DBG_ZQ_OWN (each CU loads only its GPU's z_q, wrong outputs) 92.1 / 91.6 / 92.3
    and its first W13 units as slow as b's (die 3 slowest in both): not the z_q reads.
  - nx4: b 92.8 / 94.0 / 93.4; + W2_LATE_WAIT 91.7 / 92.8 / 91.8; + NO_STEAL + LATE_WAIT 90.8 / 90.3 / 90.3; + ADAPT
    96.5 / 92.7 / 93.0; + TS16 93.2 / 93.8 / 93.6; STATIC_EARLY 95.6 / 94.8 / 96.9; latent_down 49 tasks 95.4 / 96.0 /
    97.3 (latent_down ends 1.2 us earlier, z_q 0.7, but some W13 units end 19 us later).
  - nx5 (b2 = b + NO_STEAL + LATE_WAIT): b2 90.7 / 89.8 / 94.0; W13W2_ONEQ instead of NO_STEAL 91.2 / 90.4 / 91.3; + R
    sends on 64 CUs 91.2 / 90.4 / 90.3; ONEQ + 64 CUs (b3) 90.1 / 90.4 / 90.2; TS16 92.4 / 91.7 / 90.9.
  - nx6: b3 92.7 / 90.6 / 89.2; b3 + STATIC_EARLY 93.6 / 94.2 / 94.0 (start barrier -> last CU end 92.5-93.2 vs
    83.2-85.6). EARLY's profile: GPU 0's front as fast as without, but z_q in moe's LDS 11.9 us after the task start vs
    6.5: the host launches the GPUs one after the other; the line-ups all end when the last GPU's starts, and that
    GPU's layer kernel is then still being dispatched (its front ~5 us late, everyone's z_q with it).
    -> STATIC_EARLY_RES: the layer's CU 0 posts "resident" to every GPU; each line-up kernel waits for every GPU's post
    (200 us cap), 2 us more, then writes its own GPU's go; the layer's CUs wait for that go. nx7 queued.
- Timing-build tail (b3 / ONEQ profiles): moe end -> residual end ~17 us on the slowest GPU (B300: 17.9); the rest of
  the gap to 85 is the moe body and the ~4.5 us launch -> first CU.
- 10-13 01:20 STATIC_EARLY_RES (b3 = b + LATE_WAIT + W13W2_ONEQ + R sends on 64 CUs; host resets): TP8 e2e 83.0 / 82.5 /
  82.6 (nx7) vs b3 90.2 / 90.6 / 88.6; nx9 83.6 / 84.2 / 83.7; nx11 83.6 / 83.5 / 83.7. The layer's CUs pass the start
  barrier ~0.1-0.4 us after the start event (no work before it).
- push_latency (8 GPUs, data as the flag, lockstep): R-sized 57 KB per GPU one hop 3.7 us with 32-256 sender CUs (8:
  7.3), 14 KB 3.5-3.6; an HBM stream on the other CUs barely matters; nt stores slower (4.4-4.9). In the layer R lands
  ~5 us after the sends: ~1.3 us above this floor; the tail (~17 us, B300 17.9) has little left.
- Resets in the launch's time (B300 production reset in the kernel): the host's per-launch resets are ~4.4 MB per GPU
  (z_q / K-part slots, R / S / latent_up exchange buffers, h_q, misc counters). STATIC_RESET_IN_KERNEL does not build with
  R_BLOCKS / the RMSNorm fold (no re-arm there). STATIC_RIK_ALL (+ NO_REARM: the kernel resets every buffer at its start,
  the exchange ones in the next launch's set, then a barrier over the CUs): e2e 98-102 (release / acquire barrier, the
  last CU in -> past it 9.7-13.2 us), 94-98 with a relaxed barrier (7-10 us). STATIC_DBUF_RESET: two copies of every
  reset buffer (the host points G / the maps at copy launch & 1), the launch resets the next launch's copy and exchange
  set without waiting or a barrier; all checks and the last timed launch bit for bit pass: at the start 88.6 / 88.4 /
  88.8 (the stores ahead of the first tasks' loads, vmcnt in order); before each CU's last task: nx12.
- 10-13 02:00-03:10 resets counted like B300 (TP8 e2e median, 40 launches; host resets = memsets after the 1 GB flush,
  so the reset buffers start warm and their cost is not counted):
  - The B300 harness that matched its server (test_plugin --prev-us 5): a 5 us stand-in previous kernel, the MoE layer
    launched with PDL; its in-kernel resets and their barrier ran before griddepcontrol.wait (static_megakernel/core.cuh:
    "everything up to the TMEM allocation runs before the wait"), overlapping the 5 us.
  - DBUF_RESET (two copies, the next launch's reset without a barrier) 88.2-88.8: the 4.3 MB of reset buffers are then
    cold after the flush (the front's K-part slots, pairs, counters): the front ~1.6 us slower, moe end +4.
  - STATIC_EARLY_PREWAIT (RIK_ALL + EARLY_RES: every reset and the barrier before the go; the go 5 us after every GPU's
    layer is resident, STATIC_EARLY_PREV_US=5): 87.2 / 87.0 / 87.7 (nx13), 86.8 / 87.2 / 87.4 (nx16), body = host-reset
    body (~83); the first CU past the go 3.1 us after the start event: resets + barrier ~8 us vs the 5 us window.
    Exchange resets moved to the last task: 89.7-90.2 (worse). W2_LATE_SKIP (late CUs take no W2 item): 93.6-96.8 (worse).
  - STATIC_RIK_DIE (64-KB slices in per-die pools, one claim per CU, barrier on the dies' done counts): nx16 hung (claim
    bases assumed 32 workgroups per die per launch; the one waiting for the line-up kernel's CU lands elsewhere): now
    per-launch-parity counters zeroed by the previous launch (nx17).
- Best TP8 command line (10-13; amd/tests/test_layer_one_gpu.py, from the repo root):
  --gpus 8 --wait 60 --flush-mb 1024 --gemm-shuffled --fold --e2e-reps 40 --pack 34:4 --regrid 37:64,1,1:36
  --flag=-DSTATIC_W13_STAGES=7 --flag=-DSTATIC_LINEUP
  --flag=-DSTATIC_W2_DIE_OPEN=1 --flag=-DSTATIC_W13_TICKET_IN_UNIT=1 --flag=-DSTATIC_W2_EARLY_TICKET=1
  --flag=-DSTATIC_START_BATCH=1 --flag=-DSTATIC_W2_LATE_WAIT=1 --flag=-DSTATIC_W13W2_ONEQ=1
  launch, host resets (not counted): --flag=-DSTATIC_EARLY --flag=-DSTATIC_HELLO_WAIT_ONLY --flag=-DSTATIC_EARLY_RES
  launch + resets in the kernel before the go (as B300's PDL path, 5 us stand-in; 10-13 nx22/nx23: 83.3-84.6): the above +
  --flag=-DSTATIC_RESET_IN_KERNEL --flag=-DSTATIC_NO_REARM --flag=-DSTATIC_RIK_ALL --flag=-DSTATIC_EARLY_PREWAIT
  --flag=-DSTATIC_EARLY_PREV_US=5 --flag=-DSTATIC_RIK_DIE --flag=-DSTATIC_RIK_TRIM --flag=-DSTATIC_RIK_PAR
  --flag=-DSTATIC_RIK_STATIC --flag=-DSTATIC_RIK_SLICE=16384 --flag=-DSTATIC_KARG_PREFETCH=1 --flag=-DSTATIC_TASK_PREFETCH
- 10-13 06:30 nx17 (like for like with B300, e2e median, 40 launches, cold): pw5 (resets over all CUs + barrier) 86.8 / ~87
  / 87.3; + STATIC_RIK_DIE (64-KB slices, per-die pools, per-launch-parity counters, barrier on per-die done counts) 85.4 /
  85.9 / 85.6; 24-KB slices 85.8 / 85.4 / 86.4. All checks and the last timed launch bit for bit pass. The resets now fit
  the 5 us stand-in: the CUs pass the go ~0.6 us before the start event's timestamp (the line-up's last store is its go;
  the event follows its completion), so counted from the go ~86.2. The workgroup waiting for the line-up kernel's CU
  (GPU 0: CU 255, list moe + residual_add) starts ~4 us after the others. Timing build: die 86.4 vs eres (nx7) 86.5, the
  per-node times the same once shifted; same-session normal-build A/B queued (nx18).
- 10-13 nx18 (same session, e2e median, 40 launches, cold): eres (host resets, not counted) 83.4 / 84.3 / 84.5 / 83.0;
  RIK_DIE 86.0 / 84.4 / 85.4 / 86.0. Profile: the GPU launched last (7) passes its go ~4 us after the others relative to
  its tasks (its first tasks' inputs land 5.6 us after its start reference vs 2.0 on GPU 0; its moe task start 9.2 vs
  ~5.8), its moe ends ~5 us after the others', and every GPU's sum_gpus waits for its R: the other GPUs' layers wait at
  their resident post longer than 5 us for GPU 7's, GPU 7 only gets the 5 us stand-in for its 4.3 MB of resets + barrier.
- 10-13 nx19: STATIC_RIK_LATE (only the front's buffers, ~1.4 MB, before the go; h_q, its scales, the queues, the shared
  output sum and the exchange next set by the CUs starting moe_experts, all done before the routing table): 88.8 / 88.8 /
  89.1 vs RIK_DIE 86.2 / 87.0 / 87.1, eres 83.8 / 83.6 / 83.3 (lost: z_q in LDS 7.2-7.9 us after the moe task start vs
  6.0-6.4, moe ends 68-74 vs 64-71).
- 10-13 nx20 (same session, 3 rounds): RIK_DIE 87.5 / 86.9 / 87.2; + STATIC_RIK_TRIM (no reset of the 512 KB item trace in
  normal builds) 87.0 / 86.9 / 86.9; + STATIC_RIK_LATE + STATIC_RIK_XLATE (the exchange next set, 1.5 MB, reset at moe's
  start, no wait) 86.9 / 86.8 / 86.1; 16 / 32 KB slices same; eres 84.3 / 82.8 / 83.5. The reset list (STATIC_PRINT_RESETS=1):
  4.3 MB per launch (h_q 917 KB, z_q K-part sets 917 KB, R sets 458 KB, misc 665 KB, ...); bytes are not the cost.
- 10-13 nx21, STATIC_PRE_STAMPS (normal build: per CU entry, reset barrier passed; harness prints per GPU): the CUs pass the
  reset barrier ~10 us after the GPU's first CU entered; GPU 7 (launched last) has ~8.5-10.5 us from its first CU in to the
  go, so part of its CUs pass the go late. Two round-trip cuts:
  STATIC_RIK_PAR: the 8 dies' done counts loaded together per poll (were 8 L2-bypassing loads in a row);
  STATIC_RIK_STATIC: slice s -> workgroup s (all but the last workgroup, which waits for the line-up kernel's CU), no atomic
  claim, no workgroup barrier before the stores, done adds on line sm_id % 8.
  e2e (2 rounds): tx + PAR 85.9 / 84.0; tx + PAR + STATIC 83.6 / 85.2; + 16 KB slices 83.0 / 85.4; eres 83.3 / 83.4; die
  86.4 / 86.2. GPU 7's CUs past the barrier 6.8 us after its first CU in (was 9.9). Timing build aligned at each GPU's go:
  the front 1-2 us slower with XLATE (gemm#26 last end 10.5 vs eres 8.1, die 9.2), moe end 72-76 vs eres 66-70.
- 10-13 nx22 (3 rounds, e2e median; t = RIK_DIE + RIK_TRIM + RIK_PAR + RIK_STATIC + RIK_SLICE=16384, like for like):
  eres 83.4 / 83.2 / 84.1; t 84.1 / 83.8 / 84.4; t + KARG_PREFETCH=1 83.7 / 83.6 / 84.2; t + KARG_PREFETCH=1 +
  STATIC_TASK_PREFETCH (the SM's first list entry loaded before kernel_begin; timing build: go -> first task start was
  2.2-4.4 us, one cold load of the task table) 83.4 / 83.3 / 83.8 = eres. STATIC_PREGO_PREFETCH (waves 1-3 read the first
  GEMM task's weight tile into the L2 by LDS DMA while thread 0 waits for the go): 84.7 / 85.7 / 85.2, + KARG 86.6 / 85.2 /
  84.6 (lost). Its profile: GPUs 0-6's front ~3.5 us earlier (sum_quant_send end 9.4-9.8 vs 12.9-13.5 after the go), GPU
  7 (launched last, ~5 us window, used by the resets) gains little and z_q needs every GPU's part; moe's W13 phase ~2.5 us
  longer (the layer is HBM bound: moving GPUs 0-6's front reads earlier gains nothing while GPU 7 gates). It pays only if
  the pre-go resets get much shorter on every GPU. Pre-go split (PRE_STAMPS, normal build): CUs in ~0.5 us, reset stores
  landed ~5.0 (last ~6.7), past the barrier ~8.2, GPU 7's go ~8.9-9.5 after its first CU in.
- 10-13 nx23 (same session, 5 rounds, e2e median of 40 launches each): like for like (the command line above) 84.1 / 84.4 /
  83.9 / 84.6 / 84.0 (nx22: 83.4 / 83.3 / 83.8); eres (host resets after the flush, not counted) 83.2 / 83.4 / 84.8 / 84.0 /
  83.6. All checks and the last timed launch bit for bit pass. Per-GPU e2e medians within ~1 us of each other (GPU 7 no
  longer late). Probe STATIC_COLD_RESETS_MB=512 (the host's reset buffers evicted before the launch): eres 85.9 / 86.5,
  like for like 85.2: the in-kernel resets leave their buffers warm; a cold reset set costs ~2.4 us.
- 10-13 vs SGLang (amd/bench/sglang_moe_layer.py: SGLang's own KimiK3MoE from the box image, SGLang 59eb71a8 / AITER
  e7d2453f, MI35x recipe env, TP8, HIP graph, our harness's weights and 8 tokens, 90 experts; same timing): SGLang e2e
  142.0 / 144.2 / 145.2 us, ours 83.6 / 82.7 / 83.8 in the same jobs. SGLang per kernel (profiler): front GEMM 20.1, copy 4.1,
  SiTU-mul 4.2, shared down 6.8, top-k 6.9, sort 4.1 + 5.6, MXFP8 quant 4.6, GEMM1 29.0, GEMM2 15.7, copy 4.4, all-reduce
  33.9, RMSNorm 4.4, up GEMM 10.5, add3 3.4.
  Outputs (nx27-nx30): the same 16 experts per token (8/8), routing weights within 5.7e-5; y - prefix differs by relRMS
  2.2e-2. Cause: SGLang's ROCm front GEMM writes z in bf16 (kimi_k3.py _front_fp32 is False on HIP) and MXFP8 quantizes the
  bf16 values; ours quantizes fp32 z. References (y - prefix relRMS): unquantized z and h: ours 2.58e-2, SGLang 2.62e-2;
  MXFP8 of fp32 z: ours 3.97e-3, SGLang 2.21e-2; MXFP8 of bf16 z: ours 2.18e-2, SGLang 6.04e-3. Error uniform over tokens
  and columns (no layout problem). SGLang's shared expert within 3.7e-3 of float64 (its bf16 output).
- 10-13 moez (numerics directive, no-cost change, A/B on the same harness weights vs SGLang's own KimiK3MoE): sum_quant_send
  rounds z to bf16 before MXFP8 (STATIC_Z_BF16, default 1; SGLang ROCm's front GEMM writes z in bf16). Ours vs SGLang: y relRMS
  5.48e-3 (z fp32: 1.95e-2), y - prefix 6.24e-3 (2.22e-2); bf16 values equal 24124 of 57344 (7298). All harness checks pass with
  the reference following the flag. Kept as the default.
- 10-13 server (SGLang 0.5.21 ROCm, full K3 + DSpark, TP8, rank 1 trace, medians of 8 target forwards): SGLang alone span
  20867 us, 9.38 ms/token; ours MoE on all 92 layers (z fp32 build) span 18222 us, after-KDA window (MoE + next aggregation 1)
  93.9 us/layer vs 120.1, 8.15 ms/token; 1 of 8 greedy answers the same (first difference at tokens 5-101).

## 10-13 23:05: the per-layer floor and where the server time goes
- Infinity Cache is not faster to read than HBM here (read_bw t1: 256 CUs 4.67 TB/s from MALL vs 6.0 from HBM; 192 CUs
  5.48 vs 5.38): the limit is the per-CU read rate (~24 GB/s at 64 KB in flight), so a weight prestage into MALL buys nothing.
- Bytes per GPU per KDA layer + its MoE: 114 MB + ~190 MB (MoE experts ~90 x ~2 MB MXFP4, my estimate from the shapes)
  -> ~50 us at 6 TB/s if every CU streams all the time. Server today: agg1 5.5 + KDA 56.5 + MoE 93.6 = ~155 us per layer;
  most of the difference is CUs not streaming: the attention tail (recurrence, o_proj wait, residual), the MoE front and tail
  (cross-GPU latency chains), and each kernel's start (~11 us). Hence: one kernel per layer (attention + MoE) with the MoE
  front and its weight loads started on the CUs free after the projection (test_kda_moe_layer.py --early-front,
  STATIC_GEMM_LDSW=4), the start barrier removed (STATIC_RESET_PARITY), and later the next layer's weights in the same way.
