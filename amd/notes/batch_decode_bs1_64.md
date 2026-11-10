# K3 plain decode at batch 1..64 on 8x MI355X: every optimization (10-22 .. 10-27)

Scope: Kimi K3, TP8 on 8x MI355X (gfx950), plain decode (one new token per request) at batch sizes 1, 2, 4, 8, 16, 32, 64,
our MoE + KDA + MLA megakernels inside SGLang ROCm v0.5.21 vs SGLang alone. Goal (10-22): forward span >= 1.88x faster than
SGLang alone at every batch size, in the same server session, GSM8K unchanged. Work tree `~/megakernel/mirage_amd_bs` (box
`~/mpk-amd/mirage_bs*`). Every job, number and verdict is in `~/megakernel/amd_runs/2026-10-22/STATUS_decode.md`; the rules
and tools are in the skills `amd-kernel-development` and `amd-megakernel-development` (section 8).

Terms used below:
- **T**: the batch size a library is built for (one library per kind and T). **bs**: the server's batch size.
- **Fused KDA / fused MLA kernel**: one kernel per layer pair = MoE layer L + layer L+1's aggregation 1 + layer L+1's attention
  part (server kinds `moeaggkdad` / `moeaggmla`). 68 KDA and 23 MLA layers per forward; these two kinds are ~99 % of a step.
- **Front**: router, top-k, latent_down, z_q (the quantized latent sent to every GPU), shared expert, expert table.
  **Experts**: W13 units (gate/up GEMM + SiTU + quantize to h_q) then W2 items (down GEMM into R rows).
  **Tail**: R send (R to every GPU), sum_rmsnorm, latent_up, moe_agg1, then the next layer's attention (KDA: projection, conv,
  recurrence, o_proj, attn_residual; MLA: q/kv_a, norm, q_b, absorb, attention, vgate, o_proj, attn_residual).
- **Harness**: our one-kernel test (`amd/tests/test_kda_moe_layer.py`), e2e us per kernel with 512 MB flushed before each
  launch. **Paired in-server A/B**: main library set on every layer, the variant on the layers of one parity, parity swapped
  between runs; reported as "variant - main" per kernel, two rounds (from 10-26 on, the verdict for every change).
- Numbers: "a / b" = two rounds; "x vs y" = variant vs base.

## 1. Result

Forward span ratio (SGLang alone / ours, same session), selected milestones:

| job (set) | bs 1 | bs 2 | bs 4 | bs 8 | bs 16 | bs 32 | bs 64 | what changed |
|---|---|---|---|---|---|---|---|---|
| srv4 (10-22, ms/token) | 1.30 | 1.42 | 1.66 | 1.56 | - | - | - | per-size libraries, no fusion |
| srv12 | 1.48 | | | 1.61 | 1.34 | 0.90 | 0.75 | fused at every size |
| srv21 | 1.81 | 1.78 | 1.71 | 1.69 | 1.39 | 1.06 | 0.93 | first all-7-size set |
| srv28 | 1.95 | 1.93 | 1.78 | 1.73 | 1.39 | 1.10 | 1.01 | expert interleave at T <= 4 |
| srv30 | 1.96 | 1.94 | 1.79 | 1.80 | 1.50 | 1.14 | 1.06 | RS attn_residual at T >= 32 |
| srvd42 (dlib36) | 1.964 | 1.947 | 1.880 | 1.866 | 1.478 | 1.168 | 1.088 | FLAT recurrence T 32 / 64 |
| srvd52 (dlib47) | 2.039 | 1.992 | 1.913 | 1.908 | 1.584 | 1.199 | 1.132 | resets only on the experts' CUs |
| srvd54 (dlib49) | 1.987 | 1.975 | 1.918 | 1.907 | 1.581 | 1.272 | 1.169 | AR / SRN poll bursts T >= 32 |
| srvd64 (dlib53) | 1.974 | 1.951 | 1.923 | 1.940 | 1.648 | 1.270 | 1.178 | SGLang's AITER expert layout |
| srvd136 (dlib74) | 2.023 | 1.983 | 1.950 | 1.957 | 1.668 | 1.321 | 1.206 | RB + COAL + SLOAD |
| srvd148 (dlib83) | 2.025 | 1.987 | 1.947 | 1.973 | 1.703 | 1.315 | 1.236 | + Q1, HOLD, MLA FRS, HQ_FIRST |
| srvd164 (dlib102) | 2.042 | 1.981 | 1.946 | 1.963 | 1.712 | 1.368 | 1.306 | + no W2 steals |
| srvd182 (dlib115) | 2.020 | 1.986 | 1.943 | 1.974 | 1.701 | 1.363 | 1.251 | + coop z_q, h_q die-local (current) |

- bs 1..8 pass 1.88x since srvd31..srvd53 (bs 1 / 2 from srv28). bs 16 / 32 / 64 stay at ~1.70 / ~1.36 / ~1.25-1.31.
- Two sessions of the same libraries differ by 1-2 % at bs <= 32 and up to 4 % at bs 64 (srvd164 1.306 vs srvd168 1.255).
- Full GSM8K (1319 questions): ours and SGLang alone within ~1 point either way in every run (srvd183: 0.960 vs 0.956 at
  parallel 64; srvd184: 0.949 vs 0.955 at parallel 32); GSM8K 64 = 1.000 both in every status run.
- At bs 64 our kernels went from ~600 us (first fused T 64) to ~300 us per fused KDA kernel in the server.

## 2. Serving and infrastructure

- **Per-size libraries**: kinds `kdad`, `mla`, `moe`, fused `moeaggkdad` / `moeaggmla` for T = 1, 2, 4, 8, 16, 32, 64; the hook
  (`sglang_k3.py`) picks the library from the batch size; SGLang's decode graphs only at those sizes (`CGBS`), prefill chunk 4096
  (a 16384-token prefill ran out of memory).
- **Fused everywhere** (srv12 on): the separate MoE kernel + SGLang's aggregation kernel + attention kernel per layer became one
  kernel per layer pair at every T.
- **`moeagg` for the last MoE layer** (`MPK_AMD_DEC_MAGG`): includes the model's final aggregation; bs 8 -7.5 us per forward.
- **SGLang's AITER expert layout** (`MPK_MOE_AITER=1`, srvd64): our kernels read SGLang's AITER-shuffled experts, so SGLang
  keeps its weights for prefill (before, prefill ran through our decode MoE in 64-row chunks and end-to-end GSM8K was ~2x slower).
- **Server measurement**: same-session runs from srvd40 on (SGLang alone first, every size warmed); the paired in-server A/B
  from srvd113 on (`MPK_AMD_DEC_F*_ALT`, `MPK_ALT_PARITY`, `amd/tools/trace/kdur_alt_an.py`); server stamps
  (`STATIC_NODE_END_STAMPS`, `STATIC_CSR_PHASE_STAMPS`, `MPK_DUMP_STAMPS` + `MPK_LAYER_STAMPS`).
- **Harness**: decode mode per T, synthetic routing calibrated to the server's expert counts (`MPK_TOKEN_SHARE`: 0.25 at T 16
  for ~190 experts), real routing from saved server inputs (`MPK_REAL_ROUTE`), GPUs lined up (`STATIC_LINEUP`), all-GPU expert
  traces (`MPK_TRACE_ALL=1`).

## 3. Correctness fixes needed for the new sizes

- In-kernel reset buffers must be whole 16-B pieces (T 1 attn_residual partials of 4760 B made the kernel exit).
- `tokens_plan` (the T 8 plan re-dealt for another T) treated a GEMM K split equal to 8 as a token axis (T 1 MoE 108 -> 78 us).
- MLA decode hook returned the wrong output buffer (garbage from token 1).
- `gemm_tile` activation in chunks when `[T][K part]` does not fit LDS (T 64).
- Inline-asm store of MFMA accumulators 1 wait state after the MFMA: wrong KDA / MoE values at T 32 / 64 (GSM8K 0.078);
  `st16_dev_acc` with 24 wait states. A counted GEMM read arena copy 0 while its producer wrote the launch's copy.
- top-k after the router at T >= 32; `STATIC_ARG_RELOAD` off at T 16 (router wrong with it); `STATIC_MAX_RESETS` 32 for
  builds with more 0xFF outputs; the hybrid projection's LDS limit forgot the 1 KB static LDS; the two-pair kernel's slot mask
  was one 64-bit word for ~96 slots.

## 4. MoE front (router, top-k, latent_down / z_q, shared expert, expert table)

Adopted:
- `STATIC_W13CU_EARLY` (T 1-4): the table and the first W13 weights before z_q lands (KDA T 4 -2..-4 us).
- Routing table of T 8 used at T 2 / 4 (the general table took 4.0 us at T 16).
- Shared down unpacked at T 1, 2 per CU at fused KDA T 8 and fused MLA T 8 (`MPK_MOE_PACK=2`, -2..-3 us).
- `MPK_FUSED_APART=30:26,28` at fused MLA T 16 (shared gate_up off the router / latent_down CUs, -1.3 us; loses at T 32 / 64).
- Cooperative z_q at KDA T 64 (`MPK_ZQ_FINISH=2 MPK_LDOWN_GRID=28,8,1 -DSTATIC_GEMM_QUANT_COOP=2`): latent_down's K-part tasks
  add their parts, quantize and send z_q themselves (no sum_quant_send step). Paired srvd170: KDA -2.00 / -2.15.
- z_q in 4 token groups at KDA T 32 (`MPK_MOE_REGRID=...;29:28,4,1:28`): paired srvd161 -1.52 / -1.68.
- `STATIC_CSR_PAIRS_PAR` at T 32 (the table's pair polls in flight together, harness -6; no server gain at T 64).

Measured state: the first W13 unit waits for both the expert table (~7 us after top-k, ~29-33 us after the kernel start at
T 64) and z_q (~32 us); the front reads only ~52 MB (~8 us of bandwidth).

## 5. MoE experts (W13 + W2)

Adopted:
- **CSR expert task for T > 16** (`moe_experts_csr.cuh`): a routing table in CSR form, h_q by (token, expert) pair, one CU per
  slot x 4 items, z staged per K quarter, each expert's weights read once (shared experts at T 64: 594 -> 304 us).
- `STATIC_CSR_W13_SMALL` (T 32 / 64): a slot's z_q staged for all K once, W13 weights streamed 7 steps ahead (fused KDA T 64
  -15 us harness).
- `STATIC_CSR_ZDMA` + `STATIC_SQS_SCALES_LAST` (T >= 32): z_q staging by LDS DMA (KDA T 64 -4.6, srvd47 bs 32 -535 us span).
- Expert interleave at small T: `STATIC_MOE_ILV=2` at T <= 4 (W13 units and W2 items of groups of 2 experts interleaved: W2 had
  waited ~16 us for h_q; KDA T 1 -6.5) and `STATIC_MOE_ILVW` per wave at MLA T 4 (E 4), T 8 (E 8), T 16 (E 12).
- `STATIC_W2_ONE` (T <= 2): one W2 item per wave. `STATIC_W13CU_PF` / per-wave W13 / W13CU off at T 4 depending on kind.
- `STATIC_MOE_W2_HQ_FIRST` (T 16) and `STATIC_CSR_W2_HQ_FIRST` (T 32 / 64): an item's h_q loads before its weights, so the item's
  MFMA operands do not wait for the weights in issue order. T 16 server -3.8 / -4.7 (6 runs), T 32 -3.8 / -2.5, T 64 paired
  srvd142 -2.05 / -2.17 KDA, -1.99 / -1.91 MLA.
- `STATIC_FAST_TABLE` (T 16; fewer SGPR spills): server KDA -0.8, MLA -1.25.
- **`STATIC_CSR_W2_NOSTEAL`** (T 32 / 64; the largest single gain): no W2 steals from other dies. Traces of every GPU showed the
  last ~5 experts of each die's list at 12-25 us per item instead of ~7 while every finished wave probed the 8 dies' counter
  lines; identical work then ended up to 24 us apart between GPUs. Paired srvd163: bs 32 KDA -6.50 / -6.44, MLA -6.21 /
  -7.62; bs 64 KDA -7.53 / -10.00, MLA -7.59 / -8.22 us per kernel.
- **`STATIC_CSR_HQ_LOCAL`** (T 64, needs NOSTEAL): for experts whose W13 units and W2 items are all in one die's shares, h_q is
  written with write-back stores and read by W2 from that XCD's L2 (bench `xcd_poll.hip`: an sc1 load sees a same-XCD plain store
  in 0.28 us vs 0.40 for an sc1 store). Paired srvd181: KDA -0.43 / -0.84, MLA -1.66 / -2.20.

Measured state at T 64 (server, srvd174): W13 ~6.3 TB/s, W2 ~5.5 TB/s; the slowest GPU ends the expert phase ~10 us after the
median GPU per layer (median over layers; ~2x the spread of a plain read, `gpu_var.hip`). SGLang's W13 + W2 at bs 64: 139 us.

## 6. MoE tail (R send, sum_rmsnorm, latent_up, moe_agg1)

Adopted:
- `STATIC_RBLK_PAR` (every T): R send polls its R blocks' counters with a lane per block, all in flight (KDA T 64 -5 harness).
- **Tail not folded at T >= 32** (`MPK_MOE_NFOLD`: R summed once per token by sum_rmsnorm instead of RMSNorm folded into every
  latent_up K part, S sent early, sum_gpus over the experts' CUs; `STATIC_MAX_RESETS=32`): KDA T 64 -38, MLA T 64 -28 us harness.
  Folded stays at T <= 16 (unfolded lost there).
- **sum_rmsnorm one task per token** (`MPK_MOE_REGRID=38:1,T,1:36,37`, T 32 / 64 KDA, T 64 MLA): paired srvd158 bs 64 KDA -3.94 /
  -3.79; MLA T 64 srvd160 -1.53 / -0.95.
- Poll bursts: `STATIC_GEMM_NORM_POLL2` (T <= 16: the folded-norm fill re-loaded all its polls in one burst; it had checked each
  load inside its own branch, 16 serial round trips; MLA T 16 -5.8) and `STATIC_AR_POLL2` + `STATIC_SRN_POLL2` (T >= 32: KDA T 64
  -12.9, srvd54 bs 32 -770 / bs 64 -1079 us span).
- `MPK_NOINLINE_TASKS` (T 32: sum_quant_send, sum_gpus, sum_rmsnorm; T 64: sum_rmsnorm, sum_gpus as separate functions):
  removes their spills (KDA T 32 -5.9, srvd58 bs 32 -825 / srvd59 bs 64 -810 us span).
- `MPK_LUP_APART_EXTRA=35` (latent_up off the S-send CUs; T 8 / 16, MLA T 2 / 4): srvd48 bs 16 -574 us span.
- `STATIC_AGG1_RB=8` (moe_agg1 loads 8 bank rows in flight instead of one row at a time): paired bs 64 -2.26 / -1.25 KDA.
- **`STATIC_MOE_R_RS=2`** (fused KDA T 64, 10-27): R reduce-scattered by token. The R send pushes token t's row only to GPU t / (T / 8)
  (57 KB per link instead of 458 KB); there sum_rmsnorm adds the 8 rows and RMSNorms, writes Rn and stores the Rn row into a free row of
  its own exchange buffer; the other GPUs' sum_rmsnorm tasks load it from there (remote sys-scope loads) and copy it to their Rn. Rn is
  bit-identical on every GPU. Paired srvd187: KDA T 64 -2.35 / -2.61; MLA T 64 and T 32 mixed (not used there). Mode 1 (the owner pushes
  Rn to the 7 GPUs) lost 15 us: 8 tasks' remote stores, ~0.6 us apart per wave, landed 10-16 us after the sums (dec468).
- `MPK_ATTN_OFF_AGG=1` (attention's first node off moe_agg1's CUs; KDA T <= 8, MLA T <= 16): KDA T 1 -10.6 us.

## 7. KDA attention part

Projection (q, k, v, g, f_a, beta GEMM):
- Grid 50 x 7 at T >= 32; hybrid 256-row tasks (`MPK_KDA_PROJ_HYB=4`, 25 x 7) at T 16 placed off the latent_up / moe_agg1 CUs
  with sum_gpus one per CU (`MPK_ATTN_OFF_SET=68:39+48`, `MPK_MOE_FOLLOW_LATE=38:37:35:39`): with CIR, T 16 -12.1 (srvd50 bs 16
  -834 us span).
- `STATIC_GEMM_ACT_PAD` at T 64 (activation rows padded 16 B: LDS bank conflicts, -8.5 harness). `STATIC_EARLY_L2_SECOND` at
  T 64 (the projection's second-round tasks pre-read into L2, -2.6 harness).

Conv + recurrence:
- **CIR** (conv folded into the recurrence, `MPK_KDA_CONV_IN_REC=1`) at T 8 / 16 with `MPK_KDA_SPLIT_PROJ=2` at T 8;
  `STATIC_DRD_CIR_PAR` (one round of polls for q / k / v, f_a, beta) and `STATIC_DRD_CIR_EARLY` + `G_EARLY` at T 8 / 16.
- **FLAT recurrence** (`MPK_KDA_FLAT=P`: tasks over (head, request) pairs; P 3 at T 64, 2 at T 32): the recurrence was bound by
  per-CU bandwidth (T 64 -15.0, T 32 -10.4 harness; srvd42). 4 requests per task at T 32 before FLAT (-6 harness).
- **`STATIC_DRD_COAL`** (T 32 / 64): a thread's 16 state columns are 4 kq + 32 c' instead of 16 consecutive, so one wave load /
  store covers 8 whole 128-B lines instead of 32 quarter lines (states 12.3 -> 7.4 us). Paired bs 64 -4.65 / -3.78.
- `STATIC_DRD_HOLD=72` (T 64): the last pair's 64 KB state kept in LDS; o_proj on the same CU stores it after its own loads
  (paired -1.89 / -0.20). Count posted before the last state write-back (`STATIC_DRD_POST_EARLY`).
- `STATIC_DRD_FLAT_LATE` at T 32 (all state stores after the count, -5.6 harness at the time).

o_proj and attn_residual:
- **attn_residual as a reduce-scatter at T >= 32** (`MPK_AR_RS=1`: o_proj sends each row block only to its owning GPU, every GPU
  finishes its columns, then all-gather): KDA T 64 -12, MLA T 64 -24 harness. Loses at T <= 16.
- `STATIC_AR_U_EARLY=2` (KDA T 32 / 64, MLA T 32, 10-27): the RS task pushes its u columns right after the after-u dots instead of after
  the weights, so the last step has only the out pushes (KDA T 64 weights -> end 7.1 -> 5.3 us; the dots' hop +0.9, its polls wait for
  the u stores' acks). Paired srvd188: bs 32 KDA -0.48 / -0.78, MLA -0.82 / -0.77; bs 64 KDA -2.30 / -1.53 (MLA T 64 mixed).
- o_proj in 3 K parts at T 64 (`MPK_OPROJ_FRS=1`, 56 x 3, finish by the last part; -11.6 harness) with
  `STATIC_GEMM_COOP_FINISH` (every K part shares its row block's finish, T >= 16).

## 8. MLA attention part

- `STATIC_MLA_KV_DMA` (KV chunks by LDS DMA; T <= 32, then T 64): T 1 -8, T 64 -14 harness.
- Combine loads every split's (m, l) at once (T 1 -2.6); `MPK_MLA_COMB_SPREAD=1` (every split combines its share; T 1 / 2 / 8
  / 16 / 32 / 64, -1..-4).
- KV splits per T: 16 at T <= 4, 8 at T 8..32, 2 at T 64 (4 at T 64 lost in the server, srvd158).
- `STATIC_GEMM_ACT_PAD` + `STATIC_GEMM_ACT_DMA_PAD` at T 64 (padded activation rows filled by LDS DMA, -3.5).
- o_proj in K parts: 112 x 2 + coop finish at T 64 (-16 harness); `MPK_OPROJ_FRS=1` 112 x 2 at T 32 (paired srvd140 -2.02 /
  -2.08); counted o_proj (`MPK_MLA_OPROJ_POLLED=0`) at T >= 16.
- vgate in token groups at T 16 (12 x 4, -2.2); attn_residual in 14 chunks at MLA T 4 (`MPK_AR_CHUNKS=14`, -1.5).

## 9. Code generator and runtime (every task)

- **`STATIC_TASK_SLOAD`** (every T): the task list entry read by one scalar 16-B load (`s_load_dwordx4`, waits on lgkmcnt) instead
  of 3 vector loads each waited with `vmcnt(0)` - which had made every task start wait for all stores of the task before it.
  Paired srvd132, all 12 halves negative: bs 16 -1.0..-1.4, bs 32 -1.5..-4, bs 64 -5.3 / -5.1 (KDA).
- **`STATIC_RESET_FIRST_NODE=36`** (every T): only the CUs whose list starts with the experts do the per-launch resets, after the
  start, not waited for (kernel boundary cost was ~4-6.5 us): srvd52 bs 1 2.039x, bs 16 -386 us span.
- `STATIC_TID_ASM=3` (KDA T 1-32, MLA T 1-16): the thread index through a volatile `v_mov`, so per-lane values are not hoisted
  out of the task loop and spilled (harness -2..-4; not visible in the server).

## 10. Task split and placement (plans)

Every per-size library goes through graph -> plan (grids = task split, per-CU lists = assignment) -> schedule -> generated kernel.
Each T's first plan came from the T 8 plan by `tokens_plan`; then per-T changes with the plan passes (`--regrid`, apart, follow,
off, pack; `MPK_MOE_REGRID`, `MPK_ATTN_OFF*`, `MPK_LUP_APART_*`, `MPK_FUSED_APART`, `MPK_ATTN_CAP`, `MPK_LATE_SET`). Split /
placement changes moved a T 64 kernel by ~10-25 us either way:
- helped: o_proj in K parts (-11.6), sum_rmsnorm per token (-3.9), MLA T 32 o_proj 112 x 2 (-2), z_q token groups at T 32
  (-1.6), FLAT pairs per recurrence task (-15), RS attn_residual (-12..-24), HYB4 placement at T 16 (-12 with CIR).
- hurt: shared gate_up 24 x 4 (+25), finer projection grids (+16), projection tasks off the agg1 / R-send CUs (+11..14), finer
  o_proj grids (+10), R send on 256 tasks (+8), the shared expert after the experts (+40..70).
Around today's plan, every variant tried ties or loses.

## 11. Tried and not kept (do not repeat without a new reason)

| change | where | result |
|---|---|---|
| CSR experts at T <= 16 | MoE | slower at every T (T 16 152 vs 127) |
| `STATIC_TOPK_BALLOT` (top-16 by bit search) | top-k | +5 us T 1 |
| `MPK_MOE_LOCAL_ROUTE`, `W2_WARM`, `W13_EARLY8`, `W13_PAIR`, `ILVW_BAL`, `ILVW_STEAL`, `ILVW_TABLE_FIRST` (breaks resets), `W13_EARLY` | experts | ties / losses |
| `STATIC_CSR_W2_STATIC` (fixed W2 lists) | experts | +24 us: waves reach W2 at different times |
| `STATIC_CSR_W2_AHEAD`, `TAKE_ASYNC`, `W2_PACK`, `W2_PLAIN`, `W2_LATE_STORE`, `HEAVY_FIRST`, `ZRND`, `ILV` at T 64, BSTORE switches | experts | ties / losses (AHEAD +5 at bs 32) |
| R rows in bf16 (`STATIC_CSR_R_BF16`) | experts / R send | +0.7..+1.5 |
| `STATIC_MOE_R_RS=1` (owner pushes Rn to every GPU); R_RS=2 at T 32 / MLA T 64 | tail | +15 (T 64 harness); mixed in the server |
| `STATIC_AR_U_EARLY=1` (u pushed before the dots' push) | attn_residual | dots' hop +1.1: tie |
| `STATIC_AR_DOTS_SPREAD` (dots' remote stores one per lane), `STATIC_AR_DOTS_W3` (wave 3 alone polls) | attn_residual | dots' hop unchanged (~5-6 us: the slowest GPU + one hop) |
| `STATIC_CSR_PAIRS_PAR` at T 64, router / latent_down / shared gate_up grids, shared gate_up placements (`MPK_LATE_SET`, on the router's CUs, fewer tasks) | front | ties / losses |
| sum_quant_send in token groups at T 64, `SQS_DMA`, `SQS_ROUND` | front | losses |
| latent_up in 14 K parts, latent_up computed whole on every GPU (not built: ~23 us of reads per CU) | tail | ties |
| `STATIC_POLL_SLEEP=16`, `COOP_LAST_NOWAIT`, `GEMM_FINISH_FLAG`, `NORM_DEV`, `NORM_SAMPLE`, `NORM_QR` | hand-offs | ties |
| KPAIR + `ACT_KEEP`, `ACT_L2_BUFS`, `ACT_DMA_PAD` for KDA, `ACT_SWZ`, `L2WARM_POLLED`, `GEMM_STAGES` 12..32, HYB at T 32 / 64 | projection | ties / losses |
| `DRD_FLAT_PRE`, `ST_NT` / `LD_NT`, `STATE_LATE`, `POLL_FIRST`, `L2NEXT`, FLAT P=4 at T 64, 4 pairs on 192 CUs, CIR at T 64 (fails the check) | recurrence | ties / losses (P=4 +5..6, 192 CUs +26) |
| o_proj grids 112 / 224 rows, off the recurrence CUs; RS at T <= 16; attn_residual on one CU per token | o_proj / residual | losses |
| MLA KV splits 4 at T 64, `MLA_COMB_K`, `COMB_LAST`, `COMB_FLAG`, `NORM_PAR`, norm over 2 / 4 tasks, q_b counted, vgate token groups at T 8 / 32 / 64 | MLA | ties / losses |
| two layer pairs in one kernel (T 8: -3..-4 harness, server slower; T 64: 57 VGPR + 478 SGPR spills) | kernel | dropped |
| MALL / L2 pre-stage of the next layer's weights (MALL 4.67 vs HBM 6.0 TB/s) | kernel | not built |

## 12. Measurement lessons

- The paired in-server A/B is the verdict (precision ~+-1.5 us per kernel); harness ties became server wins (Q1 -3.9,
  HQ_FIRST -2) and harness wins tied in the server (TID_ASM, cooperative z_q at T 32).
- A fixed-order harness A/B favours the first variant by ~2-4 us: rotate the order every round.
- A job helper that set `A=` without `local` made 13 A/Bs (srvd135-155) compare main against main; all were re-run (srvd158).
- Count remote store instructions per wave on a critical hand-off: they issue ~0.6 us apart, and a poll after them waits for their acks
  (vmcnt in issue order). Few-wave pushes are slow; readers pulling from the producer's exchange buffer spread the transfer.
- Server expert counts are per step: means 180 / 270 / 370 at bs 16 / 32 / 64 (the last decode step has more).
- When identical work ends at different times on different GPUs, trace every GPU: that is how the steal hot lines were found.

## 13. Where the time goes now, and the floor

T 64 fused KDA kernel in the server (srvd173 / srvd174, dlib109, max over ranks, us after the kernel start): first W13 unit
~33 (front), experts end 208, R send 220, sum_rmsnorm 232, latent_up 241, moe_agg1 255, projection 281, recurrence 302, o_proj
317, end 331. The tail after the experts reads ~35 us of bytes but takes ~124 us (dependent steps, 3-5 GPU hops, steps on few
CUs at ~30 GB/s per CU). bs 16 KDA kernel: experts end 94, end 151 (tail ~58 us).

Floor estimate per KDA layer (bytes at the measured 6.46 TB/s plus ~40 us of dependent steps): bs 16 ~135 us vs the 1.88x
target 132; bs 32 ~168 vs 155; bs 64 ~207 vs 194 (bs 64 reads ~1.08 GB per GPU per layer). SGLang's own bs 64 KDA layer is 26
kernels back to back (364.5 us) with no gaps, so fusion alone has nothing left to remove; the target needs the whole kernel at
~85 % of the read rate.

Open items (each bound at ~1-2 %): R send overlapped with W2 (needs block-ordered W2 items, and the sends must not sit on the W2
waves: remote stores block their later polls), attn_residual's dots polls not behind the u stores, bf16 KDA state (a numerics change vs SGLang, not decided),
the intermittent T 32 cooperative z_q check failure (not understood).
