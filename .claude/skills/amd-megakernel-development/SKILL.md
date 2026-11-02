---
name: amd-megakernel-development
description: How a model part (K3 MoE, KDA, MLA, ...) becomes one static megakernel on AMD MI355X in mirage_amd / mirage_amd_bs - the graph / plan / schedule / generated-kernel pipeline, writing a HIP task (dynamic expert queues, no-steal, die-local hand-offs), the cross-CU and cross-GPU hand-off and reset protocols, the one-layer and one-kernel-per-layer-pair harnesses and their checks, per-rank server libraries and the SGLang hooks, plan tuning and plan options, the flag policy and the end-of-campaign cleanup, the like-for-like timing against B300, the A/B against SGLang ROCm, and the batch-decode (bs 1..64) state, library recipes and time budget. Use when starting or porting a part to AMD, adding or changing a HIP task type, changing the AMD code generator or harness, and before reporting an AMD megakernel result. (NVIDIA/B300 work uses mirage_kda's own skills: megakernel-part-port, server-perf-testing.)
---

# Megakernel development on AMD (MI355X)

The K3 MoE layer went this way on MI355X (10-09..13): port of the B300 kernel's design, not its code, to HIP; one-layer
harness with random weights in the real shapes; plan and protocol tuning until TP8 e2e matched B300 within 10% like for
like (83.3-84.6 us); then an A/B against SGLang's own ROCm layer on the same weights (142-145 us). KDA and MLA followed the
same way (10-13..15), then per-rank libraries in SGLang ROCm, then one kernel per MoE layer + the next layer's attention
part (10-15..16): full K3 target forward 11081 vs SGLang alone 20873 us, GSM8K 0.950 vs 0.947 (10-16 final, B300 span
9035). Batch decode bs 1..64 followed (10-22..27, `~/megakernel/mirage_amd_bs`, section 8): one library per (kind, T) for
T = 1, 2, 4, 8, 16, 32, 64, forward span 2.02x / 1.99x / 1.94x / 1.97x / 1.70x / 1.36x / 1.25x vs SGLang alone (srvd182),
full GSM8K 0.960 vs 0.956. Read `amd-kernel-development` first for the hardware, toolchain and box rules this skill assumes.

## 0. Rules

- **B300 is the design reference, not code to copy.** `~/megakernel/mirage_kda` (tasks under
  `include/mirage/static_megakernel/tasks/`, specs `docs/kda_port_spec.md`, `docs/mla_port_spec.md`, results
  `docs/k3_results_2026-10-11.md`, lessons `docs/kernel_optimization_notes.md`). Read the B300 task and spec for the math,
  the serving contract and what was measured; write the HIP task for gfx950's machine (no TMA / tcgen05 / TMEM / clusters /
  mbarriers: LDS DMA, MFMA, buffer loads, sc1 hand-offs).
- **New HIP files per part; shared tasks stay unchanged.** A part gets its own tasks in
  `include/mirage/static_megakernel_hip/tasks/<part>_*.cuh`, its graph builder, its harness. If a shared task (gemm_tile,
  core.cuh, runtime.cuh, config.cuh, moe_*) needs a variant, add a flag that defaults off and prove the default code is
  identical (`.s` md5, see `amd-kernel-development` section 4), or write a new task.
- **Experiments behind `-D` flags that default off.** A loser is recorded and removed or left off; a winner becomes the
  command line, later the default. Every flag's off state is code-identical to before it existed.
- **End of a campaign: clean the flags** (10-16): remove every flag no final library sets, with its code; bake in every flag
  all libraries that use the code set the same way; keep as flags only those the libraries set differently and the
  measurement switches. Prove it as `amd-kernel-development` section 4 says (stripped `lib.s` of every final library, then
  bit-identical harness outputs against the old build's, then the server run). Record what was removed and why in the
  notes: the code is gone, the numbers stay.
- **Correct before fast.** No timing of a build whose checks fail (the harness skips e2e when a check fails).

## 1. Pipeline (python/mirage/mpk, MK_BACKEND=hip)

```
graph    StaticMegakernel(num_gpus=8); demo build_graph(mpk, weights)   layer calls: gemm_tile_layer, topk_route_layer,
                                                                         moe_experts_layer, allreduce_send_layer, ...
plan     compiler.plan_from_file(plan.json) -> {grids, lists per GPU}     amd/tools/plan_one_gpu.py: fewer_gpus_plan,
                                                                         pack_plan, regrid_plan, front_plan, phase_plan ...
schedule compiler.compile_plan(mpk, plan, out/schedules)                 schedule_gpu<g>.json: per CU its task list
kernel   mpk.compile_static(paths, gpu_tensors=ws, extra_flags=[-D...])  static_schedule.py generates build/layer.cu
                                                                         (one switch over node ids per list entry) + host
run      static_harness.Harness(sk): launch(flush_bytes), e2e_us, span_us, task_times (timing build)
rank lib amd/tools/gen_rank_lib.py --layer-cu out/layer.cu --kind K --out DIR --flag ...   layer_rank.cu = the kernel and
                                                                         task tables + rank_host.cuh (one process per GPU,
                                                                         HIP IPC exchange, capturable in a HIP graph)
check    torchrun amd/tests/test_rank_lib.py --kind K --lib DIR/lib.so --ref harness --save-outs: bit-identical + e2e
server   amd/serving/sglang_k3.py (hooks patched in by sglang_patch.py): MPK_K3_<KIND>=run MPK_AMD_LIB_<KIND>=lib.so
```
- Backend switch: `gpu_backend.backend()` ("hip"): templates `include/mirage/static_megakernel_hip`, wave 64, LDS 160 KB,
  `s_memrealtime` 10 ns per tick, no clusters.
- A layer call declares its outputs (`_scratch(name, bytes, reset, by_reader)`: reset value 0 / 0xFF before every launch (by the kernel itself, section 3),
  `exchange` buffers in the cross-GPU region with two sets alternating by launch parity), its maps (`_map`: weight views a
  task indexes), its counters (128-B lines per node) and its params (template args).
- Generated per entry: `static_mk::run_<name><StaticNode<grid,counter,buf>, StaticParams<...>, StaticBufSlots<...>,
  StaticMapSlots<...>, inputs' StaticNode...>(maps, g, L, tk)`; LDS size from `smem_<name><...>()`.
- One kernel per layer pair: `amd/tests/test_kda_moe_layer.py --order moe [--attn mla] --agg NB,WRITE,2` builds MoE layer L
  + layer L+1's aggregation 1 (`moe_agg1_layer(posted=True)`, one task per token) + layer L+1's attention part on its
  outputs in one graph; `moe_first_plan` puts each CU's MoE tasks first, then its attention tasks (so a CU starts the
  projection's weight loads when its MoE tasks are done); `MPK_SPLIT_LOOP` = the first attention node (two task loops:
  the parts are register-allocated apart). Server kinds `moeaggkda` / `moeaggmla`; `moeaggkdamix` (built with
  `MPK_AGG_MIX=1`) also writes the pre-norm mixture into SGLang's dspark capture buffer at the capture layers.

## 2. Writing a HIP task

- One workgroup of 256 threads (4 waves) per CU, one task at a time per CU, all 160 KB LDS (`L.sm`, 1024-aligned).
  `__syncthreads()` at the task start (the previous task may still read LDS).
- Inputs from other tasks: wait on the producer's counter (`cnt_wait_in` / sc1 polls) or poll the data itself when the
  buffer is 0xFF-prefilled (data as flag). Outputs other CUs read: sc1 stores (`st16_dev` / `st16_out`), then the counter
  (`cnt_post`, bypass protocol). Outputs another GPU reads: `st16_sys` / `push16` into the peer's exchange slot.
- Put independent loads first: issue the weight stream before waiting for the activation (the GEMM's `fill_act` after the
  first stages are in flight), so the two round trips overlap.
- Dynamic work (MoE experts): one counter per die (its own 128-B line) hands out the die's W13 units, then its W2 items;
  the next ticket is issued inside the current unit (after its last weight loads), so its round trip hides behind the
  unit; W13 units from the own die only. W2 items from another die (stealing) are **off in the batch libraries**
  (`STATIC_CSR_W2_NOSTEAL`, moe_experts_csr.cuh): every finished wave probing the 8 counter lines slowed the last items of
  every die 2-3x (`amd-kernel-development` section 2, hot lines). A fixed W2 assignment (no tickets, dec420) lost: waves
  reach W2 at different times. With no stealing every reader of an expert's h_q is on the producer's die, so
  `STATIC_CSR_HQ_LOCAL` writes it with write-back stores (only for experts whose units and items are all in the die's
  shares; boundary experts and the 12-B-per-pair scales stay sc1).
- Work order matters for whoever waits on it: W2 items are handed out expert by expert, so every 128-column R block is
  complete only at the end of W2 and R send cannot start earlier; the expert table (`expert_table_csr`) waits for top-k and
  z_q waits for latent_down: the first W13 unit starts at ~29-33 us at T 64 behind both.
- Timing build hooks (`STATIC_TIMING_BUILD`): stage stamps (`STAGE_STAMP`), per-item trace in a scratch buffer. They must
  not exist in the normal build.

## 3. Hand-offs, start and resets (core.cuh)

- Cross-CU on one GPU: sc1 data + `wait_mem()` + relaxed counter add (`st16_out`, `cnt_post`); readers `cnt_wait_in` /
  sc1 polls. Never a release/acquire with `buffer_wbl2`/`buffer_inv` per CU (whole-L2 operations).
- Cross-GPU: data as flag in the exchange region (two sets by launch parity, so the next launch never overwrites a set a
  slow GPU still reads); one hop 2.3-2.6 us with the slots' loads in flight together.
- Per-launch state is reset **in the kernel, always** (no flag since 10-16; counted in e2e, as B300's resets before
  `griddepcontrol.wait`): local buffers (counters, scratch, 0xFF data-as-flag outputs) live in an arena allocated twice;
  launch p uses copy p (`buf_at`, `cnt_of`) and, at its end, the CUs without a task of the graph's last node reset copy
  1 - p (`kernel_end`); the exchange buffers' next set is reset at `kernel_begin`. The launch number is each CU's own count
  (`g.arrive` lines 65..72): no barrier at the start, nothing for the host per launch, so a HIP graph replays it. The reset
  list lives in `G` (`STATIC_MAX_RESETS`, default 24: a one-kernel layer pair with extra 0xFF outputs needs 28-32; the host
  exits with "in-kernel resets: N of at most M"). A run's outputs bound to the caller's tensors are the arena copy of that
  launch, copied out after it (`static_host_par_mirror`).
- Start (harness, like for like with B300): the layer is launched early on a second stream (`STATIC_EARLY`), its CU 0 posts
  "resident" (`STATIC_EARLY_RES`, `STATIC_EARLY_PREWAIT`), a one-wave line-up kernel per GPU waits for every GPU's post,
  waits `STATIC_EARLY_PREV_US` (5 = B300's stand-in), then writes the go. `rank_host.cuh` refuses these flags (the server
  has no line-up). The WG that lands on the line-up kernel's CU starts late (~4-7 us): give it work that can be late.
- Own data may be loaded before the go (B300 PDL rule): kernel arguments (always), the first task entry
  (`STATIC_TASK_PREFETCH`). Previous-kernel outputs only after the go.

## 4. The one-layer harness and its checks

- Template: `amd/tests/test_layer_one_gpu.py` (MoE): random weights in the real shapes per GPU (`random_weights`, seeds
  `seed + g`, common tensors GPU 0's), expert banks as random bytes in the gfx950 tile layout, every step checked against a
  torch reference built from the kernel's own inputs to that step (a wrong step shows itself, not its consumers), outputs
  bit-identical on all GPUs, launch 1 and the last timed launch bit-identical to launch 0 (resets across launches).
- A new part gets the same: its own `amd/tests/test_<part>_layer.py` with step-by-step references and these checks, plus
  `--dump-y` style output for A/B with SGLang.
- The final libraries' builds (10-16, `~/mpk-amd/logs/job_fin1.sh`, all with `--gemm-shuffled`; harness check = the same
  flags + `--save-outs`, then `test_rank_lib.py --ref`):

  | kind | harness | plan options | flags |
  |---|---|---|---|
  | kda | test_kda_layer.py | `--grids '{"27":[100,7,1],"30":[112,1,1]}' --light 28:27 --apart 31,30` | `GEMM_LDSW=3 GEMM_STAGES=16` |
  | mla | test_mla_layer.py | `--grids '{"29":[34,7,1],"31":[144,1,1],"32":[24,7,1],"37":[112,1,1]}' --group 2 --kv-splits 24 --die-group 35 --rank 35,29,34` | `GEMM_LDSW=3 AR_POST_NOWAIT=1 ARG_RELOAD GEMM_STAGES=16` |
  | moe | test_layer_one_gpu.py | `--fold --pack 34:4 --regrid 37:64,1,1:36 --apart 39,37,38 --regrid 28:56,8,1 --regrid 40:64,1,1:36,37,38,39` | MF = `TASK_PREFETCH ARG_RELOAD GEMM_LANE_ASM=1` |
  | moeagg | test_layer_one_gpu.py | moe's without the last regrid + `--agg 3,0,2 --regrid 40:8,1,1:36,37,38,39` | MF |
  | moeaggkda (mix: + `MPK_AGG_MIX=1`, NODES 75) | test_kda_moe_layer.py, `MPK_KDA_PROJ_HYB=1` | `--order moe --today --grids '{"27":[50,7,1],"30":[112,1,1]}' --agg 3,0,2` | MF + `GEMM_ACT_PAD=1 GEMM_LDSW=3 GEMM_STAGES=16 GEMM_L2WARM=256 GEMM_L2WARM_NODES=74` |
  | moeaggmla | test_kda_moe_layer.py | `--order moe --attn mla --today` + mla's grids and options + `--agg 3,0,2` | MF + `GEMM_ACT_PAD=1` + mla's flags + `MAX_RESETS=28` |

  (flags as `--flag=-DSTATIC_...`; e2e harness runs add `--flush-mb 1024 --e2e-reps 40`.)
- `--solo`: GPU 0's share of the TP8 layer alone (no communication), for per-GPU work questions only.
- A rank library must be bit-identical to its harness build in 8 processes, eager and in a HIP graph, before any server
  run; the server step of a job runs only if every library's checks passed (`job_fin1.sh` srv gate). A check of the old
  build's outputs: keep each `--save-outs` ref (`rank<NN>_<kind>/ref.pt`) and compare bits after any refactor.

## 5. Tuning loop

1. Write the layer's math budget first (`amd/notes/layer_math_mi355x.md` style): bytes per GPU, the critical chain, what
   each step should cost at ~24 GB/s per CU and ~6 TB/s per GPU, the hops.
2. Profile (`--profile`, dumps in `<out>/trace/`), align GPUs at their go (`amd/tools/trace/xgpu_an.py`), find the step
   whose measured time is furthest from its budget; for the start, `STATIC_PRE_STAMPS` in the normal build.
3. Name the measured step a change removes before writing it; one `-D` flag per idea; compile all 4 builds; code identity
   of the default.
4. Same-session A/B on the box, >= 3 rounds, decisive variant first; record wins and losers with numbers in the notes. For a
   library that runs in the server, the verdict is the paired in-server A/B (`amd-kernel-development` section 6): build the
   variant set and a control set from the same tree, prove the control's stripped `lib.s` equals the current best's, then
   harness checks of the variant (correctness) and the paired A/B in one job.
5. Plans: chain GEMMs need many CUs on MI355X (`--regrid NODE:GX,GY,GZ:AFTER`), packed small tasks (`--pack NODE:PER_CU`);
   the search (`--record` cost records, `--search`) needs a starting plan. Plan options added in the batch campaign
   (harness `test_kda_moe_layer.py`, fused node numbers): `MPK_ATTN_CAP=NODE:A1+A2:CAP` (at most CAP tasks of NODE on the
   CUs holding A1 / A2), `MPK_LATE_SET=N1,N2,..:LEAD:AVOID1+AVOID2` (nodes moved after LEAD on the CUs without AVOID, before
   their next other task), `MPK_TRACE_ALL=1` (every GPU's moe trace), `MPK_MOE_REGRID` entries separated by `;`.
6. Bound before building: the bytes the change saves at ~6.4 TB/s, or the stamped wait it removes, written down first;
   one change whose bound is under ~2 us per kernel is below the server A/B's precision.

## 6. Reporting

- e2e first (events around the kernel, slowest GPU, median, cold), span beside it labeled; conditions with every number
  (box, GPUs, flush, launches, build flags). Compare with B300 like for like (in-kernel resets, 5 us stand-in).
- Against SGLang: same weights, same inputs, same timing (`amd/bench/sglang_moe_layer.py` pattern: SGLang's own module,
  its weight processing, HIP graph); outputs compared with float64 references in each rounding order.
- Harness numbers are not server numbers. The full-model result is SGLang ROCm with our kernels vs SGLang alone, same session
  (B300 method: `mirage_kda/.claude/skills/server-perf-testing`): `srv_run.sh` both ways, `step_split.py` on the two TP-1
  traces (per-part times, target forward span, step period), full GSM8K (`GSM8K=1319`, ~10 min ours, ~16 min alone),
  `compare_answers.py` (greedy texts drift from token ~4-100 on both sides' numerics; GSM8K accuracy is the check). Check
  the server log that every kernel kind was used (`mpk_lines.txt`: "ran in one kernel", "aggregation 1 from the MoE
  kernel", "dspark capture from the MoE kernel").
- Results and decisions go into `amd/notes/` (append); final results get pinned with the exact build (B300:
  `docs/k3_results_2026-10-11.md` + `~/megakernel/pinned/`).

## 7. From the B300 compiler loop (10-14), for any backend

Found on B300 (`mirage_kda/docs/kernel_optimization_notes.md` section 19, skill `megakernel-part-port` section 8); the code
paths are shared, so check them here before trusting a search or a random-plan run.
- **Check the model before a search**: predicted vs measured span of a recorded run, per phase (`mirage_kda/scripts/
  b300_session/offline_loop.py predict`, CPU only). On B300 the KDA start plan read 42.2 predicted vs 29.4 measured, the MLA
  63.0 vs 42.6, because the records counted waits the solver models elsewhere. Two rules fixed it in mirage_kda's
  `cost_records.launch_durations` (not in mirage_amd's as of 10-14): a node with inputs from every GPU (stage0_inputs) is
  ready when its deps ended on every GPU (times after each start barrier); a concurrent group's members are ready when the
  latest member is. Without them, random plans (different lists per GPU) inflate those nodes' records 3-10x.
- **A searched plan is kept only if it measures faster**, alternating with the hand plan: the B300 MoE searched plan was
  predicted 1.1 us faster and measured 1.0 us slower.
- **Tasks that wait for each other while running must be declared `concurrent_along_axis`**, or a random plan puts two on
  one CU and hangs (B300 MLA KV splits).
- **Graph tensor names unique**: the harness host binds the caller's tensors by name (mirage_amd's `host_slot_args` already
  raises on a duplicate).
- **Cost keys**: the records and the searched graph must agree on params (eps), allocated shapes and the run's grids
  (`compiler.apply_grids` before recomputing records); run-time lengths are not in the key (one cost file per length).
- **Init order**: a rare first-launch-only hang on B300 was a rank still filling its reader-reset exchange buffers after the
  exchange barrier, erasing a peer's first push; a device sync + barrier after init fixed it. Any host that prefills
  exchange buffers needs the same ordering.

## 8. Batch decode, bs 1..64 (10-22..27): state, recipes, where the time goes

- Goal (binding 10-22): >= 1.88x vs SGLang alone at every bs 1..64, same session. Reached at bs 1..8 (1.94-2.02x); bs 16
  1.70x, bs 32 1.36x, bs 64 1.25x with the current best set **dlib115** (`/mpk-amd/dlib115` on the box; srvd182, srvd183).
- Server kinds per T: `moeaggkdad_t{T}` (MoE layer + next KDA layer's attention), `moeaggmla_t{T}`, `kdad`, `mla`, `moe`,
  `moeagg`; the hook loads `MPK_AMD_DEC_<KIND>="/mpk-amd/<set>/<kind>_t{T}/lib.so"` for `MPK_K3_TOKENS=1,2,4,8,16,32,64`
  (job template: `~/mpk-amd/logs/job_srvd182.sh`, paired A/B template `job_srvd181.sh`). Builds: `logs/dlib_build_x4.sh T
  fkda|fmla` with `TREE=`, `DLIB=`, `XMF` (flags for every part), `XFKFL` / `XFMFL` (KDA / MLA part flags), `XFKENV` /
  `XFMENV` (plan env), `XFKG` / `XFMG` (grid JSON; `XFMG` needs the line `[ -n "$XFMG" ] && FMG=$XFMG` patched into a copy).
- dlib115's fused T 32 / 64 recipes (all flags `--flag=-D...`; the T <= 16 libraries are dlib74's / dlib65's):

  | kind | tree | XMF | part flags | plan env / grids |
  |---|---|---|---|---|
  | KDA T 64 | mirage_bs_hql | `STATIC_CSR_W2_HQ_FIRST=1 STATIC_CSR_W2_NOSTEAL=1 STATIC_GEMM_QUANT_COOP=2 STATIC_CSR_HQ_LOCAL=1` | `STATIC_AGG1_RB=8 STATIC_TASK_SLOAD=1 STATIC_DRD_COAL=1 STATIC_DRD_HOLD=72` | `MPK_MOE_REGRID=38:1,64,1:36,37 MPK_ZQ_FINISH=2 MPK_LDOWN_GRID=28,8,1` |
  | MLA T 64 | mirage_bs_hql | `HQ_FIRST NOSTEAL HQ_LOCAL` | `STATIC_GEMM_ACT_PAD=1 STATIC_GEMM_ACT_DMA_PAD=1 RB=8 SLOAD` | `MPK_MLA_COMB_SPREAD=1 MPK_MOE_REGRID=38:1,64,1:36,37` |
  | KDA T 32 | mirage_bs_nst | `HQ_FIRST NOSTEAL` | `RB=8 SLOAD COAL` | `MPK_MOE_REGRID=38:1,32,1:36,37;29:28,4,1:28` |
  | MLA T 32 | mirage_bs_nst | `HQ_FIRST NOSTEAL` | `RB=8 SLOAD` | `MPK_OPROJ_FRS=1`, XFMG `{"29":[34,7,1],"31":[144,1,1],"32":[24,7,1],"37":[112,2,1]}` |

- Where a T 64 KDA kernel's ~300-330 us go (server stamps srvd173 / srvd174, max over ranks): the front until the first W13
  unit ~33 us (router, top-k, expert table, z_q; ~8 us of reads); the experts ~175 us (W13 ~6.3 TB/s, W2 ~5.5; the slowest
  rank ~10 us behind the median per layer, all through the phase); the tail ~124 us (R send, sum_rmsnorm, latent_up, agg1,
  projection, conv, recurrence, o_proj, attn_residual: ~35 us of reads, the rest dependent hand-offs and 3-5 GPU hops, each
  step limited by bytes per CU on few CUs). SGLang's KDA layer at bs 64 is 26 kernels back to back (364.5 us: experts W13 +
  W2 139 us, front + shared + routing ~67, tail ~158; `~/megakernel/amd_runs/2026-10-22/` reread notes, 10-26 21:05).
- Bound: bytes per KDA layer per GPU at bs 64 ~1.08 GB (experts ~810 MB at ~370 experts, front 52, tail weights 120, states
  98) -> ~167 us at 6.46 TB/s, plus ~40 us of dependent steps -> ~207 us vs the 1.88x target ~194 us; bs 32 ~168 vs 155, bs
  16 ~135 vs 132. Changes left on the list each bound at 1-2 % (R send overlapped with W2 by block-ordered items, attn_residual
  with 2 hops, bf16 KDA state - a numerics change vs SGLang, not decided).
