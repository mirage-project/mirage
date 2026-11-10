---
name: amd-kernel-development
description: Writing, building, checking and timing HIP kernels for AMD MI355X (gfx950, CDNA4) on the shared 8-GPU box - hardware facts that change the design (8 L2s, scopes, polling bits, vmcnt, per-CU read rate, hot counter lines, LDS DMA, LDS banks, xGMI), the toolchain and code-identity checks (stripped .s md5, unifdef, leave-one-out bisects), the box lease workflow and job-file traps, the measured traps (inline-asm store hazard, task-entry vmcnt, big fused kernels' spills), and how library changes are judged (paired in-server A/B, server stamps, all-GPU traces). Use before writing or changing any gfx950 kernel or task, before any GPU run on the AMD box, when cleaning flags out of a kernel, and when a gfx950 kernel is slower than expected, hangs, gives wrong values, or ends at different times on different GPUs.
---

# AMD kernel development (MI355X, gfx950)

Everything here was measured or read during the K3 port (MoE 2026-10-09..13, KDA / MLA / one-kernel layers / server
10-13..16), the batch-decode campaign (bs 1..64, 10-22..27) and the GLM-5.3-Flash TP4 decode campaign (11-07..09,
worktree `~/megakernel/mirage_amd_glm`, every job in `amd/loop/glm53_loop.md`, frozen state
`amd/notes/glm53_frozen_2026-11-09.md`; skill `amd-megakernel-development` section 9). Sources with page numbers and labels (ISA / LLVM / asm /
inferred) are in `amd/notes/cdna4_isa_facts.md`; the MoE numbers in `amd/notes/layer_math_mi355x.md`, KDA / MLA /
one-kernel / server in `amd/notes/kda_mi355x.md` (append results there, don't start new notes files); the batch-decode
campaign's log (every job, number and verdict) in `~/megakernel/amd_runs/2026-10-22/STATUS_decode.md`. Repos:
`~/megakernel/mirage_amd` (branch `mk-amd`, bs <= 8 work) and `~/megakernel/mirage_amd_bs` (batch decode); HIP headers under
`include/mirage/static_megakernel_hip/`. `mirage_upstream` is read-only.

## 1. The machine (what changes the design)

| fact | value / source |
|---|---|
| CUs | 256 (8 XCDs x 32), wave64 only, 4 SIMDs per CU |
| registers | up to 512 VGPR per wave (256 arch + AGPRs); SGPRs <= 102; SGPR pressure spills kernel arguments to VGPR lanes |
| LDS | 160 KB per CU; one workgroup may take all of it (our kernels: 1 WG of 256 threads per CU); 64 banks of 4 B |
| L2 | 4 MB per XCD, **8 L2s per GPU, not coherent with each other** for normal stores |
| Infinity Cache (MALL) | 256 MB, memory side; a 1 GB read empties it and the L2s |
| HBM read | best measured 6.38 TB/s (`amd/bench/read_bw.hip`); 6.44-6.50 TB/s for 800 MB by a plain grid-stride nt read (1024 x 256 threads) on every GPU, 8 GPUs at once or one at a time (`gpu_var.hip`, 10-27); ~5.7 TB/s from 224 GEMM tasks (`gemm_load.hip`) |
| GPU-to-GPU spread | the same 800 MB read on the 8 GPUs at once ends within 3.2 us median (p90 7.3) of 130 us (`gpu_var.hip`): ~2.5 %. A kernel spread much larger than that is the kernel's own (see section 2, hot lines) |
| read per CU | **~17-36 GB/s with 1 WG per CU** (plain loads 21-31, nt 17-21; `l2_carry.hip`; ~36 when few CUs read, ~26 when all 256 do; B300 SM ~54): a stream on the critical path needs ~200+ CUs; 2 WGs per CU at most +25 % (`percu_waves.hip`); with only 16 KB in flight per CU ~11 GB/s. **A step that runs on few CUs is set by its bytes per CU**, not by the GPU's bandwidth (the tail steps of a decode layer) |
| MALL as a staging place | not faster than HBM for a stream (`read_bw` t1: 4.67 vs 6.0 TB/s on 256 CUs, per-CU limit) |
| L2 within one kernel | a slice pre-read by the same CU earlier in the kernel streams from L2: 64 KB per CU (2 MB per XCD) 0.64 vs 3.04 us cold, 128 KB 3.08 vs 5.16 (`l2_warm.hip`); **nt pre-reads are not kept** |
| L2 across kernels | nothing usable carries over, even with the same WG on the same XCD (`l2_carry.hip`): no next-kernel prefetch |
| clocks | `s_memrealtime` = 100 MHz constant (`wall_clock64()`); `s_memtime` = shader clock (2300-2420 MHz seen); GPUs' clocks are not shared |
| XCD of a WG | read `HW_REG_XCC_ID` (`xcc_id()`); round-robin `wg % 8` is the usual case but not documented |
| launch | two dependent full-grid kernels 3.6 us apart (`launch_gap.hip`); in a HIP graph ~2.5 us; one host thread launching 8 GPUs one after the other: ~5-8 us apart |
| cross-GPU signal | one-way 0.50-0.70 us (sc0 sc1 or sc1 polls; `ping_pong.hip`) |
| xGMI hop (data as flag, 8 GPUs) | 2.3-2.6 us from 1 KB to 115 KB per peer **with the 8 slots' loads in flight together** (the old "3.7-4 us per hop" was a bench polling the slots one after the other); in-kernel 115 KB per peer ~1.8 us at ~64 GB/s per link plus latency |
| one all-reduce burst (TP4) | every GPU pushes [T][4096] bf16 to the 3 others once, after a cross-GPU barrier (`amd/bench/hop_once.hip`, 11-09): 512 / 256 / 128 KB per peer **~14 / 8 / 5 us** from the first push to the last landed word, whatever the quiet time before. Back-to-back rounds (`hop_big.hip`) read 10 / 4.7 / 2.7 us because consecutive rounds overlap: never use them as one all-reduce's cost |
| pushes from one XCD | 32 senders all on one XCD (blocks 8 w): 512 KB per peer 17.1 vs 10.4 us spread over the 8 XCDs (`hop_big SENDER_XCD`, 11-09): an XCD's own path to the fabric limits; keep a burst's pushing tasks on many XCDs |
| reading landed data | 128 WGs x 256 threads x 4 16-B loads (2 MB) take ~1.5 us per pass, the same for sc0 sc1 / sc1 / plain and for data written locally or by another GPU (`amd/bench/sysload.hip`, 11-09) |
| remote store instructions | a wave's system-scope stores to other GPUs go out ~0.6 us apart (the next load waits in issue order for each store's ack): one instruction with a lane per GPU beats one instruction per GPU (1.57 vs 5.0-6.3 us round trip); spreading a task's pushes over its 4 waves did not bring the data earlier (kda184, kda188) |
| two kernels on two streams | B's WGs start on the CUs A freed once A's WGs are all resident; in a HIP graph only with `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1`; capturable form: `hipStreamWaitValue32` on A's resident count (`overlap.hip`). Measured for KDA -> MoE: no gain (overlap mode removed 10-16) |

## 2. Memory scopes: the rules that cost days

- `sc0/sc1` = scope (wave/group/device/system), `nt` = non-temporal. With 8 L2s, **device- and system-scope stores write
  through and device-scope loads skip the CU's vL1D**; a device-scope poll of another XCD's data is served from MALL/HBM:
  one round trip ~1-3.5 us under load.
- **Polling bits (`amd/bench/xcd_poll.hip`, 10-27, two WGs on one XCD):** a poll with plain or sc0 loads **never** sees
  another CU's store (it keeps returning the vL1D copy: 0 / 20 within 200 us), whatever the store bits. sc1 loads see a plain
  (write-back) store of a CU on the **same XCD** in 0.28 us (served from the XCD's L2) and an sc1 store in 0.40 us. So: never
  poll with plain / sc0 loads; a hand-off whose readers all run on the producer's XCD may use plain stores + sc1 loads
  (`STATIC_CSR_HQ_LOCAL`: the W13 -> W2 h_q of an expert whose units and items are all on one die, -0.4..-2.2 us per T 64
  kernel); readers on other XCDs need sc1 stores. Lines shared by two producers on different XCDs must stay sc1 (a plain
  partial line written back later overwrites the other's bytes).
- **A plain store stays in its XCD's L2 (write-back).** Another XCD will not see it, and worse, a dirty line written back later
  overwrites what another XCD wrote to the same line. Seen: a 0xFF reset left dirty in one L2 overwrote `Rn` after
  `sum_rmsnorm` wrote it; `latent_up` polled forever. Data other CUs read: `st16_dev` (sc1) stores, `ld16_dev` (sc1) loads.
- Cross-GPU data: `st16_sys` (sc0 sc1) stores into the peer's buffer, `ld16_sys` polls. **`nt` alone is not system-visible**:
  a "nt" store with no sc bits stayed in the sender's L2 and the receiver hung (`push_latency` first version). `nt sc0 sc1`
  works but was slower (4.4-4.9 us per hop vs 3.7).
- Release/acquire the LLVM way (`buffer_wbl2` + `buffer_inv` per CU) writes back / invalidates the whole XCD L2: a start
  barrier built on it took 9.7-13 us. `runtime.cuh` has only the bypass protocol now (decided by `handoff_stress.hip`):
  data with sc1 stores (`st16_out`), `wait_mem()` (vmcnt 0), a relaxed add (`cnt_post` = `cnt_add_bypass`), readers poll
  with `cnt_wait_in` and load data with sc1 (`ld16_in_at`).
- Data as its own flag: a buffer 0xFF before the launch, the reader re-loads each 16 B until no 4-B word is all-ones
  (`valid16w`): no counter round trip. Used for every cross-GPU exchange and most cross-CU hand-offs.
- Atomics bypass L1; agent-scope atomics run past the L2 (fabric latency). 256 CUs adding to one 128-B line serialize:
  spread counters over lines (one line per die or per `sm_id % 8`).
- **Hot lines at a phase end (10-27, the largest single win of the batch campaign).** In the MoE experts task every wave
  whose die queue ran out read all 8 dies' counter lines to steal W2 items; ~1000 waves on 8 lines at the end made the last
  ~5 experts of each die run 12-25 us per item instead of ~7 (their ticket atomics returned late, and each item's loads wait
  for its ticket in vmcnt order). The GPUs with the same work then ended up to 24 us apart. `STATIC_CSR_W2_NOSTEAL=1` (no W2
  steals): -6.5..-10 us per kernel at bs 32 / 64 in the server, the expert phase's per-die ends within ~6 us without it.
  Rule: no many-wave polling or probing of a few lines while other waves still need atomics on them; when identical work
  ends at different times on different GPUs, look at per-item traces of **every** GPU (`MPK_TRACE_ALL=1`, section 6).
- **Remote pushes are acknowledged before they land (11-09).** A sender's `s_waitcnt vmcnt(0)` after its system-scope pushes
  returns long before the receiver can see the data: GLM T 64 o_proj finishers' pushes acknowledged at ~73 us, an arrival
  count added after that complete at ~75, the data seen at ~87; T 32 the same ~12 us gap with half the bytes. The gap is the
  transfer (the burst's ~8-14 us, row above), not a visibility bug: a sender-side `buffer_wbl2 sc0 sc1` (`STATIC_FIN_WBL2`)
  did not shorten it and made the finishers +10..25 us slower (it writes back every dirty line of the XCD).
- **Polling floods delay the data they wait for.** 128 CUs x 256 threads polling system-scope slots of an exchange slowed the
  other GPUs' writes landing there (`STATIC_HCPM_FEWPOLL`: thread 0 polls one vector first, -12..-15 us per KDA T 32 / 64
  kernel). The more slots every thread polls, the later the same data lands (own slot only +8 us, + one peer +11..14, all four
  +23: `STATIC_HCPM_NOLAND` probes). Batching a thread's polled loads (all at once, re-poll the invalid ones) made every
  consumer later (`STATIC_ABS_BATCH`: DSA T 64 +10..23 us); per-vector poll loops act as a throttle and consume data as it
  lands - a counter that makes the reader wait for the whole input (`STATIC_DSA_OUTCNT`) lost the same way (+25 us).
  Batched loads are fine where a counter already says the data is complete (no polling rounds).
- **Dirty lines left for the next phase.** 64 MB of KDA state written with plain stores (write-back) at T 64 made the
  following MoE experts phase ~12-15 us slower in the same kernel (the experts' weight stream evicts and writes them back).
  `buffer_wbl2 sc1` by one CU per XCD at the experts' start (`STATIC_GME_WBL2`) recovers part; nt sc1 state stores
  (`STATIC_DRD_STATE_WT=2`) recover the rest **once each store instruction writes whole lines** (`STATIC_DRD_COAL`, section
  3): with quarter-line stores write-through had lost (+26..+75 us), with whole lines it won in the server (bs 32 -647 us).
- A reset of an exchange buffer by plain stores leaves dirty 0xFF lines in the resetting XCD's L2: sc1 polls on that XCD then
  read the stale 0xFF and never see the other GPUs' data (`hop_big RESET_FF=3`: every poll gave up). Clean lines (plain loads
  of landed data) cost ~0.5 us, nothing more.

## 3. Instruction-level traps

- **`vmcnt` counts stores too, in order.** `s_waitcnt vmcnt(N)` for a load also waits for every older store. Large reset
  stores issued before a task's loads delayed those loads by microseconds. Issue loads first, or keep stores on other waves.
  - **Task entries (10-26):** the generated task loop read each list entry `my[ti]` by vector loads, each waited with
    `vmcnt(0)`, so every task start waited for all stores of the previous task (remote pushes, state stores: 4-7 us gaps).
    `STATIC_TASK_SLOAD` reads the entry as one 16-B vector through an `address_space(4)` pointer: `s_load_dwordx4` + `lgkmcnt`
    (-1.5..-5 us per kernel). Read-only metadata (task lists, node ids, kernel arguments) by scalar loads. Two compile
    pitfalls: `reinterpret_cast` across address spaces is rejected (cast via `size_t`), and a struct assignment from an AS4
    lvalue fails (load an `ext_vector_type(4)` int, then build the struct).
  - `__syncthreads()` does not add `vmcnt(0)` here (58 of 99 barriers in a T 64 kernel had only `lgkmcnt` before them).
  - A ticket `atomic_fetch_add` from lane 0 is turned by the atomic optimizer into a wave reduction + `vmcnt(0)` +
    readfirstlane at once: "in flight" tickets were synchronous (10-24). Check the `.s` after writing one.
  - An item's own small loads (scales, activation pieces) issued after its weights make the next item's weights drain at its
    first MFMA: issue them first (`STATIC_CSR_W2_HQ_FIRST`, -2 us at T 64).
- **Per-thread layouts and lines:** the KDA state (128 x 128 fp32 per head) with a thread owning 16 consecutive columns made
  each wave load / store touch 32 lines a quarter each; columns `4 kq + 32 c'` give 8 whole 128-B lines per instruction
  (`STATIC_DRD_COAL`, -4.2 us per T 64 K3 KDA kernel; GLM 11-09: KDA layer T 64 106.9 -> 97.5 us, decode end 58.1 -> 45.5,
  server bs 64 -414 us per step, and it is what made write-through state stores pay). Check line efficiency of every
  per-thread layout of a big stream.
- **A stamp after a divergent poll loop waits for all 64 lanes**: thread 0's "my vector landed" stamp placed after its lane's
  loop recorded when the slowest lane of its wave got its data (GLM glmprof22 read a 14 us late landing that was not). Stamp
  inside `if (tid == 0)` with thread 0 polling alone, or record per-thread times into LDS max / min (`STATIC_HCPM_LAND_STAMP=2`).
- **Inline-asm stores wider than 8 B need `s_nop 1` inside the asm** (`runtime.cuh` `st16_dev` / `st16_sys`): the compiler
  inserts hazard wait states only after its own stores; a VALU writing the store's data VGPRs right after stored the NEW
  values (10-14: `u_out` non-finite only in the normal build). When an output is wrong but values computed from the same
  registers are right, read the `.s` for writes to the store's data VGPRs right after it.
- The compiler inserts `s_waitcnt` where results are used or registers are reused, not where you think: read the `.s`.
  Inline-asm loads are invisible to its waitcnt tracking; wait yourself before using their result.
- **LDS DMA (`__builtin_amdgcn_global_load_lds`, `global_load_lds_dwordx4`) is not in the compiler's vmcnt model**: one
  wave writes 1 KB contiguous at M0, completion is vmcnt, so wait `vmcnt(0)` yourself before reading the LDS; every later
  load waits behind the DMA in issue order (a DMA issued after a task's last MFMAs made the next task's entry and weights
  wait: kda187, +3 us).
- **LDS banks**: an activation slice whose rows are a multiple of 256 B apart puts the 8 tokens' MFMA fragments on the same
  banks: pad the rows by 16 B (`STATIC_GEMM_ACT_PAD`; `amd/bench/lds_mfma.hip`: a 24-step loop from LDS 3638 -> 2038 clocks).
- Kernel arguments: by-value `G` / `Maps` stay in SGPRs for the whole task loop and spill to VGPR lanes; `kernel_begin`
  touches every 64-B kernarg line at once (`karg_touch`, always on), `STATIC_ARG_RELOAD` re-reads the few pointers a task
  uses from the kernarg segment per task (fused KDA + MoE: 584 SGPR spills -> few).
- **Big fused kernels**: a spilled loop-invariant VGPR reloaded before a GEMM loop made the loop head wait `vmcnt(0)` every
  round (`STATIC_GEMM_LANE_ASM` recomputes the lane with a volatile `v_mbcnt`); a `sched_barrier` after each K step's refill
  keeps loads in flight (now plain code); one task loop over two parts made the first part's loop-carried values spill
  because the second part needs ~500 VGPRs: `MPK_SPLIT_LOOP=S` (two loops, nodes < S first; 475 VGPRs, no spill). Before
  judging a fused kernel, compare `.vgpr_spill_count` / `.sgpr_spill_count` / `ScratchSize` and the hot loops' `s_waitcnt`
  against the separate builds.
- Buffer loads (`__builtin_amdgcn_raw_buffer_load_b128` with a buffer resource) with aux bits for cache policy: the GEMM's
  weight streams use `GEMM_SHUF_AUX` (2, nt); the expert weights nt (`wload`, plain loads 2.3 us slower).
- MFMA: bf16 `v_mfma_f32_16x16x32_bf16`; scaled fp8/fp4 MFMA for MXFP4 experts; `ds_read_b64_tr_b16` for transposed A
  operands (`lds_tr_check.hip`). Check operand lane layouts with a bench (`mfma_layout_check.hip`) before trusting a kernel.
- Wave reductions by DPP (`mov_dpp` quad_perm / row mirrors + `readlane`), not `__shfl_xor` (an LDS round trip per step).
- `s_sleep N` sleeps 64*N cycles; poll loops use `s_sleep 1`.
- A hot loop's first execution is cold (instruction fetch): small sections can double in time in-kernel.

## 4. Toolchain and code identity

- Build environment: docker image `lmsysorg/sglang-rocm:v0.5.21-rocm10-mi35x-20261009` (hipcc, torch, SGLang, AITER), our
  venv `/mpk-amd/env` inside it for the harness. Compile-only work runs in that container **without the GPU lease**
  (`docker run --rm --name mpk-amd-NAME -v $HOME/mpk-amd:/mpk-amd IMAGE bash -c ...`); the box has 192 cores: 8
  library builds in parallel are fine.
- `bash ~/mpk-amd/logs/cgc4.sh TREE NAME flags...` builds the 4 variants (8-GPU, solo, profile solo, profile 8-GPU) on the CPU.
  It does not pass `--fold`: for spills / scratch use `amd/tools/codegen_check.py --gpus 8 <the real flags incl. --fold,
  --pack, --regrid>` and read `ScratchSize`, `.sgpr_spill_count`, `.vgpr_spill_count`.
- Server libraries: `MPK_RANK_LIB_ASM=1 python amd/tools/gen_rank_lib.py ...` also writes `<out>/lib.s`.
  `~/mpk-amd/logs/mkfinal.py TREE PREFIX` writes `final_PREFIX.sh`, which builds every library the final server run loads
  (from the job files' `libs()` steps) from one tree and prints each `lib.s` stripped md5.
- **Every new flag defaults off and must leave the default code identical.** Check: md5 of the `.s` with
  `.loc/.file/.ident/;` lines and `hip_cuid` stripped, before vs after, for the default build and for the current best
  config. Instruction scheduling can change with no semantic change (e.g. a value computed later): restore the original
  expressions under `#else` until the md5 matches.
- **Removing flags (10-16 cleanup)**: `unifdef -k -x 2 -DX=v ... -UY ...` on the headers resolves the preprocessor branches
  at the values every library builds with (never on `.py` files: Python `#` comments parse as directives). It leaves
  compound `#if A || B` with an unknown name and every C-level use (`if constexpr (STATIC_X)`, `constexpr int N =
  STATIC_X`): those by hand. Then rebuild all final libraries and compare stripped `lib.s`. **Even an edit with no
  semantic change can move register allocation** (removing an empty inline call, restructuring dead code): find which
  file does it by leave-one-out builds (all files at their unifdef-only form = identical, then each file's cleaned form
  alone; `logs/bis/run_bis.sh`). When only register numbering / spill lanes differ (same register totals, no scratch),
  prove the change by bit-identical harness outputs against the old build's saved outputs (`--save-outs` refs,
  `logs/cmp_ref.py`) and the server run, and say so.
- AITER needs the GPU arch to import: on a CPU-only container set `GPU_ARCHS=gfx950`. SGLang's `ModelConfig` still needs a GPU.

## 5. The box (the shared 8x MI355X machine)

- Everything under `~/mpk-amd` (= `/mpk-amd` in the container). **Never "xinhao" in any name on the box.** Trees: `mirage`
  (main), `mirage_stage` (compile checks, most jobs), `mirage_post` (the cleaned tree of 10-16, final libraries
  `/mpk-amd/fpost_<kind>`), `mirage_pre` (the tree before the cleanup). Keep the tree a job runs equal to local (md5 of
  every file of the changed directories, one line each side).
- **All GPU runs go through the box lease**: `cd ~/mpk-amd/logs && (nohup bash lease_runner.sh NAME 8 > runner_NAME.log 2>&1 &)`
  runs `job_NAME.sh` in container `mpk-amd-NAME`. Steps via `step.sh` (`step KEY cmd...`, results in `NAME.res`, done keys in
  `NAME.done`; rerun under a new NAME). Queue: `sudo ls /run/lock/cresco-gpu/queue`. Our tickets are priority 2: waits of
  minutes to hours. Don't raise the priority.
- Every job starts with an md5 step and a marker grep proving the code it runs. Sync = scp + md5 compare in its own call.
- Never edit a job file while its bash runs: write `.new`, `bash -n`, `mv` (only while the job has not started).
- **Shell quoting in job files**: inside double quotes `$L\_moe` keeps the backslash (`/mpk-amd/fpost\_moe/lib.so`: the
  server could not load the library, 10-16); outside quotes it is removed. Write `${L}_moe`. `bash -n` does not catch it:
  grep the expanded paths before leasing.
- Background waiter for every job (poll `NAME.res` / `runner_NAME.log`; a remote command that can exit non-zero needs `; true`
  so the waiter does not mistake it for a lost connection). Report a lost box at once.
- Kill a stuck job with `docker kill mpk-amd-NAME` (never `pkill` a pattern that is in the same command line); a lease
  runner restarts its job after a docker kill: kill the runner's PID first.
- Copy results back after every job (`~/megakernel/amd_runs/<date>/`). The box is shared: other users' jobs and files are not ours.
- **Job-file traps of the batch campaign (10-26/27):**
  - `step.sh` runs every step in the **same shell**: a helper that sets `A=...` without `local` overwrote the job's ALT
    library set; 13 paired A/Bs (srvd135-155) then compared main against main ("ALT: libraries for batch sizes []" in the
    server log). `local` for every helper variable; the run step refuses to start when the ALT set is missing; grep the
    result for `batch sizes \[\]` before reading any A/B.
  - A job file written through `ssh '... <<EOF ... EOF'` breaks on the first apostrophe in a comment (the lines then run in
    the remote shell). Write job files locally and `scp` them.
  - `env $FKENV` splits on spaces: a value that holds a list (`MPK_MOE_REGRID`, `MPK_MOE_FOLLOW_LATE`) needs `;` separators
    the harness splits on (`MPK_MOE_REGRID` accepts `;`). A variable passed through `XFKENV` / `XFKG` **replaces** the
    recipe's value of that variable (the whole grid JSON; the recipe's `MPK_MOE_FOLLOW_LATE="38:36:37 39:36:38"`): pass the
    recipe's value plus yours, or the A/B is not like for like.
  - **Start a lease runner detached** (GLM 11-09): `cd ~/mpk-amd/logs && (setsid nohup bash ./lease_runner.sh NAME 4 >
    runner_NAME.log 2>&1 < /dev/null &)` in the ssh command. A plain `nohup ... &` inside `ssh` died with the session (no log,
    no job), `./lease_runner.sh` without `bash` failed (not executable), and the waiter reads `runner_NAME.log`.
  - **A job name is used once**: `step.sh` skips every step already in `NAME.done`, so a reused name measures nothing and
    reports old results (GLM glmchain3). Check `ls ~/mpk-amd/logs | grep -c '^NAME\.'` is 0 before writing the job.
  - The job's container has no `docker`: removing root-owned folders (our old library sets, trees) is a separate
    `docker run --rm -v ... IMAGE bash -c 'rm -rf ...'` from the box host, not a job step.
  - Two 4-GPU jobs can run at once (GPUs 0-3 and 4-7): their timings differ by ~10 us per 430-us kernel between the halves and
    with the other job's load - compare only variants inside one job.
  - Library sets `/mpk-amd/dlibNN` are symlink farms into older sets (`readlink -f` inside the container: the targets are
    container paths). A new set = `mkdir` + links + **real folders** for the changed kinds; never `cp -a` into a linked
    entry (it writes through into the production set). Check that a new tree / set name is free before creating it.

## 6. Measuring

- e2e first: GPU events right before / after the kernel, slowest GPU, median of 40 launches, **1 GB read on every GPU before
  each launch** (L2s and MALL empty), GPUs lined up. Span (per-CU stamps) only beside it, labeled.
- Like for like with B300 (harness): B300's harness ran a 5 us stand-in previous kernel with PDL and the layer's resets before
  `griddepcontrol.wait`. Ours: early-launched layer (`STATIC_EARLY`), resident post + line-up (`STATIC_EARLY_RES`,
  `STATIC_EARLY_PREWAIT`, `STATIC_LINEUP`), `STATIC_EARLY_PREV_US=5`; the resets are always in the kernel. Host-side resets
  after the flush leave the reset buffers warm and uncounted: never quote that (`STATIC_COLD_RESETS_MB` probes it).
  (After the 10-16 cleanup these harness-only switches are still in the code but not yet rebuilt or run.)
- Rank libraries (the server's kernels): `torchrun --nproc-per-node 8 amd/tests/test_rank_lib.py --kind K --lib L --ref
  REF.pt`: outputs bit-identical to the harness's `--save-outs`, then e2e per rank eager and HIP graph (1 GB flushed,
  ranks lined up). 10-16 final (eager): kda 50.0-50.8, mla 47.3-48.8, moe 86.9-87.6, moeagg 89.2-90.4, moeaggkda
  131.0-132.0, moeaggmla 129.7-131.1 us.
- The GPU launched last gets the shortest pre-go window: anything before the go (resets, barrier, prefetch) must fit it there.
- Same-session A/B, >= 3 rounds per variant; run-to-run spread is ~1-2 us at 85 us, ~30 us on the server's forward span.
  Put the decisive variant first.
- Stamps: `STATIC_PRE_STAMPS` (normal build: per CU entry, stores landed, go), `STATIC_NODE_END_STAMPS=n0,n1,..` (normal
  build: per CU its last end of each listed node), `STATIC_EXPERT_STAMPS` (moe_experts' W13 / W2 phases), the timing build
  (`--profile`: per-task times, `STATIC_STAMP_SET` picks a task's 2 stage stamps, dumps under `<out>/trace/`). The timing
  build is slower and can move spills: confirm every conclusion in the normal build. Align GPUs at their go, not at clocks.
- Trace tools: `amd/tools/trace/` (`xgpu_an.py` per-GPU node ends aligned at the go, `nodeend_an.py`, `queue_an.py`,
  `pre_an.py`, `start_an.py`, `w2_an.py`, ...).
- Microbenchmarks in `amd/bench/` (`push_latency`, `ping_pong`, `hop_size`, `read_bw`, `gemm_load`, `l2_warm`, `l2_carry`,
  `lds_mfma`, `overlap`, `launch_gap`, `handoff_stress`, `sync_latency`, `hot_line`, `issue_rate`, `mfma_rate`,
  `percu_waves`, `w2_infl`, `w2_extra`, `w2_pattern`, `gpu_var`, `xcd_poll`, ...): measure a mechanism there before building
  on it, and check the bench measures what you think (`hop_size` polled serially and overstated the hop by ~1.6 us). A bench
  win is not a kernel win: the W2 benches showed h_q loads and row stores costing 25 % of the W2 rate; packing the stores
  (`STATIC_CSR_W2_PACK`) and loading two items ahead (`W2_AHEAD`) did not move the server.

### Library changes are judged in the server, paired (10-26/27)

- **Paired in-server A/B** (hook tree `mirage_bs_srv10`, `sglang_k3.py`): main set M for every layer, `MPK_AMD_DEC_FKDA_ALT` /
  `..._FMLA_ALT` = set A for the layers of one parity (`MPK_ALT_PARITY=0/1`), 4 server runs p0 / p1 / p0 / p1, each with a
  profiler trace at the batch sizes asked (`BSLIST="32 64"`); `amd/tools/trace/kdur_alt_an.py` pairs the parities: "paired
  ALT - main" per kernel kind, two numbers (rounds a / b). Precision ~+-1.5 us per kernel (layer parity alone differs by up
  to ~7 us inside one run, so never compare unpaired). Adopt only when both rounds agree in sign at the batch sizes it targets.
- **The lined-up harness is a screen, not a verdict:** harness ties became server wins (one sum_rmsnorm task per token -3.9,
  HQ_FIRST -2) and harness wins tied in the server (cooperative z_q at T 32). Use it to find causes (stage stamps, traces)
  and to reject big losers; decide in the server.
- **All-bs status** = one session, SGLang alone then ours, `BSLIST="1 2 4 8 16 32 64"`, GSM8K 64, `step_split.py` per bs
  (forward span ratio). Two sessions of the same libraries differ by 1-2 % at bs <= 32 and up to 4 % at bs 64: a 1-2 us per
  kernel change does not show there; full GSM8K (1319) per adopted set.
- **Server stamps:** a stamped build of the set (`STATIC_NODE_END_STAMPS=n0,..,n7` in the fused kernels, hook tree
  `mirage_bs_srv6` with `MPK_DUMP_STAMPS` / `MPK_LAYER_STAMPS`), then `amd/tools/trace/srvstamp_an.py PREFIX CONFIG T KDA_NODES
  MLA_NODES` (per node last end per rank, median over layers), `rankspread2.py` (per layer max - min over ranks of each node:
  a node after an exchange shows the start skew, the experts show the real spread), `STATIC_CSR_PHASE_STAMPS` + `phase_an.py`
  (per CU W13 end / first W2 / W2 end: where the slowest rank loses), `srv_go_an.py` (CU go spread, experts table times).
- **Harness traces of every GPU:** `MPK_TRACE_ALL=1` (harness option, tree `mirage_bs_gtr` and later) dumps every GPU's moe
  item trace; `gvar_an.py` (per GPU W13 / W2 item counts, durations, wave-time sums), `w2long.py` / `w2slot2.py` / `w2die.py`
  (slow items by time, slot, die), `dieend.py` (per-die ends), `expvar.py` (per-GPU node start / end distribution),
  `table_an.py` (experts table build), the read curve in the timing log (`read curve GPU g`: TB/s per 2 us).
- Kernel time vs everything else in a server step: `kfwd.py TRACE` (time per kernel name). At bs 64 the fused kernels are
  ~99 % of the step; consecutive kernels in the HIP graph run back to back (no gaps). GLM TP4 (`amd/serving/glm/glm_steps.py
  TRACE --top N`): outside the layer kernels ~230 us per step at bs 64 (logits all-gather 92.5 us at the link rate, lm_head 51).
- **Session spread decides how small a status change can be read**: two GLM STATUS sessions of the same bs 32 / 64 kernels
  differed by ~2 % (bs 32 1.203 vs 1.175); a paired A/B in one session moved by -647 us at bs 32 for a change worth
  ~-2..-14 us per kernel in the harness. Report a target "met" only when it holds in more than one session.
- **Find which side of a hand-off is slow with stamps on both sides, in steps** (GLM 11-09, the attention all-reduce): the
  sender's "pushes complete" (`STATIC_GEMM_FIN_STAMP`: vmcnt 0 after its pushes), the receiver's per-source landing of one
  vector (`STATIC_HCPM_SLOT_STAMP` 1 / 2 / 3: own slot vs remote slots, first vs last token, the thread's own vector), the
  per-thread landing spread (`LAND_STAMP=2`), the arrival of a counter posted after the pushes (`FIN_CNT` + `LAND_STAMP=3`).
  Each step ruled out one cause; three earlier days of single guesses (reset protocol, polling rate, TLB, pull) had not.
  Then reproduce the mechanism in a bench with the kernel's real pattern (synchronized one-shot burst: `hop_once.hip`).

## 7. Against SGLang on ROCm

- Layer level: `amd/bench/sglang_moe_layer.py` builds SGLang's own module (`KimiK3MoE`) under TP8 with torchrun inside the
  image (MI35x recipe env: `SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1`), loads the
  harness's weights through its parameters + `process_weights_after_loading`, captures a HIP graph, times it like the
  harness, `--kernels N` lists its kernels, `--compare-y` compares outputs with ours.
- Server level (the result that counts): `amd/serving/srv_run.sh NAME` (full K3 + DSpark, TP8, the B300 run's flags; client,
  `GSM8K=N` questions one at a time, a profiler trace), our kernels via `python amd/serving/sglang_patch.py` (hooks into
  `kimi_k3.py`, `--restore` undoes) and `MPK_K3_<KIND>=run MPK_AMD_LIB_<KIND>=lib.so`; compare with SGLang alone in the
  same session: `step_split.py OURS_TP1.trace BASE_TP1.trace /mpk-amd/k3/config.json` (per-part times, forward span, step
  period), `compare_answers.py`. 10-16 final: span 11081.4 vs 20873.0 us, step period 13670.5 vs 24889.8, full GSM8K
  (1319) 0.950 at 221.1 tok/s vs 0.947 at 141.6 (B300 span 9035).
- torchrun sets `TORCHELASTIC_USE_AGENT_STORE=True`: SGLang's `tcp://` init then makes every rank a store client and nobody
  hosts the store. Pop it before SGLang's init.
- SGLang ROCm numerics differ from SGLang CUDA in places (K3 MoE: the front GEMM writes z in bf16 on HIP, fp32 on CUDA; the
  conv + SiLU output bf16). Compare against float64 references with each rounding order before calling either side wrong.

## 8. What worked and what did not (10-11..16; details in layer_math_mi355x.md and kda_mi355x.md)

Kept (all plain code since the 10-16 cleanup unless named as a flag): shuffled GEMM weights; `--regrid` / `--pack` /
`--apart` / `--light` plans putting chain GEMMs on many CUs; per-die work queues with one counter per die handing out W13
units then W2 items, the next ticket taken inside the current unit; R in blocks; RMSNorm folded into latent_up; the
attention residual's values as their own flags and its weights per lane; DPP reductions; the KDA recurrence with the decay
folded in and in chunk form; MLA queries late, no kT transpose (`ds_read_tr`), 4 queries per wave softmax; in-kernel
resets with a two-copy arena (each launch resets the next launch's copy at its end, on the CUs without a last-node task)
and per-CU launch counts; kernarg touch and task-entry prefetch before the go (`STATIC_TASK_PREFETCH`); the next layer's
aggregation 1 and its attention part in the MoE layer's kernel (one kernel per layer pair; `MPK_SPLIT_LOOP`); flags
`GEMM_LDSW=3` (polled GEMMs' slice by LDS DMA), `GEMM_ACT_PAD`, `GEMM_L2WARM` (a waiting projection task pre-reads its
next task's first 256 KB into the L2), hybrid projection tasks (`gemm_tile_layer(hybrid=True)`), `GEMM_STAGES=16`.
Lost (code removed 10-16; don't repeat without a new reason): release/acquire barriers, double-buffered host resets, resets
at the last task, per-die claim pools with atomics, resets competing with the front, pre-go weight prefetch, W13
half-defer / quarters / CU-wide units / late skip, W2 speculation / deep / adapt / stream / take-2, start batch 2, nt
stores for exchanges, rotated pushes (`PUSH_ROT`), pushes spread over waves, next-kernel L2 prefetch, two-stream overlap
(`STATIC_RESIDENT`), activation pre-poll, carried / K-split LDS weight paths, gate-late recurrence, split projection.

Batch decode, bs 16..64 (10-22..27; every verdict a paired in-server A/B unless marked harness):
- Kept (flags, all default off, set in the best libraries): `STATIC_CSR_W2_NOSTEAL` (-6.5..-10), `STATIC_DRD_COAL` (-4.2 KDA
  T 64), `STATIC_TASK_SLOAD` (-1.5..-5), sum_rmsnorm one task per token (`MPK_MOE_REGRID=38:1,T,1:36,37`, -3.9 KDA T 64, -1..-3
  MLA T 64), `STATIC_CSR_W2_HQ_FIRST` (-2 T 64), cooperative z_q (`MPK_ZQ_FINISH=2 MPK_LDOWN_GRID=28,8,1
  -DSTATIC_GEMM_QUANT_COOP=2`, -2 KDA T 64), MLA T 32 o_proj in K parts (`MPK_OPROJ_FRS=1`, grid 112 x 2, -2), z_q in 4 token
  groups at KDA T 32 (`29:28,4,1:28`, -1.6), `STATIC_AGG1_RB=8` (~-1), `STATIC_DRD_HOLD=72` (KDA T 64, -1), `STATIC_CSR_HQ_LOCAL`
  (T 64, -0.4..-2.2).
- Lost or tied (do not repeat without a new reason): poll sleep 16 between counter polls; x through the L2 for the projection
  (`ACT_L2_BUFS`); `ACT_DMA_PAD` for the KDA projection; `COOP_LAST_NOWAIT`; `PAIRS_PAR` at T 64; `W2_AHEAD`; `TAKE_ASYNC`;
  `W2_PACK`; R rows in bf16; R send on 256 tasks; MLA 4 KV splits at T 64; KDA recurrence 192 tasks (FLAT 4); finer
  projection / o_proj / latent_down / router / shared gate_up grids; projection tasks off the agg1 CUs (`MPK_ATTN_CAP`, +11..14);
  shared gate_up apart from router / latent_down (+2.5..5), after the experts (`MPK_LATE_SET`, +40..70: S lands too late) or on
  the router's CUs; two layer pairs in one kernel at T 64 (57 VGPR + 478 SGPR spills); the hybrid projection at T 32 / 64
  (does not compile with today's grids).

GLM-5.3-Flash TP4 decode (11-07..09; details and every number in `mirage_amd_glm/amd/loop/glm53_loop.md`):
- Kept: `STATIC_DRD_COAL` (GLM default), `STATIC_DRD_STATE_WT=2` + `STATIC_GME_WBL2` in the T >= 32 KDA + MoE kernels;
  activation-bound gemms in K parts - a gemm whose task time follows its activation bytes per task (every task loading the
  whole [T][K] slice) runs as finish_all / store in 4 K parts (KDA o_proj T >= 32, DSA o_proj T >= 16 store + send, dense MLP
  down projection T >= 16: -9 us per KDA + dense kernel at T 64); MoE senders land + LDS hand-off from T 4; FEWPOLL.
- Lost (flags, off): reset / pull / wbl2 / one-poller-counter variants of the all-reduce landing (FIN_PULL, FIN_CNT,
  FIN_WBL2, HCPRE_FEWPOLL, FIN_REV); every state request up front in registers (`STATIC_DRD_PFALL`: the prologue's vmcnt(0)
  waits for all of them, +25 us); more decode tasks (rpt 2 at T 64); batched polls (ABS_BATCH) and counter-gated whole inputs
  (DSA_OUTCNT); q_b in K parts for mla_absorb (`MPK_DSA_QB_KP`: mla_absorb starts late behind the gemms anyway); count posted
  right after a W13 unit (`STATIC_GME_POSTNOW`); fewer sparse-attention splits at short contexts.

## 9. Method lessons from the B300 kernels (10-13/14; measured there)

From `mirage_kda/.claude/skills/b300-megakernel-task-kernels` (numbers in `mirage_kda/docs/kernel_optimization_notes.md`).
- Find the bound before redesigning: one timing-only build with the copies only (compute skipped) and one with both, in one
  job. B300 MLA: copies 0.55 us per chunk vs 1.05 with compute, so a deeper copy ring (proposed) would not have helped.
- Reductions over splits / rows: across lanes (shuffles; on gfx950 DPP / `ds_swizzle`), not a serial loop per thread
  (B300 combine weights 4.3 us -> shuffles).
- After every change, check the whole kernel's resources (private segment / scratch, VGPR count), not only the task changed:
  on B300 a register-held gather and a second task variant made the compiler spill other tasks' long-lived state (seen on
  gfx950 too: section 3, big fused kernels).
- Streamed data read once: no-retain cache policy; keeping it resident cost the next kernel (B300 MLA KV rows, EVICT_LAST).
- A task that waits for other tasks while it runs must be declared concurrent in its layer call (skill
  `amd-megakernel-development` section 7).
