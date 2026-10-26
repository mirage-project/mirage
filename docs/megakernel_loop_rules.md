# Rule catalog for the megakernel loop

The loop (docs/megakernel_dev_loop.md) reads this file. Every rule has:
- an id, used by the ledger and by the agents' stage prompts;
- what to do;
- why: the measurement and its source;
- how it is enforced.

**Enforcement kinds:**
- `auto:<check>`: a script fails the round (the checks are listed in the design doc, section 6);
- `review`: an item on the reviewing agent's checklist for that stage;
- `gate`: a correctness gate (stage G);
- `measure`: how a measurement must be taken (stage H);
- `model`: how the cost model or search must treat it (stages D, E).

**Status:** every rule is `verified` (measured, source given) unless marked `open` (cause unknown) or `hint` (seen once, not
isolated). A new measurement that contradicts a rule changes its status in the same round, with a ledger reference.

Numbers are B300 TP8 unless marked B200, measured during the K3 MoE / KDA / MLA work (dates are 2026).

---

## A. Reference, bound, inputs

- **A1** Only the in-server block time is compared with SGLang. Block = the end of the kernel before the part → the end of the
  part's last kernel.
  - Why: KDA harness 29.0 vs 50.1 in server; MoE harness 72.2 vs 138.0 in server (memsets).
  - Enforce: `review` (reports label harness numbers).
- **A2** Inputs match the server's statistics per call. For MoE that is the distinct-expert count: 47–117 per 8-token step,
  server mean 80.5.
  - Why: 0.31 µs per expert for us, 0.40 for SGLang.
  - Enforce: `gate` (the input set covers the server's p10/p50/p90).
- **A3** Same weights and inputs on both sides, with a routing-equality check and the load distribution reported.
  - Enforce: `gate`.
- **A4** Bound = bytes read once ÷ measured HBM bandwidth, plus the minimum hand-offs. Use measured rates: B200 read 6.39 TB/s;
  148 SMs with 16 KB TMA rings 6.8 TB/s; one SM 97 GB/s.
  - Why: MoE 83 experts ≈ 182 MB/GPU ≈ 26 µs (W13 7).
  - Enforce: `review`.
- **A5** Split SGLang's time into phases from existing traces before choosing where to work.
  - Why: front/experts/tail 27.8/40.1/15.1 vs ours 19.8/38.5/17.4 (INF).
  - Enforce: `review`.
- **A6** Read the competitor's SASS first (warp roles, ring depth, fences, schedulers), then build its structure as a 1-GPU bench
  on our data.
  - Why: w2_bench 19.17 vs 15.71 µs led to the W2 port, −1.7 µs on the layer.
  - Enforce: `review`.
- **A7** Before any TP8 number, check the fabric (dmesg FABRIC_STATE, the server's "all-reduce fusion" line).
  - Why: a broken NVLS slowed SGLang's MoE 87.2 → 100.7 µs.
  - Enforce: `auto:preflight_box`.
- **A8** Error limits come from the reference's own run-to-run noise.
  - Why: hand-picked 1e-3 fails seeds 3 and 5 at 1.02e-3 by rounding (job 74).
  - Enforce: `review` (`open` for MoE).

## B. Graph and contracts

- **B1** Candidate grids = every size whose blocks hold whole units of every cut dimension. Uneven splits and changeable axes
  that cut nothing are errors.
  - Enforce: `auto:lint_contract`.
- **B2** A minimum work per task per issuer is declared and enforced.
  - Why: gemm_tile with 1 K tile left issuer 1 idle and added a stale accumulator (relRMS 1.1); KDA split-K hit the same bug.
  - Enforce: `auto:lint_plan`.
- **B3** Dependencies come from overlapping slices only, so maps must be as fine as the data really is.
  - Why: residual_add reads o whole, waits for all 98 sum_send tasks, and its duration came out −0.77 µs.
  - Enforce: `review`.
- **B4** `combine` is a contract field the search chooses, with legality.
  - A K-split before a nonlinearity must combine first (SiLU(a+b) ≠ SiLU(a)+SiLU(b)).
  - A recurrence can't be split along time.
  - Enforce: `auto:lint_contract` (`open`: not built).
- **B5** Shared memory is a formula of split, tile and stages, plus a minimum; not one number.
  - Why: the smem plan and prefetch decide the amount (F).
  - Enforce: `auto:lint_contract`.
- **B6** Each task declares its release points and whether its output can carry a sentinel.
  - Why: about 25 of KDA's 32 µs of gains came from scheduling, sync and placement: early publish 4.5, data-as-flag 3.6+1.5+0.2.
  - Enforce: `review`.
- **B7** Release and join granularity are chosen together.
  - Why: a late publish costs +5.2 µs only with fine joins; coarse joins alone +0.3–0.4 (FIR).
  - Enforce: `model`.
- **B8** Task body order: prefetch (weights, TMEM alloc, barrier init) → wait → compute → publish. Producer-independent loads go
  before the wait.
  - Why: weights-first +4.3–5.2% single SM.
  - Enforce: `review`.
- **B9** Fuse only when the trace shows removed work, a removed GMEM round trip, or a removed critical-path edge.
  - Why: waits are conserved otherwise; conv→recurrence fusion 0.995x.
  - Enforce: `review`.
- **B10** Sparse decisions stay outside dense stream blocks.
  - Why: route inside the expert block holding all SMs: +33%.
  - Enforce: `review`.
- **B11** Persistent per-layer state is written by layer L and read by the next launch, never inside one launch. A sentinel slot
  is re-armed by its last consumer, ordered by a dependency.
  - Why: the y_rec re-arm race (CODE_MAP §3–5).
  - Enforce: `review` + `gate` (G4).
- **B12** Names describe the operation. After a rename, diff every changed line in every file, tracked or not.
  - Why: a regex rename broke the server load (job 101).
  - Enforce: `auto:lint_rename`.
- **B13** Buffer sizes are timing: any size change moves later addresses. A polled output keeps only this GPU's rows.
  - Why: full-width buffer +0.45 µs (jobs 94–97); 8-part tail 27.2 vs 23 µs.
  - Enforce: `model` (layout is part of the code tag, D9).
- **B14** The generator has nothing model-specific; its inputs are only the graph and the schedule.
  - Enforce: `auto:lint_source` (grep the generator for part constants).

## C1. Kernel design

Tensor memory and MMA:
- **C1** `tcgen05.ld/st` reach only TMEM lane slice `warp_id % 4`; a 128-lane access needs 4 warps covering 0..3.
  - Why: wrong slices silently give wrong rows.
  - Enforce: `review`.
- **C2** M=64 MMA pairs keep A and D at the same TMEM lane offset (0 or 16).
  - Why: otherwise "misaligned address".
  - Enforce: `review`.
- **C3** Allocate 512 TMEM columns.
  - Why: a 64-column allocation worked only by luck, with one CTA per SM.
  - Enforce: `review`.
- **C24** tcgen05 builds use `-gencode arch=compute_10Xa,code=sm_10Xa`. sm_100a and sm_103a features are not portable.
  - Why: plain `-arch` mis-lowers.
  - Enforce: `auto:lint_build`.

mbarriers and named barriers:
- **C4** mbarrier parity is exact only within one phase.
  - Issuers wait every stage's `full` and issue only their own; loaders wait `empty` only for their own stages.
  - Every issuer that reads a slot arrives on `empty`.
  - Why: a deadlock (loader side), "Warp Illegal Instruction" (issuer side), a hang in 1 of 150 launches from ring lapping.
  - Enforce: `review` with the per-barrier ledger of `b200-mbarrier-protocol-auditor`; hand-simulate the first, a steady and the
    last phase.
- **C5** Barrier words never share space with data; offsets derive from the layout end, never a literal.
  - Why: 0 failures in 650k launches alone, 3 of 3 with a co-running kernel.
  - Enforce: `review` + `gate` (G6).
- **C21** Named barriers can't be signalled by TMA or commit.
  - Why: `bar.sync` wakes in 38–41 ns vs 88–103 ns for a suspended mbarrier wait; there are 16 ids in total.
  - Enforce: `review`.

Shared memory:
- **C6** No static `__shared__` in task templates.
  - Why: +16 KB went over 228 KB and caused an illegal access; crossing 1 KB of `.nv.shared` lowered the dynamic maximum.
  - Enforce: `auto:lint_source`, `auto:lint_build`.

TMA and loads:
- **C7** TMA from uniform registers. The loader warp runs in lockstep and an elected lane issues.
  - Why: 91–97 vs 175–185 ns/stage.
  - Enforce: `auto:lint_sass` (R2UR count in the loader loop) + `review`.
- **C8** A TMA issue costs 60–100 ns, so a stream of many small loads needs a second loader thread.
  - Why: W13 item 10 → 7.0 µs.
  - Enforce: `review`.
- **C9** 4-bit weights expand 2x in smem: one SM streams 37–40 GB/s (smem-port bound), so M128N8 tiles on 148 SMs can't reach HBM
  bandwidth.
  - Enforce: `model` (use it in the task's bound).
- **C10** A generic-proxy smem write read by MMA/TMA needs `fence.proxy.async`.
  - Enforce: `review`.
- **C17** Batch global loads into registers before stores; no clobbered asm poll loads.
  - Why: aliasing serialised loads cost 18 µs; 28 clobbered polls 3.2 µs.
  - Enforce: `review`.

Generated code shape:
- **C11** G and Maps stay `__grid_constant__` kernel parameters, and task bodies stay `__forceinline__`.
  - Why: constant copies +2.7; separate functions +4.4 / +1.9 / +1.7; G by reference 8 → 50 loads (+4.9).
  - Enforce: `auto:lint_source` (generated code) + `auto:lint_sass` (no CALL into bodies).
- **C12** Dispatch from shared memory; copy globals and this SM's task records into smem in one burst at the start.
  - Why: three dependent global loads +1.2 µs; constant-bank misses 1.4 µs.
  - Enforce: `review`.
- **C13** No `-use_fast_math`.
  - Why: y changed in 3267 of 57344 values.
  - Enforce: `auto:lint_build`.
- **C14** No fp64 in the hot path; use `__ll2float_rn(q) * 2^-40f`.
  - Why: one `(double)` cost 1.6 µs.
  - Enforce: `auto:lint_sass` (DMUL/DFMA/DADD/F2F.F64/I2F.F64 = 0 in task bodies).
- **C15** No runtime-indexed register arrays; keep the original unroll.
  - Why: spills +54 µs/round; unrolling mbarrier-paced K loops +15% / +74%; rolling dequant loops about 10x slower.
  - Enforce: `auto:lint_sass` (LDL/STL = 0) + `review`.

Warps and signalling:
- **C16** Elected-lane code never uses `__shfl` / `__syncwarp`.
  - Why: hang at launch 1.
  - Enforce: `review`.
- **C18** "All threads wrote, one signals" = `bar.sync` + one `red.release.gpu`; no per-thread `__threadfence`.
  - Why: −1.1 / −1.5 µs; a per-thread fence stalls the next global access about 3.3 µs.
  - Enforce: `auto:lint_source` (`__threadfence` inside a task loop is flagged) + `review`.
- **C19** Split issuers by tensor-op overlap. W13: whole stages per issuer. W2: its two atoms split across both issuers.
  - Why: 0.8 vs 0.6 µs/stage.
  - Enforce: `review`.
- **C20** A thin MMA issuer, TMEM allocated once per worker.
  - Why: a 254-register MMA warp was the problem.
  - Enforce: `auto:lint_build` (registers per thread reported).

Layout:
- **C22** The static smem layout moves the span (3.4 µs from 0x200).
  - Enforce: `model` (code tag includes the layout) (`open`: cause unknown).

Flags:
- **C23** A flagged variant must leave the default SASS identical.
  - Why: an offset in front of an int index made it 64-bit.
  - Enforce: `auto:lint_sass` (default-vs-baseline diff).

## C2. Optimization record

Each entry: the change → result (conditions). The loop must not retry a "lost" entry without a new reason written in the ledger.

**Won:**
- **W1** TRT-LLM W2 structure (1 tile × 12 MMAs per stage, 3 stages, 1 MMA warp, 2 loaders) → bench 19.2 → 17.0, layer
  83.2 → 81.5.
- **W2** 15 queue variants → one path → 77.8 → 74.7, 648 instructions fewer.
- **W3** 2-tile entries + 1-tile entries for the last NSM tiles → slowest SM 3.3 → 0.9 µs after the median. Paid only after the
  per-entry costs were removed.
- **W4** Poll both vectors in one loop; load prefix/Ssum before the o poll → R 3.6 → 2.6, o 3.3 → 2.4, layer 74.7 → 72.2.
- **W5** Take the next W13 entry late (after 8 stages armed) → −3.3 µs at 16 experts.
- **W6** Epilogue math from TRT-LLM SASS (8 tokens at once, `tanh.approx`, `__expf`, exponent-bit scales).
- **W7** latent_up early weight load into free ring buffers → −1.0.
- **W8** Dual MMA issuers → KDA lone launch +2.2–3.0 without them.
- **W9** Pipeline depth over tile size for memory-bound decode GEMMs → depth 10 −6.3% pool, −18.3% splitk (B200). KDA proj ring
  plateau 8–12, cliff below 8.
- **W10** L2 hints: dense weights EVICT_FIRST (wrong default about 3% decode); expert payload hint −10.4% decode iteration. Check
  the hint reaches the kernel.
- **W11** Short-K GEMM as one monolithic task → 5.55 → 3.15 µs.
- **W12** Tensor-core recurrence about 10 µs; early publish about 6; parallel dots 1.5; aggregate counter 1.5 (KDA).
- **W13** TMA for reductions → SSQ 20.7 → 14.2 µs.

**Lost:**
- **L1** L2 prefetch of the next W2 entry +5; h_q as data-as-flag +2.5; release on the loader's idle lane +2; delaying a waited
  release +2.5.
- **L2** Stage-granular W2 issue 93–94 vs 89 µs; 4 W13 stages +0.4.
- **L3** Few-expert tricks: early table +2–3; route counter +3.1; shared down after the queue +5; half-SM plan +0.5.
- **L4** MXFP4 tile: W13 K parts (122 → 97 worse); 1D bulk + relayout 27 GB/s/SM; M-tile pairs; NACC=8; 4 issuers.
- **L5** Split-K over many SMs: KDA 148 SMs +3, 112 SMs +1.4; projection split-K at M=128 (fp32 exchange 1.7).
- **L6** Radix top-16; z_q bulk copies; TMA bulk push vs 16 B stores; `ld.relaxed` vs volatile polls (no change).
- **L7** In-loop readiness bitmaps on a saturated machine (about 55 ns/check + i-cache). Pays only if producer spread ≥ consumer
  start gain.
- **L8** List-scheduled static queue order +11.9% vs round-robin.
- **L9** Stream-K: at most 2.1–2.4 µs to win (CUTLASS minimum 8 K steps); not built.

**Open:**
- **O1** W13 bytes in flight: 80 KB (ours) vs 96 KB at the same ratio; 6th stage about 10 KB short. Keep rule: ≥2 µs at
  37–59 experts and ≤0.3 µs worse at 91/113 (INF).

## D. Cost records

- **D1** Duration = end − max(previous task's end on the SM, inputs' end); first on the SM: end − own start. End = the latest
  warp's exit.
  - Why: replays the span to 0.01 µs in 9 of 9 runs.
  - Enforce: `model`.
- **D2** Records carry context:
  - first on SM, previous node, cold/warm code, tiles in flight on the GPU;
  - launch mode, pass choices, bucket, input distribution, machine, CUDA version.
  - Why: route 4.19 / 7.10 / 8.58 by position; GEMM tile 2.3–2.5 at ≤60 in flight vs 3.2–3.7 at ≥120; first tile 7.3 vs
    8.6 by how many SMs start with a tile; cluster +0.2–0.6.
  - Enforce: `auto:lint_records`.
- **D3** Pre/post: `end = max(start + pre, inputs ready) + post` for tasks with input-free work.
  - Why: route 10.3 with no slack vs 5.4 with ≥5 µs; sact 4.9 → 2.4.
  - Enforce: `model`.
- **D4** Use quantile 0.7–0.8, not the median.
  - Why: the span is the slowest path; medians are 0.5–4.3 µs low in-sample; leave-one-out mean −0.79 µs, 6/13 within 1 µs.
  - A wave ends at its slowest task: router 5.5 predicted vs 8.0 measured.
  - Enforce: `model`.
- **D5** Bandwidth-sharing models were worse than context models (worst 10–13 µs off).
  - Enforce: `model`.
- **D6** The search never picks an option with no measured records.
  - Why: fitted options were 20–109 µs off; job 1899 chose an unmeasured K28 (107.5 vs 89.7).
  - Enforce: `auto:lint_plan`.
- **D7** Instrumentation buffers are allocated last.
  - Why: the times buffer first moved the part's buffers, +3.5–3.8 µs.
  - Enforce: `review`.
- **D8** A failed run records nothing; a SASS change invalidates the old records.
  - Enforce: `auto:lint_records`.
- **D9** The code tag hashes what changes timing, not the formatting: the task templates' source with comments and whitespace
  removed (`static_schedule.code_tag`, 10-07). Not the SASS: a body is inlined into `layer_kernel` with plan-dependent template
  arguments, so a SASS hash would differ per plan, not per template version; `lint_sass` keeps a per-build SASS summary instead.
  - Why: a whitespace format changed the old md5 tag (10-07); layout moves times (B13, C22).
  - Enforce: `auto:code_tag` (`open`: the generator's buffer / exchange layout code is not in the tag yet).
- **D10** The pool's (dynamic queue's) fixed per-SM start (7.7 µs to the first MMA) is not sharable work.
  - Enforce: `model` (`open`).

## E. Search

- **E1** Always solve from a start plan, solved fixed first and handed in complete.
  - Why: from scratch nothing at 30 / 120 / 300 s; every returned plan came from a hint; a partial hint ended worse (92.76 vs
    91.75).
  - Enforce: `auto:search_driver`.
- **E2** Time limit ≥ 60 s. FEASIBLE is not OPTIMAL; report both and the bound.
  - Why: 20 s solves were non-deterministic and poor (101.5 µs).
  - Enforce: `auto:search_driver`.
- **E3** The model is trusted for a phase only when it predicts the measured incumbents within 1 µs for that phase.
  - Why: 09-23: a free front placement 86.9 vs hand 87.1 predicted, so placement was not the gap.
  - Enforce: `model` (model-error tracking, design §5).
- **E4** Cross-GPU arrival is not modelled; the tail is predicted about 8 µs low.
  - Why: job 79 moved S sends after the queue (73.3 vs 68.8).
  - Enforce: `model` (`open`).
- **E5** Bytes are not time. LPT by weight bytes misplaced the shared tiles; an off-path branch only fills windows where the ring is
  free.
  - Why: ISSUE_K3_MOE_PLACEMENT.
  - Enforce: `review`.
- **E6** SMs are interchangeable only until a dry pass, a preload or a cluster pair depends on neighbours.
  - Enforce: `model`.
- **E7** Pool end imbalance about 2.4 µs is unmodelled.
  - Enforce: `model` (`open`).

## F. Post-processing

Dry pass:
- **F1** It runs for a node's first task on an SM: the same code on stand-in inputs, no global stores, one code copy, no run-time
  readiness check.
  - Why: cold route 6.0 → 14.0 µs; KDA o_proj 3.0 vs 1.6; tf32 first use 3.2 µs/layer; the readiness check cost 0.3–0.7
    (jobs 87/91).
  - Enforce: `review`.
- **F2** It never reads slots under TMA writes or TMEM under `tcgen05.cp`.
  - Why: two placements hung.
  - Enforce: `review`.

Cluster:
- **F3** Cluster 2 + on-chip swap only when every pair fits (`compiler.pair_clusters`).
  - Why: 82.3 vs 83.4 plain; the cluster launch alone +0.2–0.6; clusters of 4 don't fit.
  - Enforce: `auto:lint_plan`.

Shared memory and prefetch:
- **F4** A task whose smem overlaps its predecessor's live region waits for that predecessor's epilogue. The smem plan's output is
  {owner, offset, size, early} per buffer and must fit 227 KB with no space deadlock.
  - Enforce: `auto:lint_plan`.
- **F5** Preload only from ring slots measured worthless, and paced.
  - Why: ring 10 + prefill 2: 36.99 → 36.74; unpaced +0.5; ring 12 vs 11 +0.5; half the in-flight bytes +4 (FIR).
  - Enforce: `review`.
- **F6** A prefetch must not slow other kernels. Accept only if the overlapped kernels' durations are unchanged and end to end
  wins.
  - Why: KDA L2 prefetch layer −5.3, step +32 µs; MoE W2 prefetch +5; bimodal span from our own prefetch kernel.
  - Enforce: `gate` (H4).

Hand-offs:
- **F7** Hand-off costs: same SM ≈ 0; another SM ≈ 1 µs; another GPU ≈ 2.5 + 3–5 µs skew; a 148-CTA barrier 4.4 (B200).
  - Enforce: `model`.
- **F8** Data-as-flag beats counter + system fence.
  - Lamport 38.6 vs 43.4; all chunk loads in flight 37.5; slice-local poll 0.3 vs 1.8 for a full row.
  - No sys fence per tile (+7); no fence after multicast stores (about 5.5/tile).
  - Enforce: `review`.
- **F9** Sentinel rules:
  - validate every byte;
  - write unused bytes of pushed buffers;
  - e4m3 satfinite / UE8M0 ≤ 254 never give 0xFF;
  - tag single-copy buffers with the launch number;
  - the prefill must actually be the sentinel.
  - Why: the h_q check with a 0 prefill always passed (GPU 5 partial 3.2e-2).
  - Enforce: `gate` (G5 mutation).
- **F10** Polling:
  - all loads of one round in flight;
  - poll only your own slice;
  - don't add pollers or traffic during a landing (12–25 µs delays seen);
  - many SMs polling many lines slows the producers (+0.2 to +5.5);
  - one aggregate counter with `__nanosleep(64)`: −1.6.
  - Enforce: `review`.
- **F11** Two exchange sets instead of a start barrier.
  - Why: start 5.9 → 3.3, layer −1.4 to −1.6.
  - Enforce: `review`.
- **F12** Multicast bf16 partial sums for cross-GPU sums (SGLang's push all-reduce form).
  - Why: R multicast −1.0, R bf16 −1.6 to −2.1, S −0.3; y moved <0.5%.
  - Enforce: `review`.
- **F13** No two-hop collectives inside a part.
  - Why: they won 1 µs standalone and lost 1.3–2.8 in the server.
  - Enforce: `review`.
- **F14** A guessed poll design is measured before it is kept.
  - Why: one poller + a flag was 2.2 µs slower than 148 pollers.
  - Enforce: `measure`.

PDL, start, resets:
- **F15** PDL trigger point: measured only in the server window or the server.
  - Why: MoE start-trigger worse (81.9), none 78.0, at y_add 72.3; 10-07 modes 0/1/2 equal. KDA trigger at the predecessor's entry
    started 3.0 µs early; weight prefill before `griddepcontrol.wait` was later in every form.
  - Enforce: `measure`.
- **F16** Start barrier:
  - a launch generation in hello slots, not a reset (the reset raced);
  - parallel relaxed stores from 32 lanes (serial release stores: GPU 0 14 µs ahead);
  - the first multicast touch (480–760 µs) at init.
  - Enforce: `review`.
- **F17** Resets in-kernel or by the readers, never memsets before the kernel.
  - Why: 138 → 84.9 → 79.2 µs. Re-arm stores in tasks still cost about 2.2 µs (`open`).
  - Enforce: `review`.
- **F18** Prologue: issue the entry loads together, pass counters as parameters, split warps.
  - Why: 3.4 → 2.1 µs; the first TMA lands at 4.5–5.9 µs (TLB).
  - Enforce: `review`.
- **F19** Buffer and exchange order is part of the candidate; changes are measured.
  - Why: job 99 +0.2 (`open`: the buffer that matters was not found).
  - Enforce: `model`.

## G. Correctness gates

- **G1** Every timed launch checked (`--check-all`) on every input set and bucket. y vs reference within the A8 limit, y
  bit-identical on all GPUs, per-GPU partials vs fp32 references.
- **G2** Odd launches reverse the token order (catches leftover buffers).
- **G3** Determinism: same input → every kept buffer bit-identical across launches (`--dump-cmp`). No fp32 atomics in sums.
  - Why: fp32 `red.add` gave 2 of 8 identical server answers; ordered rows + int64 fixed point gave 8 of 8.
- **G4** Persistent state × seams: list each persistent buffer and every seam it crosses (task, GPU, launch); chained gates use
  reps ≥ 2.
  - Why: the conv-roll bug was invisible at reps 1; `iter_num` false hits.
- **G5** A mutation build (`-DMK_MUTATE`, a corrupted sentinel or partial) must fail the gates.
- **G6** Gates also run with a co-running kernel (contention) and after every schedule change.
  - Why: a race passed 15 times, then failed 4 of 4 with finer entries.
- **G7** CUDA graph replay == normal launch.
- **G8** The server: shadow mode on the part's layers, then `compare_answers.py` vs SGLang alone.
- **G9** Hang tooling: a host-mapped watchdog dump, cuda-gdb with lineinfo, CUDA coredumps through nvdisasm. compute-sanitizer
  doesn't work on polling kernels. Never `cudaMemcpyFromSymbol` in a hang dump.
- **G10** Scripted edits assert their replacement count; parsers are checked against one hand-read log line.
  - Why: a stale TMEM column; a wrong-column "same bits" claim (HVS).
- **G11** A build that fails any gate gives no performance conclusion.
  - Why: MK_PARK_PAIR "+2.9" withdrawn.

## H. Measurement

- **H1** L2 emptied by reading 512 MB, not writing.
  - Why: writing +3.3 µs; warm L2 understated KDA 2.3–3 µs.
  - Enforce: `auto:preflight_box`.
- **H2** Server window (`test_plugin.py --prev-us 5`) for every candidate before the server.
  - Why: 71.6 vs 71.9 µs server; host launches instead of graphs +16 µs; a serial line-up left ranks 9 µs apart.
  - Enforce: `measure`.
- **H3** A/B alternated on the same box and plan, 20 launches each, ≥4 rounds. Under about 0.3 µs is noise. Whole-step metrics need
  350+ samples with the SE quoted.
  - Why: 10-iteration medians carry ±21 µs.
  - Enforce: `measure`.
- **H4** End to end: about 20+ alternating requests per arm, and the neighbouring kernels' durations unchanged.
  - Enforce: `gate`.
- **H5** Prove the variant's code ran: a stamp or grep in the generated code, md5 on the box, our kernel present in the server trace.
  `-D` beats generated `#ifndef` defaults.
  - Why: five server runs on dead code; an empty library path silently fell back to SGLang.
  - Enforce: `auto:preflight_box`, `auto:check_trace`.
- **H6** No co-running persistent kernel; force-rebuild.
  - Why: keepalive inflated times and hung data-as-flag kernels; pip/cmake keep mtimes.
  - Enforce: `auto:preflight_box`.
- **H7** Stamps:
  - `%clock` into smem, copied out at the end; registers inside polling loops;
  - end = `atomicMax` of the warps' finishes;
  - `globaltimer` with a memory clobber;
  - a probe-free control in the same session.
  - Why: global stamps +2.7 µs and a 0.1–0.4 µs step inflated to 1.5–2; thread 0 read 1.5–2 µs early; stamp builds slowed the
    projection about 1 µs.
  - Enforce: `review` + `auto:lint_source`.
- **H8** Profiler times vs clock times are not mixed.
  - Why: the CUPTI end is about 4 µs after the last SM's clock; its start is 1.5–2 µs before the first instruction;
    profiler-on dilutes wins.
  - Enforce: `review`.
- **H9** Standalone or p10 gains are not wins until the server median moves.
  - Why: the server adds 1.6–1.8 µs of rank skew and +1.0 µs per op unexplained; in-kernel TMA ran 2x slower than isolated.
  - Enforce: `gate` (promotion only at H2/H3).
- **H10** Credit only structural wins; keep a change only on a same-lane A/B win plus all gates.
  - Enforce: `gate`.
- **H11** Single launches, not chained periods, are compared.
  - Why: KDA 29 µs existed only as a chained period; single launch 42–131 µs.
  - Enforce: `measure`.

## I. Diagnosis

- **I1** Stamp every step boundary of the slow stretch before proposing. A proposal names the step it removes, its predicted gain,
  and the test that would refute it.
  - Why: a one-stamp guess led to a reorder 1.5 µs slower.
- **I2** A slower build → diff the generated code and headers first, then SASS per source line (`-lineinfo`); experiments only to
  confirm.
  - Why: a 2-minute diff found what 4 A/B jobs didn't.
- **I3** Per-node first start / last end, two candidates side by side (job 79); per-node model error (measured − predicted last
  end).
- **I4** Count round trips, not instructions: one poll pass ≈ 1 µs.
- **I5** Diagnostic builds that remove one piece (no MMAs, no loads) to find what a phase is bound by.
  - Why: W2 17.22 / 17.06 vs 18.78, so W2 is protocol time.
- **I6** A 1-GPU bench of one phase vs the competitor's shape on the same data.
- **I7** A speedup for an unknown reason is measured further before anything builds on it.
- **I8** No conclusion about another kernel from its name; read its code. My code is the first suspect.

## Hardware numbers (for bounds and the cost model)

- **HBM:**
  - B200 read 6.39 TB/s, L2 126.5 MB;
  - MoE stream on 148 SMs 6.8 TB/s, one SM 97 GB/s;
  - KDA stream rate is SM-count-bound (99 SMs 4.6–4.8, 148 SMs 6.7–6.9 TB/s).
- **tcgen05:**
  - mma about 28 ns from one issuing warp at M=64/128 with N=8; two issuers 21–22;
  - block-scaled M128N8K32 about 53; `tcgen05.cp` about 70; commit about 49; try_wait 26.
- **Atomics and fences:** `atom.add.release.gpu` 450–580 ns stall vs `red` 96; release fence 383 ns; `fence.acq_rel.sys` 2–7 µs
  under the MoE stream.
- **NVLink:**
  - store→visible about 1.04 µs; push→land 1.1–2.3;
  - steady 640–670 GB/s; a 112 KB burst from one source about 65 GB/s; 7→1 270–380 GB/s;
  - multicast inbound 917 KB/rank about 5 µs.
- **On-chip:**
  - smem port about 208 GB/s per SM;
  - a ring without MMAs 164 ns/stage on 1 CTA, 328 at 148 CTAs;
  - L2-resident stream 13–16 TB/s (99 SMs, alone).
- **Clocks:** B300 2032 MHz pinned, the server at 1965 MHz. B200 vs B300 MoE layer the same within noise.
- **Rank skew:** GPUs 3–5 +1.1–1.4 µs on dependent L2 chains (B200, device permutation).
