# E6 — in-MPK per-task t_live decomposition of the MoE W13/W2 tasks (ENTRY / BODY / EXIT), flag OFF vs ON

Date: 2026-08-06 (box time). Box: catalyst-fleet1.cs.cmu.edu (8x B200).
Arms: GPU 5 (compute-apps checked EMPTY immediately before and after each arm — clean).
Standalone clock run: GPU 7 (GPU 5 acquired a foreign tenant before that run; the gate refused and
the run moved — header logs kept). Never GPU 0.
Worktree: `/var/tmp/e6_overhead/tree` = **mirage-combined-v024 @ 13e7e8a8** + the measurement-only
`MPK_E6_PHASES` patch (`raw/e6_instrumentation.patch`, 326-line diff; without the env var the
compile command and generated code are byte-identical to stock). Tree provenance: the brief named
`mirage-moe-v024`, but part1 provably ran `mirage-combined-v024` (`/var/tmp/alignment/capture_part1.sh`
hardcodes `TREE=.../mirage-combined-v024`; its graph contains fusion tasks 243/244 and its
`MPK_FUSE_*` flags exist only there; the moe tree also defaults to 128 workers and lacks the
136-worker + argmax-divisor commits), so the brief's mandatory ±3% perturbation gate against part1
is only satisfiable on the combined tree. The MoE impl file is byte-identical between the two trees
(`git diff 7ba0d62d 13e7e8a8 -- .../moe_fp8_blockscale_sm100.cuh` is empty) and the flag
`MPK_MOE_FP8_BLOCKSCALE_V024` exists there; both bodies (legacy v012 and v024) were instrumented.

Geometry = part1 exactly: bs16, synthetic prompts len 256 seed 20276730, msl 353, 96 new tokens,
mbt 16, page 256, 136 workers (tree default), profiler ON (120,000,000 slots), fuse flags set,
same `profile_wave.py` driver. Token sha256 of all generated ids is **identical in all four runs**
(instrumented off/on here and part1 off/on): `e09df5c67d35...`.

## 1. Phase table — medians over the [2,350) iteration window (p10–p90), live tasks (dur >= 4 µs), µs

t_live = the MPK profiler bracket (PROFILER_EVENT_START -> PROFILER_EVENT_END around the dispatch +
task body + post-body `__syncthreads`, persistent_kernel.cuh worker loop). T0..T3 are `%globaltimer`
stamps by threadIdx.x==0 inside `moe_impl_path` (both the v012 and the v024 body):
T0 = first instruction of the impl body; T1 = end of arg decode + smem carve (after `total_work`,
before ANY routing/prefetch/mbarrier-init/main-loop data operation — the same semantic point in
both bodies; v024 hoists the item-0 gather before its mbarrier init, so that gather counts as BODY
in both arms); T2 = work-item loop exited (this kernel writes its output per work item INSIDE the
loop, so there is no single pre-writeback point; EXIT is the post-loop teardown); T3 = last
instruction before return (after the mbarrier-inval tail + its `__syncthreads`).
bracket-resid (残) = t_live − (T3−T0) = the part of the t_live bracket outside the impl body
(profiler begin-entry write, the generated `_execute_task` type-dispatch chain, the `__noinline__`
call/return, the post-body `__syncthreads`, the profiler end stamp, and the instrument's own 24 B
ring write). It is reported as its own column per the brief.

| family | arm | entry T0→T1 | body T1→T2 | exit T2→T3 | bracket resid (t_live−ΣT) | t_live |
|---|---|---|---|---|---|---|
| W13 | OFF | 0.160 (0.128–0.352) | 14.528 (11.328–26.208) | 0.064 (0.032–0.064) | 0.832 (0.640–1.184) | 15.840 (12.288–27.488) |
| W13 | ON  | 0.160 (0.128–0.384) | 12.192 (9.504–23.424)  | 0.064 (0.032–0.064) | 0.896 (0.640–1.216) | 13.728 (10.432–24.896) |
| W2  | OFF | 0.160 (0.128–0.192) | 11.424 (7.264–16.832)  | 0.064 (0.032–0.064) | 0.800 (0.640–1.120) | 12.384 (8.192–17.952) |
| W2  | ON  | 0.160 (0.128–0.224) | 9.888 (6.080–14.272)   | 0.032 (0.032–0.064) | 0.800 (0.640–1.056) | 10.944 (7.040–15.424) |

Same decomposition in MEANS (means are exactly additive: entry+body+exit+resid = t_live to the ns;
part1's `t_live_us` anchor is the MEAN of live durations over this window, so this is the
anchor-comparable view). The resid splits into pre-gap (bracket begin -> T0) and post-gap (T3 ->
bracket end), medians in the last two columns:

| family | arm | entry | body | exit | resid | t_live (mean) | pre-gap med | post-gap med |
|---|---|---|---|---|---|---|---|---|
| W13 | OFF | 0.212 | 18.087 | 0.050 | 0.889 | 19.238 | 0.320 | 0.448 |
| W13 | ON  | 0.231 | 16.457 | 0.050 | 0.915 | 17.653 | 0.352 | 0.384 |
| W2  | OFF | 0.165 | 11.788 | 0.049 | 0.820 | 12.822 | 0.288 | 0.384 |
| W2  | ON  | 0.173 | 10.224 | 0.048 | 0.823 | 11.268 | 0.320 | 0.384 |

n per cell: W13 3,534,896 live-window tasks per arm; W2 3,563,520 (OFF) / 3,549,875 (ON).
`%globaltimer` quantizes at 32 ns, hence the stepped medians.

## 2. Body-only delta vs t_live delta, and the overhead share

overhead := t_live − body(T1→T2) = entry + exit + bracket-resid.

| family | estimator | body Δ off→on | t_live Δ off→on | overhead OFF µs (share) | overhead ON µs (share) |
|---|---|---|---|---|---|
| W13 | mean   | −9.01%  | −8.24%  | 1.151 (5.98%)  | 1.196 (6.78%)  |
| W13 | median | −16.08% | −13.33% | 1.312 (8.28%)  | 1.536 (11.19%) |
| W2  | mean   | −13.27% | −12.12% | 1.034 (8.06%)  | 1.044 (9.27%)  |
| W2  | median | −13.45% | −11.63% | 0.960 (7.75%)  | 1.056 (9.65%)  |

- **W13: body improved 9.01% (mean; median 16.08%), t_live improved 8.24% (mean; median 13.33%),
  overhead share 5.98% OFF / 6.78% ON (mean; medians 8.28% / 11.19%).**
- **W2: body improved 13.27% (mean; median 13.45%), t_live improved 12.12% (mean; median 11.63%),
  overhead share 8.06% OFF / 9.27% ON (mean; medians 7.75% / 9.65%).**

Reference arithmetic from the brief, side by side with the measurement: the dilution hypothesis
(fixed overhead + a ~20% faster body) required overhead ≈ 11.8 µs = 63% of W13's 18.76 µs t_live;
measured overhead is 1.151 µs = 5.98% of t_live (mean; median-based 1.312 µs = 8.28%), and the
measured in-MPK body delta itself is −9.01% (mean), not ~−20%.

## 3. Clock table (SM MHz, `nvidia-smi --query-gpu=clocks.sm` at 200 ms during the runs)

| regime | GPU | busy samples | median | p10 | p90 |
|---|---|---|---|---|---|
| in-MPK arm OFF (profiled wave, util >= 50%) | 5 | 16 | 1965 | 1965 | 1965 |
| in-MPK arm ON (profiled wave, util >= 50%)  | 5 | 16 | 1965 | 1965 | 1965 |
| standalone record instrument `bitexact_v024 --benchmark` (bursty; util >= 1%) | 7 | 8 | 1965 | 1965 | 1965 |

Idle clock on this box reads 120 MHz; every busy sample in every regime read 1965 MHz.
(In-MPK and standalone clocks were logged on different free GPUs of the same box per availability;
E1 §7 measured cross-GPU medians on this box within 0.4%.) The standalone run reproduced the
record numbers on this worktree: w13 bs16 v012 99.967 -> v024 79.446 µs (−20.53%), w2 56.114 ->
45.673 µs (−18.61%); log: `logs/standalone_bench.log`.

## 4. Perturbation check (instrumented vs part1 uninstrumented, same estimator: mean of live in window)

| quantity | part1 anchor | instrumented | Δ | budget |
|---|---|---|---|---|
| W13 t_live OFF µs | 18.764 | 19.238 | +2.53% | ~3% PASS |
| W13 t_live ON µs  | 17.375 | 17.653 | +1.60% | ~3% PASS |
| W2 t_live OFF µs  | 12.535 | 12.822 | +2.29% | ~3% PASS |
| W2 t_live ON µs   | 11.008 | 11.268 | +2.36% | ~3% PASS |
| step time OFF µs  | 8949.3 | 9015.6 | +0.74% | ~3% PASS |
| step time ON µs   | 8928.5 | 9000.1 | +0.80% | ~3% PASS |
| wave wall OFF ms  | 3151.5 | 3174.8 | +0.74% | (info) |
| wave wall ON ms   | 3144.1 | 3169.4 | +0.80% | (info) |

The brief labeled the part1 anchors medians; in the part1 pipeline they are MEANS
(`width.py: t_live_us = d[live].mean()`), so the check above compares mean to mean. No re-run with
reduced instrumentation was needed. Instrument cost note: 4 stamps + 1 ring write, threadIdx.x==0
only; the ring write executes after T3, so it lands in the post-gap/resid column, not in any phase.

## 5. QC / matching

- Profiler pairs per family per arm: 3,604,480 = 10240 tasks x 352 iterations exactly; E6 ring
  records: 3,604,480 — equal on every one of the 136 worker tracks (0 mismatched workers);
  0 dropped begin/end events; no ring wrapped (max 26,880 records/worker of 65,536 slots).
- Matching is positional per (worker, family) and every match was verified by timestamp
  containment modulo 2^32 (ring T0 inside the profiler pair): **0 failures out of 14.4 M matches**.
- fam field: OFF = {17, 18} = PATH1 v012 body (w13, w2); ON = {273, 274} = PATH1 v024 body — both
  arms ran fetch PATH 1, v024 bit set only in the ON arm.
- Live fraction in window: W13 99.20% (both arms), W2 100% OFF / 99.62% ON; dead (empty) tasks are
  excluded exactly as in part1's estimator.
- Generated dispatch instantiations (identical both arms):
  W13 `moe_fp8_blockscale_task_impl<bf16,16,8,256,512,1024,2048,true>`,
  W2 `<bf16,16,8,256,1024,2048,512,false>`, expert_stride 128.

## 6. Exact commands

```
# worktree (part1's tree+commit) + untracked build artifacts
git -C /home/muhengl/mpk-qwen35/mirage-combined-v024 worktree add /var/tmp/e6_overhead/tree 13e7e8a8
ln -s .../mirage-combined-v024/python/mirage/core.cpython-312-x86_64-linux-gnu.so /var/tmp/e6_overhead/tree/python/mirage/
ln -sfn .../mirage-combined-v024/build /var/tmp/e6_overhead/tree/build
rmdir tree/deps/{cutlass,json,z3} && ln -s .../mirage-combined-v024/deps/{cutlass,json,z3} tree/deps/
# instrumentation patch (kept at raw/e6_instrumentation.patch): moe_fp8_blockscale_sm100.cuh
# (kernel::e6 ring + T0..T3 stamps in BOTH moe_impl_path bodies), persistent_kernel.py
# (-DMPK_E6_PHASES flag + e6_dump_func exporter in the launcher boilerplate), profile_wave.py
# (dump call after the wave).

# the two arms (mirrors /var/tmp/alignment/capture_part1.sh; adds MPK_E6_PHASES/MPK_E6_OUT,
# SM-clock logging, and compute-app snapshots before AND after each arm)
CUDA_VISIBLE_DEVICES=5 /var/tmp/e6_overhead/capture_e6.sh
#   per arm: env [-u] MPK_MOE_FP8_BLOCKSCALE_V024[=1] MPK_E6_PHASES=1 MPK_E6_OUT=raw/e6_<arm>.bin \
#     MPK_FUSE_SILU_QUANT=1 MPK_FUSE_NORM_QUANT=1 MPK_FUSE_RECUR_QUANT=1 \
#     python profile_wave.py --batch-size 16 --max-seq-length 353 --max-new-tokens 96 --mbt 16 \
#       --page-size 256 --synthetic-prompt-len 256 --synthetic-seed 20276730 \
#       --out-dir <arm>/prof --kernel-dir <arm>/kernel --rep 0 --slots 120000000 --save-raw

# analysis (phase tables, matching, perturbation) -> raw/analysis.json
/home/muhengl/mpk-qwen35/venv-rm/bin/python /var/tmp/e6_overhead/analyze_e6.py

# standalone record instrument (E1 sec 8 build, from THIS worktree) + clock log
/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a \
  --expt-relaxed-constexpr -DMIRAGE_GRACE_BLACKWELL -DMPK_TARGET_CC=100 -DMODE_OFFLINE \
  -DMIRAGE_BACKEND_USE_CUDA -DMIRAGE_ENABLE_MOE_FP8_BLOCKSCALE_V024 -use_fast_math \
  -I$D/include/mirage/persistent_kernel -I$D/include/mirage/persistent_kernel/tasks -I$D/include \
  -o /var/tmp/e6_overhead/bitexact_v024_fm $D/demo/qwen3_5/accept/opt/m4i7/scripts/bitexact_v024.cu
CUDA_VISIBLE_DEVICES=7 /var/tmp/e6_overhead/standalone_clocks.sh   # ($D = the worktree)
```

## 7. Raw artifacts

- Phase rings: `/var/tmp/e6_overhead/raw/e6_{off,on}.bin` (header u64[4]; ring_n u64[160];
  24 B records {u64 t0; u32 d1,d2,d3; u32 fam} x 160x65536)
- Aggregates: `/var/tmp/e6_overhead/raw/analysis.json` (all stats above, incl. p10/p90 of every column)
- Profiler captures: `/var/tmp/e6_overhead/{off,on}/prof/raw_bs16_rep0.npz` (+ meta, tokens, task_names)
- Generated kernels: `/var/tmp/e6_overhead/{off,on}/kernel/` (test_rank0.cu shows the instantiations)
- Hygiene headers: `/var/tmp/e6_overhead/{off,on}/header.txt`, `logs/standalone_header.txt`
- Logs: `logs/capture_{off,on}.log`, `logs/clocks_{off,on,standalone}.csv`, `logs/standalone_bench.log`
- Instrumentation: `raw/e6_instrumentation.patch`; scripts `capture_e6.sh`, `analyze_e6.py`,
  `standalone_clocks.sh`; decoder copy `raw/trace_lib.py` (analyze_e6.py's OPT import path can be
  re-pointed at `.../mirage-combined-v024/demo/qwen3_5/accept/opt` now that the worktree is removed)
- part1 anchors quoted from `/var/tmp/alignment/part1/{off,on}/width136.json` (read-only)
