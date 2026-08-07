# Standalone-harness versus persistent-kernel alignment

Date: 2026-08-06 (America/New_York)  
Production measurement tree: `/home/muhengl/mpk-qwen35/mirage-combined-v024`  
Ferret tree: `/home/muhengl/mpk-qwen35/ferret`

## Bottom line

The discriminating bs16 result is **Outcome B, specifically partial-transfer B with a
scheduler interaction on top**. The v024 bodies do get faster inside MPK, but they reproduce only
**9.316%** of the standalone W13+W2 time reduction, not the claimed **17.552%**. W13 improves only
**7.402%** per live task, and its realized stage span nevertheless gets **5.596% worse** because
its self-concurrency falls. The exact post-change floor is the **critical path**, whereas the OFF
arm is work-bound in all three inspected steady iterations. W13 is the dominant post-change
straggler.

Two unlisted misalignments were at least as important as the missing fast-math flag:

1. **The checked-in analysis tools describe an obsolete 128-worker machine.** Production emitted
   136 workers plus 48 schedulers (184 persistent blocks), but M4-I5 `width.py` hard-codes
   `NW=128`, and M4-I8 `sched_gap.py` defaults to 128 workers and only fits direct/two-worker-swap
   assignments from the old 128/80 topology. Unmodified, the analysis silently discards worker
   tracks 128-135 and cannot reconstruct the realized three-worker scheduler groups. For this
   report I used the generalized wrappers
   `/var/tmp/alignment/width_realized.py` and
   `/var/tmp/alignment/sched_gap_realized.py`: all six exact iteration/arm checks report
   `assign_qc=PASS` and `identity_error_ns=0`. This is a profiler-analysis alignment defect, not a
   kernel defect.
2. **The compiler executable and language standard did not match.** Production's captured command
   used `/usr/local/cuda-12.8/bin/nvcc` 12.8.93 and `-std=c++20`; the old Ferret scripts invoked
   unqualified `nvcc`, which currently resolves through `/usr/local/cuda` to CUDA 13.2.51, and
   used C++17. This is not cosmetic: on the same immutable MoE v024 source the old CUDA 13.2/C++17
   W13 timed entry is 64 registers with a 16-byte stack frame, while the production-aligned CUDA
   12.8/C++20 build is 75 registers with a zero-byte stack frame. The shipped harness now pins the
   production compiler regime.

## Part 1 — discriminating in-MPK measurement

### Protocol and validity

- Arms: shipped/OFF and `MPK_MOE_FP8_BLOCKSCALE_V024=1`/ON, each in a distinct generated-kernel
  directory.
- Geometry: bs16, prompt 256, MSL 353, 96 new tokens, MBT 16, page size 256, seed 20276730.
- Production JIT: CUDA 12.8, C++20, `-O3 -lineinfo -use_fast_math`, one shared full megakernel TU.
- GPU: physical GPU 3, exclusively checked before each arm; GPU 0 was not used. Capture windows
  started at 07:10:21 and 07:11:41 EDT.
- Each arm retained 49,705,922 profiler events over 351 task-graph iterations. Static task-count
  anchor QC passed exactly for every ordinary task type.
- The OFF and ON token JSON files are byte-identical, SHA-256
  `4379c18095b1811523cee297bb54caf60f5e0a39b2945962e0c893f90d7c6cda`;
  their content digest is also identical (`e09df5c6...a099`).
- Exact schedule reconstruction was checked at iterations 100, 200, and 300 in each arm. Every
  run used 136 workers/48 schedulers and returned `assign_qc=PASS`, zero mismatched workers, and
  `identity_error_ns=0`.

### Per-task transfer

The standalone row is the report-of-record's direct v012-versus-v024 production-geometry timing.
The in-MPK row is MPK profiler `t_live_us`. Absolute standalone and in-MPK durations have different
launch/task aggregation, so the discriminating comparison is the percentage reduction, not an
absolute-duration ratio across the two instruments.

| family | standalone v012 → v024 (us) | standalone reduction | in-MPK OFF → ON `t_live_us` | in-MPK reduction |
|---|---:|---:|---:|---:|
| W13 | 100.814407 → 80.772797 | **19.8797%** | 18.764 → 17.375 | **7.402%** |
| W2 | 58.647198 → 50.700802 | **13.5495%** | 12.535 → 11.008 | **12.181%** |
| W13+W2 sum | 159.461609 → 131.473602 | **17.5516%** | 31.299 → 28.383 | **9.316%** |

This is not Outcome A: the claimed approximately 17% body win is not reproduced in the real
megakernel. It is also not a total loss—both live-task medians improve—which is why the precise
classification is partial-transfer B rather than a pure regression.

### Why a faster task did not become a faster stage

| stage | OFF span (us) | ON span (us) | span change | OFF → ON self-concurrency | perfect-pack ratio OFF → ON |
|---|---:|---:|---:|---:|---:|
| W13 | 1822.5 | 1924.5 | **+5.596% worse** | 104.61 → 91.75 | 1.30 → 1.48 |
| W2 | 1177.8 | 1157.4 | **1.732% better** | 108.95 → 96.68 | 1.25 → 1.41 |

W13 is the new binding realized stage. Removing W13 in the exact floor counterfactual gives the
largest CP reduction in every inspected iteration: 999/1132/1108 us OFF versus
1197/1322/1211 us ON. No different task family replaced it; rather, faster W13 tasks pack less
effectively and W13 becomes more dominant on the realized chain.

### Exact critical-path versus work floor

| iteration | OFF `cp_exact` (us) | OFF work bound (us) | OFF binds | ON `cp_exact` (us) | ON work bound (us) | ON binds |
|---:|---:|---:|:---:|---:|---:|:---:|
| 100 | 4888.4 | 4956.0 | work | 5116.6 | 4754.4 | **CP** |
| 200 | 5144.1 | 5286.8 | work | 5340.6 | 5030.4 | **CP** |
| 300 | 5138.1 | 5289.1 | work | 5189.0 | 4976.6 | **CP** |

The profiled pair's mean decoded step changes only 8949.3 → 8928.5 us (**0.232% better**), and
the 96-step wall changes 3151.5 → 3144.1 ms. This single fixed-order profile is for mechanism, not
a replacement for the prior counterbalanced E2E result, which was **0.874% worse** at bs16 and
lost 0/6 pairs.

### Memory-system context

Nsight Systems sampled whole-GPU counters at 20 kHz. Because W13/W2 are device functions in one
megakernel, no tool can attach a hardware counter to only that inlined body; the in-MPK values below
weight every sample by the number of live workers of the named task. They are contextual estimates
and still contain overlapping heterogeneous traffic. The standalone numbers are direct Nsight
Compute physical-DRAM/L2 measurements of the v024 candidate entry.

| family/context | physical DRAM (TB/s) | L2 sector hit rate |
|---|---:|---:|
| W13 standalone v024 | **2.510** | **0.71%** |
| W13 MPK OFF, task-weighted | 1.988 | 2.04% |
| W13 MPK ON, task-weighted | **1.051** | **12.97%** |
| W2 standalone v024 | **1.854** | **0.80%** |
| W2 MPK OFF, task-weighted | 0.966 | 13.45% |
| W2 MPK ON, task-weighted | **0.823** | **24.92%** |

The alternate unweighted stage-active values lead to the same result: ON W13 is 0.978 TB/s at
15.59% L2 hit; ON W2 is 0.809 TB/s at 25.44% L2 hit. Standalone byte/time details were W13
108.585 MB / 43.264 us and W2 56.014 MB / 30.208 us. The earlier 4.72 TB/s loop number is a
request/useful-byte demand estimate; the table uses hardware-observed physical DRAM traffic.

The direction is unambiguous: production does not reproduce the standalone memory regime. This is
consistent with cross-worker L2/DRAM contention and request coalescing changes, but the sampled
whole-GPU counters do not prove contention as the only cause. The exact trace separately proves a
scheduler/packing contribution. The supported causal statement is therefore: **partial body-win
loss plus W13 packing/critical-path amplification**, not correctness and not a register/spill
regression.

## Part 2 — corrected compile-flag audit

| item | production JIT | old standalone harness | verified assessment / shipped fix |
|---|---|---|---|
| compiler | captured `/usr/local/cuda-12.8/bin/nvcc` 12.8.93 | unqualified `nvcc` → CUDA 13.2.51 | **Material mismatch.** All three builds now default to CUDA 12.8, overridable only with `MPK_NVCC`. |
| `-use_fast_math` | on unless `MPK_NO_FAST_MATH=1` | absent from `./kernel` (attention put it in `kernel_fm`; dense required an extra flag; MoE had no canonical build script) | **Material.** `./kernel` is now the shipped/scored fast-math lane; precise is correctness/numerics only. |
| `-lineinfo` | yes | absent | Benign for these kernels. Added after paired proof below. |
| output / TU | `-shared`, PIC `.so`, all task bodies in one `persistent_kernel` | executable with `main()` and a small TU | **Structural and material.** Cannot be fixed by a standalone flag; periodic full-MPK validation now gates promotion. |
| `-std` | auto-detect, captured `-std=c++20` | C++17 | User's “match” finding was incorrect. Fixed to C++20. |
| `--expt-relaxed-constexpr` | yes | dense yes; attention old script no; MoE had no canonical script | Not consistently matched. Now present in all three. |
| `-rdc` | `-rdc=false` without NVSHMEM | omitted | Semantically matched: nvcc's default is non-RDC. Explicit full-MPK validation covers NVSHMEM variants. |
| `-O3`, `-gencode` | `-O3`, `compute_100a/sm_100a` | same | Matched. |
| `MPK_TARGET_CC` | set to 100 | attention set it; MoE/dense standalone did not | No direct reference in the three optimized Blackwell task bodies, but it is used by `runtime_header.h`; body-neutral, not globally inert. |
| `MAX_WORKER_PER_SCHEDULER` | 3 in the captured 136/48 build | unset | No Blackwell task-body reference; it controls scheduler queue state in `persistent_kernel.cuh`. Structural context, not a body flag. |
| `MIRAGE_USE_CUTLASS_KERNEL` | 1 | unset | No Blackwell target-body reference; only the Ampere task header consumes it. Benign for these bodies. |
| `MODE_OFFLINE` | set | attention set it; MoE/dense did not | No reference in attention/MoE/dense optimized bodies. It reaches `gdn_conv1d_sm100.cuh` and persistent runtime/admission code, so it matters to the full TU but cannot select code inside these three bodies. |
| `-Xcompiler=-fPIC`, `-lstdc++fs` | yes | absent | Host/shared-object linkage; no device-body codegen lever observed. |
| production shape/TMA/backend macros | graph-specific, including `MPK_ENABLE_TMA`, backend and shape limits | partial or absent | Structural surrounding-TU differences; covered by exact-tag in-MPK compilation rather than copied wholesale into a standalone executable. |

### `-lineinfo` proof

For each immutable candidate and each math lane I compiled with and without only `-lineinfo`.
Every selected entry had identical registers, stack, spill stores, and spill loads. Normalized
`cuobjdump -sass` was byte-identical as well (the source `identifier` metadata line was removed):

| family | precise line/no-line SASS SHA-256 | fast line/no-line SASS SHA-256 |
|---|---|---|
| attention | `c8e0a422...e3e4a` | `3b00543b...a1d3d` |
| MoE | `3136945c...1fbd0` | `7e961e00...a6fa7` |
| dense | `f97aff34...7b870` | `608183a2...f00f3` |

### Two-lane measurement protocol

- Immutable sources: attention workspace6 v024 (`9c1a928c...e0d`), MoE workspace3 v024
  (`4d433c91...15c5a9`), dense workspace4 v022 (`09fd4c44...37b`).
- Compiler/non-math flags: CUDA 12.8.93, C++20, `-O3 -lineinfo
  --expt-relaxed-constexpr`, SM100a. Only `-use_fast_math` differs.
- GPU: exclusive physical GPU 7; three order-counterbalanced paired runs per lane. Values below are
  medians of the three runs. All stack/spill fields are ptxas bytes.
- Attention uses `KV_SPLITS_CFG=1`, the unsplit body actually integrated by the Stage-1
  device-only header. MoE and dense use their v024/v022 timed candidate entries.

### Registers, stack, and spills

| optimized entry | precise regs / stack / spill-st / spill-ld | fast regs / stack / spill-st / spill-ld |
|---|---:|---:|
| attention v024 unsplit timed | **236 / 0 / 0 / 0** | **232 / 0 / 0 / 0** |
| MoE W13 v024 timed | **75 / 0 / 0 / 0** | **75 / 0 / 0 / 0** |
| MoE W2 v024 timed | **64 / 0 / 0 / 0** | **64 / 0 / 0 / 0** |
| dense v022, maximum over all 30 entries | **43 / 0 / 0 / 0** | **43 / 0 / 0 / 0** |

Dense per-family maximum registers are identical between lanes: gdnqkv 40, gdnz 33, qkvg 40,
outproj 43, gateup 42, and down 42; every specialization has zero stack and zero spill.

The previously quoted attention “228 default / 244 fast-math” is not a like-for-like lane pair.
Under the old CUDA 13.2/C++17 build, 228 is the **precise split-candidate** entry, while 244 is the
**golden** entry in both lanes; the matching fast split candidate is 236, and the matching K=1
candidate is 238 in both lanes. Under the production-aligned compiler, the integrated K=1
candidate is the 236/232 pair reported above. Thus the original 16-register claim combined two
different functions.

### Attention lane latency

| config | precise (us) | shipped fast (us) | fast reduction |
|---|---:|---:|---:|
| bs1 | 20.896 | **17.408** | 16.692% |
| bs8 | 20.544 | **17.024** | 17.134% |
| bs16 | 20.736 | **17.312** | 16.512% |

### MoE lane latency

| entry | precise (us) | shipped fast (us) | fast reduction |
|---|---:|---:|---:|
| W13 bs1 | 8.064 | 8.064 | 0.000% |
| W13 bs2 | 10.208 | 10.208 | 0.000% |
| W13 bs4 | 15.616 | 15.552 | 0.410% |
| W13 bs8 | 30.464 | 30.512 | -0.158% |
| W13 bs16 | 41.440 | 41.392 | 0.116% |
| W2 bs1 | 5.248 | 5.248 | 0.000% |
| W2 bs2 | 6.272 | 6.272 | 0.000% |
| W2 bs4 | 8.768 | 8.768 | 0.000% |
| W2 bs8 | 14.272 | 14.272 | 0.000% |
| W2 bs16 | 22.496 | 22.464 | 0.142% |

These are the same-entry cross-lane audit, not the v012→v024 reduction used in Part 1.

### Dense lane latency

Each cell lists M1/M2/M4/M8/M16 in microseconds.

| family | precise | shipped fast |
|---|---|---|
| gdnqkv | 10.272 / 10.304 / 10.240 / 10.368 / 10.240 | 10.256 / 10.304 / 10.256 / 10.320 / 10.272 |
| gdnz | 8.224 / 8.256 / 8.224 / 8.224 / 9.184 | 8.240 / 8.224 / 8.256 / 8.192 / 10.128 |
| qkvg | 10.272 / 10.368 / 10.240 / 12.256 / 10.336 | 10.272 / 10.304 / 10.240 / 12.288 / 10.304 |
| outproj | 10.240 / 10.240 / 10.240 / 10.240 / 10.240 | 10.240 / 10.240 / 10.240 / 10.240 / 10.240 |
| gateup | 8.208 / 8.192 / 8.208 / 8.224 / 8.256 | 8.192 / 8.208 / 8.224 / 8.208 / 8.224 |
| down | 8.192 / 8.192 / 8.192 / 8.160 / 8.192 | 8.192 / 8.160 / 8.192 / 8.176 / 8.192 |

Most changes are one timer bin or zero. `gdnz_M16` is bimodal in both lanes (precise reps
8.256/9.184/10.160; fast 9.120/10.128/10.176), so its median difference is not evidence of a
fast-math regression.

## Part 3 — structural gap

The user's structural diagnosis is correct. Production compiles every device task into one
`persistent_kernel` with `__launch_bounds__(WORKER_NUM_THREADS, 1)` and admits one worker block per
SM. Ptxas assigns one register/stack budget to that single entry, determined by the full inlined
control-flow graph. Consequently:

- standalone CTA/SM occupancy transitions are not production constraints;
- a standalone task-body register increase below the full megakernel allocation need not change
  production allocation at all;
- the authoritative resource gate is full-TU `persistent_kernel <= 255` registers with no new
  spill, not a per-body delta from the standalone golden;
- standalone timing cannot model 136 heterogeneous persistent workers sharing L2/DRAM or the
  realized dependency/queue schedule.

The fix therefore has two levels: make the cheap standalone lane match production flags/toolchain,
and treat its performance as provisional until an exact tagged candidate passes a real-MPK
correctness plus WALL-SPAN comparison. The second level is now routine rather than convergence-only.

## Part 4 — shipped changes

### 1. Harness builds

| path | before | after |
|---|---|---|
| `ferret/workspace3/build.sh` | absent | new SHA `e0666650...0f36`; builds `./kernel` fast first and `./kernel_precise` second |
| `ferret/workspace4/build.sh` | SHA `34efa428...cbd` | SHA `cec0eb2f...3430`; default invocation is fast, `--precise` is the opt-out lane |
| `ferret/workspace6/build.sh` | SHA `553bbe59...3c33` | SHA `f36a5dae...3325`; `./kernel` changed from precise to fast, old `kernel_fm` role is replaced by `kernel_precise` |

All three now default to `/usr/local/cuda-12.8/bin/nvcc`, C++20, `-O3 -lineinfo
--expt-relaxed-constexpr -Xptxas -v`, SM100a. Dry-run command capture proves only `./kernel`
contains `-use_fast_math` and the precise lane does not. The immutable candidate compiles used for
the two-lane tables exercised those same commands with real nvcc. All scripts are executable and
pass `bash -n`.

### 2. Canonical task constraints and frozen snapshots

I inserted one `ALIGNMENT (HARD; ...)` block with all four required axes into each canonical task,
then synchronized the complete canonical file to its retired workspace snapshot.

| task | canonical / workspace snapshot | reconstructed pre-change SHA | final SHA of **both** files | workspace marker |
|---|---|---|---|---:|
| MoE | `tasks/moe-fp8-grouped-vllm-beat.yaml` / `workspace3/task.yaml` | `eae8a0a0...b0344` | `c6552d8b...17e6e` | line 436 |
| dense | `tasks/dense-fp8-blockscale.yaml` / `workspace4/task.yaml` | `26e2b8fd...4345b` | `2e11feff...784e` | line 370 |
| attention | `tasks/attention-sm100-vllm-beat.yaml` / `workspace6/task.yaml` | `ee00ab00...7b9e9` | `75ba0f6f...baed8` | line 666 |

The pre-change SHA is reconstructed by removing only the inserted block; for MoE/dense it also
matches `HEAD:task.yaml`. Attention already carried later uncommitted coordinator rulings before
this work, so its pre-change worktree SHA intentionally differs from historical HEAD. Final
canonical/snapshot pairs are byte-identical, all six parse as YAML, and every workspace copy
contains the marker. I also normalized the older two-lane rules in all six files so they no longer
call the precise lane “default”: `./kernel` is explicitly shipped/scored and
`./kernel_precise` is explicitly diagnostic. The block states:

1. fast math is the shipped/scored latency and tag lane; precise remains a correctness differential;
2. MPK has one worker block per SM, so no standalone CTA/SM occupancy reasoning;
3. the full megakernel's shared 255-register/no-spill budget is authoritative;
4. standalone wins require periodic and final in-MPK profiler WALL-SPAN confirmation because they
   omit heterogeneous L2/DRAM contention.

### 3. Routine exact-tag in-MPK validation

Changed:

- `ferret/.claude/agents/mpk-validator.md`
- `/home/muhengl/.codex/agents/ferret-mpk-validator.toml`
- `ferret/.claude/agents/kernel-extractor.md`
- `/home/muhengl/.codex/agents/ferret-kernel-extractor.toml`
- `ferret/CLAUDE.md` (`ferret/AGENTS.md` is a symlink to this same source)
- `ferret/scripts/mpk_validate.sh`
- the Codex runner contract in `ferret/scripts/cc-run.sh`

The local and remote build templates in `ferret/CLAUDE.md` now also use the production-aligned
CUDA 12.8/C++20/fast-math/lineinfo command. This closes a final active path that still hardcoded an
unqualified `nvcc -std=c++17` build after the workspace build scripts had been fixed.

The executable workflow is now:

`immutable v### tag → CHECKPOINT extractor → tag-scoped /var/tmp kernel.cuh → MPK validator →
progress.md verdict → FINAL extractor only after integration-ready PASS`

Cadence is the first accepted tag in OPTIMIZE, every fifth score-improving tag since the preceding
checkpoint, and always the exact selected tag immediately before it is called integration-ready or
delivered. Correctness PASS alone is insufficient: a profiler task WALL-SPAN and a favorable
same-context baseline ratio are required. A flat/regressing standalone win is retained only as
historical `MPK_REJECTED` evidence.

The extractor previously could only consume the mutable worktree at convergence. It now has
CHECKPOINT and FINAL modes, requires `SOURCE_TAG`, reads `SOURCE_TAG:kernel.cu`, writes provenance
(tag plus commit), and places periodic output at
`/var/tmp/ferret-mpk-checkpoints/workspaceN/<tag>/kernel.cuh`. The validator rejects missing or
mismatched provenance. `mpk_validate.sh` now accepts `--candidate-cuh` and pins CUDA 12.8 for both
its full-scheduler and extension paths while retaining automatic exclusive-GPU selection and
self-reversion.

The Markdown and Codex TOML developer bodies are exactly identical, not merely similar:

- mpk-validator: 17,151 characters, body-identical; TOML parse PASS;
- kernel-extractor: 11,216 characters, body-identical; TOML parse PASS.

Current file SHA-256 values are `8ac8e11d...81ec` / `87a22e72...c3a` for the validator Markdown /
TOML and `f65d66b3...057a` / `18f8544e...3c5` for the extractor Markdown / TOML.

### 4. Analysis artifacts

The corrected 136-worker measurement wrappers and all raw evidence are under
`/var/tmp/alignment`. I did not edit the concurrently active `mirage-combined-v024` analyzer files
while `/var/tmp/combined` was running. The stale 128-worker defaults should be generalized in that
tree after the concurrent run lands; this report's numbers already use the corrected model and
carry the required identity checks.

## Verification summary

- Part 1: both profiler arms completed on exclusive GPU 3; token identity PASS; graph anchor QC
  PASS; six exact schedule assignments PASS with zero-nanosecond identity error.
- Flag lanes: 12 real production-aligned builds (three families × two math lanes × lineinfo on/off)
  completed; six paired SASS hashes prove lineinfo neutrality.
- Latency: three counterbalanced paired reps per family/lane completed on exclusive GPU 7.
- YAML: all six files parse; canonical/workspace hashes match exactly; all three frozen snapshots
  contain `ALIGNMENT (HARD)`.
- Shell: `bash -n` passes for `cc-run.sh`, `mpk_validate.sh`, and all three build scripts; emitted
  command dry-runs show the expected default/precise lane split.
- TOML: both installed Codex agents parse; each developer body is byte-identical to its Markdown
  counterpart.
- No `.pm/` path was created or modified. No GPU 0 work was run. The concurrent combined-tree
  process was observed and left untouched.

## Evidence map

- Raw OFF/ON profiler captures: `/var/tmp/alignment/part1/{off,on}/prof/`
- Corrected width/CP/schedule outputs: `/var/tmp/alignment/part1/{off,on}/*136*.{json,log}`
- Exact iterations: `/var/tmp/alignment/part1/{off,on}/gap136_it{100,200,300}.{json,log}`
- Whole-GPU counter correlation: `/var/tmp/alignment/part1/{off,on}/counters20k.json`
- Lane sources/build logs/run logs: `/var/tmp/alignment/lane_audit/{attention,moe,dense}/`
- Recomputable lane summary: `/var/tmp/alignment/lane_audit/summarize.py`
- Capture/analysis drivers: `/var/tmp/alignment/capture_part1.sh`,
  `/var/tmp/alignment/derive_part1_realized.sh`, `/var/tmp/alignment/check_exact_windows.sh`,
  `/var/tmp/alignment/correlate_counters.py`
