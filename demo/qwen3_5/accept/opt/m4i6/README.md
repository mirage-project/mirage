# M4-I6 — the MoE router (task 260) takes the ferret v013 winner

The router is the most serialized stage MPK runs: one task per layer per decode
step, `grid=(1,1,1)`, one CTA of 256 threads, nothing to overlap it against.
M4-I5's critical-path decomposition put it at **842.1 µs of the 7957.5 µs bs1
path (10.58 %)** at 21.053 µs/task with live/lvl = 1.0, and said it must reach
**3.697 µs/task** to hold up its end of the five-stage parity scenario.

A ferret loop on `ferret/workspace5` drove a standalone replacement to
**min_ratio 1.417** against the FlashInfer/TRT-LLM routing kernel vLLM calls —
141.7 / 150.5 / 145.6 / 170.7 / 170.3 % of its throughput at N_LIVE 1/2/4/8/16,
i.e. 29.4–41.4 % less time per call. This issue integrated that kernel.

## Result

| | bs1 | bs2 | bs4 | bs8 | bs16 |
|---|---|---|---|---|---|
| e2e wall, base → new (ms) | 1094.6 → 1049.7 | 1266.7 → 1217.5 | 1637.7 → 1559.3 | 2498.1 → 2407.2 | 3807.0 → 3688.5 |
| e2e speedup | **+4.27 %** | **+4.04 %** | **+5.03 %** | **+3.78 %** | **+3.21 %** |
| router µs/task, before → after | 20.49 → 9.27 | — | — | 24.89 → 13.48 | 23.89 → 12.47 |
| router stage wallspan (µs) | 819.4 → 370.8 | — | — | 995.7 → 539.0 | 883.8 → 461.3 |
| per-task speedup | 2.21× | — | — | 1.85× | 1.92× |

AC-3 **STABLE** at all five batch sizes, **bit-exact 10/10 per bs** against the
committed `results/dumps_final`. Tokens identical in 15/15 A/B pairs.

**How much of the 842 µs came back: 445–446 µs, 53 %.** Re-derived at bs1 with
M4-I5's own `cp_decompose.py` on both arms, twice:

| | rep0 | rep2 |
|---|---|---|
| critical path, base → new (µs) | 7905.4 → 7550.7 | 7901.0 → 7548.8 |
| router on the path (µs) | 840.6 → 395.3 | 846.8 → 400.7 |
| router share of the path | 10.63 % → 5.24 % | 10.72 % → 5.31 % |
| **recovered** | **445.3 µs (52.9 %)** | **446.1 µs (53.0 %)** |
| tax on the other stages | +90.9 µs | +93.7 µs |
| net Δcp | −354.7 µs | −352.2 µs |

Arm A's re-derived basis lands within 0.2 % of M4-I5's recorded 842.1 µs and
0.7 % of its 7957.5 µs path, so the two decompositions are measuring the same
thing and the recovery is directly comparable.

The stage is **not** at the floor M4-I5 named: 9.27 µs/task against a 3.697 µs
vLLM per-call number is 2.51× it (2.93× at bs8, 2.09× at bs16). The target is
not met.

## The cost, which is real and reproducible

`persistent_kernel` inlines every task body, so ptxas allocates one register
budget for all of them.

| | registers | stack frame | spill st/ld | smem |
|---|---|---|---|---|
| base | 238 | 144 B | 0 / 0 | 5824 B |
| new | **255** | 112 B | **4 B / 4 B** | 5856 B |

255 is the ceiling at `__launch_bounds__(256,1)` — 256 threads against a 64 K
register file — so ptxas took everything available and still spilled one 32-bit
value. `worker_kernel` moves the same way (236 → 255). Occupancy is unchanged
(MPK runs 1 CTA of 256 threads per SM by design), and smem grows by exactly the
32-byte active-expert bitmask, inside the 6 KiB static reserve.

That budget change is measurable in the graph. Stage by stage on the bs1 critical
path, arm B minus arm A, across two independent reps:

```
TASK_MOE_W13_FP8_BLOCKSCALE_SM100        +53.6    +53.2
TASK_MOE_W2_FP8_BLOCKSCALE_SM100         +14.4    +15.2
TASK_MOE_TOPK_SOFTMAX_SM100             -445.3   -446.1
TASK_ATTN_SM100                          +14.9    +16.1
TASK_MOE_MUL_SUM_ADD_SM100                +6.0     +6.1
TASK_LINEAR_FP8_BLOCKSCALE_SM100          +3.1     +2.6
  ... everything else within ±2 ...
TOTAL (= delta cp)                      -354.4   -352.4
```

Path-task counts are identical stage for stage between arms, so the chain did not
re-route — these are per-task durations. And the pattern **reproduces to within
~1 µs**, which profiled variance does not do. It is largest on the largest,
most register-hungry stages and near zero on the small elementwise ones: the
signature of the shared budget, not noise.

**So ~21 % of the recovery is given back.** Net at bs1: −353 µs of critical path,
−4.27 % end to end. Worth taking, and worth knowing: the tree now starts at the
register ceiling with a spill, so the next register-hungry integration inherits
that and must read its own ptxas delta against this baseline, not the
pre-M4-I6 one.

## What was integrated, and what was verified

Tag **v013** (`f370cbb`, "boundary-warp converged padding compute"), taken as the
**tag blob** (`git show v013:kernel.cu`) while the loop was still running on
workspace5 — never the worktree file, which carries unfinished probe code
(the M4-I2 lesson). The loop later reached a best-effort FINALIZE at the same tag,
so v013 is the definitive winner, and its own extractor's deliverable
(`workspace5/kernel.cuh`) is **code-identical** to what was imported here
(comments stripped, whitespace normalized) — two independent extractions agreeing.

Provenance was checked before anything was imported: v013's frozen `golden` block
is **byte-identical** to the 413-line `topk_softmax_task_impl` it replaces.

The five invariants the brief named, and how each was verified:

1. **M3-I5b's row-tile loop, arbitrary `num_rows`.** Preserved verbatim. Verified
   by `test_router_oracle.py` — "coverage: 1/7/9/16/17/33 rows all routed at
   VPT=8 and VPT=16" — and by `test_gate_topk.py` passing at bs 17 and 33, both
   past one pass. A reintroduced 16-row cap silently zeroes routing for surplus
   rows, so this is checked by row coverage, not by a crash.
2. **M3-I5c's barrier-separated compaction — no atomicAdd, strictly ascending.**
   Replaced by a warp-0 popcount over a shared-memory active-expert bitmask,
   which keeps both properties: rank is an exclusive prefix count evaluated in
   ascending expert order, no atomics on the count, no schedule dependence. Since
   the compaction changed, the ferret harness's ≥200-run same-input replay stress
   applies (v006/v002 comment), and the in-repo checks are `test_gate_topk`'s
   position-exact mask comparison at both expert counts plus AC-3's
   fingerprint-consistency requirement (3 reps identical key-for-key per bs). The
   contract M3-I5c earned — and *why* the atomicAdd version was wrong — is now
   written at the top of the block so a future rewrite owes the same evidence.
3. **M3-I8's `gate_padding_rows` / `num_active_rows` semantics.** `live_rows` is
   computed identically and marks stay gated by it. **Extended**: padding rows now
   also skip the load and the compute chain, and write deterministic zeros to
   their top-k weight slots — see below.
4. **The shared template (Qwen3-30B-A3B, 128 experts).** No specialized fast path
   was needed: at 128 experts `task_register.cc` picks VPT=8/THREADS_PER_ROW=16,
   which `if constexpr (VPT == 16 && THREADS_PER_ROW == 16)` excludes from the
   v009 sorted-lane path, so that instantiation takes the serial fallback with the
   proven packed-key argmax. `test_gate_topk.py` sweeps `NUM_EXPERTS_LIST =
   [128, 256]` at every batch size and passes at both.
5. **Register budget.** Measured, reported above, and the finding is that it *did*
   cost something.

### The one place bytes move

Rows at or above `num_active_rows` write **zeros** to their top-k weight slots
instead of the softmax of their residue logits. Those weights are read —
`mul_sum_add_sm100.cuh:34-45` loops `row_idx` over the whole `BATCH_SIZE` with no
liveness gate — but only into padding rows of `moe_out`. `topk_w` has exactly one
consumer in the graph (`builder.py:880`), its output is per-row, and every
downstream decode stage is per-row or per-request; M2's AC-3 established live-row
independence from padding rows empirically. Byte-identical dumps at all five bs,
plus 15/15 identical `tokens_sha256` in the A/B, is that argument confirmed on the
real checkpoint through 40 layers rather than assumed.

`live_rows == num_rows` whenever `num_active_rows` is −1, so every caller that
does not opt into M3-I8 gating — test mode, the single-layer harnesses, the
128-expert instantiation — keeps the old bytes exactly.

## Two integration defects the gates caught

**The smem storage class.** The bitmask was first taken from the megakernel's
dynamic arena, following the convention `gdn_recurrent_sm100.cuh:635-637`
records. All 16 gate-1 cells died with `cudaErrorIllegalAddress`: this task has
three standalone launchers that pass **zero** dynamic shared memory
(`sm100_moe/runtime_kernel_wrapper_sm100.cu:101` is `<<<grid, block, 0>>>`). The
arena is right for a task that wants kilobytes and wrong for 32 bytes whose only
enforcement is an illegal access. Reverted to static `__shared__` — which is also
what ferret's own extractor chose, flagging the per-worker smem budget as the
thing to check, which gate 2 then did.

**A negative probe, recorded so it is not retried.** Reversing the sorted key
array in place and dropping the second `desc[VPT]` array — a permutation of the
same unique keys, all-static indices, provably value-identical — changed nothing:
still 255 registers, still a 4-byte spill. ptxas was already coalescing
`desc[i] = kk[VPT-1-i]` into register renaming, so the second array never existed
in the allocation. The pressure is the 16 live u64 keys themselves, which *are*
the v009 win. Reverted; the imported body is byte-faithful to v013.

## Caveats

- The **profiled** `wall_ms` is not a performance measure: rep0 read
  1118.0 (A) vs 1082.5 (B) and rep2 read 1117.5 vs 1114.1, while both reps'
  `step_us` agree to 7 µs. Profiling overhead and event dumping dominate the
  wave wall. The perf claim is the `--no-profiler` A/B.
- **bs16's profiled window covers 37 router tasks, not 40.** Per-task latency is
  the normalized statistic and the recovery is computed as `Δ(µs/task) × 40 path
  tasks`; absolute window work is reported as mechanism evidence, not measurement.
- **One dirty e2e rep**: `B_bs1_rep0` started with 2656 MiB of foreign residue.
  It read 1049.7 ms between its two clean siblings' 1046.4 and 1051.9; dropping it
  gives 1049.12 ms and 1.0433× instead of 1.0427×. Residue would slow arm B, so
  the direction is conservative.
- **One lost profiled rep**: `rep1` arm B was contaminated mid-run — cp read
  35520.6 µs with `TASK_RMS_NORM_HOPPER` alone +26344 µs and the router showing
  *no* speedup, i.e. the whole kernel stalled. Excluded and named, not deleted;
  rep2 was run in its place and reproduces rep0 to within 1 µs.
- **Two AC-3 reps started on a non-clean device** (`bs4_r3` 2284 MiB, `bs16_r3`
  3646 MiB). Both accepted with identical tokens. AC-3 scores arithmetic, not
  latency, so a co-tenant footprint cannot move its verdict.
- The `bs4` arm-B e2e range is 54.0 ms (rep1 1598.2 against 1544.3 and 1559.3) on
  a clean device — run-to-run variance. The median is reported and the worst
  reading still beats every arm-A rep at that bs by 28.7 ms.
- Two ferret chains were live on the box throughout. `ferret/pick_gpu.sh` does not
  read our locks — it takes the lowest-memory GPU not already pinned by a running
  `cc-run`, in index order — which is why the guard's candidate list starts from
  the high end and why every run's `gpu_before` is audited from its own record.

## Layout

```
scripts/    setup_m4i6.sh          isolated clone at the integration commit + fresh extension build
            setup_m4i6_base.sh     the same for the parent commit (arm A)
            gpu_guard_m4i6.sh      3-sample idle guard, fails closed
            gate1_m4i6.sh          gate 1: the four router instruments, per-bs invocations
            mk_ptxas_m4i6.sh       gate 2: ptxas -v on each arm's generated megakernel TU
            gate_ac3_m4i6.sh       gate 3: full AC-3 at five bs
            sweep_router.sh        gate 4: interleaved two-clone e2e A/B
            tables_m4i6.py         the e2e tables + the per-run gpu_before audit
            stage_wallspan.sh      profiled runs -> concurrency.py, per arm and rep
            stage_tables.py        the router stage before/after vs M4-I5's basis
            critpath_m4i6.sh       re-runs M4-I5's cp_decompose.py on both arms
            cp_compare.py          the per-stage path delta across reps (the tax)
            update_backlog_m4i6.py closes the backlog lever with the measured result
tables/     m4i6_tables.txt/json, ab_per_rep.csv, stage_wallspan.txt/json,
            cp_compare_bs1.txt/json
gates/      gate1/ gate2/ gate3/ gate4/ gate5_stage/
```
