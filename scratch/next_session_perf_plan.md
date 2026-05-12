# Perf work — state at end of 2026-05-12 autonomous session

## What landed today

| Commit       | Title                                                                       | Why                                                                              |
|--------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `f44d02db`   | dpskv3 builder: document AR tile-size sweep finding in docstring            | Captures negative result so we don't re-sweep                                   |
| `af38cf42`   | dpskv3 builder: add MPK_FP8_DENSE_NUM_WORKERS env override                   | Enables the +6.3% decode win as opt-in                                          |
| `e243272c`   | dpskv3 builder: keep _fp8_dense_kv_b_proj at full num_workers (skip env)    | Avoid the runtime_m_mode=1 crash for chunked-prefill kv_b                       |
| `d6d1730a`   | allreduce: opt-in debug gates MPK_AR_SKIP_BARRIER / MPK_AR_SKIP_REDUCE      | Infrastructure to isolate AR barrier vs reduce cost                             |
| `157b9214`   | perf journal: AR phase-isolation data + A2 static-analysis hypotheses       | Records the 91 μs barrier / 221 μs reduce / 58 μs other decomposition           |
| `1ed0ea6a`   | perf: AR rewrite design doc with three options + recommendation             | `scratch/ar_rewrite_design.md`                                                  |
| `543bdc21`   | perf: draft Option A per-task barrier (out-of-build, for review)            | `scratch/ar_option_a_per_task_barrier.cuh` (180 lines, ready to wire in)        |

## Highest-value next step (staged, awaiting live validation)

**Option A (per-task barrier slots)** is committed at `3a9588cf` —
`allreduce: gated opt-in per-task barrier (MPK_AR_PER_TASK_BARRIER=1)`.
Default OFF; enable via `MPK_AR_PER_TASK_BARRIER=1` to pass
`-DMPK_AR_PER_TASK_BARRIER` to runtime nvcc.

The code is **unverified on a live GPU** — review and run validation when
GPU time is available:

1. Force the Cython relink path (only needed if you also touched .cc/.h):
   `rm python/mirage/core.*.so && touch python/mirage/_cython/*.pyx`
2. Smoke Qwen3 (TP=4, layers 0-3, mbt=1) with `MPK_AR_PER_TASK_BARRIER=1`
   → must produce same output token IDs as without the env.
3. DSv3 prefill-128 layers 0-3 with the env → per-layer cosine vs FP8 ref ≥ 0.99.
4. DSv3 prefill-128 layers 0-19 prefill+1-decode, with and without env →
   record per-token latency and AR per-task wallclock.
   - Expected with env: e2e drops toward 165-170 ms (vs 182.72 ms baseline,
     vs 164.25 ms theoretical lower-bound from `MPK_AR_SKIP_BARRIER`).
   - Expected per-task AR wallclock: 280-310 μs (down from 370 μs).

Decision tree after step 4:
- If e2e drops by ≥ 12 ms with the env: Option A succeeded; consider making
  it the default and close A3. Update `scratch/perf_optimization_journal.md`.
- If e2e drops by 3-5 ms: contention is only part of the cost; revisit
  reduce path (and skip Option B from `scratch/ar_rewrite_design.md`).
- If correctness fails on Qwen3 or DSv3: REVERT commit `3a9588cf`.
  The fault is most likely (i) the assumption that NVSHMEM zero-initializes
  the per-team psync region beyond 2*SYNC_SIZE, (ii) a P2P-write ordering
  edge case in the phase-parity slot reuse, or (iii) a stationarity
  exception I missed in the audit.

Implementation review notes:
- `mpkar_sync_block_per_task` sits at `allreduce.cuh:357-455`; layout
  documented in the header comment + `scratch/ar_option_a_per_task_barrier.cuh`.
- `MPK_AR_PER_TASK_BARRIER` env→-D propagation at `persistent_kernel.py:280-286`.
- AR kernel dispatch at `allreduce.cuh:660-672`: nested `#ifndef
  MPK_AR_SKIP_BARRIER` / `#if defined(MPK_AR_PER_TASK_BARRIER)`. If both env
  vars are set, `MPK_AR_SKIP_BARRIER` wins (no barrier at all).

## Other open work (in roughly decreasing impact order)

### A2 — fp8_gemm_dense_smallm crash at low num_workers (`Task #129`)
Static analysis shows no smoking gun. Need live bisect:
- Variant: prefill kv_b path (`runtime_m_mode=1`).
- Sweep `MPK_FP8_DENSE_NUM_WORKERS=N` for N ∈ {32, 40, 48, 56, 64, 72, 96, 128}.
  (Need to plumb a SEPARATE env override that only affects `_fp8_dense_kv_b_proj`,
  since the current global override is bypassed there.)
- Capture exact threshold and cuda-gdb backtrace at the failing run.

### A5 — TopK_sigmoid optimization (BUT first: investigate possible correctness gap)
While reading `include/mirage/persistent_kernel/tasks/blackwell/topk_sigmoid_sm100.cuh`
I noticed the kernel processes **at most 8 rows per CTA** (ROWS_PER_WARP=1, 8
warps/CTA). The builder calls it with `grid_dim=(1, 1, 1)`. For `mbt > 8`
(prefill), rows 8..mbt-1 should be unrouted (routing_indices left at zero from
Phase 0 init).

The demo runs DSv3 prefill-128 without crash, so EITHER:
- The MoE w13/w2 kernels handle "no routing" gracefully (zero output), and the
  prefill quality is just degraded silently, OR
- I'm misreading the kernel and there's an outer loop or repeated invocation
  I missed.

**Action**: trace a prefill-128 run with `printf` of `mpk_routing_indices` for
row 50 (somewhere mid-batch) to confirm whether routing is computed. If not,
this is a real prefill correctness bug to flag to the maintainer.

If correctness is fine, then A5 optimization can proceed: simplify the 3-level
nested group/expert/argmax loop in phases 3 and 5.

### A6 — Router BF16 splitk-linear at small N
Spec: `(mbt × hidden=7168) × (hidden=7168, 256)`. Currently 8.2 μs vs vLLM 3 μs.
Needs standalone bench at that exact shape to confirm kernel-vs-calling cost split.

### LM head — confirm calling overhead vs kernel time
LM head `(mbt × 7168) × (7168, 129280)` registers 505 linear_layer tasks (grid =
129280 // 256). At 128 workers, 4 waves; observed 75 μs/call in MPK. Standalone
bench needed at this shape (mbt=128 prefill, mbt=1 decode) to confirm whether
the 75 μs is dominated by dispatch+dep-spin (calling problem) or per-tile work
(kernel problem). Existing standalone test at
`tests/runtime_python/blackwell/sm100_linear/test_matmul_mpk.py` is hardcoded
to tiny shape (1×128×768); would need to rebuild the wrapper with LM-head
template params or use the looser linear_sm100 path.

### A3 — Option B (single barrier task per AR phase)
Only pursue if Option A delivers < 50% of projected savings. Multi-file
structural change (~10 files); design captured in
`scratch/ar_rewrite_design.md`.

### B1 — Q/KV phase 27% SM utilization
Root cause already documented (persistent FP8 dense kernels create
num_triggers=128 events → coarse dep chains). Fix candidates:
- B. Switch FP8 dense small/medium to non-persistent kernel for small batch
  (medium effort, no scheduler changes)
- C. Reduce number of Q/KV kernel calls by fusing (medium effort)

### B2 — EVENT_LAUNCH_TASKS path
Path 2 in journal. Hangs on both Qwen3 and DSv3 selective-layer. Needs
cuda-gdb session to diagnose the scheduler/worker handshake race. Don't
attempt without GPU + interactive debug time.

### B3 — DTensor view/slice API
Unblocks Path 1 (fuse kv_a + kv_rope into one GEMM). Needs MPK Python API
extension; design touches DTensor + KNGraph runtime. Moderate scope.

### B4, B5 — runtime scheduler changes
Deferred; high implementation risk.

## Useful artifacts

- `scratch/perf_optimization_journal.md` — running tracker; updated each session
- `scratch/ar_rewrite_design.md` — AR design with 3 options + recommendation
- `scratch/ar_option_a_per_task_barrier.cuh` — drop-in code for Option A
- `scratch/mpk_vs_vllm_perf_comparison.md` — per-kernel comparison vs vLLM/SGLang
- `outputs/ar_bench_112933/` — phase-isolation run logs (baseline / skip_barrier
  / skip_reduce)

## Memory of key facts for the next session

- vLLM target: 143 μs/layer (we're at 354 μs/layer; gap 2.5×)
- AR per-phase wall time: 370 μs (vLLM 6-8 μs)
- B200 has 148 SMs, MPK uses 128 workers max
- DSv3 hidden = 7168, padded vocab = 129280
- mbt > 8 triggers `_use_prefill` mode (chunked prefill + dual-dispatch)
- Phase-isolation gates `MPK_AR_SKIP_BARRIER` and `MPK_AR_SKIP_REDUCE`
  produce WRONG OUTPUT — never enable in correctness runs
