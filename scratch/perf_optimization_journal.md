# MPK Perf Optimization Journal — 长期维护

Started 2026-05-12. 每个优化点记录: 做了什么, 测试方法, 结果, 下一步.

## 进行中

### AllReduce kernel rewrite (A3) — Option A reverted, Options B/C still open
- **Status**: A3 Option A (per-task barrier) reverted 2026-05-12. See 「已完成」 entry. Options B/C still TBD.
- **Goal**: vLLM/SGLang style — 1 global barrier instead of 56 per-task barriers; finer-grained MLA→AR task event chains
- **Baseline measurements** (DSv3 prefill-128 TP=4 EP=2 layers 0-19, 2026-05-12 d6d1730a):
  - baseline (full AR):  370.4 μs/task, e2e 182.72 ms
  - skip_barrier:        279.0 μs/task, e2e 164.25 ms (-18.5 ms vs baseline, -10.1%)
  - skip_reduce:         149.5 μs/task, e2e 168.97 ms (-13.75 ms)
  - Decomposition: barrier=91μs/task, reduce=221μs/task, other=58μs
  - **Updated understanding (post-A3)**: barrier cost is NVLink dissemination latency (P2P signal+wait, ~91 μs for TP=4 full-radix). NOT contention as originally hypothesized — each AR task already uses its own `nvshmem_team_t` with private psync.
- **Open ideas:**
  - Option B (`scratch/ar_rewrite_design.md`): SINGLE barrier task per AR phase + 56 reduce-only tasks. Recovers 56× barrier latency but needs builder-side event-chain restructure.
  - Option C: drop explicit cross-PE barrier in favor of NVLink fence + signal-only on a small number of slots. Higher correctness risk.
- **AR_SKIP debug gates committed (opt-in)** to allreduce.cuh + persistent_kernel.py:
  - `MPK_AR_SKIP_BARRIER=1` and `MPK_AR_SKIP_REDUCE=1` env vars propagate as -D macros for measurement only.

### A2 ablation — fp8_dense_smallm low num_workers crash
- **Status**: Static code analysis done, no code-level smoking gun. Live bisect pending GPU time.
- **Goal**: Find code-level root cause of CUDA "unspecified launch failure" at num_workers<56 (decode) and <64 (prefill)
- **Method**: Bisect with cuda-gdb / printf instrumentation, vary num_workers in steps
- **Static-analysis findings (2026-05-12)**:
  - Kernel `fp8_gemm_dense_sm100_common.cuh::task_impl_tpl` is structurally fine for any `num_workers`: worker_idx + num_workers correctly parametrize tile striding (`int bidx = iter * num_workers + worker_idx; if (bidx >= total) break;`).
  - Template params `<BN=128, NS=3>` are hardcoded at registration (`task_register.cc:5501-5503`), so num_workers doesn't change generated code.
  - tcgen05 alloc (warp 2) + dealloc (warp 0) are in different warps — unusual but matches the kernel's documented role assignment (`common.cuh:26-31`); not the same restriction as CUTLASS/CuTe which the CLAUDE.md note covers.
  - The runtime_m_mode=1 (chunked-prefill kv_b path) has BIGGER `runtime_m_` → more iterations per worker when num_workers is low — but that should slow down, not crash.
- **Working hypotheses for the live bisect**:
  1. **Task graph scheduling**: with grid=(num_workers, 1, 1) and num_workers<global_workers (typically 128), some workers run other task types instead. Those tasks might race against in-flight dense-GEMM TMA/tcgen05 state in unintended ways.
  2. **Resource starvation**: if dense_smallm holds TMA descriptors / TMEM and a co-scheduled task on another worker also needs them, allocation can fail asymmetrically.
  3. **runtime_m_mode=1 only**: the dynamic M = `(lp_-fp_-1)*MPK_PAGE_SIZE + last_page_len` might compute negative/zero when fp_ or lp_ are uninitialized for the gated decode-step early-exit path — but the comment at task_register.cc:5491 already guards `if (req_id_ < 0) return`. Need to verify all kv_indptr fields are populated before the prefill chunk is dispatched.
- **Next live experiment**: run mbt=128 prefill with `MPK_FP8_DENSE_NUM_WORKERS_KV_B=N` for N in {32, 40, 48, 56, 64, 72, 96, 128} (need to plumb a separate env override for the kv_b call), capture exact failure threshold.

### LM head 75μs calling-overhead investigation
- **Status**: Planning
- **Goal**: Why does linear_sm100 at shape [batch=1, in=4096, out=32320] take 75μs/call in MPK vs presumably much faster standalone?
- **Method**: Run standalone linear_sm100 bench at LM-head shape, compare to MPK trace

### MLA decode TP4 calling-overhead investigation
- **Status**: Planning
- **Goal**: Per-call MPK 22μs vs (presumably) much faster standalone. Find what calling pattern adds overhead.
- **Method**: standalone bench mla_mtp_decode_tp4_sm100 at decode shape, compare

### SPLITK FP8 swapAB calling-overhead investigation  
- **Status**: Re-check needed — my earlier comparison may have been per-task vs per-CTA mismatch
- **Earlier measure**: MPK avg 21μs/task vs standalone 19μs per-CTA → actually only ~10% overhead. May NOT be a calling issue. Re-verify.

### MoE TopK_sigmoid (A5) — direct kernel optimization
- **Status**: Planning
- **Goal**: 13.7μs → vLLM 7μs (2× faster)
- **Method**: Read `topk_sigmoid_sm100.cuh`, identify redundancy (group-aware top-K has 3 nested levels), simplify

### Router BF16 GEMV (A6) — alternate kernel for small-N
- **Status**: Planning
- **Goal**: 8.2μs → vLLM 3μs (2.7× faster) for hidden→256 router gate matmul
- **Method**: Try tcgen05-free GEMV path or switch to splitk_linear with different config

### B1: Q/KV phase 27% SM util — deeper investigation
- **Status**: Pending after A3
- **Goal**: figure out why workers idle 73% of phase time; reduce via dependency restructuring or fine-grained events

### B2: Fix EVENT_LAUNCH_TASKS scheduler hang
- **Status**: Pending  
- **Goal**: re-enable fine-grained event-driven launch path (current code force-downgrades to EVENT_EMPTY)
- **Earlier finding**: Qwen3 also hangs (not just DSv3 selective-layer) → scheduler/worker race bug

### B3: DTensor view/slice API support
- **Status**: Pending
- **Goal**: enable Path 1 (kv_a+kv_rope fusion) and similar by letting builder create sliced views of output tensors
- **Scope**: add `tensor.view(offset, shape)` or similar to MPK Python API; runtime tracks alias

### B4: Worker queue priority/skip-blocked
- **Status**: Pending
- **Goal**: change execute_worker so workers can pull NEXT ready task instead of FIFO-blocking on head-of-queue task whose dep_event hasn't fired
- **Risk**: large runtime change; need extensive testing

### B5: Look-ahead pre-fetch (next layer's first task)
- **Status**: Pending
- **Goal**: schedule next layer's RMSnorm/q_a tasks before current layer's AR2 finishes (fills 47μs inter-layer gap)
- **Method**: builder restructure of dep events

## 已完成

### `f44d02db` (2026-05-11) — AR tile-size sweep doc
- AR tile size 128/512/1024 swept on prefill; all within noise (per-task duration barrier-bound, not bandwidth-bound)
- Default unchanged. Just docstring.

### `af38cf42` (2026-05-12) — `MPK_FP8_DENSE_NUM_WORKERS` env override
- Lower num_workers (env=64) → decode trace span 7.03ms → 6.59ms (-6.3%) ← real win
- Default unchanged (128). Opt-in via env.
- 4 callsites in `_fp8_linear_v2` & `_fp8_dense_kv_b_proj`.

### `e243272c` (2026-05-12) — keep `_fp8_dense_kv_b_proj` at full 128
- Chunked-prefill kv_b_k/v call has tighter constraint; reverted that one site to self.num_workers
- env=64 still applies to 4 decode-path sites

### Path 3 (declare partition on persistent kernel) — closed-not-viable
- Would race: persistent kernel uses strided pattern but consumer reads all of output
- Discovered semantic constraint preventing this approach

### Path 2 (EVENT_LAUNCH_TASKS re-enable) — found bug, not yet fixed
- Added `MPK_TRY_FINEGRAINED_LAUNCH=1` env gate. Qwen3 hangs after [MPK INIT] (not just DSv3 selective-layer as comment claimed)
- Reverted runtime.cc, build clean
- Indicates a deeper scheduler/worker event-handshake race bug; needs cuda-gdb debugging

### A3 Option A (AR per-task barrier `mpkar_sync_block_per_task`) — REVERTED
- Commit `3a9588cf` (2026-05-12 staged), reverted `18dc6a4b` (2026-05-12). Gated `-DMPK_AR_PER_TASK_BARRIER`.
- **Premise was wrong**: original journal claimed 56-way contention on a shared psync slot. But `nvshmem_tile_allreduce_impl` already uses `teams[task_offset]` (line 657) — each AR task gets its own NVSHMEM team and thus its own psync region. So the OLD `mpkar_sync_block(team)` already has private psync per task. The new per-task code indexes by `task_offset` within each (already-private) team's psync — semantically equivalent.
- **Correctness validation (DSv3 TP=4 EP=2 prompt=128 layers 0-3)**:
  - Baseline (env=0) run1 vs run2: bit-equal (cos=1.0000, mad=0)
  - Env=1 run1 vs run2: bit-equal (cos=1.0000, mad=0) — fully deterministic
  - Baseline vs env=1, rows 1..127: bit-equal at layer 0-1; layer 2-3 still match. **Row 0 is decode-overwrite dump artifact** (see [[feedback-row0-dump-artifact]] memory); ignoring row 0, output is bit-identical.
  - Qwen3 TP=4 layers 0-3 8 new tokens: 473 generated token IDs MATCH between baseline and env=1.
- **Perf validation (DSv3 layers 0-19)**:
  - Baseline e2e: 169.812 ms/token, slow AR task mean = 1987.1 μs
  - Env=1 e2e:   167.694 ms/token, slow AR task mean = 1846.6 μs
  - Δ e2e = -2.118 ms (-1.25%)
  - Δ slow AR task = -140.5 μs (-7.1%) — real but small
  - Predicted savings was 12-17 ms (7-10%) based on the wrong contention premise; actual is far below.
- **Decision per start.md tree**: 2.1 ms (1.25%) is below the 3-5 ms revert threshold. Revert to remove dead code.
- **Lesson learned**: re-verify the contention premise from first principles before designing fixes. `scratch/ar_rewrite_design.md` Option A's underlying analysis missed that each `task_offset` uses a different `nvshmem_team_t`. Option B (single barrier task per AR phase) and Option C (system fence) are still on the table but address a different angle.
- Phase-isolation gates (`MPK_AR_SKIP_BARRIER/REDUCE` from `d6d1730a`) remain in tree as measurement infra — those reveal the 91 μs barrier cost is **NVLink dissemination latency**, not contention.
