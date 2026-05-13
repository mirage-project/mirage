# MPK vs vLLM/SGLang per-kernel latency comparison (decode, TP=4 EP=2, hidden=7168, BF16/FP8)

来源:
- vLLM/SGLang: `~/ref_vllm_sglang.md` (2026-05-11 同学 benchmark)
- MPK: `/home/muhengl/mirage/outputs/perfetto_decode_ep2_175618/trace_rank0.csv` (TP=4 EP=2, layers 0-19, prompt-length=1, max-new-tokens=8). **Trace 只 profile 了 1 iter** (验证: TASK_BEGIN_TASK_GRAPH=1, LM head cluster=1, AR cluster=40=2×20)。

## 一句话总览 (CORRECTED — 之前错了)

**每层 wallclock**: MoE 层 ~170-280μs, dense MLP 层 ~268μs (L1/L2), L0+warmup 438μs。平均 7.07ms / 20 layers = **354μs/layer**。
**vLLM**: MoE 143μs, MLP 127μs。MPK 比 vLLM 慢 **~2.5×** per layer。
**SM 利用率**: 31% (有大 bubble) — 多数 worker idle 等待 dependency chain。

### Layer 5 (典型 MoE 层) timeline, 354μs wallclock

| 阶段 | offset μs | duration | 备注 |
|------|-----------|----------|------|
| Q/KV 投影 | 0-103 | 103 | RMSnorm + 多 FP8 dense + splitk 并发 |
| RoPE + KV gather | 67-102 | 35 | 跟 Q/KV 投影并发 |
| Attention | 103-127 | **24μs** | **8 workers active, 120 idle** (V_SPLITS=8) — vLLM 15μs |
| o_proj + AR1 | 127-200 | 73 | |
| Router | 193-217 | 24 | SPLITK+TopK — vLLM router 3μs |
| **MoE W13** | 218-296 | **78μs** | vLLM 24μs **3.25× 慢** ← load imbalance, max expert 82μs |
| SiLU + W2 | 297-331 | 34 | W2 27μs vs vLLM 16μs 1.7× 慢 |
| AR2 + combine | 340-354 | 14 | |

**关键 bubble**:
1. **Attention 阶段**: 24μs × 120 idle workers = **2880 worker-μs bubble** (MLA TP4 V_SPLITS=8 只用 8 task)
2. **MoE W13 load imbalance**: max expert 82μs vs avg 10.5μs (8× spread) → 大部分 worker 闲完成
3. **AR pre/post bubble**: post-attn AR (10μs) → router (24μs) → topk (14μs) 三个串行 = 50μs critical-path
4. **Worker 利用率**: top 59%, bottom 16% — 高方差表明 dependency chain 把一部分 worker 永久卡死

## Q/KV phase 100μs 详细分解 (2026-05-11)

**SM 利用率仅 26.9%** in Q/KV phase. Standalone kernel benches show kernels themselves are fast.

Standalone benches (single-CTA timing on B200):
- `linear_fp8_swapAB` (OLD kernel, MPK doesn't use): q_a 24.24μs, q_b 8.67μs, o_proj 64.57μs, down 16.79μs
- `linear_splitk_fp8_swapAB` (used by SPLITK calls): o_proj split_k=4 B=1 = 19.11μs per-CTA
- `fp8_gemm_dense_smallm` (PR674, MPK uses for q_a/kv_a/etc) — no standalone bench, but trace avg per-task = **6.79μs**

Trace per-event_no breakdown for layer 5 Q/KV phase (13 FP8 dense smallm calls + 9 SPLITK swapAB calls, all firing concurrently at ~8.7μs):
- Each task instance: 3-11μs avg
- Per-call wallclock: 80-90μs (tasks spread across 10-18 waves)
- Total worker-time demanded: ~2500μs (GEMM) + ~950μs (other) = ~3450μs
- Phase wallclock × workers × util = 100 × 128 × 0.27 = 3456μs ✓ matches

**Bottleneck = scheduling, NOT kernel speed**:
- Many small GEMMs (13 dense + 9 splitk) compete for 128 workers concurrently
- Each kernel's task instances are dispersed across many waves
- Util 27% — if util reached 100%, phase would shrink 100→27μs (close to user's 10μs target!)

**Path forward** (for the next iteration):
1. Profile WHY util is 27% — is it task dispatch latency? Cross-task event-chain wait? L2/HBM bandwidth saturation?
2. Reduce the number of concurrent GEMMs by consolidating Q/KV ops (e.g., fused q_a+kv_a into one larger GEMM if possible)
3. Try larger per-task output tile sizes — fewer-but-larger tiles reduce dispatch count + waste
4. Use a B=1-specific GEMV kernel for decode that doesn't waste 99% MMA throughput on B=1

## Q/KV utilization 27% — ROOT CAUSE FOUND (2026-05-12)

Deep-dive into MPK runtime + task graph dump (`outputs/perfetto_decode_fresh_*/build/task_graph_rank0.json`):

### Architectural findings

1. **MPK runtime supports fine-grained event dependency** (`FullTaskDesc.dependent_event` + `trigger_event` in `runtime_header.h`). EventType `EVENT_LAUNCH_TASKS` / `EVENT_LAUNCH_MASSIVE_TASKS` allow per-event task launch.
2. **For DSv3, this is DISABLED at runtime build time** (`runtime.cc:1011-1028`, comment says "event-driven path... currently hangs"). All EVENT_LAUNCH_TASKS downgraded to EVENT_EMPTY. Tasks are pre-allocated to worker queues at startup.
3. **Each task in a worker queue spin-waits on its `dependent_event`** before executing (`persistent_kernel.cuh:727-758`). 10ns nap loop. Workers cannot skip blocked tasks.

### Why util is 27%

Worker queues are FIFO; tasks ordered by emission. When a task in front of the queue has un-fired dep, the worker stalls **even if later tasks are ready**.

Empirical (worker 45 in Layer 5, busiest worker at 71.5%):
- 17μs initial idle (waiting for first task assignment)
- 9.4μs gap before MLA attn (waiting for kv_a/RoPE)
- **42.5μs gap before AR1** (waiting for o_proj on other workers to finish)
- **72.6μs gap before SiLU** (waiting for slowest MoE W13 expert)
- **47.6μs gap before next-layer quantize** (waiting for combine + AR2)
- Total task-time on worker 45 = ~91μs / 354μs phase = 25.7%

### Why dep_events are coarse-grained

`fp8_gemm_dense_smallm` is PERSISTENT (`grid_dim=(num_workers, 1, 1)=(128, 1, 1)`). Internally each worker strides through output tiles via `worker_idx`. From the consumer's view, ALL 128 producer tasks contribute to the output → consumer must wait for ALL 128 → `event.num_triggers=128` (= coarse).

Distribution of events from DSv3 task graph dump:
- 1220 events with num_triggers=1 (37%)
- 1140 with num_triggers=2 (35%)
- 78 with num_triggers=128 (the persistent-kernel "wait for all 128 workers" events) — these are the bottleneck
- 84 with num_triggers=56 (AR cluster boundary)

### Specific actionable opportunities

| Optimization | Mechanism | Risk |
|---|---|---|
| **A.** Re-enable EVENT_LAUNCH_TASKS path (revert runtime.cc:1011-1028 downgrade) | Fine-grained launch — workers only see ready tasks | High (comment says hangs DSv3 selective-layer) |
| **B.** Switch FP8 dense small/medium to NON-PERSISTENT kernel for small batch | grid_dim = output_tiles (not num_workers) → consumer's dep_event num_triggers = 1 per output tile | Medium (kernel rewrite, but linear_fp8_swapAB already exists) |
| **C.** Reduce number of Q/KV kernel calls by fusing | Fewer dep-chain steps → fewer bubbles | Medium (builder changes) |
| **D.** Reorder tasks in worker queue to put ready tasks first | Workers don't stall on blocked tasks | Hard (changes runtime scheduling) |
| **E.** Look-ahead: start next-layer's RMSnorm/q_a before current layer's AR2 finishes | Pre-load worker queues with future-ready work | Hard (dep graph rewrite) |

**Qwen3 comparison** (TP=1 BF16, no MoE, hidden=4096): 220 tasks per layer (vs DSv3 803/layer). Smaller task count = less queue churn → likely higher util.

## 2026-05-12 morning: NUM_WORKERS experiment win (-6.5% decode trace span)

Found that lowering `MPK_FP8_DENSE_NUM_WORKERS` from 128 → 64 gives ~6.5% decode trace span reduction. The persistent `fp8_gemm_dense_smallm` kernel uses `grid_dim=(num_workers, 1, 1)`, dispatching 128 tasks per GEMM by default. With most outputs only needing 12-56 tiles, the extra tasks early-exit but still consume scheduler dispatch. Lowering to 64 cuts task count in half + frees worker queue slots for concurrent GEMMs.

Test results (DSv3 TP=4 EP=2 decode mbt=1 layers 0-19, perfetto trace span):
| NUM_WORKERS | trace span | FP8_DENSE task count | sum_ms | per-task μs | status |
|-------------|-----------|---------------------|--------|-------------|--------|
| 128 (default) | 7.03 ms | 9,856 | 67 | 6.79 | baseline |
| 64 | 6.59 ms (-6.3%) | 4,928 | 44 | 8.99 | ✓ works |
| 56 | 6.57 ms (-6.5%) | 4,312 | 41 | 9.58 | ✓ works |
| 48 | — | — | — | — | ✗ CUDA "unspecified launch failure" |
| 32 | — | — | — | — | ✗ same crash |

**Boundary: 48-56 (decode only)**. Kernel has hidden constraint at low num_workers.

**PREFILL crash too**: tested mbt=128, NUM_WORKERS=64 also crashes (likely chunked-prefill kv_b_k/kv_b_v path with runtime_m_mode=1 has tile counts >> 64 that the kernel doesn't tolerate).

**Action**: Default stays 128. Opt-in `MPK_FP8_DENSE_NUM_WORKERS=64` for DECODE-ONLY workloads (e.g., autoregressive generation after prefill warmup). Documented + committed via `af38cf42`.

**Follow-up TODO**:
- Identify the kernel constraint at num_workers<56 (decode) and num_workers<64 (prefill mbt=128). Likely related to mbarrier ring phase counting or tcgen05 TMEM scheduling.
- If kernel can be fixed, lowering num_workers further would unlock more decode speedup AND make the default safe.

## 2026-05-12: Failed paths tried

### Path 1: Fuse kv_a + kv_rope into single FP8 GEMM
**Status**: Blocked. The MPK DTensor API has no view/slice support, so the fused GEMM's output buffer [batch, 640] can't easily be sliced into c_latent_out [batch, 512] + k_pe_out [batch, 128] for downstream consumers (rmsnorm + MLA). Would need DTensor view API or copy-slice tasks (which defeat the purpose).

### Path 2: Re-enable EVENT_LAUNCH_TASKS in runtime.cc
**Status**: Hangs even Qwen3 (smallest model). The "event-driven path is partially implemented" comment at runtime.cc:1007 understated the breakage — even adding env-var gate with NO selective layers caused Qwen3 to hang after init. Reverted. Deeper debugging required.

### Path 3: Declare output partition (1,-1,-1) on persistent kernel
**Status**: Would race. The persistent fp8_gemm_dense_smallm internally uses worker_idx-strided tile pattern, but downstream consumer kernels read ALL of producer's output (not just one stride). So declaring partition would have runtime claim "consumer worker N depends only on producer N" but actually consumer N reads all → race condition. Skip.

## Net result of 2026-05-12 morning work

**Committed**:
- `af38cf42` — `MPK_FP8_DENSE_NUM_WORKERS` env var (opt-in -6.3% decode trace span)
- `e243272c` — keep `_fp8_dense_kv_b_proj` at full self.num_workers (skip env override for chunked-prefill path)

**Recommended usage**: 
- For decode-only workloads (autoregressive generation): `export MPK_FP8_DENSE_NUM_WORKERS=64` → 6.3% faster
- For prefill or mixed: leave env unset (default 128)

**Remaining big bottlenecks unaddressed**:
- MoE W13 76μs vs vLLM 24μs — reported to kernel owner
- MoE W2 27μs vs vLLM 16μs — reported to kernel owner
- AllReduce per-task 12μs decode (and 380μs prefill barrier) — needs kernel rewrite (1 barrier vs N)
- LM head 75μs single call — grid_dim limited by vocab_size divisibility (= 64); no easy parallel-up

**Future investigation queues** (tasks #117-#129 in TodoList):
- Q/KV phase 27% util deep-dive (dispatch latency vs dep-event-wait)
- B=1 GEMV-specific kernel for decode (avoid MMA waste)
- Fix EVENT_LAUNCH_TASKS hang in runtime.cc
- DTensor view/slice API support (enables fusion paths)
- Investigate fp8_gemm_dense_smallm kernel constraint at low num_workers

## Qwen3 trace analysis (2026-05-12, `outputs/qwen3_perfetto_20260512_001933/`)

| metric | Qwen3 (TP=1 BF16) | DSv3 (TP=4 EP=2 FP8) |
|---|---|---|
| Total trace span | 6.02 ms (1 iter) | 7.03 ms (1 iter) |
| Layers | 32 | 20 |
| Per-layer wallclock | 188 μs | 354 μs |
| Total tasks | 14,380 | 36,152 |
| Total events | 2,366 | 3,294 |
| Tasks per layer | 449 | 1,808 |
| GEMMs per layer (unique event_no) | 5 LINEAR + 4 LIN_W_RES = ~9 | 13 FP8_DENSE + 9 SPLITK = 22 |
| Events with num_triggers=128 (= "wait whole kernel") | 1 | **78** |
| Events with num_triggers=2 (fine-grained) | 1728 (73%) | 1140 (35%) |

**Key architectural difference**:
- **Qwen3**: ~9 logical GEMMs per layer (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj + residuals). Single Q/K/V projection.
- **DSv3**: ~22 logical GEMMs per layer because of MLA LoRA decomposition (q_a → q_b → q_b_nope + q_b_pe; kv_a → kv_b_k + kv_b_v; kv_up_k + kv_up_v). Chain depth is **2× Qwen3**.

DSv3's MLA LoRA architecture (designed for memory savings) inherently has longer GEMM chains than vanilla MHA. Reducing chain length requires either:
1. Materializing LoRA into full Q/K/V weights (defeats the LoRA memory benefit)
2. Fusing adjacent GEMMs (e.g., q_b_nope + q_b_pe both consume q_a_out → one fused GEMM with combined output)
3. Pipelining: use fine-grained event dep so q_b can start as soon as relevant q_a tile is done

**For tomorrow — concrete improvement options**:

| Idea | Mechanism | Effort | Expected gain |
|---|---|---|---|
| Fuse q_b_nope + q_b_pe | Single FP8 GEMM with output [out_nope+out_pe] | High (kernel mods) | 30-50% Q/KV chain |
| Fuse kv_up_k + kv_up_v | Same idea, kv_up combined | High | 20-30% |
| Re-enable EVENT_LAUNCH_TASKS for non-selective-layer runs | `runtime.cc:1011-1028` revert | Medium (need to find selective-layer hang root) | Possibly large |
| Switch persistent FP8 dense → non-persistent for decode B=1 | Reverts to OLDER `linear_fp8_swapAB` path; per-task slower (24μs vs 7μs) but fine-grained dep | High (scale format mismatch) | Unknown |
| Reduce builder GEMM count by combining adjacent identical-shape projections | E.g., q_a and kv_a both read hidden, could batch as `[1, hidden] → [q_lora+kv_lora]` | Medium | 10-20% |

## Per-CALL wallclock (apples-to-apples vs vLLM, decode TP=4 EP=2)

把一次逻辑 kernel call 的 task 实例聚合成 cluster (1μs 间隔), wallclock = cluster 内 max(end) - min(begin)。这是真正可以跟 vLLM 单次 launch 比的数。

| Task | n_calls | wallclock μs | tasks/call | vLLM μs | Gap |
|------|---------|--------------|-----------|---------|-----|
| `MOE_W13_FP8` (gate_up) | 17 | **75.8** (max 82) | 128 | 24 (concurrent) | **3× 慢** ← MoE 已反馈同学 |
| `LINEAR_SM100` (LM head) | 1 | 75.5 | 127 | — | 大 vocab matmul |
| `MOE_W2_FP8` (down) | 17 | 26.7 | 112 | 16 | **1.7× 慢** ← MoE 已反馈同学 |
| `MLA_MTP_DECODE_TP4` | 20 | 22.2 | 8 | 15 | **1.5× 慢** ← V_SPLITS sweep 进行中 |
| `FP8_DENSE_SMALLM` | 94 | 19.5 | 105 | 10 (Qa) | 1.9× 慢 (per-call), 但一层 4 个 Qa 类合并起来 fast |
| `SPLITK_LINEAR_FP8_SWAPAB` | 121 | 16.0 | 51 | 7-15 (qb/qup/kvup/oproj) | 1-2× 慢 per call |
| `NVSHMEM_TILE_ALLREDUCE` | 40 | 14.7 | 56 | 6 (post-MoE) / 8 (post-attn) | **2× 慢** ← TILE sweep 完成, 不是 tile bound |
| `MOE_TOPK_SIGMOID` | 17 | 13.7 | 1 | 7 | **2× 慢** |
| `SPLITK_LINEAR` (router BF16) | 17 | 8.2 | 112 | 3 | **2.7× 慢** |
| `RMS_NORM_HOPPER` | 81 | 3.2 | 1 | 2.6 (SGLang) | 1.2× 慢 (close) |
| `MOE_MUL_SUM_ADD` (combine) | 17 | 2.4 | 56 | 10 | **4× 快** ✓ |
| `SILU_MUL` (act) | 46 | 2.3 | 5 | 3 | **1.3× 快** ✓ |
| `DEEPSEEK_MLA_ROPE` | 40 | 2.6 | 17 | — | — |
| `QUANTIZE_FP8` | 135 | 1.9 | 10 | 2 | **1.05× 快** ✓ |
| `MLA_KV_GATHER` | 20 | 2.4 | 1 | — | — |

## 新的 P0 优先级 (按 absolute saving per iter)

每 iter ≈ 0.88ms TPOT. 一个 MoE 层 wallclock ≈ 总 attn+moe kernel wallclock ≈ 200μs serial; 实际 MPK 因并发约 40-50μs。

按可省时间 (per iter, 20 layers):
1. **MoE W13** save (76-24)×17 calls = 884μs/iter ← 反馈给同学
2. **MoE W2** save (26.7-16)×17 = 182μs/iter ← 反馈给同学
3. **MLA decode** save (22.2-15)×20 = 144μs/iter — V_SPLITS sweep 跑中
4. **AllReduce** save (14.7-7)×40 = 308μs/iter (但 wallclock per-AR 已与上下游并发, 实际 saving < 308μs)
5. **MOE_TOPK_SIGMOID** save (13.7-7)×17 = 113μs/iter
6. **LM head** save (75.5-?)×1 = ?μs/iter — 不知 vLLM 数, 暂不知 gap

## Per-kernel summary (旧表, 保留作 reference)

| vLLM layer | vLLM μs | SGLang μs | MPK task type | MPK avg μs | cnt/li | MPK 状态 |
|---|---|---|---|---|---|---|
| RMSNorm | — | 2.6 | `TASK_RMS_NORM_HOPPER` | 3.2 | 0.5 | Slower 但接近 |
| quant (bf16→fp8) | 2 | 2 | `TASK_QUANTIZE_FP8_SM100` | 1.6 | 8.2 | **比 vLLM 快** |
| Qa proj (7168→1536) | 10 | 10 | `TASK_FP8_GEMM_DENSE_SMALLM_SM100` | 6.8 | 61.6 | **比 vLLM 快** |
| qb proj (1536→Hq×128) | 7 | 7 | `TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100` | 21.1 | 38.8 | **3× 慢** |
| q-up proj (1536→64) | 4.5 | 5 | `TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100` | 21.1 | (同上 bucket) | **4× 慢** |
| attention (decode) | 15 | 19 | `TASK_MLA_MTP_DECODE_TP4_SM100` + reduce | 22.5 | 1.0 | **1.5× 慢** |
| quant (post-attn) | 2 | — | `TASK_QUANTIZE_FP8_SM100` | 1.6 | (in 8.2) | Faster |
| kv up proj | 4.5 | 5 | `TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100` | 21.1 | (bucket) | **4× 慢** |
| o proj (16384→7168) | 15 | 18 | `TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100` | 21.1 | (bucket) | **1.4× 慢** |
| allreduce (post-attn) | 8 | — | `TASK_NVSHMEM_TILE_ALLREDUCE` | 11.9 | 14 | **1.5× 慢** |
| router | 3 | 8 | `TASK_SPLITK_LINEAR_SM100` (BF16) | 7.8 | 11.9 | **2.6× 慢** (vs vLLM) |
| topk | 7 | 10 | (待映射 — 应该是 sigmoid+gating 路径里的某段) | ? | ? | 待查 |
| group gemm 1 (gate_up) | 24 | 38 | `TASK_MOE_W13_FP8_SM100` | 10.6 | 13.6 | **2× 快 vs vLLM** |
| act (silu) | 3 | 6 | `TASK_SILU_MUL` | 1.8 | 1.5 | **1.6× 快** |
| group gemm 2 (down) | 16 | 21 | `TASK_MOE_W2_FP8_SM100` | 5.1 | 11.9 | **3× 快 vs vLLM** |
| combine (weighted sum) | 10 | 15 | `TASK_MOE_MUL_SUM_ADD_SM100` | 1.4 | 5.95 | **7× 快** |
| allreduce (post-MoE) | 6 | — | `TASK_NVSHMEM_TILE_ALLREDUCE` | 11.9 | (in 14) | **2× 慢** |
| Shared exp gate up | 10 | 27 | (DSv3 V3 builder 当前不走 shared expert?) | n/a | 0 | 待查 |
| Shared exp down | 22 | 27 | 同上 | n/a | 0 | 待查 |
| **per MLP layer** | **127** | **178** | — | (估算) | | 待算 |
| **per MoE layer** | **143** | **177** | — | (估算) | | 待算 |

## 主要差距 (按重要性排序)

### 🔴 P0 — 必须 tune

1. **`TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100` 慢 3-4×**  
   - MPK avg=21μs, vLLM 4.5-15μs  
   - 使用场景: qb_proj, q_up, kv_up, o_proj, 各种 dense FP8 matmul (decode 路径)  
   - 占 decode 总 SM 时间最大份额: 131ms / 总 245ms = 53%。  
   - 可能问题: pipeline stage 数, tile shape, register pressure  
   - 行动: 单独 benchmark + ncu profile, 比对 vLLM 用的 CUTLASS 或 DeepGEMM

2. **`TASK_NVSHMEM_TILE_ALLREDUCE` 慢 1.5-2×**  
   - MPK avg=11.9μs, vLLM 6-8μs  
   - 当前实验: `MPK_ALLREDUCE_TILE_SIZE` 扫 128 / 512 / 1024 中  
   - 假设: 每个 tile=128 的 barrier 跨 56 个 task 平均, barrier overhead 占大头

3. **`TASK_MLA_MTP_DECODE_TP4_SM100` 慢 1.5×**  
   - MPK avg=22μs, vLLM 15μs (TP=4 减少了每卡 head 数, 但还是慢)  
   - 行动: 看是否 V-split / head-group split 配置可以调

### 🟡 P1 — 想 tune 但量比较小

4. **`TASK_SPLITK_LINEAR_SM100` (BF16, router) 慢 2.6×**  
   - MPK avg=7.8μs, vLLM 3μs  
   - 使用场景: router gate + 任何 BF16 splitk  
   - 量级总和 ~15ms summed = 6% trace
   - 行动: 对照 vLLM router kernel 看是否需要 fuse 进 topk

5. **`TASK_RMS_NORM_HOPPER` 慢一点 (3.2 vs 2.6)**  
   - 接近 vLLM, 但 SGLang 2.6 表明可以再快  
   - 量级很小

### 🟢 已经 win

- **MoE w13/w2/combine: 全部比 vLLM 快 2-7×** — 这是 MPK 的核心优势 (fused kernels)  
- **FP8 dense smallm (Qa/kv_a)**: 6.8μs < 10μs vLLM  
- **quant**: 1.6μs < 2μs vLLM  
- **SILU**: 1.8μs < 3μs vLLM

## AllReduce TILE_SIZE sweep 结果 (prefill 128, TP=4 EP=2)

| TILE | tasks/AR | per-task μs | per-AR wallclock μs | end-to-end ms |
|------|----------|-------------|---------------------|---------------|
| 128 (default) | 56 | 370.3 | 379.8 | 166.983 |
| 512 | 14 | 374.8 | 379.9 | 171.080 (+2.5%) |
| 1024 | 7 | 376.1 | 379.1 | **164.004 (-1.8%)** |

**关键观察**:
1. per-task 耗时 (~375μs) 跟 TILE_SIZE 几乎无关 — 不是 NVLS 数据搬运 bound, 也不是 inter-rank desync (rank0/rank1 起止已验证 ±20μs 对齐)。**barrier / NVLS 多播延迟主导**。
2. TILE=1024 比 default 快 ~3ms (-1.8%) 但只小幅, 因为 AR 只占总时间 7%, 其余靠 producer→AR event chain。
3. TILE=512 反而**慢**于 128 — 可能 sweet spot 是 producer/AR tile 比例不能 4:1 (每 4 个 producer 给 1 个 AR 反而让 AR 等待更长尾的 producer 慢), 而 8:1 (1024) 或 1:1 (128) 都好。需要再跑一轮确认是不是 noise。

**结论**:
- 改 TILE_SIZE 可以拿到 ~2% 的小提升 (1024 vs 128), 不能消除 AR 的 vLLM 50× gap (per-task)
- 真正的 gap 来自 per-task 中的 NVLS 多播延迟 + dissemination barrier; 要 close gap 必须 kernel 级别改造:
  1. 合并 56 个 barrier → 1 个 (类似 vLLM 单 kernel 一次 barrier)
  2. 或者每个 task 只做 NVLS reduce, 信赖一个 global producer→consumer flag (不需要 per-task barrier)
- 但实际上, MPK **整层 wallclock ~44μs vs vLLM ~127μs**, 已经因为 persistent kernel 并发整层而胜过 vLLM。继续优化 AR 是锦上添花。

## 后续 TODO

- [x] `MPK_ALLREDUCE_TILE_SIZE` sweep: 已证明 128 default 是最优 tile_size — gap 不是来自 tile_size, 是来自 per-task barrier 本身
- [ ] 用 nsys/ncu profile `nvshmem_tile_allreduce` 单 task 找出 370μs 卡哪里
- [ ] `TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100` 单独 benchmark — 这是最大的 decode 时间消耗 (53%)
- [ ] 把 prefill trace (128 token) 也做一遍 per-kernel 对比 (vLLM/SGLang 表格是 decode, prefill 数据待问同学)
- [ ] 确认 DSv3 V3 当前 builder 是否走 shared expert 路径 — vLLM 表格里 shared exp 占 33μs (gate+act+down), 不能漏算
