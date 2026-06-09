# DeepSeek-V3 test-mode coverage — decision log

This file records the decisions made while building/refactoring test-mode tests so that every
DeepSeek-V3 demo layer is verified at real DSV3 shapes across TP∈{1,2,4,8} and bs∈{1,2,4,8,16}.
It exists so the work can be reviewed without re-deriving the reasoning. Plan file:
`~/.claude/plans/for-all-the-layers-buzzing-eich.md`.

Last updated: 2026-06-09 (PAUSED mid-run by user).

## 🔖 Progress / resumption status (as of pause, 2026-06-09)

**DONE (tests written/refactored to DSV3 shapes, swept, and PASSING on gpu0):**
- Group A — `sm100_rmsnorm` (rmsnorm + fused_rmsnorm_quantize_fp8), `sm100_quantize_fp8` (+f32 ref branch).
- Group B — `sm100_silu_mul`, `sm100_elementwise_add`, `sm100_embed`, `sm100_tensor_init`.
- Group C — `sm100_linear` (linear + linear_with_residual), `sm100_splitk_linear_bf16` (→ kernel issue #1).
- Group D — `sm100_fp8_gemm_dense` (smallm/mediumm + fp8out + decode_splitk → issue #2), `sm100_linear_fp8`
  (swapAB + residual → issue #3), `sm100_linear_fp8_bmm` (bmm + bmm_dense + assemble_q), `sm100_fp8_group_gemm_decode` (→ issue #4).
- Group E — `sm100_moe_sigmoid`, `sm100_fp8_moe` (w13/w2), `sm100_moe` (silu_mul, mul_sum_add, permute↔unpermute).
- Group F (partial) — `sm100_mla` rope (q_fused/q_split/k), `sm100_mla` kv_gather + unified,
  `sm100_mla` prefill_absorbed (→ issue #5 + build-break src fix), `sm100_mla_prefill_tp8_chunked` (→ issue #6 + bare-scale finding).

**REMAINING (not started):**
- Group F — `sm100_mla_mtp_decode` **TP1** (`mla_mtp_decode_layer` + `mla_mtp_reduce_layer`): sweep bs/q_len on
  the existing partial-level tests + ADD a fused decode→reduce final-output check vs a TP-agnostic full-MLA
  reference (add that reference to the folder's pytorch_reference.py, parametric over H — reused by TP2/4/8).
- Group F — `sm100_mla_mtp_decode` **TP2/4/8** (`mla_mtp_decode_tp{2,4,8}_layer` + matching `*_reduce_layer`):
  NEW fused decode→reduce tests vs the full-MLA reference (per-variant partial layout differs; compare final output only).
- Group G — `sm100_argmax` (argmax_partial + argmax_reduce, V=129280, ragged last chunk), `sm100_softmax_gather` (V=129280). bs-only sweeps.
- Final roll-up: run all refactored/new testmode tests serialized on gpu0; Qwen3-8B demo regression (~4.3 ms/token) as shared-path sanity.

**Orchestration reminders for resumption:** gpu0 only, GPU runs serialized (one subagent on GPU at a time);
subagents must run `test-on-gpu` in the FOREGROUND (background → orphaned jobs, hit twice); if nvcc errors
"no instance of function template matches", rebuild `.so` on gpu0 (`uv pip install -e . --no-deps`).

**⚠️ Uncommitted kernel-source changes (need user review):**
`include/mirage/persistent_kernel/tasks/blackwell/mla_prefill_sm100.cuh` + `src/kernel/task_register.cc` —
minimal, backward-compatible fix for the absorbed-prefill build break (see issue #5). All other changes are test/reference files.

## Global decisions

- **Scope = core forward path only.** Excluded the speculative/MTP family (`mtp_*`,
  `prob_scatter/extract`, `nvshmem_global_argmax`) — gated behind `--mtp>0`, which defaults to 0 and is
  not in the canonical verify command. (User decision.)
- **Collective ops skipped** (`allreduce_layer`, `nvshmem_global_argmax_layer`) — inherently multi-GPU;
  the demo's end-to-end run covers their TP>1 correctness. (User decision.)
- **TP tested as per-rank shapes on a single GPU.** All tests run `world_size=tp, mpi_rank=0` on one
  physical GPU (gpu0 only). Sharded dims (heads = 128/tp, intermediate, routed experts) are shrunk in the
  test; no MPI/NVSHMEM is used. This validates each kernel's shape handling, which is the requirement.
  (User decision: "they should handle that shape correctly".)
- **Refactor existing per-folder tests in place** under `tests/runtime_python/blackwell/sm100_*/`; do not
  build a parallel test tree. (User decision.)
- **One subagent per layer/folder for implementation + testing**, dispatched by the orchestrator. GPU runs
  are **serialized on gpu0** because the MPK megakernel assumes whole-GPU occupancy (concurrent runs hang).
  (User decision.)
- **Run autonomously to completion; log decisions here.** (User decision.)

### Test conventions
- Reference lives in each folder's `pytorch_reference.py`; reuse existing functions, extend only as noted.
  Reuse `blackwell/common/sm100_fp8_scale_layout.py` for UE8M0 quantization rather than duplicating.
- Tolerances: bf16 → `atol/rtol ≈ 1e-2`; fp8 (block-scaled) → cosine > 0.99 or relative ≈ 2–5%. A kernel
  that can't meet a sane tolerance is reported as a real kernel issue, NOT hidden by loosening tolerance.
- **Sweep only where it matters.** Per-element ops (rmsnorm, quantize, fused-rmsnorm-quant, silu_mul,
  elementwise_add, tensor_init, embed, moe_mul_sum_add, moe_unpermute, topk routing, sampling) → bs sweep
  only; HIDDEN/VOCAB are not TP-sharded. GEMM / attention / MoE-GEMM → TP × bs. Decode-only FP8 kernels are
  capped at bs≤16 (swapAB/bmm/splitk assert ≤16; decode_splitk gates ≤8).

### Test-matrix size policy (autonomous decision — to bound GPU compile time)
Each (tp, bs[, max_seq_length]) config needs its own nvcc compile of the megakernel (~1–3 min, more for
FP8). A full TP×bs cross-product (4×5=20) per GEMM-like layer across ~20 layers would be 100+ GPU-hours.
To stay tractable while still exercising **every TP value and every bs value** required:
- **bs-only layers** (per-element ops, sampling): full sweep `bs ∈ {1,2,4,8,16}` (5 configs).
- **TP×bs layers** (GEMM / attention / MoE-GEMM): a **union-of-axes** matrix, not the full cross-product:
  `{TP=1} × {bs=1,2,4,8,16}` ∪ `{bs=16} × {TP=2,4,8}` ∪ corner `{TP=8, bs=1}` = 9 configs. This hits
  every TP∈{1,2,4,8} and every bs∈{1,2,4,8,16} at least once. (Decode-only kernels cap bs at ≤16 or ≤8.)
- Dense-FP8 also varies `max_seq_length ∈ {256,4096}` to cover smallm+mediumm (add to the above where it applies).
Each subagent logs the exact matrix it ran. If a kernel fails on a specific (tp,bs), that config is reported
as a real issue rather than dropped.

### Architecture facts that drive shapes (verified from builder.py)
- `num_local_q_heads = 128 // world_size` → 128/64/32/16 for TP=1/2/4/8.
- `ep_size` defaults to 1 (pure tensor-parallel): `routed_tp_size = world_size`, `num_local_experts = 256`.
  Primary sweep uses ep=1; MoE-GEMM units add a secondary ep>1 check.
- `intermediate //= world_size`; `routed_moe_intermediate = 2048 // routed_tp_size`; HIDDEN (7168) not sharded.
- Dense-FP8 smallm-vs-mediumm selected by `max_seq_length<=512` (NOT M) → tests set `max_seq_length`
  (256 → smallm, 4096 → mediumm) to cover both kernels.
- MLA softmax uses the YARN scale `(1/sqrt(192))·mscale²`, `mscale = 0.1·ln(40)+1` — references must match,
  because the kernel applies it internally regardless of any passed scale.

### Out of scope (confirmed not called by the builder, or trivial)
`allreduce_layer`, `nvshmem_global_argmax_layer`, all `mtp_*`/`prob_*`, `identity_layer` (trivial copy),
`mla_prefill_tp8_layer` (non-chunked; not called), `mla_prefill_layer` (not called).

### Workflow gotcha (for run scripts)
Do NOT `cd /mnt/shared/.../mirage_dpskv3` inside a `.scratch/*.sh` run script. `test-on-gpu` already
`cd ~/mirage_dpskv3` on the gpu0 node; cd-ing back to the shared FS makes `__file__` resolve to the
read-only shared path (uid mismatch) and `pk.compile(output_dir=folder_of(__file__))` fails with
PermissionError. Just put `python tests/.../test_x.py` (relative) in the script.

## ⚠️ Known kernel issues surfaced by these tests (real bugs, NOT test bugs)

These are genuine correctness problems found while testing at DSV3 shapes. They are documented (not
hidden by loosening tolerance). The corresponding test configs are marked xfail with a pointer here.

1. **`splitk_linear_layer(accumulate=False)` is incorrect.** (sm100_splitk_linear_bf16, found 2026-06-09)
   - bs=16 (the only non-hanging bs for acc=False): output max_diff ≈ 3.82 vs 1e-2 tol — garbage, not the GEMM.
   - bs<16: hangs (MMA_N=16 path). The `accumulate=True` variant passes cleanly at all bs.
   - Path is used by the BF16 router-gate split-K when `_BF16_GATE_SPLITK_ENABLED=True` (current default on
     feat/dpskv3) → router-gate logits would be wrong for every MoE layer. Workaround that restores
     correctness: `_BF16_GATE_SPLITK_ENABLED=False` (falls back to `linear_layer`, which the test confirms passes).
   - Status: reported, not fixed (kernel fix is out of scope for this test-coverage task).

2. **`fp8_gemm_dense_decode_splitk_layer` crashes for `split_k >= 4` (multi-wave).** (sm100_fp8_gemm_dense,
   found 2026-06-09)
   - `cudaErrorLaunchFailure` whenever a worker must process >1 wave, i.e. when
     `total_tiles = ceil(M/128)*ceil(N/128)*split_k > num_workers`. Localized via a split_k sweep at the
     DSV3 decode O_proj shape (M=8, N=7168 → nn=56, B200 num_workers=136), K varied across TP:
     `split_k=1` (direct-store) PASS cos=1.0; `split_k=2` (112 tiles ≤ 136, single wave) PASS cos=1.0;
     **`split_k=4` (224 tiles > 136) CRASH at every K∈{16384,8192,4096,2048}; `split_k=8` (448 tiles) CRASH.**
   - Root cause (code read): the producer/consumer mbarrier phase `ph` is carried across waves but
     `nk_slice % NS (=3) != 0`, so wave 2 starts mid-phase and desyncs the A/B pipeline
     (`fp8_gemm_dense_decode_splitk_sm100.cuh` producer loops L191-264). Single-wave split_k never desyncs.
   - **Impact**: when `MPK_DSV3_DECODE_OPROJ_SPLITK=1`, the builder defaults
     `MPK_DSV3_DECODE_OPROJ_SPLITK_FACTOR=4` → the decode O_proj GEMM crashes. The path is env-gated OFF by
     default (default demo unaffected). The `e9e6c11c` "atomic scope hardening" commit fixed an *earlier*
     `cudaErrorLaunchFailure` (cross-SM atomic scope); this is a *separate* multi-wave phase bug.
   - Status: reported, not fixed (kernel fix out of scope). The test verifies the kernel + tensor_init
     prepend + reference at split_k=2 (single-wave, where the reduce-add path is CORRECT); split_k=4 runs
     only as an isolated single-config XFAIL probe (a launch failure corrupts the CUDA context).

3. **`linear_fp8_with_residual_layer` (`linear_fp8_with_residual_sm100`, non-swapAB) DEADLOCKS at DSV3
   shapes.** (sm100_linear_fp8, found 2026-06-09)
   - The residual epilogue (single-buffer `residual_full/empty` mbarriers, `init(1)`, phase=`current_iter&1`,
     `linear_fp8_sm100.cuh` L236-265, L298-312, L599-601, L728-733) hangs once there are "enough" residual
     epilogue iterations. Localized by a (grid_x, per-task-N) sweep at K=7168, bs=1, residual on
     (each forked with a 120-180 s wall guard; a hang spins the persistent workers at ~18-25% util forever):
     `gx1 N=128/384` PASS, `gx2 N=256/768 (per-task N=128/384)` PASS, **`gx1 N=4608 (288 tiles)` PASS**,
     **`gx48 N=18432 (per-task N=384)` HANG, `gx96 N=12288 (per-task N=128)` HANG, `gx1 N=36864 (2304 tiles)`
     HANG, `gx96 N=36864` HANG.** Two independent triggers: **(a) many task instances** (≥ ~48 tasks hang;
     1-2 tasks always pass even at 288 tiles/task) and **(b) a very large single-task tile count** (gx1
     hangs at 2304 tiles but passes at ≤288). The non-residual `linear_fp8_layer` with the SAME shapes /
     scale layout / grid passes the full DSV3 union-of-axes matrix 9/9 — the hang is specific to the
     residual epilogue's barrier pipeline.
   - **Impact**: EVERY DSV3-production residual config hangs — `_pick_grid_x` (mirrors builder
     `_fp8_linear_grid_x`) gives grid_x∈{36,72,96} for all gate_up (tp,bs); and even grid_x=1 hangs at the
     full tp=1 N=36864. NOTE: in the current feat/dpskv3 builder this layer is reachable only via
     `_fp8_linear_prequantized`, which is **defined but never called** (the live FP8 linears route through
     `_fp8_linear`→`_fp8_linear_v2` dense GEMM, or the swapAB decode layer), so the default demo is
     unaffected — but the layer is broken if ever wired in at DSV3 scale.
   - Status: reported, not fixed (kernel fix out of scope). A hanging config cannot be a normal pytest
     xfail (it never returns), so `test_linear_fp8_with_residual_testmode.py` records the DSV3 union-of-axes
     matrix as `HANGING_MATRIX` (documentation only, NOT executed) and asserts on a residual SMOKE
     (bs=1, N=768, K=7168, grid_x=2 = 48 total iters, in the proven-PASS region; cos=1.0) that validates
     the residual GEMM+add math + UE8M0 scale layout + harness.

4. **`fp8_group_gemm_largem_compact_sm100` no-mask tile dispatch is HARD-CODED to 128 local experts →
   OOB `m_indices` read for M_total < 16384.** (sm100_fp8_group_gemm_decode, found 2026-06-09)
   - When `active_expert_mask == nullptr` (the layer's `meta=None` path), the compact prologue's 4-warp ×
     32-lane ballot scan (`fp8_group_gemm_largem_compact_sm100.cuh` L253-287) marks **all 128 expert
     slots active** (`is_active = 1` for `expert_id ∈ [0,128)`), so `s_num_active = 128`. The tile loop then
     iterates `num_active(=128) * nn` tiles and resolves `bm = s_compact[ae]` for `ae ∈ [0,127]`, reading
     `__ldg(m_indices + bm*128)` (L327) — which requires `M_total = 128*128 = 16384` (E=128, MPE=128).
     For any smaller `M_total` (e.g. the natural decode M_total=128 at E=1), workers index
     `m_indices[bm*128]` past the end → `cudaErrorIllegalAddress`.
     **compute-sanitizer evidence** (`.scratch/probe_sanitizer.sh`, E=1/M_total=128):
     `Invalid __global__ read of size 4 bytes ... 1 bytes after the nearest allocation ... Device Frame:
     fp8_group_gemm_largem_compact ... fp8_group_gemm_largem_compact_sm100.cuh:327` (the `__ldg(m_indices…)`).
     Deterministic: crashes whenever the FIRST largem_compact launch in a process has M_total<16384;
     passed in isolation when an E≥2… run preceded it only because that earlier run left the tile loop in a
     state that masked the OOB — the bug is the hardcoded 128-expert no-mask scan, not a warmup quirk.
   - **Scan also caps at 128 experts even WITH a mask** (the ballot only covers `expert_id < 4*32 = 128`),
     so a `num_local_experts > 128` config (ep_size=1 → 256) would silently DROP experts 128-255. The
     kernel's own header comment is written around `M=16384` (128 experts). This strongly implies the
     compact largem path is only correct for **E ≤ 128** (i.e. ep_size ≥ 2 on DSV3, num_local_experts=128).
   - **Impact**: the production NEW-MoE W13/W2 group GEMMs (`MPK_DSV3_NEW_MOE=1`) route through this kernel.
     With `MPK_DSV3_ACTIVE_SKIP=1` (default) the mask IS passed, but the 128-expert scan cap still applies;
     at ep_size=1 (num_local_experts=256, the default) experts 128-255 are never dispatched. The smallm
     sibling (`fp8_group_gemm_smallm_sm100`) is M_total-driven (`total = ceil(M_total/BM)*nn`, m_indices read
     under an `m_start < M_total` guard, `fp8_group_gemm_sm100_common.cuh` L243-249) and is correct for ANY
     E/MPE — it has NO such limitation.
   - Status: reported, not fixed (kernel fix out of scope). The test therefore runs the largem arm only at
     the kernel's correct design point **E=128, MPE=128 (M_total=16384)** — production-faithful for ep≥2 —
     and sweeps the bs/decode axis through the smallm kernel (which handles small M_total correctly).

5. **`mla_prefill_sm100_task_impl` first-row-of-q-block (`q_start`) merge is NON-DETERMINISTICally WRONG
   for prefills S>64 (more than one PF_BM=64 q-block).** (sm100_mla absorbed prefill, found 2026-06-09)
   - The kernel special-cases the first row of every q-block (`q_start`) with a fix-up that runs AFTER the
     main MMA-softmax + output-write: `q_start==0` overwrites O[row0]=V[0] (`mla_prefill_sm100.cuh` L748-755),
     and `q_start>0` re-derives the diagonal score and merges it into the already-written history row
     (L758-822). For S>64 (q_block>0 exists), the row at each `q_start` (64,128,192,...) comes out
     **non-deterministic and wrong**: across two identical-seed process runs the set of corrupted (row,head)
     pairs CHANGES (e.g. S=256/H=128: run1 bad row 192 head {1,35,69,103}; run2 bad rows {64,128,192}), and
     in the full matrix the same config has produced both finite-but-wrong (max_diff≈0.32, individual q_start
     rows) and outright NaN. Only the `q_start` rows are affected — every other row is bit-stable and matches
     the reference (mean_diff≈5e-4). Likely cause (code read): the main causal mask (L510, `kv_pos<=q_pos_seq`)
     INCLUDES the diagonal key for the `q_start` row, and then the L758-822 merge ADDS the diagonal again
     (double-count), on top of `-use_fast_math` fragility on that single-score row that the comments
     themselves flag. Two speculative `__syncthreads()` (fence O before the fix-ups; barrier the tmp reduction)
     did NOT fix it — the bug is in the merge math, not just a missing barrier.
   - **Evidence**: `.scratch/probe_mla_diag.py` (determinism + per-row/per-head localization — bad row always
     == `q_start`, set of bad heads varies run-to-run); `.scratch/probe_mla_s64.py` (S<=64 is bit-stable &
     correct across all H and B>1).
   - **Scope of correctness**: deterministic + correct for **single-q-block prefills (S<=64)** at every
     H∈{128,64,32,16} and for B>1 (one q-block per request). The shared non-absorbed `mla_prefill_layer`
     and `mla_prefill_tp8_layer` are not called by the default demo (per "Out of scope" above), and the
     absorbed-prefill path is reachable only via the MTP prefill (`use_mtp_prefill_attention`, gated behind
     `--mtp>0` which defaults 0) — so the default verify command is unaffected.
   - Status: reported, not fixed (kernel fix out of scope for this test-coverage task). The test asserts on
     the sound single-q-block matrix and records S>64 as `XFAIL_MULTI_QBLOCK` (documentation only — a
     flaky-wrong config must not gate CI; NOT a tolerance loosening).
   - **Separate fix applied (in scope, REQUIRED to even compile the layer)**: commit `72cf5f76`
     ("drop unwired kpe_row_stride / kpe_offset args") removed two impl params claiming neither register
     passed them — but `register_mla_prefill_absorbed_sm100_task` DOES pass them, so the absorbed path failed
     to compile (`test.cu: too many arguments in function call`). Restored `kpe_row_stride`/`kpe_offset`
     (+ their 3 KPE read sites) in `mla_prefill_sm100.cuh` and offset the fused Q_pe base by `+d_ckv` in
     task_register.cc (mirrors `kpe_offset`) so the fused [nope512|pe64] Q/KV layout slices correctly. Both
     are backward-compatible: the non-absorbed register + the kernel-wrapper test pass 11 args and get the
     defaults (`PF_D_KPE`,`0`) = original split-layout behavior.

6. **`mla_prefill_tp8_chunked_sm100` produces ALL-NaN when the K/V buffer is sized exactly `kv_len` and
   `kv_len % BN(128) != 0` (partial last KV block) — `0 * NaN(V)`.** (sm100_mla_prefill_tp8_chunked, found
   2026-06-09)
   - The kernel loads each KV block as BN=128 rows via TMA. When the LAST block is partial (kv_len not a
     multiple of 128) AND the K/V tensor has only `kv_len` rows, the partial tail reads OOB → TMA OOB-fill
     writes NaN (`CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA`). The causal/length mask
     (`do_mask_softmax`, `mla_prefill_tp8_chunked_sm100.cuh` L289 `kvp < kv_len`) correctly sets those
     scores to `-INF` → softmax prob 0, BUT the PV MMA (`do_pv_half` L347-356) then computes
     `prob(0) * V(NaN) = NaN`, and the whole output becomes NaN. The kernel never zeroes the OOB-NaN V rows
     before the PV product, so a 0-probability column still poisons the accumulator.
   - **Evidence** (`.scratch/probe_chunked_nan.py`, exact-`kv_len` buffers, q=64): kv∈{64,96,160,192}
     (kv%128≠0) → `nan=ALL`, every output row NaN; kv∈{128,256} (kv%128==0) → clean (max_diff≤0.004). NOT
     about q_start (kv=128 q_start=0 is clean) — purely the partial last block. `.scratch/probe_chunked_pad.py`
     proves the fix: padding the K/V buffers to `ceil(kv_len/128)*128` rows (zero tail) removes the NaN for
     ALL the same cases (max_diff≤0.002, cos=0.999998) — the kernel masks `kvp>=kv_len`, so zero padding is
     numerically inert.
   - **Impact / production-faithfulness**: the DEFAULT demo is unaffected. In the live (non-test) codegen
     path the kernel reads the runtime `KV_LEN_` from the paged-KV meta tensors, while the K/V cache buffer
     is sized to `max_seq_length` (page-aligned, a multiple of 128) — so the partial-block tail is VALID
     cached data (masked out by `kvp<KV_LEN_`), never TMA-OOB-NaN, and `0*realV=0`. The NaN is a sharp edge
     for any caller that sizes the K/V buffer to exactly an unaligned `kv_len` (as a naive unit test would).
   - Status: reported, not fixed (kernel-correctness fix out of scope). The test sizes K/V to a BN-aligned
     row count (production-faithful) and asserts clean PASS, including explicit partial-last-block configs
     (kv=64→pad128, kv=192→pad256). A defensive in-kernel fix would be to zero (not NaN-fill) OOB V rows, or
     to guard the PV product so a `-INF`-masked column contributes 0 regardless of the V payload.

## Per-unit decisions

(Appended as each layer/folder is completed: shapes chosen, tolerances, references added, anything that
deviated from the plan, and any real kernel issues found.)

### sm100_rmsnorm (rmsnorm_layer + fused_rmsnorm_quantize_fp8_layer) — 2026-06-09
- **Files**: `tests/runtime_python/blackwell/sm100_rmsnorm/test_rmsnorm_testmode.py`,
  `test_fused_rmsnorm_quantize_fp8_testmode.py`, and extended `pytorch_reference.py`
  (added `fused_rmsnorm_quantize_fp8_ref` + a small f32-block-scale quantizer
  `_quantize_to_fp8_f32_scale`; reused shared UE8M0 `quantize_to_fp8_deepgemm_style`).
- **Shapes**: HIDDEN=7168, bf16, eps=1e-6 (passed explicitly; `rmsnorm_ref` default kept at 1e-5).
  bs-only sweep {1,2,4,8,16}; grid=(bs,1,1) (= builder `_rmsnorm_grid` at default
  `MPK_DSV3_RMSNORM_ROWS_PER_TASK=1`), block_dim=(128,1,1) — mirrors builder.
- **Fused matrix**: bs{1,2,4,8,16} × scale{UE8M0,f32} × emit_bf16{1,0} = 20 configs. UE8M0 scale uses
  `allocate_packed_ue8m0_scale_deepgemm_style` (col-major [packed_k=14, aligned_bs]); f32 scale is
  (bs, num_groups=56) row-major. emit_bf16=0 verified by asserting the bf16 buffer stays zero (kernel skips the store).
- **Tolerances**: bf16 compared with combined atol+rtol=1e-2 (torch.testing.assert_close semantics:
  `|out-ref| <= atol + rtol*|ref|`), NOT pure-atol — pure-atol spuriously flagged a single 1-ULP bf16
  rounding diff. UE8M0 packed scale compared bit-exact; f32 scale to ~1e-9; dequant-fp8 vs f32-normalized
  reference asserted on relative mean ≤ 5% (observed ~2.2%; per-element max 5.88% = e4m3 granularity).
- **Decision (not a loosening)**: at bs=16 emit_bf16=1, exactly 1 of 114688 bf16 elements differed by
  1.5625e-2 = exactly one bf16 ULP (probe confirmed abs_diff/ULP=1.0). Cause: kernel computes RMS via a
  256-thread tree reduction whereas torch uses `.mean()`; the slightly different `rms_rcp` pushes one f32
  product across a bf16 round-to-nearest boundary. Both values are the two nearest bf16 reps of the true
  result — benign. The atol+rtol criterion (the decision-log spec) absorbs it correctly.
- **Real kernel issues**: none. All 5 (rmsnorm) + 20 (fused) configs PASS on gpu0.

### sm100_quantize_fp8 (quantize_fp8_layer) — 2026-06-09
- **Files**: `tests/runtime_python/blackwell/sm100_quantize_fp8/test_quantize_fp8_testmode.py`
  (refactored to the DSV3-shaped, swept `_run_case(bs,K,scale_ue8m0)` + `__main__` loop idiom),
  `pytorch_reference.py` (added the f32-scale branch: `_quantize_to_fp8_f32_scale`, wired into
  `quantize_fp8_ref(scale_ue8m0=False)` — previously raised NotImplementedError; mirrors the
  sm100_rmsnorm helper but kept local to this folder). `test_quantize_fp8_multirow_testmode.py`
  default `hidden_dim` bumped 4096→7168 (DSV3 HIDDEN) so its multi-row-per-task stress path
  (bs∈{128,512,130} > num_workers) runs at a real DSV3 width — still bit-exact PASS.
- **Per-element op → bs-only sweep, no TP sweep** (K is not TP-sharded for the quantize itself).
  Matrix = `{UE8M0, f32} × {K=7168 (MoE-input HIDDEN), K=2048 (routed-MoE silu-output)} ×
  bs∈{1,2,4,8,16}` = **20 configs**. grid=(bs,1,1), block=(128,1,1) (wrapper overrides grid.y
  to drive ROWS_PER_TASK; legacy callers all pass natural (bs,1,1)).
- **UE8M0 path** (`scale_ue8m0=True`): output scale is packed UE8M0 `uint32` in the deepgemm
  col-major `[packed_k, aligned_bs]` layout (`allocate_packed_ue8m0_scale_deepgemm_style`,
  reused from the shared helper — packing logic NOT duplicated). Scale compared bit-exact
  (rtol=atol=0); fp8 compared by dequanting both sides through the shared UE8M0 decode and
  asserting relative-mean ≤ 5%.
- **f32 path** (`scale_ue8m0=False`): output scale is plain float32 `(bs, K/128)` row-major
  (kernel writes `output_s[batch*num_groups + group]`). Scale compared to 1e-6 (no UE8M0
  snapping — both sides compute `max(|grp|,1e-10)/448` in f32); fp8 dequanted with the per-group
  f32 scale and asserted relative-mean ≤ 5%.
- **Tolerances are NOT a loosening**: observed dequant rel-mean ≤ 0.015% and f32-scale max-abs-diff
  ≤ 9.3e-10 across all 20 configs — orders of magnitude inside the 5% / 1e-6 thresholds. The kernel
  and reference run identical quantization math (same 1e-10 floor, same UE8M0 IEEE-exponent encode,
  same clamp), so fp8 bytes match almost bit-for-bit. The few `fp8 max-abs-diff(float)=32` cases are
  a handful of elements straddling one e4m3 round-to-nearest boundary at large magnitude (1 fp8 ULP)
  — expected granularity, washed out in the relative mean.
- **Reference detail**: f32 reference floors the per-group max at `1e-10` to match the kernel, which
  inits `local_max = eps=1e-10` then re-floors with `1e-10`; for randn DSV3 inputs the block max
  dominates so the floor never binds, but it's replicated for fidelity.
- **Real kernel issues**: none. All 20 configs PASS on gpu0; multirow stress (3 configs) also PASS.

### sm100_silu_mul (silu_mul_layer) — 2026-06-09
- **Files**: `tests/runtime_python/blackwell/sm100_silu_mul/test_silu_mul_testmode.py`,
  `pytorch_reference.py` (defines `silu_mul_ref` with per-chunk gate||up interleaving).
- **Shapes**: DSV3 TP=1 — dense MLP `I=18432, num_tasks=48`; shared/routed expert `I=2048, num_tasks=32`.
  Input `(bs, 2*I)`, output `(bs, I)`, bf16. grid_dim=(num_tasks,1,1), block_dim=(128,1,1).
- **Matrix**: 2 configs × bs∈{1,2,4,8,16} = 10 configs. bs-only sweep (I is not TP-sharded).
- **Tolerances**: bf16 atol=rtol=1e-2 (torch.testing.assert_close). Observed max_diff ≤ 0.0625 (one
  bf16 ULP of a small-magnitude activation product) — within tolerance.
- **Fix applied**: none. The `num_tasks=32` for `I=2048` differs from the builder's actual
  `shared_split=16` but is harmless — the test creates random input with consistent num_tasks in both
  the kernel call and the reference, so the gate||up chunk boundary is self-consistent. The test
  validates the kernel's per-chunk silu(gate)*up computation across all batch sizes, not the specific
  interleave grid used in production weights.
- **Real kernel issues**: none. All 10 configs PASS on gpu0.

### sm100_elementwise_add (elementwise_add_layer) — 2026-06-09
- **Files**: `tests/runtime_python/blackwell/sm100_elementwise_add/test_elementwise_add_testmode.py`,
  `pytorch_reference.py` (defines `elementwise_add_ref`: `(a+b)` in f32 cast to bf16).
- **Shapes**: HIDDEN=7168, bf16, inputs `a,b: (bs,7168)`, output `(bs,7168)`.
  grid_dim=(bs,1,1), block_dim=(128,1,1) (mirrors builder).
- **Matrix**: bs∈{1,2,4,8,16} = 5 configs.
- **Tolerances**: bf16 atol=rtol=1e-2. Observed max_diff=0.0 (bit-exact on this dtype-preserving op).
- **Real kernel issues**: none. All 5 configs PASS on gpu0 (exact).

### sm100_embed (embed_layer) — 2026-06-09
- **Files**: `tests/runtime_python/blackwell/sm100_embed/test_embed_testmode.py`,
  `pytorch_reference.py` (defines `embed_ref`: flat-index gather from weight table).
- **Shapes**: VOCAB=129280, HIDDEN=7168, bf16. Token ids `(bs,1)` int64; output `(bs,7168)`.
  grid_dim=(56,1,1)=HIDDEN//128, block_dim=(128,1,1), input_source=1 (mirrors main DSV3 embed call).
- **Matrix**: bs∈{1,2,4,8,16} = 5 configs.
- **Tolerances**: byte-exact assertion (`torch.equal`). Embedding is a pure memory gather — no FP math.
  Observed max_diff=0.0 for all configs.
- **Decision**: `input_source=1` means the kernel reads token IDs from `task_desc->input_ptrs[0]`
  (the attached `input_ids` tensor), matching the production embed call. The reference also reads from
  the same tensor. No meta_tensor override needed in test_mode.
- **Real kernel issues**: none. All 5 configs PASS on gpu0 (byte-exact).

### sm100_tensor_init (tensor_init_layer) — 2026-06-09
- **Files**: `tests/runtime_python/blackwell/sm100_tensor_init/test_tensor_init_testmode.py`,
  `pytorch_reference.py` (defines `tensor_init_ref`: `torch.full_like(..., 0.0)`).
- **Shapes**: HIDDEN=7168, bf16. Pre-fills output with randn, then zero-fills it.
  grid_dim=(1,1,1), block_dim=(128,1,1), dummy_input_map=(-1,-1,-1), target_input_map=(-1,-1,-1)
  (mirrors builder). Kernel ignores `dummy` and vectorized-zero-fills `target`.
- **Matrix**: bs∈{1,2,4,8,16} = 5 configs.
- **Tolerances**: exact (atol=rtol=0 → assert output is identically zero). Observed max_diff=0.0.
- **Real kernel issues**: none. All 5 configs PASS on gpu0 (exact).

### sm100_linear / sm100_splitk_linear_bf16 (linear_layer, linear_with_residual_layer, splitk_linear_layer) — 2026-06-09

- **Files created**:
  - `tests/runtime_python/blackwell/sm100_linear/test_dsv3_linear_testmode.py`
  - `tests/runtime_python/blackwell/sm100_linear/test_dsv3_linear_with_residual_testmode.py`
  - `tests/runtime_python/blackwell/sm100_splitk_linear_bf16/test_dsv3_splitk_linear_bf16_testmode.py`
- **References reused**: `linear_ref`, `linear_with_residual_ref` (from `sm100_linear/pytorch_reference.py`);
  `splitk_linear_ref` (from `sm100_splitk_linear_bf16/pytorch_reference.py`). No new reference functions added.

#### linear_layer — 2 shapes, bs-only sweep

- `lm_head`: N=129280, K=7168. Non-vocab-parallel by default → N NOT TP-sharded → bs-only sweep.
  `grid_for_rmsnorm_linear_layer(129280)` → `129280//256 = 505` (size/96 > 400 path). block=(128,1,1).
- `router fallback`: N=256, K=7168. `min(grid_for_rmsnorm_linear_layer(256), 256//8) = min(64,32) = 32`.
  block=(128,1,1). (This path runs when `_BF16_GATE_SPLITK_ENABLED=False`.)
- **Matrix**: bs∈{1,2,4,8,16} × 2 shapes = 10 configs.
- **Observed max_diff**: lm_head ≤ 4.88e-4; router fallback ≤ 2.44e-4. All PASS.
- **Real kernel issues**: none. All 10 configs PASS on gpu0.

#### linear_with_residual_layer — down-proj BF16 fallback

- N=7168 (HIDDEN), K=18432 (dense MLP intermediate at TP=1). `grid_for_rmsnorm_linear_layer(7168)` = 64
  (7168%64=0). block=(128,1,1). N not TP-sharded in BF16 fallback → bs-only sweep.
- **Matrix**: bs∈{1,2,4,8,16} = 5 configs.
- **Observed max_diff** ≤ 2.0e-3. All PASS.
- **Real kernel issues**: none. All 5 configs PASS on gpu0.

#### splitk_linear_layer (BF16) — router gate shape

- N=256, K=7168. `_pick_bf16_splitk_factor` on B200 (148 SMs, 128 workers):
  `n_tiles=2`, `k_align=64`, `quotient=112`, `best_s=56` → grid=(2,56,1). block=(256,1,1).
- **accumulate=True matrix** (union-of-axes, no hang risk): bs∈{1,2,4,8,16} = 5 configs. All PASS.
  Observed max_diff ≤ 1.17e-2 (passes `atol+rtol*|ref|` with rtol=1e-2; raw max_diff exceeds
  atol=1e-2 at bs≥8 but passes the combined criterion).
- **accumulate=False matrix**: bs=16 only (bs<16 XFAIL_HANG — MMA_N=16 bug). bs=16 ran without
  hanging but produced WRONG output: **max_diff=3.824**. FAIL.
- **XFAIL_HANG**: bs∈{1,2,4,8} with acc=False not run (hang risk documented in builder lines 1526-1536
  and test_splitk_linear_bf16_accfalse_testmode.py).
- **Real kernel issue (REAL — affects production)**: `splitk_linear_layer(accumulate=False)` at bs=16
  runs to completion but produces incorrect results (max_diff=3.824, tol=1e-2). This is the SAME kernel
  path the DSV3 builder uses for the router gate (`_BF16_GATE_SPLITK_ENABLED=True`, `accumulate=False`).
  The existing standalone test previously documented a HANG for bs=1; at bs=16 the hang does not occur
  but correctness fails. Root cause: the `tensor_init` prepend zeroes the output before reduce-add but
  the split-K reduce-add appears to have a scheduling/TMA issue that produces garbage. The builder's
  gate-splitk path is therefore BROKEN for correctness at any tested bs when acc=False.
  **Impact**: Router gate logits incorrect for every MoE layer when `_BF16_GATE_SPLITK_ENABLED=True`
  (current default on feat/dpskv3). The `linear_layer` fallback path (`_BF16_GATE_SPLITK_ENABLED=False`)
  passes correctly at all bs values.

### sm100_fp8_gemm_dense (dense FP8 block-scaled GEMM family) — 2026-06-09

Covers `fp8_gemm_dense_{smallm,mediumm}_layer` (bf16 out), `fp8_gemm_dense_{smallm,mediumm}_fp8out_layer`
(fp8 out + UE8M0 scale), and `fp8_gemm_dense_decode_splitk_layer`. All run via PersistentKernel test_mode.

- **Files**:
  - `pytorch_reference.py` — NEW canonical home. Promoted the dense f32-block quantizers
    (`quantize_a_f32scale`/`quantize_b_f32scale`) + `reference_gemm` out of the inline test bodies; added
    `reference_gemm_f32`, `requantize_fp8out_ref` + `dequant_fp8out` (fp8out re-quantize ref),
    `reference_gemm_splitk` (split-K bf16-partial-accum ref), and `cosine_sim`/`rel_mean`. Reuses the shared
    UE8M0 `encode_ue8m0`/`decode_ue8m0`/`FP8_MAX` from `blackwell/common/sm100_fp8_scale_layout.py`
    (no scale logic duplicated).
  - `_build_helper.py` — re-exports the quantizers + `reference_gemm` from `pytorch_reference` for back-compat.
  - `test_fp8_gemm_dense_smallm_pk_testmode.py` — refactored to the swept DSV3-shaped smallm+mediumm matrix.
  - `test_fp8_gemm_dense_smallm_testmode.py` — refactored to import the canonical refs (was inline).
  - `test_fp8_gemm_dense_fp8out_pk_testmode.py` — NEW (fp8out variants).
  - `test_fp8_gemm_dense_decode_splitk_pk_testmode.py` — NEW (decode SplitK).
- **Kernel selection**: smallm vs mediumm picked by `max_seq_length<=512` (NOT M). Each (tp,bs) is run at
  `max_seq_length ∈ {256→smallm, 4096→mediumm}`. **TP is a SHAPE selector only** — `world_size=1`,
  per-rank-sharded N/K passed directly (no NVSHMEM; `PersistentKernel` would compile/init NVSHMEM at
  `world_size>1`). Default meta tensors (`prompt_lengths` = `max_num_batched_tokens` = bs, 1 request) give
  `active_rows = bs` → kernel `runtime_m = min(M, active_rows) = bs`; `runtime_m_mode=0` (no phase gate).
- **Shapes (DSV3, confirmed from builder.py)**: smallm/mediumm — gate_up `N=2·18432/tp, K=7168` (TP-sharded
  N: 36864/18432/9216/4608), qkv_a `N=2176, K=7168` (N unsharded), kv_b `N=4096, K=512` (small-K=512).
  fp8out — q_b_nope `N=(128/tp)·128, K=1536` (16384/8192/4096/2048). decode_splitk O_proj — `N=HIDDEN=7168,
  K=(128/tp)·128` (o_proj_original weight = hidden×H·128, so K=num_local_q_heads·128).
- **Matrix run (gpu0, single `test-on-gpu` invocation, 48 compiles total)**:
  - smallm/mediumm: union-of-axes `{tp=1}×{bs=1,2,4,8,16} ∪ {bs=16}×{tp=2,4,8} ∪ {tp=8,bs=1}` on gate_up
    × {msl 256, 4096} (18) + qkv_a & kv_b at bs∈{8,16} × {msl 256,4096} (4) = **22 configs, ALL PASS**
    (cos=1.0, rel=0.000% every config; max_abs_diff ≤ 2.0 = a few bf16 ULP at the large gate_up magnitudes).
  - fp8out: same union over q_b_nope × {msl 256,4096} = **18 configs, ALL PASS**. `scale_match=True`
    (n_diff=0 — UE8M0 exponent byte bit-exact vs ref on every group), cos=1.0, rel_vs_ref=0.000%,
    rel_vs_f32 ≈ 2.19–2.26% (e4m3 quantization error, well within 5%).
  - decode_splitk: union capped at bs≤8 (decode gate) at **split_k=2** = **8 configs, ALL PASS**
    (cos=1.0, rel=0.000%). The prepended `tensor_init` correctly zeroes a 7.0-prefilled output before the
    bf16 reduce-add (verified — no stale data; zero_rows=0).
- **fp8out scale layout (confirmed from kernel + task_register)**: output scale is flat uint32 `[bs, N/128]`
  row-major, `scale_outer_stride = N/128`; each entry's low 8 bits = `encode_ue8m0(per-128-N-group-max/448)`,
  upper 24 bits zero. Compared on the CPU int64 low-byte view (torch has no CUDA-uint32 `bitwise_and`).
- **Tolerances NOT loosened**: fp8 GEMM passes cosine>0.99 with huge margin (cos=1.0 throughout); fp8out
  dequant rel ≈ 2.2% ≪ 5%; scale compared bit-exact. No tolerance was widened to mask anything.
- **Real kernel issue found**: `fp8_gemm_dense_decode_splitk_layer` crashes for `split_k≥4` (multi-wave) —
  see Known kernel issue #2 above. Verified at split_k=2 (single-wave, correct); split_k=4 (DSV3 default)
  and split_k=8 crash with `cudaErrorLaunchFailure` at every TP/K. Localized via a split_k sweep; reported,
  not fixed (kernel fix out of scope). The other two variants (smallm/mediumm, fp8out) have NO issues.
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_fp8_gemm_dense.sh`
  (runs all 3 test files; OVERALL: `SMALLM_MEDIUMM_RC=0 FP8OUT_RC=0 DECODE_SPLITK_RC=0`).

### sm100_linear_fp8 (linear_fp8_layer + linear_fp8_with_residual_layer, non-swapAB) — 2026-06-09

- **Files**: `tests/runtime_python/blackwell/sm100_linear_fp8/test_linear_fp8_testmode.py` (refactored to
  the swept DSV3-shaped `_run_case(tp,bs)` + `MATRIX` + `test_*()` idiom),
  `test_linear_fp8_with_residual_testmode.py` (smoke + documented `HANGING_MATRIX`). Reuses
  `linear_fp8_ref` / `linear_fp8_with_residual_ref` from this folder's `pytorch_reference.py` (unchanged)
  and the shared UE8M0 `quantize_to_fp8_deepgemm_style` from `blackwell/common/sm100_fp8_scale_layout.py`
  (no scale logic duplicated). The residual file imports `MATRIX`/helpers from the no-residual file.
- **Shape**: DSV3 dense-MLP **gate_up** projection — N = `2*INTERMEDIATE_SIZE/tp` (TP-sharded:
  36864/18432/9216/4608 for tp=1/2/4/8), K = HIDDEN_SIZE = 7168 (not sharded). A projection whose output
  dim N shards by world_size, as requested. Decode-only → bs capped ≤16. `grid_x` mirrors production via a
  local copy of builder `_fp8_linear_grid_x` (largest divisor of N/128 ≤ num_workers; per-task N stays a
  multiple of 128). On B200: num_workers=136 → grid_x=96/72/72/36, per-task N=384/256/128/128.
- **Scale layout (confirmed from builder `_fp8_buffers_for_reduction` L501-510 + kernel)**: input scale
  (SFA) is `(packed_k=14, aligned_batch)` uint32 row-major contiguous (col-major UE8M0 stored transposed);
  weight scale (SFB) is deepgemm col-major logical `(N, packed_k)` stride `(1, aligned_N)`. K=7168 →
  logical_scale_k=56 → packed_k=14. `_input_scale_for_mpk` builds the `(packed_k, aligned_batch)` SFA;
  `_input_scale_dequant_view` reinterprets it for the reference's `dequant_from_packed_ue8m0`.
- **TP is a SHAPE selector only**: `world_size=1`, per-rank-sharded N passed directly (no NVSHMEM).
- **linear_fp8 matrix** (union-of-axes, bs≤16): `{tp=1}×{bs=1,2,4,8,16} ∪ {bs=16}×{tp=2,4,8} ∪ {tp=8,bs=1}`
  = **9 configs, ALL PASS** (cos=1.0, rel_mean ≤ 0.0001%, max_abs_diff ≤ 0.002 = a couple bf16 ULP at the
  gate_up magnitudes). Tolerance NOT loosened.
- **linear_fp8_with_residual**: the same DSV3 union-of-axes matrix **HANGS at every config** — real kernel
  deadlock, see Known kernel issue #3 above (fully localized by a (grid_x, per-task-N) sweep). A hanging
  config can't be a runtime xfail, so the matrix is recorded as `HANGING_MATRIX` (documentation only, not
  executed) and the test asserts on a residual **SMOKE** (bs=1, N=768, K=7168, grid_x=2 = 48 total epilogue
  iters, in the proven-PASS region) → **PASS, cos=1.0, rel=0.0%**. This validates the residual GEMM+add
  math, UE8M0 scale layout, and harness without tripping the hang.
- **Real kernel issue found**: `linear_fp8_with_residual_sm100` deadlocks at DSV3 scale (issue #3). The
  non-residual `linear_fp8_sm100` has NO issues (9/9 at production grid). Note: in the current builder the
  residual layer is reachable only via `_fp8_linear_prequantized`, which is never called → default demo
  unaffected.
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_linear_fp8.sh`
  (no-residual full matrix + residual smoke; OVERALL: `NORES_RC=0 RES_RC=0`).

### sm100_linear_fp8_bmm (linear_fp8_bmm_sm100 + linear_fp8_bmm_dense_sm100 + assemble_q_decode_sm100) — 2026-06-09

Covers the three layers of the DSV3 decode BMM path (builder `_bmm_decode_q_path` L1300/L1312 and
`_bmm_decode_o_path` L1400/L1433): per-head FP8 batched matmul (UE8M0 swapAB + dense f32-scale) and the
nope|pe assemble. All run via PersistentKernel test_mode on gpu0.

- **Files** (this folder): `pytorch_reference.py` (UE8M0 + dense-f32 per-head BMM refs, reusing shared
  `quantize_to_fp8_packed_ue8m0`/`dequant_from_packed_ue8m0`), `test_linear_fp8_bmm_testmode.py` (UE8M0),
  `test_linear_fp8_bmm_dense_testmode.py` (dense), `test_assemble_q_decode_testmode.py`. Run script
  `.scratch/run_bmm.sh` (relative paths, no cd to shared FS; per-test logs in `output/bmm_*.log`,
  `output/assemble_q.log`). **No test fixes were needed** — the previously-written files compiled, ran, and
  compared correctly on the first verified run. **No reference functions changed.**
- **Dims confirmed against builder.py**: Q-up (`kv_b_k` absorption) input `(bs,Hl,128)`→`(bs,Hl,512)`,
  weight `(Hl,512,128)`, Din=128/Dout=512, grid=(4,Hl,1); BMM2 (`kv_b_v` o-unabsorb) `(bs,Hl,512)`→
  `(bs,Hl,128)`, weight `(Hl,128,512)`, Din=512/Dout=128, grid=(1,Hl,1). `Hl=128//tp`=128/64/32/16.
  assemble_q: nope 512 ∥ pe 64 → 576, grid=(N,1,1); production uses `pe_only=True`. TP is a SHAPE selector
  only (`world_size=1`, per-rank Hl passed directly; no NVSHMEM). bs capped ≤16 (decode-only swapAB).
- **Scale layouts**: UE8M0 BMM uses row-major packed UE8M0 per (token,head) / (head,d_out) row (Din=128→
  packed_K=1, Din=512→packed_K=1 i.e. 4 logical 128-K scales in one uint32) — matches `linear_fp8_bmm_sm100`
  swapAB. Dense BMM uses plain float32 scales: input `[N,Hl,nk]` (nk=Din/128=4, per-head row stride Hl*nk),
  weight `[Hl,D_out/128=1,nk]` 128×128-block — matches `linear_fp8_bmm_dense_sm100` (grid.x=1).
- **Matrices run (gpu0, one `test-on-gpu` invocation = `.scratch/run_bmm.sh`)**:
  - **UE8M0 BMM — 18/18 PASS**: union-of-axes `{tp=1}×{bs=1,2,4,8,16} ∪ {bs=16}×{tp=2,4,8} ∪ {tp=8,bs=1}`
    (9 configs) × {qup Din=128/Dout=512, bmm2 Din=512/Dout=128}. cos=1.000000, rel-mean=0.0000%,
    max-abs ≤ 0.0010 (a couple bf16 ULP) every config.
  - **Dense f32 BMM — 9/9 PASS**: same 9-config union on bmm2 (Din=512/Dout=128, grid.x=1). cos=1.000000,
    rel-mean ≤ 0.0001%.
  - **assemble_q_decode — 24/24 PASS**: full-mode H∈{128,64,32,16}×N∈{1,2,4,8,16} (20) + pe_only spot-check
    at bs16 each H (4). Pure bf16 copy → byte-exact, max-abs = 0.000000 every config; pe_only verified to
    write only the [512:576] tail and preserve the nope sentinel. 0 FAIL.
- **Tolerances**: UE8M0/dense fp8 used cos>0.99 OR rel≤5% (decision-log fp8 spec) — passed with cos=1.0,
  huge margin (the reference dequants the same fp8 bytes the kernel consumes, so the GEMM is numerically
  near-identical to the reference einsum). assemble_q byte-exact (max-abs<1e-6). **No tolerance loosened.**
- **Real kernel issues**: **none.** All 51 configs (18+9+24) PASS.
- **Workflow note**: gpu0's CPU was heavily contended by another user's 10-hour, 10-way-parallel
  `cicc`/codex agent batch, which starved nvcc and stretched the 24-config assemble_q compile loop to ~30+
  min. The run still progressed correctly (per the decision-log "external user may slow nvcc — still
  progresses"). Verify completion via the test's own `N/N PASS` summary line in `output/*.log`
  (`grep -a`, binary-safe), not via `pgrep` over ssh — under this load ssh `pgrep`/`ps` polls flake and
  Python's stdout to the redirected log lags several configs behind actual progress.
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_bmm.sh`
  (OVERALL: `UE8M0_RC=0 DENSE_RC=0 ASM_RC=0`).

### sm100_fp8_group_gemm_decode (fp8_group_gemm_layer → smallm/largem_compact) — 2026-06-09

The permuted grouped FP8 GEMM (DSV3 NEW-MoE W13/W2, `MPK_DSV3_NEW_MOE=1`): permuted FP8 activations +
transposed UE8M0 SFA, per-expert FP8 weights + transposed UE8M0 SFB, `m_indices` selecting one expert per
BM=128 row block. Run via PersistentKernel test_mode on gpu0 (the production `fp8_group_gemm_layer` call,
which auto-dispatches smallm for `K>4096 & MPE<=8` else largem_compact).

- **Files**: `tests/runtime_python/blackwell/sm100_fp8_group_gemm_decode/pytorch_reference.py` (NEW),
  `test_fp8_group_gemm_decode_testmode.py` (NEW). The folder previously had only `test_wrapper.py`
  (kernel-wrapper for the NON-compact `fp8_group_gemm_largem_sm100`, with its own inline reference and the
  `round`-based UE8M0); it was left untouched (it tests a different kernel + build path; refactoring it is
  out of scope). `.scratch/probe_sanitizer.sh` retained as the compute-sanitizer reproducer for the kernel
  issue below.
- **SFA/SFB layout (built directly, not via moe_permute)**: driving the test through
  `moe_permute_sm100_layer` would require the full topk/routing_indices/meta/active-skip machinery — too
  much coupling for a GEMM unit test. Instead built the transposed `(num_sf_k, dim)` UE8M0-packed scales
  directly with `pytorch_reference.pack_sf_transposed`, which is **identical to builder._pack_moe_scale_ue8m0
  / test_wrapper.pack_sf** (the layer's own SFA/SFB contract). The per-128-K-block UE8M0 quantization
  REUSES the shared `encode_ue8m0`/`decode_ue8m0` (CEIL convention, matching the kernel) from
  `blackwell/common/sm100_fp8_scale_layout.py` (vectorized for speed; an import-time `_assert_vec_matches_
  shared()` proves bit-identity to the scalar helpers). The reference dequants the SAME fp8 bytes + SAME
  decoded UE8M0 scales the kernel consumes → comparison is numerically near-exact (cos=1.0, rel=0.0%).
- **Reference** (`grouped_gemm_ref`): per BM=128 block, `expert = m_indices[bm*128]` (one expert id per
  block — the kernel's contract), accumulate `sum_ki (A_blk @ B[expert]_blk.T) * sa_dec * sb_dec` over the
  128-K-blocks. Matches the kernel's per-block-scaled accumulation exactly.
- **Matrix (13 configs, ALL PASS, cos=1.000000 rel=0.0000% every config; max_abs_diff ≤ 0.5 = a few bf16
  ULP)**, one `test-on-gpu gpu0` invocation:
  - **largem (W13 gate||up, K=7168, N=2·2048/tp)** at the kernel's design point **E=128, MPE=128
    (M_total=16384)**, tp∈{1,2,4,8} → N∈{4096,2048,1024,512}. 4/4 PASS.
  - **largem (W2 down, K=2048/tp, N=7168)** same E=128/MPE=128, tp∈{1,2,4,8} → K∈{2048,1024,512,256}. 4/4 PASS.
  - **smallm (decode niche, K=7168, MPE≤8)** — the **bs/decode axis** sweep: MPE∈{1,2,4,8} at E=32, tp=1
    N=4096 (M_total=32/64/128/256) + a tp=8 corner (MPE=8, N=512). 5/5 PASS.
- **EL / M_total sizing decision (LOGGED)**: the largem arm is FIXED at E=128, MPE=128 (not the swept
  bs→E mapping originally planned) because the largem_compact no-mask tile dispatch is hard-coded to 128
  local experts and OOBs for M_total<16384 — see **Known kernel issue #4** below (found by this test). E=128
  is the kernel's actual design point (header comment "M=16384") and production-faithful for ep≥2
  (num_local_experts=128). The bs axis is therefore swept through the smallm kernel, which is M_total-driven
  and correct for any E/MPE. Full ep=1 EL=256 (M_total=32768) is both intractable per-config AND beyond the
  kernel's 128-expert scan cap.
- **Tolerances NOT loosened**: grouped fp8 spec is cos>0.99 OR rel≤5%; passed with cos=1.0 / rel=0.0% and
  huge margin (reference consumes identical fp8 bytes + UE8M0 scales).
- **Real kernel issue found**: `fp8_group_gemm_largem_compact_sm100` no-mask dispatch hard-coded to 128
  experts → OOB `m_indices` read for M_total<16384, and silently drops experts 128-255 even with a mask at
  num_local_experts>128 (ep=1). See **Known kernel issue #4** (compute-sanitizer evidence: .cuh L327). The
  smallm kernel has NO such issue. Reported, not fixed (kernel fix out of scope).
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_group_gemm_full.sh`
  (OVERALL: `GROUP_GEMM_RC=0`; on-disk log line `ALL PASS (13/13)`).

### sm100_moe_sigmoid (moe_topk_sigmoid_routing_layer → topk_sigmoid_sm100) — 2026-06-09

DeepSeek-V3 group-aware sigmoid routing: sigmoid(logits)+bias → group top-2 sum → top-K groups →
top-K experts → gather unbiased sigmoid → normalize×scale. Outputs topk_weights (bs,8) f32,
routing_indices (EL,bs) int32 (1-indexed within local range), active_expert_ids/mask (EL+1,) int32
(compacted LOCAL active-expert IDs in [0,count) + count at [EL]). Run via PersistentKernel test_mode on gpu0.

- **Files**: `tests/runtime_python/blackwell/sm100_moe_sigmoid/test_topk_sigmoid_testmode.py` (refactored
  to the swept `_run_case(bs, ep)` + `MATRIX` + `test_*()` idiom), `pytorch_reference.py`
  (extended `moe_topk_sigmoid_routing_ref` with two new kwargs `local_expert_start` / `num_local_experts`
  for ep>1 — defaults `0` / `num_experts` keep the kernel-wrapper test `test_gate_topk_sigmoid.py`
  byte-identical). Run script `.scratch/run_topk_sigmoid.sh` (relative path; log `output/topk_sigmoid.log`,
  summary line `TOPK_SIGMOID_RC=0`). No reference rewrite — only additive ep slicing.
- **Per-token op → bs-only sweep, no TP head-sharding** (per the matrix policy: topk routing is bs-only).
  The DSV3 params are real: NUM_EXPERTS=256, num_experts_per_tok=8, n_group=8, topk_group=4,
  routed_scaling_factor=2.5, scoring=sigmoid, norm_topk_prob=True (the reference's /sum normalization).
  grid=(1,1,1), block=(256,1,1) → single-CTA FUSE_COMPACTION=true path (the decode/default), 8 warps ×
  1 row/warp, exactly mirroring the builder call site (builder.py L3545).
- **Secondary ep>1 variant** (requested): ep=2 → num_local_experts=128, and the test picks ep_rank=1 so
  `local_expert_start=128` (a NON-zero offset, to actually exercise the index shift). Verified the
  reference slices to local experts and 1-indexes within the local range EXACTLY as the kernel:
  selection + weight-normalization run over ALL 256 global experts (kernel accumulates `weight_sum` over
  every selected expert regardless of locality), then locality only (a) zeroes topk_weights slots whose
  selected expert is outside [start,end), (b) writes routing_indices at `expert-start` for local experts,
  (c) emits LOCAL active-expert IDs. The dropping IS visible: at bs=16, ep=2 active=41 (local) vs ep=1
  active=75 (global) — the kernel's `node_uses_expert = expert∈[start,end)` filter is honored.
- **Matrix (10 configs, ALL PASS)**, one `test-on-gpu gpu0` invocation:
  `{ep=1}×{bs=1,2,4,8,16}` (EL=256, start=0) ∪ `{ep=2}×{bs=1,2,4,8,16}` (EL=128, start=128).
  Every config: `topk_weights max_diff = 0.000000` (tol 1e-2), `routing_indices` 0 mismatched (exact int),
  `active_expert_ids` exact count+set match. Active counts (k/ref): ep=1 {8,15,28,48,75}; ep=2 {5,9,10,18,41}.
- **Tolerance NOT loosened**: topk_weights is an f32 output and BOTH the kernel and the reference do the
  sigmoid / group-routing / normalization in f32 (the bf16 input is up-converted identically), so the
  comparison is bit-exact (max_diff exactly 0.0 every config) — far inside the 1e-2 bf16 budget.
  routing_indices / mask compared as exact int (rtol=atol=0).
- **Tie-break note**: no tie-break ambiguity surfaced. The kernel breaks expert ties toward the LOWER
  global index (`other_max==max_val && other_expert<expert`) and selects groups by strict-`>` argmax from
  g=0 (lower-index-wins), which agrees with torch `topk`'s lower-index preference; with DSV3-scale randn
  logits + bias all group/expert scores are distinct so the paths never diverge (0 mismatches at every bs).
- **Real kernel issues**: **none.** All 10 configs PASS.
- **⚠️ Stale-build trap hit + fixed (workflow note)**: the codegen for this layer's `cta_idx`/`num_ctas`
  args (added to `topk_sigmoid_sm100.cuh` + `task_register.cc` in commit `49668dea`, 2026-06-09) is a
  `src/**` change. This workspace's `test-on-gpu` has NO build freshness gate, so the FIRST run failed at
  nvcc with `no instance of function template "topk_sigmoid_task_impl" matches` (the generated `test.cu`
  passed only 11 runtime args, missing the new `cta_idx`/`num_ctas`) — the installed native `.so` was
  stale. Fixed by the documented manual rebuild on gpu0
  (`uv pip install -e . --no-deps`), after which all 10 configs pass. NOT a test or kernel bug.
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_topk_sigmoid.sh`
  (OVERALL: `TOPK_SIGMOID_RC=0`; on-disk log line `10/10 PASS` / `ALL PASS`).

### sm100_fp8_moe (moe_w13_fp8_layer + moe_w2_fp8_layer → fp8_moe_group_gemm_sm100) — 2026-06-09

The OLD-MoE routed-expert FP8 group GEMMs (`use_fp8_experts` path, builder L3740/L3820, also the MTP
sibling L4445/L4488). Each token's `routing_indices[e,token]=topk_slot(1-indexed)` + `mask[0..count-1]=`
activated local-expert IDs / `mask[EL]=count` drive a per-(token,slot) GEMM against the routed expert's
weight. Run via PersistentKernel test_mode on gpu0.

- **Files**: `tests/runtime_python/blackwell/sm100_fp8_moe/test_moe_w13_fp8_testmode.py`,
  `test_moe_w2_fp8_testmode.py` (both refactored to the swept `_run_case(tp,bs,EL)` + union-of-axes
  `main()` + `test_*()` idiom), `pytorch_reference.py` (REUSED `moe_w13_fp8_ref`/`moe_w2_fp8_ref`
  unchanged — they already do the kernel's UE8M0 floor `(__float_as_uint(sf)>>23)&0xFF`; added shared
  `cosine_sim`/`rel_mean` metrics + lifted the previously-inlined `quantize_fp8_2d/3d` + round-robin
  `make_routing` out of the two test bodies so they aren't duplicated). Run script `.scratch/run_moe_fp8.sh`
  (relative paths; logs `output/moe_w13_fp8.log`, `output/moe_w2_fp8.log`).
- **Shapes (DSV3, ep=1 → routed_tp=world_size)**: W13 gate||up — input fp8 `(bs,HIDDEN=7168)`+f32 scale,
  weight `(EL, 2·2048/tp, 7168)`+f32 scale, output bf16 `(bs,8, 2·2048/tp)`; N = 4096/2048/1024/512 for
  tp=1/2/4/8. W2 down — input fp8 `(bs,8, 2048/tp)`+scale, weight `(EL, 7168, 2048/tp)`+scale, output bf16
  `(bs,8,7168)`; K(reduction) = 2048/1024/512/256, N=7168 (unsharded). The layer takes a **float32**
  per-128-block scale (the kernel does the UE8M0 conversion internally), NOT a packed UE8M0 scale — so this
  unit uses plain `quantize_fp8_{2,3}d` f32 scales (the shared packed-UE8M0 helper does not apply here; the
  reference reproduces the same exponent-bit floor).
- **TP is a SHAPE selector only** (`world_size=1`, per-rank N(W13)/K(W2) passed directly; no NVSHMEM).
- **Grid mirrors the builder**: W13 `(_moe_expert_grid_x(bs,EL,pref=8)=min(8,bs·8), _moe_fp8_m_split(N,16),
  1)`; W2 `(min(10,bs·8), _moe_fp8_m_split(7168,14)=14, 1)`; `block=(256,1,1)` — the kernel **hard-requires
  8 warps** (warps 0-3 epilogue / 4 MMA / 5 DMA / 6 scale; .cuh L89). `m_split` adapts with N (W13: 16/16/8/4
  as N=4096/2048/1024/512; W2: 14 since 7168/14=512=4·MMA_M). `expert_offset=bid.x`, stride=grid.x.
- **EL reduction (LOGGED)**: real ep=1 has EL=256 (a (256,4096,7168) fp8 W13 weight ≈7.5 GB of which only
  the round-robin-activated experts are read). This kernel reads `mMask(EL)` directly and strides the
  activated list — it has **NO 128-expert scan cap** (unlike the decode `largem_compact` kernel, Known issue
  #4 — that is a different kernel/path). Kernel correctness is per-expert independent, so EL is reduced to
  **EL=64** for the primary sweep (matches the prior test + sm100_fp8_group_gemm_decode) with real per-expert
  N/K and a realistic activated-expert count (round-robin gives 8 at bs=1, 16 at bs=2, 32 at bs=4, 64 at
  bs≥8 — printed as `active_experts=`). Each token routes to 8 **distinct** experts.
- **Secondary ep>1 check (requested)**: ep=2 → num_local_experts=256/2=**128** at tp=1, bs=16 (W13 weight
  (128,4096,7168)≈3.75 GB) → `active_experts=128`, PASS for both W13 and W2 (production-faithful for ep≥2).
- **Matrix (per layer: union-of-axes `{tp=1}×{bs=1,2,4,8,16} ∪ {bs=16}×{tp=2,4,8} ∪ {tp=8,bs=1}` (9) +
  ep=2 (1) = 10 configs), ALL PASS**, one `test-on-gpu gpu0` invocation (20 megakernel compiles):
  - **W13 — 10/10 PASS** (cos=1.000000, rel≤0.0001%, max_abs_diff ≤ 0.001 = a couple bf16 ULP):
    tp1 bs{1,2,4,8,16} N=4096 grid=(8,16,1); tp2 bs16 N=2048 (8,16,1); tp4 bs16 N=1024 (8,8,1);
    tp8 bs16 N=512 (8,4,1); tp8 bs1 N=512 (8,4,1); **ep2** tp1 bs16 EL=128 N=4096 (8,16,1).
  - **W2 — 10/10 PASS** (cos=1.000000, rel=0.0000%, max_abs_diff=0.0000): tp1 bs{1,2,4,8,16} K=2048
    grid=(8 or 10,14,1); tp2 bs16 K=1024; tp4 bs16 K=512; tp8 bs16 K=256; tp8 bs1 K=256; **ep2** tp1 bs16
    EL=128 K=2048 (10,14,1).
- **Tolerances NOT loosened**: fp8 MoE spec is cos>0.99 OR rel≤5%; passed with cos=1.0 / rel≈0% and a huge
  margin because the reference dequants the SAME fp8 bytes through the SAME exponent-bit UE8M0 floor the
  kernel applies, so the GEMM is numerically near-identical to the reference matmul.
- **Real kernel issues**: **none.** All 20 configs (10 W13 + 10 W2) PASS. (This OLD-MoE kernel is distinct
  from the NEW-MoE `fp8_group_gemm_largem_compact` of Known issue #4 — no expert-count cap here.)
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_moe_fp8.sh`
  (OVERALL: `W13_RC=0 W2_RC=0`; each on-disk log ends `ALL PASS (10/10)`).

### sm100_moe glue (moe_silu_mul_layer + moe_mul_sum_add_layer + permute<->unpermute roundtrip) — 2026-06-09

The three BF16 MoE "glue" layers on the DSV3 forward path. All run via PersistentKernel test_mode on gpu0.

- **Files**: `tests/runtime_python/blackwell/sm100_moe/test_moe_silu_mul_testmode.py`,
  `test_moe_mul_sum_add_testmode.py` (both refactored from single-config to swept `_run_case` + `MATRIX` +
  `main()`/`test_*()` idiom), `test_permute_roundtrip_testmode.py` (the `@pytest.mark.skip` SKELETON —
  IMPLEMENTED). REUSED `moe_silu_mul_ref` / `moe_mul_sum_add_ref` from this folder's `pytorch_reference.py`
  unchanged, and the shared UE8M0 `quantize_to_fp8_deepgemm_style` / `packed_scale_k_for_reduction_size`
  from `blackwell/common/sm100_fp8_scale_layout.py` (permute's FP8 input). No reference functions added/changed.
  Run scripts `.scratch/run_moe_glue.sh` (all three) and `.scratch/run_perm.sh` (roundtrip only).
- **moe_silu_mul** — DSV3 OLD-MoE 3D path: input `(bs, TOPK=8, 2·I)` gate||up → output `(bs, 8, I)`,
  I = MOE_INTERMEDIATE(2048)/routed_tp (ep=1 → routed_tp=world_size). grid=(bs,8,1), block=(256,1,1) (mirrors
  builder L3768/L4473). I is TP-sharded as a SHAPE selector, so swept over BOTH the routed_tp shard and bs:
  union-of-axes `{rtp=1}×{bs=1,2,4,8,16} ∪ {bs=16}×{rtp=2,4,8} ∪ {rtp=8,bs=1}` = **9/9 PASS** (max_diff ≤
  0.0625 = ≤1 bf16 ULP of the small activation product). bf16 atol/rtol=1e-2.
- **moe_mul_sum_add** — HIDDEN=7168 (NOT sharded) → bs-only sweep `{1,2,4,8,16}` = **5/5 PASS** (max_diff
  0.03125). inputs moe_down_out `(bs,8,7168)`, topk_weights `(bs,8)` f32 (built as DSV3-style normalized×2.5
  sigmoid weights — the ref just consumes whatever weights are passed), shared_residual `(bs,7168)`. grid=(bs,
  _moe_hidden_split(7168,56)=56, 1), block=(128,1,1) (mirrors builder L3888/L4564). bf16 atol/rtol=1e-2.
- **permute<->unpermute roundtrip** — the IMPLEMENTED skeleton. Chains BOTH kernels in ONE graph:
  `tensor_init`(meta, zero) → `moe_permute_sm100_layer` (writes `meta`) → `moe_unpermute_sm100_layer` (reads
  `meta` + a KNOWN per-row `permuted_output` marker + `shared_residual`).
  - **Meta decode (how, exactly)** — the test generates a KNOWN routing in Python (each token picks TOPK=8
    distinct local experts; `routing_indices[e,t]=k+1` 1-indexed slot, `topk_weights[t,k]` the weight — the
    `topk_sigmoid` output contract, verified at `topk_sigmoid_sm100.cuh` L441-444). The Python reference
    decodes it IDENTICALLY to the permute kernel's deterministic token-order ballot scan
    (`moe_permute_sm100.cuh` L156-234): for each expert e, scan tokens t=0..num_active in increasing index,
    each routed token claims the next permuted row `e*BM_PADDING + slot` (slot = running count), and sets
    `tok_to_perm[t,k]=row+1`, `perm_weight[row]=topk_weights[t,k]`. unpermute then does
    `out[t]=residual[t]+Σ_k weights[t,k]·permuted_output[tok_to_perm[t,k]-1]`. The test ASSERTS the
    kernel-written `meta` (token_to_permuted int + permuted_weights f32-bits, decoded from
    `meta[0,0:M_TOTAL]` / `meta[0,M_TOTAL:M_TOTAL+MBT·TOPK]`) is BIT-EXACT to the Python decode
    (`t2p=True w=True` every config) — i.e. the token↔permuted-row mapping is proven identical, then the
    bf16 recombination is compared (uses a random bf16 `permuted_output`; the ref recombines the SAME bf16
    rows, so it's the exact inverse).
  - **num_active_rows** — set via a single prefill request `prompt_lengths=[MBT]` (+ `max_num_batched_tokens
    =max_seq_length=max_num_pages=MBT`). `prepare_next_batch` then writes `qo_indptr_buffer
    [MPK_MAX_NUM_BATCHED_REQUESTS]=MBT` (persistent_kernel.cuh L398-401), which both kernels read as
    num_active_rows → permute scans all MBT tokens, unpermute writes all MBT output rows.
  - **EL reduction (LOGGED)** — real DSV3 ep=1 has E_LOCAL=num_local_experts=256 → M_TOTAL=256·128=32768
    permuted rows. The round-trip is correctness-only and per-row independent, so E_LOCAL is reduced to small
    values (≥ TOPK so a token's 8 experts get distinct local slots) to keep M_TOTAL / compile time tractable.
    Matrix = `{(E_LOCAL,MBT)}` = `(8,1),(8,4),(16,8),(16,16)` (M_TOTAL 1024/1024/2048/2048), HIDDEN real 7168.
    **4/4 PASS**, `t2p=True w=True` (bit-exact meta), output max_diff ≤ 0.0039 (≤ a couple bf16 ULP). bf16
    atol/rtol=1e-2.
  - permute's FP8 input (`input_fp8`+`input_scale`) is irrelevant to the recombination but required by the
    kernel; built with the shared deepgemm-style UE8M0 quantizer, whose `(MBT,K_PACKED)` stride-(1,aligned_MBT)
    tensor IS the column-major `[K_PACKED,MBT_ALIGNED]` byte layout the permute kernel reads
    (`in_scale[sf·MBT_ALIGNED+t]`, K=7168→K_PACKED=14). K_PACKED matches via `packed_scale_k_for_reduction_size`.
- **Tolerances NOT loosened**: all three are dtype-preserving bf16 ops; max_diff is ≤ a couple bf16 ULP
  everywhere (and the roundtrip meta mapping is asserted BIT-EXACT, not just within tolerance).
- **Real kernel issues**: **none.** All 18 configs (9 silu_mul + 5 mul_sum_add + 4 roundtrip) PASS.
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_moe_glue.sh`
  (OVERALL: `SILU_RC=0 MSA_RC=0 PERM_RC=0`; logs end `ALL PASS`, `9/9` / `5/5` / `4/4` PASS).

### sm100_mla rope (deepseek_mla_rope_q_fused + q_split + rope_k) — 2026-06-09

The three REAL MLA RoPE layers the DSV3 builder uses (builder.py L2646/2660/2672 prefill path +
MTP sibling L4108/4117/4125). REWROTE the old `test_deepseek_mla_rope_testmode.py` (which called the
DEPRECATED combined `deepseek_mla_rope_q_layer` at num_heads=4) to call the three real layers across
the DSV3 union-of-axes matrix. All run via PersistentKernel test_mode on gpu0.

- **Files**: `tests/runtime_python/blackwell/sm100_mla/test_deepseek_mla_rope_testmode.py` (rewritten to
  swept `_run_q_fused/_run_q_split/_run_rope_k(tp,bs)` + `MATRIX` + `main()`/`test_*()` idiom),
  `pytorch_reference.py` (ADDED `build_dsv3_yarn_rope_tables` + `rope_rotate_gptj` + the three `_yarn_*`
  helpers — the prior file had no rope ref; the old test had an inline `_rotate_gptj`). Run script
  `.scratch/run_rope.sh` (relative path; log `output/rope.log`, summary `ROPE_RC=0`).
- **Rotation convention = GPT-J / interleaved** (`is_neox_style=False`), matching the kernel
  `deepseek_mla_rope_sm100.cuh` L58-106 EXACTLY: for pair p (d0=2p, d1=2p+1) at position `pos`, read
  `c=cos[pos*64+d0]`, `s=sin[pos*64+d0]` (table is `repeat_interleave(2)`-d so cos[d0]==cos[d1]) and
  `out[d0]=x0*c-x1*s; out[d1]=x1*c+x0*s`, all in f32 then round to bf16. A cheap CPU/GPU element-by-element
  kernel-emulation self-check in the run script asserts `rope_rotate_gptj == hand-rolled kernel math`
  bit-exact (`set -e` gate) before any megakernel compile — catches a convention bug for free.
- **cos/sin source = the builder's own YARN table** (`build_dsv3_yarn_rope_tables` is a line-for-line copy of
  `builder._precompute_rope_embeddings`): DSV3 config.json values rope_theta=10000, qk_rope_head_dim=64,
  factor=40, beta_fast=32, beta_slow=1, mscale=mscale_all_dim=1.0, original_max_position_embeddings=4096,
  type=yarn. **YARN scale fact**: with mscale==mscale_all_dim==1.0 the effective table mscale RATIO is
  `_yarn_get_mscale(40,1)/_yarn_get_mscale(40,1) = 1.0` (NOT 1.369) — the YARN adjustment is entirely in the
  inv_freq extrapolation/interpolation mix, not an amplitude scale. The table is a SHARED input to both the
  kernel and the reference (read at the same `[pos*64+d0]`), so YARN lives in ONE place; correctness is
  decided by the rotation convention + per-variant slice layout (what these tests verify), not by
  re-deriving YARN twice.
- **phase_gate=0** for every config (deterministic — kernel always rotates, no Q_LEN gate). bs maps to a
  single prefill request of `seq_len=bs` tokens (`prompt_lengths=[bs]`); `step=0` → positions 0..bs-1.
- **qfused_mode**: q_split run BOTH as standalone mode=0 (`[bs, H*64]`, FUSED_HEAD_DIM template=64 → rotate
  each head's 64 at offset 0) AND the row-swap mode=1 (`[bs, H*192]`, `Q_ROW_STRIDE_OVERRIDE=H*192`,
  `Q_PE_BASE_IN_ROW=H*128`, `Q_PE_HEAD_STRIDE=64` → rotate pe slice at `[H*128 + head*64 : +64]`).
- **Slice layouts (verified against the kernel .cuh + codegen task_register.cc L6805-6951)**:
  q_fused rotates the TAIL 64 of each head's 576 (`row*H*576 + head*576 + 512`); q_split mode0 each head's
  64 (`row*H*64 + head*64`); q_split mode1 the row-swap pe block; rope_k the FIRST 64 of k_pe
  (`row*128`, K_PE_STRIDE=128, NUM_HEADS=1 → grid (req,1,1)). All buffers are the layer's `output_ptrs[0]`
  (rotated in place); `input_ptrs[1]`=cos, `[2]`=sin.
- **H is TP-sharded** (`num_local_q_heads = 128 // tp` → 128/64/32/16) and drives grid.y + per-head stride,
  so tp is a genuine shape axis → union-of-axes matrix `{tp=1}×{bs=1,2,4,8,16} ∪ {bs=16}×{tp=2,4,8} ∪
  {tp=8,bs=1}` = 9 configs each for q_fused / q_split0 / q_split1. rope_k is NOT head-sharded (NUM_HEADS=1),
  so it sweeps bs∈{1,2,4,8,16} only (5 configs).
- **Matrix (32 configs total, ALL PASS, max_diff=0.000000 / bit-exact EVERY config)**, one
  `test-on-gpu gpu0` invocation (32 megakernel compiles):
  - q_fused: tp1 bs{1,2,4,8,16} (H=128) + bs16 tp{2,4,8} (H=64/32/16) + tp8 bs1 (H=16) — 9/9.
  - q_split mode0 (standalone H*64): same 9-config union — 9/9.
  - q_split mode1 (row-swap H*192): same 9-config union — 9/9.
  - rope_k (first-64 of k_pe): bs{1,2,4,8,16} — 5/5.
- **Tolerance NOT loosened**: bf16 atol/rtol=1e-2, but observed max_diff is EXACTLY 0.0 on every config.
  Both the kernel and the reference up-convert the bf16 input to f32, do the identical c/s FMA, then round
  to bf16 with the same round-to-nearest — so the bytes match bit-for-bit (same family as the embed/topk
  bit-exact cases). The 0.0 is across the REAL kernel rotation (cos/sin ≠ 1), not a no-op: the standalone
  CPU/GPU self-check independently proves the rotation math, and a wrong slice/convention would diverge.
- **Real kernel issues**: **none.** All 32 configs PASS bit-exact.
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_rope.sh`
  (OVERALL: `ROPE_RC=0`; on-disk log line `ROPE SUMMARY: 32/32 PASS` / `ALL PASS`).

### sm100_mla kv-gather (mla_kv_gather + mla_kv_gather_split + mla_kv_gather_unified) — 2026-06-09

The three DSV3 MLA paged-KV gather layers (builder.py L2725 unified / L2808 basic; split is the
prefill-layout sibling). Each appends new c_latent+k_pe to the paged cache, then gathers the full
sequence. All run via PersistentKernel test_mode on gpu0.

- **Files**: `tests/runtime_python/blackwell/sm100_mla/test_mla_kv_gather_testmode.py` (refactored bs=1 →
  bs sweep), `test_mla_kv_gather_split_testmode.py` (refactored bs=1 → bs sweep),
  `test_mla_kv_gather_unified_testmode.py` (NEW). REUSED `mla_kv_gather_ref` (contiguous [S,D_K]) +
  `mla_kv_gather_split_ref` (separate ckv [S,D_V]/kpe [S,ROPE_DIM]) from this folder's
  `pytorch_reference.py` UNCHANGED. Run script `.scratch/run_mla_gather.sh` (relative paths; logs
  `output/mla_gather*.log`).
- **⚠️ KEY page-table fact (empirically verified, `.scratch/probe_gather_meta.sh` /
  `probe_unified.sh`)**: in test_mode the offline `prepare_next_batch` runs on iter 0 and RECOMPUTES
  `qo_indptr` / `paged_kv_indptr` / `paged_kv_indices` / `last_page_len` from `prompt_lengths` +
  `total_num_requests` (= `tokens.shape[0]`) + the page_queue — it CLOBBERS any user-passed page tables
  (the prior bs=1 tests passed explicit `qo_indptr=[0,8]` etc. that were silently overwritten; with the
  default `prompt_lengths = max_num_batched_tokens = 8` the kernel saw seq_len=8, NOT the reference's
  seq_len=100 — those refactors fix a latent mismatch). So **`prompt_lengths` is the single source of
  truth**: each request bi is a FRESH prefill (step 0) of length `prompt_lengths[bi]`, the kernel appends
  ALL of bi's tokens (kv_start_pos=0) and gathers them, and pages are allocated SEQUENTIALLY from the queue
  starting at 0 → `page_indices[bi] = range(sum(pages_j, j<bi), +ceil(L_bi/page_size))`, reconstructed in
  Python. Set `max_num_batched_tokens = sum(prompt_lengths)`, `max_num_batched_requests = bs`,
  `max_num_pages = sum(pages_per)+1`, `max_seq_length = max(prompt_lengths)`, and pass
  `tokens=(bs,max_seq)` + `prompt_lengths=(bs,)`. Metadata wiring (runtime.cc L482-496): gather
  `request_id=bid.x`; unified ALSO `kv_idx=bid.y` = the gather split index.
- **Branch gate (unified, task_register.cc L6707-6719)**: `prompt_prefill = request_ids[bi]>=0 &&
  step[bi]<prompt_length[bi] && q_len>8`. True → writes ckv_sep/kpe_sep (prefill layout), contiguous_kv
  untouched; False → writes contiguous_kv (decode layout), ckv_sep/kpe_sep untouched (kernel .cuh
  L242-265). In test_mode iter 0 always has step=0<prompt_length, so the branch is selected purely by
  `q_len = prompt_length > 8`. Drove BOTH branches via prompt length: **PROMPT_LEN=40 (>8) → prefill**,
  **PROMPT_LEN=8 (<=8) → decode**. The test also asserts the UNTOUCHED branch's output stays exactly zero
  (guards against a branch-selection regression).
- **num_gather_splits (unified)**: grid.y MUST equal num_gather_splits (Python-side assert); each CTA
  strides seq_pos by N_SPLITS in both append + gather (kernel .cuh L205/L236). Swept {1,2,4} — the strided
  partition reassembles the full sequence bit-exact at every split for BOTH branches.
- **Shapes**: D_K=576 (=512 ckv + 64 kpe), NOT head-sharded → **no TP axis** (bs = #requests is the
  relevant sweep, plus num_gather_splits for unified). page_size=16, seq_len 40 (prefill, 3 pages) / 8
  (decode, 1 page). c_latent contiguous [mbt,512]; k_pe padded [mbt,128] (real rope in first 64). Offsets:
  basic-gather contiguous output at `bi*S_*D_K`; split + unified outputs at `bi*MPK_MAX_SEQ_LENGTH*{D_K|
  D_V|ROPE_DIM}` (uniform max-seq stride). Equal per-request lengths keep the basic-gather `bi*S_` slabs
  non-overlapping.
- **Matrix (ALL PASS, max_diff EXACTLY 0.000000 every config — gather is a pure bf16 memory copy →
  bit-exact, the decision-log gather-only expectation)**, one `test-on-gpu gpu0` invocation (40 compiles):
  - **mla_kv_gather (basic)** — bs∈{1,2,4,8,16} = **5/5 PASS** (contiguous_kv + paged_cache append both 0.0).
  - **mla_kv_gather_split** — bs∈{1,2,4,8,16} = **5/5 PASS** (ckv_sep + kpe_sep + append all 0.0).
  - **mla_kv_gather_unified** — bs∈{1,2,4,8,16} × num_splits∈{1,2,4} × {prefill, decode} = **30/30 PASS**.
    prefill 15/15 (ckv_sep/kpe_sep correct, contiguous_kv stays zero); decode 15/15 (contiguous_kv correct,
    ckv_sep/kpe_sep stay zero); paged_cache append 0.0 every config.
- **Correctness-not-coincidence note**: a 0-diff is trusted because (a) the gathered rows are verified
  against a Python reference that uses the SAME reconstructed per-request page_indices + the SAME fresh-
  prefill token slice the kernel consumes (per the prompt's "verify the gathered rows correspond to the
  right (request, position)"); (b) the probe confirmed `ckv_sep nonzero rows == seq_len` and the decode
  probe confirmed `contiguous_kv nonzero rows == seq_len` with the opposite branch staying zero — so the
  kernel really wrote the full window in the right layout, not a no-op; (c) num_gather_splits>1 reassembles
  the strided partition correctly (a broken stride would drop/duplicate rows → nonzero diff).
- **Tolerances NOT loosened**: gather is dtype-preserving (bf16→bf16 uint4 copy), so the comparison is
  exact (`assert max_diff == 0.0`, not within a bf16 ULP) — there is no FP math to round.
- **Real kernel issues**: **none.** All 40 configs (5 + 5 + 30) PASS bit-exact.
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_mla_gather.sh`
  (OVERALL: `BASIC_RC=0 SPLIT_RC=0 UNIFIED_RC=0`; unified on-disk log line `ALL PASS (30/30)`, basic/split
  `ALL PASS (5/5)`).
- **Workflow note**: gpu0 CPU was contended by another user's `cicc`/codex batch (load avg up to ~16),
  stretching the 40-config compile loop; the run still progressed correctly (GPU 0% = nvcc compile phase,
  NOT a megakernel hang) and finished 40/40. Tracked completion via the on-disk `ALL PASS`/`OVERALL` log
  lines (binary-safe `grep -a`), not process polling, per the gather-folder execution rule.

### sm100_mla (mla_prefill_absorbed_layer → mla_prefill_absorbed_sm100) — 2026-06-09

The DSV3 absorbed/fused-format MLA prefill: causal multi-head latent attention over compressed KV, where
Q = `q_nope_pe [sum(S_i), H·576]` (per head nope[512]‖pe[64]), KV = `[B·max_seq, 576]` (per row
ckv[512]‖kpe[64]), V = ckv[512]. Reads prefill length from `qo_indptr` and KV length from the paged-KV
meta tensors; KV is read DENSELY at `bi·MPK_MAX_SEQ_LENGTH·576`. Run via PersistentKernel test_mode on gpu0.

- **Files**: `tests/runtime_python/blackwell/sm100_mla/test_mla_prefill_absorbed_testmode.py` (refactored
  to swept `_run_case(tp, prompt_lengths, tag)` + `MATRIX`/`XFAIL_MULTI_QBLOCK` + `test_*()`). REUSED
  `mla_prefill_ref` from this folder's `pytorch_reference.py` UNCHANGED (causal MLA over fused ckv‖kpe with
  V=ckv), called with the YARN scale. Also touched (REQUIRED to compile the layer — see Known issue #5
  "Separate fix applied"): `include/.../blackwell/mla_prefill_sm100.cuh` (restored `kpe_row_stride`/
  `kpe_offset` + 3 KPE sites) and `src/kernel/task_register.cc` (offset fused Q_pe base by +d_ckv). The
  `.cc` change needs a one-time `uv pip install -e . --no-deps` on gpu0 (no build gate here); the run script
  does it after the rsync. Probes: `.scratch/probe_mla_diag.py`, `.scratch/probe_mla_s64.py`.
- **Seq setup**: drive prefill length(s) via `prompt_lengths` ONLY — `prepare_next_batch` recomputes
  qo_indptr / paged_kv_* on iter 0 (manual values would be overwritten). `page_size = max_seq`,
  `max_num_pages = B` → one page per request, `S_ == prompt length`, dense KV slabs page-aligned. The
  kernel guards `Q_LEN<=8 return` (so S>8) and `step<prompt_length` (true on iter 0). H = 128//tp.
- **Scale**: YARN `(1/sqrt(192))·mscale²`, `mscale=0.1·ln(40)+1`, applied internally by the kernel; the
  reference uses the SAME scale + causal mask. **Causal**: single-chunk prefill (`Q_LEN==S` → `q_hist=0`),
  pure causal self-attention, matching the reference's `triu(diagonal=1)` mask. **B>1**: supported — each
  request is an independent causal slab; reference run per request then concatenated. Verified at
  S=[64,64] and [48,64].
- **Tolerance**: bf16 attention → `atol=rtol=5e-2` AND cos>0.99 (both required). Observed on the asserted
  matrix: max_diff ≤ 0.0228, mean_diff ≈ 6e-4, cos ≈ 0.9995 every config — far inside tolerance, NOT loosened.
- **Asserted matrix (6/6 PASS, deterministic)**, one `test-on-gpu gpu0` invocation: union-of-axes over H —
  `S=64 × tp∈{1,2,4,8}` (H=128/64/32/16) ∪ B>1 `tp=1 S=[64,64]`, `tp=2 S=[48,64]`. All single-q-block
  (S≤PF_BM=64), which is the kernel's deterministically-correct domain.
- **Real kernel issue found (REAL, NON-DETERMINISTIC) — see Known kernel issue #5**: for S>64 (more than
  one q-block) the kernel's first-row-of-q-block fix-up corrupts the row at each `q_start` (64,128,…)
  non-deterministically (run-to-run the bad (row,head) set changes; full matrix saw both max_diff≈0.3-0.4
  and NaN). Localized to exactly the `q_start` rows (every other row is bit-stable & correct). The S>64
  configs (`S128-H128`, `S256-H128`, `S128-H64`) are recorded as `XFAIL_MULTI_QBLOCK` (documentation only,
  NOT executed in the asserting set, NOT a tolerance loosening) so a flaky-wrong config doesn't gate CI.
  Reported, not fixed (kernel-correctness fix out of scope; the absorbed-prefill path is MTP-only,
  `--mtp>0`, default OFF → default verify unaffected).
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_mla_prefill_absorbed.sh`
  (rebuilds the native ext for the task_register.cc change, then runs; on-disk log line
  `=== mla_prefill_absorbed: 6/6 PASS ===`, `MLA_PREFILL_ABSORBED_RC=0`).

### sm100_mla_prefill_tp8_chunked (mla_prefill_tp8_chunked_layer → mla_prefill_tp8_chunked_sm100) — 2026-06-09

The DSV3 **chunked** prefill MLA attention: true-unabsorbed, per-head K/V. This is the LIVE default-demo
prefill attention (builder.py L2839, `_use_prefill` path), distinct from the MTP-only absorbed prefill of
issue #5. Per head: Q_nope[q,128]‖Q_pe[q,64] vs K_nope[kv,128]‖K_rope[kv,64](shared)‖V[kv,128]; causal
mask is w.r.t. the ABSOLUTE position (`j <= q_start + i`), q_start = chunk offset. Run via PersistentKernel
test_mode on gpu0. **NEW** test (this folder previously had only the CUDA-extension `test_chunked.py`).

- **Files**: `tests/runtime_python/blackwell/sm100_mla_prefill_tp8_chunked/pytorch_reference.py` (NEW —
  LIFTED `torch_reference` from `test_chunked.py` into `mla_chunked_prefill_ref`, plus `bare_sm_scale`/
  `yarn_sm_scale`), `test_chunked_testmode.py` (NEW). Probes: `.scratch/probe_chunked_nan.py` (NaN
  localization), `.scratch/probe_chunked_pad.py` (padding-fix proof). **No kernel-src change** (the chunked
  task was already registered; the test compiled & ran against the installed `.so` with no rebuild).
- **SCALE = BARE `1/sqrt(192)`, NOT YARN.** The chunked-prefill register
  (`register_mla_prefill_tp8_chunked_sm100_task`, task_register.cc L3988-3989) hardcodes
  `sm_scale = 1/sqrtf(192)` and does NOT apply the YARN `mscale²` (= `(0.1·ln(40)+1)²≈1.874`) factor that
  EVERY sibling MLA task uses (decode L3686, non-chunked tp8 prefill L3789, absorbed L4241, mtp L7115).
  **This contradicts the task brief's "use the YARN scale (kernel applies YARN internally)" premise — for
  THIS kernel the codegen passes the bare scale, so the reference MUST match bare or it mismatches by ~1.874
  inside softmax.** Flagged as a finding: the live DSV3 prefill attention uses a different softmax scale than
  its decode/absorbed/mtp siblings. (Note: the builder DOES bake `mscale≈1.369` into the RoPE cos/sin tables
  (`_rope_*`, builder L1871-1873), so the rope components are pre-amplified in production — a partial/
  different application than a uniform `mscale²` on all scores. The unit test feeds RAW random Q/K (no rope),
  so the only scale that matters is the bare one the kernel receives.)
- **KV BUFFER PADDING (required, production-faithful)**: K_nope/K_rope/V are padded to a BN=128-aligned row
  count (zero tail). Sizing them to exactly `kv_len` when `kv_len%128≠0` → ALL-NaN output (Known kernel
  issue #6, `0*NaN(V)` on the TMA-OOB-NaN-filled partial-block V rows). Padding mirrors the real DSV3 cache
  (sized to page-aligned `max_seq_length`); the reference attends only to the first `kv_len` rows (kernel
  masks `kvp>=kv_len`), so the zero tail is numerically inert.
- **LAYOUT/grid decisions**: B=1 (the K/V TMA descriptors carry NO batch coordinate — `tma3d(KN,0,kvb,
  head*2+half)`; one descriptor addresses one batch's KV; the kernel applies `bat` only to Q/O arithmetic).
  Tensors attached 3D `[kv_len_pad,H,128]` (K_nope/V → tma.cuh 3D branch, H_local=dim[1]), K_rope 2D
  `[kv_len_pad,64]` (param_id==3 → 2D branch), Q `[q,H,128]`/`[q,H,64]`, O `[q,H,128]`. grid=(H, ceil(q/64),
  1), block=(128,1,1). In test mode the codegen passes q_len/kv_len/q_start as LITERAL params (from
  `mla_params`), NOT from meta tensors — seq lengths driven directly; metadata maps `request_id=head`,
  `kv_idx=q_block`, `merge_task_offset=batch` (runtime.cc L417-420). q_start = kv_len−q_len (tail chunk).
- **NO issue-#5 analog**: this kernel has NO first-row-of-q-block fix-up hack (the absorbed kernel's
  non-deterministic `q_start` merge bug). It applies a clean causal mask (`kvp<=qp && kvp<kv_len`) and each
  q-block is an independent CTA (grid.y). Multi-q-block (q=128 → 2 q-blocks) is DETERMINISTIC + CORRECT
  (max_diff=0.000488, no NaN, repeatable). So multi-q-block is asserted, not XFAIL'd.
- **Matrix (11/11 PASS, cos=0.999998 every config, max_diff ≤ 0.001953 = a couple bf16 ULP)**, one
  `test-on-gpu gpu0` invocation (11 megakernel compiles): union-of-axes over H=128//tp × (q,kv) —
  H128: (q64,kv64)[partial-block, kv<BN], (q64,kv128), (q64,kv192)[partial last block], (q64,kv256),
  (q128,kv128)[multi-q-block], (q128,kv256)[multi-q-block+chunked]; ∪ chunked (q64,kv256) at H={64,32,16};
  ∪ corner (q64,kv64) at H16; ∪ fused-Q spot check (q64,kv256, qfused_mode=1, row-swap [all_nope‖all_pe]).
- **Tolerance NOT loosened**: bf16 attention spec atol/rtol=3e-2 (the ext test's own atol) AND cos>0.99 AND
  no NaN; observed max_diff ≤ 0.00195 and cos=0.999998 every config — orders of magnitude inside tolerance.
  The reference dequants the same bf16 Q/K/V and does the identical bare-scale causal softmax, so the only
  divergence is f32-vs-MMA softmax accumulation order (a few bf16 ULP).
- **Real kernel issue found**: **Known kernel issue #6** — all-NaN for unaligned-`kv_len` exact-sized K/V
  buffers (`0*NaN`). The default demo is unaffected (production sizes KV to page-aligned max_seq_length).
  Reported, not fixed. NaN-masking aside, the kernel is numerically correct across all H / chunked / multi-
  q-block / fused-Q configs.
- **Final command**:
  `MIRAGE_SRC=/mnt/shared/zepengz/projects/mirage_dpskv3 test-on-gpu gpu0 .scratch/run_chunked.sh`
  (on-disk log line `=== mla_prefill_tp8_chunked: 11/11 PASS ===`, `CHUNKED_TESTMODE_RC=0`).
