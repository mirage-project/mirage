# DeepSeek V3 on B200: FP8 + MLA TP + MLA prefill + MTP

This PR brings the DeepSeek V3 FP8 inference pipeline up on B200 (SM100a) inside
MPK, adds MLA tensor parallelism (TP2/TP4/TP8), MLA chunked prefill, and MTP
speculative decoding. Demo + persistent-kernel infrastructure are cleaned up so
only the proven-working configuration ships.

## What's new

### 1. DeepSeek V3 end-to-end on B200
- FP8 MoE group GEMM on sm_100a (`fp8_group_gemm_sm100.cuh`) with 4-warp
  specialization (DMA / MMA / Scale / Epilogue) using tcgen05 + UTCCP for
  scale factors.
- MLA decode + reduce kernels (`mla_decode_sm100`, `mla_reduce_sm100`),
  MLA chunked prefill (`mla_prefill_sm100`), MLA absorbed-Q weight layout
  (`q_b_proj` is `[H*576, q_lora]`, decode consumes fused, prefill splits).
- Dense FP8 linear (`linear_fp8_sm100`) + fused residual variant
  (`linear_fp8_with_residual_sm100`) matching vLLM's fused residual semantics.
- DeepSeek V3 builder (`python/mirage/mpk/models/deepseek_v3/builder.py`) wiring
  61 MoE layers + MLA + MTP layer + shared/routed experts + LM head + AllReduce.
- Demo (`demo/deepseek_v3/demo.py`) with `--layers` selective weight loading
  (HuggingFace index-based) so you can run a prefix like `--layers 0-10` on a
  single GPU without paying the full 61-layer weight load.

### 2. MLA TP (2 / 4 / 8)
- `mla_mtp_decode_tp{2,4,8}_sm100.cuh` — MLA TP decode kernels that issue
  `tcgen05.alloc` only from warp 0 and use a named barrier `bar.sync 1, 128`
  for the active-thread sync. This is required: CuTe's TMEM allocator demands
  the same warp does alloc + dealloc, and Independent Thread Scheduling
  otherwise causes warp drift.
- TP sharding rules live in two places (`builder.py` SHARD_RULES + demo
  state-dict conversion pass). Documented in `CLAUDE.md`.

### 3. MLA chunked prefill (default-on)
- `mla_prefill_sm100` dispatched automatically when `Q_LEN >= 32`
  (prefill routing in `builder.py`). Uses 576-d fused head; splits `q_b` into
  `q_b_nope[512]` + `q_b_pe[64]` in demo.py's Phase-2 absorption.
- Prefill kernel uses the `lp_ - fp_ - 1` page-indptr formula inside the
  task-register snippet so chunked prefill works with a statically-compiled
  task graph (see `src/kernel/task_register.cc`).

### 4. MTP speculative decoding (vLLM-aligned)
- Demo flag: `--mtp N` (int, 0-3; 0 = disabled, N > 0 = draft length).
- Prefill step for the MTP layer uses ground-truth prompt tokens shifted by
  one (matches `vllm/v1/spec_decode/eagle.py` L666-669), not argmax of the
  main model's logits. New task: `TASK_MTP_BUILD_EMBED_INPUT = 294`, kernel
  `mtp_build_embed_input_kernel` in `mtp_token_ops.cuh`.
- Draft→verify loop with strict / probabilistic / synthetic rejection sampling.
- MTP decoder uses the fused `linear_fp8_with_residual_sm100` kernel for
  its residual add (previous revision used a separate elementwise_add followed
  by AllReduce, which caused an `illegal memory access` on TP>=2 due to a
  double-counted NVSHMEM `sync_counter`).

### 5. rdc=true default restored on Blackwell
- Re-verified on CUDA 13.2 + NVSHMEM 3.6.5: the 2026-04 "rdc=true hangs
  sm_100a" symptom is gone. The previous register-inflation issue (166 → 255
  regs from pulling in `libnvshmem_device.a`) does not reproduce on the
  current toolchain.
- Blackwell now uses the standard NVSHMEM link path (`-rdc=true
  -lnvshmem_device`). The old self-contained-allreduce workaround
  (`-DNVSHMEM_NO_DEVICE_LIB`, hand-rolled `__managed__
  nvshmemi_device_state_d`, `nvshmemid_hostlib_init_attr` callback,
  `ar x collective_launch.cpp.o`, host-side stubs) is kept behind
  `MPK_RDC_FALSE=1` as an escape hatch for regression isolation.
- Verification (all PASS on 2026-04-22): TP=2 9.645 ms/tok; TP=4 mbt=1 spec=1
  9.280 ms/tok; TP=4 mbt=1 spec=3 13.205 ms/tok; TP=4 mbt=64 28.614 ms/tok.
  rdc=true is slightly faster than the old rdc=false workaround.

### 6. Performance (B200, CUDA 13.2, NVSHMEM 3.6.5)

| Config | Latency |
|---|---|
| TP=1, 4K real prompt, mbt=64, seq=4500 | **21 ms/tok** end-to-end |
| TP=2, layers 0-5, mbt=1 | 9.65 ms/tok |
| TP=4, layers 0-5, mbt=1, spec=0 | ≈10 ms/tok |
| TP=4, layers 0-5, mbt=1, spec=1 | 9.28 ms/tok |
| TP=4, layers 0-5, mbt=1, spec=3 | 13.21 ms/tok |
| TP=4, layers 0-5, mbt=64, spec=0 | 28.61 ms/tok |

### 7. Bug fixes (selected — full list in `bugfix.md`)

- **Bug 14**: MLA TP decode kernels needed warp-0-only `tcgen05.alloc` +
  `bar.sync 1, 128` for internal syncs. `__syncthreads()` caused drift.
- **Bug 17-20**: MLA prefill integration — barrier ID clash with CUTLASS
  reserved range, q_start >= Q_LEN fast path, `O(N²)` decode trap when
  `max_seq_length` is over-allocated.
- **Bug 21 (Task #22)**: rdc=true assumption outdated on CUDA 13.2; switched
  back to default + kept escape hatch.
- **Bug 23**: TP=4 moe_w2_fp8 hang — scale warp overwrote `sfa_smem` /
  `sfb_smem` mid-UTCCP before MMA's read completed. Fix: scale warp waits on
  `ab_empty[smem_buf]` before writing the next k-tile.
- **MTP TP>=2 illegal memory access**: Double allreduce under fused-residual.
  Fix: use `linear_fp8_with_residual_sm100` and drop the redundant
  elementwise_add + second AllReduce (matches main-network MoE pattern).
- **MTP prefill input was argmax of main model logits**: replaced with
  ground-truth prompt tokens (vLLM-aligned).

### 8. Developer-facing cleanup

All experimental env flags / debug dumps added during the bring-up are
removed. The compiled behavior now matches what was proven working on 2026-04-22:

- **Removed** (all were ours): `MPK_USE_PREFILL`, `MPK_FUSE_RESIDUAL`,
  `MPK_SKIP_{ATTN,MLP,LAYER,MLA_ONLY,SHARED_EXPERT,ROUTED_EXPERTS,LM_HEAD,MOE,MOE_EXPERTS,MTP_DECODER,ALLREDUCE}`,
  `MPK_DUMP_{QNOPE,ATTN_PROJ,LOGITS,MOE,MOE_OUTPUT,BUFFERS,TAG,DIR}`,
  `MPK_MLA_CHECKPOINT`, `MPK_REF_{NO_QUANT,TRUE_FP8}`, `MPK_DRY_RUN`,
  `MPK_NO_RESIDUAL`, `MPK_NO_NVSHMEM`, `MPK_AR_LOCAL_COPY`,
  `MPK_ENABLE_VERBOSE`, `MPK_PTXAS_VERBOSE`, `MPK_PRECOMPILED_SO`,
  `MPK_FORCE_ALLGATHER_REDUCE`, `MPK_MOE_RAW_DEQUANT`. C++ `#ifdef` blocks
  wrapping these got their default branch kept, the debug branch deleted.
- **Kept** (upstream / necessary): `MPK_TEST_MODE` (upstream PR #652),
  `MPK_RDC_FALSE` (documented escape hatch), `MPK_SO_PATH` (internal
  Python→C++ plumbing), `MPK_TARGET_CC` / `MPK_MAX_*` / `MPK_PAGE_SIZE`
  (compile-time config), `MPK_ENABLE_TMA` / `MPK_ENABLE_PROFILING`.
- **Demo cleanup**: PyTorch reference inference path removed. The
  `--correctness` flag + reference cosine-sim comparison are gone. The
  `--use-mirage` flag is still the switch between MPK inference and native
  `AutoModelForCausalLM.generate` (kept as a pure baseline, not a
  reference).
- **MTP CLI consolidated**: `--mtp` (store_true) + `--num-speculative-tokens
  N` → single `--mtp N` (int, choices `[0, 1, 2, 3]`, default `0` = off).
- **Dead test scripts deleted**: `test_tp4_correctness.sh`,
  `test_mla_tp_matrix.sh`, `tests/runtime_python/blackwell/test_deepseek_v3_tp.sh`,
  `tests/runtime_python/blackwell/test_deepseek_v3_matrix.py`,
  `MLA_KERNEL_COMPARISON.md` (all correctness-path only).

## How to run

```bash
# TP=1, few-layer smoke
python demo/deepseek_v3/demo.py \
    --model-path /path/to/DeepSeek-V3 --use-mirage --layers 0-5 \
    --max-num-batched-tokens 1 --max-seq-length 512 --max-num-pages 8 \
    --max-new-tokens 16

# TP=4, real 4K prompt, MTP spec=3, full 40 layers
mpirun --allow-run-as-root -np 4 \
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
    python demo/deepseek_v3/demo.py \
    --model-path /path/to/DeepSeek-V3 --use-mirage --layers 0-39 \
    --max-num-batched-tokens 64 --prompt-length 4096 \
    --max-seq-length 4528 --max-num-pages 40 --max-new-tokens 512 \
    --mtp 3
```

See `demo/deepseek_v3/readme.md` for the full env setup (NVSHMEM, MPI paths).

## Test plan

- [x] TP=1 DeepSeek V3, 2 layers: megakernel compiles, runs, produces tokens
- [x] TP=2 DeepSeek V3, 6 layers: 9.65 ms/tok
- [x] TP=4 DeepSeek V3, 6 layers, mbt=1/64, MTP spec=0/1/3: all PASS, 9-29 ms/tok
- [x] TP=1 40 layers, 4K real prompt: 21 ms/tok end-to-end
- [x] `MPK_RDC_FALSE=1` escape hatch still compiles + runs
- [x] Qwen3 TP=1 demo still passes (upstream PR 661 merged in)
- [x] `demo/deepseek_v3/demo.py --help` lists only the kept args; no stale
      `--correctness` / `--num-speculative-tokens` reference
- [x] `grep -r "MPK_USE_PREFILL\|MPK_FUSE_RESIDUAL\|MPK_SKIP_\|MPK_DUMP_\|..."
      demo python include src` returns nothing

## Out of scope / follow-ups

- CUDA-IPC replacement for NVSHMEM on same-node TP (removes the
  `LD_PRELOAD=libnvshmem_host.so` requirement entirely).
- Perfetto profiler task-type dict needed new task labels (263-294); fixed
  in `python/mirage/mpk/profiler_persistent.py`. End-to-end TP=4 Perfetto
  trace matrix still to be re-run after this PR lands.
- Full 61-layer TP=4 stress (currently tested up to 40 layers).
